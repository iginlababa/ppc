// kernel_sptrsv_sycl.cpp — E5 SpTRSV: SYCL USM nd_range<1> level-set implementation.
//
// ── E5 DESIGN DECISIONS ───────────────────────────────────────────────────────
// [D3-SYCL] One q.parallel_for per level; q.wait() after each enforces inter-level
//   ordering. This is the SYCL equivalent of cudaDeviceSynchronize() between kernel
//   launches. The q.wait() call blocks the host until the device completes the level,
//   guaranteeing all x[] writes from level l are visible before level l+1 reads them.
//   Within each level: one work-item per row (same 1D design as HIP/CUDA baselines).
//   Work-group size: SPTRSV_BLOCK_SIZE (256) work-items.
// [D-USM] malloc_device for all CSR, b, x, and level_rows arrays.
//   q.memset(d_x, 0, ...) before each rep to reset solution vector.
// [D7-SYCL] Adaptive warmup: host-side chrono after q.wait() in level loop.
// [D-AMD] Compatible with AdaptiveCpp: --acpp-targets=hip:gfx942
//   Expected: SYCL will show high per-level dispatch overhead × n_levels on AMD
//   (E4 already showed ~52% efficiency for single nd_range<1>; compounded here).
// ─────────────────────────────────────────────────────────────────────────────

#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "sptrsv_common.h"

// ── One level of SpTRSV via SYCL; q.wait() enforces inter-level ordering ──────
static void run_sptrsv_level_sycl(sycl::queue& q,
                                   const int*    row_ptr,
                                   const int*    col_idx,
                                   const double* values,
                                   const double* b,
                                   double*       x,
                                   const int*    level_rows,
                                   int           level_start,
                                   int           level_size)
{
    const size_t local_sz  = static_cast<size_t>(SPTRSV_BLOCK_SIZE);
    const size_t global_sz =
        ((static_cast<size_t>(level_size) + local_sz - 1) / local_sz) * local_sz;

    q.parallel_for(
        sycl::nd_range<1>(global_sz, local_sz),
        [=](sycl::nd_item<1> item) {
            const int tid = static_cast<int>(item.get_global_id(0));
            if (tid >= level_size) return;

            const int row = level_rows[level_start + tid];
            double sum  = b[row];
            double diag = 1.0;
            const int start = row_ptr[row];
            const int end   = row_ptr[row + 1];
            for (int j = start; j < end; j++) {
                int col = col_idx[j];
                if (col == row) {
                    diag = values[j];
                } else {
                    sum -= values[j] * x[col];
                }
            }
            x[row] = sum / diag;
        });
    q.wait();  // mandatory inter-level sync — correctness depends on this
}

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  --n <N>            Number of rows (default: %d)\n"
        "  --matrix <type>    lower_triangular_laplacian|lower_triangular_random (default: lower_triangular_laplacian)\n"
        "  --warmup <W>       Max adaptive warmup iterations (default: %d)\n"
        "  --reps <R>         Timed iterations (default: %d)\n"
        "  --platform <P>     Platform tag (default: unknown)\n"
        "  --verify           Correctness check before timing\n",
        prog, SPTRSV_N_LARGE, SPTRSV_WARMUP_MAX, SPTRSV_REPS_DEFAULT);
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    int         N        = SPTRSV_N_LARGE;
    std::string matstr   = "lower_triangular_laplacian";
    int         warmup   = SPTRSV_WARMUP_MAX;
    int         reps     = SPTRSV_REPS_DEFAULT;
    std::string platform = "unknown";
    bool        verify   = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--n"        && i+1 < argc) { N        = std::stoi(argv[++i]); }
        else if (a == "--matrix"   && i+1 < argc) { matstr   = argv[++i]; }
        else if (a == "--warmup"   && i+1 < argc) { warmup   = std::stoi(argv[++i]); }
        else if (a == "--reps"     && i+1 < argc) { reps     = std::stoi(argv[++i]); }
        else if (a == "--platform" && i+1 < argc) { platform = argv[++i]; }
        else if (a == "--verify")                  { verify   = true; }
        else { print_usage(argv[0]); return 1; }
    }

    SptrsMatType mtype = parse_matrix_type(matstr);

    // ── SYCL queue ────────────────────────────────────────────────────────────
    sycl::queue q{sycl::gpu_selector_v,
                  [](const sycl::exception_list& el) {
                      for (auto& e : el) {
                          try { std::rethrow_exception(e); }
                          catch (const sycl::exception& ex) {
                              std::fprintf(stderr, "SYCL async exception: %s\n", ex.what());
                          }
                      }
                  }};
    std::fprintf(stderr, "[E5 sycl] device: %s\n",
                 q.get_device().get_info<sycl::info::device::name>().c_str());

    // ── Correctness check ─────────────────────────────────────────────────────
    if (verify) {
        SptrsCSR vcsr = generate_laplacian_lower(SPTRSV_N_SMALL);
        SptrsLevels vls = build_levels(vcsr);
        std::vector<double> vb  = make_b_vector(vcsr.nrows);
        std::vector<double> ref(vcsr.nrows, 0.0);
        sptrsv_cpu_ref(vcsr, vb.data(), ref.data());

        int*    d_rp  = sycl::malloc_device<int>   (vcsr.nrows + 1, q);
        int*    d_ci  = sycl::malloc_device<int>   (vcsr.nnz,       q);
        double* d_val = sycl::malloc_device<double>(vcsr.nnz,       q);
        double* d_b   = sycl::malloc_device<double>(vcsr.nrows,     q);
        double* d_x   = sycl::malloc_device<double>(vcsr.nrows,     q);
        int*    d_lr  = sycl::malloc_device<int>   (vcsr.nrows,     q);

        q.memcpy(d_rp,  vcsr.row_ptr.data(),   (vcsr.nrows+1)*sizeof(int)).wait();
        q.memcpy(d_ci,  vcsr.col_idx.data(),   vcsr.nnz*sizeof(int)).wait();
        q.memcpy(d_val, vcsr.values.data(),    vcsr.nnz*sizeof(double)).wait();
        q.memcpy(d_b,   vb.data(),             vcsr.nrows*sizeof(double)).wait();
        q.memcpy(d_lr,  vls.level_rows.data(), vcsr.nrows*sizeof(int)).wait();
        q.memset(d_x, 0, vcsr.nrows * sizeof(double)).wait();

        for (int l = 0; l < vls.n_levels; l++) {
            int lstart = vls.level_ptr[l];
            int lsize  = vls.level_ptr[l + 1] - lstart;
            run_sptrsv_level_sycl(q, d_rp, d_ci, d_val, d_b, d_x, d_lr, lstart, lsize);
        }

        std::vector<double> res(vcsr.nrows);
        q.memcpy(res.data(), d_x, vcsr.nrows*sizeof(double)).wait();
        sycl::free(d_rp, q); sycl::free(d_ci,  q); sycl::free(d_val, q);
        sycl::free(d_b,  q); sycl::free(d_x,   q); sycl::free(d_lr,  q);

        double max_err = 0.0;
        bool ok = sptrsv_verify(res.data(), ref.data(), vcsr.nrows, SPTRSV_CORRECT_TOL, &max_err);
        std::printf("VERIFY abstraction=sycl matrix=lower_triangular_laplacian N=%d max_rel_err=%.2e %s\n",
                    vcsr.nrows, max_err, ok ? "PASS" : "FAIL");
        if (!ok) {
            std::fprintf(stderr, "[E5 verify] sycl FAILED — aborting.\n");
            return 1;
        }
        std::fprintf(stderr, "[E5 verify] sycl PASS — proceeding to timed measurement.\n");
    }

    // ── Build matrix and level sets ───────────────────────────────────────────
    SptrsCSR csr = build_matrix(mtype, N);
    SptrsLevels ls = build_levels(csr);
    int  nrows = csr.nrows;
    long nnz   = csr.nnz;
    std::vector<double> b = make_b_vector(nrows);

    std::printf(
        "# abstraction=sycl matrix=%s N=%d nnz=%ld n_levels=%d"
        " max_lw=%d min_lw=%d warmup_max=%d reps=%d platform=%s\n",
        matrix_type_str(mtype), nrows, nnz, ls.n_levels,
        ls.max_lw, ls.min_lw, warmup, reps, platform.c_str());

    // ── Allocate device memory (USM) ──────────────────────────────────────────
    int*    d_rp  = sycl::malloc_device<int>   (nrows + 1, q);
    int*    d_ci  = sycl::malloc_device<int>   (nnz,       q);
    double* d_val = sycl::malloc_device<double>(nnz,       q);
    double* d_b   = sycl::malloc_device<double>(nrows,     q);
    double* d_x   = sycl::malloc_device<double>(nrows,     q);
    int*    d_lr  = sycl::malloc_device<int>   (nrows,     q);
    if (!d_rp || !d_ci || !d_val || !d_b || !d_x || !d_lr) {
        std::fprintf(stderr, "sycl::malloc_device failed for N=%d nnz=%ld\n", nrows, nnz);
        return 1;
    }
    q.memcpy(d_rp,  csr.row_ptr.data(),   (nrows+1)*sizeof(int)).wait();
    q.memcpy(d_ci,  csr.col_idx.data(),   nnz*sizeof(int)).wait();
    q.memcpy(d_val, csr.values.data(),    nnz*sizeof(double)).wait();
    q.memcpy(d_b,   b.data(),             nrows*sizeof(double)).wait();
    q.memcpy(d_lr,  ls.level_rows.data(), nrows*sizeof(int)).wait();

    const char* mstr = matrix_type_str(mtype);

    auto run_solve = [&]() {
        for (int l = 0; l < ls.n_levels; l++) {
            int lstart = ls.level_ptr[l];
            int lsize  = ls.level_ptr[l + 1] - lstart;
            run_sptrsv_level_sycl(q, d_rp, d_ci, d_val, d_b, d_x, d_lr, lstart, lsize);
        }
    };

    auto run_with_reset = [&]() {
        q.memset(d_x, 0, nrows * sizeof(double)).wait();
        run_solve();
    };

    int warmup_iters = sptrsv_adaptive_warmup(run_with_reset, SPTRSV_WARMUP_MIN, warmup);
    std::fprintf(stderr, "[E5] sycl: adaptive warmup done in %d iterations\n", warmup_iters);

    // ── Timed runs ────────────────────────────────────────────────────────────
    std::vector<double> gflops_vec;
    gflops_vec.reserve(reps);
    for (int r = 1; r <= reps; r++) {
        q.memset(d_x, 0, nrows * sizeof(double)).wait();
        auto t0 = std::chrono::high_resolution_clock::now();
        run_solve();
        auto t1 = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double gf = sptrsv_gflops(nnz, time_ms / 1000.0);
        gflops_vec.push_back(gf);
        sptrsv_print_run(r, nrows, nnz, ls.n_levels, ls.max_lw, ls.min_lw,
                         mstr, time_ms, gf);
    }

    auto flags = sptrsv_compute_hw_state(gflops_vec);
    for (int r = 0; r < reps; r++)
        sptrsv_print_hw_state(r + 1, flags[r]);

    SptrsStats stats = sptrsv_compute_stats(gflops_vec, flags);
    sptrsv_print_summary(nrows, nnz, ls.n_levels, ls.max_lw, ls.min_lw,
                         mstr, stats, warmup_iters);

    if (stats.n_clean > 0) {
        double ai = sptrsv_ai(nrows, nnz);
        std::printf("SPTRSV_ROOFLINE n_rows=%d nnz=%ld matrix=%s n_levels=%d"
                    " median_gflops=%.4f ai=%.4f\n",
                    nrows, nnz, mstr, ls.n_levels, stats.median_gflops, ai);
        std::fflush(stdout);
    }

    sycl::free(d_rp, q); sycl::free(d_ci,  q); sycl::free(d_val, q);
    sycl::free(d_b,  q); sycl::free(d_x,   q); sycl::free(d_lr,  q);
    return 0;
}
