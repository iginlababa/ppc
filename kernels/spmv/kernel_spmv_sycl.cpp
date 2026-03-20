// kernel_spmv_sycl.cpp — E4 SpMV: SYCL USM nd_range<1> implementation.
//
// ── E4 DESIGN DECISIONS ───────────────────────────────────────────────────────
// [D3-SYCL] nd_range<1>: work-item i computes y[i] = dot-product of row i.
//   Each work-item handles exactly one row — same one-thread-per-row design as
//   the CUDA/HIP baselines. No warp-reduction tricks.
//   Work-group size: 256 work-items (= SPMV_BLOCK_SIZE).
// [D-USM]  malloc_device for all CSR arrays; same pointer interface as HIP/CUDA.
// [D7-SYCL] Adaptive warmup: host-side chrono after queue.wait().
// Compatible with:
//   AdaptiveCpp  --acpp-targets=hip:gfx942          (AMD MI300X)
//   icpx/clang++ -fsycl -fsycl-targets=nvptx64...  (NVIDIA)
// ─────────────────────────────────────────────────────────────────────────────

#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "spmv_common.h"

// ── SpMV CSR: one work-item per row ───────────────────────────────────────────
static void run_spmv_sycl(sycl::queue& q,
                           const int*    row_ptr,
                           const int*    col_idx,
                           const double* values,
                           const double* x,
                           double*       y,
                           int           nrows)
{
    // Round global size up to a multiple of SPMV_BLOCK_SIZE
    const size_t local_sz  = static_cast<size_t>(SPMV_BLOCK_SIZE);
    const size_t global_sz = ((static_cast<size_t>(nrows) + local_sz - 1) / local_sz) * local_sz;

    q.parallel_for(
        sycl::nd_range<1>(global_sz, local_sz),
        [=](sycl::nd_item<1> item) {
            const int row = static_cast<int>(item.get_global_id(0));
            if (row >= nrows) return;
            double sum = 0.0;
            const int start = row_ptr[row];
            const int end   = row_ptr[row + 1];
            for (int j = start; j < end; j++)
                sum += values[j] * x[col_idx[j]];
            y[row] = sum;
        });
    q.wait();
}

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  --n <N>            Number of rows (default: %d)\n"
        "  --matrix <type>    Matrix type: laplacian_2d|random_sparse|power_law (default: laplacian_2d)\n"
        "  --warmup <W>       Max adaptive warmup iterations (default: %d)\n"
        "  --reps <R>         Timed iterations (default: %d)\n"
        "  --platform <P>     Platform tag (default: unknown)\n"
        "  --verify           Correctness check before timing\n",
        prog, SPMV_N_LARGE, SPMV_WARMUP_MAX, SPMV_REPS_DEFAULT);
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    int         N        = SPMV_N_LARGE;
    std::string matstr   = "laplacian_2d";
    int         warmup   = SPMV_WARMUP_MAX;
    int         reps     = SPMV_REPS_DEFAULT;
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

    SpmvMatType mtype = parse_matrix_type(matstr);

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
    std::fprintf(stderr, "[E4 sycl] device: %s\n",
                 q.get_device().get_info<sycl::info::device::name>().c_str());

    // ── Correctness check (D5) ────────────────────────────────────────────────
    if (verify) {
        SpmvCSR vcsr = generate_laplacian_2d(64);
        std::vector<double> vx  = make_x_vector(vcsr.nrows);
        std::vector<double> ref(vcsr.nrows, 0.0);
        spmv_cpu_ref(vcsr, vx.data(), ref.data());

        int*    d_rp  = sycl::malloc_device<int>   (vcsr.nrows + 1, q);
        int*    d_ci  = sycl::malloc_device<int>   (vcsr.nnz,       q);
        double* d_val = sycl::malloc_device<double>(vcsr.nnz,       q);
        double* d_x   = sycl::malloc_device<double>(vcsr.nrows,     q);
        double* d_y   = sycl::malloc_device<double>(vcsr.nrows,     q);

        q.memcpy(d_rp,  vcsr.row_ptr.data(), (vcsr.nrows+1)*sizeof(int)).wait();
        q.memcpy(d_ci,  vcsr.col_idx.data(), vcsr.nnz*sizeof(int)).wait();
        q.memcpy(d_val, vcsr.values.data(),  vcsr.nnz*sizeof(double)).wait();
        q.memcpy(d_x,   vx.data(),           vcsr.nrows*sizeof(double)).wait();
        q.memset(d_y, 0, vcsr.nrows*sizeof(double)).wait();

        run_spmv_sycl(q, d_rp, d_ci, d_val, d_x, d_y, vcsr.nrows);

        std::vector<double> res(vcsr.nrows);
        q.memcpy(res.data(), d_y, vcsr.nrows*sizeof(double)).wait();
        sycl::free(d_rp,  q); sycl::free(d_ci,  q);
        sycl::free(d_val, q); sycl::free(d_x,   q); sycl::free(d_y, q);

        double max_err = 0.0;
        bool ok = spmv_verify(res.data(), ref.data(), vcsr.nrows, SPMV_CORRECT_TOL, &max_err);
        std::printf("VERIFY abstraction=sycl matrix=laplacian_2d N=%d max_rel_err=%.2e %s\n",
                    vcsr.nrows, max_err, ok ? "PASS" : "FAIL");
        if (!ok) {
            std::fprintf(stderr, "[E4 verify] sycl FAILED — aborting.\n");
            return 1;
        }
        std::fprintf(stderr, "[E4 verify] sycl PASS — proceeding to timed measurement.\n");
    }

    // ── Build matrix ──────────────────────────────────────────────────────────
    SpmvCSR csr = build_matrix(mtype, N);
    int  nrows = csr.nrows;
    long nnz   = csr.nnz;
    std::vector<double> x = make_x_vector(nrows);

    std::printf("# abstraction=sycl matrix=%s N=%d nnz=%ld warmup_max=%d reps=%d platform=%s\n",
                matrix_type_str(mtype), nrows, nnz, warmup, reps, platform.c_str());

    // ── Allocate device memory (USM) ──────────────────────────────────────────
    int*    d_rp  = sycl::malloc_device<int>   (nrows + 1, q);
    int*    d_ci  = sycl::malloc_device<int>   (nnz,       q);
    double* d_val = sycl::malloc_device<double>(nnz,       q);
    double* d_x   = sycl::malloc_device<double>(nrows,     q);
    double* d_y   = sycl::malloc_device<double>(nrows,     q);
    if (!d_rp || !d_ci || !d_val || !d_x || !d_y) {
        std::fprintf(stderr, "sycl::malloc_device failed for N=%d nnz=%ld\n", nrows, nnz);
        return 1;
    }
    q.memcpy(d_rp,  csr.row_ptr.data(), (nrows+1)*sizeof(int)).wait();
    q.memcpy(d_ci,  csr.col_idx.data(), nnz*sizeof(int)).wait();
    q.memcpy(d_val, csr.values.data(),  nnz*sizeof(double)).wait();
    q.memcpy(d_x,   x.data(),           nrows*sizeof(double)).wait();
    q.memset(d_y, 0, nrows*sizeof(double)).wait();

    const char* mstr = matrix_type_str(mtype);

    auto run_once = [&]() {
        run_spmv_sycl(q, d_rp, d_ci, d_val, d_x, d_y, nrows);
    };

    // ── Adaptive warmup (D7) ──────────────────────────────────────────────────
    int warmup_iters = spmv_adaptive_warmup(run_once, SPMV_WARMUP_MIN, warmup);
    std::fprintf(stderr, "[E4] sycl: adaptive warmup done in %d iterations\n", warmup_iters);

    // ── Timed runs ────────────────────────────────────────────────────────────
    std::vector<double> gflops_vec;
    gflops_vec.reserve(reps);
    for (int r = 1; r <= reps; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        run_once();
        auto t1 = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double gf = spmv_gflops(nnz, time_ms / 1000.0);
        gflops_vec.push_back(gf);
        spmv_print_run(r, nrows, nnz, mstr, time_ms, gf);
    }

    // ── hw_state_verified ─────────────────────────────────────────────────────
    auto flags = spmv_compute_hw_state(gflops_vec);
    for (int r = 0; r < reps; r++)
        spmv_print_hw_state(r + 1, flags[r]);

    // ── Summary ───────────────────────────────────────────────────────────────
    SpmvStats stats = spmv_compute_stats(gflops_vec, flags);
    spmv_print_summary(nrows, nnz, mstr, stats, warmup_iters);

    if (stats.n_clean > 0) {
        double time_s = 2.0 * nnz / (stats.median_gflops * 1.0e9);
        double gbs    = spmv_gbs_effective(nrows, nnz, time_s);
        double ai     = spmv_ai(nrows, nnz);
        std::printf("SPMV_ROOFLINE n=%d nnz=%ld matrix=%s median_gbs=%.4f ai=%.4f\n",
                    nrows, nnz, mstr, gbs, ai);
        std::fflush(stdout);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    sycl::free(d_rp,  q); sycl::free(d_ci,  q);
    sycl::free(d_val, q); sycl::free(d_x,   q); sycl::free(d_y, q);
    return 0;
}
