// kernel_sptrsv_raja.cpp — E5 SpTRSV: RAJA level-set forward substitution.
//
// ── E5 DESIGN DECISIONS ───────────────────────────────────────────────────────
// [D3-RAJA] Outer loop over levels on host. Within each level:
//   RAJA::forall<cuda_exec<256>> over RangeSegment(0, level_size).
//   RAJA::synchronize<RAJA::cuda_synchronize>() between levels ensures all x
//   writes from level l are visible before level l+1 reads them.
//   RAJA::forall is the correct idiomatic choice: SpTRSV's intra-level work
//   is a flat loop over independent rows — no hierarchical parallelism needed.
//   The level loop itself is sequential on the host; RAJA does not provide
//   a level-set primitive, so host sequencing is explicit and identical to
//   native CUDA. This is a fair test: the overhead under study is the
//   per-level RAJA dispatch cost multiplied by n_levels.
// ─────────────────────────────────────────────────────────────────────────────

#include <RAJA/RAJA.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "sptrsv_common.h"

using SptrsExecPolicy = RAJA::cuda_exec<SPTRSV_BLOCK_SIZE>;

// ── CUDA helpers (RAJA kernels use raw device pointers) ───────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
                         __FILE__, __LINE__, cudaGetErrorString(_e));           \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

void run_sptrsv_level_raja(
    const int* d_row_ptr, const int* d_col_idx, const double* d_values,
    const double* d_b, double* d_x, const int* d_level_rows,
    int level_start, int level_size)
{
    RAJA::forall<SptrsExecPolicy>(
        RAJA::RangeSegment(0, level_size),
        [=] RAJA_DEVICE (int tid) {
            int row = d_level_rows[level_start + tid];
            double sum  = d_b[row];
            double diag = 1.0;
            const int start = d_row_ptr[row];
            const int end   = d_row_ptr[row + 1];
            for (int j = start; j < end; j++) {
                int col = d_col_idx[j];
                if (col == row) {
                    diag = d_values[j];
                } else {
                    sum -= d_values[j] * d_x[col];
                }
            }
            d_x[row] = sum / diag;
        });
    RAJA::synchronize<RAJA::cuda_synchronize>();
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

    // ── Correctness check ─────────────────────────────────────────────────────
    if (verify) {
        SptrsCSR vcsr = generate_laplacian_lower(SPTRSV_N_SMALL);
        SptrsLevels vls = build_levels(vcsr);
        std::vector<double> vb  = make_b_vector(vcsr.nrows);
        std::vector<double> ref(vcsr.nrows, 0.0);
        sptrsv_cpu_ref(vcsr, vb.data(), ref.data());

        int*    d_rp; int*    d_ci;  double* d_val;
        double* d_b;  double* d_x;   int*    d_lr;
        CUDA_CHECK(cudaMalloc(&d_rp,  (vcsr.nrows + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_ci,  vcsr.nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_val, vcsr.nnz * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_b,   vcsr.nrows * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_x,   vcsr.nrows * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_lr,  vcsr.nrows * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_rp,  vcsr.row_ptr.data(),   (vcsr.nrows+1)*sizeof(int),   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ci,  vcsr.col_idx.data(),   vcsr.nnz*sizeof(int),          cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_val, vcsr.values.data(),    vcsr.nnz*sizeof(double),       cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b,   vb.data(),             vcsr.nrows*sizeof(double),     cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_lr,  vls.level_rows.data(), vcsr.nrows*sizeof(int),        cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_x, 0, vcsr.nrows * sizeof(double)));

        for (int l = 0; l < vls.n_levels; l++) {
            int lstart = vls.level_ptr[l];
            int lsize  = vls.level_ptr[l + 1] - lstart;
            run_sptrsv_level_raja(d_rp, d_ci, d_val, d_b, d_x, d_lr, lstart, lsize);
        }

        std::vector<double> res(vcsr.nrows);
        CUDA_CHECK(cudaMemcpy(res.data(), d_x, vcsr.nrows*sizeof(double), cudaMemcpyDeviceToHost));
        cudaFree(d_rp); cudaFree(d_ci); cudaFree(d_val);
        cudaFree(d_b);  cudaFree(d_x);  cudaFree(d_lr);

        double max_err = 0.0;
        bool ok = sptrsv_verify(res.data(), ref.data(), vcsr.nrows, SPTRSV_CORRECT_TOL, &max_err);
        std::printf("VERIFY abstraction=raja matrix=lower_triangular_laplacian N=%d max_rel_err=%.2e %s\n",
                    vcsr.nrows, max_err, ok ? "PASS" : "FAIL");
        if (!ok) {
            std::fprintf(stderr, "[E5 verify] raja FAILED — aborting.\n");
            return 1;
        }
        std::fprintf(stderr, "[E5 verify] raja PASS — proceeding to timed measurement.\n");
    }

    // ── Build matrix and level sets ───────────────────────────────────────────
    SptrsCSR csr = build_matrix(mtype, N);
    SptrsLevels ls = build_levels(csr);
    int  nrows  = csr.nrows;
    long nnz    = csr.nnz;
    std::vector<double> b = make_b_vector(nrows);

    std::printf(
        "# abstraction=raja matrix=%s N=%d nnz=%ld n_levels=%d"
        " max_lw=%d min_lw=%d warmup_max=%d reps=%d platform=%s\n",
        matrix_type_str(mtype), nrows, nnz, ls.n_levels,
        ls.max_lw, ls.min_lw, warmup, reps, platform.c_str());

    int*    d_rp; int*    d_ci;  double* d_val;
    double* d_b;  double* d_x;   int*    d_lr;
    CUDA_CHECK(cudaMalloc(&d_rp,  (nrows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ci,  nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_val, nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b,   nrows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x,   nrows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_lr,  nrows * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_rp,  csr.row_ptr.data(),   (nrows+1)*sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ci,  csr.col_idx.data(),   nnz*sizeof(int),          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_val, csr.values.data(),    nnz*sizeof(double),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b,   b.data(),             nrows*sizeof(double),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lr,  ls.level_rows.data(), nrows*sizeof(int),        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_x, 0, nrows * sizeof(double)));

    const char* mstr = matrix_type_str(mtype);

    auto run_solve = [&]() {
        for (int l = 0; l < ls.n_levels; l++) {
            int lstart = ls.level_ptr[l];
            int lsize  = ls.level_ptr[l + 1] - lstart;
            run_sptrsv_level_raja(d_rp, d_ci, d_val, d_b, d_x, d_lr, lstart, lsize);
        }
    };

    auto run_with_reset = [&]() {
        CUDA_CHECK(cudaMemset(d_x, 0, nrows * sizeof(double)));
        run_solve();
    };

    int warmup_iters = sptrsv_adaptive_warmup(run_with_reset, SPTRSV_WARMUP_MIN, warmup);
    std::fprintf(stderr, "[E5] raja: adaptive warmup done in %d iterations\n", warmup_iters);

    std::vector<double> gflops_vec;
    gflops_vec.reserve(reps);
    for (int r = 1; r <= reps; r++) {
        CUDA_CHECK(cudaMemset(d_x, 0, nrows * sizeof(double)));
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

    cudaFree(d_rp); cudaFree(d_ci); cudaFree(d_val);
    cudaFree(d_b);  cudaFree(d_x);  cudaFree(d_lr);
    return 0;
}
