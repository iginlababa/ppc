// kernel_spmv_raja.cpp — E4 SpMV: RAJA::forall over rows.
//
// ── E4 DESIGN DECISIONS ───────────────────────────────────────────────────────
// [D3-RAJA] RAJA::forall<cuda_exec<BLOCK>> over RangeSegment(0, nrows).
//   Each lambda invocation handles one row — identical semantics to native CUDA
//   one-thread-per-row, but expressed through RAJA's execution policy abstraction.
//   Unlike E2 raja_naive (which suffered from API limitation — RAJA::forall cannot
//   express shared-memory tiling for DGEMM), RAJA::forall is the CORRECT idiomatic
//   RAJA for SpMV. No API limitation here: SpMV requires only a forall loop,
//   not hierarchical parallelism or scratchpad memory.
//   Label: "raja" (not "raja_naive") — this is an ideal fit for RAJA's model.
// ─────────────────────────────────────────────────────────────────────────────

#include <RAJA/RAJA.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "spmv_common.h"

using SpmvExecPolicy = RAJA::cuda_exec<SPMV_BLOCK_SIZE>;

void run_spmv_raja(const int* d_row_ptr, const int* d_col_idx,
                    const double* d_values, const double* d_x,
                    double* d_y, int nrows) {
    RAJA::forall<SpmvExecPolicy>(
        RAJA::RangeSegment(0, nrows),
        [=] RAJA_DEVICE (int row) {
            double sum = 0.0;
            const int start = d_row_ptr[row];
            const int end   = d_row_ptr[row + 1];
            for (int j = start; j < end; j++)
                sum += d_values[j] * d_x[d_col_idx[j]];
            d_y[row] = sum;
        });
    RAJA::synchronize<RAJA::cuda_synchronize>();
}

// ── CUDA malloc helpers (RAJA kernels operate on raw device pointers) ─────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
                         __FILE__, __LINE__, cudaGetErrorString(_e));           \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

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

    // ── Correctness check ─────────────────────────────────────────────────────
    if (verify) {
        SpmvCSR vcsr = generate_laplacian_2d(64);
        std::vector<double> vx  = make_x_vector(vcsr.nrows);
        std::vector<double> ref(vcsr.nrows, 0.0);
        spmv_cpu_ref(vcsr, vx.data(), ref.data());

        int*    d_rp; int*    d_ci;
        double* d_val; double* d_x; double* d_y;
        CUDA_CHECK(cudaMalloc(&d_rp,  (vcsr.nrows+1)*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_ci,  vcsr.nnz*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_val, vcsr.nnz*sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_x,   vcsr.nrows*sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_y,   vcsr.nrows*sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_rp,  vcsr.row_ptr.data(), (vcsr.nrows+1)*sizeof(int),   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ci,  vcsr.col_idx.data(), vcsr.nnz*sizeof(int),          cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_val, vcsr.values.data(),  vcsr.nnz*sizeof(double),       cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_x,   vx.data(),           vcsr.nrows*sizeof(double),     cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_y, 0, vcsr.nrows*sizeof(double)));

        run_spmv_raja(d_rp, d_ci, d_val, d_x, d_y, vcsr.nrows);

        std::vector<double> res(vcsr.nrows);
        CUDA_CHECK(cudaMemcpy(res.data(), d_y, vcsr.nrows*sizeof(double), cudaMemcpyDeviceToHost));
        cudaFree(d_rp); cudaFree(d_ci); cudaFree(d_val); cudaFree(d_x); cudaFree(d_y);

        double max_err = 0.0;
        bool ok = spmv_verify(res.data(), ref.data(), vcsr.nrows, SPMV_CORRECT_TOL, &max_err);
        std::printf("VERIFY abstraction=raja matrix=laplacian_2d N=%d max_rel_err=%.2e %s\n",
                    vcsr.nrows, max_err, ok ? "PASS" : "FAIL");
        if (!ok) {
            std::fprintf(stderr, "[E4 verify] raja FAILED — aborting.\n");
            return 1;
        }
        std::fprintf(stderr, "[E4 verify] raja PASS — proceeding to timed measurement.\n");
    }

    // ── Build matrix ──────────────────────────────────────────────────────────
    SpmvCSR csr = build_matrix(mtype, N);
    int  nrows  = csr.nrows;
    long nnz    = csr.nnz;
    std::vector<double> x = make_x_vector(nrows);

    std::printf("# abstraction=raja matrix=%s N=%d nnz=%ld warmup_max=%d reps=%d platform=%s\n",
                matrix_type_str(mtype), nrows, nnz, warmup, reps, platform.c_str());

    // ── Allocate device memory ────────────────────────────────────────────────
    int*    d_rp; int*    d_ci;
    double* d_val; double* d_x; double* d_y;
    if (cudaMalloc(&d_rp,  (nrows+1)*sizeof(int))  != cudaSuccess ||
        cudaMalloc(&d_ci,  nnz*sizeof(int))         != cudaSuccess ||
        cudaMalloc(&d_val, nnz*sizeof(double))      != cudaSuccess ||
        cudaMalloc(&d_x,   nrows*sizeof(double))    != cudaSuccess ||
        cudaMalloc(&d_y,   nrows*sizeof(double))    != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed for N=%d nnz=%ld\n", nrows, nnz);
        return 1;
    }
    CUDA_CHECK(cudaMemcpy(d_rp,  csr.row_ptr.data(), (nrows+1)*sizeof(int),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ci,  csr.col_idx.data(), nnz*sizeof(int),         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_val, csr.values.data(),  nnz*sizeof(double),      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x,   x.data(),           nrows*sizeof(double),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_y, 0, nrows*sizeof(double)));

    const char* mstr = matrix_type_str(mtype);

    auto run_once = [&]() {
        run_spmv_raja(d_rp, d_ci, d_val, d_x, d_y, nrows);
    };

    // ── Adaptive warmup ───────────────────────────────────────────────────────
    int warmup_iters = spmv_adaptive_warmup(run_once, SPMV_WARMUP_MIN, warmup);
    std::fprintf(stderr, "[E4] raja: adaptive warmup done in %d iterations\n", warmup_iters);

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

    auto flags = spmv_compute_hw_state(gflops_vec);
    for (int r = 0; r < reps; r++)
        spmv_print_hw_state(r + 1, flags[r]);

    SpmvStats stats = spmv_compute_stats(gflops_vec, flags);
    spmv_print_summary(nrows, nnz, mstr, stats, warmup_iters);

    cudaFree(d_rp); cudaFree(d_ci); cudaFree(d_val); cudaFree(d_x); cudaFree(d_y);
    return 0;
}
