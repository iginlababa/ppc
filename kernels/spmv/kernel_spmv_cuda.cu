// kernel_spmv_cuda.cu — E4 SpMV: native CUDA CSR implementation.
//
// ── E4 DESIGN DECISIONS ───────────────────────────────────────────────────────
// [D3-CUDA] One thread per row. Each thread accumulates its row's dot product.
//   This is the scientifically honest baseline — no warp-level reductions or
//   row-blocking that would mask abstraction overhead in comparisons.
//   Access to x[col_idx[j]] is irregular (non-coalesced) — this is inherent to
//   SpMV and expected to be the bottleneck (memory-bound, low AI ≈ 0.13 FLOP/byte).
//   Block size: SPMV_BLOCK_SIZE = 256 threads.
// [D7-CUDA] Adaptive warmup: loops until CV < 2% over last 10 timings.
//   Primary output metric: GFLOP/s = 2*nnz / time_s / 1e9.
// ─────────────────────────────────────────────────────────────────────────────

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "spmv_common.h"

// ── CUDA error helper ─────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
                         __FILE__, __LINE__, cudaGetErrorString(_e));           \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// ── SpMV CSR kernel: one thread per row ───────────────────────────────────────
__global__ void spmv_csr_kernel(const int*    __restrict__ row_ptr,
                                 const int*    __restrict__ col_idx,
                                 const double* __restrict__ values,
                                 const double* __restrict__ x,
                                 double*                    y,
                                 int nrows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;
    double sum = 0.0;
    const int start = row_ptr[row];
    const int end   = row_ptr[row + 1];
    for (int j = start; j < end; j++)
        sum += values[j] * x[col_idx[j]];
    y[row] = sum;
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

    // ── Correctness check (D5) ────────────────────────────────────────────────
    if (verify) {
        SpmvCSR vcsr = generate_laplacian_2d(64);  // 8×8 grid
        std::vector<double> vx  = make_x_vector(vcsr.nrows);
        std::vector<double> ref(vcsr.nrows, 0.0);
        spmv_cpu_ref(vcsr, vx.data(), ref.data());

        int*    d_rp;  int*    d_ci;
        double* d_val; double* d_x; double* d_y;
        CUDA_CHECK(cudaMalloc(&d_rp,  (vcsr.nrows + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_ci,  vcsr.nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_val, vcsr.nnz * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_x,   vcsr.nrows * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_y,   vcsr.nrows * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_rp,  vcsr.row_ptr.data(), (vcsr.nrows+1)*sizeof(int),   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ci,  vcsr.col_idx.data(), vcsr.nnz*sizeof(int),          cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_val, vcsr.values.data(),  vcsr.nnz*sizeof(double),       cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_x,   vx.data(),           vcsr.nrows*sizeof(double),     cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_y, 0, vcsr.nrows * sizeof(double)));

        int grid = (vcsr.nrows + SPMV_BLOCK_SIZE - 1) / SPMV_BLOCK_SIZE;
        spmv_csr_kernel<<<grid, SPMV_BLOCK_SIZE>>>(d_rp, d_ci, d_val, d_x, d_y, vcsr.nrows);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> res(vcsr.nrows);
        CUDA_CHECK(cudaMemcpy(res.data(), d_y, vcsr.nrows*sizeof(double), cudaMemcpyDeviceToHost));
        cudaFree(d_rp); cudaFree(d_ci); cudaFree(d_val); cudaFree(d_x); cudaFree(d_y);

        double max_err = 0.0;
        bool ok = spmv_verify(res.data(), ref.data(), vcsr.nrows, SPMV_CORRECT_TOL, &max_err);
        std::printf("VERIFY abstraction=native matrix=laplacian_2d N=%d max_rel_err=%.2e %s\n",
                    vcsr.nrows, max_err, ok ? "PASS" : "FAIL");
        if (!ok) {
            std::fprintf(stderr, "[E4 verify] native FAILED — aborting.\n");
            return 1;
        }
        std::fprintf(stderr, "[E4 verify] native PASS — proceeding to timed measurement.\n");
    }

    // ── Build matrix ──────────────────────────────────────────────────────────
    SpmvCSR csr = build_matrix(mtype, N);
    int  nrows = csr.nrows;
    long nnz   = csr.nnz;
    std::vector<double> x = make_x_vector(nrows);

    std::printf("# abstraction=native matrix=%s N=%d nnz=%ld warmup_max=%d reps=%d platform=%s\n",
                matrix_type_str(mtype), nrows, nnz, warmup, reps, platform.c_str());

    // ── Allocate device memory ────────────────────────────────────────────────
    int*    d_rp;  int*    d_ci;
    double* d_val; double* d_x;  double* d_y;
    if (cudaMalloc(&d_rp,  (nrows + 1) * sizeof(int))   != cudaSuccess ||
        cudaMalloc(&d_ci,  nnz * sizeof(int))             != cudaSuccess ||
        cudaMalloc(&d_val, nnz * sizeof(double))          != cudaSuccess ||
        cudaMalloc(&d_x,   nrows * sizeof(double))        != cudaSuccess ||
        cudaMalloc(&d_y,   nrows * sizeof(double))        != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed for N=%d nnz=%ld\n", nrows, nnz);
        return 1;
    }
    CUDA_CHECK(cudaMemcpy(d_rp,  csr.row_ptr.data(), (nrows+1)*sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ci,  csr.col_idx.data(), nnz*sizeof(int),           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_val, csr.values.data(),  nnz*sizeof(double),        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x,   x.data(),           nrows*sizeof(double),      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_y, 0, nrows * sizeof(double)));

    int grid = (nrows + SPMV_BLOCK_SIZE - 1) / SPMV_BLOCK_SIZE;
    const char* mstr = matrix_type_str(mtype);

    auto run_once = [&]() {
        spmv_csr_kernel<<<grid, SPMV_BLOCK_SIZE>>>(d_rp, d_ci, d_val, d_x, d_y, nrows);
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    // ── Adaptive warmup (D7) ──────────────────────────────────────────────────
    int warmup_iters = spmv_adaptive_warmup(run_once, SPMV_WARMUP_MIN, warmup);
    std::fprintf(stderr, "[E4] native: adaptive warmup done in %d iterations\n", warmup_iters);

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

    // Also print effective bandwidth for roofline context
    if (stats.n_clean > 0) {
        double time_s  = 2.0 * nnz / (stats.median_gflops * 1.0e9);
        double gbs     = spmv_gbs_effective(nrows, nnz, time_s);
        double ai      = spmv_ai(nrows, nnz);
        std::printf("SPMV_ROOFLINE n=%d nnz=%ld matrix=%s median_gbs=%.4f ai=%.4f\n",
                    nrows, nnz, mstr, gbs, ai);
        std::fflush(stdout);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    cudaFree(d_rp); cudaFree(d_ci); cudaFree(d_val);
    cudaFree(d_x);  cudaFree(d_y);
    return 0;
}
