// kernel_dgemm_hip.cpp — E2 DGEMM: native HIP tiled + rocBLAS ceiling reference.
//
// HIP port of kernel_dgemm_cuda.cu.  Kernel body (__global__, __shared__,
// __syncthreads, blockIdx, threadIdx, <<<grid,block>>>) compiles unchanged
// under hipcc.  Only runtime/library API calls are substituted:
//   cuda* → hip*,  cublas* → rocblas_*,  CUBLAS_OP_N → rocblas_operation_none
//
// ── E2 DESIGN DECISIONS ───────────────────────────────────────────────────────
// [D2] 32×32 tiled DGEMM: same tile and thread mapping as CUDA baseline.
// [D5] --mode rocblas: rocBLAS ceiling reference (native_rocblas in CSV).
//      Row-major A,B,C → rocBLAS trick: C^T = B^T * A^T in column-major
//      → rocblas_dgemm(OP_N, OP_N, N,N,N, &alpha, d_B, N, d_A, N, &beta, d_C, N)
// [D2] --verify: correctness at N=128 against CPU reference.
// ─────────────────────────────────────────────────────────────────────────────

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include "dgemm_common.h"

// ── Error helpers ─────────────────────────────────────────────────────────────
#define HIP_CHECK(call)                                                         \
    do {                                                                        \
        hipError_t _e = (call);                                                 \
        if (_e != hipSuccess) {                                                 \
            std::fprintf(stderr, "HIP error %s:%d: %s\n",                      \
                         __FILE__, __LINE__, hipGetErrorString(_e));            \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

#define ROCBLAS_CHECK(call)                                                     \
    do {                                                                        \
        rocblas_status _s = (call);                                             \
        if (_s != rocblas_status_success) {                                     \
            std::fprintf(stderr, "rocBLAS error %s:%d: status=%d\n",           \
                         __FILE__, __LINE__, static_cast<int>(_s));             \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// ── 32×32 tiled DGEMM kernel ──────────────────────────────────────────────────
__global__ void dgemm_tiled_kernel(int N, double alpha,
                                    const double* __restrict__ A,
                                    const double* __restrict__ B,
                                    double beta,
                                    double* __restrict__ C) {
    __shared__ double sA[DGEMM_TILE][DGEMM_TILE];
    __shared__ double sB[DGEMM_TILE][DGEMM_TILE];

    const int row = blockIdx.y * DGEMM_TILE + threadIdx.y;
    const int col = blockIdx.x * DGEMM_TILE + threadIdx.x;
    double acc = 0.0;

    const int num_tiles = (N + DGEMM_TILE - 1) / DGEMM_TILE;
    for (int t = 0; t < num_tiles; t++) {
        const int k_a = t * DGEMM_TILE + threadIdx.x;
        sA[threadIdx.y][threadIdx.x] = (row < N && k_a < N)
            ? A[row * N + k_a] : 0.0;

        const int k_b = t * DGEMM_TILE + threadIdx.y;
        sB[threadIdx.y][threadIdx.x] = (k_b < N && col < N)
            ? B[k_b * N + col] : 0.0;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < DGEMM_TILE; k++)
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

// ── Helpers ───────────────────────────────────────────────────────────────────
static void init_matrix(double* h, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            h[i * N + j] = 1.0 / static_cast<double>(i + j + 2);
}

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  --n <N>          Matrix dimension (default: %d)\n"
        "  --warmup <W>     Warmup iterations (default: %d)\n"
        "  --reps <R>       Timed iterations (default: %d)\n"
        "  --platform <P>   Platform tag (default: unknown)\n"
        "  --mode <M>       native | rocblas (default: native)\n"
        "  --verify         Correctness check at N=128 then exit\n",
        prog, DGEMM_N_LARGE, DGEMM_WARMUP_DEFAULT, DGEMM_REPS_DEFAULT);
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    int         N        = DGEMM_N_LARGE;
    int         warmup   = DGEMM_WARMUP_DEFAULT;
    int         reps     = DGEMM_REPS_DEFAULT;
    std::string platform = "unknown";
    std::string mode     = "native";
    bool        verify   = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--n"        && i+1 < argc) { N        = std::stoi(argv[++i]); }
        else if (a == "--warmup"   && i+1 < argc) { warmup   = std::stoi(argv[++i]); }
        else if (a == "--reps"     && i+1 < argc) { reps     = std::stoi(argv[++i]); }
        else if (a == "--platform" && i+1 < argc) { platform = argv[++i]; }
        else if (a == "--mode"     && i+1 < argc) { mode     = argv[++i]; }
        else if (a == "--verify")                  { verify   = true; }
        else { print_usage(argv[0]); return 1; }
    }

    if (mode != "native" && mode != "rocblas") {
        std::fprintf(stderr, "ERROR: --mode must be native or rocblas\n");
        return 1;
    }

    // ── rocBLAS handle ────────────────────────────────────────────────────────
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    // ── Correctness check ─────────────────────────────────────────────────────
    if (verify) {
        const int Nv = 128;
        const size_t bytes_v = static_cast<size_t>(Nv) * Nv * sizeof(double);

        double *hA = new double[Nv * Nv];
        double *hB = new double[Nv * Nv];
        double *hC = new double[Nv * Nv];
        double *hR = new double[Nv * Nv];
        init_matrix(hA, Nv);
        init_matrix(hB, Nv);
        std::fill(hC, hC + Nv * Nv, 0.0);
        dgemm_cpu_ref(Nv, DGEMM_ALPHA, hA, hB, DGEMM_BETA, hR);

        double *dA, *dB, *dC;
        HIP_CHECK(hipMalloc(&dA, bytes_v));
        HIP_CHECK(hipMalloc(&dB, bytes_v));
        HIP_CHECK(hipMalloc(&dC, bytes_v));
        HIP_CHECK(hipMemcpy(dA, hA, bytes_v, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dB, hB, bytes_v, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dC, hC, bytes_v, hipMemcpyHostToDevice));

        double alpha = DGEMM_ALPHA, beta = DGEMM_BETA;
        if (mode == "rocblas") {
            // Row-major trick: C^T = B^T * A^T in column-major
            ROCBLAS_CHECK(rocblas_dgemm(handle,
                rocblas_operation_none, rocblas_operation_none, Nv, Nv, Nv,
                &alpha, dB, Nv, dA, Nv, &beta, dC, Nv));
        } else {
            dim3 blk(DGEMM_TILE, DGEMM_TILE);
            dim3 grd((Nv + DGEMM_TILE - 1) / DGEMM_TILE,
                     (Nv + DGEMM_TILE - 1) / DGEMM_TILE);
            dgemm_tiled_kernel<<<grd, blk>>>(Nv, alpha, dA, dB, beta, dC);
        }
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(hC, dC, bytes_v, hipMemcpyDeviceToHost));

        double max_err = 0.0;
        bool ok = dgemm_verify(hC, hR, Nv, DGEMM_CORRECT_TOL, &max_err);
        std::printf("VERIFY mode=%s N=%d max_rel_err=%.2e %s\n",
                    mode.c_str(), Nv, max_err, ok ? "PASS" : "FAIL");

        HIP_CHECK(hipFree(dA)); HIP_CHECK(hipFree(dB)); HIP_CHECK(hipFree(dC));
        delete[] hA; delete[] hB; delete[] hC; delete[] hR;
        if (!ok) {
            std::fprintf(stderr, "[E2 verify] FAILED — aborting before timing.\n");
            rocblas_destroy_handle(handle);
            return 1;
        }
        std::fprintf(stderr, "[E2 verify] PASS — proceeding to timed measurement.\n");
    }

    // ── Allocate ──────────────────────────────────────────────────────────────
    const size_t bytes = static_cast<size_t>(N) * N * sizeof(double);
    double *hA = new double[static_cast<size_t>(N) * N];
    double *hB = new double[static_cast<size_t>(N) * N];
    double *hC = new double[static_cast<size_t>(N) * N];
    init_matrix(hA, N);
    init_matrix(hB, N);
    std::fill(hC, hC + static_cast<size_t>(N) * N, 0.0);

    double *dA, *dB, *dC;
    HIP_CHECK(hipMalloc(&dA, bytes));
    HIP_CHECK(hipMalloc(&dB, bytes));
    HIP_CHECK(hipMalloc(&dC, bytes));
    HIP_CHECK(hipMemcpy(dA, hA, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dC, hC, bytes, hipMemcpyHostToDevice));

    const dim3 block(DGEMM_TILE, DGEMM_TILE);
    const dim3 grid((N + DGEMM_TILE - 1) / DGEMM_TILE,
                    (N + DGEMM_TILE - 1) / DGEMM_TILE);
    double alpha = DGEMM_ALPHA, beta = DGEMM_BETA;

    auto run_once = [&]() {
        if (mode == "rocblas") {
            ROCBLAS_CHECK(rocblas_dgemm(handle,
                rocblas_operation_none, rocblas_operation_none, N, N, N,
                &alpha, dB, N, dA, N, &beta, dC, N));
        } else {
            dgemm_tiled_kernel<<<grid, block>>>(N, alpha, dA, dB, beta, dC);
        }
        HIP_CHECK(hipDeviceSynchronize());
    };

    // ── Warmup ────────────────────────────────────────────────────────────────
    std::printf("# mode=%s N=%d warmup=%d reps=%d platform=%s\n",
                mode.c_str(), N, warmup, reps, platform.c_str());
    for (int i = 0; i < warmup; i++) run_once();

    // ── Timed runs ────────────────────────────────────────────────────────────
    std::vector<double> gflops_vec;
    gflops_vec.reserve(reps);

    for (int r = 1; r <= reps; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        run_once();
        auto t1 = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double gf = dgemm_gflops(N, time_ms / 1000.0);
        gflops_vec.push_back(gf);
        dgemm_print_run(r, N, time_ms, gf);
    }

    // ── hw_state_verified ─────────────────────────────────────────────────────
    auto flags = compute_hw_state(gflops_vec);
    for (int r = 0; r < reps; r++)
        dgemm_print_hw_state(r + 1, flags[r]);

    // ── Summary ───────────────────────────────────────────────────────────────
    DgemmStats stats = compute_dgemm_stats(gflops_vec, flags);
    dgemm_print_summary(N, stats);

    // ── Cleanup ───────────────────────────────────────────────────────────────
    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));
    rocblas_destroy_handle(handle);
    delete[] hA; delete[] hB; delete[] hC;
    return 0;
}
