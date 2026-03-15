// kernel_stencil_cuda.cu — E3 3D Stencil: native CUDA implementation.
//
// ── E3 DESIGN DECISIONS ───────────────────────────────────────────────────────
// [D4-CUDA] Block=(32,4,2)=256 threads. threadIdx.x → ix (innermost in row-major
//   layout) → 32 consecutive threads read 32 consecutive doubles → coalesced.
//   No shared memory: for a memory-bound AI≈0.2 FLOP/byte stencil, shmem does
//   not improve throughput — the bottleneck is DRAM bandwidth, not L1 hits.
//   Adding shmem halos would reduce BW reuse benefit while increasing register
//   pressure. This is the scientifically correct native baseline for E3.
// [D7-CUDA] Adaptive warmup: loops until CV < 2% over last 10 timings.
//   Primary output metric: GB/s. GFLOP/s reported in STENCIL_SUMMARY for roofline.
// ─────────────────────────────────────────────────────────────────────────────

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "stencil_common.h"

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

// ── 7-point Jacobi stencil kernel ────────────────────────────────────────────
// Row-major layout: in[iz*N*N + iy*N + ix].
// threadIdx.x → ix: 32 consecutive threads access 32 consecutive doubles.
__global__ void stencil7pt_kernel(const double* __restrict__ in,
                                   double*       __restrict__ out,
                                   int N, double c0, double c1)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    const int iz = blockIdx.z * blockDim.z + threadIdx.z;

    // Guard: skip boundary cells (ghost layer, not computed)
    if (ix < 1 || ix >= N - 1 ||
        iy < 1 || iy >= N - 1 ||
        iz < 1 || iz >= N - 1) return;

    const int NN    = N * N;
    const int ctr   = iz * NN + iy * N + ix;

    out[ctr] = c0 * in[ctr]
        + c1 * (in[ctr - 1]    + in[ctr + 1]
              + in[ctr - N]     + in[ctr + N]
              + in[ctr - NN]    + in[ctr + NN]);
}

// ── Helpers ───────────────────────────────────────────────────────────────────
static void init_grid(double* h, int N) {
    // Smooth sinusoidal initial condition — avoids cancellation in verify
    for (int iz = 0; iz < N; iz++)
        for (int iy = 0; iy < N; iy++)
            for (int ix = 0; ix < N; ix++) {
                double x = static_cast<double>(ix) / N;
                double y = static_cast<double>(iy) / N;
                double z = static_cast<double>(iz) / N;
                h[iz * N * N + iy * N + ix] = std::sin(x) + std::cos(y) + std::sin(z + 0.5);
            }
}

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  --n <N>          Grid side length N (NxNxN, default: %d)\n"
        "  --warmup <W>     Max adaptive warmup iterations (default: %d)\n"
        "  --reps <R>       Timed iterations (default: %d)\n"
        "  --platform <P>   Platform tag (default: unknown)\n"
        "  --verify         Correctness check at N=16 then proceed to timing\n",
        prog, STENCIL_N_LARGE, STENCIL_WARMUP_MAX, STENCIL_REPS_DEFAULT);
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    int         N        = STENCIL_N_LARGE;
    int         warmup   = STENCIL_WARMUP_MAX;
    int         reps     = STENCIL_REPS_DEFAULT;
    std::string platform = "unknown";
    bool        verify   = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--n"        && i+1 < argc) { N        = std::stoi(argv[++i]); }
        else if (a == "--warmup"   && i+1 < argc) { warmup   = std::stoi(argv[++i]); }
        else if (a == "--reps"     && i+1 < argc) { reps     = std::stoi(argv[++i]); }
        else if (a == "--platform" && i+1 < argc) { platform = argv[++i]; }
        else if (a == "--verify")                  { verify   = true; }
        else { print_usage(argv[0]); return 1; }
    }

    // ── Correctness check ─────────────────────────────────────────────────────
    if (verify) {
        const int Nv = 16;
        const size_t bv = static_cast<size_t>(Nv) * Nv * Nv * sizeof(double);

        std::vector<double> hIn(Nv * Nv * Nv);
        std::vector<double> hOut(Nv * Nv * Nv, 0.0);
        std::vector<double> hRef(Nv * Nv * Nv, 0.0);
        init_grid(hIn.data(), Nv);
        stencil_cpu_ref(Nv, STENCIL_C0, STENCIL_C1, hIn.data(), hRef.data());

        double *dIn = nullptr, *dOut = nullptr;
        CUDA_CHECK(cudaMalloc(&dIn,  bv));
        CUDA_CHECK(cudaMalloc(&dOut, bv));
        CUDA_CHECK(cudaMemcpy(dIn,  hIn.data(),  bv, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dOut, 0, bv));

        const dim3 vblk(STENCIL_BLOCK_X, STENCIL_BLOCK_Y, STENCIL_BLOCK_Z);
        const dim3 vgrd((Nv + STENCIL_BLOCK_X - 1) / STENCIL_BLOCK_X,
                        (Nv + STENCIL_BLOCK_Y - 1) / STENCIL_BLOCK_Y,
                        (Nv + STENCIL_BLOCK_Z - 1) / STENCIL_BLOCK_Z);
        stencil7pt_kernel<<<vgrd, vblk>>>(dIn, dOut, Nv, STENCIL_C0, STENCIL_C1);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, bv, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dIn));
        CUDA_CHECK(cudaFree(dOut));

        double max_err = 0.0;
        bool ok = stencil_verify(hOut.data(), hRef.data(), Nv, STENCIL_CORRECT_TOL, &max_err);
        std::printf("VERIFY abstraction=native N=%d max_rel_err=%.2e %s\n",
                    Nv, max_err, ok ? "PASS" : "FAIL");
        if (!ok) {
            std::fprintf(stderr, "[E3 verify] native FAILED — aborting before timing.\n");
            return 1;
        }
        std::fprintf(stderr, "[E3 verify] native PASS — proceeding to timed measurement.\n");
    }

    // ── Allocate ──────────────────────────────────────────────────────────────
    const size_t bytes = static_cast<size_t>(N) * N * N * sizeof(double);
    std::vector<double> hIn(static_cast<size_t>(N) * N * N);
    init_grid(hIn.data(), N);

    double *dIn = nullptr, *dOut = nullptr;
    if (cudaMalloc(&dIn,  bytes) != cudaSuccess ||
        cudaMalloc(&dOut, bytes) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed for N=%d\n", N);
        return 1;
    }
    CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dOut, 0, bytes));

    const dim3 block(STENCIL_BLOCK_X, STENCIL_BLOCK_Y, STENCIL_BLOCK_Z);
    const dim3 grid((N + STENCIL_BLOCK_X - 1) / STENCIL_BLOCK_X,
                    (N + STENCIL_BLOCK_Y - 1) / STENCIL_BLOCK_Y,
                    (N + STENCIL_BLOCK_Z - 1) / STENCIL_BLOCK_Z);

    auto run_once = [&]() {
        stencil7pt_kernel<<<grid, block>>>(dIn, dOut, N, STENCIL_C0, STENCIL_C1);
        CUDA_CHECK(cudaDeviceSynchronize());
        // Swap pointers: each stencil step feeds its output as the next input
        std::swap(dIn, dOut);
    };

    std::printf("# abstraction=native N=%d warmup_max=%d reps=%d platform=%s\n",
                N, warmup, reps, platform.c_str());

    // ── Adaptive warmup (D7) ──────────────────────────────────────────────────
    int warmup_iters = adaptive_warmup(run_once, STENCIL_WARMUP_MIN, warmup);
    std::fprintf(stderr, "[E3] native: adaptive warmup done in %d iterations\n",
                 warmup_iters);

    // ── Timed runs ────────────────────────────────────────────────────────────
    std::vector<double> gbs_vec;
    gbs_vec.reserve(reps);

    for (int r = 1; r <= reps; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        run_once();
        auto t1 = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double gbs = stencil_gbs(N, time_ms / 1000.0);
        gbs_vec.push_back(gbs);
        stencil_print_run(r, N, time_ms, gbs);
    }

    // ── hw_state_verified ─────────────────────────────────────────────────────
    auto flags = compute_hw_state(gbs_vec);
    for (int r = 0; r < reps; r++)
        stencil_print_hw_state(r + 1, flags[r]);

    // ── Summary ───────────────────────────────────────────────────────────────
    StencilStats stats = compute_stencil_stats(gbs_vec, flags);
    stencil_print_summary(N, stats, warmup_iters);

    // Also print GFLOP/s for roofline reference
    if (stats.n_clean > 0) {
        // Approximate from median GB/s
        double gflops = stats.median_gbs *
                        static_cast<double>(STENCIL_FLOP_PER_CELL) /
                        static_cast<double>(STENCIL_BYTES_PER_CELL);
        std::printf("STENCIL_GFLOPS n=%d median_gflops=%.4f ai=%.4f\n",
                    N, gflops,
                    static_cast<double>(STENCIL_FLOP_PER_CELL) /
                    static_cast<double>(STENCIL_BYTES_PER_CELL));
        std::fflush(stdout);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    cudaFree(dIn);
    cudaFree(dOut);
    return 0;
}
