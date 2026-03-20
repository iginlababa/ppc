// kernel_stencil_hip.cpp — E3 3D Stencil: native HIP implementation (AMD).
//
// HIP port of kernel_stencil_cuda.cu. Mechanical translation:
//   cuda_runtime.h              → hip/hip_runtime.h
//   CUDA_CHECK / cudaGetError*  → HIP_CHECK / hipGetErrorString
//   cudaMalloc/Free/Memcpy/etc  → hipMalloc/Free/Memcpy/etc
//   stencil7pt_kernel<<<g,b>>>  → unchanged (HIP uses same <<<>>> syntax)
//
// [D4-HIP] Block=(32,4,2)=256 threads. threadIdx.x → ix (innermost) →
//   64 consecutive doubles per warp → coalesced on AMD wavefront (64 threads).
//   AMD wavefront = 2 CUDA warps; this 32-wide x-block fills half a wavefront
//   per row but keeps full coalescing. Same block as CUDA baseline.
// [D7-HIP] Adaptive warmup identical to CUDA version.
// Output: abstraction=native (this IS the native baseline on AMD platforms).
// ─────────────────────────────────────────────────────────────────────────────

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "stencil_common.h"

// ── HIP error helper ──────────────────────────────────────────────────────────
#define HIP_CHECK(call)                                                         \
    do {                                                                        \
        hipError_t _e = (call);                                                 \
        if (_e != hipSuccess) {                                                 \
            std::fprintf(stderr, "HIP error %s:%d: %s\n",                      \
                         __FILE__, __LINE__, hipGetErrorString(_e));            \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// ── 7-point Jacobi stencil kernel ────────────────────────────────────────────
// Row-major layout: in[iz*N*N + iy*N + ix].
// threadIdx.x → ix: coalesced access on both NVIDIA warp and AMD wavefront.
__global__ void stencil7pt_kernel(const double* __restrict__ in,
                                   double*       __restrict__ out,
                                   int N, double c0, double c1)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    const int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix < 1 || ix >= N - 1 ||
        iy < 1 || iy >= N - 1 ||
        iz < 1 || iz >= N - 1) return;

    const int NN  = N * N;
    const int ctr = iz * NN + iy * N + ix;

    out[ctr] = c0 * in[ctr]
        + c1 * (in[ctr - 1]   + in[ctr + 1]
              + in[ctr - N]    + in[ctr + N]
              + in[ctr - NN]   + in[ctr + NN]);
}

// ── Helpers ───────────────────────────────────────────────────────────────────
static void init_grid(double* h, int N) {
    for (int iz = 0; iz < N; iz++)
        for (int iy = 0; iy < N; iy++)
            for (int ix = 0; ix < N; ix++) {
                double x = static_cast<double>(ix) / N;
                double y = static_cast<double>(iy) / N;
                double z = static_cast<double>(iz) / N;
                h[iz * N * N + iy * N + ix] =
                    std::sin(x) + std::cos(y) + std::sin(z + 0.5);
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
        const int    Nv = 16;
        const size_t bv = static_cast<size_t>(Nv) * Nv * Nv * sizeof(double);

        std::vector<double> hIn(Nv * Nv * Nv);
        std::vector<double> hOut(Nv * Nv * Nv, 0.0);
        std::vector<double> hRef(Nv * Nv * Nv, 0.0);
        init_grid(hIn.data(), Nv);
        stencil_cpu_ref(Nv, STENCIL_C0, STENCIL_C1, hIn.data(), hRef.data());

        double *dIn = nullptr, *dOut = nullptr;
        HIP_CHECK(hipMalloc(&dIn,  bv));
        HIP_CHECK(hipMalloc(&dOut, bv));
        HIP_CHECK(hipMemcpy(dIn,  hIn.data(), bv, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(dOut, 0, bv));

        const dim3 vblk(STENCIL_BLOCK_X, STENCIL_BLOCK_Y, STENCIL_BLOCK_Z);
        const dim3 vgrd((Nv + STENCIL_BLOCK_X - 1) / STENCIL_BLOCK_X,
                        (Nv + STENCIL_BLOCK_Y - 1) / STENCIL_BLOCK_Y,
                        (Nv + STENCIL_BLOCK_Z - 1) / STENCIL_BLOCK_Z);
        stencil7pt_kernel<<<vgrd, vblk>>>(dIn, dOut, Nv, STENCIL_C0, STENCIL_C1);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(hOut.data(), dOut, bv, hipMemcpyDeviceToHost));
        HIP_CHECK(hipFree(dIn));
        HIP_CHECK(hipFree(dOut));

        double max_err = 0.0;
        bool ok = stencil_verify(hOut.data(), hRef.data(), Nv,
                                 STENCIL_CORRECT_TOL, &max_err);
        std::printf("VERIFY abstraction=native N=%d max_rel_err=%.2e %s\n",
                    Nv, max_err, ok ? "PASS" : "FAIL");
        if (!ok) {
            std::fprintf(stderr, "[E3 verify] native-hip FAILED — aborting.\n");
            return 1;
        }
        std::fprintf(stderr,
                     "[E3 verify] native-hip PASS — proceeding to timing.\n");
    }

    // ── Allocate ──────────────────────────────────────────────────────────────
    const size_t bytes = static_cast<size_t>(N) * N * N * sizeof(double);
    std::vector<double> hIn(static_cast<size_t>(N) * N * N);
    init_grid(hIn.data(), N);

    double *dIn = nullptr, *dOut = nullptr;
    if (hipMalloc(&dIn,  bytes) != hipSuccess ||
        hipMalloc(&dOut, bytes) != hipSuccess) {
        std::fprintf(stderr, "hipMalloc failed for N=%d\n", N);
        return 1;
    }
    HIP_CHECK(hipMemcpy(dIn, hIn.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dOut, 0, bytes));

    const dim3 block(STENCIL_BLOCK_X, STENCIL_BLOCK_Y, STENCIL_BLOCK_Z);
    const dim3 grid((N + STENCIL_BLOCK_X - 1) / STENCIL_BLOCK_X,
                    (N + STENCIL_BLOCK_Y - 1) / STENCIL_BLOCK_Y,
                    (N + STENCIL_BLOCK_Z - 1) / STENCIL_BLOCK_Z);

    auto run_once = [&]() {
        stencil7pt_kernel<<<grid, block>>>(dIn, dOut, N, STENCIL_C0, STENCIL_C1);
        HIP_CHECK(hipDeviceSynchronize());
        std::swap(dIn, dOut);
    };

    std::printf("# abstraction=native N=%d warmup_max=%d reps=%d platform=%s\n",
                N, warmup, reps, platform.c_str());

    // ── Adaptive warmup (D7) ──────────────────────────────────────────────────
    int warmup_iters = adaptive_warmup(run_once, STENCIL_WARMUP_MIN, warmup);
    std::fprintf(stderr,
                 "[E3] native-hip: adaptive warmup done in %d iterations\n",
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

    if (stats.n_clean > 0) {
        double gflops = stats.median_gbs *
                        static_cast<double>(STENCIL_FLOP_PER_CELL) /
                        static_cast<double>(STENCIL_BYTES_PER_CELL);
        std::printf("STENCIL_GFLOPS n=%d median_gflops=%.4f ai=%.4f\n",
                    N, gflops,
                    static_cast<double>(STENCIL_FLOP_PER_CELL) /
                    static_cast<double>(STENCIL_BYTES_PER_CELL));
        std::fflush(stdout);
    }

    hipFree(dIn);
    hipFree(dOut);
    return 0;
}
