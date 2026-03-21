// kernel_stencil_raja.cpp — E3 3D Stencil: RAJA::kernel 3D loop nest.
//
// ── E3 DESIGN DECISIONS ───────────────────────────────────────────────────────
// [D4-RAJA] RAJA::kernel with explicit 3D GPU mapping:
//   - Dim 0 (iz) → block_z_loop
//   - Dim 1 (iy) → block_y_loop
//   - Dim 2 (ix) → thread_x loop  (see wave64 note below)
//   ix is the innermost/fastest-varying index in row-major memory → coalesced.
//   NVIDIA: HipKernelFixed<1024> + hip_thread_x_loop (blockDim.x auto-sized to 1024).
//           For ix range ~254, threads 254..1023 are idle — acceptable on CUDA warp-32.
//   AMD:    HipKernelFixed<64>   + hip_thread_size_x_loop<64> forces blockDim.x=64=1
//           wavefront. Without this, blockDim.x=1024 → 770/1024 idle threads = ~75%
//           wavefront waste → 35–45% efficiency. With 64, all threads in each
//           wavefront are active → near-native bandwidth.
// ─────────────────────────────────────────────────────────────────────────────

#include <RAJA/RAJA.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "gpu_compat.h"
#include "stencil_common.h"

// ── RAJA 3D kernel policy ─────────────────────────────────────────────────────
// Segments: (iz, iy, ix) with dims (0,1,2).
// Thread mapping: For<2,ix> → thread_x (innermost, coalesced);
//                 For<1,iy> → block_y; For<0,iz> → block_z.
#ifdef __HIP_PLATFORM_AMD__
// HipKernelFixed<64>: blockDim.x=64 = 1 wavefront.
// hip_thread_size_x_loop<64>: each thread handles ix with stride 64 across the range.
// All 64 threads in a wavefront access consecutive ix → coalesced; no idle lanes.
using StencilPolicy3D = RAJA::KernelPolicy<
    RAJA::statement::HipKernelFixed<64,
        RAJA::statement::For<0, RAJA::hip_block_z_loop,
            RAJA::statement::For<1, RAJA::hip_block_y_loop,
                RAJA::statement::For<2, RAJA::hip_thread_size_x_loop<64>,
                    RAJA::statement::Lambda<0>>>>>>;
using SyncPolicy = RAJA::hip_synchronize;
#else
using StencilPolicy3D = RAJA::KernelPolicy<
    RAJA::statement::CudaKernel<
        RAJA::statement::For<0, RAJA::cuda_block_z_loop,
            RAJA::statement::For<1, RAJA::cuda_block_y_loop,
                RAJA::statement::For<2, RAJA::cuda_thread_x_loop,
                    RAJA::statement::Lambda<0>>>>>>;
using SyncPolicy = RAJA::cuda_synchronize;
#endif

void run_stencil_raja(int N, double c0, double c1,
                       const double* d_in, double* d_out) {
    const int NN = N * N;
    RAJA::kernel<StencilPolicy3D>(
        RAJA::make_tuple(
            RAJA::RangeSegment(1, N - 1),  // dim 0: iz
            RAJA::RangeSegment(1, N - 1),  // dim 1: iy
            RAJA::RangeSegment(1, N - 1)), // dim 2: ix
        [=] RAJA_DEVICE (int iz, int iy, int ix) {
            const int ctr = iz * NN + iy * N + ix;
            d_out[ctr] = c0 * d_in[ctr]
                + c1 * (d_in[ctr - 1]    + d_in[ctr + 1]
                      + d_in[ctr - N]     + d_in[ctr + N]
                      + d_in[ctr - NN]    + d_in[ctr + NN]);
        });
    RAJA::synchronize<SyncPolicy>();
}

// ── Helpers ───────────────────────────────────────────────────────────────────
static void init_grid(double* h, int N) {
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
        "  --n <N>          Grid side length N (default: %d)\n"
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
        std::vector<double> hIn(Nv * Nv * Nv), hOut(Nv * Nv * Nv, 0.0),
                            hRef(Nv * Nv * Nv, 0.0);
        init_grid(hIn.data(), Nv);
        stencil_cpu_ref(Nv, STENCIL_C0, STENCIL_C1, hIn.data(), hRef.data());

        double *dvIn = nullptr, *dvOut = nullptr;
        gpuMalloc(&dvIn,  bv); gpuMalloc(&dvOut, bv);
        gpuMemcpy(dvIn, hIn.data(), bv, gpuMemcpyHostToDevice);
        gpuMemset(dvOut, 0, bv);

        run_stencil_raja(Nv, STENCIL_C0, STENCIL_C1, dvIn, dvOut);

        gpuMemcpy(hOut.data(), dvOut, bv, gpuMemcpyDeviceToHost);
        gpuFree(dvIn); gpuFree(dvOut);

        double max_err = 0.0;
        bool ok = stencil_verify(hOut.data(), hRef.data(), Nv, STENCIL_CORRECT_TOL, &max_err);
        std::printf("VERIFY abstraction=raja N=%d max_rel_err=%.2e %s\n",
                    Nv, max_err, ok ? "PASS" : "FAIL");
        if (!ok) {
            std::fprintf(stderr, "[E3 verify] raja FAILED — aborting before timing.\n");
            return 1;
        }
        std::fprintf(stderr, "[E3 verify] raja PASS — proceeding to timed measurement.\n");
    }

    // ── Allocate ──────────────────────────────────────────────────────────────
    const size_t bytes = static_cast<size_t>(N) * N * N * sizeof(double);
    std::vector<double> hIn(static_cast<size_t>(N) * N * N);
    init_grid(hIn.data(), N);

    double *dIn = nullptr, *dOut = nullptr;
    if (gpuMalloc(&dIn,  bytes) != gpuSuccess ||
        gpuMalloc(&dOut, bytes) != gpuSuccess) {
        std::fprintf(stderr, "gpuMalloc failed for N=%d\n", N);
        return 1;
    }
    gpuMemcpy(dIn, hIn.data(), bytes, gpuMemcpyHostToDevice);
    gpuMemset(dOut, 0, bytes);

    auto run_once = [&]() {
        run_stencil_raja(N, STENCIL_C0, STENCIL_C1, dIn, dOut);
        std::swap(dIn, dOut);
    };

    std::printf("# abstraction=raja N=%d warmup_max=%d reps=%d platform=%s\n",
                N, warmup, reps, platform.c_str());

    int warmup_iters = adaptive_warmup(run_once, STENCIL_WARMUP_MIN, warmup);
    std::fprintf(stderr, "[E3] raja: adaptive warmup done in %d iterations\n",
                 warmup_iters);

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

    auto flags = compute_hw_state(gbs_vec);
    for (int r = 0; r < reps; r++)
        stencil_print_hw_state(r + 1, flags[r]);

    StencilStats stats = compute_stencil_stats(gbs_vec, flags);
    stencil_print_summary(N, stats, warmup_iters);

    gpuFree(dIn); gpuFree(dOut);
    return 0;
}
