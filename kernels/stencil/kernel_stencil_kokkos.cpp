// kernel_stencil_kokkos.cpp — E3 3D Stencil: Kokkos MDRangePolicy<Rank<3>>.
//
// ── E3 DESIGN DECISIONS ───────────────────────────────────────────────────────
// [D4-Kokkos] MDRangePolicy<Rank<3>> tiling hint controls blockDim layout.
//   Tile maps as {iz_tile, iy_tile, ix_tile} → {blockDim.z, blockDim.y, blockDim.x}.
//   NVIDIA (warp-32):  {2,4,32}  → blockDim=(32,4,2)=256 threads; ix-tile=32=1 warp.
//   AMD    (wave-64):  {1,8,64}  → blockDim=(64,8,1)=512 threads; ix-tile=64=1 wavefront.
//   The innermost tile extent (ix) must match the hardware SIMT width so that
//   all threads in one warp/wavefront access consecutive ix memory → coalesced.
//   A tile of 32 on AMD wastes half each wavefront; {1,8,64} fixes this.
//   Memory layout: LayoutRight (row-major) matches CUDA and RAJA baselines.
// ─────────────────────────────────────────────────────────────────────────────

#include <Kokkos_Core.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "gpu_compat.h"
#include "stencil_common.h"

using ExecSpace = Kokkos::DefaultExecutionSpace;
using ViewT     = Kokkos::View<double***, Kokkos::LayoutRight, ExecSpace,
                               Kokkos::MemoryUnmanaged>;
using ConstViewT = Kokkos::View<const double***, Kokkos::LayoutRight, ExecSpace,
                                Kokkos::MemoryUnmanaged>;

void run_stencil_kokkos(int N, double c0, double c1,
                         const double* d_in, double* d_out) {
    ConstViewT in_v(d_in,  N, N, N);
    ViewT      out_v(d_out, N, N, N);

    using MDPol = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>;

#ifdef KOKKOS_ENABLE_HIP
    constexpr int T_Z = 1, T_Y = 8, T_X = 64;  // wave64: ix-tile=64=1 wavefront
#else
    constexpr int T_Z = 2, T_Y = 4, T_X = 32;  // warp32: ix-tile=32=1 warp
#endif

    Kokkos::parallel_for(
        "stencil7pt_kokkos",
        MDPol({1, 1, 1}, {N-1, N-1, N-1}, {T_Z, T_Y, T_X}),
        KOKKOS_LAMBDA(int iz, int iy, int ix) {
            out_v(iz, iy, ix) = c0 * in_v(iz, iy, ix)
                + c1 * (in_v(iz, iy, ix-1) + in_v(iz, iy, ix+1)
                      + in_v(iz, iy-1, ix)  + in_v(iz, iy+1, ix)
                      + in_v(iz-1, iy, ix)  + in_v(iz+1, iy, ix));
        });

    Kokkos::fence();
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

    Kokkos::initialize(argc, argv);
    {
        // ── Correctness check ─────────────────────────────────────────────────
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

            run_stencil_kokkos(Nv, STENCIL_C0, STENCIL_C1, dvIn, dvOut);

            gpuMemcpy(hOut.data(), dvOut, bv, gpuMemcpyDeviceToHost);
            gpuFree(dvIn); gpuFree(dvOut);

            double max_err = 0.0;
            bool ok = stencil_verify(hOut.data(), hRef.data(), Nv, STENCIL_CORRECT_TOL, &max_err);
            std::printf("VERIFY abstraction=kokkos N=%d max_rel_err=%.2e %s\n",
                        Nv, max_err, ok ? "PASS" : "FAIL");
            if (!ok) {
                std::fprintf(stderr, "[E3 verify] kokkos FAILED — aborting before timing.\n");
                Kokkos::finalize(); return 1;
            }
            std::fprintf(stderr, "[E3 verify] kokkos PASS — proceeding to timed measurement.\n");
        }

        // ── Allocate ──────────────────────────────────────────────────────────
        const size_t bytes = static_cast<size_t>(N) * N * N * sizeof(double);
        std::vector<double> hIn(static_cast<size_t>(N) * N * N);
        init_grid(hIn.data(), N);

        double *dIn = nullptr, *dOut = nullptr;
        if (gpuMalloc(&dIn,  bytes) != gpuSuccess ||
            gpuMalloc(&dOut, bytes) != gpuSuccess) {
            std::fprintf(stderr, "gpuMalloc failed for N=%d\n", N);
            Kokkos::finalize(); return 1;
        }
        gpuMemcpy(dIn, hIn.data(), bytes, gpuMemcpyHostToDevice);
        gpuMemset(dOut, 0, bytes);

        auto run_once = [&]() {
            run_stencil_kokkos(N, STENCIL_C0, STENCIL_C1, dIn, dOut);
            std::swap(dIn, dOut);
        };

        std::printf("# abstraction=kokkos N=%d warmup_max=%d reps=%d platform=%s\n",
                    N, warmup, reps, platform.c_str());

        int warmup_iters = adaptive_warmup(run_once, STENCIL_WARMUP_MIN, warmup);
        std::fprintf(stderr, "[E3] kokkos: adaptive warmup done in %d iterations\n",
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
    }
    Kokkos::finalize();
    return 0;
}
