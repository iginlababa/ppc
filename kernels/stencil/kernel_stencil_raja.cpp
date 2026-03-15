// kernel_stencil_raja.cpp — E3 3D Stencil: RAJA::kernel 3D loop nest.
//
// ── E3 DESIGN DECISIONS ───────────────────────────────────────────────────────
// [D4-RAJA] RAJA::kernel with explicit 3D CUDA mapping:
//   - Dim 0 (iz) → cuda_block_z_loop
//   - Dim 1 (iy) → cuda_block_y_loop
//   - Dim 2 (ix) → cuda_thread_x_loop (256 threads)
//   ix is the innermost/fastest-varying index in row-major memory → coalesced.
//   This is the idiomatic RAJA::kernel for 3D structured-grid kernels with
//   a natural grid-to-block mapping, unlike the E2 raja_naive forall which
//   had no shmem and was labeled for its API limitation.
//   For E3 (memory-bound), there is no API limitation — RAJA exposes the full
//   3D parallelism efficiently. Label: "raja" (not "raja_naive").
// ─────────────────────────────────────────────────────────────────────────────

#include <RAJA/RAJA.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "stencil_common.h"

// ── RAJA 3D kernel policy ─────────────────────────────────────────────────────
// Segments: (ix, iy, iz) with dims (0,1,2).
// Thread mapping: For<2,ix> → thread_x (innermost, 256 threads per block);
//                 For<1,iy> → block_y; For<0,iz> → block_z.
using StencilPolicy3D = RAJA::KernelPolicy<
    RAJA::statement::CudaKernel<
        RAJA::statement::For<0, RAJA::cuda_block_z_loop,
            RAJA::statement::For<1, RAJA::cuda_block_y_loop,
                RAJA::statement::For<2, RAJA::cuda_thread_x_loop,
                    RAJA::statement::Lambda<0>>>>>>;

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
    RAJA::synchronize<RAJA::cuda_synchronize>();
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
        cudaMalloc(&dvIn,  bv); cudaMalloc(&dvOut, bv);
        cudaMemcpy(dvIn, hIn.data(), bv, cudaMemcpyHostToDevice);
        cudaMemset(dvOut, 0, bv);

        run_stencil_raja(Nv, STENCIL_C0, STENCIL_C1, dvIn, dvOut);

        cudaMemcpy(hOut.data(), dvOut, bv, cudaMemcpyDeviceToHost);
        cudaFree(dvIn); cudaFree(dvOut);

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
    if (cudaMalloc(&dIn,  bytes) != cudaSuccess ||
        cudaMalloc(&dOut, bytes) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed for N=%d\n", N);
        return 1;
    }
    cudaMemcpy(dIn, hIn.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemset(dOut, 0, bytes);

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

    cudaFree(dIn); cudaFree(dOut);
    return 0;
}
