// kernel_stencil_sycl.cpp — E3 3D Stencil: SYCL nd_range<3> implementation.
//
// ── E3 DESIGN DECISIONS ───────────────────────────────────────────────────────
// [D4-SYCL] Work-group (32, 4, 2) = 256 work-items. dim 2 → ix (fastest-varying
//   in row-major) so 32 consecutive work-items in dim 2 map to consecutive
//   memory addresses → coalesced on both NVIDIA (warp) and AMD (wavefront).
//   USM (malloc_device) used rather than buffers — cleaner async control and
//   same pointer-based interface as the CUDA/HIP baselines.
// [D7-SYCL] Adaptive warmup: host-side chrono after queue.wait().
// Compatible with AdaptiveCpp --acpp-targets=hip:gfx942 (AMD MI300X) and
// -fsycl -fsycl-targets=nvptx64-nvidia-cuda (NVIDIA).
// ─────────────────────────────────────────────────────────────────────────────

#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "stencil_common.h"

// ── Run one stencil step (USM, nd_range<3>) ───────────────────────────────────
static void run_stencil_sycl(sycl::queue& q,
                              const double* in, double* out, int N,
                              double c0, double c1)
{
    const int NN = N * N;
    // Pad global range to multiples of local range
    const size_t gx = ((static_cast<size_t>(N) + 31) / 32) * 32;
    const size_t gy = ((static_cast<size_t>(N) +  3) /  4) *  4;
    const size_t gz = ((static_cast<size_t>(N) +  1) /  2) *  2;

    // nd_range: {global_z, global_y, global_x}, {local_z, local_y, local_x}
    // dim 0 = z (slowest), dim 2 = x (fastest → coalesced)
    q.parallel_for(
        sycl::nd_range<3>({gz, gy, gx}, {2, 4, 32}),
        [=](sycl::nd_item<3> item) {
            const int ix = static_cast<int>(item.get_global_id(2));
            const int iy = static_cast<int>(item.get_global_id(1));
            const int iz = static_cast<int>(item.get_global_id(0));

            if (ix < 1 || ix >= N - 1 ||
                iy < 1 || iy >= N - 1 ||
                iz < 1 || iz >= N - 1) return;

            const int ctr = iz * NN + iy * N + ix;
            out[ctr] = c0 * in[ctr]
                + c1 * (in[ctr - 1]    + in[ctr + 1]
                      + in[ctr - N]     + in[ctr + N]
                      + in[ctr - NN]    + in[ctr + NN]);
        });
    q.wait();
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

    // ── SYCL queue ────────────────────────────────────────────────────────────
    sycl::queue q{sycl::gpu_selector_v,
                  [](const sycl::exception_list& el) {
                      for (auto& e : el) {
                          try { std::rethrow_exception(e); }
                          catch (const sycl::exception& ex) {
                              std::fprintf(stderr, "SYCL async exception: %s\n",
                                           ex.what());
                          }
                      }
                  }};
    std::fprintf(stderr, "[E3 sycl] device: %s\n",
                 q.get_device().get_info<sycl::info::device::name>().c_str());

    // ── Correctness check ─────────────────────────────────────────────────────
    if (verify) {
        const int    Nv = 16;
        const size_t bv = static_cast<size_t>(Nv) * Nv * Nv;

        std::vector<double> hIn(bv), hOut(bv, 0.0), hRef(bv, 0.0);
        init_grid(hIn.data(), Nv);
        stencil_cpu_ref(Nv, STENCIL_C0, STENCIL_C1, hIn.data(), hRef.data());

        double* dIn  = sycl::malloc_device<double>(bv, q);
        double* dOut = sycl::malloc_device<double>(bv, q);
        q.memcpy(dIn,  hIn.data(),  bv * sizeof(double)).wait();
        q.memset(dOut, 0, bv * sizeof(double)).wait();

        run_stencil_sycl(q, dIn, dOut, Nv, STENCIL_C0, STENCIL_C1);

        q.memcpy(hOut.data(), dOut, bv * sizeof(double)).wait();
        sycl::free(dIn,  q);
        sycl::free(dOut, q);

        double max_err = 0.0;
        bool ok = stencil_verify(hOut.data(), hRef.data(), Nv,
                                 STENCIL_CORRECT_TOL, &max_err);
        std::printf("VERIFY abstraction=sycl N=%d max_rel_err=%.2e %s\n",
                    Nv, max_err, ok ? "PASS" : "FAIL");
        if (!ok) {
            std::fprintf(stderr, "[E3 verify] sycl FAILED — aborting.\n");
            return 1;
        }
        std::fprintf(stderr,
                     "[E3 verify] sycl PASS — proceeding to timing.\n");
    }

    // ── Allocate ──────────────────────────────────────────────────────────────
    const size_t ncells = static_cast<size_t>(N) * N * N;
    std::vector<double> hIn(ncells);
    init_grid(hIn.data(), N);

    double* dIn  = sycl::malloc_device<double>(ncells, q);
    double* dOut = sycl::malloc_device<double>(ncells, q);
    if (!dIn || !dOut) {
        std::fprintf(stderr, "sycl::malloc_device failed for N=%d\n", N);
        return 1;
    }
    q.memcpy(dIn, hIn.data(), ncells * sizeof(double)).wait();
    q.memset(dOut, 0, ncells * sizeof(double)).wait();

    auto run_once = [&]() {
        run_stencil_sycl(q, dIn, dOut, N, STENCIL_C0, STENCIL_C1);
        std::swap(dIn, dOut);
    };

    std::printf("# abstraction=sycl N=%d warmup_max=%d reps=%d platform=%s\n",
                N, warmup, reps, platform.c_str());

    // ── Adaptive warmup (D7) ──────────────────────────────────────────────────
    int warmup_iters = adaptive_warmup(run_once, STENCIL_WARMUP_MIN, warmup);
    std::fprintf(stderr,
                 "[E3] sycl: adaptive warmup done in %d iterations\n",
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

    sycl::free(dIn,  q);
    sycl::free(dOut, q);
    return 0;
}
