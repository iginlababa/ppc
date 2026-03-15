// kernel_dgemm_sycl.cpp — E2 DGEMM: SYCL nd_range tiled kernel.
//
// ── E2 DESIGN DECISIONS ───────────────────────────────────────────────────────
// [D2-SYCL] Functionally identical to the CUDA tiled kernel: nd_range<2> with
//   local_accessor<double,2> replacing __shared__. local_range = {TILE, TILE}.
//   Compiled with -fsycl; SKIP at build time if no SYCL compiler found.
// [D3] alpha=1.0, beta=0.0.  Row-major storage throughout.
// ─────────────────────────────────────────────────────────────────────────────

#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "dgemm_common.h"

// ── Tiled DGEMM SYCL kernel ───────────────────────────────────────────────────
void run_dgemm_sycl(sycl::queue& q, int N, double alpha, double beta,
                     const double* d_A, const double* d_B, double* d_C) {
    constexpr int TILE = DGEMM_TILE;

    // Round up to tile boundary for nd_range requirement
    const int N_round = ((N + TILE - 1) / TILE) * TILE;

    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<double, 2> sA({TILE, TILE}, cgh);
        sycl::local_accessor<double, 2> sB({TILE, TILE}, cgh);

        cgh.parallel_for(
            sycl::nd_range<2>({static_cast<size_t>(N_round),
                               static_cast<size_t>(N_round)},
                              {static_cast<size_t>(TILE),
                               static_cast<size_t>(TILE)}),
            [=](sycl::nd_item<2> item) {
                const int row  = static_cast<int>(item.get_global_id(0));
                const int col  = static_cast<int>(item.get_global_id(1));
                const int trow = static_cast<int>(item.get_local_id(0));
                const int tcol = static_cast<int>(item.get_local_id(1));

                double acc = 0.0;
                const int num_tiles = (N + TILE - 1) / TILE;

                for (int t = 0; t < num_tiles; t++) {
                    const int k_a = t * TILE + tcol;
                    sA[trow][tcol] = (row < N && k_a < N)
                        ? d_A[row * N + k_a] : 0.0;

                    const int k_b = t * TILE + trow;
                    sB[trow][tcol] = (k_b < N && col < N)
                        ? d_B[k_b * N + col] : 0.0;

                    item.barrier(sycl::access::fence_space::local_space);

                    for (int k = 0; k < TILE; k++)
                        acc += sA[trow][k] * sB[k][tcol];

                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (row < N && col < N)
                    d_C[row * N + col] = alpha * acc + beta * d_C[row * N + col];
            });
    });
    q.wait();
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
        "  --verify         Correctness check at N=128 then proceed to timing\n",
        prog, DGEMM_N_LARGE, DGEMM_WARMUP_DEFAULT, DGEMM_REPS_DEFAULT);
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    int         N        = DGEMM_N_LARGE;
    int         warmup   = DGEMM_WARMUP_DEFAULT;
    int         reps     = DGEMM_REPS_DEFAULT;
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

    // Select GPU queue; fall back to CPU if no GPU SYCL backend available
    sycl::queue q{sycl::gpu_selector_v};
    std::printf("# abstraction=sycl N=%d warmup=%d reps=%d platform=%s device=%s\n",
                N, warmup, reps, platform.c_str(),
                q.get_device().get_info<sycl::info::device::name>().c_str());

    const size_t bytes = static_cast<size_t>(N) * N * sizeof(double);

    std::vector<double> hA(static_cast<size_t>(N) * N);
    std::vector<double> hB(static_cast<size_t>(N) * N);
    std::vector<double> hC(static_cast<size_t>(N) * N, 0.0);
    init_matrix(hA.data(), N);
    init_matrix(hB.data(), N);

    double* dA = sycl::malloc_device<double>(static_cast<size_t>(N) * N, q);
    double* dB = sycl::malloc_device<double>(static_cast<size_t>(N) * N, q);
    double* dC = sycl::malloc_device<double>(static_cast<size_t>(N) * N, q);

    q.memcpy(dA, hA.data(), bytes).wait();
    q.memcpy(dB, hB.data(), bytes).wait();
    q.memcpy(dC, hC.data(), bytes).wait();

    double alpha = DGEMM_ALPHA, beta = DGEMM_BETA;
    auto run_once = [&]() { run_dgemm_sycl(q, N, alpha, beta, dA, dB, dC); };

    // ── Correctness check ─────────────────────────────────────────────────────
    if (verify) {
        const int Nv = 128;
        const size_t bv = static_cast<size_t>(Nv) * Nv * sizeof(double);

        std::vector<double> hAv(Nv * Nv), hBv(Nv * Nv),
                            hCv(Nv * Nv, 0.0), hRv(Nv * Nv, 0.0);
        init_matrix(hAv.data(), Nv);
        init_matrix(hBv.data(), Nv);
        dgemm_cpu_ref(Nv, DGEMM_ALPHA, hAv.data(), hBv.data(), DGEMM_BETA, hRv.data());

        double* dvA = sycl::malloc_device<double>(static_cast<size_t>(Nv) * Nv, q);
        double* dvB = sycl::malloc_device<double>(static_cast<size_t>(Nv) * Nv, q);
        double* dvC = sycl::malloc_device<double>(static_cast<size_t>(Nv) * Nv, q);
        q.memcpy(dvA, hAv.data(), bv).wait();
        q.memcpy(dvB, hBv.data(), bv).wait();
        q.memcpy(dvC, hCv.data(), bv).wait();

        run_dgemm_sycl(q, Nv, DGEMM_ALPHA, DGEMM_BETA, dvA, dvB, dvC);

        q.memcpy(hCv.data(), dvC, bv).wait();
        sycl::free(dvA, q); sycl::free(dvB, q); sycl::free(dvC, q);

        double max_err = 0.0;
        bool ok = dgemm_verify(hCv.data(), hRv.data(), Nv, DGEMM_CORRECT_TOL, &max_err);
        std::printf("VERIFY abstraction=sycl N=%d max_rel_err=%.2e %s\n",
                    Nv, max_err, ok ? "PASS" : "FAIL");
        if (!ok) {
            std::fprintf(stderr, "[E2 verify] sycl FAILED — aborting before timing.\n");
            sycl::free(dA, q); sycl::free(dB, q); sycl::free(dC, q);
            return 1;
        }
        std::fprintf(stderr, "[E2 verify] sycl PASS — proceeding to timed measurement.\n");
    }

    for (int i = 0; i < warmup; i++) run_once();

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

    auto flags = compute_hw_state(gflops_vec);
    for (int r = 0; r < reps; r++)
        dgemm_print_hw_state(r + 1, flags[r]);

    DgemmStats stats = compute_dgemm_stats(gflops_vec, flags);
    dgemm_print_summary(N, stats);

    sycl::free(dA, q); sycl::free(dB, q); sycl::free(dC, q);
    return 0;
}
