// kernel_dgemm_raja.cpp — E2 DGEMM: naïve RAJA::forall (raja_naive).
//
// ── E2 DESIGN DECISIONS ───────────────────────────────────────────────────────
// [D6] This implementation deliberately uses no shared memory / tiling.
//   RAJA's execution policy model (RAJA::forall over a flat index set) does not
//   provide a clean, portable API for expressing GPU shared memory tiling in
//   RAJA 2024.x without resorting to raw CUDA intrinsics inside the lambda —
//   which defeats the abstraction goal. The RAJA::kernel + statement::CudaShmem
//   interface exists but is non-portable and underdocumented.
//
//   Scientific intent: measure the cost of this API Limitation.
//   Expected result: raja_naive ~ memory-bandwidth-limited (~44 GFLOP/s at 350
//   GB/s, arithmetic intensity = 1/8 FLOP/byte without tiling) vs native tiled
//   (~approach FP64 hardware ceiling). This ~7× gap IS the taxonomy finding.
//
//   Label in CSV: abstraction = "raja_naive"
// ─────────────────────────────────────────────────────────────────────────────

#include <RAJA/RAJA.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "dgemm_common.h"
#include "gpu_compat.h"

// ── Execution policy ──────────────────────────────────────────────────────────
// 256 threads per block: flat 1D launch over N*N output elements.
// Each thread computes one C[row, col] via an O(N) inner loop from global mem.
#ifdef __HIP_PLATFORM_AMD__
using ExecPolicy    = RAJA::hip_exec<256>;
using SyncPolicy    = RAJA::hip_synchronize;
#else
using ExecPolicy    = RAJA::cuda_exec<256>;
using SyncPolicy    = RAJA::cuda_synchronize;
#endif

// ── naïve DGEMM kernel ────────────────────────────────────────────────────────
void run_dgemm_raja(int N, double alpha, double beta,
                    const double* d_A, const double* d_B, double* d_C) {
    RAJA::forall<ExecPolicy>(
        RAJA::RangeSegment(0, N * N),
        [=] RAJA_DEVICE (int idx) {
            const int row = idx / N;
            const int col = idx % N;
            double sum = 0.0;
            // Non-tiled: each thread reads full row of A and full column of B
            // from global memory with no reuse. This is the API limitation.
            for (int k = 0; k < N; k++)
                sum += d_A[row * N + k] * d_B[k * N + col];
            d_C[idx] = alpha * sum + beta * d_C[idx];
        });
    RAJA::synchronize<SyncPolicy>();
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

    // ── Correctness check ─────────────────────────────────────────────────────
    if (verify) {
        const int Nv = 128;
        const size_t bv = static_cast<size_t>(Nv) * Nv * sizeof(double);

        std::vector<double> hAv(Nv * Nv), hBv(Nv * Nv),
                            hCv(Nv * Nv, 0.0), hRv(Nv * Nv, 0.0);
        init_matrix(hAv.data(), Nv);
        init_matrix(hBv.data(), Nv);
        dgemm_cpu_ref(Nv, DGEMM_ALPHA, hAv.data(), hBv.data(), DGEMM_BETA, hRv.data());

        double *dvA = nullptr, *dvB = nullptr, *dvC = nullptr;
        gpuMalloc(&dvA, bv); gpuMalloc(&dvB, bv); gpuMalloc(&dvC, bv);
        gpuMemcpy(dvA, hAv.data(), bv, gpuMemcpyHostToDevice);
        gpuMemcpy(dvB, hBv.data(), bv, gpuMemcpyHostToDevice);
        gpuMemcpy(dvC, hCv.data(), bv, gpuMemcpyHostToDevice);

        run_dgemm_raja(Nv, DGEMM_ALPHA, DGEMM_BETA, dvA, dvB, dvC);

        gpuMemcpy(hCv.data(), dvC, bv, gpuMemcpyDeviceToHost);
        gpuFree(dvA); gpuFree(dvB); gpuFree(dvC);

        double max_err = 0.0;
        bool ok = dgemm_verify(hCv.data(), hRv.data(), Nv, DGEMM_CORRECT_TOL, &max_err);
        std::printf("VERIFY abstraction=raja_naive N=%d max_rel_err=%.2e %s\n",
                    Nv, max_err, ok ? "PASS" : "FAIL");
        if (!ok) {
            std::fprintf(stderr, "[E2 verify] raja_naive FAILED — aborting before timing.\n");
            return 1;
        }
        std::fprintf(stderr, "[E2 verify] raja_naive PASS — proceeding to timed measurement.\n");
    }

    const size_t bytes = static_cast<size_t>(N) * N * sizeof(double);

    std::vector<double> hA(static_cast<size_t>(N) * N);
    std::vector<double> hB(static_cast<size_t>(N) * N);
    std::vector<double> hC(static_cast<size_t>(N) * N, 0.0);
    init_matrix(hA.data(), N);
    init_matrix(hB.data(), N);

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    if (gpuMalloc(&dA, bytes) != gpuSuccess ||
        gpuMalloc(&dB, bytes) != gpuSuccess ||
        gpuMalloc(&dC, bytes) != gpuSuccess) {
        std::fprintf(stderr, "gpuMalloc failed for N=%d\n", N);
        return 1;
    }
    gpuMemcpy(dA, hA.data(), bytes, gpuMemcpyHostToDevice);
    gpuMemcpy(dB, hB.data(), bytes, gpuMemcpyHostToDevice);
    gpuMemcpy(dC, hC.data(), bytes, gpuMemcpyHostToDevice);

    double alpha = DGEMM_ALPHA, beta = DGEMM_BETA;
    auto run_once = [&]() { run_dgemm_raja(N, alpha, beta, dA, dB, dC); };

    std::printf("# abstraction=raja_naive N=%d warmup=%d reps=%d platform=%s\n",
                N, warmup, reps, platform.c_str());

    // Warmup — especially important for RAJA JIT specialization
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

    gpuFree(dA); gpuFree(dB); gpuFree(dC);
    return 0;
}
