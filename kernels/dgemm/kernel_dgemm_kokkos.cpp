// kernel_dgemm_kokkos.cpp — E2 DGEMM: Kokkos TeamPolicy tiled implementation.
//
// ── E2 DESIGN DECISIONS ───────────────────────────────────────────────────────
// [D2-Kokkos] TeamPolicy: each team (= CUDA block) computes one 32×32 output
//   tile. team_size = TILE*TILE = 1024, matching the CUDA block exactly.
//   Scratch level-0 (shared memory) holds two TILE×TILE double arrays (16 KB).
//   Thread rank t = trow*TILE + tcol → each thread owns one C(grow, gcol).
// [D4] KokkosBlas::gemm not used here (ceiling reference only, per §D5).
//   kokkos_blas would be a separate entry in a future extension.
// [D6] LayoutRight = row-major, matching all other C++ implementations.
// ─────────────────────────────────────────────────────────────────────────────

#include <Kokkos_Core.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "dgemm_common.h"

using ExecSpace   = Kokkos::DefaultExecutionSpace;
using TeamPol     = Kokkos::TeamPolicy<ExecSpace>;
using MemberT     = TeamPol::member_type;
using ScratchSpace = ExecSpace::scratch_memory_space;
using ScratchView  = Kokkos::View<double*, ScratchSpace, Kokkos::MemoryUnmanaged>;

// ── Tiled DGEMM via TeamPolicy + scratch memory ───────────────────────────────
void run_dgemm_kokkos(int N, double alpha, double beta,
                       const double* d_A, const double* d_B, double* d_C) {
    constexpr int TILE = DGEMM_TILE;

    // Wrap raw device pointers (cudaMalloc) in unmanaged row-major Views
    Kokkos::View<const double**, Kokkos::LayoutRight, ExecSpace,
                 Kokkos::MemoryUnmanaged> A(d_A, N, N);
    Kokkos::View<const double**, Kokkos::LayoutRight, ExecSpace,
                 Kokkos::MemoryUnmanaged> B(d_B, N, N);
    Kokkos::View<double**, Kokkos::LayoutRight, ExecSpace,
                 Kokkos::MemoryUnmanaged> C(d_C, N, N);

    const int num_block_row = (N + TILE - 1) / TILE;
    const int num_block_col = (N + TILE - 1) / TILE;
    const int num_teams     = num_block_row * num_block_col;
    const int team_size     = TILE * TILE;   // 1024 — matches CUDA block
    const int num_tiles_k   = (N + TILE - 1) / TILE;

    // Scratch level 0 = shared memory; 2 TILE×TILE double arrays = 16 KB
    const size_t shmem = static_cast<size_t>(2 * TILE * TILE) * sizeof(double);

    Kokkos::parallel_for(
        "dgemm_kokkos",
        TeamPol(num_teams, team_size)
            .set_scratch_size(0, Kokkos::PerTeam(shmem)),
        KOKKOS_LAMBDA(const MemberT& team) {
            const int blk  = team.league_rank();
            const int brow = blk / num_block_col;
            const int bcol = blk % num_block_col;
            const int trow = team.team_rank() / TILE;
            const int tcol = team.team_rank() % TILE;

            // Split flat scratch into two TILE×TILE tiles
            ScratchView scratch(team.team_scratch(0), 2 * TILE * TILE);
            double* sA = scratch.data();
            double* sB = scratch.data() + TILE * TILE;

            const int grow = brow * TILE + trow;
            const int gcol = bcol * TILE + tcol;
            double acc = 0.0;

            for (int t = 0; t < num_tiles_k; t++) {
                // Load sA tile: row=grow, k-col=t*TILE+tcol
                const int gk_a = t * TILE + tcol;
                sA[trow * TILE + tcol] =
                    (grow < N && gk_a < N) ? A(grow, gk_a) : 0.0;

                // Load sB tile: k-row=t*TILE+trow, col=gcol
                const int gk_b = t * TILE + trow;
                sB[trow * TILE + tcol] =
                    (gk_b < N && gcol < N) ? B(gk_b, gcol) : 0.0;

                team.team_barrier();

                if (grow < N && gcol < N) {
                    #pragma unroll
                    for (int k = 0; k < TILE; k++)
                        acc += sA[trow * TILE + k] * sB[k * TILE + tcol];
                }

                team.team_barrier();
            }

            if (grow < N && gcol < N)
                C(grow, gcol) = alpha * acc + beta * C(grow, gcol);
        });

    Kokkos::fence();
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
    // Parse flags before Kokkos::initialize so --help works
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

    Kokkos::initialize(argc, argv);
    {
        // ── Correctness check ─────────────────────────────────────────────────
        if (verify) {
            const int Nv = 128;
            const size_t bv = static_cast<size_t>(Nv) * Nv * sizeof(double);

            std::vector<double> hAv(Nv * Nv), hBv(Nv * Nv),
                                hCv(Nv * Nv, 0.0), hRv(Nv * Nv, 0.0);
            init_matrix(hAv.data(), Nv);
            init_matrix(hBv.data(), Nv);
            dgemm_cpu_ref(Nv, DGEMM_ALPHA, hAv.data(), hBv.data(), DGEMM_BETA, hRv.data());

            double *dvA = nullptr, *dvB = nullptr, *dvC = nullptr;
            cudaMalloc(&dvA, bv); cudaMalloc(&dvB, bv); cudaMalloc(&dvC, bv);
            cudaMemcpy(dvA, hAv.data(), bv, cudaMemcpyHostToDevice);
            cudaMemcpy(dvB, hBv.data(), bv, cudaMemcpyHostToDevice);
            cudaMemcpy(dvC, hCv.data(), bv, cudaMemcpyHostToDevice);

            run_dgemm_kokkos(Nv, DGEMM_ALPHA, DGEMM_BETA, dvA, dvB, dvC);

            cudaMemcpy(hCv.data(), dvC, bv, cudaMemcpyDeviceToHost);
            cudaFree(dvA); cudaFree(dvB); cudaFree(dvC);

            double max_err = 0.0;
            bool ok = dgemm_verify(hCv.data(), hRv.data(), Nv, DGEMM_CORRECT_TOL, &max_err);
            std::printf("VERIFY abstraction=kokkos N=%d max_rel_err=%.2e %s\n",
                        Nv, max_err, ok ? "PASS" : "FAIL");
            if (!ok) {
                std::fprintf(stderr, "[E2 verify] kokkos FAILED — aborting before timing.\n");
                Kokkos::finalize();
                return 1;
            }
            std::fprintf(stderr, "[E2 verify] kokkos PASS — proceeding to timed measurement.\n");
        }

        const size_t bytes = static_cast<size_t>(N) * N * sizeof(double);

        // Allocate host matrices
        std::vector<double> hA(static_cast<size_t>(N) * N);
        std::vector<double> hB(static_cast<size_t>(N) * N);
        std::vector<double> hC(static_cast<size_t>(N) * N, 0.0);
        init_matrix(hA.data(), N);
        init_matrix(hB.data(), N);

        // Allocate device memory (raw CUDA — keeps memory management consistent
        // with other abstraction benchmarks)
        double *dA = nullptr, *dB = nullptr, *dC = nullptr;
        if (cudaMalloc(&dA, bytes) != cudaSuccess ||
            cudaMalloc(&dB, bytes) != cudaSuccess ||
            cudaMalloc(&dC, bytes) != cudaSuccess) {
            std::fprintf(stderr, "cudaMalloc failed for N=%d\n", N);
            Kokkos::finalize();
            return 1;
        }
        cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dC, hC.data(), bytes, cudaMemcpyHostToDevice);

        double alpha = DGEMM_ALPHA, beta = DGEMM_BETA;

        auto run_once = [&]() {
            run_dgemm_kokkos(N, alpha, beta, dA, dB, dC);
        };

        // Warmup
        std::printf("# abstraction=kokkos N=%d warmup=%d reps=%d platform=%s\n",
                    N, warmup, reps, platform.c_str());
        for (int i = 0; i < warmup; i++) run_once();

        // Timed runs
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

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
    Kokkos::finalize();
    return 0;
}
