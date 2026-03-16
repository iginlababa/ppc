// kernel_bfs_kokkos.cpp — E6 BFS: Kokkos scatter + parallel_scan compact.
//
// ── E6 DESIGN DECISIONS ───────────────────────────────────────────────────────
// [D3-Kokkos] Scatter: Kokkos::parallel_for with RangePolicy over frontier_size.
//   Compact: Kokkos::parallel_scan over n_vertices; scan functor writes to
//   d_next_frontier[update] = i when d_flags[i] != 0 (final pass); returns
//   new frontier size as the scan total.
//   Kokkos::fence() between scatter and compact, and after compact.
//   After compact: zero d_flags via Kokkos::deep_copy (reset for next level).
// [D3-compile] Two-step nvcc compilation:
//   nvcc -x cu -c kernel_bfs_kokkos.cpp -o kernel_bfs_kokkos.o
//   nvcc kernel_bfs_kokkos.o -o bfs-kokkos -lkokkoscore -lkokkoscontainers ...
// ─────────────────────────────────────────────────────────────────────────────

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "bfs_common.h"

using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace  = Kokkos::DefaultExecutionSpace::memory_space;

using IntView     = Kokkos::View<int*, MemSpace>;
using IntViewConst= Kokkos::View<const int*, MemSpace>;

// ── Main BFS driver (device views, returns time in ms) ─────────────────────
static double run_bfs_kokkos(
    const IntViewConst& v_row_ptr,
    const IntViewConst& v_col_idx,
    int                 n_vertices,
    const IntView&      v_distances,
    const IntView&      v_frontier,
    const IntView&      v_next_frontier,
    const IntView&      v_flags,
    int /* n_expected_levels */)
{
    // Init frontier: source = BFS_SOURCE
    int init_frontier[1] = { BFS_SOURCE };
    Kokkos::View<int[1], Kokkos::HostSpace> h_init(init_frontier);
    auto d_init = Kokkos::subview(v_frontier, Kokkos::pair<int,int>(0, 1));
    Kokkos::deep_copy(d_init, h_init);

    int frontier_size = 1;

    auto t0 = std::chrono::high_resolution_clock::now();

    int level = 1;
    while (frontier_size > 0) {
        // ── Scatter phase ─────────────────────────────────────────────────
        int fs = frontier_size;
        Kokkos::parallel_for(
            "bfs_scatter_kokkos",
            Kokkos::RangePolicy<ExecSpace>(0, fs),
            KOKKOS_LAMBDA(int tid) {
                int u     = v_frontier(tid);
                int start = v_row_ptr(u);
                int end   = v_row_ptr(u + 1);
                for (int j = start; j < end; j++) {
                    int v = v_col_idx(j);
                    int old = Kokkos::atomic_compare_exchange(
                                  &v_distances(v), -1, level);
                    if (old == -1) {
                        v_flags(v) = 1;
                    }
                }
            });
        Kokkos::fence();

        // ── Compact phase: parallel_scan ──────────────────────────────────
        int new_size = 0;
        Kokkos::parallel_scan(
            "bfs_compact_kokkos",
            Kokkos::RangePolicy<ExecSpace>(0, n_vertices),
            KOKKOS_LAMBDA(int i, int& update, bool final_pass) {
                if (v_flags(i)) {
                    if (final_pass) v_next_frontier(update) = i;
                    update++;
                }
            },
            new_size);
        Kokkos::fence();

        // Reset flags for next level
        Kokkos::deep_copy(v_flags, 0);
        Kokkos::fence();

        // Swap frontier pointers logically — copy next → current
        // (We avoid a raw pointer swap; instead copy only the live slice)
        if (new_size > 0) {
            auto src = Kokkos::subview(v_next_frontier,
                                        Kokkos::pair<int,int>(0, new_size));
            auto dst = Kokkos::subview(v_frontier,
                                        Kokkos::pair<int,int>(0, new_size));
            Kokkos::deep_copy(dst, src);
        }

        frontier_size = new_size;
        level++;
    }

    Kokkos::fence();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ── Reset helper ──────────────────────────────────────────────────────────────
static void reset_distances_kokkos(const IntView& v_distances, int n_vertices) {
    Kokkos::deep_copy(v_distances, -1);
    auto h = Kokkos::create_mirror_view(v_distances);
    h(BFS_SOURCE) = 0;
    Kokkos::deep_copy(v_distances, h);
}

int main(int argc, char** argv) {
    BfsConfig cfg = bfs_parse_args(argc, argv);

    Kokkos::initialize(argc, argv);
    {
        // ── Build graph ──────────────────────────────────────────────────
        CsrGraph g;
        if (cfg.graph_type == "erdos_renyi") {
            g = generate_erdos_renyi(cfg.n);
        } else if (cfg.graph_type == "2d_grid") {
            g = generate_2d_grid(cfg.n);
        } else {
            std::fprintf(stderr, "Unknown graph type: %s\n",
                         cfg.graph_type.c_str());
            Kokkos::finalize();
            return 1;
        }

        BfsResult ref = bfs_cpu_ref(g);
        bfs_print_profile(ref, cfg.graph_type, cfg.size_label);

        // ── Allocate device views ─────────────────────────────────────────
        int n_col = (int)g.col_idx.size();
        IntView v_row_ptr("row_ptr",       cfg.n + 1);
        IntView v_col_idx("col_idx",       n_col);
        IntView v_distances("distances",   cfg.n);
        IntView v_frontier("frontier",     cfg.n);
        IntView v_next_frontier("next_f",  cfg.n);
        IntView v_flags("flags",           cfg.n);

        // Mirror for host-side upload
        auto h_row_ptr = Kokkos::create_mirror_view(v_row_ptr);
        auto h_col_idx = Kokkos::create_mirror_view(v_col_idx);
        for (int i = 0; i <= cfg.n; i++) h_row_ptr(i) = g.row_ptr[i];
        for (int i = 0; i < n_col;   i++) h_col_idx(i) = g.col_idx[i];
        Kokkos::deep_copy(v_row_ptr, h_row_ptr);
        Kokkos::deep_copy(v_col_idx, h_col_idx);
        Kokkos::deep_copy(v_flags, 0);

        IntViewConst v_row_ptr_c = v_row_ptr;
        IntViewConst v_col_idx_c = v_col_idx;

        // ── Correctness check ─────────────────────────────────────────────
        if (cfg.verify) {
            reset_distances_kokkos(v_distances, cfg.n);
            Kokkos::deep_copy(v_flags, 0);
            run_bfs_kokkos(v_row_ptr_c, v_col_idx_c, cfg.n,
                           v_distances, v_frontier, v_next_frontier, v_flags,
                           ref.n_levels);
            std::vector<int> dist_gpu(cfg.n);
            auto h_dist = Kokkos::create_mirror_view(v_distances);
            Kokkos::deep_copy(h_dist, v_distances);
            for (int i = 0; i < cfg.n; i++) dist_gpu[i] = h_dist(i);
            if (!bfs_verify(dist_gpu, ref.distances)) {
                std::fprintf(stderr, "[bfs_kokkos] CORRECTNESS FAILED\n");
                Kokkos::finalize();
                return 1;
            }
            std::fprintf(stderr,
                         "[bfs_kokkos] Correctness OK (n=%d n_levels=%d)\n",
                         cfg.n, ref.n_levels);
            if (cfg.reps == 0) { Kokkos::finalize(); return 0; }
        }

        // ── Warmup ────────────────────────────────────────────────────────
        bfs_adaptive_warmup([&]() -> double {
            reset_distances_kokkos(v_distances, cfg.n);
            Kokkos::deep_copy(v_flags, 0);
            return run_bfs_kokkos(v_row_ptr_c, v_col_idx_c, cfg.n,
                                  v_distances, v_frontier, v_next_frontier,
                                  v_flags, ref.n_levels);
        }, "kokkos");

        // ── Timed runs ────────────────────────────────────────────────────
        std::vector<double> times;
        times.reserve(cfg.reps);
        for (int r = 0; r < cfg.reps; r++) {
            reset_distances_kokkos(v_distances, cfg.n);
            Kokkos::deep_copy(v_flags, 0);
            times.push_back(
                run_bfs_kokkos(v_row_ptr_c, v_col_idx_c, cfg.n,
                               v_distances, v_frontier, v_next_frontier,
                               v_flags, ref.n_levels));
        }

        // ── Report ────────────────────────────────────────────────────────
        double peak_ff = (cfg.n > 0)
            ? (double)ref.max_frontier_width / cfg.n : 0.0;
        for (int r = 0; r < cfg.reps; r++) {
            double ms    = times[r];
            double gteps = (double)g.n_edges / (ms / 1000.0) / 1e9;
            int hw_ok    = bfs_hw_state(ms,
                               std::vector<double>(times.begin(),
                                                    times.begin() + r));
            bfs_print_run(r + 1, cfg.n, g.n_edges, ref.n_levels,
                          ref.max_frontier_width, ref.min_frontier_width,
                          peak_ff, cfg.graph_type, cfg.size_label,
                          ms, gteps, hw_ok);
        }
    }
    Kokkos::finalize();
    return 0;
}
