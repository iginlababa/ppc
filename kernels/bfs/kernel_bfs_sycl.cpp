// kernel_bfs_sycl.cpp — E6 BFS: SYCL USM nd_range<1> scatter + atomic counter compact.
//
// ── E6 DESIGN DECISIONS ───────────────────────────────────────────────────────
// [D3-SYCL] Two phases per BFS level:
//   Scatter: q.parallel_for(nd_range<1>(global_sz, BLOCK_SIZE), ...) — one work-item
//            per frontier vertex. sycl::atomic_ref (acq_rel, device scope) for atomicCAS
//            on d_distances[v]; sets d_flags[v]=1 on first discovery. q.wait() after.
//   Compact: atomic counter (avoids DPL/ROCThrust dependency).
//            q.memset(d_next_size, 0) → parallel_for over N vertices → q.wait().
//            q.memcpy(&new_size, d_next_size) to read frontier size on host.
//            q.memset(d_flags, 0) after compact.
//   q.wait() is the mandatory inter-level sync barrier. Each level = 2 q.wait() calls
//   (scatter + compact) plus flag reset. For 2d_grid large (n_levels=511), that is
//   511 × 3+ q.wait() calls per BFS run → dominant HSA dispatch overhead.
// [D5-SYCL] Metric: GTEPS stored in throughput_gflops for CSV schema consistency.
// [D7-SYCL] d_distances reset via q.memset(0xFF) + q.memcpy(source=0), excluded
//           from timed region.
// AdaptiveCpp --acpp-targets=hip:gfx942 compatible (AMD MI300X).
// ─────────────────────────────────────────────────────────────────────────────

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "bfs_common.h"

// ── Main BFS driver ───────────────────────────────────────────────────────────
static double run_bfs_sycl(
    sycl::queue& q,
    const int*   d_row_ptr,
    const int*   d_col_idx,
    int          n_vertices,
    int*         d_distances,
    int*         d_frontier,
    int*         d_next_frontier,
    int*         d_flags,
    int*         d_next_size,
    int          /* n_expected_levels */)
{
    const size_t local_sz = static_cast<size_t>(BFS_BLOCK_SIZE);

    // Init frontier: source = BFS_SOURCE
    int frontier_size = 1;
    {
        int src = BFS_SOURCE;
        q.memcpy(d_frontier, &src, sizeof(int)).wait();
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    int level = 1;
    while (frontier_size > 0) {
        // ── Scatter ───────────────────────────────────────────────────────────
        {
            int fs = frontier_size;
            int lev = level;
            size_t global_sz = ((static_cast<size_t>(fs) + local_sz - 1)
                                / local_sz) * local_sz;
            q.parallel_for(
                sycl::nd_range<1>(global_sz, local_sz),
                [=](sycl::nd_item<1> item) {
                    int tid = static_cast<int>(item.get_global_id(0));
                    if (tid >= fs) return;
                    int u = d_frontier[tid];
                    int start = d_row_ptr[u];
                    int end   = d_row_ptr[u + 1];
                    for (int j = start; j < end; j++) {
                        int v = d_col_idx[j];
                        sycl::atomic_ref<int,
                            sycl::memory_order::acq_rel,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                                ref(d_distances[v]);
                        int expected = -1;
                        if (ref.compare_exchange_strong(expected, lev)) {
                            d_flags[v] = 1;
                        }
                    }
                });
            q.wait();
        }

        // ── Compact: atomic counter ────────────────────────────────────────
        q.memset(d_next_size, 0, sizeof(int)).wait();
        {
            size_t g2 = ((static_cast<size_t>(n_vertices) + local_sz - 1)
                         / local_sz) * local_sz;
            q.parallel_for(
                sycl::nd_range<1>(g2, local_sz),
                [=](sycl::nd_item<1> item) {
                    int v = static_cast<int>(item.get_global_id(0));
                    if (v >= n_vertices || !d_flags[v]) return;
                    sycl::atomic_ref<int,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                            counter(*d_next_size);
                    int pos = counter.fetch_add(1);
                    d_next_frontier[pos] = v;
                });
            q.wait();
        }

        // ── Reset flags ───────────────────────────────────────────────────
        q.memset(d_flags, 0, n_vertices * sizeof(int)).wait();

        // ── Read new frontier size (host-visible after wait) ──────────────
        int new_size = 0;
        q.memcpy(&new_size, d_next_size, sizeof(int)).wait();

        if (new_size > 0) {
            q.memcpy(d_frontier, d_next_frontier,
                     new_size * sizeof(int)).wait();
        }

        frontier_size = new_size;
        level++;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ── Reset helper ──────────────────────────────────────────────────────────────
static void reset_distances_sycl(sycl::queue& q, int* d_distances,
                                  int n_vertices) {
    q.memset(d_distances, 0xFF, n_vertices * sizeof(int)).wait();  // -1
    int zero = 0;
    q.memcpy(&d_distances[BFS_SOURCE], &zero, sizeof(int)).wait();
}

int main(int argc, char** argv) {
    BfsConfig cfg = bfs_parse_args(argc, argv);

    // ── SYCL queue ───────────────────────────────────────────────────────────
    sycl::queue q(sycl::gpu_selector_v,
                  sycl::property::queue::in_order{});

    // ── Build graph ──────────────────────────────────────────────────────────
    CsrGraph g;
    if (cfg.graph_type == "erdos_renyi") {
        g = generate_erdos_renyi(cfg.n);
    } else if (cfg.graph_type == "2d_grid") {
        g = generate_2d_grid(cfg.n);
    } else {
        std::fprintf(stderr, "Unknown graph type: %s\n", cfg.graph_type.c_str());
        return 1;
    }

    BfsResult ref = bfs_cpu_ref(g);
    bfs_print_profile(ref, cfg.graph_type, cfg.size_label);

    // ── Allocate USM device memory ────────────────────────────────────────────
    int n_col = (int)g.col_idx.size();
    int* d_row_ptr       = sycl::malloc_device<int>(cfg.n + 1,   q);
    int* d_col_idx       = sycl::malloc_device<int>(n_col,        q);
    int* d_distances     = sycl::malloc_device<int>(cfg.n,        q);
    int* d_frontier      = sycl::malloc_device<int>(cfg.n,        q);
    int* d_next_frontier = sycl::malloc_device<int>(cfg.n,        q);
    int* d_flags         = sycl::malloc_device<int>(cfg.n,        q);
    int* d_next_size     = sycl::malloc_device<int>(1,            q);

    q.memcpy(d_row_ptr, g.row_ptr.data(), (cfg.n+1)*sizeof(int)).wait();
    q.memcpy(d_col_idx, g.col_idx.data(), n_col*sizeof(int)).wait();
    q.memset(d_flags, 0, cfg.n * sizeof(int)).wait();

    // ── Correctness check ─────────────────────────────────────────────────────
    if (cfg.verify) {
        reset_distances_sycl(q, d_distances, cfg.n);
        q.memset(d_flags, 0, cfg.n * sizeof(int)).wait();
        run_bfs_sycl(q, d_row_ptr, d_col_idx, cfg.n,
                     d_distances, d_frontier, d_next_frontier,
                     d_flags, d_next_size, ref.n_levels);
        std::vector<int> dist_gpu(cfg.n);
        q.memcpy(dist_gpu.data(), d_distances, cfg.n*sizeof(int)).wait();
        q.memset(d_flags, 0, cfg.n * sizeof(int)).wait();
        if (!bfs_verify(dist_gpu, ref.distances)) {
            std::fprintf(stderr, "[bfs_sycl] CORRECTNESS FAILED\n");
            return 1;
        }
        std::fprintf(stderr, "[bfs_sycl] Correctness OK (n=%d n_levels=%d)\n",
                     cfg.n, ref.n_levels);
        if (cfg.reps == 0) return 0;
    }

    // ── Warmup ────────────────────────────────────────────────────────────────
    bfs_adaptive_warmup([&]() -> double {
        reset_distances_sycl(q, d_distances, cfg.n);
        q.memset(d_flags, 0, cfg.n * sizeof(int)).wait();
        return run_bfs_sycl(q, d_row_ptr, d_col_idx, cfg.n,
                            d_distances, d_frontier, d_next_frontier,
                            d_flags, d_next_size, ref.n_levels);
    }, "sycl");

    // ── Timed runs ────────────────────────────────────────────────────────────
    std::vector<double> times;
    times.reserve(cfg.reps);
    for (int r = 0; r < cfg.reps; r++) {
        reset_distances_sycl(q, d_distances, cfg.n);
        q.memset(d_flags, 0, cfg.n * sizeof(int)).wait();
        double ms = run_bfs_sycl(q, d_row_ptr, d_col_idx, cfg.n,
                                  d_distances, d_frontier, d_next_frontier,
                                  d_flags, d_next_size, ref.n_levels);
        times.push_back(ms);
    }

    // ── Report ────────────────────────────────────────────────────────────────
    double peak_ff = (cfg.n > 0) ? (double)ref.max_frontier_width / cfg.n : 0.0;
    for (int r = 0; r < cfg.reps; r++) {
        double ms    = times[r];
        double gteps = (double)g.n_edges / (ms / 1000.0) / 1e9;
        int hw_ok    = bfs_hw_state(ms, std::vector<double>(times.begin(),
                                                              times.begin() + r));
        bfs_print_run(r + 1, cfg.n, g.n_edges, ref.n_levels,
                      ref.max_frontier_width, ref.min_frontier_width,
                      peak_ff, cfg.graph_type, cfg.size_label,
                      ms, gteps, hw_ok);
    }

    sycl::free(d_row_ptr,       q);
    sycl::free(d_col_idx,       q);
    sycl::free(d_distances,     q);
    sycl::free(d_frontier,      q);
    sycl::free(d_next_frontier, q);
    sycl::free(d_flags,         q);
    sycl::free(d_next_size,     q);
    return 0;
}
