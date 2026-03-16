// kernel_bfs_raja.cpp — E6 BFS: RAJA scatter + exclusive_scan compact.
//
// ── E6 DESIGN DECISIONS ───────────────────────────────────────────────────────
// [D3-RAJA] Scatter: RAJA::forall<RAJA::cuda_exec<256>> over frontier_size.
//   Compact:
//     (1) RAJA::exclusive_scan<RAJA::cuda_exec<256>> on d_flags → d_scan.
//     (2) RAJA::forall to gather: if d_flags[i], d_next_frontier[d_scan[i]] = i.
//     (3) New frontier size: copy d_scan[N-1] + d_flags[N-1] from device.
//   Between scatter and compact: RAJA::synchronize<RAJA::cuda_synchronize>().
//   After compact: reset d_flags via cudaMemset; reset d_scan is not needed
//   (it is fully overwritten by exclusive_scan each level).
// [D3-compile] Two-step compile:
//   nvcc -x cu -c kernel_bfs_raja.cpp → .o
//   g++ link with -lRAJA -lcamp -lcuda -lcudart
// ─────────────────────────────────────────────────────────────────────────────

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <RAJA/RAJA.hpp>
#include <cuda_runtime.h>

#include "bfs_common.h"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                    \
                         __FILE__, __LINE__, cudaGetErrorString(_e));          \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

using RajaExec = RAJA::cuda_exec<BFS_BLOCK_SIZE>;
using RajaSync = RAJA::cuda_synchronize;

// ── Main BFS driver ──────────────────────────────────────────────────────────
static double run_bfs_raja(
    const int* d_row_ptr,
    const int* d_col_idx,
    int        n_vertices,
    int*       d_distances,
    int*       d_frontier,
    int*       d_next_frontier,
    int*       d_flags,
    int*       d_scan,
    int /* n_expected_levels */)
{
    // Init frontier
    int src = BFS_SOURCE;
    CUDA_CHECK(cudaMemcpy(d_frontier, &src, sizeof(int),
                          cudaMemcpyHostToDevice));
    int frontier_size = 1;

    auto t0 = std::chrono::high_resolution_clock::now();

    int level = 1;
    while (frontier_size > 0) {
        int fs = frontier_size;

        // ── Scatter ───────────────────────────────────────────────────────
        RAJA::forall<RajaExec>(
            RAJA::RangeSegment(0, fs),
            [=] RAJA_DEVICE(int tid) {
                int u     = d_frontier[tid];
                int start = d_row_ptr[u];
                int end   = d_row_ptr[u + 1];
                for (int j = start; j < end; j++) {
                    int v   = d_col_idx[j];
                    int old = atomicCAS(&d_distances[v], -1, level);
                    if (old == -1) {
                        d_flags[v] = 1;
                    }
                }
            });
        RAJA::synchronize<RajaSync>();

        // ── Compact: exclusive_scan → gather ──────────────────────────────
        RAJA::exclusive_scan<RajaExec>(
            d_flags, d_flags + n_vertices,
            d_scan, 0, RAJA::operators::plus<int>{});
        RAJA::synchronize<RajaSync>();

        // Gather new frontier
        RAJA::forall<RajaExec>(
            RAJA::RangeSegment(0, n_vertices),
            [=] RAJA_DEVICE(int i) {
                if (d_flags[i]) {
                    d_next_frontier[d_scan[i]] = i;
                }
            });
        RAJA::synchronize<RajaSync>();

        // New frontier size from last scan element + last flag
        int scan_last = 0, flag_last = 0;
        CUDA_CHECK(cudaMemcpy(&scan_last, d_scan    + n_vertices - 1,
                              sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&flag_last, d_flags   + n_vertices - 1,
                              sizeof(int), cudaMemcpyDeviceToHost));
        frontier_size = scan_last + flag_last;

        // Reset flags
        CUDA_CHECK(cudaMemset(d_flags, 0, n_vertices * sizeof(int)));

        // next → current
        if (frontier_size > 0) {
            CUDA_CHECK(cudaMemcpy(d_frontier, d_next_frontier,
                                  frontier_size * sizeof(int),
                                  cudaMemcpyDeviceToDevice));
        }
        level++;
    }

    RAJA::synchronize<RajaSync>();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ── Reset helper ──────────────────────────────────────────────────────────────
static void reset_distances_raja(int* d_distances, int n_vertices) {
    CUDA_CHECK(cudaMemset(d_distances, 0xFF, n_vertices * sizeof(int)));
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(&d_distances[BFS_SOURCE], &zero, sizeof(int),
                          cudaMemcpyHostToDevice));
}

int main(int argc, char** argv) {
    BfsConfig cfg = bfs_parse_args(argc, argv);

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

    // ── Allocate device memory ────────────────────────────────────────────────
    int n_col = (int)g.col_idx.size();
    int *d_row_ptr, *d_col_idx, *d_distances;
    int *d_frontier, *d_next_frontier, *d_flags, *d_scan;

    CUDA_CHECK(cudaMalloc(&d_row_ptr,       (cfg.n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx,       n_col       * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_distances,     cfg.n       * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier,      cfg.n       * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier, cfg.n       * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_flags,         cfg.n       * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scan,          cfg.n       * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, g.row_ptr.data(), (cfg.n+1)*sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, g.col_idx.data(), n_col*sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_flags, 0, cfg.n * sizeof(int)));

    // ── Correctness check ─────────────────────────────────────────────────────
    if (cfg.verify) {
        reset_distances_raja(d_distances, cfg.n);
        CUDA_CHECK(cudaMemset(d_flags, 0, cfg.n * sizeof(int)));
        run_bfs_raja(d_row_ptr, d_col_idx, cfg.n,
                     d_distances, d_frontier, d_next_frontier,
                     d_flags, d_scan, ref.n_levels);
        std::vector<int> dist_gpu(cfg.n);
        CUDA_CHECK(cudaMemcpy(dist_gpu.data(), d_distances, cfg.n*sizeof(int),
                              cudaMemcpyDeviceToHost));
        if (!bfs_verify(dist_gpu, ref.distances)) {
            std::fprintf(stderr, "[bfs_raja] CORRECTNESS FAILED\n");
            return 1;
        }
        std::fprintf(stderr,
                     "[bfs_raja] Correctness OK (n=%d n_levels=%d)\n",
                     cfg.n, ref.n_levels);
        if (cfg.reps == 0) return 0;
    }

    // ── Warmup ────────────────────────────────────────────────────────────────
    bfs_adaptive_warmup([&]() -> double {
        reset_distances_raja(d_distances, cfg.n);
        CUDA_CHECK(cudaMemset(d_flags, 0, cfg.n * sizeof(int)));
        return run_bfs_raja(d_row_ptr, d_col_idx, cfg.n,
                            d_distances, d_frontier, d_next_frontier,
                            d_flags, d_scan, ref.n_levels);
    }, "raja");

    // ── Timed runs ────────────────────────────────────────────────────────────
    std::vector<double> times;
    times.reserve(cfg.reps);
    for (int r = 0; r < cfg.reps; r++) {
        reset_distances_raja(d_distances, cfg.n);
        CUDA_CHECK(cudaMemset(d_flags, 0, cfg.n * sizeof(int)));
        times.push_back(
            run_bfs_raja(d_row_ptr, d_col_idx, cfg.n,
                         d_distances, d_frontier, d_next_frontier,
                         d_flags, d_scan, ref.n_levels));
    }

    // ── Report ────────────────────────────────────────────────────────────────
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

    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_frontier));
    CUDA_CHECK(cudaFree(d_next_frontier));
    CUDA_CHECK(cudaFree(d_flags));
    CUDA_CHECK(cudaFree(d_scan));
    return 0;
}
