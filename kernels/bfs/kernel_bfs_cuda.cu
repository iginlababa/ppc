// kernel_bfs_cuda.cu — E6 BFS: native CUDA with Thrust compact.
//
// ── E6 DESIGN DECISIONS ───────────────────────────────────────────────────────
// [D3-CUDA] Two phases per BFS level:
//   Scatter: bfs_scatter_kernel — one thread per frontier vertex.
//            Each thread iterates over its adjacency list; atomicCAS on
//            d_distances[v] to claim newly-discovered vertices; sets d_flags[v]=1.
//   Compact: thrust::copy_if over [0,N) with d_flags as stencil → d_next_frontier.
//            thrust::device policy synchronises implicitly.
//            cudaDeviceSynchronize() after scatter before Thrust compact.
//            After compact: cudaMemset d_flags to 0 (reset for next level).
// [D5-CUDA] Metric: GTEPS = n_edges / time_s / 1e9.  Stored in throughput_gflops.
// [D7-CUDA] d_distances reset to -1 (source = 0) excluded from timed region.
//           Warmup loop includes reset.
// ─────────────────────────────────────────────────────────────────────────────

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

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

// ── Scatter kernel ────────────────────────────────────────────────────────────
// One thread per frontier vertex.  For each neighbor v: atomicCAS on distances.
__global__ void bfs_scatter_kernel(
    const int* __restrict__ d_frontier,
    int                     frontier_size,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    int*                    d_distances,
    int*                    d_flags,
    int                     next_level)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    int u = d_frontier[tid];
    int start = row_ptr[u];
    int end   = row_ptr[u + 1];
    for (int j = start; j < end; j++) {
        int v = col_idx[j];
        int old = atomicCAS(&d_distances[v], -1, next_level);
        if (old == -1) {
            d_flags[v] = 1;
        }
    }
}

// ── Main BFS driver ───────────────────────────────────────────────────────────
// Returns traversal time in milliseconds.
static double run_bfs_cuda(
    const int*  d_row_ptr,
    const int*  d_col_idx,
    int         n_vertices,
    int*        d_distances,    // pre-allocated n_vertices ints (reset by caller)
    int*        d_frontier,     // pre-allocated n_vertices ints (scratch)
    int*        d_next_frontier,// pre-allocated n_vertices ints (scratch)
    int*        d_flags,        // pre-allocated n_vertices ints (scratch, zeroed)
    int         n_expected_levels)
{
    // Init: source = BFS_SOURCE, level 0
    int frontier_size = 1;
    CUDA_CHECK(cudaMemcpy(d_frontier, &BFS_SOURCE, sizeof(int),
                          cudaMemcpyHostToDevice));

    auto t0 = std::chrono::high_resolution_clock::now();

    int level = 1;
    while (frontier_size > 0) {
        int blocks = (frontier_size + BFS_BLOCK_SIZE - 1) / BFS_BLOCK_SIZE;
        bfs_scatter_kernel<<<blocks, BFS_BLOCK_SIZE>>>(
            d_frontier, frontier_size,
            d_row_ptr, d_col_idx,
            d_distances, d_flags, level);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Compact: copy vertex IDs where d_flags[v] != 0 into d_next_frontier.
        // Reconstruct device_ptrs each iteration — avoids stale pointer after any
        // previous swap or alias.
        thrust::device_ptr<int> flags_ptr(d_flags);
        thrust::device_ptr<int> next_ptr(d_next_frontier);
        auto end_ptr = thrust::copy_if(
            thrust::device,
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(n_vertices),
            flags_ptr,
            next_ptr,
            [] __device__(int f) { return f != 0; });

        frontier_size = (int)(end_ptr - next_ptr);

        // Reset flags
        CUDA_CHECK(cudaMemset(d_flags, 0, n_vertices * sizeof(int)));

        // Copy new frontier into d_frontier (no pointer swap — avoids aliasing)
        if (frontier_size > 0) {
            CUDA_CHECK(cudaMemcpy(d_frontier, d_next_frontier,
                                  frontier_size * sizeof(int),
                                  cudaMemcpyDeviceToDevice));
        }
        level++;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return ms;
}

// ── Reset helper (excluded from timed region) ─────────────────────────────────
static void reset_distances(int* d_distances, int n_vertices) {
    CUDA_CHECK(cudaMemset(d_distances, 0xFF, n_vertices * sizeof(int)));  // -1
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(&d_distances[BFS_SOURCE], &zero, sizeof(int),
                          cudaMemcpyHostToDevice));
}

int main(int argc, char** argv) {
    BfsConfig cfg = bfs_parse_args(argc, argv);

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

    // ── CPU reference BFS ─────────────────────────────────────────────────────
    BfsResult ref = bfs_cpu_ref(g);
    bfs_print_profile(ref, cfg.graph_type, cfg.size_label);

    if (cfg.verify) {
        std::fprintf(stderr, "[bfs_cuda] graph=%s n=%d n_edges=%d n_levels=%d "
                     "max_fw=%d\n", cfg.graph_type.c_str(), cfg.n, g.n_edges,
                     ref.n_levels, ref.max_frontier_width);
    }

    // ── Allocate device memory ────────────────────────────────────────────────
    int* d_row_ptr;
    int* d_col_idx;
    int* d_distances;
    int* d_frontier;
    int* d_next_frontier;
    int* d_flags;
    int  n_col = (int)g.col_idx.size();

    CUDA_CHECK(cudaMalloc(&d_row_ptr,       (cfg.n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx,       n_col       * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_distances,     cfg.n       * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier,      cfg.n       * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier, cfg.n       * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_flags,         cfg.n       * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, g.row_ptr.data(), (cfg.n+1)*sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, g.col_idx.data(), n_col*sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_flags, 0, cfg.n * sizeof(int)));

    // ── Correctness check ─────────────────────────────────────────────────────
    if (cfg.verify) {
        reset_distances(d_distances, cfg.n);
        run_bfs_cuda(d_row_ptr, d_col_idx, cfg.n,
                     d_distances, d_frontier, d_next_frontier, d_flags,
                     ref.n_levels);
        std::vector<int> dist_gpu(cfg.n);
        CUDA_CHECK(cudaMemcpy(dist_gpu.data(), d_distances, cfg.n*sizeof(int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemset(d_flags, 0, cfg.n * sizeof(int)));
        if (!bfs_verify(dist_gpu, ref.distances)) {
            std::fprintf(stderr, "[bfs_cuda] CORRECTNESS FAILED\n");
            return 1;
        }
        std::fprintf(stderr, "[bfs_cuda] Correctness OK (n=%d n_levels=%d)\n",
                     cfg.n, ref.n_levels);
        if (cfg.reps == 0) return 0;
    }

    // ── Warmup ────────────────────────────────────────────────────────────────
    bfs_adaptive_warmup([&]() -> double {
        reset_distances(d_distances, cfg.n);
        CUDA_CHECK(cudaMemset(d_flags, 0, cfg.n * sizeof(int)));
        return run_bfs_cuda(d_row_ptr, d_col_idx, cfg.n,
                            d_distances, d_frontier, d_next_frontier, d_flags,
                            ref.n_levels);
    }, "cuda");

    // ── Timed runs ────────────────────────────────────────────────────────────
    std::vector<double> times;
    times.reserve(cfg.reps);
    for (int r = 0; r < cfg.reps; r++) {
        reset_distances(d_distances, cfg.n);
        CUDA_CHECK(cudaMemset(d_flags, 0, cfg.n * sizeof(int)));
        double ms = run_bfs_cuda(d_row_ptr, d_col_idx, cfg.n,
                                 d_distances, d_frontier, d_next_frontier,
                                 d_flags, ref.n_levels);
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

    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_frontier));
    CUDA_CHECK(cudaFree(d_next_frontier));
    CUDA_CHECK(cudaFree(d_flags));
    return 0;
}
