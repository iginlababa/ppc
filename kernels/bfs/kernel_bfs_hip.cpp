// kernel_bfs_hip.cpp — E6 BFS: native HIP with atomic-counter compact.
//
// Mechanical port of kernel_bfs_cuda.cu.  Changes:
//   - #include <hip/hip_runtime.h> instead of cuda_runtime.h
//   - HIP_CHECK macro replaces CUDA_CHECK
//   - thrust::copy_if replaced by bfs_compact_kernel (atomicAdd counter)
//     avoids ROCThrust/rocprim header conflicts on ROCm 6.x installs.
//   - Kernel body (bfs_scatter_kernel) unchanged:
//     __global__, blockIdx.x, threadIdx.x, atomicCAS all have identical HIP syntax.
//
// ── E6 DESIGN DECISIONS ───────────────────────────────────────────────────────
// [D3-HIP] Two phases per BFS level:
//   Scatter: bfs_scatter_kernel — one thread per frontier vertex.
//            atomicCAS on d_distances[v]; sets d_flags[v]=1 on first discovery.
//   Compact: bfs_compact_kernel — one thread per vertex in [0,N).
//            atomicAdd on d_next_size counter; writes flagged vertices to
//            d_next_frontier.  hipDeviceSynchronize() after each phase.
//            After compact: hipMemset d_flags to 0.
// [D5-HIP] Metric: GTEPS = n_edges / time_s / 1e9.  Stored in throughput_gflops.
// [D7-HIP] d_distances reset to -1 (source = 0) excluded from timed region.
// ─────────────────────────────────────────────────────────────────────────────

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "bfs_common.h"

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t _e = (call);                                                \
        if (_e != hipSuccess) {                                                \
            std::fprintf(stderr, "HIP error %s:%d: %s\n",                     \
                         __FILE__, __LINE__, hipGetErrorString(_e));           \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// ── Scatter kernel ────────────────────────────────────────────────────────────
// Identical body to CUDA version — HIP supports __global__, blockIdx.x,
// threadIdx.x, and atomicCAS with the same syntax.
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

// ── Compact kernel ────────────────────────────────────────────────────────────
// One thread per vertex: if d_flags[v] != 0, atomicAdd to get a slot in
// d_next_frontier.  Replaces thrust::copy_if to avoid ROCThrust header issues.
__global__ void bfs_compact_kernel(
    const int* __restrict__ d_flags,
    int                     n_vertices,
    int* __restrict__       d_next_frontier,
    int*                    d_next_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_vertices) return;
    if (d_flags[tid]) {
        int pos = atomicAdd(d_next_size, 1);
        d_next_frontier[pos] = tid;
    }
}

// ── Main BFS driver ───────────────────────────────────────────────────────────
static double run_bfs_hip(
    const int*  d_row_ptr,
    const int*  d_col_idx,
    int         n_vertices,
    int*        d_distances,
    int*        d_frontier,
    int*        d_next_frontier,
    int*        d_flags,
    int*        d_next_size,
    int         /* n_expected_levels */)
{
    int frontier_size = 1;
    HIP_CHECK(hipMemcpy(d_frontier, &BFS_SOURCE, sizeof(int),
                        hipMemcpyHostToDevice));

    auto t0 = std::chrono::high_resolution_clock::now();

    int level = 1;
    int comp_blocks = (n_vertices + BFS_BLOCK_SIZE - 1) / BFS_BLOCK_SIZE;
    while (frontier_size > 0) {
        // Scatter phase
        int blocks = (frontier_size + BFS_BLOCK_SIZE - 1) / BFS_BLOCK_SIZE;
        bfs_scatter_kernel<<<blocks, BFS_BLOCK_SIZE>>>(
            d_frontier, frontier_size,
            d_row_ptr, d_col_idx,
            d_distances, d_flags, level);
        HIP_CHECK(hipDeviceSynchronize());

        // Compact phase — atomic counter, no Thrust dependency
        HIP_CHECK(hipMemset(d_next_size, 0, sizeof(int)));
        bfs_compact_kernel<<<comp_blocks, BFS_BLOCK_SIZE>>>(
            d_flags, n_vertices, d_next_frontier, d_next_size);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(&frontier_size, d_next_size, sizeof(int),
                            hipMemcpyDeviceToHost));

        // Reset flags
        HIP_CHECK(hipMemset(d_flags, 0, n_vertices * sizeof(int)));

        if (frontier_size > 0) {
            HIP_CHECK(hipMemcpy(d_frontier, d_next_frontier,
                                frontier_size * sizeof(int),
                                hipMemcpyDeviceToDevice));
        }
        level++;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ── Reset helper ──────────────────────────────────────────────────────────────
static void reset_distances(int* d_distances, int n_vertices) {
    HIP_CHECK(hipMemset(d_distances, 0xFF, n_vertices * sizeof(int)));  // -1
    int zero = 0;
    HIP_CHECK(hipMemcpy(&d_distances[BFS_SOURCE], &zero, sizeof(int),
                        hipMemcpyHostToDevice));
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

    BfsResult ref = bfs_cpu_ref(g);
    bfs_print_profile(ref, cfg.graph_type, cfg.size_label);

    // ── Allocate device memory ────────────────────────────────────────────────
    int* d_row_ptr;
    int* d_col_idx;
    int* d_distances;
    int* d_frontier;
    int* d_next_frontier;
    int* d_flags;
    int* d_next_size;
    int  n_col = (int)g.col_idx.size();

    HIP_CHECK(hipMalloc(&d_row_ptr,       (cfg.n + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_col_idx,       n_col       * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_distances,     cfg.n       * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_frontier,      cfg.n       * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_next_frontier, cfg.n       * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_flags,         cfg.n       * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_next_size,     sizeof(int)));

    HIP_CHECK(hipMemcpy(d_row_ptr, g.row_ptr.data(), (cfg.n+1)*sizeof(int),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_col_idx, g.col_idx.data(), n_col*sizeof(int),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_flags, 0, cfg.n * sizeof(int)));

    // ── Correctness check ─────────────────────────────────────────────────────
    if (cfg.verify) {
        reset_distances(d_distances, cfg.n);
        run_bfs_hip(d_row_ptr, d_col_idx, cfg.n,
                    d_distances, d_frontier, d_next_frontier, d_flags,
                    d_next_size, ref.n_levels);
        std::vector<int> dist_gpu(cfg.n);
        HIP_CHECK(hipMemcpy(dist_gpu.data(), d_distances, cfg.n*sizeof(int),
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemset(d_flags, 0, cfg.n * sizeof(int)));
        if (!bfs_verify(dist_gpu, ref.distances)) {
            std::fprintf(stderr, "[bfs_hip] CORRECTNESS FAILED\n");
            return 1;
        }
        std::fprintf(stderr, "[bfs_hip] Correctness OK (n=%d n_levels=%d)\n",
                     cfg.n, ref.n_levels);
        if (cfg.reps == 0) return 0;
    }

    // ── Warmup ────────────────────────────────────────────────────────────────
    bfs_adaptive_warmup([&]() -> double {
        reset_distances(d_distances, cfg.n);
        HIP_CHECK(hipMemset(d_flags, 0, cfg.n * sizeof(int)));
        return run_bfs_hip(d_row_ptr, d_col_idx, cfg.n,
                           d_distances, d_frontier, d_next_frontier, d_flags,
                           d_next_size, ref.n_levels);
    }, "hip");

    // ── Timed runs ────────────────────────────────────────────────────────────
    std::vector<double> times;
    times.reserve(cfg.reps);
    for (int r = 0; r < cfg.reps; r++) {
        reset_distances(d_distances, cfg.n);
        HIP_CHECK(hipMemset(d_flags, 0, cfg.n * sizeof(int)));
        double ms = run_bfs_hip(d_row_ptr, d_col_idx, cfg.n,
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

    HIP_CHECK(hipFree(d_row_ptr));
    HIP_CHECK(hipFree(d_col_idx));
    HIP_CHECK(hipFree(d_distances));
    HIP_CHECK(hipFree(d_frontier));
    HIP_CHECK(hipFree(d_next_frontier));
    HIP_CHECK(hipFree(d_flags));
    HIP_CHECK(hipFree(d_next_size));
    return 0;
}
