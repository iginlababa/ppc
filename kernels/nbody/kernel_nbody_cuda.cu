// kernel_nbody_cuda.cu — E7 N-Body native CUDA
// Two kernel variants to test P006 (shared-memory tiling):
//   native_notile: neighbor-list traversal, direct global-memory reads
//   native_tile:   classic all-pairs shared-memory tiling (cooperative warp loading)
//
// Physics: Lennard-Jones pairwise force, r_cut=2.5σ, one-sided accumulation
//          (no Newton's 3rd law, no atomics).  20 FLOPs per pair.
// Positions: float4 (w=0).  Forces: float3.
// Build:  nvcc -O3 -arch=sm_120 --extended-lambda -o bin/nbody/nbody-native kernel_nbody_cuda.cu

#include "nbody_common.h"
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>


// ══════════════════════════════════════════════════════════════════════════════
// Kernel 1 — notile: each thread processes one particle, reads neighbors from
// the prebuilt neighbor list, fetches positions from global memory directly.
// ══════════════════════════════════════════════════════════════════════════════
__global__ void nbody_notile_kernel(
    const float4* __restrict__ d_pos,
    const int*    __restrict__ d_neighbors,   // [N * MAX_NEIGHBORS]
    const int*    __restrict__ d_n_neighbors, // [N]
    float3*       __restrict__ d_forces,
    int N, float box_len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float4 pi     = d_pos[i];
    float  fx     = 0.0f, fy = 0.0f, fz = 0.0f;
    int    nni    = d_n_neighbors[i];
    float  hbox   = 0.5f * box_len;
    const int* nbrs_i = d_neighbors + (long long)i * MAX_NEIGHBORS;

    for (int k = 0; k < nni; ++k) {
        int j = nbrs_i[k];
        float4 pj = d_pos[j];
        float dx = pj.x - pi.x;
        float dy = pj.y - pi.y;
        float dz = pj.z - pi.z;
        // Minimum image convention (periodic boundary)
        if (dx >  hbox) dx -= box_len; else if (dx < -hbox) dx += box_len;
        if (dy >  hbox) dy -= box_len; else if (dy < -hbox) dy += box_len;
        if (dz >  hbox) dz -= box_len; else if (dz < -hbox) dz += box_len;
        float r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < R_CUT2 && r2 > 1e-10f) {
            float inv_r2  = 1.0f / r2;
            float inv_r6  = inv_r2 * inv_r2 * inv_r2;
            float inv_r12 = inv_r6 * inv_r6;
            float f_mag   = 24.0f * LJ_EPSILON * (2.0f * inv_r12 - inv_r6) * inv_r2;
            fx += f_mag * dx;
            fy += f_mag * dy;
            fz += f_mag * dz;
        }
    }
    d_forces[i] = {fx, fy, fz};
}


// ══════════════════════════════════════════════════════════════════════════════
// Kernel 2 — tile: classic all-pairs shared-memory tiling (P006 demonstration).
// A warp cooperatively loads TILE_SIZE=32 particle positions into __shared__
// memory per tile; every thread computes LJ forces with those TILE_SIZE particles
// (applying r_cut check).  Each j-position is loaded once per warp vs. up to
// 32 times in the notile case → data reuse ratio = TILE_SIZE.
//
// Complexity: O(N²) per launch — use only when N is small enough.
// For N=256 K: ~0.1 s/launch on RTX 5060 (dominated by 65 B comparisons).
// ══════════════════════════════════════════════════════════════════════════════
__global__ void nbody_tile_kernel(
    const float4* __restrict__ d_pos,
    float3*       __restrict__ d_forces,
    int N, float box_len)
{
    // One shared-memory tile per warp (TILE_SIZE float4 = 512 bytes per warp).
    // NBODY_BLOCK_SIZE=256, 8 warps per block → 8 × 512 = 4096 bytes per block.
    extern __shared__ float4 sh_tile[];   // [warps_per_block * TILE_SIZE]

    int i        = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id  = threadIdx.x / TILE_SIZE;
    int lane     = threadIdx.x % TILE_SIZE;

    float4 pi   = (i < N) ? d_pos[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float  hbox = 0.5f * box_len;
    float4* warp_sh = sh_tile + warp_id * TILE_SIZE;

    float fx = 0.0f, fy = 0.0f, fz = 0.0f;

    int n_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < n_tiles; ++t) {
        // Cooperative load: each lane loads one j-particle position into shared.
        int j_load = t * TILE_SIZE + lane;
        warp_sh[lane] = (j_load < N) ? d_pos[j_load] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        __syncwarp();

        if (i < N) {
            int tile_size = min(TILE_SIZE, N - t * TILE_SIZE);
            for (int k = 0; k < tile_size; ++k) {
                int j = t * TILE_SIZE + k;
                if (j == i) continue;
                float4 pj = warp_sh[k];
                float dx = pj.x - pi.x;
                float dy = pj.y - pi.y;
                float dz = pj.z - pi.z;
                // Minimum image convention
                if (dx >  hbox) dx -= box_len; else if (dx < -hbox) dx += box_len;
                if (dy >  hbox) dy -= box_len; else if (dy < -hbox) dy += box_len;
                if (dz >  hbox) dz -= box_len; else if (dz < -hbox) dz += box_len;
                float r2 = dx*dx + dy*dy + dz*dz;
                if (r2 < R_CUT2 && r2 > 1e-10f) {
                    float inv_r2  = 1.0f / r2;
                    float inv_r6  = inv_r2 * inv_r2 * inv_r2;
                    float inv_r12 = inv_r6 * inv_r6;
                    float f_mag   = 24.0f * LJ_EPSILON * (2.0f * inv_r12 - inv_r6) * inv_r2;
                    fx += f_mag * dx;
                    fy += f_mag * dy;
                    fz += f_mag * dz;
                }
            }
        }
        __syncwarp();
    }
    if (i < N) d_forces[i] = {fx, fy, fz};
}


// ══════════════════════════════════════════════════════════════════════════════
// CPU reference BFS for verification
// ══════════════════════════════════════════════════════════════════════════════
static void compute_forces_cpu_ref(
    const std::vector<float4>& pos,
    const NeighborList& nl,
    std::vector<float3>& forces,
    float box_len)
{
    int N = (int)pos.size();
    forces.assign(N, {0.0f, 0.0f, 0.0f});
    for (int i = 0; i < N; ++i) {
        float4 pi = pos[i];
        float fx = 0.0f, fy = 0.0f, fz = 0.0f;
        for (int k = 0; k < nl.count[i]; ++k) {
            int j = nl.idx[(long long)i * MAX_NEIGHBORS + k];
            float dx = min_image(pos[j].x - pi.x, box_len);
            float dy = min_image(pos[j].y - pi.y, box_len);
            float dz = min_image(pos[j].z - pi.z, box_len);
            float r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < R_CUT2 && r2 > 1e-10f) {
                float inv_r2  = 1.0f / r2;
                float inv_r6  = inv_r2 * inv_r2 * inv_r2;
                float inv_r12 = inv_r6 * inv_r6;
                float f_mag   = 24.0f * LJ_EPSILON * (2.0f * inv_r12 - inv_r6) * inv_r2;
                fx += f_mag * dx;
                fy += f_mag * dy;
                fz += f_mag * dz;
            }
        }
        forces[i] = {fx, fy, fz};
    }
}

static bool verify_forces(const std::vector<float3>& ref,
                           const std::vector<float3>& gpu,
                           float abs_tol = 1e-2f, float rel_tol = 1e-3f)
{
    int N = (int)ref.size();
    int n_err = 0;
    for (int i = 0; i < N; ++i) {
        float ex = std::fabs(ref[i].x - gpu[i].x);
        float ey = std::fabs(ref[i].y - gpu[i].y);
        float ez = std::fabs(ref[i].z - gpu[i].z);
        float err_mag = std::sqrt(ex*ex + ey*ey + ez*ez);
        float ref_mag = std::sqrt(ref[i].x*ref[i].x + ref[i].y*ref[i].y + ref[i].z*ref[i].z);
        // Accept if absolute error < abs_tol OR relative error < rel_tol
        bool ok = (err_mag < abs_tol) ||
                  (ref_mag > abs_tol && err_mag / ref_mag < rel_tol);
        if (!ok) {
            if (n_err < 5)
                fprintf(stderr, "  verify FAIL i=%d: cpu=(%.4f,%.4f,%.4f) gpu=(%.4f,%.4f,%.4f) abs=%.4f\n",
                        i, ref[i].x, ref[i].y, ref[i].z, gpu[i].x, gpu[i].y, gpu[i].z, err_mag);
            ++n_err;
        }
    }
    return n_err == 0;
}


// ══════════════════════════════════════════════════════════════════════════════
// Host driver — allocate, warmup, benchmark, report
// ══════════════════════════════════════════════════════════════════════════════
static NBodyRunResult run_notile(
    const NBodyConfig&   cfg,
    float4*              d_pos,
    int*                 d_neighbors,
    int*                 d_n_neighbors,
    float3*              d_forces,
    const NeighborStats& nbr_stats)
{
    int N = cfg.n_atoms;
    int blocks = (N + NBODY_BLOCK_SIZE - 1) / NBODY_BLOCK_SIZE;
    long long total_flops = nbr_stats.total * FLOPS_PER_PAIR;

    float box_len = cfg.box_len;

    // Warmup (E1 protocol: 50 fixed iterations, not recorded)
    for (int w = 0; w < NBODY_WARMUP; ++w)
        nbody_notile_kernel<<<blocks, NBODY_BLOCK_SIZE>>>(
            d_pos, d_neighbors, d_n_neighbors, d_forces, N, box_len);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    std::vector<double> times_ms(cfg.reps);
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    for (int r = 0; r < cfg.reps; ++r) {
        CUDA_CHECK(cudaEventRecord(t0));
        nbody_notile_kernel<<<blocks, NBODY_BLOCK_SIZE>>>(
            d_pos, d_neighbors, d_n_neighbors, d_forces, N, box_len);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        times_ms[r] = (double)ms;
    }
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));

    // Median time
    std::vector<double> sorted = times_ms;
    std::sort(sorted.begin(), sorted.end());
    double median_ms = sorted[sorted.size() / 2];

    NBodyRunResult res;
    res.time_ms      = median_ms;
    res.actual_flops = total_flops;
    res.gflops       = (median_ms > 0.0) ? (double)total_flops / (median_ms * 1e6) : 0.0;
    res.hw_state     = check_hw_state(times_ms);

    // Print all timed runs
    for (int r = 0; r < cfg.reps; ++r) {
        double g = (times_ms[r] > 0.0) ? (double)total_flops / (times_ms[r] * 1e6) : 0.0;
        NBodyRunResult rr { times_ms[r], g, total_flops, res.hw_state };
        NBODY_PRINT_RUN(r + 1, cfg, rr);
    }
    return res;
}

static NBodyRunResult run_tile(
    const NBodyConfig&   cfg,
    float4*              d_pos,
    float3*              d_forces,
    const NeighborStats& nbr_stats)
{
    int N = cfg.n_atoms;
    int blocks = (N + NBODY_BLOCK_SIZE - 1) / NBODY_BLOCK_SIZE;
    int warps_per_block = NBODY_BLOCK_SIZE / TILE_SIZE;
    size_t sh_bytes = warps_per_block * TILE_SIZE * sizeof(float4);
    // all-pairs FLOPs = N*(N-1)*FLOPS_PER_PAIR (one-sided, no Newton's 3rd in kernel)
    long long all_pairs_flops = (long long)N * (N - 1) * FLOPS_PER_PAIR;

    float box_len2 = cfg.box_len;

    // Warmup
    for (int w = 0; w < NBODY_WARMUP; ++w)
        nbody_tile_kernel<<<blocks, NBODY_BLOCK_SIZE, sh_bytes>>>(d_pos, d_forces, N, box_len2);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    std::vector<double> times_ms(cfg.reps);
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    for (int r = 0; r < cfg.reps; ++r) {
        CUDA_CHECK(cudaEventRecord(t0));
        nbody_tile_kernel<<<blocks, NBODY_BLOCK_SIZE, sh_bytes>>>(d_pos, d_forces, N, box_len2);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        times_ms[r] = (double)ms;
    }
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));

    std::vector<double> sorted = times_ms;
    std::sort(sorted.begin(), sorted.end());
    double median_ms = sorted[sorted.size() / 2];

    NBodyRunResult res;
    res.time_ms      = median_ms;
    res.actual_flops = all_pairs_flops;
    res.gflops       = (median_ms > 0.0) ? (double)all_pairs_flops / (median_ms * 1e6) : 0.0;
    res.hw_state     = check_hw_state(times_ms);

    for (int r = 0; r < cfg.reps; ++r) {
        double g = (times_ms[r] > 0.0) ? (double)all_pairs_flops / (times_ms[r] * 1e6) : 0.0;
        NBodyRunResult rr { times_ms[r], g, all_pairs_flops, res.hw_state };
        NBODY_PRINT_RUN(r + 1, cfg, rr);
    }
    return res;
}


// ══════════════════════════════════════════════════════════════════════════════
// main
// ══════════════════════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
    NBodyConfig cfg = nbody_parse_args(argc, argv);

    // Generate FCC lattice
    float box_len = 0.0f;
    std::vector<float4> h_pos = generate_fcc(cfg.m_cells, FCC_A, &box_len);
    cfg.box_len = box_len;
    int N = (int)h_pos.size();
    cfg.n_atoms = N;
    fprintf(stderr, "[nbody-native] kernel=%s  N=%d  M=%d  box=%.4f  reps=%d\n",
            cfg.kernel.c_str(), N, cfg.m_cells, box_len, cfg.reps);

    // Build neighbor list (CPU)
    NeighborList nl = build_neighbor_list(h_pos, box_len);
    NBODY_PRINT_STATS(cfg, nl.stats);

    // GPU allocations
    float4* d_pos        = nullptr;
    int*    d_neighbors  = nullptr;
    int*    d_n_nbrs     = nullptr;
    float3* d_forces     = nullptr;

    CUDA_CHECK(cudaMalloc(&d_pos,       (size_t)N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_neighbors, (long long)N * MAX_NEIGHBORS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_n_nbrs,    (size_t)N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_forces,    (size_t)N * sizeof(float3)));

    CUDA_CHECK(cudaMemcpy(d_pos,       h_pos.data(),      (size_t)N * sizeof(float4),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighbors, nl.idx.data(),     (long long)N * MAX_NEIGHBORS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_n_nbrs,    nl.count.data(),   (size_t)N * sizeof(int),     cudaMemcpyHostToDevice));

    // VRAM report
    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    fprintf(stderr, "[nbody-native] VRAM: used=%.1f MB  free=%.1f MB  total=%.1f MB\n",
            (total_mem - free_mem) / 1048576.0,
            free_mem               / 1048576.0,
            total_mem              / 1048576.0);

    // Correctness check (--verify flag)
    if (cfg.verify) {
        fprintf(stderr, "[nbody-native] Running correctness check...\n");
        std::vector<float3> ref_forces;
        compute_forces_cpu_ref(h_pos, nl, ref_forces, box_len);

        // Run GPU notile
        int blocks = (N + NBODY_BLOCK_SIZE - 1) / NBODY_BLOCK_SIZE;
        CUDA_CHECK(cudaMemset(d_forces, 0, N * sizeof(float3)));
        nbody_notile_kernel<<<blocks, NBODY_BLOCK_SIZE>>>(d_pos, d_neighbors, d_n_nbrs, d_forces, N, box_len);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float3> gpu_forces(N);
        CUDA_CHECK(cudaMemcpy(gpu_forces.data(), d_forces, N * sizeof(float3), cudaMemcpyDeviceToHost));
        bool ok_notile = verify_forces(ref_forces, gpu_forces);
        fprintf(stderr, "[nbody-native] notile correctness: %s\n", ok_notile ? "PASS" : "FAIL");
        if (!ok_notile) { cudaFree(d_pos); cudaFree(d_neighbors); cudaFree(d_n_nbrs); cudaFree(d_forces); return 1; }
    }

    // Run benchmark
    NBodyRunResult res;
    if (cfg.kernel == "tile") {
        res = run_tile(cfg, d_pos, d_forces, nl.stats);
    } else {
        res = run_notile(cfg, d_pos, d_neighbors, d_n_nbrs, d_forces, nl.stats);
    }

    NBODY_PRINT_HW_STATE(res.hw_state);

    cudaFree(d_pos);
    cudaFree(d_neighbors);
    cudaFree(d_n_nbrs);
    cudaFree(d_forces);
    return 0;
}
