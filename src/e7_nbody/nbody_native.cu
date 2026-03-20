// nbody_native.cu — E7 N-Body, native CUDA.
//
// Kernels:
//   nbody_notile_kernel — one thread per particle, direct global reads, CSR neighbor list
//   nbody_tile_kernel   — same, but stages TILE_SIZE=32 neighbor positions per-thread
//                         into dynamic shared memory (BLOCK_SIZE × TILE_SIZE × 16 = 128 KB)
//
// Usage:
//   ./nbody_native --size <small|medium|large> --kernel <notile|tile>
//                  [--reps N] [--platform STR] [--verify]

#include "nbody_common.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>

// ── notile kernel ──────────────────────────────────────────────────────────────
// Direct global reads: d_pos[j] for each CSR neighbor j of particle i.
// Minimum-image convention applied before r² check.
__global__ void nbody_notile_kernel(
    const float4* __restrict__ d_pos,     // [N] positions, .w unused
    float4*       __restrict__ d_force,   // [N] forces, .w=0
    const int*    __restrict__ d_ptr,     // [N+1] CSR row pointers
    const int*    __restrict__ d_idx,     // [total] CSR neighbor indices
    int N, float box_len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float4 pi = d_pos[i];
    float fx = 0.0f, fy = 0.0f, fz = 0.0f;
    float hbox = 0.5f * box_len;

    int start = d_ptr[i];
    int end   = d_ptr[i + 1];
    for (int k = start; k < end; ++k) {
        int    j  = d_idx[k];
        float4 pj = d_pos[j];
        float dx = pj.x - pi.x;  float dy = pj.y - pi.y;  float dz = pj.z - pi.z;
        // minimum image
        if (dx >  hbox) dx -= box_len; else if (dx < -hbox) dx += box_len;
        if (dy >  hbox) dy -= box_len; else if (dy < -hbox) dy += box_len;
        if (dz >  hbox) dz -= box_len; else if (dz < -hbox) dz += box_len;
        float r2 = dx*dx + dy*dy + dz*dz;
        if (r2 > 0.0f && r2 < NBODY_R_CUT_SQ) {
            float r2inv = 1.0f / r2;
            float r6inv = r2inv * r2inv * r2inv;
            float fscal = 48.0f * r6inv * (r6inv - 0.5f) * r2inv;
            fx += fscal * dx;  fy += fscal * dy;  fz += fscal * dz;
        }
    }
    d_force[i] = {fx, fy, fz, 0.0f};
}

// ── tile kernel ────────────────────────────────────────────────────────────────
// Per-thread staging: each thread loads its own slice of TILE_SIZE=24 neighbor
// positions into dynamic shared memory before computing forces on them.
// Layout: sh[tid * TILE_SIZE .. (tid+1)*TILE_SIZE - 1]
// Total: BLOCK_SIZE * TILE_SIZE * sizeof(float4) = 256 * 24 * 16 = 98304 bytes (96 KB).
// RTX 5060 Laptop sharedMemPerBlockOptin = 101376 (99 KB); 96 KB fits.
// Requires cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304).
__global__ void nbody_tile_kernel(
    const float4* __restrict__ d_pos,
    float4*       __restrict__ d_force,
    const int*    __restrict__ d_ptr,
    const int*    __restrict__ d_idx,
    int N, float box_len)
{
    extern __shared__ float4 sh[];

    int tid  = threadIdx.x;
    int i    = blockIdx.x * blockDim.x + tid;
    float4* my_sh = sh + tid * NBODY_TILE_SIZE;

    float fx = 0.0f, fy = 0.0f, fz = 0.0f;
    float hbox = 0.5f * box_len;
    float4 pi = {0,0,0,0};
    int start = 0, end = 0;

    if (i < N) {
        pi    = d_pos[i];
        start = d_ptr[i];
        end   = d_ptr[i + 1];
    }

    // Process CSR neighbors in tiles of TILE_SIZE
    for (int base = start; base < end; base += NBODY_TILE_SIZE) {
        int tile_len = min(NBODY_TILE_SIZE, end - base);

        // Stage tile_len neighbor positions into per-thread shared memory
        for (int t = 0; t < tile_len; ++t) {
            int j      = d_idx[base + t];
            my_sh[t]   = d_pos[j];
        }

        // Compute forces from staged positions
        for (int t = 0; t < tile_len; ++t) {
            float4 pj = my_sh[t];
            float dx = pj.x - pi.x;  float dy = pj.y - pi.y;  float dz = pj.z - pi.z;
            if (dx >  hbox) dx -= box_len; else if (dx < -hbox) dx += box_len;
            if (dy >  hbox) dy -= box_len; else if (dy < -hbox) dy += box_len;
            if (dz >  hbox) dz -= box_len; else if (dz < -hbox) dz += box_len;
            float r2 = dx*dx + dy*dy + dz*dz;
            if (r2 > 0.0f && r2 < NBODY_R_CUT_SQ) {
                float r2inv = 1.0f / r2;
                float r6inv = r2inv * r2inv * r2inv;
                float fscal = 48.0f * r6inv * (r6inv - 0.5f) * r2inv;
                fx += fscal * dx;  fy += fscal * dy;  fz += fscal * dz;
            }
        }
    }

    if (i < N)
        d_force[i] = {fx, fy, fz, 0.0f};
}

// ── Host driver ────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    NbodyConfig cfg = nbody_parse_args(argc, argv);

    // Build FCC lattice + CSR
    float box_len = 0.0f;
    auto pos = make_fcc(cfg.m_cells, NBODY_FCC_A, &box_len);
    cfg.box_len = box_len;
    int N = (int)pos.size();
    cfg.N = N;

    NbodyCSR csr = build_csr(pos, box_len);
    assert(csr.max_per_atom < NBODY_MAX_NBRS_CAP);
    NBODY_PRINT_META(cfg, csr);

    // GPU allocations
    float4 *d_pos, *d_force;
    int    *d_ptr, *d_idx;
    CUDA_CHECK(cudaMalloc(&d_pos,   N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_force, N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_ptr,   (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_idx,   csr.total * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_pos, pos.data(),       N * sizeof(float4),           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ptr, csr.ptr.data(),   (N + 1) * sizeof(int),        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_idx, csr.idx.data(),   csr.total * sizeof(int),      cudaMemcpyHostToDevice));

    // Report VRAM
    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    double used_mb = (total_mem - free_mem) / (1024.0 * 1024.0);
    std::printf("NBODY_VRAM used_mb=%.1f\n", used_mb);

    int  blocks = (N + NBODY_BLOCK_SIZE - 1) / NBODY_BLOCK_SIZE;
    bool do_tile = (cfg.kernel == "tile");

    if (do_tile) {
        size_t shmem = (size_t)NBODY_BLOCK_SIZE * NBODY_TILE_SIZE * sizeof(float4); // 128 KB
        CUDA_CHECK(cudaFuncSetAttribute(
            nbody_tile_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)shmem));
    }

    // ── Warmup (NBODY_WARMUP = 50, not timed) ──────────────────────────────────
    for (int w = 0; w < NBODY_WARMUP; ++w) {
        if (do_tile) {
            size_t shmem = (size_t)NBODY_BLOCK_SIZE * NBODY_TILE_SIZE * sizeof(float4);
            nbody_tile_kernel<<<blocks, NBODY_BLOCK_SIZE, shmem>>>(
                d_pos, d_force, d_ptr, d_idx, N, box_len);
        } else {
            nbody_notile_kernel<<<blocks, NBODY_BLOCK_SIZE>>>(
                d_pos, d_force, d_ptr, d_idx, N, box_len);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Timed runs ─────────────────────────────────────────────────────────────
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    std::vector<double> times_ms(cfg.reps);
    for (int rep = 0; rep < cfg.reps; ++rep) {
        CUDA_CHECK(cudaEventRecord(ev_start));
        if (do_tile) {
            size_t shmem = (size_t)NBODY_BLOCK_SIZE * NBODY_TILE_SIZE * sizeof(float4);
            nbody_tile_kernel<<<blocks, NBODY_BLOCK_SIZE, shmem>>>(
                d_pos, d_force, d_ptr, d_idx, N, box_len);
        } else {
            nbody_notile_kernel<<<blocks, NBODY_BLOCK_SIZE>>>(
                d_pos, d_force, d_ptr, d_idx, N, box_len);
        }
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        times_ms[rep] = (double)ms;
        NBODY_PRINT_RUN(rep + 1, cfg, csr, times_ms[rep]);
    }

    int hw = hw_state_check(times_ms);
    NBODY_PRINT_HW(hw);

    // ── Verification ───────────────────────────────────────────────────────────
    if (cfg.verify) {
        std::vector<float4> gpu_force(N);
        CUDA_CHECK(cudaMemcpy(gpu_force.data(), d_force, N * sizeof(float4), cudaMemcpyDeviceToHost));
        auto ref = cpu_ref_forces(pos, csr, box_len);
        float max_rel = verify_forces(ref, gpu_force);
        std::printf("NBODY_VERIFY max_rel_err=%.6f %s\n",
                    max_rel, (max_rel < 1e-3f) ? "PASS" : "FAIL");
    }

    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_force));
    CUDA_CHECK(cudaFree(d_ptr));
    CUDA_CHECK(cudaFree(d_idx));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    return 0;
}
