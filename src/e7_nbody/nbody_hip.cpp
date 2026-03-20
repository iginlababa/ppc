// nbody_hip.cpp — E7 N-Body, native HIP (AMD ROCm).
//
// Mechanically ported from nbody_native.cu:
//   cuda* → hip*  |  cudaEvent_t → hipEvent_t  |  CUDA_CHECK → HIP_CHECK
// Kernel logic is bit-for-bit identical to the CUDA baseline; only the
// runtime API layer differs — isolating hardware effects from abstraction cost.
//
// Only the notile kernel is compiled here (tile kernel is NVIDIA-specific
// due to the 96 KB shmem requirement — MI300X limit differs).
//
// Usage:
//   ./nbody-hip --size <small|medium|large> [--reps N] [--platform STR] [--verify]

#define NBODY_USE_HIP
#include "nbody_common.h"
#include <vector>
#include <cstdio>

// ── notile kernel (identical logic to CUDA baseline) ──────────────────────────
__global__ void nbody_notile_kernel(
    const float4* __restrict__ d_pos,
    float4*       __restrict__ d_force,
    const int*    __restrict__ d_ptr,
    const int*    __restrict__ d_idx,
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

// ── Host driver ───────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    NbodyConfig cfg = nbody_parse_args(argc, argv);
    // Default platform label for AMD if not overridden via --platform
    if (cfg.platform == "nvidia_rtx5060") cfg.platform = "amd_mi300x";
    cfg.kernel = "notile";

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
    HIP_CHECK(hipMalloc(&d_pos,   N * sizeof(float4)));
    HIP_CHECK(hipMalloc(&d_force, N * sizeof(float4)));
    HIP_CHECK(hipMalloc(&d_ptr,   (N + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_idx,   csr.total * sizeof(int)));

    HIP_CHECK(hipMemcpy(d_pos, pos.data(),     N * sizeof(float4),      hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_ptr, csr.ptr.data(), (N + 1) * sizeof(int),   hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_idx, csr.idx.data(), csr.total * sizeof(int), hipMemcpyHostToDevice));

    // Report VRAM
    size_t free_mem = 0, total_mem = 0;
    HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));
    double used_mb = (total_mem - free_mem) / (1024.0 * 1024.0);
    std::printf("NBODY_VRAM used_mb=%.1f\n", used_mb);

    int blocks = (N + NBODY_BLOCK_SIZE - 1) / NBODY_BLOCK_SIZE;

    // Warmup (not timed)
    for (int w = 0; w < NBODY_WARMUP; ++w) {
        nbody_notile_kernel<<<blocks, NBODY_BLOCK_SIZE>>>(
            d_pos, d_force, d_ptr, d_idx, N, box_len);
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Timed runs
    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    std::vector<double> times_ms(cfg.reps);
    for (int rep = 0; rep < cfg.reps; ++rep) {
        HIP_CHECK(hipEventRecord(ev_start));
        nbody_notile_kernel<<<blocks, NBODY_BLOCK_SIZE>>>(
            d_pos, d_force, d_ptr, d_idx, N, box_len);
        HIP_CHECK(hipEventRecord(ev_stop));
        HIP_CHECK(hipEventSynchronize(ev_stop));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, ev_start, ev_stop));
        times_ms[rep] = (double)ms;
        NBODY_PRINT_RUN(rep + 1, cfg, csr, times_ms[rep]);
    }

    int hw = hw_state_check(times_ms);
    NBODY_PRINT_HW(hw);

    // Verification
    if (cfg.verify) {
        std::vector<float4> gpu_force(N);
        HIP_CHECK(hipMemcpy(gpu_force.data(), d_force, N * sizeof(float4), hipMemcpyDeviceToHost));
        auto ref = cpu_ref_forces(pos, csr, box_len);
        float max_rel = verify_forces(ref, gpu_force);
        std::printf("NBODY_VERIFY max_rel_err=%.6f %s\n",
                    max_rel, (max_rel < 1e-3f) ? "PASS" : "FAIL");
    }

    HIP_CHECK(hipFree(d_pos));
    HIP_CHECK(hipFree(d_force));
    HIP_CHECK(hipFree(d_ptr));
    HIP_CHECK(hipFree(d_idx));
    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));
    return 0;
}
