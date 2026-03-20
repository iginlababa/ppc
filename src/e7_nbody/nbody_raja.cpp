// nbody_raja.cpp — E7 N-Body, RAJA abstraction.
//
// Compile: two-step via CMakeLists.txt / build_nbody.sh
//   nvcc -x cu -dc -arch=sm_120 -allow-unsupported-compiler
//        -I$RAJA_INC nbody_raja.cpp -o nbody_raja.o
//   nvcc -arch=sm_120 nbody_raja.o -L$RAJA_LIB -lRAJA -lcamp -lcuda -o nbody_raja
//
// Uses raw device pointers (float* flat layout) and RAJA::forall<cuda_exec<256>>.
// Note: harmless warning in RAJA/policy/cuda/policy.hpp:1936 with sm_120 (known).

#include "nbody_common.h"
#include <RAJA/RAJA.hpp>
#include <cuda_runtime.h>
#include <vector>

using NbodyExecPolicy = RAJA::cuda_exec<NBODY_BLOCK_SIZE>;

int main(int argc, char** argv) {
    NbodyConfig cfg = nbody_parse_args(argc, argv);
    cfg.kernel = "notile";  // RAJA: single CSR kernel, no tiling variant

    float box_len = 0.0f;
    auto pos_h = make_fcc(cfg.m_cells, NBODY_FCC_A, &box_len);
    cfg.box_len = box_len;
    int N = (int)pos_h.size();
    cfg.N = N;

    NbodyCSR csr = build_csr(pos_h, box_len);
    assert(csr.max_per_atom < NBODY_MAX_NBRS_CAP);
    NBODY_PRINT_META(cfg, csr);

    // Flat host arrays: pos4[N*4], force3[N*3]
    std::vector<float> h_pos4(N * 4);
    for (int i = 0; i < N; ++i) {
        h_pos4[i*4+0] = pos_h[i].x;
        h_pos4[i*4+1] = pos_h[i].y;
        h_pos4[i*4+2] = pos_h[i].z;
        h_pos4[i*4+3] = 0.0f;
    }

    // GPU allocations (raw device pointers)
    float *d_pos4, *d_force3;
    int   *d_ptr, *d_idx;
    CUDA_CHECK(cudaMalloc(&d_pos4,   N * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_force3, N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ptr,    (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_idx,    csr.total * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_pos4, h_pos4.data(),       N * 4 * sizeof(float),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ptr,  csr.ptr.data(),      (N + 1) * sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_idx,  csr.idx.data(),      csr.total * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_force3, 0, N * 3 * sizeof(float)));

    // Report VRAM
    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    std::printf("NBODY_VRAM used_mb=%.1f\n",
                (total_mem - free_mem) / (1024.0 * 1024.0));

    float box  = box_len;
    float hbox = 0.5f * box_len;

    auto run_kernel = [&]() {
        RAJA::forall<NbodyExecPolicy>(
            RAJA::RangeSegment(0, N),
            [=] RAJA_DEVICE (int i) {
                float px = d_pos4[i*4+0];
                float py = d_pos4[i*4+1];
                float pz = d_pos4[i*4+2];
                float fx = 0.0f, fy = 0.0f, fz = 0.0f;
                int st = d_ptr[i], en = d_ptr[i + 1];
                for (int k = st; k < en; ++k) {
                    int j = d_idx[k];
                    float dx = d_pos4[j*4+0] - px;
                    float dy = d_pos4[j*4+1] - py;
                    float dz = d_pos4[j*4+2] - pz;
                    if (dx >  hbox) dx -= box; else if (dx < -hbox) dx += box;
                    if (dy >  hbox) dy -= box; else if (dy < -hbox) dy += box;
                    if (dz >  hbox) dz -= box; else if (dz < -hbox) dz += box;
                    float r2 = dx*dx + dy*dy + dz*dz;
                    if (r2 > 0.0f && r2 < NBODY_R_CUT_SQ) {
                        float r2inv = 1.0f / r2;
                        float r6inv = r2inv * r2inv * r2inv;
                        float fscal = 48.0f * r6inv * (r6inv - 0.5f) * r2inv;
                        fx += fscal * dx;  fy += fscal * dy;  fz += fscal * dz;
                    }
                }
                d_force3[i*3+0] = fx;
                d_force3[i*3+1] = fy;
                d_force3[i*3+2] = fz;
            });
        RAJA::synchronize<RAJA::cuda_synchronize>();
    };

    // ── Warmup ─────────────────────────────────────────────────────────────────
    for (int w = 0; w < NBODY_WARMUP; ++w) run_kernel();

    // ── Timed runs ─────────────────────────────────────────────────────────────
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    std::vector<double> times_ms(cfg.reps);
    for (int rep = 0; rep < cfg.reps; ++rep) {
        CUDA_CHECK(cudaEventRecord(ev_start));
        run_kernel();
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
        std::vector<float> gpu_f3(N * 3);
        CUDA_CHECK(cudaMemcpy(gpu_f3.data(), d_force3, N * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<float4> gpu_f4(N);
        for (int i = 0; i < N; ++i)
            gpu_f4[i] = {gpu_f3[i*3+0], gpu_f3[i*3+1], gpu_f3[i*3+2], 0.0f};
        auto ref = cpu_ref_forces(pos_h, csr, box_len);
        float max_rel = verify_forces(ref, gpu_f4);
        std::printf("NBODY_VERIFY max_rel_err=%.6f %s\n",
                    max_rel, (max_rel < 1e-3f) ? "PASS" : "FAIL");
    }

    CUDA_CHECK(cudaFree(d_pos4));
    CUDA_CHECK(cudaFree(d_force3));
    CUDA_CHECK(cudaFree(d_ptr));
    CUDA_CHECK(cudaFree(d_idx));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    return 0;
}
