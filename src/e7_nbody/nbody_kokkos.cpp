// nbody_kokkos.cpp — E7 N-Body, Kokkos abstraction.
//
// Compile: two-step via CMakeLists.txt / build_nbody.sh
//   NVIDIA: nvcc -x cu -c -std=c++20 --expt-extended-lambda --expt-relaxed-constexpr
//           -arch=sm_120 -I$KOKKOS_INC nbody_kokkos.cpp -o nbody_kokkos.o
//           nvcc -arch=sm_120 nbody_kokkos.o -L$KOKKOS_LIB -lkokkoscore -lcuda -o nbody_kokkos
//   AMD:    hipcc -O3 --offload-arch=gfx942 -std=c++20
//           -DKOKKOS_ENABLE_HIP -DNBODY_USE_HIP -I$KOKKOS_INC nbody_kokkos.cpp -o nbody_kokkos.o
//           hipcc nbody_kokkos.o -L$KOKKOS_LIB -lkokkoscore -lkokkoscontainers -o nbody_kokkos
//
// Views:
//   pos   — float*[4], MemSpace: pos(i,0..2) = x,y,z
//   force — float*[3], MemSpace: force(i,0..2) = fx,fy,fz
//   ptr   — int*,      MemSpace: CSR row pointers
//   idx   — int*,      MemSpace: CSR neighbor indices

#include "nbody_common.h"
#include "gpu_compat.h"
#include <Kokkos_Core.hpp>

#ifdef KOKKOS_ENABLE_HIP
using MemSpace    = Kokkos::HIPSpace;
using ExecSpace   = Kokkos::HIP;
#else
using MemSpace    = Kokkos::CudaSpace;
using ExecSpace   = Kokkos::Cuda;
#endif
using ViewPos     = Kokkos::View<float*[4], MemSpace>;
using ViewForce   = Kokkos::View<float*[3], MemSpace>;
using ViewInt     = Kokkos::View<int*,      MemSpace>;
using RangePolicy = Kokkos::RangePolicy<ExecSpace>;

int main(int argc, char** argv) {
    NbodyConfig cfg = nbody_parse_args(argc, argv);

    // Build FCC lattice + CSR on host
    float box_len = 0.0f;
    auto pos_h = make_fcc(cfg.m_cells, NBODY_FCC_A, &box_len);
    cfg.box_len = box_len;
    int N = (int)pos_h.size();
    cfg.N = N;
    cfg.kernel = "notile";  // Kokkos single-kernel implementation (CSR, no tiling)

    NbodyCSR csr = build_csr(pos_h, box_len);
    assert(csr.max_per_atom < NBODY_MAX_NBRS_CAP);
    NBODY_PRINT_META(cfg, csr);

    Kokkos::initialize(argc, argv);
    {
        // Allocate Kokkos views
        ViewPos   d_pos("pos",   N);
        ViewForce d_force("force", N);
        ViewInt   d_ptr("ptr",   N + 1);
        ViewInt   d_idx("idx",   csr.total);

        // Host mirrors
        auto h_pos   = Kokkos::create_mirror_view(d_pos);
        auto h_ptr   = Kokkos::create_mirror_view(d_ptr);
        auto h_idx   = Kokkos::create_mirror_view(d_idx);

        for (int i = 0; i < N; ++i) {
            h_pos(i, 0) = pos_h[i].x;
            h_pos(i, 1) = pos_h[i].y;
            h_pos(i, 2) = pos_h[i].z;
            h_pos(i, 3) = 0.0f;
        }
        for (int i = 0; i <= N; ++i) h_ptr(i) = csr.ptr[i];
        for (int k = 0; k < csr.total; ++k) h_idx(k) = csr.idx[k];

        Kokkos::deep_copy(d_pos,   h_pos);
        Kokkos::deep_copy(d_ptr,   h_ptr);
        Kokkos::deep_copy(d_idx,   h_idx);

        // Report VRAM
        size_t free_mem = 0, total_mem = 0;
        GPU_CHECK(gpuMemGetInfo(&free_mem, &total_mem));
        std::printf("NBODY_VRAM used_mb=%.1f\n",
                    (total_mem - free_mem) / (1024.0 * 1024.0));

        float box   = box_len;
        float hbox  = 0.5f * box_len;
        int   total = csr.total;  (void)total;

        auto run_kernel = [&]() {
            Kokkos::parallel_for("nbody_kokkos",
                RangePolicy(0, N),
                KOKKOS_LAMBDA(int i) {
                    float px = d_pos(i, 0), py = d_pos(i, 1), pz = d_pos(i, 2);
                    float fx = 0.0f, fy = 0.0f, fz = 0.0f;
                    int st = d_ptr(i), en = d_ptr(i + 1);
                    for (int k = st; k < en; ++k) {
                        int j = d_idx(k);
                        float dx = d_pos(j, 0) - px;
                        float dy = d_pos(j, 1) - py;
                        float dz = d_pos(j, 2) - pz;
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
                    d_force(i, 0) = fx;
                    d_force(i, 1) = fy;
                    d_force(i, 2) = fz;
                });
            Kokkos::fence();
        };

        // ── Warmup ─────────────────────────────────────────────────────────────
        for (int w = 0; w < NBODY_WARMUP; ++w) run_kernel();

        // ── Timed runs ─────────────────────────────────────────────────────────
        gpuEvent_t ev_start, ev_stop;
        GPU_CHECK(gpuEventCreate(&ev_start));
        GPU_CHECK(gpuEventCreate(&ev_stop));

        std::vector<double> times_ms(cfg.reps);
        for (int rep = 0; rep < cfg.reps; ++rep) {
            GPU_CHECK(gpuEventRecord(ev_start));
            run_kernel();
            GPU_CHECK(gpuEventRecord(ev_stop));
            GPU_CHECK(gpuEventSynchronize(ev_stop));
            float ms = 0.0f;
            GPU_CHECK(gpuEventElapsedTime(&ms, ev_start, ev_stop));
            times_ms[rep] = (double)ms;
            NBODY_PRINT_RUN(rep + 1, cfg, csr, times_ms[rep]);
        }

        int hw = hw_state_check(times_ms);
        NBODY_PRINT_HW(hw);

        // ── Verification ───────────────────────────────────────────────────────
        if (cfg.verify) {
            auto h_force = Kokkos::create_mirror_view(d_force);
            Kokkos::deep_copy(h_force, d_force);
            std::vector<float4> gpu_f(N);
            for (int i = 0; i < N; ++i)
                gpu_f[i] = {h_force(i,0), h_force(i,1), h_force(i,2), 0.0f};
            auto ref = cpu_ref_forces(pos_h, csr, box_len);
            float max_rel = verify_forces(ref, gpu_f);
            std::printf("NBODY_VERIFY max_rel_err=%.6f %s\n",
                        max_rel, (max_rel < 1e-3f) ? "PASS" : "FAIL");
        }

        GPU_CHECK(gpuEventDestroy(ev_start));
        GPU_CHECK(gpuEventDestroy(ev_stop));
    }
    Kokkos::finalize();
    return 0;
}
