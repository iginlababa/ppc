// nbody_sycl.cpp — E7 N-Body, SYCL/AdaptiveCpp abstraction.
//
// Compile:
//   AMD:  acpp --acpp-targets=hip:gfx942 -O3 -I. nbody_sycl.cpp -o nbody-sycl
//
// Single kernel per rep — no level loop, no multi-launch overhead.
// Taxonomy prediction: SYCL EXCELLENT here (vs. poor in E5/E6).
// This is the direct counterpoint: bulk-parallel workload, one q.wait() per rep.
//
// Uses USM device pointers (sycl::malloc_device) — matches nbody_raja flat layout.
// pos4[N*4]: [x0,y0,z0,w0, x1,...], force3[N*3]: [fx0,fy0,fz0, fx1,...]

#include <sycl/sycl.hpp>
#include "nbody_common.h"

static void run_nbody_sycl(
    sycl::queue&  q,
    const float*  d_pos4,
    float*        d_force3,
    const int*    d_ptr,
    const int*    d_idx,
    int N, float box_len)
{
    const size_t local_sz  = NBODY_BLOCK_SIZE;
    const size_t global_sz = ((N + local_sz - 1) / local_sz) * local_sz;
    const float  hbox      = 0.5f * box_len;

    q.parallel_for(sycl::nd_range<1>(global_sz, local_sz),
        [=](sycl::nd_item<1> item) {
            int i = (int)item.get_global_id(0);
            if (i >= N) return;

            float px = d_pos4[i*4+0];
            float py = d_pos4[i*4+1];
            float pz = d_pos4[i*4+2];
            float fx = 0.0f, fy = 0.0f, fz = 0.0f;

            int st = d_ptr[i], en = d_ptr[i + 1];
            for (int k = st; k < en; ++k) {
                int   j  = d_idx[k];
                float dx = d_pos4[j*4+0] - px;
                float dy = d_pos4[j*4+1] - py;
                float dz = d_pos4[j*4+2] - pz;
                if (dx >  hbox) dx -= box_len; else if (dx < -hbox) dx += box_len;
                if (dy >  hbox) dy -= box_len; else if (dy < -hbox) dy += box_len;
                if (dz >  hbox) dz -= box_len; else if (dz < -hbox) dz += box_len;
                float r2 = dx*dx + dy*dy + dz*dz;
                if (r2 > 0.0f && r2 < NBODY_R_CUT_SQ) {
                    float r2inv = 1.0f / r2;
                    float r6inv = r2inv * r2inv * r2inv;
                    float fs    = 48.0f * r6inv * (r6inv - 0.5f) * r2inv;
                    fx += fs * dx;  fy += fs * dy;  fz += fs * dz;
                }
            }
            d_force3[i*3+0] = fx;
            d_force3[i*3+1] = fy;
            d_force3[i*3+2] = fz;
        });
    q.wait();
}

int main(int argc, char** argv) {
    NbodyConfig cfg = nbody_parse_args(argc, argv);
    cfg.kernel = "notile";

    float box_len = 0.0f;
    auto pos_h = make_fcc(cfg.m_cells, NBODY_FCC_A, &box_len);
    cfg.box_len = box_len;
    int N = (int)pos_h.size();
    cfg.N = N;

    NbodyCSR csr = build_csr(pos_h, box_len);
    assert(csr.max_per_atom < NBODY_MAX_NBRS_CAP);
    NBODY_PRINT_META(cfg, csr);

    sycl::queue q(sycl::gpu_selector_v,
                  sycl::property::queue::in_order{});

    // Flat host arrays
    std::vector<float> h_pos4(N * 4);
    for (int i = 0; i < N; ++i) {
        h_pos4[i*4+0] = pos_h[i].x;
        h_pos4[i*4+1] = pos_h[i].y;
        h_pos4[i*4+2] = pos_h[i].z;
        h_pos4[i*4+3] = 0.0f;
    }

    // USM device allocations
    float* d_pos4   = sycl::malloc_device<float>(N * 4,         q);
    float* d_force3 = sycl::malloc_device<float>(N * 3,         q);
    int*   d_ptr    = sycl::malloc_device<int>  (N + 1,         q);
    int*   d_idx    = sycl::malloc_device<int>  (csr.total,     q);

    q.memcpy(d_pos4, h_pos4.data(),     N * 4 * sizeof(float)).wait();
    q.memcpy(d_ptr,  csr.ptr.data(), (N + 1) * sizeof(int)  ).wait();
    q.memcpy(d_idx,  csr.idx.data(), csr.total * sizeof(int)).wait();
    q.memset(d_force3, 0, N * 3 * sizeof(float)).wait();

    std::printf("NBODY_VRAM used_mb=0.0\n");  // SYCL has no standard query

    // ── Warmup ─────────────────────────────────────────────────────────────────
    for (int w = 0; w < NBODY_WARMUP; ++w)
        run_nbody_sycl(q, d_pos4, d_force3, d_ptr, d_idx, N, box_len);

    // ── Timed runs ─────────────────────────────────────────────────────────────
    std::vector<double> times_ms(cfg.reps);
    for (int rep = 0; rep < cfg.reps; ++rep) {
        auto t0 = std::chrono::high_resolution_clock::now();
        run_nbody_sycl(q, d_pos4, d_force3, d_ptr, d_idx, N, box_len);
        auto t1 = std::chrono::high_resolution_clock::now();
        times_ms[rep] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        NBODY_PRINT_RUN(rep + 1, cfg, csr, times_ms[rep]);
    }

    int hw = hw_state_check(times_ms);
    NBODY_PRINT_HW(hw);

    // ── Verification ───────────────────────────────────────────────────────────
    if (cfg.verify) {
        std::vector<float> gpu_f3(N * 3);
        q.memcpy(gpu_f3.data(), d_force3, N * 3 * sizeof(float)).wait();
        std::vector<float4> gpu_f4(N);
        for (int i = 0; i < N; ++i)
            gpu_f4[i] = {gpu_f3[i*3+0], gpu_f3[i*3+1], gpu_f3[i*3+2], 0.0f};
        auto ref = cpu_ref_forces(pos_h, csr, box_len);
        float max_rel = verify_forces(ref, gpu_f4);
        std::printf("NBODY_VERIFY max_rel_err=%.6f %s\n",
                    max_rel, (max_rel < 1e-3f) ? "PASS" : "FAIL");
    }

    sycl::free(d_pos4,   q);
    sycl::free(d_force3, q);
    sycl::free(d_ptr,    q);
    sycl::free(d_idx,    q);
    return 0;
}
