// nbody_common.h — E7 N-Body shared constants, FCC lattice, CSR neighbor list,
//                  hw_state logic, and CSV output helpers.
//
// Included by all C++ abstraction implementations (native, Kokkos, RAJA).
// Julia carries equivalent definitions in-file.
//
// ── E7 DESIGN DECISIONS ───────────────────────────────────────────────────────
// [D1] Problem sizes: small=4000, medium=32000, large=256000 atoms.
//      FCC lattice: N = 4*M^3, M=10/20/40. ρ=0.8442 → a=(4/ρ)^{1/3}=1.6796 σ.
// [D2] Neighbor list: CSR layout (neigh_ptr[N+1], neigh_idx[N×54]).
//      NOT dense MAX_NBRS×N — avoids 512 MB at N=256K.
//      FCC with r_cut=2.5σ → exactly 54 neighbors per particle.
//      Positions jittered by ±0.01σ (uniform) before list construction.
//      Assert max_nbrs_per_atom < 512 before GPU upload.
// [D3] Physics: one-sided LJ force, no Newton's 3rd law, no atomics.
//      fscal = 48*r6inv*(r6inv - 0.5)*r2inv (standard LJ, 20 FLOPs/pair).
//      Minimum-image convention applied for periodic boundaries.
// [D4] Warmup: E1 protocol — 50 fixed iterations, not timed.
//      Rationale: N-body is bandwidth-bound (AI≈0.975 FLOP/byte), single
//      kernel per rep, kernel duration >> dispatch latency at N≥32K.
// [D5] hw_state_verified: 1 if max time deviation < 15% of median, else 0.
// ─────────────────────────────────────────────────────────────────────────────

#pragma once

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// Include the appropriate GPU runtime header.
// Compile with -DNBODY_USE_HIP to select the HIP path.
#ifdef NBODY_USE_HIP
#  include <hip/hip_runtime.h>
#else
#  include <cuda_runtime.h>
#endif

// ── Physical constants ─────────────────────────────────────────────────────────
static constexpr float NBODY_R_CUT      = 2.5f;
static constexpr float NBODY_R_CUT_SQ   = NBODY_R_CUT * NBODY_R_CUT;  // 6.25
static constexpr float NBODY_FCC_RHO    = 0.8442f;
static constexpr float NBODY_FCC_A      = 1.6796f;   // (4/rho)^{1/3}, precomputed
static constexpr float NBODY_JITTER     = 0.01f;     // ±0.01 σ position perturbation

// ── Kernel tuning ──────────────────────────────────────────────────────────────
static constexpr int NBODY_BLOCK_SIZE   = 256;
static constexpr int NBODY_TILE_SIZE    = 24;        // shared-memory tile width
// NOTE: BLOCK_SIZE × TILE_SIZE × sizeof(float4) must fit in sharedMemPerBlockOptin.
//       RTX 5060: sharedMemPerBlockOptin = 101376 bytes (≈99 KB).
//       256 × 24 × 16 = 98304 bytes < 101376. (32 would exceed limit on this GPU.)
static constexpr int NBODY_WARMUP       = 50;        // E1 fixed warmup iterations
static constexpr int NBODY_FLOPS_PAIR   = 20;        // canonical LJ FLOP count
static constexpr int NBODY_MAX_NBRS_CAP = 512;       // assert guard before GPU upload

// ── Problem sizes ──────────────────────────────────────────────────────────────
static constexpr int NBODY_N_SMALL      =   4000;    // M=10
static constexpr int NBODY_N_MEDIUM     =  32000;    // M=20
static constexpr int NBODY_N_LARGE      = 256000;    // M=40

// ── GPU error helpers ─────────────────────────────────────────────────────────
#ifdef NBODY_USE_HIP
#define HIP_CHECK(call)                                                            \
    do {                                                                           \
        hipError_t _e = (call);                                                    \
        if (_e != hipSuccess) {                                                    \
            std::fprintf(stderr, "HIP error at %s:%d — %s\n",                     \
                         __FILE__, __LINE__, hipGetErrorString(_e));               \
            std::exit(EXIT_FAILURE);                                               \
        }                                                                          \
    } while (0)
// Alias so shared code can use CUDA_CHECK unchanged when porting.
#define CUDA_CHECK HIP_CHECK
#else
#define CUDA_CHECK(call)                                                           \
    do {                                                                           \
        cudaError_t _e = (call);                                                   \
        if (_e != cudaSuccess) {                                                   \
            std::fprintf(stderr, "CUDA error at %s:%d — %s\n",                    \
                         __FILE__, __LINE__, cudaGetErrorString(_e));              \
            std::exit(EXIT_FAILURE);                                               \
        }                                                                          \
    } while (0)
#endif

// ── CSR neighbor list ──────────────────────────────────────────────────────────
struct NbodyCSR {
    std::vector<int> ptr;    // [N+1], prefix sums of per-particle counts
    std::vector<int> idx;    // [N × ~54], flat neighbor indices (0-based)
    int              N;
    int              total;  // sum of all neighbor counts
    int              max_per_atom;
    double           mean_per_atom;
};

// ── Config parsed from CLI ─────────────────────────────────────────────────────
struct NbodyConfig {
    int         N;
    int         m_cells;
    float       box_len;
    int         reps;
    bool        verify;
    std::string platform;
    std::string kernel;      // "notile" or "tile" (native only)
    std::string size_label;  // "small"|"medium"|"large"
};

// ── Minimum image convention ────────────────────────────────────────────────────
static inline float mi(float dx, float L) {
    if (dx >  0.5f * L) dx -= L;
    if (dx < -0.5f * L) dx += L;
    return dx;
}

// ── FCC lattice generation ──────────────────────────────────────────────────────
// Returns positions as [N] float4 (w=0).  Applies ±NBODY_JITTER uniform jitter.
// FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) × a.
static std::vector<float4> make_fcc(int m_cells, float a, float* out_box = nullptr) {
    const float bx[4] = {0.0f, 0.5f, 0.5f, 0.0f};
    const float by[4] = {0.0f, 0.5f, 0.0f, 0.5f};
    const float bz[4] = {0.0f, 0.0f, 0.5f, 0.5f};

    int N = 4 * m_cells * m_cells * m_cells;
    std::vector<float4> pos(N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> jit(-NBODY_JITTER, NBODY_JITTER);

    int k = 0;
    for (int ix = 0; ix < m_cells; ++ix)
    for (int iy = 0; iy < m_cells; ++iy)
    for (int iz = 0; iz < m_cells; ++iz)
    for (int  b = 0;  b < 4;       ++b) {
        pos[k++] = {
            (ix + bx[b]) * a + jit(rng),
            (iy + by[b]) * a + jit(rng),
            (iz + bz[b]) * a + jit(rng),
            0.0f
        };
    }
    assert(k == N);
    if (out_box) *out_box = m_cells * a;
    return pos;
}

// ── CSR cell-list neighbor builder (CPU, PBC) ──────────────────────────────────
static NbodyCSR build_csr(const std::vector<float4>& pos, float box_len) {
    int N = (int)pos.size();
    int nc = std::max(1, (int)(box_len / NBODY_R_CUT));
    float cw = box_len / nc;  // cell width ≥ R_CUT

    // Assign atoms to cells
    std::vector<std::vector<int>> cells((size_t)nc * nc * nc);
    auto cell_of = [&](float x, float y, float z) {
        int ix = std::min((int)(x / cw), nc - 1);
        int iy = std::min((int)(y / cw), nc - 1);
        int iz = std::min((int)(z / cw), nc - 1);
        return ix * nc * nc + iy * nc + iz;
    };
    for (int i = 0; i < N; ++i)
        cells[cell_of(pos[i].x, pos[i].y, pos[i].z)].push_back(i);

    // Count neighbors per atom first (for prefix sum)
    std::vector<int> cnt(N, 0);
    for (int i = 0; i < N; ++i) {
        int ix = std::min((int)(pos[i].x / cw), nc - 1);
        int iy = std::min((int)(pos[i].y / cw), nc - 1);
        int iz = std::min((int)(pos[i].z / cw), nc - 1);
        for (int ddx = -1; ddx <= 1; ++ddx)
        for (int ddy = -1; ddy <= 1; ++ddy)
        for (int ddz = -1; ddz <= 1; ++ddz) {
            int jx = (ix + ddx + nc) % nc;
            int jy = (iy + ddy + nc) % nc;
            int jz = (iz + ddz + nc) % nc;
            for (int j : cells[jx * nc * nc + jy * nc + jz]) {
                if (j == i) continue;
                float rx = mi(pos[j].x - pos[i].x, box_len);
                float ry = mi(pos[j].y - pos[i].y, box_len);
                float rz = mi(pos[j].z - pos[i].z, box_len);
                if (rx*rx + ry*ry + rz*rz < NBODY_R_CUT_SQ) ++cnt[i];
            }
        }
    }

    // Build prefix sum (neigh_ptr)
    NbodyCSR csr;
    csr.N = N;
    csr.ptr.resize(N + 1);
    csr.ptr[0] = 0;
    for (int i = 0; i < N; ++i) csr.ptr[i + 1] = csr.ptr[i] + cnt[i];
    csr.total = csr.ptr[N];
    csr.idx.resize(csr.total);

    // Fill neighbor indices
    std::vector<int> fill(N, 0);
    for (int i = 0; i < N; ++i) {
        int ix = std::min((int)(pos[i].x / cw), nc - 1);
        int iy = std::min((int)(pos[i].y / cw), nc - 1);
        int iz = std::min((int)(pos[i].z / cw), nc - 1);
        for (int ddx = -1; ddx <= 1; ++ddx)
        for (int ddy = -1; ddy <= 1; ++ddy)
        for (int ddz = -1; ddz <= 1; ++ddz) {
            int jx = (ix + ddx + nc) % nc;
            int jy = (iy + ddy + nc) % nc;
            int jz = (iz + ddz + nc) % nc;
            for (int j : cells[jx * nc * nc + jy * nc + jz]) {
                if (j == i) continue;
                float rx = mi(pos[j].x - pos[i].x, box_len);
                float ry = mi(pos[j].y - pos[i].y, box_len);
                float rz = mi(pos[j].z - pos[i].z, box_len);
                if (rx*rx + ry*ry + rz*rz < NBODY_R_CUT_SQ) {
                    csr.idx[csr.ptr[i] + fill[i]++] = j;
                }
            }
        }
    }

    // Statistics
    csr.max_per_atom  = *std::max_element(cnt.begin(), cnt.end());
    csr.mean_per_atom = (double)csr.total / N;
    return csr;
}

// ── hw_state_verified ──────────────────────────────────────────────────────────
// Returns 1 if max time deviation from median < 15%, else 0.
static int hw_state_check(const std::vector<double>& times_ms) {
    if (times_ms.size() < 3) return 1;
    std::vector<double> s = times_ms;
    std::sort(s.begin(), s.end());
    double med = s[s.size() / 2];
    if (med <= 0.0) return 1;
    for (double t : times_ms)
        if (std::fabs(t - med) / med > 0.15) return 0;
    return 1;
}

// ── Argument parsing ────────────────────────────────────────────────────────────
static NbodyConfig nbody_parse_args(int argc, char** argv) {
    NbodyConfig cfg;
    cfg.N          = NBODY_N_LARGE;
    cfg.m_cells    = 40;
    cfg.box_len    = 40.0f * NBODY_FCC_A;
    cfg.reps       = 30;
    cfg.verify     = false;
    cfg.platform   = "nvidia_rtx5060";
    cfg.kernel     = "notile";
    cfg.size_label = "large";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--n" && i+1 < argc) {
            int n = std::atoi(argv[++i]);
            cfg.N = n;
            cfg.m_cells = (int)std::round(std::cbrt(n / 4.0));
            cfg.box_len = cfg.m_cells * NBODY_FCC_A;
            if      (n <= NBODY_N_SMALL)  cfg.size_label = "small";
            else if (n <= NBODY_N_MEDIUM) cfg.size_label = "medium";
            else                           cfg.size_label = "large";
        } else if (a == "--size" && i+1 < argc) {
            cfg.size_label = argv[++i];
            if      (cfg.size_label == "small")  { cfg.N=NBODY_N_SMALL;  cfg.m_cells=10; cfg.box_len=10*NBODY_FCC_A; }
            else if (cfg.size_label == "medium") { cfg.N=NBODY_N_MEDIUM; cfg.m_cells=20; cfg.box_len=20*NBODY_FCC_A; }
            else                                  { cfg.N=NBODY_N_LARGE;  cfg.m_cells=40; cfg.box_len=40*NBODY_FCC_A; }
        } else if (a == "--reps"     && i+1 < argc) { cfg.reps     = std::atoi(argv[++i]); }
          else if (a == "--platform" && i+1 < argc) { cfg.platform = argv[++i]; }
          else if (a == "--kernel"   && i+1 < argc) { cfg.kernel   = argv[++i]; }
          else if (a == "--verify")                  { cfg.verify   = true; }
    }
    return cfg;
}

// ── CPU reference forces (for --verify) ────────────────────────────────────────
static std::vector<float4> cpu_ref_forces(
    const std::vector<float4>& pos,
    const NbodyCSR& csr, float box_len)
{
    int N = csr.N;
    std::vector<float4> f(N, {0.0f, 0.0f, 0.0f, 0.0f});
    for (int i = 0; i < N; ++i) {
        float4 pi = pos[i];
        float fx=0,fy=0,fz=0;
        for (int k = csr.ptr[i]; k < csr.ptr[i+1]; ++k) {
            int j = csr.idx[k];
            float4 pj = pos[j];
            float dx = mi(pj.x - pi.x, box_len);
            float dy = mi(pj.y - pi.y, box_len);
            float dz = mi(pj.z - pi.z, box_len);
            float r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < NBODY_R_CUT_SQ && r2 > 0.0f) {
                float r2inv = 1.0f / r2;
                float r6inv = r2inv * r2inv * r2inv;
                float fscal = 48.0f * r6inv * (r6inv - 0.5f) * r2inv;
                fx += fscal * dx; fy += fscal * dy; fz += fscal * dz;
            }
        }
        f[i] = {fx, fy, fz, 0.0f};
    }
    return f;
}

// ── Verify GPU forces vs CPU reference ─────────────────────────────────────────
// Returns max relative error. Prints first N_PRINT failures.
static float verify_forces(const std::vector<float4>& ref,
                            const std::vector<float4>& gpu,
                            int N_PRINT = 3)
{
    float max_rel = 0.0f;
    int n_fail = 0;
    for (int i = 0; i < (int)ref.size(); ++i) {
        float ex = std::fabs(ref[i].x - gpu[i].x);
        float ey = std::fabs(ref[i].y - gpu[i].y);
        float ez = std::fabs(ref[i].z - gpu[i].z);
        float err = std::sqrt(ex*ex + ey*ey + ez*ez);
        float mag = std::sqrt(ref[i].x*ref[i].x + ref[i].y*ref[i].y + ref[i].z*ref[i].z);
        float rel = (mag > 1e-6f) ? err / mag : err;
        if (rel > max_rel) max_rel = rel;
        if (rel > 1e-2f && n_fail < N_PRINT) {
            std::fprintf(stderr, "  FAIL i=%d: cpu=(%.5f,%.5f,%.5f) gpu=(%.5f,%.5f,%.5f) rel=%.4f\n",
                         i, ref[i].x, ref[i].y, ref[i].z,
                         gpu[i].x, gpu[i].y, gpu[i].z, rel);
            ++n_fail;
        }
    }
    return max_rel;
}

// ── CSV output macros ──────────────────────────────────────────────────────────
// run_nbody.sh parses lines starting with NBODY_RUN / NBODY_HW_STATE / NBODY_META
#define NBODY_PRINT_META(cfg, csr)  do {                                           \
    std::printf("NBODY_META n=%d size=%s n_nbrs_total=%d "                         \
                "max_nbrs_per_atom=%d mean_nbrs=%.2f\n",                           \
                (cfg).N, (cfg).size_label.c_str(),                                 \
                (csr).total, (csr).max_per_atom, (csr).mean_per_atom);             \
} while(0)

#define NBODY_PRINT_RUN(run, cfg, csr, time_ms)  do {                              \
    double _fl = (double)(csr).total * NBODY_FLOPS_PAIR;                           \
    double _gf = (_fl > 0 && (time_ms) > 0) ? _fl / ((time_ms) * 1e6) : 0.0;     \
    std::printf("NBODY_RUN run=%d kernel=%s size=%s n=%d "                         \
                "actual_flops=%.0f time_ms=%.6f throughput_gflops=%.6f\n",         \
                (run), (cfg).kernel.c_str(), (cfg).size_label.c_str(), (cfg).N,    \
                _fl, (time_ms), _gf);                                              \
} while(0)

#define NBODY_PRINT_HW(hw_state)  std::printf("NBODY_HW_STATE state=%d\n", (hw_state))
