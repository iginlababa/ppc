// nbody_common.h — E7 N-Body: shared constants, FCC lattice, cell-list, I/O helpers
#pragma once

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <climits>
#include <numeric>
#include <string>
#include <vector>

#include <cuda_runtime.h>

// ── LJ parameters ──────────────────────────────────────────────────────────────
static constexpr float LJ_EPSILON  = 1.0f;
static constexpr float LJ_SIGMA    = 1.0f;
static constexpr float R_CUT       = 2.5f;      // cutoff in units of sigma
static constexpr float R_CUT2      = R_CUT * R_CUT;
// LJ shift: V_shift = 4*eps*(s^12/rc^12 - s^6/rc^6)  (not added to forces)

// ── FCC parameters ─────────────────────────────────────────────────────────────
static constexpr float FCC_RHO     = 0.8442f;   // number density (#/sigma^3)
// Lattice constant a = (4/rho)^(1/3)
static constexpr float FCC_A       = 1.6796f;   // precomputed

// ── Problem sizes: N = 4 * M^3 (FCC cells per dimension) ──────────────────────
static constexpr int NBODY_N_SMALL  =   4000;   // M=10
static constexpr int NBODY_N_MEDIUM =  32000;   // M=20
static constexpr int NBODY_N_LARGE  = 256000;   // M=40

// ── Kernel parameters ──────────────────────────────────────────────────────────
static constexpr int NBODY_BLOCK_SIZE  = 256;
static constexpr int MAX_NEIGHBORS     = 512;   // per-particle cap (well above ~55 typical)
static constexpr int FLOPS_PER_PAIR    = 20;    // canonical LJ FLOP count
static constexpr int TILE_SIZE         = 32;    // shared-memory tile width (P006)
static constexpr int NBODY_WARMUP      = 50;    // E1 protocol: fixed 50 warmup iterations

// ── CUDA error helper ──────────────────────────────────────────────────────────
#define CUDA_CHECK(call) do {                                                      \
    cudaError_t _e = (call);                                                       \
    if (_e != cudaSuccess) {                                                       \
        fprintf(stderr, "CUDA error at %s:%d — %s\n",                             \
                __FILE__, __LINE__, cudaGetErrorString(_e));                       \
        exit(EXIT_FAILURE);                                                        \
    }                                                                              \
} while(0)

// ── Config ─────────────────────────────────────────────────────────────────────
struct NBodyConfig {
    int         n_atoms;
    int         m_cells;       // M in M×M×M FCC
    float       box_len;       // M * FCC_A
    int         reps;
    bool        verify;
    std::string platform;
    std::string kernel;        // "notile" or "tile"
    std::string size_label;    // "small" | "medium" | "large"
};

// ── Neighbor statistics ─────────────────────────────────────────────────────────
struct NeighborStats {
    double    mean;
    int       min_count;
    int       max_count;
    double    std_dev;
    long long total;
};

// ── Result for one timed run ────────────────────────────────────────────────────
struct NBodyRunResult {
    double    time_ms;
    double    gflops;
    long long actual_flops;    // = total_neighbors * FLOPS_PER_PAIR
    int       hw_state;        // 1=clean 0=dirty
};

// ── Neighbor list (CPU, static) ─────────────────────────────────────────────────
struct NeighborList {
    std::vector<int> idx;          // [n_atoms * MAX_NEIGHBORS] flat, row-major
    std::vector<int> count;        // [n_atoms]
    int              n_atoms;
    NeighborStats    stats;
};

// ── FCC lattice generation ──────────────────────────────────────────────────────
// Returns positions as vector<float4>; w=0.
// FCC basis vectors (fractional): (0,0,0),(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5)
inline std::vector<float4> generate_fcc(int m_cells, float a, float* out_box_len = nullptr) {
    const float bx[4] = {0.0f, 0.5f, 0.5f, 0.0f};
    const float by[4] = {0.0f, 0.5f, 0.0f, 0.5f};
    const float bz[4] = {0.0f, 0.0f, 0.5f, 0.5f};

    int n = 4 * m_cells * m_cells * m_cells;
    std::vector<float4> pos(n);
    int idx = 0;
    for (int ix = 0; ix < m_cells; ++ix)
    for (int iy = 0; iy < m_cells; ++iy)
    for (int iz = 0; iz < m_cells; ++iz)
    for (int  b = 0;  b < 4;       ++b) {
        pos[idx++] = { (ix + bx[b]) * a,
                       (iy + by[b]) * a,
                       (iz + bz[b]) * a,
                       0.0f };
    }
    assert(idx == n);
    if (out_box_len) *out_box_len = m_cells * a;
    return pos;
}

// ── Minimum image convention ───────────────────────────────────────────────────
inline float min_image(float dx, float L) {
    if (dx >  0.5f * L) dx -= L;
    if (dx < -0.5f * L) dx += L;
    return dx;
}

// ── Cell-list neighbor construction (CPU, static, PBC) ─────────────────────────
// One-sided count: both i→j and j→i pairs included (no Newton's 3rd).
inline NeighborList build_neighbor_list(const std::vector<float4>& pos, float box_len) {
    int N          = (int)pos.size();
    float cell_w   = R_CUT;
    int nc         = std::max(1, (int)(box_len / cell_w));
    float actual_w = box_len / nc;  // actual cell width >= R_CUT

    // Assign atoms to cells
    std::vector<std::vector<int>> cells((size_t)nc * nc * nc);
    auto cell_of = [&](float x, float y, float z) {
        int ix = (int)(x / actual_w); if (ix >= nc) ix = nc - 1;
        int iy = (int)(y / actual_w); if (iy >= nc) iy = nc - 1;
        int iz = (int)(z / actual_w); if (iz >= nc) iz = nc - 1;
        return ix * nc * nc + iy * nc + iz;
    };
    for (int i = 0; i < N; ++i)
        cells[cell_of(pos[i].x, pos[i].y, pos[i].z)].push_back(i);

    NeighborList nl;
    nl.n_atoms = N;
    nl.count.assign(N, 0);
    nl.idx.assign((long long)N * MAX_NEIGHBORS, -1);

    for (int i = 0; i < N; ++i) {
        int ix = (int)(pos[i].x / actual_w); if (ix >= nc) ix = nc - 1;
        int iy = (int)(pos[i].y / actual_w); if (iy >= nc) iy = nc - 1;
        int iz = (int)(pos[i].z / actual_w); if (iz >= nc) iz = nc - 1;
        int cnt = 0;
        for (int ddx = -1; ddx <= 1; ++ddx)
        for (int ddy = -1; ddy <= 1; ++ddy)
        for (int ddz = -1; ddz <= 1; ++ddz) {
            int jx = (ix + ddx + nc) % nc;
            int jy = (iy + ddy + nc) % nc;
            int jz = (iz + ddz + nc) % nc;
            for (int j : cells[jx * nc * nc + jy * nc + jz]) {
                if (j == i) continue;
                float rx = min_image(pos[j].x - pos[i].x, box_len);
                float ry = min_image(pos[j].y - pos[i].y, box_len);
                float rz = min_image(pos[j].z - pos[i].z, box_len);
                float r2 = rx*rx + ry*ry + rz*rz;
                if (r2 < R_CUT2 && cnt < MAX_NEIGHBORS)
                    nl.idx[(long long)i * MAX_NEIGHBORS + cnt++] = j;
            }
        }
        nl.count[i] = cnt;
    }

    // Statistics
    NeighborStats& s = nl.stats;
    s.min_count = INT_MAX; s.max_count = 0; s.total = 0;
    double sum2 = 0.0;
    for (int i = 0; i < N; ++i) {
        int c = nl.count[i];
        s.total += c;
        if (c < s.min_count) s.min_count = c;
        if (c > s.max_count) s.max_count = c;
        sum2 += (double)c * c;
    }
    s.mean    = (double)s.total / N;
    s.std_dev = std::sqrt(sum2 / N - s.mean * s.mean);
    return nl;
}

// ── Arithmetic intensity estimate ──────────────────────────────────────────────
// Bytes per particle: 16 (pos_i) + n_nbrs*(16+4) (pos_j + index) + 12 (force write) + 4 (count)
inline double compute_ai(const NeighborStats& s) {
    double bytes_per_particle = 16.0 + s.mean * (16.0 + 4.0) + 12.0 + 4.0;
    double flops_per_particle = s.mean * FLOPS_PER_PAIR;
    return (bytes_per_particle > 0) ? flops_per_particle / bytes_per_particle : 0.0;
}

// ── HW state check: dirty if max time deviation > 15% ──────────────────────────
inline int check_hw_state(const std::vector<double>& times_ms) {
    if (times_ms.size() < 2) return 1;
    double sum = 0.0;
    for (double t : times_ms) sum += t;
    double mean = sum / (double)times_ms.size();
    double max_dev = 0.0;
    for (double t : times_ms) {
        double dev = std::fabs(t - mean) / mean;
        if (dev > max_dev) max_dev = dev;
    }
    return (max_dev < 0.15) ? 1 : 0;
}

// ── Config parsing ─────────────────────────────────────────────────────────────
inline NBodyConfig nbody_parse_args(int argc, char** argv) {
    NBodyConfig cfg;
    cfg.n_atoms    = NBODY_N_SMALL;
    cfg.m_cells    = 10;
    cfg.box_len    = 10.0f * FCC_A;
    cfg.reps       = 30;
    cfg.verify     = false;
    cfg.platform   = "nvidia_rtx5060";
    cfg.kernel     = "notile";
    cfg.size_label = "small";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--n" || a == "--atoms") && i + 1 < argc) {
            cfg.n_atoms  = std::atoi(argv[++i]);
            cfg.m_cells  = (int)std::round(std::cbrt(cfg.n_atoms / 4.0));
            cfg.box_len  = cfg.m_cells * FCC_A;
        } else if (a == "--reps" && i + 1 < argc) {
            cfg.reps     = std::atoi(argv[++i]);
        } else if (a == "--platform" && i + 1 < argc) {
            cfg.platform = argv[++i];
        } else if (a == "--kernel" && i + 1 < argc) {
            cfg.kernel   = argv[++i];
        } else if (a == "--size" && i + 1 < argc) {
            cfg.size_label = argv[++i];
        } else if (a == "--verify") {
            cfg.verify   = true;
        }
    }
    // Infer size label from n_atoms if not explicitly set
    if (cfg.size_label.empty() || cfg.size_label == "small") {
        if      (cfg.n_atoms >= NBODY_N_LARGE)  cfg.size_label = "large";
        else if (cfg.n_atoms >= NBODY_N_MEDIUM) cfg.size_label = "medium";
        else                                     cfg.size_label = "small";
    }
    return cfg;
}

// ── CSV output macros ─────────────────────────────────────────────────────────
// run_nbody.sh parses lines starting with NBODY_RUN / NBODY_HW_STATE / NBODY_STATS
#define NBODY_PRINT_STATS(cfg, s) do {                                             \
    printf("NBODY_STATS n_atoms=%d n_nbrs_mean=%.2f "                             \
           "n_nbrs_min=%d n_nbrs_max=%d n_nbrs_std=%.2f total_nbrs=%lld\n",       \
           (cfg).n_atoms, (s).mean,                                                \
           (s).min_count, (s).max_count, (s).std_dev, (s).total);                 \
} while(0)

#define NBODY_PRINT_RUN(run, cfg, res) do {                                        \
    printf("NBODY_RUN run=%d kernel=%s n_atoms=%d size=%s "                       \
           "time_ms=%.6f throughput_gflops=%.6f\n",                               \
           (run), (cfg).kernel.c_str(), (cfg).n_atoms,                            \
           (cfg).size_label.c_str(), (res).time_ms, (res).gflops);                \
} while(0)

#define NBODY_PRINT_HW_STATE(hw)  printf("NBODY_HW_STATE state=%d\n", (hw))
