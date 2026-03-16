// sptrsv_common.h — Shared constants, types, and utilities for E5 SpTRSV.
//
// Included by all C++ abstraction implementations (CUDA, Kokkos, RAJA).
// Julia carries equivalent definitions in-file.
//
// ── E5 DESIGN DECISIONS ───────────────────────────────────────────────────────
// [D1] Problem sizes: small=256 rows, medium=2048 rows, large=8192 rows.
//      These are intentionally smaller than E4 SpMV because SpTRSV is
//      latency-bound (serial dependency chain across levels); large sizes
//      are still within 8 GB VRAM even with dense lower triangles.
// [D2] Matrix types:
//      lower_triangular_laplacian — lower triangle + diagonal of the 2D Laplacian
//                                   on sqrt(N)×sqrt(N) grid. Regular level structure:
//                                   level k contains all rows at graph distance k
//                                   from row 0 in the undirected adjacency graph.
//      lower_triangular_random   — random sparse lower triangular; diagonal=1.0,
//                                   off-diagonal avg 5 nnz/row, uniform placement,
//                                   values 0.1 (well-conditioned). Irregular levels.
// [D3] Kernel: one thread per row within each level. Sequential level launches from
//      host. No warp-per-row or persistent-kernel optimisation — honest baseline.
//      __threadfence() after each x write (defensive; inter-level ordering is
//      guaranteed by the device synchronise between kernel launches).
// [D4] Primary metric: GFLOP/s = 2*nnz / time_s / 1e9 (1 mul + 1 add per nnz,
//      consistent with E4 SpMV). SpTRSV is latency-bound, not bandwidth-bound;
//      GFLOP/s will be low — the binding constraint is level-set depth.
// [D5] Correctness: compare GPU x = L^{-1}b against sequential CPU forward
//      substitution. max_rel_err < 1e-10. Verification uses a 16×16 (N=256)
//      lower_triangular_laplacian with b = random [0.1, 1.1].
// [D6] experiment_id: sptrsv_{abs}_{platform}_{matrix}_{size_label}_n{N}_{run_id:03d}
// [D7] Warmup: adaptive CV < 2% over 10-run sliding window (same as E3/E4).
//      Max warmup = SPTRSV_WARMUP_MAX. x is reset to zero before EACH rep (solve
//      writes into x; timed region excludes the reset cudaMemset).
// ─────────────────────────────────────────────────────────────────────────────

#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// ── Problem sizes (D1) ────────────────────────────────────────────────────────
static constexpr int SPTRSV_N_SMALL  = 256;
static constexpr int SPTRSV_N_MEDIUM = 2048;
static constexpr int SPTRSV_N_LARGE  = 8192;

// ── Random lower triangular: avg off-diagonal nnz per row (D2) ───────────────
static constexpr int    SPTRSV_RANDOM_NNZ_PER_ROW = 5;
static constexpr double SPTRSV_RANDOM_OFFDIAG_VAL = 0.1; // ensures diagonal dominance
static constexpr uint64_t SPTRSV_SEED = 42ULL;

// ── Thread block (D3) ─────────────────────────────────────────────────────────
static constexpr int SPTRSV_BLOCK_SIZE = 256;

// ── Timing protocol (D7) ──────────────────────────────────────────────────────
static constexpr int    SPTRSV_WARMUP_MIN    = 10;
static constexpr int    SPTRSV_WARMUP_MAX    = 200;
static constexpr int    SPTRSV_WARMUP_WINDOW = 10;
static constexpr double SPTRSV_WARMUP_CV_CEIL = 2.0;
static constexpr int    SPTRSV_REPS_DEFAULT  = 30;

// ── Correctness tolerance (D5) ────────────────────────────────────────────────
static constexpr double SPTRSV_CORRECT_TOL = 1.0e-10;

// ── Matrix type ───────────────────────────────────────────────────────────────
enum class SptrsMatType { LAPLACIAN, RANDOM };

inline SptrsMatType parse_matrix_type(const std::string& s) {
    if (s == "lower_triangular_laplacian") return SptrsMatType::LAPLACIAN;
    if (s == "lower_triangular_random")    return SptrsMatType::RANDOM;
    std::fprintf(stderr, "Unknown matrix type: %s\n", s.c_str());
    std::exit(1);
}

inline const char* matrix_type_str(SptrsMatType t) {
    switch (t) {
        case SptrsMatType::LAPLACIAN: return "lower_triangular_laplacian";
        case SptrsMatType::RANDOM:    return "lower_triangular_random";
    }
    return "unknown";
}

// ── CSR lower-triangular matrix (host) ───────────────────────────────────────
// Standard 0-indexed CSR. row_ptr[i]..row_ptr[i+1]-1 are the nnz for row i.
// Diagonal entry is always included (L[i,i] != 0 required for forward sub).
// Column indices within each row are sorted in ascending order.
struct SptrsCSR {
    std::vector<int>    row_ptr;   // length nrows+1
    std::vector<int>    col_idx;   // length nnz (includes diagonal)
    std::vector<double> values;    // length nnz
    int   nrows = 0;
    long  nnz   = 0;
};

// ── Level-set struct ──────────────────────────────────────────────────────────
// level_ptr[l] .. level_ptr[l+1]-1 are indices into level_rows[] for level l.
// level_rows[level_ptr[l] + k] = row index of the k-th row in level l.
struct SptrsLevels {
    std::vector<int> level_ptr;   // length n_levels+1
    std::vector<int> level_rows;  // length nrows (permuted row indices)
    int n_levels      = 0;
    int max_lw        = 0;  // max level width
    int min_lw        = 0;  // min level width (excludes levels of width 0, if any)
};

// ── Level-set construction (CPU-side, shared by all abstractions) ─────────────
// BFS-like: row i is assigned to level 1 + max(level[j] for all j < i where L[i,j] != 0).
// Row 0 is always level 0 (no lower-triangular dependencies).
// Single forward pass — O(nnz) since matrix is strictly lower-triangular + diagonal.
inline SptrsLevels build_levels(const SptrsCSR& csr) {
    const int N = csr.nrows;
    std::vector<int> level(N, 0);

    for (int i = 1; i < N; i++) {
        int max_dep = -1;
        for (int j = csr.row_ptr[i]; j < csr.row_ptr[i + 1]; j++) {
            int col = csr.col_idx[j];
            if (col < i) {  // strictly lower triangular dependency
                if (level[col] > max_dep) max_dep = level[col];
            }
        }
        level[i] = (max_dep >= 0) ? max_dep + 1 : 0;
    }

    int n_levels = *std::max_element(level.begin(), level.end()) + 1;

    // Count rows per level
    std::vector<int> counts(n_levels, 0);
    for (int i = 0; i < N; i++) counts[level[i]]++;

    // Build level_ptr (prefix sum)
    SptrsLevels ls;
    ls.n_levels = n_levels;
    ls.level_ptr.resize(n_levels + 1, 0);
    for (int l = 0; l < n_levels; l++) ls.level_ptr[l + 1] = ls.level_ptr[l] + counts[l];
    ls.level_rows.resize(N);

    // Fill level_rows
    std::vector<int> pos(n_levels, 0);
    for (int l = 0; l < n_levels; l++) pos[l] = ls.level_ptr[l];
    for (int i = 0; i < N; i++) ls.level_rows[pos[level[i]]++] = i;

    // Compute max/min level width
    ls.max_lw = *std::max_element(counts.begin(), counts.end());
    ls.min_lw = *std::min_element(counts.begin(), counts.end());
    // If any zero-width levels exist (shouldn't for a connected lower triangular), ignore them
    for (int c : counts) if (c > 0 && c < ls.min_lw) ls.min_lw = c;

    return ls;
}

// ── Matrix generators ─────────────────────────────────────────────────────────

// Lower triangle + diagonal of the 2D Laplacian on Nx×Ny grid.
// Row = iy*Nx + ix. Lower neighbours in the grid have smaller row index.
// Values: -4.0 on diagonal (or boundary equivalent), +1.0 on lower off-diagonals.
inline SptrsCSR generate_laplacian_lower(int target_N) {
    int Nx = (int)std::round(std::sqrt((double)target_N));
    if (Nx < 2) Nx = 2;
    int Ny = (target_N + Nx - 1) / Nx;
    if (Ny < 2) Ny = 2;
    int N = Nx * Ny;

    SptrsCSR csr;
    csr.nrows = N;
    csr.row_ptr.resize(N + 1, 0);

    // Count nnz per row (diagonal + lower off-diagonals only)
    for (int iy = 0; iy < Ny; iy++) {
        for (int ix = 0; ix < Nx; ix++) {
            int row = iy * Nx + ix;
            int deg = 1;  // diagonal
            if (ix > 0)  ++deg;  // left neighbour: row - 1   < row
            if (iy > 0)  ++deg;  // bottom neighbour: row - Nx < row
            csr.row_ptr[row + 1] = deg;
        }
    }
    for (int i = 1; i <= N; i++) csr.row_ptr[i] += csr.row_ptr[i - 1];
    csr.nnz = csr.row_ptr[N];
    csr.col_idx.resize(csr.nnz);
    csr.values.resize(csr.nnz);

    // Fill entries (sorted ascending by column within each row)
    std::vector<int> pos(N);
    for (int i = 0; i < N; i++) pos[i] = csr.row_ptr[i];

    for (int iy = 0; iy < Ny; iy++) {
        for (int ix = 0; ix < Nx; ix++) {
            int row = iy * Nx + ix;
            std::vector<std::pair<int,double>> entries;
            // Lower off-diagonals (column < row)
            if (ix > 0)  entries.push_back({row - 1,  1.0});
            if (iy > 0)  entries.push_back({row - Nx, 1.0});
            // Diagonal (must be last in sorted order but is the largest col index = row)
            // Diagonal value: count of total neighbours (not just lower) for correct solve
            int total_deg = 1;
            if (ix > 0)      ++total_deg;
            if (ix < Nx - 1) ++total_deg;
            if (iy > 0)      ++total_deg;
            if (iy < Ny - 1) ++total_deg;
            entries.push_back({row, -(double)total_deg});
            std::sort(entries.begin(), entries.end());
            for (auto& [col, val] : entries) {
                int k = pos[row]++;
                csr.col_idx[k] = col;
                csr.values[k]  = val;
            }
        }
    }
    return csr;
}

// Random lower triangular: diagonal = 1.0, off-diagonal avg nnz_per_row entries
// placed uniformly in [0, row-1], values = SPTRSV_RANDOM_OFFDIAG_VAL.
// Diagonal dominance: |diag| = 1.0 > sum|off-diag| = nnz_per_row * 0.1 = 0.5 → stable.
inline SptrsCSR generate_random_lower(int N, int nnz_per_row, uint64_t seed) {
    std::mt19937_64 rng(seed + 7);

    SptrsCSR csr;
    csr.nrows = N;
    csr.row_ptr.resize(N + 1, 0);

    // Degrees: row 0 has only diagonal; rows 1..N-1 have min(nnz_per_row, row) off-diag + diagonal
    std::vector<int> degrees(N);
    degrees[0] = 1;  // only diagonal
    for (int i = 1; i < N; i++) {
        degrees[i] = std::min(nnz_per_row, i) + 1;  // off-diag + diagonal
    }
    for (int i = 0; i < N; i++) csr.row_ptr[i + 1] = degrees[i];
    for (int i = 1; i <= N; i++) csr.row_ptr[i] += csr.row_ptr[i - 1];
    csr.nnz = csr.row_ptr[N];
    csr.col_idx.resize(csr.nnz);
    csr.values.resize(csr.nnz, SPTRSV_RANDOM_OFFDIAG_VAL);

    // Row 0: only diagonal
    csr.col_idx[0] = 0;
    csr.values[0]  = 1.0;

    // Rows 1..N-1: sample unique columns from [0, row-1], then add diagonal
    for (int row = 1; row < N; row++) {
        int n_offdiag = degrees[row] - 1;
        int start = csr.row_ptr[row];

        // Sample n_offdiag unique columns from [0, row-1]
        std::vector<int> pool(row);
        std::iota(pool.begin(), pool.end(), 0);
        // Partial Fisher-Yates
        for (int k = 0; k < n_offdiag; k++) {
            int j = k + (int)(rng() % (row - k));
            std::swap(pool[k], pool[j]);
        }
        std::sort(pool.begin(), pool.begin() + n_offdiag);

        int k = start;
        for (int m = 0; m < n_offdiag; m++) {
            csr.col_idx[k] = pool[m];
            csr.values[k]  = SPTRSV_RANDOM_OFFDIAG_VAL;
            ++k;
        }
        // Diagonal last (largest column index = row)
        csr.col_idx[k] = row;
        csr.values[k]  = 1.0;
    }
    return csr;
}

// ── Matrix factory ────────────────────────────────────────────────────────────
inline SptrsCSR build_matrix(SptrsMatType mtype, int N) {
    switch (mtype) {
        case SptrsMatType::LAPLACIAN:
            return generate_laplacian_lower(N);
        case SptrsMatType::RANDOM:
            return generate_random_lower(N, SPTRSV_RANDOM_NNZ_PER_ROW, SPTRSV_SEED);
    }
    return {};
}

// ── RHS vector b: fixed random [0.1, 1.1] ────────────────────────────────────
inline std::vector<double> make_b_vector(int N, uint64_t seed = SPTRSV_SEED) {
    std::mt19937_64 rng(seed + 77);
    std::uniform_real_distribution<double> d(0.1, 1.1);
    std::vector<double> b(N);
    for (auto& v : b) v = d(rng);
    return b;
}

// ── Throughput formula (D4) ───────────────────────────────────────────────────
// Same formula as E4 SpMV for cross-experiment consistency.
// Note: SpTRSV is latency-bound; GFLOP/s will be << bandwidth-limited peak.
inline double sptrsv_gflops(long nnz, double time_s) {
    return 2.0 * static_cast<double>(nnz) / time_s / 1.0e9;
}

// Approximate arithmetic intensity (same access model as SpMV, plus b reads)
inline double sptrsv_ai(int nrows, long nnz) {
    double bytes = nnz * 8.0          // values (FP64)
                 + nnz * 4.0          // col_idx (int32)
                 + (nrows + 1) * 4.0  // row_ptr (int32)
                 + nrows * 8.0        // x reads (off-diag accesses ≈ nnz avg)
                 + nrows * 8.0        // b reads
                 + nrows * 8.0;       // x writes
    return (2.0 * nnz) / bytes;
}

// ── Size label (D6) ───────────────────────────────────────────────────────────
inline const char* sptrsv_size_label(int N) {
    if (N <= SPTRSV_N_SMALL)  return "small";
    if (N <= SPTRSV_N_MEDIUM) return "medium";
    return "large";
}

// ── CPU reference forward substitution (D5) ──────────────────────────────────
// Lx = b, L is lower triangular CSR with diagonal included.
// x[i] = (b[i] - sum_{j<i} L[i,j]*x[j]) / L[i,i]
inline void sptrsv_cpu_ref(const SptrsCSR& csr, const double* b, double* x) {
    for (int i = 0; i < csr.nrows; i++) {
        double sum = b[i];
        double diag = 1.0;
        for (int j = csr.row_ptr[i]; j < csr.row_ptr[i + 1]; j++) {
            int col = csr.col_idx[j];
            if (col == i) { diag = csr.values[j]; }
            else           { sum -= csr.values[j] * x[col]; }  // col < i always
        }
        x[i] = sum / diag;
    }
}

inline bool sptrsv_verify(const double* result, const double* ref,
                           int nrows, double tol, double* max_err_out) {
    double max_err = 0.0;
    for (int i = 0; i < nrows; i++) {
        double denom = (std::fabs(ref[i]) < 1.0e-14) ? 1.0 : std::fabs(ref[i]);
        double rel   = std::fabs(result[i] - ref[i]) / denom;
        if (rel > max_err) max_err = rel;
    }
    if (max_err_out) *max_err_out = max_err;
    return max_err < tol;
}

// ── Adaptive warmup (D7) ──────────────────────────────────────────────────────
// run_once() must include x-reset internally for warmup to be valid.
template <typename F>
int sptrsv_adaptive_warmup(F&& run_once,
                            int warmup_min = SPTRSV_WARMUP_MIN,
                            int warmup_max = SPTRSV_WARMUP_MAX)
{
    std::deque<double> window;
    int total = 0;
    while (total < warmup_max) {
        auto t0 = std::chrono::high_resolution_clock::now();
        run_once();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        window.push_back(ms);
        if ((int)window.size() > SPTRSV_WARMUP_WINDOW) window.pop_front();
        ++total;
        if (total >= warmup_min && (int)window.size() == SPTRSV_WARMUP_WINDOW) {
            double mean = 0.0;
            for (double v : window) mean += v;
            mean /= window.size();
            double var = 0.0;
            for (double v : window) var += (v - mean) * (v - mean);
            var /= window.size();
            double cv = (mean > 0.0) ? 100.0 * std::sqrt(var) / mean : 100.0;
            if (cv < SPTRSV_WARMUP_CV_CEIL) break;
        }
    }
    return total;
}

// ── hw_state_verified (§9.7) ──────────────────────────────────────────────────
inline std::vector<int> sptrsv_compute_hw_state(const std::vector<double>& vals) {
    if (vals.empty()) return {};
    std::vector<double> sorted = vals;
    std::sort(sorted.begin(), sorted.end());
    size_t n = sorted.size();
    double med = (n % 2 == 0)
        ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        : sorted[n / 2];
    double denom = (std::fabs(med) < 1.0e-12) ? 1.0 : std::fabs(med);
    std::vector<int> flags(vals.size());
    for (size_t i = 0; i < vals.size(); i++)
        flags[i] = (std::fabs(vals[i] - med) / denom <= 0.15) ? 1 : 0;
    return flags;
}

// ── Statistics ────────────────────────────────────────────────────────────────
struct SptrsStats {
    double median_gflops;
    double iqr_gflops;
    double mean_gflops;
    double min_gflops;
    double max_gflops;
    double cv_pct;
    int    n_clean;
};

inline SptrsStats sptrsv_compute_stats(const std::vector<double>& gflops_vec,
                                        const std::vector<int>&    hw) {
    SptrsStats s{};
    std::vector<double> clean;
    for (size_t i = 0; i < gflops_vec.size(); i++)
        if (hw[i] == 1) clean.push_back(gflops_vec[i]);
    s.n_clean = (int)clean.size();
    if (clean.empty()) return s;
    std::sort(clean.begin(), clean.end());
    size_t n = clean.size();
    s.min_gflops  = clean.front();
    s.max_gflops  = clean.back();
    s.mean_gflops = 0.0;
    for (double v : clean) s.mean_gflops += v;
    s.mean_gflops /= n;
    s.median_gflops = (n % 2 == 0)
        ? (clean[n / 2 - 1] + clean[n / 2]) / 2.0
        : clean[n / 2];
    s.iqr_gflops = clean[(3 * n) / 4] - clean[n / 4];
    s.cv_pct = (s.mean_gflops > 0.0 && n > 1)
        ? 100.0 * std::sqrt([&](){
              double var = 0.0;
              for (double v : clean) var += (v - s.mean_gflops) * (v - s.mean_gflops);
              return var / n;
          }()) / s.mean_gflops
        : 0.0;
    return s;
}

// ── Output format helpers ─────────────────────────────────────────────────────
// run_sptrsv.sh anchors on "SPTRSV_RUN", "SPTRSV_HW_STATE", "SPTRSV_SUMMARY".

inline void sptrsv_print_run(int run_id, int nrows, long nnz, int n_levels,
                              int max_lw, int min_lw, const char* matrix,
                              double time_ms, double gflops) {
    std::printf(
        "SPTRSV_RUN run=%d n_rows=%d nnz=%ld n_levels=%d max_lw=%d min_lw=%d"
        " matrix=%s time_ms=%.6f throughput_gflops=%.6f\n",
        run_id, nrows, nnz, n_levels, max_lw, min_lw, matrix, time_ms, gflops);
    std::fflush(stdout);
}

inline void sptrsv_print_hw_state(int run_id, int hw_state) {
    std::printf("SPTRSV_HW_STATE run=%d hw_state=%d\n", run_id, hw_state);
    std::fflush(stdout);
}

inline void sptrsv_print_summary(int nrows, long nnz, int n_levels,
                                  int max_lw, int min_lw, const char* matrix,
                                  const SptrsStats& s, int warmup_iters) {
    std::printf(
        "SPTRSV_SUMMARY n_rows=%d nnz=%ld n_levels=%d max_lw=%d min_lw=%d matrix=%s "
        "median_gflops=%.4f iqr_gflops=%.4f cv_pct=%.2f "
        "min_gflops=%.4f max_gflops=%.4f mean_gflops=%.4f n_clean=%d warmup_iters=%d\n",
        nrows, nnz, n_levels, max_lw, min_lw, matrix,
        s.median_gflops, s.iqr_gflops, s.cv_pct,
        s.min_gflops, s.max_gflops, s.mean_gflops,
        s.n_clean, warmup_iters);
    std::fflush(stdout);
}
