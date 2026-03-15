// spmv_common.h — Shared constants, types, and utilities for E4 SpMV.
//
// Included by all C++ abstraction implementations (CUDA, Kokkos, RAJA, SYCL).
// Julia and Numba carry equivalent definitions in-file.
//
// ── E4 DESIGN DECISIONS ───────────────────────────────────────────────────────
// [D1] Problem sizes: small=1024 rows (~5K nnz), medium=8192 rows (~40K nnz),
//      large=32768 rows (~160K nnz). Target nnz follows from matrix type.
//      For laplacian_2d the actual nrows = Nx*Ny, which may differ from --n by
//      a few rows (grid dimensions rounded). All output/CSV uses actual nrows.
// [D2] Matrix types:
//      laplacian_2d  — 5-point 2D finite-difference Laplacian on sqrt(N)×sqrt(N)
//                      grid; structured, fully regular (all rows have 3–5 nnz).
//      random_sparse — exactly SPMV_RANDOM_NNZ_PER_ROW unique random col indices
//                      per row; uniform distribution, worst-case cache behavior.
//      power_law     — degrees drawn from Pareto distribution (gamma=2.5);
//                      heavy tail creates extreme load imbalance across rows.
// [D3] Kernel: one CUDA thread per row (simple, honest baseline with no warp
//      reduction tricks — later experiments can compare vs warp-per-row).
// [D4] Primary metric: GFLOP/s = 2*nnz / time_s / 1e9 (1 mul + 1 add per nnz).
// [D5] Correctness: compare GPU y = A*x against CPU CSR SpMV with max_rel_err < 1e-10.
//      Verification uses laplacian_2d with 8×8 = 64 rows, x = random [0.1, 1.1].
// [D6] experiment_id: spmv_{abs}_{platform}_{matrix}_{size_label}_n{N}_{run_id:03d}
// [D7] Warmup: adaptive CV < 2% over 10-run sliding window (same as E2/E3).
//      Max warmup = SPMV_WARMUP_MAX iterations.
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
static constexpr int SPMV_N_SMALL  = 1024;
static constexpr int SPMV_N_MEDIUM = 8192;
static constexpr int SPMV_N_LARGE  = 32768;

// ── Random sparse: nnz per row (D2) ──────────────────────────────────────────
static constexpr int SPMV_RANDOM_NNZ_PER_ROW = 5;

// ── Power-law distribution parameters (D2) ───────────────────────────────────
static constexpr double SPMV_POWERLAW_GAMMA = 2.5;  // Pareto exponent
static constexpr int    SPMV_POWERLAW_D_MIN = 1;
static constexpr uint64_t SPMV_SEED = 42ULL;        // fixed seed for reproducibility

// ── Thread block (D3) ─────────────────────────────────────────────────────────
static constexpr int SPMV_BLOCK_SIZE = 256;  // threads per block (one thread per row)

// ── Timing protocol (D7) ──────────────────────────────────────────────────────
static constexpr int    SPMV_WARMUP_MIN    = 10;
static constexpr int    SPMV_WARMUP_MAX    = 200;
static constexpr int    SPMV_WARMUP_WINDOW = 10;
static constexpr double SPMV_WARMUP_CV_CEIL = 2.0;
static constexpr int    SPMV_REPS_DEFAULT  = 30;

// ── Correctness tolerance (D5) ────────────────────────────────────────────────
static constexpr double SPMV_CORRECT_TOL = 1.0e-10;

// ── Matrix type ───────────────────────────────────────────────────────────────
enum class SpmvMatType { LAPLACIAN_2D, RANDOM_SPARSE, POWER_LAW };

inline SpmvMatType parse_matrix_type(const std::string& s) {
    if (s == "laplacian_2d")  return SpmvMatType::LAPLACIAN_2D;
    if (s == "random_sparse") return SpmvMatType::RANDOM_SPARSE;
    if (s == "power_law")     return SpmvMatType::POWER_LAW;
    std::fprintf(stderr, "Unknown matrix type: %s\n", s.c_str());
    std::exit(1);
}

inline const char* matrix_type_str(SpmvMatType t) {
    switch (t) {
        case SpmvMatType::LAPLACIAN_2D:  return "laplacian_2d";
        case SpmvMatType::RANDOM_SPARSE: return "random_sparse";
        case SpmvMatType::POWER_LAW:     return "power_law";
    }
    return "unknown";
}

// ── CSR sparse matrix (host) ──────────────────────────────────────────────────
// Standard 0-indexed CSR. row_ptr[i]..row_ptr[i+1]-1 are the nnz indices for row i.
struct SpmvCSR {
    std::vector<int>    row_ptr;   // length nrows+1
    std::vector<int>    col_idx;   // length nnz
    std::vector<double> values;    // length nnz
    int   nrows = 0;
    long  nnz   = 0;
};

// ── Matrix generators ─────────────────────────────────────────────────────────

// 5-point 2D Laplacian on Nx×Ny grid.
// Row = iy*Nx + ix. Neighbors: ±x, ±y (Dirichlet BC: boundary rows have fewer nnz).
// Values: -4.0 on diagonal, +1.0 on off-diagonals.
inline SpmvCSR generate_laplacian_2d(int target_N) {
    // Find grid dimensions closest to target_N
    int Nx = (int)std::round(std::sqrt((double)target_N));
    if (Nx < 2) Nx = 2;
    int Ny = (target_N + Nx - 1) / Nx;
    if (Ny < 2) Ny = 2;
    int N = Nx * Ny;

    SpmvCSR csr;
    csr.nrows = N;
    csr.row_ptr.resize(N + 1, 0);

    // Count nnz per row
    for (int iy = 0; iy < Ny; iy++) {
        for (int ix = 0; ix < Nx; ix++) {
            int row = iy * Nx + ix;
            int deg = 1;  // diagonal
            if (ix > 0)      ++deg;
            if (ix < Nx - 1) ++deg;
            if (iy > 0)      ++deg;
            if (iy < Ny - 1) ++deg;
            csr.row_ptr[row + 1] = deg;
        }
    }
    // Prefix sum
    for (int i = 1; i <= N; i++)
        csr.row_ptr[i] += csr.row_ptr[i - 1];
    csr.nnz = csr.row_ptr[N];
    csr.col_idx.resize(csr.nnz);
    csr.values.resize(csr.nnz);

    // Fill entries (sorted by column within each row)
    std::vector<int> pos(N, 0);
    for (int i = 0; i < N; i++) pos[i] = csr.row_ptr[i];

    for (int iy = 0; iy < Ny; iy++) {
        for (int ix = 0; ix < Nx; ix++) {
            int row = iy * Nx + ix;
            // Collect neighbors sorted by column
            std::vector<std::pair<int,double>> entries;
            entries.push_back({row, -4.0});
            if (ix > 0)      entries.push_back({row - 1,  1.0});
            if (ix < Nx - 1) entries.push_back({row + 1,  1.0});
            if (iy > 0)      entries.push_back({row - Nx, 1.0});
            if (iy < Ny - 1) entries.push_back({row + Nx, 1.0});
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

// Random sparse: exactly nnz_per_row unique random column indices per row.
// Values: 1.0 / nnz_per_row (normalized, avoids cancellation).
inline SpmvCSR generate_random_sparse(int N, int nnz_per_row, uint64_t seed) {
    nnz_per_row = std::min(nnz_per_row, N);
    SpmvCSR csr;
    csr.nrows = N;
    csr.nnz   = (long)N * nnz_per_row;
    csr.row_ptr.resize(N + 1);
    csr.col_idx.resize(csr.nnz);
    csr.values.resize(csr.nnz, 1.0 / nnz_per_row);

    for (int i = 0; i <= N; i++) csr.row_ptr[i] = i * nnz_per_row;

    // Use Fisher-Yates partial shuffle to sample unique column indices
    std::mt19937_64 rng(seed);
    std::vector<int> perm(N);
    std::iota(perm.begin(), perm.end(), 0);

    for (int row = 0; row < N; row++) {
        // Partial Fisher-Yates for nnz_per_row samples from [0, N)
        std::vector<int> pool(perm);  // fresh copy each row (can optimize but correctness first)
        for (int k = 0; k < nnz_per_row; k++) {
            int j = k + (int)(rng() % (N - k));
            std::swap(pool[k], pool[j]);
            csr.col_idx[row * nnz_per_row + k] = pool[k];
        }
        // Sort columns within row for better cache behavior in x-access
        std::sort(csr.col_idx.begin() + row * nnz_per_row,
                  csr.col_idx.begin() + row * nnz_per_row + nnz_per_row);
    }
    return csr;
}

// Power-law degree distribution using Pareto inverse CDF.
// d_i = max(d_min, min(d_max, floor((1-u)^(-1/(gamma-1)))))
// gamma=2.5, d_min=1, d_max=min(N/4, 500).
// Column indices are sampled uniformly at random (no self-loops guaranteed).
inline SpmvCSR generate_power_law(int N, uint64_t seed) {
    std::mt19937_64 rng(seed + 1);  // different seed from random_sparse
    std::uniform_real_distribution<double> udist(0.0, 1.0);

    const double inv_alpha = 1.0 / (SPMV_POWERLAW_GAMMA - 1.0);  // 1/1.5 = 2/3
    const int    d_max     = std::min(N / 4, 500);

    // Generate degree sequence
    std::vector<int> degrees(N);
    for (int i = 0; i < N; i++) {
        double u = udist(rng);
        double d = std::pow(std::max(1.0 - u, 1e-9), -inv_alpha);
        degrees[i] = std::max(SPMV_POWERLAW_D_MIN,
                               std::min(d_max, (int)std::floor(d)));
    }

    // Build CSR
    SpmvCSR csr;
    csr.nrows = N;
    csr.row_ptr.resize(N + 1, 0);
    for (int i = 0; i < N; i++) csr.row_ptr[i + 1] = degrees[i];
    for (int i = 1; i <= N; i++) csr.row_ptr[i] += csr.row_ptr[i - 1];
    csr.nnz = csr.row_ptr[N];
    csr.col_idx.resize(csr.nnz);
    csr.values.resize(csr.nnz, 1.0);

    // Fill column indices (uniform random, avoiding self-loops, no duplicates within row)
    std::uniform_int_distribution<int> cdist(0, N - 1);
    for (int row = 0; row < N; row++) {
        int deg = degrees[row];
        int start = csr.row_ptr[row];
        if (deg == 0) continue;
        // Sample deg unique columns != row (reservoir approach)
        std::vector<int> cols;
        cols.reserve(deg);
        // Simple: rejection sampling for small degree, hash set for large
        if (deg * 4 <= N) {
            while ((int)cols.size() < deg) {
                int c = cdist(rng);
                if (c == row) continue;  // skip self-loop
                bool dup = false;
                for (int x : cols) if (x == c) { dup = true; break; }
                if (!dup) cols.push_back(c);
            }
        } else {
            // Partial shuffle for high-degree rows
            std::vector<int> pool(N);
            std::iota(pool.begin(), pool.end(), 0);
            pool.erase(pool.begin() + row);  // remove self
            for (int k = 0; k < deg && k < (int)pool.size(); k++) {
                int j = k + (int)(rng() % (pool.size() - k));
                std::swap(pool[k], pool[j]);
                cols.push_back(pool[k]);
            }
        }
        std::sort(cols.begin(), cols.end());
        for (int k = 0; k < (int)cols.size(); k++) {
            csr.col_idx[start + k] = cols[k];
            csr.values[start + k]  = 1.0;
        }
    }
    return csr;
}

// ── Matrix factory ────────────────────────────────────────────────────────────
inline SpmvCSR build_matrix(SpmvMatType mtype, int N) {
    switch (mtype) {
        case SpmvMatType::LAPLACIAN_2D:
            return generate_laplacian_2d(N);
        case SpmvMatType::RANDOM_SPARSE:
            return generate_random_sparse(N, SPMV_RANDOM_NNZ_PER_ROW, SPMV_SEED);
        case SpmvMatType::POWER_LAW:
            return generate_power_law(N, SPMV_SEED);
    }
    return {};
}

// ── Input vector: fixed random [0.1, 1.1] ────────────────────────────────────
inline std::vector<double> make_x_vector(int N, uint64_t seed = SPMV_SEED) {
    std::mt19937_64 rng(seed + 99);
    std::uniform_real_distribution<double> d(0.1, 1.1);
    std::vector<double> x(N);
    for (auto& v : x) v = d(rng);
    return x;
}

// ── Throughput formula (D4) ───────────────────────────────────────────────────
inline double spmv_gflops(long nnz, double time_s) {
    return 2.0 * static_cast<double>(nnz) / time_s / 1.0e9;
}

// Effective memory bandwidth utilised: reads values+col_idx+x, writes y, reads row_ptr
inline double spmv_gbs_effective(int nrows, long nnz, double time_s) {
    double bytes = nnz * 8.0          // values (FP64)
                 + nnz * 4.0          // col_idx (int32)
                 + (nrows + 1) * 4.0  // row_ptr (int32)
                 + nrows * 8.0        // x (FP64 reads)
                 + nrows * 8.0;       // y (FP64 writes)
    return bytes / time_s / 1.0e9;
}

// Arithmetic intensity for this configuration
inline double spmv_ai(int nrows, long nnz) {
    double bytes = nnz * 8.0 + nnz * 4.0 + (nrows + 1) * 4.0
                 + nrows * 8.0 + nrows * 8.0;
    return (2.0 * nnz) / bytes;
}

// ── Size label (D6) ───────────────────────────────────────────────────────────
inline const char* spmv_size_label(int N) {
    if (N <= SPMV_N_SMALL)  return "small";
    if (N <= SPMV_N_MEDIUM) return "medium";
    return "large";
}

// ── CPU reference SpMV (D5) ───────────────────────────────────────────────────
inline void spmv_cpu_ref(const SpmvCSR& csr, const double* x, double* y) {
    for (int row = 0; row < csr.nrows; row++) {
        double sum = 0.0;
        for (int j = csr.row_ptr[row]; j < csr.row_ptr[row + 1]; j++)
            sum += csr.values[j] * x[csr.col_idx[j]];
        y[row] = sum;
    }
}

inline bool spmv_verify(const double* result, const double* ref,
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
template <typename F>
int spmv_adaptive_warmup(F&& run_once,
                          int warmup_min = SPMV_WARMUP_MIN,
                          int warmup_max = SPMV_WARMUP_MAX)
{
    std::deque<double> window;
    int total = 0;
    while (total < warmup_max) {
        auto t0 = std::chrono::high_resolution_clock::now();
        run_once();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        window.push_back(ms);
        if ((int)window.size() > SPMV_WARMUP_WINDOW) window.pop_front();
        ++total;
        if (total >= warmup_min && (int)window.size() == SPMV_WARMUP_WINDOW) {
            double mean = 0.0;
            for (double v : window) mean += v;
            mean /= window.size();
            double var = 0.0;
            for (double v : window) var += (v - mean) * (v - mean);
            var /= window.size();
            double cv = (mean > 0.0) ? 100.0 * std::sqrt(var) / mean : 100.0;
            if (cv < SPMV_WARMUP_CV_CEIL) break;
        }
    }
    return total;
}

// ── hw_state_verified (§9.7) ──────────────────────────────────────────────────
inline std::vector<int> spmv_compute_hw_state(const std::vector<double>& vals) {
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
struct SpmvStats {
    double median_gflops;
    double iqr_gflops;
    double mean_gflops;
    double min_gflops;
    double max_gflops;
    int    n_clean;
};

inline SpmvStats spmv_compute_stats(const std::vector<double>& gflops_vec,
                                     const std::vector<int>&    hw) {
    SpmvStats s{};
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
    return s;
}

// ── Output format helpers ─────────────────────────────────────────────────────
// run_spmv.sh anchors on "SPMV_RUN", "SPMV_HW_STATE", "SPMV_SUMMARY".

inline void spmv_print_run(int run_id, int nrows, long nnz, const char* matrix,
                            double time_ms, double gflops) {
    std::printf("SPMV_RUN run=%d n=%d nnz=%ld matrix=%s time_ms=%.6f throughput_gflops=%.6f\n",
                run_id, nrows, nnz, matrix, time_ms, gflops);
    std::fflush(stdout);
}

inline void spmv_print_hw_state(int run_id, int hw_state) {
    std::printf("SPMV_HW_STATE run=%d hw_state=%d\n", run_id, hw_state);
    std::fflush(stdout);
}

inline void spmv_print_summary(int nrows, long nnz, const char* matrix,
                                const SpmvStats& s, int warmup_iters) {
    std::printf(
        "SPMV_SUMMARY n=%d nnz=%ld matrix=%s "
        "median_gflops=%.4f iqr_gflops=%.4f "
        "min_gflops=%.4f max_gflops=%.4f "
        "mean_gflops=%.4f n_clean=%d warmup_iters=%d\n",
        nrows, nnz, matrix,
        s.median_gflops, s.iqr_gflops, s.min_gflops, s.max_gflops,
        s.mean_gflops, s.n_clean, warmup_iters);
    std::fflush(stdout);
}
