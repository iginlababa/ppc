// stencil_common.h — Shared constants, types, and utilities for E3 3D Stencil.
//
// Included by all C++ abstraction implementations (CUDA, Kokkos, RAJA, SYCL).
// Julia and Numba carry equivalent definitions in-file.
//
// ── E3 DESIGN DECISIONS ───────────────────────────────────────────────────────
// [D1] Problem sizes: small=32, medium=128, large=256, xlarge=512 (all sides equal, N³ grid).
//      Memory per array: small=256KB, medium=16MB, large=128MB, xlarge=1GB (FP64).
//      Two arrays (in, out) → peak: 256 MB for large, 2 GB for xlarge. Well within MI300X 192 GB HBM.
//      large=256³ fits in MI300X L2 cache (~256 MB) → cache-bound, not DRAM-bound.
//      xlarge=512³ exceeds all GPU caches → valid DRAM-bandwidth measurement (~5 TB/s expected).
// [D2] 7-point Jacobi stencil. c0=0.5, c1=(1-c0)/6 ≈ 0.08333.
//      Sums satisfy conservation: c0 + 6*c1 = 1.0 (weighted average property).
// [D3] FP64 throughout. AI = 13 FLOP / 64 bytes ≈ 0.203 FLOP/byte (memory-bound).
//      Primary metric: GB/s (bandwidth). Secondary: GFLOP/s.
//      "~0.5 FLOP/byte" in project_spec is the optimistic estimate assuming full
//      in-plane cache reuse (2 arrays, 1 read + 1 write); hardware sees ~64 bytes/cell.
// [D4] Thread block: (BLOCK_X=32, BLOCK_Y=4, BLOCK_Z=2) = 256 threads.
//      threadIdx.x maps to ix (innermost index in row-major layout → coalesced access).
// [D5] Correctness check at N=16 against CPU Jacobi reference (max_rel_err < 1e-10).
// [D6] experiment_id format: stencil_{abstraction}_{platform}_{size_label}_n{N}_{run_id:03d}
// [D7] Warmup: adaptive — run until CV < 2% over 10-run sliding window (amended §9.1).
//      Fixed-count warmup was removed for memory-bound kernels: thermal stabilisation
//      dominates for compute-bound (E2), but for bandwidth-bound (E3) the hardware
//      reaches steady state faster. Max warmup = STENCIL_WARMUP_MAX iterations.
// ─────────────────────────────────────────────────────────────────────────────

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <deque>
#include <numeric>
#include <string>
#include <vector>

// ── Problem sizes (D1) ────────────────────────────────────────────────────────
static constexpr int STENCIL_N_SMALL  = 32;
static constexpr int STENCIL_N_MEDIUM = 128;
static constexpr int STENCIL_N_LARGE  = 256;
static constexpr int STENCIL_N_XLARGE = 512;  // exceeds L2 → DRAM-bound on all platforms

// ── Stencil coefficients (D2) ─────────────────────────────────────────────────
static constexpr double STENCIL_C0 = 0.5;
static constexpr double STENCIL_C1 = (1.0 - STENCIL_C0) / 6.0;  // ≈ 0.083333

// ── FLOP and bandwidth accounting (D3) ────────────────────────────────────────
// 7 multiplications + 6 additions = 13 FLOP per interior cell.
// 7 reads (center + 6 neighbours) + 1 write = 8 doubles = 64 bytes per cell.
static constexpr int    STENCIL_FLOP_PER_CELL  = 13;
static constexpr int    STENCIL_BYTES_PER_CELL = 64;  // 8 doubles * 8 bytes

// ── Thread block dimensions (D4) ─────────────────────────────────────────────
static constexpr int STENCIL_BLOCK_X = 32;   // must equal warp size for coalescing
static constexpr int STENCIL_BLOCK_Y = 4;
static constexpr int STENCIL_BLOCK_Z = 2;    // total = 256 threads per block

// ── Timing protocol (D7) ─────────────────────────────────────────────────────
static constexpr int    STENCIL_WARMUP_MIN    = 10;   // minimum warmup iterations
static constexpr int    STENCIL_WARMUP_MAX    = 200;  // absolute ceiling
static constexpr int    STENCIL_WARMUP_WINDOW = 10;   // sliding window for CV check
static constexpr double STENCIL_WARMUP_CV_CEIL = 2.0; // CV < 2% → stable
static constexpr int    STENCIL_REPS_DEFAULT  = 30;

// ── Correctness tolerance (D5) ────────────────────────────────────────────────
static constexpr double STENCIL_CORRECT_TOL = 1.0e-10;

// ── Throughput formulas ───────────────────────────────────────────────────────
// N_cells = (N-2)^3  — interior cells only (boundary excluded from computation).
inline long stencil_interior_cells(int N) {
    long inner = static_cast<long>(N - 2);
    return inner * inner * inner;
}

// GB/s: total bytes transferred / time_s / 1e9
inline double stencil_gbs(int N, double time_s) {
    return static_cast<double>(stencil_interior_cells(N)) *
           STENCIL_BYTES_PER_CELL / time_s / 1.0e9;
}

// GFLOP/s: secondary metric
inline double stencil_gflops(int N, double time_s) {
    return static_cast<double>(stencil_interior_cells(N)) *
           STENCIL_FLOP_PER_CELL / time_s / 1.0e9;
}

// ── Size label (D6) ───────────────────────────────────────────────────────────
inline const char* stencil_size_label(int N) {
    if (N <= STENCIL_N_SMALL)  return "small";
    if (N <= STENCIL_N_MEDIUM) return "medium";
    if (N <= STENCIL_N_LARGE)  return "large";
    return "xlarge";
}

// ── Adaptive warmup (D7) ──────────────────────────────────────────────────────
// Calls run_once() until CV of the last WINDOW timings is < CV_CEIL (%).
// Returns the number of warmup iterations executed.
template <typename F>
int adaptive_warmup(F&& run_once,
                    int warmup_min = STENCIL_WARMUP_MIN,
                    int warmup_max = STENCIL_WARMUP_MAX)
{
    std::deque<double> window;
    int total = 0;
    while (total < warmup_max) {
        auto t0 = std::chrono::high_resolution_clock::now();
        run_once();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        window.push_back(ms);
        if (static_cast<int>(window.size()) > STENCIL_WARMUP_WINDOW)
            window.pop_front();
        ++total;
        if (total >= warmup_min &&
            static_cast<int>(window.size()) == STENCIL_WARMUP_WINDOW) {
            double mean = 0.0;
            for (double v : window) mean += v;
            mean /= static_cast<double>(window.size());
            double var = 0.0;
            for (double v : window) var += (v - mean) * (v - mean);
            var /= static_cast<double>(window.size());
            double cv = (mean > 0.0) ? 100.0 * std::sqrt(var) / mean : 100.0;
            if (cv < STENCIL_WARMUP_CV_CEIL) break;
        }
    }
    return total;
}

// ── hw_state_verified (project_spec §9.7) ────────────────────────────────────
// flag[i] = 1 if |val[i] - median| / median <= 0.15, else 0
inline std::vector<int> compute_hw_state(const std::vector<double>& vals) {
    if (vals.empty()) return {};
    std::vector<double> sorted = vals;
    std::sort(sorted.begin(), sorted.end());
    size_t n = sorted.size();
    double med = (n % 2 == 0)
        ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        : sorted[n / 2];
    double denom = (std::fabs(med) < 1.0e-12) ? 1.0 : std::fabs(med);
    std::vector<int> flags(n);
    for (size_t i = 0; i < n; i++)
        flags[i] = (std::fabs(vals[i] - med) / denom <= 0.15) ? 1 : 0;
    return flags;
}

// ── Statistics ────────────────────────────────────────────────────────────────
struct StencilStats {
    double median_gbs;
    double iqr_gbs;
    double mean_gbs;
    double min_gbs;
    double max_gbs;
    int    n_clean;
};

inline StencilStats compute_stencil_stats(const std::vector<double>& gbs_vec,
                                           const std::vector<int>&    hw) {
    StencilStats s{};
    std::vector<double> clean;
    clean.reserve(gbs_vec.size());
    for (size_t i = 0; i < gbs_vec.size(); i++)
        if (hw[i] == 1) clean.push_back(gbs_vec[i]);
    s.n_clean = static_cast<int>(clean.size());
    if (clean.empty()) return s;
    std::sort(clean.begin(), clean.end());
    size_t n = clean.size();
    s.min_gbs  = clean.front();
    s.max_gbs  = clean.back();
    s.mean_gbs = 0.0;
    for (double v : clean) s.mean_gbs += v;
    s.mean_gbs /= static_cast<double>(n);
    s.median_gbs = (n % 2 == 0)
        ? (clean[n / 2 - 1] + clean[n / 2]) / 2.0
        : clean[n / 2];
    s.iqr_gbs = clean[(3 * n) / 4] - clean[n / 4];
    return s;
}

// ── CPU reference stencil (correctness only; N ≤ 32) ─────────────────────────
// Row-major: in[iz*N*N + iy*N + ix]
inline void stencil_cpu_ref(int N, double c0, double c1,
                              const double* in, double* out) {
    for (int iz = 1; iz < N - 1; iz++)
        for (int iy = 1; iy < N - 1; iy++)
            for (int ix = 1; ix < N - 1; ix++) {
                const int ctr = iz * N * N + iy * N + ix;
                out[ctr] = c0 * in[ctr]
                    + c1 * (in[ctr - 1]       + in[ctr + 1]
                          + in[ctr - N]        + in[ctr + N]
                          + in[ctr - N * N]    + in[ctr + N * N]);
            }
}

inline bool stencil_verify(const double* result, const double* ref,
                             int N, double tol, double* max_err_out) {
    double max_err = 0.0;
    for (int iz = 1; iz < N - 1; iz++)
        for (int iy = 1; iy < N - 1; iy++)
            for (int ix = 1; ix < N - 1; ix++) {
                const int idx = iz * N * N + iy * N + ix;
                double denom = (std::fabs(ref[idx]) < 1.0e-14) ? 1.0 : std::fabs(ref[idx]);
                double rel = std::fabs(result[idx] - ref[idx]) / denom;
                if (rel > max_err) max_err = rel;
            }
    if (max_err_out) *max_err_out = max_err;
    return max_err < tol;
}

// ── Output format helpers ─────────────────────────────────────────────────────
// run_stencil.sh anchors on "STENCIL_RUN", "STENCIL_HW_STATE", "STENCIL_SUMMARY".
// Keep these formats stable across all implementations.

inline void stencil_print_run(int run_id, int N, double time_ms, double gbs) {
    std::printf("STENCIL_RUN run=%d n=%d time_ms=%.6f throughput_gbs=%.6f\n",
                run_id, N, time_ms, gbs);
    std::fflush(stdout);
}

inline void stencil_print_hw_state(int run_id, int hw_state) {
    std::printf("STENCIL_HW_STATE run=%d hw_state=%d\n", run_id, hw_state);
    std::fflush(stdout);
}

inline void stencil_print_summary(int N, const StencilStats& s, int warmup_iters) {
    std::printf(
        "STENCIL_SUMMARY n=%d "
        "median_gbs=%.4f iqr_gbs=%.4f "
        "min_gbs=%.4f max_gbs=%.4f "
        "mean_gbs=%.4f n_clean=%d warmup_iters=%d\n",
        N, s.median_gbs, s.iqr_gbs, s.min_gbs, s.max_gbs,
        s.mean_gbs, s.n_clean, warmup_iters);
    std::fflush(stdout);
}
