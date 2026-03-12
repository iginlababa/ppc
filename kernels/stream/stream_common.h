// stream_common.h — Shared constants, types, and utilities for E1 STREAM Triad.
//
// Included by ALL abstraction implementations (CUDA, HIP, Kokkos, RAJA, SYCL,
// Julia wrapper, Numba wrapper).  Pure C++ / C — no GPU-specific headers.
//
// Initialization values and tolerance match BabelStream 4.0 reference and
// project_spec.md §7 (correctness tolerance 1e-6 relative error).

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

// ── Floating-point precision ──────────────────────────────────────────────────
// Build with -DSTREAM_USE_FLOAT to use single precision (non-default).
#ifdef STREAM_USE_FLOAT
using StreamFloat = float;
static constexpr const char* STREAM_PRECISION = "float";
#else
using StreamFloat = double;
static constexpr const char* STREAM_PRECISION = "double";
#endif

// ── Array initialization values ───────────────────────────────────────────────
// MUST be identical across all abstraction implementations — used by the
// cross-implementation correctness check in tests/correctness/.
static constexpr StreamFloat STREAM_INIT_A  = static_cast<StreamFloat>(0.1);
static constexpr StreamFloat STREAM_INIT_B  = static_cast<StreamFloat>(0.2);
static constexpr StreamFloat STREAM_INIT_C  = static_cast<StreamFloat>(0.0);
static constexpr StreamFloat STREAM_SCALAR  = static_cast<StreamFloat>(0.4);

// ── Default timing protocol (mirrors config.yaml) ────────────────────────────
static constexpr int STREAM_WARMUP_ITERS = 10;
static constexpr int STREAM_TIMED_ITERS  = 30;
static constexpr int STREAM_BLOCK_SIZE   = 1024;  // default GPU block size

// ── Correctness tolerance (project_spec.md §7, config.yaml) ──────────────────
static constexpr double STREAM_CORRECT_TOL = 1.0e-6;  // relative error

// ── Bandwidth formula: Triad moves 3 arrays ──────────────────────────────────
// BW (GB/s) = (3 * N * sizeof(StreamFloat)) / time_s / 1e9
inline double stream_bandwidth_gbs(size_t n, double time_s) {
    return (3.0 * static_cast<double>(n) * sizeof(StreamFloat)) / time_s / 1.0e9;
}

// Bandwidth per-operation — number of array reads+writes:
//   Copy  : 1 read + 1 write = 2 arrays
//   Mul   : 1 read + 1 write = 2 arrays
//   Add   : 2 reads + 1 write = 3 arrays
//   Triad : 2 reads + 1 write = 3 arrays  ← PRIMARY E1 metric
//   Dot   : 2 reads + 0 writes (result is scalar)
inline double copy_bandwidth_gbs(size_t n, double time_s) {
    return (2.0 * static_cast<double>(n) * sizeof(StreamFloat)) / time_s / 1.0e9;
}
inline double mul_bandwidth_gbs(size_t n, double time_s)   { return copy_bandwidth_gbs(n, time_s); }
inline double add_bandwidth_gbs(size_t n, double time_s)   { return stream_bandwidth_gbs(n, time_s); }
inline double triad_bandwidth_gbs(size_t n, double time_s) { return stream_bandwidth_gbs(n, time_s); }
inline double dot_bandwidth_gbs(size_t n, double time_s) {
    return (2.0 * static_cast<double>(n) * sizeof(StreamFloat)) / time_s / 1.0e9;
}

// ── Statistics ────────────────────────────────────────────────────────────────
struct StreamStats {
    double median;   // primary reported metric (not mean — project_spec.md §9.1)
    double iqr;      // interquartile range Q3-Q1
    double mean;
    double min;
    double max;
    int    n_outliers;   // runs > 2σ from mean
};

inline StreamStats compute_stats(std::vector<double> vals) {
    const size_t n = vals.size();
    StreamStats s{};
    if (n == 0) return s;

    std::vector<double> sorted = vals;
    std::sort(sorted.begin(), sorted.end());

    s.min  = sorted.front();
    s.max  = sorted.back();
    s.mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) / static_cast<double>(n);

    // Median
    s.median = (n % 2 == 0)
        ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        : sorted[n / 2];

    // IQR — Q3 at 75th percentile, Q1 at 25th percentile
    s.iqr = sorted[(3 * n) / 4] - sorted[n / 4];

    // Outliers: |val - mean| > 2σ
    double var = 0.0;
    for (double v : vals) var += (v - s.mean) * (v - s.mean);
    double sigma = std::sqrt(var / static_cast<double>(n));
    for (double v : vals)
        if (std::fabs(v - s.mean) > 2.0 * sigma) ++s.n_outliers;

    return s;
}

// ── Analytical expected values after N complete passes ────────────────────────
// Used by the correctness check to verify all elements without allocating a
// full reference array on the host.  Mirrors BabelStream's verify_solution().
//
// One "pass" = Copy → Mul → Add → Triad (in this order).
// Dot does not modify a/b/c so it doesn't affect expected values.
struct StreamExpected {
    double a, b, c;
};

inline StreamExpected compute_expected(int n_passes) {
    double a = static_cast<double>(STREAM_INIT_A);
    double b = static_cast<double>(STREAM_INIT_B);
    double c = static_cast<double>(STREAM_INIT_C);
    const double scalar = static_cast<double>(STREAM_SCALAR);
    for (int i = 0; i < n_passes; ++i) {
        c = a;                 // Copy
        b = scalar * c;        // Mul
        c = a + b;             // Add
        a = b + scalar * c;    // Triad
    }
    return {a, b, c};
}

// Verify a host buffer against expected value.  Returns max absolute relative
// error and a boolean pass/fail.
inline bool verify_array(const StreamFloat* buf, size_t n, double expected,
                         double tol, double* max_err_out) {
    double max_err = 0.0;
    const double denom = std::fabs(expected) < 1.0e-12 ? 1.0 : std::fabs(expected);
    for (size_t i = 0; i < n; ++i) {
        double rel = std::fabs(static_cast<double>(buf[i]) - expected) / denom;
        if (rel > max_err) max_err = rel;
    }
    if (max_err_out) *max_err_out = max_err;
    return max_err < tol;
}

// ── Output format helpers ─────────────────────────────────────────────────────
// parse_results.py regex anchors on "STREAM_RUN" and "STREAM_SUMMARY".
// Keep these formats stable across all abstraction implementations.

inline void print_run_line(const char* kernel_name, int run_id,
                            size_t n, double time_ms, double bw_gbs) {
    std::printf("STREAM_RUN kernel=%s run=%d n=%zu time_ms=%.5f bw_gbs=%.4f\n",
                kernel_name, run_id, n, time_ms, bw_gbs);
    std::fflush(stdout);
}

inline void print_summary(const char* kernel_name, const StreamStats& bw_stats) {
    std::printf(
        "STREAM_SUMMARY kernel=%s "
        "median_bw_gbs=%.4f iqr_bw_gbs=%.4f "
        "min_bw_gbs=%.4f max_bw_gbs=%.4f "
        "mean_bw_gbs=%.4f outliers=%d\n",
        kernel_name,
        bw_stats.median, bw_stats.iqr,
        bw_stats.min, bw_stats.max,
        bw_stats.mean, bw_stats.n_outliers);
    std::fflush(stdout);
}
