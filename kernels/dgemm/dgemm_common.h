// dgemm_common.h — Shared constants, types, and utilities for E2 DGEMM.
//
// Included by all C++ abstraction implementations (CUDA, Kokkos, RAJA, SYCL).
// Julia and Numba have their own equivalents in-file.
//
// ── E2 DESIGN DECISIONS ───────────────────────────────────────────────────────
// [D1] Problem sizes: small=1024, medium=4096, large=8192.
//      project_spec.md §8.2 originally said N=16384, but 3×16384²×8 ≈ 6.3 GB
//      leaves too little headroom on 8 GB VRAM. N=8192 (1.6 GB) is the override.
// [D2] Tile: 32×32 threads, each thread computes one C element.
//      Apples-to-apples across abstractions; cuBLAS is the "what's possible" ceiling.
// [D3] alpha=1.0, beta=0.0 (pure C = A*B). Simplifies all kernels identically.
// [D4] Precision: FP64 (double). This is a compute-bound FP64 kernel on purpose.
// [D5] native_cublas / julia_cublas are ceiling references, NOT PPC baselines.
//      PPC denominator = native (hand-coded tiled CUDA).
// [D6] raja_naive: naïve RAJA::forall, no shared memory. This IS the finding:
//      API Limitation pattern. Expected ~7× gap vs tiled due to no data reuse.
// [D7] experiment_id format: dgemm_{abstraction}_{platform}_{size_label}_n{N}_{run_id:03d}
// ─────────────────────────────────────────────────────────────────────────────

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

// ── Problem sizes ─────────────────────────────────────────────────────────────
static constexpr int DGEMM_N_SMALL  = 1024;
static constexpr int DGEMM_N_MEDIUM = 4096;
static constexpr int DGEMM_N_LARGE  = 8192;

// ── Tile dimensions (D2) ──────────────────────────────────────────────────────
static constexpr int DGEMM_TILE = 32;

// ── BLAS parameters (D3) ─────────────────────────────────────────────────────
static constexpr double DGEMM_ALPHA = 1.0;
static constexpr double DGEMM_BETA  = 0.0;

// ── Timing protocol ───────────────────────────────────────────────────────────
// warmup=50 required on dynamic-clock platforms (RTX 5060 Laptop, §5.5).
// HBM platforms (A100 etc.) use warmup=10; override via --warmup flag.
static constexpr int DGEMM_WARMUP_DEFAULT = 50;
static constexpr int DGEMM_REPS_DEFAULT   = 30;

// ── Correctness tolerance ─────────────────────────────────────────────────────
static constexpr double DGEMM_CORRECT_TOL = 1.0e-6;  // relative, FP64; covers tiled reorder roundoff

// ── GFLOP/s formula (project_spec.md §9.2) ───────────────────────────────────
// throughput_gflops = 2 * N^3 / (time_s * 1e9)
inline double dgemm_gflops(long N, double time_s) {
    return 2.0 * static_cast<double>(N) * static_cast<double>(N)
               * static_cast<double>(N) / time_s / 1.0e9;
}

// ── Size label (D7) ───────────────────────────────────────────────────────────
inline const char* dgemm_size_label(int N) {
    if (N <= DGEMM_N_SMALL)  return "small";
    if (N <= DGEMM_N_MEDIUM) return "medium";
    return "large";
}

// ── hw_state_verified logic (project_spec.md §9.7) ───────────────────────────
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
struct DgemmStats {
    double median;
    double iqr;
    double mean;
    double min;
    double max;
    int    n_clean;   // count where hw_state_verified == 1
};

inline DgemmStats compute_dgemm_stats(const std::vector<double>& vals,
                                       const std::vector<int>&    hw) {
    DgemmStats s{};
    std::vector<double> clean;
    clean.reserve(vals.size());
    for (size_t i = 0; i < vals.size(); i++)
        if (hw[i] == 1) clean.push_back(vals[i]);
    s.n_clean = static_cast<int>(clean.size());
    if (clean.empty()) return s;

    std::sort(clean.begin(), clean.end());
    size_t n = clean.size();
    s.min  = clean.front();
    s.max  = clean.back();
    s.mean = 0.0;
    for (double v : clean) s.mean += v;
    s.mean /= static_cast<double>(n);
    s.median = (n % 2 == 0)
        ? (clean[n / 2 - 1] + clean[n / 2]) / 2.0
        : clean[n / 2];
    s.iqr = clean[(3 * n) / 4] - clean[n / 4];
    return s;
}

// ── CPU reference DGEMM (correctness only, N <= 256) ─────────────────────────
inline void dgemm_cpu_ref(int N, double alpha,
                           const double* A, const double* B,
                           double beta, double* C) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
}

// ── Correctness check ─────────────────────────────────────────────────────────
inline bool dgemm_verify(const double* result, const double* ref,
                          int N, double tol, double* max_err_out) {
    double max_err = 0.0;
    for (int i = 0; i < N * N; i++) {
        double denom = (std::fabs(ref[i]) < 1.0e-12) ? 1.0 : std::fabs(ref[i]);
        double rel = std::fabs(result[i] - ref[i]) / denom;
        if (rel > max_err) max_err = rel;
    }
    if (max_err_out) *max_err_out = max_err;
    return max_err < tol;
}

// ── Output format helpers ─────────────────────────────────────────────────────
// run_dgemm.sh anchors on "DGEMM_RUN", "DGEMM_HW_STATE", "DGEMM_SUMMARY".
// Keep these formats stable across all implementations.

inline void dgemm_print_run(int run_id, int N, double time_ms, double gflops) {
    std::printf("DGEMM_RUN run=%d n=%d time_ms=%.6f gflops=%.6f\n",
                run_id, N, time_ms, gflops);
    std::fflush(stdout);
}

inline void dgemm_print_hw_state(int run_id, int hw_state) {
    std::printf("DGEMM_HW_STATE run=%d hw_state=%d\n", run_id, hw_state);
    std::fflush(stdout);
}

inline void dgemm_print_summary(int N, const DgemmStats& s) {
    std::printf(
        "DGEMM_SUMMARY n=%d "
        "median_gflops=%.4f iqr_gflops=%.4f "
        "min_gflops=%.4f max_gflops=%.4f "
        "mean_gflops=%.4f n_clean=%d\n",
        N, s.median, s.iqr, s.min, s.max, s.mean, s.n_clean);
    std::fflush(stdout);
}
