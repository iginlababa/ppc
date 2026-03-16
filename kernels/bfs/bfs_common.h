// bfs_common.h — Shared types and utilities for E6 BFS.
//
// Included by all C++ abstraction implementations (CUDA, Kokkos, RAJA).
// Julia carries equivalent definitions in-file.
//
// ── E6 DESIGN DECISIONS ───────────────────────────────────────────────────────
// [D1] Problem sizes: small=1024, medium=16384, large=65536 vertices (8GB VRAM safe).
// [D2] Graph types:
//      erdos_renyi — G(N, p) with p = avg_degree / N = 10 / N.  Undirected.
//                    Frontier profile: slow start, rapid expansion to ~0.3N, rapid
//                    collapse.  Highly irregular frontier widths across levels.
//      2d_grid     — sqrt(N)×sqrt(N) 4-neighbor grid.  BFS from corner (0,0).
//                    Regular diamond wavefront; peak width ≈ sqrt(N); symmetric
//                    growth+shrink around the anti-diagonal.
// [D3] Source vertex: 0 for both graph types.  For 2d_grid, vertex 0 = corner (0,0).
// [D4] Kernel: one thread per frontier vertex in the scatter phase.  One kernel
//      launch per phase (scatter + compact) per BFS level.
//      Compact: Thrust copy_if (native), parallel_scan (Kokkos), exclusive_scan
//      (RAJA), CUDA.cumsum (Julia).
// [D5] Metric: GTEPS = n_edges / time_s / 1e9.  Stored in throughput_gflops column
//      for CSV schema consistency with E2–E5.  For BFS there is no natural FLOP
//      count; "gflops" here means "giga traversal events per second".
// [D6] Correctness: compare GPU distances array against sequential CPU BFS.
//      All distances must match exactly (integer comparison).
// [D7] Warmup: adaptive CV < 2% over 10-run sliding window.  d_distances is reset
//      to -1 (source = 0) before EACH rep; reset time excluded from timed region.
// [D8] experiment_id: bfs_{abs}_{platform}_{graph}_{size_label}_n{N}_{run_id:03d}
// [D9] Frontier-width profile: computed CPU-side from bfs_cpu_ref before warmup.
//      Emitted as BFS_PROFILE line (captured by run script for fig24).
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
#include <queue>
#include <random>
#include <string>
#include <vector>

// ── Problem sizes ─────────────────────────────────────────────────────────────
static constexpr int    BFS_N_SMALL  = 1024;
static constexpr int    BFS_N_MEDIUM = 16384;
static constexpr int    BFS_N_LARGE  = 65536;

static constexpr double BFS_ERDOS_AVG_DEGREE = 10.0;
static constexpr uint64_t BFS_SEED           = 42ULL;

static constexpr int    BFS_BLOCK_SIZE   = 256;
static constexpr int    BFS_WARMUP_MAX   = 50;
static constexpr int    BFS_CV_WINDOW    = 10;
static constexpr double BFS_CV_TARGET    = 2.0;   // percent
static constexpr double BFS_HW_THRESHOLD = 0.15;  // 15% deviation → hw_state=0
static constexpr int    BFS_SOURCE       = 0;

// ── CSR graph (undirected: each undirected edge stored in both directions) ────
struct CsrGraph {
    std::vector<int> row_ptr;  // n_vertices + 1
    std::vector<int> col_idx;  // 2 * n_undirected_edges
    int n_vertices = 0;
    int n_edges    = 0;        // undirected edges (col_idx.size() / 2)
};

// ── BFS result ────────────────────────────────────────────────────────────────
struct BfsResult {
    std::vector<int> distances;       // -1 = unvisited
    std::vector<int> frontier_widths; // frontier_widths[l] = #vertices discovered at level l
    int n_levels           = 0;
    int max_frontier_width = 0;
    int min_frontier_width = 0;
    int n_visited          = 0;
};

// ── Graph generators ──────────────────────────────────────────────────────────

// Undirected Erdős–Rényi G(N, p=avg_degree/N).
// Uses geometric-skip algorithm: O(N + E) expected time.
inline CsrGraph generate_erdos_renyi(int n, uint64_t seed = BFS_SEED) {
    double p = BFS_ERDOS_AVG_DEGREE / n;
    std::mt19937_64 rng(seed);
    std::geometric_distribution<int> skip(p);

    // Collect edge list (i < j)
    std::vector<std::pair<int,int>> edges;
    edges.reserve((int)(n * BFS_ERDOS_AVG_DEGREE / 2 * 1.1));
    for (int u = 0; u < n; u++) {
        int v = u + 1 + skip(rng);
        while (v < n) {
            edges.emplace_back(u, v);
            v += 1 + skip(rng);
        }
    }

    // Build CSR (both directions)
    int m = (int)edges.size();
    std::vector<int> degree(n, 0);
    for (auto& [u, v] : edges) { degree[u]++; degree[v]++; }
    std::vector<int> row_ptr(n + 1, 0);
    for (int i = 0; i < n; i++) row_ptr[i + 1] = row_ptr[i] + degree[i];
    std::vector<int> col_idx(2 * m);
    std::vector<int> pos = row_ptr;
    for (auto& [u, v] : edges) {
        col_idx[pos[u]++] = v;
        col_idx[pos[v]++] = u;
    }
    // Sort each row for deterministic adjacency order
    for (int i = 0; i < n; i++) {
        std::sort(col_idx.begin() + row_ptr[i], col_idx.begin() + row_ptr[i + 1]);
    }
    CsrGraph g;
    g.n_vertices = n;
    g.n_edges    = m;
    g.row_ptr    = std::move(row_ptr);
    g.col_idx    = std::move(col_idx);
    return g;
}

// 2D grid graph: sqrt(N) × sqrt(N) with 4-neighbor connectivity.
// N must be a perfect square.  BFS source = vertex 0 = corner (0,0).
inline CsrGraph generate_2d_grid(int n) {
    int side = (int)std::round(std::sqrt((double)n));
    assert(side * side == n && "N must be a perfect square for 2d_grid");

    std::vector<int> degree(n, 0);
    // count degrees first
    for (int r = 0; r < side; r++) {
        for (int c = 0; c < side; c++) {
            int u = r * side + c;
            if (r > 0)        degree[u]++;
            if (r < side - 1) degree[u]++;
            if (c > 0)        degree[u]++;
            if (c < side - 1) degree[u]++;
        }
    }
    std::vector<int> row_ptr(n + 1, 0);
    for (int i = 0; i < n; i++) row_ptr[i + 1] = row_ptr[i] + degree[i];
    std::vector<int> col_idx(row_ptr[n]);
    std::vector<int> pos = row_ptr;
    for (int r = 0; r < side; r++) {
        for (int c = 0; c < side; c++) {
            int u = r * side + c;
            if (r > 0)        col_idx[pos[u]++] = (r-1)*side + c;
            if (r < side - 1) col_idx[pos[u]++] = (r+1)*side + c;
            if (c > 0)        col_idx[pos[u]++] = r*side + (c-1);
            if (c < side - 1) col_idx[pos[u]++] = r*side + (c+1);
        }
    }
    int m = row_ptr[n] / 2;
    CsrGraph g;
    g.n_vertices = n;
    g.n_edges    = m;
    g.row_ptr    = std::move(row_ptr);
    g.col_idx    = std::move(col_idx);
    return g;
}

// ── CPU BFS reference ─────────────────────────────────────────────────────────
inline BfsResult bfs_cpu_ref(const CsrGraph& g, int source = BFS_SOURCE) {
    BfsResult r;
    r.distances.assign(g.n_vertices, -1);
    r.distances[source] = 0;

    std::vector<int> frontier = {source};
    int level = 0;
    while (!frontier.empty()) {
        r.frontier_widths.push_back((int)frontier.size());
        std::vector<int> next;
        next.reserve(frontier.size() * 2);
        for (int u : frontier) {
            for (int j = g.row_ptr[u]; j < g.row_ptr[u + 1]; j++) {
                int v = g.col_idx[j];
                if (r.distances[v] == -1) {
                    r.distances[v] = level + 1;
                    next.push_back(v);
                }
            }
        }
        frontier = std::move(next);
        level++;
    }
    r.n_levels = level;
    if (!r.frontier_widths.empty()) {
        r.max_frontier_width = *std::max_element(r.frontier_widths.begin(),
                                                   r.frontier_widths.end());
        r.min_frontier_width = *std::min_element(r.frontier_widths.begin(),
                                                   r.frontier_widths.end());
    }
    r.n_visited = (int)std::count_if(r.distances.begin(), r.distances.end(),
                                      [](int d) { return d >= 0; });
    return r;
}

// ── Correctness check ─────────────────────────────────────────────────────────
inline bool bfs_verify(const std::vector<int>& dist_gpu,
                        const std::vector<int>& dist_ref)
{
    if (dist_gpu.size() != dist_ref.size()) {
        std::fprintf(stderr, "[bfs_verify] size mismatch: gpu=%zu ref=%zu\n",
                     dist_gpu.size(), dist_ref.size());
        return false;
    }
    int nerr = 0;
    for (int i = 0; i < (int)dist_ref.size() && nerr < 10; i++) {
        if (dist_gpu[i] != dist_ref[i]) {
            std::fprintf(stderr, "[bfs_verify] mismatch at v=%d: gpu=%d ref=%d\n",
                         i, dist_gpu[i], dist_ref[i]);
            nerr++;
        }
    }
    return nerr == 0;
}

// ── Emit frontier-width profile (captured by run script for fig24) ────────────
inline void bfs_print_profile(const BfsResult& ref,
                                const std::string& graph_type,
                                const std::string& size_label)
{
    std::printf("BFS_PROFILE graph=%s size=%s n_levels=%d widths=",
                graph_type.c_str(), size_label.c_str(), ref.n_levels);
    for (int i = 0; i < (int)ref.frontier_widths.size(); i++) {
        if (i) std::printf(",");
        std::printf("%d", ref.frontier_widths[i]);
    }
    std::printf("\n");
    std::fflush(stdout);
}

// ── Print one timed BFS run result ────────────────────────────────────────────
inline void bfs_print_run(int run, int n_vertices, int n_edges,
                           int n_levels, int max_fw, int min_fw,
                           double peak_ff,
                           const std::string& graph_type,
                           const std::string& size_label,
                           double time_ms, double throughput_gteps,
                           bool hw_ok)
{
    std::printf("BFS_RUN run=%d n_vertices=%d n_edges=%d n_levels=%d "
                "max_fw=%d min_fw=%d peak_ff=%.6f graph=%s size=%s "
                "time_ms=%.4f throughput_gflops=%.6f\n",
                run, n_vertices, n_edges, n_levels, max_fw, min_fw,
                peak_ff, graph_type.c_str(), size_label.c_str(),
                time_ms, throughput_gteps);
    std::printf("BFS_HW_STATE state=%d\n", hw_ok ? 1 : 0);
    std::fflush(stdout);
}

// ── Statistics helpers ────────────────────────────────────────────────────────
struct BfsRunStats {
    double mean   = 0;
    double cv_pct = 0;
};

inline BfsRunStats bfs_compute_stats(const std::deque<double>& w) {
    BfsRunStats s;
    if (w.empty()) return s;
    for (double v : w) s.mean += v;
    s.mean /= (double)w.size();
    if (w.size() < 2 || s.mean <= 0) return s;
    double var = 0;
    for (double v : w) var += (v - s.mean) * (v - s.mean);
    s.cv_pct = 100.0 * std::sqrt(var / (double)w.size()) / s.mean;
    return s;
}

// ── Adaptive warmup ───────────────────────────────────────────────────────────
// RunFn: () → double (time in ms).  Includes distance-reset overhead (excluded
// from timed reps).
template <typename RunFn>
void bfs_adaptive_warmup(RunFn run_once,
                          const std::string& label = "",
                          bool verbose = false)
{
    std::deque<double> window;
    for (int i = 0; i < BFS_WARMUP_MAX; i++) {
        double ms = run_once();
        window.push_back(ms);
        if ((int)window.size() > BFS_CV_WINDOW) window.pop_front();
        if ((int)window.size() == BFS_CV_WINDOW) {
            auto s = bfs_compute_stats(window);
            if (verbose) {
                std::fprintf(stderr, "[warmup] %s iter=%d mean=%.3f ms CV=%.2f%%\n",
                             label.c_str(), i, s.mean, s.cv_pct);
            }
            if (s.cv_pct < BFS_CV_TARGET) {
                if (verbose)
                    std::fprintf(stderr, "[warmup] %s converged at iter=%d\n",
                                 label.c_str(), i);
                break;
            }
        }
    }
}

// ── hw_state_verified (15% deviation from running median) ────────────────────
// Returns 1 if clean, 0 if dirty.
inline int bfs_hw_state(double time_ms, const std::vector<double>& history) {
    if (history.empty()) return 1;
    std::vector<double> sorted = history;
    std::sort(sorted.begin(), sorted.end());
    double med = sorted[sorted.size() / 2];
    if (med <= 0) return 1;
    double dev = std::fabs(time_ms - med) / med;
    return (dev <= BFS_HW_THRESHOLD) ? 1 : 0;
}

// ── Size label helper ─────────────────────────────────────────────────────────
inline std::string bfs_size_label(int n) {
    if (n == BFS_N_SMALL)  return "small";
    if (n == BFS_N_MEDIUM) return "medium";
    if (n == BFS_N_LARGE)  return "large";
    return "unknown";
}

// ── Command-line parsing helpers ──────────────────────────────────────────────
struct BfsConfig {
    std::string graph_type  = "erdos_renyi";  // erdos_renyi | 2d_grid
    std::string size_label  = "small";
    int n                   = BFS_N_SMALL;
    int reps                = 30;
    bool verify             = false;
    std::string platform    = "nvidia_rtx5060_laptop";
};

inline BfsConfig bfs_parse_args(int argc, char** argv) {
    BfsConfig cfg;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--graph" && i + 1 < argc) {
            cfg.graph_type = argv[++i];
        } else if (a == "--n" && i + 1 < argc) {
            cfg.n = std::atoi(argv[++i]);
            cfg.size_label = bfs_size_label(cfg.n);
        } else if (a == "--size" && i + 1 < argc) {
            cfg.size_label = argv[++i];
            if      (cfg.size_label == "small")  cfg.n = BFS_N_SMALL;
            else if (cfg.size_label == "medium") cfg.n = BFS_N_MEDIUM;
            else if (cfg.size_label == "large")  cfg.n = BFS_N_LARGE;
        } else if (a == "--reps" && i + 1 < argc) {
            cfg.reps = std::atoi(argv[++i]);
        } else if (a == "--verify") {
            cfg.verify = true;
        } else if (a == "--platform" && i + 1 < argc) {
            cfg.platform = argv[++i];
        }
    }
    return cfg;
}
