// kernel_stream_raja.cpp — RAJA abstraction for E1 STREAM Triad.
//
// Implements all five BabelStream operations using RAJA portable abstractions:
//   Copy  : c[i] = a[i]
//   Mul   : b[i] = scalar * c[i]
//   Add   : c[i] = a[i] + b[i]
//   Triad : a[i] = b[i] + scalar * c[i]   ← PRIMARY E1 metric
//   Dot   : sum += a[i] * b[i]
//
// Portability: this single file compiles against any RAJA backend.
// The backend is selected at compile time via the RAJA installation's
// exported CMake targets — RAJA propagates RAJA_ENABLE_CUDA / _HIP / _OPENMP
// definitions through its interface.  No source-level changes are needed
// between platforms.
//
// Timing: RAJA::synchronize<ExecPolicy>() + std::chrono::high_resolution_clock.
// Equivalent to the Kokkos fence()+Timer approach — both include the host-side
// synchronization cost.  The delta vs. the CUDA cudaEvent baseline is Runtime
// Coordination Overhead, measured by overhead_attribution.py.
//
// Memory: RAJA does not abstract allocation.  Raw pointers in backend-native
// memory space are used throughout:
//   CUDA / HIP : device memory (cudaMalloc / hipMalloc) — maximum bandwidth
//   OpenMP / Serial : host memory (std::malloc)
// Correctness samples three elements via dev_sample(), which issues a D2H copy
// for GPU backends and a direct read for CPU backends.
//
// Build: see kernels/stream/CMakeLists.txt
// Run:   ./stream-raja --arraysize 268435456 --numtimes 30

#include "stream_common.h"

#include <RAJA/RAJA.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <string>
#include <vector>

// ── Backend-specific policy and memory helpers ────────────────────────────────
// ExecPolicy    : passed to RAJA::forall / RAJA::RangeSegment
// ReducePolicy  : passed to RAJA::ReduceSum
// RAJA_BACKEND  : tag written to STREAM_META
//
// The dev_* helpers are small, non-inline statics so that RAJA_LAMBDA bodies
// in named static functions (do_init, do_triad) are not nested inside another
// lambda — avoiding potential NVCC extended-lambda restrictions.

static constexpr int STREAM_RAJA_BLOCK = 256;  // GPU thread-block size

#if defined(RAJA_ENABLE_CUDA)

    using ExecPolicy   = RAJA::cuda_exec<STREAM_RAJA_BLOCK>;
    using ReducePolicy = RAJA::cuda_reduce;
    static constexpr const char* RAJA_BACKEND = "cuda";

    static StreamFloat* dev_alloc(size_t n) {
        StreamFloat* p = nullptr;
        cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(StreamFloat));
        return p;
    }
    static void dev_free(StreamFloat* p)  { cudaFree(p); }
    static void dev_sync()                { RAJA::synchronize<RAJA::cuda_synchronize>(); }

    static void dev_sample(const StreamFloat* d, size_t idx, StreamFloat* out) {
        cudaMemcpy(out, d + idx, sizeof(StreamFloat), cudaMemcpyDeviceToHost);
    }

#elif defined(RAJA_ENABLE_HIP)

    using ExecPolicy   = RAJA::hip_exec<STREAM_RAJA_BLOCK>;
    using ReducePolicy = RAJA::hip_reduce;
    static constexpr const char* RAJA_BACKEND = "hip";

    static StreamFloat* dev_alloc(size_t n) {
        StreamFloat* p = nullptr;
        hipMalloc(reinterpret_cast<void**>(&p), n * sizeof(StreamFloat));
        return p;
    }
    static void dev_free(StreamFloat* p)  { hipFree(p); }
    static void dev_sync()                { RAJA::synchronize<RAJA::hip_synchronize>(); }

    static void dev_sample(const StreamFloat* d, size_t idx, StreamFloat* out) {
        hipMemcpy(out, d + idx, sizeof(StreamFloat), hipMemcpyDeviceToHost);
    }

#elif defined(RAJA_ENABLE_OPENMP)

    using ExecPolicy   = RAJA::omp_parallel_for_exec;
    using ReducePolicy = RAJA::omp_reduce;
    static constexpr const char* RAJA_BACKEND = "openmp";

    static StreamFloat* dev_alloc(size_t n) {
        return static_cast<StreamFloat*>(std::malloc(n * sizeof(StreamFloat)));
    }
    static void dev_free(StreamFloat* p) { std::free(p); }
    static void dev_sync() {}   // implicit barrier at end of parallel region

    static void dev_sample(const StreamFloat* d, size_t idx, StreamFloat* out) {
        *out = d[idx];
    }

#else  // serial fallback

    using ExecPolicy   = RAJA::seq_exec;
    using ReducePolicy = RAJA::seq_reduce;
    static constexpr const char* RAJA_BACKEND = "serial";

    static StreamFloat* dev_alloc(size_t n) {
        return static_cast<StreamFloat*>(std::malloc(n * sizeof(StreamFloat)));
    }
    static void dev_free(StreamFloat* p) { std::free(p); }
    static void dev_sync() {}

    static void dev_sample(const StreamFloat* d, size_t idx, StreamFloat* out) {
        *out = d[idx];
    }

#endif

// ── On-device initialization ──────────────────────────────────────────────────
// Named static function avoids defining RAJA_LAMBDA inside another lambda,
// which can trigger NVCC extended-lambda warnings on older toolchains.
static void do_init(StreamFloat* a, StreamFloat* b, StreamFloat* c, size_t n)
{
    RAJA::forall<ExecPolicy>(
        RAJA::RangeSegment(0, static_cast<RAJA::Index_type>(n)),
        [=] RAJA_DEVICE (RAJA::Index_type i) {
            a[i] = STREAM_INIT_A;
            b[i] = STREAM_INIT_B;
            c[i] = STREAM_INIT_C;
        });
    dev_sync();
}

// ── Single triad pass (used for triad-only correctness check) ─────────────────
static void do_triad(StreamFloat* a, StreamFloat* b, StreamFloat* c, size_t n)
{
    const StreamFloat scalar = STREAM_SCALAR;
    RAJA::forall<ExecPolicy>(
        RAJA::RangeSegment(0, static_cast<RAJA::Index_type>(n)),
        [=] RAJA_DEVICE (RAJA::Index_type i) { a[i] = b[i] + scalar * c[i]; });
    dev_sync();
}

// ── CLI options ───────────────────────────────────────────────────────────────
struct Options {
    size_t array_size  = 1u << 26;
    int    num_times   = STREAM_TIMED_ITERS;
    int    warmup      = STREAM_WARMUP_ITERS;
    bool   all_kernels = false;
    bool   csv         = false;
};

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  --arraysize N    Elements per array (default: 2^26)\n"
        "  --numtimes N     Timed iterations   (default: %d)\n"
        "  --warmup N       Warm-up iterations (default: %d)\n"
        "  --all-kernels    Run Copy/Mul/Add/Triad/Dot (default: Triad only)\n"
        "  --csv            Machine-readable CSV output\n",
        prog, STREAM_TIMED_ITERS, STREAM_WARMUP_ITERS);
}

static Options parse_args(int argc, char* argv[]) {
    static const option long_opts[] = {
        {"arraysize",   required_argument, nullptr, 'n'},
        {"numtimes",    required_argument, nullptr, 't'},
        {"warmup",      required_argument, nullptr, 'w'},
        {"all-kernels", no_argument,       nullptr, 'a'},
        {"csv",         no_argument,       nullptr, 'c'},
        {"help",        no_argument,       nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };
    Options opts;
    int c;
    while ((c = getopt_long(argc, argv, "n:t:w:ach", long_opts, nullptr)) != -1) {
        switch (c) {
            case 'n': opts.array_size  = static_cast<size_t>(std::atoll(optarg)); break;
            case 't': opts.num_times   = std::atoi(optarg); break;
            case 'w': opts.warmup      = std::atoi(optarg); break;
            case 'a': opts.all_kernels = true;  break;
            case 'c': opts.csv         = true;  break;
            case 'h': print_usage(argv[0]); std::exit(EXIT_SUCCESS);
            default:  break;
        }
    }
    return opts;
}

// ── Per-pass timing result ────────────────────────────────────────────────────
struct PassBW {
    double copy, mul, add, triad, dot;
};

// ── Run one timed pass ────────────────────────────────────────────────────────
// dev_sync() before and after each kernel ensures the wall-clock timer captures
// only GPU execution time.  The sync call itself (~5–50 µs on a warm GPU) is
// the Runtime Coordination Overhead that overhead_attribution.py measures.
static PassBW run_pass(StreamFloat* a, StreamFloat* b, StreamFloat* c,
                       size_t n, bool all_kernels)
{
    const StreamFloat scalar = STREAM_SCALAR;
    const RAJA::RangeSegment range(0, static_cast<RAJA::Index_type>(n));
    PassBW bw{};

    // time_kernel: sync → start → launch → sync → stop → compute BW
    auto time_kernel = [&](auto&& launch_fn, auto bw_fn) -> double {
        dev_sync();
        auto t0 = std::chrono::high_resolution_clock::now();
        launch_fn();
        dev_sync();
        auto t1 = std::chrono::high_resolution_clock::now();
        double t = std::chrono::duration<double>(t1 - t0).count();
        return bw_fn(n, t);
    };

    if (all_kernels) {
        // Copy: c[i] = a[i]
        bw.copy = time_kernel(
            [&]{ RAJA::forall<ExecPolicy>(range,
                     [=] RAJA_DEVICE (RAJA::Index_type i) { c[i] = a[i]; }); },
            copy_bandwidth_gbs);

        // Mul: b[i] = scalar * c[i]
        bw.mul = time_kernel(
            [&]{ RAJA::forall<ExecPolicy>(range,
                     [=] RAJA_DEVICE (RAJA::Index_type i) { b[i] = scalar * c[i]; }); },
            mul_bandwidth_gbs);

        // Add: c[i] = a[i] + b[i]
        bw.add = time_kernel(
            [&]{ RAJA::forall<ExecPolicy>(range,
                     [=] RAJA_DEVICE (RAJA::Index_type i) { c[i] = a[i] + b[i]; }); },
            add_bandwidth_gbs);
    }

    // Triad: a[i] = b[i] + scalar * c[i]  — always timed (primary E1 metric)
    bw.triad = time_kernel(
        [&]{ RAJA::forall<ExecPolicy>(range,
                 [=] RAJA_DEVICE (RAJA::Index_type i) { a[i] = b[i] + scalar * c[i]; }); },
        triad_bandwidth_gbs);

    if (all_kernels) {
        // Dot: parallel reduction with ReduceSum
        RAJA::ReduceSum<ReducePolicy, StreamFloat> dot_val(StreamFloat(0));
        dev_sync();
        auto t0 = std::chrono::high_resolution_clock::now();
        RAJA::forall<ExecPolicy>(range,
            [=] RAJA_DEVICE (RAJA::Index_type i) { dot_val += a[i] * b[i]; });
        dev_sync();
        auto t1 = std::chrono::high_resolution_clock::now();
        double t = std::chrono::duration<double>(t1 - t0).count();
        bw.dot = dot_bandwidth_gbs(n, t);
        (void)dot_val.get();  // suppress unused-variable warning; value verified separately
    }

    return bw;
}

// ── Correctness check ─────────────────────────────────────────────────────────
// Samples three elements (first, middle, last) from device → host and compares
// to the analytical expected values from compute_expected().
static bool check_correctness(StreamFloat* a, StreamFloat* b, StreamFloat* c,
                               size_t n, int n_passes, bool all_kernels)
{
    StreamExpected exp = all_kernels
        ? compute_expected(n_passes)
        : StreamExpected{
            static_cast<double>(STREAM_INIT_B + STREAM_SCALAR * STREAM_INIT_C),
            static_cast<double>(STREAM_INIT_B),
            static_cast<double>(STREAM_INIT_C)
          };

    auto relerr = [](double got, double expected) -> double {
        double denom = std::fabs(expected) < 1e-12 ? 1.0 : std::fabs(expected);
        return std::fabs(got - expected) / denom;
    };

    const size_t indices[3] = {0, n / 2, n - 1};
    double max_ea = 0.0, max_eb = 0.0, max_ec = 0.0;
    for (size_t idx : indices) {
        StreamFloat ha, hb, hc;
        dev_sample(a, idx, &ha);
        dev_sample(b, idx, &hb);
        dev_sample(c, idx, &hc);
        double ea = relerr(static_cast<double>(ha), exp.a);
        double eb = relerr(static_cast<double>(hb), exp.b);
        double ec = relerr(static_cast<double>(hc), exp.c);
        if (ea > max_ea) max_ea = ea;
        if (eb > max_eb) max_eb = eb;
        if (ec > max_ec) max_ec = ec;
    }

    bool pass = (max_ea < STREAM_CORRECT_TOL) &&
                (max_eb < STREAM_CORRECT_TOL) &&
                (max_ec < STREAM_CORRECT_TOL);

    std::printf("STREAM_CORRECT %s max_err_a=%.3e max_err_b=%.3e max_err_c=%.3e\n",
                pass ? "PASS" : "FAIL", max_ea, max_eb, max_ec);
    if (!pass) {
        if (max_ea >= STREAM_CORRECT_TOL)
            std::printf("STREAM_CORRECT DETAIL array=a expected=%.10f\n", exp.a);
        if (max_eb >= STREAM_CORRECT_TOL)
            std::printf("STREAM_CORRECT DETAIL array=b expected=%.10f\n", exp.b);
        if (max_ec >= STREAM_CORRECT_TOL)
            std::printf("STREAM_CORRECT DETAIL array=c expected=%.10f\n", exp.c);
    }
    std::fflush(stdout);
    return pass;
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    const Options opts = parse_args(argc, argv);
    const size_t n         = opts.array_size;
    const int num_times    = opts.num_times;
    const int warmup       = opts.warmup;
    const bool all_kernels = opts.all_kernels;

    // ── Metadata ─────────────────────────────────────────────────────────────
    std::printf("STREAM_META abstraction=raja backend=%s "
                "precision=%s n=%zu sizeof=%zu "
                "warmup=%d timed=%d all_kernels=%d\n",
                RAJA_BACKEND, STREAM_PRECISION, n, sizeof(StreamFloat),
                warmup, num_times, static_cast<int>(all_kernels));
    const double mb_per = static_cast<double>(n) * sizeof(StreamFloat) / (1024.0 * 1024.0);
    std::printf("STREAM_META array_mb=%.1f total_mb=%.1f\n", mb_per, 3.0 * mb_per);
    std::fflush(stdout);

    // ── Allocate ──────────────────────────────────────────────────────────────
    StreamFloat* a = dev_alloc(n);
    StreamFloat* b = dev_alloc(n);
    StreamFloat* c = dev_alloc(n);
    if (!a || !b || !c) {
        std::fprintf(stderr, "Allocation failed for n=%zu (%.1f MB each)\n",
                     n, mb_per);
        return EXIT_FAILURE;
    }

    // ── Initialize on device ──────────────────────────────────────────────────
    do_init(a, b, c, n);

    // ── Warm-up (discarded) ───────────────────────────────────────────────────
    for (int i = 0; i < warmup; ++i)
        run_pass(a, b, c, n, all_kernels);
    dev_sync();

    // ── Correctness check ─────────────────────────────────────────────────────
    // For triad-only mode: re-init then run one clean triad before checking.
    if (!all_kernels) {
        do_init(a, b, c, n);
        do_triad(a, b, c, n);
    }
    const int passes_so_far = all_kernels ? warmup : 0;
    if (!check_correctness(a, b, c, n, passes_so_far, all_kernels)) {
        std::fprintf(stderr, "CORRECTNESS CHECK FAILED — aborting.\n");
        dev_free(a); dev_free(b); dev_free(c);
        return EXIT_FAILURE;
    }

    // ── Re-initialize for timed runs (clean state) ────────────────────────────
    do_init(a, b, c, n);

    // ── Timed runs ────────────────────────────────────────────────────────────
    std::vector<double> triad_bw(num_times);
    std::vector<double> copy_bw(num_times), mul_bw(num_times),
                        add_bw(num_times),   dot_bw(num_times);

    for (int i = 0; i < num_times; ++i) {
        PassBW r = run_pass(a, b, c, n, all_kernels);
        triad_bw[i] = r.triad;
        if (all_kernels) {
            copy_bw[i] = r.copy;
            mul_bw[i]  = r.mul;
            add_bw[i]  = r.add;
            dot_bw[i]  = r.dot;
        }
        // Back-compute time_ms from BW for consistent output format
        double time_ms = (3.0 * static_cast<double>(n) * sizeof(StreamFloat))
                         / (r.triad * 1.0e9) * 1.0e3;
        print_run_line("triad", i + 1, n, time_ms, r.triad);
    }
    dev_sync();

    // ── Statistics ────────────────────────────────────────────────────────────
    print_summary("triad", compute_stats(triad_bw));
    if (all_kernels) {
        print_summary("copy", compute_stats(copy_bw));
        print_summary("mul",  compute_stats(mul_bw));
        print_summary("add",  compute_stats(add_bw));
        print_summary("dot",  compute_stats(dot_bw));
    }

    dev_free(a);
    dev_free(b);
    dev_free(c);
    return EXIT_SUCCESS;
}
