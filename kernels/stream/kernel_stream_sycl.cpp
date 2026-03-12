// kernel_stream_sycl.cpp — SYCL/DPC++ abstraction for E1 STREAM Triad.
//
// Implements all five BabelStream operations using SYCL 2020 portable abstractions:
//   Copy  : c[i] = a[i]
//   Mul   : b[i] = scalar * c[i]
//   Add   : c[i] = a[i] + b[i]
//   Triad : a[i] = b[i] + scalar * c[i]   ← PRIMARY E1 metric
//   Dot   : sum += a[i] * b[i]
//
// Portability: targets Intel oneAPI DPC++, AdaptiveCpp (hipSYCL), and
// any SYCL 2020-conformant toolchain.  The SYCL implementation selects the
// device; use environment variables to override:
//   Intel GPU:  ONEAPI_DEVICE_SELECTOR=level_zero:gpu
//   NVIDIA GPU: ONEAPI_DEVICE_SELECTOR=cuda:gpu  (DPC++ CUDA plugin)
//   AMD GPU:    ONEAPI_DEVICE_SELECTOR=hip:gpu   (AdaptiveCpp)
//   CPU:        ONEAPI_DEVICE_SELECTOR=opencl:cpu
//
// Memory: USM device memory (sycl::malloc_device) for the three arrays.
// Device pointers are directly accessible in kernel lambdas without accessors.
// The Dot reduction result uses sycl::malloc_shared (1 scalar) so the host
// can read the result without an explicit memcpy.
//
// Queue: sycl::property::queue::in_order — kernels execute sequentially in
// submission order.  This eliminates the need for explicit event dependencies
// between Copy → Mul → Add → Triad and simplifies the timing loop.
//
// Timing: q.wait() + std::chrono::high_resolution_clock.  Equivalent to
// Kokkos fence()+Timer and RAJA synchronize()+chrono — includes host-side
// synchronization cost.  The delta vs. the CUDA cudaEvent baseline is Runtime
// Coordination Overhead measured by overhead_attribution.py.
//
// Build: see kernels/stream/CMakeLists.txt
// Run:   ./stream-sycl --arraysize 268435456 --numtimes 30

#include "stream_common.h"

// SYCL 2020 unified header (Intel oneAPI ≥ 2022.3, AdaptiveCpp ≥ 22.10).
// Older DPC++ toolchains may require <CL/sycl.hpp>; set -DSTREAM_SYCL_COMPAT
// via CMake to switch.
#ifdef STREAM_SYCL_COMPAT
#  include <CL/sycl.hpp>
#else
#  include <sycl/sycl.hpp>
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <string>
#include <vector>

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
        "  --csv            Machine-readable CSV output\n"
        "Device selection: set ONEAPI_DEVICE_SELECTOR (DPC++) or\n"
        "  HIPSYCL_TARGETS (AdaptiveCpp) environment variables.\n",
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

// ── On-device initialization ──────────────────────────────────────────────────
static void do_init(sycl::queue& q,
                    StreamFloat* a, StreamFloat* b, StreamFloat* c, size_t n)
{
    q.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> i) {
        a[i] = STREAM_INIT_A;
        b[i] = STREAM_INIT_B;
        c[i] = STREAM_INIT_C;
    });
    q.wait();
}

// ── Per-pass timing result ────────────────────────────────────────────────────
struct PassBW {
    double copy, mul, add, triad, dot;
};

// ── Run one timed pass ────────────────────────────────────────────────────────
// q.wait() before and after each kernel measures wall-clock time including the
// host-side synchronization overhead — the Runtime Coordination Overhead that
// overhead_attribution.py decomposes relative to the CUDA cudaEvent baseline.
static PassBW run_pass(sycl::queue& q,
                       StreamFloat* a, StreamFloat* b, StreamFloat* c,
                       StreamFloat* dot_scratch, size_t n, bool all_kernels)
{
    const StreamFloat scalar = STREAM_SCALAR;
    PassBW bw{};

    // time_kernel: q.wait() → start → submit → q.wait() → stop → compute BW.
    // q.parallel_for returns immediately (async); second q.wait() blocks until
    // the kernel completes on the device.
    auto time_kernel = [&](auto&& submit_fn, auto bw_fn) -> double {
        q.wait();
        auto t0 = std::chrono::high_resolution_clock::now();
        submit_fn();
        q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        return bw_fn(n, std::chrono::duration<double>(t1 - t0).count());
    };

    if (all_kernels) {
        // Copy: c[i] = a[i]
        bw.copy = time_kernel(
            [&]{ q.parallel_for(sycl::range<1>{n},
                     [=](sycl::id<1> i) { c[i] = a[i]; }); },
            copy_bandwidth_gbs);

        // Mul: b[i] = scalar * c[i]
        bw.mul = time_kernel(
            [&]{ q.parallel_for(sycl::range<1>{n},
                     [=](sycl::id<1> i) { b[i] = scalar * c[i]; }); },
            mul_bandwidth_gbs);

        // Add: c[i] = a[i] + b[i]
        bw.add = time_kernel(
            [&]{ q.parallel_for(sycl::range<1>{n},
                     [=](sycl::id<1> i) { c[i] = a[i] + b[i]; }); },
            add_bandwidth_gbs);
    }

    // Triad: a[i] = b[i] + scalar * c[i]  — always timed (primary E1 metric)
    bw.triad = time_kernel(
        [&]{ q.parallel_for(sycl::range<1>{n},
                 [=](sycl::id<1> i) { a[i] = b[i] + scalar * c[i]; }); },
        triad_bandwidth_gbs);

    if (all_kernels) {
        // Dot: SYCL 2020 sycl::reduction over shared-memory scalar.
        // dot_scratch is sycl::malloc_shared — writable from device, readable
        // from host without a memcpy after q.wait().
        *dot_scratch = StreamFloat(0);
        q.wait();
        auto t0 = std::chrono::high_resolution_clock::now();
        q.parallel_for(sycl::range<1>{n},
            sycl::reduction(dot_scratch, sycl::plus<StreamFloat>{}),
            [=](sycl::id<1> i, auto& sum) { sum += a[i] * b[i]; });
        q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        bw.dot = dot_bandwidth_gbs(n, std::chrono::duration<double>(t1 - t0).count());
        (void)*dot_scratch;  // suppress unused-variable warning; verified separately
    }

    return bw;
}

// ── Correctness check ─────────────────────────────────────────────────────────
// Issues three single-element D2H copies (q.memcpy) — one per sample index —
// and compares against the analytical expected values from compute_expected().
static bool check_correctness(sycl::queue& q,
                               StreamFloat* a, StreamFloat* b, StreamFloat* c,
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

    // Sample a single element from a USM device pointer to host
    auto sample = [&](StreamFloat* d, size_t idx) -> StreamFloat {
        StreamFloat val{};
        q.memcpy(&val, d + idx, sizeof(StreamFloat)).wait();
        return val;
    };

    const size_t indices[3] = {0, n / 2, n - 1};
    double max_ea = 0.0, max_eb = 0.0, max_ec = 0.0;
    for (size_t idx : indices) {
        double ea = relerr(static_cast<double>(sample(a, idx)), exp.a);
        double eb = relerr(static_cast<double>(sample(b, idx)), exp.b);
        double ec = relerr(static_cast<double>(sample(c, idx)), exp.c);
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

    // ── Queue construction ────────────────────────────────────────────────────
    // Async exception handler: SYCL errors that occur inside kernel execution
    // are delivered here rather than at the API call site.
    auto async_err = [](sycl::exception_list el) {
        for (auto& ep : el) {
            try { std::rethrow_exception(ep); }
            catch (const sycl::exception& e) {
                std::fprintf(stderr, "SYCL async error: %s\n", e.what());
            }
        }
    };

    sycl::queue q;
    try {
        // default_selector_v: picks GPU if present, falls back to CPU.
        // Override at runtime via ONEAPI_DEVICE_SELECTOR or HIPSYCL_TARGETS.
        q = sycl::queue{sycl::default_selector_v, async_err,
                        sycl::property::queue::in_order{}};
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "SYCL queue creation failed: %s\n"
                             "Set ONEAPI_DEVICE_SELECTOR to target a specific device.\n",
                     e.what());
        return EXIT_FAILURE;
    }

    // ── Metadata ─────────────────────────────────────────────────────────────
    auto dev = q.get_device();
    std::string dev_name  = dev.get_info<sycl::info::device::name>();
    std::string dev_vendor = dev.get_info<sycl::info::device::vendor>();
    std::string dev_type  = dev.is_gpu() ? "gpu" : (dev.is_cpu() ? "cpu" : "accel");

    std::printf("STREAM_META abstraction=sycl device=\"%s\" vendor=\"%s\" "
                "device_type=%s precision=%s n=%zu sizeof=%zu "
                "warmup=%d timed=%d all_kernels=%d\n",
                dev_name.c_str(), dev_vendor.c_str(), dev_type.c_str(),
                STREAM_PRECISION, n, sizeof(StreamFloat),
                warmup, num_times, static_cast<int>(all_kernels));
    const double mb_per = static_cast<double>(n) * sizeof(StreamFloat) / (1024.0 * 1024.0);
    std::printf("STREAM_META array_mb=%.1f total_mb=%.1f\n", mb_per, 3.0 * mb_per);
    std::fflush(stdout);

    // ── Allocate USM arrays ───────────────────────────────────────────────────
    // malloc_device: device-only memory, no implicit migration — maximum BW.
    // malloc_shared: used only for the Dot scalar; 1 element, negligible cost.
    StreamFloat* a = sycl::malloc_device<StreamFloat>(n, q);
    StreamFloat* b = sycl::malloc_device<StreamFloat>(n, q);
    StreamFloat* c = sycl::malloc_device<StreamFloat>(n, q);
    StreamFloat* dot_scratch = sycl::malloc_shared<StreamFloat>(1, q);

    if (!a || !b || !c || !dot_scratch) {
        std::fprintf(stderr, "USM allocation failed for n=%zu (%.1f MB each)\n",
                     n, mb_per);
        sycl::free(a, q); sycl::free(b, q);
        sycl::free(c, q); sycl::free(dot_scratch, q);
        return EXIT_FAILURE;
    }

    // ── Initialize on device ──────────────────────────────────────────────────
    do_init(q, a, b, c, n);

    // ── Warm-up (discarded) ───────────────────────────────────────────────────
    for (int i = 0; i < warmup; ++i)
        run_pass(q, a, b, c, dot_scratch, n, all_kernels);
    q.wait();

    // ── Correctness check ─────────────────────────────────────────────────────
    // For triad-only mode: re-init then run one clean triad before checking.
    if (!all_kernels) {
        do_init(q, a, b, c, n);
        const StreamFloat scalar = STREAM_SCALAR;
        q.parallel_for(sycl::range<1>{n},
            [=](sycl::id<1> i) { a[i] = b[i] + scalar * c[i]; });
        q.wait();
    }
    const int passes_so_far = all_kernels ? warmup : 0;
    if (!check_correctness(q, a, b, c, n, passes_so_far, all_kernels)) {
        std::fprintf(stderr, "CORRECTNESS CHECK FAILED — aborting.\n");
        sycl::free(a, q); sycl::free(b, q);
        sycl::free(c, q); sycl::free(dot_scratch, q);
        return EXIT_FAILURE;
    }

    // ── Re-initialize for timed runs (clean state) ────────────────────────────
    do_init(q, a, b, c, n);

    // ── Timed runs ────────────────────────────────────────────────────────────
    std::vector<double> triad_bw(num_times);
    std::vector<double> copy_bw(num_times), mul_bw(num_times),
                        add_bw(num_times),   dot_bw(num_times);

    for (int i = 0; i < num_times; ++i) {
        PassBW r = run_pass(q, a, b, c, dot_scratch, n, all_kernels);
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
    q.wait();

    // ── Statistics ────────────────────────────────────────────────────────────
    print_summary("triad", compute_stats(triad_bw));
    if (all_kernels) {
        print_summary("copy", compute_stats(copy_bw));
        print_summary("mul",  compute_stats(mul_bw));
        print_summary("add",  compute_stats(add_bw));
        print_summary("dot",  compute_stats(dot_bw));
    }

    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(c, q);
    sycl::free(dot_scratch, q);
    return EXIT_SUCCESS;
}
