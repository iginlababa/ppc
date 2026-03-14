// kernel_stream_kokkos.cpp — Kokkos abstraction for E1 STREAM Triad.
//
// Implements all five BabelStream operations using Kokkos portable abstractions:
//   Copy  : c(i) = a(i)
//   Mul   : b(i) = scalar * c(i)
//   Add   : c(i) = a(i) + b(i)
//   Triad : a(i) = b(i) + scalar * c(i)   ← PRIMARY E1 metric
//   Dot   : sum += a(i) * b(i)
//
// Portability: this single file compiles against any Kokkos backend.
// The backend is selected at CMake time via Kokkos_ENABLE_CUDA / _HIP / _SYCL /
// _OPENMP.  No source-level changes are needed between platforms.
//
// Timing: Kokkos::Timer (wall-clock) bracketed by Kokkos::fence() calls.
// This differs from the CUDA baseline's cudaEvent timing by including the cost
// of the host-side fence call.  For large N (≥ 2^26), this overhead is
// sub-millisecond and negligible relative to kernel time.  For small N, the
// delta is measured by the overhead_attribution pipeline.
//
// Memory layout: 1D Kokkos::View — layout is irrelevant for 1D arrays
// (LayoutLeft == LayoutRight for rank-1 Views).  Layout sensitivity is
// documented for the stencil (E3) and DGEMM (E2) experiments.
//
// Build: see kernels/stream/CMakeLists.txt
// Run:   ./stream-kokkos --arraysize 268435456 --numtimes 30

#include "stream_common.h"

#include <Kokkos_Core.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <string>
#include <vector>

// ── Kokkos type aliases ───────────────────────────────────────────────────────
// DeviceView: lives in the Kokkos default memory space (device memory for GPU,
//             host memory for OpenMP/Serial).
using DeviceView  = Kokkos::View<StreamFloat*>;
using HostMirror  = DeviceView::host_mirror_type;  // Kokkos ≥ 4: use host_mirror_type

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
        "Usage: %s [options] [--kokkos-*]\n"
        "  --arraysize N    Elements per array (default: 2^26)\n"
        "  --numtimes N     Timed iterations   (default: %d)\n"
        "  --warmup N       Warm-up iterations (default: %d)\n"
        "  --all-kernels    Run Copy/Mul/Add/Triad/Dot (default: Triad only)\n"
        "  --csv            Machine-readable CSV output\n"
        "  --kokkos-*       Forwarded to Kokkos::initialize (e.g. --kokkos-num-threads)\n",
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
    // getopt skips unrecognised long options starting with "--kokkos-"
    opterr = 0;
    while ((c = getopt_long(argc, argv, "n:t:w:ach", long_opts, nullptr)) != -1) {
        switch (c) {
            case 'n': opts.array_size  = static_cast<size_t>(std::atoll(optarg)); break;
            case 't': opts.num_times   = std::atoi(optarg); break;
            case 'w': opts.warmup      = std::atoi(optarg); break;
            case 'a': opts.all_kernels = true;  break;
            case 'c': opts.csv         = true;  break;
            case 'h': print_usage(argv[0]); std::exit(EXIT_SUCCESS);
            default:  break;   // skip --kokkos-* and other unknown opts
        }
    }
    return opts;
}

// ── Per-pass timing result ────────────────────────────────────────────────────
struct PassBW {
    double copy, mul, add, triad, dot;
};

// ── Run one timed pass — returns bandwidth per kernel ─────────────────────────
// Kokkos::fence() before and after each kernel ensures the timer captures
// only GPU execution.  The fence itself costs ~5–50 µs on a warm GPU; this
// is the Runtime Coordination Overhead that overhead_attribution.py measures.
static PassBW run_pass(DeviceView& a, DeviceView& b, DeviceView& c,
                       size_t n, bool all_kernels)
{
    const StreamFloat scalar = STREAM_SCALAR;
    PassBW bw{};

    auto time_kernel = [&](auto&& launch_fn, auto bw_fn) -> double {
        Kokkos::fence();
        Kokkos::Timer timer;
        launch_fn();
        Kokkos::fence();
        double t = timer.seconds();
        return bw_fn(n, t);
    };

    if (all_kernels) {
        // Copy: c(i) = a(i)
        bw.copy = time_kernel(
            [&]{ Kokkos::parallel_for("copy", n,
                     KOKKOS_LAMBDA(const size_t i) { c(i) = a(i); }); },
            copy_bandwidth_gbs);

        // Mul: b(i) = scalar * c(i)
        bw.mul = time_kernel(
            [&]{ Kokkos::parallel_for("mul", n,
                     KOKKOS_LAMBDA(const size_t i) { b(i) = scalar * c(i); }); },
            mul_bandwidth_gbs);

        // Add: c(i) = a(i) + b(i)
        bw.add = time_kernel(
            [&]{ Kokkos::parallel_for("add", n,
                     KOKKOS_LAMBDA(const size_t i) { c(i) = a(i) + b(i); }); },
            add_bandwidth_gbs);
    }

    // Triad: a(i) = b(i) + scalar * c(i)  — always timed (primary E1 metric)
    bw.triad = time_kernel(
        [&]{ Kokkos::parallel_for("triad", n,
                 KOKKOS_LAMBDA(const size_t i) { a(i) = b(i) + scalar * c(i); }); },
        triad_bandwidth_gbs);

    if (all_kernels) {
        // Dot: parallel_reduce with default Sum reducer
        StreamFloat dot_val = 0;
        Kokkos::fence();
        Kokkos::Timer dot_timer;
        Kokkos::parallel_reduce("dot",
            Kokkos::RangePolicy<>(0, n),
            KOKKOS_LAMBDA(const size_t i, StreamFloat& local) {
                local += a(i) * b(i);
            },
            dot_val);
        Kokkos::fence();
        double dot_time = dot_timer.seconds();
        bw.dot = dot_bandwidth_gbs(n, dot_time);
        (void)dot_val;  // suppress unused-variable warning; value verified separately
    }

    return bw;
}

// ── Correctness check ─────────────────────────────────────────────────────────
// Copies three sample elements from device → host and checks against the
// analytical expected values from compute_expected().
static bool check_correctness(DeviceView& a, DeviceView& b, DeviceView& c,
                               size_t n, int n_passes, bool all_kernels)
{
    // For triad-only mode b and c are unchanged; passes = 0 gives initial values
    // but expected a = INIT_B + SCALAR * INIT_C regardless of pass count.
    // For all-kernels mode use compute_expected(n_passes).
    StreamExpected exp = all_kernels
        ? compute_expected(n_passes)
        : StreamExpected{
            static_cast<double>(STREAM_INIT_B + STREAM_SCALAR * STREAM_INIT_C),
            static_cast<double>(STREAM_INIT_B),
            static_cast<double>(STREAM_INIT_C)
          };

    // Create host mirrors and copy back a small sample
    HostMirror h_a = Kokkos::create_mirror_view(a);
    HostMirror h_b = Kokkos::create_mirror_view(b);
    HostMirror h_c = Kokkos::create_mirror_view(c);
    Kokkos::deep_copy(h_a, a);
    Kokkos::deep_copy(h_b, b);
    Kokkos::deep_copy(h_c, c);

    auto relerr = [](double got, double expected) -> double {
        double denom = std::fabs(expected) < 1e-12 ? 1.0 : std::fabs(expected);
        return std::fabs(got - expected) / denom;
    };

    // Check first, middle, and last elements
    const size_t indices[3] = {0, n / 2, n - 1};
    double max_ea = 0.0, max_eb = 0.0, max_ec = 0.0;
    for (size_t idx : indices) {
        double ea = relerr(static_cast<double>(h_a(idx)), exp.a);
        double eb = relerr(static_cast<double>(h_b(idx)), exp.b);
        double ec = relerr(static_cast<double>(h_c(idx)), exp.c);
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
    // Kokkos::initialize MUST be called before any Kokkos objects are created.
    // It parses and removes --kokkos-* arguments from argc/argv.
    Kokkos::initialize(argc, argv);

    // Parse remaining (non-Kokkos) arguments
    const Options opts = parse_args(argc, argv);
    const size_t n         = opts.array_size;
    const int num_times    = opts.num_times;
    const int warmup       = opts.warmup;
    const bool all_kernels = opts.all_kernels;

    // ── Metadata ─────────────────────────────────────────────────────────────
    std::printf("STREAM_META abstraction=kokkos execution_space=%s "
                "precision=%s n=%zu sizeof=%zu "
                "warmup=%d timed=%d all_kernels=%d\n",
                Kokkos::DefaultExecutionSpace::name(),
                STREAM_PRECISION, n, sizeof(StreamFloat),
                warmup, num_times, static_cast<int>(all_kernels));
    const double mb_per = static_cast<double>(n) * sizeof(StreamFloat) / (1024.0 * 1024.0);
    std::printf("STREAM_META array_mb=%.1f total_mb=%.1f\n", mb_per, 3.0 * mb_per);
    std::fflush(stdout);

    {   // Kokkos scope — all Views must be destroyed before Kokkos::finalize()
        // ── Allocate Views ────────────────────────────────────────────────────
        // Kokkos::View default constructor allocates in the default memory space
        // (device memory for CUDA/HIP, pinned host for OpenMP).
        DeviceView a("a", n), b("b", n), c("c", n);

        // ── Initialize on device ──────────────────────────────────────────────
        // Use parallel_for for initialization to avoid PCIe transfer overhead.
        Kokkos::parallel_for("init", n, KOKKOS_LAMBDA(const size_t i) {
            a(i) = STREAM_INIT_A;
            b(i) = STREAM_INIT_B;
            c(i) = STREAM_INIT_C;
        });
        Kokkos::fence();

        // ── Warm-up (discarded) ───────────────────────────────────────────────
        for (int i = 0; i < warmup; ++i)
            run_pass(a, b, c, n, all_kernels);
        Kokkos::fence();

        // ── Correctness check ─────────────────────────────────────────────────
        // For triad-only: re-init for a clean single-pass check.
        if (!all_kernels) {
            Kokkos::parallel_for("reinit", n, KOKKOS_LAMBDA(const size_t i) {
                a(i) = STREAM_INIT_A;
                b(i) = STREAM_INIT_B;
                c(i) = STREAM_INIT_C;
            });
            const StreamFloat scalar = STREAM_SCALAR;
            Kokkos::parallel_for("triad_check", n,
                KOKKOS_LAMBDA(const size_t i) { a(i) = b(i) + scalar * c(i); });
            Kokkos::fence();
        }
        const int passes_so_far = all_kernels ? warmup : 0;
        if (!check_correctness(a, b, c, n, passes_so_far, all_kernels)) {
            std::fprintf(stderr, "CORRECTNESS CHECK FAILED — aborting.\n");
            Kokkos::finalize();
            return EXIT_FAILURE;
        }

        // Re-initialize for the actual timing runs (clean state)
        Kokkos::parallel_for("reinit_timed", n, KOKKOS_LAMBDA(const size_t i) {
            a(i) = STREAM_INIT_A;
            b(i) = STREAM_INIT_B;
            c(i) = STREAM_INIT_C;
        });
        Kokkos::fence();

        // ── Timed runs ────────────────────────────────────────────────────────
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
            // time_ms back-computed from BW for consistent output format
            double time_ms = (3.0 * static_cast<double>(n) * sizeof(StreamFloat))
                             / (r.triad * 1.0e9) * 1.0e3;
            print_run_line("triad", i + 1, n, time_ms, r.triad);
        }
        Kokkos::fence();

        // ── Statistics ────────────────────────────────────────────────────────
        print_summary("triad", compute_stats(triad_bw));
        if (all_kernels) {
            print_summary("copy", compute_stats(copy_bw));
            print_summary("mul",  compute_stats(mul_bw));
            print_summary("add",  compute_stats(add_bw));
            print_summary("dot",  compute_stats(dot_bw));
        }

    }   // end Kokkos scope — Views destroyed here

    Kokkos::finalize();
    return EXIT_SUCCESS;
}
