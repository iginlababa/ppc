// kernel_stream_hip.cpp — HIP native baseline for E1 STREAM Triad.
//
// AMD HIP counterpart to kernel_stream_cuda.cu.  The two files share identical
// kernel logic and timing strategy; only the runtime API prefix differs
// (hip* vs cuda*).  This is intentional: any performance gap between the two
// native baselines isolates hardware differences from abstraction overhead.
//
// Implements all five BabelStream operations:
//   Copy  : c[i] = a[i]
//   Mul   : b[i] = scalar * c[i]
//   Add   : c[i] = a[i] + b[i]
//   Triad : a[i] = b[i] + scalar * c[i]   ← PRIMARY E1 metric
//   Dot   : sum += a[i] * b[i]
//
// Timing: hipEvent_t (device-side) — excludes host-launch latency.
// This is the most precise method on AMD hardware and is directly comparable
// to the CUDA cudaEvent baseline.  The Kokkos/RAJA/SYCL implementations use
// host-wall-clock timing; the delta is Runtime Coordination Overhead, measured
// by overhead_attribution.py.
//
// Metadata: prints gcnArchName (e.g., "gfx90a" for MI250X) instead of the
// CUDA cc=X.Y notation.  Requires ROCm ≥ 4.5 for hipDeviceProp_t.gcnArchName.
//
// Build: see kernels/stream/CMakeLists.txt
// Run:   ./stream-hip --arraysize 268435456 --numtimes 30

#include "stream_common.h"

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>

// ── HIP error-checking macro ──────────────────────────────────────────────────
#define HIP_CHECK(call)                                                         \
    do {                                                                        \
        hipError_t _err = (call);                                               \
        if (_err != hipSuccess) {                                               \
            std::fprintf(stderr, "HIP error at %s:%d — %s\n",                  \
                         __FILE__, __LINE__, hipGetErrorString(_err));          \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// ── GPU kernels ───────────────────────────────────────────────────────────────
// Identical to the CUDA baseline — __global__, __device__, __shared__, and
// <<<grid, block>>> launch syntax are fully supported by hipcc.

// Initialize all three arrays on-device to avoid PCIe transfer overhead.
template <typename T>
__global__ void init_kernel(T* __restrict__ a,
                             T* __restrict__ b,
                             T* __restrict__ c,
                             T va, T vb, T vc,
                             size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) { a[i] = va; b[i] = vb; c[i] = vc; }
}

// Copy: c[i] = a[i]
template <typename T>
__global__ void copy_kernel(const T* __restrict__ a,
                                   T* __restrict__ c,
                                   size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i];
}

// Mul: b[i] = scalar * c[i]
template <typename T>
__global__ void mul_kernel(      T* __restrict__ b,
                            const T* __restrict__ c,
                            T scalar,
                            size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) b[i] = scalar * c[i];
}

// Add: c[i] = a[i] + b[i]
template <typename T>
__global__ void add_kernel(const T* __restrict__ a,
                            const T* __restrict__ b,
                                  T* __restrict__ c,
                                  size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// Triad: a[i] = b[i] + scalar * c[i]   ← PRIMARY E1 KERNEL
template <typename T>
__global__ void triad_kernel(      T* __restrict__ a,
                              const T* __restrict__ b,
                              const T* __restrict__ c,
                              T scalar,
                              size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) a[i] = b[i] + scalar * c[i];
}

// Dot: two-stage reduction — grid-stride accumulation into shared memory,
// then one atomicAdd per block.  Same strategy as CUDA baseline for a
// direct apples-to-apples comparison.
template <typename T>
__global__ void dot_kernel(const T* __restrict__ a,
                            const T* __restrict__ b,
                                  T* __restrict__ partial,
                                  size_t n) {
    __shared__ T sdata[1024];
    const size_t tid    = threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    T acc = static_cast<T>(0);
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
         i < n; i += stride)
        acc += a[i] * b[i];
    sdata[tid] = acc;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(partial, sdata[0]);
}

// ── Timing helper using HIP events ────────────────────────────────────────────
// hipEventElapsedTime returns milliseconds; convert to seconds for BW formula.
static inline float event_elapsed_ms(hipEvent_t start, hipEvent_t stop) {
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    return ms;
}

// ── CLI option parsing ────────────────────────────────────────────────────────
struct Options {
    size_t array_size  = 1u << 26;
    int    num_times   = STREAM_TIMED_ITERS;
    int    warmup      = STREAM_WARMUP_ITERS;
    int    block_size  = STREAM_BLOCK_SIZE;
    bool   all_kernels = false;
    bool   csv         = false;
};

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  --arraysize N    Elements per array (default: 2^26)\n"
        "  --numtimes N     Timed iterations   (default: %d)\n"
        "  --warmup N       Warm-up iterations (default: %d)\n"
        "  --blocksize N    HIP thread-block size (default: %d)\n"
        "  --all-kernels    Run Copy/Mul/Add/Triad/Dot (default: Triad only)\n"
        "  --csv            Machine-readable CSV output\n",
        prog, STREAM_TIMED_ITERS, STREAM_WARMUP_ITERS, STREAM_BLOCK_SIZE);
}

static Options parse_args(int argc, char* argv[]) {
    static const option long_opts[] = {
        {"arraysize",   required_argument, nullptr, 'n'},
        {"numtimes",    required_argument, nullptr, 't'},
        {"warmup",      required_argument, nullptr, 'w'},
        {"blocksize",   required_argument, nullptr, 'b'},
        {"all-kernels", no_argument,       nullptr, 'a'},
        {"csv",         no_argument,       nullptr, 'c'},
        {"help",        no_argument,       nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };
    Options opts;
    int c;
    while ((c = getopt_long(argc, argv, "n:t:w:b:ach", long_opts, nullptr)) != -1) {
        switch (c) {
            case 'n': opts.array_size  = static_cast<size_t>(std::atoll(optarg)); break;
            case 't': opts.num_times   = std::atoi(optarg); break;
            case 'w': opts.warmup      = std::atoi(optarg); break;
            case 'b': opts.block_size  = std::atoi(optarg); break;
            case 'a': opts.all_kernels = true; break;
            case 'c': opts.csv         = true; break;
            case 'h': print_usage(argv[0]); std::exit(EXIT_SUCCESS);
            default:  print_usage(argv[0]); std::exit(EXIT_FAILURE);
        }
    }
    return opts;
}

// ── Device selection and info ─────────────────────────────────────────────────
// gcnArchName (e.g. "gfx90a") is more informative for AMD hardware than a
// numeric major.minor capability pair.  Available in ROCm ≥ 4.5.
static void print_device_info() {
    int dev = 0;
    HIP_CHECK(hipGetDevice(&dev));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    std::printf("STREAM_META device=\"%s\" arch=%s precision=%s\n",
                prop.name, prop.gcnArchName, STREAM_PRECISION);
}

// ── Correctness verification ──────────────────────────────────────────────────
// Mirrors kernel_stream_cuda.cu check_correctness() exactly.
static bool check_correctness(StreamFloat* d_a, StreamFloat* d_b, StreamFloat* d_c,
                               size_t n, int total_passes)
{
    StreamExpected exp = compute_expected(total_passes);

    const size_t indices[3] = {0, n / 2, n - 1};
    StreamFloat ha, hb, hc;
    double max_err_a = 0.0, max_err_b = 0.0, max_err_c = 0.0;

    auto relerr = [](double got, double expected) -> double {
        double denom = std::fabs(expected) < 1e-12 ? 1.0 : std::fabs(expected);
        return std::fabs(got - expected) / denom;
    };

    for (size_t idx : indices) {
        HIP_CHECK(hipMemcpy(&ha, d_a + idx, sizeof(StreamFloat), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&hb, d_b + idx, sizeof(StreamFloat), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&hc, d_c + idx, sizeof(StreamFloat), hipMemcpyDeviceToHost));
        double ea = relerr(static_cast<double>(ha), exp.a);
        double eb = relerr(static_cast<double>(hb), exp.b);
        double ec = relerr(static_cast<double>(hc), exp.c);
        if (ea > max_err_a) max_err_a = ea;
        if (eb > max_err_b) max_err_b = eb;
        if (ec > max_err_c) max_err_c = ec;
    }

    bool pass = (max_err_a < STREAM_CORRECT_TOL) &&
                (max_err_b < STREAM_CORRECT_TOL) &&
                (max_err_c < STREAM_CORRECT_TOL);

    std::printf("STREAM_CORRECT %s max_err_a=%.3e max_err_b=%.3e max_err_c=%.3e\n",
                pass ? "PASS" : "FAIL", max_err_a, max_err_b, max_err_c);
    if (!pass) {
        if (max_err_a >= STREAM_CORRECT_TOL)
            std::printf("STREAM_CORRECT DETAIL array=a expected=%.10f\n", exp.a);
        if (max_err_b >= STREAM_CORRECT_TOL)
            std::printf("STREAM_CORRECT DETAIL array=b expected=%.10f\n", exp.b);
        if (max_err_c >= STREAM_CORRECT_TOL)
            std::printf("STREAM_CORRECT DETAIL array=c expected=%.10f\n", exp.c);
    }
    std::fflush(stdout);
    return pass;
}

// ── One timed pass of every requested kernel ──────────────────────────────────
struct PassResult {
    double copy_bw, mul_bw, add_bw, triad_bw, dot_bw;
};

static PassResult run_timed_pass(
    StreamFloat* d_a, StreamFloat* d_b, StreamFloat* d_c, StreamFloat* d_dot,
    size_t n, int block_size, bool all_kernels,
    hipEvent_t ev_start, hipEvent_t ev_stop)
{
    const dim3 block(static_cast<unsigned>(block_size));
    const dim3 grid(static_cast<unsigned>((n + block_size - 1) / block_size));
    const StreamFloat scalar = STREAM_SCALAR;
    PassResult r{};

    if (all_kernels) {
        // Copy
        HIP_CHECK(hipEventRecord(ev_start));
        copy_kernel<<<grid, block>>>(d_a, d_c, n);
        HIP_CHECK(hipEventRecord(ev_stop));
        HIP_CHECK(hipEventSynchronize(ev_stop));
        r.copy_bw = copy_bandwidth_gbs(n, event_elapsed_ms(ev_start, ev_stop) * 1e-3);

        // Mul
        HIP_CHECK(hipEventRecord(ev_start));
        mul_kernel<<<grid, block>>>(d_b, d_c, scalar, n);
        HIP_CHECK(hipEventRecord(ev_stop));
        HIP_CHECK(hipEventSynchronize(ev_stop));
        r.mul_bw = mul_bandwidth_gbs(n, event_elapsed_ms(ev_start, ev_stop) * 1e-3);

        // Add
        HIP_CHECK(hipEventRecord(ev_start));
        add_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
        HIP_CHECK(hipEventRecord(ev_stop));
        HIP_CHECK(hipEventSynchronize(ev_stop));
        r.add_bw = add_bandwidth_gbs(n, event_elapsed_ms(ev_start, ev_stop) * 1e-3);
    }

    // Triad — always timed (primary E1 metric)
    HIP_CHECK(hipEventRecord(ev_start));
    triad_kernel<<<grid, block>>>(d_a, d_b, d_c, scalar, n);
    HIP_CHECK(hipEventRecord(ev_stop));
    HIP_CHECK(hipEventSynchronize(ev_stop));
    r.triad_bw = triad_bandwidth_gbs(n, event_elapsed_ms(ev_start, ev_stop) * 1e-3);

    if (all_kernels) {
        // Dot — reset partial sum, then accumulate
        HIP_CHECK(hipMemset(d_dot, 0, sizeof(StreamFloat)));
        HIP_CHECK(hipEventRecord(ev_start));
        dot_kernel<<<grid, block>>>(d_a, d_b, d_dot, n);
        HIP_CHECK(hipEventRecord(ev_stop));
        HIP_CHECK(hipEventSynchronize(ev_stop));
        r.dot_bw = dot_bandwidth_gbs(n, event_elapsed_ms(ev_start, ev_stop) * 1e-3);
    }

    return r;
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    const Options opts = parse_args(argc, argv);
    const size_t n         = opts.array_size;
    const int num_times    = opts.num_times;
    const int warmup       = opts.warmup;
    const int block_size   = opts.block_size;

    // ── Device info ───────────────────────────────────────────────────────────
    print_device_info();
    std::printf("STREAM_META abstraction=hip n=%zu sizeof=%zu "
                "warmup=%d timed=%d block_size=%d all_kernels=%d\n",
                n, sizeof(StreamFloat), warmup, num_times, block_size,
                static_cast<int>(opts.all_kernels));
    const double mb_per = static_cast<double>(n) * sizeof(StreamFloat) / (1024.0 * 1024.0);
    std::printf("STREAM_META array_mb=%.1f total_mb=%.1f\n", mb_per, 3.0 * mb_per);
    std::fflush(stdout);

    // ── Allocate device memory ────────────────────────────────────────────────
    StreamFloat *d_a, *d_b, *d_c, *d_dot;
    HIP_CHECK(hipMalloc(&d_a,  n * sizeof(StreamFloat)));
    HIP_CHECK(hipMalloc(&d_b,  n * sizeof(StreamFloat)));
    HIP_CHECK(hipMalloc(&d_c,  n * sizeof(StreamFloat)));
    HIP_CHECK(hipMalloc(&d_dot, sizeof(StreamFloat)));

    // ── Create timing events ──────────────────────────────────────────────────
    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    const dim3 block(static_cast<unsigned>(block_size));
    const dim3 grid(static_cast<unsigned>((n + block_size - 1) / block_size));

    // ── Initialize arrays on device ───────────────────────────────────────────
    init_kernel<<<grid, block>>>(d_a, d_b, d_c,
                                  STREAM_INIT_A, STREAM_INIT_B, STREAM_INIT_C, n);
    HIP_CHECK(hipDeviceSynchronize());

    // ── Warm-up (discarded) ───────────────────────────────────────────────────
    for (int i = 0; i < warmup; ++i)
        run_timed_pass(d_a, d_b, d_c, d_dot, n, block_size,
                       opts.all_kernels, ev_start, ev_stop);
    HIP_CHECK(hipDeviceSynchronize());

    // ── Correctness check ─────────────────────────────────────────────────────
    bool correctness_ok;
    if (opts.all_kernels) {
        correctness_ok = check_correctness(d_a, d_b, d_c, n, warmup);
    } else {
        // Triad-only: re-init then one clean triad for a deterministic check
        init_kernel<<<grid, block>>>(d_a, d_b, d_c,
                                      STREAM_INIT_A, STREAM_INIT_B, STREAM_INIT_C, n);
        HIP_CHECK(hipDeviceSynchronize());
        triad_kernel<<<grid, block>>>(d_a, d_b, d_c, STREAM_SCALAR, n);
        HIP_CHECK(hipDeviceSynchronize());

        StreamFloat ha;
        HIP_CHECK(hipMemcpy(&ha, d_a, sizeof(StreamFloat), hipMemcpyDeviceToHost));
        const StreamFloat expected_a = STREAM_INIT_B + STREAM_SCALAR * STREAM_INIT_C;
        double err = std::fabs(static_cast<double>(ha) - static_cast<double>(expected_a))
                   / std::fabs(static_cast<double>(expected_a));
        correctness_ok = (err < STREAM_CORRECT_TOL);
        std::printf("STREAM_CORRECT %s max_err_a=%.3e max_err_b=%.3e max_err_c=%.3e\n",
                    correctness_ok ? "PASS" : "FAIL", err, 0.0, 0.0);
        std::fflush(stdout);
    }

    if (!correctness_ok) {
        std::fprintf(stderr, "CORRECTNESS CHECK FAILED — aborting timing run.\n");
        std::exit(EXIT_FAILURE);
    }

    // Re-initialize for the actual timing runs (clean state after correctness check)
    init_kernel<<<grid, block>>>(d_a, d_b, d_c,
                                  STREAM_INIT_A, STREAM_INIT_B, STREAM_INIT_C, n);
    HIP_CHECK(hipDeviceSynchronize());

    // ── Timed runs ────────────────────────────────────────────────────────────
    std::vector<double> triad_bw(num_times);
    std::vector<double> copy_bw(num_times), mul_bw(num_times),
                        add_bw(num_times),   dot_bw(num_times);

    for (int i = 0; i < num_times; ++i) {
        PassResult r = run_timed_pass(d_a, d_b, d_c, d_dot, n, block_size,
                                       opts.all_kernels, ev_start, ev_stop);
        triad_bw[i] = r.triad_bw;
        if (opts.all_kernels) {
            copy_bw[i] = r.copy_bw;
            mul_bw[i]  = r.mul_bw;
            add_bw[i]  = r.add_bw;
            dot_bw[i]  = r.dot_bw;
        }
        double time_ms = (3.0 * static_cast<double>(n) * sizeof(StreamFloat))
                         / (r.triad_bw * 1.0e9) * 1.0e3;
        print_run_line("triad", i + 1, n, time_ms, r.triad_bw);
    }
    HIP_CHECK(hipDeviceSynchronize());

    // ── Statistics ────────────────────────────────────────────────────────────
    print_summary("triad", compute_stats(triad_bw));
    if (opts.all_kernels) {
        print_summary("copy", compute_stats(copy_bw));
        print_summary("mul",  compute_stats(mul_bw));
        print_summary("add",  compute_stats(add_bw));
        print_summary("dot",  compute_stats(dot_bw));
    }

    // ── Clean up ──────────────────────────────────────────────────────────────
    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));
    HIP_CHECK(hipFree(d_dot));

    return EXIT_SUCCESS;
}
