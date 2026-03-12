// kernel_stream_cuda.cu — CUDA native baseline for E1 STREAM Triad.
//
// Implements all five BabelStream operations:
//   Copy  : c[i] = a[i]
//   Mul   : b[i] = scalar * c[i]
//   Add   : c[i] = a[i] + b[i]
//   Triad : a[i] = b[i] + scalar * c[i]   ← PRIMARY E1 metric
//   Dot   : sum += a[i] * b[i]
//
// Timing is device-side (cudaEvent) — not host-wall-clock — which avoids
// launch-latency inflation for large-N runs and keeps results comparable
// to GPU-side timers in Kokkos/RAJA/SYCL implementations.
//
// Output format is parsed by scripts/parse/parse_results.py.
// Correctness is verified analytically (no host reference array).
//
// Build: see kernels/stream/CMakeLists.txt
// Run:   ./stream-cuda --arraysize 268435456 --numtimes 30

#include "stream_common.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>

// ── CUDA error-checking macro ─────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error at %s:%d — %s\n",                 \
                         __FILE__, __LINE__, cudaGetErrorString(_err));         \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// ── GPU kernels ───────────────────────────────────────────────────────────────

// Initialize all three arrays to their canonical starting values.
// Running init on-device avoids the PCIe transfer cost that would be required
// if we filled host arrays and then cudaMemcpy'd them.
template <typename T>
__global__ void init_kernel(T* __restrict__ a,
                             T* __restrict__ b,
                             T* __restrict__ c,
                             T va, T vb, T vc,
                             size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] = va;
        b[i] = vb;
        c[i] = vc;
    }
}

// Copy: c[i] = a[i]
template <typename T>
__global__ void copy_kernel(const T* __restrict__ a,
                                   T* __restrict__ c,
                                   size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i];
}

// Mul: b[i] = scalar * c[i]
template <typename T>
__global__ void mul_kernel(T* __restrict__ b,
                            const T* __restrict__ c,
                            T scalar,
                            size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        b[i] = scalar * c[i];
}

// Add: c[i] = a[i] + b[i]
template <typename T>
__global__ void add_kernel(const T* __restrict__ a,
                            const T* __restrict__ b,
                                  T* __restrict__ c,
                                  size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

// Triad: a[i] = b[i] + scalar * c[i]   ← PRIMARY E1 KERNEL
template <typename T>
__global__ void triad_kernel(      T* __restrict__ a,
                              const T* __restrict__ b,
                              const T* __restrict__ c,
                              T scalar,
                              size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        a[i] = b[i] + scalar * c[i];
}

// Dot: partial sums per block, then one atomicAdd per block.
// Two-stage: threads accumulate into registers (grid-stride) then reduce in
// shared memory.  One atomicAdd per block keeps global memory traffic low.
template <typename T>
__global__ void dot_kernel(const T* __restrict__ a,
                            const T* __restrict__ b,
                                  T* __restrict__ partial,
                                  size_t n) {
    __shared__ T sdata[1024];
    const size_t tid   = threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    // Grid-stride accumulation into register
    T acc = static_cast<T>(0);
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
         i < n; i += stride)
        acc += a[i] * b[i];
    sdata[tid] = acc;
    __syncthreads();

    // Tree reduction within block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(partial, sdata[0]);
}

// ── Timing helper using CUDA events ──────────────────────────────────────────
// Returns elapsed time in milliseconds.  Both events must be created by caller.
static inline float event_elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

// ── CLI option parsing ────────────────────────────────────────────────────────
struct Options {
    size_t array_size  = 1u << 26;   // medium (67M elements) default
    int    num_times   = STREAM_TIMED_ITERS;
    int    warmup      = STREAM_WARMUP_ITERS;
    int    block_size  = STREAM_BLOCK_SIZE;
    bool   all_kernels = false;       // if false, run only Triad (E1 primary)
    bool   csv         = false;
};

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  --arraysize N    Number of elements per array (default: 2^26 = 67108864)\n"
        "  --numtimes N     Number of timed iterations   (default: %d)\n"
        "  --warmup N       Warm-up iterations (discarded) (default: %d)\n"
        "  --blocksize N    CUDA thread block size        (default: %d)\n"
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
static void print_device_info() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("STREAM_META device=\"%s\" cc=%d.%d precision=%s\n",
                prop.name, prop.major, prop.minor, STREAM_PRECISION);
}

// ── Correctness verification ──────────────────────────────────────────────────
// Copies a sample of device data to host and checks against expected values.
// Using a sample (first/last/middle elements) avoids allocating a full host copy.
static bool check_correctness(StreamFloat* d_a, StreamFloat* d_b, StreamFloat* d_c,
                               size_t n, int total_passes) {
    // Compute expected values analytically
    StreamExpected exp = compute_expected(total_passes);

    // Sample: first element, middle element, last element
    const size_t NSAMPLE = 3;
    size_t indices[NSAMPLE] = {0, n / 2, n - 1};
    StreamFloat ha, hb, hc;
    bool pass = true;
    double max_err_a = 0.0, max_err_b = 0.0, max_err_c = 0.0;

    for (size_t k = 0; k < NSAMPLE; ++k) {
        size_t idx = indices[k];
        CUDA_CHECK(cudaMemcpy(&ha, d_a + idx, sizeof(StreamFloat), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&hb, d_b + idx, sizeof(StreamFloat), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&hc, d_c + idx, sizeof(StreamFloat), cudaMemcpyDeviceToHost));

        auto relerr = [](double got, double expected) -> double {
            double denom = std::fabs(expected) < 1e-12 ? 1.0 : std::fabs(expected);
            return std::fabs(got - expected) / denom;
        };

        double ea = relerr(static_cast<double>(ha), exp.a);
        double eb = relerr(static_cast<double>(hb), exp.b);
        double ec = relerr(static_cast<double>(hc), exp.c);
        if (ea > max_err_a) max_err_a = ea;
        if (eb > max_err_b) max_err_b = eb;
        if (ec > max_err_c) max_err_c = ec;
    }

    if (max_err_a > STREAM_CORRECT_TOL) {
        std::printf("STREAM_CORRECT FAIL array=a max_err=%.3e tol=%.3e expected=%.6f\n",
                    max_err_a, STREAM_CORRECT_TOL, exp.a);
        pass = false;
    }
    if (max_err_b > STREAM_CORRECT_TOL) {
        std::printf("STREAM_CORRECT FAIL array=b max_err=%.3e tol=%.3e expected=%.6f\n",
                    max_err_b, STREAM_CORRECT_TOL, exp.b);
        pass = false;
    }
    if (max_err_c > STREAM_CORRECT_TOL) {
        std::printf("STREAM_CORRECT FAIL array=c max_err=%.3e tol=%.3e expected=%.6f\n",
                    max_err_c, STREAM_CORRECT_TOL, exp.c);
        pass = false;
    }
    if (pass) {
        std::printf("STREAM_CORRECT PASS max_err_a=%.3e max_err_b=%.3e max_err_c=%.3e\n",
                    max_err_a, max_err_b, max_err_c);
    }
    std::fflush(stdout);
    return pass;
}

// ── Kernel launcher: runs one timed pass of every requested kernel ────────────
// Returns bandwidth in GB/s for each kernel (triad is index 3).
struct PassResult {
    double copy_bw, mul_bw, add_bw, triad_bw, dot_bw;
};

static PassResult run_timed_pass(
    StreamFloat* d_a, StreamFloat* d_b, StreamFloat* d_c, StreamFloat* d_dot,
    size_t n, int block_size, bool all_kernels,
    cudaEvent_t ev_start, cudaEvent_t ev_stop)
{
    const dim3 block(static_cast<unsigned>(block_size));
    const dim3 grid(static_cast<unsigned>((n + block_size - 1) / block_size));
    const StreamFloat scalar = STREAM_SCALAR;
    PassResult r{};

    if (all_kernels) {
        // Copy
        CUDA_CHECK(cudaEventRecord(ev_start));
        copy_kernel<<<grid, block>>>(d_a, d_c, n);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        r.copy_bw = copy_bandwidth_gbs(n, event_elapsed_ms(ev_start, ev_stop) * 1e-3);

        // Mul
        CUDA_CHECK(cudaEventRecord(ev_start));
        mul_kernel<<<grid, block>>>(d_b, d_c, scalar, n);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        r.mul_bw = mul_bandwidth_gbs(n, event_elapsed_ms(ev_start, ev_stop) * 1e-3);

        // Add
        CUDA_CHECK(cudaEventRecord(ev_start));
        add_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        r.add_bw = add_bandwidth_gbs(n, event_elapsed_ms(ev_start, ev_stop) * 1e-3);
    }

    // Triad (always measured — primary E1 metric)
    CUDA_CHECK(cudaEventRecord(ev_start));
    triad_kernel<<<grid, block>>>(d_a, d_b, d_c, scalar, n);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    r.triad_bw = triad_bandwidth_gbs(n, event_elapsed_ms(ev_start, ev_stop) * 1e-3);

    if (all_kernels) {
        // Dot — reset partial sum first
        CUDA_CHECK(cudaMemset(d_dot, 0, sizeof(StreamFloat)));
        CUDA_CHECK(cudaEventRecord(ev_start));
        dot_kernel<<<grid, block>>>(d_a, d_b, d_dot, n);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        r.dot_bw = dot_bandwidth_gbs(n, event_elapsed_ms(ev_start, ev_stop) * 1e-3);
    }

    return r;
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    const Options opts = parse_args(argc, argv);

    const size_t n          = opts.array_size;
    const int    num_times  = opts.num_times;
    const int    warmup     = opts.warmup;
    const int    block_size = opts.block_size;

    // ── Device info ──────────────────────────────────────────────────────────
    print_device_info();
    std::printf("STREAM_META abstraction=cuda n=%zu sizeof=%zu "
                "warmup=%d timed=%d block_size=%d all_kernels=%d\n",
                n, sizeof(StreamFloat), warmup, num_times, block_size,
                static_cast<int>(opts.all_kernels));
    const double mb_per_array = static_cast<double>(n) * sizeof(StreamFloat) / (1024.0 * 1024.0);
    std::printf("STREAM_META array_mb=%.1f total_mb=%.1f\n",
                mb_per_array, 3.0 * mb_per_array);
    std::fflush(stdout);

    // ── Allocate device memory ────────────────────────────────────────────────
    StreamFloat *d_a, *d_b, *d_c, *d_dot;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(StreamFloat)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(StreamFloat)));
    CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(StreamFloat)));
    CUDA_CHECK(cudaMalloc(&d_dot, sizeof(StreamFloat)));

    // ── Create timing events ──────────────────────────────────────────────────
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    const dim3 block(static_cast<unsigned>(block_size));
    const dim3 grid(static_cast<unsigned>((n + block_size - 1) / block_size));

    // ── Initialize arrays on device ───────────────────────────────────────────
    init_kernel<<<grid, block>>>(d_a, d_b, d_c,
                                  STREAM_INIT_A, STREAM_INIT_B, STREAM_INIT_C, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Warm-up (results discarded, not timed) ────────────────────────────────
    // Purpose: JIT any remaining runtime work, fill L2/TLB, exclude init costs.
    for (int i = 0; i < warmup; ++i) {
        run_timed_pass(d_a, d_b, d_c, d_dot, n, block_size,
                       opts.all_kernels, ev_start, ev_stop);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Correctness check (after warm-up, before timed runs) ──────────────────
    // total_passes: warm-up passes already ran.
    // all_kernels mode: each pass has 4 ops (Copy/Mul/Add/Triad).
    // triad-only mode: each pass has 1 op (Triad), but Triad is idempotent on
    //   b and c so expected value is fixed: a = INIT_B + SCALAR * INIT_C.
    const int passes_so_far = opts.all_kernels ? warmup : 0;
    // For triad-only the formula is: a = INIT_B + SCALAR * INIT_C each time.
    // compute_expected(0) returns initial values; for triad-only inject manually.
    bool correctness_ok;
    if (opts.all_kernels) {
        correctness_ok = check_correctness(d_a, d_b, d_c, n, passes_so_far);
    } else {
        // Triad-only: re-init and run one pass for a clean check
        init_kernel<<<grid, block>>>(d_a, d_b, d_c,
                                      STREAM_INIT_A, STREAM_INIT_B, STREAM_INIT_C, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        triad_kernel<<<grid, block>>>(d_a, d_b, d_c, STREAM_SCALAR, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        // Expected: a[i] = INIT_B + SCALAR * INIT_C
        StreamFloat ha;
        CUDA_CHECK(cudaMemcpy(&ha, d_a, sizeof(StreamFloat), cudaMemcpyDeviceToHost));
        const StreamFloat expected_a = STREAM_INIT_B + STREAM_SCALAR * STREAM_INIT_C;
        double err = std::fabs(static_cast<double>(ha) - static_cast<double>(expected_a))
                   / std::fabs(static_cast<double>(expected_a));
        correctness_ok = (err < STREAM_CORRECT_TOL);
        std::printf("STREAM_CORRECT %s max_err_a=%.3e tol=%.3e expected_a=%.6f\n",
                    correctness_ok ? "PASS" : "FAIL", err, STREAM_CORRECT_TOL,
                    static_cast<double>(expected_a));
        std::fflush(stdout);
        // Re-init for the actual timing runs
        init_kernel<<<grid, block>>>(d_a, d_b, d_c,
                                      STREAM_INIT_A, STREAM_INIT_B, STREAM_INIT_C, n);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (!correctness_ok) {
        std::fprintf(stderr, "CORRECTNESS CHECK FAILED — aborting timing run.\n");
        std::exit(EXIT_FAILURE);
    }

    // ── Timed runs ────────────────────────────────────────────────────────────
    // 30 iterations, each individually logged (project_spec.md §9.1).
    std::vector<double> triad_bw(num_times);
    std::vector<double> copy_bw(num_times), mul_bw(num_times),
                        add_bw(num_times), dot_bw(num_times);

    for (int i = 0; i < num_times; ++i) {
        PassResult r = run_timed_pass(d_a, d_b, d_c, d_dot, n, block_size,
                                       opts.all_kernels, ev_start, ev_stop);
        triad_bw[i] = r.triad_bw;
        // Compute time_ms from BW for the output line:
        // time_ms = (3 * n * sizeof(T)) / (bw_gbs * 1e9) * 1e3
        double time_ms = (3.0 * static_cast<double>(n) * sizeof(StreamFloat))
                         / (r.triad_bw * 1.0e9) * 1.0e3;
        print_run_line("triad", i + 1, n, time_ms, r.triad_bw);

        if (opts.all_kernels) {
            copy_bw[i] = r.copy_bw;
            mul_bw[i]  = r.mul_bw;
            add_bw[i]  = r.add_bw;
            dot_bw[i]  = r.dot_bw;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Statistics and summary ────────────────────────────────────────────────
    StreamStats triad_stats = compute_stats(triad_bw);
    print_summary("triad", triad_stats);

    if (opts.all_kernels) {
        print_summary("copy",  compute_stats(copy_bw));
        print_summary("mul",   compute_stats(mul_bw));
        print_summary("add",   compute_stats(add_bw));
        print_summary("dot",   compute_stats(dot_bw));
    }

    // ── Clean up ──────────────────────────────────────────────────────────────
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_dot));

    return EXIT_SUCCESS;
}
