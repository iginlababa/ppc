#!/usr/bin/env julia
# kernel_stream_julia.jl — Julia/CUDA.jl abstraction for E1 STREAM Triad.
#
# Implements all five BabelStream operations using explicit GPU kernels
# launched via the @cuda macro (CUDA.jl):
#   Copy  : c[i] = a[i]
#   Mul   : b[i] = scalar * c[i]
#   Add   : c[i] = a[i] + b[i]
#   Triad : a[i] = b[i] + scalar * c[i]   ← PRIMARY E1 metric
#   Dot   : sum += a[i] * b[i]
#
# Explicit kernels (not Julia broadcast/map) are used so that the measured
# overhead vs. the CUDA native baseline isolates Julia dispatch and JIT costs
# rather than algorithmic differences.  The Dot kernel uses the same two-stage
# shared-memory reduction as kernel_stream_cuda.cu.
#
# Timing: CUDA.jl events (CUDA.CuEvent) — device-side, identical in precision
# to the C CUDA baseline.  First-call JIT compilation is absorbed by the
# warmup phase, so timed iterations see fully compiled kernels.
#
# Memory: CuArray{T} — Julia's GPU array type, backed by cudaMalloc.
#
# Backend note: targets NVIDIA via CUDA.jl.  For AMD portability, swap
# CUDA.jl for AMDGPU.jl (ROCArray, @roc) or use KernelAbstractions.jl
# as the portable layer — see kernels/stream/README.md.
#
# Dependencies: CUDA.jl ≥ 5.0 (Julia ≥ 1.9)
# Setup:   julia --project=kernels/stream/ -e 'using Pkg; Pkg.instantiate()'
# Run:     julia --project=kernels/stream/ kernel_stream_julia.jl \
#               --arraysize 268435456 --numtimes 30

# ── Package environment ───────────────────────────────────────────────────────
# Allow running both as a standalone script and via the wrapper.
using CUDA
using Printf

# ── Constants matching stream_common.h ───────────────────────────────────────
# Values MUST be identical across all abstraction implementations.
const STREAM_FLOAT       = get(ENV, "STREAM_USE_FLOAT", "") == "1" ? Float32 : Float64
const STREAM_INIT_A      = STREAM_FLOAT(0.1)
const STREAM_INIT_B      = STREAM_FLOAT(0.2)
const STREAM_INIT_C      = STREAM_FLOAT(0.0)
const STREAM_SCALAR      = STREAM_FLOAT(0.4)
const STREAM_CORRECT_TOL = 1.0e-6
const STREAM_WARMUP      = 10
const STREAM_TIMED       = 30
const DOT_BLOCK_SIZE     = 1024   # must match shared-memory array size in dot_kernel!

# ── GPU kernels ───────────────────────────────────────────────────────────────
# All kernels are generic over T so that switching STREAM_FLOAT to Float32
# recompiles without source changes.

function init_kernel!(a, b, c, va, vb, vc)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(a)
        @inbounds a[i] = va
        @inbounds b[i] = vb
        @inbounds c[i] = vc
    end
    return
end

function copy_kernel!(a, c)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(a)
        @inbounds c[i] = a[i]
    end
    return
end

function mul_kernel!(b, c, scalar)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(b)
        @inbounds b[i] = scalar * c[i]
    end
    return
end

function add_kernel!(a, b, c)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(a)
        @inbounds c[i] = a[i] + b[i]
    end
    return
end

function triad_kernel!(a, b, c, scalar)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(a)
        @inbounds a[i] = b[i] + scalar * c[i]
    end
    return
end

# Dot: two-stage reduction mirroring kernel_stream_cuda.cu.
# Stage 1: each thread accumulates into a register (grid-stride loop).
# Stage 2: block-local tree reduction in static shared memory.
# One atomicAdd per block writes into the single-element output scalar.
# DOT_BLOCK_SIZE must match the @cuda launch parameter.
function dot_kernel!(partial, a, b)
    T = eltype(a)
    shared = CUDA.CuStaticSharedArray(T, DOT_BLOCK_SIZE)
    tid    = Int(threadIdx().x)
    stride = Int(blockDim().x) * Int(gridDim().x)

    acc = zero(T)
    i = (Int(blockIdx().x) - 1) * Int(blockDim().x) + tid
    while i <= length(a)
        @inbounds acc += a[i] * b[i]
        i += stride
    end
    @inbounds shared[tid] = acc
    sync_threads()

    s = Int(blockDim().x) >> 1
    while s > 0
        if tid <= s
            @inbounds shared[tid] += shared[tid + s]
        end
        sync_threads()
        s >>= 1
    end

    if tid == 1
        CUDA.atomic_add!(pointer(partial, 1), shared[1])
    end
    return
end

# ── Timing helper ─────────────────────────────────────────────────────────────
# CUDA.jl ≥ 5.0: CUDA.elapsed returns seconds (Float32).
# Note: CUDA.jl < 5 returned milliseconds — the * 1e-3 factor is intentionally
# absent here.
@inline function event_elapsed_s(start::CUDA.CuEvent, stop::CUDA.CuEvent)
    return Float64(CUDA.elapsed(start, stop))
end

# ── Bandwidth formulae (mirror stream_common.h) ───────────────────────────────
triad_bw_gbs(n, t_s) = 3.0 * n * sizeof(STREAM_FLOAT) / t_s / 1e9
copy_bw_gbs(n, t_s)  = 2.0 * n * sizeof(STREAM_FLOAT) / t_s / 1e9
mul_bw_gbs(n, t_s)   = 2.0 * n * sizeof(STREAM_FLOAT) / t_s / 1e9
add_bw_gbs(n, t_s)   = 3.0 * n * sizeof(STREAM_FLOAT) / t_s / 1e9
dot_bw_gbs(n, t_s)   = 2.0 * n * sizeof(STREAM_FLOAT) / t_s / 1e9

# ── Analytical expected values after n_passes ─────────────────────────────────
# One pass = Copy → Mul → Add → Triad.  Mirrors compute_expected() in
# stream_common.h so correctness checks are cross-implementation comparable.
function compute_expected(n_passes::Int)
    a = Float64(STREAM_INIT_A)
    b = Float64(STREAM_INIT_B)
    c = Float64(STREAM_INIT_C)
    s = Float64(STREAM_SCALAR)
    for _ in 1:n_passes
        c = a           # Copy
        b = s * c       # Mul
        c = a + b       # Add
        a = b + s * c   # Triad
    end
    return (a=a, b=b, c=c)
end

# ── Statistics (mirror compute_stats in stream_common.h) ──────────────────────
function compute_stats(vals::Vector{Float64})
    n      = length(vals)
    sorted = sort(vals)
    med    = n % 2 == 0 ? (sorted[n÷2] + sorted[n÷2+1]) / 2.0 : sorted[(n+1)÷2]
    q1     = sorted[max(1, n÷4 + 1)]
    q3     = sorted[min(n, 3*(n÷4) + 1)]
    iqr    = q3 - q1
    mn     = sum(vals) / n
    var    = sum((v - mn)^2 for v in vals) / n
    sigma  = sqrt(var)
    nout   = count(v -> abs(v - mn) > 2*sigma, vals)
    return (median=med, iqr=iqr, mean=mn, min=minimum(vals), max=maximum(vals), n_outliers=nout)
end

# ── Output helpers (match STREAM_RUN / STREAM_SUMMARY format) ─────────────────
function print_run_line(kernel::String, run_id::Int, n::Int, time_ms::Float64, bw_gbs::Float64)
    @printf("STREAM_RUN kernel=%s run=%d n=%d time_ms=%.5f bw_gbs=%.4f\n",
            kernel, run_id, n, time_ms, bw_gbs)
    flush(stdout)
end

function print_summary(kernel::String, s)
    @printf("STREAM_SUMMARY kernel=%s median_bw_gbs=%.4f iqr_bw_gbs=%.4f min_bw_gbs=%.4f max_bw_gbs=%.4f mean_bw_gbs=%.4f outliers=%d\n",
            kernel, s.median, s.iqr, s.min, s.max, s.mean, s.n_outliers)
    flush(stdout)
end

# ── CLI argument parsing ───────────────────────────────────────────────────────
function parse_cli()
    arraysize   = 1 << 26
    numtimes    = STREAM_TIMED
    warmup      = STREAM_WARMUP
    all_kernels = false
    blocksize   = 256

    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg in ("--arraysize", "-n") && i < length(ARGS)
            arraysize = parse(Int, ARGS[i+1]); i += 2
        elseif arg in ("--numtimes", "-t") && i < length(ARGS)
            numtimes = parse(Int, ARGS[i+1]); i += 2
        elseif arg in ("--warmup", "-w") && i < length(ARGS)
            warmup = parse(Int, ARGS[i+1]); i += 2
        elseif arg in ("--blocksize", "-b") && i < length(ARGS)
            blocksize = parse(Int, ARGS[i+1]); i += 2
        elseif arg == "--all-kernels"
            all_kernels = true; i += 1
        elseif arg in ("--help", "-h")
            println("Usage: julia kernel_stream_julia.jl [options]")
            println("  --arraysize N   Elements per array (default: 2^26)")
            println("  --numtimes  N   Timed iterations   (default: $STREAM_TIMED)")
            println("  --warmup    N   Warm-up iterations (default: $STREAM_WARMUP)")
            println("  --blocksize N   GPU thread-block size (default: 256)")
            println("  --all-kernels   Run Copy/Mul/Add/Triad/Dot")
            exit(0)
        else
            @warn "Unknown argument: $arg"
            i += 1
        end
    end
    return (; arraysize, numtimes, warmup, all_kernels, blocksize)
end

# ── On-device initialization ───────────────────────────────────────────────────
function do_init!(a::CuArray, b::CuArray, c::CuArray, block_size::Int)
    n      = length(a)
    nblocks = cld(n, block_size)
    @cuda threads=block_size blocks=nblocks init_kernel!(a, b, c,
        STREAM_INIT_A, STREAM_INIT_B, STREAM_INIT_C)
    CUDA.synchronize()
end

# ── One timed pass ─────────────────────────────────────────────────────────────
struct PassBW
    copy::Float64; mul::Float64; add::Float64; triad::Float64; dot::Float64
end

function run_pass(a::CuArray{T}, b::CuArray{T}, c::CuArray{T},
                  partial::CuArray{T}, n::Int, block_size::Int,
                  all_kernels::Bool,
                  ev_start::CUDA.CuEvent, ev_stop::CUDA.CuEvent) where T
    scalar  = STREAM_SCALAR
    nblocks = cld(n, block_size)

    # time_kernel: record start → launch → record stop → synchronize stop
    function time_kernel(launch_fn, bw_fn)
        CUDA.record(ev_start)
        launch_fn()
        CUDA.record(ev_stop)
        CUDA.synchronize(ev_stop)
        return bw_fn(n, event_elapsed_s(ev_start, ev_stop))
    end

    copy_bw = mul_bw = add_bw = dot_bw = 0.0

    if all_kernels
        copy_bw = time_kernel(copy_bw_gbs) do
            @cuda threads=block_size blocks=nblocks copy_kernel!(a, c)
        end
        mul_bw = time_kernel(mul_bw_gbs) do
            @cuda threads=block_size blocks=nblocks mul_kernel!(b, c, scalar)
        end
        add_bw = time_kernel(add_bw_gbs) do
            @cuda threads=block_size blocks=nblocks add_kernel!(a, b, c)
        end
    end

    # Triad — always timed (primary E1 metric)
    triad_bw_val = time_kernel(triad_bw_gbs) do
        @cuda threads=block_size blocks=nblocks triad_kernel!(a, b, c, scalar)
    end

    if all_kernels
        # Reset partial sum; CUDA.fill! launches a device kernel
        CUDA.fill!(partial, zero(T))
        CUDA.record(ev_start)
        @cuda threads=DOT_BLOCK_SIZE blocks=cld(n, DOT_BLOCK_SIZE) dot_kernel!(partial, a, b)
        CUDA.record(ev_stop)
        CUDA.synchronize(ev_stop)
        dot_bw = dot_bw_gbs(n, event_elapsed_s(ev_start, ev_stop))
    end

    return PassBW(copy_bw, mul_bw, add_bw, triad_bw_val, dot_bw)
end

# ── Correctness check ──────────────────────────────────────────────────────────
# Reads three sample elements from device via scalar indexing (CUDA.@allowscalar)
# and compares to analytical expected values from compute_expected().
function check_correctness(a::CuArray{T}, b::CuArray{T}, c::CuArray{T},
                            n::Int, n_passes::Int, all_kernels::Bool) where T
    exp = all_kernels ? compute_expected(n_passes) :
          (a = Float64(STREAM_INIT_B + STREAM_SCALAR * STREAM_INIT_C),
           b = Float64(STREAM_INIT_B),
           c = Float64(STREAM_INIT_C))

    relerr(got, expected) = begin
        denom = abs(expected) < 1e-12 ? 1.0 : abs(expected)
        abs(Float64(got) - expected) / denom
    end

    indices = [1, n÷2, n]   # Julia is 1-based; sample first, middle, last
    max_ea = max_eb = max_ec = 0.0

    CUDA.@allowscalar for idx in indices
        ea = relerr(a[idx], exp.a); max_ea = max(max_ea, ea)
        eb = relerr(b[idx], exp.b); max_eb = max(max_eb, eb)
        ec = relerr(c[idx], exp.c); max_ec = max(max_ec, ec)
    end

    pass = max_ea < STREAM_CORRECT_TOL &&
           max_eb < STREAM_CORRECT_TOL &&
           max_ec < STREAM_CORRECT_TOL

    @printf("STREAM_CORRECT %s max_err_a=%.3e max_err_b=%.3e max_err_c=%.3e\n",
            pass ? "PASS" : "FAIL", max_ea, max_eb, max_ec)
    if !pass
        max_ea >= STREAM_CORRECT_TOL && @printf("STREAM_CORRECT DETAIL array=a expected=%.10f\n", exp.a)
        max_eb >= STREAM_CORRECT_TOL && @printf("STREAM_CORRECT DETAIL array=b expected=%.10f\n", exp.b)
        max_ec >= STREAM_CORRECT_TOL && @printf("STREAM_CORRECT DETAIL array=c expected=%.10f\n", exp.c)
    end
    flush(stdout)
    return pass
end

# ── Main ───────────────────────────────────────────────────────────────────────
function main()
    opts = parse_cli()
    n           = opts.arraysize
    num_times   = opts.numtimes
    warmup      = opts.warmup
    all_kernels = opts.all_kernels
    block_size  = opts.blocksize

    # ── Metadata ──────────────────────────────────────────────────────────────
    dev      = CUDA.device()
    dev_name = CUDA.name(dev)
    cuda_ver = CUDA.runtime_version()
    prec_str = STREAM_FLOAT == Float32 ? "float" : "double"

    @printf("STREAM_META abstraction=julia backend=cuda device=\"%s\" cuda_runtime=%s julia=%s precision=%s n=%d sizeof=%d warmup=%d timed=%d all_kernels=%d\n",
            dev_name, string(cuda_ver), string(VERSION), prec_str,
            n, sizeof(STREAM_FLOAT), warmup, num_times, Int(all_kernels))
    mb_per = n * sizeof(STREAM_FLOAT) / (1024.0^2)
    @printf("STREAM_META array_mb=%.1f total_mb=%.1f\n", mb_per, 3.0 * mb_per)
    flush(stdout)

    # ── Allocate CuArrays ──────────────────────────────────────────────────────
    # CuArray{T}(undef, n) calls cudaMalloc; layout is identical to C float*/double*.
    a = CuArray{STREAM_FLOAT}(undef, n)
    b = CuArray{STREAM_FLOAT}(undef, n)
    c = CuArray{STREAM_FLOAT}(undef, n)
    partial = CuArray{STREAM_FLOAT}(undef, 1)   # dot reduction accumulator

    # ── Timing events ──────────────────────────────────────────────────────────
    ev_start = CUDA.CuEvent()
    ev_stop  = CUDA.CuEvent()

    # ── Initialize on device ───────────────────────────────────────────────────
    do_init!(a, b, c, block_size)

    # ── Warm-up (discarded) ────────────────────────────────────────────────────
    # Absorbs Julia JIT compilation (first @cuda call compiles the PTX) and
    # brings the GPU to steady thermal/clock state.
    for _ in 1:warmup
        run_pass(a, b, c, partial, n, block_size, all_kernels, ev_start, ev_stop)
    end
    CUDA.synchronize()

    # ── Correctness check ──────────────────────────────────────────────────────
    if !all_kernels
        do_init!(a, b, c, block_size)
        @cuda threads=block_size blocks=cld(n, block_size) triad_kernel!(a, b, c, STREAM_SCALAR)
        CUDA.synchronize()
    end
    passes_so_far = all_kernels ? warmup : 0
    if !check_correctness(a, b, c, n, passes_so_far, all_kernels)
        @error "CORRECTNESS CHECK FAILED — aborting."
        exit(1)
    end

    # ── Re-initialize for timed runs ───────────────────────────────────────────
    do_init!(a, b, c, block_size)

    # ── Timed runs ─────────────────────────────────────────────────────────────
    triad_bw_vec = Vector{Float64}(undef, num_times)
    copy_bw_vec  = Vector{Float64}(undef, num_times)
    mul_bw_vec   = Vector{Float64}(undef, num_times)
    add_bw_vec   = Vector{Float64}(undef, num_times)
    dot_bw_vec   = Vector{Float64}(undef, num_times)

    for i in 1:num_times
        r = run_pass(a, b, c, partial, n, block_size, all_kernels, ev_start, ev_stop)
        triad_bw_vec[i] = r.triad
        if all_kernels
            copy_bw_vec[i] = r.copy
            mul_bw_vec[i]  = r.mul
            add_bw_vec[i]  = r.add
            dot_bw_vec[i]  = r.dot
        end
        time_ms = 3.0 * n * sizeof(STREAM_FLOAT) / (r.triad * 1e9) * 1e3
        print_run_line("triad", i, n, time_ms, r.triad)
    end
    CUDA.synchronize()

    # ── Statistics ─────────────────────────────────────────────────────────────
    print_summary("triad", compute_stats(triad_bw_vec))
    if all_kernels
        print_summary("copy", compute_stats(copy_bw_vec))
        print_summary("mul",  compute_stats(mul_bw_vec))
        print_summary("add",  compute_stats(add_bw_vec))
        print_summary("dot",  compute_stats(dot_bw_vec))
    end
end

main()
