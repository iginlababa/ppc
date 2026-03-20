#!/usr/bin/env julia
# kernel_stream_amdgpu.jl — Julia/AMDGPU.jl abstraction for E1 STREAM Triad.
#
# AMD port of kernel_stream_julia.jl.  All five BabelStream operations are
# implemented using explicit GPU kernels launched via the @roc macro (AMDGPU.jl):
#   Copy  : c[i] = a[i]
#   Mul   : b[i] = scalar * c[i]
#   Add   : c[i] = a[i] + b[i]
#   Triad : a[i] = b[i] + scalar * c[i]   ← PRIMARY E1 metric
#   Dot   : sum += a[i] * b[i]
#
# Kernel functions are identical to kernel_stream_julia.jl — only the host-side
# API calls differ, making the abstraction overhead comparison clean.
#
# Timing: AMDGPU.@elapsed — device-side elapsed seconds, equivalent to the
# CUDA.jl event-pair approach used in the NVIDIA version.
#
# Memory: ROCArray{T} (AMDGPU.jl) backed by hipMalloc, structurally identical
# to CuArray{T} in CUDA.jl.
#
# Requires: Julia ≥ 1.12 (LLVM 18, gfx942 support) + AMDGPU.jl ≥ 2.0
# Setup:   julia -e 'using Pkg; Pkg.add("AMDGPU")'
# Run:     julia kernel_stream_amdgpu.jl --arraysize 268435456 --numtimes 30

using AMDGPU
using Printf

# ── Constants matching stream_common.h ───────────────────────────────────────
const STREAM_FLOAT       = get(ENV, "STREAM_USE_FLOAT", "") == "1" ? Float32 : Float64
const STREAM_INIT_A      = STREAM_FLOAT(0.1)
const STREAM_INIT_B      = STREAM_FLOAT(0.2)
const STREAM_INIT_C      = STREAM_FLOAT(0.0)
const STREAM_SCALAR      = STREAM_FLOAT(0.4)
const STREAM_CORRECT_TOL = 1.0e-6
const STREAM_WARMUP      = 10
const STREAM_TIMED       = 30
const DOT_BLOCK_SIZE     = 1024   # must match @ROCStaticLocalArray size in dot_kernel!

# ── GPU kernels ───────────────────────────────────────────────────────────────
# Kernel bodies are identical to kernel_stream_julia.jl — blockIdx/blockDim/
# threadIdx intrinsics are shared between CUDA.jl and AMDGPU.jl.

function init_kernel!(a, b, c, va, vb, vc)
    i = (workgroupIdx().x - Int32(1)) * workgroupDim().x + workitemIdx().x
    if i <= length(a)
        @inbounds a[i] = va
        @inbounds b[i] = vb
        @inbounds c[i] = vc
    end
    return
end

function copy_kernel!(a, c)
    i = (workgroupIdx().x - Int32(1)) * workgroupDim().x + workitemIdx().x
    if i <= length(a)
        @inbounds c[i] = a[i]
    end
    return
end

function mul_kernel!(b, c, scalar)
    i = (workgroupIdx().x - Int32(1)) * workgroupDim().x + workitemIdx().x
    if i <= length(b)
        @inbounds b[i] = scalar * c[i]
    end
    return
end

function add_kernel!(a, b, c)
    i = (workgroupIdx().x - Int32(1)) * workgroupDim().x + workitemIdx().x
    if i <= length(a)
        @inbounds c[i] = a[i] + b[i]
    end
    return
end

function triad_kernel!(a, b, c, scalar)
    i = (workgroupIdx().x - Int32(1)) * workgroupDim().x + workitemIdx().x
    if i <= length(a)
        @inbounds a[i] = b[i] + scalar * c[i]
    end
    return
end

# Dot: two-stage reduction mirroring kernel_stream_cuda.cu / kernel_stream_julia.jl.
# Stage 1: grid-stride accumulation per thread.
# Stage 2: block-local tree reduction in static LDS (local data share).
# One atomic add per block writes into the single-element output scalar.
function dot_kernel!(partial, a, b)
    T      = eltype(a)
    shared = @ROCStaticLocalArray(T, DOT_BLOCK_SIZE)
    tid    = Int(workitemIdx().x)
    stride = Int(workgroupDim().x) * Int(gridGroupDim().x)

    acc = zero(T)
    i = (Int(workgroupIdx().x) - 1) * Int(workgroupDim().x) + tid
    while i <= length(a)
        @inbounds acc += a[i] * b[i]
        i += stride
    end
    @inbounds shared[tid] = acc
    sync_workgroup()

    s = Int(workgroupDim().x) >> 1
    while s > 0
        if tid <= s
            @inbounds shared[tid] += shared[tid + s]
        end
        sync_workgroup()
        s >>= 1
    end

    if tid == 1
        AMDGPU.@atomic partial[1] += shared[1]
    end
    return
end

# ── Bandwidth formulae (mirror stream_common.h) ───────────────────────────────
triad_bw_gbs(n, t_s) = 3.0 * n * sizeof(STREAM_FLOAT) / t_s / 1e9
copy_bw_gbs(n, t_s)  = 2.0 * n * sizeof(STREAM_FLOAT) / t_s / 1e9
mul_bw_gbs(n, t_s)   = 2.0 * n * sizeof(STREAM_FLOAT) / t_s / 1e9
add_bw_gbs(n, t_s)   = 3.0 * n * sizeof(STREAM_FLOAT) / t_s / 1e9
dot_bw_gbs(n, t_s)   = 2.0 * n * sizeof(STREAM_FLOAT) / t_s / 1e9

# ── Analytical expected values after n_passes ─────────────────────────────────
function compute_expected(n_passes::Int)
    a = Float64(STREAM_INIT_A)
    b = Float64(STREAM_INIT_B)
    c = Float64(STREAM_INIT_C)
    s = Float64(STREAM_SCALAR)
    for _ in 1:n_passes
        c = a; b = s * c; c = a + b; a = b + s * c
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

# ── Output helpers ────────────────────────────────────────────────────────────
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
            println("Usage: julia kernel_stream_amdgpu.jl [options]")
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
function do_init!(a::ROCArray, b::ROCArray, c::ROCArray, block_size::Int)
    n       = length(a)
    nblocks = cld(n, block_size)
    @roc groupsize=block_size gridsize=nblocks init_kernel!(a, b, c,
        STREAM_INIT_A, STREAM_INIT_B, STREAM_INIT_C)
    AMDGPU.synchronize()
end

# ── One timed pass ─────────────────────────────────────────────────────────────
struct PassBW
    copy::Float64; mul::Float64; add::Float64; triad::Float64; dot::Float64
end

function run_pass(a::ROCArray{T}, b::ROCArray{T}, c::ROCArray{T},
                  partial::ROCArray{T}, n::Int, block_size::Int,
                  all_kernels::Bool) where T
    scalar  = STREAM_SCALAR
    nblocks = cld(n, block_size)

    # time_kernel: AMDGPU.@elapsed returns device-side elapsed seconds
    function time_kernel(launch_fn, bw_fn)
        t_s = AMDGPU.@elapsed begin
            launch_fn()
        end
        return bw_fn(n, t_s)
    end

    copy_bw = mul_bw = add_bw = dot_bw = 0.0

    if all_kernels
        copy_bw = time_kernel(copy_bw_gbs) do
            @roc groupsize=block_size gridsize=nblocks copy_kernel!(a, c)
        end
        mul_bw = time_kernel(mul_bw_gbs) do
            @roc groupsize=block_size gridsize=nblocks mul_kernel!(b, c, scalar)
        end
        add_bw = time_kernel(add_bw_gbs) do
            @roc groupsize=block_size gridsize=nblocks add_kernel!(a, b, c)
        end
    end

    triad_bw_val = time_kernel(triad_bw_gbs) do
        @roc groupsize=block_size gridsize=nblocks triad_kernel!(a, b, c, scalar)
    end

    if all_kernels
        fill!(partial, zero(T))
        dot_blocks = cld(n, DOT_BLOCK_SIZE)
        t_s = AMDGPU.@elapsed begin
            @roc groupsize=DOT_BLOCK_SIZE gridsize=dot_blocks dot_kernel!(partial, a, b)
        end
        dot_bw = dot_bw_gbs(n, t_s)
    end

    return PassBW(copy_bw, mul_bw, add_bw, triad_bw_val, dot_bw)
end

# ── Correctness check ──────────────────────────────────────────────────────────
function check_correctness(a::ROCArray{T}, b::ROCArray{T}, c::ROCArray{T},
                            n::Int, n_passes::Int, all_kernels::Bool) where T
    exp = all_kernels ? compute_expected(n_passes) :
          (a = Float64(STREAM_INIT_B + STREAM_SCALAR * STREAM_INIT_C),
           b = Float64(STREAM_INIT_B),
           c = Float64(STREAM_INIT_C))

    relerr(got, expected) = begin
        denom = abs(expected) < 1e-12 ? 1.0 : abs(expected)
        abs(Float64(got) - expected) / denom
    end

    indices = [1, n÷2, n]
    max_ea = max_eb = max_ec = 0.0

    AMDGPU.@allowscalar for idx in indices
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
    opts        = parse_cli()
    n           = opts.arraysize
    num_times   = opts.numtimes
    warmup      = opts.warmup
    all_kernels = opts.all_kernels
    block_size  = opts.blocksize

    dev      = AMDGPU.device()
    prec_str = STREAM_FLOAT == Float32 ? "float" : "double"

    @printf("STREAM_META abstraction=julia backend=amdgpu device=\"%s\" julia=%s precision=%s n=%d sizeof=%d warmup=%d timed=%d all_kernels=%d\n",
            string(dev), string(VERSION), prec_str,
            n, sizeof(STREAM_FLOAT), warmup, num_times, Int(all_kernels))
    mb_per = n * sizeof(STREAM_FLOAT) / (1024.0^2)
    @printf("STREAM_META array_mb=%.1f total_mb=%.1f\n", mb_per, 3.0 * mb_per)
    flush(stdout)

    a       = ROCArray{STREAM_FLOAT}(undef, n)
    b       = ROCArray{STREAM_FLOAT}(undef, n)
    c       = ROCArray{STREAM_FLOAT}(undef, n)
    partial = ROCArray{STREAM_FLOAT}(undef, 1)

    do_init!(a, b, c, block_size)

    # Warm-up: absorbs JIT compilation + brings GPU to steady thermal state
    for _ in 1:warmup
        run_pass(a, b, c, partial, n, block_size, all_kernels)
    end
    AMDGPU.synchronize()

    # Correctness check on a fresh init
    if !all_kernels
        do_init!(a, b, c, block_size)
        @roc groupsize=block_size gridsize=cld(n, block_size) triad_kernel!(a, b, c, STREAM_SCALAR)
        AMDGPU.synchronize()
    end
    passes_so_far = all_kernels ? warmup : 0
    if !check_correctness(a, b, c, n, passes_so_far, all_kernels)
        @error "CORRECTNESS CHECK FAILED — aborting."
        exit(1)
    end

    do_init!(a, b, c, block_size)

    triad_bw_vec = Vector{Float64}(undef, num_times)
    copy_bw_vec  = Vector{Float64}(undef, num_times)
    mul_bw_vec   = Vector{Float64}(undef, num_times)
    add_bw_vec   = Vector{Float64}(undef, num_times)
    dot_bw_vec   = Vector{Float64}(undef, num_times)

    for i in 1:num_times
        r = run_pass(a, b, c, partial, n, block_size, all_kernels)
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
    AMDGPU.synchronize()

    print_summary("triad", compute_stats(triad_bw_vec))
    if all_kernels
        print_summary("copy", compute_stats(copy_bw_vec))
        print_summary("mul",  compute_stats(mul_bw_vec))
        print_summary("add",  compute_stats(add_bw_vec))
        print_summary("dot",  compute_stats(dot_bw_vec))
    end
end

main()
