#!/usr/bin/env julia
# kernel_stream_julia.jl — Julia GPU abstraction for E1 STREAM Triad.
#
# Supports CUDA.jl (NVIDIA) and AMDGPU.jl (AMD) from a single source.
# Select backend via environment variable before running:
#   JULIA_GPU_BACKEND=cuda   ./stream-julia   [default]
#   JULIA_GPU_BACKEND=amdgpu ./stream-julia   (or use stream-amdgpu wrapper)
#
# Implements all five BabelStream operations using explicit GPU kernels:
#   Copy  : c[i] = a[i]
#   Mul   : b[i] = scalar * c[i]
#   Add   : c[i] = a[i] + b[i]
#   Triad : a[i] = b[i] + scalar * c[i]   ← PRIMARY E1 metric
#   Dot   : sum += a[i] * b[i]
#
# The five kernel functions are backend-agnostic.  The ~30-line backend
# block below is the only place CUDA.jl / AMDGPU.jl APIs differ.
#
# Requires:
#   CUDA backend  — Julia ≥ 1.9,  CUDA.jl ≥ 5.0
#   AMDGPU backend — Julia ≥ 1.12 (LLVM 18, gfx942), AMDGPU.jl ≥ 2.0
#
# Dependencies: julia -e 'using Pkg; Pkg.add(["CUDA","AMDGPU"])'
# Run: julia --startup-file=no kernel_stream_julia.jl --arraysize 268435456 --numtimes 30

using Printf

# ── Backend selection ─────────────────────────────────────────────────────────
const BACKEND = lowercase(get(ENV, "JULIA_GPU_BACKEND", "cuda"))

if BACKEND == "amdgpu"
    using AMDGPU
    const GPUArray = ROCArray
    # Device-side intrinsics — kernel bodies use these names on both backends
    @inline _tid()  = Int(workitemIdx().x)
    @inline _bid()  = Int(workgroupIdx().x)
    @inline _bdim() = Int(workgroupDim().x)
    @inline _gdim() = Int(gridGroupDim().x)
    @inline _sync() = sync_workgroup()
    # Static local (shared) memory — size must be a compile-time literal
    macro gpu_shmem(T, N);  :(@ROCStaticLocalArray($T, $N));  end
    # Atomic, launch, and elapsed use module-qualified macros (@roc, AMDGPU.@atomic,
    # AMDGPU.@elapsed).  Macro expansion in Julia happens before the conditional
    # block runs, so we defer these definitions with @eval so that AMDGPU is
    # already imported when the macros are resolved.
    @eval begin
        @inline function _gpu_atomic!(arr, val)
            AMDGPU.@atomic arr[1] += val; return
        end
        function _gpu_launch(f, threads, blocks, args...)
            @roc groupsize=threads gridsize=blocks f(args...)
        end
        _gpu_elapsed(f::Function) = AMDGPU.@elapsed f()
    end
    _gpu_sync()               = AMDGPU.synchronize()
    _gpu_device_str()         = string(AMDGPU.device())
    _backend_name()           = "amdgpu"

elseif BACKEND == "cuda"
    using CUDA
    const GPUArray = CuArray
    @inline _tid()  = Int(threadIdx().x)
    @inline _bid()  = Int(blockIdx().x)
    @inline _bdim() = Int(blockDim().x)
    @inline _gdim() = Int(gridDim().x)
    @inline _sync() = sync_threads()
    macro gpu_shmem(T, N);  :(CUDA.CuStaticSharedArray($T, $N));  end
    @inline function _gpu_atomic!(arr, val)
        CUDA.atomic_add!(pointer(arr, 1), val); return
    end
    # @cuda and CUDA.@elapsed need @eval for the same macro-expansion reason.
    @eval begin
        function _gpu_launch(f, threads, blocks, args...)
            @cuda threads=threads blocks=blocks f(args...)
        end
        _gpu_elapsed(f::Function) = CUDA.@elapsed f()
    end
    _gpu_sync()               = CUDA.synchronize()
    _gpu_device_str()         = let d = CUDA.device()
        "$(CUDA.name(d)) (cuda=$(CUDA.runtime_version()))"
    end
    _backend_name()           = "cuda"
else
    error("JULIA_GPU_BACKEND=$(BACKEND) not recognised — set to 'cuda' or 'amdgpu'")
end

# ── Constants matching stream_common.h ───────────────────────────────────────
const STREAM_FLOAT       = get(ENV, "STREAM_USE_FLOAT", "") == "1" ? Float32 : Float64
const STREAM_INIT_A      = STREAM_FLOAT(0.1)
const STREAM_INIT_B      = STREAM_FLOAT(0.2)
const STREAM_INIT_C      = STREAM_FLOAT(0.0)
const STREAM_SCALAR      = STREAM_FLOAT(0.4)
const STREAM_CORRECT_TOL = 1.0e-6
const STREAM_WARMUP      = 10
const STREAM_TIMED       = 30
const DOT_BLOCK_SIZE     = 1024   # must match @gpu_shmem size in dot_kernel!

# ── GPU kernels ───────────────────────────────────────────────────────────────
# These five functions are identical for every backend — only the helpers
# defined in the block above differ.

function init_kernel!(a, b, c, va, vb, vc)
    i = (_bid() - 1) * _bdim() + _tid()
    if i <= length(a)
        @inbounds a[i] = va
        @inbounds b[i] = vb
        @inbounds c[i] = vc
    end
    return
end

function copy_kernel!(a, c)
    i = (_bid() - 1) * _bdim() + _tid()
    if i <= length(a); @inbounds c[i] = a[i]; end
    return
end

function mul_kernel!(b, c, scalar)
    i = (_bid() - 1) * _bdim() + _tid()
    if i <= length(b); @inbounds b[i] = scalar * c[i]; end
    return
end

function add_kernel!(a, b, c)
    i = (_bid() - 1) * _bdim() + _tid()
    if i <= length(a); @inbounds c[i] = a[i] + b[i]; end
    return
end

function triad_kernel!(a, b, c, scalar)
    i = (_bid() - 1) * _bdim() + _tid()
    if i <= length(a); @inbounds a[i] = b[i] + scalar * c[i]; end
    return
end

# Dot: two-stage reduction. Stage 1: grid-stride accumulation per thread.
# Stage 2: block-local tree reduction in static shared/LDS memory.
# One atomic per block writes into the single-element output scalar.
function dot_kernel!(partial, a, b)
    T      = eltype(a)
    shared = @gpu_shmem(T, DOT_BLOCK_SIZE)
    tid    = _tid()
    stride = _bdim() * _gdim()

    acc = zero(T)
    i = (_bid() - 1) * _bdim() + tid
    while i <= length(a)
        @inbounds acc += a[i] * b[i]
        i += stride
    end
    @inbounds shared[tid] = acc
    _sync()

    s = _bdim() >> 1
    while s > 0
        if tid <= s; @inbounds shared[tid] += shared[tid + s]; end
        _sync()
        s >>= 1
    end

    if tid == 1; _gpu_atomic!(partial, shared[1]); end
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
    a = Float64(STREAM_INIT_A); b = Float64(STREAM_INIT_B)
    c = Float64(STREAM_INIT_C); s = Float64(STREAM_SCALAR)
    for _ in 1:n_passes
        c = a; b = s * c; c = a + b; a = b + s * c
    end
    return (a=a, b=b, c=c)
end

# ── Statistics ────────────────────────────────────────────────────────────────
function compute_stats(vals::Vector{Float64})
    n = length(vals); sorted = sort(vals)
    med  = n % 2 == 0 ? (sorted[n÷2] + sorted[n÷2+1]) / 2.0 : sorted[(n+1)÷2]
    mn   = sum(vals) / n
    var  = sum((v - mn)^2 for v in vals) / n
    nout = count(v -> abs(v - mn) > 2*sqrt(var), vals)
    q1   = sorted[max(1, n÷4 + 1)]; q3 = sorted[min(n, 3*(n÷4) + 1)]
    return (median=med, iqr=q3-q1, mean=mn, min=minimum(vals), max=maximum(vals), n_outliers=nout)
end

# ── Output helpers ────────────────────────────────────────────────────────────
function print_run_line(kernel, run_id, n, time_ms, bw_gbs)
    @printf("STREAM_RUN kernel=%s run=%d n=%d time_ms=%.5f bw_gbs=%.4f\n",
            kernel, run_id, n, time_ms, bw_gbs); flush(stdout)
end
function print_summary(kernel, s)
    @printf("STREAM_SUMMARY kernel=%s median_bw_gbs=%.4f iqr_bw_gbs=%.4f min_bw_gbs=%.4f max_bw_gbs=%.4f mean_bw_gbs=%.4f outliers=%d\n",
            kernel, s.median, s.iqr, s.min, s.max, s.mean, s.n_outliers); flush(stdout)
end

# ── CLI ───────────────────────────────────────────────────────────────────────
function parse_cli()
    arraysize = 1 << 26; numtimes = STREAM_TIMED; warmup = STREAM_WARMUP
    all_kernels = false; blocksize = 256
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if     a in ("--arraysize","-n") && i<length(ARGS); arraysize   = parse(Int,  ARGS[i+1]); i+=2
        elseif a in ("--numtimes", "-t") && i<length(ARGS); numtimes    = parse(Int,  ARGS[i+1]); i+=2
        elseif a in ("--warmup",   "-w") && i<length(ARGS); warmup      = parse(Int,  ARGS[i+1]); i+=2
        elseif a in ("--blocksize","-b") && i<length(ARGS); blocksize   = parse(Int,  ARGS[i+1]); i+=2
        elseif a == "--all-kernels";                         all_kernels = true; i+=1
        elseif a in ("--help","-h")
            println("Usage: julia kernel_stream_julia.jl [options]")
            println("  --arraysize N   Elements per array (default: 2^26)")
            println("  --numtimes  N   Timed iterations   (default: $STREAM_TIMED)")
            println("  --warmup    N   Warm-up iterations (default: $STREAM_WARMUP)")
            println("  --blocksize N   GPU thread-block size (default: 256)")
            println("  --all-kernels   Run Copy/Mul/Add/Triad/Dot")
            println("  JULIA_GPU_BACKEND=cuda|amdgpu  (env var, default: cuda)")
            exit(0)
        else; @warn "Unknown argument: $a"; i+=1
        end
    end
    return (; arraysize, numtimes, warmup, all_kernels, blocksize)
end

# ── On-device initialization ───────────────────────────────────────────────────
function do_init!(a::GPUArray, b::GPUArray, c::GPUArray, block_size::Int)
    n = length(a)
    _gpu_launch(init_kernel!, block_size, cld(n, block_size),
                a, b, c, STREAM_INIT_A, STREAM_INIT_B, STREAM_INIT_C)
    _gpu_sync()
end

# ── One timed pass ─────────────────────────────────────────────────────────────
struct PassBW
    copy::Float64; mul::Float64; add::Float64; triad::Float64; dot::Float64
end

function run_pass(a::GPUArray{T}, b::GPUArray{T}, c::GPUArray{T},
                  partial::GPUArray{T}, n::Int, block_size::Int,
                  all_kernels::Bool) where T
    scalar  = STREAM_SCALAR
    nblocks = cld(n, block_size)

    # do-syntax: time_kernel(bw_fn) do ... end  →  time_kernel(launch_fn, bw_fn)
    # so the do-block (launch_fn) is the FIRST positional argument.
    function time_kernel(launch_fn, bw_fn)
        return bw_fn(n, _gpu_elapsed(launch_fn))
    end

    copy_bw = mul_bw = add_bw = dot_bw = 0.0

    if all_kernels
        copy_bw = time_kernel(copy_bw_gbs) do
            _gpu_launch(copy_kernel!, block_size, nblocks, a, c)
        end
        mul_bw = time_kernel(mul_bw_gbs) do
            _gpu_launch(mul_kernel!, block_size, nblocks, b, c, scalar)
        end
        add_bw = time_kernel(add_bw_gbs) do
            _gpu_launch(add_kernel!, block_size, nblocks, a, b, c)
        end
    end

    triad_bw_val = time_kernel(triad_bw_gbs) do
        _gpu_launch(triad_kernel!, block_size, nblocks, a, b, c, scalar)
    end

    if all_kernels
        fill!(partial, zero(T))
        dot_bw = time_kernel(dot_bw_gbs) do
            _gpu_launch(dot_kernel!, DOT_BLOCK_SIZE, cld(n, DOT_BLOCK_SIZE), partial, a, b)
        end
    end

    return PassBW(copy_bw, mul_bw, add_bw, triad_bw_val, dot_bw)
end

# ── Correctness check ──────────────────────────────────────────────────────────
# Uses fancy indexing + Array() to copy 3 sample elements to CPU — works on
# both CuArray and ROCArray via GPUArrays.jl without needing @allowscalar.
function check_correctness(a::GPUArray{T}, b::GPUArray{T}, c::GPUArray{T},
                            n::Int, n_passes::Int, all_kernels::Bool) where T
    exp = all_kernels ? compute_expected(n_passes) :
          (a = Float64(STREAM_INIT_B + STREAM_SCALAR * STREAM_INIT_C),
           b = Float64(STREAM_INIT_B),
           c = Float64(STREAM_INIT_C))

    relerr(got, expected) = abs(Float64(got) - expected) /
                            max(abs(expected), 1e-12)

    sample = [1, n÷2, n]
    a_s = Array(a[sample]); b_s = Array(b[sample]); c_s = Array(c[sample])

    max_ea = maximum(relerr(a_s[k], exp.a) for k in 1:3)
    max_eb = maximum(relerr(b_s[k], exp.b) for k in 1:3)
    max_ec = maximum(relerr(c_s[k], exp.c) for k in 1:3)

    pass = max_ea < STREAM_CORRECT_TOL && max_eb < STREAM_CORRECT_TOL &&
           max_ec < STREAM_CORRECT_TOL

    @printf("STREAM_CORRECT %s max_err_a=%.3e max_err_b=%.3e max_err_c=%.3e\n",
            pass ? "PASS" : "FAIL", max_ea, max_eb, max_ec)
    if !pass
        max_ea >= STREAM_CORRECT_TOL && @printf("STREAM_CORRECT DETAIL array=a expected=%.10f\n", exp.a)
        max_eb >= STREAM_CORRECT_TOL && @printf("STREAM_CORRECT DETAIL array=b expected=%.10f\n", exp.b)
        max_ec >= STREAM_CORRECT_TOL && @printf("STREAM_CORRECT DETAIL array=c expected=%.10f\n", exp.c)
    end
    flush(stdout); return pass
end

# ── Main ───────────────────────────────────────────────────────────────────────
function main()
    opts        = parse_cli()
    n           = opts.arraysize
    num_times   = opts.numtimes
    warmup      = opts.warmup
    all_kernels = opts.all_kernels
    block_size  = opts.blocksize
    prec_str    = STREAM_FLOAT == Float32 ? "float" : "double"

    @printf("STREAM_META abstraction=julia backend=%s device=\"%s\" julia=%s precision=%s n=%d sizeof=%d warmup=%d timed=%d all_kernels=%d\n",
            _backend_name(), _gpu_device_str(), string(VERSION), prec_str,
            n, sizeof(STREAM_FLOAT), warmup, num_times, Int(all_kernels))
    mb_per = n * sizeof(STREAM_FLOAT) / (1024.0^2)
    @printf("STREAM_META array_mb=%.1f total_mb=%.1f\n", mb_per, 3.0 * mb_per)
    flush(stdout)

    a       = GPUArray{STREAM_FLOAT}(undef, n)
    b       = GPUArray{STREAM_FLOAT}(undef, n)
    c       = GPUArray{STREAM_FLOAT}(undef, n)
    partial = GPUArray{STREAM_FLOAT}(undef, 1)

    do_init!(a, b, c, block_size)

    for _ in 1:warmup
        run_pass(a, b, c, partial, n, block_size, all_kernels)
    end
    _gpu_sync()

    if !all_kernels
        do_init!(a, b, c, block_size)
        _gpu_launch(triad_kernel!, block_size, cld(n, block_size),
                    a, b, c, STREAM_SCALAR)
        _gpu_sync()
    end
    passes_so_far = all_kernels ? warmup : 0
    if !check_correctness(a, b, c, n, passes_so_far, all_kernels)
        @error "CORRECTNESS CHECK FAILED — aborting."; exit(1)
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
            copy_bw_vec[i] = r.copy; mul_bw_vec[i] = r.mul
            add_bw_vec[i]  = r.add;  dot_bw_vec[i] = r.dot
        end
        time_ms = 3.0 * n * sizeof(STREAM_FLOAT) / (r.triad * 1e9) * 1e3
        print_run_line("triad", i, n, time_ms, r.triad)
    end
    _gpu_sync()

    print_summary("triad", compute_stats(triad_bw_vec))
    if all_kernels
        print_summary("copy", compute_stats(copy_bw_vec))
        print_summary("mul",  compute_stats(mul_bw_vec))
        print_summary("add",  compute_stats(add_bw_vec))
        print_summary("dot",  compute_stats(dot_bw_vec))
    end
end

main()
