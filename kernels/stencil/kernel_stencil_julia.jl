#!/usr/bin/env julia
# kernel_stencil_julia.jl — E3 3D Stencil: Julia GPU-backend-aware kernel.
#
# E3 DESIGN DECISIONS
# [D4-Julia] @cuda/@roc kernel with 3D grid. Column-major GPU arrays.
#   For GPUArray{Float64,3} with dimensions (NX, NY, NZ), the first index (ix)
#   varies fastest in memory. threadIdx().x (CUDA) / workitemIdx().x (AMDGPU)
#   maps to ix → 32 consecutive threads access consecutive memory → coalesced.
# [D7-Julia] Adaptive warmup: tracks CV over last 10 timings, stops when CV < 2%.
# [D3] c0=0.5, c1=(1-c0)/6. FP64 throughout.
#
# Backend selection: JULIA_GPU_BACKEND env var (default: "cuda")
#   cuda   → CUDA.jl + CuArray   (--mode not applicable; @cuda launch)
#   amdgpu → AMDGPU.jl + ROCArray (@roc launch)

const BACKEND = lowercase(get(ENV, "JULIA_GPU_BACKEND", "cuda"))

if BACKEND == "amdgpu"
    using AMDGPU
    const GPUArray = ROCArray
    gpu_rand(T, dims...)  = AMDGPU.rand(T, dims...)
    gpu_zeros(T, dims...) = AMDGPU.zeros(T, dims...)
    gpu_sync()            = AMDGPU.synchronize()
    to_gpu(x)             = ROCArray(x)
    from_gpu(x)           = Array(x)
    # Backend-agnostic intrinsic wrappers (OpenCL naming for AMDGPU)
    @inline _threadIdx() = workitemIdx()
    @inline _blockIdx()  = workgroupIdx()
    @inline _blockDim()  = workgroupDim()
    # @roc needs @eval so it is resolved after AMDGPU is imported
    @eval function _gpu_launch!(out, inp, N32, c0, c1, threads, blocks)
        @roc groupsize=threads gridsize=blocks stencil7pt_kernel!(
            out, inp, N32, c0, c1)
    end

elseif BACKEND == "cuda"
    using CUDA
    const GPUArray = CuArray
    gpu_rand(T, dims...)  = CUDA.rand(T, dims...)
    gpu_zeros(T, dims...) = CUDA.zeros(T, dims...)
    gpu_sync()            = CUDA.synchronize()
    to_gpu(x)             = CuArray(x)
    from_gpu(x)           = Array(x)
    # CUDA.jl exports threadIdx/blockIdx/blockDim with CUDA naming
    @inline _threadIdx() = threadIdx()
    @inline _blockIdx()  = blockIdx()
    @inline _blockDim()  = blockDim()
    @eval function _gpu_launch!(out, inp, N32, c0, c1, threads, blocks)
        @cuda threads=threads blocks=blocks stencil7pt_kernel!(
            out, inp, N32, c0, c1)
    end

else
    error("JULIA_GPU_BACKEND=$(BACKEND) not recognised — set to 'cuda' or 'amdgpu'")
end

using Printf: @printf
using Statistics: median, mean, quantile, std

# ── 7-point stencil kernel ────────────────────────────────────────────────────
# GPUArray{Float64,3}(undef, N, N, N): column-major, A[ix, iy, iz].
# ix is the fastest-varying index → coalesced when _threadIdx().x → ix.
function stencil7pt_kernel!(out, inp, N::Int32, c0::Float64, c1::Float64)
    ix = (_blockIdx().x - Int32(1)) * _blockDim().x + _threadIdx().x
    iy = (_blockIdx().y - Int32(1)) * _blockDim().y + _threadIdx().y
    iz = (_blockIdx().z - Int32(1)) * _blockDim().z + _threadIdx().z

    # 1-indexed interior: skip boundary indices 1 and N
    if ix >= Int32(2) && ix <= N - Int32(1) &&
       iy >= Int32(2) && iy <= N - Int32(1) &&
       iz >= Int32(2) && iz <= N - Int32(1)
        @inbounds out[ix, iy, iz] = c0 * inp[ix, iy, iz] +
            c1 * (inp[ix-1, iy,   iz  ] + inp[ix+1, iy,   iz  ] +
                  inp[ix,   iy-1, iz  ] + inp[ix,   iy+1, iz  ] +
                  inp[ix,   iy,   iz-1] + inp[ix,   iy,   iz+1])
    end
    return nothing
end

# ── Performance formulas ──────────────────────────────────────────────────────
interior_cells(N::Int) = (N - 2)^3
stencil_gbs(N::Int, time_s::Float64) = interior_cells(N) * 64.0 / time_s / 1e9

# ── hw_state_verified (§9.7) ─────────────────────────────────────────────────
function compute_hw_state(vals::Vector{Float64})
    isempty(vals) && return Int[]
    med = median(vals)
    denom = abs(med) < 1e-12 ? 1.0 : abs(med)
    return [abs(v - med) / denom <= 0.15 ? 1 : 0 for v in vals]
end

# ── Adaptive warmup (D7) ──────────────────────────────────────────────────────
function adaptive_warmup!(run_once!::Function; warmup_min::Int=10,
                           warmup_max::Int=200, window_size::Int=10,
                           cv_ceil::Float64=2.0)
    window = Float64[]
    sizehint!(window, window_size + 1)
    total = 0
    while total < warmup_max
        t0 = time_ns()
        run_once!()
        t1 = time_ns()
        ms = (t1 - t0) / 1e6
        push!(window, ms)
        if length(window) > window_size
            popfirst!(window)
        end
        total += 1
        if total >= warmup_min && length(window) == window_size
            m = mean(window)
            s = std(window; corrected=false)
            cv = m > 0.0 ? 100.0 * s / m : 100.0
            cv < cv_ceil && break
        end
    end
    return total
end

# ── Correctness check ─────────────────────────────────────────────────────────
function verify_stencil(N::Int, c0::Float64, c1::Float64)::Bool
    hIn  = [sin(ix/N) + cos(iy/N) + sin(iz/N + 0.5)
            for ix in 0:N-1, iy in 0:N-1, iz in 0:N-1]
    hRef = zeros(Float64, N, N, N)
    for iz in 2:N-1, iy in 2:N-1, ix in 2:N-1
        hRef[ix,iy,iz] = c0 * hIn[ix,iy,iz] +
            c1 * (hIn[ix-1,iy,iz] + hIn[ix+1,iy,iz] +
                  hIn[ix,iy-1,iz] + hIn[ix,iy+1,iz] +
                  hIn[ix,iy,iz-1] + hIn[ix,iy,iz+1])
    end

    inp_d = to_gpu(hIn)
    out_d = gpu_zeros(Float64, N, N, N)
    N32   = Int32(N)
    bx, by, bz = 32, 4, 2
    gx = cld(N, bx); gy = cld(N, by); gz = cld(N, bz)
    _gpu_launch!(out_d, inp_d, N32, c0, c1, (bx,by,bz), (gx,gy,gz))
    gpu_sync()
    hOut = from_gpu(out_d)

    max_err = 0.0
    for iz in 2:N-1, iy in 2:N-1, ix in 2:N-1
        denom = abs(hRef[ix,iy,iz]) < 1e-14 ? 1.0 : abs(hRef[ix,iy,iz])
        err = abs(hOut[ix,iy,iz] - hRef[ix,iy,iz]) / denom
        max_err = max(max_err, err)
    end
    ok = max_err < 1e-10
    @printf("VERIFY abstraction=julia backend=%s N=%d max_rel_err=%.2e %s\n",
            BACKEND, N, max_err, ok ? "PASS" : "FAIL")
    return ok
end

# ── Run experiment ────────────────────────────────────────────────────────────
function run_stencil(N::Int, warmup_max::Int, reps::Int, platform::String)
    @printf("# abstraction=julia backend=%s N=%d warmup_max=%d reps=%d platform=%s\n",
            BACKEND, N, warmup_max, reps, platform)

    c0 = 0.5; c1 = (1.0 - c0) / 6.0
    N32 = Int32(N)
    bx, by, bz = 32, 4, 2
    gx = cld(N, bx); gy = cld(N, by); gz = cld(N, bz)

    inp_d = gpu_rand(Float64, N, N, N)
    out_d = gpu_zeros(Float64, N, N, N)

    function run_once!()
        _gpu_launch!(out_d, inp_d, N32, c0, c1, (bx,by,bz), (gx,gy,gz))
        gpu_sync()
        inp_d, out_d = out_d, inp_d
    end

    warmup_iters = adaptive_warmup!(run_once!; warmup_max=warmup_max)
    println(stderr, "[E3] julia-$(BACKEND): adaptive warmup done in $(warmup_iters) iterations")

    gbs_vec = Float64[]
    sizehint!(gbs_vec, reps)

    for r in 1:reps
        t0 = time_ns()
        run_once!()
        t1 = time_ns()
        time_ms = (t1 - t0) / 1e6
        gbs = stencil_gbs(N, time_ms / 1000.0)
        push!(gbs_vec, gbs)
        @printf("STENCIL_RUN run=%d n=%d time_ms=%.6f throughput_gbs=%.6f\n",
                r, N, time_ms, gbs)
        flush(stdout)
    end

    flags = compute_hw_state(gbs_vec)
    for r in 1:reps
        @printf("STENCIL_HW_STATE run=%d hw_state=%d\n", r, flags[r])
    end

    clean = [gbs_vec[i] for i in 1:reps if flags[i] == 1]
    n_clean = length(clean)
    if n_clean > 0
        med = median(clean)
        q1  = quantile(clean, 0.25)
        q3  = quantile(clean, 0.75)
        @printf("STENCIL_SUMMARY n=%d median_gbs=%.4f iqr_gbs=%.4f min_gbs=%.4f max_gbs=%.4f mean_gbs=%.4f n_clean=%d warmup_iters=%d\n",
                N, med, q3-q1, minimum(clean), maximum(clean), mean(clean),
                n_clean, warmup_iters)
    else
        @printf("STENCIL_SUMMARY n=%d median_gbs=0.0 iqr_gbs=0.0 min_gbs=0.0 max_gbs=0.0 mean_gbs=0.0 n_clean=0 warmup_iters=%d\n",
                N, warmup_iters)
    end
    flush(stdout)
end

# ── Entry point ───────────────────────────────────────────────────────────────
function main()
    N          = 256
    warmup_max = 200
    reps       = 30
    platform   = "unknown"
    do_verify  = false

    args = ARGS
    i = 1
    while i <= length(args)
        if     args[i] == "--n"        && i < length(args); N          = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--warmup"   && i < length(args); warmup_max = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--reps"     && i < length(args); reps       = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--platform" && i < length(args); platform   = args[i+1];              i += 2
        elseif args[i] == "--verify";                        do_verify  = true;                  i += 1
        else
            @error "Unknown argument: $(args[i])"
            exit(1)
        end
    end

    if do_verify
        ok = verify_stencil(16, 0.5, (1.0 - 0.5) / 6.0)
        if !ok
            @error "[E3 verify] julia-$(BACKEND) FAILED — aborting before timing."
            exit(1)
        end
        println(stderr, "[E3 verify] julia-$(BACKEND) PASS — proceeding to timing.")
    end

    run_stencil(N, warmup_max, reps, platform)
end

main()
