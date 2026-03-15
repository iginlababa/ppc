#!/usr/bin/env julia
# kernel_stencil_julia.jl — E3 3D Stencil: Julia/CUDA.jl @cuda kernel.
#
# E3 DESIGN DECISIONS
# [D4-Julia] @cuda kernel with 3D grid. Julia CuArrays are 1-indexed, column-major.
#   For Array{Float64,3} with dimensions (NX, NY, NZ), the first index (ix) varies
#   fastest in memory. Threads are launched with blockDim.x → ix: 32 consecutive
#   threads access consecutive memory → coalesced reads.
#   threadIdx().x, .y, .z map to ix, iy, iz respectively.
# [D7-Julia] Adaptive warmup: tracks CV over last 10 timings, stops when CV < 2%.
# [D3] c0=0.5, c1=(1-c0)/6. FP64 throughout.

using CUDA
using Printf: @printf
using Statistics: median, mean, quantile, std

const DESIGN_DECISIONS = """
# E3 DESIGN DECISIONS (Julia)
# [D4-Julia] @cuda kernel; 3D grid; column-major CuArray{Float64,3}; ix=threadIdx.x (coalesced)
# [D7-Julia] Adaptive warmup: CV < 2% over last 10 timings
# [D3] c0=0.5, c1=(1-c0)/6, FP64
"""

# ── 7-point stencil kernel ────────────────────────────────────────────────────
# CuArray{Float64,3}(undef, N, N, N): column-major, A[ix, iy, iz].
# ix is the fastest-varying index → coalesced when threadIdx().x → ix.
function stencil7pt_kernel!(out, inp, N::Int32, c0::Float64, c1::Float64)
    ix = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    iz = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    # 1-indexed interior: ix ∈ [2, N-1] (skip boundaries 1 and N)
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
# Interior cells: (N-2)^3
interior_cells(N::Int) = (N - 2)^3

# GB/s: 8 doubles (7 reads + 1 write) * 8 bytes each
stencil_gbs(N::Int, time_s::Float64) = interior_cells(N) * 64.0 / time_s / 1e9

# ── hw_state_verified (§9.7) ─────────────────────────────────────────────────
function compute_hw_state(vals::Vector{Float64})
    isempty(vals) && return Int[]
    med = median(vals)
    denom = abs(med) < 1e-12 ? 1.0 : abs(med)
    return [abs(v - med) / denom <= 0.15 ? 1 : 0 for v in vals]
end

# ── Adaptive warmup (D7) ──────────────────────────────────────────────────────
# Returns number of warmup iterations executed.
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
    # CPU reference: row-major (iz,iy,ix) indexing for parity with C++ ref
    hIn  = [sin(ix/N) + cos(iy/N) + sin(iz/N + 0.5)
            for ix in 0:N-1, iy in 0:N-1, iz in 0:N-1]  # column-major [ix,iy,iz]
    hRef = zeros(Float64, N, N, N)
    # CPU reference (column-major [ix,iy,iz])
    for iz in 2:N-1, iy in 2:N-1, ix in 2:N-1
        hRef[ix,iy,iz] = c0 * hIn[ix,iy,iz] +
            c1 * (hIn[ix-1,iy,iz] + hIn[ix+1,iy,iz] +
                  hIn[ix,iy-1,iz] + hIn[ix,iy+1,iz] +
                  hIn[ix,iy,iz-1] + hIn[ix,iy,iz+1])
    end

    inp_d = CuArray(hIn)
    out_d = CUDA.zeros(Float64, N, N, N)
    N32   = Int32(N)
    bx, by, bz = 32, 4, 2
    gx = cld(N, bx); gy = cld(N, by); gz = cld(N, bz)
    @cuda threads=(bx,by,bz) blocks=(gx,gy,gz) stencil7pt_kernel!(
        out_d, inp_d, N32, c0, c1)
    CUDA.synchronize()
    hOut = Array(out_d)

    max_err = 0.0
    for iz in 2:N-1, iy in 2:N-1, ix in 2:N-1
        denom = abs(hRef[ix,iy,iz]) < 1e-14 ? 1.0 : abs(hRef[ix,iy,iz])
        err = abs(hOut[ix,iy,iz] - hRef[ix,iy,iz]) / denom
        max_err = max(max_err, err)
    end
    ok = max_err < 1e-10
    @printf("VERIFY abstraction=julia N=%d max_rel_err=%.2e %s\n",
            N, max_err, ok ? "PASS" : "FAIL")
    return ok
end

# ── Run experiment ────────────────────────────────────────────────────────────
function run_stencil(N::Int, warmup_max::Int, reps::Int, platform::String)
    print(DESIGN_DECISIONS)
    @printf("# abstraction=julia N=%d warmup_max=%d reps=%d platform=%s\n",
            N, warmup_max, reps, platform)

    c0 = 0.5; c1 = (1.0 - c0) / 6.0
    N32 = Int32(N)
    bx, by, bz = 32, 4, 2
    gx = cld(N, bx); gy = cld(N, by); gz = cld(N, bz)

    # Allocate column-major CuArrays
    inp_d = CUDA.rand(Float64, N, N, N)
    out_d = CUDA.zeros(Float64, N, N, N)

    function run_once!()
        @cuda threads=(bx,by,bz) blocks=(gx,gy,gz) stencil7pt_kernel!(
            out_d, inp_d, N32, c0, c1)
        CUDA.synchronize()
        # Swap buffers for multi-step correctness
        inp_d, out_d = out_d, inp_d
    end

    # Adaptive warmup
    warmup_iters = adaptive_warmup!(run_once!; warmup_max=warmup_max)
    println(stderr, "[E3] julia: adaptive warmup done in $(warmup_iters) iterations")

    # Timed runs
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

    # hw_state_verified
    flags = compute_hw_state(gbs_vec)
    for r in 1:reps
        @printf("STENCIL_HW_STATE run=%d hw_state=%d\n", r, flags[r])
    end

    # Summary (clean runs only)
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
            @error "[E3 verify] julia FAILED — aborting before timing."
            exit(1)
        end
        println(stderr, "[E3 verify] julia PASS — proceeding to timed measurement.")
    end

    run_stencil(N, warmup_max, reps, platform)
end

main()
