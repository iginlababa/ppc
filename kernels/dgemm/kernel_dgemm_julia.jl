#!/usr/bin/env julia
# kernel_dgemm_julia.jl — E2 DGEMM: Julia/CUDA.jl naive + CUBLAS ceiling.
#
# E2 DESIGN DECISIONS
# [D2-Julia] --mode naive: simple @cuda kernel, no shared memory, column-major
#   CuArrays. Julia's natural layout. Each thread computes one C[row,col] via
#   O(N) inner loop from global memory. Expected behavior: memory-bandwidth-
#   limited, similar to raja_naive, exposing the same no-tiling penalty.
#   (Tiling in Julia with @cuStaticSharedMem is possible but out of scope —
#   naive is the scientifically interesting comparison here.)
# [D5-Julia] --mode cublas: CUBLAS.gemm! ceiling reference (julia_cublas).
#   Uses Julia's column-major CuArrays directly — no transpose tricks needed.
#   CUBLAS.gemm! dispatches to cuBLAS DGEMM natively.
# [D3] alpha=1.0, beta=0.0. Julia arrays are 1-indexed, column-major.
# [D7] experiment_id format: dgemm_julia_{naive|cublas}_{platform}_{size}_n{N}_{run:03d}

using CUDA
using LinearAlgebra: mul!
using Printf: @printf
using Statistics: median, mean, quantile

# ── Design decisions logged at runtime ───────────────────────────────────────
const DESIGN_DECISIONS = """
# E2 DESIGN DECISIONS (Julia)
# [D2-Julia] naive mode: no shared memory; column-major CuArrays; one thread per element
# [D5-Julia] cublas mode: CUBLAS.gemm! ceiling reference
# [D3] alpha=1.0, beta=0.0
"""

# ── Naive kernel: one thread per output element ───────────────────────────────
function dgemm_naive_kernel!(C, A, B, alpha::Float64, beta::Float64, N::Int32)
    # Thread mapping: row varies by threadIdx.x (column-major friendly)
    row = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    col = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y

    if row <= N && col <= N
        acc = 0.0
        for k = Int32(1):N
            @inbounds acc += A[row, k] * B[k, col]
        end
        @inbounds C[row, col] = alpha * acc + beta * C[row, col]
    end
    return nothing
end

# ── GFLOP/s formula ───────────────────────────────────────────────────────────
dgemm_gflops(N::Int, time_s::Float64) = 2.0 * N^3 / time_s / 1e9

# ── hw_state_verified (§9.7): |val - median| / median <= 0.15 ────────────────
function compute_hw_state(vals::Vector{Float64})
    n = length(vals)
    n == 0 && return Int[]
    med = median(vals)
    denom = abs(med) < 1e-12 ? 1.0 : abs(med)
    return [abs(v - med) / denom <= 0.15 ? 1 : 0 for v in vals]
end

# ── Run one timed experiment ──────────────────────────────────────────────────
function run_dgemm(mode::String, N::Int, warmup::Int, reps::Int, platform::String)
    print(DESIGN_DECISIONS)
    @printf("# abstraction=julia_%s N=%d warmup=%d reps=%d platform=%s\n",
            mode, N, warmup, reps, platform)

    # Allocate column-major CuArrays (Julia default)
    A_d = CUDA.rand(Float64, N, N)
    B_d = CUDA.rand(Float64, N, N)
    C_d = CUDA.zeros(Float64, N, N)

    tile = 32
    threads = (tile, tile)
    blocks  = (cld(N, tile), cld(N, tile))
    alpha   = 1.0
    beta    = 0.0
    N32     = Int32(N)

    run_once = if mode == "cublas"
        () -> begin
            CUBLAS.gemm!('N', 'N', alpha, A_d, B_d, beta, C_d)
            CUDA.synchronize()
        end
    else  # naive
        () -> begin
            @cuda threads=threads blocks=blocks dgemm_naive_kernel!(
                C_d, A_d, B_d, alpha, beta, N32)
            CUDA.synchronize()
        end
    end

    # Warmup
    for _ in 1:warmup
        run_once()
    end
    CUDA.synchronize()

    # Timed runs
    gflops_vec = Float64[]
    sizehint!(gflops_vec, reps)

    for r in 1:reps
        t0 = time_ns()
        run_once()
        t1 = time_ns()
        time_ms = (t1 - t0) / 1e6
        gf = dgemm_gflops(N, time_ms / 1000.0)
        push!(gflops_vec, gf)
        @printf("DGEMM_RUN run=%d n=%d time_ms=%.6f gflops=%.6f\n", r, N, time_ms, gf)
        flush(stdout)
    end

    # hw_state_verified
    flags = compute_hw_state(gflops_vec)
    for r in 1:reps
        @printf("DGEMM_HW_STATE run=%d hw_state=%d\n", r, flags[r])
    end

    # Summary (clean runs only)
    clean = [gflops_vec[i] for i in 1:reps if flags[i] == 1]
    n_clean = length(clean)
    if n_clean > 0
        med = median(clean)
        q1  = quantile(clean, 0.25)
        q3  = quantile(clean, 0.75)
        @printf("DGEMM_SUMMARY n=%d median_gflops=%.4f iqr_gflops=%.4f min_gflops=%.4f max_gflops=%.4f mean_gflops=%.4f n_clean=%d\n",
                N, med, q3 - q1, minimum(clean), maximum(clean), mean(clean), n_clean)
    else
        @printf("DGEMM_SUMMARY n=%d median_gflops=0.0 iqr_gflops=0.0 min_gflops=0.0 max_gflops=0.0 mean_gflops=0.0 n_clean=0\n", N)
    end
    flush(stdout)
end

# ── Entry point ───────────────────────────────────────────────────────────────
function main()
    # Defaults matching E2 protocol
    N        = 8192
    warmup   = 50
    reps     = 30
    platform = "unknown"
    mode     = "naive"   # naive | cublas
    verify   = false

    args = ARGS
    i = 1
    while i <= length(args)
        if     args[i] == "--n"        && i < length(args); N        = parse(Int,    args[i+1]); i += 2
        elseif args[i] == "--warmup"   && i < length(args); warmup   = parse(Int,    args[i+1]); i += 2
        elseif args[i] == "--reps"     && i < length(args); reps     = parse(Int,    args[i+1]); i += 2
        elseif args[i] == "--platform" && i < length(args); platform = args[i+1];                i += 2
        elseif args[i] == "--mode"     && i < length(args); mode     = args[i+1];                i += 2
        elseif args[i] == "--verify";                        verify   = true;                     i += 1
        else
            @error "Unknown argument: $(args[i])"
            exit(1)
        end
    end

    if mode ∉ ("naive", "cublas")
        @error "--mode must be naive or cublas"
        exit(1)
    end

    # ── Correctness check ─────────────────────────────────────────────────────
    if verify
        Nv    = 128
        # CPU reference: Julia mul! for column-major (equivalent to A*B)
        hAv   = [1.0 / Float64(i + j) for i in 1:Nv, j in 1:Nv]
        hBv   = [1.0 / Float64(i + j) for i in 1:Nv, j in 1:Nv]
        hRv   = hAv * hBv   # CPU reference (column-major, alpha=1, beta=0)

        # GPU run at Nv
        A_d = CuArray(hAv)
        B_d = CuArray(hBv)
        C_d = CUDA.zeros(Float64, Nv, Nv)
        tile = 32
        alpha_v = 1.0; beta_v = 0.0; Nv32 = Int32(Nv)
        if mode == "cublas"
            CUBLAS.gemm!('N', 'N', alpha_v, A_d, B_d, beta_v, C_d)
        else
            vth = (tile, tile)
            vbk = (cld(Nv, tile), cld(Nv, tile))
            @cuda threads=vth blocks=vbk dgemm_naive_kernel!(C_d, A_d, B_d, alpha_v, beta_v, Nv32)
        end
        CUDA.synchronize()
        hCv = Array(C_d)

        max_err = maximum(abs.(hCv .- hRv) ./ max.(abs.(hRv), 1e-12))
        ok = max_err < 1e-6
        pass_str = ok ? "PASS" : "FAIL"
        @printf("VERIFY abstraction=julia_%s N=%d max_rel_err=%.2e %s\n",
                mode, Nv, max_err, pass_str)
        if !ok
            @error "[E2 verify] julia_$(mode) FAILED — aborting before timing."
            exit(1)
        end
        println(stderr, "[E2 verify] julia_$(mode) PASS — proceeding to timed measurement.")
    end

    run_dgemm(mode, N, warmup, reps, platform)
end

main()
