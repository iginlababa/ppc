#!/usr/bin/env julia
# kernel_dgemm_julia.jl — E2 DGEMM: Julia GPU-backend-aware naive + library ceiling.
#
# E2 DESIGN DECISIONS
# [D2-Julia] --mode naive: simple GPU kernel, no shared memory, column-major arrays.
#   Julia's natural layout. Each thread computes one C[row,col] via O(N) inner
#   loop from global memory. Memory-bandwidth-limited, same no-tiling penalty
#   as raja_naive.
# [D5-Julia] --mode cublas / --mode rocblas: library ceiling reference.
#   Uses Julia's column-major GPU arrays directly — no transpose tricks needed.
# [D3] alpha=1.0, beta=0.0.
# [D7] experiment_id: dgemm_julia_{naive|cublas|rocblas}_{platform}_{size}_n{N}_{run:03d}
#
# Backend selection: JULIA_GPU_BACKEND env var (default: "cuda")
#   cuda   → CUDA.jl + CUBLAS  (--mode naive|cublas)
#   amdgpu → AMDGPU.jl + rocBLAS (--mode naive|rocblas)

const BACKEND = lowercase(get(ENV, "JULIA_GPU_BACKEND", "cuda"))

if BACKEND == "amdgpu"
    using AMDGPU
    import AMDGPU.rocBLAS
    const GPUArray = ROCArray
    gpu_rand(T, dims...)  = AMDGPU.rand(T, dims...)
    gpu_zeros(T, dims...) = AMDGPU.zeros(T, dims...)
    gpu_sync()            = AMDGPU.synchronize()
    to_gpu(x)             = ROCArray(x)
    from_gpu(x)           = Array(x)
    # Backend-agnostic intrinsic wrappers for use inside kernel bodies.
    # AMDGPU uses workitemIdx/workgroupIdx/workgroupDim (OpenCL naming).
    @inline _threadIdx() = workitemIdx()
    @inline _blockIdx()  = workgroupIdx()
    @inline _blockDim()  = workgroupDim()
    # @roc macro needs @eval so it is resolved after AMDGPU is imported
    @eval function _gpu_launch!(C_d, A_d, B_d, alpha, beta, N32, threads, blocks)
        @roc groupsize=threads gridsize=blocks dgemm_naive_kernel!(
            C_d, A_d, B_d, alpha, beta, N32)
    end
    function _gpu_blas_gemm!(A_d, B_d, C_d, alpha, beta)
        rocBLAS.gemm!('N', 'N', alpha, A_d, B_d, beta, C_d)
    end
    const BLAS_MODE = "rocblas"

elseif BACKEND == "cuda"
    using CUDA
    import CUDA.CUBLAS
    const GPUArray = CuArray
    gpu_rand(T, dims...)  = CUDA.rand(T, dims...)
    gpu_zeros(T, dims...) = CUDA.zeros(T, dims...)
    gpu_sync()            = CUDA.synchronize()
    to_gpu(x)             = CuArray(x)
    from_gpu(x)           = Array(x)
    # CUDA.jl exports threadIdx/blockIdx/blockDim with CUDA naming.
    @inline _threadIdx() = threadIdx()
    @inline _blockIdx()  = blockIdx()
    @inline _blockDim()  = blockDim()
    # @cuda macro needs @eval for the same reason
    @eval function _gpu_launch!(C_d, A_d, B_d, alpha, beta, N32, threads, blocks)
        @cuda threads=threads blocks=blocks dgemm_naive_kernel!(
            C_d, A_d, B_d, alpha, beta, N32)
    end
    function _gpu_blas_gemm!(A_d, B_d, C_d, alpha, beta)
        CUBLAS.gemm!('N', 'N', alpha, A_d, B_d, beta, C_d)
    end
    const BLAS_MODE = "cublas"

else
    error("JULIA_GPU_BACKEND=$(BACKEND) not recognised — set to 'cuda' or 'amdgpu'")
end

using Printf: @printf
using Statistics: median, mean, quantile

# ── Naive kernel: one thread per output element ───────────────────────────────
function dgemm_naive_kernel!(C, A, B, alpha::Float64, beta::Float64, N::Int32)
    row = (_blockIdx().x - Int32(1)) * _blockDim().x + _threadIdx().x
    col = (_blockIdx().y - Int32(1)) * _blockDim().y + _threadIdx().y
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
    @printf("# abstraction=julia_%s backend=%s N=%d warmup=%d reps=%d platform=%s\n",
            mode, BACKEND, N, warmup, reps, platform)

    A_d = gpu_rand(Float64, N, N)
    B_d = gpu_rand(Float64, N, N)
    C_d = gpu_zeros(Float64, N, N)

    tile    = 32
    threads = (tile, tile)
    blocks  = (cld(N, tile), cld(N, tile))
    alpha   = 1.0
    beta    = 0.0
    N32     = Int32(N)

    run_once = if mode == BLAS_MODE
        () -> begin
            _gpu_blas_gemm!(A_d, B_d, C_d, alpha, beta)
            gpu_sync()
        end
    else  # naive
        () -> begin
            _gpu_launch!(C_d, A_d, B_d, alpha, beta, N32, threads, blocks)
            gpu_sync()
        end
    end

    # Warmup
    for _ in 1:warmup
        run_once()
    end
    gpu_sync()

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
    N        = 8192
    warmup   = 50
    reps     = 30
    platform = "unknown"
    mode     = "naive"
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

    if mode ∉ ("naive", BLAS_MODE)
        @error "--mode must be naive or $(BLAS_MODE) for BACKEND=$(BACKEND)"
        exit(1)
    end

    # ── Correctness check ─────────────────────────────────────────────────────
    if verify
        Nv      = 128
        hAv     = [1.0 / Float64(i + j) for i in 1:Nv, j in 1:Nv]
        hBv     = [1.0 / Float64(i + j) for i in 1:Nv, j in 1:Nv]
        hRv     = hAv * hBv

        A_d = to_gpu(hAv)
        B_d = to_gpu(hBv)
        C_d = gpu_zeros(Float64, Nv, Nv)
        tile = 32
        alpha_v = 1.0; beta_v = 0.0; Nv32 = Int32(Nv)
        if mode == BLAS_MODE
            _gpu_blas_gemm!(A_d, B_d, C_d, alpha_v, beta_v)
        else
            vth = (tile, tile)
            vbk = (cld(Nv, tile), cld(Nv, tile))
            _gpu_launch!(C_d, A_d, B_d, alpha_v, beta_v, Nv32, vth, vbk)
        end
        gpu_sync()
        hCv = from_gpu(C_d)

        max_err = maximum(abs.(hCv .- hRv) ./ max.(abs.(hRv), 1e-12))
        ok = max_err < 1e-6
        @printf("VERIFY abstraction=julia_%s backend=%s N=%d max_rel_err=%.2e %s\n",
                mode, BACKEND, Nv, max_err, ok ? "PASS" : "FAIL")
        if !ok
            @error "[E2 verify] julia_$(mode) FAILED — aborting before timing."
            exit(1)
        end
        println(stderr, "[E2 verify] julia_$(mode) PASS — proceeding to timed measurement.")
    end

    run_dgemm(mode, N, warmup, reps, platform)
end

main()
