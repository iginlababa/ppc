#!/usr/bin/env julia
# kernel_spmv_julia.jl — E4 SpMV: Julia/CUDA.jl @cuda CSR kernel.
#
# E4 DESIGN DECISIONS
# [D3-Julia] @cuda kernel: one thread per row. CuArray{Int32} for row_ptr and
#   col_idx (0-indexed, C-style); CuArray{Float64} for values, x, y.
#   The kernel uses 0-indexed row numbering (threadIdx → 0-based row).
#   Access pattern to x is irregular (col_idx random) — same as all abstractions.
# [D7-Julia] Adaptive warmup: CV < 2% over last 10 timings.
#   Note: first few runs include @cuda JIT overhead — adaptive warmup absorbs this.

using CUDA
using Printf: @printf
using Statistics: median, mean, quantile, std
using Random: shuffle!, MersenneTwister

# ── Constants ──────────────────────────────────────────────────────────────────
const SPMV_BLOCK_SIZE   = 256
const SPMV_WARMUP_MIN   = 10
const SPMV_WARMUP_MAX   = 200
const SPMV_WARMUP_WIN   = 10
const SPMV_WARMUP_CV    = 2.0
const SPMV_REPS_DEFAULT = 30
const SPMV_SEED         = UInt64(42)

# ── SpMV CSR kernel: one thread per row ───────────────────────────────────────
# row_ptr, col_idx: Int32, 0-indexed C-style.
# values, x, y: Float64.
# row: 0-indexed.
function spmv_csr_kernel!(row_ptr, col_idx, values, x, y, nrows::Int32)
    row = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if row < nrows
        s = 0.0
        start = row_ptr[row + Int32(1)] + Int32(1)   # +1 to convert 0→1-indexed Julia array
        stop  = row_ptr[row + Int32(2)]               # exclusive end (0-indexed) → inclusive 1-indexed
        @inbounds for j in start:stop
            s += values[j] * x[col_idx[j] + Int32(1)]  # col_idx is 0-indexed
        end
        @inbounds y[row + Int32(1)] = s
    end
    return nothing
end

# ── Throughput ─────────────────────────────────────────────────────────────────
spmv_gflops(nnz::Int, time_s::Float64) = 2.0 * nnz / time_s / 1e9

# ── hw_state_verified ─────────────────────────────────────────────────────────
function compute_hw_state(vals::Vector{Float64})
    isempty(vals) && return Int[]
    med = median(vals)
    denom = abs(med) < 1e-12 ? 1.0 : abs(med)
    return [abs(v - med) / denom <= 0.15 ? 1 : 0 for v in vals]
end

# ── Adaptive warmup ───────────────────────────────────────────────────────────
function adaptive_warmup!(run_once!; warmup_min=SPMV_WARMUP_MIN,
                           warmup_max=SPMV_WARMUP_MAX, window_size=SPMV_WARMUP_WIN,
                           cv_ceil=SPMV_WARMUP_CV)
    window = Float64[]
    total  = 0
    while total < warmup_max
        t0 = time_ns()
        run_once!()
        t1 = time_ns()
        push!(window, (t1 - t0) / 1e6)
        length(window) > window_size && popfirst!(window)
        total += 1
        if total >= warmup_min && length(window) == window_size
            m  = mean(window)
            s  = std(window; corrected=false)
            cv = m > 0.0 ? 100.0 * s / m : 100.0
            cv < cv_ceil && break
        end
    end
    return total
end

# ── Matrix generators ──────────────────────────────────────────────────────────
# All generators return (row_ptr::Vector{Int32}, col_idx::Vector{Int32},
#                        values::Vector{Float64}, nrows::Int, nnz::Int)
# row_ptr is 0-indexed (C-style), length nrows+1.
# col_idx values are 0-indexed.

function generate_laplacian_2d(target_N::Int)
    Nx = max(2, round(Int, sqrt(target_N)))
    Ny = max(2, div(target_N + Nx - 1, Nx))
    N  = Nx * Ny

    row_ptr = zeros(Int32, N + 1)
    # Count degree per row
    for iy in 0:Ny-1, ix in 0:Nx-1
        row = iy * Nx + ix
        deg = 1
        ix > 0      && (deg += 1)
        ix < Nx - 1 && (deg += 1)
        iy > 0      && (deg += 1)
        iy < Ny - 1 && (deg += 1)
        row_ptr[row + 2] = deg
    end
    for i in 2:N+1
        row_ptr[i] += row_ptr[i-1]
    end
    nnz = row_ptr[N+1]
    col_idx = zeros(Int32, nnz)
    values  = zeros(Float64, nnz)

    pos = copy(row_ptr[1:N])  # current fill position per row (0-indexed)
    for iy in 0:Ny-1, ix in 0:Nx-1
        row = iy * Nx + ix
        entries = Tuple{Int32,Float64}[]
        push!(entries, (Int32(row), -4.0))
        ix > 0      && push!(entries, (Int32(row - 1),  1.0))
        ix < Nx - 1 && push!(entries, (Int32(row + 1),  1.0))
        iy > 0      && push!(entries, (Int32(row - Nx), 1.0))
        iy < Ny - 1 && push!(entries, (Int32(row + Nx), 1.0))
        sort!(entries, by=first)
        for (col, val) in entries
            k = pos[row + 1] + 1  # 1-indexed Julia array
            col_idx[k] = col
            values[k]  = val
            pos[row + 1] += 1
        end
    end
    return row_ptr, col_idx, values, N, Int(nnz)
end

function generate_random_sparse(N::Int, nnz_per_row::Int, seed::UInt64)
    nnz_per_row = min(nnz_per_row, N)
    nnz     = N * nnz_per_row
    row_ptr = Int32[i * nnz_per_row for i in 0:N]
    col_idx = zeros(Int32, nnz)
    values  = fill(Float64(1.0 / nnz_per_row), nnz)

    rng = MersenneTwister(seed)
    pool = collect(Int32, 0:N-1)
    for row in 0:N-1
        shuffle!(rng, pool)
        sorted_cols = sort(pool[1:nnz_per_row])
        for k in 1:nnz_per_row
            col_idx[row * nnz_per_row + k] = sorted_cols[k]
        end
    end
    return row_ptr, col_idx, values, N, nnz
end

function generate_power_law(N::Int, seed::UInt64)
    rng      = MersenneTwister(seed + 1)
    inv_a    = 1.0 / (2.5 - 1.0)   # 1/(gamma-1) = 2/3
    d_max    = min(div(N, 4), 500)

    degrees  = [max(1, min(d_max, floor(Int, (1.0 - rand(rng))^(-inv_a)))) for _ in 1:N]
    row_ptr  = Int32[0; cumsum(Int32.(degrees))]
    nnz      = Int(row_ptr[end])
    col_idx  = zeros(Int32, nnz)
    values   = ones(Float64, nnz)

    for row in 0:N-1
        deg   = degrees[row + 1]
        start = Int(row_ptr[row + 1])
        # Sample deg unique columns ≠ row
        cols = Set{Int32}()
        while length(cols) < deg
            c = rand(rng, Int32(0):Int32(N-1))
            c == row && continue
            push!(cols, c)
        end
        scols = sort(collect(cols))
        for k in 1:length(scols)
            col_idx[start + k] = scols[k]
        end
    end
    return row_ptr, col_idx, values, N, nnz
end

function build_matrix(mtype::String, N::Int)
    if mtype == "laplacian_2d"
        return generate_laplacian_2d(N)
    elseif mtype == "random_sparse"
        return generate_random_sparse(N, 5, SPMV_SEED)
    elseif mtype == "power_law"
        return generate_power_law(N, SPMV_SEED)
    else
        error("Unknown matrix type: $mtype")
    end
end

function make_x_vector(N::Int)
    rng = MersenneTwister(SPMV_SEED + 99)
    return Float64[0.1 + 1.0 * rand(rng) for _ in 1:N]
end

# ── Correctness check ──────────────────────────────────────────────────────────
function verify_spmv(mtype::String)::Bool
    rp, ci, val, nrows, nnz = generate_laplacian_2d(64)
    x   = make_x_vector(nrows)
    ref = zeros(Float64, nrows)
    # CPU reference
    for row in 0:nrows-1
        s = 0.0
        for j in (rp[row+1]+1):rp[row+2]  # 0-indexed rp, 1-indexed Julia arrays
            s += val[j] * x[ci[j] + 1]
        end
        ref[row + 1] = s
    end

    d_rp  = CuArray(rp)
    d_ci  = CuArray(ci)
    d_val = CuArray(val)
    d_x   = CuArray(x)
    d_y   = CUDA.zeros(Float64, nrows)

    grid = cld(nrows, SPMV_BLOCK_SIZE)
    @cuda threads=SPMV_BLOCK_SIZE blocks=grid spmv_csr_kernel!(
        d_rp, d_ci, d_val, d_x, d_y, Int32(nrows))
    CUDA.synchronize()
    result = Array(d_y)

    max_err = 0.0
    for i in 1:nrows
        denom = abs(ref[i]) < 1e-14 ? 1.0 : abs(ref[i])
        max_err = max(max_err, abs(result[i] - ref[i]) / denom)
    end
    ok = max_err < 1e-10
    @printf("VERIFY abstraction=julia matrix=laplacian_2d N=%d max_rel_err=%.2e %s\n",
            nrows, max_err, ok ? "PASS" : "FAIL")
    return ok
end

# ── Run experiment ─────────────────────────────────────────────────────────────
function run_spmv(mtype::String, N::Int, warmup_max::Int, reps::Int, platform::String)
    @printf("# abstraction=julia matrix=%s N_target=%d warmup_max=%d reps=%d platform=%s\n",
            mtype, N, warmup_max, reps, platform)

    rp, ci, val, nrows, nnz = build_matrix(mtype, N)
    x = make_x_vector(nrows)

    @printf("# abstraction=julia matrix=%s N=%d nnz=%d warmup_max=%d reps=%d platform=%s\n",
            mtype, nrows, nnz, warmup_max, reps, platform)
    flush(stdout)

    d_rp  = CuArray(rp)
    d_ci  = CuArray(ci)
    d_val = CuArray(val)
    d_x   = CuArray(x)
    d_y   = CUDA.zeros(Float64, nrows)

    grid  = cld(nrows, SPMV_BLOCK_SIZE)
    nrows32 = Int32(nrows)

    function run_once!()
        @cuda threads=SPMV_BLOCK_SIZE blocks=grid spmv_csr_kernel!(
            d_rp, d_ci, d_val, d_x, d_y, nrows32)
        CUDA.synchronize()
    end

    warmup_iters = adaptive_warmup!(run_once!; warmup_max=warmup_max)
    println(stderr, "[E4] julia: adaptive warmup done in $(warmup_iters) iterations")

    gflops_vec = Float64[]
    sizehint!(gflops_vec, reps)
    for r in 1:reps
        t0 = time_ns()
        run_once!()
        t1 = time_ns()
        time_ms = (t1 - t0) / 1e6
        gf = spmv_gflops(nnz, time_ms / 1000.0)
        push!(gflops_vec, gf)
        @printf("SPMV_RUN run=%d n=%d nnz=%d matrix=%s time_ms=%.6f throughput_gflops=%.6f\n",
                r, nrows, nnz, mtype, time_ms, gf)
        flush(stdout)
    end

    flags   = compute_hw_state(gflops_vec)
    for r in 1:reps
        @printf("SPMV_HW_STATE run=%d hw_state=%d\n", r, flags[r])
    end

    clean   = [gflops_vec[i] for i in 1:reps if flags[i] == 1]
    n_clean = length(clean)
    if n_clean > 0
        med = median(clean)
        q1  = quantile(clean, 0.25)
        q3  = quantile(clean, 0.75)
        @printf("SPMV_SUMMARY n=%d nnz=%d matrix=%s median_gflops=%.4f iqr_gflops=%.4f min_gflops=%.4f max_gflops=%.4f mean_gflops=%.4f n_clean=%d warmup_iters=%d\n",
                nrows, nnz, mtype, med, q3-q1, minimum(clean), maximum(clean),
                mean(clean), n_clean, warmup_iters)
    else
        @printf("SPMV_SUMMARY n=%d nnz=%d matrix=%s median_gflops=0.0 iqr_gflops=0.0 min_gflops=0.0 max_gflops=0.0 mean_gflops=0.0 n_clean=0 warmup_iters=%d\n",
                nrows, nnz, mtype, warmup_iters)
    end
    flush(stdout)
end

# ── Entry point ────────────────────────────────────────────────────────────────
function main()
    SPMV_N_SMALL  = 1024
    SPMV_N_MEDIUM = 8192
    SPMV_N_LARGE  = 32768

    N          = SPMV_N_LARGE
    matstr     = "laplacian_2d"
    warmup_max = SPMV_WARMUP_MAX
    reps       = SPMV_REPS_DEFAULT
    platform   = "unknown"
    do_verify  = false

    args = ARGS
    i = 1
    while i <= length(args)
        if     args[i] == "--n"        && i < length(args); N          = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--matrix"   && i < length(args); matstr     = args[i+1];             i += 2
        elseif args[i] == "--warmup"   && i < length(args); warmup_max = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--reps"     && i < length(args); reps       = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--platform" && i < length(args); platform   = args[i+1];             i += 2
        elseif args[i] == "--verify";                        do_verify  = true;                  i += 1
        else
            @error "Unknown argument: $(args[i])"
            exit(1)
        end
    end

    if do_verify
        ok = verify_spmv(matstr)
        if !ok
            @error "[E4 verify] julia FAILED — aborting before timing."
            exit(1)
        end
        println(stderr, "[E4 verify] julia PASS — proceeding to timed measurement.")
    end

    run_spmv(matstr, N, warmup_max, reps, platform)
end

main()
