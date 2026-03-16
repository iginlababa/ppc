#!/usr/bin/env julia
# kernel_sptrsv_julia.jl — E5 SpTRSV: Julia/CUDA.jl level-set forward substitution.
#
# E5 DESIGN DECISIONS
# [D3-Julia] @cuda kernel: one thread per row within each level. Outer loop over
#   levels in Julia on the host. CUDA.synchronize() between levels ensures all
#   x writes from level l are visible before level l+1's kernel reads them.
#   level_rows is a CuArray{Int32} of row indices (0-indexed C-style).
#   row_ptr, col_idx: Int32 (0-indexed); values, b, x: Float64.
#   The kernel accesses level_rows[level_start+tid] to get its row index, then
#   performs forward substitution over that row's CSR entries.
# [D7-Julia] Adaptive warmup: CV < 2% over last 10 timings. x is reset via
#   CUDA.fill! before each warmup iteration. Timed region excludes the fill.
# P001 note: Julia SpTRSV has n_levels kernel launches per solve (vs 1 for SpMV).
#   P001 Launch Overhead Dominance is expected to compound multiplicatively with
#   P007 Load Imbalance Amplification for irregular matrices.

using CUDA
using Printf: @printf
using Statistics: median, mean, quantile, std
using Random: MersenneTwister

# ── Constants ──────────────────────────────────────────────────────────────────
const SPTRSV_BLOCK_SIZE   = 256
const SPTRSV_WARMUP_MIN   = 10
const SPTRSV_WARMUP_MAX   = 200
const SPTRSV_WARMUP_WIN   = 10
const SPTRSV_WARMUP_CV    = 2.0
const SPTRSV_REPS_DEFAULT = 30
const SPTRSV_SEED         = UInt64(42)
const SPTRSV_N_SMALL      = 256
const SPTRSV_N_MEDIUM     = 2048
const SPTRSV_N_LARGE      = 8192

# ── SpTRSV level kernel: one thread per row within this level ──────────────────
# row_ptr, col_idx: Int32, 0-indexed C-style (Julia 1-indexed access with +1 offset).
# values, b, x: Float64.
# level_rows: Int32, 0-indexed row indices.
# level_start: 0-based index into level_rows array.
function sptrsv_level_kernel!(row_ptr, col_idx, values, b, x,
                               level_rows, level_start::Int32, level_size::Int32)
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if tid < level_size
        row   = level_rows[level_start + tid + Int32(1)]  # 0-indexed row
        s     = b[row + Int32(1)]
        diag  = 1.0
        jstart = row_ptr[row + Int32(1)] + Int32(1)   # 0-indexed ptr → 1-indexed
        jstop  = row_ptr[row + Int32(2)]               # exclusive end → inclusive 1-indexed
        @inbounds for j in jstart:jstop
            col = col_idx[j]
            if col == row
                diag = values[j]
            else
                # col < row: already solved in earlier level
                s -= values[j] * x[col + Int32(1)]
            end
        end
        @inbounds x[row + Int32(1)] = s / diag
    end
    return nothing
end

# ── Throughput ─────────────────────────────────────────────────────────────────
sptrsv_gflops(nnz::Int, time_s::Float64) = 2.0 * nnz / time_s / 1e9

# ── hw_state_verified ─────────────────────────────────────────────────────────
function compute_hw_state(vals::Vector{Float64})
    isempty(vals) && return Int[]
    med = median(vals)
    denom = abs(med) < 1e-12 ? 1.0 : abs(med)
    return [abs(v - med) / denom <= 0.15 ? 1 : 0 for v in vals]
end

# ── Adaptive warmup ───────────────────────────────────────────────────────────
function adaptive_warmup!(run_once!; warmup_min=SPTRSV_WARMUP_MIN,
                           warmup_max=SPTRSV_WARMUP_MAX, window_size=SPTRSV_WARMUP_WIN,
                           cv_ceil=SPTRSV_WARMUP_CV)
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
# Returns (row_ptr::Vector{Int32}, col_idx::Vector{Int32},
#          values::Vector{Float64}, nrows::Int, nnz::Int)
# row_ptr is 0-indexed (C-style), length nrows+1.
# col_idx values are 0-indexed; sorted ascending within each row.
# Diagonal is always included.

function generate_laplacian_lower(target_N::Int)
    Nx = max(2, round(Int, sqrt(target_N)))
    Ny = max(2, div(target_N + Nx - 1, Nx))
    N  = Nx * Ny

    row_ptr = zeros(Int32, N + 1)
    # Count: diagonal + lower off-diagonals (left: ix > 0, below: iy > 0)
    for iy in 0:Ny-1, ix in 0:Nx-1
        row = iy * Nx + ix
        deg = 1  # diagonal
        ix > 0 && (deg += 1)  # left: col = row - 1 < row
        iy > 0 && (deg += 1)  # below: col = row - Nx < row
        row_ptr[row + 2] = deg
    end
    for i in 2:N+1; row_ptr[i] += row_ptr[i-1]; end

    nnz     = Int(row_ptr[N + 1])
    col_idx = zeros(Int32, nnz)
    values  = zeros(Float64, nnz)

    pos = copy(row_ptr[1:N])  # current fill position per row (0-indexed)
    for iy in 0:Ny-1, ix in 0:Nx-1
        row = iy * Nx + ix
        entries = Tuple{Int32,Float64}[]
        ix > 0 && push!(entries, (Int32(row - 1),  1.0))
        iy > 0 && push!(entries, (Int32(row - Nx), 1.0))
        # Diagonal value: -(total degree of node in full Laplacian)
        total_deg = 1
        ix > 0      && (total_deg += 1)
        ix < Nx - 1 && (total_deg += 1)
        iy > 0      && (total_deg += 1)
        iy < Ny - 1 && (total_deg += 1)
        push!(entries, (Int32(row), -Float64(total_deg)))
        sort!(entries, by=first)
        for (col, val) in entries
            k = pos[row + 1] + 1
            col_idx[k] = col
            values[k]  = val
            pos[row + 1] += 1
        end
    end
    return row_ptr, col_idx, values, N, nnz
end

function generate_random_lower(N::Int, nnz_per_row::Int, seed::UInt64)
    rng = MersenneTwister(seed + 7)
    offdiag_val = 0.1

    # Row 0: diagonal only. Row i: min(nnz_per_row, i) off-diag + diagonal.
    degrees = Int32[i == 0 ? 1 : min(nnz_per_row, i) + 1 for i in 0:N-1]
    row_ptr = Int32[0; cumsum(degrees)]
    nnz     = Int(row_ptr[end])
    col_idx = zeros(Int32, nnz)
    values  = fill(offdiag_val, nnz)

    # Row 0: diagonal
    col_idx[1] = Int32(0)
    values[1]  = 1.0

    # Rows 1..N-1
    pool = collect(Int32, 0:N-1)
    for row in 1:N-1
        n_offdiag = Int(degrees[row + 1]) - 1
        start     = Int(row_ptr[row + 1]) + 1  # 1-indexed

        # Sample n_offdiag unique cols from [0, row-1]
        avail = pool[1:row]  # fresh copy
        for k in 1:n_offdiag
            j = k + rand(rng, 0:(row - k))
            avail[k], avail[j] = avail[j], avail[k]
        end
        sorted_cols = sort(avail[1:n_offdiag])
        for k in 1:n_offdiag
            col_idx[start + k - 1] = sorted_cols[k]
            values[start + k - 1]  = offdiag_val
        end
        # Diagonal last (largest col = row)
        col_idx[start + n_offdiag] = Int32(row)
        values[start + n_offdiag]  = 1.0
    end
    return row_ptr, col_idx, values, N, nnz
end

function build_matrix(mtype::String, N::Int)
    if mtype == "lower_triangular_laplacian"
        return generate_laplacian_lower(N)
    elseif mtype == "lower_triangular_random"
        return generate_random_lower(N, 5, SPTRSV_SEED)
    else
        error("Unknown matrix type: $mtype")
    end
end

function make_b_vector(N::Int)
    rng = MersenneTwister(SPTRSV_SEED + 77)
    return Float64[0.1 + 1.0 * rand(rng) for _ in 1:N]
end

# ── Level-set construction ─────────────────────────────────────────────────────
# Returns (level_ptr::Vector{Int32}, level_rows::Vector{Int32}, n_levels, max_lw, min_lw)
# level_ptr is 0-indexed (length n_levels+1).
# level_rows[level_ptr[l]+1 .. level_ptr[l+1]] = row indices (0-indexed) at level l.
function build_levels(row_ptr::Vector{Int32}, col_idx::Vector{Int32}, N::Int)
    level = zeros(Int32, N)
    for i in 1:N-1
        max_dep = Int32(-1)
        jstart = Int(row_ptr[i + 1]) + 1  # 1-indexed
        jstop  = Int(row_ptr[i + 2])
        for j in jstart:jstop
            col = Int(col_idx[j])
            if col < i
                max_dep = max(max_dep, level[col + 1])
            end
        end
        level[i + 1] = max_dep >= 0 ? max_dep + Int32(1) : Int32(0)
    end

    n_levels = Int(maximum(level)) + 1
    counts   = zeros(Int32, n_levels)
    for l in level; counts[l + 1] += 1; end

    # Build level_ptr (0-indexed, length n_levels+1)
    level_ptr  = zeros(Int32, n_levels + 1)
    for l in 1:n_levels; level_ptr[l + 1] = level_ptr[l] + counts[l]; end

    level_rows = zeros(Int32, N)
    pos        = copy(level_ptr[1:n_levels])  # current fill positions (0-indexed)
    for i in 0:N-1
        l = level[i + 1]
        k = pos[l + 1] + 1  # 1-indexed
        level_rows[k] = Int32(i)
        pos[l + 1] += 1
    end

    max_lw = Int(maximum(counts))
    min_lw = Int(minimum(counts[counts .> 0]))  # exclude zero-width levels if any
    return level_ptr, level_rows, n_levels, max_lw, min_lw
end

# ── Correctness check ──────────────────────────────────────────────────────────
function verify_sptrsv(mtype::String)::Bool
    rp, ci, val, nrows, nnz = generate_laplacian_lower(SPTRSV_N_SMALL)
    b   = make_b_vector(nrows)
    ref = zeros(Float64, nrows)
    # CPU reference forward substitution
    for i in 0:nrows-1
        s    = b[i + 1]
        diag = 1.0
        jstart = Int(rp[i + 1]) + 1
        jstop  = Int(rp[i + 2])
        for j in jstart:jstop
            col = Int(ci[j])
            if col == i
                diag = val[j]
            else
                s -= val[j] * ref[col + 1]
            end
        end
        ref[i + 1] = s / diag
    end

    lptr, lrows, n_levels, _, _ = build_levels(rp, ci, nrows)

    d_rp  = CuArray(rp)
    d_ci  = CuArray(ci)
    d_val = CuArray(val)
    d_b   = CuArray(b)
    d_x   = CUDA.zeros(Float64, nrows)
    d_lr  = CuArray(lrows)

    for l in 0:n_levels-1
        lstart  = lptr[l + 1]
        lsize   = lptr[l + 2] - lstart
        grid    = cld(lsize, SPTRSV_BLOCK_SIZE)
        @cuda threads=SPTRSV_BLOCK_SIZE blocks=grid sptrsv_level_kernel!(
            d_rp, d_ci, d_val, d_b, d_x, d_lr, Int32(lstart), Int32(lsize))
        CUDA.synchronize()
    end
    result = Array(d_x)

    max_err = 0.0
    for i in 1:nrows
        denom = abs(ref[i]) < 1e-14 ? 1.0 : abs(ref[i])
        max_err = max(max_err, abs(result[i] - ref[i]) / denom)
    end
    ok = max_err < 1e-10
    @printf("VERIFY abstraction=julia matrix=lower_triangular_laplacian N=%d max_rel_err=%.2e %s\n",
            nrows, max_err, ok ? "PASS" : "FAIL")
    return ok
end

# ── Run experiment ─────────────────────────────────────────────────────────────
function run_sptrsv(mtype::String, N::Int, warmup_max::Int, reps::Int, platform::String)
    rp, ci, val, nrows, nnz = build_matrix(mtype, N)
    lptr, lrows, n_levels, max_lw, min_lw = build_levels(rp, ci, nrows)
    b = make_b_vector(nrows)

    @printf("# abstraction=julia matrix=%s N=%d nnz=%d n_levels=%d max_lw=%d min_lw=%d warmup_max=%d reps=%d platform=%s\n",
            mtype, nrows, nnz, n_levels, max_lw, min_lw, warmup_max, reps, platform)
    flush(stdout)

    d_rp  = CuArray(rp)
    d_ci  = CuArray(ci)
    d_val = CuArray(val)
    d_b   = CuArray(b)
    d_x   = CUDA.zeros(Float64, nrows)
    d_lr  = CuArray(lrows)

    # Pre-compute grid sizes per level
    level_grids = [cld(Int(lptr[l + 2] - lptr[l + 1]), SPTRSV_BLOCK_SIZE) for l in 0:n_levels-1]
    level_starts = [Int32(lptr[l + 1]) for l in 0:n_levels-1]
    level_sizes  = [Int32(lptr[l + 2] - lptr[l + 1]) for l in 0:n_levels-1]

    function run_solve!()
        for l in 1:n_levels
            @cuda threads=SPTRSV_BLOCK_SIZE blocks=level_grids[l] sptrsv_level_kernel!(
                d_rp, d_ci, d_val, d_b, d_x, d_lr,
                level_starts[l], level_sizes[l])
            CUDA.synchronize()
        end
    end

    function run_with_reset!()
        CUDA.fill!(d_x, 0.0)
        run_solve!()
    end

    warmup_iters = adaptive_warmup!(run_with_reset!; warmup_max=warmup_max)
    println(stderr, "[E5] julia: adaptive warmup done in $(warmup_iters) iterations")

    gflops_vec = Float64[]
    sizehint!(gflops_vec, reps)
    mstr = mtype
    for r in 1:reps
        CUDA.fill!(d_x, 0.0)
        t0 = time_ns()
        run_solve!()
        t1 = time_ns()
        time_ms = (t1 - t0) / 1e6
        gf = sptrsv_gflops(nnz, time_ms / 1000.0)
        push!(gflops_vec, gf)
        @printf("SPTRSV_RUN run=%d n_rows=%d nnz=%d n_levels=%d max_lw=%d min_lw=%d matrix=%s time_ms=%.6f throughput_gflops=%.6f\n",
                r, nrows, nnz, n_levels, max_lw, min_lw, mstr, time_ms, gf)
        flush(stdout)
    end

    flags   = compute_hw_state(gflops_vec)
    for r in 1:reps
        @printf("SPTRSV_HW_STATE run=%d hw_state=%d\n", r, flags[r])
    end

    clean   = [gflops_vec[i] for i in 1:reps if flags[i] == 1]
    n_clean = length(clean)
    if n_clean > 0
        med = median(clean)
        q1  = quantile(clean, 0.25)
        q3  = quantile(clean, 0.75)
        iqr = q3 - q1
        mn  = mean(clean)
        cv  = length(clean) > 1 ? 100.0 * std(clean; corrected=false) / mn : 0.0
        @printf("SPTRSV_SUMMARY n_rows=%d nnz=%d n_levels=%d max_lw=%d min_lw=%d matrix=%s median_gflops=%.4f iqr_gflops=%.4f cv_pct=%.2f min_gflops=%.4f max_gflops=%.4f mean_gflops=%.4f n_clean=%d warmup_iters=%d\n",
                nrows, nnz, n_levels, max_lw, min_lw, mstr,
                med, iqr, cv, minimum(clean), maximum(clean), mn, n_clean, warmup_iters)
    else
        @printf("SPTRSV_SUMMARY n_rows=%d nnz=%d n_levels=%d max_lw=%d min_lw=%d matrix=%s median_gflops=0.0 iqr_gflops=0.0 cv_pct=0.0 min_gflops=0.0 max_gflops=0.0 mean_gflops=0.0 n_clean=0 warmup_iters=%d\n",
                nrows, nnz, n_levels, max_lw, min_lw, mstr, warmup_iters)
    end
    flush(stdout)
end

# ── Entry point ────────────────────────────────────────────────────────────────
function main()
    N          = SPTRSV_N_LARGE
    matstr     = "lower_triangular_laplacian"
    warmup_max = SPTRSV_WARMUP_MAX
    reps       = SPTRSV_REPS_DEFAULT
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
        ok = verify_sptrsv(matstr)
        if !ok
            @error "[E5 verify] julia FAILED — aborting before timing."
            exit(1)
        end
        println(stderr, "[E5 verify] julia PASS — proceeding to timed measurement.")
    end

    run_sptrsv(matstr, N, warmup_max, reps, platform)
end

main()
