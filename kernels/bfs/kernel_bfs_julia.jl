#!/usr/bin/env julia
# kernel_bfs_julia.jl — E6 BFS: Julia CUDA.jl scatter + CUDA.cumsum compact.
#
# ── E6 DESIGN DECISIONS ───────────────────────────────────────────────────────
# [D3-Julia] Scatter: custom CUDA kernel via CUDA.@cuda.
#   Compact: CUDA.cumsum on d_flags (exclusive prefix sum via cumsum-1 trick),
#            then gather via second CUDA kernel.
#   CUDA.synchronize() between scatter and compact, and after compact.
# [D5-Julia] GTEPS = n_edges / time_s / 1e9, stored as throughput_gflops.
# [D7-Julia] d_distances reset via CUDA.fill! before each rep (excluded from
#   timed region). d_flags reset via CUDA.fill! after each level.
# ─────────────────────────────────────────────────────────────────────────────

using CUDA
using Statistics
using Printf
using Random

# ── Constants ─────────────────────────────────────────────────────────────────
const BFS_N_SMALL    = 1024
const BFS_N_MEDIUM   = 16384
const BFS_N_LARGE    = 65536
const BFS_BLOCK_SIZE = 256
const BFS_WARMUP_MAX = 50
const BFS_CV_WINDOW  = 10
const BFS_CV_TARGET  = 2.0   # percent
const BFS_HW_THRESH  = 0.15
const BFS_SOURCE     = Int32(0)
const BFS_AVG_DEGREE = 10.0
const BFS_SEED       = 42

# ── Graph types ───────────────────────────────────────────────────────────────
struct CsrGraph
    row_ptr::Vector{Int32}   # n_vertices + 1
    col_idx::Vector{Int32}   # 2 * n_undirected_edges
    n_vertices::Int32
    n_edges::Int32           # undirected edge count
end

struct BfsResult
    distances::Vector{Int32}
    frontier_widths::Vector{Int32}
    n_levels::Int32
    max_frontier_width::Int32
    min_frontier_width::Int32
    n_visited::Int32
end

# ── Graph generators ──────────────────────────────────────────────────────────
function generate_erdos_renyi(n::Int, seed::Int = BFS_SEED)
    p = BFS_AVG_DEGREE / n
    rng = MersenneTwister(seed)
    adj = [Int32[] for _ in 1:n]
    for u in 1:n
        v = u  # start above u (0-indexed: v > u)
        while true
            skip = floor(Int, log(rand(rng)) / log(1.0 - p)) + 1
            v += skip
            v > n && break
            push!(adj[u], Int32(v - 1))   # 0-indexed
            push!(adj[v], Int32(u - 1))   # 0-indexed
        end
    end
    # Build CSR (0-indexed)
    row_ptr = zeros(Int32, n + 1)
    for u in 1:n
        row_ptr[u + 1] = row_ptr[u] + length(adj[u])
    end
    col_idx = zeros(Int32, row_ptr[n + 1])
    for u in 1:n
        sort!(adj[u])
        for (k, v) in enumerate(adj[u])
            col_idx[row_ptr[u] + k] = v
        end
    end
    m = div(length(col_idx), 2)
    CsrGraph(row_ptr, col_idx, Int32(n), Int32(m))
end

function generate_2d_grid(n::Int)
    side = round(Int, sqrt(n))
    @assert side * side == n "n must be a perfect square for 2d_grid"
    # vertex (r, c) -> r * side + c  (0-indexed)
    function neighbors(r, c)
        nb = Int32[]
        r > 0         && push!(nb, Int32((r-1)*side + c))
        r < side - 1  && push!(nb, Int32((r+1)*side + c))
        c > 0         && push!(nb, Int32(r*side + c - 1))
        c < side - 1  && push!(nb, Int32(r*side + c + 1))
        nb
    end
    row_ptr = zeros(Int32, n + 1)
    for r in 0:side-1, c in 0:side-1
        u = r * side + c
        row_ptr[u + 2] = length(neighbors(r, c))
    end
    for i in 2:n+1
        row_ptr[i] += row_ptr[i-1]
    end
    col_idx = zeros(Int32, row_ptr[n + 1])
    for r in 0:side-1, c in 0:side-1
        u = r * side + c
        nb = neighbors(r, c)
        for (k, v) in enumerate(nb)
            col_idx[row_ptr[u + 1] + k] = v
        end
    end
    m = div(length(col_idx), 2)
    CsrGraph(row_ptr, col_idx, Int32(n), Int32(m))
end

# ── CPU BFS reference ─────────────────────────────────────────────────────────
function bfs_cpu_ref(g::CsrGraph, source::Int32 = BFS_SOURCE)
    n = Int(g.n_vertices)
    distances = fill(Int32(-1), n)
    distances[source + 1] = Int32(0)  # 1-indexed
    frontier = [source]
    frontier_widths = Int32[]
    level = Int32(1)
    while !isempty(frontier)
        push!(frontier_widths, Int32(length(frontier)))
        next = Int32[]
        for u in frontier
            for j in (g.row_ptr[u+1]+1):g.row_ptr[u+2]  # 1-indexed
                v = g.col_idx[j]
                if distances[v + 1] == -1
                    distances[v + 1] = level
                    push!(next, v)
                end
            end
        end
        frontier = next
        level += Int32(1)
    end
    n_levels = Int32(level - 1)
    max_fw = isempty(frontier_widths) ? Int32(0) : maximum(frontier_widths)
    min_fw = isempty(frontier_widths) ? Int32(0) : minimum(frontier_widths)
    n_visited = Int32(count(d -> d >= 0, distances))
    BfsResult(distances, frontier_widths, n_levels, max_fw, min_fw, n_visited)
end

# ── CUDA kernels ──────────────────────────────────────────────────────────────
# Scatter: one thread per frontier vertex
function bfs_scatter_kernel!(d_frontier, frontier_size::Int32,
                              d_row_ptr, d_col_idx,
                              d_distances, d_flags, next_level::Int32)
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    tid > frontier_size && return nothing
    u     = d_frontier[tid]
    start = d_row_ptr[u + Int32(1)] + Int32(1)   # 1-indexed col_idx
    stop  = d_row_ptr[u + Int32(2)]
    for j in start:stop
        v = d_col_idx[j]
        old = CUDA.atomic_cas!(pointer(d_distances, v + Int32(1)), Int32(-1), next_level)
        if old == Int32(-1)
            d_flags[v + Int32(1)] = Int32(1)
        end
    end
    return nothing
end

# Gather: compact new frontier from prefix-sum scan
function bfs_gather_kernel!(d_flags, d_scan, d_next_frontier, n_vertices::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > n_vertices && return nothing
    if d_flags[i] != Int32(0)
        pos = d_scan[i]   # 1-based position from inclusive scan
        d_next_frontier[pos] = Int32(i - Int32(1))  # 0-indexed vertex
    end
    return nothing
end

# ── BFS driver ────────────────────────────────────────────────────────────────
function run_bfs_julia!(d_row_ptr, d_col_idx, n_vertices::Int32,
                         d_distances, d_frontier, d_next_frontier,
                         d_flags, d_scan)
    # Init frontier (CUDA.@allowscalar: single scalar write to CuArray from host)
    CUDA.@allowscalar d_frontier[1] = BFS_SOURCE
    frontier_size = Int32(1)

    t0 = time_ns()

    level = Int32(1)
    while frontier_size > Int32(0)
        blocks = cld(Int(frontier_size), BFS_BLOCK_SIZE)
        @cuda threads=BFS_BLOCK_SIZE blocks=blocks bfs_scatter_kernel!(
            d_frontier, frontier_size,
            d_row_ptr, d_col_idx,
            d_distances, d_flags, level)
        CUDA.synchronize()

        # Compact via inclusive cumsum then gather
        # d_scan = inclusive prefix sum of d_flags
        cumsum!(d_scan, d_flags)
        CUDA.synchronize()

        # New frontier size = last element of inclusive scan
        new_size = Int32(CUDA.@allowscalar d_scan[n_vertices])

        if new_size > Int32(0)
            blocks2 = cld(Int(n_vertices), BFS_BLOCK_SIZE)
            @cuda threads=BFS_BLOCK_SIZE blocks=blocks2 bfs_gather_kernel!(
                d_flags, d_scan, d_next_frontier, n_vertices)
            CUDA.synchronize()
        end

        # Reset flags
        CUDA.fill!(d_flags, Int32(0))
        CUDA.synchronize()

        # next → current
        if new_size > Int32(0)
            copyto!(d_frontier, 1, d_next_frontier, 1, Int(new_size))
        end

        frontier_size = new_size
        level += Int32(1)
    end

    CUDA.synchronize()
    t1 = time_ns()
    (t1 - t0) / 1e6  # ms
end

# ── Warmup ────────────────────────────────────────────────────────────────────
function adaptive_warmup!(run_fn)
    window = Float64[]
    for i in 1:BFS_WARMUP_MAX
        ms = run_fn()
        push!(window, ms)
        length(window) > BFS_CV_WINDOW && popfirst!(window)
        if length(window) == BFS_CV_WINDOW
            m = mean(window)
            cv = m > 0 ? 100.0 * std(window; corrected=false) / m : Inf
            cv < BFS_CV_TARGET && break
        end
    end
end

# ── hw_state ──────────────────────────────────────────────────────────────────
function hw_state(ms::Float64, history::Vector{Float64})
    isempty(history) && return 1
    med = median(history)
    med <= 0 && return 1
    abs(ms - med) / med <= BFS_HW_THRESH ? 1 : 0
end

# ── Correctness check ─────────────────────────────────────────────────────────
function verify_bfs(dist_gpu, dist_ref)
    n = length(dist_ref)
    length(dist_gpu) != n && return false
    for i in 1:n
        if Int(dist_gpu[i]) != Int(dist_ref[i])
            @warn "mismatch at v=$(i-1): gpu=$(dist_gpu[i]) ref=$(dist_ref[i])"
            return false
        end
    end
    true
end

# ── Main ──────────────────────────────────────────────────────────────────────
function main()
    args = Dict{String,String}()
    i = 1
    while i <= length(ARGS)
        if ARGS[i] in ("--graph", "--n", "--size", "--reps", "--platform") && i < length(ARGS)
            args[ARGS[i]] = ARGS[i+1]; i += 2
        elseif ARGS[i] == "--verify"
            args["--verify"] = "true"; i += 1
        else
            i += 1
        end
    end

    graph_type  = get(args, "--graph", "erdos_renyi")
    size_label  = get(args, "--size", "small")
    reps        = parse(Int, get(args, "--reps", "30"))
    do_verify   = get(args, "--verify", "false") == "true"

    n = if haskey(args, "--n")
        parse(Int, args["--n"])
    elseif size_label == "small"
        BFS_N_SMALL
    elseif size_label == "medium"
        BFS_N_MEDIUM
    else
        BFS_N_LARGE
    end
    size_label = (n == BFS_N_SMALL) ? "small" :
                 (n == BFS_N_MEDIUM) ? "medium" : "large"

    # Build graph
    g = if graph_type == "erdos_renyi"
        generate_erdos_renyi(n)
    elseif graph_type == "2d_grid"
        generate_2d_grid(n)
    else
        error("Unknown graph type: $graph_type")
    end

    ref = bfs_cpu_ref(g)

    # Print profile
    print("BFS_PROFILE graph=$(graph_type) size=$(size_label) n_levels=$(ref.n_levels) widths=")
    println(join(ref.frontier_widths, ","))
    flush(stdout)

    # Upload to device
    d_row_ptr       = CuArray(g.row_ptr)
    d_col_idx       = CuArray(g.col_idx)
    d_distances     = CUDA.fill(Int32(-1), Int32(n))
    d_frontier      = CUDA.zeros(Int32, n)
    d_next_frontier = CUDA.zeros(Int32, n)
    d_flags         = CUDA.zeros(Int32, n)
    d_scan          = CUDA.zeros(Int32, n)

    function reset_distances!()
        CUDA.fill!(d_distances, Int32(-1))
        CUDA.@allowscalar d_distances[BFS_SOURCE + 1] = Int32(0)
        CUDA.synchronize()
    end

    # Correctness check
    if do_verify
        reset_distances!()
        CUDA.fill!(d_flags, Int32(0))
        run_bfs_julia!(d_row_ptr, d_col_idx, Int32(n),
                       d_distances, d_frontier, d_next_frontier,
                       d_flags, d_scan)
        dist_gpu = Array(d_distances)
        if !verify_bfs(dist_gpu, ref.distances)
            @error "[bfs_julia] CORRECTNESS FAILED"
            exit(1)
        end
        @info "[bfs_julia] Correctness OK (n=$n n_levels=$(ref.n_levels))"
        reps == 0 && return
    end

    # Warmup
    adaptive_warmup!() do
        reset_distances!()
        CUDA.fill!(d_flags, Int32(0))
        run_bfs_julia!(d_row_ptr, d_col_idx, Int32(n),
                       d_distances, d_frontier, d_next_frontier,
                       d_flags, d_scan)
    end

    # Timed runs
    times = Float64[]
    for _ in 1:reps
        reset_distances!()
        CUDA.fill!(d_flags, Int32(0))
        ms = run_bfs_julia!(d_row_ptr, d_col_idx, Int32(n),
                             d_distances, d_frontier, d_next_frontier,
                             d_flags, d_scan)
        push!(times, ms)
    end

    # Report
    peak_ff  = n > 0 ? ref.max_frontier_width / n : 0.0
    n_edges  = Int(g.n_edges)
    for (r, ms) in enumerate(times)
        gteps  = n_edges / (ms / 1000.0) / 1e9
        hw_ok  = hw_state(ms, times[1:r-1])
        @printf("BFS_RUN run=%d n_vertices=%d n_edges=%d n_levels=%d max_fw=%d min_fw=%d peak_ff=%.6f graph=%s size=%s time_ms=%.4f throughput_gflops=%.6f\n",
                r, n, n_edges, ref.n_levels,
                ref.max_frontier_width, ref.min_frontier_width,
                peak_ff, graph_type, size_label, ms, gteps)
        @printf("BFS_HW_STATE state=%d\n", hw_ok)
        flush(stdout)
    end
end

main()
