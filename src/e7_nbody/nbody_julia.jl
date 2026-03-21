#!/usr/bin/env julia
# nbody_julia.jl — E7 N-Body, Julia GPU abstraction (CUDA.jl / AMDGPU.jl).
#
# Backend selection via JULIA_GPU_BACKEND env var (default: "cuda").
# Arrays:
#   d_pos       — GPU Array{Float32,2}(undef, 4, N)  column-major: pos[dim, i]
#   d_force     — GPU Array{Float32,2}(undef, 3, N)  force[dim, i]
#   d_neigh_ptr — GPU Array{Int32,1}(undef, N+1)     CSR row pointers (0-based values)
#   d_neigh_idx — GPU Array{Int32,1}(undef, total)   CSR neighbor indices (0-based)
#
# Indexing: CSR arrays hold C 0-based integers; Julia arrays are 1-indexed.
# Timing: time_ns() diff (portable — avoids CUDA.@elapsed / AMDGPU.@elapsed divergence).

using Statistics
using Random

const _GPU_BACKEND = get(ENV, "JULIA_GPU_BACKEND", "cuda")

if _GPU_BACKEND == "amdgpu"
    using AMDGPU
    @eval @inline _blockIdx()  = workgroupIdx()
    @eval @inline _threadIdx() = workitemIdx()
    @eval @inline _blockDim()  = workgroupDim()
    @eval _gpu_array(h)              = ROCArray(h)
    @eval _gpu_undef_f32_2d(d1, d2) = ROCArray{Float32,2}(undef, d1, d2)
    @eval _gpu_synchronize()         = AMDGPU.synchronize()
    @eval function _nbody_launch!(blocks, d_force, d_pos, d_neigh_ptr, d_neigh_idx, N, box_len)
        @roc groupsize=NBODY_BLOCK_SIZE gridsize=blocks nbody_kernel!(
            d_force, d_pos, d_neigh_ptr, d_neigh_idx, N, box_len)
        AMDGPU.synchronize()
    end
    @eval function _gpu_used_mb()
        try
            free  = AMDGPU.HIP.available_memory()
            total = AMDGPU.HIP.total_memory()
            return (total - free) / (1024.0 * 1024.0)
        catch
            return 0.0
        end
    end
else
    using CUDA
    @eval @inline _blockIdx()  = blockIdx()
    @eval @inline _threadIdx() = threadIdx()
    @eval @inline _blockDim()  = blockDim()
    @eval _gpu_array(h)              = CuArray(h)
    @eval _gpu_undef_f32_2d(d1, d2) = CuArray{Float32,2}(undef, d1, d2)
    @eval _gpu_synchronize()         = CUDA.synchronize()
    @eval function _nbody_launch!(blocks, d_force, d_pos, d_neigh_ptr, d_neigh_idx, N, box_len)
        @cuda threads=NBODY_BLOCK_SIZE blocks=blocks nbody_kernel!(
            d_force, d_pos, d_neigh_ptr, d_neigh_idx, N, box_len)
        CUDA.synchronize()
    end
    @eval function _gpu_used_mb()
        (CUDA.total_memory() - CUDA.available_memory()) / (1024.0 * 1024.0)
    end
end

# ── Constants ─────────────────────────────────────────────────────────────────
const NBODY_R_CUT_SQ   = 6.25f0    # (2.5σ)²
const NBODY_BLOCK_SIZE = 256
const NBODY_WARMUP     = 50
const NBODY_FLOPS_PAIR = 20
const NBODY_FCC_A      = 1.6796f0
const NBODY_JITTER     = 0.01f0

# ── Minimum image helper ───────────────────────────────────────────────────────
@inline function mi(dx::Float32, L::Float32)::Float32
    if dx >  0.5f0 * L  return dx - L  end
    if dx < -0.5f0 * L  return dx + L  end
    return dx
end

# ── GPU kernel ─────────────────────────────────────────────────────────────────
function nbody_kernel!(d_force, d_pos, d_neigh_ptr, d_neigh_idx, N::Int32, box_len::Float32)
    i = (_blockIdx().x - Int32(1)) * _blockDim().x + _threadIdx().x
    if i > N
        return nothing
    end

    px = d_pos[1, i];  py = d_pos[2, i];  pz = d_pos[3, i]
    fx = 0.0f0;  fy = 0.0f0;  fz = 0.0f0

    start = d_neigh_ptr[i]
    stop  = d_neigh_ptr[i + Int32(1)] - Int32(1)

    for k in start:stop
        j  = d_neigh_idx[k + Int32(1)] + Int32(1)   # 0-based → 1-based
        dx = mi(d_pos[1, j] - px, box_len)
        dy = mi(d_pos[2, j] - py, box_len)
        dz = mi(d_pos[3, j] - pz, box_len)
        r2 = dx*dx + dy*dy + dz*dz
        if r2 > 0.0f0 && r2 < NBODY_R_CUT_SQ
            r2inv = 1.0f0 / r2
            r6inv = r2inv * r2inv * r2inv
            fscal = 48.0f0 * r6inv * (r6inv - 0.5f0) * r2inv
            fx += fscal * dx;  fy += fscal * dy;  fz += fscal * dz
        end
    end

    d_force[1, i] = fx;  d_force[2, i] = fy;  d_force[3, i] = fz
    return nothing
end

# ── FCC lattice ────────────────────────────────────────────────────────────────
function make_fcc(m_cells::Int, a::Float32)
    bx = Float32[0, 0.5, 0.5, 0]
    by = Float32[0, 0.5, 0,   0.5]
    bz = Float32[0, 0,   0.5, 0.5]
    N  = 4 * m_cells^3
    pos = zeros(Float32, 4, N)
    rng = MersenneTwister(42)
    k = 0
    for ix in 0:m_cells-1, iy in 0:m_cells-1, iz in 0:m_cells-1, b in 1:4
        k += 1
        pos[1, k] = (ix + bx[b]) * a + (rand(rng, Float32) * 2 - 1) * NBODY_JITTER
        pos[2, k] = (iy + by[b]) * a + (rand(rng, Float32) * 2 - 1) * NBODY_JITTER
        pos[3, k] = (iz + bz[b]) * a + (rand(rng, Float32) * 2 - 1) * NBODY_JITTER
        pos[4, k] = 0.0f0
    end
    return pos
end

# ── CSR neighbor builder ───────────────────────────────────────────────────────
function build_csr(pos::Matrix{Float32}, box_len::Float32)
    N    = size(pos, 2)
    rcut = 2.5f0
    nc   = max(1, floor(Int, box_len / rcut))
    cw   = box_len / nc

    cells = [Int32[] for _ in 1:(nc*nc*nc)]
    cell_of(x,y,z) = begin
        ix = clamp(floor(Int32, x/cw), Int32(0), Int32(nc-1))
        iy = clamp(floor(Int32, y/cw), Int32(0), Int32(nc-1))
        iz = clamp(floor(Int32, z/cw), Int32(0), Int32(nc-1))
        (ix * nc + iy) * nc + iz + 1
    end
    for i in 1:N
        push!(cells[cell_of(pos[1,i], pos[2,i], pos[3,i])], Int32(i-1))
    end

    ptr = zeros(Int32, N + 1)
    idx_buf = Int32[]
    for i in 1:N
        ix = clamp(floor(Int32, pos[1,i]/cw), Int32(0), Int32(nc-1))
        iy = clamp(floor(Int32, pos[2,i]/cw), Int32(0), Int32(nc-1))
        iz = clamp(floor(Int32, pos[3,i]/cw), Int32(0), Int32(nc-1))
        for ddx in -1:1, ddy in -1:1, ddz in -1:1
            jx = mod(ix+ddx, nc);  jy = mod(iy+ddy, nc);  jz = mod(iz+ddz, nc)
            for j0 in cells[(jx*nc+jy)*nc+jz+1]
                j = j0 + 1
                if j == i  continue  end
                dx = mi(pos[1,j] - pos[1,i], box_len)
                dy = mi(pos[2,j] - pos[2,i], box_len)
                dz = mi(pos[3,j] - pos[3,i], box_len)
                if dx^2 + dy^2 + dz^2 < NBODY_R_CUT_SQ
                    push!(idx_buf, j0)
                end
            end
        end
        ptr[i+1] = Int32(length(idx_buf))
    end
    total  = Int32(length(idx_buf))
    mean_n = total / N
    max_n  = maximum(ptr[2:end] .- ptr[1:end-1])
    return ptr, idx_buf, total, max_n, mean_n
end

# ── hw_state check ─────────────────────────────────────────────────────────────
function hw_state_check(times_ms::Vector{Float64})
    length(times_ms) < 3 && return 1
    med = median(times_ms)
    med <= 0.0 && return 1
    all(abs(t - med) / med < 0.15 for t in times_ms) ? 1 : 0
end

# ── Argument parsing ───────────────────────────────────────────────────────────
function parse_args()
    args = Dict{String,String}(
        "size"     => "large",
        "reps"     => "30",
        "platform" => "amd_mi300x",
        "verify"   => "false",
    )
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if     a == "--size"     && i+1 <= length(ARGS)  args["size"]     = ARGS[i+1]; i += 2
        elseif a == "--reps"     && i+1 <= length(ARGS)  args["reps"]     = ARGS[i+1]; i += 2
        elseif a == "--platform" && i+1 <= length(ARGS)  args["platform"] = ARGS[i+1]; i += 2
        elseif a == "--verify"                            args["verify"]   = "true";    i += 1
        else                                              i += 1
        end
    end
    return args
end

# ── CPU reference (for --verify) ───────────────────────────────────────────────
function cpu_ref_forces(pos::Matrix{Float32}, ptr::Vector{Int32}, idx::Vector{Int32},
                        box_len::Float32)
    N = size(pos, 2)
    ref = zeros(Float32, 3, N)
    for i in 1:N
        px = pos[1,i]; py = pos[2,i]; pz = pos[3,i]
        fx = 0.0f0; fy = 0.0f0; fz = 0.0f0
        for k in ptr[i]:ptr[i+1]-Int32(1)
            j = idx[k+1] + Int32(1)
            dx = mi(pos[1,j]-px, box_len)
            dy = mi(pos[2,j]-py, box_len)
            dz = mi(pos[3,j]-pz, box_len)
            r2 = dx^2+dy^2+dz^2
            if r2 > 0.0f0 && r2 < NBODY_R_CUT_SQ
                r2inv = 1.0f0/r2; r6inv = r2inv^3
                fscal = 48.0f0 * r6inv * (r6inv - 0.5f0) * r2inv
                fx += fscal*dx; fy += fscal*dy; fz += fscal*dz
            end
        end
        ref[1,i]=fx; ref[2,i]=fy; ref[3,i]=fz
    end
    return ref
end

# ── Main ───────────────────────────────────────────────────────────────────────
function main()
    args      = parse_args()
    sz        = args["size"]
    reps      = parse(Int, args["reps"])
    platform  = args["platform"]
    do_verify = args["verify"] == "true"

    m_cells, N_exp = sz == "small" ? (10, 4000) : sz == "medium" ? (20, 32000) : (40, 256000)
    a       = NBODY_FCC_A
    box_len = Float32(m_cells) * a

    pos_h = make_fcc(m_cells, a)
    N     = size(pos_h, 2)
    @assert N == N_exp "Expected N=$N_exp, got N=$N"

    ptr_h, idx_h, total, max_n, mean_n = build_csr(pos_h, box_len)
    @assert max_n < 512 "max_n=$max_n exceeds cap"

    println("NBODY_META n=$N size=$sz n_nbrs_total=$total max_nbrs_per_atom=$max_n mean_nbrs=$(round(mean_n, digits=2))")

    # Upload to GPU
    d_pos       = _gpu_array(pos_h)
    d_force     = _gpu_undef_f32_2d(3, N)
    d_neigh_ptr = _gpu_array(ptr_h)
    d_neigh_idx = _gpu_array(idx_h)
    _gpu_synchronize()

    println("NBODY_VRAM used_mb=$(round(_gpu_used_mb(), digits=1))")

    N32    = Int32(N)
    blocks = cld(N, NBODY_BLOCK_SIZE)

    run_kernel!() = _nbody_launch!(blocks, d_force, d_pos, d_neigh_ptr, d_neigh_idx, N32, box_len)

    # Warmup
    for _ in 1:NBODY_WARMUP
        run_kernel!()
    end

    # Timed runs — time_ns() is portable across CUDA.jl and AMDGPU.jl
    times_ms  = Vector{Float64}(undef, reps)
    flops     = Float64(total) * NBODY_FLOPS_PAIR
    for rep in 1:reps
        t0 = time_ns()
        run_kernel!()
        t1 = time_ns()
        times_ms[rep] = (t1 - t0) / 1e6
        gflops = (times_ms[rep] > 0) ? flops / (times_ms[rep] * 1e6) : 0.0
        println("NBODY_RUN run=$rep kernel=notile size=$sz n=$N " *
                "actual_flops=$(round(flops, digits=0)) " *
                "time_ms=$(round(times_ms[rep], digits=6)) " *
                "throughput_gflops=$(round(gflops, digits=6))")
    end

    hw = hw_state_check(times_ms)
    println("NBODY_HW_STATE state=$hw")

    if do_verify
        gpu_f = Array(d_force)
        ref_f = cpu_ref_forces(pos_h, ptr_h, idx_h, box_len)
        max_rel = 0.0f0
        for i in 1:N
            err = sqrt((ref_f[1,i]-gpu_f[1,i])^2 +
                       (ref_f[2,i]-gpu_f[2,i])^2 +
                       (ref_f[3,i]-gpu_f[3,i])^2)
            mag = sqrt(ref_f[1,i]^2 + ref_f[2,i]^2 + ref_f[3,i]^2)
            rel = (mag > 1.0f-6) ? err/mag : err
            max_rel = max(max_rel, rel)
        end
        status = (max_rel < 1.0f-3) ? "PASS" : "FAIL"
        println("NBODY_VERIFY max_rel_err=$(round(max_rel, digits=6)) $status")
    end
end

main()
