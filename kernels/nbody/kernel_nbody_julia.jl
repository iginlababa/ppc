#!/usr/bin/env julia
# kernel_nbody_julia.jl — E7 N-Body Julia/CUDA.jl implementation
#
# Implements the notile LJ force kernel using CUDA.jl @cuda.
# Positions stored as CuArray{Float32,2} shape (4, N) — column-major friendly.
# Forces stored as CuArray{Float32,2} shape (3, N).
#
# One-sided accumulation, no Newton's 3rd law, no atomics.
# 20 FLOPs per pair, r_cut = 2.5σ.
# Warmup: E1 protocol (50 fixed iterations).
#
# Usage: julia --project=. kernels/nbody/kernel_nbody_julia.jl \
#            --n 4000 --kernel notile --reps 30 --platform nvidia_rtx5060_laptop

using CUDA
using Printf
using Statistics
using Random

# ── Constants ────────────────────────────────────────────────────────────────
const LJ_EPSILON  = 1.0f0
const R_CUT       = 2.5f0
const R_CUT2      = R_CUT * R_CUT
const FCC_A       = 1.6796f0
const MAX_NBRS    = 512
const BLOCK_SIZE  = 256
const FLOPS_PAIR  = 20
const N_WARMUP    = 50

# ── FCC lattice (CPU) ────────────────────────────────────────────────────────
function generate_fcc(m_cells::Int, a::Float32)
    bx = Float32[0.0, 0.5, 0.5, 0.0]
    by = Float32[0.0, 0.5, 0.0, 0.5]
    bz = Float32[0.0, 0.0, 0.5, 0.5]
    n = 4 * m_cells^3
    pos = zeros(Float32, 4, n)  # (x,y,z,0) × N column-major → pos[:,i]
    idx = 1
    for ix in 0:(m_cells-1), iy in 0:(m_cells-1), iz in 0:(m_cells-1), b in 1:4
        pos[1, idx] = (ix + bx[b]) * a
        pos[2, idx] = (iy + by[b]) * a
        pos[3, idx] = (iz + bz[b]) * a
        pos[4, idx] = 0.0f0
        idx += 1
    end
    return pos, Float32(m_cells) * a
end

# ── Minimum image convention ─────────────────────────────────────────────────
@inline function min_image(dx::Float32, L::Float32)::Float32
    dx >  0.5f0 * L && return dx - L
    dx < -0.5f0 * L && return dx + L
    return dx
end

# ── Cell-list neighbor construction (CPU) ────────────────────────────────────
function build_neighbor_list(pos::Matrix{Float32}, box_len::Float32)
    N      = size(pos, 2)
    nc     = max(1, floor(Int, box_len / R_CUT))
    cw     = box_len / nc
    cells  = [Int[] for _ in 1:(nc*nc*nc)]

    cell_of(x, y, z) = begin
        ix = min(floor(Int, x / cw), nc - 1)
        iy = min(floor(Int, y / cw), nc - 1)
        iz = min(floor(Int, z / cw), nc - 1)
        ix * nc * nc + iy * nc + iz + 1
    end

    for i in 1:N
        push!(cells[cell_of(pos[1,i], pos[2,i], pos[3,i])], i)
    end

    nbr_idx   = fill(-1, MAX_NBRS, N)  # (MAX_NBRS × N) → nbr_idx[:,i] for particle i
    nbr_count = zeros(Int32, N)

    for i in 1:N
        ix = min(floor(Int, pos[1,i] / cw), nc - 1)
        iy = min(floor(Int, pos[2,i] / cw), nc - 1)
        iz = min(floor(Int, pos[3,i] / cw), nc - 1)
        cnt = 0
        for ddx in -1:1, ddy in -1:1, ddz in -1:1
            jx = mod(ix + ddx, nc); jy = mod(iy + ddy, nc); jz = mod(iz + ddz, nc)
            ci = jx * nc * nc + jy * nc + jz + 1
            for j in cells[ci]
                j == i && continue
                dx = min_image(pos[1,j] - pos[1,i], box_len)
                dy = min_image(pos[2,j] - pos[2,i], box_len)
                dz = min_image(pos[3,j] - pos[3,i], box_len)
                r2 = dx*dx + dy*dy + dz*dz
                if r2 < R_CUT2 && cnt < MAX_NBRS
                    cnt += 1
                    nbr_idx[cnt, i] = j - 1  # 0-based for GPU kernel
                end
            end
        end
        nbr_count[i] = cnt
    end
    return nbr_idx, nbr_count
end

# ── CUDA kernel: notile LJ force ─────────────────────────────────────────────
# d_pos:      (4, N) CuArray{Float32,2}  — column-major: d_pos[:,i] = (x,y,z,0)
# d_neighbors:(MAX_NBRS, N) CuArray{Int32,2} — 0-based indices
# d_n_nbrs:   (N,) CuArray{Int32,1}
# d_forces:   (3, N) CuArray{Float32,2}  — output
function nbody_notile_kernel!(
    d_pos::CuDeviceMatrix{Float32},
    d_neighbors::CuDeviceMatrix{Int32},
    d_n_nbrs::CuDeviceVector{Int32},
    d_forces::CuDeviceMatrix{Float32},
    N::Int32)

    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > N && return

    pix = d_pos[1, i]; piy = d_pos[2, i]; piz = d_pos[3, i]
    fx  = 0.0f0;       fy  = 0.0f0;       fz  = 0.0f0
    nni = d_n_nbrs[i]

    for k in Int32(1):nni
        j = d_neighbors[k, i] + Int32(1)   # 1-based
        pjx = d_pos[1, j]; pjy = d_pos[2, j]; pjz = d_pos[3, j]
        dx  = pjx - pix;   dy  = pjy - piy;   dz  = pjz - piz
        r2  = dx*dx + dy*dy + dz*dz
        if r2 < R_CUT2 && r2 > 1.0f-10
            inv_r2  = 1.0f0 / r2
            inv_r6  = inv_r2 * inv_r2 * inv_r2
            inv_r12 = inv_r6 * inv_r6
            f_mag   = 24.0f0 * LJ_EPSILON * (2.0f0 * inv_r12 - inv_r6) * inv_r2
            fx += f_mag * dx
            fy += f_mag * dy
            fz += f_mag * dz
        end
    end
    d_forces[1, i] = fx
    d_forces[2, i] = fy
    d_forces[3, i] = fz
    return
end

# ── CPU reference for verification ──────────────────────────────────────────
function cpu_forces_ref(pos::Matrix{Float32}, nbr_idx::Matrix{Int},
                        nbr_count::Vector{Int32}, box_len::Float32)
    N = size(pos, 2)
    forces = zeros(Float32, 3, N)
    for i in 1:N
        fx = fy = fz = 0.0f0
        for k in 1:nbr_count[i]
            j  = nbr_idx[k, i] + 1  # 1-based
            dx = min_image(pos[1,j] - pos[1,i], box_len)
            dy = min_image(pos[2,j] - pos[2,i], box_len)
            dz = min_image(pos[3,j] - pos[3,i], box_len)
            r2 = dx*dx + dy*dy + dz*dz
            if r2 < R_CUT2 && r2 > 1e-10f
                inv_r2  = 1.0f0 / r2
                inv_r6  = inv_r2 * inv_r2 * inv_r2
                inv_r12 = inv_r6 * inv_r6
                f_mag   = 24.0f0 * LJ_EPSILON * (2.0f0 * inv_r12 - inv_r6) * inv_r2
                fx += f_mag * dx; fy += f_mag * dy; fz += f_mag * dz
            end
        end
        forces[1,i] = fx; forces[2,i] = fy; forces[3,i] = fz
    end
    return forces
end

# ── Driver ────────────────────────────────────────────────────────────────────
function run_nbody(n_atoms::Int, reps::Int, platform::String,
                   size_label::String, do_verify::Bool)

    m_cells = round(Int, cbrt(n_atoms / 4.0))
    pos_h, box_len = generate_fcc(m_cells, FCC_A)
    N = size(pos_h, 2)
    @printf(stderr, "[nbody-julia] N=%d  M=%d  box=%.4f  reps=%d\n",
            N, m_cells, box_len, reps)

    # Build neighbor list (CPU, 0-based indices for GPU)
    nbr_idx_h, nbr_count_h = build_neighbor_list(pos_h, box_len)
    total_nbrs = sum(nbr_count_h)
    mean_nbrs  = total_nbrs / N
    max_nbrs   = maximum(nbr_count_h)
    min_nbrs   = minimum(nbr_count_h)
    std_nbrs   = std(nbr_count_h; mean=mean_nbrs)
    @printf("NBODY_STATS n_atoms=%d n_nbrs_mean=%.2f n_nbrs_min=%d n_nbrs_max=%d n_nbrs_std=%.2f total_nbrs=%d\n",
            N, mean_nbrs, min_nbrs, max_nbrs, std_nbrs, total_nbrs)

    # Transfer to GPU
    d_pos       = CuArray(pos_h)
    d_neighbors = CuArray(Int32.(nbr_idx_h))
    d_n_nbrs    = CuArray(Int32.(nbr_count_h))
    d_forces    = CUDA.zeros(Float32, 3, N)

    blocks = cld(N, BLOCK_SIZE)
    N32    = Int32(N)

    # Correctness check
    if do_verify
        @printf(stderr, "[nbody-julia] Correctness check...\n")
        fill!(d_forces, 0.0f0)
        CUDA.@sync @cuda threads=BLOCK_SIZE blocks=blocks nbody_notile_kernel!(
            d_pos, d_neighbors, d_n_nbrs, d_forces, N32)
        gpu_forces = Array(d_forces)
        ref_forces = cpu_forces_ref(pos_h, nbr_idx_h, nbr_count_h, box_len)
        n_err = 0
        for i in 1:N
            ref_mag = sqrt(ref_forces[1,i]^2 + ref_forces[2,i]^2 + ref_forces[3,i]^2)
            err_mag = sqrt((ref_forces[1,i]-gpu_forces[1,i])^2 +
                           (ref_forces[2,i]-gpu_forces[2,i])^2 +
                           (ref_forces[3,i]-gpu_forces[3,i])^2)
            rel = ref_mag > 1e-10 ? err_mag / ref_mag : err_mag
            if rel > 1e-3
                n_err += 1
                n_err <= 3 && @printf(stderr,
                    "  FAIL i=%d cpu=(%.4f,%.4f,%.4f) gpu=(%.4f,%.4f,%.4f) rel=%.4f\n",
                    i, ref_forces[1,i], ref_forces[2,i], ref_forces[3,i],
                       gpu_forces[1,i], gpu_forces[2,i], gpu_forces[3,i], rel)
            end
        end
        @printf(stderr, "[nbody-julia] correctness: %s (%d errors)\n",
                n_err == 0 ? "PASS" : "FAIL", n_err)
        n_err > 0 && exit(1)
    end

    # VRAM report
    free_b, total_b = CUDA.memory_info()
    @printf(stderr, "[nbody-julia] VRAM: used=%.1f MB  free=%.1f MB  total=%.1f MB\n",
            (total_b - free_b) / 1048576.0, free_b / 1048576.0, total_b / 1048576.0)

    # Warmup: 50 fixed iterations (E1 protocol)
    for _ in 1:N_WARMUP
        @cuda threads=BLOCK_SIZE blocks=blocks nbody_notile_kernel!(
            d_pos, d_neighbors, d_n_nbrs, d_forces, N32)
    end
    CUDA.synchronize()

    # Timed runs
    kernel_label = "notile"
    total_flops  = Int64(total_nbrs) * FLOPS_PAIR
    times_ms     = Float64[]

    for r in 1:reps
        t0 = CUDA.@elapsed begin
            @cuda threads=BLOCK_SIZE blocks=blocks nbody_notile_kernel!(
                d_pos, d_neighbors, d_n_nbrs, d_forces, N32)
        end
        push!(times_ms, t0 * 1000.0)
    end

    hw_state = let
        μ     = mean(times_ms)
        mx    = maximum(abs.(times_ms .- μ) ./ μ)
        mx < 0.15 ? 1 : 0
    end

    for (r, t_ms) in enumerate(times_ms)
        gflops = t_ms > 0.0 ? total_flops / (t_ms * 1e6) : 0.0
        @printf("NBODY_RUN run=%d kernel=%s n_atoms=%d size=%s time_ms=%.6f throughput_gflops=%.6f\n",
                r, kernel_label, N, size_label, t_ms, gflops)
    end
    @printf("NBODY_HW_STATE state=%d\n", hw_state)
end

# ── Argument parsing ──────────────────────────────────────────────────────────
function parse_args()
    n_atoms    = 4000
    reps       = 30
    platform   = "nvidia_rtx5060_laptop"
    size_label = "small"
    do_verify  = false
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--n" || a == "--atoms"
            n_atoms = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--reps"
            reps = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--platform"
            platform = ARGS[i+1]; i += 2
        elseif a == "--size"
            size_label = ARGS[i+1]; i += 2
        elseif a == "--verify"
            do_verify = true; i += 1
        else
            i += 1
        end
    end
    return n_atoms, reps, platform, size_label, do_verify
end

n_atoms, reps, platform, size_label, do_verify = parse_args()
run_nbody(n_atoms, reps, platform, size_label, do_verify)
