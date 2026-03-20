// kernel_sptrsv_hip.cpp — E5 SpTRSV: native HIP level-set forward substitution (AMD).
//
// Mechanical port of kernel_sptrsv_cuda.cu.
// Design decisions [D3-CUDA] through [D7-CUDA] carry over unchanged:
//   one kernel launch per level, one thread per row within each level,
//   hipDeviceSynchronize() between levels (mandatory correctness barrier).
//   __threadfence() belt-and-suspenders, same as CUDA.
// Output: abstraction=native platform=amd_mi300x (or as passed via --platform).

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "sptrsv_common.h"

// ── HIP error helper ──────────────────────────────────────────────────────────
#define HIP_CHECK(call)                                                         \
    do {                                                                        \
        hipError_t _e = (call);                                                 \
        if (_e != hipSuccess) {                                                 \
            std::fprintf(stderr, "HIP error %s:%d: %s\n",                      \
                         __FILE__, __LINE__, hipGetErrorString(_e));            \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// ── SpTRSV level kernel: one thread per row within this level ─────────────────
// Identical body to CUDA version — HIP supports same __global__, blockIdx,
// threadIdx, blockDim, and __threadfence() syntax.
__global__ void sptrsv_level_kernel(
    const int*    __restrict__ row_ptr,
    const int*    __restrict__ col_idx,
    const double* __restrict__ values,
    const double* __restrict__ b,
    double*                    x,
    const int*    __restrict__ level_rows,
    int level_start,
    int level_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= level_size) return;

    int row = level_rows[level_start + tid];
    double sum  = b[row];
    double diag = 1.0;
    const int start = row_ptr[row];
    const int end   = row_ptr[row + 1];
    for (int j = start; j < end; j++) {
        int col = col_idx[j];
        if (col == row) {
            diag = values[j];
        } else {
            sum -= values[j] * x[col];
        }
    }
    x[row] = sum / diag;
    __threadfence();
}

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  --n <N>            Number of rows (default: %d)\n"
        "  --matrix <type>    lower_triangular_laplacian|lower_triangular_random (default: lower_triangular_laplacian)\n"
        "  --warmup <W>       Max adaptive warmup iterations (default: %d)\n"
        "  --reps <R>         Timed iterations (default: %d)\n"
        "  --platform <P>     Platform tag (default: unknown)\n"
        "  --verify           Correctness check before timing\n",
        prog, SPTRSV_N_LARGE, SPTRSV_WARMUP_MAX, SPTRSV_REPS_DEFAULT);
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    int         N        = SPTRSV_N_LARGE;
    std::string matstr   = "lower_triangular_laplacian";
    int         warmup   = SPTRSV_WARMUP_MAX;
    int         reps     = SPTRSV_REPS_DEFAULT;
    std::string platform = "unknown";
    bool        verify   = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--n"        && i+1 < argc) { N        = std::stoi(argv[++i]); }
        else if (a == "--matrix"   && i+1 < argc) { matstr   = argv[++i]; }
        else if (a == "--warmup"   && i+1 < argc) { warmup   = std::stoi(argv[++i]); }
        else if (a == "--reps"     && i+1 < argc) { reps     = std::stoi(argv[++i]); }
        else if (a == "--platform" && i+1 < argc) { platform = argv[++i]; }
        else if (a == "--verify")                  { verify   = true; }
        else { print_usage(argv[0]); return 1; }
    }

    SptrsMatType mtype = parse_matrix_type(matstr);

    // ── Correctness check ─────────────────────────────────────────────────────
    if (verify) {
        SptrsCSR vcsr = generate_laplacian_lower(SPTRSV_N_SMALL);
        SptrsLevels vls = build_levels(vcsr);
        std::vector<double> vb  = make_b_vector(vcsr.nrows);
        std::vector<double> ref(vcsr.nrows, 0.0);
        sptrsv_cpu_ref(vcsr, vb.data(), ref.data());

        int*    d_rp; int*    d_ci;  double* d_val;
        double* d_b;  double* d_x;   int*    d_lr;
        HIP_CHECK(hipMalloc(&d_rp,  (vcsr.nrows + 1) * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_ci,  vcsr.nnz * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_val, vcsr.nnz * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_b,   vcsr.nrows * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_x,   vcsr.nrows * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_lr,  vcsr.nrows * sizeof(int)));
        HIP_CHECK(hipMemcpy(d_rp,  vcsr.row_ptr.data(),   (vcsr.nrows+1)*sizeof(int),   hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_ci,  vcsr.col_idx.data(),   vcsr.nnz*sizeof(int),          hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_val, vcsr.values.data(),    vcsr.nnz*sizeof(double),       hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_b,   vb.data(),             vcsr.nrows*sizeof(double),     hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_lr,  vls.level_rows.data(), vcsr.nrows*sizeof(int),        hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_x, 0, vcsr.nrows * sizeof(double)));

        for (int l = 0; l < vls.n_levels; l++) {
            int lstart = vls.level_ptr[l];
            int lsize  = vls.level_ptr[l + 1] - lstart;
            int grid   = (lsize + SPTRSV_BLOCK_SIZE - 1) / SPTRSV_BLOCK_SIZE;
            sptrsv_level_kernel<<<grid, SPTRSV_BLOCK_SIZE>>>(
                d_rp, d_ci, d_val, d_b, d_x, d_lr, lstart, lsize);
            HIP_CHECK(hipDeviceSynchronize());
        }

        std::vector<double> res(vcsr.nrows);
        HIP_CHECK(hipMemcpy(res.data(), d_x, vcsr.nrows*sizeof(double), hipMemcpyDeviceToHost));
        hipFree(d_rp); hipFree(d_ci); hipFree(d_val);
        hipFree(d_b);  hipFree(d_x);  hipFree(d_lr);

        double max_err = 0.0;
        bool ok = sptrsv_verify(res.data(), ref.data(), vcsr.nrows, SPTRSV_CORRECT_TOL, &max_err);
        std::printf("VERIFY abstraction=native matrix=lower_triangular_laplacian N=%d max_rel_err=%.2e %s\n",
                    vcsr.nrows, max_err, ok ? "PASS" : "FAIL");
        if (!ok) {
            std::fprintf(stderr, "[E5 verify] native (HIP) FAILED — aborting.\n");
            return 1;
        }
        std::fprintf(stderr, "[E5 verify] native (HIP) PASS — proceeding to timed measurement.\n");
    }

    // ── Build matrix and level sets ───────────────────────────────────────────
    SptrsCSR csr = build_matrix(mtype, N);
    SptrsLevels ls = build_levels(csr);
    int  nrows = csr.nrows;
    long nnz   = csr.nnz;
    std::vector<double> b = make_b_vector(nrows);

    std::printf(
        "# abstraction=native matrix=%s N=%d nnz=%ld n_levels=%d"
        " max_lw=%d min_lw=%d warmup_max=%d reps=%d platform=%s\n",
        matrix_type_str(mtype), nrows, nnz, ls.n_levels,
        ls.max_lw, ls.min_lw, warmup, reps, platform.c_str());

    // ── Allocate device memory ────────────────────────────────────────────────
    int*    d_rp; int*    d_ci;  double* d_val;
    double* d_b;  double* d_x;   int*    d_lr;
    if (hipMalloc(&d_rp,  (nrows + 1) * sizeof(int))  != hipSuccess ||
        hipMalloc(&d_ci,  nnz * sizeof(int))            != hipSuccess ||
        hipMalloc(&d_val, nnz * sizeof(double))         != hipSuccess ||
        hipMalloc(&d_b,   nrows * sizeof(double))       != hipSuccess ||
        hipMalloc(&d_x,   nrows * sizeof(double))       != hipSuccess ||
        hipMalloc(&d_lr,  nrows * sizeof(int))          != hipSuccess) {
        std::fprintf(stderr, "hipMalloc failed for N=%d nnz=%ld\n", nrows, nnz);
        return 1;
    }
    HIP_CHECK(hipMemcpy(d_rp,  csr.row_ptr.data(),   (nrows+1)*sizeof(int),   hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_ci,  csr.col_idx.data(),   nnz*sizeof(int),          hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_val, csr.values.data(),    nnz*sizeof(double),       hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b,   b.data(),              nrows*sizeof(double),     hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_lr,  ls.level_rows.data(),  nrows*sizeof(int),        hipMemcpyHostToDevice));

    const char* mstr = matrix_type_str(mtype);

    // Pre-compute grid sizes per level
    std::vector<int> level_grids(ls.n_levels);
    for (int l = 0; l < ls.n_levels; l++) {
        int lsize = ls.level_ptr[l + 1] - ls.level_ptr[l];
        level_grids[l] = (lsize + SPTRSV_BLOCK_SIZE - 1) / SPTRSV_BLOCK_SIZE;
    }

    auto run_solve = [&]() {
        for (int l = 0; l < ls.n_levels; l++) {
            int lstart = ls.level_ptr[l];
            int lsize  = ls.level_ptr[l + 1] - lstart;
            sptrsv_level_kernel<<<level_grids[l], SPTRSV_BLOCK_SIZE>>>(
                d_rp, d_ci, d_val, d_b, d_x, d_lr, lstart, lsize);
            HIP_CHECK(hipDeviceSynchronize());
        }
    };

    auto run_with_reset = [&]() {
        HIP_CHECK(hipMemset(d_x, 0, nrows * sizeof(double)));
        run_solve();
    };

    int warmup_iters = sptrsv_adaptive_warmup(run_with_reset, SPTRSV_WARMUP_MIN, warmup);
    std::fprintf(stderr, "[E5] native (HIP): adaptive warmup done in %d iterations\n", warmup_iters);

    // ── Timed runs ────────────────────────────────────────────────────────────
    std::vector<double> gflops_vec;
    gflops_vec.reserve(reps);
    for (int r = 1; r <= reps; r++) {
        HIP_CHECK(hipMemset(d_x, 0, nrows * sizeof(double)));
        auto t0 = std::chrono::high_resolution_clock::now();
        run_solve();
        auto t1 = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double gf = sptrsv_gflops(nnz, time_ms / 1000.0);
        gflops_vec.push_back(gf);
        sptrsv_print_run(r, nrows, nnz, ls.n_levels, ls.max_lw, ls.min_lw,
                         mstr, time_ms, gf);
    }

    auto flags = sptrsv_compute_hw_state(gflops_vec);
    for (int r = 0; r < reps; r++)
        sptrsv_print_hw_state(r + 1, flags[r]);

    SptrsStats stats = sptrsv_compute_stats(gflops_vec, flags);
    sptrsv_print_summary(nrows, nnz, ls.n_levels, ls.max_lw, ls.min_lw,
                         mstr, stats, warmup_iters);

    if (stats.n_clean > 0) {
        double ai = sptrsv_ai(nrows, nnz);
        std::printf("SPTRSV_ROOFLINE n_rows=%d nnz=%ld matrix=%s n_levels=%d"
                    " median_gflops=%.4f ai=%.4f\n",
                    nrows, nnz, mstr, ls.n_levels, stats.median_gflops, ai);
        std::fflush(stdout);
    }

    hipFree(d_rp); hipFree(d_ci); hipFree(d_val);
    hipFree(d_b);  hipFree(d_x);  hipFree(d_lr);
    return 0;
}
