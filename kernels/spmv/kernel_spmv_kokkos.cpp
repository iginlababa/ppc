// kernel_spmv_kokkos.cpp — E4 SpMV: Kokkos RangePolicy over rows.
//
// ── E4 DESIGN DECISIONS ───────────────────────────────────────────────────────
// [D3-Kokkos] Kokkos::parallel_for with RangePolicy over rows. Each work item
//   corresponds to one row. The lambda captures device Views for row_ptr,
//   col_idx, values, x, y — same logical structure as native CUDA.
//   Unlike E3 (MDRangePolicy), here RangePolicy is the natural and idiomatic
//   Kokkos mapping for 1D SpMV loops. This is a fair, non-hobbled Kokkos baseline.
// ─────────────────────────────────────────────────────────────────────────────

#include <Kokkos_Core.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "spmv_common.h"

using ExecSpace  = Kokkos::DefaultExecutionSpace;
using MemSpace   = Kokkos::DefaultExecutionSpace::memory_space;
using ViewInt    = Kokkos::View<int*,    MemSpace>;
using ViewDouble = Kokkos::View<double*, MemSpace>;

void run_spmv_kokkos(const ViewInt& row_ptr, const ViewInt& col_idx,
                      const ViewDouble& values, const ViewDouble& x,
                      ViewDouble& y, int nrows) {
    Kokkos::parallel_for(
        "spmv_kokkos",
        Kokkos::RangePolicy<ExecSpace>(0, nrows),
        KOKKOS_LAMBDA(int row) {
            double sum = 0.0;
            const int start = row_ptr(row);
            const int end   = row_ptr(row + 1);
            for (int j = start; j < end; j++)
                sum += values(j) * x(col_idx(j));
            y(row) = sum;
        });
    Kokkos::fence();
}

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  --n <N>            Number of rows (default: %d)\n"
        "  --matrix <type>    Matrix type: laplacian_2d|random_sparse|power_law (default: laplacian_2d)\n"
        "  --warmup <W>       Max adaptive warmup iterations (default: %d)\n"
        "  --reps <R>         Timed iterations (default: %d)\n"
        "  --platform <P>     Platform tag (default: unknown)\n"
        "  --verify           Correctness check before timing\n",
        prog, SPMV_N_LARGE, SPMV_WARMUP_MAX, SPMV_REPS_DEFAULT);
}

int main(int argc, char** argv) {
    int         N        = SPMV_N_LARGE;
    std::string matstr   = "laplacian_2d";
    int         warmup   = SPMV_WARMUP_MAX;
    int         reps     = SPMV_REPS_DEFAULT;
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

    SpmvMatType mtype = parse_matrix_type(matstr);

    Kokkos::initialize(argc, argv);
    {
        // ── Correctness check ─────────────────────────────────────────────────
        if (verify) {
            SpmvCSR vcsr = generate_laplacian_2d(64);
            std::vector<double> vx  = make_x_vector(vcsr.nrows);
            std::vector<double> ref(vcsr.nrows, 0.0);
            spmv_cpu_ref(vcsr, vx.data(), ref.data());

            ViewInt    v_rp ("vrp",  vcsr.nrows + 1);
            ViewInt    v_ci ("vci",  vcsr.nnz);
            ViewDouble v_val("vval", vcsr.nnz);
            ViewDouble v_x  ("vx",   vcsr.nrows);
            ViewDouble v_y  ("vy",   vcsr.nrows);
            auto h_rp  = Kokkos::create_mirror_view(v_rp);
            auto h_ci  = Kokkos::create_mirror_view(v_ci);
            auto h_val = Kokkos::create_mirror_view(v_val);
            auto h_x   = Kokkos::create_mirror_view(v_x);
            for (int i = 0; i <= vcsr.nrows; i++) h_rp(i)  = vcsr.row_ptr[i];
            for (long j = 0; j < vcsr.nnz;    j++) { h_ci(j) = vcsr.col_idx[j]; h_val(j) = vcsr.values[j]; }
            for (int i = 0; i < vcsr.nrows;  i++) h_x(i)  = vx[i];
            Kokkos::deep_copy(v_rp,  h_rp);
            Kokkos::deep_copy(v_ci,  h_ci);
            Kokkos::deep_copy(v_val, h_val);
            Kokkos::deep_copy(v_x,   h_x);
            Kokkos::deep_copy(v_y,   0.0);

            run_spmv_kokkos(v_rp, v_ci, v_val, v_x, v_y, vcsr.nrows);

            auto h_y = Kokkos::create_mirror_view(v_y);
            Kokkos::deep_copy(h_y, v_y);
            std::vector<double> res(vcsr.nrows);
            for (int i = 0; i < vcsr.nrows; i++) res[i] = h_y(i);

            double max_err = 0.0;
            bool ok = spmv_verify(res.data(), ref.data(), vcsr.nrows, SPMV_CORRECT_TOL, &max_err);
            std::printf("VERIFY abstraction=kokkos matrix=laplacian_2d N=%d max_rel_err=%.2e %s\n",
                        vcsr.nrows, max_err, ok ? "PASS" : "FAIL");
            if (!ok) {
                std::fprintf(stderr, "[E4 verify] kokkos FAILED — aborting.\n");
                Kokkos::finalize(); return 1;
            }
            std::fprintf(stderr, "[E4 verify] kokkos PASS — proceeding to timed measurement.\n");
        }

        // ── Build matrix and allocate views ───────────────────────────────────
        SpmvCSR csr = build_matrix(mtype, N);
        int  nrows  = csr.nrows;
        long nnz    = csr.nnz;
        std::vector<double> x = make_x_vector(nrows);

        std::printf("# abstraction=kokkos matrix=%s N=%d nnz=%ld warmup_max=%d reps=%d platform=%s\n",
                    matrix_type_str(mtype), nrows, nnz, warmup, reps, platform.c_str());

        ViewInt    v_rp ("rp",  nrows + 1);
        ViewInt    v_ci ("ci",  nnz);
        ViewDouble v_val("val", nnz);
        ViewDouble v_x  ("x",   nrows);
        ViewDouble v_y  ("y",   nrows);

        {
            auto h_rp  = Kokkos::create_mirror_view(v_rp);
            auto h_ci  = Kokkos::create_mirror_view(v_ci);
            auto h_val = Kokkos::create_mirror_view(v_val);
            auto h_x   = Kokkos::create_mirror_view(v_x);
            for (int i = 0; i <= nrows; i++) h_rp(i)  = csr.row_ptr[i];
            for (long j = 0; j < nnz;    j++) { h_ci(j) = csr.col_idx[j]; h_val(j) = csr.values[j]; }
            for (int i = 0; i < nrows;  i++) h_x(i)  = x[i];
            Kokkos::deep_copy(v_rp,  h_rp);
            Kokkos::deep_copy(v_ci,  h_ci);
            Kokkos::deep_copy(v_val, h_val);
            Kokkos::deep_copy(v_x,   h_x);
            Kokkos::deep_copy(v_y,   0.0);
        }

        const char* mstr = matrix_type_str(mtype);

        auto run_once = [&]() {
            run_spmv_kokkos(v_rp, v_ci, v_val, v_x, v_y, nrows);
        };

        // ── Adaptive warmup ───────────────────────────────────────────────────
        int warmup_iters = spmv_adaptive_warmup(run_once, SPMV_WARMUP_MIN, warmup);
        std::fprintf(stderr, "[E4] kokkos: adaptive warmup done in %d iterations\n", warmup_iters);

        // ── Timed runs ────────────────────────────────────────────────────────
        std::vector<double> gflops_vec;
        gflops_vec.reserve(reps);
        for (int r = 1; r <= reps; r++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            run_once();
            auto t1 = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double gf = spmv_gflops(nnz, time_ms / 1000.0);
            gflops_vec.push_back(gf);
            spmv_print_run(r, nrows, nnz, mstr, time_ms, gf);
        }

        auto flags = spmv_compute_hw_state(gflops_vec);
        for (int r = 0; r < reps; r++)
            spmv_print_hw_state(r + 1, flags[r]);

        SpmvStats stats = spmv_compute_stats(gflops_vec, flags);
        spmv_print_summary(nrows, nnz, mstr, stats, warmup_iters);
    }
    Kokkos::finalize();
    return 0;
}
