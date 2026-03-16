// kernel_sptrsv_kokkos.cpp — E5 SpTRSV: Kokkos level-set forward substitution.
//
// ── E5 DESIGN DECISIONS ───────────────────────────────────────────────────────
// [D3-Kokkos] Outer loop over levels on host. Within each level:
//   Kokkos::parallel_for with RangePolicy over the level's row count.
//   Kokkos::fence() between levels ensures all writes to x from level l are
//   visible before level l+1 reads them.
//   This is the natural Kokkos pattern for level-set algorithms: the host loop
//   sequences levels, parallel_for expresses intra-level parallelism.
//   Unlike E3 (MDRangePolicy overhead), RangePolicy is correct and idiomatic here.
// [D7-Kokkos] x view is zeroed via deep_copy(v_x, 0.0) before each warmup
//   iteration (inside run_with_reset). Timed region excludes the reset.
// ─────────────────────────────────────────────────────────────────────────────

#include <Kokkos_Core.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "sptrsv_common.h"

using ExecSpace  = Kokkos::DefaultExecutionSpace;
using MemSpace   = Kokkos::DefaultExecutionSpace::memory_space;
using ViewInt    = Kokkos::View<int*,    MemSpace>;
using ViewDouble = Kokkos::View<double*, MemSpace>;

// Solve a single level: rows given by level_rows[level_start .. level_start+level_size-1]
void run_sptrsv_level_kokkos(
    const ViewInt& row_ptr, const ViewInt& col_idx,
    const ViewDouble& values, const ViewDouble& b,
    ViewDouble& x, const ViewInt& level_rows,
    int level_start, int level_size)
{
    Kokkos::parallel_for(
        "sptrsv_level_kokkos",
        Kokkos::RangePolicy<ExecSpace>(0, level_size),
        KOKKOS_LAMBDA(int tid) {
            int row = level_rows(level_start + tid);
            double sum  = b(row);
            double diag = 1.0;
            const int start = row_ptr(row);
            const int end   = row_ptr(row + 1);
            for (int j = start; j < end; j++) {
                int col = col_idx(j);
                if (col == row) {
                    diag = values(j);
                } else {
                    sum -= values(j) * x(col);
                }
            }
            x(row) = sum / diag;
        });
    Kokkos::fence();
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

    Kokkos::initialize(argc, argv);
    {
        // ── Correctness check ─────────────────────────────────────────────────
        if (verify) {
            SptrsCSR vcsr = generate_laplacian_lower(SPTRSV_N_SMALL);
            SptrsLevels vls = build_levels(vcsr);
            std::vector<double> vb  = make_b_vector(vcsr.nrows);
            std::vector<double> ref(vcsr.nrows, 0.0);
            sptrsv_cpu_ref(vcsr, vb.data(), ref.data());

            ViewInt    v_rp ("vrp",  vcsr.nrows + 1);
            ViewInt    v_ci ("vci",  vcsr.nnz);
            ViewDouble v_val("vval", vcsr.nnz);
            ViewDouble v_b  ("vb",   vcsr.nrows);
            ViewDouble v_x  ("vx",   vcsr.nrows);
            ViewInt    v_lr ("vlr",  vcsr.nrows);
            auto h_rp  = Kokkos::create_mirror_view(v_rp);
            auto h_ci  = Kokkos::create_mirror_view(v_ci);
            auto h_val = Kokkos::create_mirror_view(v_val);
            auto h_b   = Kokkos::create_mirror_view(v_b);
            auto h_lr  = Kokkos::create_mirror_view(v_lr);
            for (int i = 0; i <= vcsr.nrows; i++) h_rp(i)  = vcsr.row_ptr[i];
            for (long j = 0; j < vcsr.nnz;    j++) { h_ci(j) = vcsr.col_idx[j]; h_val(j) = vcsr.values[j]; }
            for (int i = 0; i < vcsr.nrows;  i++) { h_b(i)  = vb[i]; h_lr(i) = vls.level_rows[i]; }
            Kokkos::deep_copy(v_rp,  h_rp);
            Kokkos::deep_copy(v_ci,  h_ci);
            Kokkos::deep_copy(v_val, h_val);
            Kokkos::deep_copy(v_b,   h_b);
            Kokkos::deep_copy(v_lr,  h_lr);
            Kokkos::deep_copy(v_x,   0.0);

            for (int l = 0; l < vls.n_levels; l++) {
                int lstart = vls.level_ptr[l];
                int lsize  = vls.level_ptr[l + 1] - lstart;
                run_sptrsv_level_kokkos(v_rp, v_ci, v_val, v_b, v_x, v_lr, lstart, lsize);
            }

            auto h_y = Kokkos::create_mirror_view(v_x);
            Kokkos::deep_copy(h_y, v_x);
            std::vector<double> res(vcsr.nrows);
            for (int i = 0; i < vcsr.nrows; i++) res[i] = h_y(i);

            double max_err = 0.0;
            bool ok = sptrsv_verify(res.data(), ref.data(), vcsr.nrows, SPTRSV_CORRECT_TOL, &max_err);
            std::printf("VERIFY abstraction=kokkos matrix=lower_triangular_laplacian N=%d max_rel_err=%.2e %s\n",
                        vcsr.nrows, max_err, ok ? "PASS" : "FAIL");
            if (!ok) {
                std::fprintf(stderr, "[E5 verify] kokkos FAILED — aborting.\n");
                Kokkos::finalize(); return 1;
            }
            std::fprintf(stderr, "[E5 verify] kokkos PASS — proceeding to timed measurement.\n");
        }

        // ── Build matrix and level sets ───────────────────────────────────────
        SptrsCSR csr = build_matrix(mtype, N);
        SptrsLevels ls = build_levels(csr);
        int  nrows  = csr.nrows;
        long nnz    = csr.nnz;
        std::vector<double> b = make_b_vector(nrows);

        std::printf(
            "# abstraction=kokkos matrix=%s N=%d nnz=%ld n_levels=%d"
            " max_lw=%d min_lw=%d warmup_max=%d reps=%d platform=%s\n",
            matrix_type_str(mtype), nrows, nnz, ls.n_levels,
            ls.max_lw, ls.min_lw, warmup, reps, platform.c_str());

        // ── Allocate views ────────────────────────────────────────────────────
        ViewInt    v_rp ("rp",  nrows + 1);
        ViewInt    v_ci ("ci",  nnz);
        ViewDouble v_val("val", nnz);
        ViewDouble v_b  ("b",   nrows);
        ViewDouble v_x  ("x",   nrows);
        ViewInt    v_lr ("lr",  nrows);

        {
            auto h_rp  = Kokkos::create_mirror_view(v_rp);
            auto h_ci  = Kokkos::create_mirror_view(v_ci);
            auto h_val = Kokkos::create_mirror_view(v_val);
            auto h_b   = Kokkos::create_mirror_view(v_b);
            auto h_lr  = Kokkos::create_mirror_view(v_lr);
            for (int i = 0; i <= nrows; i++) h_rp(i)  = csr.row_ptr[i];
            for (long j = 0; j < nnz;    j++) { h_ci(j) = csr.col_idx[j]; h_val(j) = csr.values[j]; }
            for (int i = 0; i < nrows;  i++) { h_b(i)  = b[i]; h_lr(i) = ls.level_rows[i]; }
            Kokkos::deep_copy(v_rp,  h_rp);
            Kokkos::deep_copy(v_ci,  h_ci);
            Kokkos::deep_copy(v_val, h_val);
            Kokkos::deep_copy(v_b,   h_b);
            Kokkos::deep_copy(v_lr,  h_lr);
            Kokkos::deep_copy(v_x,   0.0);
        }

        const char* mstr = matrix_type_str(mtype);

        auto run_solve = [&]() {
            for (int l = 0; l < ls.n_levels; l++) {
                int lstart = ls.level_ptr[l];
                int lsize  = ls.level_ptr[l + 1] - lstart;
                run_sptrsv_level_kokkos(v_rp, v_ci, v_val, v_b, v_x, v_lr, lstart, lsize);
            }
        };

        auto run_with_reset = [&]() {
            Kokkos::deep_copy(v_x, 0.0);
            run_solve();
        };

        int warmup_iters = sptrsv_adaptive_warmup(run_with_reset, SPTRSV_WARMUP_MIN, warmup);
        std::fprintf(stderr, "[E5] kokkos: adaptive warmup done in %d iterations\n", warmup_iters);

        // ── Timed runs ────────────────────────────────────────────────────────
        std::vector<double> gflops_vec;
        gflops_vec.reserve(reps);
        for (int r = 1; r <= reps; r++) {
            Kokkos::deep_copy(v_x, 0.0);
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
    }
    Kokkos::finalize();
    return 0;
}
