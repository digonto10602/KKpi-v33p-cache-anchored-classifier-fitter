// v31b: closest-to-zero-eigenvalue midpoint zero/pole classifier.
// Algorithm requested:
//  1. Build projected F3 and (projected F3)^(-1) on a heavy coarse grid, in parallel OpenMP.
//  2. Diagonalize the projected inverse matrix and record the selected eigenvalue.
//     Default selection is the lowest algebraic eigenvalue. A config switch can use closest-to-zero for diagnostics.
//  3. Scan neighboring coarse points for sign flips.
//     left_y > 0, right_y < 0  => ZERO, E_mid = (E_left+E_right)/2
//     left_y < 0, right_y > 0  => POLE, E_mid = (E_left+E_right)/2
//  4. No refinement. No pole filtering. No index/branch tracking. No SVD validator.
// This file reuses the v30r implementation for corrected Cartesian ell=1 projector,
// cached Vsel, and parallel matrix-grid construction.

#define main v30r_disabled_main_for_v31a
#include "debug_v30r_svd_validated_cartesian_l1_100_A2_zero_scan.cpp"
#undef main

struct V31aEigPoint {
    int i=-1;
    double E=std::numeric_limits<double>::quiet_NaN();
    int success=0;
    int total_dim=0;
    int proj_dim=0;

    double y=std::numeric_limits<double>::quiet_NaN();              // selected eigenvalue used for sign flips
    int y_index=-1;
    std::string select_mode="lowest";

    double eig_lowest_alg=std::numeric_limits<double>::quiet_NaN();
    double eig_highest_alg=std::numeric_limits<double>::quiet_NaN();
    double eig_closest_zero=std::numeric_limits<double>::quiet_NaN();
    double eig_closest_abs=std::numeric_limits<double>::quiet_NaN();
    int eig_closest_index=-1;

    double herm_rel=std::numeric_limits<double>::quiet_NaN();
    double fv_equiv_F3=std::numeric_limits<double>::quiet_NaN();
    double fv_rel_leak_F3=std::numeric_limits<double>::quiet_NaN();
    std::string error="OK";
};

struct V31aFlip {
    int flip_index=-1;
    int i_left=-1;
    int i_right=-1;
    double E_left=std::numeric_limits<double>::quiet_NaN();
    double E_right=std::numeric_limits<double>::quiet_NaN();
    double E_mid=std::numeric_limits<double>::quiet_NaN();
    double y_left=std::numeric_limits<double>::quiet_NaN();
    double y_right=std::numeric_limits<double>::quiet_NaN();
    int y_index_left=-1;
    int y_index_right=-1;
    int proj_dim_left=0;
    int proj_dim_right=0;
    std::string classification="NONE"; // ZERO or POLE
    std::string status="OK";
};

static Eigen::MatrixXcd hermitize_v31a(const Eigen::MatrixXcd& M) {
    return 0.5*(M + M.adjoint());
}

static bool eigen_summary_from_projF3inv_v31a(
        const Eigen::MatrixXcd& M,
        const std::string& select_mode,
        double& y,
        int& y_index,
        double& eig_lowest,
        double& eig_highest,
        double& eig_closest,
        double& eig_closest_abs,
        int& eig_closest_index,
        double& herm_rel) {
    if(M.rows()==0 || M.cols()==0 || M.rows()!=M.cols() || !M.allFinite()) return false;
    const double nrm = M.norm();
    herm_rel = (nrm > 0.0) ? (M - M.adjoint()).norm()/nrm : 0.0;
    Eigen::MatrixXcd H = hermitize_v31a(M);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H, Eigen::EigenvaluesOnly);
    if(es.info()!=Eigen::Success || es.eigenvalues().size()==0) return false;

    eig_lowest = es.eigenvalues()[0];
    eig_highest = es.eigenvalues()[es.eigenvalues().size()-1];

    double best_abs = std::numeric_limits<double>::infinity();
    int best_idx = -1;
    for(int k=0; k<es.eigenvalues().size(); ++k) {
        const double a = std::abs(es.eigenvalues()[k]);
        if(std::isfinite(a) && a < best_abs) { best_abs = a; best_idx = k; }
    }
    if(best_idx < 0) return false;
    eig_closest = es.eigenvalues()[best_idx];
    eig_closest_abs = std::abs(eig_closest);
    eig_closest_index = best_idx;

    if(select_mode == "closest" || select_mode == "closest_zero" || select_mode == "closest-to-zero") {
        y = eig_closest;
        y_index = eig_closest_index;
    } else {
        // default requested behavior: lowest eigenvalue
        y = eig_lowest;
        y_index = 0;
    }
    return std::isfinite(y);
}

static V31aEigPoint v31a_point_from_eval(const EvalFull& ef, const std::string& select_mode) {
    V31aEigPoint p;
    p.i = ef.d.i;
    p.E = ef.d.Ecm;
    p.success = 0;
    p.total_dim = ef.d.total_dim;
    p.proj_dim = ef.d.vdim;
    p.fv_equiv_F3 = ef.d.fv_equiv_F3;
    p.fv_rel_leak_F3 = ef.d.fv_rel_leak_F3;
    p.select_mode = select_mode;
    p.error = ef.d.error;
    if(!ef.d.success || ef.projF3inv.rows()==0) return p;

    double y=NAN, lo=NAN, hi=NAN, cz=NAN, cza=NAN, herm=NAN;
    int yi=-1, czi=-1;
    if(!eigen_summary_from_projF3inv_v31a(ef.projF3inv, select_mode, y, yi, lo, hi, cz, cza, czi, herm)) {
        p.error = "EIGEN_SOLVE_FAILED";
        return p;
    }
    p.success = 1;
    p.y = y;
    p.y_index = yi;
    p.eig_lowest_alg = lo;
    p.eig_highest_alg = hi;
    p.eig_closest_zero = cz;
    p.eig_closest_abs = cza;
    p.eig_closest_index = czi;
    p.herm_rel = herm;
    p.error = "OK";
    return p;
}

static std::vector<V31aFlip> collect_v31a_flips(const std::vector<V31aEigPoint>& pts) {
    std::vector<V31aFlip> out;
    for(int i=0; i+1<(int)pts.size(); ++i) {
        const auto& L = pts[i];
        const auto& R = pts[i+1];
        if(!L.success || !R.success) continue;
        if(L.proj_dim != R.proj_dim) continue; // keep constant-projected-dimension intervals only
        if(!std::isfinite(L.y) || !std::isfinite(R.y)) continue;
        if(L.y == 0.0 || R.y == 0.0 || L.y*R.y < 0.0) {
            V31aFlip f;
            f.flip_index = (int)out.size();
            f.i_left = i;
            f.i_right = i+1;
            f.E_left = L.E;
            f.E_right = R.E;
            f.E_mid = 0.5*(L.E + R.E);
            f.y_left = L.y;
            f.y_right = R.y;
            f.y_index_left = L.y_index;
            f.y_index_right = R.y_index;
            f.proj_dim_left = L.proj_dim;
            f.proj_dim_right = R.proj_dim;
            if(L.y > 0.0 && R.y < 0.0) {
                f.classification = "ZERO";
            } else if(L.y < 0.0 && R.y > 0.0) {
                f.classification = "POLE";
            } else {
                // exact-zero endpoint convention
                f.classification = "ZERO_ENDPOINT";
            }
            out.push_back(f);
        }
    }
    return out;
}

static void write_v31a_outputs(
        const std::string& base,
        const std::vector<V31aEigPoint>& pts,
        const std::vector<V31aFlip>& flips) {
    {
        std::ofstream f(base + "_closest_eig_coarse_grid.dat");
        f << std::setprecision(17);
        f << "# columns: i Ecm success total_dim proj_dim selected_y selected_index select_mode eig_lowest_alg eig_highest_alg eig_closest_zero eig_closest_abs eig_closest_index herm_projF3inv_rel fv_equiv_F3 fv_rel_leak_F3 error\n";
        for(const auto& p: pts) {
            f << p.i << ' ' << p.E << ' ' << p.success << ' ' << p.total_dim << ' ' << p.proj_dim << ' '
              << p.y << ' ' << p.y_index << ' ' << p.select_mode << ' '
              << p.eig_lowest_alg << ' ' << p.eig_highest_alg << ' '
              << p.eig_closest_zero << ' ' << p.eig_closest_abs << ' ' << p.eig_closest_index << ' '
              << p.herm_rel << ' ' << p.fv_equiv_F3 << ' ' << p.fv_rel_leak_F3 << ' '
              << sanitize_error(p.error) << "\n";
        }
    }
    {
        std::ofstream f(base + "_signflip_zero_pole_midpoints.dat");
        f << std::setprecision(17);
        f << "# columns: flip_index classification E_mid E_left E_right y_left y_right i_left i_right selected_index_left selected_index_right proj_dim_left proj_dim_right status\n";
        for(const auto& z: flips) {
            f << z.flip_index << ' ' << z.classification << ' ' << z.E_mid << ' ' << z.E_left << ' ' << z.E_right << ' '
              << z.y_left << ' ' << z.y_right << ' ' << z.i_left << ' ' << z.i_right << ' '
              << z.y_index_left << ' ' << z.y_index_right << ' ' << z.proj_dim_left << ' ' << z.proj_dim_right << ' '
              << z.status << "\n";
        }
    }
    {
        std::ofstream f(base + "_zeros_midpoint.dat");
        f << std::setprecision(17);
        f << "# columns: zero_index E_mid E_left E_right y_left y_right selected_index_left selected_index_right\n";
        int n=0;
        for(const auto& z: flips) {
            if(z.classification == "ZERO" || z.classification == "ZERO_ENDPOINT") {
                f << n++ << ' ' << z.E_mid << ' ' << z.E_left << ' ' << z.E_right << ' '
                  << z.y_left << ' ' << z.y_right << ' ' << z.y_index_left << ' ' << z.y_index_right << "\n";
            }
        }
    }
    {
        std::ofstream f(base + "_poles_midpoint.dat");
        f << std::setprecision(17);
        f << "# columns: pole_index E_mid E_left E_right y_left y_right selected_index_left selected_index_right\n";
        int n=0;
        for(const auto& z: flips) {
            if(z.classification == "POLE") {
                f << n++ << ' ' << z.E_mid << ' ' << z.E_left << ' ' << z.E_right << ' '
                  << z.y_left << ' ' << z.y_right << ' ' << z.y_index_left << ' ' << z.y_index_right << "\n";
            }
        }
    }
}

int main(int argc, char** argv) {
    if(argc != 2) {
        std::cerr << "Usage:\n  " << argv[0] << " config/config_v31b_closest_eig_midpoint_100_A2.in\n";
        return 1;
    }
    const std::string input_file = argv[1];
    try {
        V30eScopedTimer total_timer("v31b total executable");
        PhysicsParams par;
        Options opt = read_options(input_file, par);
        #ifdef _OPENMP
        omp_set_num_threads(opt.threads);
        omp_set_max_active_levels(1);
        #endif
        Eigen::setNbThreads(1);
        std::filesystem::create_directories(opt.outdir);
        auto kv = read_kv_file(input_file);
        const std::string select_mode = get_string(kv, "eigen_select_mode", "closest");
        stage_log("[stage 1/6] 100% config read; eigen_select_mode=" + select_mode);
        for(const std::string& lab: opt.mom_labels) {
            MomentumIrrepSpec spec = parse_label(lab);
            std::vector<int> nnP = {spec.nnP[0], spec.nnP[1], spec.nnP[2]};
            const std::string base = opt.outdir + "/" + opt.prefix;

            std::ofstream meta(base + "_metadata.txt");
            meta << std::setprecision(17);
            meta << "version = v31b_closest_eig_midpoint_A2_zero_pole_scan\n";
            meta << "input_file = " << input_file << "\n";
            meta << "label = " << lab << "\n";
            meta << "nnP = " << nnP[0] << " " << nnP[1] << " " << nnP[2] << "\n";
            meta << "irrep = " << spec.irrep << "\n";
            meta << "E0 = " << opt.E0 << "\nE1 = " << opt.E1 << "\nN = " << opt.N << "\nthreads = " << opt.threads << "\n";
            meta << "matrix = inverse(projected_F3) = inverse(V^dagger F3 V), corrected Cartesian ell=1 projector\n";
            meta << "zero_pole_algorithm = coarse-only sign flip of closest-to-zero selected eigenvalue; positive-to-negative is ZERO at midpoint; negative-to-positive is POLE at midpoint; no refinement\n";
            meta << "eigen_select_mode = " << select_mode << "\n";
            meta << "expected_default_N = 1501\n";

            std::cout << "[v31b] label=" << lab << " nnP=(" << nnP[0] << "," << nnP[1] << "," << nnP[2] << ") irrep=" << spec.irrep << "\n";
            std::cout << "[v31b] coarse-only midpoint zero/pole scan with N=" << opt.N << ", E=[" << opt.E0 << "," << opt.E1 << "]\n";

            stage_log("[stage 2/6] building projected matrix grid in parallel");
            const double t0 = wall_seconds_now();
            std::vector<EvalFull> grid = evaluate_full_grid_parallel(opt, nnP, spec.irrep, par);
            stage_log("[stage 2/6] 100% projected matrix grid done, dt=" + std::to_string(wall_seconds_now()-t0) + " s");

            stage_log("[stage 3/6] extracting selected eigenvalue on coarse grid");
            std::vector<V31aEigPoint> pts(grid.size());
            int nextpct=10;
            for(int i=0; i<(int)grid.size(); ++i) {
                pts[i] = v31a_point_from_eval(grid[i], select_mode);
                progress_percent_log("v31a-eig", i+1, (int)grid.size(), nextpct, 10);
            }
            stage_log("[stage 3/6] 100% eigenvalue extraction done");

            stage_log("[stage 4/6] classifying sign flips into midpoint zeros and poles");
            auto flips = collect_v31a_flips(pts);
            int nzero=0, npole=0;
            for(const auto& f: flips) {
                if(f.classification == "ZERO" || f.classification == "ZERO_ENDPOINT") ++nzero;
                if(f.classification == "POLE") ++npole;
            }
            stage_log("[stage 4/6] 100% sign flips classified, total=" + std::to_string(flips.size()) + ", zeros=" + std::to_string(nzero) + ", poles=" + std::to_string(npole));

            stage_log("[stage 5/6] writing outputs");
            write_v31a_outputs(base, pts, flips);
            meta << "success_points = ";
            int ok=0; for(const auto& p: pts) if(p.success) ++ok;
            meta << ok << "\n";
            meta << "projector_cache_entries = " << g_projector_cache_v30q.size() << "\n";
            meta << "projector_cache_hits = " << g_projector_cache_hits_v30q << "\n";
            meta << "projector_cache_misses = " << g_projector_cache_misses_v30q << "\n";
            meta << "total_sign_flips = " << flips.size() << "\n";
            meta << "midpoint_zeros = " << nzero << "\n";
            meta << "midpoint_poles = " << npole << "\n";
            stage_log("[stage 5/6] 100% outputs written");

            stage_log("[stage 6/6] done");
        }
    } catch(const std::exception& e) {
        std::cerr << "[v31b-error] " << e.what() << std::endl;
        return 2;
    }
    return 0;
}
