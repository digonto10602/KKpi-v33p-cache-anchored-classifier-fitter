// v30z: Blind A2 eigen-min local-minimum scan with pole rejection and stability validation.
// This file reuses the v30r implementation (fixed Cartesian ell=1 projector, cached Vsel,
// parallel coarse grid builder), but replaces branch/SVD zero finding by a local
// minimization of |closest projected eigenvalue| inside non-pole sign-flip windows.
#define main v30r_disabled_main_for_v30v
#include "debug_v30r_svd_validated_cartesian_l1_100_A2_zero_scan.cpp"
#undef main

struct EigMinPointV30v {
    int i=-1;
    double E=std::numeric_limits<double>::quiet_NaN();
    int success=0;
    int total_dim=0;
    int proj_dim=0;
    double eig_closest_zero=std::numeric_limits<double>::quiet_NaN();
    double eig_closest_abs=std::numeric_limits<double>::quiet_NaN();
    int eig_index=-1;
    double eig_min_alg=std::numeric_limits<double>::quiet_NaN();
    double eig_max_alg=std::numeric_limits<double>::quiet_NaN();
    double herm_rel=std::numeric_limits<double>::quiet_NaN();
    double fv_rel_leak_F3=std::numeric_limits<double>::quiet_NaN();
    double fv_equiv_F3=std::numeric_limits<double>::quiet_NaN();
    std::string error="OK";
};

struct EigMinCandidateV30v {
    int cand_index=-1;
    int i_left=-1, i_right=-1;
    double E_left=std::numeric_limits<double>::quiet_NaN();
    double E_right=std::numeric_limits<double>::quiet_NaN();
    double y_left=std::numeric_limits<double>::quiet_NaN();
    double y_right=std::numeric_limits<double>::quiet_NaN();
    double abs_y_left=std::numeric_limits<double>::quiet_NaN();
    double abs_y_right=std::numeric_limits<double>::quiet_NaN();
    int eig_index_left=-1, eig_index_right=-1;
    double eig_min_alg_left=std::numeric_limits<double>::quiet_NaN();
    double eig_max_alg_left=std::numeric_limits<double>::quiet_NaN();
    double eig_min_alg_right=std::numeric_limits<double>::quiet_NaN();
    double eig_max_alg_right=std::numeric_limits<double>::quiet_NaN();
    double max_endpoint_abs=std::numeric_limits<double>::quiet_NaN();
    double max_endpoint_alg_range=std::numeric_limits<double>::quiet_NaN();
    int proj_dim_left=0, proj_dim_right=0;
    int near_dimension_jump=0;
    int branch_index_switch=0;
    int likely_pole=0;
    int endpoint_not_small=0;
    int keep_for_bisection=0;
    std::string status="UNREFINED";
};

struct EigZeroRecordV30v {
    int zero_index=-1;
    int cand_index=-1;
    double E=std::numeric_limits<double>::quiet_NaN();
    double y=std::numeric_limits<double>::quiet_NaN();
    double abs_y=std::numeric_limits<double>::quiet_NaN();
    double E_left=std::numeric_limits<double>::quiet_NaN();
    double E_right=std::numeric_limits<double>::quiet_NaN();
    double y_left0=std::numeric_limits<double>::quiet_NaN();
    double y_right0=std::numeric_limits<double>::quiet_NaN();
    int iter=0;
    double final_width=std::numeric_limits<double>::quiet_NaN();
    double min_overlap=std::numeric_limits<double>::quiet_NaN();
    double sigma_min=std::numeric_limits<double>::quiet_NaN();
    double final_herm_rel=std::numeric_limits<double>::quiet_NaN();
    int accepted=0;
    std::string method="local_grid_min_abs_closest_eigenvalue";
    std::string status="UNKNOWN";
};

static Eigen::MatrixXcd hermitize_v30v(const Eigen::MatrixXcd& M) {
    return 0.5*(M + M.adjoint());
}

static bool closest_zero_eigenpair_from_matrix_v30v(
        const Eigen::MatrixXcd& M,
        double& eig,
        double& abs_eig,
        int& eig_index,
        Eigen::VectorXcd* eigvec,
        double& herm_rel,
        double& eig_min_alg,
        double& eig_max_alg) {
    if(M.rows()==0 || M.cols()==0 || M.rows()!=M.cols() || !M.allFinite()) return false;
    const double nrm = M.norm();
    herm_rel = (nrm>0.0) ? (M-M.adjoint()).norm()/nrm : 0.0;
    Eigen::MatrixXcd H = hermitize_v30v(M);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H, eigvec ? Eigen::ComputeEigenvectors : Eigen::EigenvaluesOnly);
    if(es.info()!=Eigen::Success || es.eigenvalues().size()==0) return false;
    eig_min_alg = es.eigenvalues().minCoeff();
    eig_max_alg = es.eigenvalues().maxCoeff();
    double best = std::numeric_limits<double>::infinity();
    int best_i=-1;
    for(int k=0;k<es.eigenvalues().size();++k){
        const double a = std::abs(es.eigenvalues()[k]);
        if(std::isfinite(a) && a<best){ best=a; best_i=k; }
    }
    if(best_i<0) return false;
    eig = es.eigenvalues()[best_i];
    abs_eig = std::abs(eig);
    eig_index = best_i;
    if(eigvec) *eigvec = es.eigenvectors().col(best_i);
    return std::isfinite(eig) && std::isfinite(abs_eig);
}

static EigMinPointV30v eigmin_point_from_eval_v30v(const EvalFull& ef) {
    EigMinPointV30v p;
    p.i=ef.d.i; p.E=ef.d.Ecm; p.success=0; p.total_dim=ef.d.total_dim; p.proj_dim=ef.d.vdim;
    p.fv_rel_leak_F3=ef.d.fv_rel_leak_F3; p.fv_equiv_F3=ef.d.fv_equiv_F3; p.error=ef.d.error;
    if(!ef.d.success) return p;
    double eig=NAN, ae=NAN, herm=NAN, emin=NAN, emax=NAN; int idx=-1;
    if(!closest_zero_eigenpair_from_matrix_v30v(ef.projF3inv,eig,ae,idx,nullptr,herm,emin,emax)){
        p.error="EIGEN_MIN_FAILED"; return p;
    }
    p.success=1; p.eig_closest_zero=eig; p.eig_closest_abs=ae; p.eig_index=idx; p.herm_rel=herm; p.eig_min_alg=emin; p.eig_max_alg=emax; p.error="OK";
    return p;
}

static bool sign_flip_v30v(double a, double b) {
    return std::isfinite(a) && std::isfinite(b) && (a==0.0 || b==0.0 || a*b<0.0);
}

static bool eval_closest_eig_tracked_v30v(
        double E,
        const std::vector<int>& nnP,
        const std::string& irrep,
        const PhysicsParams& par,
        char debug,
        const Eigen::VectorXcd* refvec,
        double& y,
        double& abs_y,
        Eigen::VectorXcd* outvec,
        double& overlap,
        double& herm_rel) {
    EvalFull ef = evaluate_full(E,nnP,irrep,par,debug);
    if(!ef.d.success || ef.projF3inv.rows()==0) return false;
    Eigen::MatrixXcd H = hermitize_v30v(ef.projF3inv);
    const double nrm = ef.projF3inv.norm();
    herm_rel = (nrm>0.0) ? (ef.projF3inv-ef.projF3inv.adjoint()).norm()/nrm : 0.0;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H);
    if(es.info()!=Eigen::Success) return false;
    int best=-1;
    if(refvec && refvec->size()==es.eigenvectors().rows()) {
        double bov=-1.0;
        for(int k=0;k<es.eigenvalues().size();++k){
            const double ov = std::abs((refvec->adjoint()*es.eigenvectors().col(k))(0,0));
            if(std::isfinite(ov) && ov>bov){ bov=ov; best=k; }
        }
        overlap=bov;
    } else {
        double ba=std::numeric_limits<double>::infinity();
        for(int k=0;k<es.eigenvalues().size();++k){
            const double a=std::abs(es.eigenvalues()[k]);
            if(std::isfinite(a) && a<ba){ ba=a; best=k; }
        }
        overlap=1.0;
    }
    if(best<0) return false;
    y = es.eigenvalues()[best];
    abs_y = std::abs(y);
    if(outvec) *outvec = es.eigenvectors().col(best);
    return std::isfinite(y) && std::isfinite(abs_y);
}


static double g_eigen_zero_abs_tol_v30v = 1.0e-7;
static double g_eigen_sigma_cross_tol_v30v = 1.0e-5;
static double g_eigen_min_overlap_tol_v30v = 0.90;
static int    g_eigen_require_svd_cross_v30v = 1;
static int    g_eigen_prefilter_reject_index_switch_v30v = 1;
static int    g_eigen_pole_filter_enabled_v30v = 1;
static double g_eigen_pole_range_abs_max_v30v = 1.0;
static double g_eigen_prefilter_endpoint_abs_tol_v30v = 1.0e-4;

static bool eval_sigma_min_projected_v30v(
        double E,
        const std::vector<int>& nnP,
        const std::string& irrep,
        const PhysicsParams& par,
        char debug,
        double& sigma_min,
        double& herm_rel) {
    EvalFull ef = evaluate_full(E,nnP,irrep,par,debug);
    if(!ef.d.success || ef.projF3inv.rows()==0 || ef.projF3inv.rows()!=ef.projF3inv.cols() || !ef.projF3inv.allFinite()) return false;
    const double nrm = std::max(ef.projF3inv.norm(), 1.0e-300);
    herm_rel = (ef.projF3inv - ef.projF3inv.adjoint()).norm()/nrm;
    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(ef.projF3inv, Eigen::ComputeThinU | Eigen::ComputeThinV);
    if(svd.singularValues().size()==0) return false;
    sigma_min = svd.singularValues().minCoeff();
    return std::isfinite(sigma_min);
}


static int g_eigen_local_min_grid_N_v30v = 61;
static int g_eigen_local_min_double_grid_v30v = 1;
static double g_eigen_local_min_stability_E_tol_v30v = 2.0e-4;

struct LocalMinScanResultV30v {
    int ok=0;
    double E=std::numeric_limits<double>::quiet_NaN();
    double y=std::numeric_limits<double>::quiet_NaN();
    double abs_y=std::numeric_limits<double>::quiet_NaN();
    double herm_rel=std::numeric_limits<double>::quiet_NaN();
    int grid_N=0;
};

static LocalMinScanResultV30v scan_local_min_abs_closest_eig_v30v(
        double E_left,
        double E_right,
        int grid_N,
        const std::vector<int>& nnP,
        const std::string& irrep,
        const PhysicsParams& par,
        char debug) {
    LocalMinScanResultV30v r;
    r.grid_N = grid_N;
    if(grid_N < 2 || !(E_right>=E_left)) return r;
    double best_abs = std::numeric_limits<double>::infinity();
    for(int j=0;j<grid_N;++j){
        const double t = (grid_N==1) ? 0.0 : double(j)/double(grid_N-1);
        const double E = E_left + t*(E_right-E_left);
        double y=NAN, ay=NAN, ov=NAN, herm=NAN;
        if(!eval_closest_eig_tracked_v30v(E,nnP,irrep,par,debug,nullptr,y,ay,nullptr,ov,herm)) continue;
        if(std::isfinite(ay) && ay < best_abs){
            best_abs=ay; r.ok=1; r.E=E; r.y=y; r.abs_y=ay; r.herm_rel=herm;
        }
    }
    return r;
}

static EigZeroRecordV30v localmin_eigen_min_candidate_v30v(
        const EigMinCandidateV30v& c,
        const std::vector<int>& nnP,
        const std::string& irrep,
        const PhysicsParams& par,
        char debug) {
    EigZeroRecordV30v z;
    z.cand_index=c.cand_index; z.E_left=c.E_left; z.E_right=c.E_right; z.y_left0=c.y_left; z.y_right0=c.y_right;
    z.method = "local_grid_min_abs_closest_eigenvalue_pole_reject_stability_checked";
    if(!c.keep_for_bisection){ z.status=c.status; return z; }

    const int N1 = std::max(3, g_eigen_local_min_grid_N_v30v);
    const int N2 = g_eigen_local_min_double_grid_v30v ? std::max(2*N1-1, N1+2) : N1;
    LocalMinScanResultV30v r1 = scan_local_min_abs_closest_eig_v30v(c.E_left,c.E_right,N1,nnP,irrep,par,debug);
    LocalMinScanResultV30v r2 = scan_local_min_abs_closest_eig_v30v(c.E_left,c.E_right,N2,nnP,irrep,par,debug);
    if(!r1.ok || !r2.ok){ z.status="REJECT_LOCAL_MIN_EVAL_FAILED"; return z; }

    // Use the finer-grid minimum as the final candidate. Coarser grid is used only for stability.
    z.E = r2.E; z.y = r2.y; z.abs_y = r2.abs_y; z.final_herm_rel = r2.herm_rel;
    z.iter = N2;
    z.final_width = (N2>1) ? std::abs(c.E_right-c.E_left)/double(N2-1) : std::abs(c.E_right-c.E_left);
    z.min_overlap = std::abs(r2.E-r1.E); // repurpose column as stability_dE for this method
    double sig=NAN, hs=NAN;
    const bool sigma_ok_eval = eval_sigma_min_projected_v30v(z.E,nnP,irrep,par,debug,sig,hs);
    z.sigma_min = sigma_ok_eval ? sig : std::numeric_limits<double>::quiet_NaN();
    if(sigma_ok_eval && std::isfinite(hs)) z.final_herm_rel = std::max(z.final_herm_rel, hs);

    const bool eig_ok = std::isfinite(z.abs_y) && z.abs_y <= g_eigen_zero_abs_tol_v30v;
    const bool stable_ok = std::isfinite(z.min_overlap) && z.min_overlap <= g_eigen_local_min_stability_E_tol_v30v;
    z.accepted = (eig_ok && stable_ok) ? 1 : 0;
    if(z.accepted) z.status = "ACCEPT_LOCAL_MIN_ABS_EIGENVALUE_NONPOLE_STABLE";
    else if(!eig_ok) z.status = "REJECT_LOCAL_MIN_ABS_EIG_TOO_LARGE";
    else if(!stable_ok) z.status = "REJECT_LOCAL_MIN_NOT_STABLE_ON_FINE_GRID";
    else z.status = "REJECT_UNKNOWN_LOCAL_MIN_FAILURE";
    return z;
}

static std::vector<EigMinCandidateV30v> collect_eigen_min_signflips_v30v(
        const std::vector<EigMinPointV30v>& pts,
        const std::vector<double>& dim_jumps,
        double jump_guard) {
    std::vector<EigMinCandidateV30v> out;
    for(int i=0;i+1<(int)pts.size();++i){
        const auto& a=pts[i]; const auto& b=pts[i+1];
        if(!a.success || !b.success) continue;
        if(a.proj_dim!=b.proj_dim) continue;
        if(!sign_flip_v30v(a.eig_closest_zero,b.eig_closest_zero)) continue;
        EigMinCandidateV30v c;
        c.cand_index=(int)out.size(); c.i_left=i; c.i_right=i+1;
        c.E_left=a.E; c.E_right=b.E;
        c.y_left=a.eig_closest_zero; c.y_right=b.eig_closest_zero;
        c.abs_y_left=a.eig_closest_abs; c.abs_y_right=b.eig_closest_abs;
        c.eig_index_left=a.eig_index; c.eig_index_right=b.eig_index;
        c.eig_min_alg_left=a.eig_min_alg; c.eig_max_alg_left=a.eig_max_alg;
        c.eig_min_alg_right=b.eig_min_alg; c.eig_max_alg_right=b.eig_max_alg;
        c.max_endpoint_abs=std::max(a.eig_closest_abs,b.eig_closest_abs);
        c.max_endpoint_alg_range=std::max({std::abs(a.eig_min_alg),std::abs(a.eig_max_alg),std::abs(b.eig_min_alg),std::abs(b.eig_max_alg)});
        c.proj_dim_left=a.proj_dim; c.proj_dim_right=b.proj_dim;
        double nearest=std::numeric_limits<double>::infinity();
        for(double je: dim_jumps){ nearest=std::min(nearest,std::min(std::abs(c.E_left-je),std::abs(c.E_right-je))); }
        c.near_dimension_jump=(nearest<=jump_guard)?1:0;
        c.branch_index_switch=(a.eig_index!=b.eig_index)?1:0;
        c.endpoint_not_small=(std::isfinite(c.max_endpoint_abs) && c.max_endpoint_abs>g_eigen_prefilter_endpoint_abs_tol_v30v)?1:0;
        c.likely_pole=(std::isfinite(c.max_endpoint_alg_range) && c.max_endpoint_alg_range>g_eigen_pole_range_abs_max_v30v)?1:0;
        // v30z requested behavior: disregard branch/index-switch and endpoint-smallness
        // for acceptance, but USE the likely-pole flag to reject pole-like candidates
        // before local-min refinement. Pole flags remain written to the candidate file and plot.
        if(g_eigen_pole_filter_enabled_v30v && c.likely_pole){
            c.keep_for_bisection=0;
            c.status="REJECT_LIKELY_POLE_FLAG";
        } else {
            c.keep_for_bisection=1;
            c.status="KEEP_RAW_CLOSEST_EIGEN_SIGNFLIP_NONPOLE";
        }
        out.push_back(c);
    }
    return out;
}


static std::vector<EigZeroRecordV30v> refine_candidates_parallel_v30v(
        const std::vector<EigMinCandidateV30v>& cands,
        const std::vector<int>& nnP,
        const std::string& irrep,
        const PhysicsParams& par,
        char debug,
        double Etol,
        int max_iter,
        int use_tracking,
        int parallel_refine) {
    (void)Etol; (void)max_iter; (void)use_tracking;
    std::vector<EigZeroRecordV30v> out(cands.size());
    int done=0, nextpct=10;
    if(parallel_refine){
        #pragma omp parallel for schedule(dynamic,1)
        for(int i=0;i<(int)cands.size();++i){
            out[i]=localmin_eigen_min_candidate_v30v(cands[i],nnP,irrep,par,debug);
            #pragma omp critical
            { ++done; progress_percent_log("eigmin-localmin", done, (int)cands.size(), nextpct, 10); }
        }
    } else {
        for(int i=0;i<(int)cands.size();++i){ out[i]=localmin_eigen_min_candidate_v30v(cands[i],nnP,irrep,par,debug); ++done; progress_percent_log("eigmin-localmin", done, (int)cands.size(), nextpct, 10); }
    }
    int zi=0; for(auto& z: out) z.zero_index=zi++;
    return out;
}

static void write_v30v_outputs(
        const std::string& base,
        const std::vector<EigMinPointV30v>& pts,
        const std::vector<EigMinCandidateV30v>& cands,
        const std::vector<EigZeroRecordV30v>& zeros) {
    {
        std::ofstream f(base+"_eigen_min_coarse_grid.dat"); f<<std::setprecision(17);
        f << "# columns: i Ecm success total_dim proj_dim eig_closest_zero eig_closest_abs eig_index eig_min_alg eig_max_alg herm_projF3inv_rel fv_equiv_F3 fv_rel_leak_F3 error\n";
        for(const auto& p: pts) f<<p.i<<' '<<p.E<<' '<<p.success<<' '<<p.total_dim<<' '<<p.proj_dim<<' '<<p.eig_closest_zero<<' '<<p.eig_closest_abs<<' '<<p.eig_index<<' '<<p.eig_min_alg<<' '<<p.eig_max_alg<<' '<<p.herm_rel<<' '<<p.fv_equiv_F3<<' '<<p.fv_rel_leak_F3<<' '<<sanitize_error(p.error)<<"\n";
    }
    {
        std::ofstream f(base+"_eigen_min_signflip_candidates.dat"); f<<std::setprecision(17);
        f << "# columns: cand_index i_left i_right E_left E_right y_left y_right abs_y_left abs_y_right eig_index_left eig_index_right eig_min_alg_left eig_max_alg_left eig_min_alg_right eig_max_alg_right max_endpoint_abs max_endpoint_alg_range proj_dim_left proj_dim_right near_dimension_jump branch_index_switch likely_pole endpoint_not_small keep_for_bisection status\n";
        for(const auto& c: cands) f<<c.cand_index<<' '<<c.i_left<<' '<<c.i_right<<' '<<c.E_left<<' '<<c.E_right<<' '<<c.y_left<<' '<<c.y_right<<' '<<c.abs_y_left<<' '<<c.abs_y_right<<' '<<c.eig_index_left<<' '<<c.eig_index_right<<' '<<c.eig_min_alg_left<<' '<<c.eig_max_alg_left<<' '<<c.eig_min_alg_right<<' '<<c.eig_max_alg_right<<' '<<c.max_endpoint_abs<<' '<<c.max_endpoint_alg_range<<' '<<c.proj_dim_left<<' '<<c.proj_dim_right<<' '<<c.near_dimension_jump<<' '<<c.branch_index_switch<<' '<<c.likely_pole<<' '<<c.endpoint_not_small<<' '<<c.keep_for_bisection<<' '<<c.status<<"\n";
    }
    {
        std::ofstream f(base+"_eigen_min_local_min_refined.dat"); f<<std::setprecision(17);
        f << "# columns: record_index cand_index accepted Ecm eig_value abs_eig sigma_min local_grid_dE fine_grid_N stability_dE final_herm_rel E_left E_right y_left0 y_right0 method status\n";
        for(const auto& z: zeros) f<<z.zero_index<<' '<<z.cand_index<<' '<<z.accepted<<' '<<z.E<<' '<<z.y<<' '<<z.abs_y<<' '<<z.sigma_min<<' '<<z.final_width<<' '<<z.iter<<' '<<z.min_overlap<<' '<<z.final_herm_rel<<' '<<z.E_left<<' '<<z.E_right<<' '<<z.y_left0<<' '<<z.y_right0<<' '<<z.method<<' '<<z.status<<"\n";
    }
    {
        std::ofstream f(base+"_eigen_min_strict_validated_levels.dat"); f<<std::setprecision(17);
        f << "# columns: level_index Ecm eig_value abs_eig sigma_min local_grid_dE fine_grid_N stability_dE final_herm_rel method\n";
        int li=0;
        for(const auto& z: zeros) if(z.accepted) f<<li++<<' '<<z.E<<' '<<z.y<<' '<<z.abs_y<<' '<<z.sigma_min<<' '<<z.final_width<<' '<<z.iter<<' '<<z.min_overlap<<' '<<z.final_herm_rel<<' '<<z.method<<"\n";
    }
}

int main(int argc, char** argv) {
    if(argc!=2){ std::cerr<<"Usage:\n  "<<argv[0]<<" config/config_v30z_eigen_min_localmin_pole_reject_100_A2.in\n"; return 1; }
    const std::string input_file=argv[1];
    try{
        V30eScopedTimer total_timer("v30z total executable");
        PhysicsParams par; Options opt=read_options(input_file,par);
        // v30z reads mom_labels from the config. Provided configs include 100_A2 and 110_A2.
        #ifdef _OPENMP
        omp_set_num_threads(opt.threads);
        omp_set_max_active_levels(1);
        #endif
        Eigen::setNbThreads(1);
        std::filesystem::create_directories(opt.outdir);
        const double eigen_bisect_tol = get_double(read_kv_file(input_file), "eigen_bisect_tol", 1.0e-9); // retained for backward-compatible metadata only
        const int eigen_bisect_max_iter = get_int(read_kv_file(input_file), "eigen_bisect_max_iter", 80); // retained for backward-compatible metadata only
        const int eigen_use_tracking = get_int(read_kv_file(input_file), "eigen_use_tracking", 0); // not used by local-min method
        g_eigen_zero_abs_tol_v30v = get_double(read_kv_file(input_file), "eigen_zero_abs_tol", 1.0e-7);
        g_eigen_sigma_cross_tol_v30v = get_double(read_kv_file(input_file), "eigen_sigma_cross_tol", 1.0e-5);
        g_eigen_min_overlap_tol_v30v = get_double(read_kv_file(input_file), "eigen_min_overlap_tol", 0.90);
        g_eigen_require_svd_cross_v30v = get_int(read_kv_file(input_file), "eigen_require_svd_crosscheck", 1);
        g_eigen_prefilter_reject_index_switch_v30v = get_int(read_kv_file(input_file), "eigen_prefilter_reject_index_switch", 1);
        g_eigen_pole_filter_enabled_v30v = get_int(read_kv_file(input_file), "eigen_pole_filter_enabled", 1);
        g_eigen_pole_range_abs_max_v30v = get_double(read_kv_file(input_file), "eigen_pole_range_abs_max", 1.0);
        g_eigen_prefilter_endpoint_abs_tol_v30v = get_double(read_kv_file(input_file), "eigen_prefilter_endpoint_abs_tol", 1.0e-4);
        g_eigen_local_min_grid_N_v30v = get_int(read_kv_file(input_file), "eigen_local_min_grid_N", 61);
        g_eigen_local_min_double_grid_v30v = get_int(read_kv_file(input_file), "eigen_local_min_double_grid", 1);
        g_eigen_local_min_stability_E_tol_v30v = get_double(read_kv_file(input_file), "eigen_local_min_stability_E_tol", 2.0e-4);
        stage_log("[stage 1/7] 100% config read; irrep/momentum read from mom_labels");
        for(const std::string& lab: opt.mom_labels){
            MomentumIrrepSpec spec=parse_label(lab); std::vector<int> nnP={spec.nnP[0],spec.nnP[1],spec.nnP[2]};
            const std::string base=opt.outdir+"/"+opt.prefix;
            std::ofstream meta(base+"_metadata.txt"); meta<<std::setprecision(17);
            meta << "version = v30z_eigen_min_localmin_pole_reject_A2_zero_scan\n";
            meta << "input_file = "<<input_file<<"\nlabel = "<<lab<<"\nnnP = "<<nnP[0]<<" "<<nnP[1]<<" "<<nnP[2]<<"\nirrep = "<<spec.irrep<<"\n";
            meta << "E0 = "<<opt.E0<<"\nE1 = "<<opt.E1<<"\nN = "<<opt.N<<"\nthreads = "<<opt.threads<<"\n";
            meta << "zero_method = raw closest-eigenvalue sign-flip candidates followed by local minimization of |closest eigenvalue|; accept if min_abs < tolerance, non-pole, stable under finer grid\n";
            meta << "matrix = projected_F3inv = inverse(V^dagger F3 V), corrected Cartesian ell=1 projector\n";
            meta << "eigen_matrix_treatment = hermitize projected matrix before SelfAdjointEigenSolver\n";
            meta << "eigen_bisect_tol = "<<eigen_bisect_tol<<"\neigen_bisect_max_iter = "<<eigen_bisect_max_iter<<"\neigen_use_tracking = "<<eigen_use_tracking<<"\n";
            meta << "eigen_zero_abs_tol = "<<g_eigen_zero_abs_tol_v30v<<"\neigen_sigma_cross_tol = "<<g_eigen_sigma_cross_tol_v30v<<"\neigen_min_overlap_tol = "<<g_eigen_min_overlap_tol_v30v<<"\neigen_require_svd_crosscheck = "<<g_eigen_require_svd_cross_v30v<<"\n";
            meta << "eigen_prefilter_reject_index_switch = "<<g_eigen_prefilter_reject_index_switch_v30v<<"\neigen_pole_filter_enabled = "<<g_eigen_pole_filter_enabled_v30v<<"\neigen_pole_range_abs_max = "<<g_eigen_pole_range_abs_max_v30v<<"\neigen_prefilter_endpoint_abs_tol = "<<g_eigen_prefilter_endpoint_abs_tol_v30v<<"\n";
            meta << "eigen_local_min_grid_N = "<<g_eigen_local_min_grid_N_v30v<<"\neigen_local_min_double_grid = "<<g_eigen_local_min_double_grid_v30v<<"\neigen_local_min_stability_E_tol = "<<g_eigen_local_min_stability_E_tol_v30v<<"\n";
            meta << "note = v30z keeps same-dimension closest-eigenvalue sign-flip windows only if they are not flagged as likely_pole. Branch index switching and endpoint-smallness are diagnostic only. It minimizes |closest eigenvalue| on a local grid and a doubled grid, accepting only if min_abs <= eigen_zero_abs_tol and the minimum position is stable.\n";

            std::cout << "[v30z] raw closest-eigenvalue local-min scan for "<<lab<<"\n";
            stage_log("[stage 2/7] building coarse projected matrices in parallel");
            const double t0=wall_seconds_now();
            std::vector<EvalFull> fullgrid = evaluate_full_grid_parallel(opt, nnP, spec.irrep, par);
            stage_log("[stage 2/7] 100% coarse matrix cache done, dt=" + std::to_string(wall_seconds_now()-t0) + " s");

            stage_log("[stage 3/7] extracting closest-to-zero projected eigenvalue on coarse grid");
            std::vector<EigMinPointV30v> pts(fullgrid.size());
            int nextpct=10;
            for(int i=0;i<(int)fullgrid.size();++i){ pts[i]=eigmin_point_from_eval_v30v(fullgrid[i]); progress_percent_log("eigmin-coarse", i+1, (int)fullgrid.size(), nextpct, 10); }
            stage_log("[stage 3/7] 100% eigen-min coarse extraction done");

            std::vector<double> dim_jumps;
            for(size_t i=0;i+1<fullgrid.size();++i) if(fullgrid[i].d.success && fullgrid[i+1].d.success && fullgrid[i].d.vdim!=fullgrid[i+1].d.vdim) dim_jumps.push_back(0.5*(fullgrid[i].d.Ecm+fullgrid[i+1].d.Ecm));
            meta << "dimension_jump_count = "<<dim_jumps.size()<<"\n";
            meta << "projector_cache_entries = " << g_projector_cache_v30q.size() << "\n";
            meta << "projector_cache_hits = " << g_projector_cache_hits_v30q << "\n";
            meta << "projector_cache_misses = " << g_projector_cache_misses_v30q << "\n";

            stage_log("[stage 4/7] collecting raw closest-eigenvalue sign flips");
            auto cands=collect_eigen_min_signflips_v30v(pts,dim_jumps,opt.dimension_jump_guard);
            int kept=0; for(const auto& c:cands) if(c.keep_for_bisection) ++kept;
            meta << "eigen_min_raw_signflip_candidates = "<<cands.size()<<"\n";
            meta << "eigen_min_candidates_kept_for_local_min = "<<kept<<"\n";
            stage_log("[stage 4/7] 100% sign-flip collection done, raw="+std::to_string(cands.size())+", kept="+std::to_string(kept));

            stage_log("[stage 5/7] local minimization of |closest eigenvalue| for non-pole candidates");
            const double tr0=wall_seconds_now();
            auto zeros=refine_candidates_parallel_v30v(cands,nnP,spec.irrep,par,opt.debug,eigen_bisect_tol,eigen_bisect_max_iter,eigen_use_tracking,opt.parallel_refine);
            int accepted=0; for(const auto& z: zeros) if(z.accepted) ++accepted;
            meta << "eigen_min_local_min_records = "<<zeros.size()<<"\n";
            meta << "eigen_min_strict_accepted_levels = "<<accepted<<"\n";
            stage_log("[stage 5/7] 100% local minimization done, accepted="+std::to_string(accepted)+", dt="+std::to_string(wall_seconds_now()-tr0)+" s");

            stage_log("[stage 6/7] writing outputs");
            write_v30v_outputs(base,pts,cands,zeros);
            stage_log("[stage 6/7] 100% outputs written");
            stage_log("[stage 7/7] done");
        }
    } catch(const std::exception& e){ std::cerr << "[v30z-error] " << e.what() << std::endl; return 2; }
    return 0;
}
