// v30t: Blind 100_A2 eigen-min bisection with residual, overlap, and SVD cross-check validation.
// This file reuses the v30r implementation (fixed Cartesian ell=1 projector, cached Vsel,
// parallel coarse grid builder), but replaces branch/SVD zero finding by a minimal
// eigenvalue-sign-flip scan plus bisection.
#define main v30r_disabled_main_for_v30t
#include "debug_v30r_svd_validated_cartesian_l1_100_A2_zero_scan.cpp"
#undef main

struct EigMinPointV30t {
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

struct EigMinCandidateV30t {
    int cand_index=-1;
    int i_left=-1, i_right=-1;
    double E_left=std::numeric_limits<double>::quiet_NaN();
    double E_right=std::numeric_limits<double>::quiet_NaN();
    double y_left=std::numeric_limits<double>::quiet_NaN();
    double y_right=std::numeric_limits<double>::quiet_NaN();
    int proj_dim_left=0, proj_dim_right=0;
    int near_dimension_jump=0;
    std::string status="UNREFINED";
};

struct EigZeroRecordV30t {
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
    std::string method="tracked_eigenvector_bisection";
    std::string status="UNKNOWN";
};

static Eigen::MatrixXcd hermitize_v30t(const Eigen::MatrixXcd& M) {
    return 0.5*(M + M.adjoint());
}

static bool closest_zero_eigenpair_from_matrix_v30t(
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
    Eigen::MatrixXcd H = hermitize_v30t(M);
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

static EigMinPointV30t eigmin_point_from_eval_v30t(const EvalFull& ef) {
    EigMinPointV30t p;
    p.i=ef.d.i; p.E=ef.d.Ecm; p.success=0; p.total_dim=ef.d.total_dim; p.proj_dim=ef.d.vdim;
    p.fv_rel_leak_F3=ef.d.fv_rel_leak_F3; p.fv_equiv_F3=ef.d.fv_equiv_F3; p.error=ef.d.error;
    if(!ef.d.success) return p;
    double eig=NAN, ae=NAN, herm=NAN, emin=NAN, emax=NAN; int idx=-1;
    if(!closest_zero_eigenpair_from_matrix_v30t(ef.projF3inv,eig,ae,idx,nullptr,herm,emin,emax)){
        p.error="EIGEN_MIN_FAILED"; return p;
    }
    p.success=1; p.eig_closest_zero=eig; p.eig_closest_abs=ae; p.eig_index=idx; p.herm_rel=herm; p.eig_min_alg=emin; p.eig_max_alg=emax; p.error="OK";
    return p;
}

static bool sign_flip_v30t(double a, double b) {
    return std::isfinite(a) && std::isfinite(b) && (a==0.0 || b==0.0 || a*b<0.0);
}

static bool eval_closest_eig_tracked_v30t(
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
    Eigen::MatrixXcd H = hermitize_v30t(ef.projF3inv);
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


static double g_eigen_zero_abs_tol_v30t = 1.0e-7;
static double g_eigen_sigma_cross_tol_v30t = 1.0e-5;
static double g_eigen_min_overlap_tol_v30t = 0.90;
static int    g_eigen_require_svd_cross_v30t = 1;

static bool eval_sigma_min_projected_v30t(
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

static EigZeroRecordV30t bisect_eigen_min_candidate_v30t(
        const EigMinCandidateV30t& c,
        const std::vector<int>& nnP,
        const std::string& irrep,
        const PhysicsParams& par,
        char debug,
        double Etol,
        int max_iter,
        int use_tracking) {
    EigZeroRecordV30t z;
    z.cand_index=c.cand_index; z.E_left=c.E_left; z.E_right=c.E_right; z.y_left0=c.y_left; z.y_right0=c.y_right;
    if(c.near_dimension_jump){ z.status="REJECT_DIMENSION_JUMP_BRACKET"; return z; }
    double left=c.E_left, right=c.E_right;
    double fl=NAN, fr=NAN, al=NAN, ar=NAN, ov=NAN, herm=NAN;
    Eigen::VectorXcd refvec;
    if(!eval_closest_eig_tracked_v30t(left,nnP,irrep,par,debug,nullptr,fl,al,&refvec,ov,herm) ||
       !eval_closest_eig_tracked_v30t(right,nnP,irrep,par,debug,(use_tracking?&refvec:nullptr),fr,ar,nullptr,ov,herm) ||
       !sign_flip_v30t(fl,fr)) {
        z.status="REJECT_ENDPOINTS_NO_LONGER_BRACKET"; return z;
    }
    double minov=std::isfinite(ov)?ov:1.0;
    int iter=0;
    for(iter=0; iter<max_iter && std::abs(right-left)>Etol; ++iter) {
        const double mid=0.5*(left+right);
        double fm=NAN, am=NAN, ovm=NAN, hm=NAN;
        if(!eval_closest_eig_tracked_v30t(mid,nnP,irrep,par,debug,(use_tracking?&refvec:nullptr),fm,am,nullptr,ovm,hm)){
            z.status="REJECT_MIDPOINT_EVAL_FAILED"; z.iter=iter; return z;
        }
        if(std::isfinite(ovm)) minov=std::min(minov,ovm);
        if(fl==0.0){ right=left; fr=fl; break; }
        if(fr==0.0){ left=right; fl=fr; break; }
        if(fl*fm<=0.0){ right=mid; fr=fm; }
        else { left=mid; fl=fm; }
    }
    const double E=0.5*(left+right);
    double yf=NAN, ay=NAN, ovf=NAN, hf=NAN;
    eval_closest_eig_tracked_v30t(E,nnP,irrep,par,debug,(use_tracking?&refvec:nullptr),yf,ay,nullptr,ovf,hf);
    z.E=E; z.y=yf; z.abs_y=ay; z.iter=iter; z.final_width=std::abs(right-left); z.min_overlap=std::isfinite(ovf)?std::min(minov,ovf):minov; z.final_herm_rel=hf;
    double sig=NAN, hs=NAN;
    const bool sigma_ok_eval = eval_sigma_min_projected_v30t(E,nnP,irrep,par,debug,sig,hs);
    z.sigma_min = sigma_ok_eval ? sig : std::numeric_limits<double>::quiet_NaN();
    if(sigma_ok_eval && std::isfinite(hs)) z.final_herm_rel = std::max(z.final_herm_rel, hs);

    const bool width_ok   = std::isfinite(z.final_width) && z.final_width <= Etol*1.01;
    const bool eig_ok     = std::isfinite(z.abs_y) && z.abs_y <= g_eigen_zero_abs_tol_v30t;
    const bool overlap_ok = (!use_tracking) || (!std::isfinite(g_eigen_min_overlap_tol_v30t)) || (std::isfinite(z.min_overlap) && z.min_overlap >= g_eigen_min_overlap_tol_v30t);
    const bool svd_ok     = (!g_eigen_require_svd_cross_v30t) || (sigma_ok_eval && std::isfinite(z.sigma_min) && z.sigma_min <= g_eigen_sigma_cross_tol_v30t);
    z.accepted = (width_ok && eig_ok && overlap_ok && svd_ok) ? 1 : 0;
    z.method = use_tracking ? "tracked_eigenvector_bisection_residual_svd_checked" : "instant_minabs_bisection_residual_svd_checked";
    if(z.accepted) z.status = "ACCEPT_EIGEN_MIN_BISECTION_RESIDUAL_SVD_CHECKED";
    else if(!width_ok) z.status = "REJECT_BISECTION_NOT_CONVERGED";
    else if(!eig_ok) z.status = "REJECT_FINAL_EIGEN_RESIDUAL_TOO_LARGE";
    else if(!overlap_ok) z.status = "REJECT_EIGENVECTOR_OVERLAP_TOO_SMALL";
    else if(!svd_ok) z.status = "REJECT_SVD_CROSSCHECK_FAILED";
    else z.status = "REJECT_UNKNOWN_VALIDATION_FAILURE";
    return z;
}

static std::vector<EigMinCandidateV30t> collect_eigen_min_signflips_v30t(
        const std::vector<EigMinPointV30t>& pts,
        const std::vector<double>& dim_jumps,
        double jump_guard) {
    std::vector<EigMinCandidateV30t> out;
    for(int i=0;i+1<(int)pts.size();++i){
        const auto& a=pts[i]; const auto& b=pts[i+1];
        if(!a.success || !b.success) continue;
        if(a.proj_dim!=b.proj_dim) continue;
        if(!sign_flip_v30t(a.eig_closest_zero,b.eig_closest_zero)) continue;
        EigMinCandidateV30t c; c.cand_index=(int)out.size(); c.i_left=i; c.i_right=i+1; c.E_left=a.E; c.E_right=b.E; c.y_left=a.eig_closest_zero; c.y_right=b.eig_closest_zero; c.proj_dim_left=a.proj_dim; c.proj_dim_right=b.proj_dim;
        double nearest=std::numeric_limits<double>::infinity();
        for(double je: dim_jumps){ nearest=std::min(nearest,std::min(std::abs(c.E_left-je),std::abs(c.E_right-je))); }
        c.near_dimension_jump=(nearest<=jump_guard)?1:0;
        c.status=c.near_dimension_jump?"REJECT_NEAR_DIMENSION_JUMP":"KEEP_SIGNFLIP";
        out.push_back(c);
    }
    return out;
}

static std::vector<EigZeroRecordV30t> refine_candidates_parallel_v30t(
        const std::vector<EigMinCandidateV30t>& cands,
        const std::vector<int>& nnP,
        const std::string& irrep,
        const PhysicsParams& par,
        char debug,
        double Etol,
        int max_iter,
        int use_tracking,
        int parallel_refine) {
    std::vector<EigZeroRecordV30t> out(cands.size());
    int done=0, nextpct=10;
    if(parallel_refine){
        #pragma omp parallel for schedule(dynamic,1)
        for(int i=0;i<(int)cands.size();++i){
            out[i]=bisect_eigen_min_candidate_v30t(cands[i],nnP,irrep,par,debug,Etol,max_iter,use_tracking);
            #pragma omp critical
            { ++done; progress_percent_log("eigmin-bisect", done, (int)cands.size(), nextpct, 10); }
        }
    } else {
        for(int i=0;i<(int)cands.size();++i){ out[i]=bisect_eigen_min_candidate_v30t(cands[i],nnP,irrep,par,debug,Etol,max_iter,use_tracking); ++done; progress_percent_log("eigmin-bisect", done, (int)cands.size(), nextpct, 10); }
    }
    int zi=0; for(auto& z: out) z.zero_index=zi++;
    return out;
}

static void write_v30t_outputs(
        const std::string& base,
        const std::vector<EigMinPointV30t>& pts,
        const std::vector<EigMinCandidateV30t>& cands,
        const std::vector<EigZeroRecordV30t>& zeros) {
    {
        std::ofstream f(base+"_eigen_min_coarse_grid.dat"); f<<std::setprecision(17);
        f << "# columns: i Ecm success total_dim proj_dim eig_closest_zero eig_closest_abs eig_index eig_min_alg eig_max_alg herm_projF3inv_rel fv_equiv_F3 fv_rel_leak_F3 error\n";
        for(const auto& p: pts) f<<p.i<<' '<<p.E<<' '<<p.success<<' '<<p.total_dim<<' '<<p.proj_dim<<' '<<p.eig_closest_zero<<' '<<p.eig_closest_abs<<' '<<p.eig_index<<' '<<p.eig_min_alg<<' '<<p.eig_max_alg<<' '<<p.herm_rel<<' '<<p.fv_equiv_F3<<' '<<p.fv_rel_leak_F3<<' '<<sanitize_error(p.error)<<"\n";
    }
    {
        std::ofstream f(base+"_eigen_min_signflip_candidates.dat"); f<<std::setprecision(17);
        f << "# columns: cand_index i_left i_right E_left E_right y_left y_right proj_dim_left proj_dim_right near_dimension_jump status\n";
        for(const auto& c: cands) f<<c.cand_index<<' '<<c.i_left<<' '<<c.i_right<<' '<<c.E_left<<' '<<c.E_right<<' '<<c.y_left<<' '<<c.y_right<<' '<<c.proj_dim_left<<' '<<c.proj_dim_right<<' '<<c.near_dimension_jump<<' '<<c.status<<"\n";
    }
    {
        std::ofstream f(base+"_eigen_min_bisected_zeros.dat"); f<<std::setprecision(17);
        f << "# columns: zero_index cand_index accepted Ecm eig_value abs_eig sigma_min final_width iter min_overlap final_herm_rel E_left E_right y_left0 y_right0 method status\n";
        for(const auto& z: zeros) f<<z.zero_index<<' '<<z.cand_index<<' '<<z.accepted<<' '<<z.E<<' '<<z.y<<' '<<z.abs_y<<' '<<z.sigma_min<<' '<<z.final_width<<' '<<z.iter<<' '<<z.min_overlap<<' '<<z.final_herm_rel<<' '<<z.E_left<<' '<<z.E_right<<' '<<z.y_left0<<' '<<z.y_right0<<' '<<z.method<<' '<<z.status<<"\n";
    }
    {
        std::ofstream f(base+"_eigen_min_strict_validated_levels.dat"); f<<std::setprecision(17);
        f << "# columns: level_index Ecm eig_value abs_eig sigma_min final_width iter min_overlap final_herm_rel method\n";
        int li=0;
        for(const auto& z: zeros) if(z.accepted) f<<li++<<' '<<z.E<<' '<<z.y<<' '<<z.abs_y<<' '<<z.sigma_min<<' '<<z.final_width<<' '<<z.iter<<' '<<z.min_overlap<<' '<<z.final_herm_rel<<' '<<z.method<<"\n";
    }
}

int main(int argc, char** argv) {
    if(argc!=2){ std::cerr<<"Usage:\n  "<<argv[0]<<" config/config_v30t_eigen_min_bisect_residual_checked_100_A2_zero_scan.in\n"; return 1; }
    const std::string input_file=argv[1];
    try{
        V30eScopedTimer total_timer("v30t total executable");
        PhysicsParams par; Options opt=read_options(input_file,par);
        opt.mom_labels = {"100_A2"}; // blind default requested: nnP=(0,0,1), irrep=A2
        #ifdef _OPENMP
        omp_set_num_threads(opt.threads);
        omp_set_max_active_levels(1);
        #endif
        Eigen::setNbThreads(1);
        std::filesystem::create_directories(opt.outdir);
        const double eigen_bisect_tol = get_double(read_kv_file(input_file), "eigen_bisect_tol", 1.0e-9);
        const int eigen_bisect_max_iter = get_int(read_kv_file(input_file), "eigen_bisect_max_iter", 80);
        const int eigen_use_tracking = get_int(read_kv_file(input_file), "eigen_use_tracking", 1);
        g_eigen_zero_abs_tol_v30t = get_double(read_kv_file(input_file), "eigen_zero_abs_tol", 1.0e-7);
        g_eigen_sigma_cross_tol_v30t = get_double(read_kv_file(input_file), "eigen_sigma_cross_tol", 1.0e-5);
        g_eigen_min_overlap_tol_v30t = get_double(read_kv_file(input_file), "eigen_min_overlap_tol", 0.90);
        g_eigen_require_svd_cross_v30t = get_int(read_kv_file(input_file), "eigen_require_svd_crosscheck", 1);
        stage_log("[stage 1/7] 100% config read; default irrep forced to 100_A2 with nnP=(0,0,1)");
        for(const std::string& lab: opt.mom_labels){
            MomentumIrrepSpec spec=parse_label(lab); std::vector<int> nnP={spec.nnP[0],spec.nnP[1],spec.nnP[2]};
            const std::string base=opt.outdir+"/"+opt.prefix;
            std::ofstream meta(base+"_metadata.txt"); meta<<std::setprecision(17);
            meta << "version = v30t_eigen_min_bisect_residual_checked_100_A2_zero_scan\n";
            meta << "input_file = "<<input_file<<"\nlabel = "<<lab<<"\nnnP = 0 0 1\nirrep = A2\n";
            meta << "E0 = "<<opt.E0<<"\nE1 = "<<opt.E1<<"\nN = "<<opt.N<<"\nthreads = "<<opt.threads<<"\n";
            meta << "zero_method = closest_to_zero_eigenvalue_signflip_then_bisection_with_final_residual_overlap_and_svd_crosscheck\n";
            meta << "matrix = projected_F3inv = inverse(V^dagger F3 V), corrected Cartesian ell=1 projector\n";
            meta << "eigen_matrix_treatment = hermitize projected matrix before SelfAdjointEigenSolver\n";
            meta << "eigen_bisect_tol = "<<eigen_bisect_tol<<"\neigen_bisect_max_iter = "<<eigen_bisect_max_iter<<"\neigen_use_tracking = "<<eigen_use_tracking<<"\n";
            meta << "note = This is a blind 100_A2 test. It does not use determinant sign flips, SVD minima, or reference energies. Coarse grid is evaluated in parallel; Vsel/projector is cached by active basis signature and reused until shell changes. Bisection uses eigenvector-overlap tracking by default and final acceptance requires small eigenvalue residual plus SVD cross-check.\n";

            std::cout << "[v30t] blind eigen-min bisection scan for "<<lab<<"\n";
            stage_log("[stage 2/7] building coarse projected matrices in parallel");
            const double t0=wall_seconds_now();
            std::vector<EvalFull> fullgrid = evaluate_full_grid_parallel(opt, nnP, spec.irrep, par);
            stage_log("[stage 2/7] 100% coarse matrix cache done, dt=" + std::to_string(wall_seconds_now()-t0) + " s");

            stage_log("[stage 3/7] extracting closest-to-zero projected eigenvalue on coarse grid");
            std::vector<EigMinPointV30t> pts(fullgrid.size());
            int nextpct=10;
            for(int i=0;i<(int)fullgrid.size();++i){ pts[i]=eigmin_point_from_eval_v30t(fullgrid[i]); progress_percent_log("eigmin-coarse", i+1, (int)fullgrid.size(), nextpct, 10); }
            stage_log("[stage 3/7] 100% eigen-min coarse extraction done");

            std::vector<double> dim_jumps;
            for(size_t i=0;i+1<fullgrid.size();++i) if(fullgrid[i].d.success && fullgrid[i+1].d.success && fullgrid[i].d.vdim!=fullgrid[i+1].d.vdim) dim_jumps.push_back(0.5*(fullgrid[i].d.Ecm+fullgrid[i+1].d.Ecm));
            meta << "dimension_jump_count = "<<dim_jumps.size()<<"\n";
            meta << "projector_cache_entries = " << g_projector_cache_v30q.size() << "\n";
            meta << "projector_cache_hits = " << g_projector_cache_hits_v30q << "\n";
            meta << "projector_cache_misses = " << g_projector_cache_misses_v30q << "\n";

            stage_log("[stage 4/7] collecting closest-eigenvalue sign flips away from dimension jumps");
            auto cands=collect_eigen_min_signflips_v30t(pts,dim_jumps,opt.dimension_jump_guard);
            int kept=0; for(const auto& c:cands) if(!c.near_dimension_jump) ++kept;
            meta << "eigen_min_raw_signflip_candidates = "<<cands.size()<<"\n";
            meta << "eigen_min_candidates_kept_for_bisection = "<<kept<<"\n";
            stage_log("[stage 4/7] 100% sign-flip collection done, raw="+std::to_string(cands.size())+", kept="+std::to_string(kept));

            stage_log("[stage 5/7] bisection refinement of eigen-min sign flips");
            const double tr0=wall_seconds_now();
            auto zeros=refine_candidates_parallel_v30t(cands,nnP,spec.irrep,par,opt.debug,eigen_bisect_tol,eigen_bisect_max_iter,eigen_use_tracking,opt.parallel_refine);
            int accepted=0; for(const auto& z: zeros) if(z.accepted) ++accepted;
            meta << "eigen_min_bisected_records = "<<zeros.size()<<"\n";
            meta << "eigen_min_strict_accepted_levels = "<<accepted<<"\n";
            stage_log("[stage 5/7] 100% bisection done, accepted="+std::to_string(accepted)+", dt="+std::to_string(wall_seconds_now()-tr0)+" s");

            stage_log("[stage 6/7] writing outputs");
            write_v30t_outputs(base,pts,cands,zeros);
            stage_log("[stage 6/7] 100% outputs written");
            stage_log("[stage 7/7] done");
        }
    } catch(const std::exception& e){ std::cerr << "[v30t-error] " << e.what() << std::endl; return 2; }
    return 0;
}
