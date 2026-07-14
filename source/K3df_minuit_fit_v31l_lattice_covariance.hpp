#ifndef K3DF_MINUIT_FIT_V31L_LATTICE_COVARIANCE_HPP
#define K3DF_MINUIT_FIT_V31L_LATTICE_COVARIANCE_HPP

// v31l Kdf,3 fitter with corrected Cartesian ell=1 projector, F3inv-only cache, and lattice covariance.
//
// Core QC:
//   QC_proj(E) = F3inv_proj(E) + Kdf3_proj(E)
// where
//   F3inv_proj(E) = inverse( Vsel^\dagger F3(E) Vsel )
//   Kdf3_proj(E) = Vsel^\dagger Kdf3(E) Vsel
//
// The projector/Vsel is the corrected Cartesian-l1 projector from v30m/v30r.
// The expensive energy-grid cache is built once in OpenMP:
//   Only F3inv_proj(E) is stored for every irrep and energy. The basis/configuration and Vsel needed to project Kdf3 are kept as metadata, but no projected Kdf3 matrices are cached.  During Minuit FCN calls, Kdf3 is rebuilt and projected for each cached energy; then QC_proj is assembled and diagonalized.
// Zero search:
//   selected_y(E) = eigenvalue of QC_proj closest to zero
//   left_y > 0, right_y < 0 => ZERO midpoint
//   left_y < 0, right_y > 0 => POLE midpoint
// no refinement in the default v31l algorithm. Lattice energies and full covariance are loaded through lattice_data_covariance_cpp.hpp.
//
// This header intentionally includes the v30r diagnostic implementation with
// its main renamed, so that all corrected-projector and F3-building functions
// are reused in one translation unit.

#if !defined(V31L_DISABLE_MINUIT) && !defined(V31R_DISABLE_MINUIT) && !defined(V31T_DISABLE_MINUIT) && !defined(V31S_DISABLE_MINUIT) && !defined(V31T_DISABLE_MINUIT)
#include <Minuit2/FCNBase.h>
#include <Minuit2/FunctionMinimum.h>
#include <Minuit2/MnHesse.h>
#include <Minuit2/MnMigrad.h>
#include <Minuit2/MnPrint.h>
#include <Minuit2/MnUserParameters.h>
#endif

#define main v30r_disabled_main_for_v31l_fit
#include "debug_v30r_svd_validated_cartesian_l1_100_A2_zero_scan.cpp"
#undef main

#include "lattice_data_covariance_cpp.hpp"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace k3df_fit_v31l {

using VecD = std::vector<double>;
using MatrixD = Eigen::MatrixXd;

struct K3dfParameters {
    double K3iso0 = 0.0;
    double K3iso1 = 0.0;
    double K3B = 0.0;
    double K3E = 0.0;
};

struct TargetLevel {
    std::string label;
    int state = -1;
    double Ecm = std::numeric_limits<double>::quiet_NaN();
    double err = 1.0;
    std::array<int,3> nP{{0,0,0}};
    double Lbyas = std::numeric_limits<double>::quiet_NaN();
};

struct FitSettings {
    std::vector<std::string> list_of_mom = {"100_A2","110_A2"};
    double Lval = 20.0;
    double xival = 3.444;
    double scan_E0 = 0.2631;
    double scan_E1 = 0.36;
    int coarseN = 1501;
    int omp_threads = 18;
    char debug = 'n';

    // physics
    double atmpi = 0.06906;
    double atmK = 0.09698;
    double eta_1 = 1.0;
    double eta_2 = 0.5;
    double alpha = 0.5;
    double epsilon_h = 0.0;
    double max_shell_num = 20.0;
    double tolerance = 1.0e-12;
    int parity = -1;
    double eig_tol = 0.05;
    double norm_tol = 1.0e-12;
    double proj_tol = 1.0e-10;
    bool Q0norm = true;
    bool sort_orbit_flag = false;
    std::vector<int> waves_vec_1 = {0,1};
    std::vector<int> waves_vec_2 = {0};

    // fit parameters
    K3dfParameters guess{0.0,0.0,0.0,0.0};
    K3dfParameters step{10.0,10.0,10.0,10.0};
    bool use_parameter_limits = false;
    double param_lower = -1.0e8;
    double param_upper = 1.0e8;

    // lattice data/covariance input. The default path is provided by
    // lattice_data_covariance_cpp.hpp when masses_path is empty.
    int use_lattice_covariance = 1;
    std::string ensemble = "szscl21_24_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265";
    double energy_cutoff = 0.335;
    int max_state = 8;
    std::string masses_path = "";
    bool print_found_files = false;

    // Fallback/manual target file if use_lattice_covariance=0.
    std::string target_levels_file = "config/target_levels_example.dat";

    // Chi-square mode:
    // raw_cov_inv: (Edata-Emodel)^T Cov^{-1} (Edata-Emodel)
    // corr_inv: normalized residuals with Corr^{-1}
    // diagonal: sum_i ((Edata-Emodel)/sigma_i)^2
    std::string chi_square_mode = "raw_cov_inv";
    double failure_penalty = 1.0e100;

    // behavior
    bool build_cache_once = true;
    bool write_cache_grid = false; // v31l default: keep RAM cache only; do not write diagnostic cache grids unless requested.
    bool write_bestfit_qc_grid = true;

    // Optional local binary cache. This stores the actual F3inv_proj matrices plus
    // the basis metadata needed to rebuild/project Kdf,3.
    bool save_binary_f3inv_cache = false;
    bool load_binary_f3inv_cache = false;
    std::string binary_f3inv_cache_file = "cache/v31l_F3inv_projected_cache.bin";

    // v31l default: use linear interpolation inside a sign-flip bracket, rather
    // than the discontinuous midpoint. Set to "midpoint" for the old behavior.
    std::string zero_energy_mode = "linear";

    bool print_each_fcn_eval = true;
    int print_every_fcn_eval = 1;

    std::string output_dir = "output_v31l_K3df_fit_lattice_covariance_closest_eig";
    std::string output_tag = "debug_v31l_K3df_fit_lattice_covariance_closest_eig";
};

struct ProjectedQCCacheEntry {
    std::string label;
    MomentumIrrepSpec spec;
    int i = -1;
    double Ecm = std::numeric_limits<double>::quiet_NaN();
    double En = std::numeric_limits<double>::quiet_NaN();
    int success = 0;
    int total_dim = 0;
    int proj_dim = 0;

    // v31l diagnostic mode supports both conventions:
    //   old/recent: F3inv_proj = inverse(V^dagger F3 V)
    //   new test:   F3inv_full = inverse(F3), then QC_proj = V^dagger (F3inv_full + Kdf3) V
    Eigen::MatrixXcd F3inv_proj;
    Eigen::MatrixXcd F3inv_full;

    // Minimal metadata needed to rebuild and project Kdf3 during each Minuit FCN call.
    std::vector<std::vector<comp>> plm_config;
    std::vector<std::vector<comp>> klm_config;
    std::vector<comp> total_P;
    comp En_c = comp(std::numeric_limits<double>::quiet_NaN(),0.0);
    Eigen::MatrixXcd Vsel;

    double projector_idem = std::numeric_limits<double>::quiet_NaN();
    double projector_closure = std::numeric_limits<double>::quiet_NaN();
    double F3_equiv = std::numeric_limits<double>::quiet_NaN();
    double F3_leak = std::numeric_limits<double>::quiet_NaN();

    std::string error = "OK";
};

struct IrrepCache {
    std::string label;
    MomentumIrrepSpec spec;
    std::vector<ProjectedQCCacheEntry> grid;
};

struct ZeroPole {
    std::string label;
    std::string kind; // ZERO or POLE
    int i_left = -1;
    int i_right = -1;
    double E_mid = std::numeric_limits<double>::quiet_NaN();
    double E_left = std::numeric_limits<double>::quiet_NaN();
    double E_right = std::numeric_limits<double>::quiet_NaN();
    double y_left = std::numeric_limits<double>::quiet_NaN();
    double y_right = std::numeric_limits<double>::quiet_NaN();
    int eig_index_left = -1;
    int eig_index_right = -1;
};

struct FitResult {
    K3dfParameters best;
    K3dfParameters errors;
    double chi2 = std::numeric_limits<double>::quiet_NaN();
    int ndata = 0;
    int npar = 4;
    int ndof = 0;
    double chi2_dof = std::numeric_limits<double>::quiet_NaN();
    bool valid = false;
    std::map<std::string, std::vector<ZeroPole>> zero_poles_by_label;
    std::vector<TargetLevel> targets;
    std::vector<double> model_levels;
    MatrixD data_covariance;
    MatrixD data_correlation;
    MatrixD parameter_covariance;
    MatrixD parameter_correlation;
};

inline std::string clean_label(std::string s) {
    for(char& c: s) if(std::isspace((unsigned char)c)) c='_';
    return s;
}

inline PhysicsParams make_base_physics(const FitSettings& s) {
    PhysicsParams par;
    par.atmpi = s.atmpi;
    par.atmK = s.atmK;
    par.eta_1 = s.eta_1;
    par.eta_2 = s.eta_2;
    par.alpha = s.alpha;
    par.epsilon_h = s.epsilon_h;
    par.max_shell_num = s.max_shell_num;
    par.tolerance = s.tolerance;
    par.xi = s.xival;
    par.Lbyas = s.Lval;
    par.Q0norm = s.Q0norm;
    par.sort_orbit_flag = s.sort_orbit_flag;
    par.parity = s.parity;
    par.eig_tol = s.eig_tol;
    par.norm_tol = s.norm_tol;
    par.proj_tol = s.proj_tol;
    par.omp_threads = s.omp_threads;
    par.waves_vec_1 = s.waves_vec_1;
    par.waves_vec_2 = s.waves_vec_2;
    // The basis cache is independent of these values, but keep them finite.
    par.K3iso = {comp(0.0,0.0), comp(0.0,0.0)};
    par.K3B_par = comp(0.0,0.0);
    par.K3E_par = comp(0.0,0.0);
    return par;
}

inline std::vector<TargetLevel> read_target_levels(const std::string& path) {
    std::ifstream in(path);
    if(!in) throw std::runtime_error("Could not open target_levels_file: " + path);
    std::vector<TargetLevel> rows;
    std::string line;
    int lineno=0;
    while(std::getline(in,line)) {
        ++lineno;
        line = trim(strip_inline_comment(line));
        if(line.empty()) continue;
        std::istringstream is(line);
        TargetLevel r;
        if(!(is >> r.label >> r.state >> r.Ecm >> r.err)) {
            throw std::runtime_error("Bad target level line " + std::to_string(lineno) + " in " + path);
        }
        if(!(r.err>0.0) || !std::isfinite(r.err)) {
            throw std::runtime_error("Bad target error on line " + std::to_string(lineno));
        }
        rows.push_back(r);
    }
    if(rows.empty()) throw std::runtime_error("No target levels found in " + path);
    return rows;
}


inline std::array<int,3> canonical_shell_momentum(std::array<int,3> p) {
    for(int& x: p) x = std::abs(x);
    std::sort(p.begin(), p.end());
    return p;
}

inline bool same_momentum_shell_v31l(const std::array<int,3>& a, const std::array<int,3>& b) {
    return canonical_shell_momentum(a) == canonical_shell_momentum(b);
}

inline std::vector<int> infer_row_spec_index_from_covariance_order_v31l(
        const std::vector<std::string>& list_of_mom,
        const std::vector<std::array<int,3>>& nP_list) {
    std::vector<MomentumIrrepSpec> specs;
    for(const auto& label: list_of_mom) specs.push_back(parse_label(label));
    std::vector<int> row_spec_index(nP_list.size(), -1);
    std::size_t row = 0;
    for(std::size_t s=0; s<specs.size() && row<nP_list.size(); ++s) {
        while(row<nP_list.size() && same_momentum_shell_v31l(nP_list[row], specs[s].nnP)) {
            row_spec_index[row] = int(s);
            ++row;
        }
    }
    if(row != nP_list.size()) {
        std::ostringstream os;
        os << "Could not map all covariance rows to list_of_mom labels. Mapped "
           << row << " of " << nP_list.size()
           << ". Check list_of_mom ordering, energy_cutoff, max_state, and masses_path.";
        throw std::runtime_error(os.str());
    }
    return row_spec_index;
}

inline MatrixD symmetric_pseudoinverse_v31l(const MatrixD& A, double rcond=1.0e-12) {
    if(A.rows()!=A.cols()) throw std::runtime_error("symmetric_pseudoinverse_v31l requires square matrix");
    if(A.rows()==0) return MatrixD(0,0);
    MatrixD S = 0.5*(A + A.transpose());
    Eigen::SelfAdjointEigenSolver<MatrixD> es(S);
    if(es.info()!=Eigen::Success) throw std::runtime_error("Eigen decomposition failed in covariance pseudoinverse");
    const auto evals = es.eigenvalues();
    const MatrixD evecs = es.eigenvectors();
    const double max_abs = evals.cwiseAbs().maxCoeff();
    const double cutoff = std::max(rcond*max_abs, 1.0e-300);
    MatrixD Dinv = MatrixD::Zero(A.rows(),A.cols());
    for(int i=0;i<evals.size();++i) if(std::abs(evals(i)) > cutoff) Dinv(i,i)=1.0/evals(i);
    return evecs * Dinv * evecs.transpose();
}

inline MatrixD covariance_to_correlation_v31l(const MatrixD& cov) {
    MatrixD corr = MatrixD::Zero(cov.rows(),cov.cols());
    for(int i=0;i<cov.rows();++i) {
        for(int j=0;j<cov.cols();++j) {
            if(cov(i,i)>0.0 && cov(j,j)>0.0) corr(i,j)=cov(i,j)/std::sqrt(cov(i,i)*cov(j,j));
        }
    }
    return corr;
}

inline std::tuple<std::vector<TargetLevel>,MatrixD,MatrixD>
load_targets_and_covariance_v31l(const FitSettings& settings) {
    if(settings.use_lattice_covariance) {
        std::filesystem::path p = settings.masses_path.empty()
            ? lattice_covariance::default_szscl21_mass_path()
            : std::filesystem::path(settings.masses_path);

        auto [states_avg, states_err, nP_list, state_no, L_list, cov, corr]
            = covariance_between_states_szscl21_based(
                settings.ensemble,
                settings.Lval,
                settings.xival,
                settings.energy_cutoff,
                settings.list_of_mom,
                settings.max_state,
                p,
                settings.print_found_files
            );

        if(states_avg.empty()) {
            throw std::runtime_error("No lattice states passed energy_cutoff/file selection.");
        }

        std::vector<int> row_spec = infer_row_spec_index_from_covariance_order_v31l(settings.list_of_mom,nP_list);
        std::vector<MomentumIrrepSpec> specs;
        for(const auto& label: settings.list_of_mom) specs.push_back(parse_label(label));

        std::vector<TargetLevel> targets(states_avg.size());
        for(std::size_t i=0;i<states_avg.size();++i) {
            const MomentumIrrepSpec& spec = specs.at(std::size_t(row_spec[i]));
            targets[i].label = spec.label;
            targets[i].state = state_no[i];
            targets[i].Ecm = states_avg[i];
            targets[i].err = states_err[i];
            targets[i].nP = nP_list[i];
            targets[i].Lbyas = (i<L_list.size()) ? L_list[i] : settings.Lval;
        }
        return {targets,cov,corr};
    }

    std::vector<TargetLevel> targets = read_target_levels(settings.target_levels_file);
    MatrixD cov = MatrixD::Zero(targets.size(),targets.size());
    for(int i=0;i<cov.rows();++i) {
        const double e = targets[std::size_t(i)].err;
        cov(i,i)=e*e;
        targets[std::size_t(i)].Lbyas = settings.Lval;
        try { targets[std::size_t(i)].nP = parse_label(targets[std::size_t(i)].label).nnP; } catch(...) {}
    }
    MatrixD corr = covariance_to_correlation_v31l(cov);
    return {targets,cov,corr};
}

inline double chi_square_v31l(
        const std::vector<TargetLevel>& targets,
        const std::vector<double>& model,
        const MatrixD& covariance_mat,
        const MatrixD& correlation_mat,
        const std::string& mode,
        double failure_penalty) {
    const int n = int(targets.size());
    if(int(model.size()) != n) return failure_penalty;
    for(int i=0;i<n;++i) if(!(model[std::size_t(i)]>0.0) || !std::isfinite(model[std::size_t(i)])) return failure_penalty;

    if(mode == "diagonal") {
        double chi2=0.0;
        for(int i=0;i<n;++i) {
            const double sig = targets[std::size_t(i)].err;
            if(!(sig>0.0) || !std::isfinite(sig)) return failure_penalty;
            const double r = (targets[std::size_t(i)].Ecm - model[std::size_t(i)])/sig;
            chi2 += r*r;
        }
        return std::isfinite(chi2) ? chi2 : failure_penalty;
    }

    Eigen::VectorXd r(n);
    if(mode == "corr_inv") {
        for(int i=0;i<n;++i) {
            const double sig = targets[std::size_t(i)].err;
            if(!(sig>0.0) || !std::isfinite(sig)) return failure_penalty;
            r(i) = (targets[std::size_t(i)].Ecm - model[std::size_t(i)])/sig;
        }
        MatrixD W = symmetric_pseudoinverse_v31l(correlation_mat);
        const double chi2 = (r.transpose()*W*r)(0,0);
        return std::isfinite(chi2) ? chi2 : failure_penalty;
    }

    // default raw_cov_inv
    for(int i=0;i<n;++i) r(i) = targets[std::size_t(i)].Ecm - model[std::size_t(i)];
    MatrixD W = symmetric_pseudoinverse_v31l(covariance_mat);
    const double chi2 = (r.transpose()*W*r)(0,0);
    return std::isfinite(chi2) ? chi2 : failure_penalty;
}

#ifndef V31L_DISABLE_MINUIT
inline MatrixD minuit_covariance_to_eigen_v31l(const ROOT::Minuit2::FunctionMinimum& min, int npar) {
    MatrixD cov = MatrixD::Zero(npar,npar);
    try {
        const auto& m = min.UserCovariance();
        for(int i=0;i<npar;++i) for(int j=0;j<npar;++j) cov(i,j)=m(i,j);
    } catch(...) {}
    return cov;
}
#endif


inline Eigen::MatrixXcd hermitize(const Eigen::MatrixXcd& M) {
    return 0.5*(M + M.adjoint());
}

inline bool closest_zero_eigenvalue(const Eigen::MatrixXcd& M, double& eig, int& eig_index, double& herm_rel) {
    if(M.rows()==0 || M.cols()==0 || M.rows()!=M.cols() || !M.allFinite()) return false;
    const double nrm = M.norm();
    herm_rel = (nrm > 0.0) ? (M-M.adjoint()).norm()/nrm : 0.0;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(hermitize(M), Eigen::EigenvaluesOnly);
    if(es.info()!=Eigen::Success || es.eigenvalues().size()==0) return false;
    double best = std::numeric_limits<double>::infinity();
    int best_i = -1;
    for(int k=0; k<es.eigenvalues().size(); ++k) {
        const double a = std::abs(es.eigenvalues()[k]);
        if(std::isfinite(a) && a < best) { best=a; best_i=k; }
    }
    if(best_i < 0) return false;
    eig = es.eigenvalues()[best_i];
    eig_index = best_i;
    return std::isfinite(eig);
}

inline Eigen::MatrixXcd make_K3_projected(
        const std::vector<std::vector<comp>>& plm_config,
        const std::vector<std::vector<comp>>& klm_config,
        const std::vector<comp>& total_P,
        const comp& En_c,
        const PhysicsParams& par,
        const Eigen::MatrixXcd& Vsel,
        double Kiso0, double Kiso1, double KB, double KE,
        char debug) {
    const int N = int(plm_config[0].size() + klm_config[0].size());
    Eigen::MatrixXcd K3(N,N);
    std::vector<comp> Kiso = {comp(Kiso0,0.0), comp(Kiso1,0.0)};
    k3_2plus1::K3mat_2plus1(
        K3, En_c, plm_config, klm_config, total_P,
        par.atmK, par.atmpi, Kiso, comp(KB,0.0), comp(KE,0.0), debug
    );
    return Vsel.adjoint() * K3 * Vsel;
}

inline ProjectedQCCacheEntry build_cache_entry(
        int i,
        double Ecm,
        const MomentumIrrepSpec& spec,
        const FitSettings& settings,
        const PhysicsParams& par,
        char debug) {
    ProjectedQCCacheEntry out;
    out.label = spec.label;
    out.spec = spec;
    out.i = i;
    out.Ecm = Ecm;
    try {
        const std::vector<int> nnP_vec = {spec.nnP[0], spec.nnP[1], spec.nnP[2]};
        const comp pi = std::acos(-1.0);
        const double L = par.L();
        const comp twopibyL = comp(2.0,0.0)*pi/comp(L,0.0);
        std::vector<comp> total_P(3), nnP_config(3);
        for(int a=0; a<3; ++a) { total_P[a]=twopibyL*double(nnP_vec[a]); nnP_config[a]=comp(nnP_vec[a],0.0); }
        const comp Ecm_c(Ecm,0.0);
        const comp En_c = Ecm_to_E(Ecm_c,total_P);
        out.En = En_c.real();
        out.En_c = En_c;
        out.total_P = total_P;

        std::vector<std::vector<comp>> plm_config(5), klm_config(5);
        std::vector<std::vector<int>> np_config(5), nk_config(5);
        config_maker_4_momentum_first(plm_config,np_config,par.waves_vec_1,En_c,total_P,par.atmK,par.atmK,par.atmpi,L,par.epsilon_h,par.max_shell_num,par.tolerance);
        config_maker_4_momentum_first(klm_config,nk_config,par.waves_vec_2,En_c,total_P,par.atmpi,par.atmK,par.atmK,L,par.epsilon_h,par.max_shell_num,par.tolerance);
        out.plm_config = plm_config;
        out.klm_config = klm_config;

        const int A = int(plm_config[0].size());
        const int B = int(klm_config[0].size());
        const int N = A+B;
        out.total_dim = N;
        if(N <= 0) { out.error="EMPTY_CONFIG"; return out; }

        Eigen::MatrixXcd F2(N,N), G(N,N), K2inv(N,N);
        F2_2plus1_mat(F2,En_c,plm_config,klm_config,total_P,par.atmK,par.atmpi,L,par.alpha,par.epsilon_h,par.max_shell_num,par.Q0norm);
        K2inv_EREord2_2plus1_mat(K2inv,par.eta_1,par.eta_2,par.scatter_params_1,par.scatter_params_2,En_c,plm_config,klm_config,total_P,par.atmK,par.atmpi,par.epsilon_h,L);
        G_2plus1_mat(G,En_c,plm_config,klm_config,total_P,par.atmK,par.atmpi,L,par.alpha,par.epsilon_h,par.max_shell_num,par.Q0norm);

        const Eigen::MatrixXcd H = K2inv + F2 + G;
        Eigen::PartialPivLU<Eigen::MatrixXcd> luH(H);
        Eigen::MatrixXcd X1 = luH.solve(F2);
        Eigen::MatrixXcd F3 = (F2/comp(3.0,0.0)) - F2*X1;

        CachedProjectorV30q pcache = get_projector_cached_v30q(plm_config,np_config,klm_config,nk_config,spec.irrep,nnP_config,par);
        Eigen::MatrixXcd Vsel = pcache.Vsel;
        out.Vsel = Vsel;
        out.proj_dim = int(Vsel.cols());
        out.projector_idem = pcache.P_I_idem_res;
        out.projector_closure = pcache.fv_rep_closure_best;
        if(out.proj_dim <= 0) { out.error="ZERO_PROJECTED_DIM"; return out; }

        out.F3_equiv = fvproj_v30j::max_equivariance_over_little_group(
            F3, plm_config, np_config, klm_config, nk_config, nnP_config, par.parity,
            [](){ fvproj_v30j::Convention c; c.index=0; c.use_inverse_momentum=false; c.use_inverse_D=false; c.transpose_D=false; c.order_WS=false; c.parity_mode=1; return c; }()
        );

        const Eigen::MatrixXcd Pproj = Vsel*Vsel.adjoint();
        const Eigen::MatrixXcd MV = F3*Vsel;
        out.F3_leak = ((Eigen::MatrixXcd::Identity(N,N)-Pproj)*MV).norm()/std::max(MV.norm(),1.0e-300);

        // Cache both inverse conventions, scaled for numerical stability.
        // 1) old/recent convention:
        //      F3inv_proj = inverse(V^dagger F3 V)
        // 2) requested diagnostic convention:
        //      F3inv_full = inverse(F3), then QC is built in full space and projected later.
        double f3_scale = 0.0;
        for(int r=0; r<F3.rows(); ++r) for(int c=0; c<F3.cols(); ++c) {
            const double a = std::abs(F3(r,c));
            if(std::isfinite(a) && a > f3_scale) f3_scale = a;
        }
        if(!(f3_scale > 0.0) || !std::isfinite(f3_scale)) { out.error="BAD_F3_SCALE"; return out; }

        Eigen::MatrixXcd F3_scaled = F3 / f3_scale;
        Eigen::PartialPivLU<Eigen::MatrixXcd> luFullF3(F3_scaled);
        out.F3inv_full = luFullF3.solve(Eigen::MatrixXcd::Identity(N,N));
        out.F3inv_full *= f3_scale;

        Eigen::MatrixXcd projF3_scaled = Vsel.adjoint()*F3_scaled*Vsel;
        Eigen::PartialPivLU<Eigen::MatrixXcd> luProjF3(projF3_scaled);
        out.F3inv_proj = luProjF3.solve(Eigen::MatrixXcd::Identity(out.proj_dim,out.proj_dim));
        out.F3inv_proj *= f3_scale;

        if(!out.F3inv_proj.allFinite() || !out.F3inv_full.allFinite() || !out.Vsel.allFinite()) {
            out.error = "NONFINITE_F3INV_CACHE_OR_VSEL";
            return out;
        }
        out.success = 1;
        out.error = "OK";
    } catch(const std::exception& e) {
        out.success = 0;
        out.error = e.what();
    }
    return out;
}

inline std::vector<IrrepCache> build_F3inv_only_cache(const FitSettings& settings) {
    PhysicsParams par = make_base_physics(settings);
    std::vector<IrrepCache> caches;
    caches.reserve(settings.list_of_mom.size());

    #ifdef _OPENMP
    omp_set_num_threads(settings.omp_threads);
    omp_set_max_active_levels(1);
    #endif
    Eigen::setNbThreads(1);

    std::cout << "[v31l-cache] [--------------------] 0% begin F3inv-only projected cache build\n";
    int label_done = 0;
    for(const auto& lab: settings.list_of_mom) {
        MomentumIrrepSpec spec = parse_label(lab);
        IrrepCache ic;
        ic.label = lab;
        ic.spec = spec;
        ic.grid.resize(settings.coarseN);

        std::cout << "[v31l-cache] building label " << lab << " with N=" << settings.coarseN << "\n";
        int done=0, nextpct=10;
        #pragma omp parallel for schedule(dynamic,1)
        for(int i=0; i<settings.coarseN; ++i) {
            const double t = (settings.coarseN==1) ? 0.0 : double(i)/double(settings.coarseN-1);
            const double E = settings.scan_E0 + t*(settings.scan_E1-settings.scan_E0);
            ic.grid[i] = build_cache_entry(i,E,spec,settings,par,settings.debug);
            #pragma omp critical
            { ++done; progress_percent_log("v31l-cache-"+lab, done, settings.coarseN, nextpct, 10); }
        }
        caches.push_back(std::move(ic));
        ++label_done;
        std::cout << "[v31l-cache] labels complete " << label_done << "/" << settings.list_of_mom.size() << "\n";
    }
    std::cout << "[v31l-cache] [####################] 100% F3inv-only projected cache complete\n";
    return caches;
}

inline Eigen::MatrixXcd assemble_QC(const ProjectedQCCacheEntry& e, const K3dfParameters& p, const PhysicsParams& par, char debug) {
    if(!e.success) return Eigen::MatrixXcd();
    Eigen::MatrixXcd K3_proj = make_K3_projected(
        e.plm_config,
        e.klm_config,
        e.total_P,
        e.En_c,
        par,
        e.Vsel,
        p.K3iso0,
        p.K3iso1,
        p.K3B,
        p.K3E,
        debug
    );
    return e.F3inv_proj + K3_proj;
}

struct QCPointValue {
    int success=0;
    double Ecm=std::numeric_limits<double>::quiet_NaN();
    double y=std::numeric_limits<double>::quiet_NaN();
    int eig_index=-1;
    double herm_rel=std::numeric_limits<double>::quiet_NaN();
};

inline QCPointValue point_value(const ProjectedQCCacheEntry& e, const K3dfParameters& p, const PhysicsParams& par, char debug) {
    QCPointValue v;
    v.Ecm = e.Ecm;
    if(!e.success) return v;
    Eigen::MatrixXcd QC = assemble_QC(e,p,par,debug);
    double y=NAN, herm=NAN; int idx=-1;
    if(!closest_zero_eigenvalue(QC,y,idx,herm)) return v;
    v.success=1; v.y=y; v.eig_index=idx; v.herm_rel=herm;
    return v;
}

inline std::vector<ZeroPole> find_zero_poles_from_cache(const IrrepCache& ic, const K3dfParameters& p, const PhysicsParams& par, char debug, const std::string& zero_energy_mode, std::vector<QCPointValue>* grid_out=nullptr) {
    std::vector<QCPointValue> vals(ic.grid.size());
    for(std::size_t i=0; i<ic.grid.size(); ++i) vals[i] = point_value(ic.grid[i],p,par,debug);
    if(grid_out) *grid_out = vals;

    std::vector<ZeroPole> out;
    for(int i=0; i+1<(int)vals.size(); ++i) {
        const auto& L = vals[i]; const auto& R = vals[i+1];
        if(!L.success || !R.success) continue;
        if(ic.grid[i].proj_dim != ic.grid[i+1].proj_dim) continue;
        if(!std::isfinite(L.y) || !std::isfinite(R.y)) continue;
        if(L.y==0.0 || R.y==0.0 || L.y*R.y < 0.0) {
            ZeroPole z;
            z.label=ic.label; z.i_left=i; z.i_right=i+1;
            z.E_left=L.Ecm; z.E_right=R.Ecm;
            z.E_mid=0.5*(L.Ecm+R.Ecm);
            if(zero_energy_mode == "linear" && std::abs(R.y-L.y) > 1.0e-300) {
                const double E_lin = L.Ecm - L.y*(R.Ecm-L.Ecm)/(R.y-L.y);
                if(std::isfinite(E_lin) && E_lin >= std::min(L.Ecm,R.Ecm) && E_lin <= std::max(L.Ecm,R.Ecm)) {
                    z.E_mid = E_lin;
                }
            }
            z.y_left=L.y; z.y_right=R.y; z.eig_index_left=L.eig_index; z.eig_index_right=R.eig_index;
            if(L.y > 0.0 && R.y < 0.0) z.kind="ZERO";
            else if(L.y < 0.0 && R.y > 0.0) z.kind="POLE";
            else z.kind="ZERO_ENDPOINT";
            out.push_back(z);
        }
    }
    return out;
}

inline std::map<std::string,std::vector<ZeroPole>> find_all_zero_poles(
        const std::vector<IrrepCache>& caches,
        const K3dfParameters& p,
        const PhysicsParams& par,
        char debug,
        const std::string& zero_energy_mode) {
    std::map<std::string,std::vector<ZeroPole>> m;
    for(const auto& ic: caches) {
        auto z = find_zero_poles_from_cache(ic,p,par,debug,zero_energy_mode,nullptr);
        std::sort(z.begin(), z.end(), [](const ZeroPole& a, const ZeroPole& b){ return a.E_mid < b.E_mid; });
        m[ic.label] = std::move(z);
    }
    return m;
}

inline std::vector<double> model_levels_for_targets(
        const std::vector<TargetLevel>& targets,
        const std::map<std::string,std::vector<ZeroPole>>& zmap) {
    std::vector<double> model(targets.size(), 0.0);
    std::map<std::string,int> next_zero_index;
    for(std::size_t r=0; r<targets.size(); ++r) {
        const auto& t = targets[r];
        const auto it = zmap.find(t.label);
        if(it == zmap.end()) { model[r]=0.0; continue; }
        std::vector<double> zeros;
        for(const auto& z: it->second) if(z.kind=="ZERO" || z.kind=="ZERO_ENDPOINT") zeros.push_back(z.E_mid);
        std::sort(zeros.begin(), zeros.end());
        int idx = next_zero_index[t.label]++;
        if(idx >= 0 && idx < (int)zeros.size()) model[r] = zeros[idx];
        else model[r] = 0.0;
    }
    return model;
}

inline double diagonal_chi2(const std::vector<TargetLevel>& targets, const std::vector<double>& model, double failure_penalty) {
    if(model.size()!=targets.size()) return failure_penalty;
    double chi2=0.0;
    for(std::size_t i=0; i<targets.size(); ++i) {
        if(!(model[i] > 0.0) || !std::isfinite(model[i])) return failure_penalty;
        const double r = (targets[i].Ecm - model[i])/targets[i].err;
        chi2 += r*r;
    }
    return std::isfinite(chi2) ? chi2 : failure_penalty;
}

class K3dfFCN_v31l
#ifndef V31L_DISABLE_MINUIT
    final : public ROOT::Minuit2::FCNBase
#endif
{
public:
    K3dfFCN_v31l(FitSettings settings_,
                 std::vector<TargetLevel> targets_,
                 MatrixD covariance_,
                 MatrixD correlation_,
                 std::shared_ptr<std::vector<IrrepCache>> cache_)
        : settings(std::move(settings_)),
          targets(std::move(targets_)),
          covariance_mat(std::move(covariance_)),
          correlation_mat(std::move(correlation_)),
          cache(std::move(cache_)) {}

#ifndef V31L_DISABLE_MINUIT
    double Up() const override { return 1.0; }
#endif

    double operator()(const std::vector<double>& x) const {
        if(x.size() < 4) return settings.failure_penalty;
        K3dfParameters p{x[0],x[1],x[2],x[3]};
        try {
            auto zp = find_all_zero_poles(*cache,p,base_par,settings.debug,settings.zero_energy_mode);
            auto model = model_levels_for_targets(targets,zp);
            double chi2 = chi_square_v31l(targets,model,covariance_mat,correlation_mat,settings.chi_square_mode,settings.failure_penalty);
            const std::size_t id = ++eval_counter;
            if(settings.print_each_fcn_eval && settings.print_every_fcn_eval>0 && (id % (std::size_t)settings.print_every_fcn_eval)==0) {
                int nzero=0,npole=0;
                for(const auto& kv: zp) for(const auto& z: kv.second) { if(z.kind=="ZERO"||z.kind=="ZERO_ENDPOINT") ++nzero; if(z.kind=="POLE") ++npole; }
                std::cout << std::setprecision(17)
                          << "[v31l-FCN] eval=" << id
                          << " chi2=" << chi2
                          << " K3iso0=" << p.K3iso0
                          << " K3iso1=" << p.K3iso1
                          << " K3B=" << p.K3B
                          << " K3E=" << p.K3E
                          << " zeros=" << nzero
                          << " poles=" << npole
                          << "\n";
            }
            return std::isfinite(chi2) ? chi2 : settings.failure_penalty;
        } catch(const std::exception& e) {
            std::cout << "[v31l-FCN-warning] " << e.what() << "\n";
            return settings.failure_penalty;
        }
    }

private:
    FitSettings settings;
    std::vector<TargetLevel> targets;
    MatrixD covariance_mat;
    MatrixD correlation_mat;
    std::shared_ptr<std::vector<IrrepCache>> cache;
    PhysicsParams base_par = make_base_physics(settings);
    mutable std::atomic<std::size_t> eval_counter{0};
};


// ---------------- v31l binary F3inv cache save/load ----------------
template <class T>
inline void write_binary_raw_v31l(std::ostream& os, const T& x) {
    os.write(reinterpret_cast<const char*>(&x), sizeof(T));
    if(!os) throw std::runtime_error("Binary cache write failed");
}

template <class T>
inline void read_binary_raw_v31l(std::istream& is, T& x) {
    is.read(reinterpret_cast<char*>(&x), sizeof(T));
    if(!is) throw std::runtime_error("Binary cache read failed");
}

inline void write_string_v31l(std::ostream& os, const std::string& s) {
    std::uint64_t n = s.size();
    write_binary_raw_v31l(os,n);
    os.write(s.data(), std::streamsize(n));
    if(!os) throw std::runtime_error("Binary cache string write failed");
}

inline std::string read_string_v31l(std::istream& is) {
    std::uint64_t n=0;
    read_binary_raw_v31l(is,n);
    std::string s(n,'\0');
    if(n) is.read(&s[0], std::streamsize(n));
    if(!is) throw std::runtime_error("Binary cache string read failed");
    return s;
}

inline void write_comp_v31l(std::ostream& os, const comp& z) {
    double re=z.real(), im=z.imag();
    write_binary_raw_v31l(os,re);
    write_binary_raw_v31l(os,im);
}

inline comp read_comp_v31l(std::istream& is) {
    double re=0.0, im=0.0;
    read_binary_raw_v31l(is,re);
    read_binary_raw_v31l(is,im);
    return comp(re,im);
}

inline void write_vec_comp_v31l(std::ostream& os, const std::vector<comp>& v) {
    std::uint64_t n=v.size();
    write_binary_raw_v31l(os,n);
    for(const auto& z: v) write_comp_v31l(os,z);
}

inline std::vector<comp> read_vec_comp_v31l(std::istream& is) {
    std::uint64_t n=0;
    read_binary_raw_v31l(is,n);
    std::vector<comp> v(n);
    for(auto& z: v) z=read_comp_v31l(is);
    return v;
}

inline void write_nested_vec_comp_v31l(std::ostream& os, const std::vector<std::vector<comp>>& vv) {
    std::uint64_t n=vv.size();
    write_binary_raw_v31l(os,n);
    for(const auto& v: vv) write_vec_comp_v31l(os,v);
}

inline std::vector<std::vector<comp>> read_nested_vec_comp_v31l(std::istream& is) {
    std::uint64_t n=0;
    read_binary_raw_v31l(is,n);
    std::vector<std::vector<comp>> vv(n);
    for(auto& v: vv) v=read_vec_comp_v31l(is);
    return vv;
}

inline void write_matrix_v31l(std::ostream& os, const Eigen::MatrixXcd& M) {
    std::uint64_t rows=M.rows(), cols=M.cols();
    write_binary_raw_v31l(os,rows);
    write_binary_raw_v31l(os,cols);
    for(int r=0;r<M.rows();++r) for(int c=0;c<M.cols();++c) write_comp_v31l(os,M(r,c));
}

inline Eigen::MatrixXcd read_matrix_v31l(std::istream& is) {
    std::uint64_t rows=0, cols=0;
    read_binary_raw_v31l(is,rows);
    read_binary_raw_v31l(is,cols);
    Eigen::MatrixXcd M(static_cast<int>(rows), static_cast<int>(cols));
    for(int r=0;r<M.rows();++r) for(int c=0;c<M.cols();++c) M(r,c)=read_comp_v31l(is);
    return M;
}

inline void save_binary_f3inv_cache_v31l(const std::string& path, const std::vector<IrrepCache>& caches) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream os(path, std::ios::binary);
    if(!os) throw std::runtime_error("Could not open binary cache for writing: " + path);
    write_string_v31l(os, "V31L_F3INV_FULL_CACHE_1");
    std::uint64_t ncache=caches.size();
    write_binary_raw_v31l(os,ncache);
    for(const auto& ic: caches) {
        write_string_v31l(os,ic.label);
        std::uint64_t ngrid=ic.grid.size();
        write_binary_raw_v31l(os,ngrid);
        for(const auto& e: ic.grid) {
            write_string_v31l(os,e.label);
            write_binary_raw_v31l(os,e.i);
            write_binary_raw_v31l(os,e.Ecm);
            write_binary_raw_v31l(os,e.En);
            write_binary_raw_v31l(os,e.success);
            write_binary_raw_v31l(os,e.total_dim);
            write_binary_raw_v31l(os,e.proj_dim);
            write_matrix_v31l(os,e.F3inv_proj);
            write_matrix_v31l(os,e.F3inv_full);
            write_nested_vec_comp_v31l(os,e.plm_config);
            write_nested_vec_comp_v31l(os,e.klm_config);
            write_vec_comp_v31l(os,e.total_P);
            write_comp_v31l(os,e.En_c);
            write_matrix_v31l(os,e.Vsel);
            write_binary_raw_v31l(os,e.projector_idem);
            write_binary_raw_v31l(os,e.projector_closure);
            write_binary_raw_v31l(os,e.F3_equiv);
            write_binary_raw_v31l(os,e.F3_leak);
            write_string_v31l(os,e.error);
        }
    }
    std::cout << "[v31l-cache] saved binary F3inv cache: " << path << "\n";
}

inline std::vector<IrrepCache> load_binary_f3inv_cache_v31l(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if(!is) throw std::runtime_error("Could not open binary cache for reading: " + path);
    const std::string magic = read_string_v31l(is);
    if(magic != "V31L_F3INV_FULL_CACHE_1") {
        throw std::runtime_error("Bad binary cache magic/version in " + path + ": " + magic);
    }
    std::uint64_t ncache=0;
    read_binary_raw_v31l(is,ncache);
    std::vector<IrrepCache> caches(ncache);
    for(auto& ic: caches) {
        ic.label = read_string_v31l(is);
        ic.spec = parse_label(ic.label);
        std::uint64_t ngrid=0;
        read_binary_raw_v31l(is,ngrid);
        ic.grid.resize(ngrid);
        for(auto& e: ic.grid) {
            e.label = read_string_v31l(is);
            e.spec = parse_label(e.label);
            read_binary_raw_v31l(is,e.i);
            read_binary_raw_v31l(is,e.Ecm);
            read_binary_raw_v31l(is,e.En);
            read_binary_raw_v31l(is,e.success);
            read_binary_raw_v31l(is,e.total_dim);
            read_binary_raw_v31l(is,e.proj_dim);
            e.F3inv_proj = read_matrix_v31l(is);
            e.F3inv_full = read_matrix_v31l(is);
            e.plm_config = read_nested_vec_comp_v31l(is);
            e.klm_config = read_nested_vec_comp_v31l(is);
            e.total_P = read_vec_comp_v31l(is);
            e.En_c = read_comp_v31l(is);
            e.Vsel = read_matrix_v31l(is);
            read_binary_raw_v31l(is,e.projector_idem);
            read_binary_raw_v31l(is,e.projector_closure);
            read_binary_raw_v31l(is,e.F3_equiv);
            read_binary_raw_v31l(is,e.F3_leak);
            e.error = read_string_v31l(is);
        }
    }
    std::cout << "[v31l-cache] loaded binary F3inv cache: " << path << "\n";
    return caches;
}

inline std::vector<IrrepCache> get_or_build_F3inv_cache_v31l(const FitSettings& settings) {
    if(settings.load_binary_f3inv_cache) {
        return load_binary_f3inv_cache_v31l(settings.binary_f3inv_cache_file);
    }
    auto caches = build_F3inv_only_cache(settings);
    if(settings.save_binary_f3inv_cache) {
        save_binary_f3inv_cache_v31l(settings.binary_f3inv_cache_file,caches);
    }
    return caches;
}


inline void write_cache_grid_files(const FitSettings& settings, const std::vector<IrrepCache>& caches) {
    if(!settings.write_cache_grid) return;
    std::filesystem::create_directories(settings.output_dir);
    for(const auto& ic: caches) {
        const std::string path = settings.output_dir + "/" + settings.output_tag + "_" + clean_label(ic.label) + "_F3inv_only_cache_grid.dat";
        std::ofstream f(path);
        f << std::setprecision(17);
        f << "# columns: i Ecm success total_dim proj_dim projector_idem projector_closure F3_equiv F3_leak error\n";
        for(const auto& e: ic.grid) {
            f << e.i << ' ' << e.Ecm << ' ' << e.success << ' ' << e.total_dim << ' ' << e.proj_dim << ' '
              << e.projector_idem << ' ' << e.projector_closure << ' ' << e.F3_equiv << ' ' << e.F3_leak << ' '
              << sanitize_error(e.error) << "\n";
        }
        std::cout << "[v31l-write] wrote " << path << "\n";
    }
}

inline void write_qc_eig_grid_files(const FitSettings& settings, const std::vector<IrrepCache>& caches, const K3dfParameters& p, const PhysicsParams& par, const std::string& suffix) {
    std::filesystem::create_directories(settings.output_dir);
    for(const auto& ic: caches) {
        std::vector<QCPointValue> vals;
        auto zp = find_zero_poles_from_cache(ic,p,par,settings.debug,settings.zero_energy_mode,&vals);
        const std::string path = settings.output_dir + "/" + settings.output_tag + "_" + clean_label(ic.label) + "_" + suffix + "_QC_closest_eig_grid.dat";
        std::ofstream f(path);
        f << std::setprecision(17);
        f << "# columns: i Ecm success y_closest_zero eig_index herm_QC_rel\n";
        for(std::size_t i=0;i<vals.size();++i) f << i << ' ' << vals[i].Ecm << ' ' << vals[i].success << ' ' << vals[i].y << ' ' << vals[i].eig_index << ' ' << vals[i].herm_rel << "\n";
        const std::string zpath = settings.output_dir + "/" + settings.output_tag + "_" + clean_label(ic.label) + "_" + suffix + "_levels_zero_pole.dat";
        std::ofstream zf(zpath);
        zf << std::setprecision(17);
        zf << "# columns: idx kind E_mid E_left E_right y_left y_right eig_index_left eig_index_right\n";
        int idx=0;
        for(const auto& z: zp) zf << idx++ << ' ' << z.kind << ' ' << z.E_mid << ' ' << z.E_left << ' ' << z.E_right << ' ' << z.y_left << ' ' << z.y_right << ' ' << z.eig_index_left << ' ' << z.eig_index_right << "\n";
        std::cout << "[v31l-write] wrote " << path << "\n";
        std::cout << "[v31l-write] wrote " << zpath << "\n";
    }
}

inline void write_fit_summary(const FitSettings& settings, const FitResult& r) {
    std::filesystem::create_directories(settings.output_dir);
    const std::string path = settings.output_dir + "/" + settings.output_tag + "_fit_summary.dat";
    std::ofstream f(path);
    f << std::setprecision(17);
    f << "valid " << r.valid << "\n";
    f << "chi2 " << r.chi2 << "\n";
    f << "ndata " << r.ndata << "\n";
    f << "npar " << r.npar << "\n";
    f << "ndof " << r.ndof << "\n";
    f << "chi2_dof " << r.chi2_dof << "\n";
    f << "K3iso0 " << r.best.K3iso0 << " err " << r.errors.K3iso0 << "\n";
    f << "K3iso1 " << r.best.K3iso1 << " err " << r.errors.K3iso1 << "\n";
    f << "K3B " << r.best.K3B << " err " << r.errors.K3B << "\n";
    f << "K3E " << r.best.K3E << " err " << r.errors.K3E << "\n";
    std::cout << "[v31l-write] wrote " << path << "\n";

    const std::string rows_path = settings.output_dir + "/" + settings.output_tag + "_fit_levels.dat";
    std::ofstream rf(rows_path);
    rf << std::setprecision(17);
    rf << "# columns: row Lbyas label nPx nPy nPz state data_Ecm data_err model_Ecm residual_sigma\n";
    for(std::size_t i=0; i<r.targets.size(); ++i) {
        const double model = (i<r.model_levels.size()) ? r.model_levels[i] : 0.0;
        const double res = (r.targets[i].err>0.0) ? (r.targets[i].Ecm-model)/r.targets[i].err : std::numeric_limits<double>::quiet_NaN();
        rf << i << ' ' << r.targets[i].Lbyas << ' ' << r.targets[i].label << ' '
           << r.targets[i].nP[0] << ' ' << r.targets[i].nP[1] << ' ' << r.targets[i].nP[2] << ' '
           << r.targets[i].state << ' ' << r.targets[i].Ecm << ' ' << r.targets[i].err << ' ' << model << ' ' << res << "\n";
    }
    std::cout << "[v31l-write] wrote " << rows_path << "\n";

    const std::string spec_path = settings.output_dir + "/" + settings.output_tag + "_bestfit_spectrum_L_irrep_levels.dat";
    std::ofstream sf(spec_path);
    sf << std::setprecision(17);
    sf << "# columns: Lbyas label irrep kind level_index Ecm E_left E_right y_left y_right\n";
    for(const auto& kv: r.zero_poles_by_label) {
        MomentumIrrepSpec spec = parse_label(kv.first);
        int level=0;
        for(const auto& z: kv.second) {
            sf << settings.Lval << ' ' << kv.first << ' ' << spec.irrep << ' '
               << z.kind << ' ' << level++ << ' ' << z.E_mid << ' '
               << z.E_left << ' ' << z.E_right << ' ' << z.y_left << ' ' << z.y_right << "\n";
        }
    }
    std::cout << "[v31l-write] wrote " << spec_path << "\n";
}

inline FitResult fit_K3df_parameters_v31l(FitSettings settings) {
    std::filesystem::create_directories(settings.output_dir);
    #ifdef _OPENMP
    omp_set_num_threads(settings.omp_threads);
    #endif
    Eigen::setNbThreads(1);

    std::cout << "[v31l] [stage 1/7] reading lattice target levels and covariance\n";
    auto [targets,covariance_mat,correlation_mat] = load_targets_and_covariance_v31l(settings);
    std::cout << "[v31l] [stage 1/7] 100% target level count = " << targets.size()
              << " covariance_dim=" << covariance_mat.rows() << "x" << covariance_mat.cols()
              << " chi_square_mode=" << settings.chi_square_mode << "\n";

    std::cout << "[v31l] [stage 2/7] building F3inv-only projected cache in OpenMP\n";
    auto cache_ptr = std::make_shared<std::vector<IrrepCache>>(get_or_build_F3inv_cache_v31l(settings));
    write_cache_grid_files(settings,*cache_ptr);

    std::cout << "[v31l] [stage 3/7] preparing Minuit FCN\n";
    K3dfFCN_v31l fcn(settings,targets,covariance_mat,correlation_mat,cache_ptr);

    FitResult result;
    result.targets = targets;
    result.data_covariance = covariance_mat;
    result.data_correlation = correlation_mat;
    result.ndata = (int)targets.size();
    result.npar = 4;
    result.ndof = result.ndata - result.npar;

#ifndef V31L_DISABLE_MINUIT
    std::cout << "[v31l] [stage 4/7] running Minuit Migrad\n";
    ROOT::Minuit2::MnUserParameters upar;
    upar.Add("K3iso0", settings.guess.K3iso0, settings.step.K3iso0);
    upar.Add("K3iso1", settings.guess.K3iso1, settings.step.K3iso1);
    upar.Add("K3B",    settings.guess.K3B,    settings.step.K3B);
    upar.Add("K3E",    settings.guess.K3E,    settings.step.K3E);
    if(settings.use_parameter_limits) {
        for(unsigned int i=0;i<4;++i) upar.SetLimits(i,settings.param_lower,settings.param_upper);
    }
    ROOT::Minuit2::MnMigrad migrad(fcn,upar);
    ROOT::Minuit2::FunctionMinimum min = migrad();
    std::cout << "[v31l] [stage 4/7] 100% Migrad done; valid=" << min.IsValid() << "\n";
    if(min.IsValid()) {
        ROOT::Minuit2::MnHesse hesse;
        hesse(fcn,min);
    }
    const auto st = min.UserState();
    result.best.K3iso0 = st.Value("K3iso0");
    result.best.K3iso1 = st.Value("K3iso1");
    result.best.K3B    = st.Value("K3B");
    result.best.K3E    = st.Value("K3E");
    result.errors.K3iso0 = st.Error("K3iso0");
    result.errors.K3iso1 = st.Error("K3iso1");
    result.errors.K3B    = st.Error("K3B");
    result.errors.K3E    = st.Error("K3E");
    result.valid = min.IsValid();
    result.chi2 = min.Fval();
    result.parameter_covariance = minuit_covariance_to_eigen_v31l(min,4);
    result.parameter_correlation = covariance_to_correlation_v31l(result.parameter_covariance);
#else
    std::cout << "[v31l] [stage 4/7] V31L_DISABLE_MINUIT set; evaluating initial guess only\n";
    result.best = settings.guess;
    result.errors = settings.step;
    std::vector<double> x = {settings.guess.K3iso0,settings.guess.K3iso1,settings.guess.K3B,settings.guess.K3E};
    result.chi2 = fcn(x);
    result.valid = std::isfinite(result.chi2);
    result.parameter_covariance = MatrixD::Zero(4,4);
    result.parameter_correlation = MatrixD::Zero(4,4);
#endif

    result.chi2_dof = (result.ndof>0) ? result.chi2/double(result.ndof) : std::numeric_limits<double>::quiet_NaN();

    std::cout << "[v31l] [stage 5/7] solving final zero/pole spectrum with best-fit parameters\n";
    result.zero_poles_by_label = find_all_zero_poles(*cache_ptr,result.best,make_base_physics(settings),settings.debug,settings.zero_energy_mode);
    result.model_levels = model_levels_for_targets(targets,result.zero_poles_by_label);

    std::cout << "[v31l] [stage 6/7] writing best-fit projected QC grids and final levels\n";
    write_qc_eig_grid_files(settings,*cache_ptr,result.best,make_base_physics(settings),"bestfit");
    write_fit_summary(settings,result);

    std::cout << "[v31l] [stage 7/7] 100% done\n";
    return result;
}

inline void print_fit_result_summary(const FitResult& r) {
    std::cout << std::setprecision(17);
    std::cout << "[v31l-fit-summary] valid=" << r.valid
              << " chi2=" << r.chi2
              << " ndof=" << r.ndof
              << " chi2/ndof=" << r.chi2_dof << "\n";
    std::cout << "[v31l-fit-summary] K3iso0=" << r.best.K3iso0 << " +/- " << r.errors.K3iso0 << "\n";
    std::cout << "[v31l-fit-summary] K3iso1=" << r.best.K3iso1 << " +/- " << r.errors.K3iso1 << "\n";
    std::cout << "[v31l-fit-summary] K3B=" << r.best.K3B << " +/- " << r.errors.K3B << "\n";
    std::cout << "[v31l-fit-summary] K3E=" << r.best.K3E << " +/- " << r.errors.K3E << "\n";
}

} // namespace k3df_fit_v31l

#endif // K3DF_MINUIT_FIT_V31L_LATTICE_COVARIANCE_HPP
