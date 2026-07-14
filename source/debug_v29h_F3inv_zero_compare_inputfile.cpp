#include "F3_cpu_openmp_v25_K3QC_cached_core.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

// v29p: input-file driven F3^{-1} zero comparison debugger.  Changes from v29o:
//  - determinant eigenbranch candidates are de-duplicated before candidate/zero output
//  - accepted zeros are classified across scalar, full determinant, and projected determinant methods
//  - match summaries include explicit classification counts
//  - detF3inv is the full-space determinant and may contain extra zeros not present
//    in the projected irrep determinant.
//
// v29o: input-file driven F3^{-1} zero comparison debugger.  Changes from v29n:
//  - determinant/projection candidates are found by explicit adjacent-interval eigenbranch matching
//    for every coarse window [E_i,E_{i+1}] and every branch, not by one global branch track
//  - branch-scan summary files are written before any refinement/filtering
//  - this fixes missed determinant candidates near later F3iso^{-1} zeros
//  - projected determinant still uses det((V^dagger F3 V)^{-1})
//  - accepted and rejected candidate summaries are written for diagnostics.
// Compares zeros of
//   1) F3iso^{-1} = 1 / (<1,1/sqrt(2)|F3|1,1/sqrt(2)>)
//   2) det(F3^{-1})
//   3) det((V^dagger F3 V)^{-1})
// for multiple irreps supplied in the input file.

struct MomentumIrrepSpec {
    std::string label;
    std::array<int,3> nnP{{0,0,0}};
    std::string irrep;
    std::string irrep_tag;
};

static MomentumIrrepSpec parse_label(const std::string& label) {
    MomentumIrrepSpec s; s.label = label;
    if (label == "000_A1m") { s.nnP = {0,0,0}; s.irrep = "A1u"; s.irrep_tag = "A1m"; return s; }
    if (label == "100_A2")  { s.nnP = {0,0,1}; s.irrep = "A2";  s.irrep_tag = "A2";  return s; }
    if (label == "110_A2")  { s.nnP = {1,1,0}; s.irrep = "A2";  s.irrep_tag = "A2";  return s; }
    if (label == "111_A2")  { s.nnP = {1,1,1}; s.irrep = "A2";  s.irrep_tag = "A2";  return s; }
    if (label == "200_A2")  { s.nnP = {0,0,2}; s.irrep = "A2";  s.irrep_tag = "A2";  return s; }
    throw std::runtime_error("Unsupported mom_label: " + label + " . Supported: 000_A1m 100_A2 110_A2 111_A2 200_A2");
}

static std::string trim(std::string s) {
    auto not_space = [](unsigned char c){ return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}
static std::string strip_inline_comment(const std::string& line) {
    bool in_quote = false;
    for (std::size_t i=0; i<line.size(); ++i) {
        if (line[i] == '"') in_quote = !in_quote;
        if (!in_quote && line[i] == '#') return line.substr(0, i);
    }
    return line;
}
static std::map<std::string,std::string> read_kv_file(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("Could not open input file: " + filename);
    std::map<std::string,std::string> kv;
    std::string line;
    int lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        line = trim(strip_inline_comment(line));
        if (line.empty()) continue;
        std::size_t eq = line.find('=');
        if (eq == std::string::npos) throw std::runtime_error("Bad input line " + std::to_string(lineno) + ": expected key=value");
        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq+1));
        if (!val.empty() && val.front() == '"' && val.back() == '"') val = val.substr(1, val.size()-2);
        if (key.empty()) throw std::runtime_error("Bad input line " + std::to_string(lineno) + ": empty key");
        kv[key] = val;
    }
    return kv;
}
static std::vector<std::string> split_csv(std::string s) {
    for (char& c: s) if (c == ',') c = ' ';
    std::istringstream is(s);
    std::vector<std::string> out;
    std::string x;
    while (is >> x) out.push_back(x);
    return out;
}
static double get_double(const std::map<std::string,std::string>& kv, const std::string& key, double def) {
    auto it = kv.find(key); if (it == kv.end()) return def;
    try { size_t p=0; double x=std::stod(it->second,&p); if(p!=it->second.size()) throw std::runtime_error("bad"); return x; }
    catch(...) { throw std::runtime_error("Could not parse double for key " + key + " = " + it->second); }
}
static int get_int(const std::map<std::string,std::string>& kv, const std::string& key, int def) {
    auto it = kv.find(key); if (it == kv.end()) return def;
    try { size_t p=0; int x=std::stoi(it->second,&p); if(p!=it->second.size()) throw std::runtime_error("bad"); return x; }
    catch(...) { throw std::runtime_error("Could not parse int for key " + key + " = " + it->second); }
}
static std::string get_string(const std::map<std::string,std::string>& kv, const std::string& key, const std::string& def) {
    auto it = kv.find(key); return (it == kv.end()) ? def : it->second;
}
static std::vector<int> get_int_list(const std::map<std::string,std::string>& kv, const std::string& key, const std::vector<int>& def) {
    auto it = kv.find(key); if (it == kv.end()) return def;
    std::vector<int> out; for (const std::string& tok: split_csv(it->second)) out.push_back(std::stoi(tok)); return out;
}
static bool finite_complex(const comp& z) { return std::isfinite(z.real()) && std::isfinite(z.imag()); }
static std::string sanitize_error(std::string e) { for(char& c:e) if(std::isspace((unsigned char)c)) c='_'; return e.empty()?"OK":e; }

struct Options {
    std::vector<std::string> mom_labels{"000_A1m"};
    double Lval=20.0, xival=3.444, E0=0.2631, E1=0.36;
    int N=1601, threads=18;
    char debug='n';
    double atmpi=0.06906, atmK=0.09698;
    double eta_1=1.0, eta_2=0.5, alpha=0.5, epsilon_h=0.0, max_shell_num=20.0, tolerance=1.0e-12;
    int parity=-1;
    double eig_tol=0.05, norm_tol=1.0e-12, proj_tol=1.0e-10;
    bool Q0norm=true, sort_orbit_flag=false;
    std::vector<int> waves_vec_1{0,1}, waves_vec_2{0};
    int refine_iter=60;
    int local_min_refine_N=81;
    int parallel_refine=1;
    int parallel_matrix_dumps=1;
    int coarse_parallel=0; // v29k: default safe mode; set 1 to test OpenMP coarse grid
    std::string omp_schedule="dynamic";
    int omp_chunk=1;
    double duplicate_tol=1.0e-5;
    double zero_abs_tol=1.0e-2;
    int use_abs_min_candidates=0;
    int sign_refine_grid_N=101;
    double side_monotonic_rel_tol=1.0e-12;
    int accept_only_likely_zero=1;
    int write_refine_logs=1;
    int dump_matrices=1;
    int matrix_dump_stride=0;
    int max_matrix_dumps_per_irrep=20;
    std::string outdir="output_v29h";
    std::string prefix="debug_v29p_F3inv_zero_compare";
    char projector_debug='n';
    int write_candidate_logs=1;
};

static Options read_options(const std::string& input_file, PhysicsParams& par) {
    const auto kv = read_kv_file(input_file);
    Options o;
    o.mom_labels = split_csv(get_string(kv, "mom_labels", get_string(kv, "mom_label", "000_A1m")));
    if (o.mom_labels.empty()) o.mom_labels.push_back("000_A1m");
    o.Lval=get_double(kv,"Lval",o.Lval); o.xival=get_double(kv,"xival",o.xival); o.E0=get_double(kv,"E0",o.E0); o.E1=get_double(kv,"E1",o.E1);
    o.N=get_int(kv,"N",o.N); o.threads=get_int(kv,"threads",o.threads);
    o.debug = get_string(kv,"debug","n").empty() ? 'n' : get_string(kv,"debug","n")[0];
    o.atmpi=get_double(kv,"atmpi",o.atmpi); o.atmK=get_double(kv,"atmK",o.atmK);
    o.eta_1=get_double(kv,"eta_1",o.eta_1); o.eta_2=get_double(kv,"eta_2",o.eta_2); o.alpha=get_double(kv,"alpha",o.alpha);
    o.epsilon_h=get_double(kv,"epsilon_h",o.epsilon_h); o.max_shell_num=get_double(kv,"max_shell_num",o.max_shell_num); o.tolerance=get_double(kv,"tolerance",o.tolerance);
    o.parity=get_int(kv,"parity",o.parity); o.eig_tol=get_double(kv,"eig_tol",o.eig_tol); o.norm_tol=get_double(kv,"norm_tol",o.norm_tol); o.proj_tol=get_double(kv,"proj_tol",o.proj_tol);
    o.Q0norm=(get_int(kv,"Q0norm",1)!=0); o.sort_orbit_flag=(get_int(kv,"sort_orbit_flag",0)!=0);
    o.waves_vec_1=get_int_list(kv,"waves_vec_1",o.waves_vec_1); o.waves_vec_2=get_int_list(kv,"waves_vec_2",o.waves_vec_2);
    o.refine_iter=get_int(kv,"refine_iter",o.refine_iter); o.local_min_refine_N=get_int(kv,"local_min_refine_N",o.local_min_refine_N);
    o.parallel_refine=get_int(kv,"parallel_refine",o.parallel_refine);
    o.parallel_matrix_dumps=get_int(kv,"parallel_matrix_dumps",o.parallel_matrix_dumps);
    o.coarse_parallel=get_int(kv,"coarse_parallel",o.coarse_parallel);
    o.omp_schedule=get_string(kv,"omp_schedule",o.omp_schedule);
    o.omp_chunk=get_int(kv,"omp_chunk",o.omp_chunk);
    o.duplicate_tol=get_double(kv,"duplicate_tol",o.duplicate_tol); o.zero_abs_tol=get_double(kv,"zero_abs_tol",o.zero_abs_tol);
    o.use_abs_min_candidates=get_int(kv,"use_abs_min_candidates",o.use_abs_min_candidates);
    o.sign_refine_grid_N=get_int(kv,"sign_refine_grid_N",o.sign_refine_grid_N);
    o.side_monotonic_rel_tol=get_double(kv,"side_monotonic_rel_tol",o.side_monotonic_rel_tol);
    o.accept_only_likely_zero=get_int(kv,"accept_only_likely_zero",o.accept_only_likely_zero);
    o.projector_debug = get_string(kv,"projector_debug",std::string(1,o.projector_debug)).empty() ? 'n' : get_string(kv,"projector_debug",std::string(1,o.projector_debug))[0];
    o.write_candidate_logs=get_int(kv,"write_candidate_logs",o.write_candidate_logs);
    o.write_refine_logs=get_int(kv,"write_refine_logs",o.write_refine_logs);
    o.dump_matrices=get_int(kv,"dump_matrices",o.dump_matrices); o.matrix_dump_stride=get_int(kv,"matrix_dump_stride",o.matrix_dump_stride);
    o.max_matrix_dumps_per_irrep=get_int(kv,"max_matrix_dumps_per_irrep",o.max_matrix_dumps_per_irrep);
    o.outdir=get_string(kv,"outdir",o.outdir); o.prefix=get_string(kv,"prefix",o.prefix);

    par.atmpi=o.atmpi; par.atmK=o.atmK; par.xi=o.xival; par.Lbyas=o.Lval;
    par.eta_1=o.eta_1; par.eta_2=o.eta_2; par.alpha=o.alpha; par.epsilon_h=o.epsilon_h; par.max_shell_num=o.max_shell_num; par.tolerance=o.tolerance;
    par.parity=o.parity; par.eig_tol=o.eig_tol; par.norm_tol=o.norm_tol; par.proj_tol=o.proj_tol; par.Q0norm=o.Q0norm; par.sort_orbit_flag=o.sort_orbit_flag;
    par.omp_threads=o.threads; par.waves_vec_1=o.waves_vec_1; par.waves_vec_2=o.waves_vec_2;
    par.K3iso = {comp(0.0,0.0), comp(0.0,0.0)}; par.K3B_par=comp(0.0,0.0); par.K3E_par=comp(0.0,0.0);
    for (int i=0;i<4;++i) for(int j=0;j<3;++j) {
        const std::string k1="scatter1_"+std::to_string(i)+std::to_string(j);
        const std::string k2="scatter2_"+std::to_string(i)+std::to_string(j);
        par.scatter_params_1[i][j]=comp(get_double(kv,k1,par.scatter_params_1[i][j].real()),0.0);
        par.scatter_params_2[i][j]=comp(get_double(kv,k2,par.scatter_params_2[i][j].real()),0.0);
    }
    if (o.N < 2) throw std::runtime_error("N must be >= 2");
    if (!(o.E1 > o.E0)) throw std::runtime_error("E1 must be greater than E0");
    if (o.refine_iter < 1) throw std::runtime_error("refine_iter must be >= 1");
    if (o.local_min_refine_N < 5) throw std::runtime_error("local_min_refine_N must be >= 5");
    if (o.sign_refine_grid_N < 5) throw std::runtime_error("sign_refine_grid_N must be >= 5");
    if (o.omp_chunk < 1) o.omp_chunk = 1;
    return o;
}

struct EvalData {
    int i=-1; double Ecm=std::numeric_limits<double>::quiet_NaN(), En=std::numeric_limits<double>::quiet_NaN();
    int A=0, B=0, total_dim=0, vdim=0;
    comp F3iso=comp(NAN,NAN), F3iso_inv=comp(NAN,NAN), detF3=comp(NAN,NAN), detF3inv=comp(NAN,NAN), detProjF3inv=comp(NAN,NAN);
    double min_sv_F3=std::numeric_limits<double>::quiet_NaN(), max_sv_F3=std::numeric_limits<double>::quiet_NaN(), cond_F3=std::numeric_limits<double>::quiet_NaN();
    double min_sv_projF3inv=std::numeric_limits<double>::quiet_NaN(), max_sv_projF3inv=std::numeric_limits<double>::quiet_NaN(), cond_projF3inv=std::numeric_limits<double>::quiet_NaN();
    double herm_F3inv_rel=std::numeric_limits<double>::quiet_NaN(), herm_projF3inv_rel=std::numeric_limits<double>::quiet_NaN();
    bool success=false; std::string error;
};

struct EvalFull { EvalData d; Eigen::MatrixXcd F3; Eigen::MatrixXcd F3inv; Eigen::MatrixXcd Vsel; Eigen::MatrixXcd projF3; Eigen::MatrixXcd projF3inv; };

static bool finite_double_v29k(double x) {
    return std::isfinite(x);
}
static bool finite_comp_v29k(const comp& z) {
    return std::isfinite(z.real()) && std::isfinite(z.imag());
}
static bool finite_matrix_v29k(const Eigen::MatrixXcd& M) {
    if(M.rows()==0 || M.cols()==0) return false;
    return M.allFinite();
}


// v29o: determinant with overflow/underflow protection.
// Raw determinants of F3^{-1} can overflow because det(F3) can be ~1e-300 or smaller.
// For sign-flip/root diagnostics we mainly need a finite signed/complex representative.
// This returns exp(clamp(log|det|, -cap_logabs, +cap_logabs)) * phase(det),
// so the phase/sign is kept while the magnitude is capped to avoid 0/inf.
static comp determinant_scaled_partial_piv_lu(const Eigen::MatrixXcd& M, double cap_logabs=690.0) {
    if(M.rows()==0 || M.cols()==0 || M.rows()!=M.cols() || !M.allFinite()) {
        return comp(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
    }
    Eigen::PartialPivLU<Eigen::MatrixXcd> lu(M);
    const auto diag = lu.matrixLU().diagonal();
    double logabs = 0.0;
    comp phase(1.0, 0.0);
    for(int i=0; i<diag.size(); ++i) {
        const comp d = diag(i);
        const double a = std::abs(d);
        if(!std::isfinite(a)) {
            return comp(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
        }
        if(a == 0.0) {
            return comp(0.0, 0.0);
        }
        logabs += std::log(a);
        phase *= d / a;
    }
    const int psgn = int(lu.permutationP().determinant());
    phase *= double(psgn);
    if(!std::isfinite(logabs) || !std::isfinite(phase.real()) || !std::isfinite(phase.imag())) {
        return comp(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
    }
    const double clipped = std::max(-cap_logabs, std::min(cap_logabs, logabs));
    return std::exp(clipped) * phase;
}
static void mark_failure_v29k(EvalFull& out, const std::string& reason) {
    out.d.success = false;
    out.d.error = reason;
}
static bool validate_eval_v29k(EvalFull& out) {
    if(out.d.total_dim <= 0) { mark_failure_v29k(out,"BAD_DIM_EMPTY_CONFIG"); return false; }
    if(out.F3.rows()!=out.d.total_dim || out.F3.cols()!=out.d.total_dim) { mark_failure_v29k(out,"BAD_F3_DIM"); return false; }
    if(!finite_matrix_v29k(out.F3)) { mark_failure_v29k(out,"NONFINITE_F3_MATRIX"); return false; }
    if(!finite_comp_v29k(out.d.F3iso)) { mark_failure_v29k(out,"NONFINITE_F3ISO"); return false; }
    if(!(std::abs(out.d.F3iso)>0.0)) { mark_failure_v29k(out,"ZERO_F3ISO_DENOMINATOR"); return false; }
    if(!finite_comp_v29k(out.d.F3iso_inv)) { mark_failure_v29k(out,"NONFINITE_F3ISO_INV"); return false; }
    if(!finite_comp_v29k(out.d.detF3)) { mark_failure_v29k(out,"NONFINITE_DETF3"); return false; }
    if(!finite_matrix_v29k(out.F3inv)) { mark_failure_v29k(out,"NONFINITE_F3INV_MATRIX"); return false; }
    if(!finite_comp_v29k(out.d.detF3inv)) { mark_failure_v29k(out,"NONFINITE_DETF3INV"); return false; }
    if(out.d.vdim <= 0) { mark_failure_v29k(out,"BAD_PROJECTOR_ZERO_DIM"); return false; }
    if(!finite_matrix_v29k(out.projF3inv)) { mark_failure_v29k(out,"NONFINITE_PROJF3INV_MATRIX"); return false; }
    if(!finite_comp_v29k(out.d.detProjF3inv)) { mark_failure_v29k(out,"NONFINITE_DETPROJF3INV"); return false; }
    if(!finite_double_v29k(out.d.min_sv_F3) || !finite_double_v29k(out.d.max_sv_F3) || out.d.min_sv_F3 < -1.0e-14 || out.d.max_sv_F3 < -1.0e-14) { mark_failure_v29k(out,"BAD_F3_SINGULAR_VALUES"); return false; }
    if(!finite_double_v29k(out.d.min_sv_projF3inv) || !finite_double_v29k(out.d.max_sv_projF3inv) || out.d.min_sv_projF3inv < -1.0e-14 || out.d.max_sv_projF3inv < -1.0e-14) { mark_failure_v29k(out,"BAD_PROJF3INV_SINGULAR_VALUES"); return false; }
    return true;
}

static EvalFull evaluate_full(double Ecm, const std::vector<int>& nnP_vec, const std::string& irrep, const PhysicsParams& par, char debug) {
    EvalFull out; out.d.Ecm=Ecm;
    try {
        const comp pi=std::acos(-1.0); const double L=par.L(); const comp twopibyL=((comp)2.0)*pi/((comp)L);
        std::vector<comp> total_P(3), nnP_config(3);
        for(int a=0;a<3;++a){ total_P[a]=twopibyL*double(nnP_vec[a]); nnP_config[a]=comp(nnP_vec[a],0.0); }
        const comp Ecm_c(Ecm,0.0); const comp En_c=Ecm_to_E(Ecm_c,total_P); out.d.En=En_c.real();
        std::vector<std::vector<comp>> plm_config(5), klm_config(5); std::vector<std::vector<int>> np_config(5), nk_config(5);
        config_maker_4(plm_config,np_config,par.waves_vec_1,En_c,total_P,par.atmK,par.atmK,par.atmpi,L,par.epsilon_h,par.max_shell_num,par.tolerance);
        config_maker_4(klm_config,nk_config,par.waves_vec_2,En_c,total_P,par.atmpi,par.atmK,par.atmK,L,par.epsilon_h,par.max_shell_num,par.tolerance);
        const int A=int(plm_config[0].size()), B=int(klm_config[0].size()), N=A+B; out.d.A=A; out.d.B=B; out.d.total_dim=N;
        if(N<=0){ out.d.error="empty_config"; return out; }
        Eigen::MatrixXcd F2(N,N), G(N,N), K2inv(N,N);
        F2_2plus1_mat(F2,En_c,plm_config,klm_config,total_P,par.atmK,par.atmpi,L,par.alpha,par.epsilon_h,par.max_shell_num,par.Q0norm);
        K2inv_EREord2_2plus1_mat(K2inv,par.eta_1,par.eta_2,par.scatter_params_1,par.scatter_params_2,En_c,plm_config,klm_config,total_P,par.atmK,par.atmpi,par.epsilon_h,L);
        G_2plus1_mat(G,En_c,plm_config,klm_config,total_P,par.atmK,par.atmpi,L,par.alpha,par.epsilon_h,par.max_shell_num,par.Q0norm);
        const Eigen::MatrixXcd H=K2inv+F2+G;
        Eigen::PartialPivLU<Eigen::MatrixXcd> luH(H); const Eigen::MatrixXcd X1=luH.solve(F2); out.F3=(F2/comp(3.0,0.0))-F2*X1;
        Eigen::VectorXcd norm_vec(N); for(int i=0;i<A;++i) norm_vec(i)=comp(1.0,0.0); for(int i=0;i<B;++i) norm_vec(A+i)=comp(1.0/std::sqrt(2.0),0.0);
        out.d.F3iso=(norm_vec.transpose()*out.F3*norm_vec)(0,0);
        if(std::abs(out.d.F3iso)>0.0) out.d.F3iso_inv=comp(1.0,0.0)/out.d.F3iso;
        out.d.detF3=determinant_scaled_partial_piv_lu(out.F3);
        Eigen::PartialPivLU<Eigen::MatrixXcd> luF3(out.F3);
        out.F3inv=luF3.solve(Eigen::MatrixXcd::Identity(N,N));
        out.d.detF3inv=determinant_scaled_partial_piv_lu(out.F3inv);
        Eigen::JacobiSVD<Eigen::MatrixXcd> svdF3(out.F3, Eigen::ComputeThinU|Eigen::ComputeThinV);
        if(svdF3.singularValues().size()>0){ out.d.min_sv_F3=svdF3.singularValues().minCoeff(); out.d.max_sv_F3=svdF3.singularValues().maxCoeff(); if(out.d.min_sv_F3>0) out.d.cond_F3=out.d.max_sv_F3/out.d.min_sv_F3; }
        const double f3in_norm=out.F3inv.norm(); out.d.herm_F3inv_rel=(f3in_norm>0.0)?(out.F3inv-out.F3inv.adjoint()).norm()/f3in_norm:0.0;
        Eigen::MatrixXcd P_I(N,N); std::string I_mut=irrep; std::vector<comp> total_P_mut=total_P, nnP_config_mut=nnP_config;
        P_irrep_projection_2plus1(P_I,plm_config,np_config,klm_config,nk_config,I_mut,total_P_mut,nnP_config_mut,par.sort_orbit_flag,par.parity);
        Eigen::MatrixXcd Pproj; build_projector_from_eigenvectors_near_one(P_I,out.Vsel,Pproj,par.eig_tol,par.norm_tol,par.proj_tol,'n');
        out.d.vdim=int(out.Vsel.cols()); if(out.d.vdim<=0){ out.d.error="projection_produced_zero_columns"; return out; }
        // v29o: first project F3 itself, then invert the projected matrix.
        // This defines det(proj(F3^{-1})) as det((V^dagger F3 V)^{-1}),
        // instead of det(V^dagger F3^{-1} V).
        out.projF3=out.Vsel.adjoint()*out.F3*out.Vsel;
        if(!finite_matrix_v29k(out.projF3)){ out.d.error="NONFINITE_PROJF3_MATRIX"; return out; }
        Eigen::PartialPivLU<Eigen::MatrixXcd> luProjF3(out.projF3);
        out.projF3inv=luProjF3.solve(Eigen::MatrixXcd::Identity(out.d.vdim,out.d.vdim));
        out.d.detProjF3inv=determinant_scaled_partial_piv_lu(out.projF3inv);
        Eigen::JacobiSVD<Eigen::MatrixXcd> svdP(out.projF3inv,Eigen::ComputeThinU|Eigen::ComputeThinV);
        if(svdP.singularValues().size()>0){ out.d.min_sv_projF3inv=svdP.singularValues().minCoeff(); out.d.max_sv_projF3inv=svdP.singularValues().maxCoeff(); if(out.d.min_sv_projF3inv>0) out.d.cond_projF3inv=out.d.max_sv_projF3inv/out.d.min_sv_projF3inv; }
        const double pn=out.projF3inv.norm(); out.d.herm_projF3inv_rel=(pn>0.0)?(out.projF3inv-out.projF3inv.adjoint()).norm()/pn:0.0;
        if(validate_eval_v29k(out)) { out.d.success=true; out.d.error="OK"; }
    } catch(const std::exception& e) { out.d.success=false; out.d.error=e.what(); }
    return out;
}

static EvalData evaluate_data(double Ecm, const std::vector<int>& nnP_vec, const std::string& irrep, const PhysicsParams& par, char debug) {
    return evaluate_full(Ecm, nnP_vec, irrep, par, debug).d;
}

static double method_value(const EvalData& d, int method, const std::string& component) {
    comp z = (method==0) ? d.F3iso_inv : (method==1 ? d.detF3inv : d.detProjF3inv);
    if(component=="imag") return z.imag();
    if(component=="abs") return std::abs(z);
    return z.real();
}
static std::string method_name(int m) { return (m==0)?"F3iso_inv":((m==1)?"detF3inv":"detProjF3inv"); }
static std::string method_label(int m) { return (m==0)?"F3iso^{-1}":((m==1)?"det(F3^{-1})":"det(proj(F3^{-1}))"); }

struct ZeroRecord {
    std::string label, irrep, irrep_tag, method, kind, reason;
    int nPx=0,nPy=0,nPz=0,index=-1;
    double E=std::numeric_limits<double>::quiet_NaN(), E_left=std::numeric_limits<double>::quiet_NaN(), E_right=std::numeric_limits<double>::quiet_NaN();
    double y=std::numeric_limits<double>::quiet_NaN(), abs_y=std::numeric_limits<double>::quiet_NaN();
    double y_left=std::numeric_limits<double>::quiet_NaN(), y_right=std::numeric_limits<double>::quiet_NaN();
    int accepted=0;
    int crossing_index=-1;
    std::string classifier="none";
    double left_score=std::numeric_limits<double>::quiet_NaN();
    double right_score=std::numeric_limits<double>::quiet_NaN();
    int branch_index=-1;
    double branch_abs=std::numeric_limits<double>::quiet_NaN();
    int merged_count=1;
};


static bool sign_flip(double a, double b) {
    return std::isfinite(a) && std::isfinite(b) && (a==0.0 || a*b < 0.0);
}

struct SideClass {
    bool left_decreasing=false;
    bool right_decreasing=false;
    bool left_increasing=false;
    bool right_increasing=false;
    bool likely_zero=false;
    bool likely_pole=false;
    double left_score=0.0;
    double right_score=0.0;
    std::string classifier="undetermined";
};

static bool local_zero_like_crossing(const std::vector<double>& abs_y, int j, const Options& opt) {
    const int M = (int)abs_y.size();
    if(j <= 0 || j+2 >= M) return false;
    for(double v: abs_y) if(!std::isfinite(v)) return false;
    const double near_cross = std::min(abs_y[j], abs_y[j+1]);
    const double left1 = abs_y[j-1];
    const double left2 = abs_y[j];
    const double right1 = abs_y[j+1];
    const double right2 = abs_y[j+2];
    const double scale = std::max({1.0, left1, left2, right1, right2});
    const double tol = std::max(0.0,opt.side_monotonic_rel_tol) * scale;
    return near_cross <= left1 + tol && near_cross <= right2 + tol &&
           left2 <= left1 + tol && right1 <= right2 + tol;
}

static SideClass classify_side_abs(const std::vector<double>& abs_y, int j, const Options& opt) {
    const int M = (int)abs_y.size();
    SideClass c;
    if(j < 0 || j+1 >= M) { c.classifier="bad_crossing_index"; return c; }
    for(double v: abs_y) if(!std::isfinite(v)) { c.classifier="nonfinite_in_refine_grid"; return c; }

    const double tol = std::max(0.0,opt.side_monotonic_rel_tol);
    c.left_decreasing = true;
    c.right_decreasing = true;
    c.left_increasing = true;
    c.right_increasing = true;

    // zero-like: |f| decreases as we move from the left edge toward j,
    // and decreases as we move from the right edge toward j+1.
    for(int k=0;k<j;++k){
        const double scale = std::max({1.0, abs_y[k], abs_y[k+1]});
        if(abs_y[k+1] > abs_y[k] + tol*scale) c.left_decreasing=false;
        if(abs_y[k+1] < abs_y[k] - tol*scale) c.left_increasing=false;
    }
    for(int k=M-1;k>j+1;--k){
        const double scale = std::max({1.0, abs_y[k], abs_y[k-1]});
        if(abs_y[k-1] > abs_y[k] + tol*scale) c.right_decreasing=false;
        if(abs_y[k-1] < abs_y[k] - tol*scale) c.right_increasing=false;
    }

    const double left_edge  = abs_y.front();
    const double right_edge = abs_y.back();
    const double near_cross = std::min(abs_y[j], abs_y[j+1]);
    const double far_scale  = std::max({left_edge,right_edge,1.0});
    c.left_score  = (left_edge  > 0.0) ? near_cross/left_edge  : std::numeric_limits<double>::infinity();
    c.right_score = (right_edge > 0.0) ? near_cross/right_edge : std::numeric_limits<double>::infinity();

    const bool near_smaller_than_edges = (near_cross <= left_edge && near_cross <= right_edge && near_cross <= far_scale);
    c.likely_zero = c.left_decreasing && c.right_decreasing && near_smaller_than_edges;
    c.likely_pole = c.left_increasing && c.right_increasing && !near_smaller_than_edges;
    if(c.likely_zero) c.classifier="likely_zero_side_abs_decreases";
    else if(c.likely_pole) c.classifier="likely_pole_side_abs_increases";
    else c.classifier="ambiguous_side_abs_pattern";
    return c;
}

static std::vector<EvalData> evaluate_refine_grid_data_parallel(int method, double a, double b, const std::vector<int>& nnP, const std::string& irrep, const PhysicsParams& par, char debug, const Options& opt) {
    const int M = std::max(5,opt.sign_refine_grid_N);
    std::vector<EvalData> g(M);
    #pragma omp parallel for schedule(dynamic,1)
    for(int i=0;i<M;++i){
        const double t=double(i)/double(M-1);
        const double E=a+t*(b-a);
        g[i]=evaluate_data(E,nnP,irrep,par,debug);
        g[i].i=i;
    }
    (void)method;
    return g;
}

static std::vector<EvalFull> evaluate_refine_grid_full_parallel(double a, double b, const std::vector<int>& nnP, const std::string& irrep, const PhysicsParams& par, char debug, const Options& opt) {
    const int M = std::max(5,opt.sign_refine_grid_N);
    std::vector<EvalFull> g(M);
    #pragma omp parallel for schedule(dynamic,1)
    for(int i=0;i<M;++i){
        const double t=double(i)/double(M-1);
        const double E=a+t*(b-a);
        g[i]=evaluate_full(E,nnP,irrep,par,debug);
        g[i].d.i=i;
    }
    return g;
}

static ZeroRecord bisection_method_zero(int method, double a, double b, const std::vector<int>& nnP, const std::string& irrep, const PhysicsParams& par, char debug, int iter, const std::string& component) {
    // v29j: final bisection is intentionally performed on the ORIGINAL coarse-grid
    // sign-flip bracket [a,b] = [E_i,E_{i+1}], not on the smaller refined-grid
    // bracket [Eref_j,Eref_{j+1}].  The refined grid is only used to classify the
    // candidate as zero-like or pole-like.
    double fa=method_value(evaluate_data(a,nnP,irrep,par,debug),method,component);
    double fb=method_value(evaluate_data(b,nnP,irrep,par,debug),method,component);
    const double fa0=fa, fb0=fb;
    double left=a,right=b;
    if(!std::isfinite(fa) || !std::isfinite(fb) || !(fa==0.0 || fb==0.0 || fa*fb<=0.0)){
        ZeroRecord z; z.method=method_name(method); z.kind="coarse_window_bisection_failed"; z.E=0.5*(a+b); z.E_left=a; z.E_right=b;
        z.y=NAN; z.abs_y=INFINITY; z.y_left=fa0; z.y_right=fb0; z.accepted=0;
        z.reason="coarse_window_endpoints_do_not_bracket_after_re-evaluation";
        return z;
    }
    for(int k=0;k<iter;++k){
        const double mid=0.5*(left+right);
        const double fm=method_value(evaluate_data(mid,nnP,irrep,par,debug),method,component);
        if(!std::isfinite(fm)) break;
        if(fa==0.0){ right=left; fb=fa; break; }
        if(fb==0.0){ left=right; fa=fb; break; }
        if(fa*fm<=0.0){ right=mid; fb=fm; }
        else { left=mid; fa=fm; }
    }
    const double E=0.5*(left+right);
    const double y=method_value(evaluate_data(E,nnP,irrep,par,debug),method,component);
    ZeroRecord z;
    z.method=method_name(method);
    z.kind="coarse_window_bisection";
    z.E=E;
    z.E_left=a;
    z.E_right=b;
    z.y=y;
    z.abs_y=std::abs(y);
    z.y_left=fa0;
    z.y_right=fb0;
    z.reason="side_abs_likely_zero_then_bisection_on_original_coarse_signflip_window";
    return z;
}

static ZeroRecord bisection_f3iso_zero(double a, double b, const std::vector<int>& nnP, const std::string& irrep, const PhysicsParams& par, char debug, int iter, const std::string& component) {
    return bisection_method_zero(0,a,b,nnP,irrep,par,debug,iter,component);
}

static void write_refine_log_data(const std::string& file, int method, const std::vector<EvalData>& rg, const std::string& component) {
    std::ofstream out(file); out<<std::setprecision(17);
    out << "# refined cached grid for "<<method_name(method)<<"\n";
    out << "# columns: j Ecm y abs_y F3isoInv_re detF3inv_re detProjF3inv_re min_sv_F3 cond_F3 min_sv_projF3inv cond_projF3inv success error\n";
    for(const auto& d: rg){
        const double y=method_value(d,method,component);
        out<<d.i<<' '<<d.Ecm<<' '<<y<<' '<<std::abs(y)<<' '<<d.F3iso_inv.real()<<' '<<d.detF3inv.real()<<' '<<d.detProjF3inv.real()<<' '<<d.min_sv_F3<<' '<<d.cond_F3<<' '<<d.min_sv_projF3inv<<' '<<d.cond_projF3inv<<' '<<(d.success?1:0)<<' '<<sanitize_error(d.error)<<'\n';
    }
}

static std::vector<ZeroRecord> refine_f3iso_candidate_sideclass(int method, double a, double b, const std::vector<int>& nnP, const std::string& irrep, const PhysicsParams& par, char debug, const Options& opt, const std::string& component, const std::string& reflog_prefix) {
    std::vector<ZeroRecord> out;
    auto rg = evaluate_refine_grid_data_parallel(method,a,b,nnP,irrep,par,debug,opt);
    if(opt.write_refine_logs) write_refine_log_data(reflog_prefix+"_"+method_name(method)+"_refined_grid.dat",method,rg,component);
    const int M=(int)rg.size();
    std::vector<double> y(M), ay(M);
    for(int i=0;i<M;++i){ y[i]=method_value(rg[i],method,component); ay[i]=std::abs(y[i]); }
    for(int j=0;j+1<M;++j){
        if(!sign_flip(y[j],y[j+1])) continue;
        SideClass sc=classify_side_abs(ay,j,opt);
        ZeroRecord z;
        z.method=method_name(method); z.kind="sign_refined_sideclass"; z.E_left=rg[j].Ecm; z.E_right=rg[j+1].Ecm; z.y_left=y[j]; z.y_right=y[j+1]; z.crossing_index=j;
        z.classifier=sc.classifier; z.left_score=sc.left_score; z.right_score=sc.right_score; z.reason=sc.classifier;
        if(sc.likely_zero){
            z=bisection_f3iso_zero(a,b,nnP,irrep,par,debug,opt.refine_iter,component);
            z.crossing_index=j; z.classifier=sc.classifier; z.left_score=sc.left_score; z.right_score=sc.right_score;
            z.accepted = (!opt.accept_only_likely_zero || (z.abs_y <= opt.zero_abs_tol)) ? 1 : 0;
            if(!z.accepted) z.reason="rejected_after_bisection_abs_y_too_large";
        } else {
            z.E=0.5*(rg[j].Ecm+rg[j+1].Ecm); z.y=0.5*(y[j]+y[j+1]); z.abs_y=std::min(ay[j],ay[j+1]); z.accepted=0;
        }
        out.push_back(z);
    }
    return out;
}

struct BranchPoint {
    Eigen::VectorXcd eigvals;
    Eigen::MatrixXcd eigvecs;
    bool success=false;
};

static BranchPoint eigensystem_for_method(const EvalFull& ef, int method) {
    BranchPoint bp;
    if(!ef.d.success) return bp;
    const Eigen::MatrixXcd& M = (method==1) ? ef.F3inv : ef.projF3inv;
    if(M.rows()==0 || M.cols()==0 || M.rows()!=M.cols()) return bp;
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> ces(M, true);
    bp.eigvals=ces.eigenvalues();
    bp.eigvecs=ces.eigenvectors();
    for(int c=0;c<bp.eigvecs.cols();++c){ const double n=bp.eigvecs.col(c).norm(); if(n>0.0) bp.eigvecs.col(c)/=n; }
    bp.success=true;
    return bp;
}

static std::vector<std::vector<comp>> tracked_eigenvalue_branches(const std::vector<EvalFull>& rg, int method) {
    const int M=(int)rg.size();
    std::vector<BranchPoint> bp(M);
    for(int i=0;i<M;++i) bp[i]=eigensystem_for_method(rg[i],method);
    int D=0; for(int i=0;i<M;++i) if(bp[i].success){ D=(int)bp[i].eigvals.size(); break; }
    std::vector<std::vector<comp>> branches(D, std::vector<comp>(M, comp(NAN,NAN)));
    if(D==0) return branches;
    std::vector<int> prev_order(D); for(int k=0;k<D;++k) prev_order[k]=k;
    int first=-1; for(int i=0;i<M;++i) if(bp[i].success && (int)bp[i].eigvals.size()==D){ first=i; break; }
    if(first<0) return branches;
    for(int k=0;k<D;++k) branches[k][first]=bp[first].eigvals[k];
    Eigen::MatrixXcd prev_vecs=bp[first].eigvecs;
    for(int i=first+1;i<M;++i){
        if(!bp[i].success || (int)bp[i].eigvals.size()!=D) continue;
        std::vector<int> used(D,0), match(D,-1);
        for(int old=0; old<D; ++old){
            double best=-1.0; int bestj=-1;
            for(int cur=0; cur<D; ++cur){ if(used[cur]) continue; double ov=std::abs((prev_vecs.col(old).adjoint()*bp[i].eigvecs.col(cur))(0,0)); if(ov>best){best=ov; bestj=cur;} }
            if(bestj>=0){ match[old]=bestj; used[bestj]=1; }
        }
        Eigen::MatrixXcd new_prev=prev_vecs;
        for(int old=0; old<D; ++old){ if(match[old]>=0){ branches[old][i]=bp[i].eigvals[match[old]]; new_prev.col(old)=bp[i].eigvecs.col(match[old]); } }
        prev_vecs=new_prev;
    }
    return branches;
}

static void write_refine_log_branch(const std::string& file, int method, const std::vector<EvalFull>& rg, const std::vector<std::vector<comp>>& branches) {
    std::ofstream out(file); out<<std::setprecision(17);
    out << "# refined cached full-matrix/eigenbranch grid for "<<method_name(method)<<"\n";
    out << "# columns: branch j Ecm lambda_re lambda_im lambda_abs raw_det_re raw_det_abs min_sv_F3 min_sv_projF3inv cond_F3 cond_projF3inv\n";
    for(size_t b=0;b<branches.size();++b){
        for(size_t j=0;j<branches[b].size();++j){
            const auto& d=rg[j].d; const comp raw=(method==1)?d.detF3inv:d.detProjF3inv;
            out<<b<<' '<<j<<' '<<d.Ecm<<' '<<branches[b][j].real()<<' '<<branches[b][j].imag()<<' '<<std::abs(branches[b][j])<<' '<<raw.real()<<' '<<std::abs(raw)<<' '<<d.min_sv_F3<<' '<<d.min_sv_projF3inv<<' '<<d.cond_F3<<' '<<d.cond_projF3inv<<'\n';
        }
    }
}


static bool eval_minabs_eigenvalue_real(int method, double E, const std::vector<int>& nnP, const std::string& irrep, const PhysicsParams& par, char debug, double& y, double& absval) {
    EvalFull ef = evaluate_full(E,nnP,irrep,par,debug);
    if(!ef.d.success) return false;
    BranchPoint bp = eigensystem_for_method(ef,method);
    if(!bp.success || bp.eigvals.size()==0) return false;
    double best = std::numeric_limits<double>::infinity();
    int besti=-1;
    for(int q=0;q<bp.eigvals.size();++q){
        const double a = std::abs(bp.eigvals[q]);
        if(std::isfinite(a) && a<best){ best=a; besti=q; }
    }
    if(besti<0) return false;
    y = bp.eigvals[besti].real();
    absval = best;
    return std::isfinite(y) && std::isfinite(absval);
}

static bool eval_reference_tracked_eigenvalue_real(
        int method,
        double E,
        const std::vector<int>& nnP,
        const std::string& irrep,
        const PhysicsParams& par,
        char debug,
        const Eigen::VectorXcd& refvec,
        double& y,
        double& absval,
        double& overlap) {
    EvalFull ef = evaluate_full(E,nnP,irrep,par,debug);
    if(!ef.d.success) return false;
    BranchPoint bp = eigensystem_for_method(ef,method);
    if(!bp.success || bp.eigvals.size()==0 || bp.eigvecs.cols()!=bp.eigvals.size()) return false;
    int besti=-1;
    double bestov=-1.0;
    for(int q=0;q<bp.eigvals.size();++q){
        if(bp.eigvecs.rows()!=refvec.size()) continue;
        const double ov = std::abs((refvec.adjoint()*bp.eigvecs.col(q))(0,0));
        if(std::isfinite(ov) && ov>bestov){ bestov=ov; besti=q; }
    }
    if(besti<0) return false;
    y = bp.eigvals[besti].real();
    absval = std::abs(bp.eigvals[besti]);
    overlap = bestov;
    return std::isfinite(y) && std::isfinite(absval) && std::isfinite(overlap);
}

static ZeroRecord bisection_eigenbranch_reference_zero(
        int method,
        double a,
        double b,
        const std::vector<int>& nnP,
        const std::string& irrep,
        const PhysicsParams& par,
        char debug,
        int iter,
        const Eigen::VectorXcd& refvec) {
    // v29o: branch-specific bisection on the original coarse window [E_i,E_{i+1}].
    // At every bisection energy, choose the eigenvalue whose eigenvector has the
    // largest overlap with the reference eigenvector at the left endpoint.  This
    // keeps the scalar being bisected on the same local branch instead of jumping
    // to the instantaneous minimum-|lambda| branch.
    double fa=NAN, fb=NAN, aa=NAN, ab=NAN, ova=NAN, ovb=NAN;
    if(!eval_reference_tracked_eigenvalue_real(method,a,nnP,irrep,par,debug,refvec,fa,aa,ova) ||
       !eval_reference_tracked_eigenvalue_real(method,b,nnP,irrep,par,debug,refvec,fb,ab,ovb) ||
       !(fa==0.0 || fb==0.0 || fa*fb<=0.0)) {
        ZeroRecord z; z.method=method_name(method); z.kind="eigenbranch_reference_bisection_failed"; z.E=0.5*(a+b); z.E_left=a; z.E_right=b;
        z.y=NAN; z.abs_y=INFINITY; z.y_left=fa; z.y_right=fb; z.accepted=0;
        z.reason="coarse_window_endpoints_do_not_bracket_reference_tracked_eigenvalue_after_re-evaluation";
        z.left_score=ova; z.right_score=ovb;
        return z;
    }
    const double fa0=fa, fb0=fb;
    double left=a, right=b;
    double ovlast=std::min(ova,ovb);
    for(int k=0;k<iter;++k){
        const double mid=0.5*(left+right);
        double fm=NAN, am=NAN, ovm=NAN;
        if(!eval_reference_tracked_eigenvalue_real(method,mid,nnP,irrep,par,debug,refvec,fm,am,ovm)) break;
        ovlast=ovm;
        if(fa==0.0){ right=left; fb=fa; break; }
        if(fb==0.0){ left=right; fa=fb; break; }
        if(fa*fm<=0.0){ right=mid; fb=fm; }
        else { left=mid; fa=fm; }
    }
    const double E=0.5*(left+right);
    double y=NAN, ay=INFINITY, ov=NAN;
    eval_reference_tracked_eigenvalue_real(method,E,nnP,irrep,par,debug,refvec,y,ay,ov);
    ZeroRecord z;
    z.method=method_name(method);
    z.kind="eigenbranch_reference_coarse_window_bisection";
    z.E=E; z.E_left=a; z.E_right=b; z.y=y; z.abs_y=ay; z.branch_abs=ay;
    z.y_left=fa0; z.y_right=fb0;
    z.left_score=std::isfinite(ov)?ov:ovlast;
    z.reason="tracked_eigenbranch_side_abs_likely_zero_then_reference_branch_bisection_on_original_coarse_window";
    return z;
}

static ZeroRecord bisection_eigenbranch_minabs_zero(int method, double a, double b, const std::vector<int>& nnP, const std::string& irrep, const PhysicsParams& par, char debug, int iter) {
    // v29o: final determinant/eigenbranch refinement is bisection on the ORIGINAL
    // coarse-grid eigenbranch sign-flip window [E_i,E_{i+1}].  The refined grid is
    // used for branch tracking and zero-vs-pole side classification.  During the
    // bisection, the scalar function is the real part of the eigenvalue with the
    // smallest absolute value at that energy.  This avoids raw determinant overflow
    // and gives a branch-local zero residual check.
    double fa=NAN, fb=NAN, aa=NAN, ab=NAN;
    if(!eval_minabs_eigenvalue_real(method,a,nnP,irrep,par,debug,fa,aa) ||
       !eval_minabs_eigenvalue_real(method,b,nnP,irrep,par,debug,fb,ab) ||
       !(fa==0.0 || fb==0.0 || fa*fb<=0.0)) {
        ZeroRecord z; z.method=method_name(method); z.kind="eigenbranch_coarse_window_bisection_failed"; z.E=0.5*(a+b); z.E_left=a; z.E_right=b;
        z.y=NAN; z.abs_y=INFINITY; z.y_left=fa; z.y_right=fb; z.accepted=0;
        z.reason="coarse_eigenbranch_window_endpoints_do_not_bracket_minabs_eigenvalue_after_re-evaluation";
        return z;
    }
    const double fa0=fa, fb0=fb;
    double left=a, right=b;
    for(int k=0;k<iter;++k){
        const double mid=0.5*(left+right);
        double fm=NAN, am=NAN;
        if(!eval_minabs_eigenvalue_real(method,mid,nnP,irrep,par,debug,fm,am)) break;
        if(fa==0.0){ right=left; fb=fa; break; }
        if(fb==0.0){ left=right; fa=fb; break; }
        if(fa*fm<=0.0){ right=mid; fb=fm; }
        else { left=mid; fa=fm; }
    }
    const double E=0.5*(left+right);
    double y=NAN, ay=INFINITY;
    eval_minabs_eigenvalue_real(method,E,nnP,irrep,par,debug,y,ay);
    ZeroRecord z;
    z.method=method_name(method);
    z.kind="eigenbranch_coarse_window_bisection_minabs";
    z.E=E; z.E_left=a; z.E_right=b; z.y=y; z.abs_y=ay; z.branch_abs=ay;
    z.y_left=fa0; z.y_right=fb0;
    z.reason="tracked_eigenbranch_side_abs_likely_zero_then_bisection_on_original_coarse_eigenbranch_window";
    return z;
}

static std::vector<ZeroRecord> refine_det_candidate_eigenbranch(int method, double a, double b, const std::vector<int>& nnP, const std::string& irrep, const PhysicsParams& par, char debug, const Options& opt, const std::string& reflog_prefix) {
    std::vector<ZeroRecord> out;
    auto rg = evaluate_refine_grid_full_parallel(a,b,nnP,irrep,par,debug,opt);
    const auto branches = tracked_eigenvalue_branches(rg,method);
    if(opt.write_refine_logs) write_refine_log_branch(reflog_prefix+"_"+method_name(method)+"_eigenbranches.dat",method,rg,branches);
    const int M=(int)rg.size();
    for(size_t br=0;br<branches.size();++br){
        std::vector<double> y(M), ay(M);
        for(int i=0;i<M;++i){ y[i]=branches[br][i].real(); ay[i]=std::abs(branches[br][i]); }
        for(int j=0;j+1<M;++j){
            if(!sign_flip(y[j],y[j+1])) continue;
            SideClass sc=classify_side_abs(ay,j,opt);
            ZeroRecord z;
            z.method=method_name(method); z.kind="eigenbranch_sideclass"; z.E_left=rg[j].d.Ecm; z.E_right=rg[j+1].d.Ecm; z.y_left=y[j]; z.y_right=y[j+1]; z.crossing_index=j;
            z.classifier=sc.classifier; z.left_score=sc.left_score; z.right_score=sc.right_score; z.branch_index=(int)br; z.reason=sc.classifier;
            // v29o: determinant zeros are obtained from the tracked eigenbranch,
            // not by raw determinant bisection.  The coarse-grid eigenbranch sign flip opens
            // the coarse search window; the refined eigenbranch crossing classifies it.
            const bool local_zero_like = local_zero_like_crossing(ay,j,opt);
            if(sc.likely_zero || local_zero_like){
                BranchPoint bp_left = eigensystem_for_method(rg.front(), method);
                if(bp_left.success && br < (size_t)bp_left.eigvecs.cols()) {
                    z = bisection_eigenbranch_reference_zero(method,a,b,nnP,irrep,par,debug,opt.refine_iter,bp_left.eigvecs.col((int)br));
                } else {
                    z = bisection_eigenbranch_minabs_zero(method,a,b,nnP,irrep,par,debug,opt.refine_iter);
                    z.reason="reference_branch_unavailable_fallback_to_minabs_bisection";
                }
                z.crossing_index=j;
                z.classifier=sc.likely_zero ? sc.classifier : "local_zero_side_abs_decreases";
                z.left_score=sc.left_score;
                z.right_score=sc.right_score;
                z.branch_index=(int)br;
                z.branch_abs=z.abs_y;
                z.accepted = (!opt.accept_only_likely_zero || (z.abs_y <= opt.zero_abs_tol)) ? 1 : 0;
                if(z.accepted) z.reason=sc.likely_zero ? "tracked_eigenbranch_side_abs_likely_zero_reference_branch_bisection_residual_ok" : "tracked_eigenbranch_local_zero_reference_branch_bisection_residual_ok";
                else z.reason=sc.likely_zero ? "tracked_eigenbranch_likely_zero_but_reference_branch_bisection_residual_too_large" : "tracked_eigenbranch_local_zero_but_reference_branch_bisection_residual_too_large";
            } else {
                z.E=0.5*(rg[j].d.Ecm+rg[j+1].d.Ecm); z.y=0.5*(y[j]+y[j+1]); z.abs_y=std::min(ay[j],ay[j+1]); z.branch_abs=z.abs_y; z.accepted=0;
            }
            out.push_back(z);
        }
    }
    return out;
}

struct ZeroCandidate {
    int method=0;
    double a=NAN, b=NAN;
    std::string kind;
    int coarse_i=-1;
    int branch_index=-1;
};

static std::vector<ZeroCandidate> collect_zero_candidates_for_method(int method, const std::vector<EvalData>& grid, const Options& opt, const std::string& component) {
    // Scalar method only: F3iso^{-1}.
    // v29o intentionally keeps this as a normal scalar sign-flip scan.
    std::vector<ZeroCandidate> cands;
    for(size_t i=0;i+1<grid.size();++i){
        if(!grid[i].success || !grid[i+1].success) continue;
        double y0=method_value(grid[i],method,component), y1=method_value(grid[i+1],method,component);
        if(!std::isfinite(y0)||!std::isfinite(y1)) continue;
        if(sign_flip(y0,y1)) {
            ZeroCandidate z; z.method=method; z.a=grid[i].Ecm; z.b=grid[i+1].Ecm; z.kind="coarse_scalar_signflip"; z.coarse_i=(int)i;
            cands.push_back(z);
        }
    }
    (void)opt;
    return cands;
}

static std::vector<EvalFull> evaluate_full_grid_parallel(const Options& opt, const std::vector<int>& nnP, const std::string& irrep, const PhysicsParams& par) {
    std::vector<EvalFull> grid(opt.N);
    auto eval_one = [&](int i){
        double t=double(i)/double(opt.N-1);
        double E=opt.E0+t*(opt.E1-opt.E0);
        grid[i]=evaluate_full(E,nnP,irrep,par,opt.debug);
        grid[i].d.i=i;
    };
    if(!opt.coarse_parallel) {
        for(int i=0;i<opt.N;++i) eval_one(i);
        return grid;
    }
    if(opt.omp_schedule=="static") {
        #pragma omp parallel for schedule(static)
        for(int i=0;i<opt.N;++i) eval_one(i);
    } else if(opt.omp_schedule=="guided") {
        #pragma omp parallel for schedule(guided)
        for(int i=0;i<opt.N;++i) eval_one(i);
    } else {
        #pragma omp parallel for schedule(dynamic,1)
        for(int i=0;i<opt.N;++i) eval_one(i);
    }
    return grid;
}

static std::vector<EvalData> extract_eval_data_grid(const std::vector<EvalFull>& fullgrid) {
    std::vector<EvalData> out(fullgrid.size());
    for(size_t i=0;i<fullgrid.size();++i) out[i]=fullgrid[i].d;
    return out;
}

static std::vector<BranchPoint> coarse_eigensystems_for_method(const std::vector<EvalFull>& fullgrid, int method) {
    std::vector<BranchPoint> bp(fullgrid.size());
    for(size_t i=0;i<fullgrid.size();++i) bp[i]=eigensystem_for_method(fullgrid[i],method);
    return bp;
}

static std::vector<ZeroCandidate> collect_det_eigenbranch_candidates_from_adjacent_pairs(
        int method,
        const std::vector<EvalFull>& fullgrid,
        const Options& opt,
        const std::string& scan_summary_file) {
    // v29o: determinant methods are scanned locally on every coarse interval.
    // For each [E_i,E_{i+1}], we diagonalize/cache both endpoints, greedily match
    // eigenvectors by overlap, and check every matched branch for a real-part sign flip.
    // This avoids the v29n failure mode where one global branch track could lose later
    // crossings due to branch swaps/dimension changes and only create one candidate.
    std::vector<ZeroCandidate> cands;
    const int M=(int)fullgrid.size();
    const auto bp = coarse_eigensystems_for_method(fullgrid, method);

    std::ofstream scan(scan_summary_file);
    scan << std::setprecision(17);
    scan << "# v29p adjacent-pair eigenbranch scan summary for " << method_name(method) << "\n";
    scan << "# columns: method coarse_i E_left E_right dim_left dim_right left_branch right_branch lambda_left_re lambda_left_im lambda_left_abs lambda_right_re lambda_right_im lambda_right_abs overlap sign_flip candidate_created reason\n";

    for(int i=0;i+1<M;++i){
        const double Eleft  = fullgrid[i].d.Ecm;
        const double Eright = fullgrid[i+1].d.Ecm;
        if(!fullgrid[i].d.success || !fullgrid[i+1].d.success || !bp[i].success || !bp[i+1].success){
            scan << method_name(method) << ' ' << i << ' ' << Eleft << ' ' << Eright << ' '
                 << (bp[i].success?bp[i].eigvals.size():0) << ' ' << (bp[i+1].success?bp[i+1].eigvals.size():0)
                 << " -1 -1 nan nan nan nan nan nan nan 0 0 endpoint_eval_or_eigensystem_failed\n";
            continue;
        }
        const int D0=(int)bp[i].eigvals.size();
        const int D1=(int)bp[i+1].eigvals.size();
        if(D0<=0 || D1<=0 || bp[i].eigvecs.cols()!=D0 || bp[i+1].eigvecs.cols()!=D1){
            scan << method_name(method) << ' ' << i << ' ' << Eleft << ' ' << Eright << ' '
                 << D0 << ' ' << D1 << " -1 -1 nan nan nan nan nan nan nan 0 0 bad_eigensystem_dimensions\n";
            continue;
        }

        const int Dmatch = std::min(D0,D1);
        std::vector<int> used(D1,0);
        for(int old=0; old<D0; ++old){
            double best=-1.0;
            int bestj=-1;
            for(int cur=0; cur<D1; ++cur){
                if(used[cur]) continue;
                const double ov=std::abs((bp[i].eigvecs.col(old).adjoint()*bp[i+1].eigvecs.col(cur))(0,0));
                if(std::isfinite(ov) && ov>best){ best=ov; bestj=cur; }
            }
            if(bestj<0){
                scan << method_name(method) << ' ' << i << ' ' << Eleft << ' ' << Eright << ' '
                     << D0 << ' ' << D1 << ' ' << old << " -1 "
                     << bp[i].eigvals[old].real() << ' ' << bp[i].eigvals[old].imag() << ' ' << std::abs(bp[i].eigvals[old])
                     << " nan nan nan nan 0 0 no_unmatched_right_branch\n";
                continue;
            }
            used[bestj]=1;
            const comp l0=bp[i].eigvals[old];
            const comp l1=bp[i+1].eigvals[bestj];
            const double y0=l0.real();
            const double y1=l1.real();
            const bool finite = std::isfinite(y0) && std::isfinite(y1) && std::isfinite(std::abs(l0)) && std::isfinite(std::abs(l1));
            const bool flip = finite && sign_flip(y0,y1);
            int created=0;
            std::string reason = finite ? (flip ? "coarse_adjacent_branch_signflip" : "no_branch_signflip") : "nonfinite_branch_value";
            if(flip){
                ZeroCandidate z;
                z.method=method;
                z.a=Eleft;
                z.b=Eright;
                z.kind="coarse_adjacent_eigenbranch_signflip_all_intervals";
                z.coarse_i=i;
                z.branch_index=old;
                cands.push_back(z);
                created=1;
            }
            scan << method_name(method) << ' ' << i << ' ' << Eleft << ' ' << Eright << ' '
                 << D0 << ' ' << D1 << ' ' << old << ' ' << bestj << ' '
                 << l0.real() << ' ' << l0.imag() << ' ' << std::abs(l0) << ' '
                 << l1.real() << ' ' << l1.imag() << ' ' << std::abs(l1) << ' '
                 << best << ' ' << (flip?1:0) << ' ' << created << ' ' << reason << '\n';
            if(old+1>=Dmatch && D0!=D1) {
                // Continue writing matched old branches when D0>D1; unmatched left branches are logged above.
            }
        }
    }
    (void)opt;
    return cands;
}

static std::vector<ZeroRecord> refine_zero_candidates_parallel(const std::vector<ZeroCandidate>& cands, const std::vector<int>& nnP, const std::string& irrep, const PhysicsParams& par, char debug, const Options& opt, const std::string& component, const std::string& reflog_base) {
    std::vector<std::vector<ZeroRecord>> per(cands.size());
    if(cands.empty()) return {};
    if(opt.parallel_refine){
        #pragma omp parallel for schedule(dynamic,1)
        for(int i=0;i<(int)cands.size();++i){
            const auto& c=cands[i];
            std::ostringstream pref; pref<<reflog_base<<"_cand"<<std::setw(4)<<std::setfill('0')<<i<<std::setfill(' ');
            if(c.method==0) per[i]=refine_f3iso_candidate_sideclass(c.method,c.a,c.b,nnP,irrep,par,debug,opt,component,pref.str());
            else per[i]=refine_det_candidate_eigenbranch(c.method,c.a,c.b,nnP,irrep,par,debug,opt,pref.str());
        }
    } else {
        for(int i=0;i<(int)cands.size();++i){
            const auto& c=cands[i];
            std::ostringstream pref; pref<<reflog_base<<"_cand"<<std::setw(4)<<std::setfill('0')<<i<<std::setfill(' ');
            if(c.method==0) per[i]=refine_f3iso_candidate_sideclass(c.method,c.a,c.b,nnP,irrep,par,debug,opt,component,pref.str());
            else per[i]=refine_det_candidate_eigenbranch(c.method,c.a,c.b,nnP,irrep,par,debug,opt,pref.str());
        }
    }
    std::vector<ZeroRecord> out;
    for(auto& v:per) for(auto& z:v) out.push_back(z);
    return out;
}

static double zero_record_residual_for_merge(const ZeroRecord& z) {
    if(std::isfinite(z.branch_abs)) return z.branch_abs;
    if(std::isfinite(z.abs_y)) return z.abs_y;
    return std::numeric_limits<double>::infinity();
}

static bool same_coarse_interval(const ZeroRecord& a, const ZeroRecord& b, double tol) {
    return std::abs(a.E_left-b.E_left) <= tol && std::abs(a.E_right-b.E_right) <= tol;
}

static std::vector<ZeroRecord> dedup_eigenbranch_candidate_records(std::vector<ZeroRecord> zlist, const Options& opt) {
    std::sort(zlist.begin(), zlist.end(), [](const ZeroRecord& a, const ZeroRecord& b){
        if(a.E_left != b.E_left) return a.E_left < b.E_left;
        if(a.E_right != b.E_right) return a.E_right < b.E_right;
        if(a.E != b.E) return a.E < b.E;
        return zero_record_residual_for_merge(a) < zero_record_residual_for_merge(b);
    });
    std::vector<ZeroRecord> out;
    std::vector<int> used(zlist.size(),0);
    for(size_t i=0;i<zlist.size();++i){
        if(used[i]) continue;
        used[i]=1;
        ZeroRecord best=zlist[i];
        int count=1;
        const double bestE0=zlist[i].E;
        for(size_t j=i+1;j<zlist.size();++j){
            if(used[j]) continue;
            if(!same_coarse_interval(zlist[i], zlist[j], opt.duplicate_tol)) continue;
            if(!(std::isfinite(bestE0) && std::isfinite(zlist[j].E))) continue;
            if(std::abs(zlist[j].E-bestE0) > opt.duplicate_tol) continue;
            used[j]=1;
            ++count;
            if(zero_record_residual_for_merge(zlist[j]) < zero_record_residual_for_merge(best)) best=zlist[j];
        }
        best.merged_count=count;
        out.push_back(best);
    }
    std::sort(out.begin(), out.end(), [](const ZeroRecord& a, const ZeroRecord& b){ return a.E < b.E; });
    return out;
}

static std::vector<ZeroRecord> unique_sorted_zeros(std::vector<ZeroRecord> zlist, const Options& opt) {
    std::sort(zlist.begin(),zlist.end(),[](const ZeroRecord&a,const ZeroRecord&b){return a.E<b.E;});
    std::vector<ZeroRecord> unique;
    for(auto& z:zlist){
        if(opt.accept_only_likely_zero && z.accepted!=1) { /* keep rejected records out of final zero list */ continue; }
        if(unique.empty() || std::abs(z.E-unique.back().E)>opt.duplicate_tol) unique.push_back(z);
        else {
            const int merged_total = unique.back().merged_count + z.merged_count;
            if(zero_record_residual_for_merge(z) < zero_record_residual_for_merge(unique.back())) unique.back()=z;
            unique.back().merged_count = merged_total;
        }
    }
    for(size_t i=0;i<unique.size();++i) unique[i].index=int(i);
    return unique;
}

static std::vector<EvalData> evaluate_grid_parallel(const Options& opt, const std::vector<int>& nnP, const std::string& irrep, const PhysicsParams& par) {
    std::vector<EvalData> grid(opt.N);
    auto eval_one = [&](int i){
        double t=double(i)/double(opt.N-1);
        double E=opt.E0+t*(opt.E1-opt.E0);
        grid[i]=evaluate_data(E,nnP,irrep,par,opt.debug);
        grid[i].i=i;
    };

    if(!opt.coarse_parallel) {
        for(int i=0;i<opt.N;++i) eval_one(i);
        return grid;
    }

    if(opt.omp_schedule=="static") {
        #pragma omp parallel for schedule(static)
        for(int i=0;i<opt.N;++i) eval_one(i);
    } else if(opt.omp_schedule=="guided") {
        #pragma omp parallel for schedule(guided)
        for(int i=0;i<opt.N;++i) eval_one(i);
    } else {
        #pragma omp parallel for schedule(dynamic,1)
        for(int i=0;i<opt.N;++i) eval_one(i);
    }
    return grid;
}


static std::vector<ZeroRecord> find_candidate_records_for_method(int method, const std::vector<EvalData>& grid, const std::vector<int>& nnP, const std::string& irrep, const PhysicsParams& par, char debug, const Options& opt, const std::string& component, const std::string& reflog_base) {
    const auto cands = collect_zero_candidates_for_method(method, grid, opt, component);
    return refine_zero_candidates_parallel(cands, nnP, irrep, par, debug, opt, component, reflog_base + "_" + method_name(method));
}

static std::vector<ZeroRecord> find_candidate_records_for_det_method(int method, const std::vector<EvalFull>& fullgrid, const std::vector<int>& nnP, const std::string& irrep, const PhysicsParams& par, char debug, const Options& opt, const std::string& component, const std::string& reflog_base) {
    const std::string scan_summary_file = reflog_base + "_" + method_name(method) + "_coarse_eigenbranch_scan_summary.dat";
    const auto cands = collect_det_eigenbranch_candidates_from_adjacent_pairs(method, fullgrid, opt, scan_summary_file);
    return refine_zero_candidates_parallel(cands, nnP, irrep, par, debug, opt, component, reflog_base + "_" + method_name(method));
}

static std::vector<ZeroRecord> find_zeros_for_method(int method, const std::vector<EvalData>& grid, const std::vector<int>& nnP, const std::string& irrep, const PhysicsParams& par, char debug, const Options& opt, const std::string& component, const std::string& reflog_base) {
    auto zlist = find_candidate_records_for_method(method, grid, nnP, irrep, par, debug, opt, component, reflog_base);
    return unique_sorted_zeros(std::move(zlist), opt);
}


static void dump_matrix(const std::string& file, const Eigen::MatrixXcd& M) {
    std::ofstream out(file); out<<std::setprecision(17);
    out << "# rows cols " << M.rows() << ' ' << M.cols() << "\n# i j real imag abs\n";
    for(int i=0;i<M.rows();++i) for(int j=0;j<M.cols();++j) out << i << ' ' << j << ' ' << M(i,j).real() << ' ' << M(i,j).imag() << ' ' << std::abs(M(i,j)) << '\n';
}

static void write_grid_file(const std::string& file, const std::vector<EvalData>& g, const std::string& label, const MomentumIrrepSpec& spec, const std::string& input_file) {
    std::ofstream out(file); out<<std::setprecision(17);
    out << "# v29p F3inv zero comparison grid\n# input_file "<<input_file<<"\n";
    out << "# label "<<label<<" nnP "<<spec.nnP[0]<<' '<<spec.nnP[1]<<' '<<spec.nnP[2]<<" irrep "<<spec.irrep<<" irrep_tag "<<spec.irrep_tag<<"\n";
    out << "# columns:\n# i Ecm En A B total_dim vdim success "
        << "F3iso_re F3iso_im F3iso_abs F3isoInv_re F3isoInv_im F3isoInv_abs "
        << "detF3_re detF3_im detF3_abs detF3inv_re detF3inv_im detF3inv_abs "
        << "detProjF3inv_re detProjF3inv_im detProjF3inv_abs min_sv_F3 max_sv_F3 cond_F3 min_sv_projF3inv max_sv_projF3inv cond_projF3inv herm_F3inv_rel herm_projF3inv_rel error\n";
    for(const auto& r:g){
        out<<r.i<<' '<<r.Ecm<<' '<<r.En<<' '<<r.A<<' '<<r.B<<' '<<r.total_dim<<' '<<r.vdim<<' '<<(r.success?1:0)<<' '
           <<r.F3iso.real()<<' '<<r.F3iso.imag()<<' '<<std::abs(r.F3iso)<<' '
           <<r.F3iso_inv.real()<<' '<<r.F3iso_inv.imag()<<' '<<std::abs(r.F3iso_inv)<<' '
           <<r.detF3.real()<<' '<<r.detF3.imag()<<' '<<std::abs(r.detF3)<<' '
           <<r.detF3inv.real()<<' '<<r.detF3inv.imag()<<' '<<std::abs(r.detF3inv)<<' '
           <<r.detProjF3inv.real()<<' '<<r.detProjF3inv.imag()<<' '<<std::abs(r.detProjF3inv)<<' '
           <<r.min_sv_F3<<' '<<r.max_sv_F3<<' '<<r.cond_F3<<' '<<r.min_sv_projF3inv<<' '<<r.max_sv_projF3inv<<' '<<r.cond_projF3inv<<' '
           <<r.herm_F3inv_rel<<' '<<r.herm_projF3inv_rel<<' '<<sanitize_error(r.error)<<'\n';
    }
}

static void append_zero_records(std::ofstream& out, const MomentumIrrepSpec& spec, std::vector<ZeroRecord> zs) {
    for(auto& z:zs){
        out<<spec.label<<' '<<spec.nnP[0]<<' '<<spec.nnP[1]<<' '<<spec.nnP[2]<<' '<<spec.irrep<<' '<<spec.irrep_tag<<' '
           <<z.method<<' '<<z.index<<' '<<z.E<<' '<<z.E_left<<' '<<z.E_right<<' '<<z.y<<' '<<z.abs_y<<' '<<z.kind<<' '<<z.accepted<<' '<<z.reason<<' '
           <<z.classifier<<' '<<z.crossing_index<<' '<<z.left_score<<' '<<z.right_score<<' '<<z.branch_index<<' '<<z.branch_abs<<' '<<z.merged_count<<'\n';
    }
}

struct ZeroClassificationRow {
    std::string classification;
    double E_ref=NAN, E_scalar=NAN, E_full=NAN, E_projected=NAN;
    int scalar_index=-1, full_index=-1, projected_index=-1;
    int scalar_multiplicity=0, full_multiplicity=0, projected_multiplicity=0;
};

struct ZeroClassificationSummary {
    std::vector<ZeroClassificationRow> rows;
    int matched_all_three=0;
    int matched_projected_only=0;
    int full_det_extra=0;
    int projected_extra=0;
    int scalar_extra=0;
};

static int nearest_index_within(const std::vector<ZeroRecord>& v, double E, double tol) {
    int best=-1;
    double delta=std::numeric_limits<double>::infinity();
    for(int i=0;i<(int)v.size();++i){
        const double d=std::abs(v[i].E-E);
        if(std::isfinite(d) && d<delta){ delta=d; best=i; }
    }
    return (best>=0 && delta<=tol) ? best : -1;
}

static ZeroClassificationSummary classify_zeros(const std::vector<ZeroRecord>& scalar, const std::vector<ZeroRecord>& full, const std::vector<ZeroRecord>& projected, double tol) {
    ZeroClassificationSummary s;
    std::vector<int> used_full(full.size(),0), used_projected(projected.size(),0);
    for(const auto& z: scalar){
        const int ip=nearest_index_within(projected,z.E,tol);
        const int iff=nearest_index_within(full,z.E,tol);
        ZeroClassificationRow r;
        r.E_ref=z.E; r.E_scalar=z.E; r.scalar_index=z.index; r.scalar_multiplicity=z.merged_count;
        if(ip>=0){ r.E_projected=projected[ip].E; r.projected_index=projected[ip].index; r.projected_multiplicity=projected[ip].merged_count; used_projected[ip]=1; }
        if(iff>=0){ r.E_full=full[iff].E; r.full_index=full[iff].index; r.full_multiplicity=full[iff].merged_count; used_full[iff]=1; }
        if(ip>=0 && iff>=0){ r.classification="matched_all_three"; ++s.matched_all_three; s.rows.push_back(r); }
        else if(ip>=0){ r.classification="matched_projected_only"; ++s.matched_projected_only; s.rows.push_back(r); }
        else { r.classification="scalar_extra"; ++s.scalar_extra; s.rows.push_back(r); }
    }
    for(int i=0;i<(int)full.size();++i){
        if(used_full[i]) continue;
        const int is=nearest_index_within(scalar,full[i].E,tol);
        const int ip=nearest_index_within(projected,full[i].E,tol);
        if(is>=0 || ip>=0) continue;
        ZeroClassificationRow r;
        r.classification="full_det_extra"; r.E_ref=full[i].E; r.E_full=full[i].E; r.full_index=full[i].index; r.full_multiplicity=full[i].merged_count;
        ++s.full_det_extra; s.rows.push_back(r);
    }
    for(int i=0;i<(int)projected.size();++i){
        if(used_projected[i]) continue;
        const int is=nearest_index_within(scalar,projected[i].E,tol);
        if(is>=0) continue;
        ZeroClassificationRow r;
        r.classification="projected_extra"; r.E_ref=projected[i].E; r.E_projected=projected[i].E; r.projected_index=projected[i].index; r.projected_multiplicity=projected[i].merged_count;
        ++s.projected_extra; s.rows.push_back(r);
    }
    std::sort(s.rows.begin(), s.rows.end(), [](const ZeroClassificationRow& a, const ZeroClassificationRow& b){ return a.E_ref < b.E_ref; });
    return s;
}

static void write_zero_classification(const std::string& file, const MomentumIrrepSpec& spec, const ZeroClassificationSummary& cls, double tol) {
    std::ofstream out(file); out<<std::setprecision(17);
    out << "# v29p zero classification\n# label "<<spec.label<<"\n# match_tol "<<tol<<"\n";
    out << "# note detF3inv is the full-space determinant and may contain extra zeros not present in the projected irrep determinant.\n";
    out << "# columns: label classification E_ref E_F3iso_inv E_detF3inv E_detProjF3inv scalar_index full_index projected_index scalar_merged_count full_merged_count projected_merged_count\n";
    for(const auto& r: cls.rows){
        out << spec.label << ' ' << r.classification << ' ' << r.E_ref << ' ' << r.E_scalar << ' ' << r.E_full << ' ' << r.E_projected << ' '
            << r.scalar_index << ' ' << r.full_index << ' ' << r.projected_index << ' '
            << r.scalar_multiplicity << ' ' << r.full_multiplicity << ' ' << r.projected_multiplicity << '\n';
    }
}

static void write_match_summary(const std::string& file, const MomentumIrrepSpec& spec, const std::vector<ZeroRecord>& a, const std::vector<ZeroRecord>& b, const std::vector<ZeroRecord>& c, double tol, const ZeroClassificationSummary& cls) {
    std::ofstream out(file); out<<std::setprecision(17);
    out << "# v29p zero match summary\n# label "<<spec.label<<"\n# match_tol "<<tol<<"\n";
    out << "# note detF3inv is the full-space determinant and may contain extra zeros not present in the projected irrep determinant.\n";
    out << "n_F3iso_inv_zeros "<<a.size()<<"\n";
    out << "n_detF3inv_zeros "<<b.size()<<"\n";
    out << "n_detProjF3inv_zeros "<<c.size()<<"\n";
    out << "n_matched_all_three "<<cls.matched_all_three<<"\n";
    out << "n_matched_projected_only "<<cls.matched_projected_only<<"\n";
    out << "n_full_det_extra "<<cls.full_det_extra<<"\n";
    out << "n_projected_extra "<<cls.projected_extra<<"\n";
    out << "n_scalar_extra "<<cls.scalar_extra<<"\n";
    out << "n_projected_scalar_mismatches "<<(cls.projected_extra+cls.scalar_extra)<<"\n";
    out << "# columns: reference_method reference_zero_index E_ref nearest_detF3inv nearest_detF3inv_delta nearest_detProjF3inv nearest_detProjF3inv_delta all_match classification\n";
    auto nearest=[&](const std::vector<ZeroRecord>& v,double E){ double best=NAN,delta=std::numeric_limits<double>::infinity(); for(const auto& z:v){ double d=std::abs(z.E-E); if(d<delta){delta=d;best=z.E;} } return std::pair<double,double>(best,delta); };
    for(const auto& z:a){
        auto nb=nearest(b,z.E); auto nc=nearest(c,z.E);
        int ok=(std::isfinite(nb.second)&&std::isfinite(nc.second)&&nb.second<=tol&&nc.second<=tol)?1:0;
        std::string classification = ok ? "matched_all_three" : ((std::isfinite(nc.second)&&nc.second<=tol) ? "matched_projected_only" : "scalar_extra");
        out<<method_name(0)<<' '<<z.index<<' '<<z.E<<' '<<nb.first<<' '<<nb.second<<' '<<nc.first<<' '<<nc.second<<' '<<ok<<' '<<classification<<'\n';
    }
}

int main(int argc, char** argv) {
    if(argc!=2){ std::cerr<<"Usage:\n  "<<argv[0]<<" config/config_v29h_F3inv_zero_compare.in\n"; return 1; }
    const std::string input_file=argv[1];
    try{
        PhysicsParams par; Options opt=read_options(input_file,par);
        #ifdef _OPENMP
        omp_set_num_threads(opt.threads);
        omp_set_max_active_levels(1);
        #endif
        Eigen::setNbThreads(1);
        std::filesystem::create_directories(opt.outdir);
        std::filesystem::create_directories(opt.outdir+"/matrices");
        std::filesystem::create_directories(opt.outdir+"/refine_logs");
        const std::string allzero_file=opt.outdir+"/"+opt.prefix+"_all_zeros.dat";
        std::ofstream allz(allzero_file); allz<<std::setprecision(17);
        allz << "# v29p all accepted likely zeros from F3iso^{-1}, det(F3^{-1}), det(proj(F3^{-1}))\n";
        allz << "# input_file "<<input_file<<"\n# columns: label nPx nPy nPz irrep irrep_tag method zero_index Ecm E_left E_right y_real_or_component abs_y kind accepted reason classifier crossing_index left_score right_score branch_index branch_abs merged_count\n";
        std::ofstream meta(opt.outdir+"/"+opt.prefix+"_metadata.txt"); meta<<std::setprecision(17);
        meta << "version = v29p_adjacent_pair_eigenbranch_dedup_classification\ninput_file = "<<input_file<<"\nE0 = "<<opt.E0<<"\nE1 = "<<opt.E1<<"\nN = "<<opt.N<<"\nthreads = "<<opt.threads<<"\ncomponent = real\n"
             << "zero_abs_tol = "<<opt.zero_abs_tol<<"\nprojected_inverse_mode = inverse_of_projected_F3\ndet_zero_method = adjacent_pair_eigenbranch_all_coarse_intervals\nprojector_debug = n\nnote = detF3inv is the full-space determinant and may contain extra zeros not present in the projected irrep determinant.\nwrite_candidate_logs = "<<opt.write_candidate_logs<<"\nparallel_refine = "<<opt.parallel_refine<<"\nparallel_matrix_dumps = "<<opt.parallel_matrix_dumps<<"\ncoarse_parallel = "<<opt.coarse_parallel<<"\nomp_schedule = "<<opt.omp_schedule<<"\nomp_chunk = "<<opt.omp_chunk<<"\n";
        for(const std::string& lab: opt.mom_labels){
            MomentumIrrepSpec spec=parse_label(lab); std::vector<int> nnP={spec.nnP[0],spec.nnP[1],spec.nnP[2]};
            std::cout<<"[v29p-eigenbranch] scanning "<<lab<<" ...\n";
            std::cout<<"[v29p-eigenbranch] building cached coarse-grid full matrices: N="<<opt.N<<" threads="<<opt.threads<<" coarse_parallel="<<opt.coarse_parallel<<" schedule="<<opt.omp_schedule<<" ...\n";
            std::vector<EvalFull> fullgrid = evaluate_full_grid_parallel(opt, nnP, spec.irrep, par);
            std::vector<EvalData> grid = extract_eval_data_grid(fullgrid);
            const std::string base=opt.outdir+"/"+opt.prefix+"_"+lab;
            write_grid_file(base+"_grid.dat",grid,lab,spec,input_file);
            int ok=0; std::map<std::string,int> fail_counts; for(const auto& r:grid){ if(r.success) ++ok; else fail_counts[sanitize_error(r.error)]++; } meta<<"label "<<lab<<" success_points "<<ok<<" / "<<opt.N<<"\n"; for(const auto& kv:fail_counts) meta<<"label "<<lab<<" failure_count "<<kv.first<<" "<<kv.second<<"\n";
            const std::string reflog_base=opt.outdir+"/refine_logs/"+opt.prefix+"_"+lab;
            auto cand0=find_candidate_records_for_method(0,grid,nnP,spec.irrep,par,opt.debug,opt,"real",reflog_base);
            auto cand1=find_candidate_records_for_det_method(1,fullgrid,nnP,spec.irrep,par,opt.debug,opt,"real",reflog_base);
            auto cand2=find_candidate_records_for_det_method(2,fullgrid,nnP,spec.irrep,par,opt.debug,opt,"real",reflog_base);
            cand1=dedup_eigenbranch_candidate_records(std::move(cand1),opt);
            cand2=dedup_eigenbranch_candidate_records(std::move(cand2),opt);
            auto z0=unique_sorted_zeros(cand0,opt);
            auto z1=unique_sorted_zeros(cand1,opt);
            auto z2=unique_sorted_zeros(cand2,opt);
            auto cls=classify_zeros(z0,z1,z2,opt.duplicate_tol);
            if(opt.write_candidate_logs){
                std::ofstream cfile(base+"_all_candidates.dat"); cfile<<std::setprecision(17);
                cfile << "# v29p accepted and rejected candidate records after eigenbranch duplicate merging\n";
                cfile << "# columns: label nPx nPy nPz irrep irrep_tag method candidate_index Ecm E_left E_right y_real_or_branch_real abs_y kind accepted reason classifier crossing_index left_score right_score branch_index branch_abs merged_count\n";
                append_zero_records(cfile,spec,cand0); append_zero_records(cfile,spec,cand1); append_zero_records(cfile,spec,cand2);
            }
            append_zero_records(allz,spec,z0); append_zero_records(allz,spec,z1); append_zero_records(allz,spec,z2);
            std::ofstream zfile(base+"_zeros.dat"); zfile<<std::setprecision(17);
            zfile << "# columns: label nPx nPy nPz irrep irrep_tag method zero_index Ecm E_left E_right y_real_or_branch_real abs_y kind accepted reason classifier crossing_index left_score right_score branch_index branch_abs merged_count\n";
            append_zero_records(zfile,spec,z0); append_zero_records(zfile,spec,z1); append_zero_records(zfile,spec,z2);
            write_zero_classification(base+"_zero_classification.dat",spec,cls,opt.duplicate_tol);
            write_match_summary(base+"_match_summary.dat",spec,z0,z1,z2,opt.duplicate_tol,cls);
            if(opt.dump_matrices){
                std::vector<double> dumpEs; for(const auto& v:{z0,z1,z2}) for(const auto& z:v) dumpEs.push_back(z.E);
                if(opt.matrix_dump_stride>0) for(int i=0;i<opt.N;i+=opt.matrix_dump_stride) dumpEs.push_back(grid[i].Ecm);
                std::sort(dumpEs.begin(),dumpEs.end()); dumpEs.erase(std::unique(dumpEs.begin(),dumpEs.end(),[&](double a,double b){return std::abs(a-b)<opt.duplicate_tol;}),dumpEs.end());
                if((int)dumpEs.size()>opt.max_matrix_dumps_per_irrep) dumpEs.resize(opt.max_matrix_dumps_per_irrep);
                int ndump=0;
                if(opt.parallel_matrix_dumps){
                    #pragma omp parallel for schedule(dynamic,1)
                    for(int id=0; id<(int)dumpEs.size(); ++id){
                        const double E=dumpEs[id];
                        EvalFull ef=evaluate_full(E,nnP,spec.irrep,par,opt.debug);
                        if(!ef.d.success) continue;
                        std::ostringstream tag;
                        tag<<opt.outdir<<"/matrices/"<<opt.prefix<<"_"<<lab<<"_dump"<<std::setw(3)<<std::setfill('0')<<id<<std::setfill(' ')<<"_E"<<std::fixed<<std::setprecision(9)<<E;
                        dump_matrix(tag.str()+"_F3.dat",ef.F3);
                        dump_matrix(tag.str()+"_F3inv.dat",ef.F3inv);
                        dump_matrix(tag.str()+"_projF3.dat",ef.projF3);
                        dump_matrix(tag.str()+"_projF3inv.dat",ef.projF3inv);
                        #pragma omp atomic
                        ndump++;
                    }
                } else {
                    for(int id=0; id<(int)dumpEs.size(); ++id){
                        const double E=dumpEs[id];
                        EvalFull ef=evaluate_full(E,nnP,spec.irrep,par,opt.debug);
                        if(!ef.d.success) continue;
                        std::ostringstream tag;
                        tag<<opt.outdir<<"/matrices/"<<opt.prefix<<"_"<<lab<<"_dump"<<std::setw(3)<<std::setfill('0')<<id<<std::setfill(' ')<<"_E"<<std::fixed<<std::setprecision(9)<<E;
                        dump_matrix(tag.str()+"_F3.dat",ef.F3);
                        dump_matrix(tag.str()+"_F3inv.dat",ef.F3inv);
                        dump_matrix(tag.str()+"_projF3.dat",ef.projF3);
                        dump_matrix(tag.str()+"_projF3inv.dat",ef.projF3inv);
                        ++ndump;
                    }
                }
                meta<<"label "<<lab<<" matrix_dumps "<<ndump<<"\n";
            }
            std::cout<<"[v29p-eigenbranch] "<<lab<<" zeros: F3iso_inv="<<z0.size()<<" detF3inv="<<z1.size()<<" detProjF3inv="<<z2.size()<<"\n";
        }
        std::cout<<"[v29p-eigenbranch] wrote "<<opt.outdir<<"\n";
        std::cout<<"[v29p-eigenbranch] all zeros: "<<allzero_file<<"\n";
    } catch(const std::exception& e){ std::cerr<<"[v29p-eigenbranch:error] "<<e.what()<<"\n"; return 2; }
    return 0;
}
