#include "F3_cpu_openmp_v25_K3QC_cached_core.hpp"
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

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
    throw std::runtime_error("Unsupported MOM_LABEL: " + label + " . Supported: 000_A1m 100_A2 110_A2 111_A2 200_A2");
}

static double parse_double(const std::string& v, const std::string& name) {
    try { size_t p=0; double x=std::stod(v,&p); if(p!=v.size()) throw std::runtime_error("bad"); return x; }
    catch(...) { throw std::runtime_error("Could not parse double for " + name + ": " + v); }
}
static int parse_int(const std::string& v, const std::string& name) {
    try { size_t p=0; int x=std::stoi(v,&p); if(p!=v.size()) throw std::runtime_error("bad"); return x; }
    catch(...) { throw std::runtime_error("Could not parse int for " + name + ": " + v); }
}

struct Options {
    double Lval=20.0, xival=3.444, E0=0.2631, E1=0.36;
    int N=1601, threads=18;
    char debug='y';
    double atmpi=0.06906, atmK=0.09698;
    int parity=-1;
    double eig_tol=0.05;
    double K3iso0=0.0;
    double K3iso1=0.0;
    double K3B=0.0;
    double K3E=0.0;
    std::string outdir="output";
    std::string prefix="debug_v29f";
    int hermitize=1;
    double min_overlap_warn=0.60;
    double duplicate_tol=1.0e-3;
    double final_singular_abs_tol=1.0e4;
    double final_eigen_abs_tol=1.0e4;
    double final_max_cluster_width=2.0e-3;
    int final_require_filters=1;
    std::vector<std::string> labels{"111_A2"};
};

struct QCData {
    double Ecm = std::numeric_limits<double>::quiet_NaN();
    int total_dim = 0;
    int vdim = 0;
    comp det = comp(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
    double min_singular = std::numeric_limits<double>::quiet_NaN();
    double max_singular = std::numeric_limits<double>::quiet_NaN();
    double cond = std::numeric_limits<double>::quiet_NaN();
    double hermiticity_rel = std::numeric_limits<double>::quiet_NaN();
    Eigen::VectorXd evals;
    Eigen::MatrixXcd evecs;
    bool success = false;
    std::string error;
};

static void usage(const char* p) {
    std::cerr << "Usage:\n  " << p << " [options] MOM_LABEL [MOM_LABEL ...]\n\n"
              << "Options:\n"
              << "  --Lval X --xival X --E0 X --E1 X --N N\n"
              << "  --threads N --debug y|n     default debug=y in v29f\n"
              << "  --K3iso0 X --K3iso1 X --K3B X --K3E X    default all zero in v29f\n"
              << "  --outdir DIR --prefix NAME\n"
              << "  --hermitize 0|1             default 1; track eigenvalues of 0.5*(QC+QC^dagger)\n"
              << "  --threads N                 OpenMP threads for parallel energy-grid QC builds\n"
              << "  --min-overlap-warn X        default 0.60; low-overlap branch matches are flagged\n"
              << "  --duplicate-tol X           default 1e-3; clusters nearby eigen-branch zeros\n"
              << "  --final-singular-abs-tol X  default 1e4; final accepted clusters need min singular below this\n"
              << "  --final-eigen-abs-tol X     default 1e4; final accepted clusters need endpoint |lambda| below this\n"
              << "  --final-max-cluster-width X default 2e-3; reject broad clusters\n"
              << "  --final-require-filters 0|1 default 1; set 0 to accept all clusters\n"
              << "Default MOM_LABEL: 111_A2\n"
              << "Supported MOM_LABEL: 000_A1m 100_A2 110_A2 111_A2 200_A2\n";
}

static PhysicsParams make_params(const Options& o) {
    PhysicsParams par;
    par.atmpi=o.atmpi; par.atmK=o.atmK; par.xi=o.xival; par.Lbyas=o.Lval;
    par.parity=o.parity; par.eig_tol=o.eig_tol; par.omp_threads=o.threads;
    par.K3iso = {comp(o.K3iso0,0.0), comp(o.K3iso1,0.0)};
    par.K3B_par = comp(o.K3B,0.0);
    par.K3E_par = comp(o.K3E,0.0);
    return par;
}

static bool k3_parameters_are_zero(const PhysicsParams& par) {
    // Exact zero test is intentional here: these parameters come directly from parsed input.
    // If all four are zero, we skip K3mat_2plus1 completely and use QC = F3^{-1}_projected.
    return std::abs(par.K3iso[0]) == 0.0
        && std::abs(par.K3iso[1]) == 0.0
        && std::abs(par.K3B_par) == 0.0
        && std::abs(par.K3E_par) == 0.0;
}

static QCData evaluate_qc_eigen(
    double Ecm,
    const std::vector<int>& nnP_vec,
    const std::string& irrep,
    const PhysicsParams& par,
    F3iProjectedCache* cache,
    char debug,
    bool hermitize)
{
    QCData d; d.Ecm = Ecm;
    try {
        const std::string key = cache_key_v25(nnP_vec, irrep, Ecm);
        F3iProjectedCacheEntry entry;
        bool have_entry = false;
        if (cache != nullptr) {
            auto it = cache->table.find(key);
            if (it != cache->table.end()) { entry = it->second; have_entry = true; cache->hits += 1; }
        }
        if (!have_entry) {
            entry = build_F3i_projected_cache_entry_v25(Ecm, nnP_vec, irrep, par);
            if (cache != nullptr) { cache->table[key] = entry; cache->misses += 1; }
        }
        d.total_dim = entry.total_dim;
        d.vdim = entry.vdim;
        if (!entry.success) { d.success=false; d.error=entry.error; return d; }
        if (entry.vdim <= 0) { d.success=false; d.error="vdim<=0"; return d; }

        Eigen::MatrixXcd QC;
        if (k3_parameters_are_zero(par)) {
            // Important v29b/v29f change:
            // For K3iso0 = K3iso1 = K3B = K3E = 0, do NOT construct the dense
            // K3mat_2plus1 matrix. This avoids unnecessary work and avoids any
            // accidental numerical/noise contribution from a nominally zero K3.
            QC = entry.F3inv_projected;
        } else {
            Eigen::MatrixXcd K3(entry.total_dim, entry.total_dim);
            k3_2plus1::K3mat_2plus1(
                K3, comp(entry.En, 0.0), entry.plm_config, entry.klm_config, entry.total_P,
                par.atmK, par.atmpi, par.K3iso, par.K3B_par, par.K3E_par, debug);
            Eigen::MatrixXcd K3_projected = entry.Vsel.adjoint() * K3 * entry.Vsel;
            QC = entry.F3inv_projected + K3_projected;
        }

        d.det = determinant_via_partial_piv_lu(QC);

        const double qn = QC.norm();
        d.hermiticity_rel = (qn > 0.0) ? (QC - QC.adjoint()).norm()/qn : 0.0;

        Eigen::JacobiSVD<Eigen::MatrixXcd> svd(QC, Eigen::ComputeThinU | Eigen::ComputeThinV);
        if (svd.singularValues().size() > 0) {
            d.min_singular = svd.singularValues().minCoeff();
            d.max_singular = svd.singularValues().maxCoeff();
            if (d.min_singular > 0.0) d.cond = d.max_singular / d.min_singular;
        }

        Eigen::MatrixXcd QCeig = hermitize ? 0.5*(QC + QC.adjoint()) : QC;
        if (hermitize) {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(QCeig);
            if (es.info() != Eigen::Success) { d.success=false; d.error="SelfAdjointEigenSolver failed"; return d; }
            d.evals = es.eigenvalues();
            d.evecs = es.eigenvectors();
        } else {
            Eigen::ComplexEigenSolver<Eigen::MatrixXcd> ces(QCeig, true);
            if (ces.info() != Eigen::Success) { d.success=false; d.error="ComplexEigenSolver failed"; return d; }
            // For non-Hermitian matrices, use real parts only for branch plots/signs.
            // The imaginary parts are not tracked here. Prefer --hermitize 1 for QC level counting.
            d.evals.resize(ces.eigenvalues().size());
            for (int i=0;i<ces.eigenvalues().size();++i) d.evals[i] = ces.eigenvalues()[i].real();
            d.evecs = ces.eigenvectors();
        }
        d.success = true;
    } catch(const std::exception& e) {
        d.success=false; d.error=e.what();
    }
    return d;
}

static double linroot(double E0, double y0, double E1, double y1) {
    if (!std::isfinite(y0) || !std::isfinite(y1) || y1==y0) return 0.5*(E0+E1);
    return E0 - y0*(E1-E0)/(y1-y0);
}

static std::vector<int> greedy_assignment(const Eigen::MatrixXd& O) {
    const int n = (int)O.rows();
    std::vector<int> assign(n, -1);
    std::vector<char> used(n, 0);
    // Process old branches by descending best possible overlap for stability.
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b){ return O.row(a).maxCoeff() > O.row(b).maxCoeff(); });
    for (int ii : order) {
        double best=-1.0; int bestj=-1;
        for (int j=0;j<n;++j) if(!used[j] && O(ii,j)>best) { best=O(ii,j); bestj=j; }
        if (bestj>=0) { assign[ii]=bestj; used[bestj]=1; }
    }
    return assign;
}

// Exact maximum-overlap assignment by DP over bitmasks for n<=20; greedy fallback above that.
static std::vector<int> max_overlap_assignment(const Eigen::MatrixXd& O) {
    const int n = (int)O.rows();
    if (n != (int)O.cols()) throw std::runtime_error("assignment matrix must be square");
    if (n <= 0) return {};
    if (n > 20) return greedy_assignment(O);
    const int states = 1 << n;
    std::vector<double> dp(states, -1.0e300);
    std::vector<int> parent_j(states, -1), parent_prev(states, -1);
    dp[0] = 0.0;
    for (int mask=0; mask<states; ++mask) {
        int i = __builtin_popcount((unsigned)mask);
        if (i >= n || dp[mask] < -1.0e200) continue;
        for (int j=0;j<n;++j) if(!(mask & (1<<j))) {
            int nm = mask | (1<<j);
            double val = dp[mask] + O(i,j);
            if (val > dp[nm]) { dp[nm]=val; parent_j[nm]=j; parent_prev[nm]=mask; }
        }
    }
    std::vector<int> assign(n, -1);
    int mask = states-1;
    for (int i=n-1;i>=0;--i) {
        int j = parent_j[mask];
        assign[i] = j;
        mask = parent_prev[mask];
    }
    return assign;
}

struct BranchState {
    int branch_id = -1;
    double prev_E = std::numeric_limits<double>::quiet_NaN();
    double prev_lambda = std::numeric_limits<double>::quiet_NaN();
    Eigen::VectorXcd prev_vec;
};

struct ZeroEvent {
    std::string label;
    int nPx=0,nPy=0,nPz=0;
    std::string irrep, irrep_tag;
    int segment=-1;
    int branch_id=-1;
    double E_left=std::numeric_limits<double>::quiet_NaN();
    double E_right=std::numeric_limits<double>::quiet_NaN();
    double lambda_left=std::numeric_limits<double>::quiet_NaN();
    double lambda_right=std::numeric_limits<double>::quiet_NaN();
    double E_linear=std::numeric_limits<double>::quiet_NaN();
    double overlap=std::numeric_limits<double>::quiet_NaN();
    double min_singular_left=std::numeric_limits<double>::quiet_NaN();
    double min_singular_right=std::numeric_limits<double>::quiet_NaN();
};

struct ClusterSummary {
    std::string label;
    int nPx=0,nPy=0,nPz=0;
    std::string irrep, irrep_tag;
    int cluster_index=-1;
    double E_min=std::numeric_limits<double>::quiet_NaN();
    double E_max=std::numeric_limits<double>::quiet_NaN();
    double E_center=std::numeric_limits<double>::quiet_NaN();
    int members=0;
    std::string branch_ids;
    double cluster_width=std::numeric_limits<double>::quiet_NaN();
    double min_singular_cluster=std::numeric_limits<double>::quiet_NaN();
    double min_endpoint_abs_lambda=std::numeric_limits<double>::quiet_NaN();
    double max_overlap=std::numeric_limits<double>::quiet_NaN();
    int accepted=0;
    std::string reason;
};

static void write_headers(
    const Options& o,
    std::ofstream& fmeta,
    std::ofstream& ftrack,
    std::ofstream& fdet,
    std::ofstream& fdim,
    std::ofstream& fseg,
    std::ofstream& foverlap,
    std::ofstream& fzero,
    std::ofstream& fcluster,
    std::ofstream& ffinal) {
    auto common = [&](std::ofstream& f, const std::string& name){
        f << std::setprecision(17);
        f << "# " << name << "\n";
        f << "# Lval = " << o.Lval << "\n";
        f << "# xival = " << o.xival << "\n";
        f << "# L = " << o.Lval*o.xival << "\n";
        f << "# E0 = " << o.E0 << "\n";
        f << "# E1 = " << o.E1 << "\n";
        f << "# N = " << o.N << "\n";
        f << "# hermitize = " << o.hermitize << "\n";
        f << "# min_overlap_warn = " << o.min_overlap_warn << "\n";
        f << "# duplicate_tol = " << o.duplicate_tol << "\n";
        f << "# final_singular_abs_tol = " << o.final_singular_abs_tol << "\n";
        f << "# final_eigen_abs_tol = " << o.final_eigen_abs_tol << "\n";
        f << "# final_max_cluster_width = " << o.final_max_cluster_width << "\n";
        f << "# final_require_filters = " << o.final_require_filters << "\n";
        f << "# K3iso0 = " << o.K3iso0 << "\n";
        f << "# K3iso1 = " << o.K3iso1 << "\n";
        f << "# K3B = " << o.K3B << "\n";
        f << "# K3E = " << o.K3E << "\n";
        const bool k3_zero = (o.K3iso0 == 0.0 && o.K3iso1 == 0.0 && o.K3B == 0.0 && o.K3E == 0.0);
        f << "# k3_zero_skip_mode = " << (k3_zero ? 1 : 0) << "\n";
        f << "# QC_definition = " << (k3_zero ? "F3inv_projected" : "F3inv_projected + Vdagger_K3_V") << "\n";
    };
    common(fmeta, "v29f metadata: OpenMP eigensystem gather, serial max-overlap branch tracking, raw crossings -> clusters -> final accepted zeros; K3df defaults to zero, debug=y default");
    common(ftrack, "tracked eigenvalue branches using maximum eigenvector-overlap assignment; K3 construction skipped when all K3 parameters are zero");
    common(fdet, "det(projected QC) and singular diagnostics on the same energy grid");
    common(fdim, "dimension changes on the eigen-tracking grid");
    common(fseg, "constant-dimension segments used for serial eigenvector-overlap branch tracking");
    common(foverlap, "maximum-overlap assignment diagnostics between neighboring energy points");
    common(fzero, "eigenvalue-branch zero crossings");
    common(fcluster, "clustered eigenvalue-branch zero crossings with final-filter diagnostics");
    common(ffinal, "final accepted eigenvalue-branch zero clusters after min_singular/eigenvalue-scale filters");
    ftrack << "# columns: label nPx nPy nPz irrep irrep_tag grid_i Ecm segment branch_id local_eigen_index lambda overlap_from_prev low_overlap total_dim vdim det_real det_imag det_abs min_singular cond hermiticity_rel success\n";
    fdet << "# columns: label nPx nPy nPz irrep irrep_tag grid_i Ecm segment det_real det_imag det_abs total_dim vdim min_singular max_singular cond hermiticity_rel success error\n";
    fdim << "# columns: label nPx nPy nPz irrep irrep_tag jump_index E_left E_right E_mid total_dim_left total_dim_right vdim_left vdim_right\n";
    fseg << "# columns: label nPx nPy nPz irrep irrep_tag segment start_i end_i E_start E_end total_dim vdim success_points\n";
    foverlap << "# columns: label nPx nPy nPz irrep irrep_tag grid_i_prev grid_i_next E_prev E_next segment branch_id old_local_index new_local_index overlap low_overlap lambda_prev lambda_next\n";
    fzero << "# columns: label nPx nPy nPz irrep irrep_tag zero_index segment branch_id E_left E_right lambda_left lambda_right E_linear overlap min_singular_left min_singular_right\n";
    fcluster << "# columns: label nPx nPy nPz irrep irrep_tag cluster_index E_min E_max E_center members branch_ids cluster_width min_singular_cluster min_endpoint_abs_lambda max_overlap accepted reason\n";
    ffinal << "# columns: label nPx nPy nPz irrep irrep_tag final_zero_index E_center E_min E_max members branch_ids min_singular_cluster min_endpoint_abs_lambda cluster_width max_overlap reason\n";
}

static std::vector<ClusterSummary> cluster_zero_events(const std::vector<ZeroEvent>& zs, double tol) {
    std::vector<ClusterSummary> out;
    if (zs.empty()) return out;
    std::vector<ZeroEvent> v=zs;
    std::sort(v.begin(), v.end(), [](const ZeroEvent& a, const ZeroEvent& b){ return a.E_linear < b.E_linear; });
    int ci=0;
    size_t i=0;
    while(i<v.size()) {
        size_t j=i+1;
        double emin=v[i].E_linear, emax=v[i].E_linear;
        while(j<v.size() && std::abs(v[j].E_linear - emax) <= tol) {
            emin = std::min(emin, v[j].E_linear);
            emax = std::max(emax, v[j].E_linear);
            ++j;
        }
        const auto& z0=v[i];
        ClusterSummary c;
        c.label=z0.label; c.nPx=z0.nPx; c.nPy=z0.nPy; c.nPz=z0.nPz; c.irrep=z0.irrep; c.irrep_tag=z0.irrep_tag;
        c.cluster_index=ci++;
        c.E_min=emin; c.E_max=emax; c.cluster_width=emax-emin; c.members=(int)(j-i);
        c.E_center=0.0;
        c.min_singular_cluster=std::numeric_limits<double>::infinity();
        c.min_endpoint_abs_lambda=std::numeric_limits<double>::infinity();
        c.max_overlap=0.0;
        std::ostringstream branches;
        for(size_t k=i;k<j;++k) {
            const auto& z=v[k];
            c.E_center += z.E_linear;
            if (k>i) branches << ",";
            branches << z.branch_id;
            c.min_singular_cluster = std::min(c.min_singular_cluster, z.min_singular_left);
            c.min_singular_cluster = std::min(c.min_singular_cluster, z.min_singular_right);
            c.min_endpoint_abs_lambda = std::min(c.min_endpoint_abs_lambda, std::abs(z.lambda_left));
            c.min_endpoint_abs_lambda = std::min(c.min_endpoint_abs_lambda, std::abs(z.lambda_right));
            c.max_overlap = std::max(c.max_overlap, z.overlap);
        }
        c.E_center /= double(j-i);
        c.branch_ids = branches.str();
        if (!std::isfinite(c.min_singular_cluster)) c.min_singular_cluster = std::numeric_limits<double>::quiet_NaN();
        if (!std::isfinite(c.min_endpoint_abs_lambda)) c.min_endpoint_abs_lambda = std::numeric_limits<double>::quiet_NaN();
        out.push_back(c);
        i=j;
    }
    return out;
}

static void apply_final_filters(std::vector<ClusterSummary>& clusters, const Options& o) {
    for (auto& c : clusters) {
        const bool pass_singular = std::isfinite(c.min_singular_cluster) && (c.min_singular_cluster <= o.final_singular_abs_tol);
        const bool pass_eigen = std::isfinite(c.min_endpoint_abs_lambda) && (c.min_endpoint_abs_lambda <= o.final_eigen_abs_tol);
        const bool pass_width = std::isfinite(c.cluster_width) && (c.cluster_width <= o.final_max_cluster_width);
        if (!o.final_require_filters) {
            c.accepted = 1;
            c.reason = "accepted_filters_disabled";
        } else if (pass_singular && pass_eigen && pass_width) {
            c.accepted = 1;
            c.reason = "accepted:min_singular_and_endpoint_lambda_and_width";
        } else {
            c.accepted = 0;
            std::ostringstream r;
            r << "rejected:";
            if (!pass_singular) r << "min_singular(" << c.min_singular_cluster << ">" << o.final_singular_abs_tol << ");";
            if (!pass_eigen) r << "endpoint_lambda(" << c.min_endpoint_abs_lambda << ">" << o.final_eigen_abs_tol << ");";
            if (!pass_width) r << "width(" << c.cluster_width << ">" << o.final_max_cluster_width << ");";
            c.reason = r.str();
        }
    }
}

static void write_cluster_and_final_files(const std::vector<ClusterSummary>& clusters, std::ofstream& fcluster, std::ofstream& ffinal) {
    int fi=0;
    for (const auto& c : clusters) {
        fcluster << c.label << ' ' << c.nPx << ' ' << c.nPy << ' ' << c.nPz << ' ' << c.irrep << ' ' << c.irrep_tag << ' '
                 << c.cluster_index << ' ' << c.E_min << ' ' << c.E_max << ' ' << c.E_center << ' ' << c.members << ' ' << c.branch_ids << ' '
                 << c.cluster_width << ' ' << c.min_singular_cluster << ' ' << c.min_endpoint_abs_lambda << ' ' << c.max_overlap << ' '
                 << c.accepted << ' ' << c.reason << '\n';
        if (c.accepted) {
            ffinal << c.label << ' ' << c.nPx << ' ' << c.nPy << ' ' << c.nPz << ' ' << c.irrep << ' ' << c.irrep_tag << ' '
                   << fi++ << ' ' << c.E_center << ' ' << c.E_min << ' ' << c.E_max << ' ' << c.members << ' ' << c.branch_ids << ' '
                   << c.min_singular_cluster << ' ' << c.min_endpoint_abs_lambda << ' ' << c.cluster_width << ' ' << c.max_overlap << ' '
                   << c.reason << '\n';
        }
    }
}


int main(int argc, char** argv) {
    try {
        std::cout << std::setprecision(17);
        Options o;
        std::vector<std::string> labels;
        for(int i=1;i<argc;++i) {
            std::string a=argv[i];
            auto need=[&](const std::string& opt){ if(i+1>=argc) throw std::runtime_error("Missing value after "+opt); return std::string(argv[++i]); };
            if(a=="--help" || a=="-h") { usage(argv[0]); return 0; }
            else if(a=="--Lval") o.Lval=parse_double(need(a),a);
            else if(a=="--xival") o.xival=parse_double(need(a),a);
            else if(a=="--E0") o.E0=parse_double(need(a),a);
            else if(a=="--E1") o.E1=parse_double(need(a),a);
            else if(a=="--N") o.N=parse_int(need(a),a);
            else if(a=="--threads") o.threads=parse_int(need(a),a);
            else if(a=="--debug") { std::string v=need(a); o.debug=(v.empty()?'n':v[0]); }
            else if(a=="--K3iso0") o.K3iso0=parse_double(need(a),a);
            else if(a=="--K3iso1") o.K3iso1=parse_double(need(a),a);
            else if(a=="--K3B") o.K3B=parse_double(need(a),a);
            else if(a=="--K3E") o.K3E=parse_double(need(a),a);
            else if(a=="--atmpi") o.atmpi=parse_double(need(a),a);
            else if(a=="--atmK") o.atmK=parse_double(need(a),a);
            else if(a=="--parity") o.parity=parse_int(need(a),a);
            else if(a=="--eig_tol") o.eig_tol=parse_double(need(a),a);
            else if(a=="--outdir") o.outdir=need(a);
            else if(a=="--prefix") o.prefix=need(a);
            else if(a=="--hermitize") o.hermitize=parse_int(need(a),a);
            else if(a=="--min-overlap-warn") o.min_overlap_warn=parse_double(need(a),a);
            else if(a=="--duplicate-tol") o.duplicate_tol=parse_double(need(a),a);
            else if(a=="--final-singular-abs-tol") o.final_singular_abs_tol=parse_double(need(a),a);
            else if(a=="--final-eigen-abs-tol") o.final_eigen_abs_tol=parse_double(need(a),a);
            else if(a=="--final-max-cluster-width") o.final_max_cluster_width=parse_double(need(a),a);
            else if(a=="--final-require-filters") o.final_require_filters=parse_int(need(a),a);
            else if(!a.empty() && a[0]=='-') throw std::runtime_error("Unknown option: "+a);
            else labels.push_back(a);
        }
        if(!labels.empty()) o.labels=labels;
        if(!(o.E1>o.E0)) throw std::runtime_error("Need E1 > E0");
        if(o.N <= 1) throw std::runtime_error("N must be > 1");
        std::filesystem::create_directories(o.outdir);

        const std::string meta_file = o.outdir + "/" + o.prefix + "_metadata.txt";
        const std::string track_file = o.outdir + "/" + o.prefix + "_tracked_eigenvalues.dat";
        const std::string det_file = o.outdir + "/" + o.prefix + "_detprojQC_grid.dat";
        const std::string dim_file = o.outdir + "/" + o.prefix + "_dimension_jumps.dat";
        const std::string segment_file = o.outdir + "/" + o.prefix + "_dimension_segments.dat";
        const std::string overlap_file = o.outdir + "/" + o.prefix + "_overlap_quality.dat";
        const std::string zero_file = o.outdir + "/" + o.prefix + "_eigenbranch_zeros.dat";
        const std::string cluster_file = o.outdir + "/" + o.prefix + "_eigenbranch_zero_clusters.dat";
        const std::string final_file = o.outdir + "/" + o.prefix + "_final_accepted_zeros.dat";
        std::ofstream fmeta(meta_file), ftrack(track_file), fdet(det_file), fdim(dim_file), fseg(segment_file), foverlap(overlap_file), fzero(zero_file), fcluster(cluster_file), ffinal(final_file);
        if(!fmeta||!ftrack||!fdet||!fdim||!fseg||!foverlap||!fzero||!fcluster||!ffinal) throw std::runtime_error("Could not open one or more output files");
        write_headers(o, fmeta, ftrack, fdet, fdim, fseg, foverlap, fzero, fcluster, ffinal);

        PhysicsParams par = make_params(o);
#ifdef _OPENMP
        omp_set_dynamic(0);
        omp_set_num_threads(o.threads);
#endif
        // Eigen may otherwise start its own internal thread team inside each OpenMP worker.
        // Keep Eigen single-threaded and parallelize at the outer energy-grid level.
        Eigen::setNbThreads(1);

        // v29f does NOT keep a shared F3iProjectedCache during the full energy scan.
        // The cache key contains Ecm, so for a dense scan almost every point is unique.
        // Storing all entries wastes RAM and a shared map would serialize access.
        F3iProjectedCache cache;
        int global_zero_index=0;

        for(const std::string& label: o.labels) {
            MomentumIrrepSpec spec=parse_label(label);
            std::vector<int> nnP_vec={spec.nnP[0],spec.nnP[1],spec.nnP[2]};
            std::cerr << "[v29f] eigen-branch tracking label=" << label << " E=[" << o.E0 << "," << o.E1 << "] N=" << o.N << " hermitize=" << o.hermitize << "\n";

            std::vector<QCData> grid((size_t)o.N);
            std::atomic<int> done_count{0};
            const int report_step = std::max(1, o.N / 10);

#ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic,1)
#endif
            for(int i=0;i<o.N;++i) {
                double E = o.E0 + (o.E1-o.E0)*double(i)/double(o.N-1);
                // Pass nullptr: every E is unique in the grid scan, so caching full projected matrices
                // across energies gives almost no hit rate and can consume a lot of memory.
                grid[(size_t)i] = evaluate_qc_eigen(E, nnP_vec, spec.irrep, par, nullptr, o.debug, o.hermitize!=0);

                int ndone = ++done_count;
                if (ndone == o.N || ndone % report_step == 0) {
#ifdef _OPENMP
                    #pragma omp critical(v29f_progress)
#endif
                    {
                        std::cerr << "[v29f] label=" << label
                                  << " energy-build progress " << ndone << "/" << o.N
#ifdef _OPENMP
                                  << " omp_threads=" << omp_get_max_threads()
#endif
                                  << "\n";
                    }
                }
            }

            for(int i=0;i<o.N;++i) {
                const auto& q=grid[(size_t)i];
                fdet << label << ' ' << spec.nnP[0] << ' ' << spec.nnP[1] << ' ' << spec.nnP[2] << ' ' << spec.irrep << ' ' << spec.irrep_tag << ' '
                     << i << ' ' << q.Ecm << ' ' << -1 << ' ' << q.det.real() << ' ' << q.det.imag() << ' ' << std::abs(q.det) << ' '
                     << q.total_dim << ' ' << q.vdim << ' ' << q.min_singular << ' ' << q.max_singular << ' ' << q.cond << ' ' << q.hermiticity_rel << ' ' << (q.success?1:0) << ' ' << (q.error.empty()?"ok":q.error) << '\n';
            }

            int jump_index=0;
            for(int i=1;i<o.N;++i) {
                const QCData& A=grid[(size_t)i-1]; const QCData& B=grid[(size_t)i];
                if(A.success && B.success && (A.total_dim!=B.total_dim || A.vdim!=B.vdim)) {
                    fdim << label << ' ' << spec.nnP[0] << ' ' << spec.nnP[1] << ' ' << spec.nnP[2] << ' ' << spec.irrep << ' ' << spec.irrep_tag << ' '
                         << jump_index++ << ' ' << A.Ecm << ' ' << B.Ecm << ' ' << 0.5*(A.Ecm+B.Ecm) << ' '
                         << A.total_dim << ' ' << B.total_dim << ' ' << A.vdim << ' ' << B.vdim << '\n';
                }
            }

            // Precompute and save constant-dimension segments. Branch tracking is performed
            // only inside these segments and is deliberately not continued across dimension jumps.
            int seg_write = 0;
            int seg_start = -1;
            for (int i=0; i<=o.N; ++i) {
                const bool at_end = (i == o.N);
                const bool good = (!at_end && grid[(size_t)i].success);
                bool break_seg = at_end || !good;
                if (!break_seg && seg_start >= 0) {
                    const QCData& A = grid[(size_t)i-1];
                    const QCData& B = grid[(size_t)i];
                    if (!A.success || A.total_dim != B.total_dim || A.vdim != B.vdim) break_seg = true;
                }
                if (seg_start < 0 && good) seg_start = i;
                if (break_seg && seg_start >= 0) {
                    int seg_end = i - 1;
                    const QCData& S = grid[(size_t)seg_start];
                    const QCData& T = grid[(size_t)seg_end];
                    fseg << label << ' ' << spec.nnP[0] << ' ' << spec.nnP[1] << ' ' << spec.nnP[2] << ' ' << spec.irrep << ' ' << spec.irrep_tag << ' '
                         << seg_write++ << ' ' << seg_start << ' ' << seg_end << ' ' << S.Ecm << ' ' << T.Ecm << ' '
                         << S.total_dim << ' ' << S.vdim << ' ' << (seg_end - seg_start + 1) << '\n';
                    seg_start = good ? i : -1;
                }
            }

            std::vector<BranchState> branches;
            int next_branch_id=0;
            int segment=-1;
            std::vector<ZeroEvent> zeros;

            QCData* prev = nullptr;
            int prev_i = -1;

            for(int i=0;i<o.N;++i) {
                QCData& cur = grid[(size_t)i];
                if(!cur.success) { prev=nullptr; branches.clear(); continue; }
                const bool new_segment = (prev==nullptr || !prev->success || prev->vdim != cur.vdim || prev->total_dim != cur.total_dim);
                if(new_segment) {
                    segment++;
                    branches.clear();
                    for(int k=0;k<cur.vdim;++k) {
                        BranchState bs;
                        bs.branch_id = next_branch_id++;
                        bs.prev_E = cur.Ecm;
                        bs.prev_lambda = cur.evals[k];
                        bs.prev_vec = cur.evecs.col(k);
                        branches.push_back(bs);
                        ftrack << label << ' ' << spec.nnP[0] << ' ' << spec.nnP[1] << ' ' << spec.nnP[2] << ' ' << spec.irrep << ' ' << spec.irrep_tag << ' '
                               << i << ' ' << cur.Ecm << ' ' << segment << ' ' << bs.branch_id << ' ' << k << ' ' << bs.prev_lambda << ' '
                               << 1.0 << ' ' << 0 << ' ' << cur.total_dim << ' ' << cur.vdim << ' '
                               << cur.det.real() << ' ' << cur.det.imag() << ' ' << std::abs(cur.det) << ' ' << cur.min_singular << ' ' << cur.cond << ' ' << cur.hermiticity_rel << ' ' << 1 << '\n';
                    }
                    prev = &cur; prev_i = i;
                    continue;
                }

                const int n = cur.vdim;
                Eigen::MatrixXd O(n,n);
                for(int bi=0;bi<n;++bi) {
                    for(int k=0;k<n;++k) {
                        comp ov = branches[(size_t)bi].prev_vec.adjoint() * cur.evecs.col(k);
                        O(bi,k) = std::abs(ov);
                    }
                }
                std::vector<int> assign = max_overlap_assignment(O);

                std::vector<BranchState> new_branches(n);
                for(int bi=0; bi<n; ++bi) {
                    int k = assign[(size_t)bi];
                    if(k<0) continue;
                    BranchState old = branches[(size_t)bi];
                    double lam_new = cur.evals[k];
                    double ov = O(bi,k);
                    bool low = ov < o.min_overlap_warn;

                    foverlap << label << ' ' << spec.nnP[0] << ' ' << spec.nnP[1] << ' ' << spec.nnP[2] << ' ' << spec.irrep << ' ' << spec.irrep_tag << ' '
                             << prev_i << ' ' << i << ' ' << old.prev_E << ' ' << cur.Ecm << ' ' << segment << ' '
                             << old.branch_id << ' ' << bi << ' ' << k << ' ' << ov << ' ' << (low?1:0) << ' '
                             << old.prev_lambda << ' ' << lam_new << '\n';

                    if(std::isfinite(old.prev_lambda) && std::isfinite(lam_new) && old.prev_lambda*lam_new < 0.0) {
                        ZeroEvent z;
                        z.label=label; z.nPx=spec.nnP[0]; z.nPy=spec.nnP[1]; z.nPz=spec.nnP[2]; z.irrep=spec.irrep; z.irrep_tag=spec.irrep_tag;
                        z.segment=segment; z.branch_id=old.branch_id; z.E_left=old.prev_E; z.E_right=cur.Ecm; z.lambda_left=old.prev_lambda; z.lambda_right=lam_new;
                        z.E_linear=linroot(old.prev_E, old.prev_lambda, cur.Ecm, lam_new);
                        z.overlap=ov;
                        if(prev!=nullptr) z.min_singular_left=prev->min_singular;
                        z.min_singular_right=cur.min_singular;
                        zeros.push_back(z);
                        fzero << label << ' ' << spec.nnP[0] << ' ' << spec.nnP[1] << ' ' << spec.nnP[2] << ' ' << spec.irrep << ' ' << spec.irrep_tag << ' '
                              << global_zero_index++ << ' ' << segment << ' ' << old.branch_id << ' '
                              << old.prev_E << ' ' << cur.Ecm << ' ' << old.prev_lambda << ' ' << lam_new << ' ' << z.E_linear << ' ' << ov << ' '
                              << z.min_singular_left << ' ' << z.min_singular_right << '\n';
                    }

                    BranchState nb;
                    nb.branch_id = old.branch_id;
                    nb.prev_E = cur.Ecm;
                    nb.prev_lambda = lam_new;
                    nb.prev_vec = cur.evecs.col(k);
                    new_branches[(size_t)k] = nb; // store by current local eigen index for next overlap matrix columns

                    ftrack << label << ' ' << spec.nnP[0] << ' ' << spec.nnP[1] << ' ' << spec.nnP[2] << ' ' << spec.irrep << ' ' << spec.irrep_tag << ' '
                           << i << ' ' << cur.Ecm << ' ' << segment << ' ' << old.branch_id << ' ' << k << ' ' << lam_new << ' '
                           << ov << ' ' << (low?1:0) << ' ' << cur.total_dim << ' ' << cur.vdim << ' '
                           << cur.det.real() << ' ' << cur.det.imag() << ' ' << std::abs(cur.det) << ' ' << cur.min_singular << ' ' << cur.cond << ' ' << cur.hermiticity_rel << ' ' << 1 << '\n';
                }
                // Reorder branch vector by local eigen index in current eigensystem.
                branches = std::move(new_branches);
                prev=&cur; prev_i=i;
            }
            auto clusters = cluster_zero_events(zeros, o.duplicate_tol);
            apply_final_filters(clusters, o);
            write_cluster_and_final_files(clusters, fcluster, ffinal);
            int accepted_count = 0;
            for (const auto& c : clusters) if (c.accepted) ++accepted_count;
            std::cerr << "[v29f] label=" << label << " eigenbranch_zeros=" << zeros.size()
                      << " clusters=" << clusters.size() << " final_accepted=" << accepted_count
                      << " dim_jumps=" << jump_index << " cache_size=" << cache.size()
                      << " hits=" << cache.hits << " misses=" << cache.misses << "\n";
        }
        std::cerr << "[done] wrote v29f outputs under " << o.outdir << "\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "[fatal] " << e.what() << "\n";
        return 1;
    }
}
