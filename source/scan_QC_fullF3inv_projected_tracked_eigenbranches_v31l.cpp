#include "K3df_minuit_fit_v31l_lattice_covariance.hpp"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

static std::map<std::string,std::string> read_simple_kv_tr(const std::string& path) {
    std::ifstream in(path);
    if(!in) throw std::runtime_error("Could not open config: " + path);
    std::map<std::string,std::string> kv;
    std::string line;
    while(std::getline(in,line)) {
        auto hash=line.find('#'); if(hash!=std::string::npos) line=line.substr(0,hash);
        auto eq=line.find('='); if(eq==std::string::npos) continue;
        auto trim=[](std::string s) {
            while(!s.empty() && std::isspace((unsigned char)s.front())) s.erase(s.begin());
            while(!s.empty() && std::isspace((unsigned char)s.back())) s.pop_back();
            return s;
        };
        kv[trim(line.substr(0,eq))]=trim(line.substr(eq+1));
    }
    return kv;
}
static std::string gs(const std::map<std::string,std::string>& kv,const std::string& k,const std::string& d){ auto it=kv.find(k); return it==kv.end()?d:it->second; }
static double gd(const std::map<std::string,std::string>& kv,const std::string& k,double d){ auto it=kv.find(k); return it==kv.end()?d:std::stod(it->second); }
static int gi(const std::map<std::string,std::string>& kv,const std::string& k,int d){ auto it=kv.find(k); return it==kv.end()?d:std::stoi(it->second); }
static std::vector<std::string> split_ws(std::string s) {
    for(char& c:s) if(c==',') c=' ';
    std::istringstream is(s);
    std::vector<std::string> v; std::string x;
    while(is>>x) v.push_back(x);
    return v;
}

static k3df_fit_v31l::FitSettings settings_from_config(const std::string& config) {
    using namespace k3df_fit_v31l;
    auto kv = read_simple_kv_tr(config);
    FitSettings s;
    s.list_of_mom = split_ws(gs(kv,"list_of_mom","000_A1m 100_A2 110_A2 111_A2 200_A2"));
    s.Lval = gd(kv,"Lval",s.Lval);
    s.xival = gd(kv,"xival",s.xival);
    s.scan_E0 = gd(kv,"scan_E0",s.scan_E0);
    s.scan_E1 = gd(kv,"scan_E1",s.scan_E1);
    s.coarseN = gi(kv,"coarseN",s.coarseN);
    s.omp_threads = gi(kv,"omp_threads",s.omp_threads);
    s.debug = gs(kv,"debug","n").empty() ? 'n' : gs(kv,"debug","n")[0];

    s.atmpi = gd(kv,"atmpi",s.atmpi);
    s.atmK = gd(kv,"atmK",s.atmK);
    s.eta_1 = gd(kv,"eta_1",s.eta_1);
    s.eta_2 = gd(kv,"eta_2",s.eta_2);
    s.alpha = gd(kv,"alpha",s.alpha);
    s.epsilon_h = gd(kv,"epsilon_h",s.epsilon_h);
    s.max_shell_num = gd(kv,"max_shell_num",s.max_shell_num);
    s.tolerance = gd(kv,"tolerance",s.tolerance);
    s.parity = gi(kv,"parity",s.parity);
    s.eig_tol = gd(kv,"eig_tol",s.eig_tol);
    s.norm_tol = gd(kv,"norm_tol",s.norm_tol);
    s.proj_tol = gd(kv,"proj_tol",s.proj_tol);

    s.output_dir = gs(kv,"output_dir",s.output_dir);
    s.output_tag = gs(kv,"output_tag",s.output_tag);
    s.write_cache_grid = gi(kv,"write_cache_grid",s.write_cache_grid?1:0)!=0;
    s.save_binary_f3inv_cache = gi(kv,"save_binary_f3inv_cache",s.save_binary_f3inv_cache?1:0)!=0;
    s.load_binary_f3inv_cache = gi(kv,"load_binary_f3inv_cache",s.load_binary_f3inv_cache?1:0)!=0;
    s.binary_f3inv_cache_file = gs(kv,"binary_f3inv_cache_file",s.binary_f3inv_cache_file);
    return s;
}


static Eigen::MatrixXcd assemble_QC_fullF3inv_projected_v31l(
        const k3df_fit_v31l::ProjectedQCCacheEntry& e,
        const k3df_fit_v31l::K3dfParameters& p,
        const PhysicsParams& par,
        char debug) {
    using namespace k3df_fit_v31l;
    if(!e.success) return Eigen::MatrixXcd();
    if(e.F3inv_full.rows()==0 || e.F3inv_full.rows()!=e.F3inv_full.cols()) return Eigen::MatrixXcd();

    const int N = int(e.plm_config[0].size() + e.klm_config[0].size());
    if(N <= 0 || e.F3inv_full.rows() != N || e.Vsel.rows() != N) return Eigen::MatrixXcd();

    Eigen::MatrixXcd K3_full(N,N);
    std::vector<comp> Kiso = {comp(p.K3iso0,0.0), comp(p.K3iso1,0.0)};
    k3_2plus1::K3mat_2plus1(
        K3_full,
        e.En_c,
        e.plm_config,
        e.klm_config,
        e.total_P,
        par.atmK,
        par.atmpi,
        Kiso,
        comp(p.K3B,0.0),
        comp(p.K3E,0.0),
        debug
    );

    Eigen::MatrixXcd QC_full = e.F3inv_full + K3_full;
    return e.Vsel.adjoint() * QC_full * e.Vsel;
}


struct EigDecompPoint {
    int i=-1;
    double Ecm=std::numeric_limits<double>::quiet_NaN();
    int success=0;
    int proj_dim=0;
    double herm_rel=std::numeric_limits<double>::quiet_NaN();
    Eigen::VectorXd evals;
    Eigen::MatrixXcd evecs; // columns are eigenvectors in algebraic-eigenvalue order
};

struct TrackedRow {
    int segment=-1;
    int branch=-1;
    int i=-1;
    double Ecm=std::numeric_limits<double>::quiet_NaN();
    int raw_eig_index=-1;
    double eig_value=std::numeric_limits<double>::quiet_NaN();
    double overlap_from_prev=std::numeric_limits<double>::quiet_NaN();
    int proj_dim=0;
};

static EigDecompPoint eval_qc_eigendecomp(
        const k3df_fit_v31l::ProjectedQCCacheEntry& e,
        const k3df_fit_v31l::K3dfParameters& p,
        const PhysicsParams& par,
        char debug) {
    using namespace k3df_fit_v31l;
    EigDecompPoint out;
    out.i = e.i;
    out.Ecm = e.Ecm;
    out.proj_dim = e.proj_dim;
    if(!e.success) return out;

    Eigen::MatrixXcd QC = assemble_QC_fullF3inv_projected_v31l(e,p,par,debug);
    if(QC.rows()==0 || QC.cols()==0 || QC.rows()!=QC.cols() || !QC.allFinite()) return out;

    const double nrm = QC.norm();
    out.herm_rel = (nrm > 0.0) ? (QC-QC.adjoint()).norm()/nrm : 0.0;

    Eigen::MatrixXcd H = 0.5*(QC + QC.adjoint());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H, Eigen::ComputeEigenvectors);
    if(es.info()!=Eigen::Success) return out;

    out.evals = es.eigenvalues();
    out.evecs = es.eigenvectors();
    out.success = 1;
    return out;
}

static std::vector<int> hungarian_minimize_square_v31l(const std::vector<std::vector<double>>& cost) {
    // Hungarian algorithm for rectangular form n x m with n<=m, here used as square dim x dim.
    // Returns assignment[row] = col minimizing total cost.
    const int n = int(cost.size());
    if(n == 0) return {};
    const int m = int(cost[0].size());
    const double INF = 1.0e100;

    std::vector<double> u(n+1,0.0), v(m+1,0.0);
    std::vector<int> p(m+1,0), way(m+1,0);

    for(int i=1; i<=n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(m+1, INF);
        std::vector<char> used(m+1, false);
        do {
            used[j0] = true;
            const int i0 = p[j0];
            double delta = INF;
            int j1 = 0;
            for(int j=1; j<=m; ++j) {
                if(used[j]) continue;
                const double cur = cost[i0-1][j-1] - u[i0] - v[j];
                if(cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if(minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for(int j=0; j<=m; ++j) {
                if(used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while(p[j0] != 0);

        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while(j0 != 0);
    }

    std::vector<int> assignment(n, -1);
    for(int j=1; j<=m; ++j) {
        if(p[j] != 0) assignment[p[j]-1] = j-1;
    }
    return assignment;
}

static std::vector<TrackedRow> track_by_eigenvector_overlap(
        const std::vector<EigDecompPoint>& pts,
        double min_overlap_warn) {
    // v31l: global/Hungarian assignment of eigenvectors between neighboring energies.
    // We maximize total overlap by minimizing cost = 1 - |v_prev^dagger v_curr|.
    std::vector<TrackedRow> rows;
    int segment = -1;
    int global_branch_offset = 0;

    Eigen::MatrixXcd prev_vecs;
    std::vector<int> prev_branch_for_col;
    int prev_dim = -1;
    bool have_prev = false;

    for(const auto& p: pts) {
        if(!p.success || p.evals.size()==0 || p.evecs.cols()==0) {
            have_prev = false;
            prev_dim = -1;
            prev_vecs.resize(0,0);
            prev_branch_for_col.clear();
            continue;
        }

        const int dim = int(p.evals.size());

        if(!have_prev || dim != prev_dim) {
            ++segment;
            if(!prev_branch_for_col.empty()) {
                int maxb = *std::max_element(prev_branch_for_col.begin(), prev_branch_for_col.end());
                global_branch_offset = std::max(global_branch_offset, maxb+1);
            }
            prev_branch_for_col.resize(dim);
            for(int k=0;k<dim;++k) prev_branch_for_col[k] = global_branch_offset + k;

            for(int k=0;k<dim;++k) {
                rows.push_back({segment, prev_branch_for_col[k], p.i, p.Ecm, k, p.evals[k],
                                std::numeric_limits<double>::quiet_NaN(), dim});
            }

            prev_vecs = p.evecs;
            prev_dim = dim;
            have_prev = true;
            continue;
        }

        std::vector<std::vector<double>> overlap(dim, std::vector<double>(dim, 0.0));
        std::vector<std::vector<double>> cost(dim, std::vector<double>(dim, 1.0));
        for(int a=0; a<dim; ++a) {
            for(int b=0; b<dim; ++b) {
                double ov = std::abs((prev_vecs.col(a).adjoint() * p.evecs.col(b))(0,0));
                if(!std::isfinite(ov)) ov = 0.0;
                if(ov < 0.0) ov = 0.0;
                if(ov > 1.0) ov = 1.0;
                overlap[a][b] = ov;
                cost[a][b] = 1.0 - ov;
            }
        }

        // assignment[prev_col] = curr_col, globally maximizing sum overlap.
        std::vector<int> assignment = hungarian_minimize_square_v31l(cost);

        std::vector<int> curr_branch_for_col(dim, -1);
        std::vector<double> curr_overlap_for_col(dim, std::numeric_limits<double>::quiet_NaN());

        double total_overlap = 0.0;
        for(int a=0; a<dim; ++a) {
            const int b = (a < int(assignment.size())) ? assignment[a] : -1;
            if(b < 0 || b >= dim) continue;
            curr_branch_for_col[b] = prev_branch_for_col[a];
            curr_overlap_for_col[b] = overlap[a][b];
            total_overlap += overlap[a][b];
        }

        // Fallback should not be needed for square Hungarian, but keep safety.
        int next_new_branch = global_branch_offset + dim;
        for(int b=0; b<dim; ++b) {
            if(curr_branch_for_col[b] < 0) {
                curr_branch_for_col[b] = next_new_branch++;
                curr_overlap_for_col[b] = 0.0;
            }
        }

        double min_assigned_overlap = 1.0;
        for(int b=0; b<dim; ++b) {
            if(std::isfinite(curr_overlap_for_col[b])) {
                min_assigned_overlap = std::min(min_assigned_overlap, curr_overlap_for_col[b]);
            }
        }

        if(min_assigned_overlap < min_overlap_warn) {
            std::cout << "[v31l-track-warning] low global-assignment overlap"
                      << " segment=" << segment
                      << " Ecm=" << p.Ecm
                      << " min_overlap=" << min_assigned_overlap
                      << " avg_overlap=" << (dim>0 ? total_overlap/dim : 0.0)
                      << "\n";
        }

        for(int b=0; b<dim; ++b) {
            rows.push_back({segment, curr_branch_for_col[b], p.i, p.Ecm, b, p.evals[b],
                            curr_overlap_for_col[b], dim});
        }

        prev_vecs = p.evecs;
        prev_branch_for_col = curr_branch_for_col;
        prev_dim = dim;
        have_prev = true;
    }

    return rows;
}

static void write_tracked_outputs_for_label(
        const k3df_fit_v31l::FitSettings& s,
        const k3df_fit_v31l::IrrepCache& ic,
        const k3df_fit_v31l::K3dfParameters& p,
        double min_overlap_warn) {
    using namespace k3df_fit_v31l;
    PhysicsParams par = make_base_physics(s);

    std::cout << "[v31l] label=" << ic.label << " computing eigenvalues/eigenvectors in parallel\n";
    std::vector<EigDecompPoint> pts(ic.grid.size());
    int done=0, nextpct=10;

    #pragma omp parallel for schedule(dynamic,1)
    for(int i=0; i<(int)ic.grid.size(); ++i) {
        pts[i] = eval_qc_eigendecomp(ic.grid[i],p,par,s.debug);
        #pragma omp critical
        {
            ++done;
            progress_percent_log("v31l-eigendecomp-"+ic.label, done, (int)ic.grid.size(), nextpct, 10);
        }
    }

    std::cout << "[v31l] label=" << ic.label << " tracking branches by global Hungarian eigenvector-overlap assignment after full-space F3 inverse projection\n";
    auto rows = track_by_eigenvector_overlap(pts, min_overlap_warn);

    std::filesystem::create_directories(s.output_dir);

    const std::string raw_path = s.output_dir + "/" + s.output_tag + "_" + clean_label(ic.label) + "_raw_eigenvalues_with_vectors_info.dat";
    std::ofstream rf(raw_path);
    rf << std::setprecision(17);
    rf << "# columns: i Ecm success proj_dim herm_QC_rel raw_eig_index eig_value\n";
    for(const auto& pt: pts) {
        if(!pt.success) {
            rf << pt.i << ' ' << pt.Ecm << " 0 " << pt.proj_dim << ' ' << pt.herm_rel << " -1 nan\n";
            continue;
        }
        for(int k=0;k<pt.evals.size();++k) {
            rf << pt.i << ' ' << pt.Ecm << " 1 " << pt.proj_dim << ' ' << pt.herm_rel
               << ' ' << k << ' ' << pt.evals[k] << "\n";
        }
    }
    std::cout << "[v31l-write] wrote " << raw_path << "\n";

    const std::string tr_path = s.output_dir + "/" + s.output_tag + "_" + clean_label(ic.label) + "_tracked_eigenbranches.dat";
    std::ofstream tf(tr_path);
    tf << std::setprecision(17);
    tf << "# K3iso0 " << p.K3iso0 << "\n";
    tf << "# K3iso1 " << p.K3iso1 << "\n";
    tf << "# K3B " << p.K3B << "\n";
    tf << "# K3E " << p.K3E << "\n";
    tf << "# columns: segment branch i Ecm raw_eig_index eig_value overlap_from_prev proj_dim\n";
    for(const auto& r: rows) {
        tf << r.segment << ' ' << r.branch << ' ' << r.i << ' ' << r.Ecm << ' '
           << r.raw_eig_index << ' ' << r.eig_value << ' ' << r.overlap_from_prev << ' '
           << r.proj_dim << "\n";
    }
    std::cout << "[v31l-write] wrote " << tr_path << "\n";

    // Also write branch-split files for convenience.
    std::map<int,std::vector<TrackedRow>> by_branch;
    for(const auto& r: rows) by_branch[r.branch].push_back(r);
    const std::string branch_dir = s.output_dir + "/" + s.output_tag + "_" + clean_label(ic.label) + "_branches";
    std::filesystem::create_directories(branch_dir);
    for(const auto& kv: by_branch) {
        const std::string bp = branch_dir + "/branch_" + std::to_string(kv.first) + ".dat";
        std::ofstream bf(bp);
        bf << std::setprecision(17);
        bf << "# columns: segment branch i Ecm raw_eig_index eig_value overlap_from_prev proj_dim\n";
        for(const auto& r: kv.second) {
            bf << r.segment << ' ' << r.branch << ' ' << r.i << ' ' << r.Ecm << ' '
               << r.raw_eig_index << ' ' << r.eig_value << ' ' << r.overlap_from_prev << ' '
               << r.proj_dim << "\n";
        }
    }
    std::cout << "[v31l-write] wrote branch files under " << branch_dir << "\n";
}

int main(int argc, char** argv) {
    if(argc != 2) {
        std::cerr << "Usage:\n  " << argv[0] << " config/config_v31l_QC_tracked_eigenbranches.in\n";
        return 1;
    }
    try {
        using namespace k3df_fit_v31l;
        FitSettings s = settings_from_config(argv[1]);
        auto kv = read_simple_kv_tr(argv[1]);

        K3dfParameters p;
        p.K3iso0 = gd(kv,"K3iso0",0.0);
        p.K3iso1 = gd(kv,"K3iso1",0.0);
        p.K3B    = gd(kv,"K3B",0.0);
        p.K3E    = gd(kv,"K3E",0.0);

        const double min_overlap_warn = gd(kv,"min_overlap_warn",0.50);

        #ifdef _OPENMP
        omp_set_num_threads(s.omp_threads);
        omp_set_max_active_levels(1);
        #endif
        Eigen::setNbThreads(1);

        std::cout << std::setprecision(17);
        std::cout << "[v31l] full-F3inv projected-QC tracked eigenbranch scan\n";
        std::cout << "[v31l] K3iso0=" << p.K3iso0
                  << " K3iso1=" << p.K3iso1
                  << " K3B=" << p.K3B
                  << " K3E=" << p.K3E << "\n";
        std::cout << "[v31l] list_of_mom:";
        for(const auto& x: s.list_of_mom) std::cout << " " << x;
        std::cout << "\n[v31l] coarseN=" << s.coarseN
                  << " E=[" << s.scan_E0 << "," << s.scan_E1 << "]"
                  << " threads=" << s.omp_threads << "\n";

        std::cout << "[v31l] [stage 1/3] build/load F3inv projected cache\n";
        auto caches = get_or_build_F3inv_cache_v31l(s);
        if(s.write_cache_grid) write_cache_grid_files(s,caches);

        std::cout << "[v31l] [stage 2/3] eigendecompose in parallel, then track branches\n";
        for(const auto& ic: caches) {
            write_tracked_outputs_for_label(s,ic,p,min_overlap_warn);
        }

        std::cout << "[v31l] [stage 3/3] 100% done\n";
    } catch(const std::exception& e) {
        std::cerr << "[v31l-error] " << e.what() << "\n";
        return 2;
    }
    return 0;
}
