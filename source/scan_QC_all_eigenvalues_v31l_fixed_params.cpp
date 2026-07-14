#include "K3df_minuit_fit_v31l_lattice_covariance.hpp"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

static std::map<std::string,std::string> read_simple_kv_scan_all(const std::string& path) {
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
    auto kv = read_simple_kv_scan_all(config);
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
    s.zero_energy_mode = gs(kv,"zero_energy_mode",s.zero_energy_mode);
    return s;
}

struct AllEigPoint {
    int i=-1;
    double Ecm=std::numeric_limits<double>::quiet_NaN();
    int success=0;
    int proj_dim=0;
    double herm_rel=std::numeric_limits<double>::quiet_NaN();
    std::vector<double> evals;
    int closest_index=-1;
    double closest_value=std::numeric_limits<double>::quiet_NaN();
};

static AllEigPoint eval_all_eigenvalues(
        const k3df_fit_v31l::ProjectedQCCacheEntry& e,
        const k3df_fit_v31l::K3dfParameters& p,
        const PhysicsParams& par,
        char debug) {
    using namespace k3df_fit_v31l;
    AllEigPoint out;
    out.i = e.i;
    out.Ecm = e.Ecm;
    out.proj_dim = e.proj_dim;
    if(!e.success) return out;

    Eigen::MatrixXcd QC = assemble_QC(e,p,par,debug);
    if(QC.rows()==0 || QC.cols()==0 || QC.rows()!=QC.cols() || !QC.allFinite()) return out;

    const double nrm = QC.norm();
    out.herm_rel = (nrm > 0.0) ? (QC-QC.adjoint()).norm()/nrm : 0.0;
    Eigen::MatrixXcd H = 0.5*(QC + QC.adjoint());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H, Eigen::EigenvaluesOnly);
    if(es.info()!=Eigen::Success) return out;

    out.evals.resize(es.eigenvalues().size());
    double best_abs = std::numeric_limits<double>::infinity();
    int best_idx = -1;
    for(int k=0; k<es.eigenvalues().size(); ++k) {
        const double val = es.eigenvalues()[k];
        out.evals[k] = val;
        const double av = std::abs(val);
        if(std::isfinite(av) && av < best_abs) {
            best_abs = av;
            best_idx = k;
        }
    }
    out.closest_index = best_idx;
    if(best_idx >= 0) out.closest_value = out.evals[best_idx];
    out.success = 1;
    return out;
}

static void write_all_eigenvalue_outputs(
        const k3df_fit_v31l::FitSettings& s,
        const std::vector<k3df_fit_v31l::IrrepCache>& caches,
        const k3df_fit_v31l::K3dfParameters& p) {
    using namespace k3df_fit_v31l;
    std::filesystem::create_directories(s.output_dir);
    PhysicsParams par = make_base_physics(s);

    for(const auto& ic: caches) {
        std::cout << "[v31l-all-eigs] label=" << ic.label << " computing all projected-QC eigenvalues\n";
        std::vector<AllEigPoint> vals(ic.grid.size());
        int done=0, nextpct=10;

        #pragma omp parallel for schedule(dynamic,1)
        for(int i=0; i<(int)ic.grid.size(); ++i) {
            vals[i] = eval_all_eigenvalues(ic.grid[i],p,par,s.debug);
            #pragma omp critical
            {
                ++done;
                progress_percent_log("v31l-all-eigs-"+ic.label, done, (int)ic.grid.size(), nextpct, 10);
            }
        }

        const std::string long_path = s.output_dir + "/" + s.output_tag + "_" + clean_label(ic.label) + "_all_QC_eigenvalues_long.dat";
        std::ofstream lf(long_path);
        lf << std::setprecision(17);
        lf << "# K3iso0 " << p.K3iso0 << "\n";
        lf << "# K3iso1 " << p.K3iso1 << "\n";
        lf << "# K3B " << p.K3B << "\n";
        lf << "# K3E " << p.K3E << "\n";
        lf << "# columns: i Ecm success proj_dim herm_QC_rel eig_index eig_value is_closest closest_index closest_value\n";
        for(const auto& v: vals) {
            if(!v.success) {
                lf << v.i << ' ' << v.Ecm << ' ' << 0 << ' ' << v.proj_dim << ' ' << v.herm_rel
                   << " -1 nan 0 -1 nan\n";
                continue;
            }
            for(int k=0; k<(int)v.evals.size(); ++k) {
                const int is_closest = (k == v.closest_index) ? 1 : 0;
                lf << v.i << ' ' << v.Ecm << ' ' << v.success << ' ' << v.proj_dim << ' ' << v.herm_rel
                   << ' ' << k << ' ' << v.evals[k] << ' ' << is_closest << ' '
                   << v.closest_index << ' ' << v.closest_value << "\n";
            }
        }
        std::cout << "[v31l-write] wrote " << long_path << "\n";

        const std::string closest_path = s.output_dir + "/" + s.output_tag + "_" + clean_label(ic.label) + "_QC_closest_eig_grid.dat";
        std::ofstream cf(closest_path);
        cf << std::setprecision(17);
        cf << "# columns: i Ecm success y_closest_zero eig_index herm_QC_rel proj_dim\n";
        for(const auto& v: vals) {
            cf << v.i << ' ' << v.Ecm << ' ' << v.success << ' ' << v.closest_value << ' '
               << v.closest_index << ' ' << v.herm_rel << ' ' << v.proj_dim << "\n";
        }
        std::cout << "[v31l-write] wrote " << closest_path << "\n";
    }
}

int main(int argc, char** argv) {
    if(argc != 2) {
        std::cerr << "Usage:\n  " << argv[0] << " config/config_v31l_QC_all_eigenvalues.in\n";
        return 1;
    }
    try {
        using namespace k3df_fit_v31l;
        FitSettings s = settings_from_config(argv[1]);
        auto kv = read_simple_kv_scan_all(argv[1]);

        K3dfParameters p;
        p.K3iso0 = gd(kv,"K3iso0",0.0);
        p.K3iso1 = gd(kv,"K3iso1",0.0);
        p.K3B    = gd(kv,"K3B",0.0);
        p.K3E    = gd(kv,"K3E",0.0);

        #ifdef _OPENMP
        omp_set_num_threads(s.omp_threads);
        omp_set_max_active_levels(1);
        #endif
        Eigen::setNbThreads(1);

        std::cout << std::setprecision(17);
        std::cout << "[v31l] projected-QC all-eigenvalue scan\n";
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

        std::cout << "[v31l] [stage 2/3] build QC and diagonalize all eigenvalues in parallel\n";
        write_all_eigenvalue_outputs(s,caches,p);

        std::cout << "[v31l] [stage 3/3] 100% done\n";
    } catch(const std::exception& e) {
        std::cerr << "[v31l-error] " << e.what() << "\n";
        return 2;
    }
    return 0;
}
