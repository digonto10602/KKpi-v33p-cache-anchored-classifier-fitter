#include "K3df_minuit_fit_v32f_fullF3inv_QCfull_cached_classifier.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

static std::map<std::string,std::string> read_kv_v32n(const std::string& path) {
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

static k3df_fit_v32f::FitSettings settings_from_config_v32n(const std::map<std::string,std::string>& kv) {
    using namespace k3df_fit_v32f;
    FitSettings s;
    s.list_of_mom = split_ws(gs(kv,"list_of_mom","000_A1m 100_A2 110_A2 111_A2 200_A2"));
    s.Lval = gd(kv,"Lval",s.Lval);
    s.xival = gd(kv,"xival",s.xival);
    s.scan_E0 = gd(kv,"scan_E0",s.scan_E0);
    s.scan_E1 = gd(kv,"scan_E1",s.scan_E1);
    s.coarseN = gi(kv,"coarseN",s.coarseN);
    s.refineN = gi(kv,"refineN",0);
    s.iterative_refine_enable = gi(kv,"iterative_refine_enable",0);
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
    s.waves_vec_1.clear(); for(const auto& x: split_ws(gs(kv,"waves_vec_1","0 1"))) s.waves_vec_1.push_back(std::stoi(x));
    s.waves_vec_2.clear(); for(const auto& x: split_ws(gs(kv,"waves_vec_2","0"))) s.waves_vec_2.push_back(std::stoi(x));

    for(int a=0; a<4; ++a) for(int b=0; b<3; ++b) {
        s.scatter_params_1[a][b] = gd(kv,"scatter1_"+std::to_string(a)+std::to_string(b),s.scatter_params_1[a][b]);
        s.scatter_params_2[a][b] = gd(kv,"scatter2_"+std::to_string(a)+std::to_string(b),s.scatter_params_2[a][b]);
    }

    s.use_lattice_covariance = gi(kv,"use_lattice_covariance",s.use_lattice_covariance?1:0)!=0;
    s.ensemble = gs(kv,"ensemble",s.ensemble);
    s.energy_cutoff = gd(kv,"energy_cutoff",s.energy_cutoff);
    s.max_state = gi(kv,"max_state",s.max_state);
    s.masses_path = gs(kv,"masses_path",s.masses_path);
    s.print_found_files = gi(kv,"print_found_files",s.print_found_files?1:0)!=0;
    s.target_levels_file = gs(kv,"target_levels_file",s.target_levels_file);
    s.chi_square_mode = gs(kv,"chi_square_mode",s.chi_square_mode);

    s.classifier_peak_ratio_min = gd(kv,"classifier_peak_ratio_min",s.classifier_peak_ratio_min);
    s.classifier_shoulder_gap = gi(kv,"classifier_shoulder_gap",s.classifier_shoulder_gap);
    s.zero_energy_mode = gs(kv,"zero_energy_mode",s.zero_energy_mode);

    s.output_dir = gs(kv,"output_dir",s.output_dir);
    s.output_tag = gs(kv,"output_tag",s.output_tag);
    s.write_cache_grid = gi(kv,"write_cache_grid",s.write_cache_grid?1:0)!=0;
    s.write_bestfit_qc_grid = 1;
    s.save_binary_f3inv_cache = gi(kv,"save_binary_f3inv_cache",s.save_binary_f3inv_cache?1:0)!=0;
    s.load_binary_f3inv_cache = gi(kv,"load_binary_f3inv_cache",s.load_binary_f3inv_cache?1:0)!=0;
    s.require_existing_binary_f3inv_cache = gi(kv,"require_existing_binary_f3inv_cache",s.require_existing_binary_f3inv_cache?1:0)!=0;
    s.binary_f3inv_cache_file = gs(kv,"binary_f3inv_cache_file",s.binary_f3inv_cache_file);
    return s;
}

static void write_lattice_targets_v32n(const k3df_fit_v32f::FitSettings& s, const std::vector<k3df_fit_v32f::TargetLevel>& targets) {
    std::filesystem::create_directories(s.output_dir);
    const std::string path = s.output_dir + "/" + s.output_tag + "_lattice_targets.dat";
    std::ofstream f(path);
    f << std::setprecision(17);
    f << "# columns: row Lbyas label nPx nPy nPz state Ecm err\n";
    for(std::size_t i=0;i<targets.size();++i) {
        const auto& t = targets[i];
        f << i << ' ' << t.Lbyas << ' ' << t.label << ' '
          << t.nP[0] << ' ' << t.nP[1] << ' ' << t.nP[2] << ' '
          << t.state << ' ' << t.Ecm << ' ' << t.err << "\n";
    }
    std::cout << "[v32n-write] wrote lattice targets: " << path << "\n";
}

int main(int argc, char** argv) {
    if(argc != 2) {
        std::cerr << "Usage:\n  " << argv[0] << " configs/config_v32n_defaultK3df_QC_cache_plot.in\n";
        return 1;
    }
    try {
        using namespace k3df_fit_v32f;
        auto kv = read_kv_v32n(argv[1]);
        FitSettings s = settings_from_config_v32n(kv);
        g_classifier_peak_ratio_min_v32f = s.classifier_peak_ratio_min;
        g_classifier_shoulder_gap_v32f = s.classifier_shoulder_gap;
        K3dfParameters p;
        p.K3iso0 = gd(kv,"K3iso0",gd(kv,"K3iso0_guess",0.1));
        p.K3iso1 = gd(kv,"K3iso1",gd(kv,"K3iso1_guess",0.1));
        p.K3B    = gd(kv,"K3B",gd(kv,"K3B_guess",0.1));
        p.K3E    = gd(kv,"K3E",gd(kv,"K3E_guess",0.1));

        std::filesystem::create_directories(s.output_dir);
        std::cout << "[v32n] fixed/default K3df projected-QC diagnostic scan\n";
        std::cout << "[v32n] K3iso0=" << std::setprecision(17) << p.K3iso0
                  << " K3iso1=" << p.K3iso1
                  << " K3B=" << p.K3B
                  << " K3E=" << p.K3E << "\n";
        std::cout << "[v32n] Ecm range [" << s.scan_E0 << "," << s.scan_E1 << "] coarseN=" << s.coarseN << " refineN=" << s.refineN << "\n";
        std::cout << "[v32n] classifier: old digonto_classifier_v1 on signed slogdet(det(projected QC)); predicted lines are written to *_fixedK3df_levels_zero_pole.dat\n";
        std::cout << "[v32n] cache-only mode: will load existing F3inv/Vsel binary cache first and abort if none exists.\n";

        std::cout << "[v32n] [stage 1/4] loading lattice targets up to Ecm <= " << s.energy_cutoff << "\n";
        auto [targets,cov,corr] = load_targets_and_covariance_v32f(s);
        write_lattice_targets_v32n(s,targets);
        std::cout << "[v32n] target count = " << targets.size() << " covariance_dim=" << cov.rows() << "x" << cov.cols() << "\n";

        std::cout << "[v32n] [stage 2/4] loading cached F3inv_full and Vsel\n";
        auto caches = get_or_build_F3inv_cache_v32f(s);
        write_cache_grid_files(s,caches);

        std::cout << "[v32n] [stage 3/4] computing projected QC diagnostics in OpenMP\n";
        write_qc_eig_grid_files(s,caches,p,make_base_physics(s),"fixedK3df");

        std::cout << "[v32n] [stage 4/4] writing combined accepted classifier predictions\n";
        const std::string combined = s.output_dir + "/" + s.output_tag + "_classifier_predictions.dat";
        std::ofstream sf(combined);
        sf << std::setprecision(17);
        sf << "# K3iso0 " << p.K3iso0 << "\n";
        sf << "# K3iso1 " << p.K3iso1 << "\n";
        sf << "# K3B " << p.K3B << "\n";
        sf << "# K3E " << p.K3E << "\n";
        sf << "# columns: Lbyas label irrep kind level_index Ecm E_left E_right signed_slogdet_left signed_slogdet_right\n";
        auto all = find_all_zero_poles(caches,p,make_base_physics(s),s.debug,s.zero_energy_mode);
        for(const auto& kvp: all) {
            MomentumIrrepSpec spec = parse_label(kvp.first);
            int level=0;
            for(const auto& z: kvp.second) {
                sf << s.Lval << ' ' << kvp.first << ' ' << spec.irrep << ' '
                   << z.kind << ' ' << level++ << ' ' << z.E_mid << ' '
                   << z.E_left << ' ' << z.E_right << ' ' << z.y_left << ' ' << z.y_right << "\n";
            }
        }
        std::cout << "[v32n] wrote " << combined << "\n";
        std::cout << "[v32n] 100% done\n";
    } catch(const std::exception& e) {
        std::cerr << "[v32n-error] " << e.what() << "\n";
        return 2;
    }
    return 0;
}
