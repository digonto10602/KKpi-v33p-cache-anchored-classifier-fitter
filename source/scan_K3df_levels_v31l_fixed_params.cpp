#include "K3df_minuit_fit_v31l_lattice_covariance.hpp"

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

static std::map<std::string,std::string> read_simple_kv_scan(const std::string& path) {
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
    auto kv = read_simple_kv_scan(config);
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
    s.write_bestfit_qc_grid = 1;

    return s;
}

int main(int argc, char** argv) {
    if(argc != 2) {
        std::cerr << "Usage:\n  " << argv[0] << " config/config_v31l_fixed_K3df_spectrum.in\n";
        return 1;
    }

    try {
        using namespace k3df_fit_v31l;
        FitSettings s = settings_from_config(argv[1]);
        auto kv = read_simple_kv_scan(argv[1]);

        K3dfParameters p;
        p.K3iso0 = gd(kv,"K3iso0",0.0);
        p.K3iso1 = gd(kv,"K3iso1",0.0);
        p.K3B    = gd(kv,"K3B",0.0);
        p.K3E    = gd(kv,"K3E",0.0);

        std::filesystem::create_directories(s.output_dir);

        std::cout << "[v31l-scan] fixed Kdf3 spectrum scan\n";
        std::cout << "[v31l-scan] K3iso0=" << std::setprecision(17) << p.K3iso0
                  << " K3iso1=" << p.K3iso1
                  << " K3B=" << p.K3B
                  << " K3E=" << p.K3E << "\n";
        std::cout << "[v31l-scan] list_of_mom:";
        for(const auto& x: s.list_of_mom) std::cout << " " << x;
        std::cout << "\n[v31l-scan] coarseN=" << s.coarseN
                  << " E=[" << s.scan_E0 << "," << s.scan_E1 << "] threads=" << s.omp_threads << "\n";

        std::cout << "[v31l-scan] [stage 1/4] building F3inv-only projected cache\n";
        auto caches = get_or_build_F3inv_cache_v31l(s);
        write_cache_grid_files(s,caches);

        std::cout << "[v31l-scan] [stage 2/4] computing closest eigenvalue grids and zero/pole levels\n";
        write_qc_eig_grid_files(s,caches,p,make_base_physics(s),"fixedK3df");

        std::cout << "[v31l-scan] [stage 3/4] writing combined spectrum file\n";
        const std::string combined = s.output_dir + "/" + s.output_tag + "_fixedK3df_spectrum_L_irrep_levels.dat";
        std::ofstream sf(combined);
        sf << std::setprecision(17);
        sf << "# K3iso0 " << p.K3iso0 << "\n";
        sf << "# K3iso1 " << p.K3iso1 << "\n";
        sf << "# K3B " << p.K3B << "\n";
        sf << "# K3E " << p.K3E << "\n";
        sf << "# columns: Lbyas label irrep kind level_index Ecm E_left E_right y_left y_right\n";
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
        std::cout << "[v31l-scan] wrote " << combined << "\n";

        std::cout << "[v31l-scan] [stage 4/4] 100% done\n";
    } catch(const std::exception& e) {
        std::cerr << "[v31l-scan-error] " << e.what() << "\n";
        return 2;
    }
    return 0;
}
