#include "K3df_minuit_fit_v31l_lattice_covariance.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <string>

static std::map<std::string,std::string> read_simple_kv_main(const std::string& path) {
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

int main(int argc, char** argv) {
    if(argc != 2) {
        std::cerr << "Usage:\n  " << argv[0] << " config/config_v31l_K3df_fit_lattice_covariance.in\n";
        return 1;
    }
    try {
        using namespace k3df_fit_v31l;
        auto kv = read_simple_kv_main(argv[1]);
        FitSettings s;
        s.list_of_mom = split_ws(gs(kv,"list_of_mom","100_A2 110_A2"));
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

        s.use_lattice_covariance = gi(kv,"use_lattice_covariance",s.use_lattice_covariance);
        s.ensemble = gs(kv,"ensemble",s.ensemble);
        s.energy_cutoff = gd(kv,"energy_cutoff",s.energy_cutoff);
        s.max_state = gi(kv,"max_state",s.max_state);
        s.masses_path = gs(kv,"masses_path",s.masses_path);
        s.print_found_files = gi(kv,"print_found_files",s.print_found_files?1:0)!=0;
        s.target_levels_file = gs(kv,"target_levels_file",s.target_levels_file);
        s.chi_square_mode = gs(kv,"chi_square_mode",s.chi_square_mode);
        s.failure_penalty = gd(kv,"failure_penalty",s.failure_penalty);

        s.guess.K3iso0 = gd(kv,"K3iso0_guess",s.guess.K3iso0);
        s.guess.K3iso1 = gd(kv,"K3iso1_guess",s.guess.K3iso1);
        s.guess.K3B    = gd(kv,"K3B_guess",s.guess.K3B);
        s.guess.K3E    = gd(kv,"K3E_guess",s.guess.K3E);
        s.step.K3iso0 = gd(kv,"K3iso0_step",s.step.K3iso0);
        s.step.K3iso1 = gd(kv,"K3iso1_step",s.step.K3iso1);
        s.step.K3B    = gd(kv,"K3B_step",s.step.K3B);
        s.step.K3E    = gd(kv,"K3E_step",s.step.K3E);
        s.use_parameter_limits = gi(kv,"use_parameter_limits",s.use_parameter_limits?1:0)!=0;
        s.param_lower = gd(kv,"param_lower",s.param_lower);
        s.param_upper = gd(kv,"param_upper",s.param_upper);

        s.print_each_fcn_eval = gi(kv,"print_each_fcn_eval",s.print_each_fcn_eval?1:0)!=0;
        s.print_every_fcn_eval = gi(kv,"print_every_fcn_eval",s.print_every_fcn_eval);
        s.write_cache_grid = gi(kv,"write_cache_grid",s.write_cache_grid?1:0)!=0;
        s.write_bestfit_qc_grid = gi(kv,"write_bestfit_qc_grid",s.write_bestfit_qc_grid?1:0)!=0;
        s.save_binary_f3inv_cache = gi(kv,"save_binary_f3inv_cache",s.save_binary_f3inv_cache?1:0)!=0;
        s.load_binary_f3inv_cache = gi(kv,"load_binary_f3inv_cache",s.load_binary_f3inv_cache?1:0)!=0;
        s.binary_f3inv_cache_file = gs(kv,"binary_f3inv_cache_file",s.binary_f3inv_cache_file);
        s.zero_energy_mode = gs(kv,"zero_energy_mode",s.zero_energy_mode);
        s.output_dir = gs(kv,"output_dir",s.output_dir);
        s.output_tag = gs(kv,"output_tag",s.output_tag);

        std::cout << "[v31l-main] starting Kdf3 fit with lattice covariance, F3inv-only cache, and corrected Cartesian-l1 projector\n";
        std::cout << "[v31l-main] list_of_mom:";
        for(const auto& x: s.list_of_mom) std::cout << " " << x;
        std::cout << "\n[v31l-main] coarseN=" << s.coarseN << " threads=" << s.omp_threads << "\n";

        auto result = fit_K3df_parameters_v31l(s);
        print_fit_result_summary(result);
    } catch(const std::exception& e) {
        std::cerr << "[v31l-main-error] " << e.what() << "\n";
        return 2;
    }
    return 0;
}
