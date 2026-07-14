#include "K3df_minuit_fit_v32f_fullF3inv_QCfull_cached_classifier.hpp"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>

static std::map<std::string,std::string> read_kv_v32q(const std::string& path) {
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

static k3df_fit_v32f::FitSettings settings_from_config_v32q(const std::map<std::string,std::string>& kv) {
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
    s.lattice_energy_type = gs(kv,"lattice_energy_type",s.lattice_energy_type);
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

struct SLogDetInfo {
    int ok = 0;
    double det_re = std::numeric_limits<double>::quiet_NaN();
    double det_sign = std::numeric_limits<double>::quiet_NaN();
    double logabsdet = std::numeric_limits<double>::quiet_NaN();
    double signed_logabsdet = std::numeric_limits<double>::quiet_NaN();
};

static SLogDetInfo slogdet_matrix_v32q(const Eigen::MatrixXcd& M) {
    SLogDetInfo v;
    if(M.rows()==0 || M.rows()!=M.cols()) return v;
    try {
        Eigen::PartialPivLU<Eigen::MatrixXcd> lu(M);
        const comp detc = lu.determinant();
        v.det_re = detc.real();
        v.det_sign = (v.det_re>0.0) ? 1.0 : ((v.det_re<0.0) ? -1.0 : 0.0);
        const auto& LU = lu.matrixLU();
        double lga = 0.0;
        bool ok_lga = true;
        for(int d=0; d<LU.rows(); ++d) {
            const double a = std::abs(LU(d,d));
            if(!(a>0.0) || !std::isfinite(a)) { ok_lga=false; break; }
            lga += std::log(a);
        }
        if(ok_lga) v.logabsdet = lga;
        else if(std::isfinite(std::abs(detc)) && std::abs(detc)>0.0) v.logabsdet = std::log(std::abs(detc));
        if(std::isfinite(v.det_sign) && v.det_sign!=0.0 && std::isfinite(v.logabsdet)) {
            v.signed_logabsdet = v.det_sign * v.logabsdet;
            v.ok = 1;
        }
    } catch(...) {}
    return v;
}


static void write_lattice_file_audit_v32q(const k3df_fit_v32f::FitSettings& s) {
    if(!s.use_lattice_covariance) return;
    std::filesystem::create_directories(s.output_dir);
    const std::filesystem::path masses_path = s.masses_path.empty()
        ? lattice_covariance::default_szscl21_mass_path()
        : std::filesystem::path(s.masses_path);
    std::cout << "[v32q-audit] Lval/Lbyas = " << s.Lval << " xi = " << s.xival << " L*xi = " << (s.Lval*s.xival) << "\n";
    std::cout << "[v32q-audit] ensemble = " << s.ensemble << "\n";
    std::cout << "[v32q-audit] masses_path = " << masses_path.string() << "\n";
    std::cout << "[v32q-audit] energy_cutoff = " << s.energy_cutoff << " lattice_energy_type = " << s.lattice_energy_type << "\n";

    auto r = covariance_between_states_szscl21_based_result(
        s.ensemble, s.Lval, s.xival, s.energy_cutoff, s.list_of_mom,
        s.max_state, masses_path, false, s.lattice_energy_type);

    const std::string path = s.output_dir + "/" + s.output_tag + "_lattice_all_files_audit.dat";
    std::ofstream f(path);
    f << std::setprecision(17);
    f << "# v32q all discovered lattice mass files audit. This includes states above cutoff.\n";
    f << "# ensemble " << s.ensemble << "\n";
    f << "# masses_path " << masses_path.string() << "\n";
    f << "# Lval " << s.Lval << "\n";
    f << "# xi " << s.xival << "\n";
    f << "# Lbyas_times_xi_for_momentum " << (s.Lval*s.xival) << "\n";
    f << "# energy_cutoff " << s.energy_cutoff << "\n";
    f << "# lattice_energy_type " << r.lattice_energy_type << "\n";
    f << "# columns: row label irrep state nPx nPy nPz atP n_samples E_read_jk_avg E_read_jk_err Ecm_used_jk_avg Ecm_used_jk_err keep keep_reason filename\n";
    int row=0;
    for(const auto& ar : r.file_audit) {
        MomentumIrrepSpec spec = parse_label(ar.label);
        f << row++ << ' ' << ar.label << ' ' << spec.irrep << ' '
          << ar.state << ' ' << ar.nP[0] << ' ' << ar.nP[1] << ' ' << ar.nP[2] << ' '
          << ar.atP << ' ' << ar.n_samples << ' '
          << ar.jk_avg_read << ' ' << ar.jk_err_read << ' '
          << ar.jk_avg_ecm << ' ' << ar.jk_err_ecm << ' '
          << ar.keep << ' ' << ar.keep_reason << ' ' << ar.filename << "\n";
    }
    std::cout << "[v32q-audit] wrote all-files lattice audit: " << path << " rows=" << r.file_audit.size() << "\n";
}

static void write_lattice_targets_v32q(const k3df_fit_v32f::FitSettings& s, const std::vector<k3df_fit_v32f::TargetLevel>& targets) {
    std::filesystem::create_directories(s.output_dir);
    const std::string path = s.output_dir + "/" + s.output_tag + "_lattice_targets.dat";
    std::ofstream f(path);
    f << std::setprecision(17);
    f << "# v32q lattice audit. Default lattice_energy_type=Ecm means file values are used directly.\n";
    f << "# If lattice_energy_type=En_lab, Ecm_used=sqrt(E_read^2-atP^2).\n";
    f << "# columns: row Lbyas label irrep nPx nPy nPz state E_read err_read lattice_energy_type atP shifted_from_lab Ecm err\n";
    for(std::size_t i=0;i<targets.size();++i) {
        const auto& t = targets[i];
        MomentumIrrepSpec spec = parse_label(t.label);
        f << i << ' ' << t.Lbyas << ' ' << t.label << ' ' << spec.irrep << ' '
          << t.nP[0] << ' ' << t.nP[1] << ' ' << t.nP[2] << ' '
          << t.state << ' ' << t.E_read << ' ' << t.err_read << ' ' << t.lattice_energy_type << ' '
          << t.atP << ' ' << t.shifted_from_lab << ' ' << t.Ecm << ' ' << t.err << "\n";
    }
    std::cout << "[v32q-write] wrote lattice targets/audit: " << path << "\n";
}

static void write_4panel_grid_v32q(const k3df_fit_v32f::FitSettings& s,
                                   const std::vector<k3df_fit_v32f::IrrepCache>& caches,
                                   const k3df_fit_v32f::K3dfParameters& p,
                                   const PhysicsParams& par) {
    using namespace k3df_fit_v32f;
    std::filesystem::create_directories(s.output_dir);
    for(const auto& ic: caches) {
        std::vector<QCPointValue> qc_vals;
        auto zp = find_zero_poles_from_cache(ic,p,par,s.debug,s.zero_energy_mode,&qc_vals);
        const std::string grid_path = s.output_dir + "/" + s.output_tag + "_" + clean_label(ic.label) + "_fixedK3df_F3inv_QC_4panel_grid.dat";
        std::ofstream f(grid_path);
        f << std::setprecision(17);
        f << "# v32q 4-panel projected F3inv/QC diagnostic for a single irrep label.\n";
        f << "# K3iso0 " << p.K3iso0 << "\n";
        f << "# K3iso1 " << p.K3iso1 << "\n";
        f << "# K3B " << p.K3B << "\n";
        f << "# K3E " << p.K3E << "\n";
        f << "# columns: i Ecm success proj_dim signed_slogdet_projF3inv logabsdet_projF3inv det_sign_projF3inv det_re_projF3inv signed_slogdet_projQC logabsdet_projQC det_sign_projQC det_re_projQC closestEigSigned_QC eig_index_QC herm_QC_rel minAbsEigQC minSVprojQC\n";
        for(std::size_t i=0; i<ic.grid.size(); ++i) {
            const auto& e = ic.grid[i];
            SLogDetInfo f3 = e.success ? slogdet_matrix_v32q(e.F3inv_proj) : SLogDetInfo{};
            const auto& qv = qc_vals[i];
            f << i << ' ' << e.Ecm << ' ' << e.success << ' ' << e.proj_dim << ' '
              << f3.signed_logabsdet << ' ' << f3.logabsdet << ' ' << f3.det_sign << ' ' << f3.det_re << ' '
              << qv.signed_logabsdet << ' ' << qv.logabsdet << ' ' << qv.det_sign << ' ' << qv.det_re << ' '
              << qv.y << ' ' << qv.eig_index << ' ' << qv.herm_rel << ' ' << qv.min_abs_eig << ' ' << qv.min_sv << "\n";
        }
        const std::string zpath = s.output_dir + "/" + s.output_tag + "_" + clean_label(ic.label) + "_fixedK3df_levels_zero_pole.dat";
        std::ofstream zf(zpath);
        zf << std::setprecision(17);
        zf << "# columns: idx kind E_mid E_left E_right signed_slogdet_QC_left signed_slogdet_QC_right eig_index_left eig_index_right\n";
        int idx=0;
        for(const auto& z: zp) {
            zf << idx++ << ' ' << z.kind << ' ' << z.E_mid << ' ' << z.E_left << ' ' << z.E_right << ' '
               << z.y_left << ' ' << z.y_right << ' ' << z.eig_index_left << ' ' << z.eig_index_right << "\n";
        }
        std::cout << "[v32q-write] wrote " << grid_path << "\n";
        std::cout << "[v32q-write] wrote " << zpath << "\n";
    }
}

int main(int argc, char** argv) {
    if(argc != 2) {
        std::cerr << "Usage:\n  " << argv[0] << " configs/config_v32q_F3inv_QC_4panel_K3df1em5_L20ensemble_audit.in\n";
        return 1;
    }
    try {
        using namespace k3df_fit_v32f;
        auto kv = read_kv_v32q(argv[1]);
        FitSettings s = settings_from_config_v32q(kv);
        g_classifier_peak_ratio_min_v32f = s.classifier_peak_ratio_min;
        g_classifier_shoulder_gap_v32f = s.classifier_shoulder_gap;
        K3dfParameters p;
        p.K3iso0 = gd(kv,"K3iso0",0.00001);
        p.K3iso1 = gd(kv,"K3iso1",0.00001);
        p.K3B    = gd(kv,"K3B",0.00001);
        p.K3E    = gd(kv,"K3E",0.00001);

        std::filesystem::create_directories(s.output_dir);
        std::cout << "[v32q] 4-panel projected-F3inv/projected-QC diagnostic scan\n";
        std::cout << "[v32q] K3iso0=" << std::setprecision(17) << p.K3iso0
                  << " K3iso1=" << p.K3iso1
                  << " K3B=" << p.K3B
                  << " K3E=" << p.K3E << "\n";
        std::cout << "[v32q] Ecm range [" << s.scan_E0 << "," << s.scan_E1 << "] coarseN=" << s.coarseN << " refineN=" << s.refineN << "\n";
        std::cout << "[v32q] list_of_mom="; for(const auto& L: s.list_of_mom) std::cout << L << ' '; std::cout << "\n";
        std::cout << "[v32q] exact construction: QC_proj = Vsel.adjoint() * (F3inv_full + K3df_full) * Vsel\n";
        std::cout << "[v32q] cache-only mode: will load existing F3inv_full/Vsel binary cache and abort if missing.\n";

        std::cout << "[v32q] [stage 1/4] loading lattice targets with same Lbyas/xi/masses/irrep list\n";
        std::cout << "[v32q] lattice_energy_type=" << s.lattice_energy_type << " (default Ecm: no sqrt(E^2-P^2) conversion)\n";
        write_lattice_file_audit_v32q(s);
        auto [targets,cov,corr] = load_targets_and_covariance_v32f(s);
        write_lattice_targets_v32q(s,targets);
        std::cout << "[v32q] target count = " << targets.size() << " covariance_dim=" << cov.rows() << "x" << cov.cols() << "\n";

        std::cout << "[v32q] [stage 2/4] loading cached F3inv_full and Vsel\n";
        auto caches = get_or_build_F3inv_cache_v32f(s);
        write_cache_grid_files(s,caches);

        std::cout << "[v32q] [stage 3/4] computing projected F3inv and projected QC quantities\n";
        write_4panel_grid_v32q(s,caches,p,make_base_physics(s));

        std::cout << "[v32q] [stage 4/4] writing combined classifier prediction file\n";
        const std::string combined = s.output_dir + "/" + s.output_tag + "_classifier_predictions.dat";
        std::ofstream sf(combined);
        sf << std::setprecision(17);
        sf << "# K3iso0 " << p.K3iso0 << "\n";
        sf << "# K3iso1 " << p.K3iso1 << "\n";
        sf << "# K3B " << p.K3B << "\n";
        sf << "# K3E " << p.K3E << "\n";
        sf << "# columns: Lbyas label irrep kind level_index Ecm E_left E_right signed_slogdet_QC_left signed_slogdet_QC_right\n";
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
        std::cout << "[v32q] wrote " << combined << "\n";
        std::cout << "[v32q] 100% done\n";
    } catch(const std::exception& e) {
        std::cerr << "[v32q-error] " << e.what() << "\n";
        return 2;
    }
    return 0;
}
