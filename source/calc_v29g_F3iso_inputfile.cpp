#include "F3_cpu_openmp_v25_K3QC_cached_core.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

// v29g: input-file driven F3iso / det(F3) / det(projected F3) scanner.
// Input format is plain key=value text. Lines beginning with # are comments.
// This avoids external JSON/XML dependencies and keeps the runtime overhead negligible.

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
        if (eq == std::string::npos) {
            throw std::runtime_error("Bad input line " + std::to_string(lineno) + ": expected key=value");
        }
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
    auto it = kv.find(key);
    if (it == kv.end()) return def;
    try { size_t p=0; double x=std::stod(it->second,&p); if(p!=it->second.size()) throw std::runtime_error("bad"); return x; }
    catch(...) { throw std::runtime_error("Could not parse double for key " + key + " = " + it->second); }
}

static int get_int(const std::map<std::string,std::string>& kv, const std::string& key, int def) {
    auto it = kv.find(key);
    if (it == kv.end()) return def;
    try { size_t p=0; int x=std::stoi(it->second,&p); if(p!=it->second.size()) throw std::runtime_error("bad"); return x; }
    catch(...) { throw std::runtime_error("Could not parse int for key " + key + " = " + it->second); }
}

static std::string get_string(const std::map<std::string,std::string>& kv, const std::string& key, const std::string& def) {
    auto it = kv.find(key);
    return (it == kv.end()) ? def : it->second;
}

static std::vector<int> get_int_list(const std::map<std::string,std::string>& kv, const std::string& key, const std::vector<int>& def) {
    auto it = kv.find(key);
    if (it == kv.end()) return def;
    std::vector<int> out;
    for (const std::string& tok: split_csv(it->second)) out.push_back(std::stoi(tok));
    return out;
}

struct Options {
    std::string mom_label = "000_A1m";
    double Lval=20.0;
    double xival=3.444;
    double E0=0.2631;
    double E1=0.36;
    int N=1601;
    int threads=18;
    char debug='n';
    double atmpi=0.06906;
    double atmK=0.09698;
    double eta_1=1.0;
    double eta_2=0.5;
    double alpha=0.5;
    double epsilon_h=0.0;
    double max_shell_num=20.0;
    double tolerance=1.0e-12;
    int parity=-1;
    double eig_tol=0.05;
    double norm_tol=1.0e-12;
    double proj_tol=1.0e-10;
    bool Q0norm=true;
    bool sort_orbit_flag=false;
    std::vector<int> waves_vec_1{0,1};
    std::vector<int> waves_vec_2{0};
    std::string outdir="output";
    std::string prefix="debug_v29g_F3iso";
};

static Options read_options(const std::string& input_file, PhysicsParams& par) {
    const auto kv = read_kv_file(input_file);
    Options o;
    o.mom_label = get_string(kv, "mom_label", o.mom_label);
    o.Lval = get_double(kv, "Lval", o.Lval);
    o.xival = get_double(kv, "xival", o.xival);
    o.E0 = get_double(kv, "E0", o.E0);
    o.E1 = get_double(kv, "E1", o.E1);
    o.N = get_int(kv, "N", o.N);
    o.threads = get_int(kv, "threads", o.threads);
    o.debug = get_string(kv, "debug", "n").empty() ? 'n' : get_string(kv, "debug", "n")[0];
    o.atmpi = get_double(kv, "atmpi", o.atmpi);
    o.atmK = get_double(kv, "atmK", o.atmK);
    o.eta_1 = get_double(kv, "eta_1", o.eta_1);
    o.eta_2 = get_double(kv, "eta_2", o.eta_2);
    o.alpha = get_double(kv, "alpha", o.alpha);
    o.epsilon_h = get_double(kv, "epsilon_h", o.epsilon_h);
    o.max_shell_num = get_double(kv, "max_shell_num", o.max_shell_num);
    o.tolerance = get_double(kv, "tolerance", o.tolerance);
    o.parity = get_int(kv, "parity", o.parity);
    o.eig_tol = get_double(kv, "eig_tol", o.eig_tol);
    o.norm_tol = get_double(kv, "norm_tol", o.norm_tol);
    o.proj_tol = get_double(kv, "proj_tol", o.proj_tol);
    o.Q0norm = (get_int(kv, "Q0norm", 1) != 0);
    o.sort_orbit_flag = (get_int(kv, "sort_orbit_flag", 0) != 0);
    o.waves_vec_1 = get_int_list(kv, "waves_vec_1", o.waves_vec_1);
    o.waves_vec_2 = get_int_list(kv, "waves_vec_2", o.waves_vec_2);
    o.outdir = get_string(kv, "outdir", o.outdir);
    o.prefix = get_string(kv, "prefix", o.prefix);

    par.atmpi=o.atmpi; par.atmK=o.atmK; par.xi=o.xival; par.Lbyas=o.Lval;
    par.eta_1=o.eta_1; par.eta_2=o.eta_2; par.alpha=o.alpha;
    par.epsilon_h=o.epsilon_h; par.max_shell_num=o.max_shell_num; par.tolerance=o.tolerance;
    par.parity=o.parity; par.eig_tol=o.eig_tol; par.norm_tol=o.norm_tol; par.proj_tol=o.proj_tol;
    par.Q0norm=o.Q0norm; par.sort_orbit_flag=o.sort_orbit_flag; par.omp_threads=o.threads;
    par.waves_vec_1=o.waves_vec_1; par.waves_vec_2=o.waves_vec_2;

    // Keep all K3 parameters explicitly zero here because this executable is for F3, not QC = F3^{-1}+Kdf3.
    par.K3iso = {comp(0.0,0.0), comp(0.0,0.0)};
    par.K3B_par = comp(0.0,0.0);
    par.K3E_par = comp(0.0,0.0);

    // Scattering parameter defaults are PhysicsParams defaults. The input file can override each coefficient.
    for (int i=0; i<4; ++i) {
        for (int j=0; j<3; ++j) {
            const std::string k1 = "scatter1_" + std::to_string(i) + std::to_string(j);
            const std::string k2 = "scatter2_" + std::to_string(i) + std::to_string(j);
            par.scatter_params_1[i][j] = comp(get_double(kv, k1, par.scatter_params_1[i][j].real()), 0.0);
            par.scatter_params_2[i][j] = comp(get_double(kv, k2, par.scatter_params_2[i][j].real()), 0.0);
        }
    }

    if (o.N < 2) throw std::runtime_error("N must be >= 2");
    if (!(o.E1 > o.E0)) throw std::runtime_error("E1 must be greater than E0");
    return o;
}

struct F3IsoResult {
    double Ecm = std::numeric_limits<double>::quiet_NaN();
    double En = std::numeric_limits<double>::quiet_NaN();
    int dim1 = 0;
    int dim2 = 0;
    int total_dim = 0;
    int vdim = 0;
    comp F3iso = comp(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
    comp detF3 = comp(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
    comp detProjF3 = comp(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
    bool success = false;
    std::string error;
};

static F3IsoResult evaluate_F3iso_one_energy(
    double Ecm,
    const std::vector<int>& nnP_vec,
    const std::string& irrep,
    const PhysicsParams& par,
    char debug)
{
    F3IsoResult out;
    out.Ecm = Ecm;
    try {
        const comp pi = std::acos(-1.0);
        const double L = par.L();
        const comp twopibyL = ((comp)2.0) * pi / ((comp)L);

        std::vector<comp> total_P(3);
        total_P[0] = twopibyL * double(nnP_vec[0]);
        total_P[1] = twopibyL * double(nnP_vec[1]);
        total_P[2] = twopibyL * double(nnP_vec[2]);

        std::vector<comp> nnP_config(3);
        nnP_config[0] = comp(nnP_vec[0], 0.0);
        nnP_config[1] = comp(nnP_vec[1], 0.0);
        nnP_config[2] = comp(nnP_vec[2], 0.0);

        const comp Ecm_c(Ecm, 0.0);
        const comp En_c = Ecm_to_E(Ecm_c, total_P);
        out.En = En_c.real();

        std::vector<std::vector<comp>> plm_config(5);
        std::vector<std::vector<comp>> klm_config(5);
        std::vector<std::vector<int>> np_config(5);
        std::vector<std::vector<int>> nk_config(5);

        config_maker_4(plm_config, np_config, par.waves_vec_1, En_c, total_P,
                       par.atmK, par.atmK, par.atmpi, L,
                       par.epsilon_h, par.max_shell_num, par.tolerance);
        config_maker_4(klm_config, nk_config, par.waves_vec_2, En_c, total_P,
                       par.atmpi, par.atmK, par.atmK, L,
                       par.epsilon_h, par.max_shell_num, par.tolerance);

        const int A = int(plm_config[0].size());
        const int B = int(klm_config[0].size());
        const int N = A + B;
        out.dim1 = A; out.dim2 = B; out.total_dim = N;
        if (N <= 0) { out.error = "empty config"; return out; }

        Eigen::MatrixXcd F2(N, N), G(N, N), K2inv(N, N);
        F2_2plus1_mat(F2, En_c, plm_config, klm_config, total_P,
                      par.atmK, par.atmpi, L, par.alpha, par.epsilon_h,
                      par.max_shell_num, par.Q0norm);
        K2inv_EREord2_2plus1_mat(K2inv, par.eta_1, par.eta_2,
                                 par.scatter_params_1, par.scatter_params_2,
                                 En_c, plm_config, klm_config, total_P,
                                 par.atmK, par.atmpi, par.epsilon_h, L);
        G_2plus1_mat(G, En_c, plm_config, klm_config, total_P,
                     par.atmK, par.atmpi, L, par.alpha, par.epsilon_h,
                     par.max_shell_num, par.Q0norm);

        const Eigen::MatrixXcd H = K2inv + F2 + G;
        Eigen::PartialPivLU<Eigen::MatrixXcd> luH(H);
        const Eigen::MatrixXcd X1 = luH.solve(F2);
        const Eigen::MatrixXcd F3 = (F2 / comp(3.0, 0.0)) - F2 * X1;

        Eigen::VectorXcd norm_vec(N);
        for (int i=0; i<A; ++i) norm_vec(i) = comp(1.0, 0.0);
        for (int i=0; i<B; ++i) norm_vec(A+i) = comp(1.0/std::sqrt(2.0), 0.0);
        out.F3iso = (norm_vec.transpose() * F3 * norm_vec)(0,0);
        out.detF3 = determinant_via_partial_piv_lu(F3);

        Eigen::MatrixXcd P_I(N, N);
        std::string I_mut = irrep;
        std::vector<comp> total_P_mut = total_P;
        std::vector<comp> nnP_config_mut = nnP_config;
        P_irrep_projection_2plus1(P_I, plm_config, np_config, klm_config, nk_config,
                                  I_mut, total_P_mut, nnP_config_mut,
                                  par.sort_orbit_flag, par.parity);

        Eigen::MatrixXcd Vsel, Pproj;
        build_projector_from_eigenvectors_near_one(P_I, Vsel, Pproj,
                                                   par.eig_tol, par.norm_tol, par.proj_tol,
                                                   debug);
        out.vdim = int(Vsel.cols());
        if (out.vdim <= 0) { out.error = "projection produced Vsel with zero columns"; return out; }

        const Eigen::MatrixXcd projectedF3 = Vsel.adjoint() * F3 * Vsel;
        out.detProjF3 = determinant_via_partial_piv_lu(projectedF3);
        out.success = true;
    } catch (const std::exception& e) {
        out.success = false;
        out.error = e.what();
    }
    return out;
}

static void usage(const char* prog) {
    std::cerr << "Usage:\n  " << prog << " input/config_v29g_F3iso_000_A1m.in\n";
}

int main(int argc, char** argv) {
    if (argc != 2) { usage(argv[0]); return 1; }
    const std::string input_file = argv[1];

    try {
        PhysicsParams par;
        Options opt = read_options(input_file, par);
        MomentumIrrepSpec spec = parse_label(opt.mom_label);
        std::vector<int> nnP_vec = {spec.nnP[0], spec.nnP[1], spec.nnP[2]};

        #ifdef _OPENMP
        omp_set_num_threads(opt.threads);
        #endif

        std::filesystem::create_directories(opt.outdir);
        const std::string outfile = opt.outdir + "/" + opt.prefix + "_" + opt.mom_label + ".dat";
        const std::string metafile = opt.outdir + "/" + opt.prefix + "_metadata.txt";

        std::vector<F3IsoResult> results(opt.N);
        #pragma omp parallel for schedule(dynamic)
        for (int i=0; i<opt.N; ++i) {
            const double t = double(i)/double(opt.N-1);
            const double Ecm = opt.E0 + t*(opt.E1 - opt.E0);
            results[i] = evaluate_F3iso_one_energy(Ecm, nnP_vec, spec.irrep, par, opt.debug);
        }

        std::ofstream out(outfile);
        out << std::setprecision(17);
        out << "# v29g F3iso scan\n";
        out << "# input_file " << input_file << "\n";
        out << "# mom_label " << opt.mom_label << " nnP " << nnP_vec[0] << " " << nnP_vec[1] << " " << nnP_vec[2]
            << " irrep " << spec.irrep << " irrep_tag " << spec.irrep_tag << "\n";
        out << "# F3iso = norm_vec.transpose() * F3 * norm_vec\n";
        out << "# norm_vec = [1 repeated A=plm_config[0].size(), 1/sqrt(2) repeated B=klm_config[0].size()]\n";
        out << "# columns:\n";
        out << "# i Ecm En A B total_dim vdim success "
            << "F3iso_re F3iso_im F3iso_abs "
            << "detF3_re detF3_im detF3_abs "
            << "detProjF3_re detProjF3_im detProjF3_abs error\n";

        int n_success = 0;
        for (int i=0; i<opt.N; ++i) {
            const auto& r = results[i];
            if (r.success) ++n_success;
            out << i << ' ' << r.Ecm << ' ' << r.En << ' '
                << r.dim1 << ' ' << r.dim2 << ' ' << r.total_dim << ' ' << r.vdim << ' '
                << (r.success ? 1 : 0) << ' '
                << r.F3iso.real() << ' ' << r.F3iso.imag() << ' ' << std::abs(r.F3iso) << ' '
                << r.detF3.real() << ' ' << r.detF3.imag() << ' ' << std::abs(r.detF3) << ' '
                << r.detProjF3.real() << ' ' << r.detProjF3.imag() << ' ' << std::abs(r.detProjF3) << ' ';
            if (r.error.empty()) out << "OK";
            else {
                std::string e = r.error;
                for (char& c: e) if (std::isspace((unsigned char)c)) c = '_';
                out << e;
            }
            out << '\n';
        }

        std::ofstream meta(metafile);
        meta << std::setprecision(17);
        meta << "version = v29g\n";
        meta << "input_file = " << input_file << "\n";
        meta << "output_file = " << outfile << "\n";
        meta << "mom_label = " << opt.mom_label << "\n";
        meta << "irrep = " << spec.irrep << "\n";
        meta << "irrep_tag = " << spec.irrep_tag << "\n";
        meta << "E0 = " << opt.E0 << "\n";
        meta << "E1 = " << opt.E1 << "\n";
        meta << "N = " << opt.N << "\n";
        meta << "threads = " << opt.threads << "\n";
        meta << "success_points = " << n_success << "\n";
        meta << "total_points = " << opt.N << "\n";
        meta << "definition = F3iso = <1 1/sqrt(2)|F3|1 1/sqrt(2)>\n";
        meta << "det_projected_definition = det(Vsel^dagger F3 Vsel)\n";

        std::cout << "[v29g] wrote " << outfile << "\n";
        std::cout << "[v29g] wrote " << metafile << "\n";
        std::cout << "[v29g] success points: " << n_success << " / " << opt.N << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[v29g:error] " << e.what() << "\n";
        return 2;
    }
    return 0;
}
