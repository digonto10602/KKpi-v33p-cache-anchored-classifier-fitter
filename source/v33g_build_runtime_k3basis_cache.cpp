#define V32W_NO_MAIN
#include "qc_fitter_norm_refine_v2.cpp"
#undef V32W_NO_MAIN

#include "v33g_runtime_k3basis_cache.hpp"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <set>

namespace fs = std::filesystem;
using namespace k3df_fit_v32f;

namespace {

static std::string trim(std::string s) {
    while(!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    while(!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    return s;
}

static std::string strip_comment(std::string s) {
    auto p = s.find('#');
    if(p != std::string::npos) s = s.substr(0, p);
    return s;
}

static std::string ltag(double L) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(6) << L;
    std::string s = os.str();
    while(!s.empty() && s.back() == '0') s.pop_back();
    if(!s.empty() && s.back() == '.') s.pop_back();
    for(char& c : s) if(c == '.') c = 'p';
    return s;
}

static std::string xi_tag(double xi) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(3) << xi;
    std::string s = os.str();
    while(!s.empty() && s.back() == '0') s.pop_back();
    if(!s.empty() && s.back() == '.') s.pop_back();
    for(char& c : s) if(c == '.') c = 'p';
    return s;
}

static std::string key_string(double L, const std::string& lab) {
    return std::string("L") + ltag(L) + "_" + lab;
}

static std::vector<double> parse_doubles(const std::string& s) {
    std::vector<double> v;
    std::string t = s;
    for(char& c : t) if(c == ',') c = ' ';
    std::istringstream is(t);
    double x;
    while(is >> x) v.push_back(x);
    return v;
}

static std::vector<std::string> parse_words(const std::string& s) {
    std::string t = s;
    for(char& c : t) if(c == ',') c = ' ';
    std::istringstream is(t);
    std::vector<std::string> v;
    std::string x;
    while(is >> x) v.push_back(x);
    return v;
}

static std::string internal_alias(std::string lab) {
    if(lab == "001_A2") return "100_A2";
    if(lab == "010_A2") return "100_A2";
    if(lab == "100_A2") return "100_A2";
    if(lab == "011_A2") return "110_A2";
    if(lab == "101_A2") return "110_A2";
    if(lab == "110_A2") return "110_A2";
    if(lab == "002_A2") return "200_A2";
    if(lab == "020_A2") return "200_A2";
    if(lab == "200_A2") return "200_A2";
    if(lab == "000_A1g") return "000_A1p";
    if(lab == "000_A1u") return "000_A1m";
    return lab;
}

struct BuildConfig {
    FitSettings base;
    std::vector<double> Lvalues;
    std::map<double, std::vector<std::string>> irreps_by_L;
    std::string coarse_cache_root = "/media/digonto/Data/F3inv_cache";
    std::string v33g_runtime_cache_root = "cache/v33g_runtime_k3basis";
    std::map<std::pair<double, std::string>, std::string> explicit_coarse;
    int gpu_coarseN = 20000;
    double gpu_Ecm_min = 0.2631;
    double gpu_Ecm_max = 0.36;
    bool force_rebuild = false;
};

static std::map<std::pair<double, std::string>, std::string> parse_cache_block(const std::vector<std::string>& lines, const std::string& begin, const std::string& end) {
    std::map<std::pair<double, std::string>, std::string> out;
    bool active = false;
    for(std::string line : lines) {
        line = trim(strip_comment(line));
        if(line.empty()) continue;
        if(line == begin) { active = true; continue; }
        if(line == end) { active = false; continue; }
        if(!active) continue;
        std::istringstream is(line);
        double L;
        std::string lab, path;
        if(!(is >> L >> lab >> path)) throw std::runtime_error("Bad cache block line: " + line);
        out[{L, internal_alias(lab)}] = path;
    }
    return out;
}

static BuildConfig read_build_config(const std::string& cfgpath) {
    auto kv = v32w::read_kv(cfgpath);
    std::vector<std::string> raw_lines;
    {
        std::ifstream in(cfgpath);
        if(!in) throw std::runtime_error("cannot open config " + cfgpath);
        std::string line;
        while(std::getline(in, line)) raw_lines.push_back(line);
    }
    BuildConfig c;
    c.base = v32w::settings_from_config(kv);
    c.Lvalues = parse_doubles(v32w::gs(kv, "Lbyas_values", std::to_string(c.base.Lval)));
    if(c.Lvalues.empty()) c.Lvalues.push_back(c.base.Lval);
    c.coarse_cache_root = v32w::gs(kv, "coarse_cache_root", c.coarse_cache_root);
    c.v33g_runtime_cache_root = v32w::gs(kv, "v33g_runtime_cache_root", c.v33g_runtime_cache_root);
    c.gpu_coarseN = v32w::gi(kv, "gpu_cachegen_coarseN", c.base.coarseN);
    c.gpu_Ecm_min = v32w::gd(kv, "gpu_cachegen_Ecm_min", 0.2631);
    c.gpu_Ecm_max = v32w::gd(kv, "gpu_cachegen_Ecm_max", 0.36);
    c.force_rebuild = v32w::gi(kv, "force_rebuild", 0) != 0;
    const std::string global_irreps = v32w::gs(kv, "list_of_mom", "000_A1m 100_A2 110_A2 111_A2 200_A2");
    for(double L : c.Lvalues) {
        const std::string key = std::string("irreps_L") + ltag(L);
        auto labs = parse_words(v32w::gs(kv, key, global_irreps));
        for(auto& lab : labs) lab = internal_alias(lab);
        c.irreps_by_L[L] = labs;
    }
    c.explicit_coarse = parse_cache_block(raw_lines, "coarse_cache_file_list", "end_coarse_cache_file_list");
    return c;
}

static FitSettings settings_for_block(const BuildConfig& cfg, double L, const std::string& irrep) {
    FitSettings s = cfg.base;
    s.Lval = L;
    s.list_of_mom = {irrep};
    s.scan_E0 = cfg.gpu_Ecm_min;
    s.scan_E1 = cfg.gpu_Ecm_max;
    s.coarseN = cfg.gpu_coarseN;
    return s;
}

static std::string coarse_path_for(const BuildConfig& cfg, double L, const std::string& irrep) {
    auto it = cfg.explicit_coarse.find({L, irrep});
    if(it != cfg.explicit_coarse.end()) return it->second;
    fs::path dir = fs::path(cfg.coarse_cache_root) / (std::string("v32zu_Lbyas") + ltag(L) + "_gpu_cache");
    const std::string file = std::string("v32zu_Lbyas") + ltag(L) + "_xi" + xi_tag(cfg.base.xival) + "_irrep" + irrep + "_coarse" + std::to_string(cfg.gpu_coarseN) + "_F3inv_Vsel_gpu.bin";
    return (dir / file).string();
}

static std::string runtime_path_for(const BuildConfig& cfg, double L, const std::string& irrep) {
    const std::string file = std::string("v33g_Lbyas") + ltag(L) + "_xi" + xi_tag(cfg.base.xival) + "_irrep" + irrep + "_coarse" + std::to_string(cfg.gpu_coarseN) + "_runtime_K3basis.bin";
    return (fs::path(cfg.v33g_runtime_cache_root) / file).string();
}

static bool file_exists(const std::string& p) { return !p.empty() && fs::exists(fs::path(p)); }

static void read_exact(std::ifstream& is, void* p, std::size_t n) {
    is.read(reinterpret_cast<char*>(p), static_cast<std::streamsize>(n));
    if(!is) throw std::runtime_error("GPU cache read failed/truncated");
}

template<class T> static T read_scalar(std::ifstream& is) {
    T x{};
    read_exact(is, &x, sizeof(T));
    return x;
}

static IrrepCache load_gpu_coarse_cache_one(const std::string& path, const FitSettings& sblock, const PhysicsParams& par, const std::string& irrep) {
    std::ifstream is(path, std::ios::binary);
    if(!is) throw std::runtime_error("Could not open GPU coarse cache: " + path);
    std::string header;
    std::getline(is, header);
    if(header != "KKPI_F3INV_VSEL_GPU_V32ZT_EXACT_V32Y 1") {
        throw std::runtime_error("Bad GPU cache header in " + path + ": " + header);
    }
    const std::uint64_t rec_magic = 0x5653325a4f524543ULL;
    IrrepCache ic;
    ic.label = irrep;
    ic.spec = parse_label(irrep);
    while(true) {
        int c = is.peek();
        if(c == EOF) break;
        std::uint64_t magic = 0;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if(!is) break;
        if(magic != rec_magic) throw std::runtime_error("Bad GPU record magic in " + path);
        std::int32_t grid_i = read_scalar<std::int32_t>(is);
        std::int32_t dim1   = read_scalar<std::int32_t>(is);
        std::int32_t dim2   = read_scalar<std::int32_t>(is);
        std::int32_t n      = read_scalar<std::int32_t>(is);
        std::int32_t vdim   = read_scalar<std::int32_t>(is);
        double Ecm = read_scalar<double>(is);
        double En  = read_scalar<double>(is);
        (void)dim1;
        (void)dim2;
        if(n < 0 || vdim < 0) throw std::runtime_error("Negative dimensions in GPU cache " + path);
        Eigen::MatrixXcd F3inv(n,n), Vsel(n,vdim);
        for(int col = 0; col < n; ++col) for(int row = 0; row < n; ++row) {
            const double re = read_scalar<double>(is), im = read_scalar<double>(is);
            F3inv(row,col) = comp(re,im);
        }
        for(int col = 0; col < vdim; ++col) for(int row = 0; row < n; ++row) {
            const double re = read_scalar<double>(is), im = read_scalar<double>(is);
            Vsel(row,col) = comp(re,im);
        }
        ProjectedQCCacheEntry e;
        e.label = irrep;
        e.spec = ic.spec;
        e.i = grid_i;
        e.Ecm = Ecm;
        e.En = En;
        e.total_dim = n;
        e.proj_dim = vdim;
        e.F3inv_full = std::move(F3inv);
        e.Vsel = std::move(Vsel);
        try {
            const std::vector<int> nnP_vec = {ic.spec.nnP[0], ic.spec.nnP[1], ic.spec.nnP[2]};
            const comp pi = std::acos(-1.0);
            const double L = par.L();
            const comp twopibyL = comp(2.0,0.0) * pi / comp(L,0.0);
            std::vector<comp> total_P(3);
            for(int a = 0; a < 3; ++a) total_P[a] = twopibyL * double(nnP_vec[a]);
            const comp Ecm_c(Ecm,0.0);
            const comp En_c = Ecm_to_E(Ecm_c, total_P);
            e.En_c = En_c;
            e.En = En_c.real();
            e.total_P = total_P;
            std::vector<std::vector<comp>> plm_config(5), klm_config(5);
            std::vector<std::vector<int>> np_config(5), nk_config(5);
            config_maker_4_momentum_first(plm_config, np_config, par.waves_vec_1, En_c, total_P, par.atmK, par.atmK, par.atmpi, L, par.epsilon_h, par.max_shell_num, par.tolerance);
            config_maker_4_momentum_first(klm_config, nk_config, par.waves_vec_2, En_c, total_P, par.atmpi, par.atmK, par.atmK, L, par.epsilon_h, par.max_shell_num, par.tolerance);
            e.plm_config = plm_config;
            e.klm_config = klm_config;
            const int expectedN = int(plm_config[0].size() + klm_config[0].size());
            if(expectedN != n) {
                std::ostringstream os;
                os << "GPU cache dimension mismatch for " << path << " row " << grid_i << ": gpu_n=" << n << " cpu_config_N=" << expectedN;
                throw std::runtime_error(os.str());
            }
            if(vdim <= 0) {
                e.success = 0;
                e.error = "ZERO_PROJECTED_DIM_FROM_GPU";
            } else {
                e.F3inv_proj = e.Vsel.adjoint() * e.F3inv_full * e.Vsel;
                e.success = (e.F3inv_full.allFinite() && e.Vsel.allFinite() && e.F3inv_proj.allFinite()) ? 1 : 0;
                e.error = e.success ? "OK_GPU_COARSE" : "NONFINITE_GPU_CACHE";
                if(e.success) precompute_projected_k3_basis(e, par);
            }
        } catch(const std::exception& ex) {
            e.success = 0;
            e.error = ex.what();
        }
        ic.grid.push_back(std::move(e));
    }
    std::sort(ic.grid.begin(), ic.grid.end(), [](const auto& a, const auto& b) { return a.Ecm < b.Ecm; });
    std::cout << "[gpu-cache] loaded " << path << " irrep=" << irrep << " rows=" << ic.grid.size() << "\n";
    if((int)ic.grid.size() != sblock.coarseN) std::cout << "[gpu-cache-warning] row count != coarseN for " << irrep << "\n";
    return ic;
}

static v33g_runtime_k3basis::RuntimeCacheMeta make_meta(const BuildConfig& cfg, double L, const std::string& irrep, const std::string& raw_path, const IrrepCache& slim) {
    v33g_runtime_k3basis::RuntimeCacheMeta m;
    m.Lbyas = L;
    m.xi = cfg.base.xival;
    m.irrep = irrep;
    m.coarseN = cfg.gpu_coarseN;
    m.Ecm_min = cfg.gpu_Ecm_min;
    m.Ecm_max = cfg.gpu_Ecm_max;
    m.rows = slim.grid.size();
    m.atmK = cfg.base.atmK;
    m.atmpi = cfg.base.atmpi;
    m.eta_1 = cfg.base.eta_1;
    m.eta_2 = cfg.base.eta_2;
    m.alpha = cfg.base.alpha;
    m.epsilon_h = cfg.base.epsilon_h;
    m.max_shell_num = cfg.base.max_shell_num;
    m.tolerance = cfg.base.tolerance;
    m.parity = cfg.base.parity;
    m.eig_tol = cfg.base.eig_tol;
    m.norm_tol = cfg.base.norm_tol;
    m.proj_tol = cfg.base.proj_tol;
    m.waves_vec_1 = cfg.base.waves_vec_1;
    m.waves_vec_2 = cfg.base.waves_vec_2;
    m.scatter1_00 = cfg.base.scatter_params_1[0][0];
    m.scatter1_10 = cfg.base.scatter_params_1[1][0];
    m.scatter2_00 = cfg.base.scatter_params_2[0][0];
    m.source_raw_cache = raw_path;
    m.source_raw_cache_file_size = static_cast<std::uint64_t>(fs::file_size(raw_path));
    m.source_raw_cache_mtime = std::chrono::duration_cast<std::chrono::duration<double>>(fs::last_write_time(raw_path).time_since_epoch()).count();
    m.created_by = "v33g_build_runtime_k3basis_cache.cpp";
    return m;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::string cfgpath = (argc > 1 ? std::string(argv[1]) : std::string("configs/config_v33g_build_runtime_cache.in"));
        const BuildConfig cfg = read_build_config(cfgpath);
        fs::create_directories(cfg.v33g_runtime_cache_root);
        const auto t0 = std::chrono::steady_clock::now();
        std::size_t total_rows = 0;

        for(double L : cfg.Lvalues) {
            const auto it = cfg.irreps_by_L.find(L);
            if(it == cfg.irreps_by_L.end()) continue;
            for(const auto& irrep : it->second) {
                const FitSettings settings = settings_for_block(cfg, L, irrep);
                const PhysicsParams par = make_base_physics(settings);
                const std::string raw_path = coarse_path_for(cfg, L, irrep);
                if(!file_exists(raw_path)) throw std::runtime_error("missing raw v33e cache: " + raw_path);
                if(!file_exists(raw_path + ".meta.json")) throw std::runtime_error("missing raw v33e cache meta: " + raw_path + ".meta.json");
                const std::string runtime_path = runtime_path_for(cfg, L, irrep);
                if(file_exists(runtime_path) && !cfg.force_rebuild) {
                    std::cout << "[v33g-build] skip existing " << runtime_path << "\n";
                    continue;
                }

                const auto block_t0 = std::chrono::steady_clock::now();
                auto coarse = load_gpu_coarse_cache_one(raw_path, settings, par, irrep);
                IrrepCache slim;
                slim.label = irrep;
                slim.spec = parse_label(irrep);
                slim.grid.reserve(coarse.grid.size());
                for(auto& e : coarse.grid) {
                    Eigen::MatrixXcd b0, b1, bB, bE;
                    if(e.success) {
                        b0 = e.K3_proj_basis[0];
                        b1 = e.K3_proj_basis[1];
                        bB = e.K3_proj_basis[2];
                        bE = e.K3_proj_basis[3];
                    }
                    slim.grid.push_back(v33g_runtime_k3basis::to_projected_entry_v33g(
                        e.i, e.Ecm, e.En, e.success, e.total_dim, e.proj_dim,
                        e.F3inv_proj,
                        b0, b1, bB, bE,
                        irrep
                    ));
                }
                v33g_runtime_k3basis::write_runtime_cache(runtime_path, slim, make_meta(cfg, L, irrep, raw_path, slim));
                const auto block_t1 = std::chrono::steady_clock::now();
                const double sec = std::chrono::duration<double>(block_t1 - block_t0).count();
                total_rows += slim.grid.size();
                std::cout << "[v33g-build] " << key_string(L, irrep) << " rows=" << slim.grid.size() << " sec=" << std::setprecision(17) << sec << "\n";
            }
        }

        const auto t1 = std::chrono::steady_clock::now();
        std::cout << "[v33g-build] total_rows=" << total_rows << " elapsed_sec=" << std::setprecision(17)
                  << std::chrono::duration<double>(t1 - t0).count() << "\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "[v33g-build-error] " << e.what() << "\n";
        return 1;
    }
}
