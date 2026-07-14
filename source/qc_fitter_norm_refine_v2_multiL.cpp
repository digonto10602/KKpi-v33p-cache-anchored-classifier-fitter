// v32x multi-Lbyas QC fitter
//
// This translation unit intentionally reuses the working v32w normalized
// determinant / fixed-coarse / digonto_classifier_v2 implementation by
// including the v32w source with its main symbol renamed.  The new main below
// adds:
//   * combined Lbyas fits, e.g. L=20 and L=24 together;
//   * jackknife En_lab input from /home/digonto/Codes/KKpi_I2/spectrum/Ecm_data/data;
//   * full covariance within each L, block-diagonal covariance across L;
//   * per-(L,irrep) v32zu GPU coarse cache loading/generation;
//   * CPU-only refined-point generation and separate refined cache files.

// Reuse v32w implementation as a library.  The v32w source now honors
// V32W_NO_MAIN so its standalone single-L main is not compiled here.
#define V32W_NO_MAIN
#include "qc_fitter_norm_refine_v2.cpp"
#undef V32W_NO_MAIN
#include "v33g_runtime_k3basis_cache.hpp"

#include <cerrno>
#include <cstdint>
#include <chrono>
#include <cstdlib>
#include <future>
#include <regex>
#include <set>
#include <thread>
#include <sys/resource.h>

namespace v32x_multiL {
using namespace k3df_fit_v32f;
namespace fs = std::filesystem;

static std::string trim2(std::string s) {
    while(!s.empty() && std::isspace((unsigned char)s.front())) s.erase(s.begin());
    while(!s.empty() && std::isspace((unsigned char)s.back())) s.pop_back();
    return s;
}
static std::string strip_comment2(std::string s) {
    auto p=s.find('#');
    if(p!=std::string::npos) s=s.substr(0,p);
    return s;
}
static std::string ltag(double L) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(6) << L;
    std::string s=os.str();
    while(!s.empty() && s.back()=='0') s.pop_back();
    if(!s.empty() && s.back()=='.') s.pop_back();
    for(char& c:s) if(c=='.') c='p';
    return s;
}
static std::string xi_tag(double xi) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(3) << xi;
    std::string s=os.str();
    while(!s.empty() && s.back()=='0') s.pop_back();
    if(!s.empty() && s.back()=='.') s.pop_back();
    for(char& c:s) if(c=='.') c='p';
    return s;
}
static std::vector<double> parse_doubles(const std::string& s) {
    std::vector<double> v;
    std::string t=s; for(char& c:t) if(c==',') c=' ';
    std::istringstream is(t); double x; while(is>>x) v.push_back(x);
    return v;
}
static std::vector<std::string> parse_words(const std::string& s) { return v32w::sws(s); }
static bool file_exists(const std::string& p) { return !p.empty() && fs::exists(fs::path(p)); }

static std::string internal_alias(std::string lab) {
    if(lab=="001_A2") return "100_A2";
    if(lab=="010_A2") return "100_A2";
    if(lab=="100_A2") return "100_A2";
    if(lab=="011_A2") return "110_A2";
    if(lab=="101_A2") return "110_A2";
    if(lab=="110_A2") return "110_A2";
    if(lab=="002_A2") return "200_A2";
    if(lab=="020_A2") return "200_A2";
    if(lab=="200_A2") return "200_A2";
    if(lab=="000_A1g") return "000_A1p";
    if(lab=="000_A1u") return "000_A1m";
    return lab;
}

struct CacheKey {
    double L = 0.0;
    std::string label;
    bool operator<(const CacheKey& o) const {
        if(std::abs(L-o.L)>1e-12) return L < o.L;
        return label < o.label;
    }
};
static std::string key_string(double L, const std::string& lab) {
    return std::string("L") + ltag(L) + "_" + lab;
}

struct MultiTarget {
    TargetLevel t;
    std::string file_irrep;
    std::string internal_irrep;
    fs::path jack_file;
    int level_index_in_block = -1;
    double lab_mean = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> samples_ecm;
};
struct BlockInfo {
    double L = 0.0;
    std::string internal_irrep;
    std::string file_irrep_hint;
    std::vector<int> target_indices;
};
struct MultiConfig {
    FitSettings base;
    std::vector<double> Lvalues;
    std::map<double,std::vector<std::string>> irreps_by_L;
    std::string classifier_mode = "v2_refine";
    std::string cache_backend = "raw_v33e";
    std::string jack_dir;
    std::string jack_energy_type = "En_lab";
    int jack_skip_header_lines = 1;
    int jack_energy_column = 2; // 1-based; user said: index value
    double energy_cutoff = 0.335;
    double const_norm_power = 6.0;
    int ninside = 10;
    int maxdepth = 8;
    double zratio = 0.80;
    std::string coarse_cache_root = "/media/digonto/Data/F3inv_cache";
    std::string v33g_runtime_cache_root = "cache/v33g_runtime_k3basis";
    std::map<CacheKey,std::string> explicit_coarse;
    bool auto_build_missing_coarse = true;
    bool require_existing_v33g_runtime_cache = true;
    bool build_v33g_runtime_if_missing = false;
    std::string gpu_cachegen_root = "external/v32zu_gpu_cachegen";
    double gpu_Ecm_min = 0.26301;
    double gpu_Ecm_max = 0.36;
    int gpu_coarseN = 10000;
    std::string refined_cache_dir = "/media/digonto/Data/F3inv_cache/v32x_multiL_refined_cache";
    std::string refined_cache_prefix = "v32x_multiL";
    bool save_refined_cache = true;
    bool use_gpu_cache_meta_validation = true;
    int benchmark_repeat = 10;
    int benchmark_warmup = 2;
    // v33f smoke/benchmark controls.  max_total_lattice_levels=5 gives the
    // requested first-stage 000_A1m five-level lattice-spectrum test.
    int max_total_lattice_levels = 0;
    int max_lattice_levels_per_block = 0;
    int max_fcn_evals = 0;
    std::string root_search_mode = "full_scan";
    std::string det_backend = "cpu_openmp";
    bool fallback_full_scan = true;
    int window_half_rows = 50;
    int max_window_half_rows = 250;
    std::map<CacheKey,std::string> accepted_zeros_files;
    std::map<CacheKey,std::string> candidate_brackets_files;
    std::map<CacheKey,std::string> det_grid_files;
    std::array<bool,4> float_params{true,true,true,true};
};

static std::vector<std::string> read_raw_lines(const std::string& path) {
    std::ifstream in(path);
    if(!in) throw std::runtime_error("cannot open config "+path);
    std::vector<std::string> lines; std::string line;
    while(std::getline(in,line)) lines.push_back(line);
    return lines;
}

static std::map<CacheKey,std::string> parse_cache_block(const std::vector<std::string>& lines, const std::string& begin, const std::string& end) {
    std::map<CacheKey,std::string> out;
    bool active=false;
    for(std::string line: lines) {
        line=trim2(strip_comment2(line));
        if(line.empty()) continue;
        if(line==begin) { active=true; continue; }
        if(line==end) { active=false; continue; }
        if(!active) continue;
        std::istringstream is(line);
        double L; std::string lab,path;
        if(!(is>>L>>lab>>path)) throw std::runtime_error("Bad cache block line: "+line);
        CacheKey k{L,internal_alias(lab)};
        out[k]=path;
    }
    return out;
}

static MultiConfig multiconfig_from_config(const std::string& cfgpath) {
    auto kv = v32w::read_kv(cfgpath);
    auto raw = read_raw_lines(cfgpath);
    MultiConfig c;
    c.base = v32w::settings_from_config(kv);
    c.Lvalues = parse_doubles(v32w::gs(kv,"Lbyas_values",std::to_string(c.base.Lval)));
    if(c.Lvalues.empty()) c.Lvalues.push_back(c.base.Lval);
    c.jack_dir = v32w::gs(kv,"lattice_jackknife_dir","/home/digonto/Codes/KKpi_I2/spectrum/Ecm_data/data");
    c.jack_energy_type = v32w::gs(kv,"lattice_jack_energy_type","En_lab");
    c.jack_skip_header_lines = v32w::gi(kv,"jack_skip_header_lines",1);
    c.jack_energy_column = v32w::gi(kv,"jack_energy_column",2);
    c.classifier_mode = v32w::gs(kv,"classifier_mode",v32w::gs(kv,"digonto_classifier_mode","v2_refine"));
    c.cache_backend = v32w::gs(kv,"cache_backend","raw_v33e");
    c.energy_cutoff = v32w::gd(kv,"Ecm_cutoff",v32w::gd(kv,"energy_cutoff",0.335));
    c.const_norm_power = v32w::gd(kv,"const_norm_power",6.0);
    c.ninside = v32w::gi(kv,"v2_refine_points",10);
    c.maxdepth = v32w::gi(kv,"v2_max_split_depth",8);
    c.zratio = v32w::gd(kv,"v2_zero_ratio",0.80);
    c.coarse_cache_root = v32w::gs(kv,"coarse_cache_root","/media/digonto/Data/F3inv_cache");
    c.v33g_runtime_cache_root = v32w::gs(kv,"v33g_runtime_cache_root","cache/v33g_runtime_k3basis");
    c.auto_build_missing_coarse = v32w::gi(kv,"auto_build_missing_coarse",1)!=0;
    c.require_existing_v33g_runtime_cache = v32w::gi(kv,"require_existing_v33g_runtime_cache",1)!=0;
    c.build_v33g_runtime_if_missing = v32w::gi(kv,"build_v33g_runtime_if_missing",0)!=0;
    c.gpu_cachegen_root = v32w::gs(kv,"gpu_cachegen_root","external/v32zu_gpu_cachegen");
    c.gpu_Ecm_min = v32w::gd(kv,"gpu_cachegen_Ecm_min",0.26301);
    c.gpu_Ecm_max = v32w::gd(kv,"gpu_cachegen_Ecm_max",0.36);
    c.gpu_coarseN = v32w::gi(kv,"gpu_cachegen_coarseN",c.base.coarseN);
    c.refined_cache_dir = v32w::gs(kv,"refined_cache_dir","/media/digonto/Data/F3inv_cache/v32x_multiL_refined_cache");
    c.refined_cache_prefix = v32w::gs(kv,"refined_cache_prefix","v32x_multiL");
    c.save_refined_cache = v32w::gi(kv,"save_refined_cache",1)!=0;
    c.use_gpu_cache_meta_validation = v32w::gi(kv,"use_gpu_cache_meta_validation",1)!=0;
    c.benchmark_repeat = v32w::gi(kv,"benchmark_repeat",10);
    c.benchmark_warmup = v32w::gi(kv,"benchmark_warmup",2);
    c.max_total_lattice_levels = v32w::gi(kv,"max_total_lattice_levels",0);
    c.max_lattice_levels_per_block = v32w::gi(kv,"max_lattice_levels_per_block",0);
    c.max_fcn_evals = v32w::gi(kv,"max_fcn_evals",0);
    c.root_search_mode = v32w::gs(kv,"root_search_mode","full_scan");
    c.det_backend = v32w::gs(kv,"det_backend","cpu_openmp");
    c.fallback_full_scan = v32w::gi(kv,"fallback_full_scan",1)!=0;
    c.float_params = {
        v32w::gi(kv,"float_K3iso0",1)!=0, v32w::gi(kv,"float_K3iso1",1)!=0,
        v32w::gi(kv,"float_K3B",1)!=0, v32w::gi(kv,"float_K3E",1)!=0};
    c.window_half_rows = v32w::gi(kv,"window_half_rows",50);
    c.max_window_half_rows = v32w::gi(kv,"max_window_half_rows",250);
    c.base.energy_cutoff = c.energy_cutoff;
    c.base.lattice_energy_type = c.jack_energy_type;

    const std::string global_irreps = v32w::gs(kv,"list_of_mom","000_A1m 100_A2 110_A2 111_A2 200_A2");
    for(double L: c.Lvalues) {
        std::string key = std::string("irreps_L") + ltag(L);
        auto labs = parse_words(v32w::gs(kv,key,global_irreps));
        for(auto& lab: labs) lab = internal_alias(lab);
        c.irreps_by_L[L] = labs;
        for(const auto& lab: labs) {
            const CacheKey k{L,lab};
            const std::string prefix = "_" + key_string(L,lab);
            c.accepted_zeros_files[k] = v32w::gs(kv,"accepted_zeros_file"+prefix,"");
            c.candidate_brackets_files[k] = v32w::gs(kv,"candidate_brackets_file"+prefix,"");
            c.det_grid_files[k] = v32w::gs(kv,"det_grid_file"+prefix,"");
        }
    }
    c.explicit_coarse = parse_cache_block(raw,"coarse_cache_file_list","end_coarse_cache_file_list");
    return c;
}

static FitSettings settings_for_block(const MultiConfig& cfg, double L, const std::string& irrep) {
    FitSettings s = cfg.base;
    s.Lval = L;
    s.list_of_mom = {irrep};
    s.scan_E0 = cfg.gpu_Ecm_min;
    s.scan_E1 = cfg.gpu_Ecm_max;
    s.coarseN = cfg.gpu_coarseN;
    return s;
}
static double block_cnorm(const MultiConfig& cfg, double L) { return std::pow(L*cfg.base.xival,cfg.const_norm_power); }

static bool parse_jack_filename(const fs::path& p, double& L, std::string& file_irrep, std::string& internal_irrep, int& state) {
    const std::string name = p.filename().string();
    if(name.size() < 7 || p.extension() != ".jack") return false;
    const std::size_t us = name.find('_');
    const std::size_t npos = name.rfind("_n");
    if(us==std::string::npos || npos==std::string::npos || npos<=us) return false;
    try { L = std::stod(name.substr(0,us)); } catch(...) { return false; }
    file_irrep = name.substr(us+1, npos-us-1);
    std::string st = name.substr(npos+2);
    if(st.size()>=5 && st.substr(st.size()-5)==".jack") st = st.substr(0,st.size()-5);
    try { state = std::stoi(st); } catch(...) { return false; }
    internal_irrep = internal_alias(file_irrep);
    return true;
}
static bool contains_label(const std::vector<std::string>& v, const std::string& lab) {
    return std::find(v.begin(),v.end(),lab) != v.end();
}
static std::vector<double> read_jack_values(const std::string& path, int skip, int col1based) {
    std::ifstream in(path);
    if(!in) throw std::runtime_error("Could not open jackknife file: "+path);
    std::vector<double> vals; std::string line; int lineno=0;
    while(std::getline(in,line)) {
        ++lineno;
        if(lineno<=skip) continue;
        line = trim2(strip_comment2(line));
        if(line.empty()) continue;
        std::istringstream is(line);
        std::vector<std::string> cols; std::string x;
        while(is>>x) cols.push_back(x);
        if((int)cols.size() < col1based) continue;
        try { vals.push_back(std::stod(cols[std::size_t(col1based-1)])); }
        catch(...) { throw std::runtime_error("Bad numeric value in "+path+" line "+std::to_string(lineno)); }
    }
    if(vals.empty()) throw std::runtime_error("No jackknife samples found in "+path);
    return vals;
}
static double mean_vec(const std::vector<double>& v) {
    double s=0.0; for(double x:v) s+=x; return s/double(v.size());
}
static std::vector<double> convert_samples_to_ecm(const std::vector<double>& raw, const std::string& etype, double Lbyas, double xi, const std::string& internal_irrep) {
    MomentumIrrepSpec spec = parse_label(internal_irrep);
    const double n2 = double(spec.nnP[0]*spec.nnP[0] + spec.nnP[1]*spec.nnP[1] + spec.nnP[2]*spec.nnP[2]);
    const double P = 2.0*std::acos(-1.0)*std::sqrt(n2)/(xi*Lbyas);
    std::vector<double> out; out.reserve(raw.size());
    for(double E: raw) {
        if(etype=="En_lab" || etype=="Elab" || etype=="E_lab") {
            double arg=E*E-P*P;
            if(!(arg>0.0)) out.push_back(std::numeric_limits<double>::quiet_NaN());
            else out.push_back(std::sqrt(arg));
        } else {
            out.push_back(E);
        }
    }
    return out;
}

static std::tuple<std::vector<MultiTarget>,std::vector<TargetLevel>,MatrixD,MatrixD,std::vector<BlockInfo>>
load_multil_jack_targets(const MultiConfig& cfg) {
    if(!fs::exists(cfg.jack_dir)) throw std::runtime_error("lattice_jackknife_dir does not exist: "+cfg.jack_dir);
    struct Rec {
        double L;
        std::string file_irrep, internal_irrep;
        int state;
        fs::path path;
        std::vector<double> samples_ecm;
        double lab_mean;
        double ecm_mean;
    };
    std::vector<Rec> recs;
    std::set<std::string> allowedL;
    for(double L: cfg.Lvalues) allowedL.insert(ltag(L));
    for(const auto& de: fs::directory_iterator(cfg.jack_dir)) {
        if(!de.is_regular_file()) continue;
        double L; std::string fir, iir; int st;
        if(!parse_jack_filename(de.path(),L,fir,iir,st)) continue;
        bool L_ok=false; double L_match=0;
        for(double Lwant: cfg.Lvalues) if(std::abs(L-Lwant)<1e-9) { L_ok=true; L_match=Lwant; break; }
        if(!L_ok) continue;
        auto it = cfg.irreps_by_L.find(L_match);
        if(it==cfg.irreps_by_L.end() || !contains_label(it->second,iir)) continue;
        auto raw = read_jack_values(de.path().string(),cfg.jack_skip_header_lines,cfg.jack_energy_column);
        auto ecm = convert_samples_to_ecm(raw,cfg.jack_energy_type,L_match,cfg.base.xival,iir);
        for(double x: ecm) if(!std::isfinite(x)) throw std::runtime_error("Nonfinite converted Ecm in "+de.path().string());
        double lab_mean = mean_vec(raw);
        double ecm_mean = mean_vec(ecm);
        if(ecm_mean <= cfg.energy_cutoff) recs.push_back({L_match,fir,iir,st,de.path(),ecm,lab_mean,ecm_mean});
    }
    std::sort(recs.begin(),recs.end(),[&](const Rec& a,const Rec& b){
        if(std::abs(a.L-b.L)>1e-12) return a.L < b.L;
        const auto& va = cfg.irreps_by_L.at(a.L);
        auto ia = std::find(va.begin(),va.end(),a.internal_irrep);
        auto ib = std::find(va.begin(),va.end(),b.internal_irrep);
        int pa = (ia==va.end()?999:int(ia-va.begin()));
        int pb = (ib==va.end()?999:int(ib-va.begin()));
        if(pa!=pb) return pa<pb;
        if(a.state!=b.state) return a.state<b.state;
        return a.file_irrep < b.file_irrep;
    });
    if(recs.empty()) throw std::runtime_error("No jackknife states survived L/irrep selection and Ecm cutoff.");

    if(cfg.max_lattice_levels_per_block > 0) {
        std::map<CacheKey,int> kept;
        std::vector<Rec> filtered;
        for(const auto& r: recs) {
            CacheKey k{r.L,r.internal_irrep};
            if(kept[k] < cfg.max_lattice_levels_per_block) {
                filtered.push_back(r);
                kept[k]++;
            }
        }
        recs.swap(filtered);
    }
    if(cfg.max_total_lattice_levels > 0 && (int)recs.size() > cfg.max_total_lattice_levels) {
        recs.resize((std::size_t)cfg.max_total_lattice_levels);
    }
    if(recs.empty()) throw std::runtime_error("No jackknife states survived v33f level truncation.");

    std::vector<MultiTarget> mt;
    std::vector<TargetLevel> targets;
    std::map<CacheKey,int> level_count;
    for(const auto& r: recs) {
        MultiTarget m;
        m.file_irrep = r.file_irrep;
        m.internal_irrep = r.internal_irrep;
        m.jack_file = r.path;
        m.samples_ecm = r.samples_ecm;
        CacheKey k{r.L,r.internal_irrep};
        m.level_index_in_block = level_count[k]++;
        m.t.label = r.internal_irrep;
        m.t.state = r.state;
        m.t.Ecm = r.ecm_mean;
        m.t.E_read = r.lab_mean;
        m.t.lattice_energy_type = cfg.jack_energy_type;
        m.t.shifted_from_lab = (cfg.jack_energy_type=="En_lab" || cfg.jack_energy_type=="Elab" || cfg.jack_energy_type=="E_lab") ? 1 : 0;
        m.t.nP = parse_label(r.internal_irrep).nnP;
        m.t.Lbyas = r.L;
        mt.push_back(std::move(m));
    }
    const int N = int(mt.size());
    MatrixD cov = MatrixD::Zero(N,N);
    // block covariance within same L only; cross-L is zero by construction.
    for(int i=0;i<N;++i) {
        for(int j=0;j<N;++j) {
            if(std::abs(mt[std::size_t(i)].t.Lbyas - mt[std::size_t(j)].t.Lbyas)>1e-12) continue;
            const auto& ai = mt[std::size_t(i)].samples_ecm;
            const auto& aj = mt[std::size_t(j)].samples_ecm;
            if(ai.size()!=aj.size()) {
                std::ostringstream os; os << "Jackknife sample count mismatch within L=" << mt[std::size_t(i)].t.Lbyas;
                throw std::runtime_error(os.str());
            }
            const int Nj = int(ai.size());
            const double mi = mean_vec(ai), mj = mean_vec(aj);
            double s=0.0; for(int a=0;a<Nj;++a) s += (ai[std::size_t(a)]-mi)*(aj[std::size_t(a)]-mj);
            cov(i,j) = (double(Nj-1)/double(Nj))*s;
        }
    }
    for(int i=0;i<N;++i) {
        mt[std::size_t(i)].t.err = (cov(i,i)>0.0) ? std::sqrt(cov(i,i)) : 1.0;
        mt[std::size_t(i)].t.err_read = mt[std::size_t(i)].t.err;
        targets.push_back(mt[std::size_t(i)].t);
    }
    MatrixD corr = covariance_to_correlation_v32f(cov);

    std::map<CacheKey,BlockInfo> bm;
    for(int i=0;i<N;++i) {
        CacheKey k{mt[std::size_t(i)].t.Lbyas, mt[std::size_t(i)].internal_irrep};
        auto& b = bm[k];
        b.L = k.L; b.internal_irrep = k.label; b.file_irrep_hint = mt[std::size_t(i)].file_irrep;
        b.target_indices.push_back(i);
    }
    std::vector<BlockInfo> blocks;
    for(auto& kv: bm) blocks.push_back(std::move(kv.second));
    std::cout << "[multiL-data] selected targets=" << N << " from jack_dir=" << cfg.jack_dir << "\n";
    for(const auto& b: blocks) std::cout << "[multiL-data] " << key_string(b.L,b.internal_irrep) << " nlevels=" << b.target_indices.size() << "\n";
    return {mt,targets,cov,corr,blocks};
}

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for(char ch : s) {
        switch(ch) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += ch; break;
        }
    }
    return out;
}

static void dump_fitter_targets_v33k(
    const MultiConfig& cfg,
    const std::vector<MultiTarget>& mt,
    const std::vector<TargetLevel>& targets
) {
    const fs::path out_dir = fs::path("diagnostics/v33k/fitter_targets");
    fs::create_directories(out_dir);
    const fs::path csv_path = out_dir / "fitter_targets.csv";
    const fs::path json_path = out_dir / "fitter_targets.json";
    const fs::path report_path = fs::path("reports/v33k_fitter_target_dump.md");

    std::ofstream csv(csv_path);
    csv << std::setprecision(17);
    csv << "Lbyas,irrep_canonical,irrep_original,jack_file,level_index,E_lab,Ecm,Ecm_error_lower_or_sym,Ecm_error_upper_or_sym,Ecm_cutoff_used,selected_under_cutoff,momentum_label,canonical_momentum_label\n";
    for(std::size_t i=0;i<targets.size();++i) {
        const auto& t = targets[i];
        const auto& m = mt[i];
        csv << t.Lbyas << ','
            << m.internal_irrep << ','
            << m.file_irrep << ','
            << json_escape(m.jack_file.string()) << ','
            << m.level_index_in_block << ','
            << t.E_read << ','
            << t.Ecm << ','
            << t.err << ','
            << t.err << ','
            << cfg.energy_cutoff << ','
            << 1 << ','
            << m.file_irrep << ','
            << m.internal_irrep << "\n";
    }

    std::ofstream js(json_path);
    js << std::setprecision(17);
    js << "[\n";
    for(std::size_t i=0;i<targets.size();++i) {
        const auto& t = targets[i];
        const auto& m = mt[i];
        js << "  {\n";
        js << "    \"Lbyas\": " << t.Lbyas << ",\n";
        js << "    \"irrep_canonical\": \"" << json_escape(m.internal_irrep) << "\",\n";
        js << "    \"irrep_original\": \"" << json_escape(m.file_irrep) << "\",\n";
        js << "    \"jack_file\": \"" << json_escape(m.jack_file.string()) << "\",\n";
        js << "    \"level_index\": " << m.level_index_in_block << ",\n";
        js << "    \"E_lab\": " << t.E_read << ",\n";
        js << "    \"Ecm\": " << t.Ecm << ",\n";
        js << "    \"Ecm_error_lower_or_sym\": " << t.err << ",\n";
        js << "    \"Ecm_error_upper_or_sym\": " << t.err << ",\n";
        js << "    \"Ecm_cutoff_used\": " << cfg.energy_cutoff << ",\n";
        js << "    \"selected_under_cutoff\": true,\n";
        js << "    \"momentum_label\": \"" << json_escape(m.file_irrep) << "\",\n";
        js << "    \"canonical_momentum_label\": \"" << json_escape(m.internal_irrep) << "\"\n";
        js << "  }" << (i + 1 < targets.size() ? "," : "") << "\n";
    }
    js << "]\n";

    std::ofstream rpt(report_path);
    rpt << "# v33k fitter target dump\n\n";
    rpt << "selected targets: " << targets.size() << "\n\n";
    rpt << "| Lbyas | canonical | original | level_index | E_lab | Ecm | err | selected |\n";
    rpt << "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |\n";
    for(std::size_t i=0;i<targets.size();++i) {
        const auto& t = targets[i];
        const auto& m = mt[i];
        rpt << "| " << t.Lbyas << " | " << m.internal_irrep << " | " << m.file_irrep << " | "
            << m.level_index_in_block << " | " << t.E_read << " | " << t.Ecm << " | " << t.err << " | 1 |\n";
    }
    rpt << "\n## Files\n\n";
    rpt << "- " << csv_path.string() << "\n";
    rpt << "- " << json_path.string() << "\n";
}

static std::string default_gpu_cache_path(const MultiConfig& cfg, double L, const std::string& irrep) {
    fs::path dir = fs::path(cfg.coarse_cache_root) / (std::string("v32zu_Lbyas") + ltag(L) + "_gpu_cache");
    std::string file = std::string("v32zu_Lbyas") + ltag(L) + "_xi" + xi_tag(cfg.base.xival) + "_irrep" + irrep + "_coarse" + std::to_string(cfg.gpu_coarseN) + "_F3inv_Vsel_gpu.bin";
    return (dir / file).string();
}
static std::string coarse_path_for(const MultiConfig& cfg, double L, const std::string& irrep) {
    CacheKey k{L,irrep};
    auto it=cfg.explicit_coarse.find(k);
    if(it!=cfg.explicit_coarse.end()) return it->second;
    return default_gpu_cache_path(cfg,L,irrep);
}
static std::string refined_path_for(const MultiConfig& cfg, double L, const std::string& irrep) {
    std::string file = cfg.refined_cache_prefix + "_Lbyas" + ltag(L) + "_xi" + xi_tag(cfg.base.xival) + "_irrep" + irrep + "_refined_F3inv_Vsel_cpu.bin";
    return (fs::path(cfg.refined_cache_dir)/file).string();
}
static std::string runtime_path_for(const MultiConfig& cfg, double L, const std::string& irrep) {
    std::string file = std::string("v33g_Lbyas") + ltag(L) + "_xi" + xi_tag(cfg.base.xival) + "_irrep" + irrep + "_coarse" + std::to_string(cfg.gpu_coarseN) + "_runtime_K3basis.bin";
    return (fs::path(cfg.v33g_runtime_cache_root) / file).string();
}
static void write_refined_meta(const std::string& path, const MultiConfig& cfg, double L, const std::string& irrep, std::size_t nrows) {
    std::ofstream m(path + ".meta.json");
    m << std::setprecision(17);
    m << "{\n";
    m << "  \"version\": \"v32x_multiL_cpu_refined_cache\",\n";
    m << "  \"cache_kind\": \"CPU refined F3inv/Vsel points; not used for coarse sign-flip scan\",\n";
    m << "  \"Lbyas\": " << L << ",\n";
    m << "  \"xi\": " << cfg.base.xival << ",\n";
    m << "  \"irrep\": \"" << irrep << "\",\n";
    m << "  \"rows\": " << nrows << ",\n";
    m << "  \"waves_vec_1\": [0,1],\n";
    m << "  \"waves_vec_2\": [0],\n";
    m << "  \"scatter1_00\": " << cfg.base.scatter_params_1[0][0] << ",\n";
    m << "  \"scatter1_10\": " << cfg.base.scatter_params_1[1][0] << ",\n";
    m << "  \"scatter2_00\": " << cfg.base.scatter_params_2[0][0] << ",\n";
    m << "  \"source\": \"generated on demand by qc_fitter_norm_refine_v2_multiL.cpp\"\n";
    m << "}\n";
}

static int run_shell(const std::string& cmd) {
    std::cout << "[shell] " << cmd << "\n";
    int rc = std::system(cmd.c_str());
    if(rc != 0) std::cout << "[shell-warning] command returned rc=" << rc << "\n";
    return rc;
}
static void ensure_gpu_coarse_cache(const MultiConfig& cfg, double L, const std::string& irrep, const std::string& path) {
    if(file_exists(path)) {
        if(cfg.use_gpu_cache_meta_validation && !file_exists(path+".meta.json")) {
            std::cout << "[coarse-cache-warning] meta sidecar missing for " << path << "\n";
        }
        return;
    }
    if(!cfg.auto_build_missing_coarse) throw std::runtime_error("Missing coarse GPU cache and auto_build_missing_coarse=0: "+path);
    fs::create_directories(fs::path(path).parent_path());
    fs::create_directories(cfg.base.output_dir);
    fs::path config_path = fs::path(cfg.base.output_dir) / (std::string("auto_gpu_cachegen_L") + ltag(L) + "_" + irrep + ".sh");
    fs::path gpu_root = fs::path(cfg.gpu_cachegen_root);
    if(!fs::exists(gpu_root)) throw std::runtime_error("gpu_cachegen_root not found: "+gpu_root.string());
    std::ofstream gc(config_path);
    gc << "# auto-generated by v32x multi-L fitter for missing GPU coarse cache\n";
    gc << "Ecm_min=" << std::setprecision(17) << cfg.gpu_Ecm_min << "\n";
    gc << "Ecm_max=" << std::setprecision(17) << cfg.gpu_Ecm_max << "\n";
    gc << "coarseN=" << cfg.gpu_coarseN << "\n";
    gc << "xi=" << cfg.base.xival << "\n";
    gc << "output_root=" << fs::path(path).parent_path().parent_path().string() << "\n";
    gc << "irreps=" << irrep << "\n";
    gc << "debug=" << cfg.base.debug << "\n";
    gc << "use_fused_f2k2=true\nvalidate_fused_f2k2=false\nuse_kinematic_precompute=true\nuse_stream_pipeline=true\n";
    gc << "max_batch_energies=0\ngpu_memory_safety_fraction=0.85\nuse_concurrent_streams=true\nconcurrent_streams_safety_fraction=0.85\nuse_multi_gpu=false\nuse_mixed_precision_f2k2=false\n";
    gc.close();
    std::ostringstream cmd;
    cmd << "V32ZU_CONFIG='" << config_path.string() << "' bash '" << (gpu_root/"scripts/run_v32zu_gpu_cachegen.sh").string() << "' " << std::setprecision(17) << L;
    int rc = run_shell(cmd.str());
    if(rc != 0 || !file_exists(path)) throw std::runtime_error("GPU coarse cache generation failed or did not create expected file: "+path);
}

static IrrepCache load_v33g_runtime_cache_one(const MultiConfig& cfg, const FitSettings& sblock, const PhysicsParams& par, const std::string& path, const std::string& irrep) {
    if(!file_exists(path)) {
        if(cfg.build_v33g_runtime_if_missing) throw std::runtime_error("runtime cache build-on-missing not implemented yet: " + path);
        if(cfg.require_existing_v33g_runtime_cache) throw std::runtime_error("Missing v33g runtime cache: " + path);
    }
    if(!file_exists(path + ".meta.json")) throw std::runtime_error("Missing v33g runtime cache meta: " + path + ".meta.json");
    auto kv = v33g_runtime_k3basis::read_runtime_meta_kv(path);
    if(v33g_runtime_k3basis::meta_string(kv, "version", "") != "v33g_runtime_k3basis_cache") {
        throw std::runtime_error("Bad v33g runtime cache version in " + path);
    }
    if(v33g_runtime_k3basis::meta_string(kv, "irrep", irrep) != irrep) {
        throw std::runtime_error("v33g runtime cache irrep mismatch in " + path);
    }
    if(std::abs(v33g_runtime_k3basis::meta_double(kv, "Lbyas", sblock.Lval) - sblock.Lval) > 1e-12) {
        throw std::runtime_error("v33g runtime cache L mismatch in " + path);
    }
    if(std::abs(v33g_runtime_k3basis::meta_double(kv, "xi", sblock.xival) - sblock.xival) > 1e-12) {
        throw std::runtime_error("v33g runtime cache xi mismatch in " + path);
    }
    if(v33g_runtime_k3basis::meta_int(kv, "coarseN", sblock.coarseN) != sblock.coarseN) {
        throw std::runtime_error("v33g runtime cache coarseN mismatch in " + path);
    }
    if(std::abs(v33g_runtime_k3basis::meta_double(kv, "atmK", par.atmK) - par.atmK) > 1e-12) throw std::runtime_error("v33g runtime cache atmK mismatch in " + path);
    if(std::abs(v33g_runtime_k3basis::meta_double(kv, "atmpi", par.atmpi) - par.atmpi) > 1e-12) throw std::runtime_error("v33g runtime cache atmpi mismatch in " + path);
    if(std::abs(v33g_runtime_k3basis::meta_double(kv, "eta_1", par.eta_1) - par.eta_1) > 1e-12) throw std::runtime_error("v33g runtime cache eta_1 mismatch in " + path);
    if(std::abs(v33g_runtime_k3basis::meta_double(kv, "eta_2", par.eta_2) - par.eta_2) > 1e-12) throw std::runtime_error("v33g runtime cache eta_2 mismatch in " + path);
    if(std::abs(v33g_runtime_k3basis::meta_double(kv, "alpha", par.alpha) - par.alpha) > 1e-12) throw std::runtime_error("v33g runtime cache alpha mismatch in " + path);
    if(std::abs(v33g_runtime_k3basis::meta_double(kv, "epsilon_h", par.epsilon_h) - par.epsilon_h) > 1e-12) throw std::runtime_error("v33g runtime cache epsilon_h mismatch in " + path);
    if(std::abs(v33g_runtime_k3basis::meta_double(kv, "max_shell_num", par.max_shell_num) - par.max_shell_num) > 1e-12) throw std::runtime_error("v33g runtime cache max_shell_num mismatch in " + path);
    if(std::abs(v33g_runtime_k3basis::meta_double(kv, "tolerance", par.tolerance) - par.tolerance) > 1e-12) throw std::runtime_error("v33g runtime cache tolerance mismatch in " + path);
    if(v33g_runtime_k3basis::meta_int(kv, "parity", par.parity) != par.parity) throw std::runtime_error("v33g runtime cache parity mismatch in " + path);
    if(std::abs(v33g_runtime_k3basis::meta_double(kv, "eig_tol", par.eig_tol) - par.eig_tol) > 1e-12) throw std::runtime_error("v33g runtime cache eig_tol mismatch in " + path);
    if(std::abs(v33g_runtime_k3basis::meta_double(kv, "norm_tol", par.norm_tol) - par.norm_tol) > 1e-12) throw std::runtime_error("v33g runtime cache norm_tol mismatch in " + path);
    if(std::abs(v33g_runtime_k3basis::meta_double(kv, "proj_tol", par.proj_tol) - par.proj_tol) > 1e-12) throw std::runtime_error("v33g runtime cache proj_tol mismatch in " + path);
    const auto cached = v33g_runtime_k3basis::load_runtime_cache(path, irrep);
    #pragma omp critical(v33g_runtime_cache_log)
    {
        std::cout << "[v33g-cache] loaded " << path << " irrep=" << irrep << " rows=" << cached.grid.size() << "\n";
    }
    if(!cached.grid.empty() && (int)cached.grid.size() != sblock.coarseN) {
        std::cout << "[v33g-cache-warning] row count != coarseN for " << key_string(sblock.Lval,irrep) << "\n";
    }
    return cached;
}

// Read v32zu GPU cache format and enrich each row with the CPU Kdf3 metadata needed by assemble_QC.
static void read_exact(std::ifstream& is, void* p, std::size_t n) {
    is.read(reinterpret_cast<char*>(p), static_cast<std::streamsize>(n));
    if(!is) throw std::runtime_error("GPU cache read failed/truncated");
}
template<class T> static T read_scalar(std::ifstream& is) { T x; read_exact(is,&x,sizeof(T)); return x; }
static std::complex<double> swap_real_imag_for_trusted_v33h(const std::complex<double>& z) {
    return std::complex<double>(z.imag(), z.real());
}
static IrrepCache load_gpu_coarse_cache_one(const std::string& path, const FitSettings& sblock, const PhysicsParams& par, const std::string& irrep) {
    std::ifstream is(path, std::ios::binary);
    if(!is) throw std::runtime_error("Could not open GPU coarse cache: "+path);
    gpu_cache_reader_convention::require_gpu_cache_format_header(
        is,
        gpu_cache_reader_convention::GpuCacheFileFormat::CombinedRawF3invVselGpu,
        path
    );
    const std::uint64_t rec_magic = 0x5653325a4f524543ULL;
    IrrepCache ic; ic.label=irrep; ic.spec=parse_label(irrep);
    int nrec=0;
    while(true) {
        int c = is.peek();
        if(c==EOF) break;
        std::uint64_t magic=0;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if(!is) break;
        if(magic != rec_magic) throw std::runtime_error("Bad GPU record magic in "+path);
        std::int32_t grid_i = read_scalar<std::int32_t>(is);
        std::int32_t dim1   = read_scalar<std::int32_t>(is);
        std::int32_t dim2   = read_scalar<std::int32_t>(is);
        std::int32_t n      = read_scalar<std::int32_t>(is);
        std::int32_t vdim   = read_scalar<std::int32_t>(is);
        double Ecm = read_scalar<double>(is);
        double En  = read_scalar<double>(is);
        if(n<0 || vdim<0) throw std::runtime_error("Negative dimensions in GPU cache "+path);
        const auto conv = v33g_runtime_k3basis::GpuComplexReadConvention::RealImagSwappedV33dV32zu;
        Eigen::MatrixXcd F3inv = v33g_runtime_k3basis::read_gpu_cache_matrix_col_major(is, n, n, conv);
        Eigen::MatrixXcd Vsel = v33g_runtime_k3basis::read_gpu_cache_matrix_col_major(is, n, vdim, conv);
        // The trusted v33h determinant scan applies this second swap after the
        // shared variant-04 reader.  Keep the fitter numerically identical to
        // that reference; the L20 audit showed the one-swap path loses the
        // accepted sign-change bracket.
        for(int r = 0; r < F3inv.rows(); ++r)
            for(int c = 0; c < F3inv.cols(); ++c)
                F3inv(r,c) = swap_real_imag_for_trusted_v33h(F3inv(r,c));
        for(int r = 0; r < Vsel.rows(); ++r)
            for(int c = 0; c < Vsel.cols(); ++c)
                Vsel(r,c) = swap_real_imag_for_trusted_v33h(Vsel(r,c));
        ProjectedQCCacheEntry e;
        e.label=irrep; e.spec=ic.spec; e.i=grid_i; e.Ecm=Ecm; e.En=En;
        e.total_dim=n; e.proj_dim=vdim; e.F3inv_full=std::move(F3inv); e.Vsel=std::move(Vsel);
        try {
            const std::vector<int> nnP_vec = {ic.spec.nnP[0], ic.spec.nnP[1], ic.spec.nnP[2]};
            const comp pi = std::acos(-1.0);
            const double L = par.L();
            const comp twopibyL = comp(2.0,0.0)*pi/comp(L,0.0);
            std::vector<comp> total_P(3), nnP_config(3);
            for(int a=0; a<3; ++a) { total_P[a]=twopibyL*double(nnP_vec[a]); nnP_config[a]=comp(nnP_vec[a],0.0); }
            const comp Ecm_c(Ecm,0.0);
            const comp En_c = Ecm_to_E(Ecm_c,total_P);
            e.En_c = En_c;
            e.En = En_c.real();
            e.total_P = total_P;
            std::vector<std::vector<comp>> plm_config(5), klm_config(5);
            std::vector<std::vector<int>> np_config(5), nk_config(5);
            config_maker_4_momentum_first(plm_config,np_config,par.waves_vec_1,En_c,total_P,par.atmK,par.atmK,par.atmpi,L,par.epsilon_h,par.max_shell_num,par.tolerance);
            config_maker_4_momentum_first(klm_config,nk_config,par.waves_vec_2,En_c,total_P,par.atmpi,par.atmK,par.atmK,L,par.epsilon_h,par.max_shell_num,par.tolerance);
            e.plm_config = plm_config;
            e.klm_config = klm_config;
            const int expectedN = int(plm_config[0].size() + klm_config[0].size());
            if(expectedN != n) {
                std::ostringstream os; os << "GPU cache dimension mismatch for " << path << " row " << grid_i << ": gpu_n=" << n << " cpu_config_N=" << expectedN;
                throw std::runtime_error(os.str());
            }
            if(vdim <= 0) { e.success=0; e.error="ZERO_PROJECTED_DIM_FROM_GPU"; }
            else {
                e.F3inv_proj = e.Vsel.adjoint() * e.F3inv_full * e.Vsel;
                e.success = (e.F3inv_full.allFinite() && e.Vsel.allFinite() && e.F3inv_proj.allFinite()) ? 1 : 0;
                e.error = e.success ? "OK_GPU_COARSE" : "NONFINITE_GPU_CACHE";
                // Hot-window mode precomputes the K3df basis only after the
                // accepted row windows have been loaded.  The raw reader must
                // not spend startup time building basis matrices for the full
                // 20,000-row grid when FCN evaluations use local windows.
            }
        } catch(const std::exception& ex) {
            e.success=0; e.error=ex.what();
        }
        ic.grid.push_back(std::move(e));
        ++nrec;
    }
    std::sort(ic.grid.begin(),ic.grid.end(),[](auto&a,auto&b){return a.Ecm<b.Ecm;});
    std::cout << "[gpu-cache] loaded " << path << " irrep=" << irrep << " rows=" << ic.grid.size();
    if(!ic.grid.empty()) std::cout << " Ecm=[" << ic.grid.front().Ecm << "," << ic.grid.back().Ecm << "]";
    std::cout << "\n";
    if((int)ic.grid.size() != sblock.coarseN) std::cout << "[gpu-cache-warning] row count != coarseN for " << key_string(sblock.Lval,irrep) << "\n";
    return ic;
}
static IrrepCache load_refined_cache_or_empty(const std::string& path, const std::string& irrep) {
    if(!file_exists(path)) { IrrepCache r; r.label=irrep; r.spec=parse_label(irrep); return r; }
    auto caches = load_binary_f3inv_cache_v32f(path);
    for(auto& ic: caches) if(ic.label==irrep) return ic;
    if(caches.size()==1) { caches[0].label=irrep; caches[0].spec=parse_label(irrep); return caches[0]; }
    IrrepCache r; r.label=irrep; r.spec=parse_label(irrep); return r;
}

struct CandidateWithBlock { double L; std::string file_irrep; std::string internal_irrep; v32w::Cand c; };

struct BenchmarkTiming : public v32w::QCSearchTiming {
    double cache_load_sec = 0.0;
    double precompute_sec = 0.0;
    double root_assignment_sec = 0.0;
    double chisq_sec = 0.0;
    double total_sec = 0.0;
    double chi2 = NAN;
    int model_found = 0;
    long long window_rows_evaluated = 0;
    int window_expansions = 0;
    bool full_scan_occurred = false;
    bool fallback_occurred = false;
};

struct AssignmentRow {
    std::string mode;
    double Lbyas = 0.0;
    std::string irrep;
    int target_index = -1;
    double target_Ecm = NAN;
    double model_root_Ecm = NAN;
    double residual = NAN;
    std::string root_source;
    int classifier_candidate_id = -1;
    std::string assignment_status;
};

struct RuntimeBlock {
    BlockInfo info;
    FitSettings settings;
    PhysicsParams par;
    double cscale = 0.0;
    std::string coarse_path;
    std::string refined_path;
    IrrepCache coarse;
    IrrepCache refined;
};

struct AcceptedZeroWindow {
    double Lbyas = 0.0;
    std::string irrep;
    int lattice_level_index = -1;
    int target_global_index = -1;
    int bracket_id = -1;
    double E_left_bracket = NAN;
    double E_right_bracket = NAN;
    double zero_estimate_initial = NAN;
    int center_row = -1;
    int row_left = -1;
    int row_right = -1;
    int max_row_left = -1;
    int max_row_right = -1;
    double previous_model_zero = NAN;
    bool has_previous_model_zero = false;
    bool inside_Ecm_cutoff = false;
};

struct WindowGridRow {
    int row = -1;
    double Ecm = NAN;
    int Nfull = -1;
    int Nproj = -1;
    double det_real = NAN;
    double det_imag = NAN;
};

static std::vector<std::string> split_csv2(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    bool quoted = false;
    for(char c: line) {
        if(c=='"') { quoted = !quoted; continue; }
        if(c==',' && !quoted) { out.push_back(cur); cur.clear(); }
        else cur.push_back(c);
    }
    out.push_back(cur);
    return out;
}
static int csv_index2(const std::vector<std::string>& h, const std::string& name) {
    auto it = std::find(h.begin(),h.end(),name);
    return it==h.end() ? -1 : int(it-h.begin());
}
static std::string csv_at2(const std::vector<std::string>& r, int i) {
    return i>=0 && i<(int)r.size() ? trim2(r[std::size_t(i)]) : std::string();
}
static double csv_double2(const std::vector<std::string>& r, int i, double d=NAN) {
    try { return csv_at2(r,i).empty() ? d : std::stod(csv_at2(r,i)); }
    catch(...) { return d; }
}
static int csv_int2(const std::vector<std::string>& r, int i, int d=-1) {
    try { return csv_at2(r,i).empty() ? d : std::stoi(csv_at2(r,i)); }
    catch(...) { return d; }
}
static bool csv_true2(const std::vector<std::string>& r, int i) {
    const auto s = csv_at2(r,i);
    return s=="true" || s=="True" || s=="1" || s=="yes" || s=="YES";
}

static std::vector<WindowGridRow> load_window_grid(const std::string& path) {
    std::ifstream in(path);
    if(!in) throw std::runtime_error("cannot open accepted-window determinant grid: " + path);
    std::string line;
    if(!std::getline(in,line)) throw std::runtime_error("empty accepted-window determinant grid: " + path);
    const auto h = split_csv2(line);
    const int ir = csv_index2(h,"row_sorted_index"), ie = csv_index2(h,"Ecm");
    const int inf = csv_index2(h,"Nfull"), inp = csv_index2(h,"Nproj");
    const int idr = csv_index2(h,"det_real"), idi = csv_index2(h,"det_imag");
    if(ir<0 || ie<0) throw std::runtime_error("det-grid missing row_sorted_index/Ecm: " + path);
    std::vector<WindowGridRow> rows;
    while(std::getline(in,line)) {
        if(line.empty()) continue;
        const auto r = split_csv2(line);
        const int row = csv_int2(r,ir,(int)rows.size());
        rows.push_back(WindowGridRow{row,csv_double2(r,ie),csv_int2(r,inf),csv_int2(r,inp),csv_double2(r,idr),csv_double2(r,idi)});
    }
    std::sort(rows.begin(),rows.end(),[](const auto& a,const auto& b){return a.row<b.row;});
    for(int i=0;i<(int)rows.size();++i) if(rows[std::size_t(i)].row!=i) throw std::runtime_error("det-grid row indices are not contiguous: " + path);
    return rows;
}

static std::map<int,std::pair<int,int>> load_window_brackets(const std::string& path) {
    std::ifstream in(path);
    if(!in) throw std::runtime_error("cannot open accepted-window candidate brackets: " + path);
    std::string line;
    if(!std::getline(in,line)) throw std::runtime_error("empty candidate bracket file: " + path);
    const auto h = split_csv2(line);
    const int ib=csv_index2(h,"bracket_id"), il=csv_index2(h,"row_left"), ir=csv_index2(h,"row_right");
    if(ib<0 || il<0 || ir<0) throw std::runtime_error("candidate file missing bracket/row columns: " + path);
    std::map<int,std::pair<int,int>> out;
    while(std::getline(in,line)) {
        if(line.empty()) continue;
        const auto r=split_csv2(line);
        const int b=csv_int2(r,ib), l=csv_int2(r,il), rr=csv_int2(r,ir);
        if(b>0 && l>=0 && rr>=0) out[b]={l,rr};
    }
    return out;
}

static int nearest_window_row(const std::vector<WindowGridRow>& rows, double E) {
    if(rows.empty()) return -1;
    int best=0; double bd=std::abs(rows[0].Ecm-E);
    for(int i=1;i<(int)rows.size();++i) { const double d=std::abs(rows[std::size_t(i)].Ecm-E); if(d<bd){bd=d;best=i;} }
    return best;
}

static std::vector<AcceptedZeroWindow> load_accepted_windows(const MultiConfig& cfg,
                                                               const BlockInfo& block,
                                                               const RuntimeBlock& rb,
                                                               const std::vector<MultiTarget>& mt) {
    const CacheKey key{block.L,block.internal_irrep};
    const auto zpath = cfg.accepted_zeros_files.at(key);
    const auto cpath = cfg.candidate_brackets_files.at(key);
    const auto gpath = cfg.det_grid_files.at(key);
    if(zpath.empty() || cpath.empty() || gpath.empty()) throw std::runtime_error("accepted_windows requires accepted/candidate/grid files for " + key_string(block.L,block.internal_irrep));
    const auto grid = load_window_grid(gpath);
    if(grid.size()!=rb.coarse.grid.size()) throw std::runtime_error("accepted-window det-grid/cache row count mismatch for " + key_string(block.L,block.internal_irrep));
    const auto brackets = load_window_brackets(cpath);
    std::ifstream in(zpath);
    if(!in) throw std::runtime_error("cannot open accepted true-zero file: " + zpath);
    std::string line;
    if(!std::getline(in,line)) throw std::runtime_error("empty accepted true-zero file: " + zpath);
    const auto h=split_csv2(line);
    const int ib=csv_index2(h,"bracket_id"), iel=csv_index2(h,"E_left"), ier=csv_index2(h,"E_right");
    const int iz=csv_index2(h,"zero_estimate"), ilab=csv_index2(h,"user_label"), iin=csv_index2(h,"inside_Ecm_cutoff");
    if(ib<0 || iel<0 || ier<0 || iz<0 || ilab<0) throw std::runtime_error("accepted true-zero file missing required columns: " + zpath);
    struct Source { int b=-1; double el=NAN,er=NAN,z=NAN; bool inside=false; };
    std::vector<Source> src;
    while(std::getline(in,line)) {
        if(line.empty()) continue;
        const auto r=split_csv2(line);
        if(!csv_true2(r,ilab)) continue;
        Source s; s.b=csv_int2(r,ib); s.el=csv_double2(r,iel); s.er=csv_double2(r,ier); s.z=csv_double2(r,iz); s.inside=(iin>=0 ? csv_true2(r,iin) : s.z<=cfg.energy_cutoff);
        if(s.inside) src.push_back(s);
    }
    if(src.size()!=block.target_indices.size()) throw std::runtime_error("accepted zero count does not match lattice target count for " + key_string(block.L,block.internal_irrep));
    std::sort(src.begin(),src.end(),[](const auto& a,const auto& b){return a.z<b.z;});
    std::vector<AcceptedZeroWindow> out;
    for(std::size_t k=0;k<src.size();++k) {
        const auto& s=src[k];
        auto br=brackets.find(s.b);
        if(br==brackets.end()) throw std::runtime_error("accepted bracket missing from candidate file for " + key_string(block.L,block.internal_irrep));
        const int center=nearest_window_row(grid,s.z);
        int bl=std::min(br->second.first,br->second.second), rr=std::max(br->second.first,br->second.second);
        const int half=std::max(1,cfg.window_half_rows), maxhalf=std::max(half,cfg.max_window_half_rows);
        AcceptedZeroWindow w;
        w.Lbyas=block.L; w.irrep=block.internal_irrep; w.lattice_level_index=int(k); w.target_global_index=block.target_indices[k];
        w.bracket_id=s.b; w.E_left_bracket=s.el; w.E_right_bracket=s.er; w.zero_estimate_initial=s.z;
        w.center_row=center; w.row_left=std::max(0,std::min(center-half,bl)); w.row_right=std::min((int)grid.size()-1,std::max(center+half,rr));
        w.max_row_left=std::max(0,center-maxhalf); w.max_row_right=std::min((int)grid.size()-1,center+maxhalf);
        w.previous_model_zero=s.z; w.has_previous_model_zero=true; w.inside_Ecm_cutoff=s.inside;
        out.push_back(w);
    }
    return out;
}

static std::vector<v32w::Cand> evaluate_accepted_window(const RuntimeBlock& rb,
                                                        const AcceptedZeroWindow& w,
                                                        const MultiConfig& cfg,
                                                        const K3dfParameters& kp,
                                                        BenchmarkTiming* timing) {
    if(cfg.det_backend=="gpu_batched") throw std::runtime_error("gpu_batched is not implemented in the hot-window pass");
    if(cfg.det_backend!="cpu_openmp" && cfg.det_backend!="auto") throw std::runtime_error("unsupported hot-window determinant backend: " + cfg.det_backend);
    int left=w.row_left, right=w.row_right;
    for(int attempt=0; attempt<2; ++attempt) {
        std::vector<v32w::Eval> vals;
        vals.reserve(std::size_t(right-left+1));
        for(int i=left;i<=right;++i) {
            vals.push_back(v32w::eval_entry_QC(rb.coarse.grid[std::size_t(i)],kp,rb.par,rb.settings.debug,rb.cscale));
            if(timing) ++timing->window_rows_evaluated;
        }
        std::vector<v32w::Cand> candidates = v32w::find_QC_zeros_v3_from_coarse_grid(rb.info.internal_irrep,vals,cfg.zratio,timing);
        if(v32w::is_v4_like_mode(cfg.classifier_mode)) candidates=v32w::merge_QC_zeros_v4(std::move(candidates),v32w::g_v4_merge_tol_v32w,timing);
        std::vector<v32w::Cand> zeros;
        for(auto& c: candidates) if(c.kind=="true_zero") zeros.push_back(std::move(c));
        if(!zeros.empty()) {
            std::sort(zeros.begin(),zeros.end(),[&](const auto& a,const auto& b){return std::abs(a.E-w.previous_model_zero)<std::abs(b.E-w.previous_model_zero);});
            zeros.resize(1);
            return zeros;
        }
        if(attempt==0 && (left!=w.max_row_left || right!=w.max_row_right)) {
            left=w.max_row_left; right=w.max_row_right;
            if(timing) ++timing->window_expansions;
        }
    }
    return {};
}

static void assign_block_model(const RuntimeBlock& rb,
                               const std::vector<v32w::Cand>& cvec,
                               std::vector<double>& model,
                               BenchmarkTiming* timing = nullptr) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<double> zeros;
    for(const auto& c: cvec) if(c.kind == "true_zero") zeros.push_back(c.E);
    std::sort(zeros.begin(), zeros.end());
    for(std::size_t k = 0; k < rb.info.target_indices.size(); ++k) {
        const int gi = rb.info.target_indices[k];
        model[std::size_t(gi)] = (k < zeros.size()) ? zeros[k] : 0.0;
    }
    if(timing) timing->root_assignment_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static int nearest_candidate_id(const std::vector<CandidateWithBlock>& cands, double L, const std::string& irrep, double E) {
    int best = -1;
    double best_d = std::numeric_limits<double>::infinity();
    for(std::size_t i = 0; i < cands.size(); ++i) {
        const auto& c = cands[i];
        if(std::abs(c.L - L) > 1e-12 || c.internal_irrep != irrep || c.c.kind != "true_zero") continue;
        const double d = std::abs(c.c.E - E);
        if(d < best_d) { best_d = d; best = static_cast<int>(i); }
    }
    return best;
}

static std::vector<AssignmentRow> build_assignment_rows(const std::string& mode,
                                                        const std::vector<MultiTarget>& mt,
                                                        const std::vector<TargetLevel>& targets,
                                                        const std::vector<double>& model,
                                                        const std::vector<CandidateWithBlock>& cands) {
    std::vector<AssignmentRow> rows;
    rows.reserve(targets.size());
    std::map<std::string, int> block_index;
    for(std::size_t i = 0; i < targets.size(); ++i) {
        const auto& t = targets[i];
        const std::string key = key_string(t.Lbyas, t.label);
        const int idx = block_index[key]++;
        AssignmentRow row;
        row.mode = mode;
        row.Lbyas = t.Lbyas;
        row.irrep = t.label;
        row.target_index = idx;
        row.target_Ecm = t.Ecm;
        row.model_root_Ecm = model[i];
        row.residual = t.Ecm - model[i];
        row.root_source = std::isfinite(model[i]) && model[i] > 0.0 ? "true_zero" : "missing";
        row.classifier_candidate_id = nearest_candidate_id(cands, t.Lbyas, t.label, model[i]);
        row.assignment_status = (std::isfinite(model[i]) && model[i] > 0.0) ? "assigned" : "missing";
        rows.push_back(std::move(row));
    }
    return rows;
}

static void write_assignment_files(const fs::path& root,
                                   const std::string& mode,
                                   const std::vector<MultiTarget>& mt,
                                   const std::vector<TargetLevel>& targets,
                                   const std::vector<double>& model,
                                   const std::vector<CandidateWithBlock>& cands) {
    const fs::path outdir = root / "root_assignments";
    fs::create_directories(outdir);
    const auto rows = build_assignment_rows(mode, mt, targets, model, cands);
    const fs::path csv = outdir / (mode + "_assignments.csv");
    const fs::path js = outdir / (mode + "_assignments.json");
    std::ofstream ocsv(csv);
    ocsv << std::setprecision(17);
    ocsv << "mode,Lbyas,irrep,target_index,target_Ecm,model_root_Ecm,residual,root_source,classifier_candidate_id,assignment_status\n";
    for(const auto& r: rows) {
        ocsv << r.mode << "," << r.Lbyas << "," << r.irrep << "," << r.target_index << ","
             << r.target_Ecm << "," << r.model_root_Ecm << "," << r.residual << ","
             << r.root_source << "," << r.classifier_candidate_id << "," << r.assignment_status << "\n";
    }
    std::ofstream ojs(js);
    ojs << std::setprecision(17);
    ojs << "[\n";
    for(std::size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        ojs << "  {\"mode\":\"" << r.mode << "\",\"Lbyas\":" << r.Lbyas << ",\"irrep\":\"" << r.irrep
            << "\",\"target_index\":" << r.target_index << ",\"target_Ecm\":" << r.target_Ecm
            << ",\"model_root_Ecm\":" << r.model_root_Ecm << ",\"residual\":" << r.residual
            << ",\"root_source\":\"" << r.root_source << "\",\"classifier_candidate_id\":" << r.classifier_candidate_id
            << ",\"assignment_status\":\"" << r.assignment_status << "\"}" << (i + 1 < rows.size() ? "," : "") << "\n";
    }
    ojs << "]\n";
}

static void print_parameter_mask(const MultiConfig& cfg) {
    const std::string bounds = cfg.base.use_parameter_limits
        ? (std::string("[") + std::to_string(cfg.base.param_lower) + ", " + std::to_string(cfg.base.param_upper) + "]")
        : std::string("none");
    std::cout << "[parameter-mask] K3iso0 " << (cfg.float_params[0]?"floating":"fixed") << " start=" << std::setprecision(17) << cfg.base.guess.K3iso0
              << " step=" << cfg.base.step.K3iso0 << " bounds=" << bounds << "\n";
    std::cout << "[parameter-mask] K3iso1 " << (cfg.float_params[1]?"floating":"fixed") << " start=" << cfg.base.guess.K3iso1
              << " step=" << cfg.base.step.K3iso1 << " bounds=" << bounds << "\n";
    std::cout << "[parameter-mask] K3B " << (cfg.float_params[2]?"floating":"fixed") << " start=" << cfg.base.guess.K3B
              << " step=" << cfg.base.step.K3B << " bounds=" << bounds << "\n";
    std::cout << "[parameter-mask] K3E " << (cfg.float_params[3]?"floating":"fixed") << " start=" << cfg.base.guess.K3E
              << " step=" << cfg.base.step.K3E << " bounds=" << bounds << "\n";
}

static RuntimeBlock build_runtime_block(const MultiConfig& cfg, const BlockInfo& b, bool load_refined_cache) {
    RuntimeBlock rb;
    rb.info = b;
    rb.settings = settings_for_block(cfg, b.L, b.internal_irrep);
    rb.par = make_base_physics(rb.settings);
    rb.cscale = block_cnorm(cfg, b.L);
    rb.refined_path = refined_path_for(cfg, b.L, b.internal_irrep);
    if(cfg.cache_backend == "v33g_runtime") {
        if(!v32w::is_v3_like_mode(cfg.classifier_mode) && !v32w::is_v4_like_mode(cfg.classifier_mode)) {
            throw std::runtime_error("v33g_runtime requires a supported classifier_mode alias");
        }
        rb.coarse_path = runtime_path_for(cfg, b.L, b.internal_irrep);
        rb.coarse = load_v33g_runtime_cache_one(cfg, rb.settings, rb.par, rb.coarse_path, b.internal_irrep);
        rb.refined.label = b.internal_irrep;
        rb.refined.spec = parse_label(b.internal_irrep);
    } else {
        rb.coarse_path = coarse_path_for(cfg, b.L, b.internal_irrep);
        ensure_gpu_coarse_cache(cfg, b.L, b.internal_irrep, rb.coarse_path);
        rb.coarse = load_gpu_coarse_cache_one(rb.coarse_path, rb.settings, rb.par, b.internal_irrep);
        if(load_refined_cache) {
            rb.refined = load_refined_cache_or_empty(rb.refined_path, b.internal_irrep);
            for(auto& e: rb.refined.grid) {
                if(e.success && !e.has_precomputed_k3_basis) precompute_projected_k3_basis(e, rb.par);
            }
        } else {
            rb.refined.label = b.internal_irrep;
            rb.refined.spec = parse_label(b.internal_irrep);
        }
    }
    return rb;
}

class MultiLFCN final : public ROOT::Minuit2::FCNBase {
public:
    MultiLFCN(MultiConfig cfg_, std::vector<MultiTarget> mt_, std::vector<TargetLevel> targets_, MatrixD cov_, MatrixD corr_, std::vector<BlockInfo> blocks_, bool load_refined_cache_)
        : cfg(std::move(cfg_)), mt(std::move(mt_)), targets(std::move(targets_)), cov(std::move(cov_)), corr(std::move(corr_)), blocks(std::move(blocks_)), load_refined_cache(load_refined_cache_) {
        dispatch = v32w::classifier_dispatch_info(cfg.classifier_mode);
        std::cout << "[classifier-dispatch] requested_mode=" << dispatch.requested_mode
                  << " resolved_mode=" << dispatch.resolved_mode
                  << " implementation=" << dispatch.implementation_tag
                  << " version_tag=" << dispatch.version_tag
                  << " dispatch_unique=" << (dispatch.dispatch_unique ? 1 : 0)
                  << "\n";
        const auto t0 = std::chrono::steady_clock::now();
        runtime_blocks.resize(blocks.size());
        if(cfg.cache_backend != "v33g_runtime") {
            for(const auto& b: blocks) {
                const std::string coarse_path = coarse_path_for(cfg, b.L, b.internal_irrep);
                ensure_gpu_coarse_cache(cfg, b.L, b.internal_irrep, coarse_path);
            }
        }
        if(cfg.cache_backend == "v33g_runtime" && blocks.size() > 1) {
            #pragma omp parallel for schedule(dynamic,1)
            for(int bi=0; bi<(int)blocks.size(); ++bi) {
                runtime_blocks[std::size_t(bi)] = build_runtime_block(cfg, blocks[std::size_t(bi)], load_refined_cache);
            }
        } else {
            for(int bi=0; bi<(int)blocks.size(); ++bi) {
                runtime_blocks[std::size_t(bi)] = build_runtime_block(cfg, blocks[std::size_t(bi)], load_refined_cache);
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double sec = std::chrono::duration<double>(t1 - t0).count();
        setup_cache_load_sec = sec;
        setup_precompute_sec = 0.0;
        std::cout << "[v33f-perf] cache_load_and_precompute_sec=" << std::setprecision(17) << sec << "\n";
        if(cfg.root_search_mode=="accepted_windows") {
            if(cfg.fallback_full_scan) throw std::runtime_error("accepted_windows requires fallback_full_scan=0");
            if(cfg.det_backend=="gpu_batched") throw std::runtime_error("gpu_batched is not implemented in the hot-window pass");
            for(const auto& b: blocks) {
                const auto it=std::find_if(runtime_blocks.begin(),runtime_blocks.end(),[&](const auto& rb){return std::abs(rb.info.L-b.L)<1e-12 && rb.info.internal_irrep==b.internal_irrep;});
                if(it==runtime_blocks.end()) throw std::runtime_error("runtime block missing for accepted windows: " + key_string(b.L,b.internal_irrep));
                auto ws=load_accepted_windows(cfg,b,*it,mt);
                accepted_windows.insert(accepted_windows.end(),ws.begin(),ws.end());
                if(cfg.root_search_mode=="accepted_windows") {
                    for(const auto& w: ws) {
                        for(int row=w.max_row_left; row<=w.max_row_right; ++row) {
                            auto& e=it->coarse.grid[std::size_t(row)];
                            if(e.success && !e.has_precomputed_k3_basis) precompute_projected_k3_basis(e,it->par);
                        }
                    }
                }
                std::cout << "[hot-windows] " << key_string(b.L,b.internal_irrep) << " windows=" << ws.size()
                          << " initial_half_rows=" << cfg.window_half_rows << " max_half_rows=" << cfg.max_window_half_rows << "\n";
            }
            std::cout << "[hot-windows] loaded total_windows=" << accepted_windows.size()
                      << " det_backend=" << (cfg.det_backend=="auto"?"cpu_openmp":cfg.det_backend)
                      << " fallback_full_scan=0 max_fcn_evals=" << cfg.max_fcn_evals << "\n";
        }
    }
    double Up() const override { return 1.0; }
    double operator()(const std::vector<double>& x) const override {
        const std::array<double,4> fixed_values{cfg.base.guess.K3iso0,cfg.base.guess.K3iso1,cfg.base.guess.K3B,cfg.base.guess.K3E};
        std::array<double,4> values=fixed_values;
        const int nfloat = int(std::count(cfg.float_params.begin(),cfg.float_params.end(),true));
        if(static_cast<int>(x.size())==4) {
            for(int i=0;i<4;++i) values[std::size_t(i)]=x[std::size_t(i)];
        } else {
            if(static_cast<int>(x.size())<nfloat) return cfg.base.failure_penalty;
            int j=0;
            for(int i=0;i<4;++i) if(cfg.float_params[std::size_t(i)]) values[std::size_t(i)]=x[std::size_t(j++)];
        }
        const size_t id = ++evals;
        if(cfg.max_fcn_evals > 0 && id > static_cast<size_t>(cfg.max_fcn_evals)) {
            throw std::runtime_error("short-minimization FCN-call limit exceeded: " + std::to_string(cfg.max_fcn_evals));
        }
        K3dfParameters kp{values[0],values[1],values[2],values[3]};
        const auto fcn_t0 = std::chrono::steady_clock::now();
        try {
            std::vector<CandidateWithBlock> cands;
            BenchmarkTiming hot_timing;
            auto model = model_for(kp,&cands,cfg.root_search_mode=="accepted_windows" ? &hot_timing : nullptr);
            int found=0; for(double m:model) if(std::isfinite(m)&&m>0.0) ++found;
            double chi = chi_square_v32f(targets,model,cov,corr,cfg.base.chi_square_mode,cfg.base.failure_penalty);
            if(found<(int)targets.size()) chi = cfg.base.failure_penalty;
            const auto fcn_t1 = std::chrono::steady_clock::now();
            const double fcn_sec = std::chrono::duration<double>(fcn_t1 - fcn_t0).count();
            if(cfg.root_search_mode=="accepted_windows") {
                std::cout << "[hot-window-fcn] eval=" << id
                          << " backend=" << (cfg.det_backend=="auto"?"cpu_openmp":cfg.det_backend)
                          << " rows_evaluated=" << hot_timing.window_rows_evaluated
                          << " expansions=" << hot_timing.window_expansions
                          << " fallback_full_scan=0 full_scan=0"
                          << " fcn_sec=" << fcn_sec
                          << " model_found=" << found << "/" << targets.size() << "\n";
            }
            if(cfg.base.print_each_fcn_eval && cfg.base.print_every_fcn_eval>0 && id%(size_t)cfg.base.print_every_fcn_eval==0) {
                std::cout << std::setprecision(17) << "[v32x-FCN] eval=" << id << " chi2=" << chi << " model_found=" << found << "/" << targets.size()
                          << " K3iso0=" << kp.K3iso0 << " K3iso1=" << kp.K3iso1 << " K3B=" << kp.K3B << " K3E=" << kp.K3E << "\n";
                std::map<std::string,std::array<int,3>> counts;
                for(const auto& cw: cands) {
                    auto& ar = counts[key_string(cw.L,cw.internal_irrep)];
                    if(cw.c.kind=="true_zero") ar[0]++; else if(cw.c.kind=="pole") ar[1]++; else ar[2]++;
                }
                for(const auto& kv: counts) std::cout << "  [v32x-FCN] " << kv.first << " true_zero=" << kv.second[0] << " pole=" << kv.second[1] << " uncertain=" << kv.second[2] << "\n";
            }
            return std::isfinite(chi)?chi:cfg.base.failure_penalty;
        } catch(const std::exception& e) {
            std::cout << "[v32x-FCN-warning] " << e.what() << "\n";
            return cfg.base.failure_penalty;
        }
    }

    std::vector<double> model_for(const K3dfParameters& kp, std::vector<CandidateWithBlock>* out_cands, BenchmarkTiming* timing = nullptr) const {
        std::vector<double> model(targets.size(),0.0);
        std::vector<CandidateWithBlock> allc;
        if(cfg.root_search_mode=="accepted_windows") {
            for(const auto& rb: runtime_blocks) {
                std::vector<v32w::Cand> cvec;
                for(const auto& w: accepted_windows) {
                    if(std::abs(w.Lbyas-rb.info.L)>1e-12 || w.irrep!=rb.info.internal_irrep) continue;
                    auto local=evaluate_accepted_window(rb,w,cfg,kp,timing);
                    for(auto& c: local) cvec.push_back(std::move(c));
                }
                for(const auto& c: cvec) allc.push_back({rb.info.L,rb.info.file_irrep_hint,rb.info.internal_irrep,c});
                assign_block_model(rb,cvec,model,timing);
            }
            if(out_cands) *out_cands=std::move(allc);
            return model;
        }
        for(auto& rb: runtime_blocks) {
            std::vector<ProjectedQCCacheEntry> new_entries;
            auto cvec = v32w::find_QC_zeros_refined(rb.coarse,rb.refined,rb.settings,rb.par,kp,rb.cscale,cfg.ninside,cfg.maxdepth,cfg.zratio,new_entries,timing ? static_cast<v32w::QCSearchTiming*>(timing) : nullptr);
            if(!new_entries.empty()) {
                for(auto& e: rb.refined.grid) {
                    if(e.success && !e.has_precomputed_k3_basis) precompute_projected_k3_basis(e, rb.par);
                }
            }
            if(cfg.save_refined_cache && load_refined_cache && !new_entries.empty()) {
                std::sort(rb.refined.grid.begin(),rb.refined.grid.end(),[](auto&a,auto&b){return a.Ecm<b.Ecm;});
                std::vector<IrrepCache> one{rb.refined};
                save_binary_f3inv_cache_v32f(rb.refined_path,one);
                write_refined_meta(rb.refined_path,cfg,rb.info.L,rb.info.internal_irrep,rb.refined.grid.size());
            }
            for(const auto& c: cvec) allc.push_back({rb.info.L,rb.info.file_irrep_hint,rb.info.internal_irrep,c});
            assign_block_model(rb, cvec, model, timing);
        }
        if(out_cands) *out_cands = std::move(allc);
        return model;
    }

    BenchmarkTiming benchmark_fcn(const K3dfParameters& kp, const std::string& mode, const fs::path& diagnostics_root, bool write_assignments = true) const {
        BenchmarkTiming timing;
        timing.cache_load_sec = setup_cache_load_sec;
        timing.precompute_sec = setup_precompute_sec;
        const auto total_t0 = std::chrono::steady_clock::now();
        std::vector<CandidateWithBlock> cands;
        auto model = model_for(kp, &cands, &timing);
        const auto chi_t0 = std::chrono::steady_clock::now();
        int found = 0;
        for(double m : model) if(std::isfinite(m) && m > 0.0) ++found;
        double chi = chi_square_v32f(targets, model, cov, corr, cfg.base.chi_square_mode, cfg.base.failure_penalty);
        const auto chi_t1 = std::chrono::steady_clock::now();
        timing.chisq_sec = std::chrono::duration<double>(chi_t1 - chi_t0).count();
        timing.model_found = found;
        timing.chi2 = chi;
        timing.total_sec = std::chrono::duration<double>(chi_t1 - total_t0).count();
        if(write_assignments) write_assignment_files(diagnostics_root, mode, mt, targets, model, cands);
        return timing;
    }

    // Diagnostic only: reuse accepted-window model_for() after one cache load.
    // No fitting or physics path is changed by this mode.
    int write_parameter_sensitivity(const K3dfParameters& start,
                                    const K3dfParameters& best,
                                    const std::array<double,4>& steps,
                                    const fs::path& outpath) const {
        struct Point { std::vector<double> model; double chi2=NAN; int found=0; BenchmarkTiming timing; };
        auto evaluate = [&](const K3dfParameters& kp) {
            Point p;
            std::vector<CandidateWithBlock> cands;
            p.model = model_for(kp, &cands, &p.timing);
            for(double x : p.model) if(std::isfinite(x) && x>0.0) ++p.found;
            p.chi2 = chi_square_v32f(targets, p.model, cov, corr, cfg.base.chi_square_mode, cfg.base.failure_penalty);
            if(p.found != static_cast<int>(targets.size())) p.chi2 = cfg.base.failure_penalty;
            return p;
        };
        const std::array<std::string,4> names{"K3iso0","K3iso1","K3B","K3E"};
        auto values = [](const K3dfParameters& p) {
            return std::array<double,4>{p.K3iso0,p.K3iso1,p.K3B,p.K3E};
        };
        std::ofstream out(outpath);
        if(!out) throw std::runtime_error("cannot open sensitivity output " + outpath.string());
        out << std::setprecision(17)
            << "base_point,parameter,base_value,step,minus_value,plus_value,chi2_minus,chi2_base,chi2_plus,delta_chi2_minus,delta_chi2_plus,"
               "found_minus,found_base,found_plus,rows_minus,rows_base,rows_plus,expansions_minus,expansions_base,expansions_plus,"
               "full_scan_minus,full_scan_base,full_scan_plus,fallback_minus,fallback_base,fallback_plus,"
               "model0_minus,model1_minus,model2_minus,model3_minus,model0_base,model1_base,model2_base,model3_base,"
               "model0_plus,model1_plus,model2_plus,model3_plus,dE0_dparam,dE1_dparam,dE2_dparam,dE3_dparam,max_abs_dE_dparam\n";
        for(const auto& item : std::array<std::pair<std::string,K3dfParameters>,2>{{{"starting",start},{"best_logged",best}}}) {
            const auto base_values = values(item.second);
            const Point central = evaluate(item.second);
            for(int j=0;j<4;++j) {
                auto minus_values=base_values, plus_values=base_values;
                minus_values[std::size_t(j)] -= steps[std::size_t(j)];
                plus_values[std::size_t(j)] += steps[std::size_t(j)];
                const K3dfParameters minus{minus_values[0],minus_values[1],minus_values[2],minus_values[3]};
                const K3dfParameters plus{plus_values[0],plus_values[1],plus_values[2],plus_values[3]};
                const Point pm = evaluate(minus), pp = evaluate(plus);
                std::array<double,4> deriv{};
                double max_abs=0.0;
                for(int k=0;k<4;++k) {
                    deriv[std::size_t(k)] = (pp.model[std::size_t(k)]-pm.model[std::size_t(k)])/(2.0*steps[std::size_t(j)]);
                    max_abs=std::max(max_abs,std::abs(deriv[std::size_t(k)]));
                }
                out << item.first << "," << names[std::size_t(j)] << "," << base_values[std::size_t(j)] << "," << steps[std::size_t(j)] << ","
                    << minus_values[std::size_t(j)] << "," << plus_values[std::size_t(j)] << ","
                    << pm.chi2 << "," << central.chi2 << "," << pp.chi2 << "," << (pm.chi2-central.chi2) << "," << (pp.chi2-central.chi2) << ","
                    << pm.found << "," << central.found << "," << pp.found << ","
                    << pm.timing.window_rows_evaluated << "," << central.timing.window_rows_evaluated << "," << pp.timing.window_rows_evaluated << ","
                    << pm.timing.window_expansions << "," << central.timing.window_expansions << "," << pp.timing.window_expansions << ","
                    << (pm.timing.full_scan_occurred?1:0) << "," << (central.timing.full_scan_occurred?1:0) << "," << (pp.timing.full_scan_occurred?1:0) << ","
                    << (pm.timing.fallback_occurred?1:0) << "," << (central.timing.fallback_occurred?1:0) << "," << (pp.timing.fallback_occurred?1:0);
                for(double x: pm.model) out << "," << x;
                for(double x: central.model) out << "," << x;
                for(double x: pp.model) out << "," << x;
                for(double x: deriv) out << "," << x;
                out << "," << max_abs << "\n";
            }
        }
        std::cout << "[sensitivity] wrote " << outpath << " rows=8 targets=" << targets.size() << "\n";
        return 0;
    }

    double cache_load_sec() const { return setup_cache_load_sec; }
    double precompute_sec() const { return setup_precompute_sec; }
private:
    MultiConfig cfg;
    std::vector<MultiTarget> mt;
    std::vector<TargetLevel> targets;
    MatrixD cov,corr;
    std::vector<BlockInfo> blocks;
    bool load_refined_cache = true;
    std::vector<AcceptedZeroWindow> accepted_windows;
    mutable std::vector<RuntimeBlock> runtime_blocks;
    v32w::ClassifierDispatchInfo dispatch;
    double setup_cache_load_sec = 0.0;
    double setup_precompute_sec = 0.0;
    mutable std::atomic<size_t> evals{0};
};

static void write_matrix2(const std::string& path, const MatrixD& M, const std::string& hdr) { v32w::write_matrix(path,M,hdr); }

static void write_outputs_multi(const MultiConfig& cfg,
                                const std::vector<MultiTarget>& mt,
                                const std::vector<TargetLevel>& targets,
                                const std::vector<double>& model,
                                const std::vector<CandidateWithBlock>& cands,
                                const K3dfParameters& best,
                                const K3dfParameters& err,
                                double minuit_fval,
                                double final_chi2,
                                int valid,
                                int model_found,
                                const MatrixD& data_cov,
                                const MatrixD& data_corr,
                                const MatrixD& pcov,
                                const MatrixD& pcorr) {
    fs::create_directories(cfg.base.output_dir);
    const std::string pref = (fs::path(cfg.base.output_dir)/cfg.base.output_tag).string();
    int nd=(int)targets.size(), np=int(std::count(cfg.float_params.begin(),cfg.float_params.end(),true)), ndof=nd-np;
    std::ofstream sum(pref+"_fit_summary_allL.dat");
    sum << std::setprecision(17)
        << "valid " << valid << "\n"
        << "minuit_fval " << minuit_fval << "\n"
        << "recomputed_final_chi2 " << final_chi2 << "\n"
        << "chi2 " << final_chi2 << "\n"
        << "model_levels_found " << model_found << "\n"
        << "ndata " << nd << "\n"
        << "npar " << np << "\n"
        << "ndof " << ndof << "\n"
        << "chi2_dof " << (ndof>0?final_chi2/double(ndof):NAN) << "\n"
        << "K3iso0 " << best.K3iso0 << " err " << err.K3iso0 << "\n"
        << "K3iso1 " << best.K3iso1 << " err " << err.K3iso1 << "\n"
        << "K3B " << best.K3B << " err " << err.K3B << "\n"
        << "K3E " << best.K3E << " err " << err.K3E << "\n";
    write_matrix2(pref+"_covariance_allL.dat",data_cov,"# block-diagonal lattice covariance: full within each L, zero cross-L");
    write_matrix2(pref+"_correlation_allL.dat",data_corr,"# lattice correlation matrix");
    write_matrix2(pref+"_covariance_inv_allL.dat",symmetric_pseudoinverse_v32f(data_cov),"# pseudoinverse of lattice covariance");
    write_matrix2(pref+"_parameter_covariance.dat",pcov,"# rows/cols K3iso0 K3iso1 K3B K3E");
    write_matrix2(pref+"_parameter_correlation.dat",pcorr,"# rows/cols K3iso0 K3iso1 K3B K3E");

    auto write_levels = [&](const std::string& path, double Lfilter){
        std::ofstream lev(path); lev << "# row Lbyas file_irrep internal_irrep state level_index lattice_Ecm lattice_err model_Ecm residual shifted_from_lab\n" << std::setprecision(17);
        for(std::size_t i=0;i<targets.size();++i) {
            if(Lfilter>0 && std::abs(targets[i].Lbyas-Lfilter)>1e-12) continue;
            lev << i << " " << targets[i].Lbyas << " " << mt[i].file_irrep << " " << mt[i].internal_irrep << " "
                << targets[i].state << " " << mt[i].level_index_in_block << " " << targets[i].Ecm << " " << targets[i].err << " "
                << model[i] << " " << (targets[i].Ecm-model[i]) << " " << targets[i].shifted_from_lab << "\n";
        }
    };
    write_levels(pref+"_fit_levels_allL.dat",-1);
    for(double L: cfg.Lvalues) write_levels(pref+"_fit_levels_L"+ltag(L)+".dat",L);

    auto write_spec = [&](const std::string& path, double Lfilter, bool true_only){
        std::ofstream f(path);
        f << "# Lbyas file_irrep internal_irrep index Ecm kind reason BL BR FL FR yBL_scaled yBR_scaled yFL_scaled yFR_scaled\n" << std::setprecision(17);
        std::map<std::string,int> idx;
        for(const auto& cw: cands) {
            if(Lfilter>0 && std::abs(cw.L-Lfilter)>1e-12) continue;
            if(true_only && cw.c.kind!="true_zero") continue;
            std::string key=key_string(cw.L,cw.internal_irrep);
            int k=idx[key]++;
            f << cw.L << " " << cw.file_irrep << " " << cw.internal_irrep << " " << k << " " << cw.c.E << " " << cw.c.kind << " " << cw.c.reason << " "
              << cw.c.BL << " " << cw.c.BR << " " << cw.c.FL << " " << cw.c.FR << " " << cw.c.ysBL << " " << cw.c.ysBR << " " << cw.c.ysFL << " " << cw.c.ysFR << "\n";
        }
    };
    write_spec(pref+"_bestfit_QC_spectrum_allL.dat",-1,true);
    write_spec(pref+"_all_QC_candidates_allL.dat",-1,false);
    for(double L: cfg.Lvalues) {
        write_spec(pref+"_bestfit_QC_spectrum_L"+ltag(L)+".dat",L,true);
        write_spec(pref+"_all_QC_candidates_L"+ltag(L)+".dat",L,false);
    }
    // Per-L covariance/correlation submatrices
    for(double L: cfg.Lvalues) {
        std::vector<int> ids; for(int i=0;i<(int)targets.size();++i) if(std::abs(targets[std::size_t(i)].Lbyas-L)<1e-12) ids.push_back(i);
        MatrixD CL(ids.size(),ids.size()), RL(ids.size(),ids.size());
        for(int a=0;a<(int)ids.size();++a) for(int b=0;b<(int)ids.size();++b) { CL(a,b)=data_cov(ids[a],ids[b]); RL(a,b)=data_corr(ids[a],ids[b]); }
        write_matrix2(pref+"_covariance_L"+ltag(L)+".dat",CL,"# per-L covariance block");
        write_matrix2(pref+"_correlation_L"+ltag(L)+".dat",RL,"# per-L correlation block");
    }
}

static int run_l20_hot_window_audit(const MultiConfig& cfg,
                                    const std::vector<MultiTarget>& mt,
                                    const std::vector<BlockInfo>& blocks) {
    const CacheKey key{20.0,"000_A1m"};
    const auto bit=std::find_if(blocks.begin(),blocks.end(),[](const auto& b){return std::abs(b.L-20.0)<1e-12 && b.internal_irrep=="000_A1m";});
    if(bit==blocks.end()) throw std::runtime_error("L20/000_A1m target block missing");
    RuntimeBlock rb=build_runtime_block(cfg,*bit,false);
    const auto windows=load_accepted_windows(cfg,*bit,rb,mt);
    if(windows.size()!=1) throw std::runtime_error("L20 hot-window audit expected exactly one accepted window");
    const auto& w=windows.front();
    const auto brackets=load_window_brackets(cfg.candidate_brackets_files.at(key));
    const auto br_it=brackets.find(w.bracket_id);
    if(br_it==brackets.end()) throw std::runtime_error("L20 audit bracket missing");
    const int original_left=std::min(br_it->second.first,br_it->second.second);
    const int original_right=std::max(br_it->second.first,br_it->second.second);
    for(int i=w.max_row_left;i<=w.max_row_right;++i) {
        auto& e=rb.coarse.grid[std::size_t(i)];
        if(e.success && !e.has_precomputed_k3_basis) precompute_projected_k3_basis(e,rb.par);
    }
    std::map<int,v32w::Eval> local;
    const K3dfParameters kp{cfg.base.guess.K3iso0,cfg.base.guess.K3iso1,cfg.base.guess.K3B,cfg.base.guess.K3E};
    for(int i=w.max_row_left;i<=w.max_row_right;++i)
        local.emplace(i,v32w::eval_entry_QC(rb.coarse.grid[std::size_t(i)],kp,rb.par,rb.settings.debug,rb.cscale));
    // Audit-only alternate convention: undo the compatibility swap above to
    // distinguish a reader mismatch from downstream projection/scaling.
    auto one_swap_entry = rb.coarse.grid[std::size_t(w.center_row)];
    for(int r = 0; r < one_swap_entry.F3inv_full.rows(); ++r)
        for(int c = 0; c < one_swap_entry.F3inv_full.cols(); ++c)
            one_swap_entry.F3inv_full(r,c) = swap_real_imag_for_trusted_v33h(one_swap_entry.F3inv_full(r,c));
    for(int r = 0; r < one_swap_entry.Vsel.rows(); ++r)
        for(int c = 0; c < one_swap_entry.Vsel.cols(); ++c)
            one_swap_entry.Vsel(r,c) = swap_real_imag_for_trusted_v33h(one_swap_entry.Vsel(r,c));
    one_swap_entry.F3inv_proj = one_swap_entry.Vsel.adjoint() * one_swap_entry.F3inv_full * one_swap_entry.Vsel;
    one_swap_entry.has_precomputed_k3_basis = false;
    const auto one_swap_eval = v32w::eval_entry_QC(one_swap_entry,kp,rb.par,rb.settings.debug,rb.cscale);

    const auto gpath=cfg.det_grid_files.at(key);
    const auto grid=load_window_grid(gpath);
    std::filesystem::create_directories(cfg.base.output_dir);
    const auto csvpath=fs::path(cfg.base.output_dir)/"L20_hot_window_reference_vs_local_det_debug.csv";
    std::ofstream csv(csvpath);
    csv << "row_index,Ecm,Nproj,det_grid_real,det_grid_imag,local_det_real,local_det_imag,sign_grid,sign_local,in_original_bracket,in_hot_window,dimension_jump_flag\n" << std::setprecision(17);
    auto sign=[](double x){return x>0?1:(x<0?-1:0);};
    double max_abs=0.0,max_rel=0.0;
    for(int i=w.max_row_left;i<=w.max_row_right;++i) {
        const auto& g=grid[std::size_t(i)]; const auto& l=local.at(i);
        const double ar=std::abs(g.det_real-l.det.det_re), ai=std::abs(g.det_imag-l.det.det_im);
        const double ad=std::max(ar,ai), rd=ad/std::max({std::abs(g.det_real),std::abs(g.det_imag),1e-300});
        max_abs=std::max(max_abs,ad); max_rel=std::max(max_rel,rd);
        const bool jump=(i>w.max_row_left && (grid[std::size_t(i-1)].Nfull!=g.Nfull || grid[std::size_t(i-1)].Nproj!=g.Nproj));
        const bool bracket=(i>=original_left && i<=original_right);
        const bool hot=(i>=w.row_left && i<=w.row_right);
        csv << i << "," << g.Ecm << "," << g.Nproj << "," << g.det_real << "," << g.det_imag << ","
            << l.det.det_re << "," << l.det.det_im << "," << sign(g.det_real) << "," << sign(l.det.det_re) << ","
            << (bracket?"true":"false") << "," << (hot?"true":"false") << "," << (jump?"true":"false") << "\n";
    }
    auto begin_vals=[&](int left,int right){
        std::vector<v32w::Eval> v; v.reserve(std::size_t(right-left+1));
        for(int i=left;i<=right;++i) v.push_back(local.at(i));
        return v;
    };
    const auto v3c=v32w::find_QC_zeros_v3_from_coarse_grid("000_A1m",begin_vals(w.row_left,w.row_right),cfg.zratio,nullptr);
    const bool v3true=std::any_of(v3c.begin(),v3c.end(),[](const auto& c){return c.kind=="true_zero";});
    const auto v4c=v32w::merge_QC_zeros_v4(v3c,v32w::g_v4_merge_tol_v32w,nullptr);
    const bool v4true=std::any_of(v4c.begin(),v4c.end(),[](const auto& c){return c.kind=="true_zero";});
    const int bl=original_left; const int br=original_right;
    const bool ref_flip=sign(grid[std::size_t(bl)].det_real)*sign(grid[std::size_t(br)].det_real)<0;
    const bool loc_flip=sign(local.at(bl).det.det_re)*sign(local.at(br).det.det_re)<0;
    double direct_max_abs=0.0, basis_direct_max_abs=0.0;
    std::vector<std::tuple<int,double,double,double>> direct_rows;
    for(int i=std::max(w.max_row_left,bl-2);i<=std::min(w.max_row_right,br+2);++i) {
        auto e=rb.coarse.grid[std::size_t(i)];
        e.has_precomputed_k3_basis=false;
        const auto d=v32w::eval_entry_QC(e,kp,rb.par,rb.settings.debug,rb.cscale);
        const double da=std::abs(grid[std::size_t(i)].det_real-d.det.det_re);
        const double ba=std::abs(d.det.det_re-local.at(i).det.det_re);
        direct_max_abs=std::max(direct_max_abs,da); basis_direct_max_abs=std::max(basis_direct_max_abs,ba);
        direct_rows.emplace_back(i,d.det.det_re,d.det.det_im,da);
    }
    std::string reason;
    if(direct_max_abs<=1e-18 && basis_direct_max_abs>1e-18) reason="precomputed projected K3df basis path disagrees with direct full QC assembly; use direct local assembly or fix basis precompute";
    else if(max_abs>1e-18) reason="reference/local determinant mismatch; audit row/cache/scaling/projection before classifier analysis";
    else if(ref_flip && !loc_flip) reason="local determinant sign sequence does not reproduce the trusted bracket";
    else if(loc_flip && !v3true && !v4true) reason="classifier rejects the local sign-change neighborhood; inspect local context rows/method filter";
    else if(!v3true || !v4true) reason="v3/v4 local classification disagreement";
    else reason="no local rejection reproduced; failure requires FCN assignment audit";
    const auto mdpath=fs::path(cfg.base.output_dir)/"L20_HOT_WINDOW_ROOT_FAILURE_AUDIT.md";
    std::ofstream md(mdpath);
    md << "# L20 hot-window root failure audit\n\n"
       << "- sector: `L20/000_A1m`\n"
       << "- accepted zero: `0.26932269955846955`\n"
       << "- bracket_id: `" << w.bracket_id << "`\n"
       << "- bracket Ecm: `[" << w.E_left_bracket << ", " << w.E_right_bracket << "]`\n"
       << "- nearest row / center row: `" << w.center_row << "`\n"
       << "- original bracket rows: `" << bl << "-" << br << "`\n"
       << "- initial hot-window rows: `" << w.row_left << "-" << w.row_right << "`\n"
       << "- maximum audit rows: `" << w.max_row_left << "-" << w.max_row_right << "`\n"
       << "- original bracket rows: `" << bl << "-" << br << "`\n"
       << "- original bracket inside initial window: `" << ((bl>=w.row_left && br<=w.row_right)?"true":"false") << "`\n"
       << "- reference bracket sign flip: `" << (ref_flip?"true":"false") << "`\n"
       << "- local bracket sign flip: `" << (loc_flip?"true":"false") << "`\n"
       << "- v3 local true-zero: `" << (v3true?"true":"false") << "`\n"
       << "- v4 local true-zero: `" << (v4true?"true":"false") << "`\n"
       << "- max absolute determinant component difference: `" << max_abs << "`\n"
       << "- max relative determinant difference: `" << max_rel << "`\n"
       << "- direct full-assembly vs reference max absolute difference (bracket context): `" << direct_max_abs << "`\n"
       << "- direct full-assembly vs projected-basis max absolute difference (bracket context): `" << basis_direct_max_abs << "`\n"
       << "- audit-only one-swap determinant at center row: `" << one_swap_eval.det.det_re << "+i(" << one_swap_eval.det.det_im << ")`\n"
       << "- audit-only loaded (trusted-reference-compatible) determinant at center row: `" << local.at(w.center_row).det.det_re << "+i(" << local.at(w.center_row).det.det_im << ")`\n"
       << "- exact local rejection cause: **" << reason << "**\n\n"
       << "## Dimension sequence\n\n";
    int start=w.max_row_left;
    for(int i=w.max_row_left+1;i<=w.max_row_right+1;++i) {
        if(i==w.max_row_right+1 || grid[std::size_t(i)].Nfull!=grid[std::size_t(start)].Nfull || grid[std::size_t(i)].Nproj!=grid[std::size_t(start)].Nproj) {
            md << "- rows `" << start << "-" << (i-1) << "`: Nfull/Nproj `" << grid[std::size_t(start)].Nfull << "/" << grid[std::size_t(start)].Nproj << "`\n";
            start=i;
        }
    }
    md << "\n## Rows around the original bracket\n\n"
       << "| row | Ecm | reference det | local det | reference sign | local sign | dimension jump |\n|---:|---:|---:|---:|---:|---:|---|\n";
    for(int i=std::max(w.max_row_left,bl-2);i<=std::min(w.max_row_right,br+2);++i) {
        const auto& g=grid[std::size_t(i)]; const auto& l=local.at(i);
        const bool jump=(i>w.max_row_left && (grid[std::size_t(i-1)].Nfull!=g.Nfull || grid[std::size_t(i-1)].Nproj!=g.Nproj));
        md << "| " << i << " | " << g.Ecm << " | " << g.det_real << " | " << l.det.det_re << " | " << sign(g.det_real) << " | " << sign(l.det.det_re) << " | " << (jump?"yes":"no") << " |\n";
    }
    md << "\nDirect full-assembly context rows (basis disabled):\n\n";
    for(const auto& [i,re,im,ad]:direct_rows) md << "- row " << i << ": det=" << re << "+i(" << im << "), abs_diff_vs_reference=" << ad << "\n";
    md << "\nDebug CSV: `" << csvpath.string() << "`\n";
    std::cout << "[hot-window-audit] wrote " << mdpath << " and " << csvpath << "\n";
    return 0;
}

} // namespace v32x_multiL

namespace fs = std::filesystem;
using k3df_fit_v32f::K3dfParameters;
using k3df_fit_v32f::chi_square_v32f;
#if V32F_HAS_MINUIT2
using k3df_fit_v32f::minuit_covariance_to_eigen_v32f;
#endif
using k3df_fit_v32f::covariance_to_correlation_v32f;

#ifndef V33G_NO_MAIN
int main(int argc, char** argv) {
    try {
        std::string cfgpath = (argc>1 ? argv[1] : "configs/config_v32x_multiL_QC_fitter.in");
        auto cfg = v32x_multiL::multiconfig_from_config(cfgpath);
        fs::create_directories(cfg.base.output_dir);
        fs::create_directories(cfg.refined_cache_dir);
#ifdef _OPENMP
        omp_set_num_threads(cfg.base.omp_threads);
        omp_set_dynamic(0);
        omp_set_max_active_levels(1);
#endif
        Eigen::setNbThreads(1);
        std::cout << "[v32x] config=" << cfgpath << "\n";
        std::cout << "[v32x] classifier_mode=" << cfg.classifier_mode << "\n";
        std::cout << "[v32x] multi-L values:"; for(double L: cfg.Lvalues) std::cout << " " << L; std::cout << "\n";
        std::cout << "[v32x] jack_energy_type=" << cfg.jack_energy_type << " cutoff=" << cfg.energy_cutoff << " jack_dir=" << cfg.jack_dir << "\n";
        if(cfg.max_total_lattice_levels>0 || cfg.max_lattice_levels_per_block>0)
            std::cout << "[v33f] level truncation max_total_lattice_levels=" << cfg.max_total_lattice_levels
                      << " max_lattice_levels_per_block=" << cfg.max_lattice_levels_per_block << "\n";
        auto [mt,targets,cov,corr,blocks] = v32x_multiL::load_multil_jack_targets(cfg);
        std::cout << "[v32x] targets=" << targets.size() << " covariance=" << cov.rows() << "x" << cov.cols() << " blocks=" << blocks.size() << "\n";
        std::cout << "[v32x] starting K3df guesses: " << std::setprecision(17)
                  << cfg.base.guess.K3iso0 << " " << cfg.base.guess.K3iso1 << " " << cfg.base.guess.K3B << " " << cfg.base.guess.K3E << "\n";

        const std::string run_mode = (argc>2 ? std::string(argv[2]) : std::string("fit"));

        if(run_mode=="audit-l20-hot-window") {
            return v32x_multiL::run_l20_hot_window_audit(cfg,mt,blocks);
        }

        if(run_mode=="pipeline-test" || run_mode=="pipeline" || run_mode=="async-pipeline") {
            const auto kv = v32w::read_kv(cfgpath);
            const int in_flight = std::max(1, v32w::gi(kv, "pipeline_in_flight", 2));
            const auto t0 = std::chrono::steady_clock::now();
            const K3dfParameters kp{cfg.base.guess.K3iso0, cfg.base.guess.K3iso1, cfg.base.guess.K3B, cfg.base.guess.K3E};
            struct Pending {
                std::size_t idx = 0;
                std::future<v32x_multiL::RuntimeBlock> fut;
            };
            std::vector<Pending> pending;
            std::size_t next = 0;
            std::size_t done = 0;
            while(done < blocks.size()) {
                while(next < blocks.size() && pending.size() < std::size_t(in_flight)) {
                    const std::size_t bi = next++;
                    const auto launch = (in_flight == 1 ? std::launch::deferred : std::launch::async);
                    pending.push_back(Pending{
                        bi,
                        std::async(launch, [&, bi]() {
                            return v32x_multiL::build_runtime_block(cfg, blocks[bi], false);
                        })
                    });
                }

                bool progressed = false;
                for(auto it = pending.begin(); it != pending.end(); ++it) {
                    if(it->fut.wait_for(std::chrono::seconds(0)) != std::future_status::ready) continue;
                    auto rb = it->fut.get();
                    std::vector<v32x_multiL::CandidateWithBlock> cands;
                    std::vector<k3df_fit_v32f::ProjectedQCCacheEntry> new_entries;
                    auto cvec = v32w::find_QC_zeros_refined(rb.coarse, rb.refined, rb.settings, rb.par, kp, rb.cscale, cfg.ninside, cfg.maxdepth, cfg.zratio, new_entries);
                    int tz=0, np=0, nu=0;
                    for(const auto& c: cvec) {
                        if(c.kind=="true_zero") ++tz;
                        else if(c.kind=="pole") ++np;
                        else ++nu;
                    }
                    std::cout << "[pipeline-test] " << v32x_multiL::key_string(rb.info.L, rb.info.internal_irrep)
                              << " rows=" << rb.coarse.grid.size()
                              << " true_zero=" << tz
                              << " pole=" << np
                              << " uncertain=" << nu << "\n";
                    ++done;
                    pending.erase(it);
                    progressed = true;
                    break;
                }
                if(!progressed) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
            }
            const auto t1 = std::chrono::steady_clock::now();
            const double sec = std::chrono::duration<double>(t1 - t0).count();
            std::cout << "[pipeline-test] done blocks=" << blocks.size() << " elapsed_sec=" << std::setprecision(17) << sec << "\n";
            return 0;
        }

        if(run_mode=="dump-targets") {
            v32x_multiL::dump_fitter_targets_v33k(cfg, mt, targets);
            std::cout << "[v33k] dumped fitter targets to diagnostics/v33k/fitter_targets/\n";
            return 0;
        }

        const bool load_refined_cache = !(run_mode=="spectrum-only" || run_mode=="spectrum" || run_mode=="find-spectrum" || run_mode=="fcn-once");
        v32x_multiL::MultiLFCN fcn(cfg,mt,targets,cov,corr,blocks,load_refined_cache);
        v32x_multiL::print_parameter_mask(cfg);

        if(run_mode=="benchmark-fcn") {
            int repeat = cfg.benchmark_repeat;
            int warmup = cfg.benchmark_warmup;
            for(int ai=3; ai<argc; ++ai) {
                const std::string arg = argv[ai];
                if(arg=="--repeat" && ai+1<argc) repeat = std::max(1, std::stoi(argv[++ai]));
                else if(arg.rfind("--repeat=",0)==0) repeat = std::max(1, std::stoi(arg.substr(9)));
                else if(arg=="--warmup" && ai+1<argc) warmup = std::max(0, std::stoi(argv[++ai]));
                else if(arg.rfind("--warmup=",0)==0) warmup = std::max(0, std::stoi(arg.substr(9)));
            }
            const K3dfParameters kp{cfg.base.guess.K3iso0, cfg.base.guess.K3iso1, cfg.base.guess.K3B, cfg.base.guess.K3E};
            const fs::path diag_root = fs::path("diagnostics/v33m");
            const fs::path run_csv_root = diag_root / "benchmark_runs";
            fs::create_directories(run_csv_root);
            std::cout << "[benchmark-fcn] classifier_mode=" << cfg.classifier_mode << "\n";
            std::cout << "[benchmark-fcn] targets=" << targets.size() << "\n";
            std::cout << "[benchmark-fcn] warmup=" << warmup << "\n";
            std::cout << "[benchmark-fcn] repeat=" << repeat << "\n";
            for(int i=0; i<warmup; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                auto timing = fcn.benchmark_fcn(kp, cfg.classifier_mode, diag_root, false);
                const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::cout << "[benchmark-fcn-warmup] call=" << i
                          << " chi2=" << std::setprecision(17) << timing.chi2
                          << " model_found=" << timing.model_found << "/" << targets.size()
                          << " total_sec=" << timing.total_sec
                          << " wall_sec=" << wall << "\n";
            }
            std::vector<double> total_secs, det_secs, cand_secs, cls_secs, assign_secs, chisq_secs, chi2s, wall_secs;
            std::vector<v32x_multiL::BenchmarkTiming> samples;
            samples.reserve(std::size_t(repeat));
            for(int i=0; i<repeat; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                auto timing = fcn.benchmark_fcn(kp, cfg.classifier_mode, diag_root, i + 1 == repeat);
                const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                wall_secs.push_back(wall);
                total_secs.push_back(timing.total_sec);
                det_secs.push_back(timing.determinant_scan_sec);
                cand_secs.push_back(timing.candidate_generation_sec);
                cls_secs.push_back(timing.classifier_sec);
                assign_secs.push_back(timing.root_assignment_sec);
                chisq_secs.push_back(timing.chisq_sec);
                chi2s.push_back(timing.chi2);
                samples.push_back(timing);
                std::cout << "[benchmark-fcn] call=" << i
                          << " chi2=" << std::setprecision(17) << timing.chi2
                          << " model_found=" << timing.model_found << "/" << targets.size()
                          << " total_sec=" << timing.total_sec
                          << " det_scan_sec=" << timing.determinant_scan_sec
                          << " candidate_generation_sec=" << timing.candidate_generation_sec
                          << " classifier_sec=" << timing.classifier_sec
                          << " assignment_sec=" << timing.root_assignment_sec
                          << " chisq_sec=" << timing.chisq_sec
                          << " wall_sec=" << wall << "\n";
            }
            auto mean = [](const std::vector<double>& v) {
                if(v.empty()) return 0.0;
                double s = 0.0; for(double x : v) s += x; return s / double(v.size());
            };
            auto median = [](std::vector<double> v) {
                if(v.empty()) return 0.0;
                std::sort(v.begin(), v.end());
                const std::size_t n = v.size();
                return (n % 2) ? v[n/2] : 0.5 * (v[n/2 - 1] + v[n/2]);
            };
            auto stdev = [](const std::vector<double>& v) {
                if(v.size() < 2) return 0.0;
                const double m = [&](){ double s=0.0; for(double x:v) s+=x; return s/double(v.size()); }();
                double ss = 0.0; for(double x:v) ss += (x-m)*(x-m); return std::sqrt(ss / double(v.size()-1));
            };
            auto maxv = [](const std::vector<double>& v) { return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()); };
            auto minv = [](const std::vector<double>& v) { return v.empty() ? 0.0 : *std::min_element(v.begin(), v.end()); };
            const double max_rss_mb = []() {
                struct rusage ru{};
                getrusage(RUSAGE_SELF, &ru);
                return double(ru.ru_maxrss) / 1024.0;
            }();
            const auto summary_path = run_csv_root / (cfg.classifier_mode + "_benchmark.csv");
            fs::create_directories(summary_path.parent_path());
            std::ofstream csv(summary_path);
            csv << std::setprecision(17);
            csv << "mode,eligible,target_count,model_found,chi2_first,chi2_last,avg_total_sec,median_total_sec,min_total_sec,max_total_sec,std_total_sec,avg_det_scan_sec,avg_candidate_generation_sec,avg_classifier_sec,avg_assignment_sec,avg_chisq_sec,cache_load_sec,precompute_sec,wall_time_external_sec,max_rss_mb,assignment_equivalent_to_reference,rank_by_avg_total,status\n";
            const double chi2_first = chi2s.empty() ? NAN : chi2s.front();
            const double chi2_last = chi2s.empty() ? NAN : chi2s.back();
            csv << cfg.classifier_mode << ",YES," << targets.size() << "," << (samples.empty() ? 0 : samples.back().model_found) << ","
                << chi2_first << "," << chi2_last << ","
                << mean(total_secs) << "," << median(total_secs) << "," << minv(total_secs) << "," << maxv(total_secs) << "," << stdev(total_secs) << ","
                << mean(det_secs) << "," << mean(cand_secs) << "," << mean(cls_secs) << "," << mean(assign_secs) << "," << mean(chisq_secs) << ","
                << fcn.cache_load_sec() << "," << fcn.precompute_sec() << ","
                << mean(wall_secs) << "," << max_rss_mb << ","
                << "PENDING,1,PASS\n";
            std::cout << "[benchmark-fcn-summary] classifier_mode=" << cfg.classifier_mode
                      << " repeat=" << repeat
                      << " avg_total_sec=" << mean(total_secs)
                      << " median_total_sec=" << median(total_secs)
                      << " min_total_sec=" << minv(total_secs)
                      << " max_total_sec=" << maxv(total_secs)
                      << " std_total_sec=" << stdev(total_secs)
                      << " avg_det_scan_sec=" << mean(det_secs)
                      << " avg_candidate_generation_sec=" << mean(cand_secs)
                      << " avg_classifier_sec=" << mean(cls_secs)
                      << " avg_assignment_sec=" << mean(assign_secs)
                      << " avg_chisq_sec=" << mean(chisq_secs)
                      << " cache_load_sec=" << fcn.cache_load_sec()
                      << " precompute_sec=" << fcn.precompute_sec()
                      << " max_rss_mb=" << max_rss_mb << "\n";
            return 0;
        }

        if(run_mode=="sensitivity") {
            const auto kv = v32w::read_kv(cfgpath);
            auto gd = [&](const std::string& key, double fallback) { return v32w::gd(kv,key,fallback); };
            const K3dfParameters start{cfg.base.guess.K3iso0,cfg.base.guess.K3iso1,cfg.base.guess.K3B,cfg.base.guess.K3E};
            const K3dfParameters best{
                gd("sensitivity_best_K3iso0",start.K3iso0), gd("sensitivity_best_K3iso1",start.K3iso1),
                gd("sensitivity_best_K3B",start.K3B), gd("sensitivity_best_K3E",start.K3E)};
            const std::array<double,4> steps{
                gd("sensitivity_step_K3iso0",cfg.base.step.K3iso0), gd("sensitivity_step_K3iso1",cfg.base.step.K3iso1),
                gd("sensitivity_step_K3B",cfg.base.step.K3B), gd("sensitivity_step_K3E",cfg.base.step.K3E)};
            const fs::path outpath = v32w::gs(kv,"sensitivity_output_csv",(fs::path(cfg.base.output_dir)/"parameter_sensitivity.csv").string());
            return fcn.write_parameter_sensitivity(start,best,steps,outpath);
        }

        if(run_mode=="fcn-once" || run_mode=="fcn") {
            std::vector<double> x = {cfg.base.guess.K3iso0, cfg.base.guess.K3iso1, cfg.base.guess.K3B, cfg.base.guess.K3E};
            const auto t0 = std::chrono::steady_clock::now();
            const double chi = fcn(x);
            const auto t1 = std::chrono::steady_clock::now();
            const double sec = std::chrono::duration<double>(t1 - t0).count();
            std::cout << "[fcn-once] chi2=" << std::setprecision(17) << chi
                      << " fcn_sec=" << sec
                      << " params=" << x[0] << " " << x[1] << " " << x[2] << " " << x[3] << "\n";
            return std::isfinite(chi) ? 0 : 2;
        }

        if(run_mode=="spectrum-only" || run_mode=="spectrum" || run_mode=="find-spectrum") {
            std::cout << "[v32x] spectrum-only mode: skipping Minuit; using K3df guesses from config as fixed parameters\n";
            K3dfParameters best{cfg.base.guess.K3iso0,cfg.base.guess.K3iso1,cfg.base.guess.K3B,cfg.base.guess.K3E};
            K3dfParameters err{NAN,NAN,NAN,NAN};
            std::vector<v32x_multiL::CandidateWithBlock> final_cands;
            auto model = fcn.model_for(best,&final_cands);
            int model_found=0; for(double m:model) if(std::isfinite(m)&&m>0.0) ++model_found;
            double final_chi2 = chi_square_v32f(targets,model,cov,corr,cfg.base.chi_square_mode,cfg.base.failure_penalty);
            int final_valid = (model_found==(int)targets.size() && std::isfinite(final_chi2) && final_chi2<cfg.base.failure_penalty/10.0) ? 1 : 0;
            if(!final_valid) final_chi2 = cfg.base.failure_penalty;
            MatrixD pcov = MatrixD::Zero(4,4);
            MatrixD pcorr = MatrixD::Zero(4,4);
            std::cout << "[spectrum-only] recomputed_chi2=" << std::setprecision(17) << final_chi2
                      << " model_levels_found=" << model_found << "/" << targets.size()
                      << " valid=" << final_valid << "\n";
            v32x_multiL::write_outputs_multi(cfg,mt,targets,model,final_cands,best,err,final_chi2,final_chi2,final_valid,model_found,cov,corr,pcov,pcorr);
            std::cout << "[v32x] spectrum-only done. summary=" << cfg.base.output_dir << "/" << cfg.base.output_tag << "_fit_summary_allL.dat\n";
            return final_valid ? 0 : 2;
        }

        #if V32F_HAS_MINUIT2
        ROOT::Minuit2::MnUserParameters u;
        if(cfg.float_params[0]) u.Add("K3iso0",cfg.base.guess.K3iso0,cfg.base.step.K3iso0);
        if(cfg.float_params[1]) u.Add("K3iso1",cfg.base.guess.K3iso1,cfg.base.step.K3iso1);
        if(cfg.float_params[2]) u.Add("K3B",cfg.base.guess.K3B,cfg.base.step.K3B);
        if(cfg.float_params[3]) u.Add("K3E",cfg.base.guess.K3E,cfg.base.step.K3E);
        if(cfg.base.use_parameter_limits) for(unsigned int i=0;i<4;++i) u.SetLimits(i,cfg.base.param_lower,cfg.base.param_upper);
        std::cout << "[v32x] running Minuit Migrad with multi-L fixed GPU coarse cache and CPU refined points\n";
        ROOT::Minuit2::MnMigrad migrad(fcn,u);
        const unsigned int max_fcn = (unsigned int)v32w::gi(v32w::read_kv(cfgpath),"max_fcn",0);
        ROOT::Minuit2::FunctionMinimum min = (max_fcn>0) ? migrad(max_fcn) : migrad();
        if(min.IsValid()) { ROOT::Minuit2::MnHesse h; h(fcn,min); }
        auto st = min.UserState();
        auto state_value = [&](const char* name, double fallback, bool floating) { return floating ? st.Value(name) : fallback; };
        auto state_error = [&](const char* name, bool floating) { return floating ? st.Error(name) : 0.0; };
        K3dfParameters best{
            state_value("K3iso0",cfg.base.guess.K3iso0,cfg.float_params[0]),
            state_value("K3iso1",cfg.base.guess.K3iso1,cfg.float_params[1]),
            state_value("K3B",cfg.base.guess.K3B,cfg.float_params[2]),
            state_value("K3E",cfg.base.guess.K3E,cfg.float_params[3])};
        K3dfParameters err{
            state_error("K3iso0",cfg.float_params[0]), state_error("K3iso1",cfg.float_params[1]),
            state_error("K3B",cfg.float_params[2]), state_error("K3E",cfg.float_params[3])};
        std::vector<v32x_multiL::CandidateWithBlock> final_cands;
        auto model = fcn.model_for(best,&final_cands);
        int model_found=0; for(double m:model) if(std::isfinite(m)&&m>0.0) ++model_found;
        double final_chi2 = chi_square_v32f(targets,model,cov,corr,cfg.base.chi_square_mode,cfg.base.failure_penalty);
        int final_valid = (min.IsValid() && model_found==(int)targets.size() && std::isfinite(final_chi2) && final_chi2<cfg.base.failure_penalty/10.0) ? 1 : 0;
        if(!final_valid) final_chi2 = cfg.base.failure_penalty;
        auto pcov = minuit_covariance_to_eigen_v32f(min,4);
        auto pcorr = covariance_to_correlation_v32f(pcov);
        std::cout << "[final] Minuit valid=" << (min.IsValid()?1:0) << " minuit_fval=" << min.Fval() << "\n";
        std::cout << "[final] recomputed_final_chi2=" << std::setprecision(17) << final_chi2 << " model_levels_found=" << model_found << "/" << targets.size() << " final_valid=" << final_valid << "\n";
        v32x_multiL::write_outputs_multi(cfg,mt,targets,model,final_cands,best,err,min.Fval(),final_chi2,final_valid,model_found,cov,corr,pcov,pcorr);
        std::cout << "[v32x] done. summary=" << cfg.base.output_dir << "/" << cfg.base.output_tag << "_fit_summary_allL.dat\n";
        return 0;
        #else
        std::cerr << "[v32x-error] fit mode requires Minuit2; rerun with spectrum-only in this build\n";
        return 3;
        #endif
    } catch(const std::exception& e) {
        std::cerr << "[v32x-error] " << e.what() << "\n";
        return 1;
    }
}
#endif
