// v33f visual diagnostic for v33e coarseN=20000 caches.
//
// This executable reads the v33e F3inv_Vsel cache for each requested Lbyas/irrep,
// computes a scalar F3inv_isotropic diagnostic from Vsel^dagger F3inv Vsel,
// classifies its sign flips with digonto_classifier_v3, and writes data files for
// the n-scale plotting script.
//
// Usage:
//   ./bin/v33f_F3inv_iso_visual_test configs/config_v33f_F3inv_iso_visual_test.in
//
// It intentionally uses the F3inv_Vsel cache for the visual test because that is
// the most direct cache file that contains both F3inv and Vsel in the already
// validated v32zu/v33e binary layout.  The scalar diagnostic uses the legacy
// full-basis isotropic convention from the older F3 diagnostic: unit weight on
// the plm block and 1/sqrt(2) on the klm block, projected into the irrep basis.

#include "digonto_classifier_v3.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;
using comp = std::complex<double>;

static std::string trim(std::string s) {
    while(!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    while(!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    return s;
}
static std::map<std::string,std::string> read_kv(const std::string& path) {
    std::ifstream in(path);
    if(!in) throw std::runtime_error("cannot open config " + path);
    std::map<std::string,std::string> kv;
    std::string line;
    while(std::getline(in,line)) {
        const auto h=line.find('#');
        if(h!=std::string::npos) line=line.substr(0,h);
        const auto e=line.find('=');
        if(e==std::string::npos) continue;
        kv[trim(line.substr(0,e))]=trim(line.substr(e+1));
    }
    return kv;
}
static std::string gs(const std::map<std::string,std::string>& kv,const std::string& k,const std::string& d) { auto it=kv.find(k); return it==kv.end()?d:it->second; }
static int gi(const std::map<std::string,std::string>& kv,const std::string& k,int d) { auto it=kv.find(k); return it==kv.end()?d:std::stoi(it->second); }
static double gd(const std::map<std::string,std::string>& kv,const std::string& k,double d) { auto it=kv.find(k); return it==kv.end()?d:std::stod(it->second); }
static std::vector<std::string> words(std::string s) { for(char& c:s) if(c==',') c=' '; std::istringstream is(s); std::vector<std::string> v; std::string x; while(is>>x) v.push_back(x); return v; }
static std::vector<int> ints(std::string s) { std::vector<int> out; for(auto& w: words(s)) out.push_back(std::stoi(w)); return out; }
static std::string ltag(int L) { return std::to_string(L); }
static std::string xi_tag(double xi) {
    std::ostringstream os; os << std::fixed << std::setprecision(3) << xi;
    std::string s=os.str(); while(!s.empty() && s.back()=='0') s.pop_back(); if(!s.empty() && s.back()=='.') s.pop_back();
    for(char& c:s) if(c=='.') c='p'; return s;
}

template<class T> static T read_scalar(std::ifstream& is) { T x{}; is.read(reinterpret_cast<char*>(&x), sizeof(T)); if(!is) throw std::runtime_error("truncated cache"); return x; }

struct RowRaw {
    int grid_i=0, dim1=0, dim2=0, n=0, vdim=0;
    double Ecm=0.0, En=0.0;
    Eigen::MatrixXcd F3inv;
    Eigen::MatrixXcd Vsel;
};
struct RowDiag {
    int grid_i=0, total_dim=0, proj_dim=0;
    double Ecm=0.0, En=0.0;
    double F3inv_iso=std::numeric_limits<double>::quiet_NaN();
    double F3inv_proj00=std::numeric_limits<double>::quiet_NaN();
    double det_scaled_projF3inv=std::numeric_limits<double>::quiet_NaN();
    double signed_logabsdet_scaled=std::numeric_limits<double>::quiet_NaN();
};

static std::vector<RowRaw> read_merged_cache(const fs::path& cache_path) {
    std::ifstream is(cache_path, std::ios::binary);
    if(!is) throw std::runtime_error("cannot open cache " + cache_path.string());
    std::string header;
    std::getline(is,header);
    if(header != "KKPI_F3INV_VSEL_GPU_V32ZT_EXACT_V32Y 1") {
        throw std::runtime_error("unsupported F3inv_Vsel header in " + cache_path.string() + ": " + header);
    }
    const std::uint64_t rec_magic = 0x5653325a4f524543ULL;
    std::vector<RowRaw> rows;
    while(is.peek()!=EOF) {
        std::uint64_t magic = read_scalar<std::uint64_t>(is);
        if(magic != rec_magic) throw std::runtime_error("bad record magic in " + cache_path.string());
        RowRaw r;
        r.grid_i = read_scalar<std::int32_t>(is);
        r.dim1   = read_scalar<std::int32_t>(is);
        r.dim2   = read_scalar<std::int32_t>(is);
        r.n      = read_scalar<std::int32_t>(is);
        r.vdim   = read_scalar<std::int32_t>(is);
        r.Ecm    = read_scalar<double>(is);
        r.En     = read_scalar<double>(is);
        if(r.n<0 || r.vdim<0) throw std::runtime_error("negative dimensions in " + cache_path.string());
        r.F3inv.resize(r.n,r.n);
        r.Vsel.resize(r.n,r.vdim);
        for(int col=0; col<r.n; ++col) for(int row=0; row<r.n; ++row) {
            const double re=read_scalar<double>(is), im=read_scalar<double>(is);
            r.F3inv(row,col)=comp(re,im);
        }
        for(int col=0; col<r.vdim; ++col) for(int row=0; row<r.n; ++row) {
            const double re=read_scalar<double>(is), im=read_scalar<double>(is);
            r.Vsel(row,col)=comp(re,im);
        }
        rows.push_back(std::move(r));
    }
    std::sort(rows.begin(), rows.end(), [](const RowRaw& a, const RowRaw& b){ return a.Ecm < b.Ecm; });
    return rows;
}

static Eigen::VectorXcd isotropic_projected_vector(const RowRaw& r) {
    Eigen::VectorXcd vfull = Eigen::VectorXcd::Zero(r.n);
    const double klm_weight = 1.0 / std::sqrt(2.0);
    for(int i=0; i<r.dim1 && i<r.n; ++i) vfull(i) = 1.0;
    for(int i=r.dim1; i<r.dim1 + r.dim2 && i<r.n; ++i) vfull(i) = klm_weight;
    Eigen::VectorXcd u = r.Vsel.adjoint() * vfull;
    const double n = u.norm();
    if(!(n > 0.0) || !std::isfinite(n)) {
        u = Eigen::VectorXcd::Ones(r.vdim);
        const double n2 = u.norm();
        if(n2 > 0.0) u /= n2;
        return u;
    }
    return u / n;
}

static double determinant_real(const Eigen::MatrixXcd& M) {
    if(M.rows()==0 || M.rows()!=M.cols()) return std::numeric_limits<double>::quiet_NaN();
    Eigen::PartialPivLU<Eigen::MatrixXcd> lu(M);
    return lu.determinant().real();
}
static double signed_logabsdet_real(const Eigen::MatrixXcd& M) {
    if(M.rows()==0 || M.rows()!=M.cols()) return std::numeric_limits<double>::quiet_NaN();
    Eigen::PartialPivLU<Eigen::MatrixXcd> lu(M);
    const comp det = lu.determinant();
    const int s = (det.real()>0.0) ? 1 : ((det.real()<0.0) ? -1 : 0);
    if(!s) return std::numeric_limits<double>::quiet_NaN();
    double l=0.0;
    bool ok=true;
    const auto& LU = lu.matrixLU();
    for(int i=0; i<LU.rows(); ++i) {
        const double a = std::abs(LU(i,i));
        if(!(a>0.0) || !std::isfinite(a)) { ok=false; break; }
        l += std::log(a);
    }
    if(!ok) {
        const double a = std::abs(det);
        if(!(a>0.0) || !std::isfinite(a)) return std::numeric_limits<double>::quiet_NaN();
        l = std::log(a);
    }
    return s*l;
}

static RowDiag diagnose_row(const RowRaw& r, double scale) {
    RowDiag d;
    d.grid_i=r.grid_i; d.total_dim=r.n; d.proj_dim=r.vdim; d.Ecm=r.Ecm; d.En=r.En;
    if(r.vdim<=0) return d;
    Eigen::MatrixXcd P = r.Vsel.adjoint() * r.F3inv * r.Vsel;
    Eigen::VectorXcd u = isotropic_projected_vector(r);
    d.F3inv_iso = (u.adjoint() * P * u)(0,0).real();
    d.F3inv_proj00 = P(0,0).real();
    Eigen::MatrixXcd S = P / comp(scale,0.0);
    d.det_scaled_projF3inv = determinant_real(S);
    d.signed_logabsdet_scaled = signed_logabsdet_real(S);
    return d;
}

static fs::path find_cache(const fs::path& root, int L, double xi, const std::string& irrep, int coarseN) {
    const std::string fname = "v32zu_Lbyas" + ltag(L) + "_xi" + xi_tag(xi) + "_irrep" + irrep + "_coarse" + std::to_string(coarseN) + "_F3inv_Vsel_gpu.bin";
    std::vector<fs::path> hits;
    if(!fs::exists(root)) throw std::runtime_error("v33e_cache_root does not exist: " + root.string());
    for(const auto& de: fs::recursive_directory_iterator(root)) {
        if(de.is_regular_file() && de.path().filename() == fname) hits.push_back(de.path());
    }
    if(hits.empty()) throw std::runtime_error("missing v33e F3inv_Vsel cache " + fname + " under " + root.string());
    std::sort(hits.begin(), hits.end());
    return hits.front();
}

int main(int argc, char** argv) {
    try {
        const std::string cfgpath = (argc>1 ? argv[1] : "configs/config_v33f_F3inv_iso_visual_test.in");
        auto kv = read_kv(cfgpath);
        const fs::path root = gs(kv,"v33e_cache_root","/media/digonto/Data/F3inv_cache");
        const fs::path outdir = gs(kv,"output_dir","output_v33f/F3inv_iso_visual_test");
        const double xi = gd(kv,"xival",3.444);
        const int coarseN = gi(kv,"coarseN",20000);
        const auto Ls = ints(gs(kv,"Lbyas_values","20 24"));
        const auto irreps = words(gs(kv,"list_of_mom","000_A1m 100_A2 110_A2 111_A2 200_A2"));
        const int omp_threads = gi(kv,"omp_threads",18);
        digonto_v3::Params cp;
        cp.monotone_tol = gd(kv,"v3_monotone_tol",0.02);
        cp.min_drop_fraction = gd(kv,"v3_min_drop_fraction",0.15);
        cp.min_pole_rise_fraction = gd(kv,"v3_min_pole_rise_fraction",0.15);
        cp.require_both_shoulders = gi(kv,"v3_require_both_shoulders",0)!=0;
#ifdef _OPENMP
        omp_set_num_threads(omp_threads);
#endif
        fs::create_directories(outdir);
        std::ofstream manifest(outdir / "manifest_v33f_F3inv_iso_visual_test.txt");
        manifest << std::setprecision(17);
        manifest << "# cfg " << cfgpath << "\n# root " << root.string() << "\n";
        for(int L: Ls) {
            for(const std::string& irrep: irreps) {
                const fs::path cache = find_cache(root,L,xi,irrep,coarseN);
                const fs::path meta = cache.string() + ".meta.json";
                if(!fs::exists(meta)) throw std::runtime_error("missing sidecar for " + cache.string());
                std::cout << "[v33f-visual] L=" << L << " irrep=" << irrep << " cache=" << cache << "\n";
                auto rows = read_merged_cache(cache);
                const double scale = std::pow(double(L)*xi, 6.0);
                std::vector<RowDiag> diag(rows.size());
                #pragma omp parallel for schedule(dynamic,64)
                for(int i=0; i<static_cast<int>(rows.size()); ++i) {
                    diag[static_cast<std::size_t>(i)] = diagnose_row(rows[static_cast<std::size_t>(i)], scale);
                }
                std::sort(diag.begin(), diag.end(), [](const RowDiag& a, const RowDiag& b){ return a.Ecm < b.Ecm; });
                std::vector<double> x(diag.size()), y_iso(diag.size()), y_det(diag.size());
                for(std::size_t i=0; i<diag.size(); ++i) { x[i]=diag[i].Ecm; y_iso[i]=diag[i].F3inv_iso; y_det[i]=diag[i].det_scaled_projF3inv; }
                auto c_iso = digonto_v3::classify_series(x,y_iso,cp);
                auto c_det = digonto_v3::classify_series(x,y_det,cp);
                const std::string tag = "L" + ltag(L) + "_" + irrep;
                const fs::path data_path = outdir / ("v33f_F3inv_iso_" + tag + ".dat");
                const fs::path zero_path = outdir / ("v33f_F3inv_iso_" + tag + "_v3_truezeros.dat");
                const fs::path cand_path = outdir / ("v33f_F3inv_iso_" + tag + "_v3_all_candidates.dat");
                std::ofstream data(data_path), zeros(zero_path), cand(cand_path);
                data << std::setprecision(17)
                     << "# columns: grid_i Ecm En total_dim proj_dim F3inv_iso_rayleigh F3inv_proj00_real det_scaled_projF3inv_real signed_logabsdet_scaled\n";
                for(const auto& d: diag) {
                    data << d.grid_i << ' ' << d.Ecm << ' ' << d.En << ' ' << d.total_dim << ' ' << d.proj_dim << ' '
                         << d.F3inv_iso << ' ' << d.F3inv_proj00 << ' ' << d.det_scaled_projF3inv << ' ' << d.signed_logabsdet_scaled << "\n";
                }
                zeros << std::setprecision(17) << "# columns: Ecm label confidence left_index right_index E_left E_right y_left y_right reason\n";
                cand  << std::setprecision(17) << "# columns: Ecm label confidence left_index right_index E_left E_right y_left y_right reason\n";
                int nz=0,np=0,nu=0;
                for(const auto& c: c_iso) {
                    if(c.label=="true_zero") ++nz; else if(c.label=="pole") ++np; else ++nu;
                    cand << c.E_zero_linear << ' ' << c.label << ' ' << c.confidence << ' ' << c.left_index << ' ' << c.right_index << ' '
                         << c.E_left << ' ' << c.E_right << ' ' << c.y_left << ' ' << c.y_right << ' ' << c.reason << "\n";
                    if(c.label=="true_zero") {
                        zeros << c.E_zero_linear << ' ' << c.label << ' ' << c.confidence << ' ' << c.left_index << ' ' << c.right_index << ' '
                              << c.E_left << ' ' << c.E_right << ' ' << c.y_left << ' ' << c.y_right << ' ' << c.reason << "\n";
                    }
                }
                int det_nz=0; for(const auto& c: c_det) if(c.label=="true_zero") ++det_nz;
                manifest << tag << " rows " << diag.size() << " iso_candidates " << c_iso.size() << " iso_true_zero " << nz
                         << " iso_pole " << np << " iso_uncertain " << nu << " det_true_zero " << det_nz << " cache " << cache.string() << "\n";
                std::cout << "[v33f-visual] wrote " << data_path << " iso_true_zero=" << nz << " all_candidates=" << c_iso.size() << "\n";
            }
        }
        std::cout << "[v33f-visual] manifest=" << (outdir / "manifest_v33f_F3inv_iso_visual_test.txt") << "\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "[v33f-visual-error] " << e.what() << "\n";
        return 1;
    }
}
