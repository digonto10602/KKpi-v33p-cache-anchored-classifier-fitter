#include "K3df_minuit_fit_v32f_fullF3inv_QCfull_cached_classifier.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

static std::map<std::string,std::string> read_kv_v32s(const std::string& path) {
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
static std::vector<std::string> split_ws(std::string s) { for(char& c:s) if(c==',') c=' '; std::istringstream is(s); std::vector<std::string> v; std::string x; while(is>>x) v.push_back(x); return v; }

static k3df_fit_v32f::FitSettings settings_from_config_v32s(const std::map<std::string,std::string>& kv) {
    using namespace k3df_fit_v32f;
    FitSettings s;
    s.list_of_mom = split_ws(gs(kv,"list_of_mom","000_A1m 100_A2 110_A2 111_A2 200_A2"));
    s.Lval = gd(kv,"Lval",s.Lval);
    s.xival = gd(kv,"xival",s.xival);
    s.scan_E0 = gd(kv,"scan_E0",s.scan_E0);
    s.scan_E1 = gd(kv,"scan_E1",s.scan_E1);
    s.coarseN = gi(kv,"coarseN",s.coarseN);
    s.refineN = gi(kv,"refineN",0);
    s.iterative_refine_enable = 0;
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
    s.output_dir = gs(kv,"output_dir",s.output_dir);
    s.output_tag = gs(kv,"output_tag",s.output_tag);
    s.write_cache_grid = gi(kv,"write_cache_grid",s.write_cache_grid?1:0)!=0;
    s.save_binary_f3inv_cache = gi(kv,"save_binary_f3inv_cache",s.save_binary_f3inv_cache?1:0)!=0;
    s.load_binary_f3inv_cache = gi(kv,"load_binary_f3inv_cache",s.load_binary_f3inv_cache?1:0)!=0;
    s.require_existing_binary_f3inv_cache = gi(kv,"require_existing_binary_f3inv_cache",s.require_existing_binary_f3inv_cache?1:0)!=0;
    s.binary_f3inv_cache_file = gs(kv,"binary_f3inv_cache_file",s.binary_f3inv_cache_file);
    return s;
}

struct DetInfo {
    int ok=0;
    double det_re=std::numeric_limits<double>::quiet_NaN();
    int sign=0;
    double logabsdet=std::numeric_limits<double>::quiet_NaN();
    double signed_logabsdet=std::numeric_limits<double>::quiet_NaN();
};
static DetInfo det_info(const Eigen::MatrixXcd& M) {
    DetInfo d;
    if(M.rows()==0 || M.rows()!=M.cols()) return d;
    try {
        Eigen::PartialPivLU<Eigen::MatrixXcd> lu(M);
        comp detc = lu.determinant();
        d.det_re = detc.real();
        d.sign = (d.det_re>0.0) ? 1 : ((d.det_re<0.0) ? -1 : 0);
        const auto& LU=lu.matrixLU();
        double lga=0.0; bool ok=true;
        for(int i=0;i<LU.rows();++i) { double a=std::abs(LU(i,i)); if(!(a>0.0) || !std::isfinite(a)) {ok=false; break;} lga += std::log(a); }
        if(ok) d.logabsdet=lga; else if(std::isfinite(std::abs(detc)) && std::abs(detc)>0.0) d.logabsdet=std::log(std::abs(detc));
        if(d.sign==0 && std::isfinite(detc.real())) d.sign = (detc.real()>0.0) ? 1 : ((detc.real()<0.0)?-1:0);
        if(d.sign!=0 && std::isfinite(d.logabsdet)) { d.signed_logabsdet = d.sign*d.logabsdet; d.ok=1; }
    } catch(...) {}
    return d;
}

static int sign_of(double x) { return (x>0.0) ? 1 : ((x<0.0) ? -1 : 0); }

struct EvalPoint {
    double E=0.0;
    int success=0;
    int proj_dim=0;
    int from_refinement=0;
    DetInfo det;
    double y=std::numeric_limits<double>::quiet_NaN();
};
struct InitialFlip { int id=-1; int il=-1, ir=-1; double El=0,Er=0; double yl=0,yr=0; };
struct WorkBracket { int init_id=-1; int depth=0; double BL=0, BR=0; double initL=0, initR=0; };
struct FinalBracket {
    int id=-1, init_id=-1, depth=0;
    double BL=0, BR=0;
    double flipL=0, flipR=0;
    double yBL=std::numeric_limits<double>::quiet_NaN(), yBR=std::numeric_limits<double>::quiet_NaN();
    double yFlipL=std::numeric_limits<double>::quiet_NaN(), yFlipR=std::numeric_limits<double>::quiet_NaN();
    double yBL_scaled=std::numeric_limits<double>::quiet_NaN(), yBR_scaled=std::numeric_limits<double>::quiet_NaN();
    double yFlipL_scaled=std::numeric_limits<double>::quiet_NaN(), yFlipR_scaled=std::numeric_limits<double>::quiet_NaN();
    double Ezero=std::numeric_limits<double>::quiet_NaN();
    std::string kind="uncertain";
    std::string reason="";
};

static double const_norm_scale_v32s(const k3df_fit_v32f::FitSettings& s, double power) {
    return std::pow(s.Lval*s.xival, power);
}
static EvalPoint eval_from_entry(const k3df_fit_v32f::ProjectedQCCacheEntry& e, double cnorm) {
    EvalPoint p; p.E=e.Ecm; p.success=e.success; p.proj_dim=e.proj_dim;
    if(e.success && e.F3inv_proj.rows()>0 && e.F3inv_proj.cols()>0) {
        Eigen::MatrixXcd M = e.F3inv_proj / comp(cnorm,0.0);
        p.det = det_info(M);
        p.y = p.det.det_re;
    }
    return p;
}
static bool signflip_eval(const EvalPoint& a, const EvalPoint& b) {
    if(!a.success || !b.success || !a.det.ok || !b.det.ok || !std::isfinite(a.y) || !std::isfinite(b.y)) return false;
    int sa=sign_of(a.y), sb=sign_of(b.y);
    return sa!=0 && sb!=0 && sa*sb<0;
}
static double linear_zero(double x1,double y1,double x2,double y2) {
    double den=y2-y1;
    if(std::isfinite(den) && std::abs(den)>0.0) {
        double z=x1-y1*(x2-x1)/den;
        if(std::isfinite(z) && z>=std::min(x1,x2) && z<=std::max(x1,x2)) return z;
    }
    return 0.5*(x1+x2);
}
static std::vector<InitialFlip> find_initial_flips(const std::vector<EvalPoint>& g) {
    std::vector<InitialFlip> out;
    for(size_t i=0;i+1<g.size();++i) {
        if(signflip_eval(g[i],g[i+1])) {
            InitialFlip f; f.id=(int)out.size(); f.il=(int)i; f.ir=(int)i+1; f.El=g[i].E; f.Er=g[i+1].E; f.yl=g[i].y; f.yr=g[i+1].y; out.push_back(f);
        }
    }
    return out;
}

static bool almost_same_E(double a,double b,double tol=1.0e-13) { return std::abs(a-b) <= tol*std::max(1.0,std::max(std::abs(a),std::abs(b))); }

static k3df_fit_v32f::ProjectedQCCacheEntry build_refined_entry_v32s(
        int idx, double E, const k3df_fit_v32f::IrrepCache& ic, const k3df_fit_v32f::FitSettings& s,
        const PhysicsParams& par) {
    return k3df_fit_v32f::build_cache_entry(idx,E,ic.spec,s,par,s.debug);
}

static std::vector<EvalPoint> make_refined_mesh_v32s(
        const WorkBracket& b,
        int n_inside,
        const k3df_fit_v32f::IrrepCache& ic,
        std::vector<k3df_fit_v32f::ProjectedQCCacheEntry>& new_entries,
        const k3df_fit_v32f::FitSettings& s,
        const PhysicsParams& par,
        double cnorm) {
    std::vector<double> Es;
    Es.push_back(b.BL); Es.push_back(b.BR);
    for(int j=1;j<=n_inside;++j) Es.push_back(b.BL + (b.BR-b.BL)*double(j)/double(n_inside+1));
    std::sort(Es.begin(),Es.end());
    Es.erase(std::unique(Es.begin(),Es.end(),[](double a,double c){return almost_same_E(a,c);}),Es.end());

    std::vector<k3df_fit_v32f::ProjectedQCCacheEntry> entries(Es.size());
    std::vector<int> need_build(Es.size(),1);
    for(size_t ie=0; ie<Es.size(); ++ie) {
        for(const auto& e: ic.grid) {
            if(almost_same_E(Es[ie],e.Ecm)) { entries[ie]=e; need_build[ie]=0; break; }
        }
    }
    #pragma omp parallel for schedule(dynamic)
    for(int ie=0; ie<(int)Es.size(); ++ie) {
        if(need_build[ie]) entries[ie] = build_refined_entry_v32s(-1000000-ie,Es[ie],ic,s,par);
    }
    std::vector<EvalPoint> pts;
    for(size_t ie=0; ie<Es.size(); ++ie) {
        if(need_build[ie]) {
            #pragma omp critical(v32s_new_entries)
            new_entries.push_back(entries[ie]);
        }
        EvalPoint p=eval_from_entry(entries[ie],cnorm);
        p.from_refinement=need_build[ie];
        pts.push_back(p);
    }
    std::sort(pts.begin(),pts.end(),[](const EvalPoint& a,const EvalPoint& b){return a.E<b.E;});
    return pts;
}

static std::vector<std::pair<int,int>> find_flips_in_pts(const std::vector<EvalPoint>& pts) {
    std::vector<std::pair<int,int>> flips;
    for(int i=0;i+1<(int)pts.size();++i) if(signflip_eval(pts[i],pts[i+1])) flips.push_back({i,i+1});
    return flips;
}
static const EvalPoint* nearest_point(const std::vector<EvalPoint>& pts, double E) {
    if(pts.empty()) return nullptr;
    int best=0; double bd=std::abs(pts[0].E-E);
    for(int i=1;i<(int)pts.size();++i){ double d=std::abs(pts[i].E-E); if(d<bd){bd=d; best=i;} }
    return &pts[best];
}
static void classify_final_v32s(FinalBracket& fb, const std::vector<EvalPoint>& pts, double zero_ratio, double pole_threshold) {
    double maxabs=0.0;
    for(const auto& p: pts) if(p.success && p.det.ok && std::isfinite(p.y)) maxabs=std::max(maxabs,std::abs(p.y));
    if(!(maxabs>0.0) || !std::isfinite(maxabs)) { fb.kind="uncertain"; fb.reason="bad_local_scale"; return; }
    const EvalPoint* pBL=nearest_point(pts,fb.BL);
    const EvalPoint* pBR=nearest_point(pts,fb.BR);
    const EvalPoint* pFL=nearest_point(pts,fb.flipL);
    const EvalPoint* pFR=nearest_point(pts,fb.flipR);
    if(!pBL||!pBR||!pFL||!pFR) { fb.kind="uncertain"; fb.reason="missing_edge_or_flip_points"; return; }
    fb.yBL=pBL->y; fb.yBR=pBR->y; fb.yFlipL=pFL->y; fb.yFlipR=pFR->y;
    fb.yBL_scaled=fb.yBL/maxabs; fb.yBR_scaled=fb.yBR/maxabs; fb.yFlipL_scaled=fb.yFlipL/maxabs; fb.yFlipR_scaled=fb.yFlipR/maxabs;
    const double edgeL=std::abs(fb.yBL_scaled), edgeR=std::abs(fb.yBR_scaled);
    const double nearL=std::abs(fb.yFlipL_scaled), nearR=std::abs(fb.yFlipR_scaled);
    // digonto_classifier_v2_one_shoulder:
    // If either available shoulder decreases toward zero as it approaches the
    // internal refined sign flip, classify as true_zero. This is intentionally
    // more permissive than the older two-sided test, and recovers cases where
    // the refined flip lands on/near one bracket edge and one shoulder is not
    // useful. If neither shoulder decreases toward zero, classify as pole.
    const bool has_left_shoulder  = (fb.BL < fb.flipL) && std::isfinite(edgeL) && std::isfinite(nearL);
    const bool has_right_shoulder = (fb.flipR < fb.BR) && std::isfinite(edgeR) && std::isfinite(nearR);
    const bool left_to_zero  = has_left_shoulder  && (nearL <= zero_ratio*edgeL);
    const bool right_to_zero = has_right_shoulder && (nearR <= zero_ratio*edgeR);

    if(left_to_zero || right_to_zero) {
        fb.kind="true_zero";
        if(left_to_zero && right_to_zero) fb.reason="both_shoulders_decrease_toward_zero";
        else if(left_to_zero) fb.reason="left_shoulder_decreases_toward_zero";
        else fb.reason="right_shoulder_decreases_toward_zero";
    }
    else {
        fb.kind="pole";
        if(!has_left_shoulder && !has_right_shoulder) fb.reason="no_valid_shoulders_no_zero_trend_classified_pole";
        else fb.reason="no_shoulder_decreases_toward_zero_classified_pole";
    }
}

struct IrrepResult {
    std::vector<EvalPoint> coarse;
    std::vector<InitialFlip> initial;
    std::vector<FinalBracket> finals;
    std::vector<k3df_fit_v32f::ProjectedQCCacheEntry> new_entries;
};

static IrrepResult refine_irrep_v32s(k3df_fit_v32f::IrrepCache& ic, const k3df_fit_v32f::FitSettings& s,
        const PhysicsParams& par, int n_inside, int max_depth, double zero_ratio,
        double pole_threshold, double cnorm) {
    IrrepResult R;
    R.coarse.reserve(ic.grid.size());
    for(const auto& e: ic.grid) R.coarse.push_back(eval_from_entry(e,cnorm));
    std::sort(R.coarse.begin(),R.coarse.end(),[](const auto& a,const auto& b){return a.E<b.E;});
    R.initial=find_initial_flips(R.coarse);
    std::deque<WorkBracket> q;
    for(const auto& f: R.initial) q.push_back({f.id,0,f.El,f.Er,f.El,f.Er});
    int final_id=0;
    while(!q.empty()) {
        WorkBracket b=q.front(); q.pop_front();
        if(!(b.BR>b.BL) || !std::isfinite(b.BL) || !std::isfinite(b.BR)) continue;
        auto pts=make_refined_mesh_v32s(b,n_inside,ic,R.new_entries,s,par,cnorm);
        auto flips=find_flips_in_pts(pts);
        if(flips.empty()) {
            FinalBracket fb; fb.id=final_id++; fb.init_id=b.init_id; fb.depth=b.depth; fb.BL=b.BL; fb.BR=b.BR;
            fb.flipL=std::numeric_limits<double>::quiet_NaN(); fb.flipR=std::numeric_limits<double>::quiet_NaN(); fb.kind="uncertain"; fb.reason="no_signflip_after_refinement";
            R.finals.push_back(fb); continue;
        }
        if(flips.size()==1 || b.depth>=max_depth) {
            for(const auto& fl: flips) {
                FinalBracket fb; fb.id=final_id++; fb.init_id=b.init_id; fb.depth=b.depth; fb.BL=b.BL; fb.BR=b.BR;
                fb.flipL=pts[fl.first].E; fb.flipR=pts[fl.second].E;
                fb.Ezero=linear_zero(pts[fl.first].E,pts[fl.first].y,pts[fl.second].E,pts[fl.second].y);
                classify_final_v32s(fb,pts,zero_ratio,pole_threshold);
                if(flips.size()>1 && b.depth>=max_depth) fb.reason += ";max_depth_with_multiple_flips";
                R.finals.push_back(fb);
            }
        } else {
            // Split one parent bracket into multiple child brackets, separated by midpoints between adjacent internal sign flips.
            for(size_t m=0;m<flips.size();++m) {
                double childL = b.BL;
                double childR = b.BR;
                if(m>0) childL = 0.5*(pts[flips[m-1].second].E + pts[flips[m].first].E);
                if(m+1<flips.size()) childR = 0.5*(pts[flips[m].second].E + pts[flips[m+1].first].E);
                if(childR>childL) q.push_back({b.init_id,b.depth+1,childL,childR,b.initL,b.initR});
            }
        }
        // Merge newly generated entries into the working cache for possible reuse by future brackets.
        if(!R.new_entries.empty()) {
            for(const auto& e: R.new_entries) ic.grid.push_back(e);
            R.new_entries.clear();
            k3df_fit_v32f::sort_unique_cache_grid_v32f(ic,1.0e-13);
            // keep a copy of all currently refined points in R.new_entries_all by not clearing? Simpler: save final ic later.
        }
    }
    return R;
}

static void write_lattice_targets_v32s(const k3df_fit_v32f::FitSettings& s, const std::vector<k3df_fit_v32f::TargetLevel>& targets) {
    std::filesystem::create_directories(s.output_dir);
    const std::string path = s.output_dir + "/" + s.output_tag + "_lattice_targets.dat";
    std::ofstream f(path); f << std::setprecision(17);
    f << "# v32s lattice targets. Default lattice_energy_type=En_lab, so Ecm=sqrt(E_read^2-atP^2) for moving frames.\n";
    f << "# columns: row Lbyas label irrep nPx nPy nPz state E_read err_read lattice_energy_type atP shifted_from_lab Ecm err\n";
    for(size_t i=0;i<targets.size();++i) {
        const auto& t=targets[i]; MomentumIrrepSpec spec=parse_label(t.label);
        f << i << ' ' << t.Lbyas << ' ' << t.label << ' ' << spec.irrep << ' ' << t.nP[0] << ' ' << t.nP[1] << ' ' << t.nP[2] << ' ' << t.state << ' ' << t.E_read << ' ' << t.err_read << ' ' << t.lattice_energy_type << ' ' << t.atP << ' ' << t.shifted_from_lab << ' ' << t.Ecm << ' ' << t.err << "\n";
    }
    std::cout << "[v32s-write] wrote lattice targets: " << path << " rows=" << targets.size() << "\n";
}

static void write_outputs_for_irrep(const k3df_fit_v32f::FitSettings& s, const k3df_fit_v32f::IrrepCache& ic,
        const IrrepResult& R, double cnorm, double norm_power) {
    const std::string base=s.output_dir + "/" + s.output_tag + "_" + k3df_fit_v32f::clean_label(ic.label);
    MomentumIrrepSpec spec=parse_label(ic.label);
    const std::string grid_path=base + "_projF3inv_det_grid.dat";
    std::ofstream gf(grid_path); gf << std::setprecision(17);
    gf << "# v32s projected F3inv determinant grid for " << ic.label << "\n";
    gf << "# Mscaled = (Vsel^dagger F3inv_full Vsel) / const_norm_scale, const_norm_scale=(Lbyas*xi)^" << norm_power << " = " << cnorm << "\n";
    gf << "# columns: i Ecm success proj_dim det_scaledProjF3inv signed_slogdet_scaledProjF3inv logabsdet_scaledProjF3inv det_sign_scaledProjF3inv\n";
    for(size_t i=0;i<R.coarse.size();++i) {
        const auto& r=R.coarse[i];
        gf << i << ' ' << r.E << ' ' << r.success << ' ' << r.proj_dim << ' ' << r.det.det_re << ' ' << r.det.signed_logabsdet << ' ' << r.det.logabsdet << ' ' << r.det.sign << "\n";
    }
    std::cout << "[v32s-write] wrote coarse grid: " << grid_path << " rows=" << R.coarse.size() << "\n";

    const std::string init_path=base + "_digonto_classifier_v2_initial_signflips.dat";
    std::ofstream inf(init_path); inf << std::setprecision(17);
    inf << "# columns: id label irrep il ir El Er det_l det_r\n";
    for(const auto& f: R.initial) inf << f.id << ' ' << ic.label << ' ' << spec.irrep << ' ' << f.il << ' ' << f.ir << ' ' << f.El << ' ' << f.Er << ' ' << f.yl << ' ' << f.yr << "\n";
    std::cout << "[v32s-write] wrote initial signflips: " << init_path << " count=" << R.initial.size() << "\n";

    const std::string ref_path=base + "_digonto_classifier_v2_refined_candidates.dat";
    std::ofstream rf(ref_path); rf << std::setprecision(17);
    rf << "# digonto_classifier_v2 true refinement: each sign-flip bracket gets new F3inv/Vsel at 10 interior Ecm points; multiple sign flips split recursively.\n";
    rf << "# columns: id label irrep init_id depth bracket_L bracket_R flip_L flip_R E_zero y_bracket_L_scaled y_flip_L_scaled y_flip_R_scaled y_bracket_R_scaled kind reason\n";
    for(const auto& c: R.finals) rf << c.id << ' ' << ic.label << ' ' << spec.irrep << ' ' << c.init_id << ' ' << c.depth << ' ' << c.BL << ' ' << c.BR << ' ' << c.flipL << ' ' << c.flipR << ' ' << c.Ezero << ' ' << c.yBL_scaled << ' ' << c.yFlipL_scaled << ' ' << c.yFlipR_scaled << ' ' << c.yBR_scaled << ' ' << c.kind << ' ' << c.reason << "\n";
    std::cout << "[v32s-write] wrote refined candidates: " << ref_path << " count=" << R.finals.size() << "\n";
}

int main(int argc, char** argv) {
    if(argc!=2) { std::cerr << "Usage: " << argv[0] << " configs/config_v32s_projF3inv_norm_refine_v2_Enlab.in\n"; return 1; }
    try {
        using namespace k3df_fit_v32f;
        auto kv=read_kv_v32s(argv[1]);
        FitSettings s=settings_from_config_v32s(kv);
        int refine_points=gi(kv,"v2_refine_points",10);
        int max_depth=gi(kv,"v2_max_split_depth",8);
        double zero_ratio=gd(kv,"v2_zero_ratio",0.80);
        double pole_threshold=gd(kv,"v2_pole_threshold",0.75);
        double norm_power=gd(kv,"const_norm_power",6.0);
        int save_refined=gi(kv,"v2_save_refined_binary_cache",1);
        std::string refined_cache_file=gs(kv,"v2_refined_binary_cache_file","cache/v32s_refined_F3inv_Vsel_cache.bin");
        const double cnorm=const_norm_scale_v32s(s,norm_power);
        std::filesystem::create_directories(s.output_dir);
        std::cout << "[v32s] projected F3inv determinant scan + true-refined digonto_classifier_v2\n";
        std::cout << "[v32s] lattice_energy_type=" << s.lattice_energy_type << " (default should be En_lab now), ensemble=" << s.ensemble << " Lbyas=" << s.Lval << " xi=" << s.xival << "\n";
        std::cout << "[v32s] const_norm_scale=(Lbyas*xi)^" << norm_power << " = " << cnorm << "; using Mscaled=projF3inv/const_norm_scale before determinant\n";
        std::cout << "[v32s] refinement: for every coarse sign flip, build " << refine_points << " new interior Ecm points, split recursively until one sign flip per bracket or max_depth=" << max_depth << "\n";
        std::cout << "[v32s] list_of_mom="; for(const auto& L:s.list_of_mom) std::cout << L << ' '; std::cout << "\n";
        std::cout << "[v32s] stage 1/4: loading lattice targets with cutoff in Ecm after En_lab conversion\n";
        auto [targets,cov,corr]=load_targets_and_covariance_v32f(s);
        write_lattice_targets_v32s(s,targets);
        std::cout << "[v32s] lattice targets kept=" << targets.size() << "\n";
        std::cout << "[v32s] stage 2/4: loading cached coarse F3inv_full and Vsel\n";
        auto caches=get_or_build_F3inv_cache_v32f(s);
        std::cout << "[v32s] cache irreps=" << caches.size() << "\n";
        PhysicsParams par=make_base_physics(s);
        std::cout << "[v32s] stage 3/4: determinant scan, true refinement, and classification per irrep\n";
        std::vector<IrrepResult> results(caches.size());
        for(size_t ci=0; ci<caches.size(); ++ci) {
            std::cout << "[v32s] irrep " << caches[ci].label << ": coarse grid rows=" << caches[ci].grid.size() << "\n";
            results[ci]=refine_irrep_v32s(caches[ci],s,par,refine_points,max_depth,zero_ratio,pole_threshold,cnorm);
            write_outputs_for_irrep(s,caches[ci],results[ci],cnorm,norm_power);
        }
        if(save_refined) {
            std::cout << "[v32s] stage 4/4: saving refined F3inv/Vsel cache for future reuse: " << refined_cache_file << "\n";
            save_binary_f3inv_cache_v32f(refined_cache_file,caches);
        } else {
            std::cout << "[v32s] stage 4/4: refined cache saving disabled\n";
        }
        std::cout << "[v32s] 100% complete\n";
    } catch(const std::exception& e) { std::cerr << "[v32s-error] " << e.what() << "\n"; return 2; }
    return 0;
}
