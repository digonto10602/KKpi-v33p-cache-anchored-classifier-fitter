#include "K3df_minuit_fit_v32f_fullF3inv_QCfull_cached_classifier.hpp"
#include <Eigen/Dense>
#if V32F_HAS_MINUIT2
#include <Minuit2/FCNBase.h>
#include <Minuit2/FunctionMinimum.h>
#include <Minuit2/MnHesse.h>
#include <Minuit2/MnMigrad.h>
#include <Minuit2/MnUserParameters.h>
#endif
#include <algorithm>
#include <atomic>
#include <cctype>
#include <deque>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace v32w {
using namespace k3df_fit_v32f;

// v33a experimental switch: keep the old v32w/v32x refined classifier as the
// default, but allow a coarse-grid-only digonto_classifier_v3 window classifier.
// The parser sets these from config keys when present.
static std::string g_classifier_mode_v32w = "v2_refine"; // v2_refine or classifier aliases below
static double g_v3_mono_tol_v32w = 0.02;                  // relative margin for monotone tests
static double g_v3_min_drop_v32w = 0.15;                  // required fractional drop toward flip
static double g_v3_min_pole_rise_v32w = 0.15;             // required fractional rise toward flip
static int g_v3_require_both_shoulders_v32w = 0;          // 0 = one clean zero shoulder is enough
static double g_v4_merge_tol_v32w = 1.0e-3;               // collapse near-duplicate accepted roots

static bool is_v3_like_mode(const std::string& m) {
    return m=="digonto_v3_window" || m=="v3" || m=="digonto_classifier_v3" ||
           m=="raw_sign_only" || m=="algo_v7_eigenbranch" || m=="algo_v7_hybrid_det_eigenbranch" ||
           m=="eigenbranch" || m=="hybrid_det_eigenbranch";
}

static bool is_v4_like_mode(const std::string& m) {
    return m=="digonto_v4_window" || m=="v4" || m=="digonto_classifier_v4" ||
           m=="raw_sign_clustered" || m=="algo_v6_branch_count_final_select";
}

struct ClassifierDispatchInfo {
    std::string requested_mode;
    std::string resolved_mode;
    std::string implementation_tag;
    std::string version_tag;
    bool dispatch_unique = true;
    bool alias = false;
};

static ClassifierDispatchInfo classifier_dispatch_info(const std::string& mode) {
    if(mode=="raw_sign_only") return {mode, "raw_sign_only", "raw_sign_only::v33m", "v33m", true, false};
    if(mode=="raw_sign_clustered") return {mode, "raw_sign_clustered", "raw_sign_clustered::v33m", "v33m", true, false};
    if(mode=="algo_v6_branch_count_final_select") return {mode, "algo_v6_branch_count_final_select", "v6_branch_count::v33m", "v33m", true, false};
    if(mode=="algo_v7_eigenbranch") return {mode, "algo_v7_eigenbranch", "v7_eigenbranch::v33m", "v33m", true, false};
    if(mode=="algo_v7_hybrid_det_eigenbranch") return {mode, "algo_v7_hybrid_det_eigenbranch", "v7_hybrid::v33m", "v33m", true, false};
    if(mode=="eigenbranch") return {mode, "algo_v7_eigenbranch", "v7_eigenbranch::v33m", "v33m", false, true};
    if(mode=="hybrid_det_eigenbranch") return {mode, "algo_v7_hybrid_det_eigenbranch", "v7_hybrid::v33m", "v33m", false, true};
    if(mode=="v3" || mode=="digonto_classifier_v3") return {mode, "digonto_v3_window", "legacy::digonto_v3_window", "legacy", false, true};
    if(mode=="v4" || mode=="digonto_classifier_v4") return {mode, "digonto_v4_window", "legacy::digonto_v4_window", "legacy", false, true};
    if(mode=="digonto_v3_window") return {mode, "digonto_v3_window", "legacy::digonto_v3_window", "legacy", true, false};
    if(mode=="digonto_v4_window") return {mode, "digonto_v4_window", "legacy::digonto_v4_window", "legacy", true, false};
    return {mode, mode, "unknown", "v33m", true, false};
}

static std::map<std::string,std::string> read_kv(const std::string& path){std::ifstream in(path); if(!in) throw std::runtime_error("cannot open config "+path); std::map<std::string,std::string> kv; std::string line; auto trim=[](std::string s){while(!s.empty()&&std::isspace((unsigned char)s.front()))s.erase(s.begin()); while(!s.empty()&&std::isspace((unsigned char)s.back()))s.pop_back(); return s;}; while(std::getline(in,line)){auto h=line.find('#'); if(h!=std::string::npos) line=line.substr(0,h); auto e=line.find('='); if(e==std::string::npos) continue; kv[trim(line.substr(0,e))]=trim(line.substr(e+1));} return kv;}
static std::string gs(const std::map<std::string,std::string>& kv,const std::string& k,const std::string& d){auto it=kv.find(k); return it==kv.end()?d:it->second;}
static double gd(const std::map<std::string,std::string>& kv,const std::string& k,double d){auto it=kv.find(k); return it==kv.end()?d:std::stod(it->second);} static int gi(const std::map<std::string,std::string>& kv,const std::string& k,int d){auto it=kv.find(k); return it==kv.end()?d:std::stoi(it->second);} static std::vector<std::string> sws(std::string s){for(char& c:s) if(c==',') c=' '; std::istringstream is(s); std::vector<std::string> v; std::string x; while(is>>x)v.push_back(x); return v;}

static FitSettings settings_from_config(const std::map<std::string,std::string>& kv){FitSettings s; s.list_of_mom=sws(gs(kv,"list_of_mom","000_A1m 100_A2 110_A2 111_A2 200_A2")); s.Lval=gd(kv,"Lval",20); s.xival=gd(kv,"xival",3.444); s.scan_E0=gd(kv,"scan_E0",0.261); s.scan_E1=gd(kv,"scan_E1",0.36); s.coarseN=gi(kv,"coarseN",10000); s.refineN=gi(kv,"refineN",0); s.omp_threads=gi(kv,"omp_threads",18); s.debug=gs(kv,"debug","n").empty()?'n':gs(kv,"debug","n")[0]; s.atmpi=gd(kv,"atmpi",0.06906); s.atmK=gd(kv,"atmK",0.09698); s.eta_1=gd(kv,"eta_1",1.0); s.eta_2=gd(kv,"eta_2",0.5); s.alpha=gd(kv,"alpha",0.5); s.epsilon_h=gd(kv,"epsilon_h",0); s.max_shell_num=gd(kv,"max_shell_num",20); s.tolerance=gd(kv,"tolerance",1e-12); s.parity=gi(kv,"parity",-1); s.eig_tol=gd(kv,"eig_tol",0.05); s.norm_tol=gd(kv,"norm_tol",1e-12); s.proj_tol=gd(kv,"proj_tol",1e-10); s.waves_vec_1.clear(); for(auto& x:sws(gs(kv,"waves_vec_1","0 1"))) s.waves_vec_1.push_back(std::stoi(x)); s.waves_vec_2.clear(); for(auto& x:sws(gs(kv,"waves_vec_2","0"))) s.waves_vec_2.push_back(std::stoi(x)); for(int a=0;a<4;++a) for(int b=0;b<3;++b){s.scatter_params_1[a][b]=gd(kv,"scatter1_"+std::to_string(a)+std::to_string(b),s.scatter_params_1[a][b]); s.scatter_params_2[a][b]=gd(kv,"scatter2_"+std::to_string(a)+std::to_string(b),s.scatter_params_2[a][b]);}
 s.guess.K3iso0=gd(kv,"K3iso0_guess",gd(kv,"K3iso0",0.1)); s.guess.K3iso1=gd(kv,"K3iso1_guess",gd(kv,"K3iso1",0.1)); s.guess.K3B=gd(kv,"K3B_guess",gd(kv,"K3B",0.1)); s.guess.K3E=gd(kv,"K3E_guess",gd(kv,"K3E",0.1)); s.step.K3iso0=gd(kv,"K3iso0_step",0.1); s.step.K3iso1=gd(kv,"K3iso1_step",0.1); s.step.K3B=gd(kv,"K3B_step",0.1); s.step.K3E=gd(kv,"K3E_step",0.1); s.use_parameter_limits=gi(kv,"use_parameter_limits",0)!=0; s.param_lower=gd(kv,"param_lower",-1e8); s.param_upper=gd(kv,"param_upper",1e8);
 s.use_lattice_covariance=gi(kv,"use_lattice_covariance",1)!=0; s.ensemble=gs(kv,"ensemble","szscl21_20_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265"); s.energy_cutoff=gd(kv,"energy_cutoff",0.335); s.max_state=gi(kv,"max_state",8); s.masses_path=gs(kv,"masses_path",s.masses_path); s.print_found_files=gi(kv,"print_found_files",1)!=0; s.lattice_energy_type=gs(kv,"lattice_energy_type","En_lab"); s.chi_square_mode=gs(kv,"chi_square_mode","raw_cov_inv"); s.failure_penalty=gd(kv,"failure_penalty",1e100); s.print_each_fcn_eval=gi(kv,"print_each_fcn_eval",1)!=0; s.print_every_fcn_eval=gi(kv,"print_every_fcn_eval",1); s.output_dir=gs(kv,"output_dir","output_v32w_QC_fitter_norm_refine_v2"); s.output_tag=gs(kv,"output_tag","debug_v32w_QC_fitter_norm_refine_v2"); s.load_binary_f3inv_cache=gi(kv,"load_binary_f3inv_cache",1)!=0; s.save_binary_f3inv_cache=gi(kv,"save_binary_f3inv_cache",1)!=0; s.require_existing_binary_f3inv_cache=gi(kv,"require_existing_binary_f3inv_cache",0)!=0; s.binary_f3inv_cache_file=gs(kv,"binary_f3inv_cache_file","cache/cache.bin"); g_classifier_mode_v32w=gs(kv,"classifier_mode",gs(kv,"digonto_classifier_mode","v2_refine")); g_v3_mono_tol_v32w=gd(kv,"v3_monotone_tol",0.02); g_v3_min_drop_v32w=gd(kv,"v3_min_drop_fraction",0.15); g_v3_min_pole_rise_v32w=gd(kv,"v3_min_pole_rise_fraction",0.15); g_v3_require_both_shoulders_v32w=gi(kv,"v3_require_both_shoulders",0); g_v4_merge_tol_v32w=gd(kv,"v4_merge_tol",g_v4_merge_tol_v32w); return s;}

struct DetInfo{int ok=0; double det_re=NAN; double det_im=NAN; int sign=0; double logabs=NAN; double slog=NAN;};
static DetInfo det_info(const Eigen::MatrixXcd& M){DetInfo d; if(M.rows()==0||M.rows()!=M.cols()) return d; try{Eigen::PartialPivLU<Eigen::MatrixXcd> lu(M); comp dc=lu.determinant(); d.det_re=dc.real(); d.det_im=dc.imag(); d.sign=(d.det_re>0)?1:((d.det_re<0)?-1:0); const auto& LU=lu.matrixLU(); double l=0; bool ok=LU.rows()==LU.cols(); for(int i=0;i<LU.rows();++i){double a=std::abs(LU(i,i)); if(!(a>0)||!std::isfinite(a)){ok=false;break;} l+=std::log(a);} if(ok)d.logabs=l; else if(std::abs(dc)>0&&std::isfinite(std::abs(dc))) d.logabs=std::log(std::abs(dc)); if(d.sign && std::isfinite(d.logabs)){d.slog=d.sign*d.logabs; d.ok=1;}}catch(...){ } return d;}
static int sgn(double x){return (x>0)?1:((x<0)?-1:0);} static bool sameE(double a,double b,double tol=1e-13){return std::abs(a-b)<=tol*std::max(1.0,std::max(std::abs(a),std::abs(b)));}
static void progress_bar_line(const std::string& tag,int done,int total,int& nextpct,int step=10){ if(total<=0) return; int pct=int(100.0*double(done)/double(total)); if(pct>=nextpct || done==total){ int bars=pct/5; if(bars>20) bars=20; std::cout<<"["<<tag<<"] ["<<std::string(bars,'#')<<std::string(20-bars,'-')<<"] "<<pct<<"% ("<<done<<"/"<<total<<")\n"; while(nextpct<=pct) nextpct+=step; }}
static double cnorm(const FitSettings& s,double p){return std::pow(s.Lval*s.xival,p);} 
struct Eval{double E=0; int success=0; int proj_dim=0; DetInfo det; double y=NAN;};
static Eval eval_entry_QC(const ProjectedQCCacheEntry& e,const K3dfParameters& p,const PhysicsParams& par,char debug,double c){Eval v; v.E=e.Ecm; v.success=e.success; v.proj_dim=e.proj_dim; if(!e.success) return v; Eigen::MatrixXcd Q=assemble_QC(e,p,par,debug); if(Q.rows()>0){Q/=comp(c,0); v.det=det_info(Q); v.y=v.det.det_re;} return v;}
static bool flip(const Eval& a,const Eval& b){if(!a.success||!b.success||!a.det.ok||!b.det.ok||!std::isfinite(a.y)||!std::isfinite(b.y))return false; int sa=sgn(a.y), sb=sgn(b.y); return sa&&sb&&sa*sb<0;}
static double linzero(double x1,double y1,double x2,double y2){double d=y2-y1; if(std::isfinite(d)&&std::abs(d)>0){double z=x1-y1*(x2-x1)/d; if(std::isfinite(z)&&z>=std::min(x1,x2)&&z<=std::max(x1,x2)) return z;} return 0.5*(x1+x2);} 
struct Cand{std::string label; int init=-1,depth=0; double BL=0,BR=0,FL=0,FR=0,E=0; double yBL=NAN,yBR=NAN,yFL=NAN,yFR=NAN; double ysBL=NAN,ysBR=NAN,ysFL=NAN,ysFR=NAN; int merged_count=1; std::string kind="pole",reason="";};
struct QCSearchTiming {
    double determinant_scan_sec = 0.0;
    double candidate_generation_sec = 0.0;
    double classifier_sec = 0.0;
};

static ProjectedQCCacheEntry build_entry(int idx,double E,const IrrepCache& ic,const FitSettings& s,const PhysicsParams& par){return build_cache_entry(idx,E,ic.spec,s,par,s.debug);} 
static std::vector<std::pair<int,int>> flips(const std::vector<Eval>& v);
static std::vector<Eval> mesh(double BL,double BR,int n,const IrrepCache& coarse_ic,IrrepCache& refined_ic,const FitSettings& s,const PhysicsParams& par,const K3dfParameters& kp,double c,std::vector<ProjectedQCCacheEntry>& new_entries){
    std::vector<double> Es{BL,BR};
    for(int j=1;j<=n;++j) Es.push_back(BL+(BR-BL)*double(j)/double(n+1));
    std::sort(Es.begin(),Es.end());
    Es.erase(std::unique(Es.begin(),Es.end(),[](double a,double b){return sameE(a,b);}),Es.end());

    std::vector<ProjectedQCCacheEntry> ent(Es.size());
    std::vector<int> need(Es.size(),1);
    int reused=0;
    for(size_t i=0;i<Es.size();++i){
        for(const auto& e:coarse_ic.grid) if(sameE(Es[i],e.Ecm)){ent[i]=e; need[i]=0; ++reused; break;}
        if(need[i]) for(const auto& e:refined_ic.grid) if(sameE(Es[i],e.Ecm)){ent[i]=e; need[i]=0; ++reused; break;}
    }
    int need_count=0; for(int v:need) if(v) ++need_count;
    std::cout<<std::setprecision(17)
             <<"[refine-F3inv] irrep="<<coarse_ic.label<<" bracket=["<<BL<<","<<BR<<"]"
             <<" mesh_points="<<Es.size()<<" reused="<<reused<<" build="<<need_count<<"\n";

    std::atomic<int> done{0}; int nextpct=10;
    #pragma omp parallel for schedule(dynamic,1)
    for(int i=0;i<(int)Es.size();++i) if(need[i]){
        ent[i]=build_entry(-2000000-i,Es[i],coarse_ic,s,par);
        int d=++done;
        #pragma omp critical(v32w_refine_progress)
        { progress_bar_line("refine-F3inv-"+coarse_ic.label,d,need_count,nextpct,10); }
    }
    for(size_t i=0;i<Es.size();++i) {
        if(need[i]) {
            #pragma omp critical(v32w_cache_add)
            new_entries.push_back(ent[i]);
            refined_ic.grid.push_back(ent[i]);
        }
    }
    std::vector<Eval> out; out.reserve(Es.size());
    for(auto& e:ent) out.push_back(eval_entry_QC(e,kp,par,s.debug,c));
    std::sort(out.begin(),out.end(),[](const Eval&a,const Eval&b){return a.E<b.E;});
    auto fs_local=flips(out);
    std::cout<<"[refine-F3inv] irrep="<<coarse_ic.label<<" bracket signflips found="<<fs_local.size()<<"\n";
    return out;
}
static std::vector<std::pair<int,int>> flips(const std::vector<Eval>& v){std::vector<std::pair<int,int>> f; for(int i=0;i+1<(int)v.size();++i) if(flip(v[i],v[i+1])) f.push_back({i,i+1}); return f;}
static const Eval* nearest(const std::vector<Eval>& v,double E){if(v.empty()) return nullptr; int bi=0; double bd=std::abs(v[0].E-E); for(int i=1;i<(int)v.size();++i){double d=std::abs(v[i].E-E); if(d<bd){bd=d;bi=i;}} return &v[bi];}
static void classify(Cand& c,const std::vector<Eval>& pts,double zero_ratio){double m=0; for(auto& p:pts) if(p.success&&p.det.ok&&std::isfinite(p.y)) m=std::max(m,std::abs(p.y)); if(!(m>0)){c.kind="uncertain";c.reason="bad_local_scale";return;} auto pBL=nearest(pts,c.BL), pBR=nearest(pts,c.BR), pFL=nearest(pts,c.FL), pFR=nearest(pts,c.FR); if(!pBL||!pBR||!pFL||!pFR){c.kind="uncertain";c.reason="missing_points";return;} c.yBL=pBL->y;c.yBR=pBR->y;c.yFL=pFL->y;c.yFR=pFR->y;c.ysBL=c.yBL/m;c.ysBR=c.yBR/m;c.ysFL=c.yFL/m;c.ysFR=c.yFR/m; bool hL=c.BL<c.FL, hR=c.FR<c.BR; bool left=hL && std::abs(c.ysFL)<=zero_ratio*std::abs(c.ysBL); bool right=hR && std::abs(c.ysFR)<=zero_ratio*std::abs(c.ysBR); if(left||right){c.kind="true_zero"; c.reason=left&&right?"both_shoulders_decrease_toward_zero":(left?"left_shoulder_decreases_toward_zero":"right_shoulder_decreases_toward_zero");} else {c.kind="pole"; c.reason="no_shoulder_decreases_toward_zero";} c.E=linzero(c.FL,c.yFL,c.FR,c.yFR);} 

// ---------------- digonto_classifier_v3 coarse-window classifier ----------------
// The v3 classifier deliberately does not build refined points.  For a sign flip
// between i and i+1, it uses the local coarse window
//   left  shoulder: i-2, i-1, i
//   right shoulder: i+1, i+2, i+3
// and tests whether |det| decreases toward the flip.  A dynamic trimming step
// falls back from three points to the nearest two points if the three-point
// shoulder has a local extremum or a same-side sign inconsistency.
struct V3Shoulder { std::string kind="uncertain"; std::string reason=""; double score=0.0; int n=0; };
static bool v3_finite_eval(const Eval& e){return e.success&&e.det.ok&&std::isfinite(e.y)&&std::isfinite(e.E);} 
static bool v3_same_sign(double a,double b){int sa=sgn(a), sb=sgn(b); return sa&&sb&&sa==sb;}
static V3Shoulder v3_classify_shoulder(std::vector<double> vals_far_to_near, std::vector<double> signs_far_to_near, bool allow_trim){
    V3Shoulder r; if(vals_far_to_near.size()<2){r.reason="too_few_points";return r;}
    // Keep the nearest contiguous same-sign segment ending at the point adjacent to the flip.
    while(vals_far_to_near.size()>2 && !v3_same_sign(signs_far_to_near[vals_far_to_near.size()-2], signs_far_to_near.back())) {
        vals_far_to_near.erase(vals_far_to_near.begin()); signs_far_to_near.erase(signs_far_to_near.begin());
    }
    while(vals_far_to_near.size()>2 && !v3_same_sign(signs_far_to_near.front(), signs_far_to_near.back())) {
        vals_far_to_near.erase(vals_far_to_near.begin()); signs_far_to_near.erase(signs_far_to_near.begin());
    }
    if(vals_far_to_near.size()<2){r.reason="same_sign_trim_removed_window";return r;}
    auto absvec=[&](){std::vector<double> a; for(double v: vals_far_to_near) a.push_back(std::abs(v)); return a;};
    std::vector<double> a=absvec();
    // Dynamic three-to-two trimming: if the middle point is a local extremum, drop the far point and
    // judge the nearest two-point shoulder, matching the user's example y[i-2]>y[i-1] and y[i]>y[i-1].
    if(allow_trim && a.size()>=3) {
        const bool middle_local_min = (a[0] > a[1]*(1.0+g_v3_mono_tol_v32w) && a[2] > a[1]*(1.0+g_v3_mono_tol_v32w));
        const bool middle_local_max = (a[1] > a[0]*(1.0+g_v3_mono_tol_v32w) && a[1] > a[2]*(1.0+g_v3_mono_tol_v32w));
        if(middle_local_min || middle_local_max) {
            vals_far_to_near.erase(vals_far_to_near.begin()); signs_far_to_near.erase(signs_far_to_near.begin());
            a=absvec();
        }
    }
    r.n=(int)a.size();
    const double eps=1.0e-300;
    bool zero_like=true, pole_like=true;
    double zscore=1e300, pscore=1e300;
    for(size_t k=0;k+1<a.size();++k){
        const double far=a[k]+eps, near=a[k+1]+eps;
        const double drop=(far-near)/std::max(far,eps);
        const double rise=(near-far)/std::max(far,eps);
        zero_like = zero_like && (drop >= g_v3_min_drop_v32w);
        pole_like = pole_like && (rise >= g_v3_min_pole_rise_v32w);
        zscore=std::min(zscore,drop); pscore=std::min(pscore,rise);
    }
    if(zero_like){r.kind="zero"; r.score=zscore; r.reason=(r.n>=3?"monotone_abs_det_decrease_3pt":"monotone_abs_det_decrease_2pt"); return r;}
    if(pole_like){r.kind="pole"; r.score=pscore; r.reason=(r.n>=3?"monotone_abs_det_rise_3pt":"monotone_abs_det_rise_2pt"); return r;}
    r.reason="nonmonotone_or_weak_shoulders"; return r;
}
static std::vector<Cand> find_QC_zeros_v3_from_coarse_grid(const std::string& label,const std::vector<Eval>& grid,double zero_ratio,QCSearchTiming* timing=nullptr){
    std::vector<Cand> finals; int init=0;
    const auto cand_t0 = std::chrono::steady_clock::now();
    for(int i=0;i+1<(int)grid.size();++i){
        if(!flip(grid[i],grid[i+1])) continue;
        Cand c; c.label=label; c.init=init++; c.BL=(i>=2?grid[i-2].E:grid[i].E); c.BR=(i+3<(int)grid.size()?grid[i+3].E:grid[i+1].E); c.FL=grid[i].E; c.FR=grid[i+1].E; c.yFL=grid[i].y; c.yFR=grid[i+1].y; c.E=linzero(c.FL,c.yFL,c.FR,c.yFR);
        if(i<1 || i+2>=(int)grid.size()) { c.kind="uncertain"; c.reason="insufficient_v3_window"; finals.push_back(c); continue; }
        std::vector<double> local;
        for(int j=std::max(0,i-2); j<=std::min((int)grid.size()-1,i+3); ++j) if(v3_finite_eval(grid[j])) local.push_back(std::abs(grid[j].y));
        double m=0; for(double a:local) m=std::max(m,a); if(!(m>0)){c.kind="uncertain"; c.reason="bad_v3_local_scale"; finals.push_back(c); continue;}
        c.ysFL=c.yFL/m; c.ysFR=c.yFR/m; c.yBL=(i>=2?grid[i-2].y:grid[i].y); c.yBR=(i+3<(int)grid.size()?grid[i+3].y:grid[i+1].y); c.ysBL=c.yBL/m; c.ysBR=c.yBR/m;
        std::vector<double> lv,ls,rv,rs;
        for(int j=i-2;j<=i;++j) if(j>=0 && v3_finite_eval(grid[j])) { lv.push_back(grid[j].y/m); ls.push_back(grid[j].y); }
        for(int j=i+3;j>=i+1;--j) if(j<(int)grid.size() && v3_finite_eval(grid[j])) { rv.push_back(grid[j].y/m); rs.push_back(grid[j].y); }
        V3Shoulder L=v3_classify_shoulder(lv,ls,true), R=v3_classify_shoulder(rv,rs,true);
        const bool left_zero=(L.kind=="zero"), right_zero=(R.kind=="zero"), left_pole=(L.kind=="pole"), right_pole=(R.kind=="pole");
        if((g_v3_require_both_shoulders_v32w? (left_zero&&right_zero) : (left_zero||right_zero)) && !(left_pole&&right_pole)){
            c.kind="true_zero"; c.reason=std::string("v3_")+(left_zero&&right_zero?"both_zero_shoulders":(left_zero?"left_zero_shoulder":"right_zero_shoulder"))+"__L="+L.reason+"__R="+R.reason;
        } else if(left_pole || right_pole) {
            c.kind="pole"; c.reason=std::string("v3_pole_like__L=")+L.reason+"__R="+R.reason;
        } else {
            c.kind="uncertain"; c.reason=std::string("v3_uncertain__L=")+L.reason+"__R="+R.reason;
        }
        finals.push_back(c);
    }
    const auto cand_t1 = std::chrono::steady_clock::now();
    if(timing) timing->candidate_generation_sec += std::chrono::duration<double>(cand_t1 - cand_t0).count();
    std::sort(finals.begin(),finals.end(),[](const Cand&a,const Cand&b){return a.E<b.E;});
    int nz=0,np=0,nu=0; for(const auto& cc:finals){if(cc.kind=="true_zero")++nz;else if(cc.kind=="pole")++np;else++nu;}
    std::cout << "[classifier-v3] irrep=" << label << " candidates=" << finals.size() << " true_zero=" << nz << " pole=" << np << " uncertain=" << nu << " (coarse-window only; no refinement)\n";
    return finals;
}

static double cand_zero_proxy(const Cand& c) {
    double s = std::numeric_limits<double>::infinity();
    if (std::isfinite(c.ysFL)) s = std::min(s, std::abs(c.ysFL));
    if (std::isfinite(c.ysFR)) s = std::min(s, std::abs(c.ysFR));
    if (!std::isfinite(s)) s = std::numeric_limits<double>::infinity();
    return s;
}

static std::vector<Cand> merge_QC_zeros_v4(std::vector<Cand> finals, double merge_tol, QCSearchTiming* timing=nullptr) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<Cand> zeros;
    zeros.reserve(finals.size());
    for (auto& c : finals) {
        if (c.kind == "true_zero") zeros.push_back(c);
    }
    std::sort(zeros.begin(), zeros.end(), [](const Cand& a, const Cand& b) { return a.E < b.E; });

    std::vector<Cand> merged;
    for (const auto& c : zeros) {
        if (merged.empty() || std::abs(c.E - merged.back().E) > merge_tol) {
            Cand keep = c;
            keep.merged_count = 1;
            merged.push_back(std::move(keep));
            continue;
        }

        Cand& keep = merged.back();
        const double keep_score = cand_zero_proxy(keep);
        const double new_score = cand_zero_proxy(c);
        const int merged_count = keep.merged_count + 1;
        const double merged_E = (keep.E * double(keep.merged_count) + c.E) / double(merged_count);
        keep.E = merged_E;
        if (new_score < keep_score) {
            Cand rep = c;
            rep.E = merged_E;
            rep.merged_count = merged_count;
            rep.reason = keep.reason + "__v4_merge";
            merged.back() = std::move(rep);
        } else {
            keep.merged_count = merged_count;
            keep.reason += "__v4_merge";
        }
    }

    int nz = static_cast<int>(merged.size());
    std::cout << "[classifier-v4] merged true_zero=" << nz
              << " raw_true_zero=" << zeros.size()
              << " merge_tol=" << merge_tol << "\n";
    if(timing) timing->classifier_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return merged;
}

static std::vector<Cand> find_QC_zeros_refined(
        const IrrepCache& ic,IrrepCache& refined_ic,const FitSettings& s,const PhysicsParams& par,
        const K3dfParameters& kp,double c,int ninside,int maxdepth,
        double zero_ratio,std::vector<ProjectedQCCacheEntry>& new_entries,QCSearchTiming* timing=nullptr) {
    std::vector<Eval> grid(ic.grid.size());
    std::cout << "[QC-scan] irrep=" << ic.label
              << " evaluating projected QC determinant on cached grid rows=" << ic.grid.size() << "\n";

    const auto scan_t0 = std::chrono::steady_clock::now();
    std::atomic<int> grid_done{0};
    int grid_nextpct = 10;
    const int grid_total = static_cast<int>(ic.grid.size());
    #pragma omp parallel for schedule(dynamic,64)
    for(int i=0; i<grid_total; ++i) {
        grid[i] = eval_entry_QC(ic.grid[i],kp,par,s.debug,c);
        const int d = ++grid_done;
        if(d==1 || d==grid_total || d % std::max(1,grid_total/10)==0) {
            #pragma omp critical(v32w_qc_scan_progress)
            { progress_bar_line("QC-scan-" + ic.label,d,grid_total,grid_nextpct,10); }
        }
    }
    std::sort(grid.begin(),grid.end(),[](const Eval&a,const Eval&b){return a.E<b.E;});
    if(timing) timing->determinant_scan_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - scan_t0).count();

    if(is_v4_like_mode(g_classifier_mode_v32w)) {
        return merge_QC_zeros_v4(find_QC_zeros_v3_from_coarse_grid(ic.label,grid,zero_ratio,timing), g_v4_merge_tol_v32w, timing);
    }
    if(is_v3_like_mode(g_classifier_mode_v32w)) {
        return find_QC_zeros_v3_from_coarse_grid(ic.label,grid,zero_ratio,timing);
    }

    std::deque<Cand> q;
    int init=0;
    const auto cand_t0 = std::chrono::steady_clock::now();
    for(int i=0; i+1<(int)grid.size(); ++i) {
        if(flip(grid[i],grid[i+1])) {
            Cand b;
            b.label = ic.label;
            b.init = init++;
            b.BL = grid[i].E;
            b.BR = grid[i+1].E;
            q.push_back(b);
        }
    }
    if(timing) timing->candidate_generation_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - cand_t0).count();
    std::cout << "[QC-scan] irrep=" << ic.label
              << " initial signflip brackets=" << q.size() << "\n";

    std::vector<Cand> finals;
    const auto cls_t0 = std::chrono::steady_clock::now();
    while(!q.empty()) {
        Cand b = q.front();
        q.pop_front();
        auto pts = mesh(b.BL,b.BR,ninside,ic,refined_ic,s,par,kp,c,new_entries);
        auto fs = flips(pts);
        if(fs.empty()) {
            b.kind = "uncertain";
            b.reason = "lost_flip_after_refine";
            finals.push_back(b);
            continue;
        }
        if(fs.size()>1 && b.depth<maxdepth) {
            std::cout << "[refine-split] irrep=" << ic.label
                      << " parent=[" << b.BL << "," << b.BR << "] multiple_flips=" << fs.size()
                      << " depth=" << b.depth << " -> queue split brackets\n";
            for(size_t k=0; k<fs.size(); ++k) {
                const double left  = (k==0) ? b.BL : 0.5*(pts[fs[k-1].second].E + pts[fs[k].first].E);
                const double right = (k+1==fs.size()) ? b.BR : 0.5*(pts[fs[k].second].E + pts[fs[k+1].first].E);
                Cand nb;
                nb.label = ic.label;
                nb.init = b.init;
                nb.depth = b.depth + 1;
                nb.BL = left;
                nb.BR = right;
                q.push_back(nb);
            }
            continue;
        }
        for(auto pr: fs) {
            Cand cnd = b;
            cnd.FL = pts[pr.first].E;
            cnd.FR = pts[pr.second].E;
            classify(cnd,pts,zero_ratio);
            finals.push_back(cnd);
        }
    }
    if(timing) timing->classifier_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - cls_t0).count();

    std::sort(finals.begin(),finals.end(),[](const Cand&a,const Cand&b){return a.E<b.E;});
    int nz=0,np=0,nu=0;
    for(const auto& cc:finals) {
        if(cc.kind=="true_zero") ++nz;
        else if(cc.kind=="pole") ++np;
        else ++nu;
    }
    std::cout << "[classifier-v2] irrep=" << ic.label
              << " candidates=" << finals.size()
              << " true_zero=" << nz
              << " pole=" << np
              << " uncertain=" << nu
              << " newly_built=" << new_entries.size() << "\n";
    return finals;
}

class QCRefinedFCN final: public ROOT::Minuit2::FCNBase{public: QCRefinedFCN(FitSettings s_,std::vector<TargetLevel> t_,MatrixD cov_,MatrixD corr_,std::shared_ptr<std::vector<IrrepCache>> coarse_cache_,std::shared_ptr<std::vector<IrrepCache>> refined_cache_,PhysicsParams par_,int ninside_,int maxdepth_,double zratio_,double cnorm_,std::string refined_cache_path_):s(std::move(s_)),targets(std::move(t_)),cov(std::move(cov_)),corr(std::move(corr_)),coarse_cache(std::move(coarse_cache_)),refined_cache(std::move(refined_cache_)),par(std::move(par_)),ninside(ninside_),maxdepth(maxdepth_),zratio(zratio_),cn(cnorm_),refined_cache_path(std::move(refined_cache_path_)){} double Up() const override{return 1.0;} double operator()(const std::vector<double>& x) const override{if(x.size()<4) return s.failure_penalty; K3dfParameters kp{x[0],x[1],x[2],x[3]}; try{std::map<std::string,std::vector<Cand>> cmap; auto model=model_for(kp,&cmap); int found=0; for(double m:model) if(std::isfinite(m)&&m>0.0) ++found; double chi=chi_square_v32f(targets,model,cov,corr,s.chi_square_mode,s.failure_penalty); if(found<(int)targets.size()) chi=s.failure_penalty; size_t id=++evals; if(s.print_each_fcn_eval && s.print_every_fcn_eval>0 && id%(size_t)s.print_every_fcn_eval==0) { std::cout<<std::setprecision(17)<<"[v32w-FCN] eval="<<id<<" chi2="<<chi<<" model_found="<<found<<"/"<<targets.size()<<" K3iso0="<<kp.K3iso0<<" K3iso1="<<kp.K3iso1<<" K3B="<<kp.K3B<<" K3E="<<kp.K3E<<"\n"; for(const auto& kv:cmap){int nz=0,np=0,nu=0; for(const auto& cc:kv.second){ if(cc.kind=="true_zero") ++nz; else if(cc.kind=="pole") ++np; else ++nu; } std::cout<<"  [v32w-FCN] "<<kv.first<<" true_zero="<<nz<<" pole="<<np<<" uncertain="<<nu<<"\n";} } return std::isfinite(chi)?chi:s.failure_penalty;}catch(std::exception& e){std::cout<<"[v32w-FCN-warning] "<<e.what()<<"\n"; return s.failure_penalty;}} std::vector<double> model_for(const K3dfParameters& kp,std::map<std::string,std::vector<Cand>>* out) const{std::map<std::string,std::vector<Cand>> cmap; for(size_t ci=0; ci<coarse_cache->size(); ++ci){auto& ic=(*coarse_cache)[ci]; auto& ric=(*refined_cache)[ci]; std::vector<ProjectedQCCacheEntry> ne; auto c=find_QC_zeros_refined(ic,ric,s,par,kp,cn,ninside,maxdepth,zratio,ne); if(!ne.empty()){std::sort(ric.grid.begin(),ric.grid.end(),[](auto&a,auto&b){return a.Ecm<b.Ecm;});} cmap[ic.label]=std::move(c);} if(out) *out=cmap; std::vector<double> model(targets.size(),0.0); std::map<std::string,int> idx; for(size_t i=0;i<targets.size();++i){auto& t=targets[i]; std::vector<double> zeros; auto it=cmap.find(t.label); if(it!=cmap.end()) for(auto& c:it->second) if(c.kind=="true_zero") zeros.push_back(c.E); std::sort(zeros.begin(),zeros.end()); int k=idx[t.label]++; if(k>=0&&k<(int)zeros.size()) model[i]=zeros[k]; else model[i]=0.0;} return model;} void save_refined_cache_if_requested() const{ if(!refined_cache_path.empty()){ auto merged=*coarse_cache; for(size_t i=0;i<merged.size() && i<refined_cache->size();++i){ merged[i].grid.insert(merged[i].grid.end(),(*refined_cache)[i].grid.begin(),(*refined_cache)[i].grid.end()); std::sort(merged[i].grid.begin(),merged[i].grid.end(),[](auto&a,auto&b){return a.Ecm<b.Ecm;}); } save_binary_f3inv_cache_v32f(refined_cache_path,merged); } } private: FitSettings s; std::vector<TargetLevel> targets; MatrixD cov,corr; std::shared_ptr<std::vector<IrrepCache>> coarse_cache; std::shared_ptr<std::vector<IrrepCache>> refined_cache; PhysicsParams par; int ninside,maxdepth; double zratio,cn; std::string refined_cache_path; mutable std::atomic<size_t> evals{0};};


static void print_cache_summary(const std::vector<IrrepCache>& caches) {
    std::size_t total=0, ok=0, bad=0;
    std::cout << "[cache-summary] loaded irrep caches=" << caches.size() << "\n";
    for(const auto& ic: caches) {
        std::size_t lok=0,lbad=0;
        for(const auto& e: ic.grid) { ++total; if(e.success) {++ok;++lok;} else {++bad;++lbad;} }
        std::cout << "[cache-summary] " << ic.label << " rows=" << ic.grid.size()
                  << " success=" << lok << " failed=" << lbad;
        if(!ic.grid.empty()) std::cout << " Ecm_range=[" << ic.grid.front().Ecm << "," << ic.grid.back().Ecm << "]";
        std::cout << "\n";
    }
    std::cout << "[cache-summary] total rows=" << total << " success=" << ok << " failed=" << bad << "\n";
}


static std::pair<std::vector<IrrepCache>,std::vector<IrrepCache>> split_fixed_coarse_and_refined(const std::vector<IrrepCache>& loaded,const FitSettings& s){
    std::vector<IrrepCache> coarse, refined;
    const double dE = (s.coarseN>1) ? (s.scan_E1-s.scan_E0)/double(s.coarseN-1) : 0.0;
    const double tol = std::max(1e-10, std::abs(dE)*1e-3);
    std::cout << "[coarse-grid] separating fixed coarse grid from refined cache; coarseN="<<s.coarseN<<" tol="<<tol<<"\n";
    for(const auto& in: loaded){
        IrrepCache c; c.label=in.label; c.spec=in.spec;
        IrrepCache r; r.label=in.label; r.spec=in.spec;
        std::vector<char> used(in.grid.size(),0);
        for(int j=0;j<s.coarseN;++j){
            double Et = (s.coarseN>1) ? (s.scan_E0 + dE*double(j)) : s.scan_E0;
            int best=-1; double bd=1e300;
            for(int k=0;k<(int)in.grid.size();++k){ if(used[k]) continue; double dd=std::abs(in.grid[k].Ecm-Et); if(dd<bd){bd=dd; best=k;} }
            if(best>=0 && bd<=tol){ c.grid.push_back(in.grid[best]); used[best]=1; }
        }
        // Fallback: if the cache has exactly coarseN rows and matching failed because of formatting, use sorted rows.
        if((int)c.grid.size()!=s.coarseN && (int)in.grid.size()==s.coarseN){ c.grid=in.grid; std::fill(used.begin(),used.end(),1); }
        for(int k=0;k<(int)in.grid.size();++k) if(!used[k]) r.grid.push_back(in.grid[k]);
        std::sort(c.grid.begin(),c.grid.end(),[](auto&a,auto&b){return a.Ecm<b.Ecm;});
        std::sort(r.grid.begin(),r.grid.end(),[](auto&a,auto&b){return a.Ecm<b.Ecm;});
        std::cout << "[coarse-grid] "<<in.label<<" fixed_coarse_rows="<<c.grid.size()<<" refined_cache_rows="<<r.grid.size();
        if(!c.grid.empty()) std::cout << " coarse_Ecm=["<<c.grid.front().Ecm<<","<<c.grid.back().Ecm<<"]";
        std::cout << "\n";
        if((int)c.grid.size()!=s.coarseN){ std::cout << "[coarse-grid-warning] "<<in.label<<" fixed coarse rows != coarseN; initial signflip scan may be incomplete.\n"; }
        coarse.push_back(std::move(c)); refined.push_back(std::move(r));
    }
    return {coarse,refined};
}

static void print_split_cache_summary(const std::vector<IrrepCache>& coarse,const std::vector<IrrepCache>& refined){
    std::cout << "[cache-summary] fixed coarse/refined split summary\n";
    for(size_t i=0;i<coarse.size();++i){
        const auto& c=coarse[i]; const auto& r=refined[i];
        std::cout << "[cache-summary] "<<c.label<<" coarse_rows="<<c.grid.size()<<" refined_rows="<<r.grid.size()<<"\n";
    }
}

static void write_matrix(const std::string& path,const MatrixD& M,const std::string& hdr){std::ofstream f(path); f<<hdr<<"\n"<<std::setprecision(17); for(int i=0;i<M.rows();++i){for(int j=0;j<M.cols();++j) f<<(j?" ":"")<<M(i,j); f<<"\n";}}
static MatrixD cov_to_corr(const MatrixD& C){return covariance_to_correlation_v32f(C);} 
static void write_outputs(const FitSettings& s,const std::vector<TargetLevel>& targets,const std::vector<double>& model,const std::map<std::string,std::vector<Cand>>& cmap,const K3dfParameters& best,const K3dfParameters& err,double minuit_fval,double final_chi2,int valid,int model_found,const MatrixD& pcov,const MatrixD& pcorr){std::filesystem::create_directories(s.output_dir); std::ofstream sum(s.output_dir+"/"+s.output_tag+"_fit_summary.dat"); int nd=(int)targets.size(), np=4, ndof=nd-np; sum<<std::setprecision(17)<<"valid "<<valid<<"\nminuit_fval "<<minuit_fval<<"\nrecomputed_final_chi2 "<<final_chi2<<"\nchi2 "<<final_chi2<<"\nmodel_levels_found "<<model_found<<"\nndata "<<nd<<"\nnpar "<<np<<"\nndof "<<ndof<<"\nchi2_dof "<<(ndof>0?final_chi2/double(ndof):NAN)<<"\n"; sum<<"K3iso0 "<<best.K3iso0<<" err "<<err.K3iso0<<"\nK3iso1 "<<best.K3iso1<<" err "<<err.K3iso1<<"\nK3B "<<best.K3B<<" err "<<err.K3B<<"\nK3E "<<best.K3E<<" err "<<err.K3E<<"\n"; write_matrix(s.output_dir+"/"+s.output_tag+"_parameter_covariance.dat",pcov,"# rows/cols K3iso0 K3iso1 K3B K3E"); write_matrix(s.output_dir+"/"+s.output_tag+"_parameter_correlation.dat",pcorr,"# rows/cols K3iso0 K3iso1 K3B K3E"); std::ofstream lev(s.output_dir+"/"+s.output_tag+"_fit_levels.dat"); lev<<"# row label state lattice_Ecm lattice_err model_Ecm residual\n"<<std::setprecision(17); for(size_t i=0;i<targets.size();++i) lev<<i<<" "<<targets[i].label<<" "<<targets[i].state<<" "<<targets[i].Ecm<<" "<<targets[i].err<<" "<<model[i]<<" "<<(targets[i].Ecm-model[i])<<"\n"; std::ofstream spec(s.output_dir+"/"+s.output_tag+"_bestfit_QC_spectrum.dat"); spec<<"# label index Ecm kind reason BL BR FL FR yBL_scaled yBR_scaled yFL_scaled yFR_scaled merged_count\n"<<std::setprecision(17); for(auto& kv:cmap){int iz=0; for(auto& c:kv.second){if(c.kind=="true_zero") spec<<kv.first<<" "<<iz++<<" "<<c.E<<" "<<c.kind<<" "<<c.reason<<" "<<c.BL<<" "<<c.BR<<" "<<c.FL<<" "<<c.FR<<" "<<c.ysBL<<" "<<c.ysBR<<" "<<c.ysFL<<" "<<c.ysFR<<" "<<c.merged_count<<"\n";}} std::ofstream all(s.output_dir+"/"+s.output_tag+"_all_QC_candidates.dat"); all<<"# label E kind reason BL BR FL FR merged_count\n"<<std::setprecision(17); for(auto& kv:cmap) for(auto& c:kv.second) all<<kv.first<<" "<<c.E<<" "<<c.kind<<" "<<c.reason<<" "<<c.BL<<" "<<c.BR<<" "<<c.FL<<" "<<c.FR<<" "<<c.merged_count<<"\n";}

} // namespace
#ifndef V32W_NO_MAIN
int main(int argc,char** argv){try{std::string cfg=(argc>1?argv[1]:"configs/config_v32w_QC_fitter_norm_refine_v2.in"); auto kv=v32w::read_kv(cfg); auto s=v32w::settings_from_config(kv); int ninside=v32w::gi(kv,"v2_refine_points",10), maxdepth=v32w::gi(kv,"v2_max_split_depth",8); double zratio=v32w::gd(kv,"v2_zero_ratio",0.80), cpower=v32w::gd(kv,"const_norm_power",6.0); std::string refined_cache=v32w::gs(kv,"v2_refined_binary_cache_file","cache/v32w_refined_F3inv_Vsel_cache.bin"); std::filesystem::create_directories(s.output_dir);
#ifdef _OPENMP
    omp_set_num_threads(s.omp_threads);
    omp_set_dynamic(0);
    omp_set_max_active_levels(1);
#endif
    Eigen::setNbThreads(1); std::cout<<"[v32w] config="<<cfg<<"\n"; std::cout<<"[v32w] lattice_energy_type="<<s.lattice_energy_type<<" ensemble="<<s.ensemble<<"\n"; auto [targets,cov,corr]=k3df_fit_v32f::load_targets_and_covariance_v32f(s); std::cout<<"[v32w] targets="<<targets.size()<<" cov="<<cov.rows()<<"x"<<cov.cols()<<"\n"; auto loaded_cache=k3df_fit_v32f::get_or_build_F3inv_cache_v32f(s); v32w::print_cache_summary(loaded_cache); auto split=v32w::split_fixed_coarse_and_refined(loaded_cache,s); auto coarse_cache=std::make_shared<std::vector<k3df_fit_v32f::IrrepCache>>(std::move(split.first)); auto refined_cache_sp=std::make_shared<std::vector<k3df_fit_v32f::IrrepCache>>(std::move(split.second)); v32w::print_split_cache_summary(*coarse_cache,*refined_cache_sp); PhysicsParams par=k3df_fit_v32f::make_base_physics(s); double c=v32w::cnorm(s,cpower); std::cout<<"[v32w] QC normalization divides matrix by (Lbyas*xi)^"<<cpower<<" = "<<std::setprecision(17)<<c<<"\n"; v32w::QCRefinedFCN fcn(s,targets,cov,corr,coarse_cache,refined_cache_sp,par,ninside,maxdepth,zratio,c,refined_cache); ROOT::Minuit2::MnUserParameters u; u.Add("K3iso0",s.guess.K3iso0,s.step.K3iso0); u.Add("K3iso1",s.guess.K3iso1,s.step.K3iso1); u.Add("K3B",s.guess.K3B,s.step.K3B); u.Add("K3E",s.guess.K3E,s.step.K3E); if(s.use_parameter_limits) for(unsigned int i=0;i<4;++i) u.SetLimits(i,s.param_lower,s.param_upper); std::cout<<"[v32w] running Minuit Migrad with refined QC-zero classifier\n"; ROOT::Minuit2::MnMigrad migrad(fcn,u); auto min=migrad(); if(min.IsValid()){ROOT::Minuit2::MnHesse h; h(fcn,min);} auto st=min.UserState(); k3df_fit_v32f::K3dfParameters best{st.Value("K3iso0"),st.Value("K3iso1"),st.Value("K3B"),st.Value("K3E")}; k3df_fit_v32f::K3dfParameters err{st.Error("K3iso0"),st.Error("K3iso1"),st.Error("K3B"),st.Error("K3E")}; std::map<std::string,std::vector<v32w::Cand>> cmap; auto model=fcn.model_for(best,&cmap); auto pcov=k3df_fit_v32f::minuit_covariance_to_eigen_v32f(min,4); auto pcorr=k3df_fit_v32f::covariance_to_correlation_v32f(pcov); int model_found=0; for(double m:model) if(std::isfinite(m)&&m>0.0) ++model_found; double final_chi2=k3df_fit_v32f::chi_square_v32f(targets,model,cov,corr,s.chi_square_mode,s.failure_penalty); int final_valid=(min.IsValid() && model_found==(int)targets.size() && std::isfinite(final_chi2) && final_chi2<s.failure_penalty/10.0)?1:0; if(!final_valid) final_chi2=s.failure_penalty; std::cout<<"[final] Minuit valid="<<(min.IsValid()?1:0)<<" minuit_fval="<<min.Fval()<<"\n"; std::cout<<"[final] recomputed_final_chi2="<<final_chi2<<" model_levels_found="<<model_found<<"/"<<targets.size()<<" final_valid="<<final_valid<<"\n"; v32w::write_outputs(s,targets,model,cmap,best,err,min.Fval(),final_chi2,final_valid,model_found,pcov,pcorr); fcn.save_refined_cache_if_requested(); std::cout<<"[v32w] done. summary="<<s.output_dir<<"/"<<s.output_tag<<"_fit_summary.dat\n"; return 0;}catch(std::exception& e){std::cerr<<"[v32w-error] "<<e.what()<<"\n"; return 1;}}
#endif // V32W_NO_MAIN
