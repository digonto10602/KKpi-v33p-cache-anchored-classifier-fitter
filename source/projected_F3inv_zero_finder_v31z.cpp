#include "K3df_minuit_fit_v31l_lattice_covariance.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#endif

using I3 = std::array<int,3>;
using Mat3i = std::array<std::array<int,3>,3>;

static std::string trim_v31z(std::string s){ while(!s.empty()&&std::isspace((unsigned char)s.front())) s.erase(s.begin()); while(!s.empty()&&std::isspace((unsigned char)s.back())) s.pop_back(); return s; }
static std::map<std::string,std::string> read_kv(const std::string& path){ std::ifstream in(path); if(!in) throw std::runtime_error("Could not open config: "+path); std::map<std::string,std::string> kv; std::string line; while(std::getline(in,line)){ auto h=line.find('#'); if(h!=std::string::npos) line=line.substr(0,h); auto e=line.find('='); if(e==std::string::npos) continue; std::string k=trim_v31z(line.substr(0,e)); std::string v=trim_v31z(line.substr(e+1)); if(!k.empty()) kv[k]=v; } return kv; }
static std::string gs(const std::map<std::string,std::string>& kv,const std::string& k,const std::string& d){ auto it=kv.find(k); return it==kv.end()?d:it->second; }
static int gi(const std::map<std::string,std::string>& kv,const std::string& k,int d){ auto it=kv.find(k); return it==kv.end()?d:std::stoi(it->second); }
static double gd(const std::map<std::string,std::string>& kv,const std::string& k,double d){ auto it=kv.find(k); return it==kv.end()?d:std::stod(it->second); }
static std::vector<std::string> split_ws(std::string s){ for(char& c:s) if(c==',') c=' '; std::istringstream is(s); std::vector<std::string> v; std::string x; while(is>>x) v.push_back(x); return v; }
static std::vector<int> parse_int_list(std::string s, std::vector<int> def={}){ auto p=split_ws(s); if(p.empty()) return def; std::vector<int> v; for(auto& x:p) v.push_back(std::stoi(x)); return v; }
static std::string int_list_str(const std::vector<int>& v){ std::string s; for(size_t i=0;i<v.size();++i){ if(i) s+=","; s+=std::to_string(v[i]); } return s; }
static std::string wave_tag(const std::vector<int>& v){ std::string s="waves"; for(int x:v) s+="_"+std::to_string(x); return s; }
static int sign_nonzero(double x){ return (x>0.0)-(x<0.0); }

struct SLogDet {
    std::complex<double> det = {NAN,NAN};
    std::complex<double> phase = {NAN,NAN};
    double logabs = NAN;
    double signed_logabs = NAN;
    int sign_phase_re = 0;
    int sign_det_re = 0;
    int singular = 0;
};
static SLogDet slogdet_lu(const Eigen::MatrixXcd& M){
    SLogDet out; if(M.rows()==0 || M.rows()!=M.cols()) return out;
    Eigen::PartialPivLU<Eigen::MatrixXcd> lu(M);
    out.det = lu.determinant(); out.sign_det_re=sign_nonzero(out.det.real());
    auto LU=lu.matrixLU(); std::complex<double> phase((double)lu.permutationP().determinant(),0.0); double logabs=0.0;
    for(int i=0;i<LU.rows();++i){ std::complex<double> z=LU(i,i); double az=std::abs(z); if(!(az>0.0)||!std::isfinite(az)){ out.singular=1; out.phase={0,0}; out.logabs=-std::numeric_limits<double>::infinity(); return out; } logabs+=std::log(az); phase*=z/az; }
    out.phase=phase; out.logabs=logabs; out.sign_phase_re=sign_nonzero(phase.real()); out.signed_logabs=(out.sign_phase_re==0?0.0:out.sign_phase_re*logabs); return out;
}

static SLogDet inverse_scaled_slogdet_from_matrix(const Eigen::MatrixXcd& M, double scale){
    // For A = scale * M^{-1}, det(A) = scale^N / det(M).
    // This avoids forming the full inverse just to plot det(F3^{-1}).
    SLogDet sd = slogdet_lu(M);
    SLogDet out = sd;
    if(M.rows()==0 || M.rows()!=M.cols() || sd.singular){ out.singular=1; return out; }
    out.logabs = -sd.logabs + double(M.rows())*std::log(std::max(scale,1e-300));
    out.phase = std::conj(sd.phase);
    out.sign_phase_re = sign_nonzero(out.phase.real());
    out.signed_logabs = (out.sign_phase_re==0?0.0:out.sign_phase_re*out.logabs);
    out.det = {NAN,NAN};
    out.sign_det_re = out.sign_phase_re;
    return out;
}

struct OptionsV31z {
    std::vector<std::string> labels={"110_A2"};
    std::vector<int> waves1={0,1};
    std::vector<int> waves2={0};
    int nmax=6, nsq_max=12;
    double energy_group_tol=1e-10;
    std::string outdir="output";
    std::string tag="debug_v31z_projF3inv_zero_110A2_L24";
    double zero_minabs_eig_max=1e-5;
    double zero_min_sv_max=1e-5;
    double candidate_merge_tol=2e-4;
    double oracle_match_tol=2e-4;
    int refine_iter=28;
    int smoke_only=0;
    int smoke_test=0;
    int audit_eigenbranch_detail=0;   // 0 = fast: use grid min eig/singular values; 1 = recompute endpoint eigenbranches in stage 4
    int refine_zero_roots=1;          // 1 = refine accepted blind zero candidates only
    int signflip_refine_enable=1;     // v31zm/v32: refine sign-flip brackets before writing/analyzing grid
    int signflip_refine_points=1000;  // legacy one-shot mode: number of interior mesh points per sign-flip bracket
    int iterative_signflip_refine_enable=1; // v32: 1 = repeatedly refine det(proj F3inv) sign-flip windows
    int iterative_refineN=10;         // v32: number of interior points per iterative refinement round
    int iterative_refine_max_rounds=40;
    double iterative_refine_tol=1.0e-6;
};

static k3df_fit_v31l::FitSettings settings_from_config(const std::string& cfg, OptionsV31z& opt){
    using namespace k3df_fit_v31l; auto kv=read_kv(cfg); FitSettings s;
    opt.labels=split_ws(gs(kv,"list_of_mom","110_A2")); s.list_of_mom=opt.labels;
    opt.waves1=parse_int_list(gs(kv,"waves_vec_1","0,1"),{0,1}); opt.waves2=parse_int_list(gs(kv,"waves_vec_2","0"),{0}); s.waves_vec_1=opt.waves1; s.waves_vec_2=opt.waves2;
    s.Lval=gd(kv,"Lval",24.0); s.xival=gd(kv,"xival",s.xival); s.scan_E0=gd(kv,"scan_E0",s.scan_E0); s.scan_E1=gd(kv,"scan_E1",s.scan_E1); s.coarseN=gi(kv,"coarseN",s.coarseN); s.omp_threads=gi(kv,"omp_threads",s.omp_threads); s.debug=gs(kv,"debug","n").empty()?'n':gs(kv,"debug","n")[0];
    s.atmpi=gd(kv,"atmpi",s.atmpi); s.atmK=gd(kv,"atmK",s.atmK); s.eta_1=gd(kv,"eta_1",s.eta_1); s.eta_2=gd(kv,"eta_2",s.eta_2); s.alpha=gd(kv,"alpha",s.alpha); s.epsilon_h=gd(kv,"epsilon_h",s.epsilon_h); s.max_shell_num=gd(kv,"max_shell_num",s.max_shell_num); s.tolerance=gd(kv,"tolerance",s.tolerance); s.parity=gi(kv,"parity",s.parity); s.eig_tol=gd(kv,"eig_tol",s.eig_tol); s.norm_tol=gd(kv,"norm_tol",s.norm_tol); s.proj_tol=gd(kv,"proj_tol",s.proj_tol); s.Q0norm=gi(kv,"Q0norm",s.Q0norm?1:0)!=0; s.sort_orbit_flag=gi(kv,"sort_orbit_flag",s.sort_orbit_flag?1:0)!=0;
    opt.nmax=gi(kv,"nonint_nmax",6); opt.nsq_max=gi(kv,"nonint_nsq_max",12); opt.energy_group_tol=gd(kv,"energy_group_tol",1e-10);
    opt.outdir=gs(kv,"output_dir",opt.outdir); opt.tag=gs(kv,"output_tag",opt.tag); s.output_dir=opt.outdir; s.output_tag=opt.tag;
    opt.zero_minabs_eig_max=gd(kv,"zero_minabs_eig_max",opt.zero_minabs_eig_max);
    opt.zero_min_sv_max=gd(kv,"zero_min_sv_max",opt.zero_min_sv_max);
    opt.candidate_merge_tol=gd(kv,"candidate_merge_tol",opt.candidate_merge_tol);
    opt.oracle_match_tol=gd(kv,"oracle_match_tol",opt.oracle_match_tol);
    opt.refine_iter=gi(kv,"refine_iter",opt.refine_iter);
    opt.smoke_only=gi(kv,"smoke_only",opt.smoke_only);
    opt.smoke_test=gi(kv,"smoke_test",opt.smoke_test);
    opt.audit_eigenbranch_detail=gi(kv,"audit_eigenbranch_detail",opt.audit_eigenbranch_detail);
    opt.refine_zero_roots=gi(kv,"refine_zero_roots",opt.refine_zero_roots);
    opt.signflip_refine_enable=gi(kv,"signflip_refine_enable",opt.signflip_refine_enable);
    opt.signflip_refine_points=gi(kv,"signflip_refine_points",opt.signflip_refine_points);
    opt.iterative_signflip_refine_enable=gi(kv,"iterative_signflip_refine_enable",opt.iterative_signflip_refine_enable);
    opt.iterative_refineN=gi(kv,"iterative_refineN",opt.iterative_refineN);
    opt.iterative_refine_max_rounds=gi(kv,"iterative_refine_max_rounds",opt.iterative_refine_max_rounds);
    opt.iterative_refine_tol=gd(kv,"iterative_refine_tol",opt.iterative_refine_tol);
    return s;
}


static void apply_scatter_params_from_config_v31zl(const std::string& cfg, PhysicsParams& par)
{
    // v31zl critical fix:
    // make_base_physics(settings) carries older KKpi defaults such as
    // scatter1_00=4.04, scatter1_10=-43.2, scatter2_00=4.12.
    // For these toy comparison runs the scattering parameters must be fully
    // controlled by the config.  Therefore we first reset all ERE coefficients
    // to zero, then apply only entries explicitly present in the config.
    auto kv = read_kv(cfg);

    par.scatter_params_1.assign(4, std::vector<comp>(3, comp(0.0,0.0)));
    par.scatter_params_2.assign(4, std::vector<comp>(3, comp(0.0,0.0)));

    for(int ell=0; ell<4; ++ell){
        for(int paramind=0; paramind<3; ++paramind){
            const std::string ij = std::to_string(ell) + std::to_string(paramind);

            // Accepted aliases.  All mean scatter_params_i_ell_paramind.
            const std::vector<std::string> keys1 = {
                "scatter1_" + ij,
                "scatterparams_1_" + ij,
                "scatter_params_1_" + ij
            };
            const std::vector<std::string> keys2 = {
                "scatter2_" + ij,
                "scatterparams_2_" + ij,
                "scatter_params_2_" + ij
            };

            for(const auto& k : keys1){
                auto it = kv.find(k);
                if(it != kv.end()) par.scatter_params_1[ell][paramind] = comp(std::stod(it->second),0.0);
            }
            for(const auto& k : keys2){
                auto it = kv.find(k);
                if(it != kv.end()) par.scatter_params_2[ell][paramind] = comp(std::stod(it->second),0.0);
            }
        }
    }
}

static void print_active_scatter_params_v31zl(const PhysicsParams& par)
{
    std::cout << "[v31zl-scatter] ACTIVE scattering parameters after config override and zero-reset:\n";
    std::cout << "[v31zl-scatter] convention: scatter_params_i_ell_paramind; i=spectator species; paramind 0=scattering length, 1=effective range, 2=shape/P term.\n";
    std::cout << "[v31zl-scatter] scatter_params_1 -> spectator M1, interacting pair (M1,M2).\n";
    std::cout << "[v31zl-scatter] scatter_params_2 -> spectator M2, interacting pair (M1,M1).\n";
    for(int ell=0; ell<4; ++ell){
        for(int paramind=0; paramind<3; ++paramind){
            std::cout << "[v31zl-scatter] scatter1_" << ell << paramind << "=" << par.scatter_params_1[ell][paramind].real()
                      << " scatter2_" << ell << paramind << "=" << par.scatter_params_2[ell][paramind].real() << "\n";
        }
    }
}

static I3 mat_vec(const Mat3i& R,const I3& v){ I3 o{0,0,0}; for(int i=0;i<3;++i) for(int j=0;j<3;++j) o[i]+=R[i][j]*v[j]; return o; }
static int det3(const Mat3i& M){ return M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])-M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])+M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]); }
static std::vector<Mat3i> signed_perm_mats(){ std::vector<Mat3i> out; std::array<int,3> p{0,1,2}; do{ for(int sx:{-1,1}) for(int sy:{-1,1}) for(int sz:{-1,1}){ Mat3i M{}; for(auto& r:M) r={0,0,0}; M[0][p[0]]=sx; M[1][p[1]]=sy; M[2][p[2]]=sz; out.push_back(M); } }while(std::next_permutation(p.begin(),p.end())); return out; }
static std::vector<Mat3i> little_group(const I3& nP){ std::vector<Mat3i> out; for(const auto& R:signed_perm_mats()) if(mat_vec(R,nP)==nP) out.push_back(R); return out; }
static std::string canonical_key(const I3& k1,const I3& k2,const I3& k3){ auto a=k1,b=k2; if(b<a) std::swap(a,b); std::ostringstream os; os<<a[0]<<","<<a[1]<<","<<a[2]<<"|"<<b[0]<<","<<b[1]<<","<<b[2]<<"|"<<k3[0]<<","<<k3[1]<<","<<k3[2]; return os.str(); }
static I3 parse_i3(std::string s){ for(char& c:s) if(c==',') c=' '; std::istringstream is(s); I3 a{}; is>>a[0]>>a[1]>>a[2]; return a; }
static std::string transform_key(const std::string& key,const Mat3i& R){ std::stringstream ss(key); std::string s1,s2,s3; std::getline(ss,s1,'|'); std::getline(ss,s2,'|'); std::getline(ss,s3,'|'); return canonical_key(mat_vec(R,parse_i3(s1)),mat_vec(R,parse_i3(s2)),mat_vec(R,parse_i3(s3))); }
static double char_irrep(const std::string& label,const I3& nP,const Mat3i& R){ std::string ir=label.substr(label.find('_')+1); if(ir=="A1"||ir=="A1m"||ir=="A1u") return 1.0; if(ir=="A2") return det3(R)>0?1.0:-1.0; if(ir=="E"||ir=="E2"){ if(det3(R)<0) return 0.0; int tr=R[0][0]+R[1][1]+R[2][2]; if(tr==3) return 2.0; if(nP==I3{1,1,1}) return -1.0; if(nP==I3{0,0,1}||nP==I3{0,0,2}) return (tr==-1?-2.0:0.0); return 0.0;} return 1.0; }
static int irrep_dim_local(const std::string& label){ std::string ir=label.substr(label.find('_')+1); return (ir=="E"||ir=="E2")?2:1; }
static double mom2(const I3& n){ return double(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]); }
static double oneE(double m,const I3& n,double xi,double L){ double p2=std::pow(2.0*M_PI/(xi*L),2)*mom2(n); return std::sqrt(m*m+p2); }
static double three_Ecm(double mK,double mpi,const I3& k1,const I3& k2,const I3& k3,const I3& nP,double xi,double L){ double Elab=oneE(mK,k1,xi,L)+oneE(mK,k2,xi,L)+oneE(mpi,k3,xi,L); double P2=std::pow(2.0*M_PI/(xi*L),2)*mom2(nP); double x=Elab*Elab-P2; return x>0?std::sqrt(x):NAN; }
struct Shell { double E=0; int raw_mult=0, group_deg=0, projected_dim=0; double char_sum=0; std::vector<std::string> keys; };
static std::vector<Shell> nonint_shells(const std::string& label,const I3& nP,double mK,double mpi,double xi,double L,int nmax,int nsq_max,double Emin,double Emax,double tol){
    std::map<long long,std::set<std::string>> groups; std::map<long long,double> eval;
    int outer_done=0; int outer_total=2*nmax+1;
    for(int a=-nmax;a<=nmax;++a){
        ++outer_done;
        if(outer_done==1 || outer_done==outer_total || outer_done%std::max(1,outer_total/5)==0)
            std::cout<<"[v31za-nonint] outer k1-x progress "<<outer_done<<"/"<<outer_total<<" a="<<a<<" current_groups="<<groups.size()<<"\n";
        for(int b=-nmax;b<=nmax;++b) for(int c=-nmax;c<=nmax;++c){ I3 k1{a,b,c}; if(mom2(k1)>nsq_max) continue; for(int d=-nmax;d<=nmax;++d) for(int e=-nmax;e<=nmax;++e) for(int f=-nmax;f<=nmax;++f){ I3 k2{d,e,f}; if(mom2(k2)>nsq_max) continue; I3 k3{nP[0]-k1[0]-k2[0],nP[1]-k1[1]-k2[1],nP[2]-k1[2]-k2[2]}; if(mom2(k3)>nsq_max) continue; double E=three_Ecm(mK,mpi,k1,k2,k3,nP,xi,L); if(!std::isfinite(E)||E<Emin-1e-12||E>Emax+1e-12) continue; long long bin=llround(E/tol); groups[bin].insert(canonical_key(k1,k2,k3)); eval[bin]=E; } }}
    auto G=little_group(nP); std::vector<Shell> shells;
    for(auto& kv:groups){ Shell sh; sh.E=eval[kv.first]; sh.keys.assign(kv.second.begin(),kv.second.end()); sh.raw_mult=int(sh.keys.size()); double sum=0.0; for(const auto& R:G){ int fixed=0; for(const auto& key:sh.keys) if(transform_key(key,R)==key) ++fixed; sum += char_irrep(label,nP,R)*double(fixed); } sh.char_sum=sum; double mult=sum/double(G.size()); sh.group_deg=int(std::llround(mult)); sh.projected_dim=sh.group_deg*irrep_dim_local(label); shells.push_back(sh); }
    std::sort(shells.begin(),shells.end(),[](const Shell&a,const Shell&b){return a.E<b.E;}); return shells;
}

static int ivec_get(const std::vector<std::vector<int>>& cfg,int ch,int i){ if(ch<0||ch>=int(cfg.size())||i<0||i>=int(cfg[ch].size())) return 999999; return cfg[ch][i]; }
static int comp_get_int(const std::vector<std::vector<comp>>& cfg,int ch,int i){ if(ch<0||ch>=int(cfg.size())||i<0||i>=int(cfg[ch].size())) return 999999; return int(std::llround(cfg[ch][i].real())); }
static std::string basis_label(int row,int A,const std::vector<std::vector<comp>>& plm,const std::vector<std::vector<int>>& np,const std::vector<std::vector<comp>>& klm,const std::vector<std::vector<int>>& nk){ bool p=row<A; int j=p?row:row-A; const auto& lm=p?plm:klm; const auto& mm=p?np:nk; std::ostringstream os; os<<(p?"plm":"klm")<<":row="<<row<<":local="<<j<<":n="<<ivec_get(mm,0,j)<<","<<ivec_get(mm,1,j)<<","<<ivec_get(mm,2,j)<<":ell="<<comp_get_int(lm,3,j)<<":m="<<comp_get_int(lm,4,j); return os.str(); }
static double rel_matrix_res(const Eigen::MatrixXcd& A,const Eigen::MatrixXcd& B){ double den=std::max(1.0,std::max(A.norm(),B.norm())); return (A-B).norm()/den; }
static double min_abs_eig(const Eigen::VectorXcd& v){ if(v.size()==0) return NAN; double m=std::numeric_limits<double>::infinity(); for(int i=0;i<v.size();++i) m=std::min(m,std::abs(v(i))); return m; }
static double max_abs_eig(const Eigen::VectorXcd& v){ if(v.size()==0) return NAN; double m=0; for(int i=0;i<v.size();++i) m=std::max(m,std::abs(v(i))); return m; }
static int min_abs_eig_index(const Eigen::VectorXcd& v){ int idx=-1; double b=std::numeric_limits<double>::infinity(); for(int i=0;i<v.size();++i){ double a=std::abs(v(i)); if(a<b){b=a; idx=i;} } return idx; }
static int nearest_eig_index(const Eigen::VectorXcd& v, comp z){ int idx=-1; double b=std::numeric_limits<double>::infinity(); for(int i=0;i<v.size();++i){ double a=std::abs(v(i)-z); if(a<b){b=a; idx=i;} } return idx; }
static int count_near_zero(const Eigen::VectorXcd& v,double tol){ int c=0; for(int i=0;i<v.size();++i) if(std::abs(v(i))<tol) ++c; return c; }

struct Eval {
    bool ok=false; int A=0,B=0,N=0,proj_dim=0; double Ecm=0,En=0; std::string err="OK";
    SLogDet sdetFullF3inv;
    SLogDet sdetProjF3inv;
    comp F3iso=comp(NAN,NAN), F3inv_iso=comp(NAN,NAN);
    double projF3norm=NAN, projF3invnorm=NAN, fullF3inv_logabs=NAN;
    double min_sv_projF3=NAN, max_sv_projF3=NAN, cond_projF3=NAN, min_sv_projF3inv=NAN, max_sv_projF3inv=NAN, cond_projF3inv=NAN;
    Eigen::VectorXcd eigProjF3inv; Eigen::MatrixXcd Vsel; std::vector<std::string> labels;
    double P_I_idem=NAN,P_I_herm=NAN,P_I_closure=NAN,Vsel_orth=NAN,Pproj_idem=NAN,Pproj_herm=NAN,Pproj_reconstruct=NAN,F3_equiv=NAN,F3_leak=NAN;
};

static Eval eval_detail(double Ecm,const MomentumIrrepSpec& spec,const PhysicsParams& par,const std::vector<int>& waves1,const std::vector<int>& waves2){
    Eval d; d.Ecm=Ecm;
    try{
        I3 nnP{spec.nnP[0],spec.nnP[1],spec.nnP[2]}; comp pi=std::acos(-1.0); double L=par.L(); comp twopibyL=comp(2.0,0.0)*pi/comp(L,0.0);
        std::vector<comp> total_P(3), nnP_config(3); for(int a=0;a<3;++a){ total_P[a]=twopibyL*double(nnP[a]); nnP_config[a]=comp(spec.nnP[a],0.0); }
        comp En_c=Ecm_to_E(comp(Ecm,0),total_P); d.En=En_c.real();
        std::vector<std::vector<comp>> plm(5),klm(5); std::vector<std::vector<int>> np(5),nk(5);
        config_maker_4_momentum_first(plm,np,waves1,En_c,total_P,par.atmK,par.atmK,par.atmpi,L,par.epsilon_h,par.max_shell_num,par.tolerance);
        config_maker_4_momentum_first(klm,nk,waves2,En_c,total_P,par.atmpi,par.atmK,par.atmK,L,par.epsilon_h,par.max_shell_num,par.tolerance);
        d.A=int(plm[0].size()); d.B=int(klm[0].size()); d.N=d.A+d.B; if(d.N<=0){ d.err="EMPTY_BASIS"; return d; }
        for(int r=0;r<d.N;++r) d.labels.push_back(basis_label(r,d.A,plm,np,klm,nk));
        Eigen::MatrixXcd F2(d.N,d.N),G(d.N,d.N),K2inv(d.N,d.N);
        F2_2plus1_mat(F2,En_c,plm,klm,total_P,par.atmK,par.atmpi,L,par.alpha,par.epsilon_h,par.max_shell_num,par.Q0norm);
        // v31zg corrected spectator-index convention:
        // scatter_params_i_ell_paramind means i = spectator species index.
        //   scatter_params_1: spectator M1, interacting pair (M1,M2), legacy block 1/plm
        //   scatter_params_2: spectator M2, interacting pair (M1,M1), legacy block 2/klm
        // paramind: 0=scattering length, 1=effective range, 2=next ERE shape/P term.
        // The legacy K2inv_EREord2_2plus1_mat already inserts scatter_params_1 and scatter_params_2
        // into the correct internal blocks for this code's basis ordering, so DO NOT swap them here.
        K2inv_EREord2_2plus1_mat(K2inv,par.eta_1,par.eta_2,par.scatter_params_1,par.scatter_params_2,En_c,plm,klm,total_P,par.atmK,par.atmpi,par.epsilon_h,L);
        G_2plus1_mat(G,En_c,plm,klm,total_P,par.atmK,par.atmpi,L,par.alpha,par.epsilon_h,par.max_shell_num,par.Q0norm);
        Eigen::MatrixXcd H=K2inv+F2+G; Eigen::PartialPivLU<Eigen::MatrixXcd> luH(H); Eigen::MatrixXcd X1=luH.solve(F2); Eigen::MatrixXcd F3=(F2/comp(3.0,0.0))-F2*X1;
        // v31zt p-wave-safe isotropic scalar diagnostic:
                // Legacy v_iso definition: all plm rows get weight 1 and all klm rows
        // get weight 1/sqrt(2), even when waves_vec_1 includes p-wave rows.
        // This intentionally preserves the old diagnostic convention requested by Digonto.
        Eigen::VectorXcd v_iso(d.N);
        for(int rr=0; rr<d.A; ++rr) v_iso(rr)=comp(1.0,0.0);
        for(int rr=d.A; rr<d.N; ++rr) v_iso(rr)=comp(1.0/std::sqrt(2.0),0.0);
        d.F3iso = (v_iso.transpose()*F3*v_iso)(0,0);
if(std::abs(d.F3iso)>0.0 && std::isfinite(d.F3iso.real()) && std::isfinite(d.F3iso.imag())) d.F3inv_iso = comp(1.0,0.0)/d.F3iso;
        CachedProjectorV30q pcache=get_projector_cached_v30q(plm,np,klm,nk,spec.irrep,nnP_config,par);
        d.Vsel=pcache.Vsel; d.proj_dim=int(d.Vsel.cols()); d.P_I_idem=pcache.P_I_idem_res; d.P_I_herm=pcache.P_I_herm_res; d.P_I_closure=pcache.fv_rep_closure_best; d.Vsel_orth=pcache.vsel_orth_res_after; d.Pproj_idem=pcache.Pproj_idem_res; d.Pproj_herm=pcache.Pproj_herm_res;
        fvproj_v30j::Convention fvconv; fvconv.index=0; fvconv.use_inverse_momentum=false; fvconv.use_inverse_D=false; fvconv.transpose_D=false; fvconv.order_WS=false; fvconv.parity_mode=1;
        Eigen::MatrixXcd P_I=fvproj_v30j::projector_for_convention(plm,np,klm,nk,spec.irrep,nnP_config,par.parity,fvconv); d.Pproj_reconstruct=rel_matrix_res(pcache.Pproj,P_I);
        if(d.proj_dim<=0){ d.err="ZERO_PROJECTED_DIM"; return d; }
        d.F3_equiv=fvproj_v30j::max_equivariance_over_little_group(F3,plm,np,klm,nk,nnP_config,par.parity,[](){ fvproj_v30j::Convention c; c.index=0; c.use_inverse_momentum=false; c.use_inverse_D=false; c.transpose_D=false; c.order_WS=false; c.parity_mode=1; return c; }());
        Eigen::MatrixXcd Pproj=d.Vsel*d.Vsel.adjoint(); Eigen::MatrixXcd MV=F3*d.Vsel; d.F3_leak=((Eigen::MatrixXcd::Identity(d.N,d.N)-Pproj)*MV).norm()/std::max(MV.norm(),1e-300);
        double f3_scale=0.0; for(int r=0;r<F3.rows();++r) for(int c=0;c<F3.cols();++c){ double a=std::abs(F3(r,c)); if(std::isfinite(a)&&a>f3_scale) f3_scale=a; }
        if(!(f3_scale>0.0)||!std::isfinite(f3_scale)){ d.err="BAD_F3_SCALE"; return d; }
        Eigen::MatrixXcd F3_scaled=F3/f3_scale;
        d.sdetFullF3inv = inverse_scaled_slogdet_from_matrix(F3_scaled, f3_scale);
        d.fullF3inv_logabs = d.sdetFullF3inv.logabs;
        Eigen::MatrixXcd projF3=d.Vsel.adjoint()*F3_scaled*d.Vsel; d.projF3norm=projF3.norm();
        Eigen::JacobiSVD<Eigen::MatrixXcd> svdF3(projF3); if(svdF3.singularValues().size()>0){ d.min_sv_projF3=svdF3.singularValues().minCoeff(); d.max_sv_projF3=svdF3.singularValues().maxCoeff(); d.cond_projF3=(d.min_sv_projF3>0?d.max_sv_projF3/d.min_sv_projF3:NAN); }
        Eigen::PartialPivLU<Eigen::MatrixXcd> luProjF3(projF3); Eigen::MatrixXcd projF3inv=luProjF3.solve(Eigen::MatrixXcd::Identity(d.proj_dim,d.proj_dim)); projF3inv*=f3_scale; d.projF3invnorm=projF3inv.norm(); d.sdetProjF3inv=slogdet_lu(projF3inv);
        Eigen::ComplexEigenSolver<Eigen::MatrixXcd> ces(projF3inv,false); if(ces.info()==Eigen::Success) d.eigProjF3inv=ces.eigenvalues();
        Eigen::JacobiSVD<Eigen::MatrixXcd> svdInv(projF3inv); if(svdInv.singularValues().size()>0){ d.min_sv_projF3inv=svdInv.singularValues().minCoeff(); d.max_sv_projF3inv=svdInv.singularValues().maxCoeff(); d.cond_projF3inv=(d.min_sv_projF3inv>0?d.max_sv_projF3inv/d.min_sv_projF3inv:NAN); }
        d.ok=true;
    }catch(const std::exception& e){ d.ok=false; d.err=e.what(); }
    return d;
}

struct Row { int i=0,success=0,A=0,B=0,N=0,proj_dim=0; double Ecm=0,En=0; SLogDet sdFull, sd; comp F3inv_iso=comp(NAN,NAN); Eigen::VectorXcd eigs; double minAbsEig=NAN,maxAbsEig=NAN,minSV=NAN,maxSV=NAN,cond=NAN,projF3norm=NAN,projF3invnorm=NAN,minSVprojF3=NAN,condProjF3=NAN; double Pidem=NAN,Pherm=NAN,Vorth=NAN,Precon=NAN,F3equiv=NAN,F3leak=NAN; std::string err="OK"; };
static Row eval_grid(int i,double E,const MomentumIrrepSpec& spec,const PhysicsParams& par,const std::vector<int>& w1,const std::vector<int>& w2){ Row r; r.i=i; r.Ecm=E; auto d=eval_detail(E,spec,par,w1,w2); r.success=d.ok?1:0; r.A=d.A; r.B=d.B; r.N=d.N; r.proj_dim=d.proj_dim; r.En=d.En; r.sdFull=d.sdetFullF3inv; r.sd=d.sdetProjF3inv; r.F3inv_iso=d.F3inv_iso; r.eigs=d.eigProjF3inv; r.minAbsEig=min_abs_eig(d.eigProjF3inv); r.maxAbsEig=max_abs_eig(d.eigProjF3inv); r.minSV=d.min_sv_projF3inv; r.maxSV=d.max_sv_projF3inv; r.cond=d.cond_projF3inv; r.projF3norm=d.projF3norm; r.projF3invnorm=d.projF3invnorm; r.minSVprojF3=d.min_sv_projF3; r.condProjF3=d.cond_projF3; r.Pidem=d.P_I_idem; r.Pherm=d.P_I_herm; r.Vorth=d.Vsel_orth; r.Precon=d.Pproj_reconstruct; r.F3equiv=d.F3_equiv; r.F3leak=d.F3_leak; r.err=d.err; return r; }



static bool sign_flip_v31zm(int s1, int s2)
{
    return (s1 != 0 && s2 != 0 && s1*s2 < 0);
}

static void add_refined_points_for_bracket_v31zm(std::vector<double>& Es, double a, double b, int npts)
{
    if(!(std::isfinite(a) && std::isfinite(b)) || !(b > a) || npts <= 0) return;
    for(int k=1; k<=npts; ++k)
    {
        const double t = double(k) / double(npts + 1);
        Es.push_back(a + t*(b-a));
    }
}

static void add_refined_signflip_mesh_v31zm(std::vector<Row>& rows,
                                            const MomentumIrrepSpec& spec,
                                            const PhysicsParams& par,
                                            const std::vector<int>& w1,
                                            const std::vector<int>& w2,
                                            int npts_per_bracket)
{
    if(rows.size() < 2 || npts_per_bracket <= 0)
    {
        for(int ii=0; ii<int(rows.size()); ++ii) rows[size_t(ii)].i = ii;
        return;
    }

    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b){ return a.Ecm < b.Ecm; });

    std::vector<double> refined_Es;
    int n_full_phase = 0;
    int n_proj_phase = 0;
    int n_proj_re = 0;
    int n_iso_re = 0;

    for(size_t i=0; i+1<rows.size(); ++i)
    {
        const Row& Lr = rows[i];
        const Row& Rr = rows[i+1];
        if(!Lr.success || !Rr.success) continue;
        if(!(Rr.Ecm > Lr.Ecm)) continue;

        bool add_this = false;

        if(sign_flip_v31zm(Lr.sdFull.sign_phase_re, Rr.sdFull.sign_phase_re))
        {
            add_this = true;
            ++n_full_phase;
        }
        if(sign_flip_v31zm(Lr.sd.sign_phase_re, Rr.sd.sign_phase_re))
        {
            add_this = true;
            ++n_proj_phase;
        }
        if(sign_flip_v31zm(Lr.sd.sign_det_re, Rr.sd.sign_det_re))
        {
            add_this = true;
            ++n_proj_re;
        }
        const int isoL = sign_nonzero(Lr.F3inv_iso.real());
        const int isoR = sign_nonzero(Rr.F3inv_iso.real());
        if(sign_flip_v31zm(isoL, isoR))
        {
            add_this = true;
            ++n_iso_re;
        }

        if(add_this)
            add_refined_points_for_bracket_v31zm(refined_Es, Lr.Ecm, Rr.Ecm, npts_per_bracket);
    }

    if(refined_Es.empty())
    {
        for(int ii=0; ii<int(rows.size()); ++ii) rows[size_t(ii)].i = ii;
        std::cout << "[v31zm-refine] no coarse sign-flip brackets found; rows=" << rows.size() << "\n";
        return;
    }

    std::sort(refined_Es.begin(), refined_Es.end());
    std::vector<double> unique_Es;
    unique_Es.reserve(refined_Es.size());
    constexpr double etol = 1e-13;
    for(double E : refined_Es)
    {
        if(!std::isfinite(E)) continue;
        if(unique_Es.empty() || std::abs(E - unique_Es.back()) > etol)
            unique_Es.push_back(E);
    }

    std::cout << "[v31zm-refine] coarse sign-flip brackets: full_phase=" << n_full_phase
              << " proj_phase=" << n_proj_phase
              << " proj_re=" << n_proj_re
              << " F3inv_iso_re=" << n_iso_re << "\n";
    std::cout << "[v31zm-refine] evaluating refined interior points=" << unique_Es.size()
              << " using npts_per_bracket=" << npts_per_bracket << "\n";

    std::vector<Row> refined_rows(unique_Es.size());
    int refined_done = 0;
    int refined_nextpct = 10;
    const int refined_total = int(unique_Es.size());
    std::cout << "[v31zt-refine-progress] [--------------------] 0% refined points start 0/" << refined_total << "\n";

    #pragma omp parallel for schedule(dynamic)
    for(int j=0; j<int(unique_Es.size()); ++j)
    {
        refined_rows[size_t(j)] = eval_grid(-1, unique_Es[size_t(j)], spec, par, w1, w2);

        #pragma omp critical(v31zt_refined_progress)
        {
            ++refined_done;
            const int pct = (refined_total > 0) ? int((100.0 * double(refined_done)) / double(refined_total)) : 100;
            while(pct >= refined_nextpct && refined_nextpct <= 100)
            {
                const int hashes = refined_nextpct / 5;
                std::cout << "[v31zt-refine-progress] [";
                for(int h=0; h<20; ++h) std::cout << (h < hashes ? "#" : "-");
                std::cout << "] " << refined_nextpct << "% refined points "
                          << refined_done << "/" << refined_total << "\n";
                refined_nextpct += 10;
            }
        }
    }

    const size_t coarse_count = rows.size();
    rows.insert(rows.end(), refined_rows.begin(), refined_rows.end());
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b){ return a.Ecm < b.Ecm; });

    std::vector<Row> merged;
    merged.reserve(rows.size());
    for(const Row& r : rows)
    {
        if(merged.empty() || std::abs(r.Ecm - merged.back().Ecm) > etol)
        {
            merged.push_back(r);
        }
        else
        {
            // If a duplicate somehow appears, keep a successful row over an unsuccessful one.
            if(!merged.back().success && r.success) merged.back() = r;
        }
    }
    rows.swap(merged);
    for(int ii=0; ii<int(rows.size()); ++ii) rows[size_t(ii)].i = ii;

    int ok = 0;
    for(const Row& r : rows) if(r.success) ++ok;
    std::cout << "[v31zm-refine] merged sorted grid: coarse=" << coarse_count
              << " refined_unique=" << unique_Es.size()
              << " final_rows=" << rows.size()
              << " success=" << ok << "/" << rows.size() << "\n";
}

static double interp_zero(double x1,double y1,double x2,double y2){ if(std::isfinite(y1)&&std::isfinite(y2)&&std::abs(y2-y1)>0) return x1 - y1*(x2-x1)/(y2-y1); return 0.5*(x1+x2); }

static double detproj_re_for_refine_v32(const Row& r)
{
    return r.sd.det.real();
}

static int detproj_sign_for_refine_v32(const Row& r)
{
    // digonto_classifier_v1 uses signDetRe from det(projected(F3^{-1})), so the C++ iterative refinement follows the same sign.
    return r.sd.sign_det_re;
}

static bool valid_detproj_bracket_v32(const Row& a, const Row& b)
{
    return a.success && b.success && (b.Ecm > a.Ecm) && sign_flip_v31zm(detproj_sign_for_refine_v32(a), detproj_sign_for_refine_v32(b));
}

static double interp_detproj_candidate_v32(const Row& a, const Row& b)
{
    return interp_zero(a.Ecm, detproj_re_for_refine_v32(a), b.Ecm, detproj_re_for_refine_v32(b));
}

struct V32RefinedBracketRecord
{
    int bracket_id=0;
    int converged=0;
    int rounds=0;
    double E_initial_left=NAN, E_initial_right=NAN;
    double E_final_left=NAN, E_final_right=NAN;
    double E_candidate_initial=NAN;
    double E_candidate_final=NAN;
    double last_delta=NAN;
    int sign_left=0, sign_right=0;
    int final_N_left=0, final_N_right=0;
    int final_proj_dim_left=0, final_proj_dim_right=0;
};

static void add_iterative_signflip_mesh_v32(std::vector<Row>& rows,
                                            const MomentumIrrepSpec& spec,
                                            const PhysicsParams& par,
                                            const std::vector<int>& w1,
                                            const std::vector<int>& w2,
                                            int refineN,
                                            int max_rounds,
                                            double tol,
                                            const std::string& report_path)
{
    if(rows.size() < 2 || refineN <= 0)
    {
        for(int ii=0; ii<int(rows.size()); ++ii) rows[size_t(ii)].i = ii;
        return;
    }

    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b){ return a.Ecm < b.Ecm; });

    std::vector<std::pair<Row,Row>> brackets;
    for(size_t i=0; i+1<rows.size(); ++i)
    {
        if(valid_detproj_bracket_v32(rows[i], rows[i+1]))
            brackets.push_back({rows[i], rows[i+1]});
    }

    std::cout << "[v32c-refine] iterative det(projected F3^{-1}) sign-flip refinement started\n";
    std::cout << "[v32c-refine] coarse_rows=" << rows.size()
              << " coarse_detProj_signflip_brackets=" << brackets.size()
              << " refineN=" << refineN
              << " max_rounds=" << max_rounds
              << " tol=" << tol << "\n";

    std::vector<Row> extra_rows;
    std::vector<V32RefinedBracketRecord> records;
    extra_rows.reserve(size_t(std::max(1,refineN))*size_t(std::max(1,int(brackets.size())))*size_t(std::max(1,max_rounds)));
    records.reserve(brackets.size());

    int bid = 0;
    for(auto br : brackets)
    {
        Row L = br.first;
        Row R = br.second;
        V32RefinedBracketRecord rec;
        rec.bracket_id = bid;
        rec.E_initial_left = L.Ecm;
        rec.E_initial_right = R.Ecm;
        rec.E_candidate_initial = interp_detproj_candidate_v32(L,R);
        rec.E_candidate_final = rec.E_candidate_initial;
        rec.sign_left = detproj_sign_for_refine_v32(L);
        rec.sign_right = detproj_sign_for_refine_v32(R);
        double prev_candidate = rec.E_candidate_initial;

        std::cout << "[v32c-refine] bracket " << bid
                  << " initial_window=[" << L.Ecm << "," << R.Ecm << "]"
                  << " E_candidate_initial=" << prev_candidate
                  << " orientation=" << rec.sign_left << "->" << rec.sign_right << "\n";

        for(int round=1; round<=max_rounds; ++round)
        {
            std::vector<Row> local;
            local.resize(size_t(refineN)+2);
            local[0] = L;
            local[size_t(refineN)+1] = R;

            // v32c: evaluate the refineN interior grid points in parallel.
            // Each OpenMP thread writes to a unique local[k] slot; after the
            // parallel loop, the serial thread appends those rows to extra_rows.
            #pragma omp parallel for schedule(dynamic)
            for(int k=1; k<=refineN; ++k)
            {
                const double t = double(k)/double(refineN+1);
                const double E = L.Ecm + t*(R.Ecm-L.Ecm);
                local[size_t(k)] = eval_grid(-1, E, spec, par, w1, w2);
            }
            for(int k=1; k<=refineN; ++k)
                extra_rows.push_back(local[size_t(k)]);

            std::sort(local.begin(), local.end(), [](const Row& a, const Row& b){ return a.Ecm < b.Ecm; });

            int best_i = -1;
            double best_dist = std::numeric_limits<double>::infinity();
            double best_cand = NAN;
            int nflip = 0;
            for(size_t i=0; i+1<local.size(); ++i)
            {
                if(!valid_detproj_bracket_v32(local[i], local[i+1])) continue;
                ++nflip;
                double cand = interp_detproj_candidate_v32(local[i], local[i+1]);
                double dist = std::abs(cand - prev_candidate);
                if(dist < best_dist)
                {
                    best_dist = dist;
                    best_i = int(i);
                    best_cand = cand;
                }
            }

            if(best_i < 0)
            {
                std::cout << "[v32c-refine] bracket " << bid << " round=" << round
                          << " no_detProj_signflip_found_inside_window; keeping previous candidate=" << prev_candidate << "\n";
                rec.rounds = round;
                break;
            }

            L = local[size_t(best_i)];
            R = local[size_t(best_i+1)];
            const double delta = std::abs(best_cand - prev_candidate);
            rec.E_candidate_final = best_cand;
            rec.E_final_left = L.Ecm;
            rec.E_final_right = R.Ecm;
            rec.last_delta = delta;
            rec.rounds = round;
            rec.sign_left = detproj_sign_for_refine_v32(L);
            rec.sign_right = detproj_sign_for_refine_v32(R);
            rec.final_N_left = L.N;
            rec.final_N_right = R.N;
            rec.final_proj_dim_left = L.proj_dim;
            rec.final_proj_dim_right = R.proj_dim;

            const int pct = std::min(100, int(std::round(100.0*double(round)/double(std::max(1,max_rounds)))));
            std::cout << "[v32c-refine] bracket=" << bid
                      << " round=" << round
                      << " approx_progress=" << pct << "%"
                      << " local_signflips=" << nflip
                      << " window=[" << L.Ecm << "," << R.Ecm << "]"
                      << " E_candidate=" << best_cand
                      << " delta_from_previous=" << delta << "\n";

            if(delta < tol)
            {
                rec.converged = 1;
                std::cout << "[v32c-refine] bracket " << bid
                          << " converged: |E_prev-E_present|=" << delta
                          << " < tol=" << tol
                          << " final_candidate=" << best_cand << "\n";
                break;
            }
            prev_candidate = best_cand;
        }
        records.push_back(rec);
        ++bid;
    }

    const size_t coarse_count = rows.size();
    rows.insert(rows.end(), extra_rows.begin(), extra_rows.end());
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b){ return a.Ecm < b.Ecm; });

    std::vector<Row> merged;
    merged.reserve(rows.size());
    constexpr double etol = 1e-13;
    for(const Row& r : rows)
    {
        if(merged.empty() || std::abs(r.Ecm - merged.back().Ecm) > etol)
            merged.push_back(r);
        else if(!merged.back().success && r.success)
            merged.back() = r;
    }
    rows.swap(merged);
    for(int ii=0; ii<int(rows.size()); ++ii) rows[size_t(ii)].i = ii;

    int ok=0;
    for(const Row& r: rows) if(r.success) ++ok;
    std::cout << "[v32c-refine] merged sorted grid: coarse=" << coarse_count
              << " iterative_extra_rows=" << extra_rows.size()
              << " final_rows=" << rows.size()
              << " success=" << ok << "/" << rows.size() << "\n";

    std::ofstream rep(report_path);
    rep << std::setprecision(17);
    rep << "# v32 iterative det(projected F3^{-1}) sign-flip refinement report\n";
    rep << "# criterion: start from coarse detProj sign flips, add refineN interior points per round, repeat until |E_candidate_prev-E_candidate_present| < tol\n";
    rep << "# refineN " << refineN << " max_rounds " << max_rounds << " tol " << tol << "\n";
    rep << "# columns: bracket_id converged rounds E_initial_left E_initial_right E_candidate_initial E_final_left E_final_right E_candidate_final last_delta orientation N_left N_right proj_dim_left proj_dim_right\n";
    for(const auto& rec : records)
    {
        rep << rec.bracket_id << " " << rec.converged << " " << rec.rounds << " "
            << rec.E_initial_left << " " << rec.E_initial_right << " " << rec.E_candidate_initial << " "
            << rec.E_final_left << " " << rec.E_final_right << " " << rec.E_candidate_final << " "
            << rec.last_delta << " " << rec.sign_left << "->" << rec.sign_right << " "
            << rec.final_N_left << " " << rec.final_N_right << " "
            << rec.final_proj_dim_left << " " << rec.final_proj_dim_right << "\n";
    }
    std::cout << "[v32c-refine] wrote iterative refinement report " << report_path << "\n";
}

static double refine_det_root(double a,double b,int sign_a,const MomentumIrrepSpec& spec,const PhysicsParams& par,const std::vector<int>& w1,const std::vector<int>& w2,int iters){
    double lo=a,hi=b; int slo=sign_a; for(int it=0;it<iters;++it){ double mid=0.5*(lo+hi); auto dm=eval_detail(mid,spec,par,w1,w2); int sm=dm.sdetProjF3inv.sign_phase_re; if(sm==0) return mid; if(sm==slo){ lo=mid; slo=sm; } else hi=mid; } return 0.5*(lo+hi);
}
struct Candidate { double E=0; double nearest_nonint=NAN, delta=NAN; int merged_count=0; std::string source=""; };
struct EventRecord {
    int iL=0,iR=0,sameN=0,sameP=0,sL=0,sR=0,nzL=0,nzR=0;
    double Eleft=NAN,Eright=NAN,Einterp=NAN,Erefined=NAN;
    double minAbsEigL=NAN,minAbsEigR=NAN,minSVL=NAN,minSVR=NAN;
    double lamLre=NAN,lamLim=NAN,lamRre=NAN,lamRim=NAN;
    int Nleft=0,Nright=0,projDimLeft=0,projDimRight=0;
    double nearest_nonint=NAN, delta=NAN;
    std::string cl="", oracle="NO_ORACLE_MATCH";
};
static std::string classify_zero(bool same_dim, bool det_flip, bool eig_cross, double minEigMin, double minSVmin, int nearZeroCount){
    if(!same_dim) return "DIMENSION_JUMP_REJECT";
    if(det_flip && (eig_cross || minEigMin<1e-5 || minSVmin<1e-5 || nearZeroCount>0)) return "BLIND_ZERO_CANDIDATE";
    if(det_flip) return "DET_SIGN_FLIP_BUT_NO_ZERO_SIGNATURE";
    return "NO_DET_SIGN_FLIP";
}
static void write_projection_audit(const std::string& path,const std::string& cfg,const std::string& label,const std::vector<int>& w1,const std::vector<int>& w2,const MomentumIrrepSpec& spec,const PhysicsParams& par,double Eaudit){
    std::ofstream out(path); out<<std::setprecision(17);
    out<<"# v31z projected F3inv zero-finder projection audit\n";
    out<<"# generated_by write_projection_audit() in source/projected_F3inv_zero_finder_v31z.cpp\n";
    out<<"# config="<<cfg<<"\n# label="<<label<<" irrep="<<spec.irrep<<" nnP=("<<spec.nnP[0]<<","<<spec.nnP[1]<<","<<spec.nnP[2]<<") L audit_Ecm="<<Eaudit<<" waves_vec_1={"<<int_list_str(w1)<<"} waves_vec_2={"<<int_list_str(w2)<<"}\n";
    out<<"# P_I generator: fvproj_v30j::projector_for_convention(); Vsel generator: get_projector_cached_v30q()->build_cached_projector_v30q()->build_projector_from_eigenvectors_near_one()\n";
    auto d=eval_detail(Eaudit,spec,par,w1,w2);
    out<<"raw_N "<<d.N<<"\nA_plm_rows "<<d.A<<"\nB_klm_rows "<<d.B<<"\nprojected_dim "<<d.proj_dim<<"\nP_I_idempotency "<<d.P_I_idem<<"\nP_I_hermiticity "<<d.P_I_herm<<"\nP_I_closure "<<d.P_I_closure<<"\nVsel_orthonormality "<<d.Vsel_orth<<"\nPproj_idempotency "<<d.Pproj_idem<<"\nPproj_hermiticity "<<d.Pproj_herm<<"\nVselVdag_minus_PI "<<d.Pproj_reconstruct<<"\nF3_equivariance_leak_littlegroup "<<d.F3_equiv<<"\nF3_projected_subspace_leak "<<d.F3_leak<<"\n";
    out<<"# raw_row selected_by_Vsel row_weight dominant_vsel_col dominant_absV basis_label\n";
    for(int r=0;r<d.N;++r){ double w=0,best=0; int bestc=-1; if(d.Vsel.rows()>r){ for(int c=0;c<d.Vsel.cols();++c){ double a=std::norm(d.Vsel(r,c)); w+=a; double av=std::abs(d.Vsel(r,c)); if(av>best){best=av; bestc=c;} } } out<<r<<" "<<(w>1e-10?1:0)<<" "<<w<<" "<<bestc<<" "<<best<<" "<<(r<int(d.labels.size())?d.labels[r]:"NO_LABEL")<<"\n"; }
}

int main(int argc,char** argv){
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
    std::cout<<"[v31zd] executable entered main()\n";
    if(argc!=2){ std::cerr<<"Usage: "<<argv[0]<<" configs/config_v31z_projected_F3inv_zero_110A2_L24.in\n"; return 1; }
    try{
        OptionsV31z opt; std::cout<<"[v31zd] reading config "<<argv[1]<<"\n"; auto settings=settings_from_config(argv[1],opt); std::cout<<"[v31zd] config parsed; constructing PhysicsParams\n"; PhysicsParams par=k3df_fit_v31l::make_base_physics(settings);
        apply_scatter_params_from_config_v31zl(argv[1], par);
        std::filesystem::create_directories(opt.outdir);
        print_active_scatter_params_v31zl(par);
        std::cout<<"[v31zm] [stage 1/6] projected F3^{-1} zero finder started\n";
        std::cout<<"[v31zm] config="<<argv[1]<<" L="<<settings.Lval<<" scan=["<<settings.scan_E0<<","<<settings.scan_E1<<"] coarseN="<<settings.coarseN<<" waves_vec_1={"<<int_list_str(opt.waves1)<<"}\n";
        for(const auto& lab: opt.labels){
            MomentumIrrepSpec spec=parse_label(lab); I3 nnP{spec.nnP[0],spec.nnP[1],spec.nnP[2]};
            std::cout<<"[v31z] [stage 2/6] non-interacting energies and degeneracies for "<<lab<<" at L="<<settings.Lval<<"\n";
            auto shells=nonint_shells(lab,nnP,settings.atmK,settings.atmpi,settings.xival,settings.Lval,opt.nmax,opt.nsq_max,settings.scan_E0,settings.scan_E1,opt.energy_group_tol);
            std::string nonpath=opt.outdir+"/"+opt.tag+"_"+lab+"_nonint_group_degeneracies.dat"; std::ofstream no(nonpath); no<<std::setprecision(17); no<<"# label "<<lab<<" nP "<<nnP[0]<<" "<<nnP[1]<<" "<<nnP[2]<<" L "<<settings.Lval<<"\n# columns: level Ecm raw_sym_momentum_multiplicity group_theory_degeneracy projected_irrep_dim char_sum\n";
            int lev=0; for(const auto& sh:shells){ no<<lev<<" "<<sh.E<<" "<<sh.raw_mult<<" "<<sh.group_deg<<" "<<sh.projected_dim<<" "<<sh.char_sum<<"\n"; std::cout<<"[v31z-nonint] level="<<lev<<" Ecm="<<sh.E<<" raw_mult="<<sh.raw_mult<<" group_deg="<<sh.group_deg<<" projected_irrep_dim="<<sh.projected_dim<<" char_sum="<<sh.char_sum<<"\n"; ++lev; }
            std::cout<<"[v31z] wrote "<<nonpath<<"\n";
            if(opt.smoke_test){
                std::cout<<"[v31zd] [smoke] evaluating representative energies before full grid; this identifies where time is spent.\n";
                std::vector<double> smokeE{settings.scan_E0,0.5*(settings.scan_E0+settings.scan_E1),settings.scan_E1};
                for(double Esm: smokeE){
                    auto t0=std::chrono::steady_clock::now();
                    std::cout<<"[v31zd-smoke] begin eval_detail E="<<Esm<<"\n";
                    auto dd=eval_detail(Esm,spec,par,opt.waves1,opt.waves2);
                    auto t1=std::chrono::steady_clock::now();
                    double sec=std::chrono::duration<double>(t1-t0).count();
                    std::cout<<"[v31zd-smoke] done E="<<Esm<<" success="<<(dd.ok?1:0)<<" N="<<dd.N<<" proj_dim="<<dd.proj_dim<<" minAbsEig="<<min_abs_eig(dd.eigProjF3inv)<<" minSVinv="<<dd.min_sv_projF3inv<<" time_sec="<<sec;
                    if(!dd.err.empty()) std::cout<<" err="<<dd.err;
                    std::cout<<"\n";
                }
                if(opt.smoke_only){ std::cout<<"[v31zd] smoke_only=1, exiting after smoke test.\n"; continue; }
            }
            std::vector<Row> rows(settings.coarseN); int done=0,nextpct=10; std::cout<<"[v31z] [stage 3/6] scanning det(projected F3^{-1}) over grid\n";
#pragma omp parallel for schedule(dynamic,4)
            for(int i=0;i<settings.coarseN;++i){ double t=(settings.coarseN==1)?0.0:double(i)/double(settings.coarseN-1); double E=settings.scan_E0+t*(settings.scan_E1-settings.scan_E0); rows[i]=eval_grid(i,E,spec,par,opt.waves1,opt.waves2);
#pragma omp critical
                { ++done; if(done*100/settings.coarseN>=nextpct){ std::cout<<"[v31z-grid] ["<<nextpct<<"%] "<<done<<"/"<<settings.coarseN<<"\n"; nextpct+=10; } }
            }
            std::cout<<"[v31zm] coarse grid complete; coarse rows="<<rows.size()<<"\n";
            if(opt.signflip_refine_enable){
                if(opt.iterative_signflip_refine_enable){
                    std::cout<<"[v32] [stage 3b/6] iterative refinement of det(projected F3^{-1}) sign-flip windows\n";
                    std::string v32report=opt.outdir+"/"+opt.tag+"_"+lab+"_"+wave_tag(opt.waves1)+"_v32_iterative_refined_signflip_candidates.dat";
                    add_iterative_signflip_mesh_v32(rows,spec,par,opt.waves1,opt.waves2,opt.iterative_refineN,opt.iterative_refine_max_rounds,opt.iterative_refine_tol,v32report);
                } else {
                    std::cout<<"[v31zm] [stage 3b/6] adding sorted refined mesh around every coarse sign-flip bracket\n";
                    add_refined_signflip_mesh_v31zm(rows,spec,par,opt.waves1,opt.waves2,opt.signflip_refine_points);
                }
            } else {
                for(int ii=0; ii<int(rows.size()); ++ii) rows[size_t(ii)].i=ii;
                std::cout<<"[v32] signflip_refine_enable=0; using coarse grid only\n";
            }
            std::string gridpath=opt.outdir+"/"+opt.tag+"_"+lab+"_"+wave_tag(opt.waves1)+"_projF3inv_grid.dat"; std::ofstream gr(gridpath); gr<<std::setprecision(17);
            gr<<"# v32 grid generated_by main()/eval_grid()/eval_detail(); coarse grid plus iterative refined det(projected F3inv) sign-flip mesh, sorted by Ecm\n# projected F3^{-1} = inverse(Vsel^dagger * scaled_F3 * Vsel) * f3_scale\n";
            gr<<"# columns: i Ecm En success A B N proj_dim detFullF3inv_phase_re detFullF3inv_phase_im detFullF3inv_signed_logabs detFullF3inv_logabs detFullF3inv_singular detProjF3inv_re detProjF3inv_im detProjF3inv_abs phase_re phase_im signed_logabs signPhaseRe signDetRe logabs singular minAbsEig maxAbsEig minSVprojF3inv maxSVprojF3inv condProjF3inv projF3norm projF3invnorm minSVprojF3 condProjF3 F3inv_iso_re F3inv_iso_im F3inv_iso_abs P_I_idem P_I_herm Vsel_orth VselVdag_minus_PI F3_equiv F3_leak error\n";
            int ok=0; for(const auto& r:rows){ if(r.success) ++ok; std::string err=r.err; for(char&c:err) if(std::isspace((unsigned char)c)) c='_'; gr<<r.i<<" "<<r.Ecm<<" "<<r.En<<" "<<r.success<<" "<<r.A<<" "<<r.B<<" "<<r.N<<" "<<r.proj_dim<<" "<<r.sdFull.phase.real()<<" "<<r.sdFull.phase.imag()<<" "<<r.sdFull.signed_logabs<<" "<<r.sdFull.logabs<<" "<<r.sdFull.singular<<" "<<r.sd.det.real()<<" "<<r.sd.det.imag()<<" "<<std::abs(r.sd.det)<<" "<<r.sd.phase.real()<<" "<<r.sd.phase.imag()<<" "<<r.sd.signed_logabs<<" "<<r.sd.sign_phase_re<<" "<<r.sd.sign_det_re<<" "<<r.sd.logabs<<" "<<r.sd.singular<<" "<<r.minAbsEig<<" "<<r.maxAbsEig<<" "<<r.minSV<<" "<<r.maxSV<<" "<<r.cond<<" "<<r.projF3norm<<" "<<r.projF3invnorm<<" "<<r.minSVprojF3<<" "<<r.condProjF3<<" "<<r.F3inv_iso.real()<<" "<<r.F3inv_iso.imag()<<" "<<std::abs(r.F3inv_iso)<<" "<<r.Pidem<<" "<<r.Pherm<<" "<<r.Vorth<<" "<<r.Precon<<" "<<r.F3equiv<<" "<<r.F3leak<<" "<<err<<"\n"; }
            std::cout<<"[v31z] wrote "<<gridpath<<" success="<<ok<<"/"<<settings.coarseN<<"\n";
            std::string eigpath=opt.outdir+"/"+opt.tag+"_"+lab+"_"+wave_tag(opt.waves1)+"_projF3inv_eigenvalues_raw.dat";
            std::ofstream eg(eigpath); eg<<std::setprecision(17);
            eg<<"# v31zd raw projected F3^{-1} eigenvalues for independent branch tracking/plotting\n";
            eg<<"# columns: grid_i Ecm eig_local_index eig_re eig_im eig_abs proj_dim raw_N success\n";
            for(const auto& r:rows){ for(int j=0;j<r.eigs.size();++j){ auto z=r.eigs(j); eg<<r.i<<" "<<r.Ecm<<" "<<j<<" "<<z.real()<<" "<<z.imag()<<" "<<std::abs(z)<<" "<<r.proj_dim<<" "<<r.N<<" "<<r.success<<"\n"; } }
            std::cout<<"[v31zd] wrote raw eigenvalue file "<<eigpath<<"\n";
            std::string projpath=opt.outdir+"/"+opt.tag+"_"+lab+"_"+wave_tag(opt.waves1)+"_projection_matrix_Vsel_audit.dat"; write_projection_audit(projpath,argv[1],lab,opt.waves1,opt.waves2,spec,par,0.5*(settings.scan_E0+settings.scan_E1)); std::cout<<"[v31z] wrote projection audit "<<projpath<<"\n";
            std::cout<<"[v31z] [stage 4/6] classifying zero candidates from determinant/eigenbranch sign changes\n";
            std::string auditpath=opt.outdir+"/"+opt.tag+"_"+lab+"_"+wave_tag(opt.waves1)+"_zero_event_audit.dat"; std::ofstream au(auditpath); au<<std::setprecision(17);
            au<<"# v31zc zero event audit. Fast mode does not recompute endpoint eigenbranches unless audit_eigenbranch_detail=1. Blind zero classification does NOT use non-interacting oracle. Oracle columns are only comparison after prediction.\n";
            au<<"# columns: source iL iR Eleft Eright Einterp Erefined class sameN sameProjDim signL signR minAbsEigL minAbsEigR minSVinvL minSVinvR eigBranchL_re eigBranchL_im eigBranchR_re eigBranchR_im nearZeroEigCountL nearZeroEigCountR Nleft Nright projDimLeft projDimRight nearest_nonint delta_to_nonint oracle_match\n";
            std::vector<Candidate> cands;
            std::vector<EventRecord> events;
            std::cout<<"[v31zd] fast classifier: audit_eigenbranch_detail="<<opt.audit_eigenbranch_detail
                     <<" refine_zero_roots="<<opt.refine_zero_roots<<" refine_iter="<<opt.refine_iter<<"\n";
            for(int i=0;i+1<int(rows.size());++i){
                const auto& Lr=rows[i]; const auto& Rr=rows[i+1];
                if(!Lr.success||!Rr.success) continue;
                int sL=Lr.sd.sign_phase_re, sR=Rr.sd.sign_phase_re;
                if(sL==0||sR==0||sL*sR>=0) continue;
                EventRecord ev; ev.iL=i; ev.iR=i+1; ev.Eleft=Lr.Ecm; ev.Eright=Rr.Ecm; ev.sL=sL; ev.sR=sR;
                ev.sameN=(Lr.N==Rr.N)?1:0; ev.sameP=(Lr.proj_dim==Rr.proj_dim)?1:0;
                ev.Nleft=Lr.N; ev.Nright=Rr.N; ev.projDimLeft=Lr.proj_dim; ev.projDimRight=Rr.proj_dim;
                ev.minAbsEigL=Lr.minAbsEig; ev.minAbsEigR=Rr.minAbsEig; ev.minSVL=Lr.minSV; ev.minSVR=Rr.minSV;
                ev.Einterp=interp_zero(Lr.Ecm,Lr.sd.signed_logabs,Rr.Ecm,Rr.sd.signed_logabs);
                // Fast path: do not recompute endpoint matrices. The grid already stores min|eig| and min singular value.
                bool eigCross=false;
                ev.nzL = (std::isfinite(Lr.minAbsEig) && Lr.minAbsEig<opt.zero_minabs_eig_max) ? 1 : 0;
                ev.nzR = (std::isfinite(Rr.minAbsEig) && Rr.minAbsEig<opt.zero_minabs_eig_max) ? 1 : 0;
                if(opt.audit_eigenbranch_detail){
                    auto dl=eval_detail(Lr.Ecm,spec,par,opt.waves1,opt.waves2);
                    auto dr=eval_detail(Rr.Ecm,spec,par,opt.waves1,opt.waves2);
                    int idxL=min_abs_eig_index(dl.eigProjF3inv);
                    int idxR=(idxL>=0?nearest_eig_index(dr.eigProjF3inv,dl.eigProjF3inv(idxL)):min_abs_eig_index(dr.eigProjF3inv));
                    comp lamL=(idxL>=0?dl.eigProjF3inv(idxL):comp(NAN,NAN));
                    comp lamR=(idxR>=0?dr.eigProjF3inv(idxR):comp(NAN,NAN));
                    ev.lamLre=lamL.real(); ev.lamLim=lamL.imag(); ev.lamRre=lamR.real(); ev.lamRim=lamR.imag();
                    eigCross=sign_nonzero(lamL.real())*sign_nonzero(lamR.real())<0;
                    ev.nzL=count_near_zero(dl.eigProjF3inv,opt.zero_minabs_eig_max);
                    ev.nzR=count_near_zero(dr.eigProjF3inv,opt.zero_minabs_eig_max);
                }
                ev.cl=classify_zero(ev.sameN&&ev.sameP,true,eigCross,
                                    std::min(Lr.minAbsEig,Rr.minAbsEig),
                                    std::min(Lr.minSV,Rr.minSV),
                                    std::max(ev.nzL,ev.nzR));
                ev.Erefined=ev.Einterp;
                if(ev.cl=="BLIND_ZERO_CANDIDATE" && opt.refine_zero_roots){
                    ev.Erefined=refine_det_root(Lr.Ecm,Rr.Ecm,sL,spec,par,opt.waves1,opt.waves2,opt.refine_iter);
                }
                if(!shells.empty()){
                    ev.nearest_nonint=shells[0].E;
                    for(const auto& sh:shells) if(std::abs(sh.E-ev.Erefined)<std::abs(ev.nearest_nonint-ev.Erefined)) ev.nearest_nonint=sh.E;
                    ev.delta=ev.Erefined-ev.nearest_nonint;
                }
                bool om=std::isfinite(ev.delta)&&std::abs(ev.delta)<opt.oracle_match_tol;
                ev.oracle=om?"ORACLE_MATCH":"NO_ORACLE_MATCH";
                events.push_back(ev);
                if(ev.cl=="BLIND_ZERO_CANDIDATE"){ Candidate c; c.E=ev.Erefined; c.nearest_nonint=ev.nearest_nonint; c.delta=ev.delta; c.merged_count=1; c.source="detProjF3inv"; cands.push_back(c); }
            }
            for(const auto& ev: events){
                au<<"detProjF3inv "<<ev.iL<<" "<<ev.iR<<" "<<ev.Eleft<<" "<<ev.Eright<<" "<<ev.Einterp<<" "<<ev.Erefined<<" "<<ev.cl<<" "<<ev.sameN<<" "<<ev.sameP<<" "<<ev.sL<<" "<<ev.sR<<" "<<ev.minAbsEigL<<" "<<ev.minAbsEigR<<" "<<ev.minSVL<<" "<<ev.minSVR<<" "<<ev.lamLre<<" "<<ev.lamLim<<" "<<ev.lamRre<<" "<<ev.lamRim<<" "<<ev.nzL<<" "<<ev.nzR<<" "<<ev.Nleft<<" "<<ev.Nright<<" "<<ev.projDimLeft<<" "<<ev.projDimRight<<" "<<ev.nearest_nonint<<" "<<ev.delta<<" "<<ev.oracle<<"\n";
            }
            std::cout<<"[v31zd] sign-flip events="<<events.size()<<" blind_zero_candidates="<<cands.size()<<"\n";
            std::sort(cands.begin(),cands.end(),[](const Candidate&a,const Candidate&b){return a.E<b.E;}); std::vector<Candidate> merged; for(auto c:cands){ if(merged.empty()||std::abs(c.E-merged.back().E)>opt.candidate_merge_tol){ merged.push_back(c); } else { auto& m=merged.back(); m.E=(m.E*m.merged_count+c.E)/(m.merged_count+1); m.merged_count++; if(std::abs(c.delta)<std::abs(m.delta)){ m.nearest_nonint=c.nearest_nonint; m.delta=c.delta; } m.source += "+"+c.source; } }
            std::string candpath=opt.outdir+"/"+opt.tag+"_"+lab+"_"+wave_tag(opt.waves1)+"_final_zero_candidates_vs_nonint.dat"; std::ofstream co(candpath); co<<std::setprecision(17); co<<"# v31z final blind zero candidates for det(projected F3^{-1}). Non-interacting columns are oracle comparison only.\n# columns: idx E_zero merged_count nearest_nonint delta_to_nonint oracle_match source\n";
            std::cout<<"[v31z] [stage 5/6] final blind zero candidates for "<<lab<<" L="<<settings.Lval<<"\n";
            int ci=0; for(const auto& c:merged){ bool om=std::isfinite(c.delta)&&std::abs(c.delta)<opt.oracle_match_tol; co<<ci<<" "<<c.E<<" "<<c.merged_count<<" "<<c.nearest_nonint<<" "<<c.delta<<" "<<(om?"ORACLE_MATCH":"NO_ORACLE_MATCH")<<" "<<c.source<<"\n"; std::cout<<"[v31z-zero] idx="<<ci<<" E="<<c.E<<" nearest_nonint="<<c.nearest_nonint<<" delta="<<c.delta<<" "<<(om?"ORACLE_MATCH":"NO_ORACLE_MATCH")<<" merged="<<c.merged_count<<"\n"; ++ci; }
            std::cout<<"[v31z] wrote "<<auditpath<<"\n[v31z] wrote "<<candpath<<"\n";
        }
        std::cout<<"[v31z] [stage 6/6] complete\n";
    }catch(const std::exception& e){ std::cerr<<"[v31z-error] "<<e.what()<<"\n"; return 2; }
    return 0;
}
