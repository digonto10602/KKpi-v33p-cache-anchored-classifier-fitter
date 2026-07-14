#include "K3df_minuit_fit_v31l_lattice_covariance.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using I3 = std::array<int,3>;
using Mat3i = std::array<std::array<int,3>,3>;

static std::string trim_v31t(std::string s){ while(!s.empty()&&std::isspace((unsigned char)s.front())) s.erase(s.begin()); while(!s.empty()&&std::isspace((unsigned char)s.back())) s.pop_back(); return s; }
static std::map<std::string,std::string> read_kv_v31t(const std::string& path){ std::ifstream in(path); if(!in) throw std::runtime_error("Could not open config: "+path); std::map<std::string,std::string> kv; std::string line; while(std::getline(in,line)){ auto h=line.find('#'); if(h!=std::string::npos) line=line.substr(0,h); auto e=line.find('='); if(e==std::string::npos) continue; std::string k=trim_v31t(line.substr(0,e)); std::string v=trim_v31t(line.substr(e+1)); if(!k.empty()) kv[k]=v; } return kv; }
static std::string gs(const std::map<std::string,std::string>& kv,const std::string& k,const std::string& d){ auto it=kv.find(k); return it==kv.end()?d:it->second; }
static int gi(const std::map<std::string,std::string>& kv,const std::string& k,int d){ auto it=kv.find(k); return it==kv.end()?d:std::stoi(it->second); }
static double gd(const std::map<std::string,std::string>& kv,const std::string& k,double d){ auto it=kv.find(k); return it==kv.end()?d:std::stod(it->second); }
static std::vector<std::string> split_ws_v31t(std::string s){ for(char& c:s) if(c==',') c=' '; std::istringstream is(s); std::vector<std::string> v; std::string x; while(is>>x) v.push_back(x); return v; }
static std::vector<int> parse_int_list_v31t(std::string s, std::vector<int> def={}){ auto p=split_ws_v31t(s); if(p.empty()) return def; std::vector<int> v; for(auto& x:p) v.push_back(std::stoi(x)); return v; }
static std::vector<std::vector<int>> parse_wave_sets_v31t(std::string s){ std::vector<std::vector<int>> out; std::stringstream ss(s); std::string item; while(std::getline(ss,item,';')){ item=trim_v31t(item); if(!item.empty()) out.push_back(parse_int_list_v31t(item)); } if(out.empty()) out={{0},{0,1}}; return out; }
static std::string int_list_str_v31t(const std::vector<int>& v){ std::string s; for(size_t i=0;i<v.size();++i){ if(i) s+=","; s+=std::to_string(v[i]); } return s; }
static std::string wave_tag_v31t(const std::vector<int>& v){ std::string s="waves"; for(int x:v) s+="_"+std::to_string(x); return s; }

static std::complex<double> det_lu_v31t(const Eigen::MatrixXcd& M){
    if(M.rows()==0 || M.rows()!=M.cols()) return {NAN,NAN};
    Eigen::PartialPivLU<Eigen::MatrixXcd> lu(M);
    return lu.determinant();
}

struct OptionsV31r {
    std::vector<std::string> labels = {"111_A2"};
    std::vector<std::vector<int>> waves1 = {{0},{0,1}};
    std::vector<int> waves2 = {0};
    int nmax = 6;
    int nsq_max = 12;
    double energy_group_tol = 1.0e-10;
    std::string outdir = "output_v31t_nonint_degeneracy_F2Giso";
    std::string tag = "debug_v31t_nonint_degeneracy_F2Giso";
};

static k3df_fit_v31l::FitSettings settings_from_config_v31t(const std::string& cfg, OptionsV31r& opt){
    using namespace k3df_fit_v31l; auto kv=read_kv_v31t(cfg); FitSettings s;
    opt.labels=split_ws_v31t(gs(kv,"list_of_mom","111_A2")); s.list_of_mom=opt.labels;
    opt.waves1=parse_wave_sets_v31t(gs(kv,"waves_vec_1_sets","0;0,1")); opt.waves2=parse_int_list_v31t(gs(kv,"waves_vec_2","0"),{0}); s.waves_vec_2=opt.waves2;
    s.Lval=gd(kv,"Lval",s.Lval); s.xival=gd(kv,"xival",s.xival); s.scan_E0=gd(kv,"scan_E0",s.scan_E0); s.scan_E1=gd(kv,"scan_E1",s.scan_E1); s.coarseN=gi(kv,"coarseN",s.coarseN); s.omp_threads=gi(kv,"omp_threads",s.omp_threads); s.debug=gs(kv,"debug","n").empty()?'n':gs(kv,"debug","n")[0];
    s.atmpi=gd(kv,"atmpi",s.atmpi); s.atmK=gd(kv,"atmK",s.atmK); s.eta_1=gd(kv,"eta_1",s.eta_1); s.eta_2=gd(kv,"eta_2",s.eta_2); s.alpha=gd(kv,"alpha",s.alpha); s.epsilon_h=gd(kv,"epsilon_h",s.epsilon_h); s.max_shell_num=gd(kv,"max_shell_num",s.max_shell_num); s.tolerance=gd(kv,"tolerance",s.tolerance); s.parity=gi(kv,"parity",s.parity); s.eig_tol=gd(kv,"eig_tol",s.eig_tol); s.norm_tol=gd(kv,"norm_tol",s.norm_tol); s.proj_tol=gd(kv,"proj_tol",s.proj_tol); s.Q0norm=gi(kv,"Q0norm",s.Q0norm?1:0)!=0; s.sort_orbit_flag=gi(kv,"sort_orbit_flag",s.sort_orbit_flag?1:0)!=0;
    opt.nmax=gi(kv,"nonint_nmax",6); opt.nsq_max=gi(kv,"nonint_nsq_max",12); opt.energy_group_tol=gd(kv,"energy_group_tol",1.0e-10); opt.outdir=gs(kv,"output_dir",opt.outdir); opt.tag=gs(kv,"output_tag",opt.tag); s.output_dir=opt.outdir; s.output_tag=opt.tag;
    return s;
}

static I3 mat_vec(const Mat3i& R, const I3& v){ I3 o{0,0,0}; for(int i=0;i<3;++i) for(int j=0;j<3;++j) o[i]+=R[i][j]*v[j]; return o; }
static int det3(const Mat3i& M){ return M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])-M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])+M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]); }
static int trace3(const Mat3i& M){ return M[0][0]+M[1][1]+M[2][2]; }
static std::vector<Mat3i> signed_perm_mats(){ std::vector<Mat3i> out; std::array<int,3> p{0,1,2}; do{ for(int sx:{-1,1}) for(int sy:{-1,1}) for(int sz:{-1,1}){ Mat3i M{}; for(auto& r:M) r={0,0,0}; M[0][p[0]]=sx; M[1][p[1]]=sy; M[2][p[2]]=sz; out.push_back(M); } }while(std::next_permutation(p.begin(),p.end())); return out; }
static std::vector<Mat3i> little_group(const I3& nP){ std::vector<Mat3i> out; for(const auto& R: signed_perm_mats()) if(mat_vec(R,nP)==nP) out.push_back(R); return out; }

static std::string canonical_key(const I3& k1, const I3& k2, const I3& k3){
    auto a=k1,b=k2; if(b<a) std::swap(a,b); std::ostringstream os; os<<a[0]<<","<<a[1]<<","<<a[2]<<"|"<<b[0]<<","<<b[1]<<","<<b[2]<<"|"<<k3[0]<<","<<k3[1]<<","<<k3[2]; return os.str();
}
static std::string transform_key(const std::string& key, const Mat3i& R){
    std::stringstream ss(key); std::string s1,s2,s3; std::getline(ss,s1,'|'); std::getline(ss,s2,'|'); std::getline(ss,s3,'|'); auto parse=[](std::string s){ for(char& c:s) if(c==',') c=' '; std::istringstream is(s); I3 a{}; is>>a[0]>>a[1]>>a[2]; return a;}; return canonical_key(mat_vec(R,parse(s1)),mat_vec(R,parse(s2)),mat_vec(R,parse(s3)));
}

static double char_irrep(const std::string& label, const I3& nP, const Mat3i& R){
    std::string ir = label.substr(label.find('_')+1);
    if(ir=="A1" || ir=="A1m" || ir=="A1u") return 1.0;
    if(ir=="A2") {
        if(nP==I3{1,1,1}) return det3(R)>0 ? 1.0 : -1.0; // C3v: A2 is odd under reflections.
        if(nP==I3{1,1,0}) return det3(R)>0 ? 1.0 : -1.0; // C2v A2.
        if(nP==I3{0,0,1} || nP==I3{0,0,2}) return det3(R)>0 ? 1.0 : -1.0; // C4v A2, both reflection classes -1.
        return det3(R)>0 ? 1.0 : -1.0;
    }
    if(ir=="B1" || ir=="B2") {
        // Only used as a rough fallback for C2v/C4v. Prefer A1/A2/E for this diagnostic.
        return 1.0;
    }
    if(ir=="E" || ir=="E2") {
        // 2D irrep fallback for C3v/C4v: chars E=2, C3/C2=-1/-2, reflections=0.
        if(det3(R)<0) return 0.0;
        int tr=trace3(R);
        if(tr==3) return 2.0;
        if(nP==I3{1,1,1}) return -1.0;
        if(nP==I3{0,0,1} || nP==I3{0,0,2}) return (tr==-1 ? -2.0 : 0.0);
        return 0.0;
    }
    return 1.0;
}
static int irrep_dim_v31t_local(const std::string& label){ std::string ir=label.substr(label.find('_')+1); return (ir=="E" || ir=="E2") ? 2 : 1; }

static double mom2(const I3& n){ return double(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]); }
static double oneE(double m, const I3& n, double xi, double L){ const double p2 = std::pow(2.0*M_PI/(xi*L),2)*mom2(n); return std::sqrt(m*m+p2); }
static double three_Ecm(double mK,double mpi,const I3& k1,const I3& k2,const I3& k3,const I3& nP,double xi,double L){ double Elab=oneE(mK,k1,xi,L)+oneE(mK,k2,xi,L)+oneE(mpi,k3,xi,L); double P2=std::pow(2.0*M_PI/(xi*L),2)*mom2(nP); double x=Elab*Elab-P2; return x>0?std::sqrt(x):NAN; }

struct Shell { double E=0; std::vector<std::string> keys; int raw_mult=0; int group_deg=0; int projected_dim=0; double char_sum=0; };
static std::vector<Shell> nonint_shells(const std::string& label,const I3& nP,double mK,double mpi,double xi,double L,int nmax,int nsq_max,double Emin,double Emax,double tol){
    std::map<long long, std::set<std::string>> groups;
    std::map<long long, double> eval;
    for(int a=-nmax;a<=nmax;++a) for(int b=-nmax;b<=nmax;++b) for(int c=-nmax;c<=nmax;++c){ I3 k1{a,b,c}; if(mom2(k1)>nsq_max) continue; for(int d=-nmax;d<=nmax;++d) for(int e=-nmax;e<=nmax;++e) for(int f=-nmax;f<=nmax;++f){ I3 k2{d,e,f}; if(mom2(k2)>nsq_max) continue; I3 k3{nP[0]-k1[0]-k2[0],nP[1]-k1[1]-k2[1],nP[2]-k1[2]-k2[2]}; if(mom2(k3)>nsq_max) continue; double Ecm=three_Ecm(mK,mpi,k1,k2,k3,nP,xi,L); if(!std::isfinite(Ecm)||Ecm<Emin-1e-12||Ecm>Emax+1e-12) continue; long long bin=llround(Ecm/tol); groups[bin].insert(canonical_key(k1,k2,k3)); eval[bin]=Ecm; } }
    auto G=little_group(nP); std::vector<Shell> shells; for(auto& kv:groups){ Shell sh; sh.E=eval[kv.first]; sh.keys.assign(kv.second.begin(),kv.second.end()); sh.raw_mult=int(sh.keys.size()); double sum=0.0; for(const auto& R:G){ int fixed=0; for(const auto& key:sh.keys) if(transform_key(key,R)==key) ++fixed; sum += char_irrep(label,nP,R)*double(fixed); } sh.char_sum=sum; double mult = sum/double(G.size()); sh.group_deg = int(std::llround(mult)); sh.projected_dim = sh.group_deg*irrep_dim_v31t_local(label); shells.push_back(sh); }
    std::sort(shells.begin(),shells.end(),[](const Shell&a,const Shell&b){return a.E<b.E;}); return shells;
}

static comp safe_comp(const std::vector<std::vector<comp>>& c,int ch,int i){ if(ch<0||ch>=int(c.size())||i<0||i>=int(c[ch].size())) return comp(NAN,NAN); return c[ch][i]; }
static Eigen::VectorXcd iso_vec(const std::vector<std::vector<comp>>& plm,const std::vector<std::vector<comp>>& klm){ int A=plm.empty()?0:int(plm[0].size()); int B=klm.empty()?0:int(klm[0].size()); Eigen::VectorXcd v=Eigen::VectorXcd::Zero(A+B); for(int i=0;i<A;++i){ int ell=int(std::llround(safe_comp(plm,3,i).real())); int m=int(std::llround(safe_comp(plm,4,i).real())); if(ell==0 && m==0) v(i)=1.0; } for(int i=0;i<B;++i){ int ell=int(std::llround(safe_comp(klm,3,i).real())); int m=int(std::llround(safe_comp(klm,4,i).real())); if(ell==0 && m==0) v(A+i)=1.0/std::sqrt(2.0); } return v; }

struct GridRow { int i=0, success=0, A=0,B=0,N=0; double Ecm=0, En=0; comp F2iso={NAN,NAN}, Giso={NAN,NAN}, detF2={NAN,NAN}, detG={NAN,NAN}; double F2norm=NAN,Gnorm=NAN; std::string err="OK"; };
static GridRow eval_grid(int i,double Ecm,const MomentumIrrepSpec& spec,const k3df_fit_v31l::FitSettings& settings,const PhysicsParams& par,const std::vector<int>& waves1,const std::vector<int>& waves2){ GridRow r; r.i=i; r.Ecm=Ecm; try{ I3 nnP{spec.nnP[0],spec.nnP[1],spec.nnP[2]}; comp pi=std::acos(-1.0); double L=par.L(); comp twopibyL=comp(2.0,0.0)*pi/comp(L,0.0); std::vector<comp> total_P(3); for(int a=0;a<3;++a) total_P[a]=twopibyL*double(nnP[a]); comp En_c=Ecm_to_E(comp(Ecm,0),total_P); r.En=En_c.real(); std::vector<std::vector<comp>> plm(5),klm(5); std::vector<std::vector<int>> np(5),nk(5); config_maker_4_momentum_first(plm,np,waves1,En_c,total_P,par.atmK,par.atmK,par.atmpi,L,par.epsilon_h,par.max_shell_num,par.tolerance); config_maker_4_momentum_first(klm,nk,waves2,En_c,total_P,par.atmpi,par.atmK,par.atmK,L,par.epsilon_h,par.max_shell_num,par.tolerance); r.A=int(plm[0].size()); r.B=int(klm[0].size()); r.N=r.A+r.B; if(r.N<=0){ r.err="EMPTY_BASIS"; return r; } Eigen::MatrixXcd F2(r.N,r.N),G(r.N,r.N); F2_2plus1_mat(F2,En_c,plm,klm,total_P,par.atmK,par.atmpi,L,par.alpha,par.epsilon_h,par.max_shell_num,par.Q0norm); G_2plus1_mat(G,En_c,plm,klm,total_P,par.atmK,par.atmpi,L,par.alpha,par.epsilon_h,par.max_shell_num,par.Q0norm); Eigen::VectorXcd iso=iso_vec(plm,klm); r.F2iso=(iso.transpose()*F2*iso)(0,0); r.Giso=(iso.transpose()*G*iso)(0,0); r.detF2=det_lu_v31t(F2); r.detG=det_lu_v31t(G); r.F2norm=F2.norm(); r.Gnorm=G.norm(); r.success=1; }catch(const std::exception& e){ r.success=0; r.err=e.what(); } return r; }

int main(int argc,char** argv){ if(argc!=2){ std::cerr<<"Usage: "<<argv[0]<<" config/config_v31t_nonint_degeneracy_F2Giso.in\n"; return 1; } try{ OptionsV31r opt; auto settings=settings_from_config_v31t(argv[1],opt); PhysicsParams par=k3df_fit_v31l::make_base_physics(settings); std::filesystem::create_directories(opt.outdir);
#ifdef _OPENMP
    omp_set_num_threads(settings.omp_threads); omp_set_max_active_levels(1);
#endif
    Eigen::setNbThreads(1);
    std::cout<<"[v31t] non-interacting group-degeneracy + F2iso/Giso + det(F2)/det(G) diagnostic\n";
    std::cout<<"[v31t] labels="; for(auto&s:opt.labels) std::cout<<" "<<s; std::cout<<" waves_vec_1_sets="; for(auto&w:opt.waves1) std::cout<<" ["<<int_list_str_v31t(w)<<"]"; std::cout<<"\n";
    for(const auto& lab: opt.labels){ MomentumIrrepSpec spec=parse_label(lab); I3 nnP{spec.nnP[0],spec.nnP[1],spec.nnP[2]}; auto shells=nonint_shells(lab,nnP,settings.atmK,settings.atmpi,settings.xival,settings.Lval,opt.nmax,opt.nsq_max,settings.scan_E0,settings.scan_E1,opt.energy_group_tol); std::string nonpath=opt.outdir+"/"+opt.tag+"_"+lab+"_nonint_group_degeneracies.dat"; std::ofstream no(nonpath); no<<std::setprecision(17); no<<"# label "<<lab<<" nP "<<nnP[0]<<" "<<nnP[1]<<" "<<nnP[2]<<"\n"; no<<"# columns: level Ecm raw_sym_momentum_multiplicity group_theory_degeneracy projected_irrep_dim char_sum\n"; int lev=0; for(auto& sh:shells){ no<<lev++<<" "<<sh.E<<" "<<sh.raw_mult<<" "<<sh.group_deg<<" "<<sh.projected_dim<<" "<<sh.char_sum<<"\n"; std::cout<<"[v31t-nonint] "<<lab<<" Ecm="<<std::setprecision(12)<<sh.E<<" raw="<<sh.raw_mult<<" group_deg="<<sh.group_deg<<" proj_dim="<<sh.projected_dim<<"\n"; }
        for(const auto& w1: opt.waves1){ settings.waves_vec_1=w1; std::string wtag=wave_tag_v31t(w1); std::string gridpath=opt.outdir+"/"+opt.tag+"_"+lab+"_"+wtag+"_F2iso_Giso_detF2_detG_grid.dat"; std::vector<GridRow> rows(settings.coarseN); int done=0,nextpct=10; std::cout<<"[v31t] [stage] "<<lab<<" waves_vec_1="<<int_list_str_v31t(w1)<<" computing F2iso/Giso and det(F2)/det(G) over grid\n";
#pragma omp parallel for schedule(dynamic,1)
            for(int i=0;i<settings.coarseN;++i){ double t=(settings.coarseN==1)?0.0:double(i)/double(settings.coarseN-1); double E=settings.scan_E0+t*(settings.scan_E1-settings.scan_E0); rows[i]=eval_grid(i,E,spec,settings,par,w1,opt.waves2); 
#pragma omp critical
                { ++done; progress_percent_log("v31t-"+lab+"-"+wtag,done,settings.coarseN,nextpct,10); }
            }
            std::ofstream gr(gridpath); gr<<std::setprecision(17); gr<<"# label "<<lab<<" waves_vec_1 "<<int_list_str_v31t(w1)<<" waves_vec_2 "<<int_list_str_v31t(opt.waves2)<<"\n"; gr<<"# F2iso/Giso use iso vector supported only on ell=0,m=0 rows: channel1 weight 1, channel2 weight 1/sqrt(2).\n"; gr<<"# columns: i Ecm En success A B N F2iso_re F2iso_im F2iso_abs Giso_re Giso_im Giso_abs detF2_re detF2_im detF2_abs detG_re detG_im detG_abs F2norm Gnorm error\n"; int ok=0; for(const auto&r:rows){ if(r.success) ++ok; std::string err=r.err; for(char&c:err) if(std::isspace((unsigned char)c)) c='_'; gr<<r.i<<" "<<r.Ecm<<" "<<r.En<<" "<<r.success<<" "<<r.A<<" "<<r.B<<" "<<r.N<<" "<<r.F2iso.real()<<" "<<r.F2iso.imag()<<" "<<std::abs(r.F2iso)<<" "<<r.Giso.real()<<" "<<r.Giso.imag()<<" "<<std::abs(r.Giso)<<" "<<r.detF2.real()<<" "<<r.detF2.imag()<<" "<<std::abs(r.detF2)<<" "<<r.detG.real()<<" "<<r.detG.imag()<<" "<<std::abs(r.detG)<<" "<<r.F2norm<<" "<<r.Gnorm<<" "<<err<<"\n"; } std::cout<<"[v31t] wrote "<<gridpath<<" success="<<ok<<"/"<<settings.coarseN<<"\n"; }
    }
    std::cout<<"[v31t] [####################] 100% complete\n";
}catch(const std::exception& e){ std::cerr<<"[v31t-error] "<<e.what()<<"\n"; return 2; } return 0; }
