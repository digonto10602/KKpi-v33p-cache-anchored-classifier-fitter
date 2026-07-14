#include "K3df_minuit_fit_v31l_lattice_covariance.hpp"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

static std::string trim_v31p(std::string s){ while(!s.empty()&&std::isspace((unsigned char)s.front())) s.erase(s.begin()); while(!s.empty()&&std::isspace((unsigned char)s.back())) s.pop_back(); return s; }
static std::map<std::string,std::string> read_kv_v31p(const std::string& path){ std::ifstream in(path); if(!in) throw std::runtime_error("Could not open config: "+path); std::map<std::string,std::string> kv; std::string line; int n=0; while(std::getline(in,line)){ ++n; auto h=line.find('#'); if(h!=std::string::npos) line=line.substr(0,h); auto e=line.find('='); if(e==std::string::npos) continue; kv[trim_v31p(line.substr(0,e))]=trim_v31p(line.substr(e+1)); } return kv; }
static std::string gs(const std::map<std::string,std::string>& kv,const std::string& k,const std::string& d){ auto it=kv.find(k); return it==kv.end()?d:it->second; }
static int gi(const std::map<std::string,std::string>& kv,const std::string& k,int d){ auto it=kv.find(k); return it==kv.end()?d:std::stoi(it->second); }
static double gd(const std::map<std::string,std::string>& kv,const std::string& k,double d){ auto it=kv.find(k); return it==kv.end()?d:std::stod(it->second); }
static std::vector<std::string> split_ws_v31p(std::string s){ for(char& c:s) if(c==',') c=' '; std::istringstream is(s); std::vector<std::string> v; std::string x; while(is>>x) v.push_back(x); return v; }
static std::vector<int> parse_int_list_v31p(std::string s, std::vector<int> def={}){ auto p=split_ws_v31p(s); if(p.empty()) return def; std::vector<int> v; for(auto& x:p) v.push_back(std::stoi(x)); return v; }
static std::vector<std::vector<int>> parse_wave_sets_v31p(std::string s){ std::vector<std::vector<int>> out; std::stringstream ss(s); std::string item; while(std::getline(ss,item,';')){ item=trim_v31p(item); if(!item.empty()) out.push_back(parse_int_list_v31p(item)); } if(out.empty()) out={{0},{0,1}}; return out; }
static std::string list_int_str_v31p(const std::vector<int>& v){ std::string s; for(size_t i=0;i<v.size();++i){ if(i) s+=","; s+=std::to_string(v[i]); } return s; }
static std::string wave_tag_v31p(const std::vector<int>& v){ std::string s="waves"; for(int x:v) s+="_"+std::to_string(x); return s; }

struct OptionsV31p{ std::vector<std::string> labels={"111_A2"}; std::vector<std::vector<int>> waves1={{0},{0,1}}; std::vector<int> waves2={0}; int stride=1; int write_all_flat=0; double compare_tol=0.0; double large_abs=1e8; std::string outdir="output_v31p_all_QC_mixed_sp_diagnostics"; std::string tag="debug_v31p_all_QC_mixed_sp_diagnostics"; k3df_fit_v31l::K3dfParameters k3; };

static k3df_fit_v31l::FitSettings settings_from_config_v31p(const std::string& cfg, OptionsV31p& opt){
    using namespace k3df_fit_v31l; auto kv=read_kv_v31p(cfg); FitSettings s;
    opt.labels=split_ws_v31p(gs(kv,"list_of_mom","111_A2")); s.list_of_mom=opt.labels;
    opt.waves1=parse_wave_sets_v31p(gs(kv,"waves_vec_1_sets","0;0,1")); opt.waves2=parse_int_list_v31p(gs(kv,"waves_vec_2","0"),{0}); s.waves_vec_2=opt.waves2;
    s.Lval=gd(kv,"Lval",s.Lval); s.xival=gd(kv,"xival",s.xival); s.scan_E0=gd(kv,"scan_E0",s.scan_E0); s.scan_E1=gd(kv,"scan_E1",s.scan_E1); s.coarseN=gi(kv,"coarseN",s.coarseN); s.omp_threads=gi(kv,"omp_threads",s.omp_threads); s.debug=gs(kv,"debug","n").empty()?'n':gs(kv,"debug","n")[0];
    s.atmpi=gd(kv,"atmpi",s.atmpi); s.atmK=gd(kv,"atmK",s.atmK); s.eta_1=gd(kv,"eta_1",s.eta_1); s.eta_2=gd(kv,"eta_2",s.eta_2); s.alpha=gd(kv,"alpha",s.alpha); s.epsilon_h=gd(kv,"epsilon_h",s.epsilon_h); s.max_shell_num=gd(kv,"max_shell_num",s.max_shell_num); s.tolerance=gd(kv,"tolerance",s.tolerance); s.parity=gi(kv,"parity",s.parity); s.eig_tol=gd(kv,"eig_tol",s.eig_tol); s.norm_tol=gd(kv,"norm_tol",s.norm_tol); s.proj_tol=gd(kv,"proj_tol",s.proj_tol); s.Q0norm=gi(kv,"Q0norm",s.Q0norm?1:0)!=0; s.sort_orbit_flag=gi(kv,"sort_orbit_flag",s.sort_orbit_flag?1:0)!=0;
    opt.stride=std::max(1,gi(kv,"sample_stride",1)); opt.write_all_flat=gi(kv,"write_all_flatten_rows",0); opt.compare_tol=gd(kv,"compare_tol",0.0); opt.large_abs=gd(kv,"large_signflip_abs",1e8); opt.outdir=gs(kv,"output_dir",opt.outdir); opt.tag=gs(kv,"output_tag",opt.tag); s.output_dir=opt.outdir; s.output_tag=opt.tag;
    opt.k3.K3iso0=gd(kv,"K3iso0",0.0); opt.k3.K3iso1=gd(kv,"K3iso1",0.0); opt.k3.K3B=gd(kv,"K3B",0.0); opt.k3.K3E=gd(kv,"K3E",0.0);
    return s;
}
static comp safe_comp(const std::vector<std::vector<comp>>& c,int ch,int i){ if(ch<0||ch>=int(c.size())||i<0||i>=int(c[ch].size())) return comp(NAN,NAN); return c[ch][i]; }
static int safe_int(const std::vector<std::vector<int>>& c,int ch,int i){ if(ch<0||ch>=int(c.size())||i<0||i>=int(c[ch].size())) return 999999; return c[ch][i]; }
static double relnorm(const Eigen::MatrixXcd& A,const Eigen::MatrixXcd& B){ double n=B.norm(); return n>0 ? (A-B).norm()/n : A.norm(); }
static double offdiag_ell_norm(const Eigen::MatrixXcd& M,const std::vector<int>& ell){ double s=0; for(int r=0;r<M.rows();++r) for(int c=0;c<M.cols();++c) if(r<int(ell.size())&&c<int(ell.size())&&ell[r]!=ell[c]) s+=std::norm(M(r,c)); return std::sqrt(s); }
static double sp_norm(const Eigen::MatrixXcd& M,const std::vector<int>& ell){ double s=0; for(int r=0;r<M.rows();++r) for(int c=0;c<M.cols();++c) if(r<int(ell.size())&&c<int(ell.size())&&((ell[r]==0&&ell[c]==1)||(ell[r]==1&&ell[c]==0))) s+=std::norm(M(r,c)); return std::sqrt(s); }
static double block_norm(const Eigen::MatrixXcd& M,const std::vector<int>& ell,int e0,int e1){ double s=0; for(int r=0;r<M.rows();++r) for(int c=0;c<M.cols();++c) if(r<int(ell.size())&&c<int(ell.size())&&ell[r]==e0&&ell[c]==e1) s+=std::norm(M(r,c)); return std::sqrt(s); }
static std::vector<int> flattened_ell(const std::vector<std::vector<comp>>& plm,const std::vector<std::vector<comp>>& klm){ std::vector<int> ell; int A=plm.empty()?0:int(plm[0].size()); int B=klm.empty()?0:int(klm[0].size()); for(int i=0;i<A;++i) ell.push_back((int)std::llround(safe_comp(plm,3,i).real())); for(int i=0;i<B;++i) ell.push_back((int)std::llround(safe_comp(klm,3,i).real())); return ell; }

static void write_flat_compare(std::ostream& out,const std::string& label,const std::string& channel,int ei,double E,const std::vector<std::vector<comp>>& oldc,const std::vector<std::vector<int>>& oldn,const std::vector<std::vector<comp>>& newc,const std::vector<std::vector<int>>& newn,double tol,int write_all,int& mism){
    int rows=std::max(oldc.empty()?0:int(oldc[0].size()), newc.empty()?0:int(newc[0].size()));
    for(int row=0; row<rows; ++row){ comp op0=safe_comp(oldc,0,row),op1=safe_comp(oldc,1,row),op2=safe_comp(oldc,2,row),ol=safe_comp(oldc,3,row),om=safe_comp(oldc,4,row); comp np0=safe_comp(newc,0,row),np1=safe_comp(newc,1,row),np2=safe_comp(newc,2,row),nl=safe_comp(newc,3,row),nm=safe_comp(newc,4,row); double md=std::max({std::abs(op0-np0),std::abs(op1-np1),std::abs(op2-np2)}); double ld=std::abs(ol-nl), pd=std::abs(om-nm); double nd=0; for(int ch=0; ch<5; ++ch) nd=std::max(nd,double(std::abs(safe_int(oldn,ch,row)-safe_int(newn,ch,row)))); int exists=(row<(oldc.empty()?0:int(oldc[0].size())) && row<(newc.empty()?0:int(newc[0].size()))); int match=exists && md<=tol && ld<=tol && pd<=tol && nd<=tol; if(!match) ++mism; if(write_all||!match) out<<label<<' '<<ei<<' '<<E<<' '<<channel<<' '<<row<<' '<<exists<<' '<<match<<' '<<op0.real()<<' '<<op1.real()<<' '<<op2.real()<<' '<<ol.real()<<' '<<om.real()<<' '<<np0.real()<<' '<<np1.real()<<' '<<np2.real()<<' '<<nl.real()<<' '<<nm.real()<<' '<<md<<' '<<ld<<' '<<pd<<' '<<nd<<"\n";
    }
}

static Eigen::MatrixXcd assemble_QC_fullinv_projected(const k3df_fit_v31l::ProjectedQCCacheEntry& e,const k3df_fit_v31l::K3dfParameters& p,const PhysicsParams& par,char debug){
    if(!e.success) return Eigen::MatrixXcd(); int N=e.total_dim; Eigen::MatrixXcd K3(N,N); std::vector<comp> Kiso={comp(p.K3iso0,0),comp(p.K3iso1,0)}; k3_2plus1::K3mat_2plus1(K3,e.En_c,e.plm_config,e.klm_config,e.total_P,par.atmK,par.atmpi,Kiso,comp(p.K3B,0),comp(p.K3E,0),debug); return e.Vsel.adjoint()*(e.F3inv_full+K3)*e.Vsel;
}

struct RowMetrics{ int i=0,success=0,total_dim=0,proj_dim=0; double E=0; double F2_norm=0,G_norm=0,K2_norm=0,H_norm=0; double F2_sp=0,G_sp=0,K2_sp=0,F2_offell=0,G_offell=0,K2_offell=0; double V_orth=0,P_idem=0,leak_F2=0,leak_G=0,leak_K2=0,leak_H=0; double F3_equiv=0,F3_leak=0,QCproj_vs_full=0,qcproj_closest=NAN,qcfull_closest=NAN; int qcproj_idx=-1,qcfull_idx=-1; double herm_proj=NAN,herm_full=NAN; std::string error="OK"; };

static bool closest_eig_v31p(const Eigen::MatrixXcd& M,double& y,int& idx,double& herm){ if(M.rows()==0||M.rows()!=M.cols()||!M.allFinite()) return false; double n=M.norm(); herm=n>0?(M-M.adjoint()).norm()/n:0.0; Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(0.5*(M+M.adjoint())); if(es.info()!=Eigen::Success) return false; idx=0; double best=std::abs(es.eigenvalues()[0]); for(int i=1;i<es.eigenvalues().size();++i){ double a=std::abs(es.eigenvalues()[i]); if(a<best){best=a; idx=i;} } y=es.eigenvalues()[idx]; return true; }

int main(int argc,char** argv){ if(argc!=2){ std::cerr<<"Usage: "<<argv[0]<<" config/config_v31p_all_QC_mixed_sp_diagnostics.in\n"; return 1; } try{
    OptionsV31p opt; auto settings=settings_from_config_v31p(argv[1],opt); PhysicsParams par=k3df_fit_v31l::make_base_physics(settings); std::filesystem::create_directories(opt.outdir);
#ifdef _OPENMP
    omp_set_num_threads(settings.omp_threads); omp_set_max_active_levels(1);
#endif
    Eigen::setNbThreads(1);
    std::cout<<"[v31p] all mixed S/P QC diagnostics\n";
    std::cout<<"[v31p] This checks basis ordering, F2/G ell mixing, K2 ell block structure, projector leakage, inverse conventions, and QC eigen sign flips.\n";

    for(const auto& w1: opt.waves1){ settings.waves_vec_1=w1; std::string wtag=wave_tag_v31p(w1); std::cout<<"[v31p] waves_vec_1="<<list_int_str_v31p(w1)<<" waves_vec_2="<<list_int_str_v31p(opt.waves2)<<"\n";
        for(const auto& lab: opt.labels){ MomentumIrrepSpec spec=parse_label(lab); std::vector<int> nnP={spec.nnP[0],spec.nnP[1],spec.nnP[2]};
            std::string stem=opt.outdir+"/"+opt.tag+"_"+lab+"_"+wtag;
            std::ofstream flat(stem+"_flattened_basis_compare.dat"); flat<<std::setprecision(17)<<"# columns: label energy_i Ecm channel row exists match old_p0 old_p1 old_p2 old_ell old_proj_m new_p0 new_p1 new_p2 new_ell new_proj_m momdiff elldiff projmdiff intdiff\n";
            std::vector<RowMetrics> rows(settings.coarseN); int done=0,nextpct=10;
#pragma omp parallel for schedule(dynamic,1)
            for(int i=0;i<settings.coarseN;++i){ RowMetrics r; r.i=i; double t=(settings.coarseN==1)?0.0:double(i)/double(settings.coarseN-1); double E=settings.scan_E0+t*(settings.scan_E1-settings.scan_E0); r.E=E; if(i%opt.stride!=0){ rows[i]=r; continue; } try{
                comp pi=std::acos(-1.0); double L=par.L(); comp twopibyL=comp(2.0,0.0)*pi/comp(L,0.0); std::vector<comp> total_P(3); for(int a=0;a<3;++a) total_P[a]=twopibyL*double(nnP[a]); comp Ecm_c(E,0), En_c=Ecm_to_E(Ecm_c,total_P);
                std::vector<std::vector<comp>> plm_old(5),klm_old(5),plm_new(5),klm_new(5); std::vector<std::vector<int>> np_old(5),nk_old(5),np_new(5),nk_new(5);
                config_maker_4(plm_old,np_old,w1,En_c,total_P,par.atmK,par.atmK,par.atmpi,L,par.epsilon_h,par.max_shell_num,par.tolerance);
                config_maker_4(klm_old,nk_old,opt.waves2,En_c,total_P,par.atmpi,par.atmK,par.atmK,L,par.epsilon_h,par.max_shell_num,par.tolerance);
                config_maker_4_momentum_first(plm_new,np_new,w1,En_c,total_P,par.atmK,par.atmK,par.atmpi,L,par.epsilon_h,par.max_shell_num,par.tolerance);
                config_maker_4_momentum_first(klm_new,nk_new,opt.waves2,En_c,total_P,par.atmpi,par.atmK,par.atmK,L,par.epsilon_h,par.max_shell_num,par.tolerance);
                int N=int(plm_new[0].size()+klm_new[0].size()); r.total_dim=N; auto ell=flattened_ell(plm_new,klm_new); if(N>0){ Eigen::MatrixXcd F2(N,N),G(N,N),K2(N,N); F2_2plus1_mat(F2,En_c,plm_new,klm_new,total_P,par.atmK,par.atmpi,L,par.alpha,par.epsilon_h,par.max_shell_num,par.Q0norm); G_2plus1_mat(G,En_c,plm_new,klm_new,total_P,par.atmK,par.atmpi,L,par.alpha,par.epsilon_h,par.max_shell_num,par.Q0norm); K2inv_EREord2_2plus1_mat(K2,par.eta_1,par.eta_2,par.scatter_params_1,par.scatter_params_2,En_c,plm_new,klm_new,total_P,par.atmK,par.atmpi,par.epsilon_h,L); Eigen::MatrixXcd H=K2+F2+G; r.F2_norm=F2.norm(); r.G_norm=G.norm(); r.K2_norm=K2.norm(); r.H_norm=H.norm(); r.F2_sp=sp_norm(F2,ell); r.G_sp=sp_norm(G,ell); r.K2_sp=sp_norm(K2,ell); r.F2_offell=offdiag_ell_norm(F2,ell); r.G_offell=offdiag_ell_norm(G,ell); r.K2_offell=offdiag_ell_norm(K2,ell);
                    auto ce=k3df_fit_v31l::build_cache_entry(i,E,spec,settings,par,settings.debug); r.success=ce.success; r.proj_dim=ce.proj_dim; r.F3_equiv=ce.F3_equiv; r.F3_leak=ce.F3_leak; if(ce.success){ Eigen::MatrixXcd V=ce.Vsel; Eigen::MatrixXcd P=V*V.adjoint(); r.V_orth=(V.adjoint()*V-Eigen::MatrixXcd::Identity(V.cols(),V.cols())).norm(); r.P_idem=(P*P-P).norm(); auto leak=[&](const Eigen::MatrixXcd& A){ double n=A.norm(); return n>0?((Eigen::MatrixXcd::Identity(P.rows(),P.cols())-P)*A*P).norm()/n:0.0; }; r.leak_F2=leak(F2); r.leak_G=leak(G); r.leak_K2=leak(K2); r.leak_H=leak(H); Eigen::MatrixXcd QCp=k3df_fit_v31l::assemble_QC(ce,opt.k3,par,settings.debug); Eigen::MatrixXcd QCf=assemble_QC_fullinv_projected(ce,opt.k3,par,settings.debug); r.QCproj_vs_full=relnorm(QCp,QCf); closest_eig_v31p(QCp,r.qcproj_closest,r.qcproj_idx,r.herm_proj); closest_eig_v31p(QCf,r.qcfull_closest,r.qcfull_idx,r.herm_full); }
                }
            }catch(const std::exception& e){ r.success=0; r.error=e.what(); } rows[i]=r;
#pragma omp critical
                { ++done; progress_percent_log("v31p-"+lab+"-"+wtag,done,settings.coarseN,nextpct,10); }
            }
            // flattened compare sequential for deterministic file order
            int total_mismatch=0; for(int i=0;i<settings.coarseN;i+=opt.stride){ double t=(settings.coarseN==1)?0.0:double(i)/double(settings.coarseN-1); double E=settings.scan_E0+t*(settings.scan_E1-settings.scan_E0); comp pi=std::acos(-1.0); double L=par.L(); comp twopibyL=comp(2,0)*pi/comp(L,0); std::vector<comp> total_P(3); for(int a=0;a<3;++a) total_P[a]=twopibyL*double(nnP[a]); comp En_c=Ecm_to_E(comp(E,0),total_P); std::vector<std::vector<comp>> po(5),ko(5),pn(5),kn(5); std::vector<std::vector<int>> npo(5),nko(5),npn(5),nkn(5); config_maker_4(po,npo,w1,En_c,total_P,par.atmK,par.atmK,par.atmpi,L,par.epsilon_h,par.max_shell_num,par.tolerance); config_maker_4(ko,nko,opt.waves2,En_c,total_P,par.atmpi,par.atmK,par.atmK,L,par.epsilon_h,par.max_shell_num,par.tolerance); config_maker_4_momentum_first(pn,npn,w1,En_c,total_P,par.atmK,par.atmK,par.atmpi,L,par.epsilon_h,par.max_shell_num,par.tolerance); config_maker_4_momentum_first(kn,nkn,opt.waves2,En_c,total_P,par.atmpi,par.atmK,par.atmK,L,par.epsilon_h,par.max_shell_num,par.tolerance); write_flat_compare(flat,lab,"plm",i,E,po,npo,pn,npn,opt.compare_tol,opt.write_all_flat,total_mismatch); write_flat_compare(flat,lab,"klm",i,E,ko,nko,kn,nkn,opt.compare_tol,opt.write_all_flat,total_mismatch); }

            std::ofstream grid(stem+"_all_checks_grid.dat"); grid<<std::setprecision(17)<<"# i Ecm success total_dim proj_dim F2_norm G_norm K2_norm H_norm F2_sp G_sp K2_sp F2_offell G_offell K2_offell V_orth P_idem leak_F2 leak_G leak_K2 leak_H F3_equiv F3_leak QCproj_vs_full qcproj_closest qcproj_idx qcfull_closest qcfull_idx herm_proj herm_full error\n"; for(auto&r:rows){ if(r.i%opt.stride!=0) continue; std::string err=r.error; for(char&c:err) if(std::isspace((unsigned char)c)) c='_'; grid<<r.i<<' '<<r.E<<' '<<r.success<<' '<<r.total_dim<<' '<<r.proj_dim<<' '<<r.F2_norm<<' '<<r.G_norm<<' '<<r.K2_norm<<' '<<r.H_norm<<' '<<r.F2_sp<<' '<<r.G_sp<<' '<<r.K2_sp<<' '<<r.F2_offell<<' '<<r.G_offell<<' '<<r.K2_offell<<' '<<r.V_orth<<' '<<r.P_idem<<' '<<r.leak_F2<<' '<<r.leak_G<<' '<<r.leak_K2<<' '<<r.leak_H<<' '<<r.F3_equiv<<' '<<r.F3_leak<<' '<<r.QCproj_vs_full<<' '<<r.qcproj_closest<<' '<<r.qcproj_idx<<' '<<r.qcfull_closest<<' '<<r.qcfull_idx<<' '<<r.herm_proj<<' '<<r.herm_full<<' '<<err<<"\n"; }
            auto write_flips=[&](const std::string& path,bool full){ std::ofstream out(path); out<<std::setprecision(17)<<"# kind E_mid E_left E_right y_left y_right idx_left idx_right pole_like convention\n"; for(size_t i=0;i+1<rows.size();++i){ const auto&a=rows[i],&b=rows[i+1]; double y0=full?a.qcfull_closest:a.qcproj_closest, y1=full?b.qcfull_closest:b.qcproj_closest; int k0=full?a.qcfull_idx:a.qcproj_idx, k1=full?b.qcfull_idx:b.qcproj_idx; if(!a.success||!b.success||a.proj_dim!=b.proj_dim||!std::isfinite(y0)||!std::isfinite(y1)) continue; if(y0==0.0||y1==0.0||y0*y1<0.0){ std::string kind=(y0>0&&y1<0)?"ZERO":((y0<0&&y1>0)?"POLE":"ENDPOINT"); int pole_like=(std::max(std::abs(y0),std::abs(y1))>opt.large_abs)?1:0; out<<kind<<' '<<0.5*(a.E+b.E)<<' '<<a.E<<' '<<b.E<<' '<<y0<<' '<<y1<<' '<<k0<<' '<<k1<<' '<<pole_like<<' '<<(full?"fullF3inv_projected":"inverse_projectedF3")<<"\n"; } } };
            write_flips(stem+"_QC_closest_eig_signflips_inverse_projectedF3.dat",false); write_flips(stem+"_QC_closest_eig_signflips_fullF3inv_projected.dat",true);
            double maxK2off=0,maxF2sp=0,maxGsp=0,maxLeakH=0,maxQCdiff=0,maxVorth=0,maxF3eq=0; int ok=0; for(auto&r:rows){ if(r.success){++ok; maxK2off=std::max(maxK2off,r.K2_offell); maxF2sp=std::max(maxF2sp,r.F2_sp); maxGsp=std::max(maxGsp,r.G_sp); maxLeakH=std::max(maxLeakH,r.leak_H); maxQCdiff=std::max(maxQCdiff,r.QCproj_vs_full); maxVorth=std::max(maxVorth,r.V_orth); maxF3eq=std::max(maxF3eq,r.F3_equiv);} }
            std::ofstream rep(stem+"_all_checks_report.txt"); rep<<std::setprecision(17)<<"label = "<<lab<<"\nwaves_vec_1 = "<<list_int_str_v31p(w1)<<"\nwaves_vec_2 = "<<list_int_str_v31p(opt.waves2)<<"\nsuccess_points = "<<ok<<"\ntotal_points = "<<settings.coarseN<<"\nflattened_basis_mismatch_rows = "<<total_mismatch<<"\nmax_K2_offell_norm = "<<maxK2off<<"\nmax_F2_sp_norm = "<<maxF2sp<<"\nmax_G_sp_norm = "<<maxGsp<<"\nmax_projector_Vorth = "<<maxVorth<<"\nmax_H_leakage_rel = "<<maxLeakH<<"\nmax_F3_equiv = "<<maxF3eq<<"\nmax_QC_inverse_convention_reldiff = "<<maxQCdiff<<"\n";
            std::cout<<"[v31p] wrote "<<stem<<"_* ; flattened mismatches="<<total_mismatch<<" maxK2off="<<maxK2off<<" maxF2sp="<<maxF2sp<<" maxGsp="<<maxGsp<<" maxQCdiff="<<maxQCdiff<<"\n";
        }
    }
    std::cout<<"[v31p] [####################] 100% complete\n";
}catch(const std::exception& e){ std::cerr<<"[v31p-error] "<<e.what()<<"\n"; return 2; } return 0; }
