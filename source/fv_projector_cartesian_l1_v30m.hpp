#pragma once

// v30m finite-volume projector diagnostics with corrected Cartesian ell=1 block.
//
// This header implements the projector prescription of Blanton--Romero-Lopez--Sharpe
// Sec. 2.4 / App. A.3 in an explicitly testable way.  It builds the little-group
// action from separate finite-volume momentum-shift and angular-momentum matrices,
//
//   S_R : |k,ell,m> -> |R k,ell,m>
//   W_R : |k,ell,m> -> Pi(R) sum_{m'} D^ell_{m' m}(R)|k,ell,m'>
//
// and tests a controlled set of active/passive and transpose/inverse conventions.
// The best convention is selected by representation closure + P_I idempotency.

#include <Eigen/Dense>
#include <complex>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>

namespace fvproj_v30j {
// NOTE: namespace is intentionally kept as fvproj_v30j so existing v30k diagnostic
// code can call this header without invasive rewrites.  The implementation below
// is v30m: ell=1 uses the Cartesian signed-permutation action in the Blanton-Sharpe real basis.

struct Convention {
    int index = -1;
    bool use_inverse_momentum = false;  // S target momentum = R^{-1}k instead of Rk
    bool use_inverse_D = false;         // D argument R^{-1} instead of R
    bool transpose_D = false;           // D_{m_src,m_tgt} instead of D_{m_tgt,m_src}
    bool order_WS = false;              // U=W*S instead of S*W
    int parity_mode = 1;                // 0 none, 1 Pi(R), 2 Pi(R)^ell
    std::string name() const {
        std::ostringstream os;
        os << "c" << index
           << "_mom" << (use_inverse_momentum?"Rinv":"R")
           << "_D" << (use_inverse_D?"Rinv":"R")
           << (transpose_D?"T":"")
           << '_' << (order_WS?"WS":"SW")
           << "_par" << parity_mode;
        return os.str();
    }
};

struct RepDiagnostics {
    double group_size = std::numeric_limits<double>::quiet_NaN();
    double best_convention_index = std::numeric_limits<double>::quiet_NaN();
    double unitarity_max = std::numeric_limits<double>::quiet_NaN();
    double closure_AB_max = std::numeric_limits<double>::quiet_NaN();
    double closure_BA_max = std::numeric_limits<double>::quiet_NaN();
    double closure_best_max = std::numeric_limits<double>::quiet_NaN();
    double projector_herm_res = std::numeric_limits<double>::quiet_NaN();
    double projector_idem_res = std::numeric_limits<double>::quiet_NaN();
    double projector_diff_old_rel = std::numeric_limits<double>::quiet_NaN();
};

struct BestProjectorResult {
    Convention conv;
    RepDiagnostics diag;
    Eigen::MatrixXcd P;
};

inline double hermitian_rel_res(const Eigen::MatrixXcd& A) {
    if(A.rows()==0 || A.cols()==0 || A.rows()!=A.cols() || !A.allFinite()) return std::numeric_limits<double>::quiet_NaN();
    const double n = A.norm();
    return (n>0.0) ? (A-A.adjoint()).norm()/n : 0.0;
}

inline double idempotent_rel_res(const Eigen::MatrixXcd& P) {
    if(P.rows()==0 || P.cols()==0 || P.rows()!=P.cols() || !P.allFinite()) return std::numeric_limits<double>::quiet_NaN();
    const double n = P.norm();
    return (n>0.0) ? (P*P-P).norm()/n : 0.0;
}

inline bool signed_perm_equal(const std::vector<int>& a, const std::vector<int>& b) {
    return a.size()==3 && b.size()==3 && a[0]==b[0] && a[1]==b[1] && a[2]==b[2];
}

inline std::vector<int> compose_AB(const std::vector<int>& A, const std::vector<int>& B) {
    // C = A o B using cubic_transf(cubic_transf(v,B),A) = cubic_transf(v,C).
    std::vector<int> C(3,0);
    for(int j=0;j<3;++j){
        const int b = B[j];
        const int ib = std::abs(b)-1;
        const int sb = (b>0 ? 1 : -1);
        const int a = A[ib];
        const int ia = std::abs(a);
        const int sa = (a>0 ? 1 : -1);
        C[j] = sb * sa * ia;
    }
    return C;
}

inline std::vector<int> inverse_signed_perm(const std::vector<int>& R) {
    // Inverse I such that R o I = I o R = identity in the compose_AB convention.
    std::vector<int> I(3,0);
    for(int j=0;j<3;++j){
        const int r = R[j];
        const int i = std::abs(r)-1;
        const int s = (r>0 ? 1 : -1);
        I[i] = s * (j+1);
    }
    return I;
}

inline int find_group_index(const std::vector<std::vector<int>>& LG, const std::vector<int>& R) {
    for(int i=0;i<(int)LG.size();++i) if(signed_perm_equal(LG[i],R)) return i;
    return -1;
}

inline int irrep_dimension(const std::string& I) {
    if(I=="A1" || I=="A2" || I=="A1g" || I=="A2g" || I=="A1u" || I=="A2u" || I=="B1" || I=="B2") return 1;
    if(I=="E" || I=="Eg" || I=="Eu" || I=="E2") return 2;
    if(I=="T1" || I=="T2" || I=="T1g" || I=="T2g" || I=="T1u" || I=="T2u") return 3;
    throw std::runtime_error("fvproj_v30j::irrep_dimension: unknown irrep " + I);
}

inline std::vector<Convention> all_conventions() {
    std::vector<Convention> out;
    int idx=0;
    for(bool mom_inv : {false,true})
    for(bool d_inv : {false,true})
    for(bool d_T : {false,true})
    for(bool ws : {false,true})
    for(int pmode : {1,0,2}) {
        Convention c; c.index=idx++; c.use_inverse_momentum=mom_inv; c.use_inverse_D=d_inv; c.transpose_D=d_T; c.order_WS=ws; c.parity_mode=pmode;
        out.push_back(c);
    }
    return out;
}

inline double parity_factor_for_l(int parity, const std::vector<int>& R, int ell, int parity_mode) {
    if(parity != -1) return 1.0;
    if(is_in_rotations_list(R)) return 1.0;
    if(parity_mode == 0) return 1.0;
    if(parity_mode == 1) return -1.0;                 // Eq. (2.54): Pi(R)
    if(parity_mode == 2) return (ell % 2 == 0) ? 1.0 : -1.0; // test (-1)^ell convention
    return 1.0;
}

inline Eigen::MatrixXcd S_single_flavor(
        const std::vector<std::vector<comp>>& lm_config,
        const std::vector<std::vector<int>>& n_config,
        const Convention& conv,
        const std::vector<int>& R)
{
    const int dim = (int)n_config[0].size();
    Eigen::MatrixXcd S = Eigen::MatrixXcd::Zero(dim, dim);
    const std::vector<int> Rmom = conv.use_inverse_momentum ? inverse_signed_perm(R) : R;
    for(int source=0; source<dim; ++source){
        const std::vector<int> n_src = {n_config[0][source], n_config[1][source], n_config[2][source]};
        const std::vector<int> Rn = cubic_transf(n_src, Rmom);
        const int ell_src = (int)std::llround(std::real(lm_config[3][source]));
        const int m_src   = (int)std::llround(std::real(lm_config[4][source]));
        for(int target=0; target<dim; ++target){
            const std::vector<int> n_tgt = {n_config[0][target], n_config[1][target], n_config[2][target]};
            if(Rn != n_tgt) continue;
            const int ell_tgt = (int)std::llround(std::real(lm_config[3][target]));
            const int m_tgt   = (int)std::llround(std::real(lm_config[4][target]));
            if(ell_tgt==ell_src && m_tgt==m_src) S(target,source)=comp(1.0,0.0);
        }
    }
    return S;
}


inline int cart_axis_from_m_v30m(int m) {
    // Blanton--Romero-Lopez--Sharpe App. A.1 real harmonics:
    //   Y_11 ~ x, Y_10 ~ z, Y_1,-1 ~ y.
    // Code ordering is usually m=-1,0,+1, therefore axes are [y,z,x].
    if(m ==  1) return 0; // x
    if(m ==  0) return 2; // z
    if(m == -1) return 1; // y
    return -1;
}

inline double cartesian_signed_perm_element_v30m(int m_tgt, int m_src, const std::vector<int>& R) {
    const int a_t = cart_axis_from_m_v30m(m_tgt);
    const int a_s = cart_axis_from_m_v30m(m_src);
    if(a_t < 0 || a_s < 0 || R.size()!=3) return 0.0;
    // cubic_transf implements out[row] = sign(R[row])*vec[abs(R[row])-1].
    const int src_axis_for_target = std::abs(R[a_t]) - 1;
    const int sign = (R[a_t] > 0) ? 1 : -1;
    return (src_axis_for_target == a_s) ? double(sign) : 0.0;
}

inline double D_real_element_v30m(int ell, int m_tgt, int m_src, const std::vector<int>& R) {
    if(ell == 0) return (m_tgt == 0 && m_src == 0) ? 1.0 : 0.0;
    if(ell == 1) return cartesian_signed_perm_element_v30m(m_tgt, m_src, R);
    // Fallback for future ell>1 tests. The current v30m package is intended for s+p.
    return real_wigner_d::D_real_element(ell, m_tgt, m_src, R);
}

inline Eigen::MatrixXcd W_single_flavor(
        const std::vector<std::vector<comp>>& lm_config,
        const std::vector<std::vector<int>>& n_config,
        int parity,
        const Convention& conv,
        const std::vector<int>& R)
{
    const int dim = (int)n_config[0].size();
    Eigen::MatrixXcd W = Eigen::MatrixXcd::Zero(dim, dim);
    const std::vector<int> Rd = conv.use_inverse_D ? inverse_signed_perm(R) : R;
    for(int source=0; source<dim; ++source){
        const std::vector<int> n_src = {n_config[0][source], n_config[1][source], n_config[2][source]};
        const int ell_src = (int)std::llround(std::real(lm_config[3][source]));
        const int m_src   = (int)std::llround(std::real(lm_config[4][source]));
        for(int target=0; target<dim; ++target){
            const std::vector<int> n_tgt = {n_config[0][target], n_config[1][target], n_config[2][target]};
            if(n_tgt != n_src) continue;
            const int ell_tgt = (int)std::llround(std::real(lm_config[3][target]));
            if(ell_tgt != ell_src) continue;
            const int m_tgt = (int)std::llround(std::real(lm_config[4][target]));
            const double pf = parity_factor_for_l(parity,R,ell_src,conv.parity_mode);
            const double d = conv.transpose_D
                ? D_real_element_v30m(ell_src, m_src, m_tgt, Rd)
                : D_real_element_v30m(ell_src, m_tgt, m_src, Rd);
            W(target,source) += comp(pf*d,0.0);
        }
    }
    return W;
}

inline Eigen::MatrixXcd U_single_flavor(
        const std::vector<std::vector<comp>>& lm_config,
        const std::vector<std::vector<int>>& n_config,
        int parity,
        const Convention& conv,
        const std::vector<int>& R)
{
    const Eigen::MatrixXcd S = S_single_flavor(lm_config,n_config,conv,R);
    const Eigen::MatrixXcd W = W_single_flavor(lm_config,n_config,parity,conv,R);
    return conv.order_WS ? (W*S) : (S*W);
}

inline Eigen::MatrixXcd U_2plus1(
        const std::vector<std::vector<comp>>& plm_config,
        const std::vector<std::vector<int>>& np_config,
        const std::vector<std::vector<comp>>& klm_config,
        const std::vector<std::vector<int>>& nk_config,
        int parity,
        const Convention& conv,
        const std::vector<int>& R)
{
    const Eigen::MatrixXcd U1 = U_single_flavor(plm_config,np_config,parity,conv,R);
    const Eigen::MatrixXcd U2 = U_single_flavor(klm_config,nk_config,parity,conv,R);
    Eigen::MatrixXcd U = Eigen::MatrixXcd::Zero(U1.rows()+U2.rows(), U1.cols()+U2.cols());
    U.block(0,0,U1.rows(),U1.cols()) = U1;
    U.block(U1.rows(),U1.cols(),U2.rows(),U2.cols()) = U2;
    return U;
}

inline Eigen::MatrixXcd projector_for_convention(
        const std::vector<std::vector<comp>>& plm_config,
        const std::vector<std::vector<int>>& np_config,
        const std::vector<std::vector<comp>>& klm_config,
        const std::vector<std::vector<int>>& nk_config,
        const std::string& irrep,
        const std::vector<comp>& nnP_config,
        int parity,
        const Convention& conv)
{
    std::vector<std::vector<int>> LG = little_group(const_cast<std::vector<comp>&>(nnP_config));
    const int dI = irrep_dimension(irrep);
    const int dim = (int)np_config[0].size() + (int)nk_config[0].size();
    Eigen::MatrixXcd P = Eigen::MatrixXcd::Zero(dim, dim);
    for(const auto& R : LG){
        const double chiR = (double)chi(R, irrep, const_cast<std::vector<comp>&>(nnP_config));
        P += chiR * U_2plus1(plm_config,np_config,klm_config,nk_config,parity,conv,R);
    }
    if(!LG.empty()) P *= ((double)dI / (double)LG.size());
    // Eq. (2.56) should already be Hermitian for real irreps; symmetrize only to
    // suppress roundoff while preserving convention tests via closure separately.
    P = 0.5*(P + P.adjoint());
    return P;
}

inline RepDiagnostics diagnose_convention(
        const std::vector<std::vector<comp>>& plm_config,
        const std::vector<std::vector<int>>& np_config,
        const std::vector<std::vector<comp>>& klm_config,
        const std::vector<std::vector<int>>& nk_config,
        const std::vector<comp>& nnP_config,
        int parity,
        const std::string& irrep,
        const Convention& conv,
        const Eigen::MatrixXcd* P_old = nullptr)
{
    RepDiagnostics d;
    d.best_convention_index = conv.index;
    std::vector<std::vector<int>> LG = little_group(const_cast<std::vector<comp>&>(nnP_config));
    d.group_size = (double)LG.size();
    std::vector<Eigen::MatrixXcd> Ulist;
    Ulist.reserve(LG.size());
    double unit = 0.0, closeAB = 0.0, closeBA = 0.0, closeBest = 0.0;
    for(const auto& R : LG){
        Eigen::MatrixXcd U = U_2plus1(plm_config,np_config,klm_config,nk_config,parity,conv,R);
        Ulist.push_back(U);
        const Eigen::MatrixXcd I = Eigen::MatrixXcd::Identity(U.rows(), U.cols());
        unit = std::max(unit, (U.adjoint()*U - I).norm()/std::max(I.norm(),1.0e-300));
    }
    for(int a=0; a<(int)LG.size(); ++a){
        for(int b=0; b<(int)LG.size(); ++b){
            const auto CAB = compose_AB(LG[a], LG[b]);
            const auto CBA = compose_AB(LG[b], LG[a]);
            const int iAB = find_group_index(LG, CAB);
            const int iBA = find_group_index(LG, CBA);
            double rAB = std::numeric_limits<double>::infinity();
            double rBA = std::numeric_limits<double>::infinity();
            if(iAB >= 0) rAB = (Ulist[a]*Ulist[b] - Ulist[iAB]).norm() / std::max(Ulist[iAB].norm(), 1.0e-300);
            if(iBA >= 0) rBA = (Ulist[a]*Ulist[b] - Ulist[iBA]).norm() / std::max(Ulist[iBA].norm(), 1.0e-300);
            closeAB = std::max(closeAB, rAB);
            closeBA = std::max(closeBA, rBA);
            closeBest = std::max(closeBest, std::min(rAB,rBA));
        }
    }
    const Eigen::MatrixXcd Pnew = projector_for_convention(plm_config,np_config,klm_config,nk_config,irrep,nnP_config,parity,conv);
    d.unitarity_max = unit;
    d.closure_AB_max = closeAB;
    d.closure_BA_max = closeBA;
    d.closure_best_max = closeBest;
    d.projector_herm_res = hermitian_rel_res(Pnew);
    d.projector_idem_res = idempotent_rel_res(Pnew);
    if(P_old && P_old->rows()==Pnew.rows() && P_old->cols()==Pnew.cols() && P_old->allFinite()) {
        d.projector_diff_old_rel = (Pnew - *P_old).norm() / std::max(Pnew.norm(), 1.0e-300);
    }
    return d;
}

inline BestProjectorResult best_projector(
        const std::vector<std::vector<comp>>& plm_config,
        const std::vector<std::vector<int>>& np_config,
        const std::vector<std::vector<comp>>& klm_config,
        const std::vector<std::vector<int>>& nk_config,
        const std::vector<comp>& nnP_config,
        int parity,
        const std::string& irrep,
        const Eigen::MatrixXcd* P_old = nullptr)
{
    BestProjectorResult best;
    double best_score = std::numeric_limits<double>::infinity();
    for(const Convention& c : all_conventions()){
        RepDiagnostics d = diagnose_convention(plm_config,np_config,klm_config,nk_config,nnP_config,parity,irrep,c,P_old);
        // prioritize closure, then projector idempotency, then unitarity
        const double score = 1.0e6*d.closure_best_max + 1.0e3*d.projector_idem_res + d.unitarity_max;
        if(std::isfinite(score) && score < best_score){
            best_score = score;
            best.conv = c;
            best.diag = d;
            best.P = projector_for_convention(plm_config,np_config,klm_config,nk_config,irrep,nnP_config,parity,c);
        }
    }
    return best;
}

inline void P_irrep_projection_2plus1_best(
        Eigen::MatrixXcd& P_I,
        const std::vector<std::vector<comp>>& plm_config,
        const std::vector<std::vector<int>>& np_config,
        const std::vector<std::vector<comp>>& klm_config,
        const std::vector<std::vector<int>>& nk_config,
        const std::string& irrep,
        const std::vector<comp>& nnP_config,
        int parity)
{
    BestProjectorResult b = best_projector(plm_config,np_config,klm_config,nk_config,nnP_config,parity,irrep,nullptr);
    P_I = b.P;
}

inline double equivariance_rel_res(const Eigen::MatrixXcd& M, const Eigen::MatrixXcd& U) {
    if(M.rows()==0 || M.cols()==0 || M.rows()!=M.cols() || U.rows()!=M.rows() || U.cols()!=M.cols() || !M.allFinite() || !U.allFinite())
        return std::numeric_limits<double>::quiet_NaN();
    const double n = M.norm();
    const Eigen::MatrixXcd rotated = U.adjoint() * M * U;
    return (n>0.0) ? (rotated - M).norm()/n : (rotated - M).norm();
}

inline double max_equivariance_over_little_group(
        const Eigen::MatrixXcd& M,
        const std::vector<std::vector<comp>>& plm_config,
        const std::vector<std::vector<int>>& np_config,
        const std::vector<std::vector<comp>>& klm_config,
        const std::vector<std::vector<int>>& nk_config,
        const std::vector<comp>& nnP_config,
        int parity,
        const Convention& conv)
{
    std::vector<std::vector<int>> LG = little_group(const_cast<std::vector<comp>&>(nnP_config));
    double out = 0.0;
    for(const auto& R : LG){
        const Eigen::MatrixXcd U = U_2plus1(plm_config,np_config,klm_config,nk_config,parity,conv,R);
        out = std::max(out, equivariance_rel_res(M,U));
    }
    return out;
}

inline double max_equivariance_best_over_little_group(
        const Eigen::MatrixXcd& M,
        const std::vector<std::vector<comp>>& plm_config,
        const std::vector<std::vector<int>>& np_config,
        const std::vector<std::vector<comp>>& klm_config,
        const std::vector<std::vector<int>>& nk_config,
        const std::vector<comp>& nnP_config,
        int parity,
        const std::string& irrep)
{
    BestProjectorResult b = best_projector(plm_config,np_config,klm_config,nk_config,nnP_config,parity,irrep,nullptr);
    return max_equivariance_over_little_group(M,plm_config,np_config,klm_config,nk_config,nnP_config,parity,b.conv);
}

} // namespace fvproj_v30j // v30m implementation
