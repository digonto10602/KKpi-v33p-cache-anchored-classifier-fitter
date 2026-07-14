// projections.hpp
// Full C++ translation of projections.py + group_theory_defns.py
// Uses Eigen::MatrixXcd throughout (complex double matrices)
// Requires: Eigen3

#pragma once
#include <Eigen/Dense>
#include <Eigen/QR>
#include <vector>
#include <array>
#include <string>
#include <complex>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <cmath>

using MatC   = Eigen::MatrixXcd;
using VecC   = Eigen::VectorXcd;
using Vec3   = std::array<int, 3>;
using GroupElem = std::array<int, 3>;
using comp   = std::complex<double>;

static const double PI = M_PI;

enum class Waves { S, SP };

//==============================================================================
// UTILITY
//==============================================================================

// Zero out entries with |x| < tol
MatC chop(const MatC& A, double tol = 1e-13) {
    return A.unaryExpr([tol](comp x) -> comp {
        double re = std::abs(x.real()) < tol ? 0.0 : x.real();
        double im = std::abs(x.imag()) < tol ? 0.0 : x.imag();
        return comp(re, im);
    });
}

// Real matrix version (used internally for D-matrices)
Eigen::MatrixXd chop_real(const Eigen::MatrixXd& A, double tol = 1e-13) {
    return A.unaryExpr([tol](double x){ return std::abs(x) < tol ? 0.0 : x; });
}

// Block-diagonal stacking of complex matrices
MatC block_diag(const std::vector<MatC>& blocks) {
    int total_r = 0, total_c = 0;
    for (const auto& b : blocks) { total_r += b.rows(); total_c += b.cols(); }
    MatC out = MatC::Zero(total_r, total_c);
    int r = 0, c = 0;
    for (const auto& b : blocks) {
        out.block(r, c, b.rows(), b.cols()) = b;
        r += b.rows(); c += b.cols();
    }
    return out;
}

// Block-diagonal for square matrices
MatC block_diag_sq(const std::vector<MatC>& blocks) {
    int total = 0;
    for (const auto& b : blocks) total += b.rows();
    MatC out = MatC::Zero(total, total);
    int offset = 0;
    for (const auto& b : blocks) {
        out.block(offset, offset, b.rows(), b.cols()) = b;
        offset += b.rows();
    }
    return out;
}

//==============================================================================
// GROUP THEORY
//==============================================================================

// Apply signed permutation R=[i,j,k] to integer 3-vector
Vec3 cubic_transf(const Vec3& vec, const GroupElem& R) {
    Vec3 out;
    for (int a = 0; a < 3; a++)
        out[a] = (R[a] > 0 ? 1 : -1) * vec[std::abs(R[a]) - 1];
    return out;
}

// 24 proper rotations of O
std::vector<GroupElem> rotations_list() {
    return {
        {1,2,3},{2,3,1},{3,1,2},{1,3,-2},{2,-1,3},{3,2,-1},
        {1,-2,-3},{2,-3,-1},{3,-1,-2},{1,-3,2},{2,1,-3},{3,-2,1},
        {-1,2,-3},{-2,3,-1},{-3,1,-2},{-1,3,2},{-2,-1,-3},{-3,2,1},
        {-1,-2,3},{-2,-3,1},{-3,-1,2},{-1,-3,-2},{-2,1,3},{-3,-2,-1}
    };
}

bool is_proper_rotation(const GroupElem& R) {
    auto rots = rotations_list();
    return std::find(rots.begin(), rots.end(), R) != rots.end();
}

// Full Oh group (48 elements)
std::vector<GroupElem> Oh_list() {
    std::vector<GroupElem> out;
    std::array<int,3> perm = {1,2,3};
    do {
        for (int m = 0; m < 8; m++) {
            out.push_back({
                ((m>>0)&1 ? -1:1)*perm[0],
                ((m>>1)&1 ? -1:1)*perm[1],
                ((m>>2)&1 ? -1:1)*perm[2]
            });
        }
    } while (std::next_permutation(perm.begin(), perm.end()));
    return out;  // 48 elements
}

// Little group for given nnP
std::vector<GroupElem> little_group(const Vec3& nnP) {
    int x = nnP[0], y = nnP[1], z = nnP[2];
    if (x==0 && y==0 && z==0) return Oh_list();
    if (x==0 && y==0)         return {{1,2,3},{-1,2,3},{1,-2,3},{-1,-2,3},
                                       {2,1,3},{-2,1,3},{2,-1,3},{-2,-1,3}};
    if (x==y && z==0)         return {{1,2,3},{1,2,-3},{2,1,3},{2,1,-3}};
    if (x==y && y==z)         return {{1,2,3},{1,3,2},{2,1,3},{2,3,1},{3,1,2},{3,2,1}};
    if (z==0 && x!=0 && y!=0 && x!=y) return {{1,2,3},{1,2,-3}};
    if (x==y && z!=x)         return {{1,2,3},{2,1,3}};
    return {{1,2,3}};
}

// Irrep list for given nnP
std::vector<std::string> irrep_list(const Vec3& nnP) {
    int x = nnP[0], y = nnP[1], z = nnP[2];
    if (x==0&&y==0&&z==0) return {"A1g","A2g","Eg","T1g","T2g","A1u","A2u","Eu","T1u","T2u"};
    if (x==0&&y==0)        return {"A1","A2","B1","B2","E"};
    if (x==y&&z==0)        return {"A1","A2","B1","B2"};
    if (x==y&&y==z)        return {"A1","A2","E"};
    if ((z==0&&x!=y)||(x==y&&z!=x)) return {"A1","A2"};
    return {};
}

// Irrep dimension
int irrep_dim(const std::string& I) {
    if (I=="A1g"||I=="A1"||I=="A2g"||I=="A2"||I=="A1u"||I=="A2u"||I=="B1"||I=="B2")
        return 1;
    if (I=="Eg"||I=="E"||I=="Eu"||I=="E2") return 2;
    if (I=="T1g"||I=="T1"||I=="T2g"||I=="T2"||I=="T1u"||I=="T2u") return 3;
    throw std::invalid_argument("irrep_dim: unknown irrep " + I);
}

// get_lm_size
int get_lm_size(Waves waves) {
    if (waves == Waves::S)  return 1;
    if (waves == Waves::SP) return 4;
    throw std::invalid_argument("get_lm_size: unknown waves");
}

// Conjugacy class of group element R
std::string conj_class(const GroupElem& R) {
    int N_negs = 0, N_correct = 0;
    for (int i = 0; i < 3; i++) {
        if (R[i] < 0) N_negs++;
        if (std::abs(R[i]) == i+1) N_correct++;
    }
    if (N_correct == 3) {
        if (N_negs == 0) return "E";
        if (N_negs == 2) return "C4^2";
        if (N_negs == 3) return "i";
        if (N_negs == 1) return "sigma_h";
    }
    if (N_correct == 0) return (N_negs%2==0) ? "C3" : "S6";
    if (N_correct == 1) {
        int i_c = -1;
        for (int i = 0; i < 3; i++) if (std::abs(R[i])==i+1) { i_c=i; break; }
        if (N_negs%2 == 1) return (R[i_c] < 0) ? "C2" : "C4";
        else               return (R[i_c] > 0) ? "sigma_d" : "S4";
    }
    throw std::runtime_error("conj_class: unreachable");
}

// Character chi(R, I, nnP) — returns double (all real characters in these groups)
double chi(const GroupElem& R, const std::string& I, const Vec3& nnP) {
    std::string cc = conj_class(R);
    int x=nnP[0], y=nnP[1], z=nnP[2];

    // Oh (000)
    if (x==0&&y==0&&z==0) {
        if (I=="A1g"||I=="A1") return 1;
        if (I=="A2g"||I=="A2") return (cc=="C2"||cc=="C4"||cc=="sigma_d"||cc=="S4") ? -1 : 1;
        if (I=="Eg"||I=="E") {
            if (cc=="E"||cc=="C4^2"||cc=="i"||cc=="sigma_h") return 2;
            if (cc=="C3"||cc=="S6") return -1;
            return 0;
        }
        if (I=="T1g"||I=="T1") {
            if (cc=="E"||cc=="i") return 3;
            if (cc=="C3"||cc=="S6") return 0;
            if (cc=="C4"||cc=="S4") return 1;
            return -1;
        }
        if (I=="T2g"||I=="T2") {
            if (cc=="E"||cc=="i") return 3;
            if (cc=="C3"||cc=="S6") return 0;
            if (cc=="C2"||cc=="sigma_d") return 1;
            return -1;
        }
        if (I=="A1u") return (cc=="E"||cc=="C3"||cc=="C4^2"||cc=="C4"||cc=="C2") ? 1 : -1;
        if (I=="A2u") return (cc=="E"||cc=="C3"||cc=="C4^2"||cc=="S4"||cc=="sigma_d") ? 1 : -1;
        if (I=="Eu") {
            if (cc=="E"||cc=="C4^2") return 2;
            if (cc=="i"||cc=="sigma_h") return -2;
            if (cc=="S6") return 1;
            if (cc=="C3") return -1;
            return 0;
        }
        if (I=="T1u") {
            if (cc=="E") return 3; if (cc=="i") return -3;
            if (cc=="C4"||cc=="sigma_h"||cc=="sigma_d") return 1;
            if (cc=="C2"||cc=="C4^2"||cc=="S4") return -1;
            return 0;
        }
        if (I=="T2u") {
            if (cc=="E") return 3; if (cc=="i") return -3;
            if (cc=="C2"||cc=="S4"||cc=="sigma_h") return 1;
            if (cc=="C4"||cc=="C4^2"||cc=="sigma_d") return -1;
            return 0;
        }
        throw std::invalid_argument("chi: unknown Oh irrep: " + I);
    }

    // C4v (00z)
    if (x==0&&y==0) {
        if (I=="A1") return 1;
        if (I=="A2") return (cc=="sigma_h"||cc=="sigma_d") ? -1 : 1;
        if (I=="B1") return (cc=="C4"||cc=="sigma_d") ? -1 : 1;
        if (I=="B2") return (cc=="C4"||cc=="sigma_h") ? -1 : 1;
        if (I=="E"||I=="E2") {
            if (cc=="E") return 2;
            if (cc=="C4^2") return -2;
            return 0;
        }
        throw std::invalid_argument("chi: unknown C4v irrep: " + I);
    }

    // C2v (aa0)
    if (x==y&&z==0) {
        if (I=="A1") return 1;
        if (I=="A2") return (cc=="sigma_h"||cc=="sigma_d") ? -1 : 1;
        if (I=="B1") return (cc=="C2"||cc=="sigma_h") ? -1 : 1;
        if (I=="B2") return (cc=="C2"||cc=="sigma_d") ? -1 : 1;
        throw std::invalid_argument("chi: unknown C2v irrep: " + I);
    }

    // C3v (aaa)
    if (x==y&&y==z) {
        if (I=="A1") return 1;
        if (I=="A2") return (cc=="sigma_d") ? -1 : 1;
        if (I=="E"||I=="E2") {
            if (cc=="E") return 2;
            if (cc=="C3") return -1;
            return 0;
        }
        throw std::invalid_argument("chi: unknown C3v irrep: " + I);
    }

    // C2 (ab0 or aab)
    if (I=="A1") return 1;
    if (I=="A2") return (cc=="sigma_h"||cc=="sigma_d") ? -1 : 1;

    throw std::invalid_argument("chi: unhandled irrep/nnP: " + I
                                 + " nnP=(" + std::to_string(x) + ","
                                 + std::to_string(y) + ","
                                 + std::to_string(z) + ")");
}

// p-wave Wigner D-matrix (3x3 real) for group element R
// Returned as Eigen::MatrixXd (real) — cast to complex at use site
Eigen::MatrixXd Dmat11_real(const GroupElem& Rin) {
    std::array<int,3> R = {Rin[0], Rin[1], Rin[2]};
    int s = (R[0]>0) ? 1 : -1;

    auto swap_mat = [](int i, int j) {
        Eigen::MatrixXd U = Eigen::MatrixXd::Identity(3,3);
        U(i,i)=0; U(j,j)=0; U(i,j)=1; U(j,i)=1;
        return U;
    };

    if ((R[0]==1&&R[1]==2&&R[2]==3)||(R[0]==-1&&R[1]==-2&&R[2]==-3))
        return s * Eigen::MatrixXd::Identity(3,3);
    if ((R[0]==2&&R[1]==1&&R[2]==3)||(R[0]==-2&&R[1]==-1&&R[2]==-3))
        return s * swap_mat(0,2);
    if ((R[0]==1&&R[1]==3&&R[2]==2)||(R[0]==-1&&R[1]==-3&&R[2]==-2))
        return s * swap_mat(0,1);
    if ((R[0]==3&&R[1]==2&&R[2]==1)||(R[0]==-3&&R[1]==-2&&R[2]==-1))
        return s * swap_mat(1,2);

    // Cyclic permutations via composition
    if ((R[0]==2&&R[1]==3&&R[2]==1)||(R[0]==-2&&R[1]==-3&&R[2]==-1)) {
        GroupElem r1={1,3,2}, r2={2,1,3};
        return s * chop_real(Dmat11_real(r1)*Dmat11_real(r2));
    }
    if ((R[0]==3&&R[1]==1&&R[2]==2)||(R[0]==-3&&R[1]==-1&&R[2]==-2)) {
        GroupElem r1={3,2,1}, r2={2,1,3};
        return s * chop_real(Dmat11_real(r1)*Dmat11_real(r2));
    }

    // Single negation [1,2,-3]
    if ((R[0]==1&&R[1]==2&&R[2]==-3)||(R[0]==-1&&R[1]==-2&&R[2]==3))
        return s * chop_real(Eigen::Vector3d(1,-1,1).asDiagonal().toDenseMatrix());

    // Composite: one negation + transposition
    int r0=R[0], r1=R[1], r2=R[2];
    if (r0*r1>0 && r0*r2<0) {
        GroupElem a={1,2,-3}, b={r0,r1,-r2};
        return chop_real(Dmat11_real(a)*Dmat11_real(b));
    }
    if (r0*r2>0 && r0*r1<0) {
        GroupElem a={1,3,2}, b={r0,r2,r1};
        return chop_real(Dmat11_real(a)*Dmat11_real(b));
    }
    if (r1*r2>0 && r0*r1<0) {
        GroupElem a={3,2,1}, b={r2,r1,r0};
        return chop_real(Dmat11_real(a)*Dmat11_real(b));
    }
    throw std::runtime_error("Dmat11_real: unhandled case");
}

// Complex wrapper for Dmat11
MatC Dmat11(const GroupElem& R) {
    return Dmat11_real(R).cast<comp>();
}

//==============================================================================
// CORE PROJECTOR BUILDER
// Works directly from a flat nnk_list + lm_list (no orbit grouping needed)
// Basis order: nnk outer, lm inner
//==============================================================================

// Build full (N x N) Hermitian projector P^I in the given basis
MatC build_full_projector(
    const Vec3&                            nnP,
    const std::string&                     I,
    const std::vector<Vec3>&               nnk_list,
    const std::vector<std::pair<int,int>>& lm_list,
    int parity = -1)
{
    int d_I  = irrep_dim(I);
    auto LG  = little_group(nnP);
    int  Nk  = (int)nnk_list.size();
    int  Nlm = (int)lm_list.size();
    int  N   = Nk * Nlm;

    MatC P = MatC::Zero(N, N);

    for (int ik2 = 0; ik2 < Nk; ik2++) {
    for (int im2 = 0; im2 < Nlm; im2++) {
        int row          = ik2*Nlm + im2;
        auto [ell2, m2]  = lm_list[im2];

    for (int ik1 = 0; ik1 < Nk; ik1++) {
    for (int im1 = 0; im1 < Nlm; im1++) {
        int col          = ik1*Nlm + im1;
        auto [ell1, m1]  = lm_list[im1];

        if (ell1 != ell2) continue;  // D-matrix is block-diagonal in l

        comp val = 0.0;
        for (const auto& R : LG) {
            if (cubic_transf(nnk_list[ik1], R) != nnk_list[ik2]) continue;

            double par = (parity==-1 && !is_proper_rotation(R)) ? -1.0 : 1.0;
            double ch  = chi(R, I, nnP);

            comp D_val = 0.0;
            if (ell1 == 0) {
                D_val = comp(1.0, 0.0);
            } else if (ell1 == 1) {
                // Dmat11: 3x3 in (m=-1,0,+1) → index = m+1
                MatC D = Dmat11(R);
                D_val = D(m2+1, m1+1);
            }
            // Extend here for ell==2 using Dmat22 if needed

            val += comp(par * ch, 0.0) * D_val;
        }
        P(row, col) = val;
    }}}}

    return chop((comp(d_I,0)/comp((int)LG.size(),0)) * P);
}

// Extract eigenvalue-1 subspace and orthonormalize via QR
// Returns rectangular (N x N_irrep) matrix
MatC subspace_from_projector(const MatC& P_I) {
    int n = (int)P_I.rows();

    // Use self-adjoint solver (P_I should be Hermitian)
    Eigen::SelfAdjointEigenSolver<MatC> eig(P_I);
    const auto& evals = eig.eigenvalues();  // real
    const auto& evecs = eig.eigenvectors(); // complex

    std::vector<int> ivec;
    for (int i = 0; i < n; i++)
        if (std::abs(evals(i) - 1.0) < 1e-13) ivec.push_back(i);

    int expected = (int)std::round(P_I.trace().real());
    if ((int)ivec.size() != expected) {
        std::cerr << "subspace_from_projector: got " << ivec.size()
                  << " expected " << expected << "\n";
        throw std::runtime_error("Subspace dimension mismatch");
    }
    if (ivec.empty()) return MatC::Zero(n, 0);

    // Gather eigenvectors
    MatC Psub(n, (int)ivec.size());
    for (int j = 0; j < (int)ivec.size(); j++)
        Psub.col(j) = evecs.col(ivec[j]);
    Psub = chop(Psub);

    // QR orthonormalization (thin Q)
    Eigen::HouseholderQR<MatC> qr(Psub);
    MatC Q = qr.householderQ() * MatC::Identity(n, (int)ivec.size());
    return chop(Q);
}

//==============================================================================
// HIGH-LEVEL API: project QC3 matrix given orbit lists (Python-style interface)
//==============================================================================

// Build subspace projector for one flavor (all orbits, given waves)
MatC P_irrep_subspace_flavor(
    const Vec3&                 nnP,
    const std::string&          I,
    const std::vector<Vec3>&    nnk_list,  // ALL momenta for this flavor (flat)
    Waves                       waves,
    int                         parity)
{
    // Build lm_list from waves
    std::vector<std::pair<int,int>> lm_list;
    lm_list.push_back({0, 0});  // s-wave always
    if (waves == Waves::SP) {
        lm_list.push_back({1,-1});
        lm_list.push_back({1, 0});
        lm_list.push_back({1, 1});
    }

    MatC P_full = build_full_projector(nnP, I, nnk_list, lm_list, parity);
    return subspace_from_projector(P_full);
}

// Full 2+1 projection:  P_I = block_diag(P_flavor1, P_flavor2)
// Then returns P_I^dag * M * P_I
MatC irrep_proj_2plus1(
    const MatC&              M,
    const Vec3&              nnP,
    const std::string&       I,
    const std::vector<Vec3>& nnk_list_1,  // flavor-1 momenta (s+p)
    const std::vector<Vec3>& nnk_list_2,  // flavor-2 momenta (s only)
    int parity = -1)
{
    MatC Psub1 = P_irrep_subspace_flavor(nnP, I, nnk_list_1, Waves::SP, parity);
    MatC Psub2 = P_irrep_subspace_flavor(nnP, I, nnk_list_2, Waves::S,  parity);
    MatC P_I   = block_diag({Psub1, Psub2});

    if (P_I.cols() == 0) return MatC::Zero(0, 0);

    return chop(P_I.adjoint() * M * P_I);
}
