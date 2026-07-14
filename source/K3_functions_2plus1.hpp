#ifndef K3_FUNCTIONS_2PLUS1_HPP
#define K3_FUNCTIONS_2PLUS1_HPP

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "functions.h"             // comp, omega_func, sigma_pvec_based, kallentriangle, boost, etc.
#include "spherical_functions.h"   // spherical_harmonics == Python defns.y1real for ell=1

namespace k3_2plus1
{

using Vec3 = std::array<comp, 3>;

inline Vec3 make_vec3(comp x, comp y, comp z)
{
    return Vec3{x, y, z};
}

inline Vec3 vec_from_config(const std::vector<std::vector<comp>>& cfg, int a)
{
    return make_vec3(cfg[0][a], cfg[1][a], cfg[2][a]);
}

inline int ell_from_config(const std::vector<std::vector<comp>>& cfg, int a)
{
    return static_cast<int>(std::llround(std::real(cfg[3][a])));
}

inline int m_from_config(const std::vector<std::vector<comp>>& cfg, int a)
{
    return static_cast<int>(std::llround(std::real(cfg[4][a])));
}

inline std::vector<comp> to_stdvec(const Vec3& v)
{
    return std::vector<comp>{v[0], v[1], v[2]};
}

inline Vec3 from_stdvec(const std::vector<comp>& v)
{
    return make_vec3(v[0], v[1], v[2]);
}

inline Vec3 operator+(const Vec3& a, const Vec3& b)
{
    return make_vec3(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}

inline Vec3 operator-(const Vec3& a, const Vec3& b)
{
    return make_vec3(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

inline Vec3 operator*(const comp c, const Vec3& a)
{
    return make_vec3(c * a[0], c * a[1], c * a[2]);
}

inline Vec3 operator*(const double c, const Vec3& a)
{
    return make_vec3(comp(c, 0.0) * a[0], comp(c, 0.0) * a[1], comp(c, 0.0) * a[2]);
}

inline Vec3 operator/(const Vec3& a, const comp c)
{
    return make_vec3(a[0] / c, a[1] / c, a[2] / c);
}

inline comp dot3(const Vec3& a, const Vec3& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline comp norm2(const Vec3& a)
{
    return dot3(a, a);
}

inline comp norm3(const Vec3& a)
{
    return std::sqrt(norm2(a));
}

inline comp omega_vec(const Vec3& p, double m)
{
    return std::sqrt(norm2(p) + comp(m * m, 0.0));
}

inline comp y1real_cpp(const Vec3& v, int m)
{
    // Python defns.y1real(v,m) == C++ spherical_harmonics(v,1,m)
    return spherical_harmonics(to_stdvec(v), 1, m);
}

inline Vec3 boost_cpp(comp p0, const Vec3& pvec, comp E2, const Vec3& P2vec)
{
    return from_stdvec(::boost(p0, to_stdvec(pvec), E2, to_stdvec(P2vec)));
}

inline comp sigma_i_cpp(comp E, const Vec3& Pvec, const Vec3& pvec_i, double Mi)
{
    return sigma_pvec_based(E, to_stdvec(pvec_i), Mi, to_stdvec(Pvec));
}

inline comp qst2_i_cpp(comp E, const Vec3& Pvec, const Vec3& pvec_i,
                       double Mi, double Mj, double Mk)
{
    const comp sig_i = sigma_i_cpp(E, Pvec, pvec_i, Mi);
    return kallentriangle(sig_i, comp(Mj * Mj, 0.0), comp(Mk * Mk, 0.0)) / (comp(4.0, 0.0) * sig_i);
}

inline bool is_s(int ell, int m)
{
    return ell == 0 && m == 0;
}

inline bool is_p(int ell, int m)
{
    return ell == 1 && (m == -1 || m == 0 || m == 1);
}

inline double flavor_mass(int flavor, double M1, double M2)
{
    return (flavor == 1) ? M1 : M2;
}

inline double third_mass_for_E_piece(int i, int j, double M1, double M2)
{
    // This follows Python K3E.py:
    // if i==j==1: Mk=M2 else Mk=M1
    return (i == 1 && j == 1) ? M2 : M1;
}

inline comp K3B_element(
    comp E,
    const Vec3& Pvec,
    const Vec3& pvec,
    const Vec3& kvec,
    int i,
    int j,
    int ell_out,
    int m_out,
    int ell_in,
    int m_in,
    double M1,
    double M2)
{
    const double Mtot_d = 2.0 * M1 + M2;
    const comp Mtot2 = comp(Mtot_d * Mtot_d, 0.0);

    const double Mi = flavor_mass(i, M1, M2);
    const double Mj = flavor_mass(j, M1, M2);

    const comp Ecm2 = E * E - norm2(Pvec);
    const Vec3 P2p_vec = Pvec - pvec;
    const Vec3 P2k_vec = Pvec - kvec;

    const comp om_pi = omega_vec(pvec, Mi);
    const comp om_kj = omega_vec(kvec, Mj);

    comp q2_p1 = 0.0;
    comp q2_k1 = 0.0;
    comp om_qp1_1 = 0.0, om_qp1_2 = 0.0, pms_0 = 0.0, om_pp1 = 0.0;
    comp om_qk1_1 = 0.0, om_qk1_2 = 0.0, kms_0 = 0.0, om_kk1 = 0.0;
    Vec3 pp_vec1 = make_vec3(0.0, 0.0, 0.0);
    Vec3 kk_vec1 = make_vec3(0.0, 0.0, 0.0);

    if (i == 1)
    {
        q2_p1 = qst2_i_cpp(E, Pvec, pvec, M1, M1, M2);
        om_qp1_1 = std::sqrt(q2_p1 + comp(M1 * M1, 0.0));
        om_qp1_2 = std::sqrt(q2_p1 + comp(M2 * M2, 0.0));
        pms_0 = om_qp1_1 - om_qp1_2;
        pp_vec1 = boost_cpp(om_pi, pvec, E - om_pi, P2p_vec);
        om_pp1 = std::sqrt(norm2(pp_vec1) + comp(M1 * M1, 0.0));
    }

    if (j == 1)
    {
        q2_k1 = qst2_i_cpp(E, Pvec, kvec, M1, M1, M2);
        om_qk1_1 = std::sqrt(q2_k1 + comp(M1 * M1, 0.0));
        om_qk1_2 = std::sqrt(q2_k1 + comp(M2 * M2, 0.0));
        kms_0 = om_qk1_1 - om_qk1_2;
        kk_vec1 = boost_cpp(om_kj, kvec, E - om_kj, P2k_vec);
        om_kk1 = std::sqrt(norm2(kk_vec1) + comp(M1 * M1, 0.0));
    }

    comp val = 0.0;

    if (i == 2 && j == 2)
    {
        if (is_s(ell_out, m_out) && is_s(ell_in, m_in))
        {
            val += comp(2.0, 0.0) * (
                Ecm2 - E * (om_pi + om_kj) + dot3(Pvec, pvec + kvec)
                + comp(M2 * M2 - 4.0 * M1 * M1, 0.0));
        }
        return val / Mtot2;
    }

    if (i == 1 && j == 1)
    {
        if (is_s(ell_out, m_out) && is_s(ell_in, m_in))
        {
            val += E * (om_pi + om_kj) - dot3(Pvec, pvec + kvec)
                   - comp(6.0 * M1 * M1, 0.0)
                   + om_pp1 * pms_0 + om_kk1 * kms_0;
        }
        else if (is_p(ell_out, m_out) && is_s(ell_in, m_in))
        {
            val += comp(-2.0 / 3.0, 0.0) * y1real_cpp(pp_vec1, m_out);
        }
        else if (is_s(ell_out, m_out) && is_p(ell_in, m_in))
        {
            val += comp(-2.0 / 3.0, 0.0) * y1real_cpp(kk_vec1, m_in);
        }
        return val / Mtot2;
    }

    if (i == 1 && j == 2)
    {
        if (is_s(ell_out, m_out) && is_s(ell_in, m_in))
        {
            val += E * om_pi - dot3(pvec, Pvec) + Ecm2
                   - comp(2.0, 0.0) * E * om_kj
                   + comp(2.0, 0.0) * dot3(Pvec, kvec)
                   + om_pp1 * pms_0
                   + comp(M2 * M2 - 7.0 * M1 * M1, 0.0);
        }
        else if (is_p(ell_out, m_out) && is_s(ell_in, m_in))
        {
            val += comp(-2.0 / 3.0, 0.0) * y1real_cpp(pp_vec1, m_out);
        }
        return val / Mtot2;
    }

    if (i == 2 && j == 1)
    {
        if (is_s(ell_out, m_out) && is_s(ell_in, m_in))
        {
            val += E * om_kj - dot3(kvec, Pvec) + Ecm2
                   - comp(2.0, 0.0) * E * om_pi
                   + comp(2.0, 0.0) * dot3(Pvec, pvec)
                   + om_kk1 * kms_0
                   + comp(M2 * M2 - 7.0 * M1 * M1, 0.0);
        }
        else if (is_s(ell_out, m_out) && is_p(ell_in, m_in))
        {
            val += comp(-2.0 / 3.0, 0.0) * y1real_cpp(kk_vec1, m_in);
        }
        return val / Mtot2;
    }

    return comp(0.0, 0.0);
}

inline comp K3E_element(
    comp E,
    const Vec3& Pvec,
    const Vec3& pvec,
    const Vec3& kvec,
    int i,
    int j,
    int ell_out,
    int m_out,
    int ell_in,
    int m_in,
    double M1,
    double M2)
{
    const double Mtot_d = 2.0 * M1 + M2;
    const comp Mtot2 = comp(Mtot_d * Mtot_d, 0.0);

    const double Mi = flavor_mass(i, M1, M2);
    const double Mj = flavor_mass(j, M1, M2);
    const double Mk = third_mass_for_E_piece(i, j, M1, M2);

    const comp Ecm2 = E * E - norm2(Pvec);
    const Vec3 P2p_vec = Pvec - pvec;
    const Vec3 P2k_vec = Pvec - kvec;

    const comp om_pi = omega_vec(pvec, Mi);
    const comp om_kj = omega_vec(kvec, Mj);

    const comp E2p = E - om_pi;
    const comp E2k = E - om_kj;

    const comp sig_pi = sigma_i_cpp(E, Pvec, pvec, Mi);
    const comp sig_kj = sigma_i_cpp(E, Pvec, kvec, Mj);

    const comp q2_pi = kallentriangle(sig_pi, comp(Mj * Mj, 0.0), comp(Mk * Mk, 0.0)) / (comp(4.0, 0.0) * sig_pi);
    const comp q2_kj = kallentriangle(sig_kj, comp(Mi * Mi, 0.0), comp(Mk * Mk, 0.0)) / (comp(4.0, 0.0) * sig_kj);

    const comp om_qpi_1 = std::sqrt(q2_pi + comp(M1 * M1, 0.0));
    const comp om_qpi_2 = std::sqrt(q2_pi + comp(M2 * M2, 0.0));
    const comp om_qkj_1 = std::sqrt(q2_kj + comp(M1 * M1, 0.0));
    const comp om_qkj_2 = std::sqrt(q2_kj + comp(M2 * M2, 0.0));

    const comp pms_0 = om_qpi_1 - om_qpi_2;
    const comp kms_0 = om_qkj_1 - om_qkj_2;

    const Vec3 psk_vec = boost_cpp(om_pi, pvec, E2k, P2k_vec);
    const Vec3 ksp_vec = boost_cpp(om_kj, kvec, E2p, P2p_vec);

    const comp om_psk = std::sqrt(norm2(psk_vec) + comp(Mi * Mi, 0.0));
    const comp om_ksp = std::sqrt(norm2(ksp_vec) + comp(Mj * Mj, 0.0));

    const Vec3 P2psk_vec = boost_cpp(E2p, P2p_vec, E2k, P2k_vec);
    const Vec3 P2ksp_vec = boost_cpp(E2k, P2k_vec, E2p, P2p_vec);

    const comp P2psk_0 = std::sqrt(sig_pi + norm2(P2psk_vec));
    const comp P2ksp_0 = std::sqrt(sig_kj + norm2(P2ksp_vec));

    const Vec3 beta_p_vec = P2p_vec / E2p;
    const Vec3 beta_k_vec = P2k_vec / E2k;

    const comp gam_p = std::sqrt(comp(1.0, 0.0) / (comp(1.0, 0.0) - norm2(beta_p_vec)));
    const comp gam_k = std::sqrt(comp(1.0, 0.0) / (comp(1.0, 0.0) - norm2(beta_k_vec)));

    const comp beta_p = norm3(beta_p_vec);
    const comp beta_k = norm3(beta_k_vec);

    Vec3 beta_p_hat = beta_p_vec;
    Vec3 beta_k_hat = beta_k_vec;
    if (std::abs(beta_p) != 0.0) beta_p_hat = beta_p_vec / beta_p;
    if (std::abs(beta_k) != 0.0) beta_k_hat = beta_k_vec / beta_k;

    const comp bhat_dot = dot3(beta_k_hat, beta_p_hat);
    const comp bvec_dot = dot3(beta_p_vec, beta_k_vec);

    const Vec3 V = (-pms_0 * gam_p) * (
        beta_p_vec
        + dot3(beta_k_hat, beta_p_vec) * (gam_k - comp(1.0, 0.0)) * beta_k_hat
        - gam_k * beta_k_vec);

    const Vec3 Vp = (-kms_0 * gam_k) * (
        beta_k_vec
        + dot3(beta_p_hat, beta_k_vec) * (gam_p - comp(1.0, 0.0)) * beta_p_hat
        - gam_p * beta_p_vec);

    comp t[3][3] = {};
    for (int I = 0; I < 3; ++I)
    {
        const int Im = (I + 2) % 3; // Python: (I-1)%3 maps x,y,z -> +1,-1,0 local indices 2,0,1
        t[Im][Im] = comp(-1.0, 0.0);
        for (int J = 0; J < 3; ++J)
        {
            const int Jm = (J + 2) % 3;
            t[Im][Jm] += beta_p_hat[I] * beta_k_hat[J]
                * (gam_k * beta_k * gam_p * beta_p - bhat_dot * (gam_k - comp(1.0, 0.0)) * (gam_p - comp(1.0, 0.0)))
                - beta_k_hat[I] * beta_k_hat[J] * (gam_k - comp(1.0, 0.0))
                - beta_p_hat[I] * beta_p_hat[J] * (gam_p - comp(1.0, 0.0));
        }
    }

    comp val = 0.0;

    if (i == 2 && j == 2)
    {
        if (is_s(ell_out, m_out) && is_s(ell_in, m_in))
        {
            val += comp(2.0, 0.0) * (comp(M2 * M2, 0.0) - om_pi * om_kj + dot3(pvec, kvec));
        }
        return val / Mtot2;
    }

    if (i == 1 && j == 1)
    {
        if (is_s(ell_out, m_out) && is_s(ell_in, m_in))
        {
            val += comp(2.0 * M2 * M2, 0.0)
                - comp(0.5, 0.0) * (
                    Ecm2 - E * (om_pi + om_kj) + dot3(Pvec, pvec + kvec)
                    + om_pi * om_kj - dot3(pvec, kvec)
                    - P2psk_0 * kms_0 - P2ksp_0 * pms_0
                    + gam_p * gam_k * (comp(1.0, 0.0) - bvec_dot) * pms_0 * kms_0);
        }
        else if (is_p(ell_out, m_out) && is_s(ell_in, m_in))
        {
            val += comp(-1.0 / 3.0, 0.0) * y1real_cpp(P2ksp_vec + Vp, m_out);
        }
        else if (is_s(ell_out, m_out) && is_p(ell_in, m_in))
        {
            val += comp(-1.0 / 3.0, 0.0) * y1real_cpp(P2psk_vec + V, m_in);
        }
        else if (is_p(ell_out, m_out) && is_p(ell_in, m_in))
        {
            const int ro = m_out + 1; // m=-1,0,+1 -> 0,1,2
            const int ci = m_in + 1;
            val += comp(-2.0 / 3.0, 0.0) * t[ro][ci];
        }
        return val / Mtot2;
    }

    if (i == 1 && j == 2)
    {
        if (is_s(ell_out, m_out) && is_s(ell_in, m_in))
        {
            val += comp(2.0 * M2 * M2, 0.0)
                - (E - om_pi) * om_kj
                + dot3(P2p_vec, kvec)
                + pms_0 * om_ksp;
        }
        else if (is_p(ell_out, m_out) && is_s(ell_in, m_in))
        {
            val += comp(-2.0 / 3.0, 0.0) * y1real_cpp(ksp_vec, m_out);
        }
        return val / Mtot2;
    }

    if (i == 2 && j == 1)
    {
        if (is_s(ell_out, m_out) && is_s(ell_in, m_in))
        {
            val += comp(2.0 * M2 * M2, 0.0)
                - (E - om_kj) * om_pi
                + dot3(P2k_vec, pvec)
                + kms_0 * om_psk;
        }
        else if (is_s(ell_out, m_out) && is_p(ell_in, m_in))
        {
            val += comp(-2.0 / 3.0, 0.0) * y1real_cpp(psk_vec, m_in);
        }
        return val / Mtot2;
    }

    return comp(0.0, 0.0);
}

inline comp K3_element_raw(
    comp E,
    const Vec3& Pvec,
    const Vec3& pvec,
    const Vec3& kvec,
    int i,
    int j,
    int ell_out,
    int m_out,
    int ell_in,
    int m_in,
    const std::vector<comp>& Kiso,
    comp K3B_par,
    comp K3E_par,
    double M1,
    double M2)
{
    const double Mtot_d = 2.0 * M1 + M2;
    const comp Ecm2 = E * E - norm2(Pvec);
    const comp Delta = Ecm2 / comp(Mtot_d * Mtot_d, 0.0) - comp(1.0, 0.0);

    comp out = 0.0;

    if (is_s(ell_out, m_out) && is_s(ell_in, m_in))
    {
        comp powD = 1.0;
        for (std::size_t a = 0; a < Kiso.size(); ++a)
        {
            out += Kiso[a] * powD;
            powD *= Delta;
        }
    }

    if (std::abs(K3B_par) != 0.0)
    {
        out += K3B_par * K3B_element(E, Pvec, pvec, kvec, i, j,
                                     ell_out, m_out, ell_in, m_in, M1, M2);
    }

    if (std::abs(K3E_par) != 0.0)
    {
        out += K3E_par * K3E_element(E, Pvec, pvec, kvec, i, j,
                                     ell_out, m_out, ell_in, m_in, M1, M2);
    }

    return out;
}

inline double two_plus_one_block_factor(int i, int j)
{
    if (i == 1 && j == 1) return 1.0;
    if (i == 1 && j == 2) return 1.0 / std::sqrt(2.0);
    if (i == 2 && j == 1) return 1.0 / std::sqrt(2.0);
    if (i == 2 && j == 2) return 0.5;
    throw std::runtime_error("two_plus_one_block_factor: invalid flavor");
}

inline void K3mat_2plus1(
    Eigen::MatrixXcd& K3,
    comp E,
    const std::vector<std::vector<comp>>& plm_config,
    const std::vector<std::vector<comp>>& klm_config,
    const std::vector<comp>& total_P_vec,
    double M1,
    double M2,
    const std::vector<comp>& Kiso,
    comp K3B_par,
    comp K3E_par,
    char debug = 'n')
{
    const int dim1 = static_cast<int>(plm_config[0].size());
    const int dim2 = static_cast<int>(klm_config[0].size());
    const int N = dim1 + dim2;

    if (static_cast<int>(total_P_vec.size()) < 3)
        throw std::runtime_error("K3mat_2plus1: total_P_vec must have size >= 3");

    K3 = Eigen::MatrixXcd::Zero(N, N);
    const Vec3 Pvec = make_vec3(total_P_vec[0], total_P_vec[1], total_P_vec[2]);

    for (int row = 0; row < N; ++row)
    {
        const bool row_is_flav1 = (row < dim1);
        const int i = row_is_flav1 ? 1 : 2;
        const int rr = row_is_flav1 ? row : row - dim1;
        const auto& row_cfg = row_is_flav1 ? plm_config : klm_config;

        const Vec3 pvec = vec_from_config(row_cfg, rr);
        const int ell_out = ell_from_config(row_cfg, rr);
        const int m_out = m_from_config(row_cfg, rr);

        for (int col = 0; col < N; ++col)
        {
            const bool col_is_flav1 = (col < dim1);
            const int j = col_is_flav1 ? 1 : 2;
            const int cc = col_is_flav1 ? col : col - dim1;
            const auto& col_cfg = col_is_flav1 ? plm_config : klm_config;

            const Vec3 kvec = vec_from_config(col_cfg, cc);
            const int ell_in = ell_from_config(col_cfg, cc);
            const int m_in = m_from_config(col_cfg, cc);

            const comp raw = K3_element_raw(
                E, Pvec, pvec, kvec, i, j,
                ell_out, m_out, ell_in, m_in,
                Kiso, K3B_par, K3E_par, M1, M2);

            K3(row, col) = comp(two_plus_one_block_factor(i, j), 0.0) * raw;
        }
    }

    if (debug == 'y')
    {
        std::cout << "[K3mat_2plus1] dim1=" << dim1
                  << " dim2=" << dim2
                  << " N=" << N
                  << " max|K3|=" << K3.cwiseAbs().maxCoeff()
                  << "\n";
    }
}

} // namespace k3_2plus1

#endif // K3_FUNCTIONS_2PLUS1_HPP
