#ifndef REAL_WIGNER_D_HPP
#define REAL_WIGNER_D_HPP

#include <Eigen/Dense>
#include <complex>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <string>

namespace real_wigner_d
{
    using comp = std::complex<double>;

    // ============================================================
    // basic helpers
    // ============================================================
    inline int midx(const int l, const int m)
    {
        return m + l; // maps m in [-l,l] -> [0,2l]
    }

    inline void check_m_range(const int l, const int m)
    {
        if (m < -l || m > l) {
            throw std::runtime_error("check_m_range: m out of range for given l");
        }
    }

    inline double factorial_int(const int n)
    {
        if (n < 0) {
            throw std::runtime_error("factorial_int: negative input");
        }

        double out = 1.0;
        for (int k = 2; k <= n; ++k) {
            out *= static_cast<double>(k);
        }
        return out;
    }

    inline bool approx_equal(const double a, const double b, const double tol = 1e-12)
    {
        return std::abs(a - b) < tol;
    }

    inline bool approx_equal_matrix(const Eigen::MatrixXd& A,
                                    const Eigen::MatrixXd& B,
                                    const double tol = 1e-12)
    {
        if (A.rows() != B.rows() || A.cols() != B.cols()) {
            return false;
        }

        for (int i = 0; i < A.rows(); ++i) {
            for (int j = 0; j < A.cols(); ++j) {
                if (!approx_equal(A(i,j), B(i,j), tol)) {
                    return false;
                }
            }
        }
        return true;
    }

    // ============================================================
    // Convert signed permutation to 3x3 orthogonal matrix
    //
    // Convention:
    //   op = {a,b,c}
    // means
    //   (O r)_0 = sign(a) * r_{|a|-1}
    //   (O r)_1 = sign(b) * r_{|b|-1}
    //   (O r)_2 = sign(c) * r_{|c|-1}
    //
    // Example:
    //   { 1, 2, 3} -> identity
    //   {-1, 2, 3} -> reflect x
    //   { 2, 1, 3} -> swap x and y
    // ============================================================
    inline Eigen::Matrix3d signed_perm_to_matrix(const std::vector<int>& op)
    {
        if (op.size() != 3) {
            throw std::runtime_error("signed_perm_to_matrix: op must have size 3");
        }

        Eigen::Matrix3d O = Eigen::Matrix3d::Zero();
        std::vector<int> used(3, 0);

        for (int i = 0; i < 3; ++i) {
            const int a = op[i];
            const int j = std::abs(a) - 1;

            if (j < 0 || j > 2) {
                throw std::runtime_error("signed_perm_to_matrix: entries must be ±1, ±2, ±3");
            }

            used[j] += 1;
            if (used[j] > 1) {
                throw std::runtime_error("signed_perm_to_matrix: invalid signed permutation");
            }

            O(i, j) = (a > 0 ? 1.0 : -1.0);
        }

        return O;
    }

    // ============================================================
    // Extract ZYZ Euler angles from proper rotation matrix
    // Convention:
    //   R = Rz(alpha) Ry(beta) Rz(gamma)
    // ============================================================
    inline void rotation_matrix_to_zyz(const Eigen::Matrix3d& R,
                                       double& alpha, double& beta, double& gamma)
    {
        const double cb = std::max(-1.0, std::min(1.0, R(2,2)));
        beta = std::acos(cb);

        const double sb = std::sin(beta);

        if (std::abs(sb) > 1e-14) {
            alpha = std::atan2(R(1,2),  R(0,2));
            gamma = std::atan2(R(2,1), -R(2,0));
        } else {
            gamma = 0.0;
            alpha = std::atan2(R(1,0), R(0,0));
        }
    }

    // ============================================================
    // Small Wigner d^l_{m,mp}(beta)
    // ============================================================
    inline double wigner_small_d(const int l, const int m, const int mp, const double beta)
    {
        check_m_range(l, m);
        check_m_range(l, mp);

        const double pref =
            std::sqrt(
                factorial_int(l + m) *
                factorial_int(l - m) *
                factorial_int(l + mp) *
                factorial_int(l - mp)
            );

        double sum = 0.0;

        const int kmin = std::max(0, m - mp);
        const int kmax = std::min(l + m, l - mp);

        const double cb2 = std::cos(beta / 2.0);
        const double sb2 = std::sin(beta / 2.0);

        for (int k = kmin; k <= kmax; ++k) {
            const int a = l + m - k;
            const int b = k;
            const int c = mp - m + k;
            const int d = l - mp - k;

            if (a < 0 || b < 0 || c < 0 || d < 0) {
                continue;
            }

            const double denom =
                factorial_int(a) *
                factorial_int(b) *
                factorial_int(c) *
                factorial_int(d);

            const double sign = (((k - m + mp) % 2) == 0) ? 1.0 : -1.0;

            const int p1 = 2 * l + m - mp - 2 * k;
            const int p2 = mp - m + 2 * k;

            const double term = sign * pref / denom
                              * std::pow(cb2, p1)
                              * std::pow(sb2, p2);

            sum += term;
        }

        return sum;
    }

    // ============================================================
    // Complex Wigner D element from Euler angles
    // D^l_{m,mp}(alpha,beta,gamma) = exp(-i m alpha) d^l_{m,mp}(beta) exp(-i mp gamma)
    // ============================================================
    inline comp wigner_D_complex_element_from_euler(const int l,
                                                    const int m,
                                                    const int mp,
                                                    const double alpha,
                                                    const double beta,
                                                    const double gamma)
    {
        const double d = wigner_small_d(l, m, mp, beta);
        const comp p1 = std::exp(comp(0.0, -static_cast<double>(m ) * alpha));
        const comp p2 = std::exp(comp(0.0, -static_cast<double>(mp) * gamma));
        return p1 * d * p2;
    }

    // ============================================================
    // Real basis -> complex basis change of basis matrix
    // basis order in both cases: m = -l,...,+l
    // ============================================================
    inline Eigen::MatrixXcd real_to_complex_U(const int l)
    {
        const int dim = 2 * l + 1;
        Eigen::MatrixXcd U = Eigen::MatrixXcd::Zero(dim, dim);

        U(midx(l,0), midx(l,0)) = 1.0;

        for (int m = 1; m <= l; ++m) {
            const double s = ((m % 2) == 0) ? 1.0 : -1.0; // (-1)^m
            const double invsqrt2 = 1.0 / std::sqrt(2.0);

            // column for +m real basis vector
            U(midx(l,-m), midx(l,+m)) = invsqrt2;
            U(midx(l,+m), midx(l,+m)) = s * invsqrt2;

            // column for -m real basis vector
            U(midx(l,-m), midx(l,-m)) = comp(0.0, -s * invsqrt2);
            U(midx(l,+m), midx(l,-m)) = comp(0.0,  invsqrt2);
        }

        return U;
    }

    // ============================================================
    // Build full complex D matrix for a proper rotation R in SO(3)
    // ============================================================
    inline Eigen::MatrixXcd wigner_D_complex_matrix_from_rotation(const int l,
                                                                  const Eigen::Matrix3d& R)
    {
        double alpha, beta, gamma;
        rotation_matrix_to_zyz(R, alpha, beta, gamma);

        const int dim = 2 * l + 1;
        Eigen::MatrixXcd D = Eigen::MatrixXcd::Zero(dim, dim);

        for (int m = -l; m <= l; ++m) {
            for (int mp = -l; mp <= l; ++mp) {
                D(midx(l,m), midx(l,mp)) =
                    wigner_D_complex_element_from_euler(l, m, mp, alpha, beta, gamma);
            }
        }

        return D;
    }

    // ============================================================
    // Full real-basis D matrix for a proper rotation R in SO(3)
    // ============================================================
    inline Eigen::MatrixXd D_real_matrix_proper(const int l, const Eigen::Matrix3d& R)
    {
        const Eigen::MatrixXcd Dc = wigner_D_complex_matrix_from_rotation(l, R);
        const Eigen::MatrixXcd U  = real_to_complex_U(l);

        const Eigen::MatrixXcd Dr_c = U.adjoint() * Dc * U;

        const int dim = 2 * l + 1;
        Eigen::MatrixXd Dr(dim, dim);

        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                Dr(i,j) = std::real(Dr_c(i,j));
            }
        }

        return Dr;
    }

    // ============================================================
    // Full real-basis matrix for a general orthogonal transformation
    // represented by signed permutation op.
    //
    // If det(O)=+1, use O directly.
    // If det(O)=-1, write O = (-I)(Rproper), with Rproper = -O in SO(3),
    // and multiply by parity factor (-1)^l.
    // ============================================================
    inline Eigen::MatrixXd D_real_matrix(const int l, const std::vector<int>& op)
    {
        const Eigen::Matrix3d O = signed_perm_to_matrix(op);
        const double detO = O.determinant();

        if (std::abs(detO - 1.0) < 1e-12) {
            return D_real_matrix_proper(l, O);
        }
        else if (std::abs(detO + 1.0) < 1e-12) {
            const Eigen::Matrix3d Rproper = -O;
            const double parity_factor = ((l % 2) == 0) ? 1.0 : -1.0;
            return parity_factor * D_real_matrix_proper(l, Rproper);
        }
        else {
            throw std::runtime_error("D_real_matrix: determinant is not ±1");
        }
    }

    // ============================================================
    // Single element D_real^(l)_{m,mp}
    // ============================================================
    inline double D_real_element(const int l,
                                 const int m,
                                 const int mp,
                                 const std::vector<int>& op)
    {
        check_m_range(l, m);
        check_m_range(l, mp);

        const Eigen::MatrixXd D = D_real_matrix(l, op);
        return D(midx(l,m), midx(l,mp));
    }

    // ============================================================
    // pretty print
    // ============================================================
    inline void print_matrix(const Eigen::MatrixXd& M, const std::string& name = "Matrix")
    {
        std::cout << name << " =\n";
        for (int i = 0; i < M.rows(); ++i) {
            for (int j = 0; j < M.cols(); ++j) {
                std::cout << std::setw(14) << M(i,j) << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    // ============================================================
    // TESTS
    // ============================================================

    inline void test_signed_perm_to_matrix()
    {
        std::cout << "Running test_signed_perm_to_matrix()\n";

        {
            const std::vector<int> op = {1,2,3};
            const Eigen::Matrix3d O = signed_perm_to_matrix(op);
            const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

            std::cout << "identity check: "
                      << (approx_equal_matrix(O, I) ? "PASS" : "FAIL") << "\n";
        }

        {
            const std::vector<int> op = {-1,2,3};
            const Eigen::Matrix3d O = signed_perm_to_matrix(op);

            std::cout << "det({-1,2,3}) = " << O.determinant()
                      << "  expected = -1\n";
        }

        {
            const std::vector<int> op = {2,1,3};
            const Eigen::Matrix3d O = signed_perm_to_matrix(op);

            std::cout << "matrix for {2,1,3}:\n" << O << "\n\n";
        }
    }

    inline void test_D_real_identity(const int l)
    {
        std::cout << "Running test_D_real_identity(l=" << l << ")\n";

        const std::vector<int> op = {1,2,3};
        const Eigen::MatrixXd D = D_real_matrix(l, op);
        const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(2*l+1, 2*l+1);

        std::cout << "identity matrix check: "
                  << (approx_equal_matrix(D, I) ? "PASS" : "FAIL") << "\n\n";

        if (!approx_equal_matrix(D, I)) {
            print_matrix(D, "D_real(identity)");
        }
    }

    inline void test_D_real_reflection_parity(const int l)
    {
        std::cout << "Running test_D_real_reflection_parity(l=" << l << ")\n";

        const std::vector<int> op_reflect_x = {-1,2,3};
        const Eigen::MatrixXd D = D_real_matrix(l, op_reflect_x);

        std::cout << "det = "
                  << signed_perm_to_matrix(op_reflect_x).determinant()
                  << "\n";

        print_matrix(D, "D_real({-1,2,3})");
    }

    inline void test_D_real_swap_xy(const int l)
    {
        std::cout << "Running test_D_real_swap_xy(l=" << l << ")\n";

        const std::vector<int> op = {2,1,3};
        const Eigen::MatrixXd D = D_real_matrix(l, op);

        print_matrix(D, "D_real({2,1,3})");
    }

    inline void test_D_real_orthogonality(const int l, const std::vector<int>& op)
    {
        std::cout << "Running test_D_real_orthogonality(l=" << l << ")\n";

        const Eigen::MatrixXd D = D_real_matrix(l, op);
        const Eigen::MatrixXd check = D.transpose() * D;
        const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(2*l+1, 2*l+1);

        std::cout << "orthogonality check: "
                  << (approx_equal_matrix(check, I, 1e-10) ? "PASS" : "FAIL") << "\n\n";

        if (!approx_equal_matrix(check, I, 1e-10)) {
            print_matrix(check, "D^T D");
        }
    }

    inline void test_D_real_element_vs_matrix(const int l,
                                              const int m,
                                              const int mp,
                                              const std::vector<int>& op)
    {
        std::cout << "Running test_D_real_element_vs_matrix(l=" << l
                  << ", m=" << m << ", mp=" << mp << ")\n";

        const Eigen::MatrixXd D = D_real_matrix(l, op);
        const double a = D(midx(l,m), midx(l,mp));
        const double b = D_real_element(l, m, mp, op);

        std::cout << "matrix entry  = " << a << "\n";
        std::cout << "element func  = " << b << "\n";
        std::cout << "consistency   = "
                  << (approx_equal(a, b, 1e-12) ? "PASS" : "FAIL") << "\n\n";
    }

    inline void run_basic_tests()
    {
        test_signed_perm_to_matrix();

        test_D_real_identity(0);
        test_D_real_identity(1);
        test_D_real_identity(2);

        test_D_real_reflection_parity(1);
        test_D_real_reflection_parity(2);

        test_D_real_swap_xy(1);
        test_D_real_swap_xy(2);

        test_D_real_orthogonality(1, {2,1,3});
        test_D_real_orthogonality(2, {2,1,3});
        test_D_real_orthogonality(2, {-1,2,3});

        test_D_real_element_vs_matrix(2, 1, -1, {2,1,3});
    }

} // namespace real_wigner_d

#endif // REAL_WIGNER_D_HPP