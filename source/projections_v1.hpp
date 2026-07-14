#ifndef PROJECTIONS_V1_H
#define PROJECTIONS_V1_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <complex>
#include <Eigen/Dense>
#include <Eigen/QR>
#include "spherical_functions.h"
#include "functions.h"
#include "F2_functions_v2.h"
#include "K2_functions_v2.h"
#include "QC_functions_v2.h"
#include <omp.h>
//#include "gpu_varsize_batched_inverse.cu"
#include "dig_tools.hpp"
//#include "projections_from_config.hpp"
//#include "eigenvalue_tracker.hpp"

#include <type_traits>
#include "real_wigner_d.hpp"

#include <limits>
#include <cmath>
#include <tuple>
#include <type_traits>
#include <iostream>

typedef std::complex<double> comp; 

using MatC   = Eigen::MatrixXcd;
using VecC   = Eigen::VectorXcd;
using Vec3   = std::array<int, 3>;
using GroupElem = std::array<int, 3>;
using comp   = std::complex<double>;

static const double PI = M_PI;

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

// -----------------------------------------------------------------------------
// Scalar chop
// -----------------------------------------------------------------------------
template <typename T>
T chop_scalar(const T& x, double tol = 1e-16)
{
    if constexpr (std::is_arithmetic_v<T>) {
        return (std::abs(x) < tol) ? T(0) : x;
    }
    else {
        return (std::abs(x) < tol) ? T(0.0, 0.0) : x;
    }
}

// -----------------------------------------------------------------------------
// std::vector<T>
// -----------------------------------------------------------------------------
template <typename T>
std::vector<T> chop(const std::vector<T>& arr, double tol = 1e-16)
{
    std::vector<T> out = arr;
    for (auto& x : out) {
        x = chop_scalar(x, tol);
    }
    return out;
}

// -----------------------------------------------------------------------------
// std::vector<std::vector<T>>
// -----------------------------------------------------------------------------
template <typename T>
std::vector<std::vector<T>> chop(const std::vector<std::vector<T>>& arr, double tol = 1e-16)
{
    std::vector<std::vector<T>> out = arr;
    for (auto& row : out) {
        for (auto& x : row) {
            x = chop_scalar(x, tol);
        }
    }
    return out;
}

// -----------------------------------------------------------------------------
// Eigen matrices / vectors
// Works for MatrixXcd, VectorXcd, MatrixXd, etc.
// -----------------------------------------------------------------------------
template <typename Derived>
auto chop(const Eigen::MatrixBase<Derived>& arr, double tol = 1e-16)
    -> typename Derived::PlainObject
{
    using Plain = typename Derived::PlainObject;
    Plain out = arr.derived();

    for (int i = 0; i < out.rows(); ++i) {
        for (int j = 0; j < out.cols(); ++j) {
            out(i,j) = chop_scalar(out(i,j), tol);
        }
    }

    return out;
}

Eigen::MatrixXcd blockDiag(const std::vector<Eigen::MatrixXcd>& blocks) {
    int total = 0;
    for (const auto& b : blocks) total += b.rows();

    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(total, total);

    int offset = 0;
    for (const auto& b : blocks) {
        int sz = b.rows();
        result.block(offset, offset, sz, sz) = b;
        offset += sz;
    }
    return result;
}



std::vector<std::vector<int>> rotations_list()
{
    return {
        { 1,  2,  3}, { 2,  3,  1}, { 3,  1,  2},
        { 1,  3, -2}, { 2, -1,  3}, { 3,  2, -1},

        { 1, -2, -3}, { 2, -3, -1}, { 3, -1, -2},
        { 1, -3,  2}, { 2,  1, -3}, { 3, -2,  1},

        {-1,  2, -3}, {-2,  3, -1}, {-3,  1, -2},
        {-1,  3,  2}, {-2, -1, -3}, {-3,  2,  1},

        {-1, -2,  3}, {-2, -3,  1}, {-3, -1,  2},
        {-1, -3, -2}, {-2,  1,  3}, {-3, -2, -1}
    };
}

bool is_rotation(const std::vector<int>& R)
{
    static const std::vector<std::vector<int>> rots = {
        { 1,  2,  3}, { 2,  3,  1}, { 3,  1,  2},
        { 1,  3, -2}, { 2, -1,  3}, { 3,  2, -1},

        { 1, -2, -3}, { 2, -3, -1}, { 3, -1, -2},
        { 1, -3,  2}, { 2,  1, -3}, { 3, -2,  1},

        {-1,  2, -3}, {-2,  3, -1}, {-3,  1, -2},
        {-1,  3,  2}, {-2, -1, -3}, {-3,  2,  1},

        {-1, -2,  3}, {-2, -3,  1}, {-3, -1,  2},
        {-1, -3, -2}, {-2,  1,  3}, {-3, -2, -1}
    };

    for (const auto& rot : rots) {
        if (rot == R) {
            return true;
        }
    }
    return false;
}



std::vector<int> cubic_transf(const std::vector<int>& vec,
                             const std::vector<int>& p)
{
    if (vec.size() != 3 || p.size() != 3) {
        throw std::runtime_error("cubic_transf: vec and p must both have size 3");
    }

    std::vector<int> out(3);

    for (int j = 0; j < 3; ++j) {
        int idx = std::abs(p[j]) - 1;
        int sgn = (p[j] > 0) ? 1 : -1;
        out[j] = sgn * vec[idx];
    }

    return out;
}

// -----------------------------------------------------------------------------
// Generate all permutations of a 3-vector like [1,2,3], [1,2,-3], etc.
// -----------------------------------------------------------------------------
std::vector<std::vector<int>> all_perms_of_3(std::vector<int> v)
{
    std::vector<std::vector<int>> out;
    std::sort(v.begin(), v.end());

    do {
        out.push_back(v);
    } while (std::next_permutation(v.begin(), v.end()));

    return out;
}

// -----------------------------------------------------------------------------
// Python:
//
// def Oh_list():
//   Oh_list = list(perms([1,2,3]))
//   Oh_list += list(perms([1,2,-3]))
//   Oh_list += list(perms([1,-2,3]))
//   Oh_list += list(perms([-1,2,3]))
//   Oh_list += list(perms([1,-2,-3]))
//   Oh_list += list(perms([-1,2,-3]))
//   Oh_list += list(perms([-1,-2,3]))
//   Oh_list += list(perms([-1,-2,-3]))
//   Oh_list = [ list(R) for R in Oh_list ]
//   return Oh_list
// -----------------------------------------------------------------------------
std::vector<std::vector<int>> Oh_list()
{
    std::vector<std::vector<int>> out;

    auto append_perms = [&](const std::vector<int>& v) {
        auto p = all_perms_of_3(v);
        out.insert(out.end(), p.begin(), p.end());
    };

    append_perms({ 1,  2,  3});
    append_perms({ 1,  2, -3});
    append_perms({ 1, -2,  3});
    append_perms({-1,  2,  3});
    append_perms({ 1, -2, -3});
    append_perms({-1,  2, -3});
    append_perms({-1, -2,  3});
    append_perms({-1, -2, -3});

    return out;
}

// -----------------------------------------------------------------------------
// Little group for given shell type
// shell is total_nP = std::vector<comp>
// nPx = int(total_nP[0].real()), etc.
// -----------------------------------------------------------------------------
std::vector<std::vector<int>> little_group(const std::vector<comp>& total_nP)
{
    if (total_nP.size() != 3) {
        throw std::runtime_error("little_group: total_nP must have size 3");
    }

    const int x = static_cast<int>(total_nP[0].real());
    const int y = static_cast<int>(total_nP[1].real());
    const int z = static_cast<int>(total_nP[2].real());
    
    // 000
    if (x == 0 && y == 0 && z == 0) {
        return Oh_list();
    }
    // 00a
    else if (x == 0 && y == 0) {
        return {
            { 1,  2,  3}, {-1,  2,  3}, { 1, -2,  3}, {-1, -2,  3},
            { 2,  1,  3}, {-2,  1,  3}, { 2, -1,  3}, {-2, -1,  3}
        };
    }
    // aa0
    else if (x == y && z == 0) {
        return {
            {1, 2,  3}, {1, 2, -3},
            {2, 1,  3}, {2, 1, -3}
        };
    }
    // aaa
    else if (x == y && y == z) {
        return {
            {1, 2, 3}, {1, 3, 2}, {2, 1, 3},
            {2, 3, 1}, {3, 1, 2}, {3, 2, 1}
        };
    }
    // ab0
    else if (x != 0 && y != 0 && x != y && z == 0) {
        return {
            {1, 2,  3},
            {1, 2, -3}
        };
    }
    // aab
    else if (x == y && y != z) {
        return {
            {1, 2, 3},
            {2, 1, 3}
        };
    }
    // abc
    else if (x != 0 && y != 0 && z != 0 &&
             x != y && y != z && z != x) {
        return {
            {1, 2, 3}
        };
    }
    else {
        throw std::runtime_error("little_group: invalid shell input");
    }
}

Eigen::Matrix3d Dmat11(const std::vector<int>& R)
{
    auto sign = [](int x) -> int {
        return (x > 0) - (x < 0);
    };

    auto eye3 = []() -> Eigen::Matrix3d {
        return Eigen::Matrix3d::Identity();
    };

    auto valid_Dmat_input = [](const std::vector<int>& v) -> bool {
    if (v.size() != 3) return false;

    std::array<bool,4> seen = {false, false, false, false};

    for (int x : v) {
            int ax = std::abs(x);
            if (ax < 1 || ax > 3) return false;
            if (seen[ax]) return false;
            seen[ax] = true;
        }
        return true;
    };

    if (R.size() != 3) {
        throw std::runtime_error("Error in Dmat11: R must have size 3");
    }

    /*
    if (!is_rotation(R)) {
        throw std::runtime_error("Error in Dmat11: invalid input");
    }
    */

    if (!valid_Dmat_input(R)) {
        throw std::runtime_error("Error in Dmat11: invalid input");
    }
    

    // Trivial transformation
    if (R == std::vector<int>{1,2,3} || R == std::vector<int>{-1,-2,-3}) {
        return sign(R[0]) * eye3();
    }

    // Single permutation
    else if (R == std::vector<int>{2,1,3} || R == std::vector<int>{-2,-1,-3}) {
        Eigen::Matrix3d U = Eigen::Matrix3d::Identity();
        U(0,0) = 0.0; U(2,2) = 0.0;
        U(0,2) = 1.0; U(2,0) = 1.0;
        return sign(R[0]) * U;
    }
    else if (R == std::vector<int>{1,3,2} || R == std::vector<int>{-1,-3,-2}) {
        Eigen::Matrix3d U = Eigen::Matrix3d::Identity();
        U(0,0) = 0.0; U(1,1) = 0.0;
        U(0,1) = 1.0; U(1,0) = 1.0;
        return sign(R[0]) * U;
    }
    else if (R == std::vector<int>{3,2,1} || R == std::vector<int>{-3,-2,-1}) {
        Eigen::Matrix3d U = Eigen::Matrix3d::Identity();
        U(1,1) = 0.0; U(2,2) = 0.0;
        U(1,2) = 1.0; U(2,1) = 1.0;
        return sign(R[0]) * U;
    }

    // Cyclic permutation
    else if (R == std::vector<int>{2,3,1} || R == std::vector<int>{-2,-3,-1}) {
        return sign(R[0]) * chop(Dmat11({1,3,2}) * Dmat11({2,1,3}));
    }
    else if (R == std::vector<int>{3,1,2} || R == std::vector<int>{-3,-1,-2}) {
        return sign(R[0]) * chop(Dmat11({3,2,1}) * Dmat11({2,1,3}));
    }

    // Single negation
    else if (R == std::vector<int>{1,2,-3} || R == std::vector<int>{-1,-2,3}) {
        Eigen::Matrix3d U = Eigen::Matrix3d::Zero();
        U(0,0) =  1.0;
        U(1,1) = -1.0;
        U(2,2) =  1.0;
        return sign(R[0]) * chop(U);
    }
    else if (R[0]*R[1] > 0 && R[0]*R[2] < 0) {
        return chop(Dmat11({1,2,-3}) * Dmat11({R[0], R[1], -R[2]}));
    }
    else if (R[0]*R[2] > 0 && R[0]*R[1] < 0) {
        return chop(Dmat11({1,3,2}) * Dmat11({R[0], R[2], R[1]}));
    }
    else if (R[1]*R[2] > 0 && R[0]*R[1] < 0) {
        return chop(Dmat11({3,2,1}) * Dmat11({R[2], R[1], R[0]}));
    }

    throw std::runtime_error("Error in Dmat11: This should never trigger");
}

Eigen::Matrix<double,5,5> Dmat22(const std::vector<int>& R)
{
    using Mat5 = Eigen::Matrix<double,5,5>;
    using Mat6 = Eigen::Matrix<double,6,6>;

    auto sign = [](int x) -> int {
        return (x > 0) - (x < 0);
    };
    auto valid_Dmat_input = [](const std::vector<int>& v) -> bool {
    if (v.size() != 3) return false;

    std::array<bool,4> seen = {false, false, false, false};

    for (int x : v) {
            int ax = std::abs(x);
            if (ax < 1 || ax > 3) return false;
            if (seen[ax]) return false;
            seen[ax] = true;
        }
        return true;
    };

    if (R.size() != 3) {
        throw std::runtime_error("Error in Dmat22: R must have size 3");
    }

    /*
    if (!is_rotation(R)) {
        throw std::runtime_error("Error in Dmat11: invalid input");
    }
    */

    if (!valid_Dmat_input(R)) {
        throw std::runtime_error("Error in Dmat22: invalid input");
    }

    // Trivial transformation
    if (R == std::vector<int>{1,2,3} || R == std::vector<int>{-1,-2,-3}) {
        return Mat5::Identity();
    }

    // Single permutation
    else if (R == std::vector<int>{2,1,3} || R == std::vector<int>{-2,-1,-3}) {
        Mat6 U = Mat6::Identity();
        U(2,2) = 0.0; U(2,4) = 1.0;
        U(4,2) = 1.0; U(4,4) = 0.0;
        U(5,5) = -1.0;
        return U.block<5,5>(1,1);
    }

    else if (R == std::vector<int>{1,3,2} || R == std::vector<int>{-1,-3,-2}) {
        Mat6 U = Mat6::Zero();
        U(0,0) = 1.0;
        U(1,4) = 1.0; U(2,2) = 1.0; U(4,1) = 1.0;
        U(3,3) = -0.5;               U(3,5) = -std::sqrt(3.0)/2.0;
        U(5,3) = -std::sqrt(3.0)/2.0; U(5,5) =  0.5;
        return U.block<5,5>(1,1);
    }

    else if (R == std::vector<int>{3,2,1} || R == std::vector<int>{-3,-2,-1}) {
        Mat6 U = Mat6::Zero();
        U(0,0) = 1.0;
        U(1,2) = 1.0; U(2,1) = 1.0; U(4,4) = 1.0;
        U(3,3) = -0.5;              U(3,5) =  std::sqrt(3.0)/2.0;
        U(5,3) =  std::sqrt(3.0)/2.0; U(5,5) = 0.5;
        return U.block<5,5>(1,1);
    }

    // Cyclic permutation
    else if (R == std::vector<int>{2,3,1} || R == std::vector<int>{-2,-3,-1}) {
        return chop(Dmat22({1,3,2}) * Dmat22({2,1,3}));
    }

    else if (R == std::vector<int>{3,1,2} || R == std::vector<int>{-3,-1,-2}) {
        return chop(Dmat22({3,2,1}) * Dmat22({2,1,3}));
    }

    // Single negation
    else if (R == std::vector<int>{1,2,-3} || R == std::vector<int>{-1,-2,3}) {
        Mat5 U = Mat5::Zero();
        U(0,0) =  1.0;
        U(1,1) = -1.0;
        U(2,2) =  1.0;
        U(3,3) = -1.0;
        U(4,4) =  1.0;
        return chop(U);
    }

    else if (R[0]*R[1] > 0 && R[0]*R[2] < 0) {
        return chop(Dmat22({1,2,-3}) * Dmat22({R[0], R[1], -R[2]}));
    }

    else if (R[0]*R[2] > 0 && R[0]*R[1] < 0) {
        return chop(Dmat22({1,3,2}) * Dmat22({R[0], R[2], R[1]}));
    }

    else if (R[1]*R[2] > 0 && R[0]*R[1] < 0) {
        return chop(Dmat22({3,2,1}) * Dmat22({R[2], R[1], R[0]}));
    }

    throw std::runtime_error("Error in Dmat22: This should never trigger");
}

Eigen::Matrix4d Dmat(const std::vector<int>& R)
{
    Eigen::Matrix4d D = Eigen::Matrix4d::Zero();
    D(0,0) = 1.0;
    D.block<3,3>(1,1) = Dmat11(R);
    return D;
}



int irrep_dim(const std::string& I)
{
    if (I == "A1g" || I == "A1" || I == "A2g" || I == "A2" ||
        I == "A1u" || I == "A2u" || I == "B1" || I == "B2")
    {
        return 1;
    }
    else if (I == "Eg" || I == "E" || I == "Eu" || I == "E2")
    {
        return 2;
    }
    else if (I == "T1g" || I == "T1" || I == "T2g" || I == "T2" ||
             I == "T1u" || I == "T2u")
    {
        return 3;
    }

    throw std::runtime_error("Error: invalid irrep in irrep_dim -- \"" + I + "\"");
}

std::string conj_class(const std::vector<int>& p)
{
    if (p.size() != 3) {
        throw std::runtime_error("conj_class: p must have size 3");
    }

    std::vector<int> p_abs = {std::abs(p[0]), std::abs(p[1]), std::abs(p[2])};

    int N_negs = 0;
    int N_correct = 0;

    for (int i = 0; i < 3; ++i) {
        if (p[i] < 0) {
            ++N_negs;
        }
        if (p_abs[i] == i + 1) {
            ++N_correct;
        }
    }

    if (N_correct == 3) {
        if (N_negs == 0) {
            return "E";
        }
        else if (N_negs == 2) {
            return "C4^2";
        }
        else if (N_negs == 3) {
            return "i";
        }
        else if (N_negs == 1) {
            return "sigma_h";
        }
    }
    else if (N_correct == 0) {
        if ((N_negs % 2) == 0) {
            return "C3";
        }
        else {
            return "S6";
        }
    }
    else if (N_correct == 1) {
        int i_correct = -1;
        for (int i = 0; i < 3; ++i) {
            if (p_abs[i] == i + 1) {
                i_correct = i;
                break;
            }
        }

        if (i_correct == -1) {
            throw std::runtime_error("conj_class: internal error finding correct index");
        }

        if ((N_negs % 2) == 1) {
            if (p[i_correct] < 0) {
                return "C2";
            }
            else {
                return "C4";
            }
        }
        else {
            if (p[i_correct] > 0) {
                return "sigma_d";
            }
            else {
                return "S4";
            }
        }
    }

    throw std::runtime_error("Error in conj_class: should never reach here");
}

inline int chi(const std::vector<int>& p,
               const std::string& I,
               const std::vector<comp>& Pvec)
{
    const bool debug = false;

    if (Pvec.size() != 3) {
        throw std::runtime_error("chi: Pvec must have size 3");
    }

    const std::string cc = conj_class(p);

    if (debug) {
        std::cout << "conj_class__________________\n";
        std::cout << "cc: " << cc << "\n";
        std::cout << "____________________________\n";
    }

    const int Px = static_cast<int>(Pvec[0].real());
    const int Py = static_cast<int>(Pvec[1].real());
    const int Pz = static_cast<int>(Pvec[2].real());

    // ------------------------------------------------------------
    // 000 (Oh)
    // ------------------------------------------------------------
    if (Px == 0 && Py == 0 && Pz == 0) {

        if (I == "A1g" || I == "A1") {
            return 1;
        }

        else if (I == "A2g" || I == "A2") {
            if (cc == "C2" || cc == "C4" || cc == "sigma_d" || cc == "S4") {
                return -1;
            } else {
                return 1;
            }
        }

        else if (I == "Eg" || I == "E") {
            if (cc == "E" || cc == "C4^2" || cc == "i" || cc == "sigma_h") {
                return 2;
            } else if (cc == "C3" || cc == "S6") {
                return -1;
            } else {
                return 0;
            }
        }

        else if (I == "T1g" || I == "T1") {
            if (cc == "E" || cc == "i") {
                return 3;
            } else if (cc == "C3" || cc == "S6") {
                return 0;
            } else if (cc == "C4" || cc == "S4") {
                return 1;
            } else {
                return -1;
            }
        }

        else if (I == "T2g" || I == "T2") {
            if (cc == "E" || cc == "i") {
                return 3;
            } else if (cc == "C3" || cc == "S6") {
                return 0;
            } else if (cc == "C2" || cc == "sigma_d") {
                return 1;
            } else {
                return -1;
            }
        }

        else if (I == "A1u") {
            if (cc == "E" || cc == "C3" || cc == "C4^2" || cc == "C4" || cc == "C2") {
                return 1;
            } else {
                return -1;
            }
        }

        else if (I == "A2u") {
            if (cc == "E" || cc == "C3" || cc == "C4^2" || cc == "S4" || cc == "sigma_d") {
                return 1;
            } else {
                return -1;
            }
        }

        else if (I == "Eu") {
            if (cc == "E" || cc == "C4^2") {
                return 2;
            } else if (cc == "i" || cc == "sigma_h") {
                return -2;
            } else if (cc == "S6") {
                return 1;
            } else if (cc == "C3") {
                return -1;
            } else {
                return 0;
            }
        }

        else if (I == "T1u") {
            if (cc == "E") {
                return 3;
            } else if (cc == "i") {
                return -3;
            } else if (cc == "C4" || cc == "sigma_h" || cc == "sigma_d") {
                return 1;
            } else if (cc == "C2" || cc == "C4^2" || cc == "S4") {
                return -1;
            } else {
                return 0;
            }
        }

        else if (I == "T2u") {
            if (cc == "E") {
                return 3;
            } else if (cc == "i") {
                return -3;
            } else if (cc == "C2" || cc == "S4" || cc == "sigma_h") {
                return 1;
            } else if (cc == "C4" || cc == "C4^2" || cc == "sigma_d") {
                return -1;
            } else {
                return 0;
            }
        }

        else {
            throw std::runtime_error("chi: invalid irrep for Oh");
        }
    }

    // ------------------------------------------------------------
    // 00a (C4v)
    // ------------------------------------------------------------
    else if (Px == 0 && Py == 0) {

        if (I == "A1") {
            return 1;
        }

        else if (I == "A2") {
            if (cc == "sigma_h" || cc == "sigma_d") {
                return -1;
            } else if (cc == "E" || cc == "C4" || cc == "C4^2") {
                return 1;
            } else {
                throw std::runtime_error("chi: " + cc + " not in LG(00a)");
            }
        }

        else if (I == "B1") {
            if (cc == "C4" || cc == "sigma_d") {
                return -1;
            } else if (cc == "E" || cc == "C4^2" || cc == "sigma_h") {
                return 1;
            } else {
                throw std::runtime_error("chi: " + cc + " not in LG(00a)");
            }
        }

        else if (I == "B2") {
            if (cc == "C4" || cc == "sigma_h") {
                return -1;
            } else if (cc == "E" || cc == "C4^2" || cc == "sigma_d") {
                return 1;
            } else {
                throw std::runtime_error("chi: " + cc + " not in LG(00a)");
            }
        }

        else if (I == "E" || I == "E2") {
            if (cc == "E") {
                return 2;
            } else if (cc == "C4^2") {
                return -2;
            } else if (cc == "C4" || cc == "sigma_h" || cc == "sigma_d") {
                return 0;
            } else {
                throw std::runtime_error("chi: " + cc + " not in LG(00a)");
            }
        }

        else {
            throw std::runtime_error("chi: invalid irrep for C4v");
        }
    }

    // ------------------------------------------------------------
    // aa0 (C2v)
    // ------------------------------------------------------------
    else if (Px == Py && Py != Pz && Pz == 0) {

        if (I == "A1") {
            return 1;
        }

        else if (I == "A2") {
            if (cc == "sigma_h" || cc == "sigma_d") {
                return -1;
            } else if (cc == "E" || cc == "C2") {
                return 1;
            } else {
                throw std::runtime_error("chi: " + cc + " not in LG(aa0)");
            }
        }

        else if (I == "B1") {
            if (cc == "C2" || cc == "sigma_h") {
                return -1;
            } else if (cc == "E" || cc == "sigma_d") {
                return 1;
            } else {
                throw std::runtime_error("chi: " + cc + " not in LG(aa0)");
            }
        }

        else if (I == "B2") {
            if (cc == "C2" || cc == "sigma_d") {
                return -1;
            } else if (cc == "E" || cc == "sigma_h") {
                return 1;
            } else {
                throw std::runtime_error("chi: " + cc + " not in LG(aa0)");
            }
        }

        else {
            throw std::runtime_error("chi: invalid irrep for C2v");
        }
    }

    // ------------------------------------------------------------
    // aaa (C3v)
    // ------------------------------------------------------------
    else if (Px == Py && Py == Pz) {

        if (I == "A1") {
            return 1;
        }

        else if (I == "A2") {
            if (cc == "sigma_d") {
                return -1;
            } else if (cc == "E" || cc == "C3") {
                return 1;
            } else {
                throw std::runtime_error("chi: " + cc + " not in LG(aaa)");
            }
        }

        else if (I == "E" || I == "E2") {
            if (cc == "E") {
                return 2;
            } else if (cc == "C3") {
                return -1;
            } else if (cc == "sigma_d") {
                return 0;
            } else {
                throw std::runtime_error("chi: " + cc + " not in LG(aaa)");
            }
        }

        else {
            throw std::runtime_error("chi: invalid irrep for C3v");
        }
    }

    // ------------------------------------------------------------
    // ab0, aab (C2)
    // ------------------------------------------------------------
    else if ((Px != 0 && Px != Py && Py != 0 && Pz == 0) ||
             (Px == Py && Py != Pz)) {

        if (I == "A1" || I == "A") {
            return 1;
        }

        else if (I == "A2" || I == "B") {
            if (cc == "sigma_h" || cc == "sigma_d") {
                return -1;
            } else if (cc == "E") {
                return 1;
            } else {
                throw std::runtime_error("chi: " + cc + " not in LG(ab0/aab)");
            }
        }

        else {
            throw std::runtime_error("chi: invalid irrep for C2");
        }
    }

    // ------------------------------------------------------------
    // abc (C1 trivial) or invalid
    // ------------------------------------------------------------
    else {
        throw std::runtime_error("chi: invalid Pvec input");
    }
}

inline bool is_in_rotations_list(const std::vector<int>& R)
{
    auto rots = rotations_list();
    return std::find(rots.begin(), rots.end(), R) != rots.end();
}

std::vector<int> get_orbit(const std::vector<std::vector<comp>>& plm_vec,
                           const std::vector<comp>& total_nP)
{
    if (plm_vec.size() < 3) {
        throw std::runtime_error("get_orbit: plm_vec must have at least 3 entries");
    }
    if (plm_vec[0].empty() || plm_vec[1].empty() || plm_vec[2].empty()) {
        throw std::runtime_error("get_orbit: plm_vec[0], plm_vec[1], plm_vec[2] must be nonempty");
    }
    if (total_nP.size() < 3) {
        throw std::runtime_error("get_orbit: total_nP must have size at least 3");
    }

    int x = static_cast<int>(std::real(total_nP[0]));
    int y = static_cast<int>(std::real(total_nP[1]));
    int z = static_cast<int>(std::real(total_nP[2]));

    int a = static_cast<int>(std::real(plm_vec[0][0]));
    int b = static_cast<int>(std::real(plm_vec[1][0]));
    int c = static_cast<int>(std::real(plm_vec[2][0]));
    // if your code stores the 3rd component in plm_vec[3][0], use:
    // int c = static_cast<int>(std::real(plm_vec[3][0]));

    // P = 000
    if (x == 0 && y == 0 && z == 0) {
        std::vector<int> tmp = {std::abs(a), std::abs(b), std::abs(c)};
        std::sort(tmp.begin(), tmp.end());
        a = tmp[0];
        b = tmp[1];
        c = tmp[2];

        if ((a == 0 && b > 0) || (a < b && b == c)) {
            return {b, c, a};
        } else {
            return {a, b, c};
        }
    }

    // 00z
    else if (x == 0 && y == 0 && z > 0) {
        if (a == 0 || b == 0) {
            b = std::max(std::abs(a), std::abs(b));
            return {b, 0, c};
        } else {
            std::vector<int> tmp = {std::abs(a), std::abs(b)};
            std::sort(tmp.begin(), tmp.end());
            a = tmp[0];
            b = tmp[1];
            return {a, b, c};
        }
    }

    // xx0
    else if (x == y && x > 0 && z == 0) {
        std::vector<int> tmp = {a, b};
        std::sort(tmp.begin(), tmp.end());
        a = tmp[0];
        b = tmp[1];
        return {a, b, std::abs(c)};
    }

    // xxx
    else if (x == y && y == z && x > 0) {
        std::vector<int> tmp = {a, b, c};

        std::sort(tmp.begin(), tmp.end());
        std::sort(tmp.begin(), tmp.end(),
                  [](int u, int v) { return std::abs(u) < std::abs(v); });

        a = tmp[0];
        b = tmp[1];
        c = tmp[2];

        if (a == 0 && b != 0) {
            std::vector<int> bc = {b, c};
            std::sort(bc.begin(), bc.end());
            b = bc[0];
            c = bc[1];
            return {b, c, 0};
        }
        else if (a < b && b == c) {
            return {b, b, a};
        }
        else {
            return {a, b, c};
        }
    }

    // xy0
    else if (z == 0 && x > 0 && x < y) {
        return {a, b, std::abs(c)};
    }

    // xxz
    else if (x > 0 && x == y && z > 0 && x != z) {
        std::vector<int> tmp = {a, b};
        std::sort(tmp.begin(), tmp.end());
        a = tmp[0];
        b = tmp[1];
        return {a, b, c};
    }

    throw std::runtime_error("get_orbit: momentum class not implemented");
}

void unique_sort_orbits(std::vector<std::vector<comp>>& orbits)
{
    if (orbits.size() != 3) {
        throw std::runtime_error("orbits must have size 3");
    }
    if (orbits[0].size() != orbits[1].size() || orbits[1].size() != orbits[2].size()) {
        throw std::runtime_error("orbits[0], orbits[1], orbits[2] must have same size");
    }

    using Triplet = std::tuple<int,int,int>;
    std::vector<Triplet> pts;
    pts.reserve(orbits[0].size());

    for (size_t i = 0; i < orbits[0].size(); ++i) {
        int nx = static_cast<int>(orbits[0][i].real());
        int ny = static_cast<int>(orbits[1][i].real());
        int nz = static_cast<int>(orbits[2][i].real());
        pts.emplace_back(nx, ny, nz);
    }

    std::sort(pts.begin(), pts.end());

    pts.erase(std::unique(pts.begin(), pts.end()), pts.end());

    orbits[0].clear();
    orbits[1].clear();
    orbits[2].clear();

    orbits[0].reserve(pts.size());
    orbits[1].reserve(pts.size());
    orbits[2].reserve(pts.size());

    for (const auto& [nx, ny, nz] : pts) {
        orbits[0].push_back(comp(nx, 0.0));
        orbits[1].push_back(comp(ny, 0.0));
        orbits[2].push_back(comp(nz, 0.0));
    }
}

void orbit_maker(   std::vector<std::vector<comp>> &orbit, 
                    std::vector<std::vector<int>> &nplm_config, 
                    std::vector<comp> &total_nP,
                    bool sort_it  )
{
    char debug = 'n'; 
    for(int i=0; i<nplm_config[0].size(); ++i)
    {
        std::vector<std::vector<comp>> temp_nplm(3,std::vector<comp>(1)); 
        int npx = nplm_config[0][i]; 
        int npy = nplm_config[1][i]; 
        int npz = nplm_config[2][i]; 
        temp_nplm[0][0] = npx; 
        temp_nplm[1][0] = npy; 
        temp_nplm[2][0] = npz; 
        int ell = nplm_config[3][i]; 
        int proj_m = nplm_config[4][i]; 

        std::vector<int> temp_orbit = get_orbit(temp_nplm, total_nP); 

        orbit[0].push_back(temp_orbit[0]);
        orbit[1].push_back(temp_orbit[1]);
        orbit[2].push_back(temp_orbit[2]);

        if(debug=='y')
        {
            std::cout<< "np mom = "<<std::setw(5); 
            std::cout<< npx << "," << npy << "," << npz << '\t';
            std::cout<< "orbit = "<<std::setw(5); 
            std::cout<< temp_orbit[0] << "," << temp_orbit[1] << "," << temp_orbit[2] << '\t';  
            std::cout<< "ell = " << ell << " proj_m = " << proj_m << std::endl;
        }
        
    }

    if(sort_it)
    {
        unique_sort_orbits(orbit);
    }



}

void wigner_d_tests()
{
    using namespace real_wigner_d;

    std::vector<int> R = {2,1,3};

    double val = D_real_element(2, 1, -1, R);
    std::cout << "D^(2)_{1,-1} = " << val << "\n\n";

    Eigen::MatrixXd D = D_real_matrix(2, R);
    print_matrix(D, "D_real");

    run_basic_tests();

    
}

void test_projector_idempotency(const Eigen::MatrixXcd& P, double tol = 1e-12)
{
    Eigen::MatrixXcd diff = P * P - P;
    double err = diff.norm();

    std::cout << "||P^2 - P|| = " << err << "\n";

    if (err < tol) {
        std::cout << "PASS: P is a projector within tolerance.\n";
    } else {
        std::cout << "FAIL: P is not a projector within tolerance.\n";
    }
}

void P_irrep_projection_single_flavor(  Eigen::MatrixXcd &P_I,
                                        std::vector<std::vector<comp>> &plm_config,
                                        std::vector<std::vector<int>> &np_config, 
                                        std::vector<std::vector<comp>> &klm_config,
                                        std::vector<std::vector<int>> &nk_config,
                                        std::string &irrep, 
                                        std::vector<comp> &total_P, 
                                        std::vector<comp> &nnP_config,
                                        bool sort_orbit_flag,
                                        int parity 
                                    )
{
    char debug = 'n';
    std::string I = irrep; 
    auto d_I = irrep_dim(I); 
    auto LG = little_group(nnP_config); 
    if(debug=='y')
    {
        for (const auto& row : LG) {
        std::cout << "{ ";
        for (const auto& x : row) {
            std::cout << x << " ";
        }
        std::cout << "}\n";
        
        }
        std::cout<<"LG size = "<<LG.size()<<std::endl; 
    }

    std::vector<std::vector<comp>> orbit_np(3);
    bool sort_flag1 = sort_orbit_flag;  
    orbit_maker(orbit_np, np_config, nnP_config, sort_flag1);

    std::vector<std::vector<comp>> orbit_nk(3);
    bool sort_flag2 = sort_orbit_flag;  
    orbit_maker(orbit_nk, nk_config, nnP_config, sort_flag2);

    for(int i=0; i<plm_config[0].size(); ++i)
    {
        std::vector<comp> pvec(3); 
        std::vector<int> npvec(3); 
        pvec[0] = plm_config[0][i]; 
        pvec[1] = plm_config[1][i]; 
        pvec[2] = plm_config[2][i];
        
        npvec[0] = np_config[0][i]; 
        npvec[1] = np_config[1][i]; 
        npvec[2] = np_config[2][i]; 

        int ell_p = static_cast<int> (std::real(plm_config[3][i])); 
        int proj_m_p = static_cast<int> (std::real(plm_config[4][i])); 

        std::vector<comp> orbit_p(3); 
        orbit_p[0] = orbit_np[0][i]; 
        orbit_p[1] = orbit_np[1][i]; 
        orbit_p[2] = orbit_np[2][i]; 

        for(int j=0; j<klm_config[0].size(); ++j)
        {
            std::vector<comp> kvec(3); 
            std::vector<int> nkvec(3); 
            kvec[0] = klm_config[0][j]; 
            kvec[1] = klm_config[1][j]; 
            kvec[2] = klm_config[2][j]; 
        
            nkvec[0] = nk_config[0][j]; 
            nkvec[1] = nk_config[1][j]; 
            nkvec[2] = nk_config[2][j]; 



            int ell_k = static_cast<int> (std::real(klm_config[3][j])); 
            int proj_m_k = static_cast<int> (std::real(klm_config[4][j])); 


            std::vector<comp> orbit_k(3); 
            orbit_k[0] = orbit_nk[0][j]; 
            orbit_k[1] = orbit_nk[1][j]; 
            orbit_k[2] = orbit_nk[2][j]; 

            comp P_I_element = {0.0, 0.0}; 

            double LG_bar = LG.size(); 
            
            for(int k=0; k<LG.size(); ++k)
            {
                std::vector<int> R(3); 
                R[0] = LG[k][0];
                R[1] = LG[k][1];
                R[2] = LG[k][2]; 

                auto Rp = cubic_transf(npvec, R); 

                if(debug=='y')
                {   
                    std::cout << "Projection Tests:___________________________________ " << std::endl; 
                    std::cout << "i:" << i << '\t' << "j:" << j << '\t' << "k:" << k << std::endl;
                    std::cout << "pvec: [" << pvec[0] << ","
                                           << pvec[1] << ","
                                           << pvec[2] << "]" << std::endl; 
                    std::cout << "npvec: [" << npvec[0] << ","
                                            << npvec[1] << ","
                                            << npvec[2] << "]" << std::endl; 
                    std::cout << "kvec: [" << kvec[0] << ","
                                           << kvec[1] << ","
                                           << kvec[2] << "]" << std::endl; 
                    std::cout << "nkvec: [" << nkvec[0] << ","
                                            << nkvec[1] << ","
                                            << nkvec[2] << "]" << std::endl; 
                    std::cout << "orbit_p: [ " << orbit_p[0] << ","
                                               << orbit_p[1] << ","
                                               << orbit_p[2] << "]"  << std::endl; 
                    std::cout << "orbit_k: [ " << orbit_k[0] << ","
                                               << orbit_k[1] << ","
                                               << orbit_k[2] << "]"  << std::endl; 
                    std::cout << "ell_p: " << ell_p << '\t' << "ell_k: " << ell_k << std::endl; 
                    std::cout << "proj_m_p: " << proj_m_p << '\t' << "proj_m_k: " << proj_m_k << std::endl; 
                    std::cout << "R: [" << R[0] << "," 
                                       << R[1] << "," 
                                       << R[2] << "]" << std::endl; 
                    std::cout << "Rp: [" << Rp[0] << "," 
                                        << Rp[1] << "," 
                                        << Rp[2] << "]" << std::endl; 
                                        
                }

                if( Rp == nkvec )
                {
                    if(debug=='y')
                    {
                        std::cout << "Rp=nkvec : True" << std::endl; 
                    }
                    if( ell_p == ell_k )
                    {
                        if(debug=='y')
                        {
                            std::cout << "ell_p = ell_k : True" << std::endl; 
                        }
                        if( orbit_p == orbit_k ) 
                        {
                            if(debug=='y')
                            {
                                std::cout << "orbit_p = orbit_k : True" << std::endl; 
                            }
                            int par = (parity == -1 && !is_in_rotations_list(R)) ? -1 : 1;
                            int chi_val = chi(R, I, nnP_config); 
                            double wigner_d_val = real_wigner_d::D_real_element(ell_k, proj_m_p, proj_m_k, R); 
                            P_I_element += par * chi_val * wigner_d_val; 

                            if(debug=='y')
                            {
                                std::cout << "par: " << par << '\t' 
                                          << "chi: " << chi_val << '\t' 
                                          << "D^ell_mm': " << wigner_d_val << std::endl; 
                            }
                        }
                        
                    }
                }

                

                

            }

            P_I_element = P_I_element * ((comp)d_I)/LG_bar; 
        
            P_I(i,j) = P_I_element; 

            if(debug=='y')
            {
                std::cout << "d_I: " << d_I << '\t' 
                            << "|LG(P)|: " << LG_bar << std::endl; 
                std::cout << "P_I(" << i << "," << j << ") = " << P_I(i,j) << std::endl;
            }

        }
    }

    if(debug=='y')
    {
        std::cout << "Projection Matrix P_I = \n"; 
        Eigen::MatrixXd PIreal = P_I.real(); 
        std::cout << PIreal << std::endl; 
        std::cout << "____________________________________________________ " << std::endl; 
    }
    
    if(debug=='y')
    {
        double tol = 1e-12;
        test_projector_idempotency(P_I, tol);
        std::cout << "____________________________________________________ " << std::endl; 

    }
    


}

// ------------------------------------------------------------
// Extract orthonormal basis of projector image from P_I
// Equivalent to the Python P_irrep_subspace_o logic
// ------------------------------------------------------------
void projector_subspace_basis(Eigen::MatrixXcd& Psub,
                              const Eigen::MatrixXcd& P_I,
                              double eval_tol = 1e-13,
                              double chop_tol = 1e-12,
                              char debug = 'n')
{
    if (P_I.rows() != P_I.cols()) {
        throw std::runtime_error("projector_subspace_basis: P_I must be square");
    }

    const int N = P_I.rows();

    // eigh equivalent for Hermitian matrix
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(P_I);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error("projector_subspace_basis: eigensolver failed");
    }

    const auto& evals = es.eigenvalues();
    const auto& evecs = es.eigenvectors();

    std::vector<int> ivec;
    for (int i = 0; i < evals.size(); ++i) {
        if (std::abs(evals(i) - 1.0) < eval_tol) {
            ivec.push_back(i);
        }
    }

    // trace check
    double tr = P_I.trace().real();
    int tr_rounded = static_cast<int>(std::round(tr));

    
    if (static_cast<int>(ivec.size()) != tr_rounded) {
        std::cout << "Error in projector_subspace_basis: wrong subspace dimension\n";
        std::cout << "number of eigenvalues near 1 = " << ivec.size() << "\n";
        std::cout << "round(trace(P_I))           = " << tr_rounded << "\n";
        throw std::runtime_error("projector_subspace_basis: wrong subspace dimension");
    }

    if (ivec.empty()) {
        Psub = Eigen::MatrixXcd(N, 0);
        return;
    }

    // collect eigenvectors with eigenvalue ~ 1
    Eigen::MatrixXcd V(N, static_cast<int>(ivec.size()));
    for (int j = 0; j < static_cast<int>(ivec.size()); ++j) {
        V.col(j) = evecs.col(ivec[j]);
    }

    // Python used .real before QR
    Eigen::MatrixXd Vreal = V.real();

    // QR orthonormalization, equivalent to LA.qr(Psub)[0]
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(Vreal);
    Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(N, static_cast<int>(ivec.size()));

    Psub = chop(Q.cast<comp>(), chop_tol);

    if (debug == 'y') {
        std::cout << "Eigenvalues of P_I:\n" << evals.transpose() << "\n\n";
        std::cout << "Subspace dimension = " << ivec.size() << "\n";
        std::cout << "Psub = \n" << Psub.real() << "\n\n";
    }
}

//eigenvalue decomposition and diagonalization are done inherently 
void P_irrep_projection_2plus1( Eigen::MatrixXcd &P_I,
                                std::vector<std::vector<comp>> &plm_config,
                                std::vector<std::vector<int>> &np_config, 
                                std::vector<std::vector<comp>> &klm_config,
                                std::vector<std::vector<int>> &nk_config,
                                std::string &irrep, 
                                std::vector<comp> &total_P, 
                                std::vector<comp> &nnP_config,
                                bool sort_orbit_flag,
                                int parity 
                            )
{
    char debug = 'n';
    std::string I = irrep; 
    int dim1 = plm_config[0].size(); 
    int dim2 = klm_config[0].size(); 

    Eigen::MatrixXcd P_I_1(dim1, dim1); 
    Eigen::MatrixXcd P_I_2(dim2, dim2); 
    P_irrep_projection_single_flavor(P_I_1, plm_config, np_config, 
                                            plm_config, np_config,
                                            I, 
                                            total_P, nnP_config, 
                                            sort_orbit_flag, parity );

    Eigen::MatrixXcd Psub_I_1; 
    //projector_subspace_basis( Psub_I_1, P_I_1, 1e-16, 1e-16, 'n'); 
    P_irrep_projection_single_flavor(P_I_2, klm_config, nk_config, 
                                            klm_config, nk_config,
                                            I, 
                                            total_P, nnP_config, 
                                            sort_orbit_flag, parity );
    Eigen::MatrixXcd Psub_I_2; 
    //projector_subspace_basis( Psub_I_2, P_I_2, 1e-16, 1e-16, 'n'); 
    
    auto P_I_1_chopped = chop(P_I_1);
    auto P_I_2_chopped = chop(P_I_2); 

    P_I = blockDiag({P_I_1_chopped, P_I_2_chopped});

    if(debug=='y')
    {
        std::cout << "P_I :\n" ; 
        std::cout << P_I.real() << std::endl; 
    }

    if(debug=='y')
    {
        double tol = 1e-12;
        std::cout << "idempotency of the total P_I" << std::endl;
        test_projector_idempotency(P_I, tol);
        std::cout << "____________________________________________________ " << std::endl; 

    }
    
}

void analyze_projector_eigensystem(const Eigen::MatrixXcd& P_I,
                                   double zero_tol = 1e-12,
                                   bool normalize_evecs = true,
                                   char debug = 'y')
{
    if (P_I.rows() != P_I.cols()) {
        throw std::runtime_error("analyze_projector_eigensystem: P_I must be square.");
    }

    const int N = P_I.rows();

    // Since P_I is expected to be Hermitian/projector-like, use SelfAdjointEigenSolver
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(P_I);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error("analyze_projector_eigensystem: eigensolver failed.");
    }

    Eigen::VectorXd evals = es.eigenvalues();   // real eigenvalues
    Eigen::MatrixXcd evecs = es.eigenvectors(); // columns are eigenvectors

    int total_evals = evals.size();
    int zero_count = 0;
    int positive_count = 0;
    int negative_count = 0;

    for (int i = 0; i < total_evals; ++i) {
        double lam = evals(i);

        if (std::abs(lam) < zero_tol) {
            ++zero_count;
        }
        else if (lam > zero_tol) {
            ++positive_count;
        }
        else if (lam < -zero_tol) {
            ++negative_count;
        }
    }

    // Normalize eigenvectors to norm 1 if requested
    if (normalize_evecs) {
        for (int i = 0; i < evecs.cols(); ++i) {
            double nrm = evecs.col(i).norm();
            if (nrm > zero_tol) {
                evecs.col(i) /= nrm;
            }
        }
    }

    std::cout << "====================================================\n";
    std::cout << "Eigen-analysis of P_I\n";
    std::cout << "Matrix dimension              = " << N << " x " << N << "\n";
    std::cout << "Number of eigenvalues found   = " << total_evals << "\n";
    std::cout << "Number of ~zero eigenvalues   = " << zero_count << "\n";
    std::cout << "Number of >0 eigenvalues      = " << positive_count << "\n";
    std::cout << "Number of <0 eigenvalues      = " << negative_count << "\n";
    std::cout << "====================================================\n";

    std::cout << "Eigenvalues = \n" << evals << "\n";
    std::cout << "----------------------------------------------------\n";

    std::cout << "Eigenvector matrix (columns are eigenvectors) = \n";
    //std::cout << evecs << "\n";
    std::cout << "====================================================\n";

    if (normalize_evecs) {
        std::cout << "Norms of eigenvectors:\n";
        for (int i = 0; i < total_evals; ++i) {
            std::cout << "||v_" << i << "|| = " << evecs.col(i).norm() << "\n";
        }
        std::cout << "====================================================\n";
    }

    if (debug == 'y') {
        // Extra check: P_I * v = lambda * v
        std::cout << "Residual check ||P_I v - lambda v|| for each eigenpair:\n";
        for (int i = 0; i < total_evals; ++i) {
            Eigen::VectorXcd resid = P_I * evecs.col(i) - evals(i) * evecs.col(i);
            std::cout << "i = " << i << "   residual norm = " << resid.norm() << "\n";
        }
        std::cout << "====================================================\n";
    }
}

void build_projector_from_eigenvectors_near_one(const Eigen::MatrixXcd& P_I,
                                                Eigen::MatrixXcd& Vsel,
                                                Eigen::MatrixXcd& Pproj,
                                                double eig_tol = 0.05,
                                                double norm_tol = 1e-12,
                                                double proj_tol = 1e-10,
                                                char debug = 'y')
{
    if (P_I.rows() != P_I.cols()) {
        throw std::runtime_error("build_projector_from_eigenvectors_near_one: P_I must be square.");
    }

    const int N = P_I.rows();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(P_I);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error("build_projector_from_eigenvectors_near_one: eigensolver failed.");
    }

    Eigen::VectorXd evals = es.eigenvalues();
    Eigen::MatrixXcd evecs = es.eigenvectors(); // columns are eigenvectors

    // pick eigenvalues in [1-eig_tol, 1+eig_tol]
    std::vector<int> keep_inds;
    for (int i = 0; i < evals.size(); ++i) {
        if (evals(i) >= 1.0 - eig_tol && evals(i) <= 1.0 + eig_tol) {
            keep_inds.push_back(i);
        }
    }

    const int r = static_cast<int>(keep_inds.size());

    if (debug=='y')
    {
        std::cout << "====================================================\n";
        std::cout << "Selecting eigenvectors with eigenvalues in ["
                << 1.0 - eig_tol << ", " << 1.0 + eig_tol << "]\n";
        std::cout << "Total eigenvalues found = " << evals.size() << "\n";
        std::cout << "Number selected         = " << r << "\n";
        std::cout << "====================================================\n";
    }

    if (r == 0) {
        Vsel = Eigen::MatrixXcd(N, 0);
        Pproj = Eigen::MatrixXcd::Zero(N, N);
        if(debug=='y')
        {
            std::cout << "No eigenvalues found near 1.\n";
        }
        return;
    }

    // Build matrix of selected eigenvectors
    Vsel.resize(N, r);
    for (int j = 0; j < r; ++j) {
        Vsel.col(j) = evecs.col(keep_inds[j]);

        // normalize column to norm 1
        double nrm = Vsel.col(j).norm();
        if (nrm > norm_tol) {
            Vsel.col(j) /= nrm;
        }
    }

    // Build projector from selected orthonormal eigenvectors
    Pproj = Vsel * Vsel.adjoint();

    // Check projector properties
    double herm_res = (Pproj - Pproj.adjoint()).norm();
    double idem_res = (Pproj * Pproj - Pproj).norm();

    bool is_hermitian = (herm_res < proj_tol);
    bool is_idempotent = (idem_res < proj_tol);
    bool is_projector = is_hermitian && is_idempotent;

    if (debug=='y')
    {
        // print selected eigenvalues
        std::cout << "Selected eigenvalues:\n";
        for (int j = 0; j < r; ++j) {
            std::cout << "eval[" << keep_inds[j] << "] = " << evals(keep_inds[j]) << "\n";
        }
        std::cout << "====================================================\n";

        // print Vsel matrix
        std::cout << "Matrix of selected normalized eigenvectors Vsel\n";
        std::cout << "(each column is an eigenvector with eigenvalue near 1):\n";
        std::cout << Vsel << "\n";
        std::cout << "Vsel_real:" << "\n";
        std::cout << chop(Vsel, 1e-10).real() << "\n";
    }
    
    if (debug=='y')
    {
        // Python used .real before QR
        Eigen::MatrixXd Vreal = Vsel.real();

        // QR orthonormalization, equivalent to LA.qr(Psub)[0]
        
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(Vreal);
        Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(N, static_cast<int>(r));
        std::cout << "Vsel_real_qr:" << std::endl;
        std::cout << chop(Q,1e-10) <<std::endl; 
        
        
        std::cout << "====================================================\n";

        // print norms of selected eigenvectors
        std::cout << "Norms of selected eigenvectors:\n";
        for (int j = 0; j < r; ++j) {
            std::cout << "||Vsel.col(" << j << ")|| = " << Vsel.col(j).norm() << "\n";
        }
        std::cout << "====================================================\n";

        // print projector
        std::cout << "Projector built from selected eigenvectors:\n";
        std::cout << Pproj << "\n";
        std::cout << "====================================================\n";
    }
    if (debug == 'y')
    {
        std::cout << "====================================================\n";
        std::cout << "Projector checks:\n";
        std::cout << "||Pproj - Pproj^dagger|| = " << herm_res << "\n";
        std::cout << "||Pproj^2 - Pproj||      = " << idem_res << "\n";
        std::cout << "Hermitian?  " << (is_hermitian ? "yes" : "no") << "\n";
        std::cout << "Idempotent? " << (is_idempotent ? "yes" : "no") << "\n";
        std::cout << "Is projector? " << (is_projector ? "yes" : "no") << "\n";
        std::cout << "====================================================\n";
    }

    if (debug == 'y') {
        std::cout << "Residuals ||P_I v - lambda v|| for selected eigenpairs:\n";
        for (int j = 0; j < r; ++j) {
            int idx = keep_inds[j];
            Eigen::VectorXcd resid = P_I * Vsel.col(j) - evals(idx) * Vsel.col(j);
            std::cout << "selected j = " << j
                      << "   original eig index = " << idx
                      << "   residual norm = " << resid.norm() << "\n";
        }
        std::cout << "====================================================\n";
    }
}

//Here we tested how to make the P_irrep for the flavor 
//of the spectator, this matches python implementation 
void test_P_irrep_maker_v1()
{
    char debug = 'n'; 
    comp pi = std::acos(-1.0); 
    double atmpi = 0.5;//0.06906;
    double atmK  = 1.0;//0.09698;

    double eta_1 = 1.0;
    double eta_2 = 0.5;

    double alpha         = 0.5;
    double max_shell_num = 20.0;

    std::vector<int> waves_vec_1 = {0, 1};
    std::vector<int> waves_vec_2 = {0};

    double epsilon_h = 0;
    bool   Q0norm    = true;

    double xi    = 3.444;
    double Lbyas = 20;
    double L     = 4.0;//xi * Lbyas;

    comp twopibyL = ((comp)2.0) * pi / ((comp)L);

    Vec3 nnP = {0, 0, 1};
    std::vector<comp> total_P(3);
    total_P[0] = twopibyL * (double)nnP[0];
    total_P[1] = twopibyL * (double)nnP[1];
    total_P[2] = twopibyL * (double)nnP[2];
    
    std::vector<comp> total_nP(3); 
    total_nP[0] = ((comp)nnP[0]); 
    total_nP[1] = ((comp)nnP[1]);
    total_nP[2] = ((comp)nnP[2]); 

    std::cout << "total_nP = " << total_nP[0] << "," << total_nP[1] << "," << total_nP[2] << std::endl; 

    double tolerance = 0.0;

    std::vector<std::vector<comp>> scatter_params_1(4, std::vector<comp>(3));
    scatter_params_1[0][0] = 4.04;
    scatter_params_1[1][0] = -43.2;

    std::vector<std::vector<comp>> scatter_params_2(4, std::vector<comp>(3));
    scatter_params_2[0][0] = 4.12;

    double En_initial = 0.26310;
    double En_final   = 0.36;
    int    En_points  = 1000;
    double del_En     = std::abs(En_initial - En_final) / (double)En_points;

    double En = 3.9286896161789544;//3.601;//0.27; 

    std::vector<std::string> irreps;// = irrep_list(nnP);
    irreps.push_back("A2");
    const double SINGULAR_COND_THRESHOLD = 1e10;

    std::vector<std::vector<comp>> plm_config(5), klm_config(5);
    std::vector<std::vector<int>> np_config(5), nk_config(5); 
    config_maker_4(plm_config, np_config, waves_vec_1, En, total_P,
                atmK, atmK, atmpi, L, epsilon_h, max_shell_num, tolerance);
    config_maker_4(klm_config, nk_config, waves_vec_2, En, total_P,
                atmpi, atmK, atmK, L, epsilon_h, max_shell_num, tolerance);

    
    std::vector<comp> nnP_config(3); 
    for(int i=0; i<nnP.size(); ++i)
    {
        nnP_config[i] = ((comp) nnP[i]); 
    }
    std::vector<std::vector<comp>> orbit(3);
    bool sort_flag = false;  
    orbit_maker(orbit, nk_config, nnP_config, sort_flag); 
    //auto np_config = get_orbit(plm_config, total_P); 

    std::cout << "nk_config = [ ";

    for(size_t i = 0; i < nk_config[0].size(); ++i)
    {
        auto a = nk_config[0][i]; 
        auto b = nk_config[1][i]; 
        auto c = nk_config[2][i]; 

        std::cout << a << "," << b << "," << c << "\n";
    }
    std::cout << "]" << std::endl; 

    std::cout << "nk_orbit = [ ";
    for (size_t i = 0; i < orbit[0].size(); ++i) {
        auto a = orbit[0][i];
        auto b = orbit[1][i];
        auto c = orbit[2][i];


        std::cout << a << "," << b << "," << c << "\n";
    }
    std::cout << "]" << std::endl;

    
    std::string I = "A2"; 
    auto d_I = irrep_dim(I);
    std::cout << "d_I = " << d_I << std::endl; 

    //abort(); 
    auto LG = little_group(nnP_config); 
    for (const auto& row : LG) {
    std::cout << "{ ";
    for (const auto& x : row) {
        std::cout << x << " ";
    }
    std::cout << "}\n";
    
    }
    std::cout<<"LG size = "<<LG.size()<<std::endl; 
    for(int i=0; i<klm_config[0].size(); ++i)
    {
        std::vector<comp> pvec(3); 
        pvec[0] = klm_config[0][i]; 
        pvec[1] = klm_config[1][i]; 
        pvec[2] = klm_config[2][i]; 

        for(int j=0; j<LG.size(); ++j)
        {
            std::vector<int> R(3); 
            R[0] = LG[j][0];
            R[1] = LG[j][1];
            R[2] = LG[j][2]; 
            Eigen::Matrix3d D11 = Dmat11(R);
            Eigen::Matrix<double,5,5> D22 = Dmat22(R); 
            
            std::cout << "i,j = " << i << "," << j << std::endl;
            std::cout << "p = " << pvec[0] << "," << pvec[1] << "," << pvec[2] << '\t';
            std::cout << "R = " << R[0] << "," << R[1] << "," << R[2] << std::endl; 
            std::cout << "D11(R):\n";
            std::cout << D11 << std::endl; 
            std::cout << "D22(R):\n";
            std::cout << D22 << std::endl; 
            std::cout << "____________________________________" << std::endl;  
        }
    }
    

    wigner_d_tests();

    int parity = -1; 
    int proj_m_p = 0; 
    int proj_m_k = 0; 
    int ell_p = 0; 
    int ell_k = 0; 
    std::vector<comp> orbit_p(3);
    std::vector<comp> orbit_k(3);
    int dim1 = plm_config[0].size();
    int dim2 = klm_config[0].size(); 

    Eigen::MatrixXcd P_I(dim2, dim2); 
    std::cout << "check from here:___________________________________ " << std::endl; 


    for(int i=0; i<klm_config[0].size(); ++i)
    {
        std::vector<comp> pvec(3); 
        std::vector<int> npvec(3); 
        pvec[0] = klm_config[0][i]; 
        pvec[1] = klm_config[1][i]; 
        pvec[2] = klm_config[2][i];
        
        npvec[0] = nk_config[0][i]; 
        npvec[1] = nk_config[1][i]; 
        npvec[2] = nk_config[2][i]; 

        ell_p = static_cast<int> (std::real(klm_config[3][i])); 
        proj_m_p = static_cast<int> (std::real(klm_config[4][i])); 

        orbit_p[0] = orbit[0][i]; 
        orbit_p[1] = orbit[1][i]; 
        orbit_p[2] = orbit[2][i]; 

        for(int j=0; j<klm_config[0].size(); ++j)
        {
            std::vector<comp> kvec(3); 
            std::vector<int> nkvec(3); 
            kvec[0] = klm_config[0][j]; 
            kvec[1] = klm_config[1][j]; 
            kvec[2] = klm_config[2][j]; 
        
            nkvec[0] = nk_config[0][j]; 
            nkvec[1] = nk_config[1][j]; 
            nkvec[2] = nk_config[2][j]; 



            ell_k = static_cast<int> (std::real(klm_config[3][j])); 
            proj_m_k = static_cast<int> (std::real(klm_config[4][j])); 


            orbit_k[0] = orbit[0][j]; 
            orbit_k[1] = orbit[1][j]; 
            orbit_k[2] = orbit[2][j]; 

            comp P_I_element = {0.0, 0.0}; 

            double LG_bar = LG.size(); 
            
            for(int k=0; k<LG.size(); ++k)
            {
                std::vector<int> R(3); 
                R[0] = LG[k][0];
                R[1] = LG[k][1];
                R[2] = LG[k][2]; 

                auto Rp = cubic_transf(npvec, R); 

                if(debug=='y')
                {   
                    std::cout << "Projection Tests:___________________________________ " << std::endl; 
                    std::cout << "i:" << i << '\t' << "j:" << j << '\t' << "k:" << k << std::endl;
                    std::cout << "pvec: [" << pvec[0] << ","
                                           << pvec[1] << ","
                                           << pvec[2] << "]" << std::endl; 
                    std::cout << "npvec: [" << npvec[0] << ","
                                            << npvec[1] << ","
                                            << npvec[2] << "]" << std::endl; 
                    std::cout << "kvec: [" << kvec[0] << ","
                                           << kvec[1] << ","
                                           << kvec[2] << "]" << std::endl; 
                    std::cout << "nkvec: [" << nkvec[0] << ","
                                            << nkvec[1] << ","
                                            << nkvec[2] << "]" << std::endl; 
                    std::cout << "orbit_p: [ " << orbit_p[0] << ","
                                               << orbit_p[1] << ","
                                               << orbit_p[2] << "]"  << std::endl; 
                    std::cout << "orbit_k: [ " << orbit_k[0] << ","
                                               << orbit_k[1] << ","
                                               << orbit_k[2] << "]"  << std::endl; 
                    std::cout << "ell_p: " << ell_p << '\t' << "ell_k: " << ell_k << std::endl; 
                    std::cout << "proj_m_p: " << proj_m_p << '\t' << "proj_m_k: " << proj_m_k << std::endl; 
                    std::cout << "R: [" << R[0] << "," 
                                       << R[1] << "," 
                                       << R[2] << "]" << std::endl; 
                    std::cout << "Rp: [" << Rp[0] << "," 
                                        << Rp[1] << "," 
                                        << Rp[2] << "]" << std::endl; 
                                        
                }

                if( Rp == nkvec )
                {
                    if(debug=='y')
                    {
                        std::cout << "Rp=nkvec : True" << std::endl; 
                    }
                    if( ell_p == ell_k )
                    {
                        if(debug=='y')
                        {
                            std::cout << "ell_p = ell_k : True" << std::endl; 
                        }
                        if( orbit_p == orbit_k ) 
                        {
                            if(debug=='y')
                            {
                                std::cout << "orbit_p = orbit_k : True" << std::endl; 
                            }
                            int par = (parity == -1 && !is_in_rotations_list(R)) ? -1 : 1;
                            int chi_val = chi(R, I, nnP_config); 
                            double wigner_d_val = real_wigner_d::D_real_element(ell_k, proj_m_p, proj_m_k, R); 
                            P_I_element += par * chi_val * wigner_d_val; 

                            if(debug=='y')
                            {
                                std::cout << "par: " << par << '\t' 
                                          << "chi: " << chi_val << '\t' 
                                          << "D^ell_mm': " << wigner_d_val << std::endl; 
                            }
                        }
                        
                    }
                }

                

                

            }

            P_I_element = P_I_element * ((comp)d_I)/LG_bar; 
        
            P_I(i,j) = P_I_element; 

            if(debug=='y')
            {
                std::cout << "d_I: " << d_I << '\t' 
                            << "|LG(P)|: " << LG_bar << std::endl; 
                std::cout << "P_I(" << i << "," << j << ") = " << P_I(i,j) << std::endl;
            }

        }
    }

    if(debug=='y')
    {
        std::cout << "Projection Matrix P_I = \n"; 
        Eigen::MatrixXd PIreal = P_I.real(); 
        std::cout << PIreal << std::endl; 
        std::cout << "____________________________________________________ " << std::endl; 
    }
    
    double tol = 1e-12;
    
    test_projector_idempotency(P_I, tol);

    
}

std::string complex_to_string(const comp& z, int precision = 12)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision)
        << z.real()
        << (z.imag() >= 0 ? "+" : "")
        << z.imag() << "i";
    return oss.str();
}

void print_nonzero_elements_and_counts(const Eigen::MatrixXcd& P_I, double tol = 1e-12)
{
    std::vector<comp> nonzero_elements;
    std::map<std::string, int> counts;

    for (int i = 0; i < P_I.rows(); ++i)
    {
        for (int j = 0; j < P_I.cols(); ++j)
        {
            comp val = P_I(i, j);

            if (std::abs(val) > tol)
            {
                nonzero_elements.push_back(val);
                counts[complex_to_string(val)]++;
            }
        }
    }

    std::cout << "All non-zero elements:" << std::endl;
    for (const auto& val : nonzero_elements)
    {
        std::cout << val << std::endl;
    }

    std::cout << "\nCount of each distinct non-zero element:" << std::endl;
    for (const auto& [elem, count] : counts)
    {
        std::cout << elem << " appears " << count << " times" << std::endl;
    }
}

Eigen::MatrixXcd make_permutation_matrix_18()
{
    Eigen::MatrixXcd Pi = Eigen::MatrixXcd::Zero(18, 18);

    // 1-based permutation:
    // p = [1,3,2,4,5,7,6,8,10,15,11,16,9,14,12,17,13,18]

    std::vector<int> p = {
        0, 2, 1, 3, 4, 6, 5, 7,
        9, 14, 10, 15, 8, 13, 11, 16, 12, 17
    }; // converted to 0-based

    for (int i = 0; i < 18; ++i)
    {
        Pi(i, p[i]) = 1.0;
    }

    return Pi;
}

template <typename Derived>
auto slogdet(const Eigen::EigenBase<Derived>& Aexpr)
{
    using Scalar     = typename Derived::Scalar;
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    using PlainMat   = typename Derived::PlainObject;
    using SignType   = std::conditional_t<
        Eigen::NumTraits<Scalar>::IsComplex,
        Scalar,
        RealScalar
    >;

    if (Aexpr.rows() != Aexpr.cols()) {
        throw std::runtime_error("slogdet: matrix must be square");
    }

    PlainMat A = Aexpr.derived().eval();
    Eigen::PartialPivLU<PlainMat> lu(A);
    const auto& LU = lu.matrixLU();

    SignType sign = static_cast<RealScalar>(lu.permutationP().determinant());
    RealScalar logabsdet = RealScalar(0);

    for (int i = 0; i < A.rows(); ++i) {
        Scalar d = LU(i, i);
        RealScalar ad = std::abs(d);

        if constexpr (Eigen::NumTraits<Scalar>::IsComplex) {
            if (std::isnan(d.real()) || std::isnan(d.imag()) ||
                std::isinf(d.real()) || std::isinf(d.imag()) ||
                std::isnan(ad) || std::isinf(ad)) {
                std::cout << "Bad LU pivot at i = " << i
                          << "  d = " << d
                          << "  |d| = " << ad << "\n";
                return std::make_pair(
                    SignType(std::numeric_limits<RealScalar>::quiet_NaN(),
                             std::numeric_limits<RealScalar>::quiet_NaN()),
                    std::numeric_limits<RealScalar>::quiet_NaN()
                );
            }
        } else {
            if (std::isnan(d) || std::isinf(d) ||
                std::isnan(ad) || std::isinf(ad)) {
                std::cout << "Bad LU pivot at i = " << i
                          << "  d = " << d
                          << "  |d| = " << ad << "\n";
                return std::make_pair(
                    std::numeric_limits<RealScalar>::quiet_NaN(),
                    std::numeric_limits<RealScalar>::quiet_NaN()
                );
            }
        }

        if (ad == RealScalar(0)) {
            if constexpr (Eigen::NumTraits<Scalar>::IsComplex) {
                return std::make_pair(
                    SignType(RealScalar(0), RealScalar(0)),
                    -std::numeric_limits<RealScalar>::infinity()
                );
            } else {
                return std::make_pair(
                    RealScalar(0),
                    -std::numeric_limits<RealScalar>::infinity()
                );
            }
        }

        if constexpr (Eigen::NumTraits<Scalar>::IsComplex) {
            sign *= d / ad;
        } else {
            if (d < RealScalar(0)) sign = -sign;
        }

        logabsdet += std::log(ad);
    }

    return std::make_pair(sign, logabsdet);
}

void print_bad_entries(const Eigen::MatrixXcd& A, const std::string& name)
{
    bool found = false;
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            auto z = A(i,j);
            if (std::isnan(z.real()) || std::isnan(z.imag()) ||
                std::isinf(z.real()) || std::isinf(z.imag())) {
                std::cout << name << "(" << i << "," << j << ") = " << z << "\n";
                found = true;
            }
        }
    }
    if (!found) {
        std::cout << name << " has no NaN/Inf entries\n";
    }
}


void test_P_I_v1()
{
    char debug = 'n'; 
    comp pi = std::acos(-1.0); 
    double atmpi = 0.06906;//0.5;//0.06906;
    double atmK  = 0.09698;//1.0;//0.09698;

    double eta_1 = 1.0;
    double eta_2 = 0.5;

    double alpha         = 0.5;
    double max_shell_num = 20.0;

    std::vector<int> waves_vec_1 = {0, 1};
    std::vector<int> waves_vec_2 = {0};

    double epsilon_h = 0;
    bool   Q0norm    = true;

    double xi    = 3.444;
    double Lbyas = 20;
    double L     = xi * Lbyas; //4; //xi * Lbyas;

    comp twopibyL = ((comp)2.0) * pi / ((comp)L);

    Vec3 nnP = {0, 0, 1};
    std::vector<comp> total_P(3);
    total_P[0] = twopibyL * (double)nnP[0];
    total_P[1] = twopibyL * (double)nnP[1];
    total_P[2] = twopibyL * (double)nnP[2];
    
    std::vector<comp> total_nP(3); 
    total_nP[0] = ((comp)nnP[0]); 
    total_nP[1] = ((comp)nnP[1]);
    total_nP[2] = ((comp)nnP[2]); 

    std::cout << "total_nP = " << total_nP[0] << "," << total_nP[1] << "," << total_nP[2] << std::endl; 

    double tolerance = 0.0;

    std::vector<std::vector<comp>> scatter_params_1(4, std::vector<comp>(3));
    scatter_params_1[0][0] = 4.04;
    scatter_params_1[1][0] = -43.2;

    std::vector<std::vector<comp>> scatter_params_2(4, std::vector<comp>(3));
    scatter_params_2[0][0] = 4.12;

    double En_initial = 0.26310;
    double En_final   = 0.36;
    int    En_points  = 1000;
    double del_En     = std::abs(En_initial - En_final) / (double)En_points;

    double En = 0.30400815858956826;//3.9286896161789544;//0.27;//3.9286896161789544;//3.601;//0.27; 

    std::vector<std::string> irreps;// = irrep_list(nnP);
    irreps.push_back("A2");
    const double SINGULAR_COND_THRESHOLD = 1e10;

    std::vector<std::vector<comp>> plm_config(5), klm_config(5);
    std::vector<std::vector<int>> np_config(5), nk_config(5); 
    config_maker_4(plm_config, np_config, waves_vec_1, En, total_P,
                atmK, atmK, atmpi, L, epsilon_h, max_shell_num, tolerance);
    config_maker_4(klm_config, nk_config, waves_vec_2, En, total_P,
                atmpi, atmK, atmK, L, epsilon_h, max_shell_num, tolerance);

    
    std::vector<comp> nnP_config(3); 
    for(int i=0; i<nnP.size(); ++i)
    {
        nnP_config[i] = ((comp) nnP[i]); 
    }

    for(int i=0; i<np_config[0].size(); ++i)
    {
        int npx = np_config[0][i]; 
        int npy = np_config[1][i]; 
        int npz = np_config[2][i]; 
        int ell = np_config[3][i]; 
        int proj_m = np_config[4][i]; 
        std::cout << "np: [" << npx << ","
                             << npy << ","
                             << npz << "]\t"
                  << "ell:"  << ell << ","
                  << "proj_m:" << proj_m 
                  << std::endl; 
    }
    for(int i=0; i<nk_config[0].size(); ++i)
    {
        int npx = nk_config[0][i]; 
        int npy = nk_config[1][i]; 
        int npz = nk_config[2][i]; 
        int ell = nk_config[3][i]; 
        int proj_m = nk_config[4][i]; 
        std::cout << "np: [" << npx << ","
                             << npy << ","
                             << npz << "]\t"
                  << "ell:"  << ell << ","
                  << "proj_m:" << proj_m 
                  << std::endl; 
    }

    int dim1 = plm_config[0].size(); 
    int dim2 = klm_config[0].size(); 
    int total_dim = dim1 + dim2; 


    Eigen::MatrixXcd P_I_1(dim1, dim1); 
    Eigen::MatrixXcd P_I_2(dim2, dim2); 

    std::string I = irreps[0]; 

    bool sort_orbit_flag = false; 
    int parity = -1; 
    double chop_tol = 1e-10; 
    P_irrep_projection_single_flavor(P_I_1, plm_config, np_config, plm_config, np_config, I, total_P, nnP_config, sort_orbit_flag, parity);
    print_nonzero_elements_and_counts(P_I_1);
    
    P_irrep_projection_single_flavor(P_I_2, klm_config, nk_config, klm_config, nk_config, I, total_P, nnP_config, sort_orbit_flag, parity);
    print_nonzero_elements_and_counts(P_I_2);
    auto chopped_P_I_1 = chop(P_I_1, chop_tol); 
    auto chopped_P_I_2 = chop(P_I_2, chop_tol); 
    std::cout << "chopped P_I_1:\n" << chopped_P_I_1.real() << std::endl; 
    std::cout << "chopped P_I_2:\n" << chopped_P_I_2.real() << std::endl; 

    double eig_tol = 0.05;
    double norm_tol = 1e-12;
    double proj_tol = 1e-10; 
    Eigen::MatrixXcd Vsel1;
    Eigen::MatrixXcd Pproj1;
    build_projector_from_eigenvectors_near_one(P_I_1, Vsel1, Pproj1, eig_tol, norm_tol, proj_tol, debug='y'); 
    Eigen::MatrixXcd Vsel2;
    Eigen::MatrixXcd Pproj2;
    build_projector_from_eigenvectors_near_one(P_I_2, Vsel2, Pproj2, eig_tol, norm_tol, proj_tol, debug='y'); 
    
    
    
    Eigen::MatrixXcd P_I(dim1+dim2, dim1+dim2); 

    P_irrep_projection_2plus1(P_I, plm_config, np_config, klm_config, nk_config, I, total_P, nnP_config, sort_orbit_flag, parity); 
    Eigen::MatrixXcd Vsel;
    Eigen::MatrixXcd Pproj;
    build_projector_from_eigenvectors_near_one(P_I, Vsel, Pproj, eig_tol, norm_tol, proj_tol, debug='y'); 
    std::cout << "Vsel of P_I:" << std::endl;
    std::cout << chop(Vsel.real(),1e-3) << std::endl; 
    
    /*
    Eigen::MatrixXcd Pi = make_permutation_matrix_18();

    auto P_I_perm = Pi.transpose() * P_I * Pi; 
    std::cout << "permutated P_I:\n" << P_I_perm.real() << std::endl;
    std::cout << "unpermutated P_I:\n" << P_I.real() << std::endl;

    Eigen::MatrixXcd Vselperm;
    Eigen::MatrixXcd Pprojperm;
    build_projector_from_eigenvectors_near_one(P_I_perm, Vselperm, Pprojperm, eig_tol, norm_tol, proj_tol, debug='y'); 
    std::cout << "Vsel of permutated P_I:" << std::endl;
    std::cout << chop(Vselperm.real(),1e-3) << std::endl; 
    */
    
    /*
    //analyze_projector_eigensystem(P_I, 1e-16, true, 'y');
    Eigen::MatrixXcd Psub = P_I;
    
    Eigen::MatrixXcd Vsel;
    Eigen::MatrixXcd Pproj;
    build_projector_from_eigenvectors_near_one(P_I, Vsel, Pproj, eig_tol, norm_tol, proj_tol, debug='y'); 
    
    std::cout << "chopped P_I:\n"; 
    std::cout << chop(P_I).real() << std::endl; 
    */
   
    //projector_subspace_basis(Psub, P_I, 1e-16, 1e-16, 'n');

    /*
    std::cout << "Psub:" << std::endl; 
    std::cout << Psub << std::endl; 
    */

    Eigen::MatrixXcd F3mat(total_dim, total_dim);
    comp F3iso;
    Eigen::VectorXcd state_vec(total_dim);
    Eigen::MatrixXcd F2mat(total_dim, total_dim);
    Eigen::MatrixXcd K2imat(total_dim, total_dim);
    Eigen::MatrixXcd Gmat(total_dim, total_dim); 
    Eigen::MatrixXcd Hmatinv(total_dim, total_dim); 
    test_F3iso_ND_2plus1_mat_with_normalization_single_En(  F3mat, F3iso, 
                                                            state_vec, 
                                                            F2mat, K2imat, Gmat, Hmatinv, 
                                                            En, 
                                                            plm_config, klm_config, 
                                                            total_P, 
                                                            eta_1, eta_2, 
                                                            scatter_params_1, scatter_params_2,
                                                            atmK, atmpi, 
                                                            alpha, epsilon_h, 
                                                            L, max_shell_num, 
                                                            Q0norm );
    
    
    Eigen::MatrixXcd F3matinv = F3mat.inverse();               
    auto projF3 = Vsel.transpose() * F3matinv * Vsel;                       
    
    std::cout << "F3 mat :" << std::endl; 
    std::cout << F3mat.real() << std::endl;
    std::cout << "_________________________________" << std::endl;
    std::cout << "F3 mat inverse :" << std::endl; 
    std::cout << F3matinv.real() << std::endl; 
    std::cout << "_________________________________" << std::endl;
    std::cout << "F3 mat determinant :" << std::endl; 
    std::cout << F3matinv.determinant() << std::endl; 
    std::cout << "_________________________________" << std::endl;
    std::cout << "projected F3inv :" << std::endl; 
    std::cout << projF3.real() << std::endl;
    std::cout << "det projected F3inv :" << std::endl; 
    std::cout << projF3.determinant() << std::endl;
    std::cout << "_________________________________" << std::endl;

    auto [signB, logabsdetB] = slogdet(F3matinv);

    std::cout << "Complex matrix:\n";
    std::cout << "sign      = " << signB << "\n";
    std::cout << "logabsdet = " << logabsdetB << "\n";
    std::cout << "det recon = " << signB * std::exp(logabsdetB) << "\n";
    std::cout << "det exact = " << F3matinv.determinant() << "\n";
    std::cout << "_________________________________" << std::endl;
    
}



#endif 

/*
int main()
{
    test_P_I_v1();

    return 0; 
}
*/