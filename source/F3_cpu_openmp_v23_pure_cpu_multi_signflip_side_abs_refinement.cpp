// F3_cpu_openmp_v23_pure_cpu_multi_signflip_side_abs_refinement.cpp
//
// Pure CPU/OpenMP version:
//   - No CUDA
//   - No cuBLAS
//   - No cuSOLVER
//   - Matrix/config/projection building: CPU/OpenMP + Eigen + original functions
//   - F3 solves/products/determinants: CPU Eigen only
//
// Adaptive refinement:
//   1. Coarse scan.
//   2. Find sign flips.
//   3. Refine each sign-flip interval exactly [E_i, E_{i+1}].
//   4. If refined mesh has multiple sign flips, classify each sub-flip separately.
//   5. For one/multiple sign flips, use side-wise |det| trends:
//        likely_zero: |det| decreases inward from both sides.
//        likely_pole: |det| increases inward from both sides.
//        ambiguous: otherwise, then refine exactly [E_j, E_{j+1}] again.
//
// Compile with:
//   bash compile_F3_cpu_openmp_v23_pure_cpu_multi_signflip_side_abs_refinement.sh perf
//
// Run:
//   ./test_F3_cpu_openmp_v23_pure_cpu_multi_signflip_side_abs_refinement \
//      0 0 1 A2 A2 1000 50 0.263101 0.36 20 n

#include <Eigen/Dense>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// Your existing physics headers.
#include "functions.h"
#include "F2_functions_v2.h"
#include "K2_functions_v2.h"
#include "G_functions_v2.h"
#include "QC_functions_v2.h"
#include "projections_v1.hpp"

using comp = std::complex<double>;

struct LocalScopedTimer
{
    std::string name;
    char debug;
    std::chrono::high_resolution_clock::time_point t0;

    LocalScopedTimer(std::string name_, char debug_)
        : name(std::move(name_)), debug(debug_), t0(std::chrono::high_resolution_clock::now())
    {}

    ~LocalScopedTimer()
    {
        if (debug == 'y')
        {
            const auto t1 = std::chrono::high_resolution_clock::now();
            const double sec = std::chrono::duration<double>(t1 - t0).count();
            std::cout << "[timer] " << name << " : " << std::setprecision(16) << sec << " sec\n";
        }
    }
};

struct PhysicsParams
{
    double atmpi = 0.06906;
    double atmK  = 0.09698;
    double eta_1 = 1.0;
    double eta_2 = 0.5;
    double alpha = 0.5;
    double epsilon_h = 0.0;
    double max_shell_num = 20.0;
    double tolerance = 1.0e-12;
    double xi = 3.444;
    double Lbyas = 20.0;
    bool Q0norm = true;
    bool sort_orbit_flag = false;
    int parity = -1;
    double eig_tol  = 0.05;
    double norm_tol = 1.0e-12;
    double proj_tol = 1.0e-10;
    int omp_threads = 18;

    std::vector<int> waves_vec_1 = {0, 1};
    std::vector<int> waves_vec_2 = {0};

    std::vector<std::vector<comp>> scatter_params_1;
    std::vector<std::vector<comp>> scatter_params_2;

    PhysicsParams()
    {
        scatter_params_1.assign(4, std::vector<comp>(3, comp(0.0, 0.0)));
        scatter_params_2.assign(4, std::vector<comp>(3, comp(0.0, 0.0)));

        scatter_params_1[0][0] = comp(4.04, 0.0);
        scatter_params_1[1][0] = comp(-43.2, 0.0);

        scatter_params_2[0][0] = comp(4.12, 0.0);
    }

    double L() const { return xi * Lbyas; }
};

struct EvalResult
{
    int i = -1;
    double Ecm = std::numeric_limits<double>::quiet_NaN();
    double En = std::numeric_limits<double>::quiet_NaN();

    int dim1 = 0;
    int dim2 = 0;
    int total_dim = 0;
    int vdim = 0;

    comp det = comp(std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN());

    comp eig_min_proj = comp(std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::quiet_NaN());

    bool success = false;
    std::string error;
};

struct ScanPoint
{
    int i = -1;
    double Ecm = std::numeric_limits<double>::quiet_NaN();
    comp det = comp(std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN());
    double y = std::numeric_limits<double>::quiet_NaN();
    double y_norm = std::numeric_limits<double>::quiet_NaN();
    double abs_norm = std::numeric_limits<double>::quiet_NaN();
    int total_dim = 0;
    int vdim = 0;
    bool success = false;
};

struct SignFlip
{
    int left = -1;
    int right = -1;
    double E_linear = std::numeric_limits<double>::quiet_NaN();
};

struct ClassifyResult
{
    std::string classification = "ambiguous";
    int flip_index = -1;
    int j = -1;
    int jp1 = -1;
    double E_linear = std::numeric_limits<double>::quiet_NaN();

    int left_start = -1;
    int left_end = -1;
    int right_start = -1;
    int right_end = -1;

    double left_dec_frac = 0.0;
    double left_inc_frac = 0.0;
    double right_dec_frac = 0.0;
    double right_inc_frac = 0.0;
    double min_abs_local = std::numeric_limits<double>::quiet_NaN();
    double edge_abs_scale = std::numeric_limits<double>::quiet_NaN();
    double zero_depth_ratio = std::numeric_limits<double>::quiet_NaN();
};

struct RefineTask
{
    int candidate = -1;
    int branch = 0;
    int round = 0;
    double E_left = 0.0;
    double E_right = 0.0;
};

static std::string nnP_tag_string(const std::vector<int>& nnP)
{
    std::ostringstream os;
    os << nnP[0] << nnP[1] << nnP[2];
    return os.str();
}

static std::string L_tag(double Lbyas)
{
    std::ostringstream os;
    os << "L" << std::setprecision(12) << Lbyas;
    std::string s = os.str();
    for (char& c : s)
    {
        if (c == '.') c = 'p';
    }
    return s;
}

static comp determinant_via_partial_piv_lu(const Eigen::MatrixXcd& A)
{
    if (A.rows() != A.cols() || A.rows() == 0)
    {
        return comp(std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN());
    }

    Eigen::PartialPivLU<Eigen::MatrixXcd> lu(A);
    return lu.determinant();
}

static EvalResult evaluate_one_energy_cpu_eigen(
    int i,
    double Ecm,
    const std::vector<int>& nnP_vec,
    const std::string& irrep,
    const PhysicsParams& par,
    char debug)
{
    EvalResult out;
    out.i = i;
    out.Ecm = Ecm;

    try
    {
        const comp pi = std::acos(-1.0);
        const double L = par.L();
        const comp twopibyL = ((comp)2.0) * pi / ((comp)L);

        std::vector<comp> total_P(3);
        total_P[0] = twopibyL * double(nnP_vec[0]);
        total_P[1] = twopibyL * double(nnP_vec[1]);
        total_P[2] = twopibyL * double(nnP_vec[2]);

        std::vector<comp> nnP_config(3);
        nnP_config[0] = comp(nnP_vec[0], 0.0);
        nnP_config[1] = comp(nnP_vec[1], 0.0);
        nnP_config[2] = comp(nnP_vec[2], 0.0);

        const comp Ecm_c(Ecm, 0.0);
        const comp En_c = Ecm_to_E(Ecm_c, total_P);
        out.En = En_c.real();

        std::vector<std::vector<comp>> plm_config(5);
        std::vector<std::vector<comp>> klm_config(5);

        std::vector<std::vector<int>> np_config(5);
        std::vector<std::vector<int>> nk_config(5);

        config_maker_4(
            plm_config,
            np_config,
            par.waves_vec_1,
            En_c,
            total_P,
            par.atmK,
            par.atmK,
            par.atmpi,
            L,
            par.epsilon_h,
            par.max_shell_num,
            par.tolerance
        );

        config_maker_4(
            klm_config,
            nk_config,
            par.waves_vec_2,
            En_c,
            total_P,
            par.atmpi,
            par.atmK,
            par.atmK,
            L,
            par.epsilon_h,
            par.max_shell_num,
            par.tolerance
        );

        const int dim1 = int(plm_config[0].size());
        const int dim2 = int(klm_config[0].size());
        const int N = dim1 + dim2;

        out.dim1 = dim1;
        out.dim2 = dim2;
        out.total_dim = N;

        if (N <= 0)
        {
            out.error = "empty config";
            return out;
        }

        Eigen::MatrixXcd F2(N, N);
        Eigen::MatrixXcd G(N, N);
        Eigen::MatrixXcd K2inv(N, N);

        F2_2plus1_mat(
            F2,
            En_c,
            plm_config,
            klm_config,
            total_P,
            par.atmK,
            par.atmpi,
            L,
            par.alpha,
            par.epsilon_h,
            par.max_shell_num,
            par.Q0norm
        );

        K2inv_EREord2_2plus1_mat(
            K2inv,
            par.eta_1,
            par.eta_2,
            par.scatter_params_1,
            par.scatter_params_2,
            En_c,
            plm_config,
            klm_config,
            total_P,
            par.atmK,
            par.atmpi,
            par.epsilon_h,
            L
        );

        G_2plus1_mat(
            G,
            En_c,
            plm_config,
            klm_config,
            total_P,
            par.atmK,
            par.atmpi,
            L,
            par.alpha,
            par.epsilon_h,
            par.max_shell_num,
            par.Q0norm
        );

        Eigen::MatrixXcd H = K2inv + F2 + G;

        Eigen::MatrixXcd P_I(N, N);
        std::string I_mut = irrep;
        std::vector<comp> total_P_mut = total_P;
        std::vector<comp> nnP_config_mut = nnP_config;

        P_irrep_projection_2plus1(
            P_I,
            plm_config,
            np_config,
            klm_config,
            nk_config,
            I_mut,
            total_P_mut,
            nnP_config_mut,
            par.sort_orbit_flag,
            par.parity
        );

        Eigen::MatrixXcd Vsel;
        Eigen::MatrixXcd Pproj;

        build_projector_from_eigenvectors_near_one(
            P_I,
            Vsel,
            Pproj,
            par.eig_tol,
            par.norm_tol,
            par.proj_tol,
            'n'
        );

        out.vdim = int(Vsel.cols());

        if (out.vdim <= 0)
        {
            out.error = "projection produced Vsel with zero columns";
            return out;
        }

        // CPU Eigen solve path:
        //   H  X1 = F2
        //   F3 = F2/3 - F2 X1
        //   F3 X2 = Vsel
        //   F3inv_projected = Vsel^H X2
        //   det = det(F3inv_projected)
        Eigen::PartialPivLU<Eigen::MatrixXcd> luH(H);
        Eigen::MatrixXcd X1 = luH.solve(F2);

        Eigen::MatrixXcd F3 = (F2 / comp(3.0, 0.0)) - F2 * X1;

        Eigen::PartialPivLU<Eigen::MatrixXcd> luF3(F3);
        Eigen::MatrixXcd X2 = luF3.solve(Vsel);

        Eigen::MatrixXcd F3inv_projected = Vsel.adjoint() * X2;

        out.det = determinant_via_partial_piv_lu(F3inv_projected);

        // Optional: smallest eigenvalue if your functions.h has it. If not, leave NaN.
        // out.eig_min_proj = smallest_eigenvalue(F3inv_projected);

        out.success = std::isfinite(out.det.real()) && std::isfinite(out.det.imag());

        if (debug == 'y')
        {
            #pragma omp critical
            {
                std::cout << "[cpu-only] i=" << i
                          << " Ecm=" << std::setprecision(17) << Ecm
                          << " dim=(" << dim1 << "," << dim2 << ")"
                          << " N=" << N
                          << " vdim=" << out.vdim
                          << " det=(" << out.det.real() << "," << out.det.imag() << ")"
                          << " success=" << out.success
                          << "\n";
            }
        }
    }
    catch (const std::exception& e)
    {
        out.success = false;
        out.error = e.what();

        if (debug == 'y')
        {
            #pragma omp critical
            {
                std::cerr << "[cpu-only] ERROR i=" << i
                          << " Ecm=" << std::setprecision(17) << Ecm
                          << " : " << e.what() << "\n";
            }
        }
    }

    return out;
}

static std::vector<ScanPoint> run_scan_cpu_openmp(
    const std::vector<int>& nnP_vec,
    const std::string& irrep,
    const std::string& irrep_tag,
    double E0,
    double E1,
    int Npts,
    const std::string& scan_tag,
    const PhysicsParams& par,
    char debug)
{
    if (Npts <= 1)
    {
        throw std::runtime_error("Npts must be > 1");
    }

    const double dE = (E1 - E0) / double(Npts - 1);

    std::vector<EvalResult> evals(static_cast<std::size_t>(Npts));

    {
        LocalScopedTimer timer("CPU-only OpenMP scan " + scan_tag, debug);

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < Npts; ++i)
        {
            const double Ecm = E0 + double(i) * dE;
            evals[size_t(i)] = evaluate_one_energy_cpu_eigen(
                i, Ecm, nnP_vec, irrep, par, debug
            );
        }
    }

    std::vector<ScanPoint> pts(static_cast<std::size_t>(Npts));

    double max_abs_y = 0.0;
    for (int i = 0; i < Npts; ++i)
    {
        const auto& e = evals[size_t(i)];
        pts[size_t(i)].i = i;
        pts[size_t(i)].Ecm = e.Ecm;
        pts[size_t(i)].det = e.det;
        pts[size_t(i)].y = e.det.real();
        pts[size_t(i)].total_dim = e.total_dim;
        pts[size_t(i)].vdim = e.vdim;
        pts[size_t(i)].success = e.success;

        if (e.success && std::isfinite(pts[size_t(i)].y))
        {
            max_abs_y = std::max(max_abs_y, std::abs(pts[size_t(i)].y));
        }
    }

    if (!(max_abs_y > 0.0) || !std::isfinite(max_abs_y))
    {
        max_abs_y = 1.0;
    }

    for (auto& p : pts)
    {
        if (p.success && std::isfinite(p.y))
        {
            p.y_norm = p.y / max_abs_y;
            p.abs_norm = std::abs(p.det) / max_abs_y;
        }
    }

    const std::string tag = nnP_tag_string(nnP_vec);
    const std::string ltag = L_tag(par.Lbyas);

    const std::string raw_name =
        "det_proj_F3i_raw_" + tag + "_" + irrep_tag + scan_tag +
        "_cpu_only_v23_" + ltag + ".dat";

    const std::string norm_name =
        "det_proj_F3i_normalized_" + tag + "_" + irrep_tag + scan_tag +
        "_cpu_only_v23_" + ltag + ".dat";

    {
        std::ofstream f(raw_name);
        f << std::setprecision(40);
        f << "# i\tEcm\tEn_dummy\tdet_real\tdet_imag\tabs_det\ttotal_dim\tvdim\tsuccess\n";
        for (const auto& p : pts)
        {
            f << p.i << '\t'
              << p.Ecm << '\t'
              << 0.0 << '\t'
              << p.det.real() << '\t'
              << p.det.imag() << '\t'
              << std::abs(p.det) << '\t'
              << p.total_dim << '\t'
              << p.vdim << '\t'
              << int(p.success) << '\n';
        }
    }

    {
        std::ofstream f(norm_name);
        f << std::setprecision(40);
        f << "# i\tEcm\tEcm_imag\tdet_norm_real\tdet_norm_imag\tabs_det_norm\ttotal_dim\tvdim\tsuccess\n";
        for (const auto& p : pts)
        {
            f << p.i << '\t'
              << p.Ecm << '\t'
              << 0.0 << '\t'
              << p.y_norm << '\t'
              << (p.det.imag() / max_abs_y) << '\t'
              << p.abs_norm << '\t'
              << p.total_dim << '\t'
              << p.vdim << '\t'
              << int(p.success) << '\n';
        }
    }

    if (debug == 'y')
    {
        std::cout << "[cpu-only] wrote " << raw_name << "\n";
        std::cout << "[cpu-only] wrote " << norm_name << "\n";
    }

    return pts;
}

static std::vector<SignFlip> find_sign_flips(const std::vector<ScanPoint>& pts)
{
    std::vector<SignFlip> flips;

    for (int i = 0; i + 1 < int(pts.size()); ++i)
    {
        const auto& a = pts[size_t(i)];
        const auto& b = pts[size_t(i + 1)];

        if (!a.success || !b.success) continue;
        if (!std::isfinite(a.y_norm) || !std::isfinite(b.y_norm)) continue;

        if (a.y_norm == 0.0)
        {
            flips.push_back({i, i, a.Ecm});
        }
        else if (a.y_norm * b.y_norm < 0.0)
        {
            double ez = a.Ecm;
            if (b.y_norm != a.y_norm)
            {
                ez = a.Ecm - a.y_norm * (b.Ecm - a.Ecm) / (b.y_norm - a.y_norm);
            }
            else
            {
                ez = 0.5 * (a.Ecm + b.Ecm);
            }

            flips.push_back({i, i + 1, ez});
        }
    }

    return flips;
}

static std::pair<double, double> trend_fraction(
    const std::vector<ScanPoint>& pts,
    int start,
    int end)
{
    // Walk from start -> end. Decreasing means |y| goes down as we walk inward.
    if (start < 0 || end < 0 || start >= int(pts.size()) || end >= int(pts.size()) || start == end)
    {
        return {0.0, 0.0};
    }

    const int step = (end > start) ? 1 : -1;

    int dec = 0;
    int inc = 0;
    int flat = 0;
    int total = 0;

    for (int i = start; i != end; i += step)
    {
        const int j = i + step;

        const double a = std::abs(pts[size_t(i)].y_norm);
        const double b = std::abs(pts[size_t(j)].y_norm);

        if (!std::isfinite(a) || !std::isfinite(b)) continue;

        if (b < a) dec++;
        else if (b > a) inc++;
        else flat++;

        total++;
    }

    if (total <= 0) return {0.0, 0.0};

    return {double(dec) / double(total), double(inc) / double(total)};
}

static ClassifyResult classify_one_subflip_side_abs(
    const std::vector<ScanPoint>& pts,
    const std::vector<SignFlip>& flips,
    int flip_index,
    double trend_fraction_required)
{
    ClassifyResult cr;
    cr.flip_index = flip_index;

    const int n = int(pts.size());
    const SignFlip& f = flips[size_t(flip_index)];

    cr.j = f.left;
    cr.jp1 = f.right;
    cr.E_linear = f.E_linear;

    if (cr.j < 0 || cr.jp1 < 0 || cr.j >= n || cr.jp1 >= n)
    {
        cr.classification = "ambiguous";
        return cr;
    }

    if (flips.size() == 1)
    {
        cr.left_start = 0;
        cr.left_end = cr.j;
        cr.right_start = n - 1;
        cr.right_end = cr.jp1;
    }
    else
    {
        if (flip_index == 0)
        {
            cr.left_start = 0;
            cr.left_end = cr.j;

            const auto& next = flips[size_t(flip_index + 1)];
            cr.right_start = (cr.jp1 + next.left) / 2;
            cr.right_end = cr.jp1;
        }
        else if (flip_index == int(flips.size()) - 1)
        {
            const auto& prev = flips[size_t(flip_index - 1)];
            cr.left_start = (prev.right + cr.j) / 2;
            cr.left_end = cr.j;

            cr.right_start = n - 1;
            cr.right_end = cr.jp1;
        }
        else
        {
            const auto& prev = flips[size_t(flip_index - 1)];
            const auto& next = flips[size_t(flip_index + 1)];

            cr.left_start = (prev.right + cr.j) / 2;
            cr.left_end = cr.j;

            cr.right_start = (cr.jp1 + next.left) / 2;
            cr.right_end = cr.jp1;
        }
    }

    cr.left_start = std::clamp(cr.left_start, 0, n - 1);
    cr.left_end = std::clamp(cr.left_end, 0, n - 1);
    cr.right_start = std::clamp(cr.right_start, 0, n - 1);
    cr.right_end = std::clamp(cr.right_end, 0, n - 1);

    const auto [ldec, linc] = trend_fraction(pts, cr.left_start, cr.left_end);
    const auto [rdec, rinc] = trend_fraction(pts, cr.right_start, cr.right_end);

    cr.left_dec_frac = ldec;
    cr.left_inc_frac = linc;
    cr.right_dec_frac = rdec;
    cr.right_inc_frac = rinc;

    double min_abs = std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i)
    {
        if (pts[size_t(i)].success && std::isfinite(pts[size_t(i)].y_norm))
        {
            min_abs = std::min(min_abs, std::abs(pts[size_t(i)].y_norm));
        }
    }
    cr.min_abs_local = min_abs;

    const double edge_l = std::abs(pts[size_t(cr.left_start)].y_norm);
    const double edge_r = std::abs(pts[size_t(cr.right_start)].y_norm);
    cr.edge_abs_scale = 0.5 * (edge_l + edge_r);
    if (cr.edge_abs_scale > 0.0 && std::isfinite(cr.edge_abs_scale))
    {
        cr.zero_depth_ratio = cr.min_abs_local / cr.edge_abs_scale;
    }

    const bool both_decrease =
        (cr.left_dec_frac >= trend_fraction_required &&
         cr.right_dec_frac >= trend_fraction_required);

    const bool both_increase =
        (cr.left_inc_frac >= trend_fraction_required &&
         cr.right_inc_frac >= trend_fraction_required);

    if (both_decrease)
    {
        cr.classification = "likely_zero";
    }
    else if (both_increase)
    {
        cr.classification = "likely_pole";
    }
    else
    {
        cr.classification = "ambiguous";
    }

    return cr;
}

static bool add_unique_zero(std::vector<double>& zeros, double z, double tol)
{
    if (!std::isfinite(z)) return false;

    for (double old : zeros)
    {
        if (std::abs(old - z) <= tol)
        {
            return false;
        }
    }

    zeros.push_back(z);
    std::sort(zeros.begin(), zeros.end());
    return true;
}

static void run_adaptive_cpu_only_v23(
    const std::vector<int>& nnP_vec,
    const std::string& irrep,
    const std::string& irrep_tag,
    int coarseN,
    int refineN,
    double E0,
    double E1,
    PhysicsParams par,
    char debug)
{
    if (coarseN <= 1 || refineN <= 1)
    {
        throw std::runtime_error("coarseN and refineN must be > 1");
    }

    omp_set_num_threads(par.omp_threads);
    Eigen::setNbThreads(1);

    std::cout << "# F3 pure CPU/OpenMP/Eigen v23 multi-signflip side-abs refinement\n";
    std::cout << "# nnP=[" << nnP_vec[0] << "," << nnP_vec[1] << "," << nnP_vec[2] << "]"
              << " irrep=" << irrep
              << " irrep_tag=" << irrep_tag
              << " coarseN=" << coarseN
              << " refineN=" << refineN
              << " Ecm=[" << std::setprecision(17) << E0 << "," << E1 << "]"
              << " Lbyas=" << par.Lbyas
              << " omp_threads=" << par.omp_threads
              << "\n";
    std::cout << "# classifier=multi_signflip_side_abs_cpu_only_eigen\n";

    const std::string coarse_tag =
        "_cpu_v23_coarse_N" + std::to_string(coarseN);

    std::vector<ScanPoint> coarse_pts = run_scan_cpu_openmp(
        nnP_vec,
        irrep,
        irrep_tag,
        E0,
        E1,
        coarseN,
        coarse_tag,
        par,
        debug
    );

    std::vector<SignFlip> coarse_flips = find_sign_flips(coarse_pts);

    std::cout << "[adaptive-cpu-only] coarse sign flips = "
              << coarse_flips.size() << "\n";

    const int max_rounds = 12;
    const double min_window_width = 1.0e-14;
    const double trend_fraction_required = 0.65;
    const double zero_duplicate_tol = 1.0e-10;

    std::vector<RefineTask> tasks;
    tasks.reserve(coarse_flips.size());

    for (int c = 0; c < int(coarse_flips.size()); ++c)
    {
        const auto& fl = coarse_flips[size_t(c)];
        double a = coarse_pts[size_t(fl.left)].Ecm;
        double b = coarse_pts[size_t(fl.right)].Ecm;

        if (a > b) std::swap(a, b);

        tasks.push_back({c, 0, 0, a, b});
    }

    std::vector<double> finalized_zeros;

    int next_branch_id = 1;

    while (!tasks.empty())
    {
        RefineTask task = tasks.back();
        tasks.pop_back();

        if (task.round > max_rounds)
        {
            std::cout << "[adaptive-cpu-only] candidate=" << task.candidate
                      << " branch=" << task.branch
                      << " reached max_rounds; stopping ambiguous\n";
            continue;
        }

        if (std::abs(task.E_right - task.E_left) < min_window_width)
        {
            std::cout << "[adaptive-cpu-only] candidate=" << task.candidate
                      << " branch=" << task.branch
                      << " round=" << task.round
                      << " window too small; stopping ambiguous width="
                      << std::setprecision(17) << std::abs(task.E_right - task.E_left)
                      << "\n";
            continue;
        }

        std::ostringstream stag;
        stag << "_cpu_v23_refine_" << task.candidate
             << "_branch" << task.branch
             << "_round" << task.round
             << "_N" << refineN;

        std::vector<ScanPoint> ref_pts = run_scan_cpu_openmp(
            nnP_vec,
            irrep,
            irrep_tag,
            task.E_left,
            task.E_right,
            refineN,
            stag.str(),
            par,
            debug
        );

        std::vector<SignFlip> ref_flips = find_sign_flips(ref_pts);

        if (ref_flips.empty())
        {
            std::cout << "[adaptive-cpu-only] candidate=" << task.candidate
                      << " branch=" << task.branch
                      << " round=" << task.round
                      << " N=" << refineN
                      << " found_sign_flips=0 class=no_sign_flip"
                      << " window=[" << std::setprecision(17)
                      << task.E_left << "," << task.E_right << "]\n";
            continue;
        }

        for (int sf = 0; sf < int(ref_flips.size()); ++sf)
        {
            ClassifyResult cr = classify_one_subflip_side_abs(
                ref_pts,
                ref_flips,
                sf,
                trend_fraction_required
            );

            std::cout << "[adaptive-cpu-only] candidate=" << task.candidate
                      << " branch=" << task.branch
                      << " round=" << task.round
                      << " N=" << refineN
                      << " found_sign_flips=" << ref_flips.size()
                      << " subflip=" << sf
                      << " class=" << cr.classification
                      << " flip=(" << cr.j << "," << cr.jp1 << ")"
                      << " left_range=(" << cr.left_start << "->" << cr.left_end << ")"
                      << " right_range=(" << cr.right_start << "->" << cr.right_end << ")"
                      << " left_dec=" << cr.left_dec_frac
                      << " left_inc=" << cr.left_inc_frac
                      << " right_dec=" << cr.right_dec_frac
                      << " right_inc=" << cr.right_inc_frac
                      << " min_abs=" << cr.min_abs_local
                      << " edge_abs=" << cr.edge_abs_scale
                      << " depth_ratio=" << cr.zero_depth_ratio
                      << " E=" << std::setprecision(17) << cr.E_linear
                      << " window=[" << task.E_left << "," << task.E_right << "]"
                      << "\n";

            if (cr.classification == "likely_zero")
            {
                add_unique_zero(finalized_zeros, cr.E_linear, zero_duplicate_tol);
            }
            else if (cr.classification == "ambiguous")
            {
                double a = ref_pts[size_t(cr.j)].Ecm;
                double b = ref_pts[size_t(cr.jp1)].Ecm;
                if (a > b) std::swap(a, b);

                if (std::abs(b - a) >= min_window_width)
                {
                    tasks.push_back({
                        task.candidate,
                        next_branch_id++,
                        task.round + 1,
                        a,
                        b
                    });
                }
                else
                {
                    std::cout << "[adaptive-cpu-only] candidate=" << task.candidate
                              << " branch=" << task.branch
                              << " subflip=" << sf
                              << " ambiguous but exact sign-flip window too small; stop\n";
                }
            }
        }
    }

    std::sort(finalized_zeros.begin(), finalized_zeros.end());

    const std::string tag = nnP_tag_string(nnP_vec);
    const std::string ltag = L_tag(par.Lbyas);
    const std::string zero_file =
        "finalized_Ecm_zeros_" + tag + "_" + irrep_tag +
        "_cpu_only_v23_multi_side_abs_" + ltag + ".dat";

    {
        std::ofstream f(zero_file);
        f << std::setprecision(40);
        f << "# Lbyas\tEcm_zero\n";
        for (double z : finalized_zeros)
        {
            f << par.Lbyas << '\t' << z << "\n";
        }
    }

    std::cout << "[adaptive-cpu-only] Finalized Ecm zeros\n";
    std::cout << "# Lbyas Ecm_zero\n";
    for (double z : finalized_zeros)
    {
        std::cout << std::setprecision(17)
                  << par.Lbyas << "      " << z << "\n";
    }
    std::cout << "[adaptive-cpu-only] saved " << zero_file << "\n";
}

int main(int argc, char** argv)
{
    try
    {
        std::vector<int> nnP_vec = {0, 0, 1};
        std::string irrep = "A2";
        std::string irrep_tag = "A2";
        int coarseN = 1000;
        int refineN = 50;
        double E0 = 0.263101;
        double E1 = 0.36;
        double Lbyas = 20.0;
        char debug = 'n';
        int omp_threads = 18;

        if (argc > 1) nnP_vec[0] = std::stoi(argv[1]);
        if (argc > 2) nnP_vec[1] = std::stoi(argv[2]);
        if (argc > 3) nnP_vec[2] = std::stoi(argv[3]);
        if (argc > 4) irrep = argv[4];
        if (argc > 5) irrep_tag = argv[5];
        if (argc > 6) coarseN = std::stoi(argv[6]);
        if (argc > 7) refineN = std::stoi(argv[7]);
        if (argc > 8) E0 = std::stod(argv[8]);
        if (argc > 9) E1 = std::stod(argv[9]);
        if (argc > 10) Lbyas = std::stod(argv[10]);
        if (argc > 11) debug = argv[11][0];
        if (argc > 12) omp_threads = std::stoi(argv[12]);

        PhysicsParams par;
        par.Lbyas = Lbyas;
        par.omp_threads = omp_threads;

        // Match your trusted v9/v20 behavior for moving-frame irreps.
        par.parity = -1;

        run_adaptive_cpu_only_v23(
            nnP_vec,
            irrep,
            irrep_tag,
            coarseN,
            refineN,
            E0,
            E1,
            par,
            debug
        );

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
