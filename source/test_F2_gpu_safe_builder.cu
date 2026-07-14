// ============================================================================
// test_F2_gpu_safe_builder.cu
//
// Standalone validation test for F2_gpu_safe_builder.cuh.
//
// What it does:
//   1. Builds plm/klm configs using your CPU config_maker_4.
//   2. Builds F2 using your CPU F2_2plus1_mat.
//   3. Builds F2 using f2gpu::F2_2plus1_mat_gpu_safe.
//   4. Compares element-by-element.
//   5. Writes summary and detailed mismatch files.
//
// Expected source tree placement:
//   ./source/test_F2_gpu_safe_builder.cu
//   ./source/F2_gpu_safe_builder.cuh
//   ./source/functions.h
//   ./source/F2_functions_v2.h
//   ./source/Faddeeva.cc
//   ./source/Faddeeva.hh
//   ./source/spherical_functions.h
//
// Compile using the accompanying compile_test_F2_gpu_safe_builder.sh.
// ============================================================================

#include <cuda_runtime.h>
#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "functions.h"
#include "F2_functions_v2.h"
#include "F2_gpu_safe_builder.cuh"

using comp = std::complex<double>;

struct TestParams
{
    double atmpi = 0.06906;
    double atmK  = 0.09698;

    double alpha = 0.5;
    int max_shell_num = 20;
    double epsilon_h = 0.0;
    double tolerance = 1.0e-12;
    bool Q0norm = true;

    double xi = 3.444;
    double Lbyas = 20.0;

    int nnP0 = 0;
    int nnP1 = 0;
    int nnP2 = 0;

    double Ecm_initial = 0.26310;
    double Ecm_final   = 0.36;
    int Ecm_points = 10;

    // Compare every compare_stride point. Set 1 for every point.
    int compare_stride = 1;

    // Absolute tolerance used for summary pass/fail.
    double abs_tol = 1.0e-7;
    double rel_tol = 1.0e-6;

    char debug = 'n';

    // GPU approximation controls.
    int erfi_max_terms = 120;
    double erfi_tol = 1.0e-15;
};

static inline double wall_seconds_since(
    const std::chrono::high_resolution_clock::time_point& t0)
{
    const auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

static inline std::vector<comp> make_total_P(const TestParams& p)
{
    const comp pi = std::acos(-1.0);
    const double L = p.xi * p.Lbyas;
    const comp twopibyL = ((comp)2.0) * pi / ((comp)L);

    std::vector<comp> total_P(3);
    total_P[0] = twopibyL * static_cast<double>(p.nnP0);
    total_P[1] = twopibyL * static_cast<double>(p.nnP1);
    total_P[2] = twopibyL * static_cast<double>(p.nnP2);
    return total_P;
}

struct MatrixCompareStats
{
    bool shape_match = true;
    int rows = 0;
    int cols = 0;
    int gpu_rows = 0;
    int gpu_cols = 0;

    double max_abs = 0.0;
    double max_rel = 0.0;
    double frob_abs = 0.0;
    double frob_cpu = 0.0;
    double frob_rel = 0.0;

    int max_i = -1;
    int max_j = -1;
    int n_bad_abs = 0;
    int n_bad_rel = 0;
    int n_nan_inf = 0;
};

static MatrixCompareStats compare_matrices(
    const Eigen::MatrixXcd& A,
    const Eigen::MatrixXcd& B,
    double abs_tol,
    double rel_tol,
    std::ofstream* mismatch_out,
    int energy_index,
    double Ecm,
    double En)
{
    MatrixCompareStats s;
    s.rows = A.rows();
    s.cols = A.cols();
    s.gpu_rows = B.rows();
    s.gpu_cols = B.cols();

    if (A.rows() != B.rows() || A.cols() != B.cols())
    {
        s.shape_match = false;
        return s;
    }

    long double frob_diff_sq = 0.0L;
    long double frob_cpu_sq = 0.0L;

    for (int i = 0; i < A.rows(); ++i)
    {
        for (int j = 0; j < A.cols(); ++j)
        {
            const comp a = A(i, j);
            const comp b = B(i, j);
            const comp d = a - b;

            const bool bad =
                !std::isfinite(a.real()) || !std::isfinite(a.imag()) ||
                !std::isfinite(b.real()) || !std::isfinite(b.imag());

            if (bad)
            {
                s.n_nan_inf += 1;
            }

            const double ad = std::abs(d);
            const double denom = std::max(1.0, std::abs(a));
            const double rd = ad / denom;

            frob_diff_sq += static_cast<long double>(ad) * static_cast<long double>(ad);
            frob_cpu_sq += static_cast<long double>(std::abs(a)) * static_cast<long double>(std::abs(a));

            if (ad > s.max_abs)
            {
                s.max_abs = ad;
                s.max_rel = rd;
                s.max_i = i;
                s.max_j = j;
            }

            if (ad > abs_tol)
            {
                s.n_bad_abs += 1;
            }

            if (rd > rel_tol)
            {
                s.n_bad_rel += 1;
            }

            if (mismatch_out && (ad > abs_tol || rd > rel_tol || bad))
            {
                (*mismatch_out)
                    << energy_index << '\t'
                    << std::setprecision(17) << Ecm << '\t'
                    << std::setprecision(17) << En << '\t'
                    << i << '\t' << j << '\t'
                    << std::setprecision(17) << a.real() << '\t'
                    << std::setprecision(17) << a.imag() << '\t'
                    << std::setprecision(17) << b.real() << '\t'
                    << std::setprecision(17) << b.imag() << '\t'
                    << std::setprecision(17) << d.real() << '\t'
                    << std::setprecision(17) << d.imag() << '\t'
                    << std::setprecision(17) << ad << '\t'
                    << std::setprecision(17) << rd << '\n';
            }
        }
    }

    s.frob_abs = std::sqrt(static_cast<double>(frob_diff_sq));
    s.frob_cpu = std::sqrt(static_cast<double>(frob_cpu_sq));
    s.frob_rel = s.frob_abs / std::max(1.0, s.frob_cpu);

    return s;
}

static void print_usage(const char* exe)
{
    std::cout
        << "Usage:\n"
        << "  " << exe << " [Ecm_initial Ecm_final Ecm_points nnP0 nnP1 nnP2 compare_stride]\n\n"
        << "Defaults:\n"
        << "  Ecm_initial    = 0.26310\n"
        << "  Ecm_final      = 0.36\n"
        << "  Ecm_points     = 10\n"
        << "  nnP            = 0 0 0\n"
        << "  compare_stride = 1\n\n"
        << "Example:\n"
        << "  " << exe << " 0.26310 0.36 50 0 0 0 5\n";
}

int main(int argc, char** argv)
{
    try
    {
        TestParams par;

        if (argc == 2)
        {
            const std::string arg1 = argv[1];
            if (arg1 == "-h" || arg1 == "--help")
            {
                print_usage(argv[0]);
                return 0;
            }
        }

        if (argc >= 4)
        {
            par.Ecm_initial = std::stod(argv[1]);
            par.Ecm_final   = std::stod(argv[2]);
            par.Ecm_points  = std::stoi(argv[3]);
        }

        if (argc >= 7)
        {
            par.nnP0 = std::stoi(argv[4]);
            par.nnP1 = std::stoi(argv[5]);
            par.nnP2 = std::stoi(argv[6]);
        }

        if (argc >= 8)
        {
            par.compare_stride = std::max(1, std::stoi(argv[7]));
        }

        if (par.Ecm_points <= 0)
        {
            throw std::runtime_error("Ecm_points must be positive.");
        }

        const double L = par.xi * par.Lbyas;
        const std::vector<comp> total_P = make_total_P(par);

        std::vector<int> waves_vec_1 = {0, 1};
        std::vector<int> waves_vec_2 = {0};

        f2gpu::F2GpuOptions gpu_opt;
        gpu_opt.debug = par.debug;
        gpu_opt.threads_per_block = 256;
        gpu_opt.erfi_max_terms = par.erfi_max_terms;
        gpu_opt.erfi_tol = par.erfi_tol;

        int device = 0;
        cudaDeviceProp prop{};
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&prop, device);

        size_t freeB = 0;
        size_t totalB = 0;
        cudaMemGetInfo(&freeB, &totalB);

        std::cout << std::scientific << std::setprecision(17);
        std::cout << "[test F2 GPU] device = " << device << "  " << prop.name << "\n";
        std::cout << "[test F2 GPU] free VRAM  = " << double(freeB) / (1024.0 * 1024.0) << " MiB\n";
        std::cout << "[test F2 GPU] total VRAM = " << double(totalB) / (1024.0 * 1024.0) << " MiB\n";
        std::cout << "[test F2 GPU] Ecm range = [" << par.Ecm_initial << ", " << par.Ecm_final
                  << "], points = " << par.Ecm_points << "\n";
        std::cout << "[test F2 GPU] nnP = [" << par.nnP0 << ", " << par.nnP1 << ", " << par.nnP2 << "]\n";
        std::cout << "[test F2 GPU] compare_stride = " << par.compare_stride << "\n";
        std::cout << "[test F2 GPU] abs_tol = " << par.abs_tol << " rel_tol = " << par.rel_tol << "\n";
        std::cout << "[test F2 GPU] erfi_max_terms = " << par.erfi_max_terms
                  << " erfi_tol = " << par.erfi_tol << "\n";

        std::ofstream summary("F2_gpu_safe_compare_summary.dat");
        if (!summary.is_open())
        {
            throw std::runtime_error("Could not open F2_gpu_safe_compare_summary.dat");
        }

        std::ofstream mismatch("F2_gpu_safe_mismatches.dat");
        if (!mismatch.is_open())
        {
            throw std::runtime_error("Could not open F2_gpu_safe_mismatches.dat");
        }

        summary
            << "# i\tEcm\tEn\tdim1\tdim2\ttotal_dim\t"
            << "cpu_sec\tgpu_sec\tmax_abs\tmax_rel\tfrob_abs\tfrob_rel\t"
            << "max_i\tmax_j\tn_bad_abs\tn_bad_rel\tn_nan_inf\tstatus\n";

        mismatch
            << "# i\tEcm\tEn\trow\tcol\t"
            << "cpu_re\tcpu_im\tgpu_re\tgpu_im\tdiff_re\tdiff_im\tabs_diff\trel_diff\n";

        const double del_Ecm = (par.Ecm_points == 1)
            ? 0.0
            : (par.Ecm_final - par.Ecm_initial) / double(par.Ecm_points - 1);

        int tested = 0;
        int passed = 0;
        int failed = 0;
        int shape_failed = 0;

        for (int i = 0; i < par.Ecm_points; ++i)
        {
            if (i % par.compare_stride != 0 && i != par.Ecm_points - 1)
            {
                continue;
            }

            const double Ecm = par.Ecm_initial + double(i) * del_Ecm;
            const double En = Ecm_to_E(comp(Ecm, 0.0), total_P).real();

            std::vector<std::vector<comp>> plm_config(5), klm_config(5);
            std::vector<std::vector<int>> np_config(5), nk_config(5);

            config_maker_4(
                plm_config,
                np_config,
                waves_vec_1,
                comp(En, 0.0),
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
                waves_vec_2,
                comp(En, 0.0),
                total_P,
                par.atmpi,
                par.atmK,
                par.atmK,
                L,
                par.epsilon_h,
                par.max_shell_num,
                par.tolerance
            );

            const int dim1 = static_cast<int>(plm_config[0].size());
            const int dim2 = static_cast<int>(klm_config[0].size());
            const int total_dim = dim1 + dim2;

            Eigen::MatrixXcd F2_cpu;
            Eigen::MatrixXcd F2_gpu;

            const auto t_cpu = std::chrono::high_resolution_clock::now();
            F2_2plus1_mat(
                F2_cpu,
                comp(En, 0.0),
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
            const double cpu_sec = wall_seconds_since(t_cpu);

            const auto t_gpu = std::chrono::high_resolution_clock::now();
            f2gpu::F2_2plus1_mat_gpu_safe(
                F2_gpu,
                comp(En, 0.0),
                plm_config,
                klm_config,
                total_P,
                par.atmK,
                par.atmpi,
                L,
                par.alpha,
                par.epsilon_h,
                par.max_shell_num,
                par.Q0norm,
                gpu_opt
            );
            const double gpu_sec = wall_seconds_since(t_gpu);

            MatrixCompareStats stats = compare_matrices(
                F2_cpu,
                F2_gpu,
                par.abs_tol,
                par.rel_tol,
                &mismatch,
                i,
                Ecm,
                En
            );

            std::string status = "PASS";
            if (!stats.shape_match)
            {
                status = "SHAPE_FAIL";
                shape_failed += 1;
                failed += 1;
            }
            else if (stats.max_abs > par.abs_tol && stats.max_rel > par.rel_tol)
            {
                status = "FAIL";
                failed += 1;
            }
            else
            {
                passed += 1;
            }

            tested += 1;

            summary
                << i << '\t'
                << std::setprecision(17) << Ecm << '\t'
                << std::setprecision(17) << En << '\t'
                << dim1 << '\t'
                << dim2 << '\t'
                << total_dim << '\t'
                << std::setprecision(17) << cpu_sec << '\t'
                << std::setprecision(17) << gpu_sec << '\t'
                << std::setprecision(17) << stats.max_abs << '\t'
                << std::setprecision(17) << stats.max_rel << '\t'
                << std::setprecision(17) << stats.frob_abs << '\t'
                << std::setprecision(17) << stats.frob_rel << '\t'
                << stats.max_i << '\t'
                << stats.max_j << '\t'
                << stats.n_bad_abs << '\t'
                << stats.n_bad_rel << '\t'
                << stats.n_nan_inf << '\t'
                << status << '\n';

            std::cout
                << "[" << status << "] i=" << i
                << " Ecm=" << std::setprecision(12) << Ecm
                << " En=" << std::setprecision(12) << En
                << " dim1=" << dim1
                << " dim2=" << dim2
                << " total=" << total_dim
                << " cpu_sec=" << std::setprecision(6) << cpu_sec
                << " gpu_sec=" << std::setprecision(6) << gpu_sec
                << " max_abs=" << std::scientific << std::setprecision(3) << stats.max_abs
                << " max_rel=" << std::scientific << std::setprecision(3) << stats.max_rel
                << " bad_abs=" << stats.n_bad_abs
                << " bad_rel=" << stats.n_bad_rel
                << " max_at=(" << stats.max_i << "," << stats.max_j << ")"
                << std::defaultfloat << '\n';
        }

        summary.close();
        mismatch.close();

        std::cout << "\n[test F2 GPU] tested = " << tested
                  << " passed = " << passed
                  << " failed = " << failed
                  << " shape_failed = " << shape_failed << "\n";
        std::cout << "[test F2 GPU] wrote F2_gpu_safe_compare_summary.dat\n";
        std::cout << "[test F2 GPU] wrote F2_gpu_safe_mismatches.dat\n";

        return (failed == 0) ? 0 : 2;
    }
    catch (const std::exception& e)
    {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
