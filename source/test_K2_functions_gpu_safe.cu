#include <bits/stdc++.h>
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <omp.h>

#include "spherical_functions.h"
#include "functions.h"
#include "F2_functions_v2.h"
#include "K2_functions_v2.h"
#include "K2_functions_gpu_safe.cuh"

using comp = std::complex<double>;

static inline std::string pass_fail(const k2gpu::MatrixCompareStats& st,
                                    double abs_tol,
                                    double rel_tol)
{
    if (st.n_nan_inf == 0 && st.n_bad_abs == 0 && st.n_bad_rel == 0 &&
        st.max_abs <= abs_tol && st.max_rel <= rel_tol) {
        return "PASS";
    }
    return "FAIL";
}

static void write_mismatches(
    std::ofstream& fout,
    int energy_index,
    double Ecm,
    double En,
    const Eigen::MatrixXcd& cpu,
    const Eigen::MatrixXcd& gpu,
    double abs_tol,
    double rel_tol,
    int max_print = 200)
{
    int printed = 0;
    for (int i = 0; i < cpu.rows(); ++i) {
        for (int j = 0; j < cpu.cols(); ++j) {
            const comp a = cpu(i,j);
            const comp b = gpu(i,j);
            const double adiff = std::abs(a - b);
            const double rdiff = adiff / std::max(1.0, std::abs(a));
            const bool bad = (!std::isfinite(a.real()) || !std::isfinite(a.imag()) ||
                              !std::isfinite(b.real()) || !std::isfinite(b.imag()) ||
                              adiff > abs_tol || rdiff > rel_tol);
            if (bad) {
                fout << energy_index << '\t'
                     << std::setprecision(17) << Ecm << '\t'
                     << std::setprecision(17) << En << '\t'
                     << i << '\t' << j << '\t'
                     << std::setprecision(17) << a.real() << '\t' << a.imag() << '\t'
                     << std::setprecision(17) << b.real() << '\t' << b.imag() << '\t'
                     << std::setprecision(17) << adiff << '\t'
                     << std::setprecision(17) << rdiff << '\n';
                printed++;
                if (printed >= max_print) return;
            }
        }
    }
}

int main(int argc, char** argv)
{
    double Ecm_initial = 0.26310;
    double Ecm_final   = 0.36;
    int    Ecm_points  = 50;
    int nnP0 = 0, nnP1 = 0, nnP2 = 0;
    int compare_stride = 5;

    if (argc >= 4) {
        Ecm_initial = std::stod(argv[1]);
        Ecm_final   = std::stod(argv[2]);
        Ecm_points  = std::stoi(argv[3]);
    }
    if (argc >= 7) {
        nnP0 = std::stoi(argv[4]);
        nnP1 = std::stoi(argv[5]);
        nnP2 = std::stoi(argv[6]);
    }
    if (argc >= 8) {
        compare_stride = std::max(1, std::stoi(argv[7]));
    }

    const double atmpi = 0.06906;
    const double atmK  = 0.09698;
    const double eta_1 = 1.0;
    const double eta_2 = 0.5;
    const double epsilon_h = 0.0;
    const double max_shell_num = 20.0;
    const double tolerance = 0.0;
    const double xi = 3.444;
    const double Lbyas = 20.0;
    const double L = xi * Lbyas;

    const double pi = std::acos(-1.0);
    const comp twopibyL = comp(2.0*pi/L, 0.0);

    std::vector<int> waves_vec_1 = {0, 1};
    std::vector<int> waves_vec_2 = {0};

    std::vector<comp> total_P(3);
    total_P[0] = twopibyL * double(nnP0);
    total_P[1] = twopibyL * double(nnP1);
    total_P[2] = twopibyL * double(nnP2);

    std::vector<std::vector<comp>> scatter_params_1(4, std::vector<comp>(3, comp(0.0,0.0)));
    scatter_params_1[0][0] = comp(4.04, 0.0);
    scatter_params_1[0][1] = comp(0.0, 0.0);
    scatter_params_1[0][2] = comp(0.0, 0.0);
    scatter_params_1[1][0] = comp(-43.2, 0.0);
    scatter_params_1[1][1] = comp(0.0, 0.0);
    scatter_params_1[1][2] = comp(0.0, 0.0);

    std::vector<std::vector<comp>> scatter_params_2(4, std::vector<comp>(3, comp(0.0,0.0)));
    scatter_params_2[0][0] = comp(4.12, 0.0);
    scatter_params_2[0][1] = comp(0.0, 0.0);
    scatter_params_2[0][2] = comp(0.0, 0.0);

    const double abs_tol = 1.0e-10;
    const double rel_tol = 1.0e-10;

    k2gpu::Options opt;
    opt.threads_per_block = 256;
    opt.debug = 'n';

    std::ofstream summary("K2_functions_gpu_safe_compare_summary.dat");
    std::ofstream mism("K2_functions_gpu_safe_mismatches.dat");

    summary << "# i\tEcm\tEn\tdim1\tdim2\ttotal_dim\tcpu_sec\tgpu_sec\tmax_abs\tmax_rel\tfrob_abs\tfrob_rel\tmax_i\tmax_j\tn_bad_abs\tn_bad_rel\tn_nan_inf\tstatus\n";
    mism << "# i\tEcm\tEn\trow\tcol\tcpu_re\tcpu_im\tgpu_re\tgpu_im\tabs_diff\trel_diff\n";

    std::cout << "# K2_functions_gpu_safe test\n";
    std::cout << "# Ecm_initial=" << std::setprecision(17) << Ecm_initial
              << " Ecm_final=" << Ecm_final
              << " Ecm_points=" << Ecm_points
              << " nnP=[" << nnP0 << "," << nnP1 << "," << nnP2 << "]"
              << " compare_stride=" << compare_stride << "\n";

    int n_tested = 0;
    int n_pass = 0;
    int n_fail = 0;

    const double del_Ecm = (Ecm_points <= 1) ? 0.0 : std::abs(Ecm_final - Ecm_initial) / double(Ecm_points);

    for (int i = 0; i < Ecm_points; ++i) {
        if (i % compare_stride != 0) continue;

        const double Ecm = Ecm_initial + double(i) * del_Ecm;
        const double En = Ecm_to_E(comp(Ecm,0.0), total_P).real();

        std::vector<std::vector<comp>> plm_config(5), klm_config(5);
        std::vector<std::vector<int>> np_config(5), nk_config(5);

        config_maker_4(plm_config, np_config, waves_vec_1, comp(En,0.0), total_P,
                       atmK, atmK, atmpi, L, epsilon_h, max_shell_num, tolerance);
        config_maker_4(klm_config, nk_config, waves_vec_2, comp(En,0.0), total_P,
                       atmpi, atmK, atmK, L, epsilon_h, max_shell_num, tolerance);

        const int dim1 = static_cast<int>(plm_config[0].size());
        const int dim2 = static_cast<int>(klm_config[0].size());
        const int total_dim = dim1 + dim2;

        Eigen::MatrixXcd K2_cpu;
        Eigen::MatrixXcd K2_gpu;

        const double t0 = omp_get_wtime();
        K2inv_EREord2_2plus1_mat(K2_cpu,
                                 eta_1, eta_2,
                                 scatter_params_1, scatter_params_2,
                                 comp(En,0.0),
                                 plm_config, klm_config,
                                 total_P,
                                 atmK, atmpi,
                                 epsilon_h, L);
        const double t1 = omp_get_wtime();

        const double t2 = omp_get_wtime();
        k2gpu::K2inv_EREord2_2plus1_mat_gpu_safe(K2_gpu,
                                                 eta_1, eta_2,
                                                 scatter_params_1, scatter_params_2,
                                                 comp(En,0.0),
                                                 plm_config, klm_config,
                                                 total_P,
                                                 atmK, atmpi,
                                                 epsilon_h, L,
                                                 opt);
        const double t3 = omp_get_wtime();

        const auto st = k2gpu::compare_matrices(K2_cpu, K2_gpu, abs_tol, rel_tol);
        const std::string status = pass_fail(st, abs_tol, rel_tol);

        n_tested++;
        if (status == "PASS") n_pass++;
        else {
            n_fail++;
            write_mismatches(mism, i, Ecm, En, K2_cpu, K2_gpu, abs_tol, rel_tol);
        }

        summary << i << '\t'
                << std::setprecision(17) << Ecm << '\t'
                << std::setprecision(17) << En << '\t'
                << dim1 << '\t' << dim2 << '\t' << total_dim << '\t'
                << std::setprecision(17) << (t1 - t0) << '\t'
                << std::setprecision(17) << (t3 - t2) << '\t'
                << std::setprecision(17) << st.max_abs << '\t'
                << std::setprecision(17) << st.max_rel << '\t'
                << std::setprecision(17) << st.frob_abs << '\t'
                << std::setprecision(17) << st.frob_rel << '\t'
                << st.max_i << '\t' << st.max_j << '\t'
                << st.n_bad_abs << '\t'
                << st.n_bad_rel << '\t'
                << st.n_nan_inf << '\t'
                << status << '\n';

        std::cout << "i=" << i
                  << " Ecm=" << std::setprecision(17) << Ecm
                  << " En=" << En
                  << " dims=(" << dim1 << "," << dim2 << ")"
                  << " cpu_sec=" << std::setprecision(17) << (t1 - t0)
                  << " gpu_sec=" << std::setprecision(17) << (t3 - t2)
                  << " max_abs=" << std::setprecision(17) << st.max_abs
                  << " max_rel=" << std::setprecision(17) << st.max_rel
                  << " status=" << status
                  << "\n";
    }

    summary.close();
    mism.close();

    std::cout << "\nTested: " << n_tested << "\n";
    std::cout << "PASS  : " << n_pass << "\n";
    std::cout << "FAIL  : " << n_fail << "\n";
    std::cout << "Wrote : K2_functions_gpu_safe_compare_summary.dat\n";
    std::cout << "Wrote : K2_functions_gpu_safe_mismatches.dat\n";

    return (n_fail == 0) ? 0 : 1;
}
