#include <bits/stdc++.h>
#include <omp.h>

#include "functions.h"
#include "F2_functions_v2.h"
#include "K2_functions_v2.h"
#include "functions_gpu_config_maker4.cuh"
#include "K2_gpu_safe_builder.cuh"

using comp = std::complex<double>;

static inline double wall_time_sec()
{
    return omp_get_wtime();
}

static std::string status_from_compare(const k2gpu::CompareResult& c,
                                       double abs_tol,
                                       double rel_tol)
{
    if (!c.shape_match) return "SHAPE_MISMATCH";
    if (c.n_nan_inf > 0) return "NAN_INF";
    if (c.max_abs <= abs_tol || c.max_rel <= rel_tol) return "PASS";
    return "FAIL";
}

static void write_mismatch_details(std::ofstream& fout,
                                   int i_energy,
                                   double Ecm,
                                   double En,
                                   const Eigen::MatrixXcd& cpu,
                                   const Eigen::MatrixXcd& gpu,
                                   double abs_tol,
                                   double rel_tol,
                                   int max_print = 50)
{
    if (cpu.rows() != gpu.rows() || cpu.cols() != gpu.cols()) {
        fout << "# energy_i=" << i_energy << " Ecm=" << std::setprecision(17) << Ecm
             << " En=" << En << " SHAPE_MISMATCH cpu="
             << cpu.rows() << "x" << cpu.cols() << " gpu="
             << gpu.rows() << "x" << gpu.cols() << "\n";
        return;
    }

    int printed = 0;
    for (int r = 0; r < cpu.rows(); ++r) {
        for (int c = 0; c < cpu.cols(); ++c) {
            const comp a = cpu(r, c);
            const comp b = gpu(r, c);
            const double ad = std::abs(a - b);
            const double rd = ad / std::max(1.0, std::abs(a));
            if (ad > abs_tol && rd > rel_tol) {
                if (printed == 0) {
                    fout << "# energy_i=" << i_energy
                         << " Ecm=" << std::setprecision(17) << Ecm
                         << " En=" << En << "\n";
                    fout << "# row col cpu_re cpu_im gpu_re gpu_im abs_diff rel_diff\n";
                }
                fout << r << '\t' << c << '\t'
                     << std::setprecision(17)
                     << a.real() << '\t' << a.imag() << '\t'
                     << b.real() << '\t' << b.imag() << '\t'
                     << ad << '\t' << rd << '\n';
                printed++;
                if (printed >= max_print) {
                    fout << "# ... truncated mismatches for this energy ...\n";
                    return;
                }
            }
        }
    }
}

int main(int argc, char** argv)
{
    double Ecm_initial = 0.26310;
    double Ecm_final   = 0.36;
    int Ecm_points     = 50;
    int nnP0 = 0;
    int nnP1 = 0;
    int nnP2 = 0;
    int compare_stride = 5;

    if (argc >= 2) Ecm_initial = std::stod(argv[1]);
    if (argc >= 3) Ecm_final   = std::stod(argv[2]);
    if (argc >= 4) Ecm_points  = std::stoi(argv[3]);
    if (argc >= 7) {
        nnP0 = std::stoi(argv[4]);
        nnP1 = std::stoi(argv[5]);
        nnP2 = std::stoi(argv[6]);
    }
    if (argc >= 8) compare_stride = std::max(1, std::stoi(argv[7]));

    const double pi = std::acos(-1.0);
    const double atmpi = 0.06906;
    const double atmK  = 0.09698;

    const double eta_1 = 1.0;
    const double eta_2 = 0.5;

    const double epsilon_h = 0.0;
    const double xi = 3.444;
    const double Lbyas = 20.0;
    const double L = xi * Lbyas;
    const double max_shell_num = 20.0;
    const double tolerance = 1.0e-12;

    const std::vector<int> waves_vec_1 = {0, 1};
    const std::vector<int> waves_vec_2 = {0};

    const comp twopibyL = comp(2.0 * pi / L, 0.0);
    std::vector<comp> total_P(3);
    total_P[0] = twopibyL * double(nnP0);
    total_P[1] = twopibyL * double(nnP1);
    total_P[2] = twopibyL * double(nnP2);

    std::vector<std::vector<comp>> scatter_params_1(4, std::vector<comp>(3, comp(0.0, 0.0)));
    scatter_params_1[0][0] = 4.04;
    scatter_params_1[0][1] = 0.0;
    scatter_params_1[0][2] = 0.0;
    scatter_params_1[1][0] = -43.2;
    scatter_params_1[1][1] = 0.0;
    scatter_params_1[1][2] = 0.0;

    std::vector<std::vector<comp>> scatter_params_2(4, std::vector<comp>(3, comp(0.0, 0.0)));
    scatter_params_2[0][0] = 4.12;
    scatter_params_2[0][1] = 0.0;
    scatter_params_2[0][2] = 0.0;

    const double P2 = std::norm(total_P[0]) + std::norm(total_P[1]) + std::norm(total_P[2]);
    const double del_Ecm = (Ecm_points <= 1) ? 0.0 : std::abs(Ecm_final - Ecm_initial) / double(Ecm_points - 1);

    const double abs_tol = 1.0e-9;
    const double rel_tol = 1.0e-8;

    std::ofstream summary("K2_gpu_safe_compare_summary.dat");
    std::ofstream mismatch("K2_gpu_safe_mismatches.dat");

    if (!summary.is_open() || !mismatch.is_open()) {
        throw std::runtime_error("Could not open output files");
    }

    summary << "# i\tEcm\tEn\tdim1\tdim2\ttotal_dim\tcpu_sec\tgpu_sec\tmax_abs\tmax_rel\tfrob_abs\tfrob_rel\tmax_i\tmax_j\tn_bad_abs\tn_bad_rel\tn_nan_inf\tstatus\n";
    mismatch << "# K2 GPU mismatches\n";

    k2gpu::K2GpuOptions opt;
    opt.debug = 'n';
    opt.threads_per_block = 256;

    std::cout << std::setprecision(17);
    std::cout << "# K2 GPU safe builder test\n";
    std::cout << "# Ecm_initial=" << Ecm_initial
              << " Ecm_final=" << Ecm_final
              << " Ecm_points=" << Ecm_points
              << " nnP=[" << nnP0 << "," << nnP1 << "," << nnP2 << "]"
              << " compare_stride=" << compare_stride << "\n";

    int n_pass = 0;
    int n_fail = 0;
    int n_tested = 0;

    for (int i = 0; i < Ecm_points; ++i) {
        if (i % compare_stride != 0 && i != Ecm_points - 1) continue;

        const double Ecm = Ecm_initial + double(i) * del_Ecm;
        const double En = std::sqrt(Ecm * Ecm + P2);

        std::vector<std::vector<comp>> plm_config(5), klm_config(5);
        std::vector<std::vector<int>> np_config(5), nk_config(5);

        config_maker_4(plm_config, np_config, waves_vec_1, En, total_P,
                       atmK, atmK, atmpi, L, epsilon_h, max_shell_num, tolerance);

        config_maker_4(klm_config, nk_config, waves_vec_2, En, total_P,
                       atmpi, atmK, atmK, L, epsilon_h, max_shell_num, tolerance);

        const int dim1 = static_cast<int>(plm_config[0].size());
        const int dim2 = static_cast<int>(klm_config[0].size());
        const int total_dim = dim1 + dim2;

        Eigen::MatrixXcd K2_cpu;
        Eigen::MatrixXcd K2_gpu;

        const double t0 = wall_time_sec();
        K2inv_EREord2_2plus1_mat(K2_cpu,
                                  eta_1, eta_2,
                                  scatter_params_1, scatter_params_2,
                                  comp(En, 0.0),
                                  plm_config, klm_config,
                                  total_P,
                                  atmK, atmpi,
                                  epsilon_h,
                                  L);
        const double t1 = wall_time_sec();

        const double t2 = wall_time_sec();
        k2gpu::K2inv_EREord2_2plus1_mat_gpu_safe(K2_gpu,
                                                  eta_1, eta_2,
                                                  scatter_params_1, scatter_params_2,
                                                  comp(En, 0.0),
                                                  plm_config, klm_config,
                                                  total_P,
                                                  atmK, atmpi,
                                                  epsilon_h,
                                                  L,
                                                  opt);
        const double t3 = wall_time_sec();

        const auto cmp = k2gpu::compare_matrices(K2_cpu, K2_gpu, abs_tol, rel_tol);
        const std::string status = status_from_compare(cmp, abs_tol, rel_tol);

        if (status == "PASS") n_pass++;
        else {
            n_fail++;
            write_mismatch_details(mismatch, i, Ecm, En, K2_cpu, K2_gpu, abs_tol, rel_tol, 100);
        }
        n_tested++;

        summary << i << '\t'
                << std::setprecision(17) << Ecm << '\t'
                << En << '\t'
                << dim1 << '\t'
                << dim2 << '\t'
                << total_dim << '\t'
                << (t1 - t0) << '\t'
                << (t3 - t2) << '\t'
                << cmp.max_abs << '\t'
                << cmp.max_rel << '\t'
                << cmp.frob_abs << '\t'
                << cmp.frob_rel << '\t'
                << cmp.max_i << '\t'
                << cmp.max_j << '\t'
                << cmp.n_bad_abs << '\t'
                << cmp.n_bad_rel << '\t'
                << cmp.n_nan_inf << '\t'
                << status << '\n';

        std::cout << "i=" << i
                  << " Ecm=" << Ecm
                  << " En=" << En
                  << " dims=(" << dim1 << "," << dim2 << ")"
                  << " cpu_sec=" << (t1 - t0)
                  << " gpu_sec=" << (t3 - t2)
                  << " max_abs=" << cmp.max_abs
                  << " max_rel=" << cmp.max_rel
                  << " status=" << status
                  << "\n";
    }

    summary.close();
    mismatch.close();

    std::cout << "\nDone. tested=" << n_tested
              << " pass=" << n_pass
              << " fail=" << n_fail << "\n";
    std::cout << "Wrote K2_gpu_safe_compare_summary.dat\n";
    std::cout << "Wrote K2_gpu_safe_mismatches.dat\n";

    return (n_fail == 0) ? 0 : 1;
}
