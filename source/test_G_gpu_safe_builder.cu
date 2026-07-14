#include <bits/stdc++.h>
#include <Eigen/Dense>
#include <omp.h>

#include "spherical_functions.h"
#include "functions.h"
#include "F2_functions_v2.h"
#include "K2_functions_v2.h"
#include "G_functions_v2.h"
#include "G_gpu_safe_builder.cuh"

using comp = std::complex<double>;

static inline std::string status_from_stats(const ggpu::MatrixCompareStats& s)
{
    if (s.n_nan_inf > 0) return "NAN_INF";
    if (s.n_bad_abs == 0 && s.n_bad_rel == 0) return "PASS";
    return "FAIL";
}

static void write_mismatch_entries(
    std::ofstream& fout,
    int eidx,
    double Ecm,
    double En,
    const Eigen::MatrixXcd& A,
    const Eigen::MatrixXcd& B,
    double abs_tol,
    double rel_tol,
    int max_print)
{
    int printed = 0;
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            const comp a = A(i,j);
            const comp b = B(i,j);
            const double d = std::abs(a - b);
            const double r = d / std::max(1.0, std::abs(a));
            if (d > abs_tol || r > rel_tol ||
                !std::isfinite(a.real()) || !std::isfinite(a.imag()) ||
                !std::isfinite(b.real()) || !std::isfinite(b.imag())) {
                fout << eidx << '\t' << std::setprecision(17) << Ecm << '\t' << En << '\t'
                     << i << '\t' << j << '\t'
                     << A.rows() << '\t' << A.cols() << '\t'
                     << std::setprecision(17)
                     << a.real() << '\t' << a.imag() << '\t'
                     << b.real() << '\t' << b.imag() << '\t'
                     << d << '\t' << r << '\n';
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
    int Ecm_points     = 20;
    int nnP0 = 0, nnP1 = 0, nnP2 = 0;
    int compare_stride = 1;

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
    const double alpha = 0.5;
    const int max_shell_num = 20;
    const double epsilon_h = 0.0;
    const double tolerance = 1.0e-12;
    const bool Q0norm = true;
    const double xi = 3.444;
    const double Lbyas = 20.0;
    const double L = xi * Lbyas;

    std::vector<int> waves_vec_1 = {0, 1};
    std::vector<int> waves_vec_2 = {0};

    const comp pi = std::acos(-1.0);
    const comp twopibyL = ((comp)2.0) * pi / ((comp)L);

    std::vector<comp> total_P(3);
    total_P[0] = twopibyL * (double)nnP0;
    total_P[1] = twopibyL * (double)nnP1;
    total_P[2] = twopibyL * (double)nnP2;

    const comp P2 = total_P[0]*total_P[0] + total_P[1]*total_P[1] + total_P[2]*total_P[2];
    const double del_Ecm = (Ecm_points <= 1) ? 0.0 : (Ecm_final - Ecm_initial) / double(Ecm_points - 1);

    ggpu::GGpuOptions gopt;
    gopt.debug = 'n';
    gopt.threads_per_block = 256;

    const double abs_tol = 1.0e-9;
    const double rel_tol = 1.0e-7;

    std::ofstream summary("G_gpu_safe_compare_summary.dat");
    std::ofstream mism("G_gpu_safe_mismatches.dat");

    summary << "# i\tEcm\tEn\tdim1\tdim2\ttotal_dim\tcpu_sec\tgpu_sec\tmax_abs\tmax_rel\tfrob_abs\tfrob_rel\tmax_i\tmax_j\tn_bad_abs\tn_bad_rel\tn_nan_inf\tstatus\n";
    mism << "# i\tEcm\tEn\trow\tcol\trows\tcols\tcpu_re\tcpu_im\tgpu_re\tgpu_im\tabs_diff\trel_diff\n";

    std::cout << std::setprecision(17);
    std::cout << "Testing G GPU-safe builder\n";
    std::cout << "Ecm range: " << Ecm_initial << " to " << Ecm_final
              << "  points=" << Ecm_points << "  nnP=[" << nnP0 << "," << nnP1 << "," << nnP2 << "]\n";

    int n_pass = 0;
    int n_fail = 0;

    for (int i = 0; i < Ecm_points; ++i) {
        if (i % compare_stride != 0) continue;

        const double Ecm = Ecm_initial + double(i) * del_Ecm;
        const double En = std::sqrt(Ecm*Ecm + P2.real());
        const comp En_c(En, 0.0);

        std::vector<std::vector<comp>> plm_config(5), klm_config(5);
        std::vector<std::vector<int>> np_config(5), nk_config(5);

        config_maker_4(plm_config, np_config, waves_vec_1, En_c, total_P,
                       atmK, atmK, atmpi, L, epsilon_h, max_shell_num, tolerance);
        config_maker_4(klm_config, nk_config, waves_vec_2, En_c, total_P,
                       atmpi, atmK, atmK, L, epsilon_h, max_shell_num, tolerance);

        const int dim1 = static_cast<int>(plm_config[0].size());
        const int dim2 = static_cast<int>(klm_config[0].size());
        const int total_dim = dim1 + dim2;

        Eigen::MatrixXcd G_cpu;
        Eigen::MatrixXcd G_gpu;

        const auto t0 = std::chrono::high_resolution_clock::now();
        G_2plus1_mat(G_cpu, En_c, plm_config, klm_config, total_P,
                     atmK, atmpi, L, alpha, epsilon_h, max_shell_num, Q0norm);
        const auto t1 = std::chrono::high_resolution_clock::now();

        ggpu::G_2plus1_mat_gpu_safe(G_gpu, En_c, plm_config, klm_config, total_P,
                                    atmK, atmpi, L, alpha, epsilon_h, max_shell_num, Q0norm, gopt);
        const auto t2 = std::chrono::high_resolution_clock::now();

        const double cpu_sec = std::chrono::duration<double>(t1 - t0).count();
        const double gpu_sec = std::chrono::duration<double>(t2 - t1).count();

        ggpu::MatrixCompareStats st = ggpu::compare_matrices(G_cpu, G_gpu, abs_tol, rel_tol);
        const std::string status = status_from_stats(st);

        if (status == "PASS") n_pass++; else n_fail++;

        summary << i << '\t' << std::setprecision(17)
                << Ecm << '\t' << En << '\t'
                << dim1 << '\t' << dim2 << '\t' << total_dim << '\t'
                << cpu_sec << '\t' << gpu_sec << '\t'
                << st.max_abs << '\t' << st.max_rel << '\t'
                << st.frob_abs << '\t' << st.frob_rel << '\t'
                << st.max_i << '\t' << st.max_j << '\t'
                << st.n_bad_abs << '\t' << st.n_bad_rel << '\t' << st.n_nan_inf << '\t'
                << status << '\n';

        if (status != "PASS") {
            write_mismatch_entries(mism, i, Ecm, En, G_cpu, G_gpu, abs_tol, rel_tol, 200);
        }

        std::cout << "i=" << i
                  << " Ecm=" << Ecm
                  << " dim=(" << dim1 << "+" << dim2 << ")=" << total_dim
                  << " cpu=" << cpu_sec << "s"
                  << " gpu=" << gpu_sec << "s"
                  << " max_abs=" << st.max_abs
                  << " max_rel=" << st.max_rel
                  << " status=" << status << "\n";
    }

    summary.close();
    mism.close();

    std::cout << "\nPASS=" << n_pass << " FAIL=" << n_fail << "\n";
    std::cout << "Wrote G_gpu_safe_compare_summary.dat\n";
    std::cout << "Wrote G_gpu_safe_mismatches.dat\n";

    return (n_fail == 0) ? 0 : 2;
}
