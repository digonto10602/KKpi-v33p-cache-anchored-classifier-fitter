#include <bits/stdc++.h>
#include <Eigen/Dense>
#include <cuda_runtime.h>

#include "spherical_functions.h"
#include "functions.h"
#include "F2_functions_v2.h"
#include "K2_functions_v2.h"
#include "G_functions_v2.h"
#include "QC_functions_v2.h"
#include "dig_tools.hpp"
#include "projections_v1.hpp"
#include "projections_gpu_safe.cuh"

using comp = std::complex<double>;

struct CompareStats
{
    double max_abs = 0.0;
    double max_rel = 0.0;
    double frob_abs = 0.0;
    double frob_ref = 0.0;
    int max_i = -1;
    int max_j = -1;
    int n_bad = 0;
};

CompareStats compare_matrix(const Eigen::MatrixXcd& A, const Eigen::MatrixXcd& B, double atol, double rtol)
{
    CompareStats s;
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        s.n_bad = -1;
        return s;
    }
    double sum2 = 0.0, ref2 = 0.0;
    for (int i=0;i<A.rows();++i) {
        for (int j=0;j<A.cols();++j) {
            double err = std::abs(A(i,j)-B(i,j));
            double ref = std::max(1.0, std::abs(A(i,j)));
            double rel = err/ref;
            sum2 += err*err;
            ref2 += std::norm(A(i,j));
            if (err > s.max_abs) { s.max_abs = err; s.max_rel = rel; s.max_i=i; s.max_j=j; }
            if (err > atol && rel > rtol) s.n_bad++;
        }
    }
    s.frob_abs = std::sqrt(sum2);
    s.frob_ref = std::sqrt(ref2);
    return s;
}

int main(int argc, char** argv)
{
    double Ecm_initial = 0.26310;
    double Ecm_final   = 0.36;
    int Ecm_points     = 20;
    int nnP0 = 0, nnP1 = 0, nnP2 = 0;
    int compare_stride = 1;
    std::string irrep = "A1u";

    if (argc > 1) Ecm_initial = std::stod(argv[1]);
    if (argc > 2) Ecm_final   = std::stod(argv[2]);
    if (argc > 3) Ecm_points  = std::stoi(argv[3]);
    if (argc > 4) nnP0 = std::stoi(argv[4]);
    if (argc > 5) nnP1 = std::stoi(argv[5]);
    if (argc > 6) nnP2 = std::stoi(argv[6]);
    if (argc > 7) compare_stride = std::stoi(argv[7]);
    if (argc > 8) irrep = argv[8];

    const double atmpi = 0.06906;
    const double atmK  = 0.09698;
    const double xi = 3.444;
    const double Lbyas = 20.0;
    const double L = xi * Lbyas;
    const double epsilon_h = 0.0;
    const double max_shell_num = 20.0;
    const double tolerance = 0.0;
    const int parity = -1;
    const bool sort_orbit_flag = false;

    std::vector<int> waves_vec_1 = {0,1};
    std::vector<int> waves_vec_2 = {0};

    comp pi = std::acos(-1.0);
    comp twopibyL = ((comp)2.0*pi)/((comp)L);
    std::vector<comp> total_P(3);
    total_P[0] = twopibyL * (double)nnP0;
    total_P[1] = twopibyL * (double)nnP1;
    total_P[2] = twopibyL * (double)nnP2;
    std::vector<comp> nnP_config = {(comp)nnP0, (comp)nnP1, (comp)nnP2};

    double del_Ecm = (Ecm_points == 1) ? 0.0 : std::abs(Ecm_final - Ecm_initial) / double(Ecm_points - 1);

    std::ofstream sum("projections_gpu_safe_compare_summary.dat");
    sum << "# i\tEcm\tEn\tdim1\tdim2\ttotal_dim\tP_max_abs\tP_max_rel\tP_bad\tV_cpu_cols\tV_gpu_cols\tPproj_max_abs\tPproj_max_rel\tPproj_bad\tstatus\n";

    std::cout << "# projections GPU-safe test\n";
    std::cout << "# range " << Ecm_initial << " " << Ecm_final << " points=" << Ecm_points
              << " nnP=[" << nnP0 << "," << nnP1 << "," << nnP2 << "] irrep=" << irrep
              << " stride=" << compare_stride << "\n";

    pgpu::Options opt;
    opt.debug = 'n';
    opt.eig_tol = 0.05;
    opt.chop_tol = 1.0e-13;

    int n_fail = 0;
    int n_test = 0;

    for (int i=0; i<Ecm_points; i += compare_stride) {
        double Ecm = Ecm_initial + i * del_Ecm;
        double En = Ecm_to_E((comp)Ecm, total_P).real();

        std::vector<std::vector<comp>> plm_config(5), klm_config(5);
        std::vector<std::vector<int>> np_config(5), nk_config(5);

        config_maker_4(plm_config, np_config, waves_vec_1, (comp)En, total_P, atmK, atmK, atmpi, L, epsilon_h, max_shell_num, tolerance);
        config_maker_4(klm_config, nk_config, waves_vec_2, (comp)En, total_P, atmpi, atmK, atmK, L, epsilon_h, max_shell_num, tolerance);

        int dim1 = (int)np_config[0].size();
        int dim2 = (int)nk_config[0].size();
        int N = dim1 + dim2;
        if (N == 0) continue;

        Eigen::MatrixXcd P_cpu, V_cpu, Pproj_cpu;
        P_irrep_projection_2plus1(P_cpu, plm_config, np_config, klm_config, nk_config, irrep, total_P, nnP_config, sort_orbit_flag, parity);
        build_projector_from_eigenvectors_near_one(P_cpu, V_cpu, Pproj_cpu, 0.05, 1.0e-12, 1.0e-10, 'n');

        Eigen::MatrixXcd P_gpu, V_gpu, Pproj_gpu;
        pgpu::P_irrep_projection_2plus1_and_Vsel_gpu_safe(P_gpu, V_gpu, Pproj_gpu, np_config, nk_config, irrep, nnP0,nnP1,nnP2, sort_orbit_flag, parity, opt);

        CompareStats pstat = compare_matrix(P_cpu, P_gpu, 1.0e-10, 1.0e-10);
        CompareStats qstat = compare_matrix(Pproj_cpu, Pproj_gpu, 1.0e-8, 1.0e-8);

        bool pass = (pstat.n_bad == 0 && qstat.n_bad == 0 && V_cpu.cols() == V_gpu.cols());
        if (!pass) n_fail++;
        n_test++;

        std::cout << std::setprecision(17)
                  << "i=" << i << " Ecm=" << Ecm << " En=" << En
                  << " dims=(" << dim1 << "," << dim2 << ")"
                  << " P_max_abs=" << pstat.max_abs
                  << " Pproj_max_abs=" << qstat.max_abs
                  << " Vcols cpu/gpu=" << V_cpu.cols() << "/" << V_gpu.cols()
                  << " status=" << (pass ? "PASS" : "FAIL") << "\n";

        sum << i << '\t' << std::setprecision(17) << Ecm << '\t' << En << '\t'
            << dim1 << '\t' << dim2 << '\t' << N << '\t'
            << pstat.max_abs << '\t' << pstat.max_rel << '\t' << pstat.n_bad << '\t'
            << V_cpu.cols() << '\t' << V_gpu.cols() << '\t'
            << qstat.max_abs << '\t' << qstat.max_rel << '\t' << qstat.n_bad << '\t'
            << (pass ? "PASS" : "FAIL") << '\n';
    }

    sum.close();
    std::cout << "Tested " << n_test << " points. failures=" << n_fail << "\n";
    std::cout << "Wrote projections_gpu_safe_compare_summary.dat\n";
    return (n_fail == 0) ? 0 : 2;
}
