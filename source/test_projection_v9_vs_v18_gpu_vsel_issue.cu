// test_projection_v9_vs_v18_gpu_vsel_issue.cu
//
// Purpose:
//   Diagnostic comparison for the moving-frame projection/Vsel issue.
//   This does NOT build F2/G/K2/F3. It only compares:
//       CPU/v9 projection path: projections_v1.hpp + Eigen
//       GPU/v18 projection path: projections_gpu_safe.cuh + cuSOLVER
//
// Typical failing case:
//   nnP = [0,0,1], irrep = A2
//
// Compile with:
//   bash compile_test_projection_v9_vs_v18_gpu_vsel_issue.sh perf
//
// Run examples:
//   ./test_projection_v9_vs_v18_gpu_vsel_issue 0.263101 0.36 2000 0 0 1 A2 1 n
//   ./test_projection_v9_vs_v18_gpu_vsel_issue 0.263101 0.27 50   0 0 1 A2 1 y
//
// Args:
//   Ecm_initial Ecm_final Ecm_points nnP0 nnP1 nnP2 irrep stride debug
//
// Output files:
//   projection_v9_vs_v18_summary.dat
//   projection_v9_vs_v18_failures.dat
//   projection_v9_vs_v18_eigenvalues.dat
//   projection_v9_vs_v18_config_shapes.dat

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
    long long n_bad = 0;
    bool shape_mismatch = false;
};

static CompareStats compare_matrix(
    const Eigen::MatrixXcd& A,
    const Eigen::MatrixXcd& B,
    double atol,
    double rtol)
{
    CompareStats s;

    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        s.shape_mismatch = true;
        s.n_bad = -1;
        return s;
    }

    double sum2 = 0.0;
    double ref2 = 0.0;

    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            const double err = std::abs(A(i,j) - B(i,j));
            const double ref = std::max(1.0, std::abs(A(i,j)));
            const double rel = err / ref;

            sum2 += err * err;
            ref2 += std::norm(A(i,j));

            if (err > s.max_abs) {
                s.max_abs = err;
                s.max_rel = rel;
                s.max_i = i;
                s.max_j = j;
            }

            if (err > atol && rel > rtol) {
                s.n_bad++;
            }
        }
    }

    s.frob_abs = std::sqrt(sum2);
    s.frob_ref = std::sqrt(ref2);

    return s;
}

struct EigenSummary
{
    int n = 0;
    int count_near_one_005 = 0;
    int count_near_one_1e8 = 0;
    int count_gt_half = 0;
    double trace_real = 0.0;
    double eval_min = 0.0;
    double eval_max = 0.0;
    double closest_to_one = 1.0e300;
    double closest_eval_to_one = 0.0;
    double idempotency_error = 0.0;
    std::vector<double> evals;
};

static EigenSummary summarize_projector(const Eigen::MatrixXcd& P)
{
    EigenSummary s;
    const int N = static_cast<int>(P.rows());
    s.n = N;

    if (N == 0 || P.cols() != N) {
        return s;
    }

    s.trace_real = P.trace().real();

    Eigen::MatrixXcd Ph = 0.5 * (P + P.adjoint());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(Ph);

    if (es.info() != Eigen::Success) {
        s.eval_min = std::numeric_limits<double>::quiet_NaN();
        s.eval_max = std::numeric_limits<double>::quiet_NaN();
        return s;
    }

    s.evals.resize(N);
    for (int i = 0; i < N; ++i) {
        const double ev = es.eigenvalues()(i);
        s.evals[i] = ev;

        if (i == 0 || ev < s.eval_min) s.eval_min = ev;
        if (i == 0 || ev > s.eval_max) s.eval_max = ev;

        if (std::abs(ev - 1.0) <= 0.05) s.count_near_one_005++;
        if (std::abs(ev - 1.0) <= 1.0e-8) s.count_near_one_1e8++;
        if (ev > 0.5) s.count_gt_half++;

        const double d = std::abs(ev - 1.0);
        if (d < s.closest_to_one) {
            s.closest_to_one = d;
            s.closest_eval_to_one = ev;
        }
    }

    Eigen::MatrixXcd err = P * P - P;
    s.idempotency_error = err.norm();

    return s;
}

static std::string last_eigs_string(const std::vector<double>& evals, int nlast = 12)
{
    if (evals.empty()) return "";
    std::ostringstream os;
    os << std::setprecision(17);

    const int N = static_cast<int>(evals.size());
    const int start = std::max(0, N - nlast);

    for (int i = start; i < N; ++i) {
        if (i > start) os << ",";
        os << evals[i];
    }

    return os.str();
}

static std::string first_eigs_string(const std::vector<double>& evals, int nfirst = 8)
{
    if (evals.empty()) return "";
    std::ostringstream os;
    os << std::setprecision(17);

    const int N = static_cast<int>(evals.size());
    const int stop = std::min(N, nfirst);

    for (int i = 0; i < stop; ++i) {
        if (i > 0) os << ",";
        os << evals[i];
    }

    return os.str();
}

static void print_config_sample(
    std::ofstream& out,
    int i,
    double Ecm,
    const std::string& label,
    const std::vector<std::vector<int>>& cfg,
    int max_rows = 12)
{
    const int n = static_cast<int>(cfg[0].size());
    out << "# sample " << label << " i=" << i
        << " Ecm=" << std::setprecision(17) << Ecm
        << " n=" << n << "\n";
    out << "# row nx ny nz ell m\n";

    for (int r = 0; r < std::min(n, max_rows); ++r) {
        out << r << " "
            << cfg[0][r] << " "
            << cfg[1][r] << " "
            << cfg[2][r] << " "
            << cfg[3][r] << " "
            << cfg[4][r] << "\n";
    }
}

int main(int argc, char** argv)
{
    double Ecm_initial = 0.263101;
    double Ecm_final   = 0.36;
    int Ecm_points     = 2000;
    int nnP0 = 0;
    int nnP1 = 0;
    int nnP2 = 1;
    std::string irrep = "A2";
    int compare_stride = 1;
    char debug = 'n';

    if (argc > 1) Ecm_initial = std::stod(argv[1]);
    if (argc > 2) Ecm_final   = std::stod(argv[2]);
    if (argc > 3) Ecm_points  = std::stoi(argv[3]);
    if (argc > 4) nnP0 = std::stoi(argv[4]);
    if (argc > 5) nnP1 = std::stoi(argv[5]);
    if (argc > 6) nnP2 = std::stoi(argv[6]);
    if (argc > 7) irrep = argv[7];
    if (argc > 8) compare_stride = std::stoi(argv[8]);
    if (argc > 9) debug = argv[9][0];

    if (Ecm_points <= 0) {
        std::cerr << "Ecm_points must be positive.\n";
        return 1;
    }

    if (compare_stride <= 0) compare_stride = 1;

    const double atmpi = 0.06906;
    const double atmK  = 0.09698;
    const double xi = 3.444;
    const double Lbyas = 20.0;
    const double L = xi * Lbyas;
    const double epsilon_h = 0.0;
    const double max_shell_num = 20.0;
    const double tolerance = 0.0;

    // Current v9/v18 choices used in your tests.
    const int parity = -1;
    const bool sort_orbit_flag = false;

    std::vector<int> waves_vec_1 = {0, 1};
    std::vector<int> waves_vec_2 = {0};

    const comp pi = std::acos(-1.0);
    const comp twopibyL = ((comp)2.0 * pi) / ((comp)L);

    std::vector<comp> total_P(3);
    total_P[0] = twopibyL * (double)nnP0;
    total_P[1] = twopibyL * (double)nnP1;
    total_P[2] = twopibyL * (double)nnP2;

    std::vector<comp> nnP_config = {(comp)nnP0, (comp)nnP1, (comp)nnP2};

    const double del_Ecm =
        (Ecm_points == 1)
        ? 0.0
        : (Ecm_final - Ecm_initial) / double(Ecm_points - 1);

    pgpu::Options pgopt;
    pgopt.debug = debug;
    pgopt.eig_tol = 0.05;
    pgopt.chop_tol = 1.0e-13;

    std::ofstream summary("projection_v9_vs_v18_summary.dat");
    std::ofstream failures("projection_v9_vs_v18_failures.dat");
    std::ofstream eigout("projection_v9_vs_v18_eigenvalues.dat");
    std::ofstream cfgout("projection_v9_vs_v18_config_shapes.dat");

    summary << "# i\tEcm\tEn\tdim1\tdim2\ttotal_dim\t"
            << "cpu_Vcols\tgpu_Vcols\t"
            << "cpu_trace\tgpu_trace\t"
            << "cpu_near1_005\tgpu_near1_005\t"
            << "cpu_gt_half\tgpu_gt_half\t"
            << "cpu_eval_max\tgpu_eval_max\t"
            << "cpu_closest_eval1\tgpu_closest_eval1\t"
            << "P_max_abs\tP_max_rel\tP_bad\t"
            << "Pproj_max_abs\tPproj_max_rel\tPproj_bad\t"
            << "status\n";

    failures << "# Projection failures / mismatches for v9 CPU Vsel vs v18 GPU Vsel\n";
    failures << "# i Ecm En dim1 dim2 total_dim cpu_Vcols gpu_Vcols reason\n";

    eigout << "# i\tEcm\tN\tpath\ttrace\tnear1_005\tnear1_1e8\tgt_half\t"
           << "eval_min\teval_max\tclosest_eval_to_one\tclosest_dist_to_one\t"
           << "idempotency_error\tfirst_eigs\tlast_eigs\n";

    cfgout << "# Config shape transitions and samples\n";
    cfgout << "# i Ecm En dim1 dim2 total_dim\n";

    std::cout << "# projection v9 CPU vs v18 GPU Vsel diagnostic\n";
    std::cout << "# Ecm=[" << std::setprecision(17) << Ecm_initial << ", " << Ecm_final
              << "] points=" << Ecm_points
              << " nnP=[" << nnP0 << "," << nnP1 << "," << nnP2 << "]"
              << " irrep=" << irrep
              << " stride=" << compare_stride
              << " debug=" << debug << "\n";

    int previous_dim1 = -1;
    int previous_dim2 = -1;

    int n_tested = 0;
    int n_fail = 0;
    int n_vcol_mismatch = 0;
    int n_gpu_zero_cpu_nonzero = 0;
    int n_projector_bad = 0;

    for (int i = 0; i < Ecm_points; i += compare_stride) {
        const double Ecm = Ecm_initial + double(i) * del_Ecm;
        const double En = Ecm_to_E((comp)Ecm, total_P).real();

        std::vector<std::vector<comp>> plm_config(5), klm_config(5);
        std::vector<std::vector<int>> np_config(5), nk_config(5);

        config_maker_4(
            plm_config,
            np_config,
            waves_vec_1,
            (comp)En,
            total_P,
            atmK,
            atmK,
            atmpi,
            L,
            epsilon_h,
            max_shell_num,
            tolerance);

        config_maker_4(
            klm_config,
            nk_config,
            waves_vec_2,
            (comp)En,
            total_P,
            atmpi,
            atmK,
            atmK,
            L,
            epsilon_h,
            max_shell_num,
            tolerance);

        const int dim1 = static_cast<int>(np_config[0].size());
        const int dim2 = static_cast<int>(nk_config[0].size());
        const int N = dim1 + dim2;

        if (N <= 0) continue;

        const bool shape_transition = (dim1 != previous_dim1 || dim2 != previous_dim2);

        if (shape_transition || debug == 'y') {
            cfgout << i << "\t" << std::setprecision(17) << Ecm << "\t" << En << "\t"
                   << dim1 << "\t" << dim2 << "\t" << N << "\n";
            print_config_sample(cfgout, i, Ecm, "np_config", np_config, 10);
            print_config_sample(cfgout, i, Ecm, "nk_config", nk_config, 10);
            cfgout << "\n";
        }

        previous_dim1 = dim1;
        previous_dim2 = dim2;

        Eigen::MatrixXcd P_cpu;
        Eigen::MatrixXcd V_cpu;
        Eigen::MatrixXcd Pproj_cpu;

        Eigen::MatrixXcd P_gpu;
        Eigen::MatrixXcd V_gpu;
        Eigen::MatrixXcd Pproj_gpu;

        std::string status = "PASS";
        std::string reason = "";

        try {
            P_irrep_projection_2plus1(
                P_cpu,
                plm_config,
                np_config,
                klm_config,
                nk_config,
                irrep,
                total_P,
                nnP_config,
                sort_orbit_flag,
                parity);

            build_projector_from_eigenvectors_near_one(
                P_cpu,
                V_cpu,
                Pproj_cpu,
                0.05,
                1.0e-12,
                1.0e-10,
                'n');
        }
        catch (const std::exception& e) {
            status = "CPU_EXCEPTION";
            reason = e.what();
            failures << i << " " << std::setprecision(17) << Ecm << " " << En << " "
                     << dim1 << " " << dim2 << " " << N << " "
                     << -1 << " " << -1 << " "
                     << "CPU_EXCEPTION: " << e.what() << "\n";
            n_fail++;
            continue;
        }

        try {
            pgpu::P_irrep_projection_2plus1_and_Vsel_gpu_safe(
                P_gpu,
                V_gpu,
                Pproj_gpu,
                np_config,
                nk_config,
                irrep,
                nnP0,
                nnP1,
                nnP2,
                sort_orbit_flag,
                parity,
                pgopt);
        }
        catch (const std::exception& e) {
            status = "GPU_EXCEPTION";
            reason = e.what();

            // Still summarize CPU projector if possible.
            const EigenSummary ecs = summarize_projector(P_cpu);

            summary << i << '\t' << std::setprecision(17) << Ecm << '\t' << En << '\t'
                    << dim1 << '\t' << dim2 << '\t' << N << '\t'
                    << V_cpu.cols() << '\t' << -1 << '\t'
                    << ecs.trace_real << '\t' << "nan" << '\t'
                    << ecs.count_near_one_005 << '\t' << -1 << '\t'
                    << ecs.count_gt_half << '\t' << -1 << '\t'
                    << ecs.eval_max << '\t' << "nan" << '\t'
                    << ecs.closest_eval_to_one << '\t' << "nan" << '\t'
                    << "nan\tnan\t-1\tnan\tnan\t-1\t"
                    << status << '\n';

            failures << i << " " << std::setprecision(17) << Ecm << " " << En << " "
                     << dim1 << " " << dim2 << " " << N << " "
                     << V_cpu.cols() << " " << -1 << " "
                     << "GPU_EXCEPTION: " << e.what() << "\n";

            n_fail++;
            continue;
        }

        const EigenSummary ecs = summarize_projector(P_cpu);
        const EigenSummary egs = summarize_projector(P_gpu);

        eigout << i << '\t' << std::setprecision(17) << Ecm << '\t' << N << '\t'
               << "CPU_v9" << '\t'
               << ecs.trace_real << '\t'
               << ecs.count_near_one_005 << '\t'
               << ecs.count_near_one_1e8 << '\t'
               << ecs.count_gt_half << '\t'
               << ecs.eval_min << '\t'
               << ecs.eval_max << '\t'
               << ecs.closest_eval_to_one << '\t'
               << ecs.closest_to_one << '\t'
               << ecs.idempotency_error << '\t'
               << first_eigs_string(ecs.evals) << '\t'
               << last_eigs_string(ecs.evals) << '\n';

        eigout << i << '\t' << std::setprecision(17) << Ecm << '\t' << N << '\t'
               << "GPU_v18" << '\t'
               << egs.trace_real << '\t'
               << egs.count_near_one_005 << '\t'
               << egs.count_near_one_1e8 << '\t'
               << egs.count_gt_half << '\t'
               << egs.eval_min << '\t'
               << egs.eval_max << '\t'
               << egs.closest_eval_to_one << '\t'
               << egs.closest_to_one << '\t'
               << egs.idempotency_error << '\t'
               << first_eigs_string(egs.evals) << '\t'
               << last_eigs_string(egs.evals) << '\n';

        CompareStats pstat = compare_matrix(P_cpu, P_gpu, 1.0e-10, 1.0e-10);
        CompareStats qstat = compare_matrix(Pproj_cpu, Pproj_gpu, 1.0e-8, 1.0e-8);

        bool pass = true;

        if (V_cpu.cols() != V_gpu.cols()) {
            pass = false;
            n_vcol_mismatch++;
            if (V_cpu.cols() > 0 && V_gpu.cols() == 0) {
                n_gpu_zero_cpu_nonzero++;
            }
            reason += "Vcols_mismatch;";
        }

        if (pstat.shape_mismatch || pstat.n_bad != 0) {
            pass = false;
            n_projector_bad++;
            reason += "P_matrix_mismatch;";
        }

        if (qstat.shape_mismatch || qstat.n_bad != 0) {
            pass = false;
            reason += "Pproj_mismatch;";
        }

        if (!pass) {
            status = "FAIL";
            n_fail++;

            failures << i << " " << std::setprecision(17) << Ecm << " " << En << " "
                     << dim1 << " " << dim2 << " " << N << " "
                     << V_cpu.cols() << " " << V_gpu.cols() << " "
                     << reason
                     << " cpu_trace=" << ecs.trace_real
                     << " gpu_trace=" << egs.trace_real
                     << " cpu_near1=" << ecs.count_near_one_005
                     << " gpu_near1=" << egs.count_near_one_005
                     << " cpu_evalmax=" << ecs.eval_max
                     << " gpu_evalmax=" << egs.eval_max
                     << " Pmaxabs=" << pstat.max_abs
                     << " Pmaxloc=(" << pstat.max_i << "," << pstat.max_j << ")"
                     << "\n";
        }

        summary << i << '\t' << std::setprecision(17) << Ecm << '\t' << En << '\t'
                << dim1 << '\t' << dim2 << '\t' << N << '\t'
                << V_cpu.cols() << '\t' << V_gpu.cols() << '\t'
                << ecs.trace_real << '\t' << egs.trace_real << '\t'
                << ecs.count_near_one_005 << '\t' << egs.count_near_one_005 << '\t'
                << ecs.count_gt_half << '\t' << egs.count_gt_half << '\t'
                << ecs.eval_max << '\t' << egs.eval_max << '\t'
                << ecs.closest_eval_to_one << '\t' << egs.closest_eval_to_one << '\t'
                << pstat.max_abs << '\t' << pstat.max_rel << '\t' << pstat.n_bad << '\t'
                << qstat.max_abs << '\t' << qstat.max_rel << '\t' << qstat.n_bad << '\t'
                << status << '\n';

        if (debug == 'y' || !pass || shape_transition) {
            std::cout << std::setprecision(17)
                      << "i=" << i
                      << " Ecm=" << Ecm
                      << " En=" << En
                      << " dims=(" << dim1 << "," << dim2 << ")"
                      << " N=" << N
                      << " Vcols cpu/gpu=" << V_cpu.cols() << "/" << V_gpu.cols()
                      << " trace cpu/gpu=" << ecs.trace_real << "/" << egs.trace_real
                      << " near1 cpu/gpu=" << ecs.count_near_one_005 << "/" << egs.count_near_one_005
                      << " evalmax cpu/gpu=" << ecs.eval_max << "/" << egs.eval_max
                      << " Pmaxabs=" << pstat.max_abs
                      << " status=" << status;

            if (!reason.empty()) std::cout << " reason=" << reason;
            std::cout << "\n";
        }

        n_tested++;
    }

    summary.close();
    failures.close();
    eigout.close();
    cfgout.close();

    std::cout << "\n# Done projection diagnostic\n";
    std::cout << "# tested = " << n_tested << "\n";
    std::cout << "# failures = " << n_fail << "\n";
    std::cout << "# Vcol mismatches = " << n_vcol_mismatch << "\n";
    std::cout << "# GPU zero while CPU nonzero = " << n_gpu_zero_cpu_nonzero << "\n";
    std::cout << "# projector matrix mismatches = " << n_projector_bad << "\n";
    std::cout << "# wrote projection_v9_vs_v18_summary.dat\n";
    std::cout << "# wrote projection_v9_vs_v18_failures.dat\n";
    std::cout << "# wrote projection_v9_vs_v18_eigenvalues.dat\n";
    std::cout << "# wrote projection_v9_vs_v18_config_shapes.dat\n";

    // Return nonzero if mismatch is found, because this is a diagnostic test.
    return (n_fail == 0) ? 0 : 2;
}
