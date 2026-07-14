// ============================================================================
// test_gpu_config_maker4_two_flavor.cu
//
// Standalone/drop-in tester for functions_gpu_config_maker4.cuh.
//
// Purpose:
//   Compare CPU config_maker_4 against GPU-safe config_maker_4 for the two
//   flavor channels used in your KKpi/2+1 setup:
//
//     flavor 1 / plm: waves_vec_1 = {0,1}, masses = (atmK, atmK, atmpi)
//     flavor 2 / klm: waves_vec_2 = {0},   masses = (atmpi, atmK, atmK)
//
// It tests over an Ecm grid, writes per-energy dimension summaries, and writes
// detailed mismatch reports if CPU/GPU configs differ.
//
// Required local headers in your source directory:
//   functions.h
//   spherical_functions.h
//   functions_gpu_config_maker4.cuh
//
// Compile example:
//   nvcc -O3 -g -lineinfo -std=c++17 --expt-relaxed-constexpr \
//        -DEIGEN_NO_CUDA \
//        -I/usr/include/eigen3 -I./source \
//        test_gpu_config_maker4_two_flavor.cu \
//        -lcudart -o test_gpu_config_maker4
//
// Run examples:
//   ./test_gpu_config_maker4
//   ./test_gpu_config_maker4 0.26310 0.36 5000 0 0 0
//   ./test_gpu_config_maker4 0.320 0.335 200 0 0 0
//
// Output files:
//   gpu_config_maker4_two_flavor_summary.dat
//   gpu_config_maker4_mismatch_flavor1_plm.dat
//   gpu_config_maker4_mismatch_flavor2_klm.dat
//
// ============================================================================

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "functions.h"
#include "functions_gpu_config_maker4.cuh"

using comp = std::complex<double>;

#ifndef F4_TEST_CUDA_CHECK
#define F4_TEST_CUDA_CHECK(call)                                                     \
    do {                                                                            \
        cudaError_t err__ = (call);                                                   \
        if (err__ != cudaSuccess) {                                                   \
            std::ostringstream os__;                                                  \
            os__ << "CUDA error at " << __FILE__ << ":" << __LINE__               \
                 << " in " << #call << " : " << cudaGetErrorString(err__);         \
            throw std::runtime_error(os__.str());                                    \
        }                                                                             \
    } while (0)
#endif

struct ConfigCompareStats
{
    int nE = 0;
    int count_mismatches = 0;
    int n_mismatched_energies = 0;
    int first_bad_i = -1;
    double max_abs_p_diff = 0.0;
};

static inline double cabs_diff(comp a, comp b)
{
    return std::abs(a - b);
}

static inline std::string vec3_to_string(int a, int b, int c)
{
    return std::to_string(a) + std::to_string(b) + std::to_string(c);
}

static inline bool compare_one_config(
    const std::vector<std::vector<comp>>& cpu_plm,
    const std::vector<std::vector<int>>& cpu_n,
    const std::vector<std::vector<comp>>& gpu_plm,
    const std::vector<std::vector<int>>& gpu_n,
    double momentum_tol,
    ConfigCompareStats& stats,
    int iE,
    double Ecm,
    double En,
    const std::string& label,
    std::ofstream& mismatch_out)
{
    bool ok = true;

    const std::size_t cpu_size = cpu_plm.empty() ? 0 : cpu_plm[0].size();
    const std::size_t gpu_size = gpu_plm.empty() ? 0 : gpu_plm[0].size();

    if (cpu_plm.size() != 5 || gpu_plm.size() != 5 || cpu_n.size() != 5 || gpu_n.size() != 5)
    {
        mismatch_out << "# BAD_CONTAINER_SIZE " << label
                     << " i=" << iE
                     << " Ecm=" << std::setprecision(17) << Ecm
                     << " En=" << En
                     << " cpu_plm_rows=" << cpu_plm.size()
                     << " gpu_plm_rows=" << gpu_plm.size()
                     << " cpu_n_rows=" << cpu_n.size()
                     << " gpu_n_rows=" << gpu_n.size()
                     << '\n';
        ok = false;
    }

    if (cpu_size != gpu_size)
    {
        mismatch_out << "# SIZE_MISMATCH " << label
                     << " i=" << iE
                     << " Ecm=" << std::setprecision(17) << Ecm
                     << " En=" << En
                     << " cpu_size=" << cpu_size
                     << " gpu_size=" << gpu_size
                     << '\n';
        ok = false;
    }

    const std::size_t ncheck = std::min(cpu_size, gpu_size);

    for (std::size_t s = 0; s < ncheck; ++s)
    {
        double max_p_diff_this = 0.0;

        for (int r = 0; r < 5; ++r)
        {
            if (r < 3)
            {
                const double d = cabs_diff(cpu_plm[r][s], gpu_plm[r][s]);
                max_p_diff_this = std::max(max_p_diff_this, d);
                stats.max_abs_p_diff = std::max(stats.max_abs_p_diff, d);

                if (d > momentum_tol)
                {
                    ok = false;
                }
            }
            else
            {
                const int cpu_lm = static_cast<int>(std::llround(cpu_plm[r][s].real()));
                const int gpu_lm = static_cast<int>(std::llround(gpu_plm[r][s].real()));
                if (cpu_lm != gpu_lm)
                {
                    ok = false;
                }
            }

            if (cpu_n[r][s] != gpu_n[r][s])
            {
                ok = false;
            }
        }

        if (!ok)
        {
            mismatch_out << std::setprecision(17)
                         << "MISMATCH\t" << label
                         << "\ti=" << iE
                         << "\tEcm=" << Ecm
                         << "\tEn=" << En
                         << "\tstate=" << s
                         << "\tcpu_n=(" << cpu_n[0][s] << ',' << cpu_n[1][s] << ',' << cpu_n[2][s]
                         << "; ell=" << cpu_n[3][s] << "; m=" << cpu_n[4][s] << ")"
                         << "\tgpu_n=(" << gpu_n[0][s] << ',' << gpu_n[1][s] << ',' << gpu_n[2][s]
                         << "; ell=" << gpu_n[3][s] << "; m=" << gpu_n[4][s] << ")"
                         << "\tcpu_p=(" << cpu_plm[0][s] << ',' << cpu_plm[1][s] << ',' << cpu_plm[2][s] << ")"
                         << "\tgpu_p=(" << gpu_plm[0][s] << ',' << gpu_plm[1][s] << ',' << gpu_plm[2][s] << ")"
                         << "\tmax_p_diff_state=" << max_p_diff_this
                         << '\n';
            break;
        }
    }

    if (!ok)
    {
        stats.count_mismatches += 1;
        stats.n_mismatched_energies += 1;
        if (stats.first_bad_i < 0)
        {
            stats.first_bad_i = iE;
        }
    }

    return ok;
}

static inline void print_usage(const char* exe)
{
    std::cout << "Usage:\n"
              << "  " << exe << " [Ecm_initial Ecm_final Ecm_points nPx nPy nPz]\n\n"
              << "Defaults:\n"
              << "  Ecm_initial = 0.26310\n"
              << "  Ecm_final   = 0.36\n"
              << "  Ecm_points  = 5000\n"
              << "  nnP         = 0 0 0\n";
}

void test_gpu_config_maker4_two_flavor_over_energy_region(
    double Ecm_initial = 0.26310,
    double Ecm_final   = 0.36,
    int Ecm_points     = 5000,
    int nPx            = 0,
    int nPy            = 0,
    int nPz            = 0,
    char debug         = 'y')
{
    if (Ecm_points <= 0)
    {
        throw std::runtime_error("Ecm_points must be > 0");
    }

    const double pi = std::acos(-1.0);

    const double atmpi = 0.06906;
    const double atmK  = 0.09698;

    const double xi    = 3.444;
    const double Lbyas = 20.0;
    const double L     = xi * Lbyas;

    const double epsilon_h = 0.0;
    const double max_shell_num = 20.0;
    const double tolerance = 1.0e-12;

    const std::vector<int> waves_vec_1 = {0, 1};
    const std::vector<int> waves_vec_2 = {0};

    const comp twopibyL = comp(2.0 * pi / L, 0.0);

    std::vector<comp> total_P(3);
    total_P[0] = twopibyL * double(nPx);
    total_P[1] = twopibyL * double(nPy);
    total_P[2] = twopibyL * double(nPz);

    const double P2 = std::norm(total_P[0]) + std::norm(total_P[1]) + std::norm(total_P[2]);

    const double del_Ecm = (Ecm_points == 1)
        ? 0.0
        : std::abs(Ecm_final - Ecm_initial) / double(Ecm_points - 1);

    std::vector<double> Ecm_values(static_cast<std::size_t>(Ecm_points));
    std::vector<double> En_values(static_cast<std::size_t>(Ecm_points));

    for (int i = 0; i < Ecm_points; ++i)
    {
        const double Ecm = Ecm_initial + double(i) * del_Ecm;
        const double En = std::sqrt(Ecm * Ecm + P2);
        Ecm_values[std::size_t(i)] = Ecm;
        En_values[std::size_t(i)] = En;
    }

    if (debug == 'y')
    {
        int device = 0;
        F4_TEST_CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        F4_TEST_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

        std::size_t freeB = 0;
        std::size_t totalB = 0;
        F4_TEST_CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));

        std::cout << std::setprecision(17);
        std::cout << "[test] GPU device = " << device << " : " << prop.name << '\n';
        std::cout << "[test] free VRAM bytes = " << freeB << '\n';
        std::cout << "[test] total VRAM bytes = " << totalB << '\n';
        std::cout << "[test] Ecm range = [" << Ecm_initial << ", " << Ecm_final << "]\n";
        std::cout << "[test] Ecm_points = " << Ecm_points << '\n';
        std::cout << "[test] nnP = (" << nPx << ',' << nPy << ',' << nPz << ")\n";
        std::cout << "[test] L = " << L << " Lbyas = " << Lbyas << '\n';
    }

    // ------------------------------------------------------------------------
    // GPU build for both flavors over all energies.
    // ------------------------------------------------------------------------
    f4gpu::Options opt;
    opt.debug = debug;
    opt.vram_safety_fraction = 0.35;
    opt.max_energies_per_chunk = 4096;
    opt.cuda_threads_per_block = 256;

    std::vector<std::vector<std::vector<comp>>> gpu_plm_vec;
    std::vector<std::vector<std::vector<int>>>  gpu_np_vec;
    std::vector<std::vector<std::vector<comp>>> gpu_klm_vec;
    std::vector<std::vector<std::vector<int>>>  gpu_nk_vec;

    auto t0_gpu = std::chrono::high_resolution_clock::now();

    f4gpu::gpu_config_maker_4_many_to_cpu_vectors(
        gpu_plm_vec,
        gpu_np_vec,
        waves_vec_1,
        En_values,
        total_P,
        atmK,
        atmK,
        atmpi,
        L,
        epsilon_h,
        max_shell_num,
        tolerance,
        opt
    );

    f4gpu::gpu_config_maker_4_many_to_cpu_vectors(
        gpu_klm_vec,
        gpu_nk_vec,
        waves_vec_2,
        En_values,
        total_P,
        atmpi,
        atmK,
        atmK,
        L,
        epsilon_h,
        max_shell_num,
        tolerance,
        opt
    );

    auto t1_gpu = std::chrono::high_resolution_clock::now();
    const double gpu_sec = std::chrono::duration<double>(t1_gpu - t0_gpu).count();

    // ------------------------------------------------------------------------
    // CPU reference build + comparison.
    // ------------------------------------------------------------------------
    ConfigCompareStats stats_flavor1;
    ConfigCompareStats stats_flavor2;
    stats_flavor1.nE = Ecm_points;
    stats_flavor2.nE = Ecm_points;

    std::ofstream summary("gpu_config_maker4_two_flavor_summary.dat");
    std::ofstream mismatch1("gpu_config_maker4_mismatch_flavor1_plm.dat");
    std::ofstream mismatch2("gpu_config_maker4_mismatch_flavor2_klm.dat");

    if (!summary.is_open() || !mismatch1.is_open() || !mismatch2.is_open())
    {
        throw std::runtime_error("Could not open one or more output files.");
    }

    summary << "# i\tEcm\tEn\tcpu_dim1\tgpu_dim1\tcpu_dim2\tgpu_dim2\t"
            << "cpu_total_dim\tgpu_total_dim\tmatch_flavor1\tmatch_flavor2\n";

    mismatch1 << "# Detailed mismatches for flavor1/plm, waves={0,1}, masses=(K,K,pi)\n";
    mismatch2 << "# Detailed mismatches for flavor2/klm, waves={0}, masses=(pi,K,K)\n";

    const double momentum_tol = 5.0e-14;

    auto t0_cpu = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < Ecm_points; ++i)
    {
        const double Ecm = Ecm_values[std::size_t(i)];
        const double En  = En_values[std::size_t(i)];

        std::vector<std::vector<comp>> cpu_plm(5), cpu_klm(5);
        std::vector<std::vector<int>>  cpu_np(5),  cpu_nk(5);

        config_maker_4(
            cpu_plm,
            cpu_np,
            waves_vec_1,
            comp(En, 0.0),
            total_P,
            atmK,
            atmK,
            atmpi,
            L,
            epsilon_h,
            max_shell_num,
            tolerance
        );

        config_maker_4(
            cpu_klm,
            cpu_nk,
            waves_vec_2,
            comp(En, 0.0),
            total_P,
            atmpi,
            atmK,
            atmK,
            L,
            epsilon_h,
            max_shell_num,
            tolerance
        );

        bool match1 = compare_one_config(
            cpu_plm,
            cpu_np,
            gpu_plm_vec[std::size_t(i)],
            gpu_np_vec[std::size_t(i)],
            momentum_tol,
            stats_flavor1,
            i,
            Ecm,
            En,
            "flavor1_plm",
            mismatch1
        );

        bool match2 = compare_one_config(
            cpu_klm,
            cpu_nk,
            gpu_klm_vec[std::size_t(i)],
            gpu_nk_vec[std::size_t(i)],
            momentum_tol,
            stats_flavor2,
            i,
            Ecm,
            En,
            "flavor2_klm",
            mismatch2
        );

        const int cpu_dim1 = int(cpu_plm[0].size());
        const int gpu_dim1 = int(gpu_plm_vec[std::size_t(i)][0].size());
        const int cpu_dim2 = int(cpu_klm[0].size());
        const int gpu_dim2 = int(gpu_klm_vec[std::size_t(i)][0].size());

        summary << std::setprecision(17)
                << i << '\t'
                << Ecm << '\t'
                << En << '\t'
                << cpu_dim1 << '\t'
                << gpu_dim1 << '\t'
                << cpu_dim2 << '\t'
                << gpu_dim2 << '\t'
                << cpu_dim1 + cpu_dim2 << '\t'
                << gpu_dim1 + gpu_dim2 << '\t'
                << int(match1) << '\t'
                << int(match2) << '\n';

        if (debug == 'y' && (i < 5 || i == Ecm_points - 1 || i % 250 == 0 || !match1 || !match2))
        {
            std::cout << std::setprecision(17)
                      << "[compare] i=" << i
                      << " Ecm=" << Ecm
                      << " En=" << En
                      << " dim1 CPU/GPU=" << cpu_dim1 << '/' << gpu_dim1
                      << " dim2 CPU/GPU=" << cpu_dim2 << '/' << gpu_dim2
                      << " total CPU/GPU=" << (cpu_dim1 + cpu_dim2) << '/' << (gpu_dim1 + gpu_dim2)
                      << " match1=" << match1
                      << " match2=" << match2
                      << '\n';
        }
    }

    auto t1_cpu = std::chrono::high_resolution_clock::now();
    const double cpu_sec = std::chrono::duration<double>(t1_cpu - t0_cpu).count();

    summary.close();
    mismatch1.close();
    mismatch2.close();

    std::cout << "\n========== GPU config_maker_4 two-flavor test summary ==========" << '\n';
    std::cout << std::setprecision(17);
    std::cout << "Ecm range              = [" << Ecm_initial << ", " << Ecm_final << "]\n";
    std::cout << "Ecm_points             = " << Ecm_points << '\n';
    std::cout << "nnP                    = (" << nPx << ',' << nPy << ',' << nPz << ")\n";
    std::cout << "GPU build time seconds = " << gpu_sec << '\n';
    std::cout << "CPU ref+compare seconds= " << cpu_sec << '\n';
    std::cout << "flavor1 mismatched energies = " << stats_flavor1.n_mismatched_energies
              << " first_bad_i = " << stats_flavor1.first_bad_i
              << " max_abs_p_diff = " << stats_flavor1.max_abs_p_diff << '\n';
    std::cout << "flavor2 mismatched energies = " << stats_flavor2.n_mismatched_energies
              << " first_bad_i = " << stats_flavor2.first_bad_i
              << " max_abs_p_diff = " << stats_flavor2.max_abs_p_diff << '\n';
    std::cout << "summary file           = gpu_config_maker4_two_flavor_summary.dat\n";
    std::cout << "mismatch flavor1 file  = gpu_config_maker4_mismatch_flavor1_plm.dat\n";
    std::cout << "mismatch flavor2 file  = gpu_config_maker4_mismatch_flavor2_klm.dat\n";
    std::cout << "===============================================================\n";
}

int main(int argc, char** argv)
{
    try
    {
        double Ecm_initial = 0.26310;
        double Ecm_final   = 0.36;
        int Ecm_points     = 5000;
        int nPx = 0;
        int nPy = 0;
        int nPz = 0;

        if (argc == 2)
        {
            const std::string arg1 = argv[1];
            if (arg1 == "-h" || arg1 == "--help")
            {
                print_usage(argv[0]);
                return 0;
            }
        }

        if (argc == 7)
        {
            Ecm_initial = std::stod(argv[1]);
            Ecm_final   = std::stod(argv[2]);
            Ecm_points  = std::stoi(argv[3]);
            nPx         = std::stoi(argv[4]);
            nPy         = std::stoi(argv[5]);
            nPz         = std::stoi(argv[6]);
        }
        else if (argc != 1)
        {
            print_usage(argv[0]);
            return 1;
        }

        test_gpu_config_maker4_two_flavor_over_energy_region(
            Ecm_initial,
            Ecm_final,
            Ecm_points,
            nPx,
            nPy,
            nPz,
            'y'
        );
    }
    catch (const std::exception& e)
    {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
