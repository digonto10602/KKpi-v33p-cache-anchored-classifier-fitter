#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "functions.h"
#include "K3_functions_2plus1.hpp"

using comp = std::complex<double>;

static void write_config_file(
    const std::string& filename,
    const std::vector<std::vector<comp>>& plm_config,
    const std::vector<std::vector<int>>& np_config,
    const std::vector<std::vector<comp>>& klm_config,
    const std::vector<std::vector<int>>& nk_config)
{
    std::ofstream f(filename);
    f << std::setprecision(17);
    f << "# global_index flavor local_index nx ny nz px py pz ell m\n";

    int g = 0;
    for (int a = 0; a < static_cast<int>(plm_config[0].size()); ++a, ++g)
    {
        f << g << '\t' << 1 << '\t' << a << '\t'
          << np_config[0][a] << '\t' << np_config[1][a] << '\t' << np_config[2][a] << '\t'
          << std::real(plm_config[0][a]) << '\t'
          << std::real(plm_config[1][a]) << '\t'
          << std::real(plm_config[2][a]) << '\t'
          << static_cast<int>(std::llround(std::real(plm_config[3][a]))) << '\t'
          << static_cast<int>(std::llround(std::real(plm_config[4][a]))) << '\n';
    }

    for (int a = 0; a < static_cast<int>(klm_config[0].size()); ++a, ++g)
    {
        f << g << '\t' << 2 << '\t' << a << '\t'
          << nk_config[0][a] << '\t' << nk_config[1][a] << '\t' << nk_config[2][a] << '\t'
          << std::real(klm_config[0][a]) << '\t'
          << std::real(klm_config[1][a]) << '\t'
          << std::real(klm_config[2][a]) << '\t'
          << static_cast<int>(std::llround(std::real(klm_config[3][a]))) << '\t'
          << static_cast<int>(std::llround(std::real(klm_config[4][a]))) << '\n';
    }
}

static void write_matrix_file(const std::string& filename, const Eigen::MatrixXcd& M)
{
    std::ofstream f(filename);
    f << std::setprecision(17);
    f << "# row col real imag abs\n";
    for (int r = 0; r < M.rows(); ++r)
    {
        for (int c = 0; c < M.cols(); ++c)
        {
            f << r << '\t' << c << '\t'
              << std::real(M(r,c)) << '\t'
              << std::imag(M(r,c)) << '\t'
              << std::abs(M(r,c)) << '\n';
        }
    }
}

int main(int argc, char** argv)
{
    // Defaults matching your v23 setup.
    int nPx = 0, nPy = 0, nPz = 1;
    double Ecm = 0.28;
    double Lbyas = 20.0;
    char debug = 'n';

    if (argc > 1) nPx = std::atoi(argv[1]);
    if (argc > 2) nPy = std::atoi(argv[2]);
    if (argc > 3) nPz = std::atoi(argv[3]);
    if (argc > 4) Ecm = std::atof(argv[4]);
    if (argc > 5) Lbyas = std::atof(argv[5]);
    if (argc > 6) debug = argv[6][0];

    const double atmpi = 0.06906;
    const double atmK  = 0.09698;
    const double xi = 3.444;
    const double L = xi * Lbyas;
    const double epsilon_h = 0.0;
    const double max_shell_num = 20.0;
    const double tolerance = 1.0e-12;

    const comp pi = std::acos(-1.0);
    std::vector<comp> nnP_config = {comp(nPx,0), comp(nPy,0), comp(nPz,0)};
    std::vector<comp> total_P = {
        comp(2.0,0) * pi * nnP_config[0] / comp(L,0),
        comp(2.0,0) * pi * nnP_config[1] / comp(L,0),
        comp(2.0,0) * pi * nnP_config[2] / comp(L,0)
    };

    const comp En = Ecm_to_E(comp(Ecm,0.0), total_P);

    std::vector<int> waves_vec_1 = {0, 1};
    std::vector<int> waves_vec_2 = {0};

    std::vector<std::vector<comp>> plm_config(5), klm_config(5);
    std::vector<std::vector<int>>  np_config(5),  nk_config(5);

    // flavor 1 spectator: K spectator; interacting pair K pi
    config_maker_4(plm_config, np_config, waves_vec_1, En, total_P,
                   atmK, atmK, atmpi, L, epsilon_h, max_shell_num, tolerance);

    // flavor 2 spectator: pi spectator; interacting pair K K
    config_maker_4(klm_config, nk_config, waves_vec_2, En, total_P,
                   atmpi, atmK, atmK, L, epsilon_h, max_shell_num, tolerance);

    std::vector<comp> Kiso = {comp(200.0,0.0), comp(400.0,0.0)};
    comp K3B_par = comp(400.0,0.0);
    comp K3E_par = comp(300.0,0.0);

    Eigen::MatrixXcd K3;
    k3_2plus1::K3mat_2plus1(K3, En, plm_config, klm_config, total_P,
                             atmK, atmpi, Kiso, K3B_par, K3E_par, debug);

    write_config_file("k3_cpp_configs.dat", plm_config, np_config, klm_config, nk_config);
    write_matrix_file("k3_cpp_matrix.dat", K3);

    std::cout << std::setprecision(17);
    std::cout << "Ecm = " << Ecm << "\n";
    std::cout << "En  = " << std::real(En) << "\n";
    std::cout << "L   = " << L << "\n";
    std::cout << "dim1 = " << plm_config[0].size() << "\n";
    std::cout << "dim2 = " << klm_config[0].size() << "\n";
    std::cout << "K3 shape = " << K3.rows() << " x " << K3.cols() << "\n";
    std::cout << "wrote k3_cpp_configs.dat and k3_cpp_matrix.dat\n";

    return 0;
}
