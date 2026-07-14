#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#include "F3_gpu_full_matrix_builder_v10_adaptive_zeros.cu"

int main(int argc, char** argv)
{
    try
    {
        int n0 = 0, n1 = 0, n2 = 0;
        std::string irrep = "A1m";
        std::string irrep_tag = "A1m";

        if (argc >= 4)
        {
            n0 = std::stoi(argv[1]);
            n1 = std::stoi(argv[2]);
            n2 = std::stoi(argv[3]);
        }
        if (argc >= 5) irrep = argv[4];
        if (argc >= 6) irrep_tag = argv[5];
        else irrep_tag = irrep;

        std::vector<int> nnP_vec = {n0, n1, n2};

        std::cout << "# F3 v10 GPU-matrix-builder adaptive zero search\n";
        std::cout << "# nnP=[" << n0 << "," << n1 << "," << n2 << "] irrep=" << irrep
                  << " irrep_tag=" << irrep_tag << "\n";

        auto t0 = std::chrono::high_resolution_clock::now();

        test_F3_with_pwave_all_energy_gpu_omp_cublas_adaptive_zero_search_v1(
            nnP_vec,
            irrep,
            irrep_tag
        );

        auto t1 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();

        std::cout << "# total_wall_sec " << sec << "\n";

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
