//This is the new printer code to check codes and run executables 

#include <bits/stdc++.h>
#include "spherical_functions.h"
#include "functions.h"
#include "F2_functions_v2.h"
#include "K2_functions_v2.h"
#include "QC_functions_v2.h"
#include<omp.h>
//#include "gpu_solvers_batched_streams_v2.cpp"
#include "gpu_varsize_batched_inverse.cu"
#include "dig_tools.hpp"
#include "projections_v1.hpp"
//#include "file_processor.hpp" //need to be commented for F3 v7
#include <vector>
#include <complex>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <stdexcept>

//F3 with solvers based includes 
//#include "F3_gpu_omp_cublas_pipeline_v4_include_safe.cu"
//#include "F3_gpu_omp_cublas_pipeline_v6_oom_safe_chunks.cu" 
//#include "F3_gpu_omp_cublas_pipeline_v7_adaptive_zero_search.cu"
//#include "F3_gpu_omp_cublas_pipeline_v8_matrix_saving.cu"
//#include "F3_gpu_omp_cublas_pipeline_v8_adaptive_matrix_saving.cu"
#include "F3_gpu_omp_cublas_pipeline_v9_inward_shape_refinement.cu"


void test_projections_gpu_v2() {
    
    comp pi = std::acos(-1.0); 
    double atmpi = 0.06906;
    double atmK  = 0.09698;

    double eta_1 = 1.0;
    double eta_2 = 0.5;

    double alpha         = 0.5;
    double max_shell_num = 20.0;

    std::vector<int> waves_vec_1 = {0, 1};
    std::vector<int> waves_vec_2 = {0};

    double epsilon_h = 0;
    bool   Q0norm    = true;

    double xi    = 3.444;
    double Lbyas = 20;
    double L     = xi * Lbyas;

    comp twopibyL = ((comp)2.0) * pi / ((comp)L);

    Vec3 nnP = {0, 0, 0};
    std::vector<comp> total_P(3);
    total_P[0] = twopibyL * (double)nnP[0];
    total_P[1] = twopibyL * (double)nnP[1];
    total_P[2] = twopibyL * (double)nnP[2];

    double tolerance = 0.0;

    std::vector<std::vector<comp>> scatter_params_1(4, std::vector<comp>(3));
    scatter_params_1[0][0] = 4.04;
    scatter_params_1[1][0] = -43.2;

    std::vector<std::vector<comp>> scatter_params_2(4, std::vector<comp>(3));
    scatter_params_2[0][0] = 4.12;

    double En_initial = 0.26310;
    double En_final   = 0.36;
    int    En_points  = 5000;
    double del_En     = std::abs(En_initial - En_final) / (double)En_points;

    std::vector<std::string> irreps = {"A1u"};

    //==========================================================================
    // STEP 1: Build all F3 ingredients + GPU inversion of Hmat
    //==========================================================================
    std::vector<std::vector<std::vector<comp>>> plm_vec(En_points,
        std::vector<std::vector<comp>>(5));
    std::vector<std::vector<std::vector<comp>>> klm_vec(En_points,
        std::vector<std::vector<comp>>(5));

    std::vector<Eigen::MatrixXcd> F2_vec(En_points);
    std::vector<Eigen::MatrixXcd> G_vec(En_points);
    std::vector<Eigen::MatrixXcd> K2inv_vec(En_points);
    std::vector<Eigen::MatrixXcd> Hmat_vec(En_points);
    std::vector<Eigen::MatrixXcd> Hmatinv_vec(En_points);
    std::vector<Eigen::MatrixXcd> F3_vec(En_points);

    auto F3_ing_builder = [&](int i, double En, Eigen::MatrixXcd& Hmat) {
        std::vector<std::vector<comp>> plm_config(5), klm_config(5);
        std::vector<std::vector<int>> np_config(5), nk_config(5);

        config_maker_4(plm_config, np_config, waves_vec_1, En, total_P,
                       atmK, atmK, atmpi, L, epsilon_h, max_shell_num, tolerance);
        config_maker_4(klm_config, nk_config, waves_vec_2, En, total_P,
                       atmpi, atmK, atmK, L, epsilon_h, max_shell_num, tolerance);

        plm_vec[i] = std::move(plm_config);
        klm_vec[i] = std::move(klm_config);

        int dim1 = (int)plm_vec[i][0].size();
        int dim2 = (int)klm_vec[i][0].size();
        int tot  = dim1 + dim2;

        F2_vec[i].resize(tot, tot);
        G_vec[i].resize(tot, tot);
        K2inv_vec[i].resize(tot, tot);
        Hmat_vec[i].resize(tot, tot);

        F2_2plus1_mat(F2_vec[i], En, plm_vec[i], klm_vec[i],
                      total_P, atmK, atmpi, L, alpha, epsilon_h, max_shell_num, Q0norm);

        K2inv_EREord2_2plus1_mat(K2inv_vec[i], eta_1, eta_2,
                                 scatter_params_1, scatter_params_2,
                                 En, plm_vec[i], klm_vec[i],
                                 total_P, atmK, atmpi, epsilon_h, L);

        G_2plus1_mat(G_vec[i], En, plm_vec[i], klm_vec[i],
                     total_P, atmK, atmpi, L, alpha, epsilon_h, max_shell_num, Q0norm);

        Hmat.resize(tot, tot);
        Hmat = K2inv_vec[i] + F2_vec[i] + G_vec[i];
        Hmat_vec[i] = Hmat;
    };

    int chunkSize = suggest_chunk_size_from_En_final(
        En_final, waves_vec_1, waves_vec_2, total_P,
        atmK, atmpi, L, epsilon_h, max_shell_num, tolerance,
        0.90, 0.70, 6);
    printer("suggested chunkSize : ", chunkSize);

    build_and_invert_energy_sweep_varsize_batched_gpu_v3(
        chunkSize, En_initial, En_final, En_points,
        F3_ing_builder, Hmatinv_vec, 0.90, 16);

    // Build F3_vec = F2/3 - F2 * Hmatinv * F2
    for (int i = 0; i < En_points; i++) {
        Eigen::MatrixXcd& F2 = F2_vec[i];
        F3_vec[i] = F2 / 3.0 - F2 * Hmatinv_vec[i] * F2;
    }
    std::cout << "F3_vec built for all energies\n" << std::endl;

    //==========================================================================
    // STEP 2: GPU inversion of all F3 matrices
    //==========================================================================
    std::vector<Eigen::MatrixXcd> F3inv_vec(En_points);

    std::cout << "Inverting all F3 matrices on GPU...\n" << std::endl;

    {
        std::vector<Eigen::MatrixXcd> F3inv_valid(En_points);

        auto F3inv_builder = [&](int j, double /*En*/, Eigen::MatrixXcd& Hmat) {
            Hmat = F3_vec[j];
        };

        build_and_invert_energy_sweep_varsize_batched_gpu_v3(
            chunkSize,
            0.0,
            (double)(En_points - 1),
            En_points,
            F3inv_builder,
            F3inv_valid,
            0.90, 16);

        for (int j = 0; j < En_points; j++)
            F3inv_vec[j] = std::move(F3inv_valid[j]);
    }

    std::cout << "F3inv_vec built for all energies\n" << std::endl;

    //==========================================================================
    // STEP 3: Build n_config for each energy
    //==========================================================================
    std::vector<std::vector<std::vector<int>>> np_vec(En_points,
        std::vector<std::vector<int>>(5));
    std::vector<std::vector<std::vector<int>>> nk_vec(En_points,
        std::vector<std::vector<int>>(5));

    std::cout << "Building n_configs...\n" << std::endl;
    for (int i = 0; i < En_points; i++) {
        double En  = En_initial + i * del_En;
        comp   En_c(En, 0.0);

        std::vector<std::vector<comp>> plm_tmp(5), klm_tmp(5);
        config_maker_4(plm_tmp, np_vec[i], waves_vec_1, En_c, total_P,
                       atmK, atmK, atmpi, L, epsilon_h, max_shell_num, tolerance);
        config_maker_4(klm_tmp, nk_vec[i], waves_vec_2, En_c, total_P,
                       atmpi, atmK, atmK, L, epsilon_h, max_shell_num, tolerance);
    }
    std::cout << "n_configs done\n" << std::endl;

    //==========================================================================
    // STEP 4: Open output file for fixed irrep
    //==========================================================================
    auto nnP_str = [&]() {
        return std::to_string(std::abs(nnP[0]))
             + std::to_string(std::abs(nnP[1]))
             + std::to_string(std::abs(nnP[2]));
    };

    std::string irrep = irreps[0];
    std::string fname = nnP_str() + "_" + irrep + "_det.dat";
    std::ofstream outfile(fname);

    if (!outfile.is_open())
        throw std::runtime_error("Cannot open: " + fname);

    outfile << "# Ecm\tEn\tReDet\tImDet\n";
    std::cout << "Opened: " << fname << "\n";

    //==========================================================================
    // STEP 5: Irrep projection using Psub and GPU-inverted F3
    //==========================================================================
    std::cout << "Starting irrep projections...\n" << std::endl;

    std::vector<int> nnP_config = {nnP[0], nnP[1], nnP[2]};
    std::vector<comp> nnP_config1 = {((comp)nnP[0]), ((comp)nnP[1]), ((comp)nnP[2])};
    bool sort_orbit_flag = false;
    int parity = -1;

    for (int i = 0; i < En_points; i++) {
        double En  = En_initial + i * del_En;
        comp   En_c(En, 0.0);
        comp   P2  = total_P[0]*total_P[0]
                   + total_P[1]*total_P[1]
                   + total_P[2]*total_P[2];
        double Ecm = std::real(std::sqrt(En_c*En_c - P2));

        auto& plm_config = plm_vec[i];
        auto& klm_config = klm_vec[i];
        auto& np_config  = np_vec[i];
        auto& nk_config  = nk_vec[i];

        int dim1 = (int)plm_config[0].size();
        int dim2 = (int)klm_config[0].size();

        comp det_projF3inv(std::numeric_limits<double>::quiet_NaN(), 0.0);

        try {
            if (dim1 + dim2 > 0) {
                Eigen::MatrixXcd P_I(dim1 + dim2, dim1 + dim2);

                P_irrep_projection_2plus1(
                    P_I,
                    plm_config, np_config,
                    klm_config, nk_config,
                    irrep, total_P, nnP_config1,
                    sort_orbit_flag, parity
                );

                Eigen::MatrixXcd Psub;
                projector_subspace_basis(Psub, P_I, 1e-16, 1e-16, 'n');

                Eigen::MatrixXcd F3matinv = F3inv_vec[i];
                Eigen::MatrixXcd projF3inv = Psub.transpose() * F3matinv * Psub;

                det_projF3inv = projF3inv.determinant();
            }
        } catch (const std::exception& e) {
            std::cerr << "Projection error i=" << i
                      << " irrep=" << irrep
                      << ": " << e.what() << "\n";
        }

        outfile << std::fixed << std::setprecision(20)
                << Ecm << '\t'
                << En  << '\t'
                << det_projF3inv.real() << '\t'
                << det_projF3inv.imag() << "\n";

        std::cout << std::fixed << std::setprecision(20)
                  << Ecm << '\t'
                  << En  << '\t'
                  << det_projF3inv.real() << '\t'
                  << det_projF3inv.imag() << "\n";

        if (i % 100 == 0)
            std::cout << "Progress: " << i << "/" << En_points
                      << "  Ecm=" << std::fixed << std::setprecision(6)
                      << Ecm << "\n" << std::endl;
    }

    outfile.close();

    std::cout << "\nDone.\n";
    std::cout << "Output: " << fname << "\n";
}

void test_projections_gpu_v3() {
    
    comp pi = std::acos(-1.0); 
    double atmpi = 0.06906;
    double atmK  = 0.09698;

    double eta_1 = 1.0;
    double eta_2 = 0.5;

    double alpha         = 0.5;
    double max_shell_num = 20.0;

    std::vector<int> waves_vec_1 = {0, 1};
    std::vector<int> waves_vec_2 = {0};

    double epsilon_h = 0;
    bool   Q0norm    = true;

    double xi    = 3.444;
    double Lbyas = 20;
    double L     = xi * Lbyas;

    comp twopibyL = ((comp)2.0) * pi / ((comp)L);

    Vec3 nnP = {0, 0, 1};
    std::vector<comp> total_P(3);
    total_P[0] = twopibyL * (double)nnP[0];
    total_P[1] = twopibyL * (double)nnP[1];
    total_P[2] = twopibyL * (double)nnP[2];

    double tolerance = 0.0;

    std::vector<std::vector<comp>> scatter_params_1(4, std::vector<comp>(3));
    scatter_params_1[0][0] = 4.04;
    scatter_params_1[1][0] = -43.2;

    std::vector<std::vector<comp>> scatter_params_2(4, std::vector<comp>(3));
    scatter_params_2[0][0] = 4.12;

    double En_initial = 0.26310;
    double En_final   = 0.36;
    int    En_points  = 5000;
    double del_En     = std::abs(En_initial - En_final) / (double)En_points;

    std::vector<std::string> irreps = {"A2"};

    //==========================================================================
    // STEP 1: Build all F3 ingredients + GPU inversion of Hmat
    //==========================================================================
    std::vector<std::vector<std::vector<comp>>> plm_vec(En_points,
        std::vector<std::vector<comp>>(5));
    std::vector<std::vector<std::vector<comp>>> klm_vec(En_points,
        std::vector<std::vector<comp>>(5));

    std::vector<Eigen::MatrixXcd> F2_vec(En_points);
    std::vector<Eigen::MatrixXcd> G_vec(En_points);
    std::vector<Eigen::MatrixXcd> K2inv_vec(En_points);
    std::vector<Eigen::MatrixXcd> Hmat_vec(En_points);
    std::vector<Eigen::MatrixXcd> Hmatinv_vec(En_points);
    std::vector<Eigen::MatrixXcd> F3_vec(En_points);

    auto F3_ing_builder = [&](int i, double En, Eigen::MatrixXcd& Hmat) {
        std::vector<std::vector<comp>> plm_config(5), klm_config(5);
        std::vector<std::vector<int>> np_config(5), nk_config(5);

        config_maker_4(plm_config, np_config, waves_vec_1, En, total_P,
                       atmK, atmK, atmpi, L, epsilon_h, max_shell_num, tolerance);
        config_maker_4(klm_config, nk_config, waves_vec_2, En, total_P,
                       atmpi, atmK, atmK, L, epsilon_h, max_shell_num, tolerance);

        plm_vec[i] = std::move(plm_config);
        klm_vec[i] = std::move(klm_config);

        int dim1 = (int)plm_vec[i][0].size();
        int dim2 = (int)klm_vec[i][0].size();
        int tot  = dim1 + dim2;

        F2_vec[i].resize(tot, tot);
        G_vec[i].resize(tot, tot);
        K2inv_vec[i].resize(tot, tot);
        Hmat_vec[i].resize(tot, tot);

        F2_2plus1_mat(F2_vec[i], En, plm_vec[i], klm_vec[i],
                      total_P, atmK, atmpi, L, alpha, epsilon_h, max_shell_num, Q0norm);

        K2inv_EREord2_2plus1_mat(K2inv_vec[i], eta_1, eta_2,
                                 scatter_params_1, scatter_params_2,
                                 En, plm_vec[i], klm_vec[i],
                                 total_P, atmK, atmpi, epsilon_h, L);

        G_2plus1_mat(G_vec[i], En, plm_vec[i], klm_vec[i],
                     total_P, atmK, atmpi, L, alpha, epsilon_h, max_shell_num, Q0norm);

        Hmat.resize(tot, tot);
        Hmat = K2inv_vec[i] + F2_vec[i] + G_vec[i];
        Hmat_vec[i] = Hmat;
    };

    int chunkSize = suggest_chunk_size_from_En_final(
        En_final, waves_vec_1, waves_vec_2, total_P,
        atmK, atmpi, L, epsilon_h, max_shell_num, tolerance,
        0.90, 0.70, 6);
    printer("suggested chunkSize : ", chunkSize);

    build_and_invert_energy_sweep_varsize_batched_gpu_v3(
        chunkSize, En_initial, En_final, En_points,
        F3_ing_builder, Hmatinv_vec, 0.90, 16);

    // Build F3_vec = F2/3 - F2 * Hmatinv * F2
    for (int i = 0; i < En_points; i++) {
        Eigen::MatrixXcd& F2 = F2_vec[i];
        F3_vec[i] = F2 / 3.0 - F2 * Hmatinv_vec[i] * F2;
    }
    std::cout << "F3_vec built for all energies\n" << std::endl;

    //==========================================================================
    // STEP 2: Build valid_indices for invertible F3, invert only those on GPU
    //==========================================================================
    std::vector<Eigen::MatrixXcd> F3inv_vec(En_points);
    std::vector<int> valid_indices;
    valid_indices.reserve(En_points);

    std::cout << "Checking which F3 matrices are valid for inversion...\n" << std::endl;

    for (int i = 0; i < En_points; i++) {
        int dim1 = (int)plm_vec[i][0].size();
        int dim2 = (int)klm_vec[i][0].size();
        int tot  = dim1 + dim2;

        if (tot == 0) continue;

        Eigen::FullPivLU<Eigen::MatrixXcd> lu(F3_vec[i]);
        if (lu.isInvertible()) {
            valid_indices.push_back(i);
        }
    }

    std::cout << "Valid F3 matrices: " << valid_indices.size()
              << " / " << En_points << "\n" << std::endl;

    if (!valid_indices.empty()) {
        int N_valid = (int)valid_indices.size();
        std::vector<Eigen::MatrixXcd> F3inv_valid(N_valid);

        auto F3inv_builder = [&](int j, double /*En*/, Eigen::MatrixXcd& Hmat) {
            int i = valid_indices[j];
            Hmat = F3_vec[i];
        };

        build_and_invert_energy_sweep_varsize_batched_gpu_v3(
            chunkSize,
            0.0,
            (double)(N_valid - 1),
            N_valid,
            F3inv_builder,
            F3inv_valid,
            0.90, 16
        );

        for (int j = 0; j < N_valid; j++) {
            F3inv_vec[valid_indices[j]] = std::move(F3inv_valid[j]);
        }
    }

    std::cout << "F3inv_vec built for all valid energies\n" << std::endl;

    //==========================================================================
    // STEP 3: Build n_config for each energy
    //==========================================================================
    std::vector<std::vector<std::vector<int>>> np_vec(En_points,
        std::vector<std::vector<int>>(5));
    std::vector<std::vector<std::vector<int>>> nk_vec(En_points,
        std::vector<std::vector<int>>(5));

    std::cout << "Building n_configs...\n" << std::endl;
    for (int i = 0; i < En_points; i++) {
        double En  = En_initial + i * del_En;
        comp   En_c(En, 0.0);

        std::vector<std::vector<comp>> plm_tmp(5), klm_tmp(5);
        config_maker_4(plm_tmp, np_vec[i], waves_vec_1, En_c, total_P,
                       atmK, atmK, atmpi, L, epsilon_h, max_shell_num, tolerance);
        config_maker_4(klm_tmp, nk_vec[i], waves_vec_2, En_c, total_P,
                       atmpi, atmK, atmK, L, epsilon_h, max_shell_num, tolerance);
    }
    std::cout << "n_configs done\n" << std::endl;

    //==========================================================================
    // STEP 4: Open output file for fixed irrep
    //==========================================================================
    auto nnP_str = [&]() {
        return std::to_string(std::abs(nnP[0]))
             + std::to_string(std::abs(nnP[1]))
             + std::to_string(std::abs(nnP[2]));
    };

    std::string irrep = irreps[0];
    std::string fname = nnP_str() + "_" + irrep + "_det.dat";
    std::ofstream outfile(fname);

    if (!outfile.is_open())
        throw std::runtime_error("Cannot open: " + fname);

    outfile << "# Ecm\tEn\tReDet\tImDet\n";
    std::cout << "Opened: " << fname << "\n";

    //==========================================================================
    // STEP 5: Irrep projection only for valid energies
    //==========================================================================
    std::cout << "Starting irrep projections for valid energies only...\n" << std::endl;

    std::vector<int> nnP_config = {nnP[0], nnP[1], nnP[2]};
    std::vector<comp> nnP_config1 = {((comp)nnP[0]), ((comp)nnP[1]), ((comp)nnP[2])};
    int sort_orbit_flag = 0;
    int parity = -1;

    for (int idx = 0; idx < (int)valid_indices.size(); idx++) {
        int i = valid_indices[idx];

        double En  = En_initial + i * del_En;
        comp   En_c(En, 0.0);
        comp   P2  = total_P[0]*total_P[0]
                   + total_P[1]*total_P[1]
                   + total_P[2]*total_P[2];
        double Ecm = std::real(std::sqrt(En_c*En_c - P2));

        auto& plm_config = plm_vec[i];
        auto& klm_config = klm_vec[i];
        auto& np_config  = np_vec[i];
        auto& nk_config  = nk_vec[i];

        int dim1 = (int)plm_config[0].size();
        int dim2 = (int)klm_config[0].size();

        try {
            Eigen::MatrixXcd P_I(dim1 + dim2, dim1 + dim2);
            Eigen::MatrixXcd Psub;

            P_irrep_projection_2plus1(
                Psub,
                plm_config, np_config,
                klm_config, nk_config,
                irrep, total_P, nnP_config1,
                sort_orbit_flag, parity
            );

            //projector_subspace_basis(Psub, P_I, 1e-16, 1e-16, 'n');

            Eigen::MatrixXcd F3matinv = F3inv_vec[i];
            Eigen::MatrixXcd projF3inv = Psub.transpose() * F3matinv * Psub;

            comp det_projF3inv(std::numeric_limits<double>::quiet_NaN(),
                               std::numeric_limits<double>::quiet_NaN());

            if (projF3inv.rows() > 0 && projF3inv.cols() > 0) {
                det_projF3inv = projF3inv.determinant();
            }

            outfile << std::fixed << std::setprecision(20)
                    << Ecm << '\t'
                    << En  << '\t'
                    << det_projF3inv.real() << '\t'
                    << det_projF3inv.imag() << "\n";

            std::cout << std::fixed << std::setprecision(20)
                      << Ecm << '\t'
                      << En  << '\t'
                      << det_projF3inv.real() << '\t'
                      << det_projF3inv.imag() << "\n";
        }
        catch (const std::exception& e) {
            std::cerr << "Projection error i=" << i
                      << " irrep=" << irrep
                      << ": " << e.what() << "\n";
        }

        if (idx % 100 == 0)
            std::cout << "Progress: " << idx << "/" << valid_indices.size()
                      << "  Ecm=" << std::fixed << std::setprecision(6)
                      << Ecm << "\n" << std::endl;
    }

    outfile.close();

    std::cout << "\nDone.\n";
    std::cout << "Output: " << fname << "\n";
}


void test_projections_cpu_v3() {
    
    comp pi = std::acos(-1.0); 
    double atmpi = 0.06906;
    double atmK  = 0.09698;

    double eta_1 = 1.0;
    double eta_2 = 0.5;

    double alpha         = 0.5;
    double max_shell_num = 20.0;

    std::vector<int> waves_vec_1 = {0, 1};
    std::vector<int> waves_vec_2 = {0};

    double epsilon_h = 0;
    bool   Q0norm    = true;

    double xi    = 3.444;
    double Lbyas = 20;
    double L     = xi * Lbyas;

    comp twopibyL = ((comp)2.0) * pi / ((comp)L);

    Vec3 nnP = {0, 0, 1};
    std::vector<comp> total_P(3);
    total_P[0] = twopibyL * (double)nnP[0];
    total_P[1] = twopibyL * (double)nnP[1];
    total_P[2] = twopibyL * (double)nnP[2];

    double tolerance = 0.0;

    std::vector<std::vector<comp>> scatter_params_1(4, std::vector<comp>(3));
    scatter_params_1[0][0] = 4.04;
    scatter_params_1[1][0] = -43.2;

    std::vector<std::vector<comp>> scatter_params_2(4, std::vector<comp>(3));
    scatter_params_2[0][0] = 4.12;

    double En_initial = 0.26310;
    double En_final   = 0.36;
    int    En_points  = 10;
    double del_En     = std::abs(En_initial - En_final) / (double)En_points;

    std::vector<std::string> irreps = {"A2"};

    //==========================================================================
    // STEP 1: Build all F3 ingredients + CPU inversion of Hmat
    //==========================================================================
    std::vector<std::vector<std::vector<comp>>> plm_vec(
        En_points, std::vector<std::vector<comp>>(5));
    std::vector<std::vector<std::vector<comp>>> klm_vec(
        En_points, std::vector<std::vector<comp>>(5));

    std::vector<Eigen::MatrixXcd> F2_vec(En_points);
    std::vector<Eigen::MatrixXcd> G_vec(En_points);
    std::vector<Eigen::MatrixXcd> K2inv_vec(En_points);
    std::vector<Eigen::MatrixXcd> Hmat_vec(En_points);
    std::vector<Eigen::MatrixXcd> Hmatinv_vec(En_points);
    std::vector<Eigen::MatrixXcd> F3_vec(En_points);

    std::cout << "Building Hmat and inverting on CPU...\n" << std::endl;

    for (int i = 0; i < En_points; i++) {
        double En = En_initial + i * del_En;

        std::vector<std::vector<comp>> plm_config(5), klm_config(5);
        std::vector<std::vector<int>> np_config(5), nk_config(5);

        config_maker_4(plm_config, np_config, waves_vec_1, En, total_P,
                       atmK, atmK, atmpi, L, epsilon_h, max_shell_num, tolerance);
        config_maker_4(klm_config, nk_config, waves_vec_2, En, total_P,
                       atmpi, atmK, atmK, L, epsilon_h, max_shell_num, tolerance);

        plm_vec[i] = std::move(plm_config);
        klm_vec[i] = std::move(klm_config);

        int dim1 = (int)plm_vec[i][0].size();
        int dim2 = (int)klm_vec[i][0].size();
        int tot  = dim1 + dim2;

        F2_vec[i].resize(tot, tot);
        G_vec[i].resize(tot, tot);
        K2inv_vec[i].resize(tot, tot);
        Hmat_vec[i].resize(tot, tot);

        F2_2plus1_mat(F2_vec[i], En, plm_vec[i], klm_vec[i],
                      total_P, atmK, atmpi, L, alpha, epsilon_h, max_shell_num, Q0norm);

        K2inv_EREord2_2plus1_mat(K2inv_vec[i], eta_1, eta_2,
                                 scatter_params_1, scatter_params_2,
                                 En, plm_vec[i], klm_vec[i],
                                 total_P, atmK, atmpi, epsilon_h, L);

        G_2plus1_mat(G_vec[i], En, plm_vec[i], klm_vec[i],
                     total_P, atmK, atmpi, L, alpha, epsilon_h, max_shell_num, Q0norm);

        Hmat_vec[i] = K2inv_vec[i] + F2_vec[i] + G_vec[i];

        if (tot > 0) {
            Hmatinv_vec[i] = Hmat_vec[i].inverse();
        } else {
            Hmatinv_vec[i].resize(0, 0);
        }

        if (i % 100 == 0) {
            std::cout << "Hmat CPU inversion progress: " << i << "/" << En_points
                      << "  size=" << tot << "\n";
        }
    }

    // Build F3_vec = F2/3 - F2 * Hmatinv * F2
    for (int i = 0; i < En_points; i++) {
        if (F2_vec[i].rows() > 0) {
            Eigen::MatrixXcd& F2 = F2_vec[i];
            F3_vec[i] = F2 / 3.0 - F2 * Hmatinv_vec[i] * F2;
        } else {
            F3_vec[i].resize(0, 0);
        }
    }
    std::cout << "F3_vec built for all energies\n" << std::endl;

    //==========================================================================
    // STEP 2: Build valid_indices for invertible F3, invert only those on CPU
    //==========================================================================
    std::vector<Eigen::MatrixXcd> F3inv_vec(En_points);
    std::vector<int> valid_indices;
    valid_indices.reserve(En_points);

    std::cout << "Checking which F3 matrices are valid for inversion...\n" << std::endl;

    for (int i = 0; i < En_points; i++) {
        int dim1 = (int)plm_vec[i][0].size();
        int dim2 = (int)klm_vec[i][0].size();
        int tot  = dim1 + dim2;

        if (tot == 0) continue;

        Eigen::FullPivLU<Eigen::MatrixXcd> lu(F3_vec[i]);
        if (lu.isInvertible()) {
            valid_indices.push_back(i);
        }
    }

    std::cout << "Valid F3 matrices: " << valid_indices.size()
              << " / " << En_points << "\n" << std::endl;

    for (int idx = 0; idx < (int)valid_indices.size(); idx++) {
        int i = valid_indices[idx];
        F3inv_vec[i] = F3_vec[i].inverse();

        if (idx % 100 == 0) {
            std::cout << "F3 CPU inversion progress: " << idx << "/"
                      << valid_indices.size()
                      << "  original_i=" << i
                      << "  size=" << F3_vec[i].rows() << "\n";
        }
    }

    std::cout << "F3inv_vec built for all valid energies\n" << std::endl;

    //==========================================================================
    // STEP 3: Build n_config for each energy
    //==========================================================================
    std::vector<std::vector<std::vector<int>>> np_vec(
        En_points, std::vector<std::vector<int>>(5));
    std::vector<std::vector<std::vector<int>>> nk_vec(
        En_points, std::vector<std::vector<int>>(5));

    std::cout << "Building n_configs...\n" << std::endl;
    for (int i = 0; i < En_points; i++) {
        double En  = En_initial + i * del_En;
        comp   En_c(En, 0.0);

        std::vector<std::vector<comp>> plm_tmp(5), klm_tmp(5);
        config_maker_4(plm_tmp, np_vec[i], waves_vec_1, En_c, total_P,
                       atmK, atmK, atmpi, L, epsilon_h, max_shell_num, tolerance);
        config_maker_4(klm_tmp, nk_vec[i], waves_vec_2, En_c, total_P,
                       atmpi, atmK, atmK, L, epsilon_h, max_shell_num, tolerance);
    }
    std::cout << "n_configs done\n" << std::endl;

    //==========================================================================
    // STEP 4: Open output file for fixed irrep
    //==========================================================================
    auto nnP_str = [&]() {
        return std::to_string(std::abs(nnP[0]))
             + std::to_string(std::abs(nnP[1]))
             + std::to_string(std::abs(nnP[2]));
    };

    std::string irrep = irreps[0];
    std::string fname = nnP_str() + "_" + irrep + "_det_cpu.dat";
    std::ofstream outfile(fname);

    if (!outfile.is_open())
        throw std::runtime_error("Cannot open: " + fname);

    outfile << "# Ecm\tEn\tReDet\tImDet\n";
    std::cout << "Opened: " << fname << "\n";

    //==========================================================================
    // STEP 5: Irrep projection only for valid energies
    //==========================================================================
    std::cout << "Starting irrep projections for valid energies only...\n" << std::endl;

    std::vector<int> nnP_config = {nnP[0], nnP[1], nnP[2]};
    std::vector<comp> nnP_config1 = {((comp)nnP[0]), ((comp)nnP[1]), ((comp)nnP[2])};
    bool sort_orbit_flag = false; 
    int parity = -1;

    for (int idx = 0; idx < (int)valid_indices.size(); idx++) {
        int i = valid_indices[idx];

        double En  = En_initial + i * del_En;
        comp   En_c(En, 0.0);
        comp   P2  = total_P[0]*total_P[0]
                   + total_P[1]*total_P[1]
                   + total_P[2]*total_P[2];
        double Ecm = std::real(std::sqrt(En_c*En_c - P2));

        auto& plm_config = plm_vec[i];
        auto& klm_config = klm_vec[i];
        auto& np_config  = np_vec[i];
        auto& nk_config  = nk_vec[i];

        int dim1 = (int)plm_config[0].size();
        int dim2 = (int)klm_config[0].size();

        try {
            Eigen::MatrixXcd Psub;

            P_irrep_projection_2plus1(
                Psub,
                plm_config, np_config,
                klm_config, nk_config,
                irrep, total_P, nnP_config1,
                sort_orbit_flag, parity
            );

            Eigen::MatrixXcd F3matinv = F3inv_vec[i];
            Eigen::MatrixXcd projF3inv = Psub.transpose() * F3matinv * Psub;

            comp det_projF3inv(std::numeric_limits<double>::quiet_NaN(),
                               std::numeric_limits<double>::quiet_NaN());

            if (projF3inv.rows() > 0 && projF3inv.cols() > 0) {
                det_projF3inv = projF3inv.determinant();
            }

            outfile << std::fixed << std::setprecision(20)
                    << Ecm << '\t'
                    << En  << '\t'
                    << det_projF3inv.real() << '\t'
                    << det_projF3inv.imag() << "\n";

            std::cout << std::fixed << std::setprecision(20)
                      << Ecm << '\t'
                      << En  << '\t'
                      << det_projF3inv.real() << '\t'
                      << det_projF3inv.imag() << "\n";
        }
        catch (const std::exception& e) {
            std::cerr << "Projection error i=" << i
                      << " irrep=" << irrep
                      << ": " << e.what() << "\n";
        }

        if (idx % 100 == 0) {
            std::cout << "Progress: " << idx << "/" << valid_indices.size()
                      << "  Ecm=" << std::fixed << std::setprecision(6)
                      << Ecm << "\n" << std::endl;
        }
    }

    outfile.close();

    std::cout << "\nDone.\n";
    std::cout << "Output: " << fname << "\n";
}

void print_total_dim_vs_energy(
    double Ecm_initial,
    double Ecm_final,
    int Ecm_points,
    const Vec3& nnP,
    const std::vector<int>& waves_vec_1,
    const std::vector<int>& waves_vec_2,
    double atmK,
    double atmpi,
    double L,
    double epsilon_h,
    double max_shell_num,
    double tolerance
)
{
    comp pi = std::acos(-1.0);
    comp twopibyL = ((comp)2.0) * pi / ((comp)L);

    std::vector<comp> total_P(3);
    total_P[0] = twopibyL * (double)nnP[0];
    total_P[1] = twopibyL * (double)nnP[1];
    total_P[2] = twopibyL * (double)nnP[2];

    double del_Ecm = std::abs(Ecm_initial - Ecm_final) / (double)Ecm_points;

    std::ofstream fout("total_dim_vs_energy.dat");

    if (!fout.is_open()) {
        throw std::runtime_error("Could not open total_dim_vs_energy.dat");
    }

    fout << "# i\tEn\tEcm\tdim1\tdim2\ttotal_dim\n";

    std::cout << std::scientific << std::setprecision(17);
    std::cout << "# i\tEn\tEcm\tdim1\tdim2\ttotal_dim\n";

    for (int i = 0; i < Ecm_points; ++i)
    {
        double Ecm = Ecm_initial + i * del_Ecm;
        double En  = Ecm_to_E(Ecm, total_P).real();

        std::vector<std::vector<comp>> plm_config(5), klm_config(5);
        std::vector<std::vector<int>> np_config(5), nk_config(5);

        config_maker_4(
            plm_config,
            np_config,
            waves_vec_1,
            En,
            total_P,
            atmK, atmK, atmpi,
            L,
            epsilon_h,
            max_shell_num,
            tolerance
        );

        config_maker_4(
            klm_config,
            nk_config,
            waves_vec_2,
            En,
            total_P,
            atmpi, atmK, atmK,
            L,
            epsilon_h,
            max_shell_num,
            tolerance
        );

        int dim1 = static_cast<int>(plm_config[0].size());
        int dim2 = static_cast<int>(klm_config[0].size());
        int total_dim = dim1 + dim2;

        std::cout << i << '\t'
                  << En << '\t'
                  << Ecm << '\t'
                  << dim1 << '\t'
                  << dim2 << '\t'
                  << total_dim << '\n';

        fout << i << '\t'
             << std::setprecision(17)
             << En << '\t'
             << Ecm << '\t'
             << dim1 << '\t'
             << dim2 << '\t'
             << total_dim << '\n';
    }

    fout.close();

    std::cout << "Wrote total_dim_vs_energy.dat\n";
}

void test_F3_with_pwave_all_energy_v1()
{
    char debug = 'n'; 
    comp pi = std::acos(-1.0); 
    double atmpi = 0.06906;//0.5;//0.06906;
    double atmK  = 0.09698;//1.0;//0.09698;

    double eta_1 = 1.0;
    double eta_2 = 0.5;

    double alpha         = 0.5;
    double max_shell_num = 20.0;

    std::vector<int> waves_vec_1 = {0, 1};
    std::vector<int> waves_vec_2 = {0};

    double epsilon_h = 0;
    bool   Q0norm    = true;

    double xi    = 3.444;
    double Lbyas = 20;
    double L     = xi * Lbyas; //4; //xi * Lbyas;

    comp twopibyL = ((comp)2.0) * pi / ((comp)L);

    Vec3 nnP = {0, 0, 0};
    std::vector<comp> total_P(3);
    total_P[0] = twopibyL * (double)nnP[0];
    total_P[1] = twopibyL * (double)nnP[1];
    total_P[2] = twopibyL * (double)nnP[2];
    
    std::vector<comp> total_nP(3); 
    total_nP[0] = ((comp)nnP[0]); 
    total_nP[1] = ((comp)nnP[1]);
    total_nP[2] = ((comp)nnP[2]); 

    std::cout << "total_nP = " << total_nP[0] << "," << total_nP[1] << "," << total_nP[2] << std::endl; 

    double tolerance = 1.0e-12;//0.0;

    std::vector<std::vector<comp>> scatter_params_1(4, std::vector<comp>(3));
    scatter_params_1[0][0] = 4.04;
    scatter_params_1[0][1] = 0.0;
    scatter_params_1[0][2] = 0.0;
    scatter_params_1[1][0] = -43.2;
    scatter_params_1[1][1] = 0.0;
    scatter_params_1[1][2] = 0.0;

    std::vector<std::vector<comp>> scatter_params_2(4, std::vector<comp>(3));
    scatter_params_2[0][0] = 4.12;
    scatter_params_2[0][1] = 0.0;
    scatter_params_2[0][2] = 0.0;


    double Ecm_initial = 0.26310;
    double Ecm_final   = 0.36;
    int    Ecm_points  = 1000;
    double del_Ecm     = std::abs(Ecm_initial - Ecm_final) / (double)Ecm_points;
  
    

    std::ofstream fout("000_A1m_F3_check_changedK2.dat"); 
    

    std::vector<comp> Ecm_vec; 
    std::vector<comp> det_proj_F3i_vec; 

    for(int i=0; i<(int)Ecm_points; ++i)
    {
        double Ecm = Ecm_initial + i*del_Ecm; 
        double En = Ecm_to_E( Ecm, total_P ).real();

        std::vector<std::string> irreps;// = irrep_list(nnP);
        irreps.push_back("A1u");
        //irreps.push_back("A2");
        const double SINGULAR_COND_THRESHOLD = 1e10;

        std::vector<std::vector<comp>> plm_config(5), klm_config(5);
        std::vector<std::vector<int>> np_config(5), nk_config(5); 
        config_maker_4(plm_config, np_config, waves_vec_1, En, total_P,
                    atmK, atmK, atmpi, L, epsilon_h, max_shell_num, tolerance);
        config_maker_4(klm_config, nk_config, waves_vec_2, En, total_P,
                    atmpi, atmK, atmK, L, epsilon_h, max_shell_num, tolerance);

        
        std::vector<comp> nnP_config(3); 
        for(int i=0; i<nnP.size(); ++i)
        {
            nnP_config[i] = ((comp) nnP[i]); 
        }

        int dim1 = plm_config[0].size(); 
        int dim2 = klm_config[0].size(); 
        int total_dim = dim1 + dim2; 


        std::string I = irreps[0]; 

        bool sort_orbit_flag = false; 
        int parity = -1; 
        double chop_tol = 1e-16; 
        
        double eig_tol = 0.05;
        double norm_tol = 1e-12;
        double proj_tol = 1e-10; 
        
        Eigen::MatrixXcd P_I(dim1+dim2, dim1+dim2); 

        P_irrep_projection_2plus1(P_I, plm_config, np_config, klm_config, nk_config, I, total_P, nnP_config, sort_orbit_flag, parity); 
        Eigen::MatrixXcd Vsel;
        Eigen::MatrixXcd Pproj;
        build_projector_from_eigenvectors_near_one(P_I, Vsel, Pproj, eig_tol, norm_tol, proj_tol, debug='n'); 
        
        /*
        std::cout << "P_I shape = "
          << P_I.rows() << " x " << P_I.cols() << "\n";

        std::cout << "Vsel shape = "
                << Vsel.rows() << " x " << Vsel.cols() << "\n";

        std::cout << "Pproj shape = "
                << Pproj.rows() << " x " << Pproj.cols() << "\n";

        Eigen::ComplexEigenSolver<Eigen::MatrixXcd> ces(P_I);

        std::cout << std::scientific << std::setprecision(17);

        for (int i = 0; i < ces.eigenvalues().size(); ++i) {
            auto lam = ces.eigenvalues()(i);
            std::cout << "eig[" << i << "] = " << lam
                    << "  |eig-1| = " << std::abs(lam - comp(1.0, 0.0))
                    << "  |eig| = " << std::abs(lam)
                    << "\n";
        } 
        */       


        Eigen::MatrixXcd F3mat(total_dim, total_dim);
        comp F3iso;
        Eigen::VectorXcd state_vec(total_dim);
        Eigen::MatrixXcd F2mat(total_dim, total_dim);
        Eigen::MatrixXcd K2imat(total_dim, total_dim);
        Eigen::MatrixXcd Gmat(total_dim, total_dim); 
        Eigen::MatrixXcd Hmatinv(total_dim, total_dim); 
        test_F3iso_ND_2plus1_mat_with_normalization_single_En(  F3mat, F3iso, 
                                                                state_vec, 
                                                                F2mat, K2imat, Gmat, Hmatinv, 
                                                                En, 
                                                                plm_config, klm_config, 
                                                                total_P, 
                                                                eta_1, eta_2, 
                                                                scatter_params_1, scatter_params_2,
                                                                atmK, atmpi, 
                                                                alpha, epsilon_h, 
                                                                L, max_shell_num, 
                                                                Q0norm );
        
        
        Eigen::MatrixXcd F3matinv;// = F3mat.inverse();               
        Eigen::MatrixXcd tempIdentity(dim1+dim2,dim1+dim2);
        tempIdentity.setIdentity(); 
        double relerror = 0.0;
	    LinearSolver_4(F3mat, F3matinv, tempIdentity, relerror);

        Eigen::MatrixXcd projF3i = Vsel.transpose() * F3matinv * Vsel; 
        Eigen::MatrixXcd projF2 = Vsel.transpose() * F2mat * Vsel;
        Eigen::MatrixXcd projG = Vsel.transpose() * Gmat * Vsel;

        comp det_projF3i = projF3i.determinant(); 
        comp det_projF2 = projF2.determinant(); 
        comp det_projG = projG.determinant(); 

        comp eig_val = smallest_eigenvalue(projF3i);

        
        //auto projF3 = Vsel.transpose() * F3matinv * Vsel; 
        /*
        auto [signF3, logabsdetF3] = slogdet(F3mat);
        std::cout << "En:" << En << '\t'
                  << "Ecm:" << Ecm << '\t'
                  << "F3: sign=" << signF3 << " log|det(F3)|= " << logabsdetF3 << std::endl; 
        
        auto [signHmi, logabsdetHmi] = slogdet(Hmatinv);
        std::cout << "En:" << En << '\t'
                  << "Ecm:" << Ecm << '\t'
                  << "Hmat: sign=" << signHmi << " log|det(H^{-1})|= " << logabsdetHmi << std::endl; 
    
        auto [signK2i, logabsdetK2i] = slogdet(K2imat);
        std::cout << "En:" << En << '\t'
                  << "Ecm:" << Ecm << '\t'
                  << "K2i: sign=" << signK2i << " log|det(K2i)|= " << logabsdetK2i << std::endl; 
        
        auto [signG, logabsdetG] = slogdet(Gmat);
        std::cout << "En:" << En << '\t'
                  << "Ecm:" << Ecm << '\t'
                  << "G: sign=" << signG << " log|det(G)|= " << logabsdetG << std::endl; 
        */
        std::cout << "En:" << En << '\t'
                  << "Ecm:" << Ecm << '\t'
                  << "eigval = " << eig_val << std::endl; 
        

        std::cout << std::setprecision(30); 
        std::cout << "En:" << En << '\t'
                  << "Ecm:" << Ecm << '\t'
                  << "detF3i = " << det_projF3i << std::endl;

        std::cout << "En:" << En << '\t'
                  << "Ecm:" << Ecm << '\t'
                  << "detF2 = " << det_projF2 << std::endl;
        
        std::cout << "En:" << En << '\t'
                  << "Ecm:" << Ecm << '\t'
                  << "detG = " << det_projG << std::endl;
        
        Ecm_vec.push_back(Ecm); 
        det_proj_F3i_vec.push_back(det_projF3i);
        
        fout << std::setprecision(40); 
        fout << En << '\t'
             << Ecm << '\t'
             //<< (signF3*logabsdetF3).real() << '\t'
             //<< (signHmi*logabsdetHmi).real() << '\t'
             //<< (signK2i*logabsdetK2i).real() << '\t'
             //<< (signG*logabsdetG).real() << '\t'
             << (eig_val).real() << '\t'
             << (eig_val).imag() << '\t'
             << det_projF3i.real() << '\t'
             << det_projF3i.imag() << '\t'
             << det_projF2.real()  << '\t'
             << det_projF2.imag()  << '\t'
             << det_projG.real()   << '\t'
             << det_projG.imag()  
             <<std::endl; 
        std::cout << "___________________________" << std::endl; 
    }
    fout.close(); 

    debug = 'y';

    std::vector<comp> det_proj_F3i_vec_normalized =
        normalize_det_vector_by_max(det_proj_F3i_vec, Ecm_vec, 2.0, debug);

    
    std::string output_filename = "det_proj_F3i_normalized_000_A1m_L20.dat";

    //std::ofstream 
    fout.open(output_filename.c_str());

    if (!fout.is_open())
    {
        throw std::runtime_error("Could not open output file: " + output_filename);
    }

    fout << std::setprecision(16);
    std::cout << std::setprecision(16);

    fout << "# i"
        << '\t' << "Ecm_real"
        << '\t' << "Ecm_imag"
        << '\t' << "det_norm_real"
        << '\t' << "det_norm_imag"
        << '\t' << "abs_det_norm"
        << '\n';

    if (debug == 'y')
    {
        std::cout << "# i"
                << '\t' << "Ecm_real"
                << '\t' << "Ecm_imag"
                << '\t' << "det_norm_real"
                << '\t' << "det_norm_imag"
                << '\t' << "abs_det_norm"
                << '\n';
    }

    for (std::size_t i = 0; i < det_proj_F3i_vec_normalized.size(); ++i)
    {
        fout << i
            << '\t' << Ecm_vec[i].real()
            << '\t' << Ecm_vec[i].imag()
            << '\t' << det_proj_F3i_vec_normalized[i].real()
            << '\t' << det_proj_F3i_vec_normalized[i].imag()
            << '\t' << std::abs(det_proj_F3i_vec_normalized[i])
            << '\n';

        if (debug == 'y')
        {
            std::cout << i
                    << '\t' << Ecm_vec[i].real()
                    << '\t' << Ecm_vec[i].imag()
                    << '\t' << det_proj_F3i_vec_normalized[i].real()
                    << '\t' << det_proj_F3i_vec_normalized[i].imag()
                    << '\t' << std::abs(det_proj_F3i_vec_normalized[i])
                    << '\n';
        }
    }

    fout.close();

    if (debug == 'y')
    {
        std::cout << "Saved normalized determinant vector to: "
                << output_filename << std::endl;
    }

    
}


void print_nan_inf_entries(const Eigen::MatrixXcd& A, const std::string& name)
{
    bool found = false;

    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            double re = A(i,j).real();
            double im = A(i,j).imag();

            bool bad = std::isnan(re) || std::isnan(im) ||
                       std::isinf(re) || std::isinf(im);

            if (bad) {
                std::cout << name << "(" << i << "," << j << ") = "
                          << A(i,j) << "  <-- bad entry\n";
                found = true;
            }
        }
    }

    if (!found) {
        std::cout << name << " has no NaN/Inf entries.\n";
    }
}

void debug_Gij_and_Ylm_for_ijk(
    comp En,
    comp sigma_p,
    std::vector<comp> nnp,
    std::vector<comp> nnk,
    std::vector<comp> nnP,
    int i,
    int j,
    int k,
    int ell_f,
    int proj_mf,
    int ell_i,
    int proj_mi,
    double mi,
    double mj,
    double mk,
    double L,
    double alpha
)
{
    std::cout << std::setprecision(30);

    comp pi = std::acos(-1.0);
    comp twopibyL = ((comp)2.0 * pi) / ((comp)L);
    comp Lby2pi = ((comp)L) / ((comp)2.0 * pi);

    // --------------------------------------------------
    // Convert lattice vectors to physical momenta
    // --------------------------------------------------
    std::vector<comp> pvec(3), kvec(3), Pvec(3);

    for(int a = 0; a < 3; ++a)
    {
        pvec[a] = nnp[a] * twopibyL;
        kvec[a] = nnk[a] * twopibyL;
        Pvec[a] = nnP[a] * twopibyL;
    }

    comp spec_p = std::sqrt(pvec[0]*pvec[0] + pvec[1]*pvec[1] + pvec[2]*pvec[2]);
    comp spec_k = std::sqrt(kvec[0]*kvec[0] + kvec[1]*kvec[1] + kvec[2]*kvec[2]);

    comp omp = omega_func(spec_p, mi);
    comp omk = omega_func(spec_k, mj);

    // --------------------------------------------------
    // G_ij boosts
    // --------------------------------------------------
    std::vector<comp> P_minus_p(3), P_minus_k(3);

    for(int a = 0; a < 3; ++a)
    {
        P_minus_p[a] = Pvec[a] - pvec[a];
        P_minus_k[a] = Pvec[a] - kvec[a];
    }

    std::vector<comp> nnks = boost(omk, kvec, En - omp, P_minus_p);
    std::vector<comp> nnps = boost(omp, pvec, En - omk, P_minus_k);

    // --------------------------------------------------
    // I-sum / Ylm debug part
    // This follows your print_Ylm1_Ylm2_for_ijk logic
    // --------------------------------------------------
    comp px = pvec[0], py = pvec[1], pz = pvec[2];
    comp Px = Pvec[0], Py = Pvec[1], Pz = Pvec[2];

    comp gamma = (En - omega_func(spec_p, mi)) / std::sqrt(sigma_p);

    comp xii =
        ((comp)0.5) *
        (((comp)1.0) + ((comp)(mj*mj - mk*mk)) / sigma_p);

    comp npPx = (Px - px) * Lby2pi;
    comp npPy = (Py - py) * Lby2pi;
    comp npPz = (Pz - pz) * Lby2pi;

    comp npPsq = npPx*npPx + npPy*npPy + npPz*npPz;

    comp nax = (comp)i;
    comp nay = (comp)j;
    comp naz = (comp)k;

    comp rx = nax;
    comp ry = nay;
    comp rz = naz;

    comp na_dot_npP = nax*npPx + nay*npPy + naz*npPz;
    comp prod1 = (comp)0.0;

    if(std::abs(npPsq) > 1.0e-14)
    {
        prod1 =
            (na_dot_npP / npPsq) * (((comp)1.0) / gamma - ((comp)1.0))
            - xii / gamma;

        rx = nax + npPx * prod1;
        ry = nay + npPy * prod1;
        rz = naz + npPz * prod1;
    }

    std::vector<comp> rvec(3);
    rvec[0] = rx;
    rvec[1] = ry;
    rvec[2] = rz;

    comp r2 = rx*rx + ry*ry + rz*rz;

    comp q2_p = q2psq_star(sigma_p, mj, mk);
    comp x = std::sqrt(q2_p) * Lby2pi;
    comp x2 = x*x;

    comp prop_den = x2 - r2;
    comp UV_term = std::exp(((comp)alpha) * prop_den);

    comp Ylm1 = spherical_harmonics(rvec, ell_f, proj_mf);
    comp Ylm2 = spherical_harmonics(rvec, ell_i, proj_mi);

    comp Ylm_term = Ylm1 * Ylm2;
    comp summand = UV_term * Ylm_term / prop_den;

    comp pow_term = std::pow(twopibyL, (comp)(ell_f + ell_i));
    comp total_contribution = pow_term * summand;

    // --------------------------------------------------
    // Print output like Python
    // --------------------------------------------------
    std::cout << "\n================ FULL DEBUG C++ =================\n";

    std::cout << "\n--- INPUT ---\n";
    std::cout << "E = " << En << "\n";
    std::cout << "L = " << L << "\n";
    std::cout << "Mijk = [" << mi << ", " << mj << ", " << mk << "]\n";

    std::cout << "\n--- INDICES ---\n";
    std::cout << "nnP = [" << nnP[0] << " " << nnP[1] << " " << nnP[2] << "]\n";
    std::cout << "nnk = [" << nnk[0] << " " << nnk[1] << " " << nnk[2] << "]\n";
    std::cout << "nna = [" << nax << " " << nay << " " << naz << "]\n";

    std::cout << "\n--- MOMENTA ---\n";
    std::cout << "twopibyL = " << twopibyL << "\n";
    std::cout << "pvec = [" << pvec[0] << " " << pvec[1] << " " << pvec[2] << "]\n";
    std::cout << "kvec = [" << kvec[0] << " " << kvec[1] << " " << kvec[2] << "]\n";
    std::cout << "Pvec = [" << Pvec[0] << " " << Pvec[1] << " " << Pvec[2] << "]\n";
    std::cout << "spec_p = " << spec_p << "\n";
    std::cout << "spec_k = " << spec_k << "\n";

    std::cout << "\n--- ENERGIES ---\n";
    std::cout << "omega_p = " << omp << "\n";
    std::cout << "omega_k = " << omk << "\n";
    std::cout << "E - omega_p = " << En - omp << "\n";
    std::cout << "E - omega_k = " << En - omk << "\n";

    std::cout << "\n--- RELATIVE MOMENTUM FOR YLM ---\n";
    std::cout << "npP = (P - p) * L/(2*pi) = ["
              << npPx << " " << npPy << " " << npPz << "]\n";
    std::cout << "|npP|^2 = " << npPsq << "\n";

    std::cout << "\n--- SIGMA / X / BOOST FACTORS ---\n";
    std::cout << "sigma_p = " << sigma_p << "\n";
    std::cout << "q2_p = " << q2_p << "\n";
    std::cout << "x = " << x << "\n";
    std::cout << "x2 = " << x2 << "\n";
    std::cout << "gamma = " << gamma << "\n";
    std::cout << "xii = " << xii << "\n";

    std::cout << "\n--- RVEC BUILD ---\n";
    std::cout << "dot(nna, npP) = " << na_dot_npP << "\n";
    std::cout << "prod1 = " << prod1 << "\n";
    std::cout << "rvec = [" << rx << " " << ry << " " << rz << "]\n";
    std::cout << "r2 = " << r2 << "\n";

    std::cout << "\n--- PROPAGATOR ---\n";
    std::cout << "prop_den = x2 - r2 = " << prop_den << "\n";
    std::cout << "UV_term = " << UV_term << "\n";
    std::cout << "UV/prop = " << UV_term / prop_den << "\n";

    std::cout << "\n--- SPHERICAL HARMONICS ---\n";
    std::cout << "[ell_f, m_f] = [" << ell_f << ", " << proj_mf << "]\n";
    std::cout << "[ell_i, m_i] = [" << ell_i << ", " << proj_mi << "]\n";
    std::cout << "Ylm1 = " << Ylm1 << "\n";
    std::cout << "Ylm2 = " << Ylm2 << "\n";
    std::cout << "Ylm1 * Ylm2 = " << Ylm_term << "\n";

    std::cout << "\n--- FINAL TERMS ---\n";
    std::cout << "pow_term = (2*pi/L)^(ell_f+ell_i) = " << pow_term << "\n";
    std::cout << "summand = " << summand << "\n";
    std::cout << "total contribution = " << total_contribution << "\n";

    std::cout << "\n============================================\n";

    std::cout << "\ninput nnp = ["
              << nnp[0] << " " << nnp[1] << " " << nnp[2] << "]\n";

    std::cout << "input nnk = ["
              << nnk[0] << " " << nnk[1] << " " << nnk[2] << "]\n";

    std::cout << "input nnP = ["
              << nnP[0] << " " << nnP[1] << " " << nnP[2] << "]\n\n";

    std::cout << "pvec = ["
              << pvec[0] << " " << pvec[1] << " " << pvec[2] << "]\n";

    std::cout << "kvec = ["
              << kvec[0] << " " << kvec[1] << " " << kvec[2] << "]\n";

    std::cout << "Pvec = ["
              << Pvec[0] << " " << Pvec[1] << " " << Pvec[2] << "]\n\n";

    std::cout << "omega_p = " << omp << "\n";
    std::cout << "omega_k = " << omk << "\n\n";

    std::cout << "nnks = boost(k):\n";
    std::cout << "["
              << nnks[0] << " "
              << nnks[1] << " "
              << nnks[2] << "]\n\n";

    std::cout << "nnps = boost(p):\n";
    std::cout << "["
              << nnps[0] << " "
              << nnps[1] << " "
              << nnps[2] << "]\n";
}

void print_F2_2plus1_component_for_lm(
    const Eigen::MatrixXcd &F2,
    const std::vector<std::vector<comp> > &plm_config,
    const std::vector<std::vector<comp> > &klm_config,
    int target_ell_f,
    int target_proj_mf,
    int target_ell_i,
    int target_proj_mi,
    double zero_tol = 1.0e-14
)
{
    std::cout << std::setprecision(30);

    const int size1 = static_cast<int>(plm_config[0].size());
    const int size2 = static_cast<int>(klm_config[0].size());
    const int total_size = size1 + size2;

    if(F2.rows() != total_size || F2.cols() != total_size)
    {
        std::cerr << "ERROR: F2 matrix size does not match config sizes.\n";
        std::cerr << "F2.rows() = " << F2.rows()
                  << ", F2.cols() = " << F2.cols() << "\n";
        std::cerr << "expected size = "
                  << total_size << " x " << total_size << "\n";
        return;
    }

    std::cout << "============================================================\n";
    std::cout << "Printing F2 matrix component with:\n";
    std::cout << "ell_f   = " << target_ell_f   << "\n";
    std::cout << "proj_mf = " << target_proj_mf << "\n";
    std::cout << "ell_i   = " << target_ell_i   << "\n";
    std::cout << "proj_mi = " << target_proj_mi << "\n";
    std::cout << "============================================================\n";

    int count = 0;

    // ============================================================
    // Flavor block 1: F2_1
    // global indices: [0, size1)
    // config used: plm_config
    // ============================================================

    std::cout << "\nFlavor block 1: F2_1, Mijk = [m1, m1, m2]\n";
    std::cout << "------------------------------------------------------------\n";

    for(int i = 0; i < size1; ++i)
    {
        int ell_f = static_cast<int>(plm_config[3][i].real());
        int proj_mf = static_cast<int>(plm_config[4][i].real());

        if(ell_f != target_ell_f || proj_mf != target_proj_mf)
        {
            continue;
        }

        for(int j = 0; j < size1; ++j)
        {
            int ell_i = static_cast<int>(plm_config[3][j].real());
            int proj_mi = static_cast<int>(plm_config[4][j].real());

            if(ell_i != target_ell_i || proj_mi != target_proj_mi)
            {
                continue;
            }

            comp val = F2(i, j);

            std::cout << "local indices  : i = " << i
                      << ", j = " << j << "\n";

            std::cout << "global indices : i = " << i
                      << ", j = " << j << "\n";

            std::cout << "p_f = ["
                      << plm_config[0][i] << ", "
                      << plm_config[1][i] << ", "
                      << plm_config[2][i] << "]\n";

            std::cout << "p_i = ["
                      << plm_config[0][j] << ", "
                      << plm_config[1][j] << ", "
                      << plm_config[2][j] << "]\n";

            std::cout << "ell_f, proj_mf = "
                      << ell_f << ", " << proj_mf << "\n";

            std::cout << "ell_i, proj_mi = "
                      << ell_i << ", " << proj_mi << "\n";

            std::cout << "F2(i,j) = " << val << "\n";
            std::cout << "real    = " << val.real() << "\n";
            std::cout << "imag    = " << val.imag() << "\n";
            std::cout << "abs     = " << std::abs(val) << "\n";

            if(std::abs(val) < zero_tol)
            {
                std::cout << "status  = numerically zero under tol = "
                          << zero_tol << "\n";
            }

            std::cout << "------------------------------------------------------------\n";

            ++count;
        }
    }

    // ============================================================
    // Flavor block 2: F2_2
    // global indices: [size1, size1 + size2)
    // config used: klm_config
    // ============================================================

    std::cout << "\nFlavor block 2: F2_2, Mijk = [m2, m1, m1]\n";
    std::cout << "------------------------------------------------------------\n";

    for(int i = 0; i < size2; ++i)
    {
        int ell_f = static_cast<int>(klm_config[3][i].real());
        int proj_mf = static_cast<int>(klm_config[4][i].real());

        if(ell_f != target_ell_f || proj_mf != target_proj_mf)
        {
            continue;
        }

        for(int j = 0; j < size2; ++j)
        {
            int ell_i = static_cast<int>(klm_config[3][j].real());
            int proj_mi = static_cast<int>(klm_config[4][j].real());

            if(ell_i != target_ell_i || proj_mi != target_proj_mi)
            {
                continue;
            }

            const int gi = size1 + i;
            const int gj = size1 + j;

            comp val = F2(gi, gj);

            std::cout << "local indices  : i = " << i
                      << ", j = " << j << "\n";

            std::cout << "global indices : i = " << gi
                      << ", j = " << gj << "\n";

            std::cout << "p_f = ["
                      << klm_config[0][i] << ", "
                      << klm_config[1][i] << ", "
                      << klm_config[2][i] << "]\n";

            std::cout << "p_i = ["
                      << klm_config[0][j] << ", "
                      << klm_config[1][j] << ", "
                      << klm_config[2][j] << "]\n";

            std::cout << "ell_f, proj_mf = "
                      << ell_f << ", " << proj_mf << "\n";

            std::cout << "ell_i, proj_mi = "
                      << ell_i << ", " << proj_mi << "\n";

            std::cout << "F2(i,j) = " << val << "\n";
            std::cout << "real    = " << val.real() << "\n";
            std::cout << "imag    = " << val.imag() << "\n";
            std::cout << "abs     = " << std::abs(val) << "\n";

            if(std::abs(val) < zero_tol)
            {
                std::cout << "status  = numerically zero under tol = "
                          << zero_tol << "\n";
            }

            std::cout << "------------------------------------------------------------\n";

            ++count;
        }
    }

    std::cout << "Total matching F2 components printed = " << count << "\n";
    std::cout << "============================================================\n";
}


void test_F3_with_pwave_single_energy_v1()
{
    char debug = 'n'; 
    comp pi = std::acos(-1.0); 
    double atmpi = 0.06906;//0.5;//0.06906;
    double atmK  = 0.09698;//1.0;//0.09698;

    double M1 = atmK;///atmK; 
    double M2 = atmpi;///atmK; 
    double normalizer = 1.0; 
    double eta_1 = 1.0;
    double eta_2 = 0.5;

    double alpha         = 0.5;
    double max_shell_num = 20.0;

    std::vector<int> waves_vec_1 = {0, 1};
    std::vector<int> waves_vec_2 = {0};

    double epsilon_h = 0;
    bool   Q0norm    = true;

    double xi    = 3.444;
    double Lbyas = 20;
    double L     = xi * Lbyas * normalizer;// M1;//xi * Lbyas; //4; //xi * Lbyas;

    comp twopibyL = ((comp)2.0) * pi / ((comp)L);

    Vec3 nnP = {0, 0, 1};
    std::vector<comp> total_P(3);
    total_P[0] = twopibyL * (double)nnP[0];
    total_P[1] = twopibyL * (double)nnP[1];
    total_P[2] = twopibyL * (double)nnP[2];
    
    std::vector<comp> total_nP(3); 
    total_nP[0] = ((comp)nnP[0]); 
    total_nP[1] = ((comp)nnP[1]);
    total_nP[2] = ((comp)nnP[2]); 

    std::cout << "total_nP = " << total_nP[0] << "," << total_nP[1] << "," << total_nP[2] << std::endl; 

    double tolerance = 1.0e-10;//0.0;

    std::vector<std::vector<comp>> scatter_params_1(4, std::vector<comp>(3));
    scatter_params_1[0][0] = 4.04/normalizer;//4.04;
    scatter_params_1[0][1] = 0.0;
    scatter_params_1[0][2] = 0.0;
    scatter_params_1[1][0] = -43.2/normalizer; //-43.2;
    scatter_params_1[1][1] = 0.0;
    scatter_params_1[1][2] = 0.0;

    std::vector<std::vector<comp>> scatter_params_2(4, std::vector<comp>(3));
    scatter_params_2[0][0] = 4.12/normalizer;//4.12;
    scatter_params_2[0][1] = 0.0;
    scatter_params_2[0][2] = 0.0;

    double Ecm_initial = 0.26310/normalizer;//0.26310;
    double Ecm_final   = 0.36/normalizer;//0.36;
    int    Ecm_points  = 5000;
    double del_Ecm     = std::abs(Ecm_initial - Ecm_final) / (double)Ecm_points;

    print_total_dim_vs_energy(
        Ecm_initial,
        Ecm_final,
        Ecm_points,
        nnP,
        waves_vec_1,
        waves_vec_2,
        atmK,
        atmpi,
        L,
        epsilon_h,
        max_shell_num,
        tolerance
    );

    abort(); 

    std::ofstream fout("temp_F3i_check.dat"); 

    //for(int i=0; i<(int)Ecm_points; ++i)
    {
        double Ecm = 0.333832;//0.2712/atmK;//0.2712;//0.2637783;//Ecm_initial + i*del_Ecm; 
        double En = Ecm_to_E( Ecm, total_P ).real();
        std::cout << "Ecm = " << Ecm << " En = " << En << std::endl; 
        std::vector<std::string> irreps;// = irrep_list(nnP);
        irreps.push_back("A1u");
        const double SINGULAR_COND_THRESHOLD = 1e10;

        std::vector<std::vector<comp>> plm_config(5), klm_config(5);
        std::vector<std::vector<int>> np_config(5), nk_config(5); 
        config_maker_4(plm_config, np_config, waves_vec_1, En, total_P,
                    M1, M2, M1, L, epsilon_h, max_shell_num, tolerance);
        
        //abort(); 
        
        config_maker_4(klm_config, nk_config, waves_vec_2, En, total_P,
                    M2, M1, M1, L, epsilon_h, max_shell_num, tolerance);

        print_config_5col(np_config,   "np_config");
        print_config_5col(nk_config,   "nk_config");
        std::vector<std::vector<comp>> orbit_np(3);
        bool sort_flag1 = true;  
        orbit_maker(orbit_np, np_config, total_nP, sort_flag1);

        for(int i=0; i<orbit_np[0].size(); ++i)
        {
            int opx = static_cast<int>(orbit_np[0][i].real()); 
            int opy = static_cast<int>(orbit_np[1][i].real()); 
            int opz = static_cast<int>(orbit_np[2][i].real()); 

            std::cout << "orbit = [" << opx << ','
                                     << opy << ','
                                     << opz << "]" 
                                     << std::endl;
        }


        std::cout << "plm size:" << plm_config[0].size() << std::endl;
        std::cout << "klm size:" << klm_config[0].size() << std::endl; 
        std::vector<comp> nnP_config(3); 
        for(int i=0; i<nnP.size(); ++i)
        {
            nnP_config[i] = ((comp) nnP[i]); 
        }

        int dim1 = plm_config[0].size(); 
        int dim2 = klm_config[0].size(); 
        int total_dim = dim1 + dim2; 


        std::string I = irreps[0]; 

        bool sort_orbit_flag = false; 
        int parity = -1; 
        double chop_tol = 1e-10; 
        
        double eig_tol = 0.05;
        double norm_tol = 1e-12;
        double proj_tol = 1e-10; 
        
        //Eigen::MatrixXcd P_I(dim1+dim2, dim1+dim2); 

        //P_irrep_projection_2plus1(P_I, plm_config, np_config, klm_config, nk_config, I, total_P, nnP_config, sort_orbit_flag, parity); 
        //Eigen::MatrixXcd Vsel;
        //Eigen::MatrixXcd Pproj;
        //build_projector_from_eigenvectors_near_one(P_I, Vsel, Pproj, eig_tol, norm_tol, proj_tol, debug='y'); 
        
        /*
        
        Eigen::MatrixXcd F3mat(total_dim, total_dim);
        comp F3iso;
        Eigen::VectorXcd state_vec(total_dim);
        Eigen::MatrixXcd F2mat(total_dim, total_dim);
        Eigen::MatrixXcd K2imat(total_dim, total_dim);
        Eigen::MatrixXcd Gmat(total_dim, total_dim); 
        Eigen::MatrixXcd Hmatinv(total_dim, total_dim); 
        test_F3iso_ND_2plus1_mat_with_normalization_single_En(  F3mat, F3iso, 
                                                                state_vec, 
                                                                F2mat, K2imat, Gmat, Hmatinv, 
                                                                En, 
                                                                plm_config, klm_config, 
                                                                total_P, 
                                                                eta_1, eta_2, 
                                                                scatter_params_1, scatter_params_2,
                                                                atmK, atmpi, 
                                                                alpha, epsilon_h, 
                                                                L, max_shell_num, 
                                                                Q0norm );
        
        
        Eigen::MatrixXcd F3matinv;// = F3mat.inverse();               
        Eigen::MatrixXcd tempIdentity(dim1+dim2,dim1+dim2);
        tempIdentity.setIdentity(); 
        double relerror = 0.0;
	    LinearSolver_4(F3mat, F3matinv, tempIdentity, relerror);

        
        //auto projF3 = Vsel.transpose() * F3matinv * Vsel; 
        print_bad_entries(F3mat, "F3mat");
        auto [signF3, logabsdetF3] = slogdet(F3mat);
        std::cout << "En:" << En << '\t'
                  << "Ecm:" << Ecm << '\t'
                  << "F3mat: sign=" << signF3 << "log|det(F3)|=" << logabsdetF3 << std::endl; 
        
        print_bad_entries(F3mat, "F2mat");
        auto [signF2, logabsdetF2] = slogdet(F2mat);
        std::cout << "En:" << En << '\t'
                  << "Ecm:" << Ecm << '\t'
                  << "F2mat: sign=" << signF2 << "log|det(F2)|=" << logabsdetF2 << std::endl; 
        
                          
        print_bad_entries(Hmatinv, "Hmatinv");
        auto [signHmi, logabsdetHmi] = slogdet(Hmatinv);
        std::cout << "En:" << En << '\t'
                  << "Ecm:" << Ecm << '\t'
                  << "Hmat: sign=" << signHmi << "log|det(H^{-1})|=" << logabsdetHmi << std::endl; 
    
        print_bad_entries(K2imat, "K2imat");
        auto [signK2i, logabsdetK2i] = slogdet(K2imat);
        std::cout << "En:" << En << '\t'
                  << "Ecm:" << Ecm << '\t'
                  << "K2imat: sign=" << signK2i << "log|det(K2i)|=" << logabsdetK2i << std::endl; 
        
        print_bad_entries(Gmat, "Gmat");
        auto [signG, logabsdetG] = slogdet(Gmat);
        std::cout << "En:" << En << '\t'
                  << "Ecm:" << Ecm << '\t'
                  << "Gmat: sign=" << signG << "log|det(G)|=" << logabsdetG << std::endl; 
        
        fout << En << '\t'
             << Ecm << '\t'
             << (signF3*logabsdetF3).real() << '\t'
             << (signHmi*logabsdetHmi).real() << '\t'
             << (signK2i*logabsdetK2i).real() << '\t'
             << (signG*logabsdetG).real() << '\t'
             <<std::endl; 
        std::cout << "___________________________" << std::endl; 
        */
        std::vector<std::vector<comp>> plm_config1(5), klm_config1(5);
        std::vector<std::vector<int>> np_config1(5), nk_config1(5); 
        std::vector<comp> k_test1(3); 
        std::vector<comp> p_test1(3); 
        
        comp twopibyL1 = 2.0*pi/L; 
        comp t_px = 1.0*twopibyL1; 
        comp t_py = 0.0*twopibyL1;
        comp t_pz = 1.0*twopibyL1; 
        int ell_p1 = 0; 
        int proj_p1 = 0; 
        int ell_p2 = 1; 
        int proj_p2 = -1;
        int ell_p3 = 1; 
        int proj_p3 = 0;  
        int ell_p4 = 1; 
        int proj_p4 = 1; 

        p_test1[0] = t_px; 
        p_test1[1] = t_py; 
        p_test1[2] = t_pz; 
        k_test1 = p_test1; 
        
        plm_config1[0].push_back(t_px);
        plm_config1[1].push_back(t_py);
        plm_config1[2].push_back(t_pz);
        plm_config1[3].push_back((comp)ell_p1);
        plm_config1[4].push_back((comp)proj_p1);
        
        plm_config1[0].push_back(t_px);
        plm_config1[1].push_back(t_py);
        plm_config1[2].push_back(t_pz);
        plm_config1[3].push_back((comp)ell_p2);
        plm_config1[4].push_back((comp)proj_p2);

        plm_config1[0].push_back(t_px);
        plm_config1[1].push_back(t_py);
        plm_config1[2].push_back(t_pz);
        plm_config1[3].push_back((comp)ell_p3);
        plm_config1[4].push_back((comp)proj_p3);

        plm_config1[0].push_back(t_px);
        plm_config1[1].push_back(t_py);
        plm_config1[2].push_back(t_pz);
        plm_config1[3].push_back((comp)ell_p4);
        plm_config1[4].push_back((comp)proj_p4);

        klm_config1[0].push_back(t_px);
        klm_config1[1].push_back(t_py);
        klm_config1[2].push_back(t_pz);
        klm_config1[3].push_back((comp)ell_p1);
        klm_config1[4].push_back((comp)proj_p1);

        int dim11 = plm_config[0].size();
        int dim22 = klm_config[0].size(); 
        int totdim12 = dim11 + dim22; 

        int ell_f = 0; 
        int proj_mf = 0; 
        int ell_i = 1; 
        int proj_mi = -1; 

        std::vector<comp> p = p_test1; 
        comp sigma_p = sigma_pvec_based(En, p, M1, total_P); 
        double mi = M1; 
        double mj = M1; 
        double mk = M2; 

        std::vector<comp> nnp(3); 
        nnp[0] = 1.0; 
        nnp[1] = 0.0; 
        nnp[2] = 1.0; 

        std::vector<comp> nnk(3); 
        nnk[0] = 1.0; 
        nnk[1] = 0.0; 
        nnk[2] = 1.0; 

        std::vector<comp> nnP1(3); 
        nnP1[0] = (comp) nnP[0];
        nnP1[1] = (comp) nnP[1]; 
        nnP1[2] = (comp) nnP[2]; 

        plm_config1 = plm_config;
        klm_config1 = klm_config; 

        /*
        debug_Gij_and_Ylm_for_ijk(
                    En,
                    sigma_p,
                    nnp,
                    nnk,
                    nnP1,
                    0, 2, 2,
                    ell_f,
                    proj_mf,
                    ell_i,
                    proj_mi,
                    mi,
                    mj,
                    mk,
                    L,
                    alpha
                );
        
        //abort(); 

        print_Ylm1_Ylm2_for_ijk(
                    En,
                    sigma_p,
                    p,
                    total_P,
                    0, 2, 2,
                    ell_f,
                    proj_mf,
                    ell_i,
                    proj_mi,
                    mi,
                    mj,
                    mk,
                    L
                );

        print_Gij_boosts(
                    En,
                    p_test1,
                    k_test1,
                    total_P,
                    M1,
                    M1
                );

        std::cout << "++++++================================++++++++++" << std::endl; 
        */

        Eigen::MatrixXcd F2mat1(totdim12, totdim12);
        F2_2plus1_mat( F2mat1, En, plm_config1, klm_config1, total_P, M1, M2, L, alpha, epsilon_h, 11, Q0norm);
        std::cout << "++++++================================++++++++++" << std::endl; std::cout << "++++++================================++++++++++" << std::endl; 

        Eigen::MatrixXcd Gmat1(totdim12, totdim12); 
        G_2plus1_mat(Gmat1, En, plm_config1, klm_config1, total_P, M1, M2, L, alpha, epsilon_h, max_shell_num, Q0norm);
        std::cout << "++++++================================++++++++++" << std::endl; std::cout << "++++++================================++++++++++" << std::endl; 
        
        Eigen::MatrixXcd K2imat1(totdim12, totdim12); 
        K2inv_EREord2_2plus1_mat(K2imat1, eta_1, eta_2, scatter_params_1, scatter_params_2, En, plm_config1, klm_config1, total_P, M1, M2, epsilon_h, L);
        
        std::cout << "________________________________" << std::endl; 
        std::cout << "F2mat : " << std::endl; 
        std::cout << F2mat1.real() << std::endl;
        std::cout << "F2mat shape = "
          << F2mat1.rows() << " x " << F2mat1.cols()
          << std::endl;
        std::cout << "________________________________" << std::endl; 
        
        /*
        std::cout << "++++++================================++++++++++" << std::endl; std::cout << "++++++================================++++++++++" << std::endl; 
        std::cout << "++++++================================++++++++++" << std::endl; std::cout << "++++++================================++++++++++" << std::endl; 
        std::cout << "++++++================================++++++++++" << std::endl; std::cout << "++++++================================++++++++++" << std::endl; 
        */
        // Example: print ell_f = 0, m_f = 0, ell_i = 1, m_i = -1
        /*
        print_F2_2plus1_component_for_lm(
            F2mat1,
            plm_config1,
            klm_config1,
            0,  0,
            1, -1
        );*/

        // Example: debug ell_f=0, mf=0, ell_i=1, mi=-1

        /*
        print_function_output_to_file("F2_debug_log.txt", [&]() {
            debug_F2_2plus1_component_for_lm_steps(
                F2mat1,
                plm_config1,
                klm_config1,
                En,
                total_P,
                M1,
                M2,
                L,
                alpha,
                epsilon_h,
                11,
                Q0norm,
                0,  0,
                1, -1,
                -1,      // max_terms_to_print; use -1 to print all terms
                true,    // print_each_sum_term
                1.0e-10  // comparison tolerance
            );
        });
        std::cout << "++++++================================++++++++++" << std::endl; std::cout << "++++++================================++++++++++" << std::endl; 
        std::cout << "++++++================================++++++++++" << std::endl; std::cout << "++++++================================++++++++++" << std::endl; 
        std::cout << "++++++================================++++++++++" << std::endl; std::cout << "++++++================================++++++++++" << std::endl; 
        
        */
        
        std::cout << "Gmat : " << std::endl; 
        std::cout << Gmat1.real() << std::endl;
        std::cout << "Gmat shape = "
          << Gmat1.rows() << " x " << Gmat1.cols()
          << std::endl;
        std::cout << "________________________________" << std::endl; 

        std::cout << "K2imat : " << std::endl; 
        std::cout << K2imat1.real() << std::endl;
        std::cout << "K2imat shape = "
          << K2imat1.rows() << " x " << K2imat1.cols()
          << std::endl;
        std::cout << "________________________________" << std::endl; 

        Eigen::MatrixXcd F3mat2(totdim12, totdim12);
        comp F3iso2;
        Eigen::VectorXcd state_vec2(totdim12);
        Eigen::MatrixXcd F2mat2(totdim12, totdim12);
        Eigen::MatrixXcd K2imat2(totdim12, totdim12);
        Eigen::MatrixXcd Gmat2(totdim12, totdim12); 
        Eigen::MatrixXcd Hmatinv2(totdim12, totdim12); 
        test_F3iso_ND_2plus1_mat_with_normalization_single_En(  F3mat2, F3iso2, 
                                                                state_vec2, 
                                                                F2mat2, K2imat2, Gmat2, Hmatinv2, 
                                                                En, 
                                                                plm_config1, klm_config1, 
                                                                total_P, 
                                                                eta_1, eta_2, 
                                                                scatter_params_1, scatter_params_2,
                                                                M1, M2, 
                                                                alpha, epsilon_h, 
                                                                L, max_shell_num, 
                                                                Q0norm );
        
        std::cout << "F3mat:" << std::endl; 
        std::cout << F3mat2 << std::endl; 
        double tol;      
        Eigen::MatrixXcd chopped_F3 = chop(F3mat2, tol=1e-16); 
        
        Eigen::MatrixXcd F3matinv2 = F3mat2.inverse(); 
        std::cout << "F3inv_mat:" << std::endl; 
        std::cout << F3matinv2 << std::endl; 

        std::cout << "chopped F3inv_mat:" << std::endl; 
        std::cout << chop(F3matinv2, tol=1e-16) << std::endl; 

        std::cout << "inverted chopped F3:" << std::endl; 
        std::cout << chopped_F3.inverse() << std::endl; 

        std::cout << "det F3matinv:" << std::endl;
        std::cout << F3matinv2.determinant() << std::endl; 
        
        std::cout << "det chopped F3matinv:" << std::endl;
        std::cout << (chopped_F3.inverse()).determinant() << std::endl; 



        
        abort(); 
        
        comp F2val_test = F2_ang_mom(En, k_test1, k_test1, total_P, 0, 0, 0, 0, L, M1, M1, M2, alpha, epsilon_h, max_shell_num, Q0norm );

        std::cout<< "F2 val test = " << F2val_test << std::endl; 

        comp sigp1 = sigma_pvec_based(En,p_test1,M1,total_P);
        comp sigp2 = sigma_pvec_based(En,k_test1,M2,total_P);

        std::cout << "sig1:" << sigp1 << '\t' << "sig2:" << sigp2 << std::endl; 
        
        comp Gvaltest = G_ij_lm(  En, p_test1, p_test1, total_P, 1, 0, 1, -1, M1, M1, M2, L, epsilon_h, Q0norm); 
        
        std::cout << "Gtest = " << Gvaltest << std::endl; 
        
        /*
        
        F2_2plus1_mat( F2mat, En, plm_config, klm_config, total_P, m1, m2, L, alpha, epsilon_h, max_shell_num, Q0norm);

	    K2inv_EREord2_2plus1_mat(K2imat, eta_i_1, eta_i_2, scatter_params_1, scatter_params_2, En, plm_config, klm_config, total_P, m1, m2, epsilon_h, L);

	    G_2plus1_mat(Gmat, En, plm_config, klm_config, total_P, m1, m2, L, alpha, epsilon_h, max_shell_num, Q0norm);
            
	
        */
    }
    fout.close(); 
    
}

void test_F3_with_pwave_all_energy_gpu_omp_normalized_v1( 
    std::vector<int> &nnP_vec, 
    std::string irrep, 
    std::string irrep_tag)
{
    char debug = 'y';

    comp pi = std::acos(-1.0);

    double atmpi = 0.06906;
    double atmK  = 0.09698;

    double eta_1 = 1.0;
    double eta_2 = 0.5;

    double alpha         = 0.5;
    double max_shell_num = 20.0;

    std::vector<int> waves_vec_1 = {0, 1}; // flavor 1: s + p wave
    std::vector<int> waves_vec_2 = {0};    // flavor 2: s wave

    double epsilon_h = 0.0;
    bool Q0norm = true;

    double xi    = 3.444;
    double Lbyas = 20.0;
    double L     = xi * Lbyas;

    comp twopibyL = ((comp)2.0) * pi / ((comp)L);

    Vec3 nnP = {nnP_vec[0], nnP_vec[1], nnP_vec[2]};

    std::vector<comp> total_P(3);
    total_P[0] = twopibyL * (double)nnP[0];
    total_P[1] = twopibyL * (double)nnP[1];
    total_P[2] = twopibyL * (double)nnP[2];

    std::vector<comp> nnP_config(3);
    nnP_config[0] = (comp)nnP[0];
    nnP_config[1] = (comp)nnP[1];
    nnP_config[2] = (comp)nnP[2];

    double tolerance = 1.0e-12;

    std::vector<std::vector<comp>> scatter_params_1(4, std::vector<comp>(3));
    scatter_params_1[0][0] = 4.04;
    scatter_params_1[0][1] = 0.0;
    scatter_params_1[0][2] = 0.0;

    scatter_params_1[1][0] = -43.2;
    scatter_params_1[1][1] = 0.0;
    scatter_params_1[1][2] = 0.0;

    std::vector<std::vector<comp>> scatter_params_2(4, std::vector<comp>(3));
    scatter_params_2[0][0] = 4.12;
    scatter_params_2[0][1] = 0.0;
    scatter_params_2[0][2] = 0.0;

    double Ecm_initial = 0.26310;
    double Ecm_final   = 0.36;
    int    Ecm_points  = 10000;

    double del_Ecm = std::abs(Ecm_initial - Ecm_final) / (double)Ecm_points;

    std::string I = irrep;//"A2";//"A1u";
    std::string irrep_tag_for_file = irrep_tag;//"A2";//"A1m";

    bool sort_orbit_flag = false;
    int parity = -1;

    double eig_tol  = 0.05;
    double norm_tol = 1.0e-12;
    double proj_tol = 1.0e-10;

    double max_norm_value = 2.0;

    if (debug == 'y')
    {
        std::cout << std::setprecision(16);
        std::cout << "total_nP = "
                  << nnP_config[0] << ", "
                  << nnP_config[1] << ", "
                  << nnP_config[2] << std::endl;

        std::cout << "L = " << L << std::endl;
        std::cout << "Ecm range = [" << Ecm_initial << ", "
                  << Ecm_final << "]" << std::endl;
        std::cout << "Ecm_points = " << Ecm_points << std::endl;
        std::cout << "irrep = " << I << std::endl;
    }

    //==========================================================================
    // STEP 0: Build Ecm_vec and En_vec
    //==========================================================================

    std::vector<comp> Ecm_vec(Ecm_points);
    std::vector<comp> En_vec(Ecm_points);

    for (int i = 0; i < Ecm_points; ++i)
    {
        double Ecm = Ecm_initial + i * del_Ecm;
        comp Ecm_c(Ecm, 0.0);

        comp En_c = Ecm_to_E(Ecm, total_P);

        Ecm_vec[i] = Ecm_c;
        En_vec[i]  = En_c;
    }

    //==========================================================================
    // STEP 1: Allocate all energy-dependent objects
    //==========================================================================

    std::vector<std::vector<std::vector<comp>>> plm_vec(
        Ecm_points, std::vector<std::vector<comp>>(5)
    );

    std::vector<std::vector<std::vector<comp>>> klm_vec(
        Ecm_points, std::vector<std::vector<comp>>(5)
    );

    std::vector<std::vector<std::vector<int>>> np_vec(
        Ecm_points, std::vector<std::vector<int>>(5)
    );

    std::vector<std::vector<std::vector<int>>> nk_vec(
        Ecm_points, std::vector<std::vector<int>>(5)
    );

    std::vector<Eigen::MatrixXcd> F2_vec(Ecm_points);
    std::vector<Eigen::MatrixXcd> G_vec(Ecm_points);
    std::vector<Eigen::MatrixXcd> K2inv_vec(Ecm_points);
    std::vector<Eigen::MatrixXcd> Hmat_vec(Ecm_points);
    std::vector<Eigen::MatrixXcd> Hmatinv_vec(Ecm_points);
    std::vector<Eigen::MatrixXcd> F3_vec(Ecm_points);
    std::vector<Eigen::MatrixXcd> F3inv_vec(Ecm_points);

    //==========================================================================
    // STEP 2: Build configs, F2, K2inv, G, and Hmat using OpenMP on CPU
    //==========================================================================

    if (debug == 'y')
    {
        std::cout << "Building F2, K2inv, G, Hmat for all energies using OpenMP..."
                  << std::endl;
    }

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < Ecm_points; ++i)
    {
        comp En_c = En_vec[i];

        std::vector<std::vector<comp>> plm_config(5);
        std::vector<std::vector<comp>> klm_config(5);

        std::vector<std::vector<int>> np_config(5);
        std::vector<std::vector<int>> nk_config(5);

        config_maker_4(
            plm_config, np_config,
            waves_vec_1,
            En_c,
            total_P,
            atmK, atmK, atmpi,
            L,
            epsilon_h,
            max_shell_num,
            tolerance
        );

        config_maker_4(
            klm_config, nk_config,
            waves_vec_2,
            En_c,
            total_P,
            atmpi, atmK, atmK,
            L,
            epsilon_h,
            max_shell_num,
            tolerance
        );

        plm_vec[i] = plm_config;
        klm_vec[i] = klm_config;

        np_vec[i] = np_config;
        nk_vec[i] = nk_config;

        int dim1 = (int)plm_vec[i][0].size();
        int dim2 = (int)klm_vec[i][0].size();
        int total_dim = dim1 + dim2;

        F2_vec[i].resize(total_dim, total_dim);
        G_vec[i].resize(total_dim, total_dim);
        K2inv_vec[i].resize(total_dim, total_dim);
        Hmat_vec[i].resize(total_dim, total_dim);

        F2_2plus1_mat(
            F2_vec[i],
            En_c,
            plm_vec[i],
            klm_vec[i],
            total_P,
            atmK,
            atmpi,
            L,
            alpha,
            epsilon_h,
            max_shell_num,
            Q0norm
        );

        K2inv_EREord2_2plus1_mat(
            K2inv_vec[i],
            eta_1,
            eta_2,
            scatter_params_1,
            scatter_params_2,
            En_c,
            plm_vec[i],
            klm_vec[i],
            total_P,
            atmK,
            atmpi,
            epsilon_h,
            L
        );

        G_2plus1_mat(
            G_vec[i],
            En_c,
            plm_vec[i],
            klm_vec[i],
            total_P,
            atmK,
            atmpi,
            L,
            alpha,
            epsilon_h,
            max_shell_num,
            Q0norm
        );

        Hmat_vec[i] = K2inv_vec[i] + F2_vec[i] + G_vec[i];
    }

    if (debug == 'y')
    {
        std::cout << "Finished building Hmat_vec." << std::endl;
    }

    //==========================================================================
    // STEP 3: GPU batch invert Hmat
    //
    // Hmatinv_vec[i] = Hmat_vec[i]^{-1}
    //
    // Then X = H^{-1} F2, so
    //
    // F3 = F2/3 - F2 * H^{-1} * F2
    //==========================================================================

    int chunkSize = suggest_chunk_size_from_En_final(
        Ecm_final,
        waves_vec_1,
        waves_vec_2,
        total_P,
        atmK,
        atmpi,
        L,
        epsilon_h,
        max_shell_num,
        tolerance,
        0.90,
        0.70,
        6
    );

    if (debug == 'y')
    {
        std::cout << "suggested chunkSize = " << chunkSize << std::endl;
        std::cout << "Starting GPU batched inversion of Hmat_vec..." << std::endl;
    }

    auto Hmat_builder = [&](int i, double /*dummy_E*/, Eigen::MatrixXcd& Hmat)
    {
        Hmat = Hmat_vec[i];
    };

    build_and_invert_energy_sweep_varsize_batched_gpu_v3(
        chunkSize,
        0.0,
        (double)(Ecm_points - 1),
        Ecm_points,
        Hmat_builder,
        Hmatinv_vec,
        0.90,
        16
    );

    if (debug == 'y')
    {
        std::cout << "Finished GPU batched inversion of Hmat." << std::endl;
    }

    //==========================================================================
    // STEP 4: Build F3 using OpenMP
    //
    // old CPU logic:
    //     X = solve H X = F2
    //     F3 = F2/3 - F2 * X
    //
    // here:
    //     X = H^{-1} * F2
    //     F3 = F2/3 - F2 * H^{-1} * F2
    //==========================================================================

    if (debug == 'y')
    {
        std::cout << "Building F3_vec using OpenMP..." << std::endl;
    }

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < Ecm_points; ++i)
    {
        F3_vec[i] = F2_vec[i] / 3.0 - F2_vec[i] * Hmatinv_vec[i] * F2_vec[i];
    }

    if (debug == 'y')
    {
        std::cout << "Finished building F3_vec." << std::endl;
    }

    //==========================================================================
    // STEP 5: Select valid F3 matrices for inversion
    //==========================================================================

    std::vector<int> valid_indices;

    if (debug == 'y')
    {
        std::cout << "Checking invertibility of F3 matrices..." << std::endl;
    }

    for (int i = 0; i < Ecm_points; ++i)
    {
        int dim1 = (int)plm_vec[i][0].size();
        int dim2 = (int)klm_vec[i][0].size();
        int total_dim = dim1 + dim2;

        if (total_dim == 0)
        {
            continue;
        }

        Eigen::FullPivLU<Eigen::MatrixXcd> lu(F3_vec[i]);

        if (lu.isInvertible())
        {
            valid_indices.push_back(i);
        }
        else
        {
            if (debug == 'y')
            {
                std::cout << "Skipping non-invertible F3 at i = " << i
                          << ", Ecm = " << Ecm_vec[i]
                          << ", En = " << En_vec[i] << std::endl;
            }
        }
    }

    if (debug == 'y')
    {
        std::cout << "Valid F3 matrices = "
                  << valid_indices.size()
                  << " / "
                  << Ecm_points
                  << std::endl;
    }

    //==========================================================================
    // STEP 6: GPU batch invert only valid F3 matrices
    //==========================================================================

    if (!valid_indices.empty())
    {
        int N_valid = (int)valid_indices.size();

        std::vector<Eigen::MatrixXcd> F3inv_valid(N_valid);

        auto F3mat_builder = [&](int j, double /*dummy_E*/, Eigen::MatrixXcd& F3mat)
        {
            int i = valid_indices[j];
            F3mat = F3_vec[i];
        };

        if (debug == 'y')
        {
            std::cout << "Starting GPU batched inversion of valid F3 matrices..."
                      << std::endl;
        }

        build_and_invert_energy_sweep_varsize_batched_gpu_v3(
            chunkSize,
            0.0,
            (double)(N_valid - 1),
            N_valid,
            F3mat_builder,
            F3inv_valid,
            0.90,
            16
        );

        for (int j = 0; j < N_valid; ++j)
        {
            int i = valid_indices[j];
            F3inv_vec[i] = std::move(F3inv_valid[j]);
        }

        if (debug == 'y')
        {
            std::cout << "Finished GPU batched inversion of F3." << std::endl;
        }
    }

    //==========================================================================
    // STEP 7: Project F3^{-1}, F2, G and compute determinants
    //==========================================================================

    std::vector<comp> det_proj_F3i_vec;
    std::vector<comp> Ecm_valid_vec;

    det_proj_F3i_vec.reserve(valid_indices.size());
    Ecm_valid_vec.reserve(valid_indices.size());

    std::vector<comp> det_proj_F3i_all(Ecm_points, comp(
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN()
    ));

    std::vector<comp> eig_proj_F3i_all(Ecm_points, comp(
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN()
    ));

    std::vector<comp> det_proj_F2_all(Ecm_points, comp(
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN()
    ));

    std::vector<comp> det_proj_G_all(Ecm_points, comp(
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN()
    ));

    if (debug == 'y')
    {
        std::cout << "Starting irrep projection and determinant calculation..."
                  << std::endl;
    }

    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < (int)valid_indices.size(); ++idx)
    {
        int i = valid_indices[idx];

        try
        {
            auto& plm_config = plm_vec[i];
            auto& klm_config = klm_vec[i];
            auto& np_config  = np_vec[i];
            auto& nk_config  = nk_vec[i];

            int dim1 = (int)plm_config[0].size();
            int dim2 = (int)klm_config[0].size();
            int total_dim = dim1 + dim2;

            if (total_dim == 0)
            {
                continue;
            }

            Eigen::MatrixXcd P_I(total_dim, total_dim);

            P_irrep_projection_2plus1(
                P_I,
                plm_config,
                np_config,
                klm_config,
                nk_config,
                I,
                total_P,
                nnP_config,
                sort_orbit_flag,
                parity
            );

            Eigen::MatrixXcd Vsel;
            Eigen::MatrixXcd Pproj;

            build_projector_from_eigenvectors_near_one(
                P_I,
                Vsel,
                Pproj,
                eig_tol,
                norm_tol,
                proj_tol,
                'n'
            );

            if (Vsel.cols() == 0)
            {
                continue;
            }

            Eigen::MatrixXcd projF3i = Vsel.transpose() * F3inv_vec[i] * Vsel;
            Eigen::MatrixXcd projF2  = Vsel.transpose() * F2_vec[i]    * Vsel;
            Eigen::MatrixXcd projG   = Vsel.transpose() * G_vec[i]     * Vsel;

            comp det_projF3i = comp(
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN()
            );

            comp det_projF2 = comp(
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN()
            );

            comp det_projG = comp(
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN()
            );

            comp eig_val = comp(
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN()
            );

            if (projF3i.rows() > 0 && projF3i.cols() > 0)
            {
                det_projF3i = projF3i.determinant();
                eig_val = smallest_eigenvalue(projF3i);
            }

            if (projF2.rows() > 0 && projF2.cols() > 0)
            {
                det_projF2 = projF2.determinant();
            }

            if (projG.rows() > 0 && projG.cols() > 0)
            {
                det_projG = projG.determinant();
            }

            det_proj_F3i_all[i] = det_projF3i;
            eig_proj_F3i_all[i] = eig_val;
            det_proj_F2_all[i]  = det_projF2;
            det_proj_G_all[i]   = det_projG;
        }
        catch (const std::exception& e)
        {
            if (debug == 'y')
            {
                #pragma omp critical
                {
                    std::cout << "Projection error at valid idx = " << idx
                              << ", global i = " << i
                              << ", Ecm = " << Ecm_vec[i]
                              << ", En = " << En_vec[i]
                              << ", error = " << e.what()
                              << std::endl;
                }
            }
        }
    }

    if (debug == 'y')
    {
        std::cout << "Finished irrep projection and determinant calculation."
                  << std::endl;
    }

    //==========================================================================
    // STEP 8: Save raw determinant output and build valid determinant vector
    //==========================================================================

    std::string raw_output_filename =
        "det_proj_F3i_raw_" + std::to_string(nnP[0]) + std::to_string(nnP[1]) + std::to_string(nnP[2]) + "_" + irrep_tag_for_file + "_L20.dat";

    std::ofstream fout_raw(raw_output_filename.c_str());

    if (!fout_raw.is_open())
    {
        throw std::runtime_error("Could not open output file: " + raw_output_filename);
    }

    fout_raw << std::setprecision(40);

    fout_raw << "# i"
             << '\t' << "En_real"
             << '\t' << "En_imag"
             << '\t' << "Ecm_real"
             << '\t' << "Ecm_imag"
             << '\t' << "eig_projF3i_real"
             << '\t' << "eig_projF3i_imag"
             << '\t' << "det_projF3i_real"
             << '\t' << "det_projF3i_imag"
             << '\t' << "det_projF2_real"
             << '\t' << "det_projF2_imag"
             << '\t' << "det_projG_real"
             << '\t' << "det_projG_imag"
             << '\n';

    for (int idx = 0; idx < (int)valid_indices.size(); ++idx)
    {
        int i = valid_indices[idx];

        comp det_projF3i = det_proj_F3i_all[i];

        if (!std::isfinite(det_projF3i.real()) || !std::isfinite(det_projF3i.imag()))
        {
            continue;
        }

        Ecm_valid_vec.push_back(Ecm_vec[i]);
        det_proj_F3i_vec.push_back(det_projF3i);

        fout_raw << i
                 << '\t' << En_vec[i].real()
                 << '\t' << En_vec[i].imag()
                 << '\t' << Ecm_vec[i].real()
                 << '\t' << Ecm_vec[i].imag()
                 << '\t' << eig_proj_F3i_all[i].real()
                 << '\t' << eig_proj_F3i_all[i].imag()
                 << '\t' << det_proj_F3i_all[i].real()
                 << '\t' << det_proj_F3i_all[i].imag()
                 << '\t' << det_proj_F2_all[i].real()
                 << '\t' << det_proj_F2_all[i].imag()
                 << '\t' << det_proj_G_all[i].real()
                 << '\t' << det_proj_G_all[i].imag()
                 << '\n';

        if (debug == 'y')
        {
            std::cout << std::setprecision(30);
            std::cout << "i = " << i
                      << '\t' << "En = " << En_vec[i]
                      << '\t' << "Ecm = " << Ecm_vec[i]
                      << '\t' << "eigval = " << eig_proj_F3i_all[i]
                      << '\t' << "detF3i = " << det_proj_F3i_all[i]
                      << '\t' << "detF2 = " << det_proj_F2_all[i]
                      << '\t' << "detG = " << det_proj_G_all[i]
                      << std::endl;
        }
    }

    fout_raw.close();

    if (debug == 'y')
    {
        std::cout << "Saved raw determinant file to: "
                  << raw_output_filename << std::endl;
    }

    //==========================================================================
    // STEP 9: Normalize det_proj_F3i_vec and save final normalized output
    //==========================================================================

    debug = 'y';

    std::vector<comp> det_proj_F3i_vec_normalized =
        normalize_det_vector_by_max(
            det_proj_F3i_vec,
            Ecm_valid_vec,
            max_norm_value,
            debug
        );

    std::string output_filename =
        "det_proj_F3i_normalized_" + std::to_string(nnP[0]) + std::to_string(nnP[1]) + std::to_string(nnP[2]) + "_" + irrep_tag_for_file + "_L20.dat";

    std::ofstream fout(output_filename.c_str());

    if (!fout.is_open())
    {
        throw std::runtime_error("Could not open output file: " + output_filename);
    }

    fout << std::setprecision(16);

    fout << "# i"
         << '\t' << "Ecm_real"
         << '\t' << "Ecm_imag"
         << '\t' << "det_norm_real"
         << '\t' << "det_norm_imag"
         << '\t' << "abs_det_norm"
         << '\n';

    if (debug == 'y')
    {
        std::cout << std::setprecision(16);

        std::cout << "# i"
                  << '\t' << "Ecm_real"
                  << '\t' << "Ecm_imag"
                  << '\t' << "det_norm_real"
                  << '\t' << "det_norm_imag"
                  << '\t' << "abs_det_norm"
                  << '\n';
    }

    for (std::size_t i = 0; i < det_proj_F3i_vec_normalized.size(); ++i)
    {
        fout << i
             << '\t' << Ecm_valid_vec[i].real()
             << '\t' << Ecm_valid_vec[i].imag()
             << '\t' << det_proj_F3i_vec_normalized[i].real()
             << '\t' << det_proj_F3i_vec_normalized[i].imag()
             << '\t' << std::abs(det_proj_F3i_vec_normalized[i])
             << '\n';

        if (debug == 'y')
        {
            std::cout << i
                      << '\t' << Ecm_valid_vec[i].real()
                      << '\t' << Ecm_valid_vec[i].imag()
                      << '\t' << det_proj_F3i_vec_normalized[i].real()
                      << '\t' << det_proj_F3i_vec_normalized[i].imag()
                      << '\t' << std::abs(det_proj_F3i_vec_normalized[i])
                      << '\n';
        }
    }

    fout.close();

    if (debug == 'y')
    {
        std::cout << "Saved normalized determinant vector to: "
                  << output_filename << std::endl;
    }
}




struct QCPoint
{
    double Ecm1;
    double Ecm2;
    double Ecm_zero;
    double abs_gap_det_norm_real;
};

std::string make_QC_output_filename(
    const std::vector<int>& nnP,
    const std::string& irrep_tag
)
{
    if (nnP.size() != 3)
    {
        throw std::runtime_error("nnP must have size 3.");
    }

    return "QC_points_"
         + std::to_string(nnP[0])
         + std::to_string(nnP[1])
         + std::to_string(nnP[2])
         + "_"
         + irrep_tag
         + ".dat";
}

std::vector<QCPoint> find_QC_points_from_normalized_file(
    const std::string& input_filename,
    const std::vector<int>& nnP,
    const std::string& irrep_tag,
    char debug = 'n'
)
{
    std::ifstream fin(input_filename.c_str());

    if (!fin.is_open())
    {
        throw std::runtime_error("Could not open input file: " + input_filename);
    }

    std::vector<double> Ecm_real_vec;
    std::vector<double> det_norm_real_vec;

    std::string line;

    while (std::getline(fin, line))
    {
        if (line.empty())
        {
            continue;
        }

        if (line[0] == '#')
        {
            continue;
        }

        std::istringstream iss(line);

        int i;
        double Ecm_real;
        double Ecm_imag;
        double det_norm_real;
        double det_norm_imag;
        double abs_det_norm;

        if (!(iss >> i
                  >> Ecm_real
                  >> Ecm_imag
                  >> det_norm_real
                  >> det_norm_imag
                  >> abs_det_norm))
        {
            continue;
        }

        Ecm_real_vec.push_back(Ecm_real);
        det_norm_real_vec.push_back(det_norm_real);
    }

    fin.close();

    if (Ecm_real_vec.size() != det_norm_real_vec.size())
    {
        throw std::runtime_error("Ecm_real_vec and det_norm_real_vec size mismatch.");
    }

    if (Ecm_real_vec.size() < 2)
    {
        throw std::runtime_error("Need at least two data points to search for sign flips.");
    }

    std::vector<QCPoint> qc_points;

    for (std::size_t i = 0; i + 1 < det_norm_real_vec.size(); ++i)
    {
        double x1 = Ecm_real_vec[i];
        double x2 = Ecm_real_vec[i + 1];

        double y1 = det_norm_real_vec[i];
        double y2 = det_norm_real_vec[i + 1];

        if (!std::isfinite(x1) || !std::isfinite(x2) ||
            !std::isfinite(y1) || !std::isfinite(y2))
        {
            continue;
        }

        bool exact_zero_1 = (y1 == 0.0);
        bool exact_zero_2 = (y2 == 0.0);
        bool sign_flip = (y1 * y2 < 0.0);

        if (!sign_flip && !exact_zero_1 && !exact_zero_2)
        {
            continue;
        }

        double interpolated_Ecm = std::numeric_limits<double>::quiet_NaN();

        if (exact_zero_1)
        {
            interpolated_Ecm = x1;
        }
        else if (exact_zero_2)
        {
            interpolated_Ecm = x2;
        }
        else
        {
            // Linear interpolation between:
            //
            // y(x1) = y1
            // y(x2) = y2
            //
            // The zero crossing is:
            //
            // x0 = x1 - y1 * (x2 - x1)/(y2 - y1)
            interpolated_Ecm = x1 - y1 * (x2 - x1) / (y2 - y1);
        }

        double abs_gap = std::abs(y2 - y1);

        QCPoint point;
        point.Ecm1 = x1;
        point.Ecm2 = x2;
        point.Ecm_zero = interpolated_Ecm;
        point.abs_gap_det_norm_real = abs_gap;

        qc_points.push_back(point);

        if (debug == 'y')
        {
            std::cout << std::setprecision(17);
            std::cout << "sign flip / zero found at index pair "
                      << i << ", " << i + 1 << "\n";
            std::cout << "Ecm1 = " << x1
                      << ", Ecm2 = " << x2
                      << ", y1 = " << y1
                      << ", y2 = " << y2
                      << ", interpolated Ecm zero = " << interpolated_Ecm
                      << ", abs_gap = " << abs_gap
                      << "\n";
        }
    }

    std::sort(
        qc_points.begin(),
        qc_points.end(),
        [](const QCPoint& a, const QCPoint& b)
        {
            return a.abs_gap_det_norm_real < b.abs_gap_det_norm_real;
        }
    );

    std::string output_filename = make_QC_output_filename(nnP, irrep_tag);

    std::ofstream fout(output_filename.c_str());

    if (!fout.is_open())
    {
        throw std::runtime_error("Could not open output file: " + output_filename);
    }

    fout << std::setprecision(17);

    fout << "# Ecm_real_1"
         << '\t' << "Ecm_real_2"
         << '\t' << "interpolated_Ecm_real_zero"
         << '\t' << "abs_gap_det_norm_real"
         << '\n';

    for (const auto& point : qc_points)
    {
        fout << point.Ecm1
             << '\t' << point.Ecm2
             << '\t' << point.Ecm_zero
             << '\t' << point.abs_gap_det_norm_real
             << '\n';
    }

    fout.close();

    if (debug == 'y')
    {
        std::cout << "Found " << qc_points.size()
                  << " sign-flip / zero-crossing points.\n";

        std::cout << "Saved sorted QC points to: "
                  << output_filename << std::endl;
    }

    return qc_points;
}



int main()
{

    //From this file :
    //test_projections_gpu_v3();
    //test_projections_cpu_v3();
    //test_F3_with_pwave_all_energy_v1();
    //test_F3_with_pwave_single_energy_v1(); 


    //testing GPU F3 codes
    std::vector<int> nnP_vec1 = {0,0,0}; 
    std::vector<int> nnP_vec2 = {0,0,1}; 
    std::vector<int> nnP_vec3 = {1,1,0}; 
    std::vector<int> nnP_vec4 = {1,1,1}; 
    std::vector<int> nnP_vec5 = {0,0,2}; 
    std::string irrep1 = "A1u"; 
    std::string irrep1tag = "A1m"; 
    std::string irrep2 = "A2"; 
    std::string irrep2tag = "A2"; 
    //test_F3_with_pwave_all_energy_gpu_omp_cublas_pipeline_v3(nnP_vec1, irrep1, irrep1tag);
    //test_F3_with_pwave_all_energy_gpu_omp_cublas_pipeline_v3(nnP_vec2, irrep2, irrep2tag);
    //test_F3_with_pwave_all_energy_gpu_omp_cublas_pipeline_v3(nnP_vec3, irrep2, irrep2tag);
    //test_F3_with_pwave_all_energy_gpu_omp_cublas_pipeline_v3(nnP_vec4, irrep2, irrep2tag);
    //test_F3_with_pwave_all_energy_gpu_omp_cublas_pipeline_v3(nnP_vec5, irrep2, irrep2tag);

    //v7
    //test_F3_with_pwave_all_energy_gpu_omp_cublas_adaptive_zero_search_v1(nnP_vec1, irrep1, irrep1tag);


    //v8
    test_F3_with_pwave_all_energy_gpu_omp_cublas_adaptive_zero_search_v1(
        nnP_vec1,
        irrep1,
        irrep1tag
    );

    /*
    //testing QC_points code
    char debug = 'y';

    std::vector<int> nnP = {1, 1, 0};
    std::string irrep_tag = "A2";

    std::string input_filename =
        "det_proj_F3i_normalized_110_A2_L20.dat";

    std::vector<QCPoint> qc_points =
        find_QC_points_from_normalized_file(
            input_filename,
            nnP,
            irrep_tag,
            debug
        );

    //testing QC_points code 2
    debug = 'y';

    std::vector<SignFlipCandidate> candidates =
        classify_sign_flips_zero_vs_pole(
            "det_proj_F3i_normalized_110_A2_L20.dat",
            "classified_zero_pole_candidates_110_A2.dat",
            3,        // window size
            1.0e-9,   // small_y_threshold
            1.0e1,    // spike_ratio_threshold
            1.0e1,    // slope_ratio_threshold
            debug
        );

    */

    /*
    char debug = 'y'; 
    std::vector<comp> test_vec; 
    print_and_test_vector_sum( 2147483, test_vec); 
    comp test_vec_sum =
    precise_vector_sum_debug(
        test_vec,
        "test_vec debug",
        debug
    );
    */

    //From projections_v1.hpp :
    //test_P_I_v1();
    
    return 0; 
}