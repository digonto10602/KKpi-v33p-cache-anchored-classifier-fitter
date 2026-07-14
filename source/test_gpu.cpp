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
#include "projections_from_config.hpp"
#include "eigenvalue_tracker.hpp"
#include "projections_v1.hpp"


void test_spherical_functions()
{
    comp px = 1.0; 
    comp py = 0.0; 
    comp pz = 0.0; 

    std::vector<comp> p(3);
    p[0] = px; 
    p[1] = py; 
    p[2] = pz; 

    for(int ell=0; ell<3; ++ell)
    {
        std::vector<int> m_vec; 

        ell_m_vector(ell, m_vec);

        for(int m_ind=0; m_ind<m_vec.size(); ++m_ind)
        {
            int m = m_vec[m_ind]; 
            std::cout<<"ell = "<<ell<<'\t'
                     <<"m = "<<m_vec[m_ind]<<'\t'
                     <<spherical_harmonics(p, ell, m).real()<<'\t'
                     <<std::endl; 
        }

    }
}

void test_F2_ang_mom_function()
{
    double atmpi = 0.06906;
    double atmK = 0.09698;

    double mi = atmK; 
    double mj = atmK; 
    double mk = atmpi; 

    double En = 0.22; 
    //double total_P = 0.0; 
    double alpha = 0.5; 
    double max_shell_num = 20.0; 
    double kx = 0.0; 
    double ky = 0.0; 
    double kz = 0.0; 
    double px = 0.0; 
    double py = 0.0; 
    double pz = 0.0; 


    std::vector<comp> p(3); 
    std::vector<comp> k(3); 
    std::vector<comp> total_P(3); 
    p[0] = px; 
    p[1] = py; 
    p[2] = pz; 
    k[0] = kx; 
    k[1] = ky; 
    k[2] = kz; 
    total_P[0] = 0.0; 
    total_P[1] = 0.0; 
    total_P[2] = 0.0; 


    int ell_f = 1; 
    int proj_mf = +1; 
    int ell_i = 2; 
    int proj_mi = -1; 

    double epsilon_h = 0; 
    bool Q0norm = true; 

    double Lbyas = 20; 
    double xi = 3.444; 
    double L = xi*Lbyas; 

    comp F2 = F2_ang_mom(En, k, p, total_P, ell_f, proj_mf, ell_i, proj_mi, L, mi, mj, mk, alpha, epsilon_h, max_shell_num, Q0norm);

    std::cout<<"F2 = "<<F2<<std::endl; 




}

void test_config_maker_3()
{
    double atmpi = 0.06906;
    double atmK = 0.09698;

    double mi = atmK; 
    double mj = atmK; 
    double mk = atmpi; 

    double En = 0.22; 
    //double total_P = 0.0; 
    double alpha = 0.5; 
    double max_shell_num = 20.0; 
    
    std::vector<comp> total_P(3); 
    
    total_P[0] = 0.0; 
    total_P[1] = 0.0; 
    total_P[2] = 0.0; 


    std::vector<int> waves_vec(2); 
    waves_vec[0] = 0; 
    waves_vec[1] = 1; 

    double epsilon_h = 0; 
    bool Q0norm = true; 

    double Lbyas = 20; 
    double xi = 3.444; 
    double L = xi*Lbyas; 

    double max_num_shell = 20.0; 
    double tolerance = 0.0; 

    std::vector< std::vector<comp> > plm_config(5, std::vector<comp> ()); 

    config_maker_3(plm_config, waves_vec, En, total_P, mi, mj, mk, L, epsilon_h, max_num_shell, tolerance); 

    for(int i=0; i<plm_config[0].size(); ++i)
    {
        comp px = plm_config[0][i]; 
        comp py = plm_config[1][i]; 
        comp pz = plm_config[2][i]; 
        int ell = static_cast<int>(plm_config[3][i].real()); 
        int proj_m = static_cast<int>(plm_config[4][i].real()); 

        std::cout << "px = " << px << " "
                  << "py = " << py << " "
                  << "pz = " << pz << " "
                  << "ell = " << ell << " "
                  << "proj_m = " << proj_m << std::endl; 


    }
}

void test_F2_ang_mat()
{
    double atmpi = 0.06906;
    double atmK = 0.09698;

    double mi = atmK; 
    double mj = atmK; 
    double mk = atmpi; 

    double En = 0.22; 
    //double total_P = 0.0; 
    double alpha = 0.5; 
    double max_shell_num = 2.0; 
    
    std::vector<comp> total_P(3); 
    
    total_P[0] = 0.0; 
    total_P[1] = 0.0; 
    total_P[2] = 0.0; 


    std::vector<int> waves_vec(2); 
    waves_vec[0] = 0; 
    waves_vec[1] = 1; 

    double epsilon_h = 0; 
    bool Q0norm = true; 

    double Lbyas = 20; 
    double xi = 3.444; 
    double L = xi*Lbyas; 

    double tolerance = 0.0; 

    std::vector< std::vector<comp> > plm_config(5, std::vector<comp> ()); 

    config_maker_3(plm_config, waves_vec, En, total_P, mi, mj, mk, L, epsilon_h, max_shell_num, tolerance); 

    int dim = plm_config[0].size(); 

    Eigen::MatrixXcd F2_1_mat(dim, dim); 
    F2_i_ang_mom_mat(F2_1_mat, En, plm_config, plm_config,
                     total_P, mi, mj, mk, L, alpha, 
                     epsilon_h, max_shell_num, Q0norm); 

    std::cout<<"F2_i_mat: "<<std::endl; 
    std::cout<<F2_1_mat<<std::endl; 

}

void test_F2_2plus1_mat()
{
    double atmpi = 0.06906;
    double atmK = 0.09698;

    double mi = atmK; 
    double mj = atmK; 
    double mk = atmpi; 

    double En = 0.22; 
    //double total_P = 0.0; 
    double alpha = 0.5; 
    double max_shell_num = 2.0; 
    
    std::vector<comp> total_P(3); 
    
    total_P[0] = 0.0; 
    total_P[1] = 0.0; 
    total_P[2] = 0.0; 


    std::vector<int> waves_vec_1(2); 
    waves_vec_1[0] = 0; 
    waves_vec_1[1] = 1; 

    std::vector<int> waves_vec_2(1); 
    waves_vec_2[0] = 0; 

    double epsilon_h = 0; 
    bool Q0norm = true; 

    double Lbyas = 20; 
    double xi = 3.444; 
    double L = xi*Lbyas; 

    double tolerance = 0.0; 

    std::vector< std::vector<comp> > plm_config(5, std::vector<comp> ()); 

    config_maker_3(plm_config, waves_vec_1, En, total_P, mi, mj, mk, L, epsilon_h, max_shell_num, tolerance); 

    int dim1 = plm_config[0].size(); 

    
    Eigen::MatrixXcd F2_1_mat(dim1, dim1); 
    F2_i_ang_mom_mat(F2_1_mat, En, plm_config, plm_config,
                     total_P, mi, mj, mk, L, alpha, 
                     epsilon_h, max_shell_num, Q0norm); 

    std::cout<<"F2_i_mat: "<<std::endl; 
    std::cout<<F2_1_mat<<std::endl; 

    mi = atmpi; 
    mj = atmK; 
    mk = atmK; 

    std::vector< std::vector<comp> > klm_config(5, std::vector<comp> ()); 

    config_maker_3(klm_config, waves_vec_1, En, total_P, mi, mj, mk, L, epsilon_h, max_shell_num, tolerance); 

    int dim2 = klm_config[0].size(); 

    Eigen::MatrixXcd F2_2_mat(dim2, dim2); 
    F2_i_ang_mom_mat(F2_2_mat, En, plm_config, plm_config,
                     total_P, mi, mj, mk, L, alpha, 
                     epsilon_h, max_shell_num, Q0norm); 

    std::cout<<"F2_i_mat: "<<std::endl; 
    std::cout<<F2_2_mat<<std::endl; 

    Eigen::MatrixXcd F2_mat(dim1 + dim2, dim1 + dim2); 
    
    F2_2plus1_mat(F2_mat, En, plm_config, klm_config, total_P, atmK, atmpi, L, alpha, epsilon_h, max_shell_num, Q0norm); 

    std::cout<<"F2 2plus1 mat: "<<std::endl; 
    std::cout<<F2_mat<<std::endl; 


    




}

void test_K2inv_2plus1_mat()
{
    double atmpi = 0.06906;
    double atmK = 0.09698;

    double eta_1 = 1.0;
    double eta_2 = 0.5; 

    double mi = atmK; 
    double mj = atmK; 
    double mk = atmpi; 

    double En = 0.22; 
    //double total_P = 0.0; 
    double alpha = 0.5; 
    double max_shell_num = 20.0; 
    
    std::vector<comp> total_P(3); 
    
    total_P[0] = 0.0; 
    total_P[1] = 0.0; 
    total_P[2] = 0.0; 


    std::vector<int> waves_vec_1(2); 
    waves_vec_1[0] = 0; 
    waves_vec_1[1] = 1; 

    std::vector<int> waves_vec_2(1); 
    waves_vec_2[0] = 0; 

    double epsilon_h = 0; 
    bool Q0norm = true; 

    double Lbyas = 20; 
    double xi = 3.444; 
    double L = xi*Lbyas; 

    double tolerance = 0.0; 

    std::vector< std::vector<comp> > plm_config(5, std::vector<comp> ()); 

    config_maker_3(plm_config, waves_vec_1, En, total_P, mi, mj, mk, L, epsilon_h, max_shell_num, tolerance); 

    int dim1 = plm_config[0].size(); 

    
    mi = atmpi; 
    mj = atmK; 
    mk = atmK; 

    std::vector< std::vector<comp> > klm_config(5, std::vector<comp> ()); 

    config_maker_3(klm_config, waves_vec_2, En, total_P, mi, mj, mk, L, epsilon_h, max_shell_num, tolerance); 

    int dim2 = klm_config[0].size(); 

    


    std::vector< std::vector<comp> > scatter_params_1(4, std::vector<comp>(3)); 
    double a1_0 = 4.05;
    double a1_1 = 4.02;
    
    double zeroval1 = 0; 
    scatter_params_1[0][0] = a1_0; 
    scatter_params_1[1][0] = a1_1; 
    
    
    
    std::vector< std::vector<comp> > scatter_params_2(4, std::vector<comp>(3)); 
    double a2_0 = 4.03;
    
    //double zeroval1 = 0; 
    scatter_params_2[0][0] = a2_0; 


    Eigen::MatrixXcd K2inv_mat(dim1 + dim2, dim1 + dim2); 

    K2inv_EREord2_2plus1_mat(K2inv_mat, eta_1, eta_2, scatter_params_1, scatter_params_2, En, plm_config, klm_config, total_P, atmK, atmpi, epsilon_h, L);

    std::cout<< "K2inv = " << std::endl; 
    std::cout<< K2inv_mat << std::endl; 


    




}

void test_F3iso_2plus1_mat()
{
    double atmpi = 0.06906;
    double atmK = 0.09698;

    double eta_1 = 1.0;
    double eta_2 = 0.5; 

    double mi = atmK; 
    double mj = atmK; 
    double mk = atmpi; 

    double En = 0.2631; 
    //double total_P = 0.0; 
    double alpha = 0.5; 
    double max_shell_num = 20.0; 
    
    std::vector<comp> total_P(3); 
    
    total_P[0] = 0.0; 
    total_P[1] = 0.0; 
    total_P[2] = 0.0; 


    std::vector<int> waves_vec_1(2); 
    waves_vec_1[0] = 0; 
    waves_vec_1[1] = 1; 

    std::vector<int> waves_vec_2(1); 
    waves_vec_2[0] = 0; 

    double epsilon_h = 0; 
    bool Q0norm = true; 

    double Lbyas = 20; 
    double xi = 3.444; 
    double L = xi*Lbyas; 

    double tolerance = 0.0; 

    std::vector< std::vector<comp> > plm_config(5, std::vector<comp> ()); 

    config_maker_3(plm_config, waves_vec_1, En, total_P, mi, mj, mk, L, epsilon_h, max_shell_num, tolerance); 

    int dim1 = plm_config[0].size(); 

    
    mi = atmpi; 
    mj = atmK; 
    mk = atmK; 

    std::vector< std::vector<comp> > klm_config(5, std::vector<comp> ()); 

    config_maker_3(klm_config, waves_vec_2, En, total_P, mi, mj, mk, L, epsilon_h, max_shell_num, tolerance); 

    int dim2 = klm_config[0].size(); 

    


    std::vector< std::vector<comp> > scatter_params_1(4, std::vector<comp>(3)); 
    double a1_0 = 4.05;
    double a1_1 = 4.02;
    
    double zeroval1 = 0; 
    scatter_params_1[0][0] = a1_0; 
    scatter_params_1[1][0] = a1_1; 
    
    
    
    std::vector< std::vector<comp> > scatter_params_2(4, std::vector<comp>(3)); 
    double a2_0 = 4.03;
    
    //double zeroval1 = 0; 
    scatter_params_2[0][0] = a2_0; 



    Eigen::MatrixXcd F3mat(dim1 + dim2, dim1 + dim2);
    Eigen::MatrixXcd F2mat(dim1 + dim2, dim1 + dim2);
    Eigen::MatrixXcd K2inv_mat(dim1 + dim2, dim1 + dim2); 

    Eigen::MatrixXcd Gmat(dim1 + dim2, dim1 + dim2);
    Eigen::MatrixXcd Hmatinv(dim1 + dim2, dim1 + dim2);
    comp F3iso = {0.0,0.0}; 
    Eigen::VectorXcd state_vec(dim1 + dim2); 

    test_F3iso_ND_2plus1_mat_with_normalization_single_En(F3mat, F3iso, state_vec, F2mat, K2inv_mat, Gmat, Hmatinv, En, plm_config, klm_config, total_P, eta_1, eta_2, scatter_params_1, scatter_params_2, atmK, atmpi, alpha, epsilon_h, L, max_shell_num, Q0norm );

    std::cout<<"F3iso = "<<F3iso<<std::endl; 
    




}

//does not work

/*
void test_F3iso_gpu()
{
    double atmpi = 0.06906;
    double atmK = 0.09698;

    double eta_1 = 1.0;
    double eta_2 = 0.5; 

    double mi = atmK; 
    double mj = atmK; 
    double mk = atmpi; 

    double En = 0.2631; 
    //double total_P = 0.0; 
    double alpha = 0.5; 
    double max_shell_num = 20.0; 
    
    std::vector<comp> total_P(3); 
    
    total_P[0] = 0.0; 
    total_P[1] = 0.0; 
    total_P[2] = 0.0; 


    std::vector<int> waves_vec_1(2); 
    waves_vec_1[0] = 0; 
    waves_vec_1[1] = 1; 

    std::vector<int> waves_vec_2(1); 
    waves_vec_2[0] = 0; 

    double epsilon_h = 0; 
    bool Q0norm = true; 

    double Lbyas = 20; 
    double xi = 3.444; 
    double L = xi*Lbyas; 

    double tolerance = 0.0; 

    std::vector< std::vector<comp> > scatter_params_1(4, std::vector<comp>(3)); 
    double a1_0 = 4.05;
    double a1_1 = 4.02;
    
    double zeroval1 = 0; 
    scatter_params_1[0][0] = a1_0; 
    scatter_params_1[1][0] = a1_1; 
    
    
    
    std::vector< std::vector<comp> > scatter_params_2(4, std::vector<comp>(3)); 
    double a2_0 = 4.03;
    
    //double zeroval1 = 0; 
    scatter_params_2[0][0] = a2_0; 

    double En_initial = 0.26310;
    double En_final = 0.36;
    double En_points = 100;
    double del_En = std::abs(En_initial - En_final)/En_points; 

    std::vector<Eigen::MatrixXcd> F3mat_vec((int)En_points); 
    std::vector<Eigen::MatrixXcd> F2mat_vec((int)En_points); 
    std::vector<Eigen::MatrixXcd> K2invmat_vec((int)En_points); 
    std::vector<Eigen::MatrixXcd> Gmat_vec((int)En_points); 
    std::vector<Eigen::MatrixXcd> Hmat_vec((int)En_points); 
    std::vector<Eigen::MatrixXcd> Hinvmat_vec((int)En_points); 
    std::vector<Eigen::VectorXcd> statevec_vec((int)En_points); 

    int total_systems = (int) En_points; 
    int batch_size = 20;
    int size = 105; 

    auto matrix_generator = [&](int idx, Eigen::MatrixXcd& A, Eigen::MatrixXcd& B) {
        comp En = En_initial + (double)idx * del_En; 
        mi = atmK; 
        mj = atmK; 
        mk = atmpi; 
        std::vector< std::vector<comp> > plm_config(5, std::vector<comp> ()); 

        config_maker_3(plm_config, waves_vec_1, En, total_P, mi, mj, mk, L, epsilon_h, max_shell_num, tolerance); 

        int dim1 = plm_config[0].size(); 

    
        mi = atmpi; 
        mj = atmK; 
        mk = atmK; 

        std::vector< std::vector<comp> > klm_config(5, std::vector<comp> ()); 

        config_maker_3(klm_config, waves_vec_2, En, total_P, mi, mj, mk, L, epsilon_h, max_shell_num, tolerance); 

        int dim2 = klm_config[0].size(); 

        int tot_dim = dim1 + dim2;
        Eigen::MatrixXcd F3mat(tot_dim, tot_dim); 
        Eigen::MatrixXcd F2mat(tot_dim, tot_dim); 
        Eigen::MatrixXcd K2invmat(tot_dim, tot_dim); 
        Eigen::MatrixXcd Gmat(tot_dim, tot_dim);
        Eigen::MatrixXcd Hmat(tot_dim, tot_dim); 
        Eigen::MatrixXcd Hmatinv(tot_dim, tot_dim); 
        Eigen::VectorXcd state_vec(tot_dim, tot_dim);

        F2_2plus1_mat( F2mat, En, plm_config, klm_config, total_P, atmK, atmpi, L, alpha, epsilon_h, max_shell_num, Q0norm);

	    K2inv_EREord2_2plus1_mat(K2invmat, eta_1, eta_2, scatter_params_1, scatter_params_2, En, plm_config, klm_config, total_P, atmK, atmpi, epsilon_h, L);

	    G_2plus1_mat(Gmat, En, plm_config, klm_config, total_P, atmK, atmpi, L, alpha, epsilon_h, max_shell_num, Q0norm);

        int eigvec_counter = 0; 
        Eigen::VectorXcd EigVec(dim1 + dim2); 
        for(int i=0;i<dim1;++i)
        {
            EigVec(i) = 1.0;
            eigvec_counter += 1; 
        }
        for(int i=eigvec_counter ;i<(dim1 + dim2); ++i)
        {
            EigVec(i) = 1.0/std::sqrt(2.0); 
        }
        
        F2mat_vec[idx] = F2mat; 
        K2invmat_vec[idx] = K2invmat;
        Gmat_vec[idx] = Gmat; 
        statevec_vec[idx] = EigVec; 

        
	    Hmat = K2invmat + F2mat + Gmat; 
        A = Hmat; 
        Eigen::MatrixXcd temp_Id(tot_dim, tot_dim);
        temp_Id.setIdentity(); 
        B = temp_Id; 

    };

    cusolverBatchedQR_pipeline(total_systems, batch_size, size, size, 
                               matrix_generator, Hinvmat_vec);



}
*/

/*
void test_F3iso_gpu_2()
{
    double atmpi = 0.06906;
    double atmK  = 0.09698;

    double eta_1 = 1.0;
    double eta_2 = 0.5;

    double alpha = 0.5;
    double max_shell_num = 20.0;

    std::vector<comp> total_P(3);
    total_P[0] = 0.0; total_P[1] = 0.0; total_P[2] = 0.0;

    std::vector<int> waves_vec_1(2); waves_vec_1[0] = 0; waves_vec_1[1] = 1;
    std::vector<int> waves_vec_2(1); waves_vec_2[0] = 0;

    double epsilon_h = 0;
    bool Q0norm = true;

    double Lbyas = 20;
    double xi = 3.444;
    double L = xi * Lbyas;

    double tolerance = 0.0;

    std::vector<std::vector<comp>> scatter_params_1(4, std::vector<comp>(3));
    scatter_params_1[0][0] = 4.05;
    scatter_params_1[1][0] = 4.02;

    std::vector<std::vector<comp>> scatter_params_2(4, std::vector<comp>(3));
    scatter_params_2[0][0] = 4.03;

    double En_initial = 0.26310;
    double En_final   = 0.36;
    int En_points     = 100;
    double del_En     = std::abs(En_initial - En_final) / (double)En_points;

    int total_systems = En_points;
    int batch_streams = 20;

    std::vector<Eigen::MatrixXcd> Hinvmat_vec(total_systems);

    // (A) size generator
    auto size_generator = [&](int idx) -> int {
        comp En = En_initial + (double)idx * del_En;

        // plm_config
        double mi = atmK, mj = atmK, mk = atmpi;
        std::vector<std::vector<comp>> plm_config(5, std::vector<comp>());
        config_maker_3(plm_config, waves_vec_1, En, total_P, mi, mj, mk, L, epsilon_h, max_shell_num, tolerance);
        int dim1 = (int)plm_config[0].size();

        // klm_config
        mi = atmpi; mj = atmK; mk = atmK;
        std::vector<std::vector<comp>> klm_config(5, std::vector<comp>());
        config_maker_3(klm_config, waves_vec_2, En, total_P, mi, mj, mk, L, epsilon_h, max_shell_num, tolerance);
        int dim2 = (int)klm_config[0].size();

        return dim1 + dim2;
    };

    // (B) matrix generator that pads to m_fixed if needed
    auto matrix_generator_padded =
        [&](int idx, int m_fixed, Eigen::MatrixXcd& A, Eigen::MatrixXcd& B)
    {
        comp En = En_initial + (double)idx * del_En;

        // true configs
        double mi = atmK, mj = atmK, mk = atmpi;
        std::vector<std::vector<comp>> plm_config(5, std::vector<comp>());
        config_maker_3(plm_config, waves_vec_1, En, total_P, mi, mj, mk, L, epsilon_h, max_shell_num, tolerance);
        int dim1 = (int)plm_config[0].size();

        mi = atmpi; mj = atmK; mk = atmK;
        std::vector<std::vector<comp>> klm_config(5, std::vector<comp>());
        config_maker_3(klm_config, waves_vec_2, En, total_P, mi, mj, mk, L, epsilon_h, max_shell_num, tolerance);
        int dim2 = (int)klm_config[0].size();

        int tot_dim = dim1 + dim2;

        Eigen::MatrixXcd F2mat(tot_dim, tot_dim);
        Eigen::MatrixXcd K2invmat(tot_dim, tot_dim);
        Eigen::MatrixXcd Gmat(tot_dim, tot_dim);

        F2_2plus1_mat(F2mat, En, plm_config, klm_config, total_P, atmK, atmpi, L, alpha, epsilon_h, max_shell_num, Q0norm);
        K2inv_EREord2_2plus1_mat(K2invmat, eta_1, eta_2, scatter_params_1, scatter_params_2, En,
                                plm_config, klm_config, total_P, atmK, atmpi, epsilon_h, L);
        G_2plus1_mat(Gmat, En, plm_config, klm_config, total_P, atmK, atmpi, L, alpha, epsilon_h, max_shell_num, Q0norm);

        Eigen::MatrixXcd Hmat = K2invmat + F2mat + Gmat;

        // Build B = Identity (so solution gives H^{-1})
        // Now pad to m_fixed if needed: A_pad = diag(H, I), B_pad = diag(I, I)
        A.setZero(m_fixed, m_fixed);
        B.setZero(m_fixed, m_fixed);

        // top-left true blocks
        A.block(0, 0, tot_dim, tot_dim) = Hmat;
        B.block(0, 0, tot_dim, tot_dim).setIdentity();

        // padded diagonal = Identity (keeps physical solution unchanged)
        for (int k = tot_dim; k < m_fixed; ++k) {
            A(k, k) = 1.0;
            B(k, k) = 1.0;
        }
    };

    // (C) Solve variable sizes
    // bucket=0 means exact grouping (best compute). If you get too many unique sizes, try bucket=16.
    int bucket = 0; // or 16
    cusolverBatchedQR_pipeline_variable_m(
        total_systems,
        batch_streams,
        /*nrhs=*///0 /*placeholder*/,
        //size_generator,
        /*matrix_generator_padded=*///nullptr,
        //Hinvmat_vec
    //);
//}


void test_F3iso_gpu_3()
{
    double atmpi = 0.06906;
    double atmK  = 0.09698;

    double eta_1 = 1.0;
    double eta_2 = 0.5;

    double alpha = 0.5;
    double max_shell_num = 20.0;

    std::vector<comp> total_P(3);
    total_P[0] = 0.0; total_P[1] = 0.0; total_P[2] = 0.0;

    std::vector<int> waves_vec_1(2); 
    waves_vec_1[0] = 0; 
    waves_vec_1[1] = 1;
    std::vector<int> waves_vec_2(1); 
    waves_vec_2[0] = 0;

    double epsilon_h = 0;
    bool Q0norm = true;

    double Lbyas = 20;
    double xi = 3.444;
    double L = xi * Lbyas;

    double tolerance = 0.0;

    std::vector<std::vector<comp>> scatter_params_1(4, std::vector<comp>(3));
    scatter_params_1[0][0] = 4.05;
    scatter_params_1[1][0] = 4.02;

    std::vector<std::vector<comp>> scatter_params_2(4, std::vector<comp>(3));
    scatter_params_2[0][0] = 4.03;

    double En_initial = 0.26310;
    double En_final   = 0.36;
    int En_points     = 10000;
    double del_En     = std::abs(En_initial - En_final) / (double)En_points;

    // Making the global vectors 
    std::vector< std::vector< std::vector<comp> > > plm_vec(En_points, 
        std::vector< std::vector<comp> >(
            5, std::vector<comp>()
        )
    ); 
    std::vector< std::vector< std::vector<comp> > > klm_vec(En_points, 
        std::vector< std::vector<comp> >(
            5, std::vector<comp>()
        )
    ); 
    std::vector< Eigen::MatrixXcd > F2_vec(En_points); 
    std::vector< Eigen::MatrixXcd > G_vec(En_points); 
    std::vector< Eigen::MatrixXcd > K2inv_vec(En_points); 
    std::vector< Eigen::MatrixXcd > Hmat_vec(En_points); 
    std::vector< Eigen::MatrixXcd > Hmatinv_vec(En_points); 
    std::vector< Eigen::MatrixXcd > F3_vec(En_points); 

    auto F3_ing_builder = [&](int i, double En, Eigen::MatrixXcd &Hmat) {

        // Build configs into locals OR directly into global storage
        std::vector<std::vector<comp>> plm_config(5), klm_config(5);
        config_maker_3(plm_config, waves_vec_1, En, total_P, atmK, atmK, atmpi, L, epsilon_h, max_shell_num, tolerance);
        config_maker_3(klm_config, waves_vec_2, En, total_P, atmpi, atmK, atmK, L, epsilon_h, max_shell_num, tolerance);

        // MOVE them into global vectors (avoid deep copy)
        plm_vec[size_t(i)] = std::move(plm_config);
        klm_vec[size_t(i)] = std::move(klm_config);

        int dim1 = int(plm_vec[size_t(i)][0].size());
        int dim2 = int(klm_vec[size_t(i)][0].size());
        int tot  = dim1 + dim2;

        // Resize global matrices once and fill in-place
        F2_vec[size_t(i)].resize(tot, tot);
        G_vec[size_t(i)].resize(tot, tot);
        K2inv_vec[size_t(i)].resize(tot, tot);
        Hmat_vec[size_t(i)].resize(tot, tot);

        F2_2plus1_mat(F2_vec[size_t(i)], En, plm_vec[size_t(i)], klm_vec[size_t(i)], total_P, atmK, atmpi, L, alpha, epsilon_h, max_shell_num, Q0norm);
        K2inv_EREord2_2plus1_mat(K2inv_vec[size_t(i)], eta_1, eta_2, scatter_params_1, scatter_params_2, En, plm_vec[size_t(i)], klm_vec[size_t(i)], total_P, atmK, atmpi, epsilon_h, L);
        G_2plus1_mat(G_vec[size_t(i)], En, plm_vec[size_t(i)], klm_vec[size_t(i)], total_P, atmK, atmpi, L, alpha, epsilon_h, max_shell_num, Q0norm);

        Hmat.resize(tot, tot);
        Hmat = K2inv_vec[size_t(i)] + F2_vec[size_t(i)] + G_vec[size_t(i)];

        Hmat_vec[size_t(i)] = Hmat;  // copy once (or move if you change API)
    };

    /*
    build_and_invert_energy_sweep_varsize_batched_gpu(
      En_initial, En_final, En_points, F3_ing_builder, Hmatinv_vec, 0.90, 16
    );
    */

    int chunkSize = suggest_chunk_size_from_En_final(
    En_final, waves_vec_1, waves_vec_2, total_P,
    atmK, atmpi, L, epsilon_h, max_shell_num, tolerance,
    0.90, 0.70, 6);
    printer("suggested chunkSize : ", chunkSize); 
    int blocksize = chunkSize; 
    printer("implemented chunksize : ", blocksize); 
    build_and_invert_energy_sweep_varsize_batched_gpu_v3(
      blocksize, 
      En_initial, En_final, En_points, F3_ing_builder, Hmatinv_vec, 0.90, 16
    );

    int stride = 100;
    for(int i=0; i<En_points; ++i)
    {
        int dim1 = plm_vec[i][0].size(); 
        int dim2 = klm_vec[i][0].size(); 
        int totsize = dim1 + dim2; 

        Eigen::MatrixXcd F3mat(totsize, totsize); 
        Eigen::MatrixXcd &F2 = F2_vec[i]; 
        
        F3mat = F2/3.0 - F2 * Hmatinv_vec[i] * F2; 
        if(i % stride == 0)
        {
            std::cout << "==============================================" << '\n'; 
            std::cout << "i: " << i << '\t'
                    << "dim1: " << dim1 << '\t'
                    << "dim2: " << dim2 << '\t'
                    << "matsize: " << totsize << "x" << totsize << '\t'
                    << "detF3: " << 
                    F3mat.sum() << '\n'; 
            std::cout << "F2: " << F2_vec[i].sum() << '\t'
                    << "G: " << G_vec[i].sum() << '\t'
                    << "K2inv: " << K2inv_vec[i].sum() << '\n'; 

            Eigen::JacobiSVD<Eigen::MatrixXcd> svd(Hmat_vec[i]); 
            comp cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1); 
            std::cout << "Hmat: " << Hmat_vec[i].sum() << '\t' 
                    << "cond no.: " << cond << '\n'; 

            Eigen::JacobiSVD<Eigen::MatrixXcd> svd1(Hmatinv_vec[i]); 
            comp cond1 = svd1.singularValues()(0) / svd1.singularValues()(svd1.singularValues().size() - 1); 
            std::cout << "Hmatinv: " << Hmatinv_vec[i].sum() << '\t'
                    << "cond no.:" << cond1 << '\n'; 
            
            std::cout << "F3: " << F3mat.sum() << '\n'; 
            std::cout << "==============================================" << '\n'; 
        }
        
    }

}

void matrix_size_generator()
{
    double atmpi = 0.06906;
    double atmK  = 0.09698;

    double eta_1 = 1.0;
    double eta_2 = 0.5;

    double alpha = 0.5;
    double max_shell_num = 20.0;

    std::vector<comp> total_P(3);
    total_P[0] = 0.0; total_P[1] = 0.0; total_P[2] = 0.0;

    std::vector<int> waves_vec_1(2); 
    waves_vec_1[0] = 0; 
    waves_vec_1[1] = 1;
    std::vector<int> waves_vec_2(1); 
    waves_vec_2[0] = 0;

    double epsilon_h = 0;
    bool Q0norm = true;

    double Lbyas = 20;
    double xi = 3.444;
    double L = xi * Lbyas;

    double tolerance = 0.0;

    std::vector<std::vector<comp>> scatter_params_1(4, std::vector<comp>(3));
    scatter_params_1[0][0] = 4.05;
    scatter_params_1[1][0] = 4.02;

    std::vector<std::vector<comp>> scatter_params_2(4, std::vector<comp>(3));
    scatter_params_2[0][0] = 4.03;

    double En_initial = 0.26310;
    double En_final   = 0.36;
    int En_points     = 1000;
    double del_En     = std::abs(En_initial - En_final) / (double)En_points;

    // Making the global vectors 
    std::vector< std::vector< std::vector<comp> > > plm_vec(En_points, 
        std::vector< std::vector<comp> >(
            5, std::vector<comp>()
        )
    ); 
    std::vector< std::vector< std::vector<comp> > > klm_vec(En_points, 
        std::vector< std::vector<comp> >(
            5, std::vector<comp>()
        )
    ); 

    
    int prev_matsize = INT_MIN; 
    std::vector<int> matsize; 
    std::vector<int> matsizecount; 
    for(int i=0; i<En_points; ++i)
    {
        double En = En_initial + i*del_En; 

        std::vector<std::vector<comp>> plm_config(5), klm_config(5);
        std::vector<std::vector<int>> np_config(3), nk_config(3);
        config_maker_4(plm_config, np_config, waves_vec_1, En, total_P, atmK, atmK, atmpi, L, epsilon_h, max_shell_num, tolerance);
        config_maker_4(klm_config, nk_config, waves_vec_2, En, total_P, atmpi, atmK, atmK, L, epsilon_h, max_shell_num, tolerance);

        int dim1 = plm_config[0].size(); 
        int dim2 = klm_config[0].size(); 
        int tot = dim1 + dim2;
        //std::cout << "i : " << i << '\t' << "tot : " << tot << std::endl; 
        int matsize_counter = 0; 
        if(i==0)
        {
            matsize.push_back(tot);
            matsize_counter = 1; 
            matsizecount.push_back(matsize_counter); 
        }
        else 
        {
            int indmatsize = 0; 
            int indmat=0; 
            for(int j=0; j<matsize.size(); ++j)
            {
                //std::cout << " j : " << j << '\t'
                //          << "matsize : " << matsize[j] << std::endl; 

                if(tot==matsize[j])
                {
                    indmatsize = 1; 
                    indmat = j; 
                }

                
                //std::cout << "indmat : " << indmatsize << std::endl;
                //std::cout << "j ind : " << indmat << std::endl; 
               
            }

            if(indmatsize==1)
            {
                matsizecount[indmat] += 1; 
            }
            else 
            {
                matsize.push_back(tot); 
                matsizecount.push_back(1); 
            }
        }
        
    }

    for(int i=0; i<matsize.size(); ++i)
    {
        std::cout << "matsize : " << matsize[i] << '\t'
                  << "count : " << matsizecount[i] << std::endl; 
    }
}

void nconfig_check()
{
    double pi = std::acos(-1.0); 
    double atmpi = 6.0/4.0; //0.06906;
    double atmK  = 1.0;//6.0/4.0;//0.09698;

    double eta_1 = 1.0;
    double eta_2 = 0.5;

    double alpha = 0.5;
    double max_shell_num = 20.0;

    

    std::vector<int> waves_vec_1(2); 
    waves_vec_1[0] = 0; 
    waves_vec_1[1] = 1;
    std::vector<int> waves_vec_2(1); 
    waves_vec_2[0] = 0;

    double epsilon_h = 0;
    bool Q0norm = true;

    double Lbyas = 20;
    double xi = 3.444;
    double L = 4.0;//xi * Lbyas;
    double twopibyL = 2.0*pi/L; 
    std::vector<comp> total_P(3);
    total_P[0] = 0.0; total_P[1] = 0.0; total_P[2] = 1.0*twopibyL;

    double tolerance = 0.0;

    std::vector<std::vector<comp>> scatter_params_1(4, std::vector<comp>(3));
    scatter_params_1[0][0] = 4.05;
    scatter_params_1[1][0] = 4.02;

    std::vector<std::vector<comp>> scatter_params_2(4, std::vector<comp>(3));
    scatter_params_2[0][0] = 4.03;

    double En_initial = 0.26310;
    double En_final   = 0.36;
    int En_points     = 1000;
    double del_En     = std::abs(En_initial - En_final) / (double)En_points;

    // Making the global vectors 
    std::vector< std::vector< std::vector<comp> > > plm_vec(En_points, 
        std::vector< std::vector<comp> >(
            5, std::vector<comp>()
        )
    ); 
    std::vector< std::vector< std::vector<comp> > > klm_vec(En_points, 
        std::vector< std::vector<comp> >(
            5, std::vector<comp>()
        )
    ); 
    comp Ecm = 4.0; 
    
    comp En = Ecm_to_E(Ecm, total_P); 

    std::vector<std::vector<comp>> plm_config(5), klm_config(5);
    std::vector<std::vector<int>> np_config(3), nk_config(3);
    config_maker_4(plm_config, np_config, waves_vec_1, En, total_P, atmK, atmK, atmpi, L, epsilon_h, max_shell_num, tolerance);
    config_maker_4(klm_config, nk_config, waves_vec_2, En, total_P, atmpi, atmK, atmK, L, epsilon_h, max_shell_num, tolerance);

    std::cout << "flavor 1"<< std::endl; 
    for(int i=0; i<np_config[0].size(); ++i)
    {
        int nx = np_config[0][i]; 
        int ny = np_config[1][i]; 
        int nz = np_config[2][i]; 

        std::cout << nx << "," << ny << "," << nz << std::endl; 
    }
    std::cout << "flavor 2"<< std::endl; 
    for(int i=0; i<nk_config[0].size(); ++i)
    {
        int nx = nk_config[0][i]; 
        int ny = nk_config[1][i]; 
        int nz = nk_config[2][i]; 

        std::cout << nx << "," << ny << "," << nz << std::endl; 
    }
    
    
    int dim1 = plm_config[0].size(); 
    int dim2 = klm_config[0].size(); 
    int tot = dim1 + dim2;
    //std::cout << "i : " << i << '\t' << "tot : " << tot << std::endl; 
      
}



void test_projections() {
    

    double pi = std::acos(-1.0); 
    double atmpi = 6.0/4.0; //0.06906;
    double atmK  = 1.0;//6.0/4.0;//0.09698;

    double eta_1 = 1.0;
    double eta_2 = 0.5;

    double alpha = 0.5;
    double max_shell_num = 20.0;

    

    std::vector<int> waves_vec_1(2); 
    waves_vec_1[0] = 0; 
    waves_vec_1[1] = 1;
    std::vector<int> waves_vec_2(1); 
    waves_vec_2[0] = 0;

    double epsilon_h = 0;
    bool Q0norm = true;

    double Lbyas = 20;
    double xi = 3.444;
    double L = 4.0;//xi * Lbyas;
    //double twopibyL = 2.0*pi/L; 
    //std::vector<comp> total_P(3);
    Vec3  nnP     = {0, 0, 1};
    double twopiL = 2.0 * PI / L;

    std::vector<comp> total_P = {
        comp(twopiL * nnP[0], 0),
        comp(twopiL * nnP[1], 0),
        comp(twopiL * nnP[2], 0)
    };

    double tolerance = 0.0;

    std::vector<std::vector<comp>> scatter_params_1(4, std::vector<comp>(3));
    scatter_params_1[0][0] = 0.15;
    scatter_params_1[1][0] = 0.2;

    std::vector<std::vector<comp>> scatter_params_2(4, std::vector<comp>(3));
    scatter_params_2[0][0] = 0.1;

    double En_initial = 0.26310;
    double En_final   = 0.36;
    int En_points     = 1000;
    double del_En     = std::abs(En_initial - En_final) / (double)En_points;

    // Making the global vectors 
    std::vector< std::vector< std::vector<comp> > > plm_vec(En_points, 
        std::vector< std::vector<comp> >(
            5, std::vector<comp>()
        )
    ); 
    std::vector< std::vector< std::vector<comp> > > klm_vec(En_points, 
        std::vector< std::vector<comp> >(
            5, std::vector<comp>()
        )
    ); 
    comp Ecm = 2.6; 
    
    comp En = Ecm_to_E(Ecm, total_P); 

    std::vector<std::vector<comp>> plm_config(5), klm_config(5);
    std::vector<std::vector<int>> np_config(3), nk_config(3);
    config_maker_4(plm_config, np_config, waves_vec_1, En, total_P, atmK, atmK, atmpi, L, epsilon_h, max_shell_num, tolerance);
    config_maker_4(klm_config, nk_config, waves_vec_2, En, total_P, atmpi, atmK, atmK, L, epsilon_h, max_shell_num, tolerance);
    int dim1 = plm_config[0].size();
    int dim2 = klm_config[0].size(); 
    // Filled by your config_maker_4 calls:
    FlavorConfig cfg1, cfg2;
    cfg1.plm_config = plm_config;
    cfg1.n_config   = np_config;
    cfg2.plm_config = klm_config;
    cfg2.n_config   = nk_config;
    
    // config_maker_4(..., waves_vec={0,1}, ...) → cfg1 (flavor-1, s+p)
    // config_maker_4(..., waves_vec={0},   ...) → cfg2 (flavor-2, s only)

    // QC3_mat: your F3inv + K3 matrix (MatrixXcd)
    Eigen::MatrixXcd F3mat(dim1 + dim2, dim1 + dim2);
    Eigen::MatrixXcd F2mat(dim1 + dim2, dim1 + dim2);
    Eigen::MatrixXcd K2inv_mat(dim1 + dim2, dim1 + dim2); 

    Eigen::MatrixXcd Gmat(dim1 + dim2, dim1 + dim2);
    Eigen::MatrixXcd Hmatinv(dim1 + dim2, dim1 + dim2);
    comp F3iso = {0.0,0.0}; 
    Eigen::VectorXcd state_vec(dim1 + dim2); 

    for(int i=0; i<dim1; ++i)
    {
        comp px = plm_config[0][i];
        comp py = plm_config[1][i]; 
        comp pz = plm_config[2][i]; 
        int ell = static_cast<int>(plm_config[3][i].real()); 
        int proj_m = static_cast<int>(plm_config[4][i].real()); 
        std::cout << "mom1: " << px << '\t' << py << '\t' << pz << '\t' << "ell:" << ell << '\t' << "m:" << proj_m << std::endl; 
    }
    for(int i=0; i<dim2; ++i)
    {
        comp px = klm_config[0][i];
        comp py = klm_config[1][i]; 
        comp pz = klm_config[2][i]; 
        int ell = static_cast<int>(klm_config[3][i].real()); 
        int proj_m = static_cast<int>(klm_config[4][i].real()); 
        
        std::cout << "mom1: " << px << '\t' << py << '\t' << pz << '\t' << "ell:" << ell << '\t' << "m:" << proj_m << std::endl; 
        
    }

    test_F3iso_ND_2plus1_mat_with_normalization_single_En(F3mat, F3iso, state_vec, F2mat, K2inv_mat, Gmat, Hmatinv, En, plm_config, klm_config, total_P, eta_1, eta_2, scatter_params_1, scatter_params_2, atmK, atmpi, alpha, epsilon_h, L, max_shell_num, Q0norm );

    std::cout<<"F3mat = \n"<<F3mat<<std::endl; 
    std::cout<<"F3matinv = \n"<<F3mat.inverse()<<std::endl;

    MatC QC3_mat = F3mat.inverse();

    //Vec3 nnP = {0, 0, 1};


    // Single irrep
    MatC M_proj = irrep_proj_2plus1_from_config(QC3_mat, nnP, "A2",
                                                cfg1, cfg2, /*parity=*/-1);

    std::cout << "F3i projected:" << std::endl; 
    std::cout << M_proj << std::endl; 
    
    /*
    if (M_A1u.cols() > 0) {
        Eigen::SelfAdjointEigenSolver<MatC> eig(M_A1u);
        std::cout << "A1u eigenvalues:\n" << eig.eigenvalues() << "\n";
    }*/

    // Or loop all irreps at once:
    print_irrep_eigenvalues(QC3_mat, nnP, cfg1, cfg2, -1);
}

void test_projections_1() {
    double pi = std::acos(-1.0); 
    double atmpi = 6.0/4.0; //0.06906;
    double atmK  = 1.0;//6.0/4.0;//0.09698;

    double eta_1 = 1.0;
    double eta_2 = 0.5;

    double alpha = 0.5;
    double max_shell_num = 20.0;

    

    std::vector<int> waves_vec_1(2); 
    waves_vec_1[0] = 0; 
    waves_vec_1[1] = 1;
    std::vector<int> waves_vec_2(1); 
    waves_vec_2[0] = 0;

    double epsilon_h = 0;
    bool Q0norm = true;

    double Lbyas = 20;
    double xi = 3.444;
    double L = 4.0;//xi * Lbyas;
    //double twopibyL = 2.0*pi/L; 
    //std::vector<comp> total_P(3);
    Vec3  nnP     = {0, 0, 1};
    double twopiL = 2.0 * PI / L;

    std::vector<comp> total_P = {
        comp(twopiL * nnP[0], 0),
        comp(twopiL * nnP[1], 0),
        comp(twopiL * nnP[2], 0)
    };

    double tolerance = 0.0;

    std::vector<std::vector<comp>> scatter_params_1(4, std::vector<comp>(3));
    scatter_params_1[0][0] = 0.15;
    scatter_params_1[1][0] = 0.2;

    std::vector<std::vector<comp>> scatter_params_2(4, std::vector<comp>(3));
    scatter_params_2[0][0] = 0.1;

    

    
    
    
    // config_maker_4(..., waves_vec={0,1}, ...) → cfg1 (flavor-1, s+p)
    // config_maker_4(..., waves_vec={0},   ...) → cfg2 (flavor-2, s only)

    

    //Vec3  nnP    = {0, 0, 1};
    std::string irrep = "A2";

    // Lambda: rebuild everything at each Ecm
    auto build_MI = [&](double Ecm) -> MatC {

        comp En = Ecm_to_E(comp(Ecm,0), total_P);

        // Rebuild configs at this energy
        std::vector<std::vector<comp>> plm_config(5), klm_config(5);
        std::vector<std::vector<int>>  np_config(3),  nk_config(3);
        std::cout << "1. Starting test_projections\n" << std::endl;
        config_maker_4(plm_config, np_config, waves_vec_1,
                       En, total_P, atmK, atmK, atmpi,
                       L, epsilon_h, max_shell_num, tolerance);
        std::cout << "2. cfg1 done, size=" << plm_config[0].size() << "\n" << std::endl;
        config_maker_4(klm_config, nk_config, waves_vec_2,
                       En, total_P, atmpi, atmK, atmK,
                       L, epsilon_h, max_shell_num, tolerance);
        std::cout << "3. cfg2 done, size=" << klm_config[0].size() << "\n" << std::endl;
        FlavorConfig cfg1, cfg2;
        cfg1.plm_config = plm_config; cfg1.n_config = np_config;
        cfg2.plm_config = klm_config; cfg2.n_config = nk_config;

        int dim1 = plm_config[0].size();
        int dim2 = klm_config[0].size();

        // Rebuild F3mat at this energy
        MatC F3mat(dim1+dim2, dim1+dim2);
        // QC3_mat: your F3inv + K3 matrix (MatrixXcd)
        //Eigen::MatrixXcd F3mat(dim1 + dim2, dim1 + dim2);
        Eigen::MatrixXcd F2mat(dim1 + dim2, dim1 + dim2);
        Eigen::MatrixXcd K2inv_mat(dim1 + dim2, dim1 + dim2); 

        Eigen::MatrixXcd Gmat(dim1 + dim2, dim1 + dim2);
        Eigen::MatrixXcd Hmatinv(dim1 + dim2, dim1 + dim2);
        comp F3iso = {0.0,0.0}; 
        Eigen::VectorXcd state_vec(dim1 + dim2); 
        test_F3iso_ND_2plus1_mat_with_normalization_single_En(F3mat, F3iso, state_vec, F2mat, K2inv_mat, Gmat, Hmatinv, En, plm_config, klm_config, total_P, eta_1, eta_2, scatter_params_1, scatter_params_2, atmK, atmpi, alpha, epsilon_h, L, max_shell_num, Q0norm );

        /*
        
        std::cout << "5. Parsing configs\n" << std::endl;
        auto basis1 = parse_config(cfg1.plm_config, cfg1.n_config);
        auto basis2 = parse_config(cfg2.plm_config, cfg2.n_config);

        auto nnk1   = unique_nnk_list(basis1);
        auto nnk2   = unique_nnk_list(basis2);
        auto lm1    = unique_lm_list(basis1);
        auto lm2    = unique_lm_list(basis2);

        std::cout << "6. nnk1=" << nnk1.size()
                << " lm1=" << lm1.size()
                << " nnk2=" << nnk2.size()
                << " lm2=" << lm2.size() << "\n" << std::endl;

        // Sanity check sizes
        int N1 = nnk1.size() * lm1.size();
        int N2 = nnk2.size() * lm2.size();
        std::cout << "7. N1=" << N1 << " N2=" << N2
                << " N1+N2=" << N1+N2
                << " F3mat=" << F3mat.rows() << "\n" << std::endl;

        MatC P_full1 = build_full_projector(nnP, "A1", nnk1, lm1, -1);
        std::cout << "8. P_full1 built " << P_full1.rows()
                << "x" << P_full1.cols() << "\n" << std::endl;

        MatC P_full2 = build_full_projector(nnP, "A1", nnk2, lm2, -1);
        std::cout << "9. P_full2 built\n" << std::endl;

        

        MatC Psub1 = subspace_from_projector(P_full1);
        std::cout << "10. subspace done, cols=" << Psub1.cols() << "\n" << std::endl;
        MatC Psub2 = subspace_from_projector(P_full2);
        std::cout << "11. subspace done, cols=" << Psub2.cols() << "\n" << std::endl;
        */
        MatC F3inv = F3mat.inverse(); 

        return irrep_proj_2plus1_from_config(F3inv, nnP, irrep,
                                              cfg1, cfg2, -1);
    };

    // Run tracker
    TrackedSpectrum spec = track_eigenvalues(build_MI, 2.6, 3.0, 10);
    //print_spectrum(spec, irrep);
    if (spec.tracks.empty()) {
    std::cout << "tracks is empty — irrep has no subspace at any energy\n";
    } else {
    std::cout << "spec size: " << spec.tracks[0].size() << "\n";
    std::cout << "num levels: " << spec.tracks.size() << "\n";
    } 

    print_eigenvalue_tracks(spec, irrep);
}

void test_projections_gpu() {
    
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

    std::vector<std::string> irreps = {"A2"};// = irrep_list(nnP);

    // Singularity threshold: F3 with condition number above this is flagged
    // and bypassed for inversion — these are near energy levels
    const double SINGULAR_COND_THRESHOLD = 1e10;

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
        config_maker_3(plm_config, waves_vec_1, En, total_P,
                       atmK, atmK, atmpi, L, epsilon_h, max_shell_num, tolerance);
        config_maker_3(klm_config, waves_vec_2, En, total_P,
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
    // STEP 2: Singularity check on F3 — flag near-singular matrices
    //         These correspond to energy levels (det F3 = 0)
    //==========================================================================
    std::vector<bool>   F3_is_singular(En_points, false);
    std::vector<double> F3_cond(En_points, 0.0);
    int n_singular = 0;

    std::cout << "Checking F3 singularity...\n" << std::endl;
    for (int i = 0; i < En_points; i++) {
        int dim1 = (int)plm_vec[i][0].size();
        int dim2 = (int)klm_vec[i][0].size();
        if (dim1 == 0 || dim2 == 0) {
            F3_is_singular[i] = true;
            continue;
        }

        // SVD-based condition number
        Eigen::JacobiSVD<Eigen::MatrixXcd> svd(F3_vec[i],
            Eigen::ComputeThinU | Eigen::ComputeThinV);
        double smin = svd.singularValues().tail(1)(0);
        double smax = svd.singularValues()(0);
        F3_cond[i]  = (smin > 1e-300) ? smax / smin
                                       : std::numeric_limits<double>::infinity();

        if (F3_cond[i] > SINGULAR_COND_THRESHOLD) {
            F3_is_singular[i] = true;
            n_singular++;
            double En  = En_initial + i * del_En;
            comp   En_c(En, 0.0);
            comp   P2  = total_P[0]*total_P[0]
                       + total_P[1]*total_P[1]
                       + total_P[2]*total_P[2];
            double Ecm = std::real(std::sqrt(En_c*En_c - P2));
            std::cout << "  [SINGULAR] i=" << i
                      << " En="  << std::fixed << std::setprecision(8) << En
                      << " Ecm=" << Ecm
                      << " cond=" << std::scientific << F3_cond[i]
                      << "\n";
        }
    }
    std::cout << "Singularity check done. "
              << n_singular << "/" << En_points
              << " matrices flagged as singular (near energy levels)\n"
              << std::endl;

    //==========================================================================
    // STEP 3: GPU inversion of F3 — only for non-singular matrices
    //         Build a filtered list, invert, scatter results back
    //==========================================================================
    std::vector<Eigen::MatrixXcd> F3inv_vec(En_points);  // final results

    // Collect non-singular indices
    std::vector<int> valid_indices;
    for (int i = 0; i < En_points; i++)
        if (!F3_is_singular[i]) valid_indices.push_back(i);

    std::cout << "Inverting " << valid_indices.size()
              << " non-singular F3 matrices on GPU...\n" << std::endl;

    if (!valid_indices.empty()) {
        // Temporary storage sized to valid subset
        int N_valid = (int)valid_indices.size();
        std::vector<Eigen::MatrixXcd> F3inv_valid(N_valid);

        // Builder: maps sequential index j → original index valid_indices[j]
        auto F3inv_builder = [&](int j, double /*En*/, Eigen::MatrixXcd& Hmat) {
            int i = valid_indices[j];
            Hmat  = F3_vec[i];
        };

        // Dummy energy sweep over [0, N_valid-1] with unit spacing
        // The builder ignores En and uses j directly
        build_and_invert_energy_sweep_varsize_batched_gpu_v3(
            chunkSize,
            0.0,                       // En_initial (dummy)
            (double)(N_valid - 1),     // En_final   (dummy)
            N_valid,                   // En_points  = number of valid matrices
            F3inv_builder,
            F3inv_valid,
            0.90, 16);

        // Scatter valid results back into full F3inv_vec
        for (int j = 0; j < N_valid; j++)
            F3inv_vec[valid_indices[j]] = std::move(F3inv_valid[j]);
    }

    // Leave F3inv_vec[i] as default-constructed (size 0x0) for singular i
    std::cout << "F3inv_vec built for all non-singular energies\n" << std::endl;

    //==========================================================================
    // STEP 4: Build n_config for each energy (config_maker_4 fills n_config)
    //==========================================================================
    std::vector<std::vector<std::vector<int>>> np_vec(En_points,
        std::vector<std::vector<int>>(3));
    std::vector<std::vector<std::vector<int>>> nk_vec(En_points,
        std::vector<std::vector<int>>(3));

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
    // STEP 5: Open output files — one per irrep
    // filename: "nnP_irrep_eigval.dat"  e.g. "011_A1_eigval.dat"
    //==========================================================================
    auto nnP_str = [&]() {
        return std::to_string(std::abs(nnP[0]))
             + std::to_string(std::abs(nnP[1]))
             + std::to_string(std::abs(nnP[2]));
    };

    std::map<std::string, std::ofstream> outfiles;
    for (const auto& irrep : irreps) {
        std::string fname = nnP_str() + "_" + irrep + "_eigval.dat";
        outfiles[irrep].open(fname);
        if (!outfiles[irrep].is_open())
            throw std::runtime_error("Cannot open: " + fname);

        outfiles[irrep] << std::setw(20) << "# En"
                        << std::setw(20) << "Ecm"
                        << std::setw(20) << "min_eigenvalue"
                        << std::setw(20) << "F3_cond"
                        << std::setw(12) << "singular"
                        << "\n";
        std::cout << "Opened: " << fname << "\n";
    }

    //==========================================================================
    // STEP 6: Irrep projection + smallest eigenvalue
    //         For singular F3: write NaN and flag in output
    //         For non-singular F3: project F3inv and extract smallest eig
    //==========================================================================
    std::cout << "Starting irrep projections...\n" << std::endl;

    for (int i = 0; i < En_points; i++) {
        double En  = En_initial + i * del_En;
        comp   En_c(En, 0.0);
        comp   P2  = total_P[0]*total_P[0]
                   + total_P[1]*total_P[1]
                   + total_P[2]*total_P[2];
        double Ecm = std::real(std::sqrt(En_c*En_c - P2));

        int dim1 = (int)plm_vec[i][0].size();
        int dim2 = (int)klm_vec[i][0].size();

        // Write NaN for empty configs or singular F3
        if (dim1 == 0 || dim2 == 0 || F3_is_singular[i]) {
            for (const auto& irrep : irreps) {
                outfiles[irrep] << std::fixed    << std::setprecision(20)
                                << En << '\t'
                                << Ecm << '\t'
                                << "NaN" << '\t'
                                << F3_cond[i] << '\t'
                                << (F3_is_singular[i] ? 1 : 0) << '\t'
                                << "\n";
            }
            continue;
        }

        // Hermiticity check at first valid index
        if (i == valid_indices[0]) {
            double herr = (F3inv_vec[i] - F3inv_vec[i].adjoint()).norm()
                        /  F3inv_vec[i].norm();
            std::cout << "F3inv Hermiticity error (first valid i="
                      << i << "): " << herr << "\n";
        }

        FlavorConfig cfg1, cfg2;
        cfg1.plm_config = plm_vec[i]; cfg1.n_config = np_vec[i];
        cfg2.plm_config = klm_vec[i]; cfg2.n_config = nk_vec[i];

        for (const auto& irrep : irreps) {
            double eig_min = std::numeric_limits<double>::quiet_NaN();
            comp detMI = {0.0, 0.0};
            try {
                MatC M_I = irrep_proj_2plus1_from_config(
                    F3inv_vec[i], nnP, irrep, cfg1, cfg2, -1);

                if (M_I.cols() > 0) {
                    Eigen::SelfAdjointEigenSolver<MatC> eig(M_I);

                    // Copy eigenvalues into a sortable vector
                    std::vector<double> evals(eig.eigenvalues().data(),
                                            eig.eigenvalues().data()
                                            + eig.eigenvalues().size());

                    // Sort ascending by value (not absolute value)
                    std::sort(evals.begin(), evals.end());

                    // First element is the smallest
                    eig_min = evals[0];
                    comp detMI = M_I.determinant();
                }
            } catch (const std::exception& e) {
                std::cerr << "  Projection error i=" << i
                          << " irrep=" << irrep
                          << ": " << e.what() << "\n";
            }
            
             


            /*
            outfiles[irrep] << std::fixed    << std::setprecision(10)
                            << std::setw(20) << En
                            << std::setw(20) << Ecm
                            << std::setw(20) << eig_min
                            << std::setw(20) << F3_cond[i]
                            << std::setw(12) << 0
                            << "\n";
            */
            outfiles[irrep] << std::fixed    << std::setprecision(20)
                      << En << '\t'
                      << Ecm << '\t'
                      << detMI.real() << '\t' 
                      << F3_cond[i] << '\t'
                      << 0<< "\n";
            
            std::cout << std::fixed    << std::setprecision(20)
                      << En << '\t'
                      << Ecm << '\t'
                      << detMI.real() << '\t' 
                      << F3_cond[i] << '\t'
                      << 0<< "\n";
        }

        if (i % 100 == 0)
            std::cout << "Progress: " << i << "/" << En_points
                      << "  Ecm=" << std::fixed << std::setprecision(6)
                      << Ecm << "\n" << std::endl;
    }

    for (auto& [irrep, f] : outfiles) f.close();

    std::cout << "\nDone.\n";
    std::cout << "Singular (near-level) points: " << n_singular << "\n";
    std::cout << "Output: " << nnP_str() << "_<irrep>_eigval.dat\n";
}

//This is uses the new projections_v1.hpp file 
//for irrep projections

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
        std::vector<std::vector<comp>> np_config(5), nk_config(5);

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
        std::vector<std::vector<int>>(3));
    std::vector<std::vector<std::vector<int>>> nk_vec(En_points,
        std::vector<std::vector<int>>(3));

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
    int sort_orbit_flag = 0;
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
                    irrep, total_P, nnP_config,
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


void test_projections_cpu() {
    
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

    Vec3 nnP = {1, 0, 0};
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
    int    En_points  = 1000;
    double del_En     = std::abs(En_initial - En_final) / (double)En_points;

    std::vector<std::string> irreps;// = irrep_list(nnP);
    irreps.push_back("A2");
    const double SINGULAR_COND_THRESHOLD = 1e10;

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
        config_maker_3(plm_config, waves_vec_1, En, total_P,
                       atmK, atmK, atmpi, L, epsilon_h, max_shell_num, tolerance);
        config_maker_3(klm_config, waves_vec_2, En, total_P,
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
    // STEP 2: Build n_config for each energy
    //==========================================================================
    std::vector<std::vector<std::vector<int>>> np_vec(En_points,
        std::vector<std::vector<int>>(3));
    std::vector<std::vector<std::vector<int>>> nk_vec(En_points,
        std::vector<std::vector<int>>(3));

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
    // STEP 3: Open output files
    //==========================================================================
    auto nnP_str = [&]() {
        return std::to_string(std::abs(nnP[0]))
             + std::to_string(std::abs(nnP[1]))
             + std::to_string(std::abs(nnP[2]));
    };

    std::map<std::string, std::ofstream> outfiles;
    for (const auto& irrep : irreps) {
        std::string fname = nnP_str() + "_" + irrep + "_eigval.dat";
        outfiles[irrep].open(fname);
        if (!outfiles[irrep].is_open())
            throw std::runtime_error("Cannot open: " + fname);

        outfiles[irrep] << std::setw(20) << "# En"
                        << std::setw(20) << "Ecm"
                        << std::setw(20) << "min_eigenvalue"
                        << std::setw(20) << "F3_cond"
                        << std::setw(12) << "singular"
                        << "\n";
        std::cout << "Opened: " << fname << "\n";
    }
    for (const auto& irrep : irreps) {
    std::string fname = nnP_str() + "_" + irrep + "_eigval.dat";
    outfiles[irrep].open(fname);
    if (!outfiles[irrep].is_open()) {
        std::cerr << "ERROR: Cannot open file: " << fname
                  << " reason: " << std::strerror(errno) << "\n";
        throw std::runtime_error("Cannot open: " + fname);
    }
    std::cout << "Opened: " << fname << "\n" << std::endl;
    }


    //==========================================================================
    // STEP 4: For each energy — Eigen inversion + irrep projection + eigenvalue
    //==========================================================================
    std::cout << "Starting CPU inversion + irrep projections...\n" << std::endl;

    int n_singular  = 0;
    int n_projected = 0;
    std::ofstream fout; 
    fout.open("tmp.dat");
    for (int i = 0; i < En_points; i++) {
        double En  = En_initial + i * del_En;
        comp   En_c(En, 0.0);
        comp   P2  = total_P[0]*total_P[0]
                   + total_P[1]*total_P[1]
                   + total_P[2]*total_P[2];
        double Ecm = std::real(std::sqrt(En_c*En_c - P2));

        int dim1 = (int)plm_vec[i][0].size();
        int dim2 = (int)klm_vec[i][0].size();

        //----------------------------------------------------------------------
        // Empty config
        //----------------------------------------------------------------------
        if (dim1 == 0 || dim2 == 0) {
            for (const auto& irrep : irreps) {
                outfiles[irrep] << std::fixed    << std::setprecision(10)
                                << std::setw(20) << En
                                << std::setw(20) << Ecm
                                << std::setw(20) << "NaN"
                                << std::setw(20) << "NaN"
                                << std::setw(12) << 0
                                << "\n";
            }
            continue;
        }

        //----------------------------------------------------------------------
        // Singularity check via SVD condition number
        //----------------------------------------------------------------------
        Eigen::JacobiSVD<Eigen::MatrixXcd> svd(F3_vec[i],
            Eigen::ComputeThinU | Eigen::ComputeThinV);
        double smin  = svd.singularValues().tail(1)(0);
        double smax  = svd.singularValues()(0);
        double cond  = (smin > 1e-300) ? smax / smin
                                       : std::numeric_limits<double>::infinity();
        bool singular = (cond > SINGULAR_COND_THRESHOLD);

        if (singular) {
            n_singular++;
            std::cout << "  [SINGULAR] i=" << i
                      << " En="  << std::fixed << std::setprecision(8) << En
                      << " Ecm=" << Ecm
                      << " cond=" << std::scientific << cond << "\n";

            for (const auto& irrep : irreps) {
                outfiles[irrep] << std::fixed    << std::setprecision(10)
                                << En
                                << '\t' << Ecm
                                << '\t' << "NaN"
                                << '\t' << cond
                                << '\t' << 1
                                << "\n";
            }
            continue;
        }

        //----------------------------------------------------------------------
        // Eigen inversion of F3 (FullPivLU — robust for complex matrices)
        //----------------------------------------------------------------------
        Eigen::MatrixXcd F3inv = F3_vec[i].fullPivLu().inverse();

        //----------------------------------------------------------------------
        // Hermiticity check (first valid i only)
        //----------------------------------------------------------------------
        if (n_projected == 0) {
            double herr = (F3inv - F3inv.adjoint()).norm() / F3inv.norm();
            std::cout << "F3inv Hermiticity error (first valid i="
                      << i << "): " << herr << "\n" << std::endl;
        }

        //----------------------------------------------------------------------
        // Build FlavorConfig
        //----------------------------------------------------------------------
        FlavorConfig cfg1, cfg2;
        cfg1.plm_config = plm_vec[i]; cfg1.n_config = np_vec[i];
        cfg2.plm_config = klm_vec[i]; cfg2.n_config = nk_vec[i];

        //----------------------------------------------------------------------
        // Irrep projection + smallest eigenvalue
        //----------------------------------------------------------------------
        
         
        for (const auto& irrep : irreps) {
            double eig_min = std::numeric_limits<double>::quiet_NaN();
            comp detMI = {0.0, 0.0}; 
            try {
                MatC M_I = irrep_proj_2plus1_from_config(
                    F3inv, nnP, irrep, cfg1, cfg2, -1);
                
                if (M_I.cols() > 0) {
                    Eigen::SelfAdjointEigenSolver<MatC> eig(M_I);
                    const auto& evals = eig.eigenvalues();
                    int    min_idx = 0;
                    double min_abs = std::abs(evals(0));
                    for (int k = 1; k < (int)evals.size(); k++) {
                        double a = std::abs(evals(k));
                        if (a < min_abs) { min_abs = a; min_idx = k; }
                    }
                    eig_min = evals(min_idx);
                    detMI = M_I.determinant();
                }
            } catch (const std::exception& e) {
                std::cerr << "  Projection error i=" << i
                          << " irrep=" << irrep
                          << ": " << e.what() << "\n";
            }

             

            
            outfiles[irrep] << std::setprecision(10)
                            << En
                            << '\t' << Ecm
                            << '\t' << detMI.real()
                            << '\t' << cond
                            << '\t' << 0
                            << "\n";
            fout << std::setprecision(10)
                            << En
                            << '\t' << Ecm
                            << '\t' << detMI.real()
                            << '\t' << cond
                            << '\t' << 0
                            << "\n";

            std::cout << std::setprecision(5)
                      << En << '\t'
                      << Ecm << '\t'
                      << detMI.real() << '\t' 
                      << cond << '\t'
                      << 0<< "\n";
        }
        
        n_projected++;

        if (i % 100 == 0)
            std::cout << "Progress: " << i << "/" << En_points
                      << "  Ecm=" << std::fixed << std::setprecision(6)
                      << Ecm << "\n" << std::endl;
    }
    fout.close();
    for (auto& [irrep, f] : outfiles) f.close();
     

    std::cout << "\nDone.\n"
              << "Projected:  " << n_projected << "/" << En_points << "\n"
              << "Singular:   " << n_singular  << "/" << En_points
              << " (near energy levels)\n"
              << "Output: "     << nnP_str()   << "_<irrep>_eigval.dat\n";
}


int main()
{
    //test_spherical_functions();

    //test_F2_ang_mom_function();

    //test_config_maker_3(); 

    //test_F2_ang_mat();

    //test_F2_2plus1_mat();

    //test_K2inv_2plus1_mat();

    //test_F3iso_2plus1_mat();

    //test_F3iso_gpu_2();

    //test_F3iso_gpu_3();

    //matrix_size_generator();

    //nconfig_check();

    //test_projections();

    //test_projections();
    //test_projections_1();

    //test_projections_gpu();

    test_projections_gpu_v2();

    //test_projections_cpu();
    return 0; 
}
