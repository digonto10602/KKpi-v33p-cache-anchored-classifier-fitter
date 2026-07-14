//This is the new printer code to check codes and run executables 

#include <bits/stdc++.h>
#include "spherical_functions.h"
#include "functions.h"
#include "F2_functions_v2.h"
#include "K2_functions_v2.h"
#include "QC_functions_v2.h"
#include<omp.h>

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

    
    


    std::vector< std::vector<comp> > scatter_params_1(4, std::vector<comp>(3)); 
    double a1_0 = 4.04;
    double a1_1 = 4.12;
    
    double zeroval1 = 0; 
    scatter_params_1[0][0] = a1_0; 
    scatter_params_1[1][0] = a1_1; 
    
    
    
    std::vector< std::vector<comp> > scatter_params_2(4, std::vector<comp>(3)); 
    double a2_0 = 43.2;
    
    //double zeroval1 = 0; 
    scatter_params_2[0][0] = a2_0; 



    

    double En_initial = 0.2631;
    double En_final = 0.35;
    double En_points = 1000; 
    double del_En = abs(En_initial - En_final)/En_points; 

    for(int i=0; i<En_points; ++i)
    {
        double En = En_initial + i*del_En; 
        comp F3iso = {0.0, 0.0}; 

        std::vector< std::vector<comp> > plm_config(5, std::vector<comp> ()); 

        config_maker_3(plm_config, waves_vec_1, En, total_P, mi, mj, mk, L, epsilon_h, max_shell_num, tolerance); 

        int dim1 = plm_config[0].size(); 

        
        mi = atmpi; 
        mj = atmK; 
        mk = atmK; 

        std::vector< std::vector<comp> > klm_config(5, std::vector<comp> ()); 

        config_maker_3(klm_config, waves_vec_2, En, total_P, mi, mj, mk, L, epsilon_h, max_shell_num, tolerance); 

        int dim2 = klm_config[0].size(); 

        Eigen::MatrixXcd F3mat(dim1 + dim2, dim1 + dim2);
        Eigen::MatrixXcd F2mat(dim1 + dim2, dim1 + dim2);
        Eigen::MatrixXcd K2inv_mat(dim1 + dim2, dim1 + dim2); 

        Eigen::MatrixXcd Gmat(dim1 + dim2, dim1 + dim2);
        Eigen::MatrixXcd Hmatinv(dim1 + dim2, dim1 + dim2);
        Eigen::VectorXcd state_vec(dim1 + dim2); 


        test_F3iso_ND_2plus1_mat_with_normalization_single_En(F3mat, F3iso, state_vec, F2mat, K2inv_mat, Gmat, Hmatinv, En, plm_config, klm_config, total_P, eta_1, eta_2, scatter_params_1, scatter_params_2, atmK, atmpi, alpha, epsilon_h, L, max_shell_num, Q0norm );

        std::cout<<En<<'\t'<<F3iso.real()<<std::endl;
    
    }
    
    //std::cout<<"F3iso = "<<F3iso<<std::endl; 
    




}





int main()
{
    //test_spherical_functions();

    //test_F2_ang_mom_function();

    //test_config_maker_3(); 

    //test_F2_ang_mat();

    //test_F2_2plus1_mat();

    //test_K2inv_2plus1_mat();

    test_F3iso_2plus1_mat();


    return 0; 
}
