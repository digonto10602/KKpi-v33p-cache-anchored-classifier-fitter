#ifndef K2FUNCTIONS_V2_H
#define K2FUNCTIONS_V2_H

#include "functions.h"
#include "F2_functions_v2.h"
#include<Eigen/Dense>


/* We code the K2 functions here, K2inv denotes the first definition that goes in to the
tilde_K2 function defined in 2.15 and 2.14 equations of the paper https://arxiv.org/pdf/2111.12734.pdf
*/

typedef std::complex<double> comp;

comp K2_inv_00( double eta_i, //this is the symmetry factor, eta=1 for piK (i=1), and eta=1/2 for KK (i=2)
                double scattering_length,
                comp sigma_i,
                double mj, 
                double mk,
                double epsilon_h    )
{
    double pi = std::acos(-1.0);
    comp A = eta_i/(8.0*pi*std::sqrt(sigma_i));
    comp B = -1.0/scattering_length; 

    //std::cout<<std::setprecision(25);
    //std::cout<<"qcotdel0 = "<<B<<std::endl;
    comp C = std::abs(std::sqrt(q2psq_star(sigma_i, mj, mk)));
    comp D = 1.0 - cutoff_function_1(sigma_i, mj, mk, epsilon_h);

    //std::cout << "q = " << C << std::endl; 
    //std::cout << "cutoff = "<< cutoff_function_1(sigma_i, mj, mk, epsilon_h) << std::endl; 
    //std::cout << "CD = " <<C*D << std::endl;
    return A*(B + C*D);
}

comp K2_inv_ERE_ang_mom( 
                        double eta_i, //this is the symmetry factor, eta=1 for piK (i=1), and eta=1/2 for KK (i=2)
                        std::vector< std::vector<comp> > scatter_params,
                        int ell_f,
                        int proj_mf, 
                        int ell_i, 
                        int proj_mi, 
                        comp sigma_i,
                        double mj, 
                        double mk,
                        double epsilon_h    
                    )
{
    char debug = 'n'; 
    double pi = std::acos(-1.0);
    comp A = eta_i/(8.0*pi*std::sqrt(sigma_i));
    comp B = {0.0, 0.0}; //-1.0/scattering_length; 

    //std::cout<<std::setprecision(25);
    //std::cout<<"qcotdel0 = "<<B<<std::endl;
    comp C = std::abs(std::sqrt(q2psq_star(sigma_i, mj, mk)));
    comp D = 1.0 - cutoff_function_1(sigma_i, mj, mk, epsilon_h);
    comp q2psq = q2psq_star(sigma_i, mj, mk);
    
    if(debug=='y')
    {
        std::cout << "A = " << A << std::endl;
        std::cout << "C = " << C << std::endl;
        std::cout << "q2psq = " << q2psq << std::endl; 
    }

    comp out = {0.0,0.0};
    comp zero = {0.0,0.0};

    if(ell_f==ell_i)
    {
        if(proj_mf==proj_mi)
        {
            int ell = ell_f; 

            if(ell==0)
            {
                comp a0 = scatter_params[ell][0]; 
                comp r0 = scatter_params[ell][1]; 
                comp P0 = scatter_params[ell][2];

                comp qcotdel = - 1.0/a0 + 0.5*r0*q2psq + P0*q2psq*q2psq; 
                comp B = qcotdel; 

                out = A*(B + C*D);

                if(debug=='y')
                {
                    std::cout << "ell = " << ell << std::endl; 
                    std::cout << "scat par = " << a0 << '\t' << r0 << '\t' << P0 << std:: endl; 
                    std::cout << "B = " << B << std::endl;
                    std::cout << "out = " << out << std::endl;
                }
            }
            else if(ell==1)
            {
                comp a1 = scatter_params[ell][0]; 
                comp r1 = scatter_params[ell][1]; 
                comp P1 = scatter_params[ell][2];

                comp qcotdel = - 1.0/(a1*q2psq) + 0.5*r1 + P1*q2psq; 
                //comp qcotdel = - 1.0/(a1) + 0.5*r1*q2psq + P1*q2psq*q2psq; 
                comp B = qcotdel; 

                out = A*(B + q2psq*C*D);
                //stable equation 
                if(debug=='y')
                {
                    std::cout << "ell = " << ell << std::endl; 
                    std::cout << "B = " << B << std::endl;
                    std::cout << "out = " << out << std::endl;
                }
            }
            else if(ell==2)
            {
                comp a2 = scatter_params[ell][0]; 
                comp r2 = scatter_params[ell][1]; 
                comp P2 = scatter_params[ell][2];

                comp qcotdel = - 1.0/(a2*q2psq*q2psq) + 0.5*r2/q2psq + P2; 
                comp B = qcotdel; 

                out = A*(B + C*D);
                if(debug=='y')
                {
                    std::cout << "ell = " << ell << std::endl; 
                    std::cout << "B = " << B << std::endl;
                    std::cout << "out = " << out << std::endl;
                }
            }
            else if(ell==3)
            {
                comp a3 = scatter_params[ell][0]; 
                comp r3 = scatter_params[ell][1]; 
                comp P3 = scatter_params[ell][2];

                comp qcotdel = - 1.0/(a3*q2psq*q2psq*q2psq) + 0.5*r3/(q2psq*q2psq) + P3/q2psq; 
                comp B = qcotdel; 

                out = A*(B + C*D);
                if(debug=='y')
                {
                    std::cout << "ell = " << ell << std::endl; 
                    std::cout << "B = " << B << std::endl;
                    std::cout << "out = " << out << std::endl;
                }
            }
            else
            {
                std::cerr << "ell>3 is not coded yet" << std::endl; 
                out = zero;
            }
        }
        else
        {
            out = zero;
        }
    }
    else
    {
        out = zero; 
    }

    //std::cout << "q = " << C << std::endl; 
    //std::cout << "cutoff = "<< cutoff_function_1(sigma_i, mj, mk, epsilon_h) << std::endl; 
    //std::cout << "CD = " <<C*D << std::endl;
    return out;
}

/*  This is K2 matrix we use to build K2_hat matrix for the KKpi 
project, this function builds a matrix for 1/(2 omega_p L^3 )*K2_inv_00
This function assumes that pvec and kvec are same, if they are different 
then this function will not work for them and have to be redefined */

void K2inv_EREord2_i_mat( 
                            Eigen::MatrixXcd &K2inv,
                            double eta_i, 
                            std::vector< std::vector<comp> > scatter_params,
                            comp En,  
                            std::vector< std::vector<comp> > &plm_config, 
                            std::vector< std::vector<comp> > &klm_config,
                            std::vector<comp> total_P,
                            double mi, 
                            double mj, 
                            double mk, 
                            double epsilon_h, 
                            double L            
                        )
{

    char debug = 'n'; 
    int size1 = plm_config[0].size();
    int size2 = klm_config[0].size();
    for(int i=0; i<size1; ++i)
    {
        for(int j=0; j<size2; ++j)
        {
            if(i==j)
            {
                int ell_f = static_cast<int> (plm_config[3][i].real()); 
                int proj_mf = static_cast<int> (plm_config[4][i].real()); 
                
                int ell_i = static_cast<int> (klm_config[3][j].real()); 
                int proj_mi = static_cast<int> (klm_config[4][j].real()); 
                
                comp px = plm_config[0][i];
                comp py = plm_config[1][i];
                comp pz = plm_config[2][i];

                comp spec_p = std::sqrt(px*px + py*py + pz*pz);

                comp kx = klm_config[0][j];
                comp ky = klm_config[1][j];
                comp kz = klm_config[2][j];

                comp spec_k = std::sqrt(kx*kx + ky*ky + kz*kz);

                comp Px = total_P[0];
                comp Py = total_P[1];
                comp Pz = total_P[2];

                comp total_P_val = std::sqrt(Px*Px + Py*Py + Pz*Pz);

                comp Pminusk_x = Px - kx;
                comp Pminusk_y = Py - ky;
                comp Pminusk_z = Pz - kz; 
                comp Pminusk = std::sqrt(Pminusk_x*Pminusk_x + Pminusk_y*Pminusk_y + Pminusk_z*Pminusk_z);


                comp sig_k = (En - omega_func(spec_k,mi))*(En - omega_func(spec_k,mi)) - Pminusk*Pminusk;

                

                //comp sig_k = sigma(En, spec_k, mi, total_P_val);

                comp K2_inv_val = K2_inv_ERE_ang_mom(eta_i, scatter_params, ell_f, proj_mf, ell_i, proj_mi, sig_k, mj, mk, epsilon_h);
            
                if(debug=='y')
                {
                    std::cout << "K2 inv val = " << K2_inv_val << std::endl; 
                }

                comp omega_k = omega_func(spec_k, mi);

                comp constval = 1.0/(2.0*omega_k*L*L*L);
                
                K2inv(i,j) = constval*K2_inv_val; 
            }
            else 
            {
                K2inv(i,j) = 0.0;
            }
        }
    }
}

//This function builds the K2inv based on the 
//ERE of order 0, for the 2+1 system
//Assumes m1 and m2 are such that 2m1 + m2 system
//K2inv = | K2inv_1   0 |
//        | 0  2xK2inv_2|

void K2inv_EREord2_2plus1_mat( 
                                Eigen::MatrixXcd &K2inv,
                                double eta_1,
                                double eta_2,  
                                std::vector< std::vector<comp> > scatter_params_1,
                                std::vector< std::vector<comp> > scatter_params_2,
                                comp En,  
                                std::vector< std::vector<comp> > &plm_config, 
                                std::vector< std::vector<comp> > &klm_config,
                                std::vector<comp> total_P,
                                double m1, 
                                double m2, 
                                double epsilon_h, 
                                double L            
                        )
{
    char debug = 'n'; 
    double mi, mj, mk = 0.0; 
    int size1 = plm_config[0].size(); 
    int size2 = klm_config[0].size(); 

    //when i=1
    mi = m1; 
    mj = m1; 
    mk = m2; 

    Eigen::MatrixXcd K2inv_1(size1, size1); 
    Eigen::MatrixXcd K2inv_2(size2, size2); 

    Eigen::MatrixXcd Filler0_12(size1,size2);
    Eigen::MatrixXcd Filler0_21(size2,size1);
    

    Filler0_12 = Eigen::MatrixXcd::Zero(size1,size2);
    Filler0_21 = Eigen::MatrixXcd::Zero(size2,size1);
    
    K2inv_EREord2_i_mat( K2inv_1, eta_1, scatter_params_1, En, plm_config, plm_config, total_P, mi, mj, mk, epsilon_h, L); 

    //when i=2
    mi = m2; 
    mj = m1; 
    mk = m1; 

    K2inv_EREord2_i_mat( K2inv_2, eta_2, scatter_params_2, En, klm_config, klm_config, total_P, mi, mj, mk, epsilon_h, L); 

    if(debug=='y')
    {
        std::cout<< "K2inv_2 mat = " << std::endl; 
        std::cout<< K2inv_2 << std::endl;
    }

    Eigen::MatrixXcd K2inv_mat(size1 + size2, size1 + size2); 

    K2inv_mat << K2inv_1 , Filler0_12, 
                 Filler0_21, 2.0*K2inv_2; 

    K2inv = K2inv_mat; 

    

}


#endif