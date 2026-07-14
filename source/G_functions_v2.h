#ifndef GFUNCTIONS_V2_H
#define GFUNCTIONS_V2_H
#include "functions.h"
#include "F2_functions_v2.h"
#include "K2_functions_v2.h"
#include<Eigen/Dense>

typedef std::complex<double> comp;

void print_Gij_boosts(
    comp En,
    std::vector<comp> p,
    std::vector<comp> k,
    std::vector<comp> total_P,
    double mi,
    double mj
)
{
    comp px = p[0], py = p[1], pz = p[2];
    comp kx = k[0], ky = k[1], kz = k[2];

    comp Px = total_P[0], Py = total_P[1], Pz = total_P[2];

    comp spec_p = std::sqrt(px*px + py*py + pz*pz);
    comp spec_k = std::sqrt(kx*kx + ky*ky + kz*kz);

    comp omg_p = omega_func(spec_p, mi);
    comp omg_k = omega_func(spec_k, mj);

    std::vector<comp> Pp(3);
    Pp[0] = Px - px;
    Pp[1] = Py - py;
    Pp[2] = Pz - pz;

    std::vector<comp> Pk(3);
    Pk[0] = Px - kx;
    Pk[1] = Py - ky;
    Pk[2] = Pz - kz;

    std::vector<comp> k_star = boost(omg_k, k, En - omg_p, Pp);
    std::vector<comp> p_star = boost(omg_p, p, En - omg_k, Pk);

    std::cout << "p = [ "
              << p[0] << ", " << p[1] << ", " << p[2] << " ]\n";

    std::cout << "k = [ "
              << k[0] << ", " << k[1] << ", " << k[2] << " ]\n";

    std::cout << "total_P = [ "
              << total_P[0] << ", " << total_P[1] << ", " << total_P[2] << " ]\n\n";

    std::cout << "omega_p = " << omg_p << "\n";
    std::cout << "omega_k = " << omg_k << "\n\n";

    std::cout << "Pp = P - p = [ "
              << Pp[0] << ", " << Pp[1] << ", " << Pp[2] << " ]\n";

    std::cout << "Pk = P - k = [ "
              << Pk[0] << ", " << Pk[1] << ", " << Pk[2] << " ]\n\n";

    std::cout << "k_star = boost(omega_k, k, En - omega_p, P - p):\n";
    std::cout << "[ "
              << k_star[0] << ", "
              << k_star[1] << ", "
              << k_star[2] << " ]\n\n";

    std::cout << "p_star = boost(omega_p, p, En - omega_k, P - k):\n";
    std::cout << "[ "
              << p_star[0] << ", "
              << p_star[1] << ", "
              << p_star[2] << " ]\n";
}

comp G_ij_lm(  
                comp En, 
                std::vector<comp> p,
                std::vector<comp> k,
                std::vector<comp> total_P,
                int ell_f, 
                int proj_mf, 
                int ell_i, 
                int proj_mi, 
                double mi,
                double mj,
                double mk,
                double L,
                double epsilon_h,
                bool Q0norm
           )
{
    comp px = p[0];
    comp py = p[1];
    comp pz = p[2];

    comp spec_p = std::sqrt(px*px + py*py + pz*pz);

    comp kx = k[0];
    comp ky = k[1];
    comp kz = k[2];

    comp spec_k = std::sqrt(kx*kx + ky*ky + kz*kz);

    comp Px = total_P[0];
    comp Py = total_P[1];
    comp Pz = total_P[2];

    comp total_P_val = std::sqrt(Px*Px + Py*Py + Pz*Pz);

    comp omg_p = omega_func(spec_p,mi);
    comp omg_k = omega_func(spec_k,mj);

    comp Ppx = Px - px; 
    comp Ppy = Py - py; 
    comp Ppz = Pz - pz; 

    std::vector<comp> Pp(3); 
    Pp[0] = Ppx; 
    Pp[1] = Ppy;
    Pp[2] = Ppz; 

    comp Pkx = Px - kx; 
    comp Pky = Py - ky; 
    comp Pkz = Pz - kz; 

    std::vector<comp> Pk(3); 
    Pk[0] = Pkx; 
    Pk[1] = Pky;
    Pk[2] = Pkz; 

    std::vector<comp> k_star = boost(omg_k, k, En - omg_p, Pp );
    std::vector<comp> p_star = boost(omg_p, p, En - omg_k, Pk );

    comp Ylm1 = spherical_harmonics(k_star, ell_f, proj_mf); 
    comp Ylm2 = spherical_harmonics(p_star, ell_i, proj_mi); 


    comp oneby2omegapLcube = 1.0/(2.0*omega_func(spec_p,mi)*L*L*L);
    comp oneby2omegakLcube = 1.0/(2.0*omega_func(spec_k,mj)*L*L*L);

    comp sig_i = sigma_pvec_based(En,p,mi,total_P);//sigma(En,spec_p,mi,total_P_val);
    comp sig_j = sigma_pvec_based(En,k,mj,total_P);//sigma(En,spec_k,mj,total_P_val);
    comp cutoff1 = cutoff_function_1(sig_i,mj,mk,epsilon_h);
    comp cutoff2 = cutoff_function_1(sig_j,mi,mk,epsilon_h);
    comp q2psq = q2psq_star(sig_i,mj,mk);
    comp q2ksq = q2psq_star(sig_j,mi,mk);

    char debug = 'n';
    

    comp mom_P_p_k_x = Px - px - kx;
    comp mom_P_p_k_y = Py - py - ky;
    comp mom_P_p_k_z = Pz - pz - kz; 

    comp mom_P_p_k = std::sqrt(mom_P_p_k_x*mom_P_p_k_x + mom_P_p_k_y*mom_P_p_k_y + mom_P_p_k_z*mom_P_p_k_z);


    comp denom = (En - omega_func(spec_p,mi) - omega_func(spec_k,mj))*(En - omega_func(spec_p,mi) - omega_func(spec_k,mj))
                -mom_P_p_k*mom_P_p_k - mk*mk; 

    comp tol = {0.0,1.0e-10};
    //comp denom = (En - omega_func(spec_p,mi) - omega_func(spec_k,mj) - omega_func(mom_P_p_k,mk))*(En - omega_func(spec_p,mi) - omega_func(spec_k,mj) + omega_func(mom_P_p_k,mk) + tol);
    //comp denom = 2.0*omega_func(mom_P_p_k,mk)*(En - omega_func(spec_p,mi) - omega_func(spec_k,mj) - omega_func(mom_P_p_k,mk));

    comp onebydenom = 1.0/denom; 

    //std::cout<<"sigi="<<sig_i<<'\t'<<"sigj="<<sig_j<<'\t'<<"cutoff1="<<cutoff1<<'\t'<<"cutoff2="<<cutoff2<<"denom="<<denom<<std::endl;

    comp result = Ylm1*oneby2omegapLcube*cutoff1*cutoff2*onebydenom*oneby2omegakLcube*Ylm2;

    if(debug=='y')
    {
        std::cout << "=============================================" << std::endl; 
        std::cout << "p = " << px << '\t' << py << '\t' << pz << std::endl; 
        std::cout << "sigp = " << sig_i << std::endl; 
        std::cout << "cutoff1 = " << cutoff1 << std::endl;
        std::cout << "k = " << kx << '\t' << ky << '\t' << kz << std::endl; 
        std::cout << "sigk = " << sig_j << std::endl; 
        std::cout << "cutoff2 = " << cutoff2 << std::endl; 
        std::cout << "eff_f = " << ell_f << " " << "proj_m_f = " << proj_mf << std::endl; 
        std::cout << "eff_i = " << ell_i << " " << "proj_m_i = " << proj_mi << std::endl; 
        std::cout << "kstar = " << k_star[0] << "," << k_star[1] << "," << k_star[2] << std::endl; 
        std::cout << "pstar = " << p_star[0] << "," << p_star[1] << "," << p_star[2] << std::endl; 
        std::cout << "ylm1 = " << Ylm1 << std::endl; 
        std::cout << "ylm2 = " << Ylm2 << std::endl; 
        std::cout << "=============================================" << std::endl; 
    }

    if(Q0norm)
    {
        return result;
    }
    else
    {
        comp leftterm = 1.0/std::pow(q2psq, ell_f);
        comp rightterm = 1.0/std::pow(q2ksq, ell_i);
        result = leftterm*result*rightterm; 
        return result; 
    }

    
    //return oneby2omegapLcube;

}

Eigen::MatrixXcd get_parity_block_from_plm_config(
    const std::vector<std::vector<comp>>& plm_config,
    double tol = 1e-12)
{
    if (plm_config.size() < 5) {
        throw std::runtime_error("get_parity_block_from_plm_config: plm_config must have at least 5 rows");
    }

    const std::size_t Ntot = plm_config[0].size();

    for (std::size_t r = 1; r < 5; ++r) {
        if (plm_config[r].size() != Ntot) {
            throw std::runtime_error("get_parity_block_from_plm_config: inconsistent row sizes in plm_config");
        }
    }

    Eigen::MatrixXcd PL = Eigen::MatrixXcd::Identity(Ntot, Ntot);

    for (std::size_t i = 0; i < Ntot; ++i) {
        comp ell_c = plm_config[3][i];

        if (std::abs(ell_c.imag()) > tol) {
            throw std::runtime_error("get_parity_block_from_plm_config: ell has nonzero imaginary part");
        }

        int ell = static_cast<int>(std::llround(ell_c.real()));

        if (std::abs(ell_c.real() - ell) > tol) {
            throw std::runtime_error("get_parity_block_from_plm_config: ell is not close to an integer");
        }

        if (ell % 2 != 0) {
            PL(i, i) = comp(-1.0, 0.0);
        }
    }

    return PL;
}


void G_2plus1_mat(
                    Eigen::MatrixXcd &G,
                    comp En, 
                    std::vector<std::vector<comp> > &plm_config,
                    std::vector<std::vector<comp> > &klm_config,
                    std::vector<comp> total_P,
                    double m1,
                    double m2, 
                    double L, 
                    double alpha, 
                    double epsilon_h,
                    int max_shell_num,
                    bool Q0norm
)
{
    char debug = 'n';
    
    double mi, mj, mk = 0.0; 

    int size1 = plm_config[0].size();
    int size2 = klm_config[0].size(); 

    int total_size = size1 + size2; 

    Eigen::MatrixXcd G11(size1, size1); 
    Eigen::MatrixXcd G12(size1, size2); 
    Eigen::MatrixXcd G21(size2, size1); 
    Eigen::MatrixXcd Filler0_22(size2, size2); 
    Filler0_22 = Eigen::MatrixXcd::Zero(size2, size2); 

    //when i,j = 1,1 
    mi = m1; 
    mj = m1; 
    mk = m2; 

    for(int i=0; i<size1; ++i)
    {
        for(int j=0; j<size1; ++j)
        {
            comp px = plm_config[0][i]; 
            comp py = plm_config[1][i];
            comp pz = plm_config[2][i]; 

            comp spec_p = std::sqrt(px*px + py*py + pz*pz);
            std::vector<comp> p(3);
            p[0] = px; 
            p[1] = py; 
            p[2] = pz;  

            comp kx = plm_config[0][j];
            comp ky = plm_config[1][j];
            comp kz = plm_config[2][j]; 

            comp spec_k = std::sqrt(kx*kx + ky*ky + kz*kz);

            std::vector<comp> k(3);
            k[0] = kx; 
            k[1] = ky; 
            k[2] = kz; 

            comp Px = total_P[0];
            comp Py = total_P[1];
            comp Pz = total_P[2];

            int ell_f = static_cast<int>(plm_config[3][i].real());
            int proj_mf = static_cast<int>(plm_config[4][i].real());

            int ell_i = static_cast<int>(plm_config[3][j].real()); 
            int proj_mi = static_cast<int>(plm_config[4][j].real()); 

            G11(i,j) = G_ij_lm(En, p, k, total_P, ell_f, proj_mf, ell_i, proj_mi, mi, mj, mk, L, epsilon_h, Q0norm); 
            if(debug=='y')
            {
                double const1 = L/(2.0*std::acos(-1.0));
                std::cout << "i:" << i << " j:" << j << std::endl; 
                std::cout << "px:" << const1*px.real() << ' '
                          << "py:" << const1*py.real() << ' '
                          << "pz:" << const1*pz.real() << ' '
                          << "kx:" << const1*kx.real() << ' '
                          << "ky:" << const1*ky.real() << ' '
                          << "kz:" << const1*kz.real() << std::endl; 
                std::cout << "ell_f:" << ell_f << ' '
                          << "proj_m_f:" << proj_mf << ' '
                          << "ell_i:" << ell_i << ' '
                          << "proj_m_i:" << proj_mi << std::endl; 
                std::cout << "G11:" << G11(i,j) << std::endl; 
            }
        }
    }


    //when i,j = 1,2 
    mi = m1; 
    mj = m2; 
    mk = m1; 

    for(int i=0; i<size1; ++i)
    {
        for(int j=0; j<size2; ++j)
        {
            comp px = plm_config[0][i]; 
            comp py = plm_config[1][i];
            comp pz = plm_config[2][i]; 

            comp spec_p = std::sqrt(px*px + py*py + pz*pz);
            std::vector<comp> p(3);
            p[0] = px; 
            p[1] = py; 
            p[2] = pz;  

            comp kx = klm_config[0][j];
            comp ky = klm_config[1][j];
            comp kz = klm_config[2][j]; 

            comp spec_k = std::sqrt(kx*kx + ky*ky + kz*kz);

            std::vector<comp> k(3);
            k[0] = kx; 
            k[1] = ky; 
            k[2] = kz; 

            comp Px = total_P[0];
            comp Py = total_P[1];
            comp Pz = total_P[2];

            int ell_f = static_cast<int>(plm_config[3][i].real());
            int proj_mf = static_cast<int>(plm_config[4][i].real());

            int ell_i = static_cast<int>(klm_config[3][j].real()); 
            int proj_mi = static_cast<int>(klm_config[4][j].real()); 

            comp projector = std::pow(-1.0, ell_f); 
            
            G12(i,j) = std::sqrt(2.0)*projector*G_ij_lm(En, p, k, total_P, ell_f, proj_mf, ell_i, proj_mi, mi, mj, mk, L, epsilon_h, Q0norm);

            if(debug=='y')
            {
                std::cout << "i:" << i << " j:" << j << std::endl; 
                std::cout << "px:" << px << ' '
                          << "py:" << py << ' '
                          << "pz:" << pz << ' '
                          << "kx:" << kx << ' '
                          << "ky:" << ky << ' '
                          << "kz:" << kz << std::endl; 
                std::cout << "ell_f:" << ell_f << ' '
                          << "proj_m_f:" << proj_mf << ' '
                          << "ell_i:" << ell_i << ' '
                          << "proj_m_i:" << proj_mi << std::endl; 
                std::cout << "parity term:" << projector << std::endl; 
                std::cout << "G12:" << G11(i,j) << std::endl; 
            }
            
            
        }
    }

     //when i,j = 2,1 
    mi = m2; 
    mj = m1; 
    mk = m1; 

    for(int i=0; i<size2; ++i)
    {
        for(int j=0; j<size1; ++j)
        {
            comp px = klm_config[0][i]; 
            comp py = klm_config[1][i];
            comp pz = klm_config[2][i]; 

            comp spec_p = std::sqrt(px*px + py*py + pz*pz);
            std::vector<comp> p(3);
            p[0] = px; 
            p[1] = py; 
            p[2] = pz;  

            comp kx = plm_config[0][j];
            comp ky = plm_config[1][j];
            comp kz = plm_config[2][j]; 

            comp spec_k = std::sqrt(kx*kx + ky*ky + kz*kz);

            std::vector<comp> k(3);
            k[0] = kx; 
            k[1] = ky; 
            k[2] = kz; 

            comp Px = total_P[0];
            comp Py = total_P[1];
            comp Pz = total_P[2];

            int ell_f = static_cast<int>(klm_config[3][i].real());
            int proj_mf = static_cast<int>(klm_config[4][i].real());

            int ell_i = static_cast<int>(plm_config[3][j].real()); 
            int proj_mi = static_cast<int>(plm_config[4][j].real()); 

            comp projector = std::pow(-1.0, ell_i); 
            
            G21(i,j) = std::sqrt(2.0)*G_ij_lm(En, p, k, total_P, ell_f, proj_mf, ell_i, proj_mi, mi, mj, mk, L, epsilon_h, Q0norm)*projector;

            if(debug=='y')
            {
                std::cout << "i:" << i << " j:" << j << std::endl; 
                std::cout << "px:" << px << ' '
                          << "py:" << py << ' '
                          << "pz:" << pz << ' '
                          << "kx:" << kx << ' '
                          << "ky:" << ky << ' '
                          << "kz:" << kz << std::endl; 
                std::cout << "ell_f:" << ell_f << ' '
                          << "proj_m_f:" << proj_mf << ' '
                          << "ell_i:" << ell_i << ' '
                          << "proj_m_i:" << proj_mi << std::endl; 
                std::cout << "parity term:" << projector << std::endl; 
                std::cout << "G21:" << G11(i,j) << std::endl; 
            }
            
            
        }
    }

    Eigen::MatrixXcd Gmat(size1 + size2, size1 + size2); 

    Gmat << G11, G12,
            G21, Filler0_22; 

    G = Gmat; 

}





#endif