#ifndef FUNCTIONS_H
#define FUNCTIONS_H 

#include <algorithm>
#include<bits/stdc++.h>
#include<cmath>
#include<Eigen/Dense>
#include "spherical_functions.h"

typedef std::complex<double> comp;

comp mysqrt(    comp x  )
{
    comp ii = {0.0,1.0};
    return ii*std::sqrt(-x);
}

comp omega_func(    comp p, 
                    double m    )
{
    comp mcomp = (comp) m; 
    return std::sqrt(p*p + mcomp*mcomp);
}

comp sigma( comp En,
            comp spec_p,
            double mi,
            comp total_P    )
{
    comp A = En - omega_func(spec_p,mi);
    comp B = total_P - spec_p;

    return A*A - B*B;
}

/* This sigma takes the spectator momenta as a vector */
comp sigma_pvec_based(  comp En, 
                        std::vector<comp> p,
                        double mi, 
                        std::vector<comp> total_P   )
{
    comp px = p[0];
    comp py = p[1];
    comp pz = p[2];

    comp spec_p = std::sqrt(px*px + py*py + pz*pz);

    comp Px = total_P[0];
    comp Py = total_P[1];
    comp Pz = total_P[2];

    comp Pminusp_x = Px - px; 
    comp Pminusp_y = Py - py; 
    comp Pminusp_z = Pz - pz; 

    comp Pminusp_sq = Pminusp_x*Pminusp_x + Pminusp_y*Pminusp_y + Pminusp_z*Pminusp_z; 
    comp A = En - omega_func(spec_p,mi);

    return A*A - Pminusp_sq; 
}

comp kallentriangle(    comp x, 
                        comp y, 
                        comp z  )
{
    //return x*x + y*y + z*z - 2.0*x*y - 2.0*y*z - 2.0*z*x;
    return x*x + y*y + z*z - ((comp)2.0)*(x*y + y*z + z*x);

}

comp q2psq_star(    comp sigma_i,
                    double mj, 
                    double mk )
{
    comp mjcomp = (comp) mj; 
    comp mkcomp = (comp) mk; 
    return kallentriangle(sigma_i, mjcomp*mjcomp, mkcomp*mkcomp)/(((comp)4.0)*sigma_i);
}

comp pmom(  comp En,
            comp sigk,
            double m    )
{
    return std::sqrt(kallentriangle(En*En,sigk,m*m))/(((comp)2.0)*sqrt(En*En));
}

comp kmax_for_P0(   comp En, 
                    double m )
{
    comp mcomp = (comp) m; 
    comp A = (En*En + mcomp*mcomp)/(((comp)2.0)*En);

    return A*A - mcomp*mcomp;
}


comp Jfunc( comp z  )
{
    if(std::real(z)<=0.0)
    {
        return ((comp)0.0);
    }
    else if(std::real(z)>0.0 && std::real(z)<1.0)
    {
        comp A = -((comp)1.0)/z;
        comp B = std::exp(-((comp)1.0)/(((comp)1.0)-z));
        return std::exp(A*B);
    }
    else
    {
        return ((comp)1.0);
    }
    
    

}

comp cutoff_function_1( comp sigma_i,
                        double mj, 
                        double mk, 
                        double epsilon_h   )
{
    comp mjcomp = (comp) mj;
    comp mkcomp = (comp) mk; 

    if(mj==mk && epsilon_h==0.0)
    {
        comp Z = sigma_i/(((comp)4.0)*mjcomp*mjcomp);
        return Jfunc(Z); 
    }
    else 
    {
        //std::cout<<"went to else"<<std::endl; 
        comp Z = (comp) (((comp)1.0) + ((comp)epsilon_h))*( sigma_i - (comp) std::abs(mjcomp*mjcomp - mkcomp*mkcomp) )/( (mjcomp + mkcomp)*(mjcomp + mkcomp) - std::abs(mjcomp*mjcomp - mkcomp*mkcomp) );
        return Jfunc(Z);
        
    }

    //return 1.0; 

}

comp E_to_Ecm(  comp En,
                std::vector<comp> total_P )
{
    comp Px = total_P[0];
    comp Py = total_P[1];
    comp Pz = total_P[2];

    comp tot_P = std::sqrt(Px*Px + Py*Py + Pz*Pz);

    return std::sqrt(En*En - tot_P*tot_P);
}

comp Ecm_to_E(  comp En_cm,
                std::vector<comp> total_P )
{
    comp Px = total_P[0];
    comp Py = total_P[1];
    comp Pz = total_P[2];

    comp tot_P = std::sqrt(Px*Px + Py*Py + Pz*Pz);

    return std::sqrt(En_cm*En_cm + tot_P*tot_P);
}

/*

    S-wave F2 function for i = 1

*/

//This dawson function has imaginary argument for the purpose
//of generality of the code but the definition need to be changed 
//if we are using true complex x's, the definition would be changed
//to use fadeeva function w(z). For now since we are only using 
//real energies and real momentum we can use this defintion with 
//the argument being complex without creating any errors.

comp dawson_func(comp x)
{
    double steps = 1000000;
    comp summ = 0.0;

    comp y = 0.0;
    comp dely = x/((comp)steps);
    for(int i=0;i<(int)steps;++i)
    {
        summ = summ + std::exp(y*y)*dely;
        y=y+dely;
    }

    return std::exp(-x*x)*summ;
}

comp ERFI_func(comp x)
{
    comp pi = std::acos(-1.0);
    return dawson_func(x)*((comp)2.0)*std::exp(x*x)/std::sqrt(pi);
}

/* Configuration vectors for the spectator momentum, these are 
generated with only the integer numbers first, then multiplied with
2pi/L, and they go upto where H(k) becomes zero */

void config_maker(  std::vector< std::vector<comp> > &p_config,
                    comp En,
                    double mi,
                    double L    )
{
    comp pi = std::acos(-1.0);
    comp twopibyL = ((comp) 2.0*pi)/((comp) L);
    comp Lby2pi = ((comp) L)/((comp) 2.0*pi);

    comp kmax = pmom(En,((comp)0.0),mi);

    int nmax = (int)ceil((Lby2pi.real())*abs(kmax));

    int nmaxsq = nmax*nmax; 

    for(int i=-nmax; i<nmax + 1; ++i)
    {
        for(int j=-nmax; j<nmax + 1; ++j)
        {
            for(int k=-nmax; k<nmax + 1; ++k)
            {
                int nsq = i*i + j*j + k*k; 
                if(nsq<=nmaxsq)
                {
                    comp px = (((comp)2.0)*pi/((comp)L))*((comp)i);
                    comp py = (((comp)2.0)*pi/((comp)L))*((comp)j);
                    comp pz = (((comp)2.0)*pi/((comp)L))*((comp)k); 


                    comp p = std::sqrt(px*px + py*py + pz*pz);
                    
                    if(abs(p)<=abs(kmax))
                    {
                        p_config[0].push_back(px);
                        p_config[1].push_back(py);
                        p_config[2].push_back(pz);
                        //std::cout << "n = " << i << j << k << " nsq = " << nsq << " nmaxsq = " << nmaxsq << std::endl; 
                        //std::cout << "px = " << px << " py = " << py << " pz = " << pz << std::endl;
                        //std::cout << "p = " << p << " kmax = " << kmax << std::endl; 
                 
                    }
                    else 
                    {
                        continue; 
                    }
                }
            }
        }
    }
    //std::cout << "nmax = " << nmax << '\t' << "nval = " << (L/(2.0*pi))*abs(kmax) << '\t' 
    //          << "kmax = " << kmax << std::endl;
}



/* This config maker is based upon the cutoff function H(k) instead of kmax */
void config_maker_1(  std::vector< std::vector<comp> > &p_config,
                    comp En,
                    std::vector<comp> total_P,
                    double mi,
                    double mj, 
                    double mk,
                    double L,
                    double xi, 
                    double epsilon_h,
                    double tolerance    )
{
    char debug = 'n';
    comp pi = std::acos(-1.0); 
    comp Lby2pi = ((comp) L)/((comp) 2.0*pi);
    comp twopibyL = ((comp) 2.0*pi)/((comp) L);

    //comp cutoff = cutoff_function_1(sig_k,mj, mk, epsilon_h);
    //comp kmax = pmom(En,0.0,mi);
    comp kmax = kmax_for_P0(En, mi);

    int nmax = 20;//(int)ceil((L/(2.0*pi))*abs(kmax));
    //std::cout<<"Nmax = "<<nmax<<std::endl;

    int nmaxsq = nmax*nmax; 

    for(int i=-nmax; i<nmax + 1; ++i)
    {
        for(int j=-nmax; j<nmax + 1; ++j)
        {
            for(int k=-nmax; k<nmax + 1; ++k)
            {
                int nsq = i*i + j*j + k*k; 
                //if(nsq<=nmaxsq)
                {
                    comp px = (((comp)2.0)*pi/((comp)xi*(comp)L))*((comp)i);
                    comp py = (((comp)2.0)*pi/((comp)xi*(comp)L))*((comp)j);
                    comp pz = (((comp)2.0)*pi/((comp)xi*(comp)L))*((comp)k); 
                    comp p = std::sqrt(px*px + py*py + pz*pz);

                    comp Px = total_P[0];
                    comp Py = total_P[1];
                    comp Pz = total_P[2];

                    comp Pminusp_x = Px - px;
                    comp Pminusp_y = Py - py;
                    comp Pminusp_z = Pz - pz; 
                    comp Pminusp = std::sqrt(Pminusp_x*Pminusp_x + Pminusp_y*Pminusp_y + Pminusp_z*Pminusp_z);


                    comp sig_k = (En - omega_func(p,mi))*(En - omega_func(p,mi)) - Pminusp*Pminusp; 

                    comp cutoff = cutoff_function_1(sig_k, mj, mk, epsilon_h);

                    //std::cout<<"kmax = "<<kmax<<'\t'<<"p = "<<p<<'\t'<<"sigi = "<<sig_k<<'\t'<<"cutoff = "<<cutoff<<std::endl; 
                    //std::cout<<"mi="<<mi<<'\t'<<"mj="<<mj<<'\t'<<"mk="<<mk<<std::endl;
                    double tmp = real(cutoff);
                    //std::cout<<"cutoff = "<<cutoff<<std::endl; 
                    if(tmp<tolerance) tmp = 0.0;
                    //if(abs(p)<=abs(kmax))

                    //this was set last time 
                    //if(tmp>0.0 && abs(p)<abs(kmax)) 
                    
                    if(tmp>0.0) 
                    {
                        p_config[0].push_back(px);
                        p_config[1].push_back(py);
                        p_config[2].push_back(pz);
                        if(debug=='y')
                        {
                            std::cout << "cutoff = " << cutoff << std::endl;
                            std::cout << "n = " << i << j << k << " nsq = " << nsq << " nmaxsq = " << nmaxsq << std::endl; 
                            std::cout << "px = " << px << " py = " << py << " pz = " << pz << std::endl;
                            std::cout << "p = " << p << " kmax = " << kmax << std::endl;
                        } 
                 
                    }
                    else 
                    {
                        continue; 
                    }
                }
            }
        }
    }
    int check_p_size = 0;
    int psize0 = p_config[0].size();
    int psize1 = p_config[1].size();
    int psize2 = p_config[2].size();

    if(debug=='y')
    {
        std::cout<<"size1 = "<<psize0<<'\t'<<"size2 = "<<psize1<<'\t'<<"size3 = "<<psize2<<std::endl; 
    }

    if(psize0==0 || psize1==0 || psize2==0)
    {
        p_config[0].push_back(0.0);
        p_config[1].push_back(0.0);
        p_config[2].push_back(0.0);
    }
    //std::cout << "nmax = " << nmax << '\t' << "nval = " << (L/(2.0*pi))*abs(kmax) << '\t' 
    //          << "kmax = " << kmax << std::endl;
}

/* This config maker is based upon the cutoff function H(k) instead of kmax
   This config maker returns the shells as well the p_config  */
void config_maker_2(  std::vector< std::vector<comp> > &p_config,
                    std::vector< std::vector<int> > &n_config, 
                    comp En,
                    std::vector<comp> total_P,
                    double mi,
                    double mj, 
                    double mk,
                    double L,
                    double xi, 
                    double epsilon_h,
                    double tolerance    )
{
    char debug = 'n';
    comp pi = std::acos(-1.0);

    //comp cutoff = cutoff_function_1(sig_k,mj, mk, epsilon_h);
    //comp kmax = pmom(En,0.0,mi);
    comp kmax = kmax_for_P0(En, mi);

    int nmax = 20;//(int)ceil((L/(2.0*pi))*abs(kmax));
    //std::cout<<"Nmax = "<<nmax<<std::endl;

    int nmaxsq = nmax*nmax; 

    for(int i=-nmax; i<nmax + 1; ++i)
    {
        for(int j=-nmax; j<nmax + 1; ++j)
        {
            for(int k=-nmax; k<nmax + 1; ++k)
            {
                int nsq = i*i + j*j + k*k; 
                //if(nsq<=nmaxsq)
                {
                    comp px = (((comp)2.0)*pi/((comp)xi*(comp)L))*((comp)i);
                    comp py = (((comp)2.0)*pi/((comp)xi*(comp)L))*((comp)j);
                    comp pz = (((comp)2.0)*pi/((comp)xi*(comp)L))*((comp)k); 


                    comp p = std::sqrt(px*px + py*py + pz*pz);

                    comp Px = total_P[0];
                    comp Py = total_P[1];
                    comp Pz = total_P[2];

                    comp Pminusp_x = Px - px;
                    comp Pminusp_y = Py - py;
                    comp Pminusp_z = Pz - pz; 
                    comp Pminusp = std::sqrt(Pminusp_x*Pminusp_x + Pminusp_y*Pminusp_y + Pminusp_z*Pminusp_z);


                    comp sig_k = (En - omega_func(p,mi))*(En - omega_func(p,mi)) - Pminusp*Pminusp; 

                    comp cutoff = cutoff_function_1(sig_k, mj, mk, epsilon_h);

                    //std::cout<<"kmax = "<<kmax<<'\t'<<"p = "<<p<<'\t'<<"sigi = "<<sig_k<<'\t'<<"cutoff = "<<cutoff<<std::endl; 
                    //std::cout<<"mi="<<mi<<'\t'<<"mj="<<mj<<'\t'<<"mk="<<mk<<std::endl;
                    double tmp = real(cutoff);
                    //std::cout<<"cutoff = "<<cutoff<<std::endl; 
                    if(tmp<tolerance) tmp = 0.0;
                    //if(abs(p)<=abs(kmax))

                    //this was set last time 
                    //if(tmp>0.0 && abs(p)<abs(kmax)) 
                    
                    if(tmp>0.0) 
                    {
                        p_config[0].push_back(px);
                        p_config[1].push_back(py);
                        p_config[2].push_back(pz);
                        n_config[0].push_back(i);
                        n_config[1].push_back(j); 
                        n_config[2].push_back(k);
                        if(debug=='y')
                        {
                            std::cout << "cutoff = " << cutoff << std::endl;
                            std::cout << "n = " << i << j << k << " nsq = " << nsq << " nmaxsq = " << nmaxsq << std::endl; 
                            std::cout << "px = " << px << " py = " << py << " pz = " << pz << std::endl;
                            std::cout << "p = " << p << " kmax = " << kmax << std::endl;
                        } 
                 
                    }
                    else 
                    {
                        continue; 
                    }
                }
            }
        }
    }
    int check_p_size = 0;
    int psize0 = p_config[0].size();
    int psize1 = p_config[1].size();
    int psize2 = p_config[2].size();

    if(debug=='y')
    {
        std::cout<<"size1 = "<<psize0<<'\t'<<"size2 = "<<psize1<<'\t'<<"size3 = "<<psize2<<std::endl; 
    }

    if(psize0==0 || psize1==0 || psize2==0)
    {
        p_config[0].push_back(0.0);
        p_config[1].push_back(0.0);
        p_config[2].push_back(0.0);
        n_config[0].push_back(0);
        n_config[0].push_back(0); 
        n_config[0].push_back(0); 
    }
    //std::cout << "nmax = " << nmax << '\t' << "nval = " << (L/(2.0*pi))*abs(kmax) << '\t' 
    //          << "kmax = " << kmax << std::endl;
}


comp particle_energy(   comp spec_k,
                        double m    )
{
    comp omgk = omega_func(spec_k, m); 

    return omgk; 
}

void non_int_spectrum_config_maker( std::vector<std::vector<int> > &n_config,
                                    int nmax, //maximum shell that it is going to loop over
                                    int nsq_max ) //nsq_max is set as a cut off for the shell maker
{
    for(int i=-nmax; i<nmax+1; ++i)
    {
        for(int j=-nmax; j<nmax+1; ++j)
        {
            for(int k=-nmax; k<nmax+1; ++k)
            {
                int nx = i;
                int ny = j;
                int nz = k; 

                int nsq = i*i + j*j + k*k; 
                if(nsq<=nsq_max)
                {
                    n_config[0].push_back(i);
                    n_config[1].push_back(j);
                    n_config[2].push_back(k); 
                }
            }
        }
    }
}

//This config maker makes the plm_config vectors which
//will have 5 elements: px, py, pz, ell, proj_m 
//based on the waves_vec supplied which contains ell
//values as waves_vec = [0,1], 0 means 'S' wave and 
//1 means 'P' wave and so on, this function will create
//config with corresponding ell and proj_m values 
void config_maker_3(  
                    std::vector< std::vector<comp> > &plm_config,
                    std::vector<int> waves_vec, 
                    comp En,
                    std::vector<comp> total_P,
                    double mi,
                    double mj, 
                    double mk,
                    double L,
                    double epsilon_h,
                    double max_num_shell, 
                    double tolerance    
                )
{
    char debug = 'n';
    comp pi = std::acos(-1.0); 
    comp Lby2pi = ((comp) L)/((comp) 2.0*pi);
    comp twopibyL = ((comp) 2.0*pi)/((comp) L);

    //comp cutoff = cutoff_function_1(sig_k,mj, mk, epsilon_h);
    //comp kmax = pmom(En,0.0,mi);
    comp kmax = kmax_for_P0(En, mi);

    int nmax = max_num_shell;//(int)ceil((L/(2.0*pi))*abs(kmax));
    //std::cout<<"Nmax = "<<nmax<<std::endl;

    std::vector<int> ell_vec;
    std::vector<int> proj_m_vec; 

    std::vector< std::vector<int> > ell_m_vec(2, std::vector<int>()); 

    for(int i=0; i<waves_vec.size(); ++i)
    {
        int ell = waves_vec[i]; 
         

        std::vector<int> m_vec;
        ell_m_vector(ell, m_vec); 
        for(int j=0; j<m_vec.size(); ++j)
        {
            int proj_m = m_vec[j]; 
            ell_m_vec[0].push_back(ell);
            ell_m_vec[1].push_back(proj_m);
        }
    }

    int nmaxsq = nmax*nmax; 

    for(int lm=0; lm<ell_m_vec[0].size(); ++lm)
    {
        int ell = ell_m_vec[0][lm];
        int proj_m = ell_m_vec[1][lm]; 

        for(int i=-nmax; i<nmax + 1; ++i)
        {
            for(int j=-nmax; j<nmax + 1; ++j)
            {
                for(int k=-nmax; k<nmax + 1; ++k)
                {
                    int nsq = i*i + j*j + k*k; 
                    //if(nsq<=nmaxsq)
                    {
                        comp px = (((comp)2.0)*pi/((comp)L))*((comp)i);
                        comp py = (((comp)2.0)*pi/((comp)L))*((comp)j);
                        comp pz = (((comp)2.0)*pi/((comp)L))*((comp)k); 

                        comp p = std::sqrt(px*px + py*py + pz*pz);

                        comp Px = total_P[0];
                        comp Py = total_P[1];
                        comp Pz = total_P[2];

                        comp Pminusp_x = Px - px;
                        comp Pminusp_y = Py - py;
                        comp Pminusp_z = Pz - pz; 
                        comp Pminusp = std::sqrt(Pminusp_x*Pminusp_x + Pminusp_y*Pminusp_y + Pminusp_z*Pminusp_z);


                        comp sig_k = (En - omega_func(p,mi))*(En - omega_func(p,mi)) - Pminusp*Pminusp; 

                        comp cutoff = cutoff_function_1(sig_k, mj, mk, epsilon_h);

                        //std::cout<<"kmax = "<<kmax<<'\t'<<"p = "<<p<<'\t'<<"sigi = "<<sig_k<<'\t'<<"cutoff = "<<cutoff<<std::endl; 
                        //std::cout<<"mi="<<mi<<'\t'<<"mj="<<mj<<'\t'<<"mk="<<mk<<std::endl;
                        double tmp = real(cutoff);
                        //std::cout<<"cutoff = "<<cutoff<<std::endl; 
                        if(tmp<tolerance) tmp = 0.0;
                        //if(abs(p)<=abs(kmax))

                        //this was set last time 
                        //if(tmp>0.0 && abs(p)<abs(kmax)) 

                        
                        if(tmp>0.0) 
                        {
                            plm_config[0].push_back(px);
                            plm_config[1].push_back(py);
                            plm_config[2].push_back(pz);
                            plm_config[3].push_back(ell); 
                            plm_config[4].push_back(proj_m); 
                            
                            if(debug=='y')
                            {
                                std::cout << "cutoff = " << cutoff << std::endl;
                                std::cout << "n = " << i << j << k << " nsq = " << nsq << " nmaxsq = " << nmaxsq << std::endl; 
                                std::cout << "px = " << px << " py = " << py << " pz = " << pz << std::endl;
                                std::cout << "p = " << p << " kmax = " << kmax << std::endl;
                                std::cout << "ell = " << ell << " proj_m = " << proj_m << std::endl; 
                            } 
                    
                        }
                        else 
                        {
                            continue; 
                        }
                    }
                }
            }
        }
    }
    int check_p_size = 0;
    int psize0 = plm_config[0].size();
    int psize1 = plm_config[1].size();
    int psize2 = plm_config[2].size();
    int psize3 = plm_config[3].size();
    int psize4 = plm_config[4].size();

    if(debug=='y')
    {
        std::cout<<"size1 = "<<psize0<<'\t'<<"size2 = "<<psize1<<'\t'<<"size3 = "<<psize2<<'\t'<<"size4 = "<<psize3<<'\t'<<"size5 = "<<psize4<<std::endl; 
    }

    /*
    if(psize0==0 || psize1==0 || psize2==0)
    {
        plm_config[0].push_back(0.0);
        plm_config[1].push_back(0.0);
        plm_config[2].push_back(0.0);
        plm_config[3].push_back(0.0);
        plm_config[4].push_back(0.0);
    }
    */ 
    //std::cout << "nmax = " << nmax << '\t' << "nval = " << (L/(2.0*pi))*abs(kmax) << '\t' 
    //          << "kmax = " << kmax << std::endl;
}

template <typename T>
void print_config_5col(const std::vector<std::vector<T>>& config,
                       const std::string& name = "config",
                       double tol = 1e-12)
{
    if (config.size() < 5) {
        throw std::runtime_error("print_config_5col: config must have at least 5 rows");
    }

    const std::size_t N = config[0].size();

    for (std::size_t r = 1; r < 5; ++r) {
        if (config[r].size() != N) {
            throw std::runtime_error("print_config_5col: inconsistent row sizes in config");
        }
    }

    std::cout << name << ":\n";
    std::cout << std::setprecision(16);

    for (std::size_t i = 0; i < N; ++i) {
        if constexpr (std::is_same_v<T, comp>) {
            if (std::abs(config[3][i].imag()) > tol || std::abs(config[4][i].imag()) > tol) {
                throw std::runtime_error("print_config_5col: ell or proj_m has nonzero imaginary part");
            }

            int ell    = static_cast<int>(std::llround(config[3][i].real()));
            int proj_m = static_cast<int>(std::llround(config[4][i].real()));

            std::cout << "(" << i
                      << ", p=[" << config[0][i]
                      << ", "    << config[1][i]
                      << ", "    << config[2][i]
                      << "], ell=" << ell
                      << ", proj_m=" << proj_m
                      << ")\n";
        }
        else if constexpr (std::is_integral_v<T>) {
            int ell    = config[3][i];
            int proj_m = config[4][i];

            std::cout << "(" << i
                      << ", p=[" << config[0][i]
                      << ", "    << config[1][i]
                      << ", "    << config[2][i]
                      << "], ell=" << ell
                      << ", proj_m=" << proj_m
                      << ")\n";
        }
        else {
            std::cout << "(" << i
                      << ", p=[" << config[0][i]
                      << ", "    << config[1][i]
                      << ", "    << config[2][i]
                      << "], ell=" << config[3][i]
                      << ", proj_m=" << config[4][i]
                      << ")\n";
        }
    }
}

//There are two En_min for configs, one with plus sign
//and the other with minus sign, we take the plus sign 
//to be the only En_min
comp En_min_plus_for_config(    
    std::vector<comp> k, //spectator momentum 
    std::vector<comp> total_P,
    double mi, 
    double mj, 
    double mk, 
    double zval, // this is set to be 1.0e-2, sigma/4m 
    double epsilon_h
)
{
    comp kx = k[0]; 
    comp ky = k[1]; 
    comp kz = k[2]; 
    comp Px = total_P[0]; 
    comp Py = total_P[1]; 
    comp Pz = total_P[2]; 

    comp ksq = kx*kx + ky*ky + kz*kz; 
    comp omgk = std::sqrt(ksq + mi*mi); 

    comp Pminksq = (Px - kx)*(Px - kx) + (Py - ky)*(Py - ky) + (Pz - kz)*(Pz - kz);
    comp sigma_min_i = std::abs(mj*mj - mk*mk) + (2.0*zval*(mj + mk)*std::min(mj,mk))/(1.0 + epsilon_h); 
    
    comp En_result = omgk + std::sqrt(sigma_min_i + Pminksq);

    return En_result; 
}

comp En_min_minus_for_config(    
    std::vector<comp> k, //spectator momentum 
    std::vector<comp> total_P,
    double mi, 
    double mj, 
    double mk, 
    double zval, // this is set to be 1.0e-2, sigma/4m 
    double epsilon_h
)
{
    comp kx = k[0]; 
    comp ky = k[1]; 
    comp kz = k[2]; 
    comp Px = total_P[0]; 
    comp Py = total_P[1]; 
    comp Pz = total_P[2]; 

    comp ksq = kx*kx + ky*ky + kz*kz; 
    comp omgk = std::sqrt(ksq + mi*mi); 

    comp Pminksq = (Px - kx)*(Px - kx) + (Py - ky)*(Py - ky) + (Pz - kz)*(Pz - kz);
    comp sigma_min_i = std::abs(mj*mj - mk*mk) + (2.0*zval*(mj + mk)*std::min(mj,mk))/(1.0 + epsilon_h); 
    
    comp En_result = omgk - std::sqrt(sigma_min_i + Pminksq);

    return En_result; 
}

//This config maker makes the plm_config vectors which
//will have 5 elements: px, py, pz, ell, proj_m 
//based on the waves_vec supplied which contains ell
//values as waves_vec = [0,1], 0 means 'S' wave and 
//1 means 'P' wave and so on, this function will create
//config with corresponding ell and proj_m values 
//this includes the integer n list as well
void config_maker_4(  
                    std::vector< std::vector<comp> > &plm_config,
                    std::vector< std::vector<int> > &n_config, 
                    std::vector<int> waves_vec, 
                    comp En,
                    std::vector<comp> total_P,
                    double mi,
                    double mj, 
                    double mk,
                    double L,
                    double epsilon_h,
                    double max_num_shell, 
                    double tolerance    
                )
{
    char debug = 'n';
    comp pi = std::acos(-1.0); 
    comp Lby2pi = ((comp) L)/((comp) 2.0*pi);
    comp twopibyL = ((comp) 2.0*pi)/((comp) L);

    //comp cutoff = cutoff_function_1(sig_k,mj, mk, epsilon_h);
    //comp kmax = pmom(En,0.0,mi);
    comp kmax = kmax_for_P0(En, mi);

    int nmax = max_num_shell;//(int)ceil((L/(2.0*pi))*abs(kmax));
    //std::cout<<"Nmax = "<<nmax<<std::endl;

    std::vector<int> ell_vec;
    std::vector<int> proj_m_vec; 

    std::vector< std::vector<int> > ell_m_vec(2, std::vector<int>()); 

    for(int i=0; i<waves_vec.size(); ++i)
    {
        int ell = waves_vec[i]; 
         

        std::vector<int> m_vec;
        ell_m_vector(ell, m_vec); 
        for(int j=0; j<m_vec.size(); ++j)
        {
            int proj_m = m_vec[j]; 
            ell_m_vec[0].push_back(ell);
            ell_m_vec[1].push_back(proj_m);
        }
    }

    int nmaxsq = nmax*nmax; 

    for(int lm=0; lm<ell_m_vec[0].size(); ++lm)
    {
        int ell = ell_m_vec[0][lm];
        int proj_m = ell_m_vec[1][lm]; 

        for(int i=-nmax; i<nmax + 1; ++i)
        {
            for(int j=-nmax; j<nmax + 1; ++j)
            {
                for(int k=-nmax; k<nmax + 1; ++k)
                {
                    int nsq = i*i + j*j + k*k; 
                    //if(nsq<=nmaxsq)
                    {
                        comp px = (((comp)2.0)*pi/((comp)L))*((comp)i);
                        comp py = (((comp)2.0)*pi/((comp)L))*((comp)j);
                        comp pz = (((comp)2.0)*pi/((comp)L))*((comp)k); 

                        std::vector<comp> pvec(3); 
                        pvec[0] = px; 
                        pvec[1] = py; 
                        pvec[2] = pz; 

                        comp p = std::sqrt(px*px + py*py + pz*pz);

                        comp Px = total_P[0];
                        comp Py = total_P[1];
                        comp Pz = total_P[2];

                        comp Pminusp_x = Px - px;
                        comp Pminusp_y = Py - py;
                        comp Pminusp_z = Pz - pz; 
                        comp Pminusp = std::sqrt(Pminusp_x*Pminusp_x + Pminusp_y*Pminusp_y + Pminusp_z*Pminusp_z);


                        comp sig_k = (En - omega_func(p,mi))*(En - omega_func(p,mi)) - Pminusp*Pminusp; 

                        comp cutoff = cutoff_function_1(sig_k, mj, mk, epsilon_h);

                        //std::cout<<"kmax = "<<kmax<<'\t'<<"p = "<<p<<'\t'<<"sigi = "<<sig_k<<'\t'<<"cutoff = "<<cutoff<<std::endl; 
                        //std::cout<<"mi="<<mi<<'\t'<<"mj="<<mj<<'\t'<<"mk="<<mk<<std::endl;
                        double tmp = real(cutoff);
                        //std::cout<<"cutoff = "<<cutoff<<std::endl; 
                        if(tmp<tolerance) tmp = 0.0;
                        //if(abs(p)<=abs(kmax))

                        //this was set last time 
                        //if(tmp>0.0 && abs(p)<abs(kmax)) 

                        comp En_min_plus = En_min_plus_for_config( pvec, total_P, mi, mj, mk, 0.02, epsilon_h); 

                        comp En_min_minus = En_min_minus_for_config( pvec, total_P, mi, mj, mk, 0.0000001, epsilon_h); 

                        
                        //if(tmp>0.0) 
                        if(En.real()>En_min_plus.real())
                        {
                            plm_config[0].push_back(px);
                            plm_config[1].push_back(py);
                            plm_config[2].push_back(pz);
                            plm_config[3].push_back(ell); 
                            plm_config[4].push_back(proj_m); 
                            n_config[0].push_back(i); 
                            n_config[1].push_back(j);
                            n_config[2].push_back(k);
                            n_config[3].push_back(ell); 
                            n_config[4].push_back(proj_m); 
                            
                            if(debug=='y')
                            {
                                std::cout << "config selected" << std::endl; 
                                std::cout << "cutoff = " << cutoff << std::endl;
                                std::cout << "n = " << i << "," << j << "," << k << " nsq = " << nsq << " nmaxsq = " << nmaxsq << std::endl;  
                                std::cout << "px = " << px << " py = " << py << " pz = " << pz << std::endl;
                                std::cout << "p = " << p << " kmax = " << kmax << std::endl;
                                std::cout << "ell = " << ell << " proj_m = " << proj_m << std::endl; 
                                std::cout << "En = " << En << " Emin+ = " << En_min_plus << " Emin- = " << En_min_minus << std::endl; 
                                std::cout << "_________________________________________" << std::endl;
                            } 
                    
                        }
                        else 
                        {
                            if(debug=='y')
                            {
                                std::cout << "not selected" << std::endl; 
                                std::cout << "cutoff = " << cutoff << std::endl;
                                std::cout << "n = " << i << "," << j << "," << k << " nsq = " << nsq << " nmaxsq = " << nmaxsq << std::endl; 
                                std::cout << "px = " << px << " py = " << py << " pz = " << pz << std::endl;
                                std::cout << "p = " << p << " kmax = " << kmax << std::endl;
                                std::cout << "ell = " << ell << " proj_m = " << proj_m << std::endl; 
                                std::cout << "En = " << En << " Emin+ = " << En_min_plus << " Emin- = " << En_min_minus << std::endl; 
                                std::cout << "---------------------------------------------------" << std::endl;
                            } 
                            continue; 
                        }
                    }
                }
            }
        }
    }
    

    if(debug=='y')
    {
        int check_p_size = 0;
        int psize0 = plm_config[0].size();
        int psize1 = plm_config[1].size();
        int psize2 = plm_config[2].size();
        int psize3 = plm_config[3].size();
        int psize4 = plm_config[4].size();
        std::cout<<"size1 = "<<psize0<<'\t'<<"size2 = "<<psize1<<'\t'<<"size3 = "<<psize2<<'\t'<<"size4 = "<<psize3<<'\t'<<"size5 = "<<psize4<<std::endl; 
    }

    /*
    if(psize0==0 || psize1==0 || psize2==0)
    {
        plm_config[0].push_back(0.0);
        plm_config[1].push_back(0.0);
        plm_config[2].push_back(0.0);
        plm_config[3].push_back(0.0);
        plm_config[4].push_back(0.0);
    }
    */ 
    //std::cout << "nmax = " << nmax << '\t' << "nval = " << (L/(2.0*pi))*abs(kmax) << '\t' 
    //          << "kmax = " << kmax << std::endl;
}

//This config maker keeps momentum outermost and then appends all (ell,m)
//for each accepted spectator momentum.  It is intended for the v30/v30b
//p-wave projection/Vsel diagnostic studies only.  The original
//config_maker_4 above is intentionally left unchanged for legacy code.
void config_maker_4_momentum_first(  
                    std::vector< std::vector<comp> > &plm_config,
                    std::vector< std::vector<int> > &n_config, 
                    std::vector<int> waves_vec, 
                    comp En,
                    std::vector<comp> total_P,
                    double mi,
                    double mj, 
                    double mk,
                    double L,
                    double epsilon_h,
                    double max_num_shell, 
                    double tolerance    
                )
{
    // v30 change:
    // Previous order was lm outermost, then nx,ny,nz.  For waves_vec={0,1},
    // that produced all s-wave momenta first, then all p-wave m=-1,0,+1
    // momenta.  The p-wave projection/Vsel branch tracking is easier to debug
    // if all partial-wave components for a selected spectator momentum are
    // contiguous.  Therefore v30 uses momentum outermost:
    //   for nx,ny,nz selected by the energy condition:
    //       for every (ell,m) in waves_vec-expanded order:
    //           append (px,py,pz,ell,m)
    // This does not change the set of states; it changes only basis ordering.
    char debug = 'n';
    comp pi = std::acos(-1.0); 
    comp twopibyL = ((comp) 2.0*pi)/((comp) L);

    int nmax = static_cast<int>(max_num_shell);

    std::vector< std::vector<int> > ell_m_vec(2, std::vector<int>()); 
    for(int iw=0; iw<static_cast<int>(waves_vec.size()); ++iw)
    {
        int ell = waves_vec[iw]; 
        std::vector<int> m_vec;
        ell_m_vector(ell, m_vec); 
        for(int jm=0; jm<static_cast<int>(m_vec.size()); ++jm)
        {
            int proj_m = m_vec[jm]; 
            ell_m_vec[0].push_back(ell);
            ell_m_vec[1].push_back(proj_m);
        }
    }

    int candidate_momenta = 0;
    int selected_momenta = 0;

    for(int i=-nmax; i<nmax + 1; ++i)
    {
        for(int j=-nmax; j<nmax + 1; ++j)
        {
            for(int k=-nmax; k<nmax + 1; ++k)
            {
                ++candidate_momenta;

                comp px = twopibyL*((comp)i);
                comp py = twopibyL*((comp)j);
                comp pz = twopibyL*((comp)k); 

                std::vector<comp> pvec(3); 
                pvec[0] = px; 
                pvec[1] = py; 
                pvec[2] = pz; 

                comp Pminusp_x = total_P[0] - px;
                comp Pminusp_y = total_P[1] - py;
                comp Pminusp_z = total_P[2] - pz; 
                comp Pminusp = std::sqrt(Pminusp_x*Pminusp_x + Pminusp_y*Pminusp_y + Pminusp_z*Pminusp_z);

                comp sig_k = (En - omega_func(std::sqrt(px*px + py*py + pz*pz),mi))
                           * (En - omega_func(std::sqrt(px*px + py*py + pz*pz),mi))
                           - Pminusp*Pminusp; 
                comp cutoff = cutoff_function_1(sig_k, mj, mk, epsilon_h);
                double tmp = real(cutoff);
                if(tmp<tolerance) tmp = 0.0;

                comp En_min_plus = En_min_plus_for_config( pvec, total_P, mi, mj, mk, 0.02, epsilon_h); 
                comp En_min_minus = En_min_minus_for_config( pvec, total_P, mi, mj, mk, 0.0000001, epsilon_h); 

                if(En.real()>En_min_plus.real())
                {
                    ++selected_momenta;
                    for(int lm=0; lm<static_cast<int>(ell_m_vec[0].size()); ++lm)
                    {
                        int ell = ell_m_vec[0][lm];
                        int proj_m = ell_m_vec[1][lm]; 

                        plm_config[0].push_back(px);
                        plm_config[1].push_back(py);
                        plm_config[2].push_back(pz);
                        plm_config[3].push_back(ell); 
                        plm_config[4].push_back(proj_m); 
                        n_config[0].push_back(i); 
                        n_config[1].push_back(j);
                        n_config[2].push_back(k);
                        n_config[3].push_back(ell); 
                        n_config[4].push_back(proj_m); 

                        if(debug=='y')
                        {
                            std::cout << "config selected v30 momentum-first" << std::endl; 
                            std::cout << "cutoff = " << cutoff << std::endl;
                            std::cout << "n = " << i << "," << j << "," << k << std::endl;  
                            std::cout << "px = " << px << " py = " << py << " pz = " << pz << std::endl;
                            std::cout << "ell = " << ell << " proj_m = " << proj_m << std::endl; 
                            std::cout << "En = " << En << " Emin+ = " << En_min_plus << " Emin- = " << En_min_minus << std::endl; 
                            std::cout << "_________________________________________" << std::endl;
                        } 
                    }
                }
                else if(debug=='y')
                {
                    std::cout << "not selected v30 momentum-first" << std::endl; 
                    std::cout << "cutoff = " << cutoff << std::endl;
                    std::cout << "n = " << i << "," << j << "," << k << std::endl; 
                    std::cout << "px = " << px << " py = " << py << " pz = " << pz << std::endl;
                    std::cout << "En = " << En << " Emin+ = " << En_min_plus << " Emin- = " << En_min_minus << std::endl; 
                    std::cout << "---------------------------------------------------" << std::endl;
                }
            }
        }
    }

    if(debug=='y')
    {
        std::cout << "[config_maker_4_momentum_first v30] order = momentum_first_then_lm" << std::endl;
        std::cout << "[config_maker_4_momentum_first v30] candidate_momenta = " << candidate_momenta << std::endl;
        std::cout << "[config_maker_4_momentum_first v30] selected_momenta = " << selected_momenta << std::endl;
        std::cout << "[config_maker_4_momentum_first v30] lm_count = " << ell_m_vec[0].size() << std::endl;
        std::cout << "[config_maker_4_momentum_first v30] total rows = " << plm_config[0].size() << std::endl;
        std::cout << "size1 = " << plm_config[0].size() << '\t'
                  << "size2 = " << plm_config[1].size() << '\t'
                  << "size3 = " << plm_config[2].size() << '\t'
                  << "size4 = " << plm_config[3].size() << '\t'
                  << "size5 = " << plm_config[4].size() << std::endl; 
    }
}

//This config maker makes the plm_config vectors which
//will have 5 elements: px, py, pz, ell, proj_m 
//based on the waves_vec supplied which contains ell
//values as waves_vec = [0,1], 0 means 'S' wave and 
//1 means 'P' wave and so on, this function will create
//config with corresponding ell and proj_m values 
//this includes the integer n list called n_config 
//as well as the orbit called n_orbit 
//This is not finished yet!!! DO NOT USE 
void config_maker_5(  
                    std::vector< std::vector<comp> > &plm_config,
                    std::vector< std::vector<int> > &n_config, 
                    std::vector< std::vector<int> > &n_orbit, 
                    std::vector<int> waves_vec, 
                    comp En,
                    std::vector<comp> total_P,
                    double mi,
                    double mj, 
                    double mk,
                    double L,
                    double epsilon_h,
                    double max_num_shell, 
                    double tolerance    
                )
{
    char debug = 'n';
    comp pi = std::acos(-1.0); 
    comp Lby2pi = ((comp) L)/((comp) 2.0*pi);
    comp twopibyL = ((comp) 2.0*pi)/((comp) L);

    //comp cutoff = cutoff_function_1(sig_k,mj, mk, epsilon_h);
    //comp kmax = pmom(En,0.0,mi);
    comp kmax = kmax_for_P0(En, mi);

    int nmax = max_num_shell;//(int)ceil((L/(2.0*pi))*abs(kmax));
    //std::cout<<"Nmax = "<<nmax<<std::endl;

    std::vector<int> ell_vec;
    std::vector<int> proj_m_vec; 

    std::vector< std::vector<int> > ell_m_vec(2, std::vector<int>()); 

    for(int i=0; i<waves_vec.size(); ++i)
    {
        int ell = waves_vec[i]; 
         

        std::vector<int> m_vec;
        ell_m_vector(ell, m_vec); 
        for(int j=0; j<m_vec.size(); ++j)
        {
            int proj_m = m_vec[j]; 
            ell_m_vec[0].push_back(ell);
            ell_m_vec[1].push_back(proj_m);
        }
    }

    int nmaxsq = nmax*nmax; 

    for(int lm=0; lm<ell_m_vec[0].size(); ++lm)
    {
        int ell = ell_m_vec[0][lm];
        int proj_m = ell_m_vec[1][lm]; 

        for(int i=-nmax; i<nmax + 1; ++i)
        {
            for(int j=-nmax; j<nmax + 1; ++j)
            {
                for(int k=-nmax; k<nmax + 1; ++k)
                {
                    int nsq = i*i + j*j + k*k; 
                    //if(nsq<=nmaxsq)
                    {
                        comp px = (((comp)2.0)*pi/((comp)L))*((comp)i);
                        comp py = (((comp)2.0)*pi/((comp)L))*((comp)j);
                        comp pz = (((comp)2.0)*pi/((comp)L))*((comp)k); 

                        comp p = std::sqrt(px*px + py*py + pz*pz);

                        comp Px = total_P[0];
                        comp Py = total_P[1];
                        comp Pz = total_P[2];

                        comp Pminusp_x = Px - px;
                        comp Pminusp_y = Py - py;
                        comp Pminusp_z = Pz - pz; 
                        comp Pminusp = std::sqrt(Pminusp_x*Pminusp_x + Pminusp_y*Pminusp_y + Pminusp_z*Pminusp_z);


                        comp sig_k = (En - omega_func(p,mi))*(En - omega_func(p,mi)) - Pminusp*Pminusp; 

                        comp cutoff = cutoff_function_1(sig_k, mj, mk, epsilon_h);

                        //std::cout<<"kmax = "<<kmax<<'\t'<<"p = "<<p<<'\t'<<"sigi = "<<sig_k<<'\t'<<"cutoff = "<<cutoff<<std::endl; 
                        //std::cout<<"mi="<<mi<<'\t'<<"mj="<<mj<<'\t'<<"mk="<<mk<<std::endl;
                        double tmp = real(cutoff);
                        //std::cout<<"cutoff = "<<cutoff<<std::endl; 
                        if(tmp<tolerance) tmp = 0.0;
                        //if(abs(p)<=abs(kmax))

                        //this was set last time 
                        //if(tmp>0.0 && abs(p)<abs(kmax)) 

                        
                        if(tmp>0.0) 
                        {
                            plm_config[0].push_back(px);
                            plm_config[1].push_back(py);
                            plm_config[2].push_back(pz);
                            plm_config[3].push_back(ell); 
                            plm_config[4].push_back(proj_m); 
                            n_config[0].push_back(i); 
                            n_config[1].push_back(j);
                            n_config[2].push_back(k);
                            n_config[3].push_back(ell); 
                            n_config[4].push_back(proj_m); 
                            
                            if(debug=='y')
                            {
                                std::cout << "cutoff = " << cutoff << std::endl;
                                std::cout << "n = " << i << j << k << " nsq = " << nsq << " nmaxsq = " << nmaxsq << std::endl; 
                                std::cout << "px = " << px << " py = " << py << " pz = " << pz << std::endl;
                                std::cout << "p = " << p << " kmax = " << kmax << std::endl;
                                std::cout << "ell = " << ell << " proj_m = " << proj_m << std::endl; 
                            } 
                    
                        }
                        else 
                        {
                            continue; 
                        }
                    }
                }
            }
        }
    }
    

    if(debug=='y')
    {
        int check_p_size = 0;
        int psize0 = plm_config[0].size();
        int psize1 = plm_config[1].size();
        int psize2 = plm_config[2].size();
        int psize3 = plm_config[3].size();
        int psize4 = plm_config[4].size();
        std::cout<<"size1 = "<<psize0<<'\t'<<"size2 = "<<psize1<<'\t'<<"size3 = "<<psize2<<'\t'<<"size4 = "<<psize3<<'\t'<<"size5 = "<<psize4<<std::endl; 
    }

    /*
    if(psize0==0 || psize1==0 || psize2==0)
    {
        plm_config[0].push_back(0.0);
        plm_config[1].push_back(0.0);
        plm_config[2].push_back(0.0);
        plm_config[3].push_back(0.0);
        plm_config[4].push_back(0.0);
    }
    */ 
    //std::cout << "nmax = " << nmax << '\t' << "nval = " << (L/(2.0*pi))*abs(kmax) << '\t' 
    //          << "kmax = " << kmax << std::endl;
}


//Send this vector p and omega_p
//with E-wk and P-k to get the boosted p*
std::vector<comp> boost(
    comp p0,
    std::vector<comp> p,
    comp E2p,
    std::vector<comp> P2p
)
{
    comp px = p[0];
    comp py = p[1];
    comp pz = p[2];

    comp P2px = P2p[0];
    comp P2py = P2p[1];
    comp P2pz = P2p[2];

    comp zeroval = {0.0, 0.0};

    if(P2px == zeroval && P2py == zeroval && P2pz == zeroval)
    {
        return p;
    }

    comp P2norm = std::sqrt(P2px*P2px + P2py*P2py + P2pz*P2pz);

    std::vector<comp> P2hat(3);
    P2hat[0] = P2px / P2norm;
    P2hat[1] = P2py / P2norm;
    P2hat[2] = P2pz / P2norm;

    comp beta2 = P2norm / E2p;
    comp gam2 = 1.0 / std::sqrt(1.0 - beta2*beta2);

    comp dot_p_P2hat =
        px * P2hat[0] +
        py * P2hat[1] +
        pz * P2hat[2];

    comp common =
        (gam2 - 1.0) * dot_p_P2hat
        - gam2 * beta2 * p0;

    std::vector<comp> out(3);
    out[0] = px + common * P2hat[0];
    out[1] = py + common * P2hat[1];
    out[2] = pz + common * P2hat[2];

    return out;
}



comp threebody_non_int_energy_lab(  double m1, 
                                double m2, 
                                double m3, 
                                comp k1,
                                comp k2, 
                                comp k3 )
{
    comp particle_en1 = particle_energy(k1, m1);
    comp particle_en2 = particle_energy(k2, m2);
    comp particle_en3 = particle_energy(k3, m3);

    comp total_energy = particle_en1 + particle_en2 + particle_en3; 

    return total_energy; 

}

void threebody_non_int_spectrum(    std::string filename, 
                                    double m1, 
                                    double m2, 
                                    double m3, 
                                    std::vector<comp> total_P,
                                    double xi, 
                                    double L,
                                    int nmax, 
                                    int nsq_max )
{
    std::ofstream fout; 
    fout.open(filename.c_str());

    std::vector<std::vector<int> > n_config(3,std::vector<int> ());

    non_int_spectrum_config_maker(n_config, nmax, nsq_max); 

    int size = n_config[0].size(); 

    std::vector<double> energy; 

    double pi = std::acos(-1.0); 
    double twopibyxiL = 2.0*pi/(xi*L);
    
    /* This has been proven multiple times that this is the formulation 
    of non-int spectrum that REDSTAR uses to calculate the non-int spectrum and operators */
    for(int i=0; i<size; ++i)
    {
        for(int j=0; j<size; ++j)
        {
            comp Px = total_P[0];
            comp Py = total_P[1];
            comp Pz = total_P[2];
            comp spec_P = std::sqrt(Px*Px + Py*Py + Pz*Pz);

            comp px1 = twopibyxiL*n_config[0][i];
            comp py1 = twopibyxiL*n_config[1][i];
            comp pz1 = twopibyxiL*n_config[2][i];
            comp spec_p1 = std::sqrt(px1*px1 + py1*py1 + pz1*pz1);
            
            comp px2 = twopibyxiL*n_config[0][j];
            comp py2 = twopibyxiL*n_config[1][j];
            comp pz2 = twopibyxiL*n_config[2][j];
            comp spec_p2 = std::sqrt(px2*px2 + py2*py2 + pz2*pz2);

            comp px3 = Px - px1 - px2; 
            comp py3 = Py - py1 - py2; 
            comp pz3 = Pz - pz1 - pz2; 
            comp spec_p3 = std::sqrt(px3*px3 + py3*py3 + pz3*pz3);

            comp threebodyenergy = threebody_non_int_energy_lab(m1,m2,m3,spec_p1,spec_p2,spec_p3);

            comp Ecm = std::sqrt(threebodyenergy*threebodyenergy - spec_P*spec_P); 

            energy.push_back(real(Ecm)); 
        }
    }
    
    
    std::sort( energy.begin(), energy.end() );
    energy.erase( std::unique( energy.begin(), energy.end() ), energy.end() );

    for(int i=0;i<energy.size();++i)
    {
        //std::cout<<energy[i]<<std::endl; 
        fout<<std::setprecision(20)<<energy[i]<<std::endl;
    }
    fout.close(); 
    std::cout<<"non-int spectrum generated with filename = "<<filename<<std::endl; 


}

void threebody_non_int_spectrum_with_multiplicity(
                                    std::string filename, 
                                    double m1, 
                                    double m2, 
                                    double m3, 
                                    std::vector<comp> total_P,
                                    double xi, 
                                    double L,
                                    int nmax, 
                                    int nsq_max )
{
    std::ofstream fout; 
    fout.open(filename.c_str());

    std::vector<std::vector<int> > n_config(3,std::vector<int> ());

    non_int_spectrum_config_maker(n_config, nmax, nsq_max); 

    int size = n_config[0].size(); 

    std::vector<double> energy; 

    double pi = std::acos(-1.0); 
    double twopibyxiL = 2.0*pi/(xi*L);
    
    /* This has been proven multiple times that this is the formulation 
    of non-int spectrum that REDSTAR uses to calculate the non-int spectrum and operators */
    for(int i=0; i<size; ++i)
    {
        for(int j=0; j<size; ++j)
        {
            comp Px = total_P[0];
            comp Py = total_P[1];
            comp Pz = total_P[2];
            comp spec_P = std::sqrt(Px*Px + Py*Py + Pz*Pz);

            comp px1 = twopibyxiL*n_config[0][i];
            comp py1 = twopibyxiL*n_config[1][i];
            comp pz1 = twopibyxiL*n_config[2][i];
            comp spec_p1 = std::sqrt(px1*px1 + py1*py1 + pz1*pz1);
            
            comp px2 = twopibyxiL*n_config[0][j];
            comp py2 = twopibyxiL*n_config[1][j];
            comp pz2 = twopibyxiL*n_config[2][j];
            comp spec_p2 = std::sqrt(px2*px2 + py2*py2 + pz2*pz2);

            comp px3 = Px - px1 - px2; 
            comp py3 = Py - py1 - py2; 
            comp pz3 = Pz - pz1 - pz2; 
            comp spec_p3 = std::sqrt(px3*px3 + py3*py3 + pz3*pz3);

            comp threebodyenergy = threebody_non_int_energy_lab(m1,m2,m3,spec_p1,spec_p2,spec_p3);

            comp Ecm = std::sqrt(threebodyenergy*threebodyenergy - spec_P*spec_P); 

            energy.push_back(real(Ecm)); 
        }
    }
    
    std::vector<double> energy_full_set = energy; 
    std::sort( energy.begin(), energy.end() );
    energy.erase( std::unique( energy.begin(), energy.end() ), energy.end() );

    for(int i=0;i<energy.size();++i)
    {
        //std::cout<<energy[i]<<std::endl; 
        double given_energy = energy[i];
        int energy_counter = 0; 
        for(int j=0;j<energy_full_set.size();++j)
        {
            if(given_energy==energy_full_set[j])
            {
                energy_counter = energy_counter + 1; 
            }
        }

        fout<<std::setprecision(20)<<energy[i]<<'\t'<<energy_counter<<std::endl;
    }
    fout.close(); 
    std::cout<<"non-int spectrum generated with filename = "<<filename<<std::endl; 


}



void threebody_Gpoles(    std::string filename, 
                                    double m1, 
                                    double m2, 
                                    double m3, 
                                    std::vector<comp> total_P,
                                    double xi, 
                                    double L,
                                    int nmax, 
                                    int nsq_max )
{
    std::ofstream fout; 
    fout.open(filename.c_str());

    std::vector<std::vector<int> > n_config(3,std::vector<int> ());

    non_int_spectrum_config_maker(n_config, nmax, nsq_max); 

    int size = n_config[0].size(); 

    std::vector<double> energy; 

    double pi = std::acos(-1.0); 
    double twopibyxiL = 2.0*pi/(xi*L);
    
    for(int i=0; i<size; ++i)
    {
        for(int j=0; j<size; ++j)
        {
            comp Px = total_P[0];
            comp Py = total_P[1];
            comp Pz = total_P[2];
            comp spec_P = std::sqrt(Px*Px + Py*Py + Pz*Pz);

            comp px1 = twopibyxiL*n_config[0][i];
            comp py1 = twopibyxiL*n_config[1][i];
            comp pz1 = twopibyxiL*n_config[2][i];
            comp spec_p1 = std::sqrt(px1*px1 + py1*py1 + pz1*pz1);
            
            comp px2 = twopibyxiL*n_config[0][j];
            comp py2 = twopibyxiL*n_config[1][j];
            comp pz2 = twopibyxiL*n_config[2][j];
            comp spec_p2 = std::sqrt(px2*px2 + py2*py2 + pz2*pz2);

            comp px3 = Px - px1 - px2; 
            comp py3 = Py - py1 - py2; 
            comp pz3 = Pz - pz1 - pz2; 
            comp spec_p3 = std::sqrt(px3*px3 + py3*py3 + pz3*pz3);

            comp particle_energy1 = particle_energy(spec_p1, m1);
            comp particle_energy2 = particle_energy(spec_p2, m2);
            comp particle_energy3 = particle_energy(spec_p3, m3);

            comp pole_energy1 = particle_energy1 + particle_energy2 + particle_energy3; 
            comp pole_energy2 = particle_energy1 + particle_energy2 - particle_energy3; 

            comp Ecm1 = std::sqrt(pole_energy1*pole_energy1 - spec_P*spec_P); 
            comp Ecm2 = std::sqrt(pole_energy2*pole_energy1 - spec_P*spec_P); 

            energy.push_back(real(Ecm1)); 
            energy.push_back(real(Ecm2)); 
        }
    }
    
   
    std::sort( energy.begin(), energy.end() );
    energy.erase( std::unique( energy.begin(), energy.end() ), energy.end() );

    for(int i=0;i<energy.size();++i)
    {
        //std::cout<<energy[i]<<std::endl; 
        fout<<std::setprecision(20)<<energy[i]<<std::endl;
    }
    fout.close(); 
    std::cout<<"poles of G generated with filename = "<<filename<<std::endl; 


}




#endif