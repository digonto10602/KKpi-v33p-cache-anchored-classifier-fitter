#ifndef SPHERICAL_FUNCTIONS_H
#define SPHERICAL_FUNCTIONS_H

#include<bits/stdc++.h>
#include<cmath>

typedef std::complex<double> comp;

//This is the caligraphic Ylm function seen in 
//arxiv: https://arxiv.org/pdf/2111.12734 
//we use this for general 2+1 QC 
//this is defined using the list of REAL spherical harmonics found in 
//https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
//this function is hard-coded 
comp spherical_harmonics(   std::vector<comp> p,
                            int ell, 
                            int m   )
{
    comp pi = std::acos(-1.0); 
    comp norm = std::sqrt((comp) 4.0*pi); 

    comp px = p[0]; 
    comp py = p[1]; 
    comp pz = p[2]; 
    comp term = {0.0, 0.0}; 

    //Here we calculate |p|^{ell} Y_{ell m}(\hat{p}) 
    if(ell==0)
    {
        term = (comp) 1.0;
    }
    else if(ell==1)
    {
        if(m==-1)
        {
            term = std::sqrt((comp) 3.0)*py; 
        }
        else if(m==0)
        {
            term = std::sqrt((comp) 3.0)*pz;
        }
        else if(m==+1)
        {
            term = std::sqrt((comp) 3.0)*px; 
        }
        else
        {
            std::cerr << "Invalid m=" << m << " for ell=1 in spherical_harmonics\n";
            return comp{0.0, 0.0};
        }
         

    }
    else if(ell==2)
    {
        if(m==-2)
        {
            term = std::sqrt((comp) 15.0)*px*py; 
        }
        else if(m==-1)
        {
            term = std::sqrt((comp) 15.0)*py*pz; 
        }
        else if(m==0)
        {
            double val = 5.0/4.0; 
            term = std::sqrt((comp) val)*(((comp) 2.0)*pz*pz - px*px - py*py); 
        }
        else if(m==+1)
        {
            term = std::sqrt((comp) 15.0)*px*pz; 
        }
        else if(m==+2)
        {
            double val = 5.0/4.0;
            term = std::sqrt((comp) val)*(px*px - py*py); 
        }
        else
        {
            std::cerr << "Invalid m=" << m << " for ell=2 in spherical_harmonics\n";
            return comp{0.0, 0.0};
        }
    }
    else
    {
        std::cerr << "Invalid ell=" << ell << " not added to the package yet, only upto ell=2 available!!!\n";
        return comp{0.0, 0.0};
    }

    return term; 
}

//For a particular value of ell, this generates a
//vector of m values going from -|ell| -> +|ell| in 
//step of 1 
void ell_m_vector(  int ell,
                    std::vector<int> &m_vec )
{
    if(ell==0)
    {
        m_vec.push_back(0); 
    }
    else
    {
        int ini = -std::abs(ell);
        int fin = +std::abs(ell); 
        int del = 1; 
        int points = std::abs(ini - fin)/del;

        for(int i=0; i<points+1; ++i)
        {
            int m = ini + i*del; 
            m_vec.push_back(m); 
            //std::cout<<"m val = "<<m<<std::endl; 
        }
    }
}

#endif 