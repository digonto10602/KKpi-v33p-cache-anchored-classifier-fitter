#ifndef F2FUNCTIONSV2_H
#define F2FUNCTIONSV2_H


#include<bits/stdc++.h>
#include<cmath>
#include<Eigen/Dense>
#include <vector>
#include <complex>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <limits>
#include <stdexcept>





#include "functions.h"
//#include "gsl/gsl_sf_dawson.h"
//#include<Eigen/Dense>

//#include "Faddeeva.cc"
#include "Faddeeva.hh"
#include "spherical_functions.h"



using comp = std::complex<double>;

comp smallest_eigenvalue(const Eigen::MatrixXcd& A)
{
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("smallest_eigenvalue: matrix must be square");
    }

    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> ces;
    ces.compute(A, false);  // false = eigenvalues only

    if (ces.info() != Eigen::Success) {
        throw std::runtime_error("smallest_eigenvalue: Eigen decomposition failed");
    }

    Eigen::VectorXcd eigvals = ces.eigenvalues();

    int min_idx = 0;
    double min_abs = std::abs(eigvals(0));

    for (int i = 1; i < eigvals.size(); ++i) {
        double abs_val = std::abs(eigvals(i));

        if (abs_val < min_abs) {
            min_abs = abs_val;
            min_idx = i;
        }
    }

    return eigvals(min_idx);
}

comp sorted_neumaier_sum(std::vector<comp> bare_sum_vec)
{
    long double sum_re = 0.0L;
    long double sum_im = 0.0L;

    long double c_re = 0.0L;
    long double c_im = 0.0L;

    // Sort small magnitude to large magnitude
    std::sort(
        bare_sum_vec.begin(),
        bare_sum_vec.end(),
        [](const comp& a, const comp& b) {
            return std::norm(a) < std::norm(b);
        }
    );

    for (const auto& z : bare_sum_vec)
    {
        long double x_re = static_cast<long double>(z.real());
        long double x_im = static_cast<long double>(z.imag());

        // Neumaier compensation for real part
        long double t_re = sum_re + x_re;

        if (std::abs(sum_re) >= std::abs(x_re)) {
            c_re += (sum_re - t_re) + x_re;
        } else {
            c_re += (x_re - t_re) + sum_re;
        }

        sum_re = t_re;

        // Neumaier compensation for imaginary part
        long double t_im = sum_im + x_im;

        if (std::abs(sum_im) >= std::abs(x_im)) {
            c_im += (sum_im - t_im) + x_im;
        } else {
            c_im += (x_im - t_im) + sum_im;
        }

        sum_im = t_im;
    }

    return comp(
        static_cast<double>(sum_re + c_re),
        static_cast<double>(sum_im + c_im)
    );
}

void print_and_test_vector_sum( int n,
                                std::vector<comp> &v
                            ) {
    //std::vector<comp> v;

    for (int i = 1; i <= n; ++i) {
        v.push_back(comp(static_cast<double>(i), 0.0));
    }

    comp sum = 0.0;

    std::cout << std::scientific << std::setprecision(17);

    std::cout << "vector contents:\n";
    for (int i = 0; i < n; ++i) {
        std::cout << "v[" << i << "] = " << v[i] << "\n";
        sum += v[i];
    }

    comp expected = comp(static_cast<double>(n) * static_cast<double>(n + 1) / 2.0, 0.0);

    std::cout << "\ncomputed sum = " << sum << "\n";
    std::cout << "expected sum = " << expected << "\n";
    std::cout << "difference   = " << sum - expected << "\n";
    std::cout << "abs diff     = " << std::abs(sum - expected) << "\n";

    if (std::abs(sum - expected) < 1.0e-12) {
        std::cout << "PASS: sum is correct\n";
    } else {
        std::cout << "FAIL: sum is not correct\n";
    }
}

comp precise_vector_sum_debug(
    const std::vector<comp>& input_vec,
    const std::string& label = "precise_vector_sum_debug",
    char debug = 'y'
) {
    // ------------------------------------------------------------
    // If debug is off, return the most precise estimate silently:
    // sorted-by-magnitude Neumaier compensated sum.
    // ------------------------------------------------------------

    struct NeumaierComplexSum {
        long double sum_re = 0.0L;
        long double sum_im = 0.0L;

        long double c_re = 0.0L;
        long double c_im = 0.0L;

        void add(comp z) {
            long double x_re = static_cast<long double>(z.real());
            long double x_im = static_cast<long double>(z.imag());

            long double t_re = sum_re + x_re;
            if (std::abs(sum_re) >= std::abs(x_re)) {
                c_re += (sum_re - t_re) + x_re;
            } else {
                c_re += (x_re - t_re) + sum_re;
            }
            sum_re = t_re;

            long double t_im = sum_im + x_im;
            if (std::abs(sum_im) >= std::abs(x_im)) {
                c_im += (sum_im - t_im) + x_im;
            } else {
                c_im += (x_im - t_im) + sum_im;
            }
            sum_im = t_im;
        }

        comp result() const {
            return comp(
                static_cast<double>(sum_re + c_re),
                static_cast<double>(sum_im + c_im)
            );
        }
    };

    auto sequential_sum = [](const std::vector<comp>& v) -> comp {
        comp s = 0.0;
        for (const auto& z : v) {
            s += z;
        }
        return s;
    };

    auto reverse_sum = [](std::vector<comp> v) -> comp {
        std::reverse(v.begin(), v.end());

        comp s = 0.0;
        for (const auto& z : v) {
            s += z;
        }
        return s;
    };

    auto sorted_plain_sum = [](std::vector<comp> v) -> comp {
        std::sort(
            v.begin(),
            v.end(),
            [](const comp& a, const comp& b) {
                return std::norm(a) < std::norm(b);
            }
        );

        comp s = 0.0;
        for (const auto& z : v) {
            s += z;
        }
        return s;
    };

    auto sorted_neumaier_sum = [](std::vector<comp> v) -> comp {
        std::sort(
            v.begin(),
            v.end(),
            [](const comp& a, const comp& b) {
                return std::norm(a) < std::norm(b);
            }
        );

        NeumaierComplexSum s;
        for (const auto& z : v) {
            s.add(z);
        }

        return s.result();
    };

    auto unsorted_neumaier_sum = [](const std::vector<comp>& v) -> comp {
        NeumaierComplexSum s;
        for (const auto& z : v) {
            s.add(z);
        }

        return s.result();
    };

    auto sorted_large_first_plain_sum = [](std::vector<comp> v) -> comp {
        std::sort(
            v.begin(),
            v.end(),
            [](const comp& a, const comp& b) {
                return std::norm(a) > std::norm(b);
            }
        );

        comp s = 0.0;
        for (const auto& z : v) {
            s += z;
        }
        return s;
    };

    comp s_seq                 = sequential_sum(input_vec);
    comp s_reverse             = reverse_sum(input_vec);
    comp s_sorted_plain        = sorted_plain_sum(input_vec);
    comp s_sorted_large_plain  = sorted_large_first_plain_sum(input_vec);
    comp s_unsorted_neumaier   = unsorted_neumaier_sum(input_vec);
    comp s_sorted_neumaier     = sorted_neumaier_sum(input_vec);

    if (debug != 'y') {
        return s_sorted_neumaier;
    }

    if (debug == 'y')
    {

        std::cout << std::scientific << std::setprecision(17);

        std::cout << "\n=====================================================\n";
        std::cout << label << "\n";
        std::cout << "=====================================================\n";

        std::cout << "vector size                         = "
                << input_vec.size() << "\n";
    
    }

    if (input_vec.empty()) {
        std::cout << "WARNING: input vector is empty. Returning zero.\n";
        std::cout << "=====================================================\n\n";
        return comp(0.0, 0.0);
    }

    long double sum_abs_terms = 0.0L;
    double max_abs = 0.0;
    double min_nonzero_abs = std::numeric_limits<double>::infinity();
    size_t max_idx = 0;
    size_t min_nonzero_idx = 0;
    size_t zero_count = 0;
    size_t nan_count = 0;
    size_t inf_count = 0;

    for (size_t i = 0; i < input_vec.size(); ++i) {
        const comp& z = input_vec[i];

        bool bad_re_nan = std::isnan(z.real());
        bool bad_im_nan = std::isnan(z.imag());
        bool bad_re_inf = std::isinf(z.real());
        bool bad_im_inf = std::isinf(z.imag());

        if (bad_re_nan || bad_im_nan) {
            ++nan_count;
        }

        if (bad_re_inf || bad_im_inf) {
            ++inf_count;
        }

        double a = std::abs(z);

        if (a == 0.0) {
            ++zero_count;
        }

        if (std::isfinite(a)) {
            sum_abs_terms += static_cast<long double>(a);

            if (a > max_abs) {
                max_abs = a;
                max_idx = i;
            }

            if (a > 0.0 && a < min_nonzero_abs) {
                min_nonzero_abs = a;
                min_nonzero_idx = i;
            }
        }
    }

    if (debug == 'y')
    {

        std::cout << "zero terms                          = "
                << zero_count << "\n";

        std::cout << "NaN terms                           = "
                << nan_count << "\n";

        std::cout << "Inf terms                           = "
                << inf_count << "\n";

        std::cout << "sum |terms|                         = "
                << static_cast<double>(sum_abs_terms) << "\n";

        std::cout << "max |term|                          = "
                << max_abs << "\n";

        std::cout << "max |term| index                    = "
                << max_idx << "\n";

        std::cout << "max |term| value                    = "
                << input_vec[max_idx] << "\n";

        if (std::isfinite(min_nonzero_abs)) {
            std::cout << "min nonzero |term|                  = "
                    << min_nonzero_abs << "\n";

            std::cout << "min nonzero |term| index            = "
                    << min_nonzero_idx << "\n";

            std::cout << "min nonzero |term| value            = "
                    << input_vec[min_nonzero_idx] << "\n";
        } else {
            std::cout << "min nonzero |term|                  = none\n";
        }

        std::cout << "\n------------------ sums ------------------\n";

        std::cout << "sequential sum                      = "
                << s_seq << "\n";

        std::cout << "reverse sequential sum              = "
                << s_reverse << "\n";

        std::cout << "sorted small-to-large plain sum      = "
                << s_sorted_plain << "\n";

        std::cout << "sorted large-to-small plain sum      = "
                << s_sorted_large_plain << "\n";

        std::cout << "unsorted Neumaier sum               = "
                << s_unsorted_neumaier << "\n";

        std::cout << "sorted small-to-large Neumaier sum   = "
                << s_sorted_neumaier << "\n";

        std::cout << "\n------------------ absolute values ------------------\n";

        std::cout << "|sequential|                        = "
                << std::abs(s_seq) << "\n";

        std::cout << "|reverse sequential|                = "
                << std::abs(s_reverse) << "\n";

        std::cout << "|sorted small-to-large plain|        = "
                << std::abs(s_sorted_plain) << "\n";

        std::cout << "|sorted large-to-small plain|        = "
                << std::abs(s_sorted_large_plain) << "\n";

        std::cout << "|unsorted Neumaier|                 = "
                << std::abs(s_unsorted_neumaier) << "\n";

        std::cout << "|sorted small-to-large Neumaier|     = "
                << std::abs(s_sorted_neumaier) << "\n";

        std::cout << "\n------------------ differences relative to sorted Neumaier ------------------\n";

        std::cout << "|seq - sorted neum|                 = "
                << std::abs(s_seq - s_sorted_neumaier) << "\n";

        std::cout << "|reverse - sorted neum|             = "
                << std::abs(s_reverse - s_sorted_neumaier) << "\n";

        std::cout << "|sorted plain - sorted neum|         = "
                << std::abs(s_sorted_plain - s_sorted_neumaier) << "\n";

        std::cout << "|large-first plain - sorted neum|    = "
                << std::abs(s_sorted_large_plain - s_sorted_neumaier) << "\n";

        std::cout << "|unsorted neum - sorted neum|        = "
                << std::abs(s_unsorted_neumaier - s_sorted_neumaier) << "\n";

        std::cout << "\n------------------ pairwise order sensitivity ------------------\n";

        std::cout << "|seq - reverse|                     = "
                << std::abs(s_seq - s_reverse) << "\n";

        std::cout << "|seq - sorted plain|                = "
                << std::abs(s_seq - s_sorted_plain) << "\n";

        std::cout << "|seq - large-first plain|           = "
                << std::abs(s_seq - s_sorted_large_plain) << "\n";

        std::cout << "|sorted plain - large-first plain|   = "
                << std::abs(s_sorted_plain - s_sorted_large_plain) << "\n";

        std::cout << "\n------------------ cancellation diagnostics ------------------\n";

        auto print_cancellation_ratio = [&](const std::string& name, comp s) {
            double abs_s = std::abs(s);

            if (abs_s > 0.0) {
                std::cout << "sum|terms| / |" << name << "|              = "
                        << static_cast<double>(sum_abs_terms) / abs_s << "\n";
            } else {
                std::cout << "sum|terms| / |" << name << "|              = inf because |"
                        << name << "| = 0\n";
            }
        };

        print_cancellation_ratio("seq", s_seq);
        print_cancellation_ratio("sorted_neum", s_sorted_neumaier);

        std::cout << "\n------------------ first few terms ------------------\n";

        size_t nprint = std::min<size_t>(input_vec.size(), 10);

        for (size_t i = 0; i < nprint; ++i) {
            std::cout << "input_vec[" << i << "] = "
                    << input_vec[i]
                    << " |term| = "
                    << std::abs(input_vec[i])
                    << "\n";
        }

        std::cout << "\n------------------ first few sorted terms ------------------\n";

        std::vector<comp> sorted_vec = input_vec;

        std::sort(
            sorted_vec.begin(),
            sorted_vec.end(),
            [](const comp& a, const comp& b) {
                return std::norm(a) < std::norm(b);
            }
        );

        for (size_t i = 0; i < nprint; ++i) {
            std::cout << "sorted_vec[" << i << "] = "
                    << sorted_vec[i]
                    << " |term| = "
                    << std::abs(sorted_vec[i])
                    << "\n";
        }

        std::cout << "\n------------------ last few sorted terms ------------------\n";

        size_t start_last = sorted_vec.size() > nprint ? sorted_vec.size() - nprint : 0;

        for (size_t i = start_last; i < sorted_vec.size(); ++i) {
            std::cout << "sorted_vec[" << i << "] = "
                    << sorted_vec[i]
                    << " |term| = "
                    << std::abs(sorted_vec[i])
                    << "\n";
        }

        std::cout << "=====================================================\n\n";
    }

    return s_sorted_neumaier;
}

struct NeumaierComplexSum {
    long double sum_re = 0.0L;
    long double sum_im = 0.0L;

    long double c_re = 0.0L;
    long double c_im = 0.0L;

    void add(comp z) {
        long double x_re = static_cast<long double>(z.real());
        long double x_im = static_cast<long double>(z.imag());

        long double t_re = sum_re + x_re;
        if (std::abs(sum_re) >= std::abs(x_re)) {
            c_re += (sum_re - t_re) + x_re;
        } else {
            c_re += (x_re - t_re) + sum_re;
        }
        sum_re = t_re;

        long double t_im = sum_im + x_im;
        if (std::abs(sum_im) >= std::abs(x_im)) {
            c_im += (sum_im - t_im) + x_im;
        } else {
            c_im += (x_im - t_im) + sum_im;
        }
        sum_im = t_im;
    }

    comp result() const {
        return comp(
            static_cast<double>(sum_re + c_re),
            static_cast<double>(sum_im + c_im)
        );
    }
};

comp precise_vector_sum(std::vector<comp> terms) {
    std::sort(
        terms.begin(),
        terms.end(),
        [](const comp& a, const comp& b) {
            return std::norm(a) < std::norm(b);
        }
    );

    NeumaierComplexSum s;

    for (const auto& z : terms) {
        s.add(z);
    }

    return s.result();
}


struct NeumaierDouble {
    double sum = 0.0;
    double c   = 0.0;

    void add(double x)
    {
        double t = sum + x;

        if (std::abs(sum) >= std::abs(x)) {
            c += (sum - t) + x;
        } else {
            c += (x - t) + sum;
        }

        sum = t;
    }

    double value() const
    {
        return sum + c;
    }
};

struct NeumaierComplexDouble {
    NeumaierDouble real_sum;
    NeumaierDouble imag_sum;

    void add(const std::complex<double>& z)
    {
        real_sum.add(z.real());
        imag_sum.add(z.imag());
    }

    std::complex<double> value() const
    {
        return {
            real_sum.value(),
            imag_sum.value()
        };
    }
};



/* First we code all the function needed for F2 functions, we start with a single F in S-wave
as needed to check. We follow the paper = https://arxiv.org/pdf/2111.12734.pdf */
/* This is the v2 of this code base, here we add the spherical harmonics to F2 functions 
such that in future it can take in any partial wave (ell<3) can build the F2 function or 
matrix as needed for the three-body QC */

typedef std::complex<double> comp;

comp I0F(   comp En, 
            comp sigma_p,
            comp p,
            comp total_P, 
            double alpha,
            double mi,
            double mj,
            double mk, 
            double L )
{
    comp pi = std::acos(-1.0);
    comp Lby2pi = ((comp) L)/((comp) 2.0*pi); 
    comp gamma = (En - omega_func(p,mi))/std::sqrt(sigma_p);
    comp x = std::sqrt(q2psq_star(sigma_p,mj,mk))*Lby2pi;

    comp A = ((comp)4.0)*pi*gamma;
    comp B = -std::sqrt(pi/((comp)alpha))*((comp)1.0/(comp)2.0)*std::exp((comp)alpha*x*x);
    //comp C = 0.5*(pi*x)*ERFI_func(std::sqrt(alpha*x*x));

    double relerr = 0.0;
    comp term1 = ((comp)alpha)*x*x; 
    
    std::complex<double> term1comp(term1.real(), term1.imag()); 
    comp fadeeva_erfi = Faddeeva::erfi(std::sqrt(term1comp),relerr);

    comp C = ((comp)0.5)*(pi*x)*fadeeva_erfi;

    char debug = 'n';
    if(debug=='y')
    {
        std::cout<<std::setprecision(30);
        std::cout<<"constant = "<<A<<std::endl;
        std::cout<<"Fadeeva Erfi = "<<fadeeva_erfi<<std::endl; 
        std::cout<<"factor1 = "<<B<<std::endl; 
        std::cout<<"factor2 = "<<C<<std::endl; 
    }
    return A*(B + C);

}

comp I1F(   comp En, 
            comp sigma_p,
            comp p,
            comp total_P, 
            double alpha,
            double mi,
            double mj,
            double mk, 
            double L )
{
    comp pi = std::acos(-1.0);
    comp Lby2pi = ((comp) L)/((comp) 2.0*pi); 
    comp gamma = (En - omega_func(p,mi))/std::sqrt(sigma_p);
    comp x = std::sqrt(q2psq_star(sigma_p,mj,mk))*Lby2pi;

    comp A = ((comp)4.0)*pi*gamma;
    comp term = pi/(((comp) alpha)*((comp) alpha)*((comp) alpha)); 
    comp B = -std::sqrt(term) * (((comp)1.0) + ((comp)2.0)*((comp)alpha)*x*x)/((comp)4.0) * std::exp(((comp)alpha)*x*x); 
    
    //comp C = 0.5*(pi*x)*ERFI_func(std::sqrt(alpha*x*x));

    comp term1 = ((comp)alpha)*x*x; 
    double relerr = 0.0;

    
    std::complex<double> term1comp(term1.real(), term1.imag()); 
    comp fadeeva_erfi = Faddeeva::erfi(std::sqrt(term1comp),relerr);

    comp C = ((comp)0.5)*(pi*x*x*x)*fadeeva_erfi;

    char debug = 'n';
    if(debug=='y')
    {
        std::cout<<std::setprecision(30);
        std::cout<<"constant = "<<A<<std::endl;
        std::cout<<"factor1 = "<<B<<std::endl; 
        std::cout<<"factor2 = "<<C<<std::endl; 
    }
    return A*(B + C);

}

comp I2F(   comp En, 
            comp sigma_p,
            comp p,
            comp total_P, 
            double alpha,
            double mi,
            double mj,
            double mk, 
            double L )
{
    comp pi = std::acos(-1.0);
    comp Lby2pi = ((comp) L)/((comp) 2.0*pi); 
    comp gamma = (En - omega_func(p,mi))/std::sqrt(sigma_p);
    comp x = std::sqrt(q2psq_star(sigma_p,mj,mk))*Lby2pi;

    comp A = ((comp)4.0)*pi*gamma;
    comp B = -std::sqrt(pi/(((comp)alpha)*((comp)alpha)*((comp)alpha)*((comp)alpha)*((comp)alpha))) * (((comp)3.0) + ((comp)2.0)*((comp)alpha)*x*x + ((comp)4.0)*((comp)alpha)*((comp)alpha)*x*x*x*x)/((comp)8.0) * std::exp(((comp)alpha)*x*x); 
    
    //comp C = 0.5*(pi*x)*ERFI_func(std::sqrt(alpha*x*x));
    comp term1 = ((comp)alpha)*x*x;
    double relerr = 0.0;

    std::complex<double> term1comp(term1.real(), term1.imag()); 
    comp fadeeva_erfi = Faddeeva::erfi(std::sqrt(term1comp),relerr);

    comp C = ((comp)0.5)*(pi*x*x*x*x*x)*fadeeva_erfi;

    char debug = 'n';
    if(debug=='y')
    {
        std::cout<<std::setprecision(25);
        std::cout<<"constant = "<<A<<std::endl;
        std::cout<<"factor1 = "<<B<<std::endl; 
        std::cout<<"factor2 = "<<C<<std::endl; 
    }
    return A*(B + C);

}

comp I_int_ang_mom( 
                    comp En, 
                    comp sigma_p,
                    comp p,
                    comp total_P, 
                    int ell_f,
                    int proj_mf, 
                    int ell_i, 
                    int proj_mi, 
                    double alpha,
                    double mi,
                    double mj,
                    double mk, 
                    double L,
                    bool Q0norm //this is the Q0 normalization that removes q*^{ell} barrier factors 
                                //if true then it will use Q0 normalization, the terms will have explicit q*^{ell} dependence 
                )
{
    comp pi = std::acos(-1.0); 
    comp Lby2pi = ((comp) L)/((comp) 2.0*pi);
    comp twopibyL = ((comp) 2.0*pi)/((comp) L);
    comp x = std::sqrt(q2psq_star(sigma_p,mj,mk))*Lby2pi;
    
    comp zero_val = {0.0, 0.0};
    
    if(ell_f==ell_i)
    {
        if(proj_mf==proj_mi)
        {
            if(ell_i==0)
            {
                comp I0 = I0F(En, sigma_p, p, total_P, alpha, mi, mj, mk, L); 
                return I0; 
            }
            else if(ell_i==1)
            {
                comp I1 = I1F(En, sigma_p, p, total_P, alpha, mi, mj, mk, L); 
                if(Q0norm)
                {
                    I1 = I1*std::pow(twopibyL, 2.0*ell_i);
                }
                else
                {
                    I1 = I1/(std::pow(x,2.0*ell_i));
                }
                return I1;
            }
            else if(ell_i==2)
            {
                comp I2 = I2F(En, sigma_p, p, total_P, alpha, mi, mj, mk, L); 
                if(Q0norm)
                {
                    I2 = I2*(std::pow(twopibyL, 2.0*ell_i));
                }
                else
                {
                    I2 = I2/(std::pow(x,2.0*ell_i));
                }
                return I2;
            }
            else 
            {
                std::cerr << "Invalid ell=" << ell_i << " not added to the package yet, only upto ell=2 available!!!\n";
                return zero_val;
            }
        }
        else 
        {
            return zero_val; 
        }
    }
    else 
    {
        return zero_val;
    }
}

void print_Ylm1_Ylm2_for_ijk(
    comp En,
    comp sigma_p,
    std::vector<comp> p,
    std::vector<comp> total_P,
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
    double L
)
{
    comp pi = std::acos(-1.0);
    comp Lby2pi = ((comp)L) / ((comp)2.0 * pi);

    comp px = p[0], py = p[1], pz = p[2];
    comp Px = total_P[0], Py = total_P[1], Pz = total_P[2];

    comp spec_p = std::sqrt(px*px + py*py + pz*pz);

    comp gamma = (En - omega_func(spec_p, mi)) / std::sqrt(sigma_p);

    comp xii = ((comp)0.5) *
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

    if (std::abs(npPsq) > 1.0e-14)
    {
        comp na_dot_npP = nax*npPx + nay*npPy + naz*npPz;

        comp prod1 =
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

    comp Ylm1 = spherical_harmonics(rvec, ell_f, proj_mf);
    comp Ylm2 = spherical_harmonics(rvec, ell_i, proj_mi);

    std::cout << std::setprecision(30);
    std::cout << "----------------------------------------\n";
    std::cout << "input ijk = [" << i << ", " << j << ", " << k << "]\n";
    std::cout << "na_vec = [" << nax << ", " << nay << ", " << naz << "]\n";
    std::cout << "rvec   = [" << rx  << ", " << ry  << ", " << rz  << "]\n";
    std::cout << "ell_f, m_f = " << ell_f << ", " << proj_mf << "\n";
    std::cout << "ell_i, m_i = " << ell_i << ", " << proj_mi << "\n";
    std::cout << "Ylm1 = " << Ylm1 << "\n";
    std::cout << "Ylm2 = " << Ylm2 << "\n";
    std::cout << "Ylm1 * Ylm2 = " << Ylm1 * Ylm2 << "\n";
    std::cout << "----------------------------------------\n";
}



comp I_sum_ang_mom(   
                    comp En, 
                    comp sigma_p,
                    std::vector<comp> p,
                    std::vector<comp> total_P, 
                    int ell_f, 
                    int proj_mf, 
                    int ell_i, 
                    int proj_mi, 
                    double alpha,
                    double mi,
                    double mj,
                    double mk, 
                    double L,
                    int max_shell_num,
                    bool Q0norm
                )
{
    char debug = 'n';
    double tolerance = 1.0e-11;
    comp pi = std::acos(-1.0); 
    comp Lby2pi = ((comp) L)/((comp) 2.0*pi);
    comp twopibyL = ((comp) 2.0*pi)/((comp) L);

    comp px = p[0];
    comp py = p[1];
    comp pz = p[2];

    comp spec_p = std::sqrt(px*px + py*py + pz*pz);

    comp Px = total_P[0];
    comp Py = total_P[1];
    comp Pz = total_P[2];

    comp total_P_val = std::sqrt(Px*Px + Py*Py + Pz*Pz);

    comp gamma = (En - omega_func(spec_p,mi))/std::sqrt(sigma_p);
    //std::cout<<"gamma = "<<gamma<<std::endl;
    comp x = std::sqrt(q2psq_star(sigma_p,mj,mk))*Lby2pi;
    comp xii = ((comp)0.5)*(((comp)1.0) + ((comp)(mj*mj - mk*mk))/sigma_p); //this is another xi which is used inside the F sum function

    if(debug=='y')
    {
        std::cout<<"x = "<<x<<'\t'<<"sig_p = "<<sigma_p<<std::endl;

        //std::cout<<"x = "<<x<<'\t'<<"sig_p = "<<sigma_p<<std::endl;
    }

    comp npPx = (Px - px)*Lby2pi; 
    comp npPy = (Py - py)*Lby2pi;
    comp npPz = (Pz - pz)*Lby2pi;

    comp npP = std::sqrt(npPx*npPx + npPy*npPy + npPz*npPz);

    int c1 = 0;
    int c2 = 0; //these two are for checking if p and P are zero or not

    if(abs(spec_p)<1.0e-10 || abs(spec_p)==0.0 ) c1 = 1;
    if(abs(total_P_val)<1.0e-10 || abs(total_P_val)==0.0) c2 = 1;

    comp xibygamma = xii/gamma; 
    
    //int max_shell_num = 50;
    int na_x_initial = -max_shell_num;
    int na_x_final = +max_shell_num;
    int na_y_initial = -max_shell_num;
    int na_y_final = +max_shell_num;
    int na_z_initial = -max_shell_num;
    int na_z_final = +max_shell_num;

    comp summ = {0.0,0.0};
    comp temp_summ = {0.0,0.0};
    std::vector<comp> bare_sum_vec; 

    for(int i=na_x_initial;i<na_x_final+1;++i)
    {
        for(int j=na_y_initial;j<na_y_final+1;++j)
        {
            for(int k=na_z_initial;k<na_z_final+1;++k)
            {
                comp na = (comp) std::sqrt(i*i + j*j + k*k);

                comp nax = (comp) i;
                comp nay = (comp) j;
                comp naz = (comp) k;

                comp nax_npPx = nax*npPx;
                comp nay_npPy = nay*npPy;
                comp naz_npPz = naz*npPz; 

                comp na_dot_npP = nax_npPx + nay_npPy + naz_npPz;
                comp npPsq = npP*npP; 

                //we have change the sign before xi/gamma to minus according to the python code 
                comp prod1 = ( (na_dot_npP/npPsq)*(((comp)1.0)/gamma - ((comp)1.0)) - xibygamma );
                
                comp rx = 0.0;
                comp ry = 0.0;
                comp rz = 0.0;
                
                if(c1==1 && c2==1)
                {
                    rx = nax;
                    ry = nay;
                    rz = naz; 
                }
                else 
                {
                    if(abs(npPsq)==0)
                    {
                        rx = nax;
                        ry = nay;
                        rz = naz;
                    }
                    else 
                    {
                        rx = nax + npPx*prod1;
                        ry = nay + npPy*prod1;
                        rz = naz + npPz*prod1;
                    }
                    
                }

                comp r = std::sqrt(rx*rx + ry*ry + rz*rz);
                std::vector<comp> rvec(3); 
                rvec[0] = rx; 
                rvec[1] = ry; 
                rvec[2] = rz; 

                
                comp Ylm1 = spherical_harmonics(rvec, ell_f, proj_mf); 
                comp Ylm2 = spherical_harmonics(rvec, ell_i, proj_mi);
                comp term =  Ylm1 * std::exp(((comp)alpha)*(x*x - r*r))/(x*x - r*r) * Ylm2;
                comp prev_term = summ; 
                summ = summ + Ylm1 * std::exp(((comp)alpha)*(x*x - r*r))/(x*x - r*r) * Ylm2;

                bare_sum_vec.push_back(term); 

                if(debug=='y')
                {
                    if((int) nax.real()==0 && (int) nay.real()==2 && (int) naz.real()==2)
                    {
                    if(ell_f==1 && proj_mf==-1 && ell_i==1 && proj_mi==-1)
                    {
                        std::cout << "-------------------------" << std::endl;
                        std::cout << std::setprecision(30);
                        std::cout << "na_vec = [" << nax << "," 
                                                  << nay << "," 
                                                  << naz << "]" 
                                                  << std::endl; 
                        std::cout << "rvec = [" << rx << "," 
                                                << ry << "," 
                                                << rz << "]" 
                                                << std::endl; 
                        std::cout << "Ylm1 : " << Ylm1 << '\t'
                                  << "Ylm2 : " << Ylm2 << std::endl; 
                        std::cout << "pow term : " << (std::pow(twopibyL, (comp) ell_f +  (comp) ell_i)) << std::endl; 
                        std::cout << "UV+prop : " << std::exp((comp)alpha*(x*x - r*r))/(x*x - r*r) << std::endl;
                        std::cout << "summand term : " << term << std::endl; 
                        std::cout << "prev out : " << prev_term * (std::pow(twopibyL, ell_f + ell_i)) << std::endl; 
                        std::cout << "out : " << term*(std::pow(twopibyL, ell_f + ell_i)) << std::endl; 
                        comp term_some_A = term * (std::pow(twopibyL, ell_f + ell_i));
                        comp term_some_B = prev_term * (std::pow(twopibyL, ell_f + ell_i));
                        
                        std::cout << "prev sum term : " << term_some_B << std::endl;
                        std::cout << "curr sum term : " << term_some_A << std::endl; 
                        std::cout << "sum check : " <<  term_some_A.real() + term_some_B.real() << std::endl; 
                        std::cout << "sum term : " << summ*(std::pow(twopibyL, ell_f + ell_i)) <<std::endl;
                    }
                    
                     
                    //std::cout<<i<<'\t'<<j<<'\t'<<k<<'\t'<<x*x - r*r<<'\t'<<prod1<<'\t'<<summ<<std::endl;
                    if(!std::isnan(abs(prod1)))
                    {
                        //std::cout<<"npP = "<<npP<<'\t'
                        //     <<"prod1 = "<<prod1<<'\t'
                        //     <<"r = "<<r<<'\t'<<"rx = "<<rx<<'\t'<<"ry = "<<ry<<'\t'<<"rz = "<<rz<<std::endl;
                    }

                    std::cout << "-------------------------" << std::endl;
                    }
                }
                
            }
        }
    }

    /*
    comp final_sum_neum_from_bare =
    precise_vector_sum_debug(
        bare_sum_vec,
        "bare_sum_vec debug",
        debug
    );
    */

    comp final_bare_sum = sorted_neumaier_sum(bare_sum_vec); 

    comp final_result = {0.0,0.0}; 

    if(Q0norm)
    {
        //summ = summ*(std::pow(twopibyL, ell_f + ell_i));
        final_result = final_bare_sum*(std::pow(twopibyL, ell_f + ell_i));
    }
    else
    {
        //summ = summ/(std::pow(x, ell_f + ell_i));
        final_result = final_bare_sum/(std::pow(x, ell_f + ell_i));
    }

    if(debug=='y')
    {
        std::cout << "out = " << summ << std::endl; 
    }

    //return summ ;
    return final_result ;
}

//We write the F2(p, ell_f, m_f; k, ell_i, m_i) function here

comp F2_ang_mom(    
                    comp En, 
                    std::vector<comp> k, //we assume that k,p,P are a 3-vector
                    std::vector<comp> p,
                    std::vector<comp> total_P,
                    int ell_f, 
                    int proj_mf,
                    int ell_i, 
                    int proj_mi, 
                    double L,
                    double mi,
                    double mj, 
                    double mk, 
                    double alpha,
                    double epsilon_h,
                    int max_shell_num,
                    bool Q0norm    
                )
{
    char debug = 'n';
    comp kx = k[0];
    comp ky = k[1];
    comp kz = k[2];

    comp px = p[0];
    comp py = p[1];
    comp pz = p[2];

    comp Px = total_P[0];
    comp Py = total_P[1];
    comp Pz = total_P[2];

    comp spec_k = std::sqrt(kx*kx + ky*ky + kz*kz);
    comp spec_p = std::sqrt(px*px + py*py + pz*pz);
    comp total_P_val = std::sqrt(Px*Px + Py*Py + Pz*Pz);

    comp sigp = sigma_pvec_based(En,p,mi,total_P);//sigma(En, spec_p, mi, total_P_val);

    comp cutoff = cutoff_function_1(sigp, mj, mk, epsilon_h);

    comp omega_p = omega_func(spec_p,mi);

    if(debug=='y')
    {
        std::cout << "================================" << std::endl; 
        
        std::cout << " F2_ang_mom function : " << std::endl; 
        std::cout << "--------------------------------" << std::endl; 
        std::cout << "p1x = " << px << '\t'
                  << "p1y = " << py << '\t'
                  << "p1z = " << pz << std::endl; 
        std::cout << "ell_f = " << ell_f << '\t'
                  << "proj_mf = " << proj_mf << std::endl; 
        std::cout << "k1x = " << kx << '\t'
                  << "k1y = " << ky << '\t'
                  << "k1z = " << kz << std::endl;
        std::cout << "ell_i = " << ell_i << '\t'
                  << "proj_mi = " << proj_mi << std::endl; 
        std::cout << "spec_k = " << spec_k << '\t'
                  << "spec_p = " << spec_p << std::endl; 
        std::cout << "total_P = " << total_P_val << std::endl; 
        std::cout << "sig_p = " << sigp << '\t'
                  << "cutoff = " << cutoff << '\t' 
                  << "omega_p = " << omega_p << std::endl; 
    }

    //condition for the delta function
    int condition_delta = 0;

    if(px==kx && py==ky && pz==kz)
    {
        condition_delta = 0;
    }
    else 
    {
        condition_delta = 1;
    }

    if(debug=='y')
    {
        std::cout << "condition delta = " << condition_delta << std::endl; 
    }
    
    if(condition_delta==1) return 0.0;
    else 
    {
        comp pi = std::acos(-1.0);
        comp A = cutoff/(((comp)16.0)*pi*pi*((comp)L)*((comp)L)*((comp)L)*((comp)L)*omega_p*(En - omega_p));

        comp B = I_sum_ang_mom(En, sigp, p, total_P, ell_f, proj_mf, ell_i, proj_mi, alpha, mi, mj, mk, L, max_shell_num, Q0norm);

        comp C = I_int_ang_mom(En, sigp, spec_p, total_P_val, ell_f, proj_mf, ell_i, proj_mi, alpha, mi, mj, mk, L, Q0norm);
        //std::cout<<A<<'\t'<<B<<'\t'<<C<<std::endl; 

        if(debug=='y')
        {
            std::cout << "cutoff = " << cutoff << std::endl; 
            std::cout << "cutoff times constant = " << A << std::endl;  
            std::cout << "sum = " << B << std::endl; 
            std::cout << "analytical res = " << C << std::endl; 
            std::cout << std::endl; 
            std::cout << "================================" << std::endl; 
        
        }

        if(abs(A)==0)
        {
            return 0.0;
        }
        else
        return A*(B - C);
    }

}


//This function builds the F2_i matrix in the channel, momentum and angular momentum l,m space
void F2_i_ang_mom_mat(  
                        Eigen::MatrixXcd &F2,
                        comp En, 
                        std::vector<std::vector<comp> > &plm_config,
                        std::vector<std::vector<comp> > &klm_config,
                        std::vector<comp> total_P,
                        double mi,
                        double mj, 
                        double mk, 
                        double L, 
                        double alpha, 
                        double epsilon_h,
                        int max_shell_num,
                        bool Q0norm
                      )
{
    char debug = 'n';
    char special_check = 'n'; 

    if(debug=='y')
    {
        std::cout << "We will print out the components of F2 matrix" << std::endl;
        std::cout << "(F2_i_mat function from F2_functions.h)" << std::endl; 
    }
    for(int i=0; i<plm_config[0].size(); ++i)
    {
        if(debug=='y')
        {
            std::cout << "-------------------------------------" << std::endl; 
        }
        for(int j=0; j<klm_config[0].size(); ++j)
        {
            comp px = plm_config[0][i];
            comp py = plm_config[1][i];
            comp pz = plm_config[2][i];
            int ell_f = static_cast<int> (plm_config[3][i].real()); 
            int proj_mf = static_cast<int> (plm_config[4][i].real()); 


            comp spec_p = std::sqrt(px*px + py*py + pz*pz);
            std::vector<comp> p(3);
            p[0] = px;
            p[1] = py;
            p[2] = pz; 

            comp kx = klm_config[0][j];
            comp ky = klm_config[1][j];
            comp kz = klm_config[2][j];
            int ell_i = static_cast<int> (klm_config[3][j].real()); 
            int proj_mi = static_cast<int> (klm_config[4][j].real()); 

            if(debug=='y' && special_check=='y')
            {
                std::cout << "special check" << std::endl; 
                std::cout << "ell_i = " << ell_i << '\t' 
                          << "klmconfig = " << klm_config[3][j] << std::endl;
            }

            comp spec_k = std::sqrt(kx*kx + ky*ky + kz*kz);
            std::vector<comp> k(3);
            k[0] = kx;
            k[1] = ky;
            k[2] = kz; 

            comp Px = total_P[0];
            comp Py = total_P[1];
            comp Pz = total_P[2];

            comp total_P_val = std::sqrt(Px*Px + Py*Py + Pz*Pz);
            
            comp F2_val = F2_ang_mom(En, k, p, total_P, ell_f, proj_mf, ell_i, proj_mi, L, mi, mj, mk, alpha, epsilon_h, max_shell_num, Q0norm);

            F2(i,j) = F2_val;

            if(debug=='y')
            {
                std::cout << "i = " << i << '\t' << "j = " << j << std::endl; 
                std::cout << "px = " << px << '\t'
                          << "py = " << py << '\t'
                          << "pz = " << pz << std::endl; 
                std::cout << "ell_f = " << ell_f << '\t'
                          << "proj_mf = " << proj_mf << std::endl; 
                std::cout << "kx = " << kx << '\t'
                          << "ky = " << ky << '\t'
                          << "kz = " << kz << std::endl;
                std::cout << "ell_i = " << ell_i << '\t'
                          << "proj_mi = " << proj_mi << std::endl;
                std::cout << "F2 val = " << F2_val << std::endl;
                std::cout << "-------------------------------------" << std::endl;
            }

            


        }
        if(debug=='y')
        {
            std::cout << "-------------------------------------" << std::endl; 
        }
    }

    if(debug=='y')
    {
        std::cout << "=========================================" << std::endl; 
    }
}

//This builds the F2 function as a matrix in channel space 
//We pass the waves for particular channel i, and it build the
//F1tilde and F2tilde matrices to form F2 matrix for the 
//(2+1) system, 
/*
    F2 = | F2_1 0 .. |
         | 0    F2_2 | 

*/
// The system has m1 and m2 particles such that threshold = 2m1 + m2
void F2_2plus1_mat(
                        Eigen::MatrixXcd &F2,
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
    if(debug=='y')
    {
        std::cout << "  F2 angular momentum matrix in channel space " << std::endl;
        std::cout << "(F2_ang_mom_mat function from F2_functions.h)" << std::endl; 
    }

    int size1 = plm_config[0].size(); 
    int size2 = klm_config[0].size(); 

    int total_size = size1 + size2; 

    Eigen::MatrixXcd F2_1(size1, size1); 
    Eigen::MatrixXcd F2_2(size2, size2); 

    //when i=1 
    mi = m1; 
    mj = m1; 
    mk = m2; 

    F2_i_ang_mom_mat(F2_1, En, plm_config, plm_config, total_P, mi, mj, mk, L, alpha, epsilon_h, max_shell_num, Q0norm); 

    //when i=2
    mi = m2; 
    mj = m1; 
    mk = m1; 

    F2_i_ang_mom_mat(F2_2, En, klm_config, klm_config, total_P, mi, mj, mk, L, alpha, epsilon_h, max_shell_num, Q0norm); 

    Eigen::MatrixXcd Filler0_12(size1,size2);
    Eigen::MatrixXcd Filler0_21(size2,size1);
    Eigen::MatrixXcd Filler0_22(size2,size2);

    Filler0_12 = Eigen::MatrixXcd::Zero(size1,size2);
    Filler0_21 = Eigen::MatrixXcd::Zero(size2,size1);
    Filler0_22 = Eigen::MatrixXcd::Zero(size2,size2);

    Eigen::MatrixXcd F2_mat(size1 + size2,size1 + size2);
    F2_mat <<   F2_1,   Filler0_12,
                Filler0_21, F2_2; 

    F2 = F2_mat; 

}

// ============================================================================
// Debug helpers for F2_2plus1 lm components
// Paste these after your existing F2 functions.
// ============================================================================

void print_comp_vec3_debug(
    const std::string &name,
    const std::vector<comp> &v
)
{
    std::cout << name << " = ["
              << v[0] << ", "
              << v[1] << ", "
              << v[2] << "]\n";
}


// ============================================================================
// Debug version of I_int_ang_mom
// Prints all steps entering the integral subtraction part.
// ============================================================================

comp debug_I_int_ang_mom_steps(
    comp En,
    comp sigma_p,
    comp p_mag,
    comp total_P_mag,
    int ell_f,
    int proj_mf,
    int ell_i,
    int proj_mi,
    double alpha,
    double mi,
    double mj,
    double mk,
    double L,
    bool Q0norm
)
{
    std::cout << std::setprecision(30);

    comp pi = std::acos(-1.0);
    comp Lby2pi = ((comp)L) / (((comp)2.0) * pi);
    comp twopibyL = (((comp)2.0) * pi) / ((comp)L);

    comp gamma = (En - omega_func(p_mag, mi)) / std::sqrt(sigma_p);
    comp q2star = q2psq_star(sigma_p, mj, mk);
    comp x = std::sqrt(q2star) * Lby2pi;

    std::cout << "\n";
    std::cout << "================ I_int_ang_mom DEBUG ================\n";
    std::cout << "En            = " << En << "\n";
    std::cout << "sigma_p       = " << sigma_p << "\n";
    std::cout << "p_mag         = " << p_mag << "\n";
    std::cout << "total_P_mag   = " << total_P_mag << "\n";
    std::cout << "mi,mj,mk      = " << mi << ", " << mj << ", " << mk << "\n";
    std::cout << "L             = " << L << "\n";
    std::cout << "alpha         = " << alpha << "\n";
    std::cout << "Q0norm        = " << Q0norm << "\n";
    std::cout << "ell_f,mf      = " << ell_f << ", " << proj_mf << "\n";
    std::cout << "ell_i,mi      = " << ell_i << ", " << proj_mi << "\n";
    std::cout << "gamma         = " << gamma << "\n";
    std::cout << "q2star        = " << q2star << "\n";
    std::cout << "x             = sqrt(q2star) * L/(2pi) = " << x << "\n";
    std::cout << "2pi/L         = " << twopibyL << "\n";

    comp zero_val = {0.0, 0.0};

    if(ell_f != ell_i)
    {
        std::cout << "Integral part is ZERO because ell_f != ell_i.\n";
        std::cout << "I_int         = " << zero_val << "\n";
        std::cout << "=====================================================\n";
        return zero_val;
    }

    if(proj_mf != proj_mi)
    {
        std::cout << "Integral part is ZERO because proj_mf != proj_mi.\n";
        std::cout << "I_int         = " << zero_val << "\n";
        std::cout << "=====================================================\n";
        return zero_val;
    }

    comp Iraw = zero_val;
    comp Inorm = zero_val;

    if(ell_i == 0)
    {
        Iraw = I0F(En, sigma_p, p_mag, total_P_mag,
                   alpha, mi, mj, mk, L);

        Inorm = Iraw;

        std::cout << "ell = 0 integral selected.\n";
        std::cout << "I0F raw       = " << Iraw << "\n";
        std::cout << "normalization = none for ell=0\n";
    }
    else if(ell_i == 1)
    {
        Iraw = I1F(En, sigma_p, p_mag, total_P_mag,
                   alpha, mi, mj, mk, L);

        std::cout << "ell = 1 integral selected.\n";
        std::cout << "I1F raw       = " << Iraw << "\n";

        if(Q0norm)
        {
            comp factor = std::pow(twopibyL, 2.0 * ell_i);
            Inorm = Iraw * factor;

            std::cout << "Q0norm true.\n";
            std::cout << "factor        = (2pi/L)^(2ell) = " << factor << "\n";
        }
        else
        {
            comp factor = std::pow(x, 2.0 * ell_i);
            Inorm = Iraw / factor;

            std::cout << "Q0norm false.\n";
            std::cout << "factor        = x^(2ell) = " << factor << "\n";
        }
    }
    else if(ell_i == 2)
    {
        Iraw = I2F(En, sigma_p, p_mag, total_P_mag,
                   alpha, mi, mj, mk, L);

        std::cout << "ell = 2 integral selected.\n";
        std::cout << "I2F raw       = " << Iraw << "\n";

        if(Q0norm)
        {
            comp factor = std::pow(twopibyL, 2.0 * ell_i);
            Inorm = Iraw * factor;

            std::cout << "Q0norm true.\n";
            std::cout << "factor        = (2pi/L)^(2ell) = " << factor << "\n";
        }
        else
        {
            comp factor = std::pow(x, 2.0 * ell_i);
            Inorm = Iraw / factor;

            std::cout << "Q0norm false.\n";
            std::cout << "factor        = x^(2ell) = " << factor << "\n";
        }
    }
    else
    {
        std::cerr << "Invalid ell = " << ell_i
                  << ". Only ell=0,1,2 are implemented.\n";
        std::cout << "I_int         = " << zero_val << "\n";
        std::cout << "=====================================================\n";
        return zero_val;
    }

    std::cout << "I_int final   = " << Inorm << "\n";
    std::cout << "real          = " << Inorm.real() << "\n";
    std::cout << "imag          = " << Inorm.imag() << "\n";
    std::cout << "abs           = " << std::abs(Inorm) << "\n";
    std::cout << "=====================================================\n";

    return Inorm;
}


// ============================================================================
// Debug version of I_sum_ang_mom
// Prints rvec, Ylm1, Ylm2, UV/propgator, summand, and running sum.
// ============================================================================

comp debug_I_sum_ang_mom_steps(
    comp En,
    comp sigma_p,
    std::vector<comp> p,
    std::vector<comp> total_P,
    int ell_f,
    int proj_mf,
    int ell_i,
    int proj_mi,
    double alpha,
    double mi,
    double mj,
    double mk,
    double L,
    int max_shell_num,
    bool Q0norm,
    int max_terms_to_print = 50,
    bool print_each_term = true
)
{
    std::cout << std::setprecision(30);

    comp pi = std::acos(-1.0);
    comp Lby2pi = ((comp)L) / (((comp)2.0) * pi);
    comp twopibyL = (((comp)2.0) * pi) / ((comp)L);

    comp px = p[0];
    comp py = p[1];
    comp pz = p[2];

    comp Px = total_P[0];
    comp Py = total_P[1];
    comp Pz = total_P[2];

    comp spec_p = std::sqrt(px*px + py*py + pz*pz);
    comp total_P_val = std::sqrt(Px*Px + Py*Py + Pz*Pz);

    comp gamma = (En - omega_func(spec_p, mi)) / std::sqrt(sigma_p);
    comp q2star = q2psq_star(sigma_p, mj, mk);
    comp x = std::sqrt(q2star) * Lby2pi;

    comp xii = ((comp)0.5) *
        (((comp)1.0) + ((comp)(mj*mj - mk*mk)) / sigma_p);

    comp npPx = (Px - px) * Lby2pi;
    comp npPy = (Py - py) * Lby2pi;
    comp npPz = (Pz - pz) * Lby2pi;

    comp npP = std::sqrt(npPx*npPx + npPy*npPy + npPz*npPz);
    comp npPsq = npP * npP;

    int c1 = 0;
    int c2 = 0;

    if(std::abs(spec_p) < 1.0e-10) c1 = 1;
    if(std::abs(total_P_val) < 1.0e-10) c2 = 1;

    comp xibygamma = xii / gamma;

    int na_x_initial = -max_shell_num;
    int na_x_final   =  max_shell_num;
    int na_y_initial = -max_shell_num;
    int na_y_final   =  max_shell_num;
    int na_z_initial = -max_shell_num;
    int na_z_final   =  max_shell_num;

    std::cout << "\n";
    std::cout << "================ I_sum_ang_mom DEBUG ================\n";
    std::cout << "En              = " << En << "\n";
    std::cout << "sigma_p         = " << sigma_p << "\n";
    print_comp_vec3_debug("p", p);
    print_comp_vec3_debug("total_P", total_P);
    std::cout << std::setprecision(30); 
    std::cout << "spec_p              = " << spec_p << "\n";
    std::cout << "total_P_val         = " << total_P_val << "\n";
    std::cout << "mi,mj,mk            = " << mi << ", " << mj << ", " << mk << "\n";
    std::cout << "L                   = " << L << "\n";
    std::cout << "alpha               = " << alpha << "\n";
    std::cout << "Q0norm              = " << Q0norm << "\n";
    std::cout << "max_shell_num       = " << max_shell_num << "\n";
    std::cout << "ell_f,mf            = " << ell_f << ", " << proj_mf << "\n";
    std::cout << "ell_i,mi            = " << ell_i << ", " << proj_mi << "\n";
    std::cout << "gamma               = " << gamma << "\n";
    std::cout << "q2star              = " << q2star << "\n";
    std::cout << "x                   = " << x << "\n";
    std::cout << "x^2                 = " << x*x << "\n";
    std::cout << "xii                 = " << xii << "\n";
    std::cout << "xii/gamma           = " << xibygamma << "\n";
    std::cout << "npP                 = [" << npPx << ", " << npPy << ", " << npPz << "]\n";
    std::cout << "npPsq               = " << npPsq << "\n";
    std::cout << "2pi/L               = " << twopibyL << "\n";
    std::cout << "2pi/L^{ell_i+ell_f} = " << std::pow(twopibyL,ell_i + ell_f) << "\n";
    
    std::cout << "=====================================================\n";

    comp summ = {0.0, 0.0};
    comp scaled_summ = {0.0, 0.0}; 
    NeumaierComplexDouble scaled_sum_neum;
    std::vector<comp> bare_sum_vec;
    std::vector<comp> scaled_sum_vec;

    //Kahan Summation
    comp kahan_c = {0.0, 0.0};  // compensation

    int accepted_terms = 0;
    int printed_terms = 0;

    for(int i = na_x_initial; i <= na_x_final; ++i)
    {
        for(int j = na_y_initial; j <= na_y_final; ++j)
        {
            for(int k = na_z_initial; k <= na_z_final; ++k)
            {
                comp nax = (comp)i;
                comp nay = (comp)j;
                comp naz = (comp)k;

                comp na_dot_npP = nax*npPx + nay*npPy + naz*npPz;

                comp prod1 = {0.0, 0.0};

                if(std::abs(npPsq) > 0.0)
                {
                    prod1 =
                        (na_dot_npP / npPsq) *
                        ((((comp)1.0) / gamma) - ((comp)1.0))
                        - xibygamma;
                }

                comp rx = {0.0, 0.0};
                comp ry = {0.0, 0.0};
                comp rz = {0.0, 0.0};

                if(c1 == 1 && c2 == 1)
                {
                    rx = nax;
                    ry = nay;
                    rz = naz;
                }
                else
                {
                    if(std::abs(npPsq) == 0.0)
                    {
                        rx = nax;
                        ry = nay;
                        rz = naz;
                    }
                    else
                    {
                        rx = nax + npPx * prod1;
                        ry = nay + npPy * prod1;
                        rz = naz + npPz * prod1;
                    }
                }

                comp r = std::sqrt(rx*rx + ry*ry + rz*rz);
                comp r2 = r*r;

                std::vector<comp> rvec(3);
                rvec[0] = rx;
                rvec[1] = ry;
                rvec[2] = rz;

                comp Ylm1 = spherical_harmonics(rvec, ell_f, proj_mf);
                comp Ylm2 = spherical_harmonics(rvec, ell_i, proj_mi);

                comp prop_den = x*x - r*r;
                comp prop = ((comp)1.0) / prop_den;
                comp UV = std::exp(((comp)alpha) * prop_den);

                comp term = Ylm1 * UV * prop * Ylm2;

                comp prev_summ = summ;
                summ = summ + term;

                comp twopibyL = ((comp)2.0 * pi) / ((comp)L);
                comp norm = std::pow(twopibyL, (comp)(ell_f + ell_i));

                comp scaled_term = norm * term;
                bare_sum_vec.push_back(term);
                scaled_sum_vec.push_back(scaled_term);
                comp prev_scaled_summ = scaled_summ; 

                comp y = scaled_term - kahan_c;
                comp t = scaled_summ + y;
                kahan_c = (t - scaled_summ) - y;
                scaled_summ = t;

                comp prev_scaled_summ_neum = scaled_sum_neum.value(); 
                scaled_sum_neum.add(scaled_term); 

                comp scaled_summ_neum = scaled_sum_neum.value(); 

                //scaled_summ = scaled_summ + scaled_term; 


                ++accepted_terms;

                bool do_print = print_each_term;
                if(max_terms_to_print >= 0 && printed_terms >= max_terms_to_print)
                {
                    do_print = false;
                }

                if(do_print)
                {
                    std::cout << "-----------------------------------------------------\n";
                    std::cout << "term number       = " << accepted_terms << "\n";
                    std::cout << "na_vec            = ["
                              << nax << ", " << nay << ", " << naz << "]\n";
                    std::cout << "na_dot_npP        = " << na_dot_npP << "\n";
                    std::cout << "prod1             = " << prod1 << "\n";
                    std::cout << "rvec              = ["
                              << rx << ", " << ry << ", " << rz << "]\n";
                    std::cout << "r                 = " << r << "\n";
                    std::cout << "r^2               = " << r2 << "\n";
                    std::cout << "x^2 - r^2         = " << prop_den << "\n";
                    std::cout << "1/(x^2-r^2)       = " << prop << "\n";
                    std::cout << "UV exp            = " << UV << "\n";
                    std::cout << "Ylm1              = Y_("
                              << ell_f << "," << proj_mf << ") = "
                              << Ylm1 << "\n";
                    std::cout << "Ylm2              = Y_("
                              << ell_i << "," << proj_mi << ") = "
                              << Ylm2 << "\n";
                    std::cout << "Ylm1*Ylm2             = " << Ylm1 * Ylm2 << "\n";
                    std::cout << "bare summand          = " << term << "\n";
                    std::cout << "previous bare sum     = " << prev_summ << "\n";
                    std::cout << "new bare sum          = " << summ << "\n";
                    std::cout << "scaled term           = " << scaled_term << "\n"; 
                    std::cout << "prev. scaled_sum      = " << prev_scaled_summ << "\n";
                    std::cout << "new scaled sum        = " << scaled_summ << "\n"; 
                    std::cout << "prev. scaled sum neum = " << prev_scaled_summ_neum << "\n";
                    std::cout << "scaled sum neum       = " << scaled_summ_neum << "\n";
                    ++printed_terms;
                }
            }
        }
    }

    comp norm_factor = {1.0, 0.0};
    comp final_summ = summ;

    comp bare_sum_final_neum = precise_vector_sum(bare_sum_vec); 
    comp scaled_sum_final_neum = precise_vector_sum(scaled_sum_vec); 
    char debug = 'y'; 
    /*
    std::vector<comp> test_vec; 
    print_and_test_vector_sum( 10000000, test_vec); 
    comp test_vec_sum =
    precise_vector_sum_debug(
        test_vec,
        "test_vec debug",
        debug
    );
    */
   
    comp final_sum_neum_from_bare =
    precise_vector_sum_debug(
        bare_sum_vec,
        "bare_sum_vec debug",
        debug
    );

    comp final_sum_neum_from_scaled =
        precise_vector_sum_debug(
            scaled_sum_vec,
            "scaled_sum_vec debug",
            debug
        );
    
    //comp final_sum_neum_from_bare;
    //comp final_sum_neum_from_scaled; 

    /*std::cout << "HEY : " << scaled_sum_final_neum << std::endl; 
    for(int i=0;i<bare_sum_vec.size(); ++i)
    {
        std::cout << "i:" << i << " bsv:" << bare_sum_vec[i] << ' , ';
    }
    std::cout << std::endl;*/




    if(Q0norm)
    {
        norm_factor = std::pow(twopibyL, ell_f + ell_i);
        final_summ = summ * norm_factor;

        final_sum_neum_from_bare = norm_factor * final_sum_neum_from_bare; 
        final_sum_neum_from_scaled = final_sum_neum_from_scaled; 

        std::cout << "-----------------------------------------------------\n";
        std::cout << "Q0norm true.\n";
        std::cout << "normalization factor = (2pi/L)^(ell_f+ell_i)\n";
    }
    else
    {
        norm_factor = std::pow(x, ell_f + ell_i);
        final_summ = summ / norm_factor;

        comp qnorm = std::pow(q2star, ell_f + ell_i); 

        final_sum_neum_from_bare = bare_sum_final_neum / norm_factor ; 
        final_sum_neum_from_scaled = scaled_sum_final_neum / qnorm; 

        std::cout << "-----------------------------------------------------\n";
        std::cout << "Q0norm false.\n";
        std::cout << "normalization factor = x^(ell_f+ell_i)\n";
    }

    std::cout << std::scientific << std::setprecision(30);
    std::cout << "accepted terms       = " << accepted_terms << "\n";
    std::cout << "printed terms        = " << printed_terms << "\n";
    std::cout << "bare sum             = " << summ << "\n";
    std::cout << "scaled sum (kahan)   = " << scaled_summ << "\n";
    std::cout << "normalization factor = " << norm_factor << "\n";
    std::cout << "I_sum final (seq)    = " << final_summ << "\n";
    std::cout << "I_sum neum from bare = " << final_sum_neum_from_bare << "\n";
    std::cout << "I_sum neum  scaled   = " << final_sum_neum_from_scaled << "\n\n";

    std::cout << "diff |seq - neum bare|     = "
            << std::abs(final_summ - final_sum_neum_from_bare) << "\n";

    std::cout << "diff |seq - neum scaled|   = "
            << std::abs(final_summ - final_sum_neum_from_scaled) << "\n";

    std::cout << "diff |neum bare - scaled|  = "
            << std::abs(final_sum_neum_from_bare - final_sum_neum_from_scaled) << "\n\n";

    std::cout << "real                 = " << final_summ.real() << "\n";
    std::cout << "imag                 = " << final_summ.imag() << "\n";
    std::cout << "abs                  = " << std::abs(final_summ) << "\n";
    std::cout << "=====================================================\n";

    return final_summ;
}


// ============================================================================
// Debug version of F2_ang_mom
// Prints p,k,P, delta condition, sigma, cutoff, prefactor, I_sum, I_int,
// and final F2 value.
// ============================================================================

comp debug_F2_ang_mom_steps(
    comp En,
    std::vector<comp> k,
    std::vector<comp> p,
    std::vector<comp> total_P,
    int ell_f,
    int proj_mf,
    int ell_i,
    int proj_mi,
    double L,
    double mi,
    double mj,
    double mk,
    double alpha,
    double epsilon_h,
    int max_shell_num,
    bool Q0norm,
    int max_terms_to_print = 50,
    bool print_each_sum_term = true
)
{
    std::cout << std::setprecision(30);

    comp pi = std::acos(-1.0);

    comp kx = k[0];
    comp ky = k[1];
    comp kz = k[2];

    comp px = p[0];
    comp py = p[1];
    comp pz = p[2];

    comp Px = total_P[0];
    comp Py = total_P[1];
    comp Pz = total_P[2];

    comp spec_k = std::sqrt(kx*kx + ky*ky + kz*kz);
    comp spec_p = std::sqrt(px*px + py*py + pz*pz);
    comp total_P_val = std::sqrt(Px*Px + Py*Py + Pz*Pz);

    comp sigp = sigma_pvec_based(En, p, mi, total_P);
    comp cutoff = cutoff_function_1(sigp, mj, mk, epsilon_h);
    comp omega_p = omega_func(spec_p, mi);

    std::cout << "\n";
    std::cout << "#####################################################\n";
    std::cout << "############### F2_ang_mom DEBUG ###################\n";
    std::cout << "#####################################################\n";
    std::cout << "En              = " << En << "\n";
    print_comp_vec3_debug("k initial", k);
    print_comp_vec3_debug("p final  ", p);
    print_comp_vec3_debug("total_P  ", total_P);
    std::cout << "ell_f,mf        = " << ell_f << ", " << proj_mf << "\n";
    std::cout << "ell_i,mi        = " << ell_i << ", " << proj_mi << "\n";
    std::cout << "mi,mj,mk        = " << mi << ", " << mj << ", " << mk << "\n";
    std::cout << "L               = " << L << "\n";
    std::cout << "alpha           = " << alpha << "\n";
    std::cout << "epsilon_h       = " << epsilon_h << "\n";
    std::cout << "max_shell_num   = " << max_shell_num << "\n";
    std::cout << "Q0norm          = " << Q0norm << "\n";
    std::cout << "spec_k          = " << spec_k << "\n";
    std::cout << "spec_p          = " << spec_p << "\n";
    std::cout << "total_P_val     = " << total_P_val << "\n";
    std::cout << "sigma_p         = " << sigp << "\n";
    std::cout << "cutoff h        = " << cutoff << "\n";
    std::cout << "omega_p         = " << omega_p << "\n";
    std::cout << "En - omega_p    = " << En - omega_p << "\n";

    int condition_delta = 0;

    if(px == kx && py == ky && pz == kz)
    {
        condition_delta = 0;
    }
    else
    {
        condition_delta = 1;
    }

    std::cout << "delta condition = " << condition_delta << "\n";

    if(condition_delta == 1)
    {
        std::cout << "p != k, so this F2 component is exactly zero by spectator delta.\n";
        std::cout << "F2 value        = " << comp(0.0, 0.0) << "\n";
        std::cout << "#####################################################\n";
        return comp(0.0, 0.0);
    }

    comp A = cutoff /
        (((comp)16.0) * pi*pi *
         ((comp)L) * ((comp)L) * ((comp)L) * ((comp)L) *
         omega_p * (En - omega_p));

    std::cout << "prefactor A     = h/(16*pi^2*L^4*omega_p*(En-omega_p))\n";
    std::cout << "A               = " << A << "\n";

    comp B = debug_I_sum_ang_mom_steps(
        En,
        sigp,
        p,
        total_P,
        ell_f,
        proj_mf,
        ell_i,
        proj_mi,
        alpha,
        mi,
        mj,
        mk,
        L,
        max_shell_num,
        Q0norm,
        max_terms_to_print,
        print_each_sum_term
    );

    comp C = debug_I_int_ang_mom_steps(
        En,
        sigp,
        spec_p,
        total_P_val,
        ell_f,
        proj_mf,
        ell_i,
        proj_mi,
        alpha,
        mi,
        mj,
        mk,
        L,
        Q0norm
    );

    comp F2_val = comp(0.0, 0.0);

    if(std::abs(A) == 0.0)
    {
        F2_val = comp(0.0, 0.0);
    }
    else
    {
        F2_val = A * (B - C);
    }

    std::cout << "\n";
    std::cout << "================ FINAL F2 ASSEMBLY ==================\n";
    std::cout << "A prefactor     = " << A << "\n";
    std::cout << "B = I_sum       = " << B << "\n";
    std::cout << "C = I_int       = " << C << "\n";
    std::cout << "B - C           = " << B - C << "\n";
    std::cout << "F2 = A*(B-C)    = " << F2_val << "\n";
    std::cout << "real            = " << F2_val.real() << "\n";
    std::cout << "imag            = " << F2_val.imag() << "\n";
    std::cout << "abs             = " << std::abs(F2_val) << "\n";
    std::cout << "#####################################################\n";

    return F2_val;
}


// ============================================================================
// Debug full 2+1 F2 matrix component for arbitrary lm target.
// This loops through both flavor blocks and rebuilds every matching matrix entry.
// ============================================================================

void debug_F2_2plus1_component_for_lm_steps(
    const Eigen::MatrixXcd &F2,
    const std::vector<std::vector<comp> > &plm_config,
    const std::vector<std::vector<comp> > &klm_config,
    comp En,
    std::vector<comp> total_P,
    double m1,
    double m2,
    double L,
    double alpha,
    double epsilon_h,
    int max_shell_num,
    bool Q0norm,
    int target_ell_f,
    int target_proj_mf,
    int target_ell_i,
    int target_proj_mi,
    int max_terms_to_print = 50,
    bool print_each_sum_term = true,
    double compare_tol = 1.0e-10
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

    std::cout << "\n";
    std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
    std::cout << "%%%% DEBUG FULL F2_2PLUS1 LM COMPONENT CALCULATION %%\n";
    std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
    std::cout << "target ell_f   = " << target_ell_f << "\n";
    std::cout << "target proj_mf = " << target_proj_mf << "\n";
    std::cout << "target ell_i   = " << target_ell_i << "\n";
    std::cout << "target proj_mi = " << target_proj_mi << "\n";
    std::cout << "size1          = " << size1 << "\n";
    std::cout << "size2          = " << size2 << "\n";
    std::cout << "total_size     = " << total_size << "\n";
    std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";

    int count = 0;

    // ------------------------------------------------------------------------
    // Flavor block 1: F2_1
    // Mijk = [m1, m1, m2]
    // config = plm_config
    // ------------------------------------------------------------------------

    std::cout << "\n";
    std::cout << "=====================================================\n";
    std::cout << "FLAVOR BLOCK 1: F2_1, Mijk = [m1, m1, m2]\n";
    std::cout << "=====================================================\n";

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

            std::vector<comp> p(3);
            p[0] = plm_config[0][i];
            p[1] = plm_config[1][i];
            p[2] = plm_config[2][i];

            std::vector<comp> k(3);
            k[0] = plm_config[0][j];
            k[1] = plm_config[1][j];
            k[2] = plm_config[2][j];

            comp built_val = F2(i, j);

            std::cout << "\n";
            std::cout << "*************************************************************\n";
            std::cout << "MATCHING COMPONENT IN FLAVOR BLOCK 1\n";
            std::cout << "local indices  : i = " << i << ", j = " << j << "\n";
            std::cout << "global indices : i = " << i << ", j = " << j << "\n";
            print_comp_vec3_debug("p final", p);
            print_comp_vec3_debug("k initial", k);
            std::cout << "matrix stored F2(i,j) = " << built_val << "\n";
            std::cout << "*************************************************************\n";

            comp debug_val = debug_F2_ang_mom_steps(
                En,
                k,
                p,
                total_P,
                ell_f,
                proj_mf,
                ell_i,
                proj_mi,
                L,
                m1,
                m1,
                m2,
                alpha,
                epsilon_h,
                max_shell_num,
                Q0norm,
                max_terms_to_print,
                print_each_sum_term
            );

            comp diff = built_val - debug_val;

            std::cout << "\n";
            std::cout << "*************** COMPARISON WITH STORED MATRIX ***************\n";
            std::cout << "stored F2(i,j)       = " << built_val << "\n";
            std::cout << "debug recomputed F2  = " << debug_val << "\n";
            std::cout << "difference           = " << diff << "\n";
            std::cout << "|difference|         = " << std::abs(diff) << "\n";

            if(std::abs(diff) < compare_tol)
            {
                std::cout << "status               = MATCH within tol = "
                          << compare_tol << "\n";
            }
            else
            {
                std::cout << "status               = MISMATCH above tol = "
                          << compare_tol << "\n";
            }

            std::cout << "*************************************************************\n";

            ++count;
        }
    }

    // ------------------------------------------------------------------------
    // Flavor block 2: F2_2
    // Mijk = [m2, m1, m1]
    // config = klm_config
    // ------------------------------------------------------------------------

    std::cout << "\n";
    std::cout << "=====================================================\n";
    std::cout << "FLAVOR BLOCK 2: F2_2, Mijk = [m2, m1, m1]\n";
    std::cout << "=====================================================\n";

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

            std::vector<comp> p(3);
            p[0] = klm_config[0][i];
            p[1] = klm_config[1][i];
            p[2] = klm_config[2][i];

            std::vector<comp> k(3);
            k[0] = klm_config[0][j];
            k[1] = klm_config[1][j];
            k[2] = klm_config[2][j];

            const int gi = size1 + i;
            const int gj = size1 + j;

            comp built_val = F2(gi, gj);

            std::cout << "\n";
            std::cout << "*************************************************************\n";
            std::cout << "MATCHING COMPONENT IN FLAVOR BLOCK 2\n";
            std::cout << "local indices  : i = " << i << ", j = " << j << "\n";
            std::cout << "global indices : i = " << gi << ", j = " << gj << "\n";
            print_comp_vec3_debug("p final", p);
            print_comp_vec3_debug("k initial", k);
            std::cout << "matrix stored F2(gi,gj) = " << built_val << "\n";
            std::cout << "*************************************************************\n";

            comp debug_val = debug_F2_ang_mom_steps(
                En,
                k,
                p,
                total_P,
                ell_f,
                proj_mf,
                ell_i,
                proj_mi,
                L,
                m2,
                m1,
                m1,
                alpha,
                epsilon_h,
                max_shell_num,
                Q0norm,
                max_terms_to_print,
                print_each_sum_term
            );

            comp diff = built_val - debug_val;

            std::cout << "\n";
            std::cout << "*************** COMPARISON WITH STORED MATRIX ***************\n";
            std::cout << "stored F2(gi,gj)     = " << built_val << "\n";
            std::cout << "debug recomputed F2  = " << debug_val << "\n";
            std::cout << "difference           = " << diff << "\n";
            std::cout << "|difference|         = " << std::abs(diff) << "\n";

            if(std::abs(diff) < compare_tol)
            {
                std::cout << "status               = MATCH within tol = "
                          << compare_tol << "\n";
            }
            else
            {
                std::cout << "status               = MISMATCH above tol = "
                          << compare_tol << "\n";
            }

            std::cout << "*************************************************************\n";

            ++count;
        }
    }

    std::cout << "\n";
    std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
    std::cout << "Total matching components debugged = " << count << "\n";
    std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
}
            





#endif