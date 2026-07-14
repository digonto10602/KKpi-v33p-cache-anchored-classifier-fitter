#ifndef F2_GPU_SAFE_BUILDER_CUH
#define F2_GPU_SAFE_BUILDER_CUH

/*
  F2_gpu_safe_builder.cuh

  CUDA companion implementation for GPU-side construction of the 2+1 F2 matrix.

  This file is intended to be included from a .cu translation unit compiled by nvcc.

  What it does:
    - accepts your existing CPU-side configs:
          std::vector<std::vector<std::complex<double>>> plm_config, klm_config
    - flattens them into GPU-safe arrays
    - builds the full block-diagonal F2 matrix on the GPU:
          F2 = [ F2_1   0  ]
               [  0    F2_2]
    - copies the result back to Eigen::MatrixXcd

  Important numerical note:
    Your CPU F2 integral part calls Faddeeva::erfi. That is host-only.
    This GPU version uses a device-side complex erfi series approximation.
    For production, validate element-by-element against CPU F2 for representative
    energies and increase erfi_max_terms / adjust erfi_tol if needed.

  Compile with:
      nvcc -O3 -std=c++17 --expt-relaxed-constexpr -DEIGEN_NO_CUDA ...

  Dependencies on host side:
      #include <Eigen/Dense>
      #include <vector>
      #include <complex>
      #include <cuda_runtime.h>
      #include <cuComplex.h>
*/

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace f2gpu {

// =============================================================================
// Error checking
// =============================================================================

inline void check_cuda(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
    {
        throw std::runtime_error(std::string("CUDA error at ") + what + ": " + cudaGetErrorString(e));
    }
}

#define F2GPU_CUDA_CHECK(call) ::f2gpu::check_cuda((call), #call)

// =============================================================================
// Options
// =============================================================================

struct F2GpuOptions
{
    int threads_per_block = 256;

    // Device erfi approximation parameters.
    int erfi_max_terms = 80;
    double erfi_tol = 1.0e-15;

    // If true, prints rough timing and dimensions.
    char debug = 'n';
};

// =============================================================================
// GPU-safe complex arithmetic helpers
// =============================================================================

__host__ __device__ inline cuDoubleComplex zmake(double re, double im)
{
    return make_cuDoubleComplex(re, im);
}

__host__ __device__ inline double zre(cuDoubleComplex z) { return cuCreal(z); }
__host__ __device__ inline double zim(cuDoubleComplex z) { return cuCimag(z); }

__host__ __device__ inline cuDoubleComplex zadd(cuDoubleComplex a, cuDoubleComplex b)
{
    return make_cuDoubleComplex(cuCreal(a) + cuCreal(b), cuCimag(a) + cuCimag(b));
}

__host__ __device__ inline cuDoubleComplex zsub(cuDoubleComplex a, cuDoubleComplex b)
{
    return make_cuDoubleComplex(cuCreal(a) - cuCreal(b), cuCimag(a) - cuCimag(b));
}

__host__ __device__ inline cuDoubleComplex zmul(cuDoubleComplex a, cuDoubleComplex b)
{
    return make_cuDoubleComplex(
        cuCreal(a) * cuCreal(b) - cuCimag(a) * cuCimag(b),
        cuCreal(a) * cuCimag(b) + cuCimag(a) * cuCreal(b)
    );
}

__host__ __device__ inline cuDoubleComplex zmul_real(cuDoubleComplex a, double x)
{
    return make_cuDoubleComplex(cuCreal(a) * x, cuCimag(a) * x);
}

__host__ __device__ inline cuDoubleComplex zdiv(cuDoubleComplex a, cuDoubleComplex b)
{
    const double br = cuCreal(b);
    const double bi = cuCimag(b);
    const double den = br * br + bi * bi;

    return make_cuDoubleComplex(
        (cuCreal(a) * br + cuCimag(a) * bi) / den,
        (cuCimag(a) * br - cuCreal(a) * bi) / den
    );
}

__host__ __device__ inline cuDoubleComplex zdiv_real(cuDoubleComplex a, double x)
{
    return make_cuDoubleComplex(cuCreal(a) / x, cuCimag(a) / x);
}

__host__ __device__ inline double zabs(cuDoubleComplex a)
{
    return hypot(cuCreal(a), cuCimag(a));
}

__host__ __device__ inline cuDoubleComplex zneg(cuDoubleComplex a)
{
    return make_cuDoubleComplex(-cuCreal(a), -cuCimag(a));
}

__host__ __device__ inline cuDoubleComplex zsqrt(cuDoubleComplex z)
{
    const double x = cuCreal(z);
    const double y = cuCimag(z);

    if (y == 0.0)
    {
        if (x >= 0.0) return make_cuDoubleComplex(sqrt(x), 0.0);
        return make_cuDoubleComplex(0.0, sqrt(-x));
    }

    const double r = hypot(x, y);
    const double u = sqrt(0.5 * (r + x));
    const double v = copysign(sqrt(0.5 * (r - x)), y);
    return make_cuDoubleComplex(u, v);
}

__host__ __device__ inline cuDoubleComplex zexp(cuDoubleComplex z)
{
    const double ex = exp(cuCreal(z));
    return make_cuDoubleComplex(ex * cos(cuCimag(z)), ex * sin(cuCimag(z)));
}

__host__ __device__ inline cuDoubleComplex zpow_int(cuDoubleComplex z, int n)
{
    cuDoubleComplex out = make_cuDoubleComplex(1.0, 0.0);
    for (int i = 0; i < n; ++i) out = zmul(out, z);
    return out;
}

__host__ __device__ inline cuDoubleComplex zpow_real_positive_base(double base, double exponent)
{
    return make_cuDoubleComplex(pow(base, exponent), 0.0);
}

// =============================================================================
// Device physics scalar functions matching functions.h/F2_functions_v2.h logic
// =============================================================================

__host__ __device__ inline cuDoubleComplex omega_func_d(cuDoubleComplex p, double m)
{
    return zsqrt(zadd(zmul(p, p), make_cuDoubleComplex(m * m, 0.0)));
}

__host__ __device__ inline cuDoubleComplex kallentriangle_d(
    cuDoubleComplex x,
    cuDoubleComplex y,
    cuDoubleComplex z)
{
    const cuDoubleComplex two = make_cuDoubleComplex(2.0, 0.0);
    return zsub(
        zadd(zadd(zmul(x, x), zmul(y, y)), zmul(z, z)),
        zmul(two, zadd(zadd(zmul(x, y), zmul(y, z)), zmul(z, x)))
    );
}

__host__ __device__ inline cuDoubleComplex q2psq_star_d(
    cuDoubleComplex sigma_i,
    double mj,
    double mk)
{
    const cuDoubleComplex mj2 = make_cuDoubleComplex(mj * mj, 0.0);
    const cuDoubleComplex mk2 = make_cuDoubleComplex(mk * mk, 0.0);
    return zdiv(
        kallentriangle_d(sigma_i, mj2, mk2),
        zmul_real(sigma_i, 4.0)
    );
}

__host__ __device__ inline cuDoubleComplex Jfunc_d(cuDoubleComplex z)
{
    const double zr = cuCreal(z);

    if (zr <= 0.0)
    {
        return make_cuDoubleComplex(0.0, 0.0);
    }
    else if (zr > 0.0 && zr < 1.0)
    {
        const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex A = zdiv(make_cuDoubleComplex(-1.0, 0.0), z);
        cuDoubleComplex B = zexp(zdiv(make_cuDoubleComplex(-1.0, 0.0), zsub(one, z)));
        return zexp(zmul(A, B));
    }
    else
    {
        return make_cuDoubleComplex(1.0, 0.0);
    }
}

__host__ __device__ inline cuDoubleComplex cutoff_function_1_d(
    cuDoubleComplex sigma_i,
    double mj,
    double mk,
    double epsilon_h)
{
    if (mj == mk && epsilon_h == 0.0)
    {
        const double den = 4.0 * mj * mj;
        return Jfunc_d(zdiv_real(sigma_i, den));
    }

    const double absdiff = fabs(mj * mj - mk * mk);
    const double denom = (mj + mk) * (mj + mk) - absdiff;

    cuDoubleComplex num = zsub(sigma_i, make_cuDoubleComplex(absdiff, 0.0));
    num = zmul_real(num, 1.0 + epsilon_h);

    return Jfunc_d(zdiv_real(num, denom));
}

__host__ __device__ inline cuDoubleComplex sigma_pvec_based_d(
    cuDoubleComplex En,
    double px,
    double py,
    double pz,
    double mi,
    double Px,
    double Py,
    double Pz)
{
    const cuDoubleComplex p2 = make_cuDoubleComplex(px * px + py * py + pz * pz, 0.0);
    const cuDoubleComplex spec_p = zsqrt(p2);
    const cuDoubleComplex A = zsub(En, omega_func_d(spec_p, mi));

    const double dx = Px - px;
    const double dy = Py - py;
    const double dz = Pz - pz;
    const double Pminusp_sq = dx * dx + dy * dy + dz * dz;

    return zsub(zmul(A, A), make_cuDoubleComplex(Pminusp_sq, 0.0));
}

__host__ __device__ inline cuDoubleComplex spherical_harmonics_d(
    double px,
    double py,
    double pz,
    int ell,
    int m)
{
    if (ell == 0)
    {
        return make_cuDoubleComplex(1.0, 0.0);
    }

    if (ell == 1)
    {
        const double s3 = 1.73205080756887729352744634151;
        if (m == -1) return make_cuDoubleComplex(s3 * py, 0.0);
        if (m ==  0) return make_cuDoubleComplex(s3 * pz, 0.0);
        if (m == +1) return make_cuDoubleComplex(s3 * px, 0.0);
        return make_cuDoubleComplex(0.0, 0.0);
    }

    if (ell == 2)
    {
        const double s15 = 3.87298334620741688517926539978;
        const double s5over4 = 1.11803398874989484820458683437;
        if (m == -2) return make_cuDoubleComplex(s15 * px * py, 0.0);
        if (m == -1) return make_cuDoubleComplex(s15 * py * pz, 0.0);
        if (m ==  0) return make_cuDoubleComplex(s5over4 * (2.0 * pz * pz - px * px - py * py), 0.0);
        if (m == +1) return make_cuDoubleComplex(s15 * px * pz, 0.0);
        if (m == +2) return make_cuDoubleComplex(s5over4 * (px * px - py * py), 0.0);
        return make_cuDoubleComplex(0.0, 0.0);
    }

    return make_cuDoubleComplex(0.0, 0.0);
}

// erfi(z) = 2/sqrt(pi) sum_{n=0}^inf z^{2n+1}/(n!(2n+1))
// This replaces host-only Faddeeva::erfi inside I0F/I1F/I2F.
__device__ inline cuDoubleComplex erfi_series_d(
    cuDoubleComplex z,
    int max_terms,
    double tol)
{
    const double two_over_sqrt_pi = 1.12837916709551257389615890312;

    cuDoubleComplex z2 = zmul(z, z);
    cuDoubleComplex term = z;  // n = 0: z/(0!*1)
    cuDoubleComplex sum = term;

    for (int n = 0; n + 1 < max_terms; ++n)
    {
        // term_{n+1} / term_n = z^2 * (2n+1) / ((n+1)(2n+3))
        const double num = double(2 * n + 1);
        const double den = double((n + 1) * (2 * n + 3));
        term = zmul(term, zmul_real(z2, num / den));
        sum = zadd(sum, term);

        if (zabs(term) < tol * fmax(1.0, zabs(sum)))
        {
            break;
        }
    }

    return zmul_real(sum, two_over_sqrt_pi);
}

__device__ inline cuDoubleComplex I0F_d(
    cuDoubleComplex En,
    cuDoubleComplex sigma_p,
    double p_mag,
    double total_P_mag,
    double alpha,
    double mi,
    double mj,
    double mk,
    double L,
    int erfi_max_terms,
    double erfi_tol)
{
    (void)total_P_mag;
    const double pi = 3.141592653589793238462643383279502884;
    const double Lby2pi = L / (2.0 * pi);

    cuDoubleComplex gamma = zdiv(zsub(En, omega_func_d(make_cuDoubleComplex(p_mag, 0.0), mi)), zsqrt(sigma_p));
    cuDoubleComplex x = zmul_real(zsqrt(q2psq_star_d(sigma_p, mj, mk)), Lby2pi);

    cuDoubleComplex A = zmul_real(gamma, 4.0 * pi);
    cuDoubleComplex x2 = zmul(x, x);
    cuDoubleComplex alphax2 = zmul_real(x2, alpha);

    cuDoubleComplex B = zmul_real(zexp(alphax2), -sqrt(pi / alpha) * 0.5);
    cuDoubleComplex erfi_val = erfi_series_d(zsqrt(alphax2), erfi_max_terms, erfi_tol);
    cuDoubleComplex C = zmul(zmul_real(x, 0.5 * pi), erfi_val);

    return zmul(A, zadd(B, C));
}

__device__ inline cuDoubleComplex I1F_d(
    cuDoubleComplex En,
    cuDoubleComplex sigma_p,
    double p_mag,
    double total_P_mag,
    double alpha,
    double mi,
    double mj,
    double mk,
    double L,
    int erfi_max_terms,
    double erfi_tol)
{
    (void)total_P_mag;
    const double pi = 3.141592653589793238462643383279502884;
    const double Lby2pi = L / (2.0 * pi);

    cuDoubleComplex gamma = zdiv(zsub(En, omega_func_d(make_cuDoubleComplex(p_mag, 0.0), mi)), zsqrt(sigma_p));
    cuDoubleComplex x = zmul_real(zsqrt(q2psq_star_d(sigma_p, mj, mk)), Lby2pi);
    cuDoubleComplex x2 = zmul(x, x);
    cuDoubleComplex x3 = zmul(x2, x);
    cuDoubleComplex alphax2 = zmul_real(x2, alpha);

    cuDoubleComplex A = zmul_real(gamma, 4.0 * pi);

    cuDoubleComplex one_plus = zadd(make_cuDoubleComplex(1.0, 0.0), zmul_real(x2, 2.0 * alpha));
    const double pref = -sqrt(pi / (alpha * alpha * alpha)) / 4.0;
    cuDoubleComplex B = zmul_real(zmul(one_plus, zexp(alphax2)), pref);

    cuDoubleComplex erfi_val = erfi_series_d(zsqrt(alphax2), erfi_max_terms, erfi_tol);
    cuDoubleComplex C = zmul(zmul_real(x3, 0.5 * pi), erfi_val);

    return zmul(A, zadd(B, C));
}

__device__ inline cuDoubleComplex I2F_d(
    cuDoubleComplex En,
    cuDoubleComplex sigma_p,
    double p_mag,
    double total_P_mag,
    double alpha,
    double mi,
    double mj,
    double mk,
    double L,
    int erfi_max_terms,
    double erfi_tol)
{
    (void)total_P_mag;
    const double pi = 3.141592653589793238462643383279502884;
    const double Lby2pi = L / (2.0 * pi);

    cuDoubleComplex gamma = zdiv(zsub(En, omega_func_d(make_cuDoubleComplex(p_mag, 0.0), mi)), zsqrt(sigma_p));
    cuDoubleComplex x = zmul_real(zsqrt(q2psq_star_d(sigma_p, mj, mk)), Lby2pi);
    cuDoubleComplex x2 = zmul(x, x);
    cuDoubleComplex x4 = zmul(x2, x2);
    cuDoubleComplex x5 = zmul(x4, x);
    cuDoubleComplex alphax2 = zmul_real(x2, alpha);

    cuDoubleComplex A = zmul_real(gamma, 4.0 * pi);

    cuDoubleComplex poly = make_cuDoubleComplex(3.0, 0.0);
    poly = zadd(poly, zmul_real(x2, 2.0 * alpha));
    poly = zadd(poly, zmul_real(x4, 4.0 * alpha * alpha));

    const double pref = -sqrt(pi / pow(alpha, 5.0)) / 8.0;
    cuDoubleComplex B = zmul_real(zmul(poly, zexp(alphax2)), pref);

    cuDoubleComplex erfi_val = erfi_series_d(zsqrt(alphax2), erfi_max_terms, erfi_tol);
    cuDoubleComplex C = zmul(zmul_real(x5, 0.5 * pi), erfi_val);

    return zmul(A, zadd(B, C));
}

__device__ inline cuDoubleComplex I_int_ang_mom_d(
    cuDoubleComplex En,
    cuDoubleComplex sigma_p,
    double p_mag,
    double total_P_mag,
    int ell_f,
    int proj_mf,
    int ell_i,
    int proj_mi,
    double alpha,
    double mi,
    double mj,
    double mk,
    double L,
    bool Q0norm,
    int erfi_max_terms,
    double erfi_tol)
{
    if (ell_f != ell_i || proj_mf != proj_mi)
    {
        return make_cuDoubleComplex(0.0, 0.0);
    }

    const double pi = 3.141592653589793238462643383279502884;
    const double twopibyL = 2.0 * pi / L;
    const double Lby2pi = L / (2.0 * pi);

    cuDoubleComplex x = zmul_real(zsqrt(q2psq_star_d(sigma_p, mj, mk)), Lby2pi);

    cuDoubleComplex I = make_cuDoubleComplex(0.0, 0.0);

    if (ell_i == 0)
    {
        I = I0F_d(En, sigma_p, p_mag, total_P_mag, alpha, mi, mj, mk, L, erfi_max_terms, erfi_tol);
    }
    else if (ell_i == 1)
    {
        I = I1F_d(En, sigma_p, p_mag, total_P_mag, alpha, mi, mj, mk, L, erfi_max_terms, erfi_tol);
        if (Q0norm) I = zmul_real(I, pow(twopibyL, 2.0 * ell_i));
        else I = zdiv(I, zpow_int(x, 2 * ell_i));
    }
    else if (ell_i == 2)
    {
        I = I2F_d(En, sigma_p, p_mag, total_P_mag, alpha, mi, mj, mk, L, erfi_max_terms, erfi_tol);
        if (Q0norm) I = zmul_real(I, pow(twopibyL, 2.0 * ell_i));
        else I = zdiv(I, zpow_int(x, 2 * ell_i));
    }

    return I;
}

__device__ inline cuDoubleComplex I_sum_ang_mom_d(
    cuDoubleComplex En,
    cuDoubleComplex sigma_p,
    double px,
    double py,
    double pz,
    double Px,
    double Py,
    double Pz,
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
    bool Q0norm)
{
    (void)Q0norm;

    const double pi = 3.141592653589793238462643383279502884;
    const double Lby2pi = L / (2.0 * pi);
    const double twopibyL = 2.0 * pi / L;

    const double spec_p = sqrt(px * px + py * py + pz * pz);
    const double total_P_val = sqrt(Px * Px + Py * Py + Pz * Pz);

    cuDoubleComplex gamma = zdiv(
        zsub(En, omega_func_d(make_cuDoubleComplex(spec_p, 0.0), mi)),
        zsqrt(sigma_p)
    );

    cuDoubleComplex x = zmul_real(zsqrt(q2psq_star_d(sigma_p, mj, mk)), Lby2pi);
    cuDoubleComplex x2 = zmul(x, x);

    cuDoubleComplex xii = zmul_real(
        zadd(make_cuDoubleComplex(1.0, 0.0), zdiv_real(make_cuDoubleComplex(mj * mj - mk * mk, 0.0), 1.0)),
        0.0
    );
    // Explicitly reproduce: xii = 0.5 * (1 + (mj^2 - mk^2)/sigma_p)
    xii = zmul_real(zadd(make_cuDoubleComplex(1.0, 0.0), zdiv(make_cuDoubleComplex(mj * mj - mk * mk, 0.0), sigma_p)), 0.5);

    const double npPx = (Px - px) * Lby2pi;
    const double npPy = (Py - py) * Lby2pi;
    const double npPz = (Pz - pz) * Lby2pi;
    const double npPsq = npPx * npPx + npPy * npPy + npPz * npPz;

    const int c1 = (fabs(spec_p) < 1.0e-10) ? 1 : 0;
    const int c2 = (fabs(total_P_val) < 1.0e-10) ? 1 : 0;

    cuDoubleComplex xibygamma = zdiv(xii, gamma);

    cuDoubleComplex summ = make_cuDoubleComplex(0.0, 0.0);

    for (int ia = -max_shell_num; ia <= max_shell_num; ++ia)
    {
        for (int ja = -max_shell_num; ja <= max_shell_num; ++ja)
        {
            for (int ka = -max_shell_num; ka <= max_shell_num; ++ka)
            {
                const double nax = double(ia);
                const double nay = double(ja);
                const double naz = double(ka);

                double rx = nax;
                double ry = nay;
                double rz = naz;

                if (!(c1 == 1 && c2 == 1) && fabs(npPsq) != 0.0)
                {
                    const double na_dot_npP = nax * npPx + nay * npPy + naz * npPz;
                    cuDoubleComplex prod1 = zsub(
                        zmul_real(zsub(zdiv(make_cuDoubleComplex(1.0, 0.0), gamma), make_cuDoubleComplex(1.0, 0.0)), na_dot_npP / npPsq),
                        xibygamma
                    );

                    // For your real-energy use case, prod1 is expected real-like.
                    const double pr = cuCreal(prod1);
                    rx = nax + npPx * pr;
                    ry = nay + npPy * pr;
                    rz = naz + npPz * pr;
                }

                const double r2_real = rx * rx + ry * ry + rz * rz;
                cuDoubleComplex prop = zsub(x2, make_cuDoubleComplex(r2_real, 0.0));
                cuDoubleComplex UV = zexp(zmul_real(prop, alpha));

                cuDoubleComplex Ylm1 = spherical_harmonics_d(rx, ry, rz, ell_f, proj_mf);
                cuDoubleComplex Ylm2 = spherical_harmonics_d(rx, ry, rz, ell_i, proj_mi);

                cuDoubleComplex term = zdiv(zmul(zmul(Ylm1, UV), Ylm2), prop);
                summ = zadd(summ, term);
            }
        }
    }

    const double pow_term = pow(twopibyL, double(ell_f + ell_i));
    return zmul_real(summ, pow_term);
}

__device__ inline cuDoubleComplex F2_ang_mom_d(
    cuDoubleComplex En,
    double px,
    double py,
    double pz,
    int ell_f,
    int proj_mf,
    double kx,
    double ky,
    double kz,
    int ell_i,
    int proj_mi,
    double Px,
    double Py,
    double Pz,
    double mi,
    double mj,
    double mk,
    double L,
    double alpha,
    double epsilon_h,
    int max_shell_num,
    bool Q0norm,
    int erfi_max_terms,
    double erfi_tol)
{
    // Exact CPU logic: return zero unless spectator momentum p == k.
    // Since both are generated as exact integer multiples of 2pi/L, exact compare is fine
    // when copied from CPU configs. Keep a small tolerance to be safe.
    const double tol = 1.0e-14;
    if (fabs(px - kx) > tol || fabs(py - ky) > tol || fabs(pz - kz) > tol)
    {
        return make_cuDoubleComplex(0.0, 0.0);
    }

    const double spec_p = sqrt(px * px + py * py + pz * pz);
    const double total_P_val = sqrt(Px * Px + Py * Py + Pz * Pz);

    cuDoubleComplex sigp = sigma_pvec_based_d(En, px, py, pz, mi, Px, Py, Pz);
    cuDoubleComplex cutoff = cutoff_function_1_d(sigp, mj, mk, epsilon_h);
    cuDoubleComplex omega_p = omega_func_d(make_cuDoubleComplex(spec_p, 0.0), mi);

    const double pi = 3.141592653589793238462643383279502884;

    cuDoubleComplex denom = zmul(
        zmul_real(omega_p, 16.0 * pi * pi * L * L * L * L),
        zsub(En, omega_p)
    );

    cuDoubleComplex A = zdiv(cutoff, denom);

    if (zabs(A) == 0.0)
    {
        return make_cuDoubleComplex(0.0, 0.0);
    }

    cuDoubleComplex B = I_sum_ang_mom_d(
        En, sigp,
        px, py, pz,
        Px, Py, Pz,
        ell_f, proj_mf,
        ell_i, proj_mi,
        alpha,
        mi, mj, mk,
        L,
        max_shell_num,
        Q0norm
    );

    cuDoubleComplex C = I_int_ang_mom_d(
        En, sigp,
        spec_p,
        total_P_val,
        ell_f, proj_mf,
        ell_i, proj_mi,
        alpha,
        mi, mj, mk,
        L,
        Q0norm,
        erfi_max_terms,
        erfi_tol
    );

    return zmul(A, zsub(B, C));
}

// =============================================================================
// Flat config representation
// =============================================================================

struct FlatConfig
{
    int n = 0;
    std::vector<double> px, py, pz;
    std::vector<int> ell, m;
};

inline FlatConfig flatten_config(const std::vector<std::vector<std::complex<double>>>& cfg)
{
    if (cfg.size() < 5)
    {
        throw std::runtime_error("flatten_config: config must have 5 rows");
    }

    FlatConfig out;
    out.n = static_cast<int>(cfg[0].size());
    out.px.resize(out.n);
    out.py.resize(out.n);
    out.pz.resize(out.n);
    out.ell.resize(out.n);
    out.m.resize(out.n);

    for (int i = 0; i < out.n; ++i)
    {
        out.px[static_cast<std::size_t>(i)] = cfg[0][static_cast<std::size_t>(i)].real();
        out.py[static_cast<std::size_t>(i)] = cfg[1][static_cast<std::size_t>(i)].real();
        out.pz[static_cast<std::size_t>(i)] = cfg[2][static_cast<std::size_t>(i)].real();
        out.ell[static_cast<std::size_t>(i)] = static_cast<int>(std::llround(cfg[3][static_cast<std::size_t>(i)].real()));
        out.m[static_cast<std::size_t>(i)] = static_cast<int>(std::llround(cfg[4][static_cast<std::size_t>(i)].real()));
    }

    return out;
}

// =============================================================================
// Device memory RAII
// =============================================================================

template <typename T>
struct DeviceBuffer
{
    T* ptr = nullptr;
    std::size_t count = 0;

    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t n) { allocate(n); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
    {
        ptr = other.ptr;
        count = other.count;
        other.ptr = nullptr;
        other.count = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other)
        {
            release();
            ptr = other.ptr;
            count = other.count;
            other.ptr = nullptr;
            other.count = 0;
        }
        return *this;
    }

    ~DeviceBuffer() { release(); }

    void allocate(std::size_t n)
    {
        release();
        count = n;
        if (n > 0)
        {
            F2GPU_CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
        }
    }

    void release()
    {
        if (ptr)
        {
            cudaFree(ptr);
            ptr = nullptr;
            count = 0;
        }
    }

    void copy_from_host(const std::vector<T>& h)
    {
        if (count != h.size()) allocate(h.size());
        if (!h.empty()) F2GPU_CUDA_CHECK(cudaMemcpy(ptr, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copy_to_host(std::vector<T>& h) const
    {
        h.resize(count);
        if (count > 0) F2GPU_CUDA_CHECK(cudaMemcpy(h.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
};

struct DeviceFlatConfig
{
    int n = 0;
    DeviceBuffer<double> px, py, pz;
    DeviceBuffer<int> ell, m;

    explicit DeviceFlatConfig(const FlatConfig& h)
    {
        n = h.n;
        px.copy_from_host(h.px);
        py.copy_from_host(h.py);
        pz.copy_from_host(h.pz);
        ell.copy_from_host(h.ell);
        m.copy_from_host(h.m);
    }
};

// =============================================================================
// Kernels
// =============================================================================

__global__ void build_F2_2plus1_kernel(
    cuDoubleComplex* F2_out,
    int total_dim,
    cuDoubleComplex En,
    const double* p1x,
    const double* p1y,
    const double* p1z,
    const int* ell1,
    const int* m1cfg,
    int size1,
    const double* p2x,
    const double* p2y,
    const double* p2z,
    const int* ell2,
    const int* m2cfg,
    int size2,
    double Px,
    double Py,
    double Pz,
    double mK,
    double mpi,
    double L,
    double alpha,
    double epsilon_h,
    int max_shell_num,
    bool Q0norm,
    int erfi_max_terms,
    double erfi_tol)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elems = total_dim * total_dim;

    if (tid >= total_elems)
    {
        return;
    }

    const int row = tid % total_dim;
    const int col = tid / total_dim; // column-major for Eigen/cuBLAS compatibility

    cuDoubleComplex val = make_cuDoubleComplex(0.0, 0.0);

    if (row < size1 && col < size1)
    {
        // Flavor/channel 1: mi=mK, mj=mK, mk=mpi
        val = F2_ang_mom_d(
            En,
            p1x[row], p1y[row], p1z[row], ell1[row], m1cfg[row],
            p1x[col], p1y[col], p1z[col], ell1[col], m1cfg[col],
            Px, Py, Pz,
            mK, mK, mpi,
            L, alpha, epsilon_h,
            max_shell_num, Q0norm,
            erfi_max_terms, erfi_tol
        );
    }
    else if (row >= size1 && col >= size1)
    {
        // Flavor/channel 2: mi=mpi, mj=mK, mk=mK
        const int rr = row - size1;
        const int cc = col - size1;

        val = F2_ang_mom_d(
            En,
            p2x[rr], p2y[rr], p2z[rr], ell2[rr], m2cfg[rr],
            p2x[cc], p2y[cc], p2z[cc], ell2[cc], m2cfg[cc],
            Px, Py, Pz,
            mpi, mK, mK,
            L, alpha, epsilon_h,
            max_shell_num, Q0norm,
            erfi_max_terms, erfi_tol
        );
    }

    F2_out[tid] = val;
}

// =============================================================================
// Host API: one energy point
// =============================================================================

inline void F2_2plus1_mat_gpu_safe(
    Eigen::MatrixXcd& F2,
    std::complex<double> En,
    const std::vector<std::vector<std::complex<double>>>& plm_config,
    const std::vector<std::vector<std::complex<double>>>& klm_config,
    const std::vector<std::complex<double>>& total_P,
    double m1,
    double m2,
    double L,
    double alpha,
    double epsilon_h,
    int max_shell_num,
    bool Q0norm,
    const F2GpuOptions& opt = F2GpuOptions())
{
    if (total_P.size() < 3)
    {
        throw std::runtime_error("F2_2plus1_mat_gpu_safe: total_P must have size 3");
    }

    FlatConfig h1 = flatten_config(plm_config);
    FlatConfig h2 = flatten_config(klm_config);

    const int size1 = h1.n;
    const int size2 = h2.n;
    const int total_dim = size1 + size2;

    F2.resize(total_dim, total_dim);

    if (total_dim == 0)
    {
        return;
    }

    DeviceFlatConfig d1(h1);
    DeviceFlatConfig d2(h2);

    DeviceBuffer<cuDoubleComplex> dF2(static_cast<std::size_t>(total_dim) * static_cast<std::size_t>(total_dim));

    const int total_elems = total_dim * total_dim;
    const int threads = opt.threads_per_block;
    const int blocks = (total_elems + threads - 1) / threads;

    cuDoubleComplex En_d = make_cuDoubleComplex(En.real(), En.imag());

    build_F2_2plus1_kernel<<<blocks, threads>>>(
        dF2.ptr,
        total_dim,
        En_d,
        d1.px.ptr, d1.py.ptr, d1.pz.ptr, d1.ell.ptr, d1.m.ptr, size1,
        d2.px.ptr, d2.py.ptr, d2.pz.ptr, d2.ell.ptr, d2.m.ptr, size2,
        total_P[0].real(), total_P[1].real(), total_P[2].real(),
        m1, m2,
        L, alpha, epsilon_h,
        max_shell_num,
        Q0norm,
        opt.erfi_max_terms,
        opt.erfi_tol
    );

    F2GPU_CUDA_CHECK(cudaGetLastError());
    F2GPU_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<cuDoubleComplex> hF2;
    dF2.copy_to_host(hF2);

    for (int col = 0; col < total_dim; ++col)
    {
        for (int row = 0; row < total_dim; ++row)
        {
            const std::size_t idx = static_cast<std::size_t>(col) * static_cast<std::size_t>(total_dim) + static_cast<std::size_t>(row);
            F2(row, col) = std::complex<double>(cuCreal(hF2[idx]), cuCimag(hF2[idx]));
        }
    }

    if (opt.debug == 'y')
    {
        std::cout << "[F2 GPU] built one F2 matrix: size1=" << size1
                  << " size2=" << size2
                  << " total_dim=" << total_dim
                  << " blocks=" << blocks
                  << " threads=" << threads << '\n';
    }
}

// =============================================================================
// Host API: many energies, existing CPU configs already built
// =============================================================================

inline void F2_2plus1_many_mat_gpu_safe(
    std::vector<Eigen::MatrixXcd>& F2_vec,
    const std::vector<std::complex<double>>& En_vec,
    const std::vector<std::vector<std::vector<std::complex<double>>>>& plm_vec,
    const std::vector<std::vector<std::vector<std::complex<double>>>>& klm_vec,
    const std::vector<std::complex<double>>& total_P,
    double m1,
    double m2,
    double L,
    double alpha,
    double epsilon_h,
    int max_shell_num,
    bool Q0norm,
    const F2GpuOptions& opt = F2GpuOptions())
{
    const std::size_t N = En_vec.size();

    if (plm_vec.size() != N || klm_vec.size() != N)
    {
        throw std::runtime_error("F2_2plus1_many_mat_gpu_safe: inconsistent vector sizes");
    }

    F2_vec.assign(N, Eigen::MatrixXcd());

    for (std::size_t i = 0; i < N; ++i)
    {
        F2_2plus1_mat_gpu_safe(
            F2_vec[i],
            En_vec[i],
            plm_vec[i],
            klm_vec[i],
            total_P,
            m1,
            m2,
            L,
            alpha,
            epsilon_h,
            max_shell_num,
            Q0norm,
            opt
        );

        if (opt.debug == 'y' && (i % 100 == 0 || i + 1 == N))
        {
            std::cout << "[F2 GPU] progress " << i + 1 << " / " << N << '\n';
        }
    }
}

// =============================================================================
// Comparison helper against CPU F2
// =============================================================================

inline void compare_F2_cpu_gpu(
    const Eigen::MatrixXcd& F2_cpu,
    const Eigen::MatrixXcd& F2_gpu,
    const std::string& label = "F2",
    double warn_abs_tol = 1.0e-8)
{
    if (F2_cpu.rows() != F2_gpu.rows() || F2_cpu.cols() != F2_gpu.cols())
    {
        std::cout << "[F2 compare] " << label << " shape mismatch: CPU "
                  << F2_cpu.rows() << "x" << F2_cpu.cols()
                  << " GPU " << F2_gpu.rows() << "x" << F2_gpu.cols() << '\n';
        return;
    }

    double max_abs = 0.0;
    double max_rel = 0.0;
    int max_i = -1;
    int max_j = -1;

    for (int i = 0; i < F2_cpu.rows(); ++i)
    {
        for (int j = 0; j < F2_cpu.cols(); ++j)
        {
            const std::complex<double> diff = F2_cpu(i, j) - F2_gpu(i, j);
            const double ad = std::abs(diff);
            const double denom = std::max(1.0, std::abs(F2_cpu(i, j)));
            const double rd = ad / denom;

            if (ad > max_abs)
            {
                max_abs = ad;
                max_rel = rd;
                max_i = i;
                max_j = j;
            }
        }
    }

    std::cout << std::scientific << std::setprecision(17);
    std::cout << "[F2 compare] " << label
              << " max_abs=" << max_abs
              << " max_rel=" << max_rel
              << " at (" << max_i << "," << max_j << ")";

    if (max_abs > warn_abs_tol)
    {
        std::cout << "  WARNING above tol=" << warn_abs_tol;
    }

    std::cout << '\n';
}

} // namespace f2gpu

#endif // F2_GPU_SAFE_BUILDER_CUH
