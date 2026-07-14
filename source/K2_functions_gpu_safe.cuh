#ifndef K2_FUNCTIONS_GPU_SAFE_CUH
#define K2_FUNCTIONS_GPU_SAFE_CUH

/*
  K2_functions_gpu_safe.cuh

  GPU-safe companion for K2_functions_v2.h.

  Target CPU reference structure:

      K2inv_EREord2_2plus1_mat
          K2inv_EREord2_i_mat
              K2_inv_ERE_ang_mom

  This file intentionally does NOT include F2_functions_v2.h or K2_functions_v2.h.
  It rewrites the needed scalar functions as __host__ __device__ safe routines and
  builds the full two-flavor K2inv matrix on the GPU.

  Output block structure:

      K2inv = [ K2inv_1       0      ]
              [   0       2*K2inv_2 ]

  where flavor 1 uses masses (mi,mj,mk) = (m1,m1,m2), eta_1, scatter_params_1
  and flavor 2 uses masses (mi,mj,mk) = (m2,m1,m1), eta_2, scatter_params_2.

  Notes:
    - One CUDA thread computes one matrix element.
    - Off-diagonal blocks are explicitly set to zero.
    - Within each flavor block K2inv is diagonal in the current CPU definition.
    - This file is suitable for inclusion in .cu files compiled by nvcc.
*/

#include <cuda_runtime.h>
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace k2gpu {

using host_comp = std::complex<double>;

#ifndef K2GPU_CUDA_CHECK
#define K2GPU_CUDA_CHECK(call)                                                       \
    do {                                                                             \
        cudaError_t k2gpu_err__ = (call);                                             \
        if (k2gpu_err__ != cudaSuccess) {                                             \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ +      \
                                     ":" + std::to_string(__LINE__) + " : " +       \
                                     cudaGetErrorString(k2gpu_err__));                \
        }                                                                            \
    } while (0)
#endif

struct Cx
{
    double re;
    double im;

    __host__ __device__ Cx() : re(0.0), im(0.0) {}
    __host__ __device__ Cx(double r, double i = 0.0) : re(r), im(i) {}
};

__host__ __device__ inline Cx make_cx(double r, double i = 0.0)
{
    return Cx(r, i);
}

__host__ __device__ inline Cx operator+(Cx a, Cx b) { return Cx(a.re + b.re, a.im + b.im); }
__host__ __device__ inline Cx operator-(Cx a, Cx b) { return Cx(a.re - b.re, a.im - b.im); }
__host__ __device__ inline Cx operator-(Cx a) { return Cx(-a.re, -a.im); }
__host__ __device__ inline Cx operator*(Cx a, Cx b)
{
    return Cx(a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re);
}
__host__ __device__ inline Cx operator/(Cx a, Cx b)
{
    const double den = b.re*b.re + b.im*b.im;
    return Cx((a.re*b.re + a.im*b.im)/den, (a.im*b.re - a.re*b.im)/den);
}

__host__ __device__ inline Cx operator+(Cx a, double b) { return Cx(a.re + b, a.im); }
__host__ __device__ inline Cx operator+(double a, Cx b) { return Cx(a + b.re, b.im); }
__host__ __device__ inline Cx operator-(Cx a, double b) { return Cx(a.re - b, a.im); }
__host__ __device__ inline Cx operator-(double a, Cx b) { return Cx(a - b.re, -b.im); }
__host__ __device__ inline Cx operator*(Cx a, double b) { return Cx(a.re*b, a.im*b); }
__host__ __device__ inline Cx operator*(double a, Cx b) { return Cx(a*b.re, a*b.im); }
__host__ __device__ inline Cx operator/(Cx a, double b) { return Cx(a.re/b, a.im/b); }
__host__ __device__ inline Cx operator/(double a, Cx b)
{
    const double den = b.re*b.re + b.im*b.im;
    return Cx(a*b.re/den, -a*b.im/den);
}

__host__ __device__ inline double norm2(Cx z) { return z.re*z.re + z.im*z.im; }
__host__ __device__ inline double abs_cx(Cx z) { return sqrt(norm2(z)); }

__host__ __device__ inline Cx sqrt_cx(Cx z)
{
    const double x = z.re;
    const double y = z.im;

    if (y == 0.0) {
        if (x >= 0.0) return Cx(sqrt(x), 0.0);
        return Cx(0.0, sqrt(-x));
    }

    const double r = hypot(x, y);
    const double u = sqrt(0.5*(r + x));
    const double v = copysign(sqrt(0.5*(r - x)), y);
    return Cx(u, v);
}

__host__ __device__ inline Cx exp_cx(Cx z)
{
    const double e = exp(z.re);
    return Cx(e*cos(z.im), e*sin(z.im));
}

__host__ __device__ inline Cx pow_int_cx(Cx z, int n)
{
    if (n == 0) return Cx(1.0, 0.0);
    Cx out(1.0, 0.0);
    int m = n;
    if (m < 0) m = -m;
    for (int i = 0; i < m; ++i) out = out * z;
    if (n < 0) out = Cx(1.0, 0.0) / out;
    return out;
}

__host__ __device__ inline Cx omega_func_gpu(Cx p, double m)
{
    return sqrt_cx(p*p + m*m);
}

__host__ __device__ inline Cx kallen_gpu(Cx x, Cx y, Cx z)
{
    return x*x + y*y + z*z - 2.0*(x*y + y*z + z*x);
}

__host__ __device__ inline Cx q2psq_star_gpu(Cx sigma_i, double mj, double mk)
{
    const Cx mj2(mj*mj, 0.0);
    const Cx mk2(mk*mk, 0.0);
    return kallen_gpu(sigma_i, mj2, mk2) / (4.0*sigma_i);
}

__host__ __device__ inline Cx Jfunc_gpu(Cx z)
{
    if (z.re <= 0.0) {
        return Cx(0.0, 0.0);
    }
    else if (z.re > 0.0 && z.re < 1.0) {
        const Cx A = -1.0 / z;
        const Cx B = exp_cx(-1.0 / (Cx(1.0, 0.0) - z));
        return exp_cx(A * B);
    }
    else {
        return Cx(1.0, 0.0);
    }
}

__host__ __device__ inline Cx cutoff_function_1_gpu(Cx sigma_i, double mj, double mk, double epsilon_h)
{
    const double mj2 = mj*mj;
    const double mk2 = mk*mk;

    if (mj == mk && epsilon_h == 0.0) {
        Cx Z = sigma_i / (4.0*mj2);
        return Jfunc_gpu(Z);
    }
    else {
        const double absdiff = fabs(mj2 - mk2);
        const double denom = (mj + mk)*(mj + mk) - absdiff;
        Cx Z = ((1.0 + epsilon_h) * (sigma_i - absdiff)) / denom;
        return Jfunc_gpu(Z);
    }
}

struct ConfigView
{
    int n;
    const double* px;
    const double* py;
    const double* pz;
    const int* ell;
    const int* m;
};

struct ScatterParamsView
{
    // row-major [ell][0..2], max ell = 3, total 12 complex values
    const double* re;
    const double* im;

    __device__ inline Cx get(int ell, int idx) const
    {
        const int off = 3*ell + idx;
        return Cx(re[off], im[off]);
    }
};

class DeviceConfig
{
public:
    int n = 0;
    double* d_px = nullptr;
    double* d_py = nullptr;
    double* d_pz = nullptr;
    int* d_ell = nullptr;
    int* d_m = nullptr;

    DeviceConfig() = default;
    DeviceConfig(const DeviceConfig&) = delete;
    DeviceConfig& operator=(const DeviceConfig&) = delete;

    DeviceConfig(DeviceConfig&& other) noexcept { move_from(other); }
    DeviceConfig& operator=(DeviceConfig&& other) noexcept
    {
        if (this != &other) {
            release();
            move_from(other);
        }
        return *this;
    }

    ~DeviceConfig() { release(); }

    ConfigView view() const
    {
        ConfigView v;
        v.n = n;
        v.px = d_px;
        v.py = d_py;
        v.pz = d_pz;
        v.ell = d_ell;
        v.m = d_m;
        return v;
    }

    void upload(const std::vector<std::vector<host_comp>>& cfg, double int_tol = 1.0e-12)
    {
        release();

        if (cfg.size() < 5) {
            throw std::runtime_error("DeviceConfig::upload: config must have 5 rows");
        }

        const std::size_t N = cfg[0].size();
        for (int r = 1; r < 5; ++r) {
            if (cfg[r].size() != N) {
                throw std::runtime_error("DeviceConfig::upload: inconsistent config row sizes");
            }
        }

        n = static_cast<int>(N);
        if (n == 0) return;

        std::vector<double> h_px(N), h_py(N), h_pz(N);
        std::vector<int> h_ell(N), h_m(N);

        for (std::size_t i = 0; i < N; ++i) {
            h_px[i] = cfg[0][i].real();
            h_py[i] = cfg[1][i].real();
            h_pz[i] = cfg[2][i].real();

            if (std::abs(cfg[3][i].imag()) > int_tol ||
                std::abs(cfg[4][i].imag()) > int_tol) {
                throw std::runtime_error("DeviceConfig::upload: ell/m has nonzero imaginary part");
            }

            h_ell[i] = static_cast<int>(std::llround(cfg[3][i].real()));
            h_m[i]   = static_cast<int>(std::llround(cfg[4][i].real()));
        }

        K2GPU_CUDA_CHECK(cudaMalloc(&d_px,  N*sizeof(double)));
        K2GPU_CUDA_CHECK(cudaMalloc(&d_py,  N*sizeof(double)));
        K2GPU_CUDA_CHECK(cudaMalloc(&d_pz,  N*sizeof(double)));
        K2GPU_CUDA_CHECK(cudaMalloc(&d_ell, N*sizeof(int)));
        K2GPU_CUDA_CHECK(cudaMalloc(&d_m,   N*sizeof(int)));

        K2GPU_CUDA_CHECK(cudaMemcpy(d_px,  h_px.data(),  N*sizeof(double), cudaMemcpyHostToDevice));
        K2GPU_CUDA_CHECK(cudaMemcpy(d_py,  h_py.data(),  N*sizeof(double), cudaMemcpyHostToDevice));
        K2GPU_CUDA_CHECK(cudaMemcpy(d_pz,  h_pz.data(),  N*sizeof(double), cudaMemcpyHostToDevice));
        K2GPU_CUDA_CHECK(cudaMemcpy(d_ell, h_ell.data(), N*sizeof(int),    cudaMemcpyHostToDevice));
        K2GPU_CUDA_CHECK(cudaMemcpy(d_m,   h_m.data(),   N*sizeof(int),    cudaMemcpyHostToDevice));
    }

    void release()
    {
        if (d_px)  cudaFree(d_px);
        if (d_py)  cudaFree(d_py);
        if (d_pz)  cudaFree(d_pz);
        if (d_ell) cudaFree(d_ell);
        if (d_m)   cudaFree(d_m);
        d_px = d_py = d_pz = nullptr;
        d_ell = d_m = nullptr;
        n = 0;
    }

private:
    void move_from(DeviceConfig& other)
    {
        n = other.n;
        d_px = other.d_px;
        d_py = other.d_py;
        d_pz = other.d_pz;
        d_ell = other.d_ell;
        d_m = other.d_m;
        other.n = 0;
        other.d_px = other.d_py = other.d_pz = nullptr;
        other.d_ell = other.d_m = nullptr;
    }
};

class DeviceScatterParams
{
public:
    double* d_re = nullptr;
    double* d_im = nullptr;

    DeviceScatterParams() = default;
    DeviceScatterParams(const DeviceScatterParams&) = delete;
    DeviceScatterParams& operator=(const DeviceScatterParams&) = delete;
    ~DeviceScatterParams() { release(); }

    ScatterParamsView view() const
    {
        ScatterParamsView v;
        v.re = d_re;
        v.im = d_im;
        return v;
    }

    void upload(const std::vector<std::vector<host_comp>>& sp)
    {
        release();

        std::vector<double> h_re(12, 0.0), h_im(12, 0.0);
        for (int ell = 0; ell < 4; ++ell) {
            if (ell < static_cast<int>(sp.size())) {
                for (int j = 0; j < 3; ++j) {
                    if (j < static_cast<int>(sp[ell].size())) {
                        h_re[3*ell + j] = sp[ell][j].real();
                        h_im[3*ell + j] = sp[ell][j].imag();
                    }
                }
            }
        }

        K2GPU_CUDA_CHECK(cudaMalloc(&d_re, 12*sizeof(double)));
        K2GPU_CUDA_CHECK(cudaMalloc(&d_im, 12*sizeof(double)));
        K2GPU_CUDA_CHECK(cudaMemcpy(d_re, h_re.data(), 12*sizeof(double), cudaMemcpyHostToDevice));
        K2GPU_CUDA_CHECK(cudaMemcpy(d_im, h_im.data(), 12*sizeof(double), cudaMemcpyHostToDevice));
    }

    void release()
    {
        if (d_re) cudaFree(d_re);
        if (d_im) cudaFree(d_im);
        d_re = d_im = nullptr;
    }
};

__device__ inline Cx K2_inv_ERE_ang_mom_gpu(
    double eta_i,
    ScatterParamsView scatter_params,
    int ell_f,
    int proj_mf,
    int ell_i,
    int proj_mi,
    Cx sigma_i,
    double mj,
    double mk,
    double epsilon_h)
{
    const double pi = 3.141592653589793238462643383279502884;
    Cx zero(0.0, 0.0);

    if (ell_f != ell_i) return zero;
    if (proj_mf != proj_mi) return zero;

    const int ell = ell_f;
    if (ell < 0 || ell > 3) return zero;

    const Cx q2psq = q2psq_star_gpu(sigma_i, mj, mk);
    const Cx sqrt_sigma = sqrt_cx(sigma_i);
    const Cx A = eta_i / (8.0*pi*sqrt_sigma);
    const Cx C(abs_cx(sqrt_cx(q2psq)), 0.0);
    const Cx D = Cx(1.0, 0.0) - cutoff_function_1_gpu(sigma_i, mj, mk, epsilon_h);

    Cx out(0.0, 0.0);

    if (ell == 0) {
        const Cx a0 = scatter_params.get(0, 0);
        const Cx r0 = scatter_params.get(0, 1);
        const Cx P0 = scatter_params.get(0, 2);
        const Cx qcotdel = -1.0/a0 + 0.5*r0*q2psq + P0*q2psq*q2psq;
        out = A * (qcotdel + C*D);
    }
    else if (ell == 1) {
        const Cx a1 = scatter_params.get(1, 0);
        const Cx r1 = scatter_params.get(1, 1);
        const Cx P1 = scatter_params.get(1, 2);
        const Cx qcotdel = -1.0/(a1*q2psq) + 0.5*r1 + P1*q2psq;
        out = A * (qcotdel + q2psq*C*D);
    }
    else if (ell == 2) {
        const Cx a2 = scatter_params.get(2, 0);
        const Cx r2 = scatter_params.get(2, 1);
        const Cx P2 = scatter_params.get(2, 2);
        const Cx qcotdel = -1.0/(a2*q2psq*q2psq) + 0.5*r2/q2psq + P2;
        out = A * (qcotdel + C*D);
    }
    else if (ell == 3) {
        const Cx a3 = scatter_params.get(3, 0);
        const Cx r3 = scatter_params.get(3, 1);
        const Cx P3 = scatter_params.get(3, 2);
        const Cx qcotdel = -1.0/(a3*q2psq*q2psq*q2psq) + 0.5*r3/(q2psq*q2psq) + P3/q2psq;
        out = A * (qcotdel + C*D);
    }

    return out;
}

__device__ inline Cx K2inv_diag_element_gpu(
    double En,
    ConfigView cfg,
    int i,
    double Px,
    double Py,
    double Pz,
    double eta_i,
    ScatterParamsView scatter_params,
    double mi,
    double mj,
    double mk,
    double epsilon_h,
    double L)
{
    const double kx = cfg.px[i];
    const double ky = cfg.py[i];
    const double kz = cfg.pz[i];

    const int ell = cfg.ell[i];
    const int m = cfg.m[i];

    const Cx spec_k = sqrt_cx(Cx(kx*kx + ky*ky + kz*kz, 0.0));

    const double Pmkx = Px - kx;
    const double Pmky = Py - ky;
    const double Pmkz = Pz - kz;
    const double Pminksq = Pmkx*Pmkx + Pmky*Pmky + Pmkz*Pmkz;

    const Cx omega_k = omega_func_gpu(spec_k, mi);
    const Cx sig_k = (Cx(En, 0.0) - omega_k)*(Cx(En, 0.0) - omega_k) - Pminksq;

    const Cx K2_inv_val = K2_inv_ERE_ang_mom_gpu(
        eta_i,
        scatter_params,
        ell,
        m,
        ell,
        m,
        sig_k,
        mj,
        mk,
        epsilon_h);

    const Cx constval = 1.0 / (2.0*omega_k*L*L*L);
    return constval * K2_inv_val;
}

__global__ void build_K2inv_2plus1_kernel(
    double En,
    ConfigView plm,
    ConfigView klm,
    double Px,
    double Py,
    double Pz,
    double eta_1,
    double eta_2,
    ScatterParamsView sp1,
    ScatterParamsView sp2,
    double m1,
    double m2,
    double epsilon_h,
    double L,
    Cx* out,
    int total_dim)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elems = total_dim * total_dim;
    if (tid >= total_elems) return;

    const int row = tid / total_dim;
    const int col = tid - row * total_dim;
    const int size1 = plm.n;
    const int size2 = klm.n;

    Cx val(0.0, 0.0);

    // Upper-left flavor-1 block: K2inv_1, diagonal only.
    if (row < size1 && col < size1) {
        if (row == col) {
            val = K2inv_diag_element_gpu(
                En, plm, row,
                Px, Py, Pz,
                eta_1, sp1,
                m1, m1, m2,
                epsilon_h, L);
        }
    }
    // Lower-right flavor-2 block: 2*K2inv_2, diagonal only.
    else if (row >= size1 && col >= size1) {
        const int i2 = row - size1;
        const int j2 = col - size1;
        if (i2 < size2 && j2 < size2 && i2 == j2) {
            val = 2.0 * K2inv_diag_element_gpu(
                En, klm, i2,
                Px, Py, Pz,
                eta_2, sp2,
                m2, m1, m1,
                epsilon_h, L);
        }
    }
    // Off-diagonal flavor blocks remain zero.

    out[tid] = val;
}

struct Options
{
    int threads_per_block = 256;
    char debug = 'n';
};

inline void K2inv_EREord2_2plus1_mat_gpu_safe(
    Eigen::MatrixXcd& K2inv,
    double eta_1,
    double eta_2,
    const std::vector<std::vector<host_comp>>& scatter_params_1,
    const std::vector<std::vector<host_comp>>& scatter_params_2,
    host_comp En,
    const std::vector<std::vector<host_comp>>& plm_config,
    const std::vector<std::vector<host_comp>>& klm_config,
    const std::vector<host_comp>& total_P,
    double m1,
    double m2,
    double epsilon_h,
    double L,
    const Options& opt = Options())
{
    if (total_P.size() < 3) {
        throw std::runtime_error("K2 GPU: total_P must have 3 elements");
    }

    DeviceConfig d_plm;
    DeviceConfig d_klm;
    DeviceScatterParams d_sp1;
    DeviceScatterParams d_sp2;

    d_plm.upload(plm_config);
    d_klm.upload(klm_config);
    d_sp1.upload(scatter_params_1);
    d_sp2.upload(scatter_params_2);

    const int size1 = d_plm.n;
    const int size2 = d_klm.n;
    const int total_dim = size1 + size2;

    K2inv.resize(total_dim, total_dim);
    if (total_dim == 0) {
        return;
    }

    const int total_elems = total_dim * total_dim;
    Cx* d_out = nullptr;
    K2GPU_CUDA_CHECK(cudaMalloc(&d_out, static_cast<std::size_t>(total_elems)*sizeof(Cx)));

    const int threads = opt.threads_per_block > 0 ? opt.threads_per_block : 256;
    const int blocks = (total_elems + threads - 1) / threads;

    if (opt.debug == 'y') {
        std::cout << "[K2 GPU launch] size1=" << size1
                  << " size2=" << size2
                  << " total_dim=" << total_dim
                  << " total_elems=" << total_elems
                  << " blocks=" << blocks
                  << " threads=" << threads
                  << " En=" << std::setprecision(17) << En.real()
                  << std::endl;
    }

    build_K2inv_2plus1_kernel<<<blocks, threads>>>(
        En.real(),
        d_plm.view(),
        d_klm.view(),
        total_P[0].real(), total_P[1].real(), total_P[2].real(),
        eta_1,
        eta_2,
        d_sp1.view(),
        d_sp2.view(),
        m1,
        m2,
        epsilon_h,
        L,
        d_out,
        total_dim);

    K2GPU_CUDA_CHECK(cudaPeekAtLastError());
    K2GPU_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<Cx> h_out(static_cast<std::size_t>(total_elems));
    K2GPU_CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                                static_cast<std::size_t>(total_elems)*sizeof(Cx),
                                cudaMemcpyDeviceToHost));

    cudaFree(d_out);

    for (int i = 0; i < total_dim; ++i) {
        for (int j = 0; j < total_dim; ++j) {
            const Cx z = h_out[static_cast<std::size_t>(i*total_dim + j)];
            K2inv(i, j) = host_comp(z.re, z.im);
        }
    }
}

inline void K2inv_EREord2_2plus1_many_mat_gpu_safe(
    std::vector<Eigen::MatrixXcd>& K2inv_vec,
    const std::vector<host_comp>& En_vec,
    const std::vector<std::vector<std::vector<host_comp>>>& plm_vec,
    const std::vector<std::vector<std::vector<host_comp>>>& klm_vec,
    double eta_1,
    double eta_2,
    const std::vector<std::vector<host_comp>>& scatter_params_1,
    const std::vector<std::vector<host_comp>>& scatter_params_2,
    const std::vector<host_comp>& total_P,
    double m1,
    double m2,
    double epsilon_h,
    double L,
    const Options& opt = Options())
{
    const std::size_t N = En_vec.size();
    if (plm_vec.size() != N || klm_vec.size() != N) {
        throw std::runtime_error("K2 many GPU: En_vec/plm_vec/klm_vec size mismatch");
    }

    K2inv_vec.resize(N);
    for (std::size_t i = 0; i < N; ++i) {
        K2inv_EREord2_2plus1_mat_gpu_safe(
            K2inv_vec[i],
            eta_1,
            eta_2,
            scatter_params_1,
            scatter_params_2,
            En_vec[i],
            plm_vec[i],
            klm_vec[i],
            total_P,
            m1,
            m2,
            epsilon_h,
            L,
            opt);
    }
}

struct MatrixCompareStats
{
    double max_abs = 0.0;
    double max_rel = 0.0;
    double frob_abs = 0.0;
    double frob_ref = 0.0;
    double frob_rel = 0.0;
    int max_i = -1;
    int max_j = -1;
    int n_bad_abs = 0;
    int n_bad_rel = 0;
    int n_nan_inf = 0;
};

inline MatrixCompareStats compare_matrices(
    const Eigen::MatrixXcd& A,
    const Eigen::MatrixXcd& B,
    double abs_tol = 1.0e-10,
    double rel_tol = 1.0e-10)
{
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::runtime_error("compare_matrices: shape mismatch");
    }

    MatrixCompareStats st;
    long double frob_diff2 = 0.0L;
    long double frob_ref2 = 0.0L;

    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            const host_comp a = A(i,j);
            const host_comp b = B(i,j);
            const double adiff = std::abs(a - b);
            const double aref = std::max(1.0, std::abs(a));
            const double rdiff = adiff / aref;

            if (!std::isfinite(a.real()) || !std::isfinite(a.imag()) ||
                !std::isfinite(b.real()) || !std::isfinite(b.imag())) {
                st.n_nan_inf++;
            }

            if (adiff > st.max_abs) {
                st.max_abs = adiff;
                st.max_rel = rdiff;
                st.max_i = i;
                st.max_j = j;
            }
            if (adiff > abs_tol) st.n_bad_abs++;
            if (rdiff > rel_tol) st.n_bad_rel++;

            frob_diff2 += static_cast<long double>(std::norm(a - b));
            frob_ref2 += static_cast<long double>(std::norm(a));
        }
    }

    st.frob_abs = std::sqrt(static_cast<double>(frob_diff2));
    st.frob_ref = std::sqrt(static_cast<double>(frob_ref2));
    st.frob_rel = st.frob_abs / std::max(1.0, st.frob_ref);
    return st;
}

} // namespace k2gpu

#endif // K2_FUNCTIONS_GPU_SAFE_CUH
