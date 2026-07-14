#ifndef K2_GPU_SAFE_BUILDER_CUH
#define K2_GPU_SAFE_BUILDER_CUH

#include <cuda_runtime.h>
#include <Eigen/Dense>

#include <complex>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <limits>

namespace k2gpu
{

using host_comp = std::complex<double>;

#define K2GPU_CUDA_CHECK(call)                                                       \
    do {                                                                             \
        cudaError_t _err = (call);                                                   \
        if (_err != cudaSuccess) {                                                   \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ +     \
                                     ":" + std::to_string(__LINE__) + " : " +       \
                                     cudaGetErrorString(_err));                      \
        }                                                                            \
    } while (0)

struct CudaComplex
{
    double re;
    double im;

    __host__ __device__ CudaComplex() : re(0.0), im(0.0) {}
    __host__ __device__ CudaComplex(double r, double i = 0.0) : re(r), im(i) {}
};

__host__ __device__ inline CudaComplex make_c(double r, double i = 0.0)
{
    return CudaComplex(r, i);
}

__host__ __device__ inline CudaComplex operator+(CudaComplex a, CudaComplex b)
{
    return make_c(a.re + b.re, a.im + b.im);
}

__host__ __device__ inline CudaComplex operator-(CudaComplex a, CudaComplex b)
{
    return make_c(a.re - b.re, a.im - b.im);
}

__host__ __device__ inline CudaComplex operator-(CudaComplex a)
{
    return make_c(-a.re, -a.im);
}

__host__ __device__ inline CudaComplex operator*(CudaComplex a, CudaComplex b)
{
    return make_c(a.re * b.re - a.im * b.im,
                  a.re * b.im + a.im * b.re);
}

__host__ __device__ inline CudaComplex operator/(CudaComplex a, CudaComplex b)
{
    const double den = b.re * b.re + b.im * b.im;
    return make_c((a.re * b.re + a.im * b.im) / den,
                  (a.im * b.re - a.re * b.im) / den);
}

__host__ __device__ inline CudaComplex operator+(CudaComplex a, double b)
{
    return make_c(a.re + b, a.im);
}

__host__ __device__ inline CudaComplex operator+(double a, CudaComplex b)
{
    return make_c(a + b.re, b.im);
}

__host__ __device__ inline CudaComplex operator-(CudaComplex a, double b)
{
    return make_c(a.re - b, a.im);
}

__host__ __device__ inline CudaComplex operator-(double a, CudaComplex b)
{
    return make_c(a - b.re, -b.im);
}

__host__ __device__ inline CudaComplex operator*(CudaComplex a, double b)
{
    return make_c(a.re * b, a.im * b);
}

__host__ __device__ inline CudaComplex operator*(double a, CudaComplex b)
{
    return make_c(a * b.re, a * b.im);
}

__host__ __device__ inline CudaComplex operator/(CudaComplex a, double b)
{
    return make_c(a.re / b, a.im / b);
}

__host__ __device__ inline CudaComplex operator/(double a, CudaComplex b)
{
    return make_c(a, 0.0) / b;
}

__host__ __device__ inline double abs_c(CudaComplex z)
{
    return hypot(z.re, z.im);
}

__host__ __device__ inline CudaComplex sqrt_c(CudaComplex z)
{
    const double r = hypot(z.re, z.im);
    const double u = sqrt(0.5 * (r + z.re));
    const double v_abs = sqrt(fmax(0.0, 0.5 * (r - z.re)));
    const double v = (z.im >= 0.0) ? v_abs : -v_abs;
    return make_c(u, v);
}

__host__ __device__ inline CudaComplex exp_c(CudaComplex z)
{
    const double e = exp(z.re);
    return make_c(e * cos(z.im), e * sin(z.im));
}

__host__ __device__ inline CudaComplex pow_int_c(CudaComplex z, int n)
{
    if (n == 0) return make_c(1.0, 0.0);
    CudaComplex out = make_c(1.0, 0.0);
    for (int i = 0; i < n; ++i) out = out * z;
    return out;
}

__host__ __device__ inline CudaComplex omega_func_dev(CudaComplex p, double m)
{
    return sqrt_c(p * p + m * m);
}

__host__ __device__ inline CudaComplex kallen_dev(CudaComplex x, CudaComplex y, CudaComplex z)
{
    return x * x + y * y + z * z - 2.0 * (x * y + y * z + z * x);
}

__host__ __device__ inline CudaComplex q2psq_star_dev(CudaComplex sigma_i,
                                                       double mj,
                                                       double mk)
{
    const CudaComplex mj2 = make_c(mj * mj, 0.0);
    const CudaComplex mk2 = make_c(mk * mk, 0.0);
    return kallen_dev(sigma_i, mj2, mk2) / (4.0 * sigma_i);
}

__host__ __device__ inline CudaComplex Jfunc_dev(CudaComplex z)
{
    if (z.re <= 0.0) {
        return make_c(0.0, 0.0);
    } else if (z.re > 0.0 && z.re < 1.0) {
        CudaComplex A = -1.0 / z;
        CudaComplex B = exp_c(-1.0 / (1.0 - z));
        return exp_c(A * B);
    } else {
        return make_c(1.0, 0.0);
    }
}

__host__ __device__ inline CudaComplex cutoff_function_1_dev(CudaComplex sigma_i,
                                                             double mj,
                                                             double mk,
                                                             double epsilon_h)
{
    const double mj2 = mj * mj;
    const double mk2 = mk * mk;

    if (mj == mk && epsilon_h == 0.0) {
        CudaComplex Z = sigma_i / (4.0 * mj2);
        return Jfunc_dev(Z);
    } else {
        const double denom = (mj + mk) * (mj + mk) - fabs(mj2 - mk2);
        CudaComplex Z = ((1.0 + epsilon_h) * (sigma_i - fabs(mj2 - mk2))) / denom;
        return Jfunc_dev(Z);
    }
}

__host__ __device__ inline CudaComplex sigma_pvec_based_dev(double En,
                                                            double px,
                                                            double py,
                                                            double pz,
                                                            double mi,
                                                            double Px,
                                                            double Py,
                                                            double Pz)
{
    const double psq = px * px + py * py + pz * pz;
    CudaComplex spec_p = sqrt_c(make_c(psq, 0.0));

    const double Pminusp_x = Px - px;
    const double Pminusp_y = Py - py;
    const double Pminusp_z = Pz - pz;
    const double Pminusp_sq = Pminusp_x * Pminusp_x +
                              Pminusp_y * Pminusp_y +
                              Pminusp_z * Pminusp_z;

    CudaComplex A = make_c(En, 0.0) - omega_func_dev(spec_p, mi);
    return A * A - Pminusp_sq;
}

__host__ __device__ inline CudaComplex K2_inv_ERE_ang_mom_dev(
    double eta_i,
    const CudaComplex* scatter_params_flat, // 4 x 3, row-major: ell*3 + par
    int ell_f,
    int proj_mf,
    int ell_i,
    int proj_mi,
    CudaComplex sigma_i,
    double mj,
    double mk,
    double epsilon_h)
{
    const double pi = 3.141592653589793238462643383279502884;
    const CudaComplex zero = make_c(0.0, 0.0);

    if (ell_f != ell_i) return zero;
    if (proj_mf != proj_mi) return zero;

    const int ell = ell_f;
    if (ell < 0 || ell > 3) return zero;

    CudaComplex A = eta_i / (8.0 * pi * sqrt_c(sigma_i));
    CudaComplex q2psq = q2psq_star_dev(sigma_i, mj, mk);
    CudaComplex sqrt_q2 = sqrt_c(q2psq);
    CudaComplex C = make_c(abs_c(sqrt_q2), 0.0);
    CudaComplex D = make_c(1.0, 0.0) - cutoff_function_1_dev(sigma_i, mj, mk, epsilon_h);

    const CudaComplex a = scatter_params_flat[ell * 3 + 0];
    const CudaComplex r = scatter_params_flat[ell * 3 + 1];
    const CudaComplex P = scatter_params_flat[ell * 3 + 2];

    CudaComplex B = zero;
    CudaComplex out = zero;

    if (ell == 0) {
        B = -1.0 / a + 0.5 * r * q2psq + P * q2psq * q2psq;
        out = A * (B + C * D);
    } else if (ell == 1) {
        B = -1.0 / (a * q2psq) + 0.5 * r + P * q2psq;
        out = A * (B + q2psq * C * D);
    } else if (ell == 2) {
        B = -1.0 / (a * q2psq * q2psq) + 0.5 * r / q2psq + P;
        out = A * (B + C * D);
    } else if (ell == 3) {
        B = -1.0 / (a * q2psq * q2psq * q2psq)
            + 0.5 * r / (q2psq * q2psq)
            + P / q2psq;
        out = A * (B + C * D);
    }

    return out;
}

struct FlatConfigHost
{
    int n = 0;
    std::vector<double> px, py, pz;
    std::vector<int> ell, m;
};

inline FlatConfigHost flatten_config(const std::vector<std::vector<host_comp>>& cfg,
                                     const std::string& name)
{
    if (cfg.size() < 5) {
        throw std::runtime_error(name + ": config must have 5 rows");
    }

    const int n = static_cast<int>(cfg[0].size());
    for (int r = 1; r < 5; ++r) {
        if (static_cast<int>(cfg[r].size()) != n) {
            throw std::runtime_error(name + ": inconsistent config row sizes");
        }
    }

    FlatConfigHost out;
    out.n = n;
    out.px.resize(n);
    out.py.resize(n);
    out.pz.resize(n);
    out.ell.resize(n);
    out.m.resize(n);

    for (int i = 0; i < n; ++i) {
        out.px[static_cast<std::size_t>(i)] = cfg[0][static_cast<std::size_t>(i)].real();
        out.py[static_cast<std::size_t>(i)] = cfg[1][static_cast<std::size_t>(i)].real();
        out.pz[static_cast<std::size_t>(i)] = cfg[2][static_cast<std::size_t>(i)].real();
        out.ell[static_cast<std::size_t>(i)] = static_cast<int>(std::llround(cfg[3][static_cast<std::size_t>(i)].real()));
        out.m[static_cast<std::size_t>(i)] = static_cast<int>(std::llround(cfg[4][static_cast<std::size_t>(i)].real()));
    }

    return out;
}

inline std::vector<CudaComplex>
flatten_scatter_params(const std::vector<std::vector<host_comp>>& scatter_params)
{
    std::vector<CudaComplex> flat(12, make_c(0.0, 0.0));

    const int rows = std::min<int>(4, static_cast<int>(scatter_params.size()));
    for (int ell = 0; ell < rows; ++ell) {
        const int cols = std::min<int>(3, static_cast<int>(scatter_params[static_cast<std::size_t>(ell)].size()));
        for (int p = 0; p < cols; ++p) {
            const host_comp z = scatter_params[static_cast<std::size_t>(ell)][static_cast<std::size_t>(p)];
            flat[static_cast<std::size_t>(ell * 3 + p)] = make_c(z.real(), z.imag());
        }
    }

    return flat;
}

struct DeviceConfig
{
    int n = 0;
    double* px = nullptr;
    double* py = nullptr;
    double* pz = nullptr;
    int* ell = nullptr;
    int* m = nullptr;

    void allocate_and_copy(const FlatConfigHost& h)
    {
        n = h.n;
        if (n <= 0) return;

        K2GPU_CUDA_CHECK(cudaMalloc(&px, sizeof(double) * static_cast<std::size_t>(n)));
        K2GPU_CUDA_CHECK(cudaMalloc(&py, sizeof(double) * static_cast<std::size_t>(n)));
        K2GPU_CUDA_CHECK(cudaMalloc(&pz, sizeof(double) * static_cast<std::size_t>(n)));
        K2GPU_CUDA_CHECK(cudaMalloc(&ell, sizeof(int) * static_cast<std::size_t>(n)));
        K2GPU_CUDA_CHECK(cudaMalloc(&m, sizeof(int) * static_cast<std::size_t>(n)));

        K2GPU_CUDA_CHECK(cudaMemcpy(px, h.px.data(), sizeof(double) * static_cast<std::size_t>(n), cudaMemcpyHostToDevice));
        K2GPU_CUDA_CHECK(cudaMemcpy(py, h.py.data(), sizeof(double) * static_cast<std::size_t>(n), cudaMemcpyHostToDevice));
        K2GPU_CUDA_CHECK(cudaMemcpy(pz, h.pz.data(), sizeof(double) * static_cast<std::size_t>(n), cudaMemcpyHostToDevice));
        K2GPU_CUDA_CHECK(cudaMemcpy(ell, h.ell.data(), sizeof(int) * static_cast<std::size_t>(n), cudaMemcpyHostToDevice));
        K2GPU_CUDA_CHECK(cudaMemcpy(m, h.m.data(), sizeof(int) * static_cast<std::size_t>(n), cudaMemcpyHostToDevice));
    }

    void release()
    {
        if (px) cudaFree(px);
        if (py) cudaFree(py);
        if (pz) cudaFree(pz);
        if (ell) cudaFree(ell);
        if (m) cudaFree(m);
        px = py = pz = nullptr;
        ell = m = nullptr;
        n = 0;
    }

    ~DeviceConfig()
    {
        release();
    }
};

struct K2GpuOptions
{
    int threads_per_block = 256;
    char debug = 'n';
};

__global__ void k2_fill_full_kernel(CudaComplex* K2,
                                    int total_dim,
                                    DeviceConfig plm,
                                    DeviceConfig klm,
                                    const CudaComplex* scatter1,
                                    const CudaComplex* scatter2,
                                    double eta1,
                                    double eta2,
                                    double En,
                                    double Px,
                                    double Py,
                                    double Pz,
                                    double m1,
                                    double m2,
                                    double epsilon_h,
                                    double L)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_total = total_dim * total_dim;
    if (tid >= n_total) return;

    const int row = tid / total_dim;
    const int col = tid - row * total_dim;

    CudaComplex val = make_c(0.0, 0.0);

    const int size1 = plm.n;
    const int size2 = klm.n;

    // Block 11: K2inv_1, masses mi=m1, mj=m1, mk=m2, eta1
    if (row < size1 && col < size1) {
        if (row == col) {
            const double px = plm.px[row];
            const double py = plm.py[row];
            const double pz = plm.pz[row];
            const int ell_f = plm.ell[row];
            const int mf = plm.m[row];
            const int ell_i = plm.ell[col];
            const int mi_m = plm.m[col];

            const double psq = px * px + py * py + pz * pz;
            CudaComplex spec_p = sqrt_c(make_c(psq, 0.0));
            CudaComplex sigma_i = sigma_pvec_based_dev(En, px, py, pz, m1, Px, Py, Pz);
            CudaComplex K2val = K2_inv_ERE_ang_mom_dev(eta1, scatter1,
                                                        ell_f, mf, ell_i, mi_m,
                                                        sigma_i, m1, m2, epsilon_h);
            CudaComplex omega_p = omega_func_dev(spec_p, m1);
            CudaComplex constval = 1.0 / (2.0 * omega_p * L * L * L);
            val = constval * K2val;
        }
    }
    // Block 22: 2 * K2inv_2, masses mi=m2, mj=m1, mk=m1, eta2
    else if (row >= size1 && col >= size1) {
        const int r2 = row - size1;
        const int c2 = col - size1;
        if (r2 < size2 && c2 < size2 && r2 == c2) {
            const double px = klm.px[r2];
            const double py = klm.py[r2];
            const double pz = klm.pz[r2];
            const int ell_f = klm.ell[r2];
            const int mf = klm.m[r2];
            const int ell_i = klm.ell[c2];
            const int mi_m = klm.m[c2];

            const double psq = px * px + py * py + pz * pz;
            CudaComplex spec_p = sqrt_c(make_c(psq, 0.0));
            CudaComplex sigma_i = sigma_pvec_based_dev(En, px, py, pz, m2, Px, Py, Pz);
            CudaComplex K2val = K2_inv_ERE_ang_mom_dev(eta2, scatter2,
                                                        ell_f, mf, ell_i, mi_m,
                                                        sigma_i, m1, m1, epsilon_h);
            CudaComplex omega_p = omega_func_dev(spec_p, m2);
            CudaComplex constval = 1.0 / (2.0 * omega_p * L * L * L);
            val = 2.0 * constval * K2val;
        }
    }

    // Eigen is column-major by default, so store column-major.
    K2[row + col * total_dim] = val;
}

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
    const K2GpuOptions& opt = K2GpuOptions())
{
    if (total_P.size() < 3) {
        throw std::runtime_error("K2 GPU: total_P must have size 3");
    }

    FlatConfigHost h_plm = flatten_config(plm_config, "plm_config");
    FlatConfigHost h_klm = flatten_config(klm_config, "klm_config");

    const int size1 = h_plm.n;
    const int size2 = h_klm.n;
    const int total_dim = size1 + size2;

    K2inv.resize(total_dim, total_dim);
    if (total_dim == 0) return;

    DeviceConfig d_plm;
    DeviceConfig d_klm;
    d_plm.allocate_and_copy(h_plm);
    d_klm.allocate_and_copy(h_klm);

    std::vector<CudaComplex> h_scatter1 = flatten_scatter_params(scatter_params_1);
    std::vector<CudaComplex> h_scatter2 = flatten_scatter_params(scatter_params_2);

    CudaComplex* d_scatter1 = nullptr;
    CudaComplex* d_scatter2 = nullptr;
    CudaComplex* d_K2 = nullptr;

    K2GPU_CUDA_CHECK(cudaMalloc(&d_scatter1, sizeof(CudaComplex) * h_scatter1.size()));
    K2GPU_CUDA_CHECK(cudaMalloc(&d_scatter2, sizeof(CudaComplex) * h_scatter2.size()));
    K2GPU_CUDA_CHECK(cudaMemcpy(d_scatter1, h_scatter1.data(), sizeof(CudaComplex) * h_scatter1.size(), cudaMemcpyHostToDevice));
    K2GPU_CUDA_CHECK(cudaMemcpy(d_scatter2, h_scatter2.data(), sizeof(CudaComplex) * h_scatter2.size(), cudaMemcpyHostToDevice));

    const std::size_t matrix_size = static_cast<std::size_t>(total_dim) * static_cast<std::size_t>(total_dim);
    K2GPU_CUDA_CHECK(cudaMalloc(&d_K2, sizeof(CudaComplex) * matrix_size));

    const int threads = opt.threads_per_block > 0 ? opt.threads_per_block : 256;
    const int blocks = static_cast<int>((matrix_size + static_cast<std::size_t>(threads) - 1) / static_cast<std::size_t>(threads));

    k2_fill_full_kernel<<<blocks, threads>>>(
        d_K2,
        total_dim,
        d_plm,
        d_klm,
        d_scatter1,
        d_scatter2,
        eta_1,
        eta_2,
        En.real(),
        total_P[0].real(),
        total_P[1].real(),
        total_P[2].real(),
        m1,
        m2,
        epsilon_h,
        L
    );

    K2GPU_CUDA_CHECK(cudaGetLastError());
    K2GPU_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<CudaComplex> h_K2(matrix_size);
    K2GPU_CUDA_CHECK(cudaMemcpy(h_K2.data(), d_K2, sizeof(CudaComplex) * matrix_size, cudaMemcpyDeviceToHost));

    for (int col = 0; col < total_dim; ++col) {
        for (int row = 0; row < total_dim; ++row) {
            const CudaComplex z = h_K2[static_cast<std::size_t>(row + col * total_dim)];
            K2inv(row, col) = host_comp(z.re, z.im);
        }
    }

    if (d_scatter1) cudaFree(d_scatter1);
    if (d_scatter2) cudaFree(d_scatter2);
    if (d_K2) cudaFree(d_K2);

    if (opt.debug == 'y') {
        std::cout << "[K2 GPU] built K2inv: size1=" << size1
                  << " size2=" << size2
                  << " total_dim=" << total_dim
                  << " matrix_size=" << matrix_size
                  << std::endl;
    }
}

inline void K2inv_EREord2_2plus1_many_mat_gpu_safe(
    std::vector<Eigen::MatrixXcd>& K2inv_vec,
    const std::vector<host_comp>& En_vec,
    double eta_1,
    double eta_2,
    const std::vector<std::vector<host_comp>>& scatter_params_1,
    const std::vector<std::vector<host_comp>>& scatter_params_2,
    const std::vector<std::vector<std::vector<host_comp>>>& plm_vec,
    const std::vector<std::vector<std::vector<host_comp>>>& klm_vec,
    const std::vector<host_comp>& total_P,
    double m1,
    double m2,
    double epsilon_h,
    double L,
    const K2GpuOptions& opt = K2GpuOptions())
{
    const std::size_t N = En_vec.size();
    if (plm_vec.size() != N || klm_vec.size() != N) {
        throw std::runtime_error("K2 many GPU: En/plm/klm size mismatch");
    }

    K2inv_vec.assign(N, Eigen::MatrixXcd());

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
            opt
        );
    }
}

struct CompareResult
{
    double max_abs = 0.0;
    double max_rel = 0.0;
    double frob_abs = 0.0;
    double frob_rel = 0.0;
    int max_i = -1;
    int max_j = -1;
    int n_bad_abs = 0;
    int n_bad_rel = 0;
    int n_nan_inf = 0;
    bool shape_match = true;
};

inline CompareResult compare_matrices(const Eigen::MatrixXcd& A,
                                      const Eigen::MatrixXcd& B,
                                      double abs_tol = 1.0e-10,
                                      double rel_tol = 1.0e-8)
{
    CompareResult r;

    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        r.shape_match = false;
        return r;
    }

    long double diff2 = 0.0L;
    long double norm2 = 0.0L;

    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            const host_comp a = A(i, j);
            const host_comp b = B(i, j);
            const host_comp d = a - b;
            const double ad = std::abs(d);
            const double scale = std::max(1.0, std::abs(a));
            const double rd = ad / scale;

            if (!std::isfinite(a.real()) || !std::isfinite(a.imag()) ||
                !std::isfinite(b.real()) || !std::isfinite(b.imag())) {
                r.n_nan_inf++;
            }

            if (ad > r.max_abs) {
                r.max_abs = ad;
                r.max_rel = rd;
                r.max_i = i;
                r.max_j = j;
            }

            if (ad > abs_tol) r.n_bad_abs++;
            if (rd > rel_tol) r.n_bad_rel++;

            diff2 += static_cast<long double>(std::norm(d));
            norm2 += static_cast<long double>(std::norm(a));
        }
    }

    r.frob_abs = std::sqrt(static_cast<double>(diff2));
    r.frob_rel = r.frob_abs / std::max(1.0, std::sqrt(static_cast<double>(norm2)));

    return r;
}

} // namespace k2gpu

#endif // K2_GPU_SAFE_BUILDER_CUH
