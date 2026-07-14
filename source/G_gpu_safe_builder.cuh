#ifndef G_GPU_SAFE_BUILDER_CUH
#define G_GPU_SAFE_BUILDER_CUH

#include <cuda_runtime.h>
#include <Eigen/Dense>
#include <vector>
#include <complex>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <limits>
#include <algorithm>

namespace ggpu {

struct GpuComplex {
    double x; // real
    double y; // imag
};

struct ConfigEntry {
    double px;
    double py;
    double pz;
    int ell;
    int m;
};

struct Vec3d {
    double x;
    double y;
    double z;
};

struct GGpuOptions {
    int threads_per_block = 256;
    char debug = 'n';
};

#define GGPU_CUDA_CHECK(call)                                                     \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ +  \
                                     ":" + std::to_string(__LINE__) + " : " +   \
                                     cudaGetErrorString(err__));                  \
        }                                                                         \
    } while (0)

static inline std::vector<ConfigEntry>
flatten_config_host(const std::vector<std::vector<std::complex<double>>>& cfg)
{
    if (cfg.size() < 5) {
        throw std::runtime_error("flatten_config_host: config must have 5 rows");
    }

    const std::size_t N = cfg[0].size();
    for (std::size_t r = 1; r < 5; ++r) {
        if (cfg[r].size() != N) {
            throw std::runtime_error("flatten_config_host: inconsistent config row sizes");
        }
    }

    std::vector<ConfigEntry> out(N);
    for (std::size_t i = 0; i < N; ++i) {
        out[i].px = cfg[0][i].real();
        out[i].py = cfg[1][i].real();
        out[i].pz = cfg[2][i].real();
        out[i].ell = static_cast<int>(std::llround(cfg[3][i].real()));
        out[i].m   = static_cast<int>(std::llround(cfg[4][i].real()));
    }
    return out;
}

__host__ __device__ inline GpuComplex make_c(double a, double b = 0.0) { return {a, b}; }
__host__ __device__ inline GpuComplex cadd(GpuComplex a, GpuComplex b) { return {a.x + b.x, a.y + b.y}; }
__host__ __device__ inline GpuComplex csub(GpuComplex a, GpuComplex b) { return {a.x - b.x, a.y - b.y}; }
__host__ __device__ inline GpuComplex cneg(GpuComplex a) { return {-a.x, -a.y}; }
__host__ __device__ inline GpuComplex cmul(GpuComplex a, GpuComplex b) { return {a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x}; }
__host__ __device__ inline GpuComplex cmul_d(GpuComplex a, double b) { return {a.x*b, a.y*b}; }
__host__ __device__ inline GpuComplex cdiv(GpuComplex a, GpuComplex b) {
    double den = b.x*b.x + b.y*b.y;
    return {(a.x*b.x + a.y*b.y)/den, (a.y*b.x - a.x*b.y)/den};
}
__host__ __device__ inline GpuComplex cdiv_d(GpuComplex a, double b) { return {a.x/b, a.y/b}; }
__host__ __device__ inline double cabs2(GpuComplex a) { return a.x*a.x + a.y*a.y; }
__host__ __device__ inline double cabs(GpuComplex a) { return sqrt(cabs2(a)); }

__host__ __device__ inline GpuComplex csqrt_c(GpuComplex z) {
    if (z.y == 0.0) {
        if (z.x >= 0.0) return {sqrt(z.x), 0.0};
        return {0.0, sqrt(-z.x)};
    }
    double r = hypot(z.x, z.y);
    double u = sqrt(0.5 * (r + z.x));
    double v = copysign(sqrt(0.5 * (r - z.x)), z.y);
    return {u, v};
}

__host__ __device__ inline GpuComplex cexp_c(GpuComplex z) {
    double e = exp(z.x);
    return {e*cos(z.y), e*sin(z.y)};
}

__host__ __device__ inline GpuComplex cpow_int(GpuComplex a, int n) {
    if (n == 0) return {1.0, 0.0};
    if (n < 0) return cdiv(make_c(1.0), cpow_int(a, -n));
    GpuComplex out = {1.0, 0.0};
    GpuComplex base = a;
    int p = n;
    while (p > 0) {
        if (p & 1) out = cmul(out, base);
        base = cmul(base, base);
        p >>= 1;
    }
    return out;
}

__host__ __device__ inline GpuComplex omega_gpu(GpuComplex p, double m) {
    return csqrt_c(cadd(cmul(p, p), make_c(m*m)));
}

__host__ __device__ inline GpuComplex kallen_gpu(GpuComplex x, GpuComplex y, GpuComplex z) {
    GpuComplex x2 = cmul(x, x);
    GpuComplex y2 = cmul(y, y);
    GpuComplex z2 = cmul(z, z);
    GpuComplex xy = cmul(x, y);
    GpuComplex yz = cmul(y, z);
    GpuComplex zx = cmul(z, x);
    return csub(cadd(cadd(x2, y2), z2), cmul_d(cadd(cadd(xy, yz), zx), 2.0));
}

__host__ __device__ inline GpuComplex q2psq_star_gpu(GpuComplex sigma_i, double mj, double mk) {
    GpuComplex y = make_c(mj*mj);
    GpuComplex z = make_c(mk*mk);
    return cdiv(kallen_gpu(sigma_i, y, z), cmul_d(sigma_i, 4.0));
}

__host__ __device__ inline GpuComplex Jfunc_gpu(GpuComplex z) {
    if (z.x <= 0.0) return make_c(0.0);
    if (z.x > 0.0 && z.x < 1.0) {
        GpuComplex A = cdiv(make_c(-1.0), z);
        GpuComplex B = cexp_c(cdiv(make_c(-1.0), csub(make_c(1.0), z)));
        return cexp_c(cmul(A, B));
    }
    return make_c(1.0);
}

__host__ __device__ inline GpuComplex cutoff_gpu(GpuComplex sigma_i, double mj, double mk, double epsilon_h) {
    if (mj == mk && epsilon_h == 0.0) {
        return Jfunc_gpu(cdiv(sigma_i, make_c(4.0*mj*mj)));
    }
    double absdiff = fabs(mj*mj - mk*mk);
    double denom = (mj + mk)*(mj + mk) - absdiff;
    GpuComplex num = csub(sigma_i, make_c(absdiff));
    GpuComplex Z = cmul_d(cdiv(num, make_c(denom)), 1.0 + epsilon_h);
    return Jfunc_gpu(Z);
}

__host__ __device__ inline GpuComplex sigma_pvec_gpu(
    GpuComplex En, double px, double py, double pz,
    double mi, Vec3d P)
{
    GpuComplex spec_p = csqrt_c(make_c(px*px + py*py + pz*pz));
    GpuComplex A = csub(En, omega_gpu(spec_p, mi));
    double dx = P.x - px;
    double dy = P.y - py;
    double dz = P.z - pz;
    double pminus_sq = dx*dx + dy*dy + dz*dz;
    return csub(cmul(A, A), make_c(pminus_sq));
}

__host__ __device__ inline void boost_gpu(
    GpuComplex p0,
    double px, double py, double pz,
    GpuComplex E2p,
    double P2px, double P2py, double P2pz,
    GpuComplex& ox, GpuComplex& oy, GpuComplex& oz)
{
    double P2norm_real = sqrt(P2px*P2px + P2py*P2py + P2pz*P2pz);
    if (P2norm_real < 1.0e-300) {
        ox = make_c(px); oy = make_c(py); oz = make_c(pz);
        return;
    }

    double hx = P2px / P2norm_real;
    double hy = P2py / P2norm_real;
    double hz = P2pz / P2norm_real;

    GpuComplex P2norm = make_c(P2norm_real);
    GpuComplex beta2 = cdiv(P2norm, E2p);
    GpuComplex beta2sq = cmul(beta2, beta2);
    GpuComplex gam2 = cdiv(make_c(1.0), csqrt_c(csub(make_c(1.0), beta2sq)));

    GpuComplex dot = make_c(px*hx + py*hy + pz*hz);

    GpuComplex common = csub(
        cmul(csub(gam2, make_c(1.0)), dot),
        cmul(cmul(gam2, beta2), p0)
    );

    ox = cadd(make_c(px), cmul_d(common, hx));
    oy = cadd(make_c(py), cmul_d(common, hy));
    oz = cadd(make_c(pz), cmul_d(common, hz));
}

__host__ __device__ inline GpuComplex spherical_harmonics_gpu(GpuComplex px, GpuComplex py, GpuComplex pz, int ell, int m) {
    if (ell == 0) return make_c(1.0);

    if (ell == 1) {
        const double s3 = 1.7320508075688772935274463415058724;
        if (m == -1) return cmul_d(py, s3);
        if (m ==  0) return cmul_d(pz, s3);
        if (m == +1) return cmul_d(px, s3);
        return make_c(0.0);
    }

    if (ell == 2) {
        const double s15 = 3.8729833462074168851792653997823996;
        const double s54 = 1.1180339887498948482045868343656381; // sqrt(5/4)
        if (m == -2) return cmul_d(cmul(px, py), s15);
        if (m == -1) return cmul_d(cmul(py, pz), s15);
        if (m ==  0) {
            GpuComplex val = csub(cmul_d(cmul(pz, pz), 2.0), cadd(cmul(px, px), cmul(py, py)));
            return cmul_d(val, s54);
        }
        if (m == +1) return cmul_d(cmul(px, pz), s15);
        if (m == +2) return cmul_d(csub(cmul(px, px), cmul(py, py)), s54);
        return make_c(0.0);
    }

    return make_c(0.0);
}

__device__ inline GpuComplex G_ij_lm_gpu(
    GpuComplex En,
    const ConfigEntry& p_entry,
    const ConfigEntry& k_entry,
    Vec3d P,
    double mi, double mj, double mk,
    double L,
    double epsilon_h,
    bool Q0norm)
{
    const double px = p_entry.px;
    const double py = p_entry.py;
    const double pz = p_entry.pz;
    const double kx = k_entry.px;
    const double ky = k_entry.py;
    const double kz = k_entry.pz;

    const int ell_f = p_entry.ell;
    const int proj_mf = p_entry.m;
    const int ell_i = k_entry.ell;
    const int proj_mi = k_entry.m;

    GpuComplex spec_p = csqrt_c(make_c(px*px + py*py + pz*pz));
    GpuComplex spec_k = csqrt_c(make_c(kx*kx + ky*ky + kz*kz));

    GpuComplex omg_p = omega_gpu(spec_p, mi);
    GpuComplex omg_k = omega_gpu(spec_k, mj);

    GpuComplex kst_x, kst_y, kst_z;
    GpuComplex pst_x, pst_y, pst_z;

    boost_gpu(
        omg_k,
        kx, ky, kz,
        csub(En, omg_p),
        P.x - px, P.y - py, P.z - pz,
        kst_x, kst_y, kst_z
    );

    boost_gpu(
        omg_p,
        px, py, pz,
        csub(En, omg_k),
        P.x - kx, P.y - ky, P.z - kz,
        pst_x, pst_y, pst_z
    );

    GpuComplex Ylm1 = spherical_harmonics_gpu(kst_x, kst_y, kst_z, ell_f, proj_mf);
    GpuComplex Ylm2 = spherical_harmonics_gpu(pst_x, pst_y, pst_z, ell_i, proj_mi);

    GpuComplex sig_i = sigma_pvec_gpu(En, px, py, pz, mi, P);
    GpuComplex sig_j = sigma_pvec_gpu(En, kx, ky, kz, mj, P);

    GpuComplex cutoff1 = cutoff_gpu(sig_i, mj, mk, epsilon_h);
    GpuComplex cutoff2 = cutoff_gpu(sig_j, mi, mk, epsilon_h);

    double mom_x = P.x - px - kx;
    double mom_y = P.y - py - ky;
    double mom_z = P.z - pz - kz;
    double mom_sq = mom_x*mom_x + mom_y*mom_y + mom_z*mom_z;

    GpuComplex A = csub(csub(En, omega_gpu(spec_p, mi)), omega_gpu(spec_k, mj));
    GpuComplex denom = csub(cmul(A, A), make_c(mom_sq + mk*mk));

    GpuComplex oneby2omegapLcube = cdiv(make_c(1.0), cmul_d(omega_gpu(spec_p, mi), 2.0*L*L*L));
    GpuComplex oneby2omegakLcube = cdiv(make_c(1.0), cmul_d(omega_gpu(spec_k, mj), 2.0*L*L*L));

    GpuComplex result = Ylm1;
    result = cmul(result, oneby2omegapLcube);
    result = cmul(result, cutoff1);
    result = cmul(result, cutoff2);
    result = cmul(result, cdiv(make_c(1.0), denom));
    result = cmul(result, oneby2omegakLcube);
    result = cmul(result, Ylm2);

    if (!Q0norm) {
        GpuComplex q2psq = q2psq_star_gpu(sig_i, mj, mk);
        GpuComplex q2ksq = q2psq_star_gpu(sig_j, mi, mk);
        result = cmul(cdiv(make_c(1.0), cpow_int(q2psq, ell_f)), result);
        result = cmul(result, cdiv(make_c(1.0), cpow_int(q2ksq, ell_i)));
    }

    return result;
}

__global__ void G_2plus1_kernel(
    GpuComplex* Gout,
    int total,
    int size1,
    int size2,
    GpuComplex En,
    const ConfigEntry* plm,
    const ConfigEntry* klm,
    Vec3d P,
    double m1,
    double m2,
    double L,
    double epsilon_h,
    bool Q0norm)
{
    long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long n_elem = static_cast<long long>(total) * static_cast<long long>(total);
    if (tid >= n_elem) return;

    int row = static_cast<int>(tid / total);
    int col = static_cast<int>(tid - static_cast<long long>(row) * total);

    GpuComplex val = make_c(0.0);

    if (row < size1 && col < size1) {
        val = G_ij_lm_gpu(En, plm[row], plm[col], P, m1, m1, m2, L, epsilon_h, Q0norm);
    }
    else if (row < size1 && col >= size1) {
        int j2 = col - size1;
        val = G_ij_lm_gpu(En, plm[row], klm[j2], P, m1, m2, m1, L, epsilon_h, Q0norm);
        double projector = (plm[row].ell % 2 == 0) ? 1.0 : -1.0;
        val = cmul_d(val, 1.4142135623730950488016887242096981 * projector);
    }
    else if (row >= size1 && col < size1) {
        int i2 = row - size1;
        val = G_ij_lm_gpu(En, klm[i2], plm[col], P, m2, m1, m1, L, epsilon_h, Q0norm);
        double projector = (plm[col].ell % 2 == 0) ? 1.0 : -1.0;
        val = cmul_d(val, 1.4142135623730950488016887242096981 * projector);
    }
    else {
        val = make_c(0.0);
    }

    Gout[tid] = val;
}

inline void G_2plus1_mat_gpu_safe(
    Eigen::MatrixXcd& G,
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
    const GGpuOptions& opt = GGpuOptions{})
{
    (void)alpha;
    (void)max_shell_num;

    if (total_P.size() < 3) throw std::runtime_error("G_2plus1_mat_gpu_safe: total_P size < 3");

    std::vector<ConfigEntry> h_plm = flatten_config_host(plm_config);
    std::vector<ConfigEntry> h_klm = flatten_config_host(klm_config);

    const int size1 = static_cast<int>(h_plm.size());
    const int size2 = static_cast<int>(h_klm.size());
    const int total = size1 + size2;

    G.resize(total, total);
    if (total == 0) return;

    ConfigEntry* d_plm = nullptr;
    ConfigEntry* d_klm = nullptr;
    GpuComplex* d_G = nullptr;

    const std::size_t bytes_plm = static_cast<std::size_t>(size1) * sizeof(ConfigEntry);
    const std::size_t bytes_klm = static_cast<std::size_t>(size2) * sizeof(ConfigEntry);
    const std::size_t bytes_G = static_cast<std::size_t>(total) * static_cast<std::size_t>(total) * sizeof(GpuComplex);

    if (size1 > 0) GGPU_CUDA_CHECK(cudaMalloc(&d_plm, bytes_plm));
    if (size2 > 0) GGPU_CUDA_CHECK(cudaMalloc(&d_klm, bytes_klm));
    GGPU_CUDA_CHECK(cudaMalloc(&d_G, bytes_G));

    if (size1 > 0) GGPU_CUDA_CHECK(cudaMemcpy(d_plm, h_plm.data(), bytes_plm, cudaMemcpyHostToDevice));
    if (size2 > 0) GGPU_CUDA_CHECK(cudaMemcpy(d_klm, h_klm.data(), bytes_klm, cudaMemcpyHostToDevice));

    Vec3d P{total_P[0].real(), total_P[1].real(), total_P[2].real()};
    GpuComplex En_gpu{En.real(), En.imag()};

    const long long n_elem = static_cast<long long>(total) * static_cast<long long>(total);
    const int threads = opt.threads_per_block > 0 ? opt.threads_per_block : 256;
    const int blocks = static_cast<int>((n_elem + threads - 1) / threads);

    G_2plus1_kernel<<<blocks, threads>>>(
        d_G, total, size1, size2, En_gpu, d_plm, d_klm, P, m1, m2, L, epsilon_h, Q0norm
    );
    GGPU_CUDA_CHECK(cudaGetLastError());
    GGPU_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<GpuComplex> h_G(static_cast<std::size_t>(total) * static_cast<std::size_t>(total));
    GGPU_CUDA_CHECK(cudaMemcpy(h_G.data(), d_G, bytes_G, cudaMemcpyDeviceToHost));

    for (int i = 0; i < total; ++i) {
        for (int j = 0; j < total; ++j) {
            const GpuComplex z = h_G[static_cast<std::size_t>(i) * static_cast<std::size_t>(total) + static_cast<std::size_t>(j)];
            G(i, j) = std::complex<double>(z.x, z.y);
        }
    }

    if (d_plm) cudaFree(d_plm);
    if (d_klm) cudaFree(d_klm);
    if (d_G) cudaFree(d_G);

    if (opt.debug == 'y') {
        std::cout << "[G_gpu_safe] built G with size " << total << " x " << total
                  << "  size1=" << size1 << " size2=" << size2
                  << "  matrix MiB=" << (double(bytes_G) / (1024.0 * 1024.0)) << "\n";
    }
}

inline void G_2plus1_many_mat_gpu_safe(
    std::vector<Eigen::MatrixXcd>& G_vec,
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
    const GGpuOptions& opt = GGpuOptions{})
{
    const std::size_t N = En_vec.size();
    if (plm_vec.size() != N || klm_vec.size() != N) {
        throw std::runtime_error("G_2plus1_many_mat_gpu_safe: vector size mismatch");
    }

    G_vec.resize(N);
    for (std::size_t i = 0; i < N; ++i) {
        G_2plus1_mat_gpu_safe(
            G_vec[i], En_vec[i], plm_vec[i], klm_vec[i], total_P,
            m1, m2, L, alpha, epsilon_h, max_shell_num, Q0norm, opt
        );
    }
}

struct MatrixCompareStats {
    double max_abs = 0.0;
    double max_rel = 0.0;
    double frob_abs = 0.0;
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
    double rel_tol = 1.0e-8)
{
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::runtime_error("compare_matrices: shape mismatch");
    }

    MatrixCompareStats s;
    long double diff2 = 0.0L;
    long double ref2 = 0.0L;

    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            const auto a = A(i, j);
            const auto b = B(i, j);
            if (!std::isfinite(a.real()) || !std::isfinite(a.imag()) ||
                !std::isfinite(b.real()) || !std::isfinite(b.imag())) {
                s.n_nan_inf++;
                continue;
            }
            const double d = std::abs(a - b);
            const double scale = std::max(1.0, std::abs(a));
            const double r = d / scale;
            diff2 += static_cast<long double>(d) * static_cast<long double>(d);
            ref2 += static_cast<long double>(std::abs(a)) * static_cast<long double>(std::abs(a));
            if (d > s.max_abs) { s.max_abs = d; s.max_i = i; s.max_j = j; }
            if (r > s.max_rel) { s.max_rel = r; }
            if (d > abs_tol) s.n_bad_abs++;
            if (r > rel_tol) s.n_bad_rel++;
        }
    }

    s.frob_abs = std::sqrt(static_cast<double>(diff2));
    s.frob_rel = (ref2 > 0.0L) ? std::sqrt(static_cast<double>(diff2 / ref2)) : s.frob_abs;
    return s;
}

} // namespace ggpu

#endif // G_GPU_SAFE_BUILDER_CUH
