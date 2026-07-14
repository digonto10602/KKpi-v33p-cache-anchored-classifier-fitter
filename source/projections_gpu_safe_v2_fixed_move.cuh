#ifndef PROJECTIONS_GPU_SAFE_CUH
#define PROJECTIONS_GPU_SAFE_CUH

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <Eigen/Dense>
#include <vector>
#include <array>
#include <complex>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include "real_wigner_d.hpp"

namespace pgpu
{
using comp = std::complex<double>;

#define PGPU_CUDA_CHECK(call) do { cudaError_t err__ = (call); if (err__ != cudaSuccess) { throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + std::to_string(__LINE__) + " : " + cudaGetErrorString(err__)); } } while(0)
#define PGPU_CUSOLVER_CHECK(call) do { cusolverStatus_t st__ = (call); if (st__ != CUSOLVER_STATUS_SUCCESS) { throw std::runtime_error(std::string("cuSOLVER error at ") + __FILE__ + ":" + std::to_string(__LINE__) + " status=" + std::to_string((int)st__)); } } while(0)

struct Options
{
    int threads_per_block = 256;
    double eig_tol = 0.05;
    double chop_tol = 1.0e-14;
    char debug = 'n';
};

struct HostConfigFlat
{
    int n = 0;
    std::vector<int> nx, ny, nz, ell, m;
};

struct DeviceConfig
{
    int n = 0;
    int* nx = nullptr;
    int* ny = nullptr;
    int* nz = nullptr;
    int* ell = nullptr;
    int* m = nullptr;

    DeviceConfig() = default;
    DeviceConfig(const DeviceConfig&) = delete;
    DeviceConfig& operator=(const DeviceConfig&) = delete;
    DeviceConfig(DeviceConfig&& o) noexcept { *this = std::move(o); }
    DeviceConfig& operator=(DeviceConfig&& o) noexcept
    {
        if (this != &o) {
            release();
            n = o.n; nx = o.nx; ny = o.ny; nz = o.nz; ell = o.ell; m = o.m;
            o.n = 0; o.nx = o.ny = o.nz = o.ell = o.m = nullptr;
        }
        return *this;
    }
    ~DeviceConfig() { release(); }
    void release()
    {
        if (nx) cudaFree(nx);
        if (ny) cudaFree(ny);
        if (nz) cudaFree(nz);
        if (ell) cudaFree(ell);
        if (m) cudaFree(m);
        nx = ny = nz = ell = m = nullptr;
        n = 0;
    }
};

struct DeviceConfigView
{
    int n;
    const int* nx;
    const int* ny;
    const int* nz;
    const int* ell;
    const int* m;
};

struct DeviceProjectionTables
{
    int LG = 0;
    int* R = nullptr;          // length 3*LG
    double* weight = nullptr;  // par * chi, length LG
    double* D1 = nullptr;      // LG*9
    double* D2 = nullptr;      // LG*25
    DeviceProjectionTables() = default;
    DeviceProjectionTables(const DeviceProjectionTables&) = delete;
    DeviceProjectionTables& operator=(const DeviceProjectionTables&) = delete;

    DeviceProjectionTables(DeviceProjectionTables&& o) noexcept
    {
        *this = std::move(o);
    }

    DeviceProjectionTables& operator=(DeviceProjectionTables&& o) noexcept
    {
        if (this != &o)
        {
            release();
            LG = o.LG;
            R = o.R;
            weight = o.weight;
            D1 = o.D1;
            D2 = o.D2;

            o.LG = 0;
            o.R = nullptr;
            o.weight = nullptr;
            o.D1 = nullptr;
            o.D2 = nullptr;
        }
        return *this;
    }

    ~DeviceProjectionTables() { release(); }
    void release()
    {
        if (R) cudaFree(R);
        if (weight) cudaFree(weight);
        if (D1) cudaFree(D1);
        if (D2) cudaFree(D2);
        R = nullptr; weight = D1 = D2 = nullptr; LG = 0;
    }
};

inline HostConfigFlat flatten_n_config(const std::vector<std::vector<int>>& n_config)
{
    if (n_config.size() < 5) throw std::runtime_error("flatten_n_config: n_config must have 5 rows");
    const int n = static_cast<int>(n_config[0].size());
    HostConfigFlat h; h.n = n;
    h.nx.resize(n); h.ny.resize(n); h.nz.resize(n); h.ell.resize(n); h.m.resize(n);
    for (int i=0;i<n;++i) {
        h.nx[i] = n_config[0][i];
        h.ny[i] = n_config[1][i];
        h.nz[i] = n_config[2][i];
        h.ell[i] = n_config[3][i];
        h.m[i] = n_config[4][i];
    }
    return h;
}

inline DeviceConfig upload_config(const HostConfigFlat& h)
{
    DeviceConfig d; d.n = h.n;
    if (h.n == 0) return d;
    const size_t bytes = sizeof(int) * static_cast<size_t>(h.n);
    PGPU_CUDA_CHECK(cudaMalloc(&d.nx, bytes));
    PGPU_CUDA_CHECK(cudaMalloc(&d.ny, bytes));
    PGPU_CUDA_CHECK(cudaMalloc(&d.nz, bytes));
    PGPU_CUDA_CHECK(cudaMalloc(&d.ell, bytes));
    PGPU_CUDA_CHECK(cudaMalloc(&d.m, bytes));
    PGPU_CUDA_CHECK(cudaMemcpy(d.nx, h.nx.data(), bytes, cudaMemcpyHostToDevice));
    PGPU_CUDA_CHECK(cudaMemcpy(d.ny, h.ny.data(), bytes, cudaMemcpyHostToDevice));
    PGPU_CUDA_CHECK(cudaMemcpy(d.nz, h.nz.data(), bytes, cudaMemcpyHostToDevice));
    PGPU_CUDA_CHECK(cudaMemcpy(d.ell, h.ell.data(), bytes, cudaMemcpyHostToDevice));
    PGPU_CUDA_CHECK(cudaMemcpy(d.m, h.m.data(), bytes, cudaMemcpyHostToDevice));
    return d;
}

inline DeviceConfigView view(const DeviceConfig& d)
{
    return DeviceConfigView{d.n, d.nx, d.ny, d.nz, d.ell, d.m};
}

inline int sign_int(int x) { return (x > 0) - (x < 0); }

inline std::vector<std::array<int,3>> rotations_list_host()
{
    return {{
        { 1,  2,  3}, { 2,  3,  1}, { 3,  1,  2},
        { 1,  3, -2}, { 2, -1,  3}, { 3,  2, -1},
        { 1, -2, -3}, { 2, -3, -1}, { 3, -1, -2},
        { 1, -3,  2}, { 2,  1, -3}, { 3, -2,  1},
        {-1,  2, -3}, {-2,  3, -1}, {-3,  1, -2},
        {-1,  3,  2}, {-2, -1, -3}, {-3,  2,  1},
        {-1, -2,  3}, {-2, -3,  1}, {-3, -1,  2},
        {-1, -3, -2}, {-2,  1,  3}, {-3, -2, -1}
    }};
}

inline bool is_rotation_host(const std::array<int,3>& R)
{
    auto rots = rotations_list_host();
    return std::find(rots.begin(), rots.end(), R) != rots.end();
}

inline std::vector<std::array<int,3>> all_perms_host(std::array<int,3> a)
{
    std::vector<int> v = {a[0],a[1],a[2]};
    std::sort(v.begin(), v.end());
    std::vector<std::array<int,3>> out;
    do { out.push_back({v[0],v[1],v[2]}); } while(std::next_permutation(v.begin(), v.end()));
    return out;
}

inline std::vector<std::array<int,3>> Oh_list_host()
{
    std::vector<std::array<int,3>> out;
    auto append = [&](std::array<int,3> a){ auto p=all_perms_host(a); out.insert(out.end(), p.begin(), p.end()); };
    append({ 1,  2,  3}); append({ 1,  2, -3}); append({ 1, -2,  3}); append({-1,  2,  3});
    append({ 1, -2, -3}); append({-1,  2, -3}); append({-1, -2,  3}); append({-1, -2, -3});
    return out;
}

inline std::vector<std::array<int,3>> little_group_host(int Px, int Py, int Pz)
{
    if (Px==0 && Py==0 && Pz==0) return Oh_list_host();
    if (Px==0 && Py==0) return {{ {1,2,3},{-1,2,3},{1,-2,3},{-1,-2,3},{2,1,3},{-2,1,3},{2,-1,3},{-2,-1,3} }};
    if (Px==Py && Pz==0) return {{ {1,2,3},{1,2,-3},{2,1,3},{2,1,-3} }};
    if (Px==Py && Py==Pz) return {{ {1,2,3},{1,3,2},{2,1,3},{2,3,1},{3,1,2},{3,2,1} }};
    if (Px!=0 && Py!=0 && Px!=Py && Pz==0) return {{ {1,2,3},{1,2,-3} }};
    if (Px==Py && Py!=Pz) return {{ {1,2,3},{2,1,3} }};
    if (Px!=0 && Py!=0 && Pz!=0 && Px!=Py && Py!=Pz && Pz!=Px) return {{ {1,2,3} }};
    throw std::runtime_error("little_group_host: unsupported total momentum shell");
}

inline std::string conj_class_host(const std::array<int,3>& p)
{
    int N_negs=0, N_correct=0;
    for (int i=0;i<3;++i) { if (p[i]<0) ++N_negs; if (std::abs(p[i])==i+1) ++N_correct; }
    if (N_correct==3) {
        if (N_negs==0) return "E";
        if (N_negs==2) return "C4^2";
        if (N_negs==3) return "i";
        if (N_negs==1) return "sigma_h";
    } else if (N_correct==0) {
        return (N_negs%2==0) ? "C3" : "S6";
    } else if (N_correct==1) {
        int ic=-1; for (int i=0;i<3;++i) if (std::abs(p[i])==i+1) {ic=i; break;}
        if (N_negs%2==1) return (p[ic]<0) ? "C2" : "C4";
        return (p[ic]>0) ? "sigma_d" : "S4";
    }
    throw std::runtime_error("conj_class_host failed");
}

inline int irrep_dim_host(const std::string& I)
{
    if (I=="A1g"||I=="A1"||I=="A2g"||I=="A2"||I=="A1u"||I=="A2u"||I=="B1"||I=="B2") return 1;
    if (I=="Eg"||I=="E"||I=="Eu"||I=="E2") return 2;
    if (I=="T1g"||I=="T1"||I=="T2g"||I=="T2"||I=="T1u"||I=="T2u") return 3;
    throw std::runtime_error("irrep_dim_host: invalid irrep " + I);
}

inline int chi_host(const std::array<int,3>& p, const std::string& I, int Px, int Py, int Pz)
{
    const std::string cc = conj_class_host(p);
    if (Px==0 && Py==0 && Pz==0) {
        if (I=="A1g"||I=="A1") return 1;
        if (I=="A2g"||I=="A2") return (cc=="C2"||cc=="C4"||cc=="sigma_d"||cc=="S4") ? -1 : 1;
        if (I=="Eg"||I=="E") { if (cc=="E"||cc=="C4^2"||cc=="i"||cc=="sigma_h") return 2; if (cc=="C3"||cc=="S6") return -1; return 0; }
        if (I=="T1g"||I=="T1") { if (cc=="E"||cc=="i") return 3; if (cc=="C3"||cc=="S6") return 0; if (cc=="C4"||cc=="S4") return 1; return -1; }
        if (I=="T2g"||I=="T2") { if (cc=="E"||cc=="i") return 3; if (cc=="C3"||cc=="S6") return 0; if (cc=="C2"||cc=="sigma_d") return 1; return -1; }
        if (I=="A1u") return (cc=="E"||cc=="C3"||cc=="C4^2"||cc=="C4"||cc=="C2") ? 1 : -1;
        if (I=="A2u") return (cc=="E"||cc=="C3"||cc=="C4^2"||cc=="S4"||cc=="sigma_d") ? 1 : -1;
        if (I=="Eu") { if (cc=="E"||cc=="C4^2") return 2; if (cc=="i"||cc=="sigma_h") return -2; if (cc=="S6") return 1; if (cc=="C3") return -1; return 0; }
        if (I=="T1u") { if (cc=="E") return 3; if (cc=="i") return -3; if (cc=="C4"||cc=="sigma_h"||cc=="sigma_d") return 1; if (cc=="C2"||cc=="C4^2"||cc=="S4") return -1; return 0; }
        if (I=="T2u") { if (cc=="E") return 3; if (cc=="i") return -3; if (cc=="C2"||cc=="S4"||cc=="sigma_h") return 1; if (cc=="C4"||cc=="C4^2"||cc=="sigma_d") return -1; return 0; }
    } else if (Px==0 && Py==0) {
        if (I=="A1") return 1;
        if (I=="A2") return (cc=="sigma_h"||cc=="sigma_d") ? -1 : 1;
        if (I=="B1") return (cc=="C4"||cc=="sigma_d") ? -1 : 1;
        if (I=="B2") return (cc=="C4"||cc=="sigma_h") ? -1 : 1;
        if (I=="E"||I=="E2") { if (cc=="E") return 2; if (cc=="C4^2") return -2; return 0; }
    } else if (Px==Py && Py!=Pz && Pz==0) {
        if (I=="A1") return 1;
        if (I=="A2") return (cc=="sigma_h"||cc=="sigma_d") ? -1 : 1;
        if (I=="B1") return (cc=="C2"||cc=="sigma_h") ? -1 : 1;
        if (I=="B2") return (cc=="C2"||cc=="sigma_d") ? -1 : 1;
    } else if (Px==Py && Py==Pz) {
        if (I=="A1") return 1;
        if (I=="A2") return (cc=="sigma_d") ? -1 : 1;
        if (I=="E"||I=="E2") { if (cc=="E") return 2; if (cc=="C3") return -1; return 0; }
    } else if ((Px!=0 && Px!=Py && Py!=0 && Pz==0) || (Px==Py && Py!=Pz)) {
        if (I=="A1"||I=="A") return 1;
        if (I=="A2"||I=="B") return (cc=="sigma_h"||cc=="sigma_d") ? -1 : 1;
    } else {
        return 1;
    }
    throw std::runtime_error("chi_host: invalid irrep/shell combination: " + I);
}

inline DeviceProjectionTables upload_projection_tables(const std::string& irrep, int nnP0, int nnP1, int nnP2, int parity)
{
    auto LG = little_group_host(nnP0, nnP1, nnP2);
    DeviceProjectionTables d;
    d.LG = static_cast<int>(LG.size());

    std::vector<int> hR(3*d.LG);
    std::vector<double> hw(d.LG);
    std::vector<double> hD1(d.LG*9, 0.0), hD2(d.LG*25, 0.0);

    for (int g=0; g<d.LG; ++g) {
        auto R = LG[g];
        hR[3*g+0] = R[0]; hR[3*g+1] = R[1]; hR[3*g+2] = R[2];
        int par = (parity == -1 && !is_rotation_host(R)) ? -1 : 1;
        int ch = chi_host(R, irrep, nnP0, nnP1, nnP2);
        hw[g] = static_cast<double>(par * ch);

        std::vector<int> Rv = {R[0], R[1], R[2]};
        for (int mf=-1; mf<=1; ++mf)
            for (int mi=-1; mi<=1; ++mi)
                hD1[g*9 + (mf+1)*3 + (mi+1)] = real_wigner_d::D_real_element(1, mf, mi, Rv);
        for (int mf=-2; mf<=2; ++mf)
            for (int mi=-2; mi<=2; ++mi)
                hD2[g*25 + (mf+2)*5 + (mi+2)] = real_wigner_d::D_real_element(2, mf, mi, Rv);
    }

    PGPU_CUDA_CHECK(cudaMalloc(&d.R, sizeof(int)*hR.size()));
    PGPU_CUDA_CHECK(cudaMalloc(&d.weight, sizeof(double)*hw.size()));
    PGPU_CUDA_CHECK(cudaMalloc(&d.D1, sizeof(double)*hD1.size()));
    PGPU_CUDA_CHECK(cudaMalloc(&d.D2, sizeof(double)*hD2.size()));
    PGPU_CUDA_CHECK(cudaMemcpy(d.R, hR.data(), sizeof(int)*hR.size(), cudaMemcpyHostToDevice));
    PGPU_CUDA_CHECK(cudaMemcpy(d.weight, hw.data(), sizeof(double)*hw.size(), cudaMemcpyHostToDevice));
    PGPU_CUDA_CHECK(cudaMemcpy(d.D1, hD1.data(), sizeof(double)*hD1.size(), cudaMemcpyHostToDevice));
    PGPU_CUDA_CHECK(cudaMemcpy(d.D2, hD2.data(), sizeof(double)*hD2.size(), cudaMemcpyHostToDevice));
    return d;
}

__device__ inline int sign_dev(int x) { return (x>0) - (x<0); }

__device__ inline void sort2_int(int& a, int& b) { if (b < a) { int t=a; a=b; b=t; } }
__device__ inline void sort3_int(int& a, int& b, int& c) { sort2_int(a,b); sort2_int(b,c); sort2_int(a,b); }
__device__ inline void sort3_abs_order(int& a, int& b, int& c)
{
    for (int pass=0; pass<3; ++pass) {
        if (abs(b) < abs(a)) { int t=a; a=b; b=t; }
        if (abs(c) < abs(b)) { int t=b; b=c; c=t; }
    }
}

__device__ inline void cubic_transform_dev(int vx, int vy, int vz, int r0, int r1, int r2, int& ox, int& oy, int& oz)
{
    int arr[3] = {vx, vy, vz};
    int R[3] = {r0, r1, r2};
    int* out[3] = {&ox, &oy, &oz};
    for (int j=0;j<3;++j) {
        int idx = abs(R[j]) - 1;
        int sgn = (R[j] > 0) ? 1 : -1;
        *out[j] = sgn * arr[idx];
    }
}

__device__ inline void orbit_key_dev(int a0, int b0, int c0, int Px, int Py, int Pz, int& oa, int& ob, int& oc)
{
    int a=a0,b=b0,c=c0;
    if (Px==0 && Py==0 && Pz==0) {
        a=abs(a); b=abs(b); c=abs(c); sort3_int(a,b,c);
        if ((a==0 && b>0) || (a<b && b==c)) { oa=b; ob=c; oc=a; }
        else { oa=a; ob=b; oc=c; }
    } else if (Px==0 && Py==0 && Pz>0) {
        if (a==0 || b==0) { int bb = max(abs(a), abs(b)); oa=bb; ob=0; oc=c; }
        else { a=abs(a); b=abs(b); sort2_int(a,b); oa=a; ob=b; oc=c; }
    } else if (Px==Py && Px>0 && Pz==0) {
        sort2_int(a,b); oa=a; ob=b; oc=abs(c);
    } else if (Px==Py && Py==Pz && Px>0) {
        sort3_int(a,b,c); sort3_abs_order(a,b,c);
        if (a==0 && b!=0) { sort2_int(b,c); oa=b; ob=c; oc=0; }
        else if (a<b && b==c) { oa=b; ob=b; oc=a; }
        else { oa=a; ob=b; oc=c; }
    } else if (Pz==0 && Px>0 && Px<Py) {
        oa=a; ob=b; oc=abs(c);
    } else if (Px>0 && Px==Py && Pz>0 && Px!=Pz) {
        sort2_int(a,b); oa=a; ob=b; oc=c;
    } else {
        oa=a; ob=b; oc=c;
    }
}

__device__ inline double D_lookup_dev(int ell, int mf, int mi, int g, const double* D1, const double* D2)
{
    if (ell == 0) return 1.0;
    if (ell == 1) return D1[g*9 + (mf+1)*3 + (mi+1)];
    if (ell == 2) return D2[g*25 + (mf+2)*5 + (mi+2)];
    return 0.0;
}

__global__ void build_PI_kernel(
    double* PI, int N, int size1, int dI,
    int nnP0, int nnP1, int nnP2,
    DeviceConfigView cfg1, DeviceConfigView cfg2,
    int LG, const int* R, const double* weight, const double* D1, const double* D2)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * N;
    if (tid >= total) return;

    const int row = tid % N;
    const int col = tid / N; // column-major flat index

    bool in11 = (row < size1 && col < size1);
    bool in22 = (row >= size1 && col >= size1);
    if (!(in11 || in22)) { PI[tid] = 0.0; return; }

    DeviceConfigView cfg;
    int i, j;
    if (in11) { cfg = cfg1; i = row; j = col; }
    else { cfg = cfg2; i = row - size1; j = col - size1; }

    const int npx = cfg.nx[i], npy = cfg.ny[i], npz = cfg.nz[i];
    const int nkx = cfg.nx[j], nky = cfg.ny[j], nkz = cfg.nz[j];
    const int ell_p = cfg.ell[i], ell_k = cfg.ell[j];
    const int mp = cfg.m[i], mk = cfg.m[j];

    int opx, opy, opz, okx, oky, okz;
    orbit_key_dev(npx,npy,npz, nnP0,nnP1,nnP2, opx,opy,opz);
    orbit_key_dev(nkx,nky,nkz, nnP0,nnP1,nnP2, okx,oky,okz);

    double val = 0.0;
    if (ell_p == ell_k && opx==okx && opy==oky && opz==okz) {
        for (int g=0; g<LG; ++g) {
            int r0 = R[3*g+0], r1 = R[3*g+1], r2 = R[3*g+2];
            int rx, ry, rz;
            cubic_transform_dev(npx,npy,npz, r0,r1,r2, rx,ry,rz);
            if (rx==nkx && ry==nky && rz==nkz) {
                val += weight[g] * D_lookup_dev(ell_k, mp, mk, g, D1, D2);
            }
        }
        val *= static_cast<double>(dI) / static_cast<double>(LG);
    }

    if (fabs(val) < 1.0e-15) val = 0.0;
    PI[tid] = val;
}

inline Eigen::MatrixXd P_irrep_projection_2plus1_gpu_real(
    const std::vector<std::vector<int>>& np_config,
    const std::vector<std::vector<int>>& nk_config,
    const std::string& irrep,
    int nnP0, int nnP1, int nnP2,
    bool sort_orbit_flag,
    int parity,
    const Options& opt = Options())
{
    if (sort_orbit_flag) {
        throw std::runtime_error("P_irrep_projection_2plus1_gpu_real: sort_orbit_flag=true is not supported in the GPU path. Use false, matching your production path.");
    }

    HostConfigFlat h1 = flatten_n_config(np_config);
    HostConfigFlat h2 = flatten_n_config(nk_config);
    const int size1 = h1.n;
    const int size2 = h2.n;
    const int N = size1 + size2;
    if (N == 0) return Eigen::MatrixXd(0,0);

    DeviceConfig d1 = upload_config(h1);
    DeviceConfig d2 = upload_config(h2);
    DeviceProjectionTables tabs = upload_projection_tables(irrep, nnP0, nnP1, nnP2, parity);
    const int dI = irrep_dim_host(irrep);

    double* d_PI = nullptr;
    PGPU_CUDA_CHECK(cudaMalloc(&d_PI, sizeof(double)*static_cast<size_t>(N)*N));
    int threads = opt.threads_per_block;
    int blocks = (N*N + threads - 1) / threads;
    if (opt.debug == 'y') {
        std::cout << "[PGPU] build P_I N=" << N << " size1=" << size1 << " size2=" << size2
                  << " LG=" << tabs.LG << " blocks=" << blocks << " threads=" << threads << "\n";
    }
    build_PI_kernel<<<blocks, threads>>>(d_PI, N, size1, dI, nnP0,nnP1,nnP2, view(d1), view(d2), tabs.LG, tabs.R, tabs.weight, tabs.D1, tabs.D2);
    PGPU_CUDA_CHECK(cudaGetLastError());
    PGPU_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> hPI(static_cast<size_t>(N)*N);
    PGPU_CUDA_CHECK(cudaMemcpy(hPI.data(), d_PI, sizeof(double)*hPI.size(), cudaMemcpyDeviceToHost));
    cudaFree(d_PI);

    Eigen::MatrixXd PI(N,N);
    for (int col=0; col<N; ++col)
        for (int row=0; row<N; ++row)
            PI(row,col) = hPI[static_cast<size_t>(row) + static_cast<size_t>(col)*N];
    return PI;
}

inline void build_projector_from_eigenvectors_near_one_gpu_real(
    const Eigen::MatrixXd& PI_host,
    Eigen::MatrixXcd& Vsel,
    Eigen::MatrixXcd& Pproj,
    const Options& opt = Options())
{
    const int N = static_cast<int>(PI_host.rows());
    if (PI_host.cols() != N) throw std::runtime_error("build_projector_gpu: PI must be square");
    if (N == 0) { Vsel.resize(0,0); Pproj.resize(0,0); return; }

    std::vector<double> hA(static_cast<size_t>(N)*N);
    for (int col=0; col<N; ++col)
        for (int row=0; row<N; ++row)
            hA[static_cast<size_t>(row)+static_cast<size_t>(col)*N] = PI_host(row,col);

    double* dA = nullptr;
    double* dW = nullptr;
    int* devInfo = nullptr;
    PGPU_CUDA_CHECK(cudaMalloc(&dA, sizeof(double)*hA.size()));
    PGPU_CUDA_CHECK(cudaMalloc(&dW, sizeof(double)*N));
    PGPU_CUDA_CHECK(cudaMalloc(&devInfo, sizeof(int)));
    PGPU_CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeof(double)*hA.size(), cudaMemcpyHostToDevice));

    cusolverDnHandle_t solver = nullptr;
    PGPU_CUSOLVER_CHECK(cusolverDnCreate(&solver));

    int lwork = 0;
    PGPU_CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, N, dA, N, dW, &lwork));
    double* work = nullptr;
    PGPU_CUDA_CHECK(cudaMalloc(&work, sizeof(double)*static_cast<size_t>(lwork)));
    PGPU_CUSOLVER_CHECK(cusolverDnDsyevd(solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, N, dA, N, dW, work, lwork, devInfo));
    PGPU_CUDA_CHECK(cudaDeviceSynchronize());

    int info = 0;
    PGPU_CUDA_CHECK(cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {
        cudaFree(work); cudaFree(dA); cudaFree(dW); cudaFree(devInfo); cusolverDnDestroy(solver);
        throw std::runtime_error("cuSOLVER Dsyevd failed with devInfo=" + std::to_string(info));
    }

    std::vector<double> evals(N);
    std::vector<double> evecs(static_cast<size_t>(N)*N);
    PGPU_CUDA_CHECK(cudaMemcpy(evals.data(), dW, sizeof(double)*N, cudaMemcpyDeviceToHost));
    PGPU_CUDA_CHECK(cudaMemcpy(evecs.data(), dA, sizeof(double)*evecs.size(), cudaMemcpyDeviceToHost));

    cudaFree(work); cudaFree(dA); cudaFree(dW); cudaFree(devInfo); cusolverDnDestroy(solver);

    std::vector<int> keep;
    for (int i=0; i<N; ++i) {
        if (evals[i] >= 1.0 - opt.eig_tol && evals[i] <= 1.0 + opt.eig_tol) keep.push_back(i);
    }

    const int r = static_cast<int>(keep.size());
    Vsel = Eigen::MatrixXcd::Zero(N, r);
    for (int jc=0; jc<r; ++jc) {
        int src = keep[jc];
        double norm2 = 0.0;
        for (int row=0; row<N; ++row) {
            double v = evecs[static_cast<size_t>(row)+static_cast<size_t>(src)*N];
            norm2 += v*v;
        }
        double invn = (norm2 > 0.0) ? 1.0/std::sqrt(norm2) : 1.0;
        for (int row=0; row<N; ++row) {
            double v = evecs[static_cast<size_t>(row)+static_cast<size_t>(src)*N] * invn;
            if (std::abs(v) < opt.chop_tol) v = 0.0;
            Vsel(row,jc) = comp(v,0.0);
        }
    }
    Pproj = Vsel * Vsel.adjoint();

    if (opt.debug == 'y') {
        std::cout << "[PGPU] eigvals near one selected r=" << r << " / N=" << N << "\n";
    }
}

inline void P_irrep_projection_2plus1_and_Vsel_gpu_safe(
    Eigen::MatrixXcd& P_I,
    Eigen::MatrixXcd& Vsel,
    Eigen::MatrixXcd& Pproj,
    const std::vector<std::vector<int>>& np_config,
    const std::vector<std::vector<int>>& nk_config,
    const std::string& irrep,
    int nnP0, int nnP1, int nnP2,
    bool sort_orbit_flag,
    int parity,
    const Options& opt = Options())
{
    Eigen::MatrixXd PI_real = P_irrep_projection_2plus1_gpu_real(np_config, nk_config, irrep, nnP0,nnP1,nnP2, sort_orbit_flag, parity, opt);
    P_I = PI_real.cast<comp>();
    build_projector_from_eigenvectors_near_one_gpu_real(PI_real, Vsel, Pproj, opt);
}

} // namespace pgpu

#endif // PROJECTIONS_GPU_SAFE_CUH
