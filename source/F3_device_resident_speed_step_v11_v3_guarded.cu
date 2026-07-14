// ============================================================================
// F3_device_resident_speed_step_v11_v3_guarded.cu
//
// Device-resident speed-step pipeline for the 2+1 F3 determinant scan.
//
// Main goal:
//   Build F2, G, K2inv, and Vsel on the GPU and keep them on the GPU until
//   after the F3 solve/projection/determinant is complete. Only small final
//   results are copied back to the CPU.
//
// What is still CPU-side:
//   - The accepted config lists are compacted to CPU vectors by your current
//     GPU config maker. This preserves the exact deterministic config ordering.
//   - The projection eigensolver copies only the real eigenvalues to CPU so we
//     know which eigenvectors are near 1. The Vsel matrix itself is gathered on
//     the GPU and is not copied back before F3.
//
// What stays device-resident before F3 construction:
//   - F2
//   - G
//   - K2inv
//   - H = F2 + G + K2inv
//   - Vsel
//   - X1, F3, X2, F3inv_projected
//
// Compile with the provided compile script.
// ============================================================================

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <csignal>
#include <execinfo.h>
#include <unistd.h>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "functions_gpu_config_maker4.cuh"
#include "F2_gpu_safe_builder.cuh"
#include "G_gpu_safe_builder.cuh"
#include "K2_functions_gpu_safe.cuh"
#include "projections_gpu_safe.cuh"

using comp = std::complex<double>;

// ---------------- crash/stage diagnostics ----------------
static const char* g_v11_stage = "startup";
static inline void v11_set_stage(const char* s) { g_v11_stage = s; }
static void v11_sigsegv_handler(int sig)
{
    void* array[64];
    int n = backtrace(array, 64);
    const char msg1[] = "\n[V11 fatal] caught SIGSEGV while in stage: ";
    write(STDERR_FILENO, msg1, sizeof(msg1)-1);
    if (g_v11_stage) write(STDERR_FILENO, g_v11_stage, strlen(g_v11_stage));
    const char msg2[] = "\nBacktrace:\n";
    write(STDERR_FILENO, msg2, sizeof(msg2)-1);
    backtrace_symbols_fd(array, n, STDERR_FILENO);
    _exit(128 + sig);
}
#define V11_STAGE(name) do { v11_set_stage(name); } while(0)
#define V11_SYNC_STAGE(name) do { V11_STAGE(name); V11_CUDA_CHECK(cudaDeviceSynchronize()); } while(0)

#define V11_CUDA_CHECK(call) do { cudaError_t err__=(call); if(err__!=cudaSuccess){ throw std::runtime_error(std::string("CUDA error at ")+__FILE__+":"+std::to_string(__LINE__)+" in " #call " : "+cudaGetErrorString(err__)); } } while(0)
#define V11_CUBLAS_CHECK(call) do { cublasStatus_t st__=(call); if(st__!=CUBLAS_STATUS_SUCCESS){ throw std::runtime_error(std::string("cuBLAS error at ")+__FILE__+":"+std::to_string(__LINE__)+" in " #call " status="+std::to_string((int)st__)); } } while(0)
#define V11_CUSOLVER_CHECK(call) do { cusolverStatus_t st__=(call); if(st__!=CUSOLVER_STATUS_SUCCESS){ throw std::runtime_error(std::string("cuSOLVER error at ")+__FILE__+":"+std::to_string(__LINE__)+" in " #call " status="+std::to_string((int)st__)); } } while(0)

struct ScopedTimer
{
    std::string name;
    char debug;
    std::chrono::high_resolution_clock::time_point t0;
    ScopedTimer(std::string n, char d) : name(std::move(n)), debug(d), t0(std::chrono::high_resolution_clock::now()) {}
    ~ScopedTimer()
    {
        if (debug == 'y') {
            auto t1 = std::chrono::high_resolution_clock::now();
            double sec = std::chrono::duration<double>(t1 - t0).count();
            std::cout << "[timer] " << name << " : " << sec << " sec\n";
        }
    }
};

template <typename T>
struct DevicePtr
{
    T* p = nullptr;
    size_t n = 0;
    DevicePtr() = default;
    explicit DevicePtr(size_t n_) { alloc(n_); }
    DevicePtr(const DevicePtr&) = delete;
    DevicePtr& operator=(const DevicePtr&) = delete;
    DevicePtr(DevicePtr&& o) noexcept { p=o.p; n=o.n; o.p=nullptr; o.n=0; }
    DevicePtr& operator=(DevicePtr&& o) noexcept { if(this!=&o){ free(); p=o.p; n=o.n; o.p=nullptr; o.n=0; } return *this; }
    ~DevicePtr(){ free(); }
    void alloc(size_t n_){ free(); n=n_; if(n) V11_CUDA_CHECK(cudaMalloc(&p, n*sizeof(T))); }
    void free(){ if(p){ cudaFree(p); p=nullptr; n=0; } }
};

static inline cuDoubleComplex c_make(double r, double i=0.0){ return make_cuDoubleComplex(r,i); }

__global__ void combine_H_colmajor_kernel(
    cuDoubleComplex* H_col,
    const cuDoubleComplex* F2_col,
    const ggpu::GpuComplex* G_row,
    const k2gpu::Cx* K_row,
    int n)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n*n;
    if (tid >= total) return;
    const int row = tid % n;
    const int col = tid / n;
    const int row_idx = row*n + col;

    const cuDoubleComplex f = F2_col[tid];
    const ggpu::GpuComplex g = G_row[row_idx];
    const k2gpu::Cx k = K_row[row_idx];
    H_col[tid] = make_cuDoubleComplex(cuCreal(f) + g.x + k.re,
                                      cuCimag(f) + g.y + k.im);
}

__global__ void build_F3_from_F2_and_F2X1_kernel(
    cuDoubleComplex* F3,
    const cuDoubleComplex* F2,
    const cuDoubleComplex* F2X1,
    int total_elems)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elems) return;
    const cuDoubleComplex a = F2[tid];
    const cuDoubleComplex b = F2X1[tid];
    F3[tid] = make_cuDoubleComplex((1.0/3.0)*cuCreal(a) - cuCreal(b),
                                   (1.0/3.0)*cuCimag(a) - cuCimag(b));
}

__global__ void gather_real_evecs_to_complex_Vsel_kernel(
    cuDoubleComplex* Vsel,
    const double* evecs_colmajor,
    const int* selected_cols,
    int N,
    int vdim,
    double chop_tol)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * vdim;
    if (tid >= total) return;
    const int row = tid % N;
    const int csel = tid / N;
    const int src_col = selected_cols[csel];
    double v = evecs_colmajor[row + src_col*N];
    if (fabs(v) < chop_tol) v = 0.0;
    Vsel[tid] = make_cuDoubleComplex(v, 0.0);
}

static void device_lu_solve_inplace(
    cublasHandle_t handle,
    int n,
    int nrhs,
    cuDoubleComplex* dA_lu,
    cuDoubleComplex* dB_inout)
{
    DevicePtr<cuDoubleComplex*> dAptr(1), dBptr(1);
    DevicePtr<int> piv(n), info(1);
    cuDoubleComplex* hA = dA_lu;
    cuDoubleComplex* hB = dB_inout;
    V11_CUDA_CHECK(cudaMemcpy(dAptr.p, &hA, sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice));
    V11_CUDA_CHECK(cudaMemcpy(dBptr.p, &hB, sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice));

    V11_CUBLAS_CHECK(cublasZgetrfBatched(handle, n, dAptr.p, n, piv.p, info.p, 1));
    int hinfo = 0;
    V11_CUDA_CHECK(cudaMemcpy(&hinfo, info.p, sizeof(int), cudaMemcpyDeviceToHost));
    if (hinfo != 0) throw std::runtime_error("LU factorization failed, info=" + std::to_string(hinfo));

    V11_CUBLAS_CHECK(cublasZgetrsBatched(handle, CUBLAS_OP_N, n, nrhs,
                                         (const cuDoubleComplex**)dAptr.p, n,
                                         piv.p, dBptr.p, n, info.p, 1));
    V11_CUDA_CHECK(cudaMemcpy(&hinfo, info.p, sizeof(int), cudaMemcpyDeviceToHost));
    if (hinfo != 0) throw std::runtime_error("LU solve failed, info=" + std::to_string(hinfo));
}

static comp determinant_colmajor_device_matrix(cublasHandle_t handle, const cuDoubleComplex* dA, int n)
{
    if (n <= 0) return comp(std::numeric_limits<double>::quiet_NaN(), 0.0);
    DevicePtr<cuDoubleComplex> dLU(static_cast<size_t>(n) * static_cast<size_t>(n));
    V11_CUDA_CHECK(cudaMemcpy(dLU.p, dA, sizeof(cuDoubleComplex)*size_t(n)*size_t(n), cudaMemcpyDeviceToDevice));

    DevicePtr<cuDoubleComplex*> dAptr(1);
    DevicePtr<int> piv(n), info(1);
    cuDoubleComplex* hA = dLU.p;
    V11_CUDA_CHECK(cudaMemcpy(dAptr.p, &hA, sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice));
    V11_CUBLAS_CHECK(cublasZgetrfBatched(handle, n, dAptr.p, n, piv.p, info.p, 1));
    int hinfo=0;
    V11_CUDA_CHECK(cudaMemcpy(&hinfo, info.p, sizeof(int), cudaMemcpyDeviceToHost));
    if (hinfo != 0) return comp(std::numeric_limits<double>::quiet_NaN(), 0.0);

    std::vector<cuDoubleComplex> hLU(size_t(n)*size_t(n));
    std::vector<int> hpiv(n);
    V11_CUDA_CHECK(cudaMemcpy(hLU.data(), dLU.p, sizeof(cuDoubleComplex)*hLU.size(), cudaMemcpyDeviceToHost));
    V11_CUDA_CHECK(cudaMemcpy(hpiv.data(), piv.p, sizeof(int)*hpiv.size(), cudaMemcpyDeviceToHost));

    comp det(1.0,0.0);
    int sign = 1;
    for (int i=0;i<n;++i) {
        cuDoubleComplex z = hLU[size_t(i) + size_t(i)*size_t(n)];
        det *= comp(cuCreal(z), cuCimag(z));
        if (hpiv[i] != i+1) sign = -sign;
    }
    return double(sign)*det;
}

struct DeviceResidentResult
{
    int i = -1;
    comp Ecm;
    comp En;
    int dim1 = 0;
    int dim2 = 0;
    int total_dim = 0;
    int vdim = 0;
    comp det = comp(std::numeric_limits<double>::quiet_NaN(), 0.0);
    double build_sec = 0.0;
    double solve_sec = 0.0;
};

struct PhysicsParams
{
    double atmpi = 0.06906;
    double atmK = 0.09698;
    double eta_1 = 1.0;
    double eta_2 = 0.5;
    double alpha = 0.5;
    double max_shell_num = 20.0;
    double epsilon_h = 0.0;
    double xi = 3.444;
    double Lbyas = 20.0;
    double tolerance = 1.0e-12;
    bool Q0norm = true;
    int parity = +1;
    bool sort_orbit_flag = false;
};

static std::vector<std::vector<comp>> make_scatter_params_1()
{
    std::vector<std::vector<comp>> s(4, std::vector<comp>(3, comp(0.0,0.0)));
    s[0][0] = 4.04; s[0][1] = 0.0; s[0][2] = 0.0;
    s[1][0] = -43.2; s[1][1] = 0.0; s[1][2] = 0.0;
    return s;
}

static std::vector<std::vector<comp>> make_scatter_params_2()
{
    std::vector<std::vector<comp>> s(4, std::vector<comp>(3, comp(0.0,0.0)));
    s[0][0] = 4.12; s[0][1] = 0.0; s[0][2] = 0.0;
    return s;
}

static void build_projection_Vsel_device(
    cuDoubleComplex*& dVsel_out,
    int& vdim_out,
    int N,
    int size1,
    const std::vector<std::vector<int>>& np_config,
    const std::vector<std::vector<int>>& nk_config,
    const std::string& irrep,
    int nnP0,
    int nnP1,
    int nnP2,
    int parity,
    double eig_tol,
    double chop_tol,
    int threads,
    char debug)
{
    using namespace pgpu;
    HostConfigFlat h1 = flatten_n_config(np_config);
    HostConfigFlat h2 = flatten_n_config(nk_config);
    if (h1.n + h2.n != N) throw std::runtime_error("projection config dimension mismatch");

    DeviceConfig d1 = upload_config(h1);
    DeviceConfig d2 = upload_config(h2);
    DeviceProjectionTables tabs = upload_projection_tables(irrep, nnP0, nnP1, nnP2, parity);
    const int dI = irrep_dim_host(irrep);

    DevicePtr<double> dPI(static_cast<size_t>(N) * static_cast<size_t>(N));
    int blocks = (N*N + threads - 1)/threads;
    build_PI_kernel<<<blocks, threads>>>(dPI.p, N, size1, dI, nnP0, nnP1, nnP2,
                                         view(d1), view(d2), tabs.LG, tabs.R, tabs.weight, tabs.D1, tabs.D2);
    V11_CUDA_CHECK(cudaGetLastError());

    DevicePtr<double> dW(N);
    int lwork = 0;
    cusolverDnHandle_t solver = nullptr;
    V11_CUSOLVER_CHECK(cusolverDnCreate(&solver));
    V11_CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                                                   N, dPI.p, N, dW.p, &lwork));
    DevicePtr<double> work(lwork);
    DevicePtr<int> devInfo(1);
    V11_CUSOLVER_CHECK(cusolverDnDsyevd(solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                                        N, dPI.p, N, dW.p, work.p, lwork, devInfo.p));
    int info = 0;
    V11_CUDA_CHECK(cudaMemcpy(&info, devInfo.p, sizeof(int), cudaMemcpyDeviceToHost));
    V11_CUSOLVER_CHECK(cusolverDnDestroy(solver));
    if (info != 0) throw std::runtime_error("projection Dsyevd failed, info=" + std::to_string(info));

    std::vector<double> evals(N);
    V11_CUDA_CHECK(cudaMemcpy(evals.data(), dW.p, sizeof(double)*N, cudaMemcpyDeviceToHost));
    std::vector<int> selected;
    for (int i=0;i<N;++i) {
        if (evals[i] >= 1.0 - eig_tol && evals[i] <= 1.0 + eig_tol) selected.push_back(i);
    }
    vdim_out = static_cast<int>(selected.size());
    if (vdim_out <= 0) throw std::runtime_error("projection produced Vsel with zero columns");

    DevicePtr<int> dSel(vdim_out);
    V11_CUDA_CHECK(cudaMemcpy(dSel.p, selected.data(), sizeof(int)*selected.size(), cudaMemcpyHostToDevice));
    V11_CUDA_CHECK(cudaMalloc(&dVsel_out, sizeof(cuDoubleComplex)*size_t(N)*size_t(vdim_out)));
    int vblocks = (N*vdim_out + threads - 1)/threads;
    gather_real_evecs_to_complex_Vsel_kernel<<<vblocks, threads>>>(dVsel_out, dPI.p, dSel.p, N, vdim_out, chop_tol);
    V11_CUDA_CHECK(cudaGetLastError());

    if (debug == 'y') {
        std::cout << "[projection device-resident] N=" << N << " vdim=" << vdim_out << " irrep=" << irrep << "\n";
    }
}

static DeviceResidentResult compute_one_energy_device_resident(
    int idx,
    double Ecm,
    const std::array<int,3>& nnP,
    const std::string& irrep,
    const std::string& irrep_tag,
    const PhysicsParams& par,
    cublasHandle_t cublas,
    char debug = 'n')
{
    (void)irrep_tag;
    DeviceResidentResult out;
    V11_STAGE("compute_one_energy: enter");
    out.i = idx;
    out.Ecm = comp(Ecm,0.0);

    const double pi = std::acos(-1.0);
    const double L = par.xi * par.Lbyas;
    const double twopibyL = 2.0*pi/L;
    std::vector<comp> total_P(3);
    total_P[0] = comp(twopibyL * nnP[0], 0.0);
    total_P[1] = comp(twopibyL * nnP[1], 0.0);
    total_P[2] = comp(twopibyL * nnP[2], 0.0);
    const double P2 = std::norm(total_P[0]) + std::norm(total_P[1]) + std::norm(total_P[2]);
    const double En_real = std::sqrt(Ecm*Ecm + P2);
    out.En = comp(En_real, 0.0);

    std::vector<int> waves_vec_1 = {0,1};
    std::vector<int> waves_vec_2 = {0};
    std::vector<std::vector<comp>> plm_config(5), klm_config(5);
    std::vector<std::vector<int>> np_config(5), nk_config(5);

    auto tbuild0 = std::chrono::high_resolution_clock::now();

    V11_STAGE("config_maker_4_gpu_safe: begin");
    {
        ScopedTimer timer("GPU config compact", debug);
        f4gpu::gpu_config_maker_4_single_to_cpu_vectors(plm_config, np_config, waves_vec_1, En_real, total_P,
                                                         par.atmK, par.atmK, par.atmpi, L, par.epsilon_h,
                                                         par.max_shell_num, par.tolerance, debug);
        f4gpu::gpu_config_maker_4_single_to_cpu_vectors(klm_config, nk_config, waves_vec_2, En_real, total_P,
                                                         par.atmpi, par.atmK, par.atmK, L, par.epsilon_h,
                                                         par.max_shell_num, par.tolerance, debug);
    }

    V11_STAGE("config_maker_4_gpu_safe: finished");
    out.dim1 = static_cast<int>(plm_config[0].size());
    out.dim2 = static_cast<int>(klm_config[0].size());
    const int n = out.dim1 + out.dim2;
    out.total_dim = n;
    if (n <= 0) return out;

    const int threads = 256;
    const int elems = n*n;
    const int blocks = (elems + threads - 1)/threads;

    // ---------------- F2 on device, column-major ----------------
    V11_STAGE("F2 flatten/upload/build: begin");
    f2gpu::FlatConfig h_f2_1 = f2gpu::flatten_config(plm_config);
    f2gpu::FlatConfig h_f2_2 = f2gpu::flatten_config(klm_config);
    f2gpu::DeviceFlatConfig d_f2_1(h_f2_1), d_f2_2(h_f2_2);
    DevicePtr<cuDoubleComplex> dF2(static_cast<size_t>(elems));
    f2gpu::build_F2_2plus1_kernel<<<blocks, threads>>>(
        dF2.p, n, make_cuDoubleComplex(En_real,0.0),
        d_f2_1.px.ptr, d_f2_1.py.ptr, d_f2_1.pz.ptr, d_f2_1.ell.ptr, d_f2_1.m.ptr, out.dim1,
        d_f2_2.px.ptr, d_f2_2.py.ptr, d_f2_2.pz.ptr, d_f2_2.ell.ptr, d_f2_2.m.ptr, out.dim2,
        total_P[0].real(), total_P[1].real(), total_P[2].real(),
        par.atmK, par.atmpi, L, par.alpha, par.epsilon_h, int(par.max_shell_num), par.Q0norm,
        100, 1.0e-15);
    V11_CUDA_CHECK(cudaGetLastError());
    V11_SYNC_STAGE("F2 kernel synchronize");

    // ---------------- G on device, row-major in ggpu header ----------------
    V11_STAGE("G flatten/upload/build: begin");
    std::vector<ggpu::ConfigEntry> hG1 = ggpu::flatten_config_host(plm_config);
    std::vector<ggpu::ConfigEntry> hG2 = ggpu::flatten_config_host(klm_config);
    ggpu::ConfigEntry* dG1 = nullptr;
    ggpu::ConfigEntry* dG2 = nullptr;
    ggpu::GpuComplex* dG = nullptr;
    if (!hG1.empty()) V11_CUDA_CHECK(cudaMalloc(&dG1, sizeof(ggpu::ConfigEntry)*hG1.size()));
    if (!hG2.empty()) V11_CUDA_CHECK(cudaMalloc(&dG2, sizeof(ggpu::ConfigEntry)*hG2.size()));
    V11_CUDA_CHECK(cudaMalloc(&dG, sizeof(ggpu::GpuComplex)*size_t(elems)));
    if (!hG1.empty()) V11_CUDA_CHECK(cudaMemcpy(dG1, hG1.data(), sizeof(ggpu::ConfigEntry)*hG1.size(), cudaMemcpyHostToDevice));
    if (!hG2.empty()) V11_CUDA_CHECK(cudaMemcpy(dG2, hG2.data(), sizeof(ggpu::ConfigEntry)*hG2.size(), cudaMemcpyHostToDevice));
    ggpu::Vec3d Pgg{total_P[0].real(), total_P[1].real(), total_P[2].real()};
    ggpu::G_2plus1_kernel<<<blocks, threads>>>(dG, n, out.dim1, out.dim2, ggpu::GpuComplex{En_real,0.0}, dG1, dG2, Pgg,
                                               par.atmK, par.atmpi, L, par.epsilon_h, par.Q0norm);
    V11_CUDA_CHECK(cudaGetLastError());
    V11_SYNC_STAGE("G kernel synchronize");

    // ---------------- K2 on device, row-major in k2 header ----------------
    V11_STAGE("K2 flatten/upload/build: begin");
    k2gpu::DeviceConfig dK1; dK1.upload(plm_config);
    k2gpu::DeviceConfig dK2; dK2.upload(klm_config);
    k2gpu::DeviceScatterParams dSP1; dSP1.upload(make_scatter_params_1());
    k2gpu::DeviceScatterParams dSP2; dSP2.upload(make_scatter_params_2());
    k2gpu::Cx* dK = nullptr;
    V11_CUDA_CHECK(cudaMalloc(&dK, sizeof(k2gpu::Cx)*size_t(elems)));
    k2gpu::build_K2inv_2plus1_kernel<<<blocks, threads>>>(
        En_real, dK1.view(), dK2.view(), total_P[0].real(), total_P[1].real(), total_P[2].real(),
        par.eta_1, par.eta_2, dSP1.view(), dSP2.view(), par.atmK, par.atmpi, par.epsilon_h, L, dK, n);
    V11_CUDA_CHECK(cudaGetLastError());
    V11_SYNC_STAGE("K2 kernel synchronize");

    // ---------------- H = F2 + G + K2, column-major ----------------
    V11_STAGE("combine H: begin");
    DevicePtr<cuDoubleComplex> dH(static_cast<size_t>(elems));
    combine_H_colmajor_kernel<<<blocks, threads>>>(dH.p, dF2.p, dG, dK, n);
    V11_CUDA_CHECK(cudaGetLastError());
    V11_SYNC_STAGE("combine H synchronize");

    // ---------------- Vsel on device ----------------
    V11_STAGE("projection/Vsel device build: begin");
    cuDoubleComplex* dVsel_raw = nullptr;
    int vdim = 0;
    build_projection_Vsel_device(dVsel_raw, vdim, n, out.dim1, np_config, nk_config, irrep,
                                  nnP[0], nnP[1], nnP[2], par.parity,
                                  0.05, 1.0e-14, threads, debug);
    DevicePtr<cuDoubleComplex> dVsel; dVsel.p = dVsel_raw; dVsel.n = size_t(n)*size_t(vdim);
    V11_STAGE("projection/Vsel device build: finished");
    out.vdim = vdim;

    auto tbuild1 = std::chrono::high_resolution_clock::now();
    out.build_sec = std::chrono::duration<double>(tbuild1 - tbuild0).count();

    // Free config G device arrays after H is built.
    if (dG1) cudaFree(dG1);
    if (dG2) cudaFree(dG2);
    if (dG) cudaFree(dG);
    if (dK) cudaFree(dK);

    auto tsolve0 = std::chrono::high_resolution_clock::now();

    // H X1 = F2
    V11_STAGE("solve H X1 = F2: begin");
    DevicePtr<cuDoubleComplex> dH_LU(static_cast<size_t>(elems));
    DevicePtr<cuDoubleComplex> dX1(static_cast<size_t>(elems));
    V11_CUDA_CHECK(cudaMemcpy(dH_LU.p, dH.p, sizeof(cuDoubleComplex)*size_t(elems), cudaMemcpyDeviceToDevice));
    V11_CUDA_CHECK(cudaMemcpy(dX1.p, dF2.p, sizeof(cuDoubleComplex)*size_t(elems), cudaMemcpyDeviceToDevice));
    device_lu_solve_inplace(cublas, n, n, dH_LU.p, dX1.p);
    V11_SYNC_STAGE("solve H X1 = F2 synchronize");

    // F2X1 and F3
    V11_STAGE("build F3: begin");
    DevicePtr<cuDoubleComplex> dF2X1(static_cast<size_t>(elems));
    DevicePtr<cuDoubleComplex> dF3(static_cast<size_t>(elems));
    cuDoubleComplex alpha1 = c_make(1.0,0.0), beta0 = c_make(0.0,0.0);
    V11_CUBLAS_CHECK(cublasZgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                 &alpha1, dF2.p, n, dX1.p, n, &beta0, dF2X1.p, n));
    build_F3_from_F2_and_F2X1_kernel<<<blocks, threads>>>(dF3.p, dF2.p, dF2X1.p, elems);
    V11_CUDA_CHECK(cudaGetLastError());
    V11_SYNC_STAGE("build F3 synchronize");

    // F3 X2 = Vsel
    V11_STAGE("solve F3 X2 = Vsel: begin");
    DevicePtr<cuDoubleComplex> dF3_LU(static_cast<size_t>(elems));
    DevicePtr<cuDoubleComplex> dX2(static_cast<size_t>(n) * static_cast<size_t>(vdim));
    V11_CUDA_CHECK(cudaMemcpy(dF3_LU.p, dF3.p, sizeof(cuDoubleComplex)*size_t(elems), cudaMemcpyDeviceToDevice));
    V11_CUDA_CHECK(cudaMemcpy(dX2.p, dVsel.p, sizeof(cuDoubleComplex)*size_t(n)*size_t(vdim), cudaMemcpyDeviceToDevice));
    device_lu_solve_inplace(cublas, n, vdim, dF3_LU.p, dX2.p);
    V11_SYNC_STAGE("solve F3 X2 = Vsel synchronize");

    // F3inv_projected = Vsel^H X2
    V11_STAGE("projected determinant: begin");
    DevicePtr<cuDoubleComplex> dProj(static_cast<size_t>(vdim) * static_cast<size_t>(vdim));
    V11_CUBLAS_CHECK(cublasZgemm(cublas, CUBLAS_OP_C, CUBLAS_OP_N, vdim, vdim, n,
                                 &alpha1, dVsel.p, n, dX2.p, n, &beta0, dProj.p, vdim));

    V11_SYNC_STAGE("projection GEMM synchronize");
    out.det = determinant_colmajor_device_matrix(cublas, dProj.p, vdim);
    V11_STAGE("projected determinant: finished");

    auto tsolve1 = std::chrono::high_resolution_clock::now();
    out.solve_sec = std::chrono::duration<double>(tsolve1 - tsolve0).count();

    if (debug == 'y') {
        std::cout << "[device-resident] i=" << idx
                  << " Ecm=" << std::setprecision(17) << Ecm
                  << " dim=(" << out.dim1 << "," << out.dim2 << ")"
                  << " total_dim=" << n
                  << " vdim=" << vdim
                  << " det=" << out.det
                  << " build_sec=" << out.build_sec
                  << " solve_sec=" << out.solve_sec << "\n";
    }

    return out;
}

struct DetRow
{
    int i;
    double Ecm;
    comp det_raw;
    comp det_norm;
};

static std::vector<DetRow> run_scan_device_resident(
    const std::array<int,3>& nnP,
    const std::string& irrep,
    const std::string& irrep_tag,
    double Ecm_initial,
    double Ecm_final,
    int Ecm_points,
    const std::string& tag,
    char debug)
{
    PhysicsParams par;
    if (irrep.size() > 0 && irrep.back() == 'u') par.parity = -1;
    else par.parity = +1;

    cublasHandle_t cublas = nullptr;
    V11_CUBLAS_CHECK(cublasCreate(&cublas));

    std::vector<DetRow> rows;
    rows.reserve(Ecm_points);
    double max_abs = 0.0;

    const double del = (Ecm_points == 1) ? 0.0 : (Ecm_final - Ecm_initial)/double(Ecm_points - 1);

    std::ofstream fraw("det_proj_F3i_raw_" + std::to_string(nnP[0]) + std::to_string(nnP[1]) + std::to_string(nnP[2]) + "_" + irrep_tag + tag + "_L20.dat");
    fraw << "# i Ecm_real Ecm_imag det_real det_imag abs_det build_sec solve_sec total_dim vdim\n";

    for (int i=0;i<Ecm_points;++i) {
        double Ecm = Ecm_initial + double(i)*del;
        std::cout << "[scan " << tag << "] starting i=" << i << "/" << Ecm_points
                  << " Ecm=" << std::setprecision(17) << Ecm << std::endl;
        V11_STAGE("run_scan: compute_one_energy_device_resident");
        DeviceResidentResult r;
        try {
            r = compute_one_energy_device_resident(i, Ecm, nnP, irrep, irrep_tag, par, cublas, debug);
        } catch (const std::exception& e) {
            std::cerr << "[scan " << tag << "] FAILED at i=" << i
                      << " Ecm=" << std::setprecision(17) << Ecm
                      << " stage=" << (g_v11_stage ? g_v11_stage : "unknown")
                      << " error=" << e.what() << std::endl;
            throw;
        }
        DetRow row; row.i=i; row.Ecm=Ecm; row.det_raw=r.det; row.det_norm=r.det;
        rows.push_back(row);
        max_abs = std::max(max_abs, std::abs(r.det));
        fraw << std::setprecision(17)
             << i << '\t' << Ecm << '\t' << 0.0 << '\t'
             << r.det.real() << '\t' << r.det.imag() << '\t' << std::abs(r.det) << '\t'
             << r.build_sec << '\t' << r.solve_sec << '\t' << r.total_dim << '\t' << r.vdim << '\n';
        std::cout << "[scan " << tag << "] i=" << i << "/" << Ecm_points
                  << " Ecm=" << std::setprecision(17) << Ecm
                  << " det=" << r.det
                  << " dim=" << r.total_dim
                  << " vdim=" << r.vdim
                  << " build=" << r.build_sec
                  << " solve=" << r.solve_sec << "\n";
    }
    fraw.close();

    V11_CUBLAS_CHECK(cublasDestroy(cublas));

    if (max_abs == 0.0 || !std::isfinite(max_abs)) max_abs = 1.0;
    const double scale = 2.0/max_abs;
    std::ofstream fnorm("det_proj_F3i_normalized_" + std::to_string(nnP[0]) + std::to_string(nnP[1]) + std::to_string(nnP[2]) + "_" + irrep_tag + tag + "_L20.dat");
    fnorm << "# i Ecm_real Ecm_imag det_norm_real det_norm_imag abs_det_norm\n";
    for (auto& row: rows) {
        row.det_norm = row.det_raw * scale;
        fnorm << std::setprecision(17)
              << row.i << '\t' << row.Ecm << '\t' << 0.0 << '\t'
              << row.det_norm.real() << '\t' << row.det_norm.imag() << '\t'
              << std::abs(row.det_norm) << '\n';
    }
    fnorm.close();
    return rows;
}

static bool sign_flip(double a, double b)
{
    return (a == 0.0 || b == 0.0 || (a > 0.0 && b < 0.0) || (a < 0.0 && b > 0.0));
}

static double linear_zero(double x1, double y1, double x2, double y2)
{
    double den = y2 - y1;
    if (std::abs(den) < 1e-300) return 0.5*(x1+x2);
    return x1 - y1*(x2-x1)/den;
}

static std::string classify_inward_shape(const std::vector<DetRow>& rows, double& ezero, double trend_required=0.65, double margin_required=0.20)
{
    int N = static_cast<int>(rows.size());
    int flip = -1;
    double center = 0.5*double(N-1);
    double best = 1e300;
    for (int i=0;i+1<N;++i) {
        if (sign_flip(rows[i].det_norm.real(), rows[i+1].det_norm.real())) {
            double d = std::abs((double(i)+0.5) - center);
            if (d < best) { best=d; flip=i; }
        }
    }
    if (flip < 0) return "no_sign_flip";
    ezero = linear_zero(rows[flip].Ecm, rows[flip].det_norm.real(), rows[flip+1].Ecm, rows[flip+1].det_norm.real());

    std::vector<double> gaps;
    for (int k=0;k<N/2;++k) {
        gaps.push_back(std::abs(rows[k].det_norm.real()) + std::abs(rows[N-1-k].det_norm.real()));
    }
    int dec=0, inc=0, flat=0;
    for (int k=0;k+1<(int)gaps.size();++k) {
        double g0=gaps[k], g1=gaps[k+1];
        double eps = 1e-12*std::max({1.0,std::abs(g0),std::abs(g1)});
        if (g1 < g0 - eps) dec++;
        else if (g1 > g0 + eps) inc++;
        else flat++;
    }
    int tot = dec+inc+flat;
    if (tot <= 0) return "ambiguous";
    double df = double(dec)/double(tot);
    double inf = double(inc)/double(tot);
    double margin = std::abs(df-inf);
    if (df >= trend_required && margin >= margin_required) return "likely_zero";
    if (inf >= trend_required && margin >= margin_required) return "likely_pole";
    return "ambiguous";
}

static void adaptive_final_zeros_device_resident(
    const std::array<int,3>& nnP,
    const std::string& irrep,
    const std::string& irrep_tag,
    double Ecm_initial,
    double Ecm_final,
    int coarse_points,
    int refine_points,
    char debug)
{
    PhysicsParams par;
    std::vector<DetRow> coarse = run_scan_device_resident(nnP, irrep, irrep_tag, Ecm_initial, Ecm_final, coarse_points,
                                                          "_device_coarse_N" + std::to_string(coarse_points), debug);

    std::vector<int> flips;
    for (int i=0;i+1<(int)coarse.size();++i) {
        if (sign_flip(coarse[i].det_norm.real(), coarse[i+1].det_norm.real())) flips.push_back(i);
    }

    std::cout << "[adaptive-device] coarse sign flips = " << flips.size() << "\n";
    std::vector<double> zeros;
    int rid = 0;
    for (int fi: flips) {
        int il = std::max(0, fi - 5);
        int ir = std::min((int)coarse.size()-1, fi + 6);
        double EL = coarse[il].Ecm;
        double ER = coarse[ir].Ecm;
        bool done = false;
        int Nref = refine_points;
        for (int round=0; round<4 && !done; ++round) {
            auto ref = run_scan_device_resident(nnP, irrep, irrep_tag, EL, ER, Nref,
                                                "_device_refine_" + std::to_string(rid) + "_round" + std::to_string(round) + "_N" + std::to_string(Nref), debug);
            double ez = std::numeric_limits<double>::quiet_NaN();
            std::string cls = classify_inward_shape(ref, ez, 0.65, 0.20);
            std::cout << "[adaptive-device] candidate=" << rid << " round=" << round
                      << " N=" << Nref << " class=" << cls
                      << " E=" << std::setprecision(17) << ez << "\n";
            if (cls == "likely_zero") {
                bool dup=false;
                for (double z: zeros) if (std::abs(z-ez) < 1e-7) dup=true;
                if (!dup) zeros.push_back(ez);
                done = true;
            } else if (cls == "likely_pole" || cls == "no_sign_flip") {
                done = true;
            } else {
                Nref *= 2;
            }
        }
        rid++;
    }
    std::sort(zeros.begin(), zeros.end());
    std::string fname = "finalized_Ecm_zeros_" + std::to_string(nnP[0]) + std::to_string(nnP[1]) + std::to_string(nnP[2]) + "_" + irrep_tag + "_device_resident_L20.dat";
    std::ofstream fout(fname);
    fout << "# Lbyas\tEcm_zero\n";
    std::cout << "[adaptive-device] Finalized Ecm zeros\n# Lbyas\tEcm_zero\n";
    for (double z: zeros) {
        fout << std::setprecision(17) << par.Lbyas << '\t' << z << '\n';
        std::cout << std::setprecision(17) << par.Lbyas << '\t' << z << '\n';
    }
    std::cout << "[adaptive-device] saved " << fname << "\n";
}

int main(int argc, char** argv)
{
    std::signal(SIGSEGV, v11_sigsegv_handler);
    try {
        V11_STAGE("main: parse args");
        int n0=0,n1=0,n2=0;
        std::string irrep="A1g";
        std::string irrep_tag="A1p";
        double E0 = 0.26310;
        double E1 = 0.36;
        int coarseN = 2000;
        int refineN = 300;
        char debug = 'n';

        if (argc >= 4) { n0=std::stoi(argv[1]); n1=std::stoi(argv[2]); n2=std::stoi(argv[3]); }
        if (argc >= 5) irrep = argv[4];
        if (argc >= 6) irrep_tag = argv[5];
        if (argc >= 7) coarseN = std::stoi(argv[6]);
        if (argc >= 8) refineN = std::stoi(argv[7]);
        if (argc >= 10) { E0 = std::stod(argv[8]); E1 = std::stod(argv[9]); }
        if (argc >= 11) debug = argv[10][0];

        std::array<int,3> nnP{n0,n1,n2};
        std::cout << "# F3 device-resident speed-step v11\n";
        std::cout << "# nnP=["<<n0<<","<<n1<<","<<n2<<"] irrep="<<irrep
                  << " irrep_tag="<<irrep_tag
                  << " coarseN="<<coarseN<<" refineN="<<refineN
                  << " Ecm=["<<std::setprecision(17)<<E0<<","<<E1<<"]\n";

        V11_STAGE("main: adaptive_final_zeros_device_resident");
        adaptive_final_zeros_device_resident(nnP, irrep, irrep_tag, E0, E1, coarseN, refineN, debug);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
