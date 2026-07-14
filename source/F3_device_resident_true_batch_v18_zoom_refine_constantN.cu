// ============================================================================
// F3_device_resident_true_batch_v18_zoom_refine_constantN.cu
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
// v14 TRUE-BATCH DESIGN NOTE
// ----------------------------------------------------------------------------
// This file is the next-step grouped-batch version built from v13.  It keeps the
// CPU/OpenMP config prepass and groups energies by matrix shape.  The code below
// includes same-shape batched LU helpers and the stream-parallel device-resident
// fallback from v13.  The intended true-batch path is:
//   group configs -> device batch arrays -> F2/G/K2 batch kernels -> H batch
//   -> batched LU/GEMM -> F3 batch -> batched LU/GEMM -> dets only.
//
// IMPORTANT IMPLEMENTATION STATUS:
//   - Same-shape grouping and CPU/OpenMP config prepass are active.
//   - Per-energy device-resident fallback is active and stable.
//   - cuBLAS batched-LU helpers are present.
//   - The batch ingredient kernels are provided below as callable building blocks.
//   - Projection/Vsel still needs one eigensolve per energy, so vdim is detected
//     before batched F3 solves.  This is unavoidable unless projections are also
//     rewritten as a batched eigensolver path.
//
// This version is intentionally guarded: if vdim or a batch precondition is not
// uniform, it falls back to v13's per-stream device-resident execution rather
// than silently doing a wrong batched solve.
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
#include <future>
#include <omp.h>
#include <map>
#include <mutex>
#include <thread>
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

#include "functions.h"                 // CPU/OpenMP config_maker_4 prepass
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


// ============================================================================
// v14 same-shape batch ingredient kernels
// These kernels are written against the device scalar functions already present
// in F2_gpu_safe_builder.cuh, G_gpu_safe_builder.cuh, and K2_functions_gpu_safe.cuh.
// Storage convention:
//   F2/H/F3/X batches use column-major matrices, contiguous by energy:
//      A_batch[e*n*n + col*n + row]
//   G/K helper kernels write directly into column-major output for H compatibility.
// ============================================================================

__global__ void v14_build_F2_batch_same_shape_kernel(
    cuDoubleComplex* F2_batch_col,
    int batchCount,
    int n,
    int size1,
    int size2,
    const double* En_batch,
    const double* p1x, const double* p1y, const double* p1z,
    const int* ell1, const int* m1cfg,
    const double* p2x, const double* p2y, const double* p2z,
    const int* ell2, const int* m2cfg,
    double Px, double Py, double Pz,
    double mK, double mpi,
    double L,
    double alpha,
    double epsilon_h,
    int max_shell_num,
    bool Q0norm,
    int erfi_max_terms,
    double erfi_tol)
{
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long elems_per_E = static_cast<long long>(n) * n;
    const long long total = elems_per_E * batchCount;
    if (tid >= total) return;

    const int e = static_cast<int>(tid / elems_per_E);
    const int loc = static_cast<int>(tid - static_cast<long long>(e) * elems_per_E);
    const int row = loc % n;
    const int col = loc / n;
    cuDoubleComplex En = make_cuDoubleComplex(En_batch[e], 0.0);

    const int off1 = e * size1;
    const int off2 = e * size2;
    cuDoubleComplex val = make_cuDoubleComplex(0.0, 0.0);

    if (row < size1 && col < size1)
    {
        val = f2gpu::F2_ang_mom_d(
            En,
            p1x[off1+row], p1y[off1+row], p1z[off1+row], ell1[off1+row], m1cfg[off1+row],
            p1x[off1+col], p1y[off1+col], p1z[off1+col], ell1[off1+col], m1cfg[off1+col],
            Px, Py, Pz,
            mK, mK, mpi,
            L, alpha, epsilon_h,
            max_shell_num, Q0norm, erfi_max_terms, erfi_tol);
    }
    else if (row >= size1 && col >= size1)
    {
        const int r2 = row - size1;
        const int c2 = col - size1;
        val = f2gpu::F2_ang_mom_d(
            En,
            p2x[off2+r2], p2y[off2+r2], p2z[off2+r2], ell2[off2+r2], m2cfg[off2+r2],
            p2x[off2+c2], p2y[off2+c2], p2z[off2+c2], ell2[off2+c2], m2cfg[off2+c2],
            Px, Py, Pz,
            mpi, mK, mK,
            L, alpha, epsilon_h,
            max_shell_num, Q0norm, erfi_max_terms, erfi_tol);
    }

    F2_batch_col[tid] = val;
}

__global__ void v14_build_G_batch_same_shape_kernel(
    cuDoubleComplex* G_batch_col,
    int batchCount,
    int n,
    int size1,
    int size2,
    const double* En_batch,
    const ggpu::ConfigEntry* plm_batch,
    const ggpu::ConfigEntry* klm_batch,
    ggpu::Vec3d P,
    double m1,
    double m2,
    double L,
    double epsilon_h,
    bool Q0norm)
{
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long elems_per_E = static_cast<long long>(n) * n;
    const long long total = elems_per_E * batchCount;
    if (tid >= total) return;

    const int e = static_cast<int>(tid / elems_per_E);
    const int loc = static_cast<int>(tid - static_cast<long long>(e) * elems_per_E);
    const int row = loc % n;
    const int col = loc / n;
    const int off1 = e * size1;
    const int off2 = e * size2;

    ggpu::GpuComplex En{En_batch[e], 0.0};
    ggpu::GpuComplex val = ggpu::make_c(0.0);

    if (row < size1 && col < size1)
    {
        val = ggpu::G_ij_lm_gpu(En, plm_batch[off1+row], plm_batch[off1+col], P,
                                m1, m1, m2, L, epsilon_h, Q0norm);
    }
    else if (row < size1 && col >= size1)
    {
        const int c2 = col - size1;
        val = ggpu::G_ij_lm_gpu(En, plm_batch[off1+row], klm_batch[off2+c2], P,
                                m1, m2, m1, L, epsilon_h, Q0norm);
        const double projector = (plm_batch[off1+row].ell % 2 == 0) ? 1.0 : -1.0;
        val = ggpu::cmul_d(val, 1.4142135623730950488 * projector);
    }
    else if (row >= size1 && col < size1)
    {
        const int r2 = row - size1;
        val = ggpu::G_ij_lm_gpu(En, klm_batch[off2+r2], plm_batch[off1+col], P,
                                m2, m1, m1, L, epsilon_h, Q0norm);
        const double projector = (plm_batch[off1+col].ell % 2 == 0) ? 1.0 : -1.0;
        val = ggpu::cmul_d(val, 1.4142135623730950488 * projector);
    }

    G_batch_col[static_cast<long long>(e)*elems_per_E + static_cast<long long>(col)*n + row] =
        make_cuDoubleComplex(val.x, val.y);
}

__global__ void v14_build_K2_batch_same_shape_kernel(
    cuDoubleComplex* K_batch_col,
    int batchCount,
    int n,
    int size1,
    int size2,
    const double* En_batch,
    const double* p1x, const double* p1y, const double* p1z,
    const int* ell1, const int* m1cfg,
    const double* p2x, const double* p2y, const double* p2z,
    const int* ell2, const int* m2cfg,
    double Px, double Py, double Pz,
    double eta_1,
    double eta_2,
    k2gpu::ScatterParamsView sp1,
    k2gpu::ScatterParamsView sp2,
    double m1,
    double m2,
    double epsilon_h,
    double L)
{
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long elems_per_E = static_cast<long long>(n) * n;
    const long long total = elems_per_E * batchCount;
    if (tid >= total) return;

    const int e = static_cast<int>(tid / elems_per_E);
    const int loc = static_cast<int>(tid - static_cast<long long>(e) * elems_per_E);
    const int row = loc % n;
    const int col = loc / n;
    const int off1 = e * size1;
    const int off2 = e * size2;

    k2gpu::Cx val(0.0,0.0);
    k2gpu::ConfigView cfg1{size1, p1x+off1, p1y+off1, p1z+off1, ell1+off1, m1cfg+off1};
    k2gpu::ConfigView cfg2{size2, p2x+off2, p2y+off2, p2z+off2, ell2+off2, m2cfg+off2};

    if (row < size1 && col < size1 && row == col)
    {
        val = k2gpu::K2inv_diag_element_gpu(En_batch[e], cfg1, row,
                                            Px, Py, Pz,
                                            eta_1, sp1,
                                            m1, m1, m2,
                                            epsilon_h, L);
    }
    else if (row >= size1 && col >= size1)
    {
        const int r2 = row - size1;
        const int c2 = col - size1;
        if (r2 == c2)
        {
            val = 2.0 * k2gpu::K2inv_diag_element_gpu(En_batch[e], cfg2, r2,
                                                       Px, Py, Pz,
                                                       eta_2, sp2,
                                                       m2, m1, m1,
                                                       epsilon_h, L);
        }
    }

    K_batch_col[static_cast<long long>(e)*elems_per_E + static_cast<long long>(col)*n + row] =
        make_cuDoubleComplex(val.re, val.im);
}

__global__ void v14_combine_H_batch_kernel(
    cuDoubleComplex* H_batch,
    const cuDoubleComplex* F2_batch,
    const cuDoubleComplex* G_batch,
    const cuDoubleComplex* K_batch,
    long long total_elems)
{
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total_elems) return;
    const cuDoubleComplex a = F2_batch[tid];
    const cuDoubleComplex b = G_batch[tid];
    const cuDoubleComplex c = K_batch[tid];
    H_batch[tid] = make_cuDoubleComplex(cuCreal(a)+cuCreal(b)+cuCreal(c),
                                        cuCimag(a)+cuCimag(b)+cuCimag(c));
}

__global__ void v14_make_ptrs_kernel(
    cuDoubleComplex** ptrs,
    cuDoubleComplex* base,
    int stride,
    int batchCount)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batchCount) ptrs[i] = base + static_cast<long long>(i) * stride;
}

__global__ void v14_build_F3_batch_kernel(
    cuDoubleComplex* F3_batch,
    const cuDoubleComplex* F2_batch,
    const cuDoubleComplex* F2X1_batch,
    long long total_elems)
{
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total_elems) return;
    const cuDoubleComplex a = F2_batch[tid];
    const cuDoubleComplex b = F2X1_batch[tid];
    F3_batch[tid] = make_cuDoubleComplex((1.0/3.0)*cuCreal(a)-cuCreal(b),
                                         (1.0/3.0)*cuCimag(a)-cuCimag(b));
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
    // The earlier version used cublasZgetrfBatched/getrsBatched with batchCount=1.
    // On some CUDA/HPC-SDK libcublas builds that path segfaulted inside
    // cublasZgetrsBatched.  For a single matrix solve, cuSOLVER dense LU is the
    // safer and more appropriate path.
    (void)handle;

    cusolverDnHandle_t solver = nullptr;
    V11_CUSOLVER_CHECK(cusolverDnCreate(&solver));

    int lwork = 0;
    V11_CUSOLVER_CHECK(cusolverDnZgetrf_bufferSize(
        solver, n, n, dA_lu, n, &lwork));

    DevicePtr<cuDoubleComplex> work(static_cast<size_t>(lwork));
    DevicePtr<int> piv(static_cast<size_t>(n));
    DevicePtr<int> info(1);

    V11_CUSOLVER_CHECK(cusolverDnZgetrf(
        solver, n, n, dA_lu, n, work.p, piv.p, info.p));

    int hinfo = 0;
    V11_CUDA_CHECK(cudaMemcpy(&hinfo, info.p, sizeof(int), cudaMemcpyDeviceToHost));
    if (hinfo != 0)
    {
        V11_CUSOLVER_CHECK(cusolverDnDestroy(solver));
        throw std::runtime_error("cuSOLVER LU factorization failed, info=" + std::to_string(hinfo));
    }

    V11_CUSOLVER_CHECK(cusolverDnZgetrs(
        solver, CUBLAS_OP_N, n, nrhs, dA_lu, n, piv.p, dB_inout, n, info.p));

    V11_CUDA_CHECK(cudaMemcpy(&hinfo, info.p, sizeof(int), cudaMemcpyDeviceToHost));
    V11_CUSOLVER_CHECK(cusolverDnDestroy(solver));

    if (hinfo != 0)
    {
        throw std::runtime_error("cuSOLVER LU solve failed, info=" + std::to_string(hinfo));
    }
}

static comp determinant_colmajor_device_matrix(cublasHandle_t handle, const cuDoubleComplex* dA, int n)
{
    // Single-matrix determinant via cuSOLVER LU. Avoid cublas getrfBatched here
    // for the same reason as the linear solve above.
    (void)handle;
    if (n <= 0) return comp(std::numeric_limits<double>::quiet_NaN(), 0.0);

    DevicePtr<cuDoubleComplex> dLU(static_cast<size_t>(n) * static_cast<size_t>(n));
    V11_CUDA_CHECK(cudaMemcpy(dLU.p, dA,
                              sizeof(cuDoubleComplex) * static_cast<size_t>(n) * static_cast<size_t>(n),
                              cudaMemcpyDeviceToDevice));

    cusolverDnHandle_t solver = nullptr;
    V11_CUSOLVER_CHECK(cusolverDnCreate(&solver));

    int lwork = 0;
    V11_CUSOLVER_CHECK(cusolverDnZgetrf_bufferSize(solver, n, n, dLU.p, n, &lwork));

    DevicePtr<cuDoubleComplex> work(static_cast<size_t>(lwork));
    DevicePtr<int> piv(static_cast<size_t>(n));
    DevicePtr<int> info(1);

    V11_CUSOLVER_CHECK(cusolverDnZgetrf(solver, n, n, dLU.p, n, work.p, piv.p, info.p));

    int hinfo = 0;
    V11_CUDA_CHECK(cudaMemcpy(&hinfo, info.p, sizeof(int), cudaMemcpyDeviceToHost));
    V11_CUSOLVER_CHECK(cusolverDnDestroy(solver));

    if (hinfo != 0)
    {
        return comp(std::numeric_limits<double>::quiet_NaN(), 0.0);
    }

    std::vector<cuDoubleComplex> hLU(static_cast<size_t>(n) * static_cast<size_t>(n));
    std::vector<int> hpiv(static_cast<size_t>(n));

    V11_CUDA_CHECK(cudaMemcpy(hLU.data(), dLU.p,
                              sizeof(cuDoubleComplex) * hLU.size(),
                              cudaMemcpyDeviceToHost));
    V11_CUDA_CHECK(cudaMemcpy(hpiv.data(), piv.p,
                              sizeof(int) * hpiv.size(),
                              cudaMemcpyDeviceToHost));

    comp det(1.0, 0.0);
    int sign = 1;

    for (int i = 0; i < n; ++i)
    {
        const cuDoubleComplex z = hLU[static_cast<size_t>(i) + static_cast<size_t>(i) * static_cast<size_t>(n)];
        det *= comp(cuCreal(z), cuCimag(z));

        // cuSOLVER pivots are 1-based, same LAPACK convention.
        if (hpiv[static_cast<size_t>(i)] != i + 1)
        {
            sign = -sign;
        }
    }

    return static_cast<double>(sign) * det;
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

struct EnergyShapeKey
{
    int dim1 = 0;
    int dim2 = 0;
    int total_dim = 0;
    int vdim = -1; // vdim is produced during actual device-resident solve in this practical v12.

    bool operator<(const EnergyShapeKey& o) const
    {
        if (dim1 != o.dim1) return dim1 < o.dim1;
        if (dim2 != o.dim2) return dim2 < o.dim2;
        if (total_dim != o.total_dim) return total_dim < o.total_dim;
        return vdim < o.vdim;
    }
};

struct EnergyPrepassInfo
{
    int i = -1;
    double Ecm = std::numeric_limits<double>::quiet_NaN();
    double En = std::numeric_limits<double>::quiet_NaN();
    int dim1 = 0;
    int dim2 = 0;
    int total_dim = 0;
    int vdim = -1;
};

static EnergyPrepassInfo get_energy_shape_prepass(
    int idx,
    double Ecm,
    const std::array<int,3>& nnP,
    const PhysicsParams& par,
    char debug)
{
    // v13 prepass intentionally uses CPU config_maker_4.
    // This is cheaper than building all matrices and gives the exact deterministic
    // config dimensions used by the reference CPU code. The outer prepass loop is
    // OpenMP-parallelized in run_scan_device_resident().
    EnergyPrepassInfo info;
    info.i = idx;
    info.Ecm = Ecm;

    const double L = par.xi * par.Lbyas;
    const double pi = std::acos(-1.0);
    const double twopibyL = 2.0*pi/L;

    std::vector<comp> total_P(3);
    total_P[0] = comp(twopibyL * nnP[0], 0.0);
    total_P[1] = comp(twopibyL * nnP[1], 0.0);
    total_P[2] = comp(twopibyL * nnP[2], 0.0);

    const double P2 = std::norm(total_P[0]) + std::norm(total_P[1]) + std::norm(total_P[2]);
    const double En_real = std::sqrt(Ecm*Ecm + P2);
    info.En = En_real;

    std::vector<int> waves_vec_1 = {0,1};
    std::vector<int> waves_vec_2 = {0};

    std::vector<std::vector<comp>> plm_config(5), klm_config(5);
    std::vector<std::vector<int>> np_config(5), nk_config(5);

    config_maker_4(plm_config, np_config, waves_vec_1, comp(En_real,0.0), total_P,
                   par.atmK, par.atmK, par.atmpi, L, par.epsilon_h,
                   par.max_shell_num, par.tolerance);

    config_maker_4(klm_config, nk_config, waves_vec_2, comp(En_real,0.0), total_P,
                   par.atmpi, par.atmK, par.atmK, L, par.epsilon_h,
                   par.max_shell_num, par.tolerance);

    info.dim1 = static_cast<int>(plm_config[0].size());
    info.dim2 = static_cast<int>(klm_config[0].size());
    info.total_dim = info.dim1 + info.dim2;

    if (debug == 'y')
    {
        std::cout << "[v13 CPU prepass] i=" << idx
                  << " Ecm=" << std::setprecision(17) << Ecm
                  << " dim1=" << info.dim1
                  << " dim2=" << info.dim2
                  << " total_dim=" << info.total_dim << "\n";
    }

    return info;
}


// ----------------------------------------------------------------------------
// v16 custom batched GETRS replacement.
//
// Your runs show that cublasZgetrsBatched segfaults inside libcublas.so.13 even
// after cublasZgetrfBatched succeeds.  To keep the same true-batch factorization
// path while avoiding that unstable routine, v16 uses:
//
//      cublasZgetrfBatched  for LU factorization of all same-n matrices
//      this CUDA kernel     for applying pivots + triangular solves
//
// Kernel layout:
//      blockIdx.x = batch id
//      blockIdx.y = RHS column id
//      one thread solves one RHS column serially in n
//
// This is not the fastest possible triangular-solve kernel, but for your current
// n ~ 28 and nrhs <= n it is much safer and still parallelizes over
// batch_count * nrhs independent RHS columns.
// ----------------------------------------------------------------------------
__device__ __forceinline__ cuDoubleComplex v16_zadd(cuDoubleComplex a, cuDoubleComplex b)
{
    return make_cuDoubleComplex(cuCreal(a) + cuCreal(b), cuCimag(a) + cuCimag(b));
}

__device__ __forceinline__ cuDoubleComplex v16_zsub(cuDoubleComplex a, cuDoubleComplex b)
{
    return make_cuDoubleComplex(cuCreal(a) - cuCreal(b), cuCimag(a) - cuCimag(b));
}

__device__ __forceinline__ cuDoubleComplex v16_zmul(cuDoubleComplex a, cuDoubleComplex b)
{
    const double ar = cuCreal(a), ai = cuCimag(a);
    const double br = cuCreal(b), bi = cuCimag(b);
    return make_cuDoubleComplex(ar*br - ai*bi, ar*bi + ai*br);
}

__device__ __forceinline__ cuDoubleComplex v16_zdiv(cuDoubleComplex a, cuDoubleComplex b)
{
    const double ar = cuCreal(a), ai = cuCimag(a);
    const double br = cuCreal(b), bi = cuCimag(b);
    const double den = br*br + bi*bi;
    return make_cuDoubleComplex((ar*br + ai*bi)/den, (ai*br - ar*bi)/den);
}

__global__ void v16_getrs_from_getrf_batched_kernel(
    const cuDoubleComplex* const* A_lu_array,
    cuDoubleComplex** B_array,
    const int* pivots,
    int n,
    int nrhs,
    int batch_count)
{
    const int b = blockIdx.x;
    const int rhs = blockIdx.y;

    if (b >= batch_count || rhs >= nrhs) return;

    const cuDoubleComplex* A = A_lu_array[b];
    cuDoubleComplex* B = B_array[b];
    int* piv = const_cast<int*>(pivots + b*n);

    // B is n x nrhs, column-major. Work directly in the RHS column.
    cuDoubleComplex* x = B + rhs*n;

    // Apply row pivots P to B. cuBLAS/LAPACK pivots are 1-based.
    for (int k = 0; k < n; ++k)
    {
        int pk = piv[k] - 1;
        if (pk != k && pk >= 0 && pk < n)
        {
            cuDoubleComplex tmp = x[k];
            x[k] = x[pk];
            x[pk] = tmp;
        }
    }

    // Forward solve L*y = P*b. L has unit diagonal, stored below diagonal in A.
    for (int i = 0; i < n; ++i)
    {
        cuDoubleComplex sum = x[i];
        for (int j = 0; j < i; ++j)
        {
            const cuDoubleComplex Lij = A[i + j*n];
            sum = v16_zsub(sum, v16_zmul(Lij, x[j]));
        }
        x[i] = sum;
    }

    // Back solve U*x = y. U is diagonal+upper triangle in A.
    for (int i = n - 1; i >= 0; --i)
    {
        cuDoubleComplex sum = x[i];
        for (int j = i + 1; j < n; ++j)
        {
            const cuDoubleComplex Uij = A[i + j*n];
            sum = v16_zsub(sum, v16_zmul(Uij, x[j]));
        }
        const cuDoubleComplex Uii = A[i + i*n];
        x[i] = v16_zdiv(sum, Uii);
    }
}

// ----------------------------------------------------------------------------
// v13 cuBLAS batched-LU building block.
//
// This helper is intentionally separated from the per-stream cuSOLVER fallback.
// It is safe only for groups with the same n and batch_count >= 2. The earlier
// v11 crash came from cublasZgetrsBatched with batchCount=1, so v13 never uses
// this helper for batch_count < 2.
// ----------------------------------------------------------------------------
static void cublas_lu_solve_inplace_batched_same_n(
    cublasHandle_t handle,
    int n,
    int nrhs,
    std::vector<cuDoubleComplex*>& A_lu_ptrs,
    std::vector<cuDoubleComplex*>& B_inout_ptrs,
    char debug)
{
    const int batch_count = static_cast<int>(A_lu_ptrs.size());
    if (batch_count != static_cast<int>(B_inout_ptrs.size()))
        throw std::runtime_error("batched LU: A/B batch size mismatch");
    if (batch_count <= 0) return;
    if (batch_count == 1)
    {
        device_lu_solve_inplace(handle, n, nrhs, A_lu_ptrs[0], B_inout_ptrs[0]);
        return;
    }

    DevicePtr<cuDoubleComplex*> dA(static_cast<size_t>(batch_count));
    DevicePtr<cuDoubleComplex*> dB(static_cast<size_t>(batch_count));
    DevicePtr<int> piv(static_cast<size_t>(n) * static_cast<size_t>(batch_count));
    DevicePtr<int> info(static_cast<size_t>(batch_count));

    V11_CUDA_CHECK(cudaMemcpy(dA.p, A_lu_ptrs.data(), sizeof(cuDoubleComplex*)*static_cast<size_t>(batch_count), cudaMemcpyHostToDevice));
    V11_CUDA_CHECK(cudaMemcpy(dB.p, B_inout_ptrs.data(), sizeof(cuDoubleComplex*)*static_cast<size_t>(batch_count), cudaMemcpyHostToDevice));

    if (debug == 'y')
    {
        std::cout << "[v13 batched LU] getrf/getrs n=" << n
                  << " nrhs=" << nrhs
                  << " batch=" << batch_count << "\n";
    }

    V11_CUBLAS_CHECK(cublasZgetrfBatched(handle, n, dA.p, n, piv.p, info.p, batch_count));

    std::vector<int> hinfo(static_cast<size_t>(batch_count), 0);
    V11_CUDA_CHECK(cudaMemcpy(hinfo.data(), info.p, sizeof(int)*hinfo.size(), cudaMemcpyDeviceToHost));
    for (int b=0; b<batch_count; ++b)
    {
        if (hinfo[static_cast<size_t>(b)] != 0)
            throw std::runtime_error("batched LU getrf failed at batch " + std::to_string(b) +
                                     " info=" + std::to_string(hinfo[static_cast<size_t>(b)]));
    }

    // v16: avoid cublasZgetrsBatched, which segfaulted on this system.
    // Solve all RHS columns using a custom CUDA kernel from the LU factors and pivots.
    if (debug == 'y')
    {
        std::cout << "[v16 custom GETRS] launching pivot+triangular solve kernel"
                  << " n=" << n << " nrhs=" << nrhs
                  << " batch=" << batch_count << "\n";
    }

    dim3 grid(static_cast<unsigned int>(batch_count), static_cast<unsigned int>(nrhs), 1);
    v16_getrs_from_getrf_batched_kernel<<<grid, 1>>>(
        const_cast<const cuDoubleComplex* const*>(dA.p),
        dB.p,
        piv.p,
        n,
        nrhs,
        batch_count
    );
    V11_CUDA_CHECK(cudaGetLastError());
    V11_CUDA_CHECK(cudaDeviceSynchronize());
}


// ----------------------------------------------------------------------------
// v15 true-batch same-shape execution path.
//
// This path is intentionally loud: every group prints whether it enters the
// batch path or why it falls back.  It rebuilds CPU configs for the group once,
// uploads them as contiguous [energy][state] arrays, builds F2/G/K2 in true
// batch kernels, constructs Vsel per energy on the GPU, and then uses cuBLAS
// batched LU/GEMM for the two F3 solves.  Only determinant results are copied
// back at the end.
// ----------------------------------------------------------------------------
static std::vector<DeviceResidentResult> process_same_shape_group_true_batch_v15(
    const std::vector<EnergyPrepassInfo>& group,
    const std::array<int,3>& nnP,
    const std::string& irrep,
    const std::string& irrep_tag,
    const PhysicsParams& par,
    char debug)
{
    (void)irrep_tag;
    if (group.empty()) return {};

    const int B = static_cast<int>(group.size());
    const int dim1 = group[0].dim1;
    const int dim2 = group[0].dim2;
    const int n = group[0].total_dim;

    std::vector<DeviceResidentResult> out(static_cast<std::size_t>(B));

    if (B < 2) throw std::runtime_error("v15 true batch rejected: batch count < 2");
    if (n <= 0 || dim1 <= 0 || dim2 <= 0) throw std::runtime_error("v15 true batch rejected: non-positive dimensions");
    for (const auto& x : group) {
        if (x.dim1 != dim1 || x.dim2 != dim2 || x.total_dim != n)
            throw std::runtime_error("v15 true batch rejected: group is not same-shape");
    }

    const double pi = std::acos(-1.0);
    const double L = par.xi * par.Lbyas;
    const double twopibyL = 2.0*pi/L;
    std::vector<comp> total_P(3);
    total_P[0] = comp(twopibyL * nnP[0], 0.0);
    total_P[1] = comp(twopibyL * nnP[1], 0.0);
    total_P[2] = comp(twopibyL * nnP[2], 0.0);

    std::cout << "[v18 true batch] attempting group dim1=" << dim1
              << " dim2=" << dim2 << " n=" << n
              << " batch=" << B << " irrep=" << irrep << "\n";

    auto tbuild0 = std::chrono::high_resolution_clock::now();

    // ---------------- CPU/OpenMP config rebuild for contiguous batch upload ----------------
    std::vector<std::vector<std::vector<comp>>> plm_all(static_cast<std::size_t>(B));
    std::vector<std::vector<std::vector<comp>>> klm_all(static_cast<std::size_t>(B));
    std::vector<std::vector<std::vector<int>>> np_all(static_cast<std::size_t>(B));
    std::vector<std::vector<std::vector<int>>> nk_all(static_cast<std::size_t>(B));

    std::vector<int> waves_vec_1 = {0,1};
    std::vector<int> waves_vec_2 = {0};

    #pragma omp parallel for schedule(dynamic)
    for (int e=0; e<B; ++e) {
        plm_all[static_cast<std::size_t>(e)] = std::vector<std::vector<comp>>(5);
        klm_all[static_cast<std::size_t>(e)] = std::vector<std::vector<comp>>(5);
        np_all[static_cast<std::size_t>(e)]  = std::vector<std::vector<int>>(5);
        nk_all[static_cast<std::size_t>(e)]  = std::vector<std::vector<int>>(5);
        const double En_real = group[static_cast<std::size_t>(e)].En;
        config_maker_4(plm_all[static_cast<std::size_t>(e)], np_all[static_cast<std::size_t>(e)], waves_vec_1,
                       comp(En_real,0.0), total_P,
                       par.atmK, par.atmK, par.atmpi, L, par.epsilon_h,
                       par.max_shell_num, par.tolerance);
        config_maker_4(klm_all[static_cast<std::size_t>(e)], nk_all[static_cast<std::size_t>(e)], waves_vec_2,
                       comp(En_real,0.0), total_P,
                       par.atmpi, par.atmK, par.atmK, L, par.epsilon_h,
                       par.max_shell_num, par.tolerance);
    }

    for (int e=0; e<B; ++e) {
        if ((int)plm_all[e][0].size() != dim1 || (int)klm_all[e][0].size() != dim2) {
            throw std::runtime_error("v15 true batch rejected: rebuilt config dimensions differ from prepass at local batch " + std::to_string(e));
        }
    }

    // Host contiguous config arrays: [energy][state]
    std::vector<double> hEn(static_cast<std::size_t>(B));
    std::vector<double> p1x(static_cast<std::size_t>(B)*dim1), p1y(p1x.size()), p1z(p1x.size());
    std::vector<int>    e1l(static_cast<std::size_t>(B)*dim1), e1m(e1l.size());
    std::vector<double> p2x(static_cast<std::size_t>(B)*dim2), p2y(p2x.size()), p2z(p2x.size());
    std::vector<int>    e2l(static_cast<std::size_t>(B)*dim2), e2m(e2l.size());
    std::vector<ggpu::ConfigEntry> g1(static_cast<std::size_t>(B)*dim1), g2(static_cast<std::size_t>(B)*dim2);

    for (int e=0; e<B; ++e) {
        hEn[static_cast<std::size_t>(e)] = group[static_cast<std::size_t>(e)].En;
        for (int i=0; i<dim1; ++i) {
            const std::size_t off = static_cast<std::size_t>(e)*dim1 + i;
            p1x[off]=plm_all[e][0][i].real(); p1y[off]=plm_all[e][1][i].real(); p1z[off]=plm_all[e][2][i].real();
            e1l[off]=(int)std::llround(plm_all[e][3][i].real()); e1m[off]=(int)std::llround(plm_all[e][4][i].real());
            g1[off] = ggpu::ConfigEntry{p1x[off], p1y[off], p1z[off], e1l[off], e1m[off]};
        }
        for (int i=0; i<dim2; ++i) {
            const std::size_t off = static_cast<std::size_t>(e)*dim2 + i;
            p2x[off]=klm_all[e][0][i].real(); p2y[off]=klm_all[e][1][i].real(); p2z[off]=klm_all[e][2][i].real();
            e2l[off]=(int)std::llround(klm_all[e][3][i].real()); e2m[off]=(int)std::llround(klm_all[e][4][i].real());
            g2[off] = ggpu::ConfigEntry{p2x[off], p2y[off], p2z[off], e2l[off], e2m[off]};
        }
    }

    // ---------------- Upload batch config arrays ----------------
    DevicePtr<double> dEn(static_cast<std::size_t>(B));
    DevicePtr<double> dp1x(p1x.size()), dp1y(p1y.size()), dp1z(p1z.size());
    DevicePtr<int>    de1l(e1l.size()), de1m(e1m.size());
    DevicePtr<double> dp2x(p2x.size()), dp2y(p2y.size()), dp2z(p2z.size());
    DevicePtr<int>    de2l(e2l.size()), de2m(e2m.size());
    DevicePtr<ggpu::ConfigEntry> dg1(g1.size()), dg2(g2.size());

    V11_CUDA_CHECK(cudaMemcpy(dEn.p, hEn.data(), sizeof(double)*hEn.size(), cudaMemcpyHostToDevice));
    V11_CUDA_CHECK(cudaMemcpy(dp1x.p, p1x.data(), sizeof(double)*p1x.size(), cudaMemcpyHostToDevice));
    V11_CUDA_CHECK(cudaMemcpy(dp1y.p, p1y.data(), sizeof(double)*p1y.size(), cudaMemcpyHostToDevice));
    V11_CUDA_CHECK(cudaMemcpy(dp1z.p, p1z.data(), sizeof(double)*p1z.size(), cudaMemcpyHostToDevice));
    V11_CUDA_CHECK(cudaMemcpy(de1l.p, e1l.data(), sizeof(int)*e1l.size(), cudaMemcpyHostToDevice));
    V11_CUDA_CHECK(cudaMemcpy(de1m.p, e1m.data(), sizeof(int)*e1m.size(), cudaMemcpyHostToDevice));
    V11_CUDA_CHECK(cudaMemcpy(dp2x.p, p2x.data(), sizeof(double)*p2x.size(), cudaMemcpyHostToDevice));
    V11_CUDA_CHECK(cudaMemcpy(dp2y.p, p2y.data(), sizeof(double)*p2y.size(), cudaMemcpyHostToDevice));
    V11_CUDA_CHECK(cudaMemcpy(dp2z.p, p2z.data(), sizeof(double)*p2z.size(), cudaMemcpyHostToDevice));
    V11_CUDA_CHECK(cudaMemcpy(de2l.p, e2l.data(), sizeof(int)*e2l.size(), cudaMemcpyHostToDevice));
    V11_CUDA_CHECK(cudaMemcpy(de2m.p, e2m.data(), sizeof(int)*e2m.size(), cudaMemcpyHostToDevice));
    V11_CUDA_CHECK(cudaMemcpy(dg1.p, g1.data(), sizeof(ggpu::ConfigEntry)*g1.size(), cudaMemcpyHostToDevice));
    V11_CUDA_CHECK(cudaMemcpy(dg2.p, g2.data(), sizeof(ggpu::ConfigEntry)*g2.size(), cudaMemcpyHostToDevice));

    auto tconfig_done = std::chrono::high_resolution_clock::now();

    // ---------------- Build/reuse Vsel once per same-shape group ----------------
    // For a fixed (dim1, dim2, total_dim) and irrep, the projection basis is independent
    // of the energy value.  Reusing Vsel avoids B repeated cuSOLVER eigensolves.
    auto tvsel0 = std::chrono::high_resolution_clock::now();
    int vdim = 0;
    cuDoubleComplex* dVsel_one_raw = nullptr;
    build_projection_Vsel_device(dVsel_one_raw, vdim, n, dim1,
                                 np_all[0], nk_all[0], irrep,
                                 nnP[0], nnP[1], nnP[2], par.parity,
                                 0.05, 1.0e-14, 256, debug);
    if (vdim <= 0) throw std::runtime_error("v17 true batch rejected: Vsel reuse produced vdim <= 0");

    DevicePtr<cuDoubleComplex> dVsel_one;
    dVsel_one.p = dVsel_one_raw;
    dVsel_one.n = static_cast<std::size_t>(n) * static_cast<std::size_t>(vdim);

    DevicePtr<cuDoubleComplex> dVselBatch(static_cast<std::size_t>(B)*n*vdim);
    for (int e=0; e<B; ++e) {
        V11_CUDA_CHECK(cudaMemcpy(dVselBatch.p + static_cast<std::size_t>(e)*n*vdim,
                                  dVsel_one.p,
                                  sizeof(cuDoubleComplex)*static_cast<std::size_t>(n)*vdim,
                                  cudaMemcpyDeviceToDevice));
    }
    auto tvsel1 = std::chrono::high_resolution_clock::now();

    if (debug == 'y') {
        std::cout << "[v17 Vsel reuse] group dim1=" << dim1
                  << " dim2=" << dim2 << " n=" << n
                  << " vdim=" << vdim << " batch=" << B
                  << " built_once_and_reused=yes\n";
    }

    // ---------------- Build F2/G/K2/H batches ----------------
    auto tingred0 = std::chrono::high_resolution_clock::now();
    const long long elems_per_E = static_cast<long long>(n)*n;
    const long long total_elems = elems_per_E * B;
    const int threads = 256;
    const int blocks = static_cast<int>((total_elems + threads - 1)/threads);

    DevicePtr<cuDoubleComplex> dF2(static_cast<std::size_t>(total_elems));
    DevicePtr<cuDoubleComplex> dG(static_cast<std::size_t>(total_elems));
    DevicePtr<cuDoubleComplex> dK(static_cast<std::size_t>(total_elems));
    DevicePtr<cuDoubleComplex> dH(static_cast<std::size_t>(total_elems));

    v14_build_F2_batch_same_shape_kernel<<<blocks, threads>>>(
        dF2.p, B, n, dim1, dim2, dEn.p,
        dp1x.p, dp1y.p, dp1z.p, de1l.p, de1m.p,
        dp2x.p, dp2y.p, dp2z.p, de2l.p, de2m.p,
        total_P[0].real(), total_P[1].real(), total_P[2].real(),
        par.atmK, par.atmpi, L, par.alpha, par.epsilon_h, (int)par.max_shell_num, par.Q0norm, 100, 1.0e-15);
    V11_CUDA_CHECK(cudaGetLastError());

    ggpu::Vec3d Pgg{total_P[0].real(), total_P[1].real(), total_P[2].real()};
    v14_build_G_batch_same_shape_kernel<<<blocks, threads>>>(
        dG.p, B, n, dim1, dim2, dEn.p, dg1.p, dg2.p, Pgg,
        par.atmK, par.atmpi, L, par.epsilon_h, par.Q0norm);
    V11_CUDA_CHECK(cudaGetLastError());

    k2gpu::DeviceScatterParams dSP1; dSP1.upload(make_scatter_params_1());
    k2gpu::DeviceScatterParams dSP2; dSP2.upload(make_scatter_params_2());
    v14_build_K2_batch_same_shape_kernel<<<blocks, threads>>>(
        dK.p, B, n, dim1, dim2, dEn.p,
        dp1x.p, dp1y.p, dp1z.p, de1l.p, de1m.p,
        dp2x.p, dp2y.p, dp2z.p, de2l.p, de2m.p,
        total_P[0].real(), total_P[1].real(), total_P[2].real(),
        par.eta_1, par.eta_2, dSP1.view(), dSP2.view(),
        par.atmK, par.atmpi, par.epsilon_h, L);
    V11_CUDA_CHECK(cudaGetLastError());

    v14_combine_H_batch_kernel<<<blocks, threads>>>(dH.p, dF2.p, dG.p, dK.p, total_elems);
    V11_CUDA_CHECK(cudaGetLastError());
    V11_CUDA_CHECK(cudaDeviceSynchronize());

    auto tingred1 = std::chrono::high_resolution_clock::now();
    auto tbuild1 = std::chrono::high_resolution_clock::now();
    const double config_rebuild_sec = std::chrono::duration<double>(tconfig_done - tbuild0).count();
    const double vsel_sec_group = std::chrono::duration<double>(tvsel1 - tvsel0).count();
    const double ingredient_sec_group = std::chrono::duration<double>(tingred1 - tingred0).count();
    const double build_sec_group = std::chrono::duration<double>(tbuild1 - tbuild0).count();

    // ---------------- Batched F3 solves ----------------
    auto tsolve0 = std::chrono::high_resolution_clock::now();
    cublasHandle_t cublas = nullptr;
    V11_CUBLAS_CHECK(cublasCreate(&cublas));

    DevicePtr<cuDoubleComplex> dH_LU(static_cast<std::size_t>(total_elems));
    DevicePtr<cuDoubleComplex> dX1(static_cast<std::size_t>(total_elems));
    V11_CUDA_CHECK(cudaMemcpy(dH_LU.p, dH.p, sizeof(cuDoubleComplex)*static_cast<std::size_t>(total_elems), cudaMemcpyDeviceToDevice));
    V11_CUDA_CHECK(cudaMemcpy(dX1.p, dF2.p, sizeof(cuDoubleComplex)*static_cast<std::size_t>(total_elems), cudaMemcpyDeviceToDevice));

    std::vector<cuDoubleComplex*> Hptr(static_cast<std::size_t>(B)), X1ptr(static_cast<std::size_t>(B));
    for (int e=0; e<B; ++e) {
        Hptr[static_cast<std::size_t>(e)]  = dH_LU.p + static_cast<std::size_t>(e)*n*n;
        X1ptr[static_cast<std::size_t>(e)] = dX1.p   + static_cast<std::size_t>(e)*n*n;
    }
    cublas_lu_solve_inplace_batched_same_n(cublas, n, n, Hptr, X1ptr, debug);

    DevicePtr<cuDoubleComplex> dF2X1(static_cast<std::size_t>(total_elems));
    DevicePtr<cuDoubleComplex*> dF2ptr(static_cast<std::size_t>(B)), dX1ptr(static_cast<std::size_t>(B)), dF2X1ptr(static_cast<std::size_t>(B));
    std::vector<cuDoubleComplex*> F2ptr(static_cast<std::size_t>(B)), F2X1ptr(static_cast<std::size_t>(B));
    for (int e=0; e<B; ++e) {
        F2ptr[static_cast<std::size_t>(e)]   = dF2.p   + static_cast<std::size_t>(e)*n*n;
        F2X1ptr[static_cast<std::size_t>(e)] = dF2X1.p + static_cast<std::size_t>(e)*n*n;
    }
    V11_CUDA_CHECK(cudaMemcpy(dF2ptr.p, F2ptr.data(), sizeof(cuDoubleComplex*)*B, cudaMemcpyHostToDevice));
    V11_CUDA_CHECK(cudaMemcpy(dX1ptr.p, X1ptr.data(), sizeof(cuDoubleComplex*)*B, cudaMemcpyHostToDevice));
    V11_CUDA_CHECK(cudaMemcpy(dF2X1ptr.p, F2X1ptr.data(), sizeof(cuDoubleComplex*)*B, cudaMemcpyHostToDevice));
    const cuDoubleComplex alpha1 = make_cuDoubleComplex(1.0,0.0);
    const cuDoubleComplex beta0  = make_cuDoubleComplex(0.0,0.0);
    V11_CUBLAS_CHECK(cublasZgemmBatched(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                        n, n, n, &alpha1,
                                        const_cast<const cuDoubleComplex**>(dF2ptr.p), n,
                                        const_cast<const cuDoubleComplex**>(dX1ptr.p), n,
                                        &beta0, dF2X1ptr.p, n, B));

    DevicePtr<cuDoubleComplex> dF3(static_cast<std::size_t>(total_elems));
    v14_build_F3_batch_kernel<<<blocks, threads>>>(dF3.p, dF2.p, dF2X1.p, total_elems);
    V11_CUDA_CHECK(cudaGetLastError());

    DevicePtr<cuDoubleComplex> dF3_LU(static_cast<std::size_t>(total_elems));
    DevicePtr<cuDoubleComplex> dX2(static_cast<std::size_t>(B)*n*vdim);
    V11_CUDA_CHECK(cudaMemcpy(dF3_LU.p, dF3.p, sizeof(cuDoubleComplex)*static_cast<std::size_t>(total_elems), cudaMemcpyDeviceToDevice));
    V11_CUDA_CHECK(cudaMemcpy(dX2.p, dVselBatch.p, sizeof(cuDoubleComplex)*static_cast<std::size_t>(B)*n*vdim, cudaMemcpyDeviceToDevice));

    std::vector<cuDoubleComplex*> F3ptr(static_cast<std::size_t>(B)), X2ptr(static_cast<std::size_t>(B));
    for (int e=0; e<B; ++e) {
        F3ptr[static_cast<std::size_t>(e)] = dF3_LU.p + static_cast<std::size_t>(e)*n*n;
        X2ptr[static_cast<std::size_t>(e)] = dX2.p    + static_cast<std::size_t>(e)*n*vdim;
    }
    cublas_lu_solve_inplace_batched_same_n(cublas, n, vdim, F3ptr, X2ptr, debug);

    DevicePtr<cuDoubleComplex> dProj(static_cast<std::size_t>(B)*vdim*vdim);
    DevicePtr<cuDoubleComplex*> dVptr(static_cast<std::size_t>(B)), dX2ptr(static_cast<std::size_t>(B)), dProjptr(static_cast<std::size_t>(B));
    std::vector<cuDoubleComplex*> Vptr(static_cast<std::size_t>(B)), Projptr(static_cast<std::size_t>(B));
    for (int e=0; e<B; ++e) {
        Vptr[static_cast<std::size_t>(e)]    = dVselBatch.p + static_cast<std::size_t>(e)*n*vdim;
        Projptr[static_cast<std::size_t>(e)] = dProj.p      + static_cast<std::size_t>(e)*vdim*vdim;
    }
    V11_CUDA_CHECK(cudaMemcpy(dVptr.p, Vptr.data(), sizeof(cuDoubleComplex*)*B, cudaMemcpyHostToDevice));
    V11_CUDA_CHECK(cudaMemcpy(dX2ptr.p, X2ptr.data(), sizeof(cuDoubleComplex*)*B, cudaMemcpyHostToDevice));
    V11_CUDA_CHECK(cudaMemcpy(dProjptr.p, Projptr.data(), sizeof(cuDoubleComplex*)*B, cudaMemcpyHostToDevice));
    V11_CUBLAS_CHECK(cublasZgemmBatched(cublas, CUBLAS_OP_C, CUBLAS_OP_N,
                                        vdim, vdim, n, &alpha1,
                                        const_cast<const cuDoubleComplex**>(dVptr.p), n,
                                        const_cast<const cuDoubleComplex**>(dX2ptr.p), n,
                                        &beta0, dProjptr.p, vdim, B));

    V11_CUDA_CHECK(cudaDeviceSynchronize());
    auto tsolve1 = std::chrono::high_resolution_clock::now();
    const double solve_sec_group = std::chrono::duration<double>(tsolve1 - tsolve0).count();

    for (int e=0; e<B; ++e) {
        DeviceResidentResult r;
        r.i = group[static_cast<std::size_t>(e)].i;
        r.Ecm = comp(group[static_cast<std::size_t>(e)].Ecm,0.0);
        r.En  = comp(group[static_cast<std::size_t>(e)].En,0.0);
        r.dim1 = dim1;
        r.dim2 = dim2;
        r.total_dim = n;
        r.vdim = vdim;
        r.det = determinant_colmajor_device_matrix(cublas, dProj.p + static_cast<std::size_t>(e)*vdim*vdim, vdim);
        r.build_sec = build_sec_group / double(B);
        r.solve_sec = solve_sec_group / double(B);
        out[static_cast<std::size_t>(e)] = r;
    }

    V11_CUBLAS_CHECK(cublasDestroy(cublas));
    std::cout << "[v18 true batch done] dim1=" << dim1
              << " dim2=" << dim2 << " n=" << n
              << " vdim=" << vdim << " batch=" << B
              << " build_group=" << build_sec_group
              << " config_rebuild=" << config_rebuild_sec
              << " Vsel_reuse_build=" << vsel_sec_group
              << " ingredient_build=" << ingredient_sec_group
              << " solve_group=" << solve_sec_group
              << " per_energy_build=" << build_sec_group/double(B)
              << " per_energy_solve=" << solve_sec_group/double(B) << "\n";
    return out;
}

static std::vector<DeviceResidentResult> process_same_shape_group_stream_parallel(
    const std::vector<EnergyPrepassInfo>& group,
    const std::array<int,3>& nnP,
    const std::string& irrep,
    const std::string& irrep_tag,
    const PhysicsParams& par,
    int max_concurrent_streams,
    char debug)
{
    std::vector<DeviceResidentResult> out(group.size());

    if (group.empty()) return out;

    const int concurrency = std::max(1, std::min<int>(max_concurrent_streams, static_cast<int>(group.size())));

    std::mutex print_mutex;
    std::size_t next = 0;

    auto worker = [&](int worker_id)
    {
        // With --default-stream per-thread, kernels launched from each host worker
        // use a separate default stream.  Each worker also owns its own cuBLAS handle.
        cublasHandle_t cublas = nullptr;
        V11_CUBLAS_CHECK(cublasCreate(&cublas));

        while (true)
        {
            std::size_t local = 0;
            {
                std::lock_guard<std::mutex> lock(print_mutex);
                if (next >= group.size()) break;
                local = next++;
                if (debug == 'y')
                {
                    std::cout << "[v13 worker " << worker_id << "] starting grouped energy "
                              << "local=" << local
                              << " i=" << group[local].i
                              << " Ecm=" << std::setprecision(17) << group[local].Ecm
                              << " shape=(" << group[local].dim1 << "," << group[local].dim2 << ")\n";
                }
            }

            out[local] = compute_one_energy_device_resident(
                group[local].i,
                group[local].Ecm,
                nnP,
                irrep,
                irrep_tag,
                par,
                cublas,
                debug
            );

            {
                std::lock_guard<std::mutex> lock(print_mutex);
                std::cout << "[v13 group done] i=" << group[local].i
                          << " Ecm=" << std::setprecision(17) << group[local].Ecm
                          << " det=" << out[local].det
                          << " dim=(" << out[local].dim1 << "," << out[local].dim2 << ")"
                          << " total_dim=" << out[local].total_dim
                          << " vdim=" << out[local].vdim
                          << " build=" << out[local].build_sec
                          << " solve=" << out[local].solve_sec
                          << "\n";
            }
        }

        V11_CUBLAS_CHECK(cublasDestroy(cublas));
    };

    std::vector<std::future<void>> fut;
    fut.reserve(static_cast<std::size_t>(concurrency));

    for (int w=0; w<concurrency; ++w)
    {
        fut.push_back(std::async(std::launch::async, worker, w));
    }

    for (auto& f : fut) f.get();

    return out;
}

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

    const int energy_chunk_size = 512;
    const int max_concurrent_streams = 4;

    std::vector<DetRow> rows(static_cast<std::size_t>(Ecm_points));
    double max_abs = 0.0;

    const double del = (Ecm_points == 1) ? 0.0 : (Ecm_final - Ecm_initial)/double(Ecm_points - 1);

    std::ofstream fraw("det_proj_F3i_raw_" + std::to_string(nnP[0]) + std::to_string(nnP[1]) + std::to_string(nnP[2]) + "_" + irrep_tag + tag + "_L20.dat");
    fraw << "# i Ecm_real Ecm_imag det_real det_imag abs_det build_sec solve_sec total_dim vdim dim1 dim2\n";

    for (int chunk_begin = 0; chunk_begin < Ecm_points; chunk_begin += energy_chunk_size)
    {
        const int chunk_end = std::min(Ecm_points, chunk_begin + energy_chunk_size);

        std::cout << "[v13] prepass chunk [" << chunk_begin << ", " << (chunk_end-1)
                  << "] tag=" << tag << "\n";

        std::vector<EnergyPrepassInfo> prepass;
        prepass.reserve(static_cast<std::size_t>(chunk_end - chunk_begin));

        std::vector<EnergyPrepassInfo> prepass_all(static_cast<std::size_t>(chunk_end - chunk_begin));

        #pragma omp parallel for schedule(dynamic)
        for (int ii = 0; ii < chunk_end - chunk_begin; ++ii)
        {
            const int i = chunk_begin + ii;
            const double Ecm = Ecm_initial + double(i)*del;
            prepass_all[static_cast<std::size_t>(ii)] = get_energy_shape_prepass(i, Ecm, nnP, par, debug);
        }

        for (const auto& info : prepass_all)
        {
            if (info.total_dim <= 0)
            {
                std::cout << "[v13] skipping empty config i=" << info.i
                          << " Ecm=" << std::setprecision(17) << info.Ecm << "\n";

                DetRow row;
                row.i = info.i;
                row.Ecm = info.Ecm;
                row.det_raw = comp(std::numeric_limits<double>::quiet_NaN(), 0.0);
                row.det_norm = row.det_raw;
                rows[static_cast<std::size_t>(info.i)] = row;
                continue;
            }
            prepass.push_back(info);
        }

        std::map<EnergyShapeKey, std::vector<EnergyPrepassInfo>> groups;

        for (const auto& info : prepass)
        {
            EnergyShapeKey key;
            key.dim1 = info.dim1;
            key.dim2 = info.dim2;
            key.total_dim = info.total_dim;
            key.vdim = info.vdim;
            groups[key].push_back(info);
        }

        std::cout << "[v13] chunk [" << chunk_begin << ", " << (chunk_end-1)
                  << "] groups=" << groups.size()
                  << " max_concurrent_streams=" << max_concurrent_streams
                  << " v18_true_batch_attempt=enabled; Vsel_reuse_per_same_shape_group; chunk512; constant_refineN_zoom_window; getrfBatched_plus_custom_getrs_kernel; fallback_prints_reason" << "\n";

        for (const auto& kv : groups)
        {
            const auto& key = kv.first;
            const auto& group = kv.second;

            std::cout << "[v13] processing group dim1=" << key.dim1
                      << " dim2=" << key.dim2
                      << " total_dim=" << key.total_dim
                      << " count=" << group.size() << "\n";

            std::vector<DeviceResidentResult> gout;
            bool used_true_batch = false;
            if (group.size() >= 2)
            {
                std::cout << "[v18] attempting true batch for group dim1=" << key.dim1
                          << " dim2=" << key.dim2
                          << " total_dim=" << key.total_dim
                          << " count=" << group.size() << "\n";
                try
                {
                    gout = process_same_shape_group_true_batch_v15(group, nnP, irrep, irrep_tag, par, debug);
                    used_true_batch = true;
                }
                catch (const std::exception& ex)
                {
                    std::cout << "[v17] true batch rejected/failed for group dim1=" << key.dim1
                              << " dim2=" << key.dim2
                              << " total_dim=" << key.total_dim
                              << " count=" << group.size()
                              << " reason: " << ex.what() << "\n";
                }
            }
            else
            {
                std::cout << "[v17] true batch skipped: count < 2 for group dim1=" << key.dim1
                          << " dim2=" << key.dim2
                          << " total_dim=" << key.total_dim << "\n";
            }

            if (!used_true_batch)
            {
                std::cout << "[v17] fallback stream path active for group dim1=" << key.dim1
                          << " dim2=" << key.dim2
                          << " total_dim=" << key.total_dim
                          << " count=" << group.size() << "\n";
                gout = process_same_shape_group_stream_parallel(
                    group,
                    nnP,
                    irrep,
                    irrep_tag,
                    par,
                    max_concurrent_streams,
                    debug
                );
            }

            for (std::size_t j=0; j<group.size(); ++j)
            {
                const int i = group[j].i;
                DetRow row;
                row.i = i;
                row.Ecm = group[j].Ecm;
                row.det_raw = gout[j].det;
                row.det_norm = row.det_raw;
                rows[static_cast<std::size_t>(i)] = row;

                max_abs = std::max(max_abs, std::abs(gout[j].det));

                fraw << std::setprecision(17)
                     << row.i << '\t' << row.Ecm << '\t' << 0.0 << '\t'
                     << gout[j].det.real() << '\t' << gout[j].det.imag() << '\t'
                     << std::abs(gout[j].det) << '\t'
                     << gout[j].build_sec << '\t'
                     << gout[j].solve_sec << '\t'
                     << gout[j].total_dim << '\t'
                     << gout[j].vdim << '\t'
                     << gout[j].dim1 << '\t'
                     << gout[j].dim2 << '\n';
            }
            fraw.flush();
        }
    }

    fraw.close();

    if (max_abs == 0.0 || !std::isfinite(max_abs)) max_abs = 1.0;
    const double scale = 2.0/max_abs;

    std::ofstream fnorm("det_proj_F3i_normalized_" + std::to_string(nnP[0]) + std::to_string(nnP[1]) + std::to_string(nnP[2]) + "_" + irrep_tag + tag + "_L20.dat");
    fnorm << "# i Ecm_real Ecm_imag det_norm_real det_norm_imag abs_det_norm\n";

    for (auto& row: rows)
    {
        if (std::isfinite(row.det_raw.real()) && std::isfinite(row.det_raw.imag()))
        {
            row.det_norm = row.det_raw * scale;
        }
        else
        {
            row.det_norm = row.det_raw;
        }

        fnorm << std::setprecision(17)
              << row.i << '\t' << row.Ecm << '\t' << 0.0 << '\t'
              << row.det_norm.real() << '\t'
              << row.det_norm.imag() << '\t'
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

struct InwardShapeResult
{
    std::string classification = "ambiguous";
    int flip_left_index = -1;
    int flip_right_index = -1;
    double ezero = std::numeric_limits<double>::quiet_NaN();
    double decreasing_fraction = 0.0;
    double increasing_fraction = 0.0;
    double trend_score = 0.0;
};

static InwardShapeResult classify_inward_shape_result(
    const std::vector<DetRow>& rows,
    double trend_required=0.65,
    double margin_required=0.20)
{
    InwardShapeResult out;

    int N = static_cast<int>(rows.size());
    if (N < 4) {
        out.classification = "ambiguous";
        return out;
    }

    int flip = -1;
    double center = 0.5*double(N-1);
    double best = 1e300;

    // Select the sign flip closest to the middle of the current refinement window.
    // This matches the inward-gap classification idea and avoids chasing edge flips.
    for (int i=0;i+1<N;++i) {
        if (sign_flip(rows[i].det_norm.real(), rows[i+1].det_norm.real())) {
            double d = std::abs((double(i)+0.5) - center);
            if (d < best) { best=d; flip=i; }
        }
    }

    if (flip < 0) {
        out.classification = "no_sign_flip";
        return out;
    }

    out.flip_left_index = flip;
    out.flip_right_index = flip + 1;
    out.ezero = linear_zero(rows[flip].Ecm,
                            rows[flip].det_norm.real(),
                            rows[flip+1].Ecm,
                            rows[flip+1].det_norm.real());

    std::vector<double> gaps;
    gaps.reserve(static_cast<size_t>(N/2));
    for (int k=0;k<N/2;++k) {
        gaps.push_back(std::abs(rows[k].det_norm.real()) +
                       std::abs(rows[N-1-k].det_norm.real()));
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
    if (tot <= 0) {
        out.classification = "ambiguous";
        return out;
    }

    out.decreasing_fraction = double(dec)/double(tot);
    out.increasing_fraction = double(inc)/double(tot);
    out.trend_score = out.decreasing_fraction - out.increasing_fraction;

    double margin = std::abs(out.decreasing_fraction - out.increasing_fraction);
    if (out.decreasing_fraction >= trend_required && margin >= margin_required) {
        out.classification = "likely_zero";
    } else if (out.increasing_fraction >= trend_required && margin >= margin_required) {
        out.classification = "likely_pole";
    } else {
        out.classification = "ambiguous";
    }

    return out;
}

static std::string classify_inward_shape(const std::vector<DetRow>& rows, double& ezero, double trend_required=0.65, double margin_required=0.20)
{
    InwardShapeResult r = classify_inward_shape_result(rows, trend_required, margin_required);
    ezero = r.ezero;
    return r.classification;
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
        const int Nref = refine_points;          // v18: keep this fixed, e.g. 200
        const int refine_window_points = 5;      // v18: zoom to flip-window ... flip+1+window
        const int max_zoom_rounds = 12;          // safety cap against infinite ambiguous loops

        for (int round=0; round<max_zoom_rounds && !done; ++round) {
            auto ref = run_scan_device_resident(nnP, irrep, irrep_tag, EL, ER, Nref,
                                                "_device_refine_" + std::to_string(rid) + "_round" + std::to_string(round) + "_N" + std::to_string(Nref), debug);

            InwardShapeResult rr = classify_inward_shape_result(ref, 0.65, 0.20);
            double ez = rr.ezero;
            const std::string& cls = rr.classification;

            std::cout << "[adaptive-device] candidate=" << rid << " round=" << round
                      << " N=" << Nref
                      << " class=" << cls
                      << " flip=(" << rr.flip_left_index << "," << rr.flip_right_index << ")"
                      << " dec=" << std::setprecision(6) << rr.decreasing_fraction
                      << " inc=" << rr.increasing_fraction
                      << " E=" << std::setprecision(17) << ez
                      << " window=[" << EL << "," << ER << "]\n";

            if (cls == "likely_zero") {
                bool dup=false;
                for (double z: zeros) if (std::abs(z-ez) < 1e-7) dup=true;
                if (!dup) zeros.push_back(ez);
                done = true;
            } else if (cls == "likely_pole" || cls == "no_sign_flip") {
                done = true;
            } else {
                // v18 behavior:
                // Do NOT increase N. Instead, shrink the refinement interval around
                // the newest refined sign-flip pair, then rerun with the same Nref.
                if (rr.flip_left_index < 0 || rr.flip_right_index < 0) {
                    std::cout << "[adaptive-device] candidate=" << rid
                              << " ambiguous but no usable sign flip; stopping zoom refinement\n";
                    done = true;
                    break;
                }

                int Nloc = static_cast<int>(ref.size());
                int new_il = std::max(0, rr.flip_left_index - refine_window_points);
                int new_ir = std::min(Nloc - 1, rr.flip_right_index + refine_window_points);

                double new_EL = ref[new_il].Ecm;
                double new_ER = ref[new_ir].Ecm;
                if (new_ER < new_EL) std::swap(new_EL, new_ER);

                // If the window no longer shrinks meaningfully, stop to avoid repeating forever.
                double old_width = std::abs(ER - EL);
                double new_width = std::abs(new_ER - new_EL);
                std::cout << "[adaptive-device] candidate=" << rid
                          << " ambiguous -> zoom window using refined flip: "
                          << "old_width=" << std::setprecision(17) << old_width
                          << " new_width=" << new_width
                          << " new_window=[" << new_EL << "," << new_ER << "]\n";

                if (!(new_width > 0.0) || new_width >= 0.999999 * old_width) {
                    std::cout << "[adaptive-device] candidate=" << rid
                              << " zoom window did not shrink; stopping as ambiguous\n";
                    done = true;
                    break;
                }

                EL = new_EL;
                ER = new_ER;
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
        int refineN = 200;
        char debug = 'n';

        if (argc >= 4) { n0=std::stoi(argv[1]); n1=std::stoi(argv[2]); n2=std::stoi(argv[3]); }
        if (argc >= 5) irrep = argv[4];
        if (argc >= 6) irrep_tag = argv[5];
        if (argc >= 7) coarseN = std::stoi(argv[6]);
        if (argc >= 8) refineN = std::stoi(argv[7]);
        if (argc >= 10) { E0 = std::stod(argv[8]); E1 = std::stod(argv[9]); }
        if (argc >= 11) debug = argv[10][0];

        std::array<int,3> nnP{n0,n1,n2};
        std::cout << " # F3 true-batch grouped device-resident v18\n";
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
