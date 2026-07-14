// F3_gpu_omp_cublas_pipeline_v3.cu
//
// Purpose:
//   Full CPU(OpenMP) producer + GPU(cuBLAS/CUDA) consumer pipeline for
//
//       H  * X1 = F2
//       F3 = F2/3 - F2 * X1
//       F3 * X2 = Vsel
//       F3inv_projected = Vsel^H * X2
//       det = det(F3inv_projected)
//
// Notes:
//   1. This file assumes your existing physics functions are available from functions.h:
//        - Ecm_to_E
//        - config_maker_4
//        - F2_2plus1_mat
//        - K2inv_EREord2_2plus1_mat
//        - G_2plus1_mat
//        - P_irrep_projection_2plus1
//        - build_projector_from_eigenvectors_near_one
//        - smallest_eigenvalue
//        - normalize_det_vector_by_max
//      and your Vec3 type is available there as in your current code.
//
//   2. The GPU uses cuBLAS batched LU/solve and GEMM. No CPU LU invertibility precheck is used.
//
//   3. Determinant of F3inv_projected is computed using GPU batched LU of the projected matrices.
//      The diagonal/product/pivot parity is evaluated on the GPU by a small CUDA kernel.
//
//   4. Compile example:
//        nvcc -O3 -std=c++17 -DEIGEN_NO_CUDA --expt-relaxed-constexpr \
//             -Xcompiler -fopenmp -lineinfo -I/usr/include/eigen3 \
//             F3_gpu_omp_cublas_pipeline_v3.cu -lcublas -lcudart -o run_f3_pipeline
//
//      If compiling as object:
//        nvcc -O3 -std=c++17 -DEIGEN_NO_CUDA --expt-relaxed-constexpr \
//             -Xcompiler -fopenmp -lineinfo -I/usr/include/eigen3 \
//             -c F3_gpu_omp_cublas_pipeline_v3.cu -o F3_gpu_omp_cublas_pipeline_v3.o

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>

#include <Eigen/Dense>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "functions.h"

using comp = std::complex<double>;

//==============================================================================
// Error checking
//==============================================================================

static inline void f3_throwOnCuda(cudaError_t st, const char* msg, const char* file, int line)
{
    if (st != cudaSuccess)
    {
        std::ostringstream os;
        os << "CUDA error: " << msg << " | " << cudaGetErrorString(st)
           << " @ " << file << ":" << line;
        throw std::runtime_error(os.str());
    }
}

static inline void f3_throwOnCublas(cublasStatus_t st, const char* msg, const char* file, int line)
{
    if (st != CUBLAS_STATUS_SUCCESS)
    {
        std::ostringstream os;
        os << "cuBLAS error: " << msg << " | status=" << int(st)
           << " @ " << file << ":" << line;
        throw std::runtime_error(os.str());
    }
}

#define F3_CUDA_CHECK(x, msg)   f3_throwOnCuda((x), (msg), __FILE__, __LINE__)
#define F3_CUBLAS_CHECK(x, msg) f3_throwOnCublas((x), (msg), __FILE__, __LINE__)

static inline void f3_cudaSyncCheck(const char* where)
{
    F3_CUDA_CHECK(cudaDeviceSynchronize(), where);
    F3_CUDA_CHECK(cudaGetLastError(), where);
}

//==============================================================================
// Timer
//==============================================================================

struct F3ScopedTimer
{
    std::string name;
    char debug;
    std::chrono::high_resolution_clock::time_point t0;

    F3ScopedTimer(const std::string& name_, char debug_)
        : name(name_), debug(debug_), t0(std::chrono::high_resolution_clock::now())
    {}

    ~F3ScopedTimer()
    {
        if (debug == 'y')
        {
            auto t1 = std::chrono::high_resolution_clock::now();
            double sec = std::chrono::duration<double>(t1 - t0).count();
            std::cout << "[TIMER] " << name << " : " << sec << " s\n";
        }
    }
};

//==============================================================================
// Thread-safe queue
//==============================================================================

template <class T>
class ThreadSafeQueue
{
private:
    std::queue<T> q;
    std::mutex m;
    std::condition_variable cv;
    bool closed = false;

public:
    void push(T item)
    {
        {
            std::lock_guard<std::mutex> lock(m);
            if (closed)
            {
                throw std::runtime_error("Cannot push to closed queue.");
            }
            q.push(std::move(item));
        }
        cv.notify_one();
    }

    bool pop(T& item)
    {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [&]() { return closed || !q.empty(); });

        if (q.empty())
        {
            return false;
        }

        item = std::move(q.front());
        q.pop();
        return true;
    }

    void close()
    {
        {
            std::lock_guard<std::mutex> lock(m);
            closed = true;
        }
        cv.notify_all();
    }
};

//==============================================================================
// Data containers
//==============================================================================

struct EnergyMatrixPack
{
    int i = -1;

    comp Ecm;
    comp En;

    int total_dim = 0;
    int vdim = 0;

    Eigen::MatrixXcd F2;
    Eigen::MatrixXcd G;
    Eigen::MatrixXcd K2inv;
    Eigen::MatrixXcd H;
    Eigen::MatrixXcd Vsel;
};

struct EnergyGpuResult
{
    int i = -1;

    comp Ecm = comp(
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN()
    );

    comp En = comp(
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN()
    );

    int total_dim = 0;
    int vdim = 0;

    Eigen::MatrixXcd X1;
    Eigen::MatrixXcd F3;
    Eigen::MatrixXcd X2;
    Eigen::MatrixXcd F3inv_projected;

    comp det_F3inv_projected = comp(
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN()
    );

    comp eig_F3inv_projected = comp(
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN()
    );

    bool success = false;
    int gpu_info_H = 0;
    int gpu_info_F3 = 0;
    int gpu_info_proj = 0;
};

struct CpuGpuChunk
{
    int chunk_id = -1;

    int i_start = -1;
    int i_end = -1;

    comp Ecm_start;
    comp Ecm_end;

    int total_dim = 0;
    int count = 0;

    std::vector<EnergyMatrixPack> items;
};

//==============================================================================
// Basic pack/unpack helpers
//==============================================================================

static inline void f3_packEigenToHostCublas(const Eigen::MatrixXcd& A, cuDoubleComplex* dst)
{
    const std::complex<double>* src = A.data();
    const int elems = int(A.size());

    for (int i = 0; i < elems; ++i)
    {
        dst[i] = make_cuDoubleComplex(src[i].real(), src[i].imag());
    }
}

static inline void f3_unpackHostCublasToEigen(const cuDoubleComplex* src, Eigen::MatrixXcd& A)
{
    std::complex<double>* dst = A.data();
    const int elems = int(A.size());

    for (int i = 0; i < elems; ++i)
    {
        dst[i] = std::complex<double>(cuCreal(src[i]), cuCimag(src[i]));
    }
}

static inline size_t bytes_complex_matrix(int rows, int cols)
{
    return size_t(rows) * size_t(cols) * sizeof(cuDoubleComplex);
}

static inline size_t estimate_gpu_bytes_one_energy(int n, int vdim)
{
    size_t B = 0;

    B += bytes_complex_matrix(n, n);       // H / LU_H
    B += bytes_complex_matrix(n, n);       // F2
    B += bytes_complex_matrix(n, n);       // X1
    B += bytes_complex_matrix(n, n);       // F2X1 product
    B += bytes_complex_matrix(n, n);       // F3 original
    B += bytes_complex_matrix(n, n);       // F3 LU copy
    B += bytes_complex_matrix(n, vdim);    // Vsel
    B += bytes_complex_matrix(n, vdim);    // X2
    B += bytes_complex_matrix(vdim, vdim); // projected original
    B += bytes_complex_matrix(vdim, vdim); // projected LU copy

    B += size_t(n) * sizeof(int);          // H pivots per matrix
    B += size_t(n) * sizeof(int);          // F3 pivots per matrix
    B += size_t(vdim) * sizeof(int);       // projected pivots per matrix
    B += 3 * sizeof(int);                  // info arrays

    B += 16ull * 1024ull * 1024ull;        // safety/workspace pad per energy-ish

    return B;
}

//==============================================================================
// No custom CUDA kernels in this include-safe version.
// Pointer arrays are built on host; F3 uses cublasZgeam; determinant is computed
// on CPU from the cuBLAS LU factors.
//==============================================================================

//==============================================================================
// GPU buffer cache for same total_dim and max vdim
//==============================================================================

struct PipelineGpuCache
{
    int n = 0;
    int max_vdim = 0;
    int cap = 0;

    cuDoubleComplex* dH = nullptr;
    cuDoubleComplex* dF2 = nullptr;
    cuDoubleComplex* dX1 = nullptr;
    cuDoubleComplex* dF2X1 = nullptr;
    cuDoubleComplex* dF3 = nullptr;
    cuDoubleComplex* dF3LU = nullptr;
    cuDoubleComplex* dVsel = nullptr;
    cuDoubleComplex* dX2 = nullptr;
    cuDoubleComplex* dProj = nullptr;
    cuDoubleComplex* dProjLU = nullptr;
    cuDoubleComplex* dDet = nullptr;

    int* dPivH = nullptr;
    int* dPivF3 = nullptr;
    int* dPivProj = nullptr;
    int* dInfoH = nullptr;
    int* dInfoF3 = nullptr;
    int* dInfoProj = nullptr;

    cuDoubleComplex** dH_array = nullptr;
    cuDoubleComplex** dF2_array = nullptr;
    cuDoubleComplex** dX1_array = nullptr;
    cuDoubleComplex** dF2X1_array = nullptr;
    cuDoubleComplex** dF3_array = nullptr;
    cuDoubleComplex** dF3LU_array = nullptr;
    cuDoubleComplex** dVsel_array = nullptr;
    cuDoubleComplex** dX2_array = nullptr;
    cuDoubleComplex** dProj_array = nullptr;
    cuDoubleComplex** dProjLU_array = nullptr;

    void release()
    {
        if (dH) cudaFree(dH), dH = nullptr;
        if (dF2) cudaFree(dF2), dF2 = nullptr;
        if (dX1) cudaFree(dX1), dX1 = nullptr;
        if (dF2X1) cudaFree(dF2X1), dF2X1 = nullptr;
        if (dF3) cudaFree(dF3), dF3 = nullptr;
        if (dF3LU) cudaFree(dF3LU), dF3LU = nullptr;
        if (dVsel) cudaFree(dVsel), dVsel = nullptr;
        if (dX2) cudaFree(dX2), dX2 = nullptr;
        if (dProj) cudaFree(dProj), dProj = nullptr;
        if (dProjLU) cudaFree(dProjLU), dProjLU = nullptr;
        if (dDet) cudaFree(dDet), dDet = nullptr;

        if (dPivH) cudaFree(dPivH), dPivH = nullptr;
        if (dPivF3) cudaFree(dPivF3), dPivF3 = nullptr;
        if (dPivProj) cudaFree(dPivProj), dPivProj = nullptr;
        if (dInfoH) cudaFree(dInfoH), dInfoH = nullptr;
        if (dInfoF3) cudaFree(dInfoF3), dInfoF3 = nullptr;
        if (dInfoProj) cudaFree(dInfoProj), dInfoProj = nullptr;

        if (dH_array) cudaFree(dH_array), dH_array = nullptr;
        if (dF2_array) cudaFree(dF2_array), dF2_array = nullptr;
        if (dX1_array) cudaFree(dX1_array), dX1_array = nullptr;
        if (dF2X1_array) cudaFree(dF2X1_array), dF2X1_array = nullptr;
        if (dF3_array) cudaFree(dF3_array), dF3_array = nullptr;
        if (dF3LU_array) cudaFree(dF3LU_array), dF3LU_array = nullptr;
        if (dVsel_array) cudaFree(dVsel_array), dVsel_array = nullptr;
        if (dX2_array) cudaFree(dX2_array), dX2_array = nullptr;
        if (dProj_array) cudaFree(dProj_array), dProj_array = nullptr;
        if (dProjLU_array) cudaFree(dProjLU_array), dProjLU_array = nullptr;

        n = 0;
        max_vdim = 0;
        cap = 0;
    }

    ~PipelineGpuCache()
    {
        release();
    }

    void ensure(int n_in, int max_vdim_in, int batchCount)
    {
        if (batchCount <= 0)
        {
            return;
        }

        if (n == n_in && max_vdim >= max_vdim_in && cap >= batchCount)
        {
            return;
        }

        int newCap = std::max(batchCount, std::max(32, cap > 0 ? 2 * cap : 32));
        int newV = std::max(max_vdim_in, std::max(max_vdim, 1));

        release();

        n = n_in;
        max_vdim = newV;
        cap = newCap;

        size_t nn = size_t(cap) * size_t(n) * size_t(n);
        size_t nv = size_t(cap) * size_t(n) * size_t(max_vdim);
        size_t vv = size_t(cap) * size_t(max_vdim) * size_t(max_vdim);

        F3_CUDA_CHECK(cudaMalloc((void**)&dH, nn * sizeof(cuDoubleComplex)), "cudaMalloc dH");
        F3_CUDA_CHECK(cudaMalloc((void**)&dF2, nn * sizeof(cuDoubleComplex)), "cudaMalloc dF2");
        F3_CUDA_CHECK(cudaMalloc((void**)&dX1, nn * sizeof(cuDoubleComplex)), "cudaMalloc dX1");
        F3_CUDA_CHECK(cudaMalloc((void**)&dF2X1, nn * sizeof(cuDoubleComplex)), "cudaMalloc dF2X1");
        F3_CUDA_CHECK(cudaMalloc((void**)&dF3, nn * sizeof(cuDoubleComplex)), "cudaMalloc dF3");
        F3_CUDA_CHECK(cudaMalloc((void**)&dF3LU, nn * sizeof(cuDoubleComplex)), "cudaMalloc dF3LU");
        F3_CUDA_CHECK(cudaMalloc((void**)&dVsel, nv * sizeof(cuDoubleComplex)), "cudaMalloc dVsel");
        F3_CUDA_CHECK(cudaMalloc((void**)&dX2, nv * sizeof(cuDoubleComplex)), "cudaMalloc dX2");
        F3_CUDA_CHECK(cudaMalloc((void**)&dProj, vv * sizeof(cuDoubleComplex)), "cudaMalloc dProj");
        F3_CUDA_CHECK(cudaMalloc((void**)&dProjLU, vv * sizeof(cuDoubleComplex)), "cudaMalloc dProjLU");
        F3_CUDA_CHECK(cudaMalloc((void**)&dDet, size_t(cap) * sizeof(cuDoubleComplex)), "cudaMalloc dDet");

        F3_CUDA_CHECK(cudaMalloc((void**)&dPivH, size_t(cap) * size_t(n) * sizeof(int)), "cudaMalloc dPivH");
        F3_CUDA_CHECK(cudaMalloc((void**)&dPivF3, size_t(cap) * size_t(n) * sizeof(int)), "cudaMalloc dPivF3");
        F3_CUDA_CHECK(cudaMalloc((void**)&dPivProj, size_t(cap) * size_t(max_vdim) * sizeof(int)), "cudaMalloc dPivProj");
        F3_CUDA_CHECK(cudaMalloc((void**)&dInfoH, size_t(cap) * sizeof(int)), "cudaMalloc dInfoH");
        F3_CUDA_CHECK(cudaMalloc((void**)&dInfoF3, size_t(cap) * sizeof(int)), "cudaMalloc dInfoF3");
        F3_CUDA_CHECK(cudaMalloc((void**)&dInfoProj, size_t(cap) * sizeof(int)), "cudaMalloc dInfoProj");

        F3_CUDA_CHECK(cudaMalloc((void**)&dH_array, size_t(cap) * sizeof(cuDoubleComplex*)), "cudaMalloc dH_array");
        F3_CUDA_CHECK(cudaMalloc((void**)&dF2_array, size_t(cap) * sizeof(cuDoubleComplex*)), "cudaMalloc dF2_array");
        F3_CUDA_CHECK(cudaMalloc((void**)&dX1_array, size_t(cap) * sizeof(cuDoubleComplex*)), "cudaMalloc dX1_array");
        F3_CUDA_CHECK(cudaMalloc((void**)&dF2X1_array, size_t(cap) * sizeof(cuDoubleComplex*)), "cudaMalloc dF2X1_array");
        F3_CUDA_CHECK(cudaMalloc((void**)&dF3_array, size_t(cap) * sizeof(cuDoubleComplex*)), "cudaMalloc dF3_array");
        F3_CUDA_CHECK(cudaMalloc((void**)&dF3LU_array, size_t(cap) * sizeof(cuDoubleComplex*)), "cudaMalloc dF3LU_array");
        F3_CUDA_CHECK(cudaMalloc((void**)&dVsel_array, size_t(cap) * sizeof(cuDoubleComplex*)), "cudaMalloc dVsel_array");
        F3_CUDA_CHECK(cudaMalloc((void**)&dX2_array, size_t(cap) * sizeof(cuDoubleComplex*)), "cudaMalloc dX2_array");
        F3_CUDA_CHECK(cudaMalloc((void**)&dProj_array, size_t(cap) * sizeof(cuDoubleComplex*)), "cudaMalloc dProj_array");
        F3_CUDA_CHECK(cudaMalloc((void**)&dProjLU_array, size_t(cap) * sizeof(cuDoubleComplex*)), "cudaMalloc dProjLU_array");
    }
};

//==============================================================================
// GPU same-dim and same-vdim subgroup processor
//==============================================================================

static void gpu_process_same_dim_same_vdim_group_cublas(
    cublasHandle_t handle,
    PipelineGpuCache& cache,
    int n,
    int vdim,
    const std::vector<const EnergyMatrixPack*>& group,
    std::vector<EnergyGpuResult>& group_results,
    bool save_large_matrices,
    char debug)
{
    const int batchCount = int(group.size());
    if (batchCount == 0)
    {
        return;
    }

    if (n <= 0 || vdim <= 0)
    {
        throw std::runtime_error("gpu_process_same_dim_same_vdim_group_cublas got invalid dimensions.");
    }

    cache.ensure(n, vdim, batchCount);

    const size_t nn_elems = size_t(batchCount) * size_t(n) * size_t(n);
    const size_t nv_elems = size_t(batchCount) * size_t(n) * size_t(vdim);
    const size_t vv_elems = size_t(batchCount) * size_t(vdim) * size_t(vdim);

    std::vector<cuDoubleComplex> hH(nn_elems);
    std::vector<cuDoubleComplex> hF2(nn_elems);
    std::vector<cuDoubleComplex> hVsel(nv_elems);

    for (int b = 0; b < batchCount; ++b)
    {
        const auto& item = *group[size_t(b)];

        if (item.H.rows() != n || item.H.cols() != n ||
            item.F2.rows() != n || item.F2.cols() != n ||
            item.Vsel.rows() != n || item.Vsel.cols() != vdim)
        {
            throw std::runtime_error("group contains wrong matrix shape.");
        }

        f3_packEigenToHostCublas(item.H, hH.data() + size_t(b) * size_t(n) * size_t(n));
        f3_packEigenToHostCublas(item.F2, hF2.data() + size_t(b) * size_t(n) * size_t(n));
        f3_packEigenToHostCublas(item.Vsel, hVsel.data() + size_t(b) * size_t(n) * size_t(vdim));
    }

    {
        F3ScopedTimer timer("GPU H2D chunk copy", debug);
        F3_CUDA_CHECK(cudaMemcpy(cache.dH, hH.data(), nn_elems * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), "H2D H");
        F3_CUDA_CHECK(cudaMemcpy(cache.dF2, hF2.data(), nn_elems * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), "H2D F2");
        F3_CUDA_CHECK(cudaMemcpy(cache.dX1, hF2.data(), nn_elems * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), "H2D X1 initial RHS=F2");
        F3_CUDA_CHECK(cudaMemcpy(cache.dVsel, hVsel.data(), nv_elems * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), "H2D Vsel");
        F3_CUDA_CHECK(cudaMemcpy(cache.dX2, hVsel.data(), nv_elems * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), "H2D X2 initial RHS=Vsel");
    }

    const int threads = 256; // only used for loop granularity messages; no CUDA kernels launched here

    // Build cuBLAS pointer arrays on the host, then copy them to device.
    // This keeps the file include-safe even when included from a .cpp translation unit.
    std::vector<cuDoubleComplex*> hHptr(batchCount), hF2ptr(batchCount), hX1ptr(batchCount), hF2X1ptr(batchCount);
    std::vector<cuDoubleComplex*> hF3ptr(batchCount), hF3LUptr(batchCount), hVselptr(batchCount), hX2ptr(batchCount);
    std::vector<cuDoubleComplex*> hProjptr(batchCount), hProjLUptr(batchCount);

    for (int b = 0; b < batchCount; ++b)
    {
        hHptr[b]      = cache.dH      + size_t(b) * size_t(n) * size_t(n);
        hF2ptr[b]     = cache.dF2     + size_t(b) * size_t(n) * size_t(n);
        hX1ptr[b]     = cache.dX1     + size_t(b) * size_t(n) * size_t(n);
        hF2X1ptr[b]   = cache.dF2X1   + size_t(b) * size_t(n) * size_t(n);
        hF3ptr[b]     = cache.dF3     + size_t(b) * size_t(n) * size_t(n);
        hF3LUptr[b]   = cache.dF3LU   + size_t(b) * size_t(n) * size_t(n);
        hVselptr[b]   = cache.dVsel   + size_t(b) * size_t(n) * size_t(vdim);
        hX2ptr[b]     = cache.dX2     + size_t(b) * size_t(n) * size_t(vdim);
        hProjptr[b]   = cache.dProj   + size_t(b) * size_t(vdim) * size_t(vdim);
        hProjLUptr[b] = cache.dProjLU + size_t(b) * size_t(vdim) * size_t(vdim);
    }

    F3_CUDA_CHECK(cudaMemcpy(cache.dH_array, hHptr.data(), size_t(batchCount) * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice), "H2D H ptrs");
    F3_CUDA_CHECK(cudaMemcpy(cache.dF2_array, hF2ptr.data(), size_t(batchCount) * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice), "H2D F2 ptrs");
    F3_CUDA_CHECK(cudaMemcpy(cache.dX1_array, hX1ptr.data(), size_t(batchCount) * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice), "H2D X1 ptrs");
    F3_CUDA_CHECK(cudaMemcpy(cache.dF2X1_array, hF2X1ptr.data(), size_t(batchCount) * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice), "H2D F2X1 ptrs");
    F3_CUDA_CHECK(cudaMemcpy(cache.dF3_array, hF3ptr.data(), size_t(batchCount) * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice), "H2D F3 ptrs");
    F3_CUDA_CHECK(cudaMemcpy(cache.dF3LU_array, hF3LUptr.data(), size_t(batchCount) * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice), "H2D F3LU ptrs");
    F3_CUDA_CHECK(cudaMemcpy(cache.dVsel_array, hVselptr.data(), size_t(batchCount) * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice), "H2D Vsel ptrs");
    F3_CUDA_CHECK(cudaMemcpy(cache.dX2_array, hX2ptr.data(), size_t(batchCount) * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice), "H2D X2 ptrs");
    F3_CUDA_CHECK(cudaMemcpy(cache.dProj_array, hProjptr.data(), size_t(batchCount) * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice), "H2D Proj ptrs");
    F3_CUDA_CHECK(cudaMemcpy(cache.dProjLU_array, hProjLUptr.data(), size_t(batchCount) * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice), "H2D ProjLU ptrs");

    std::vector<int> hInfoH(batchCount, 0);
    std::vector<int> hInfoF3(batchCount, 0);
    std::vector<int> hInfoProj(batchCount, 0);
    std::vector<int> hInfoGetrsH(batchCount, 0);
    std::vector<int> hInfoGetrsF3(batchCount, 0);

    {
        F3ScopedTimer timer("GPU LU solve H X1 = F2", debug);

        F3_CUDA_CHECK(cudaMemset(cache.dInfoH, 0, size_t(batchCount) * sizeof(int)), "memset dInfoH");

        F3_CUBLAS_CHECK(
            cublasZgetrfBatched(
                handle,
                n,
                cache.dH_array,
                n,
                cache.dPivH,
                cache.dInfoH,
                batchCount
            ),
            "cublasZgetrfBatched H"
        );
        f3_cudaSyncCheck("after H getrfBatched");

        F3_CUDA_CHECK(cudaMemcpy(hInfoH.data(), cache.dInfoH, size_t(batchCount) * sizeof(int), cudaMemcpyDeviceToHost), "D2H infoH");

        F3_CUBLAS_CHECK(
            cublasZgetrsBatched(
                handle,
                CUBLAS_OP_N,
                n,
                n,
                (const cuDoubleComplex**)cache.dH_array,
                n,
                cache.dPivH,
                cache.dX1_array,
                n,
                hInfoGetrsH.data(),
                batchCount
            ),
            "cublasZgetrsBatched H X1=F2"
        );
        f3_cudaSyncCheck("after H getrsBatched");
    }

    {
        F3ScopedTimer timer("GPU GEMM F2 * X1 and build F3", debug);

        const cuDoubleComplex alpha_gemm = make_cuDoubleComplex(1.0, 0.0);
        const cuDoubleComplex beta_gemm  = make_cuDoubleComplex(0.0, 0.0);

        F3_CUBLAS_CHECK(
            cublasZgemmBatched(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n,
                n,
                n,
                &alpha_gemm,
                (const cuDoubleComplex**)cache.dF2_array,
                n,
                (const cuDoubleComplex**)cache.dX1_array,
                n,
                &beta_gemm,
                cache.dF2X1_array,
                n,
                batchCount
            ),
            "cublasZgemmBatched F2X1"
        );
        f3_cudaSyncCheck("after F2X1 gemm");

        // F3 = (1/3) F2 - F2X1.
        // Use cublasZgeam per matrix to avoid a custom CUDA kernel.
        const cuDoubleComplex alpha_f3 = make_cuDoubleComplex(1.0 / 3.0, 0.0);
        const cuDoubleComplex beta_f3  = make_cuDoubleComplex(-1.0, 0.0);

        for (int b = 0; b < batchCount; ++b)
        {
            F3_CUBLAS_CHECK(
                cublasZgeam(
                    handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    n,
                    n,
                    &alpha_f3,
                    cache.dF2 + size_t(b) * size_t(n) * size_t(n),
                    n,
                    &beta_f3,
                    cache.dF2X1 + size_t(b) * size_t(n) * size_t(n),
                    n,
                    cache.dF3 + size_t(b) * size_t(n) * size_t(n),
                    n
                ),
                "cublasZgeam F3 = F2/3 - F2X1"
            );
        }
        f3_cudaSyncCheck("after F3 geam loop");
    }

    {
        F3ScopedTimer timer("GPU LU solve F3 X2 = Vsel", debug);

        F3_CUDA_CHECK(cudaMemcpy(cache.dF3LU, cache.dF3, nn_elems * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice), "D2D F3 to F3LU");

        F3_CUDA_CHECK(cudaMemset(cache.dInfoF3, 0, size_t(batchCount) * sizeof(int)), "memset dInfoF3");

        F3_CUBLAS_CHECK(
            cublasZgetrfBatched(
                handle,
                n,
                cache.dF3LU_array,
                n,
                cache.dPivF3,
                cache.dInfoF3,
                batchCount
            ),
            "cublasZgetrfBatched F3"
        );
        f3_cudaSyncCheck("after F3 getrfBatched");

        F3_CUDA_CHECK(cudaMemcpy(hInfoF3.data(), cache.dInfoF3, size_t(batchCount) * sizeof(int), cudaMemcpyDeviceToHost), "D2H infoF3");

        F3_CUBLAS_CHECK(
            cublasZgetrsBatched(
                handle,
                CUBLAS_OP_N,
                n,
                vdim,
                (const cuDoubleComplex**)cache.dF3LU_array,
                n,
                cache.dPivF3,
                cache.dX2_array,
                n,
                hInfoGetrsF3.data(),
                batchCount
            ),
            "cublasZgetrsBatched F3 X2=Vsel"
        );
        f3_cudaSyncCheck("after F3 getrsBatched");
    }

    {
        F3ScopedTimer timer("GPU GEMM Vsel^H * X2", debug);

        const cuDoubleComplex alpha_gemm = make_cuDoubleComplex(1.0, 0.0);
        const cuDoubleComplex beta_gemm  = make_cuDoubleComplex(0.0, 0.0);

        F3_CUBLAS_CHECK(
            cublasZgemmBatched(
                handle,
                CUBLAS_OP_C,
                CUBLAS_OP_N,
                vdim,
                vdim,
                n,
                &alpha_gemm,
                (const cuDoubleComplex**)cache.dVsel_array,
                n,
                (const cuDoubleComplex**)cache.dX2_array,
                n,
                &beta_gemm,
                cache.dProj_array,
                vdim,
                batchCount
            ),
            "cublasZgemmBatched VselH_X2"
        );
        f3_cudaSyncCheck("after projected gemm");
    }

    std::vector<cuDoubleComplex> hDet(static_cast<size_t>(batchCount));

    {
        F3ScopedTimer timer("GPU determinant LU projected matrix", debug);

        F3_CUDA_CHECK(cudaMemcpy(cache.dProjLU, cache.dProj, vv_elems * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice), "D2D Proj to ProjLU");

        F3_CUDA_CHECK(cudaMemset(cache.dInfoProj, 0, size_t(batchCount) * sizeof(int)), "memset dInfoProj");

        F3_CUBLAS_CHECK(
            cublasZgetrfBatched(
                handle,
                vdim,
                cache.dProjLU_array,
                vdim,
                cache.dPivProj,
                cache.dInfoProj,
                batchCount
            ),
            "cublasZgetrfBatched projected"
        );
        f3_cudaSyncCheck("after projected getrfBatched");

        F3_CUDA_CHECK(cudaMemcpy(hInfoProj.data(), cache.dInfoProj, size_t(batchCount) * sizeof(int), cudaMemcpyDeviceToHost), "D2H infoProj");

        // Compute determinant from LU on CPU. The LU itself is still computed by cuBLAS.
        // This avoids a custom CUDA kernel, so this file can be included from .cpp files.
        std::vector<cuDoubleComplex> hProjLU(vv_elems);
        std::vector<int> hPivProj(size_t(batchCount) * size_t(vdim));

        F3_CUDA_CHECK(cudaMemcpy(hProjLU.data(), cache.dProjLU, vv_elems * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost), "D2H projected LU");
        F3_CUDA_CHECK(cudaMemcpy(hPivProj.data(), cache.dPivProj, size_t(batchCount) * size_t(vdim) * sizeof(int), cudaMemcpyDeviceToHost), "D2H projected pivots");

        for (int b = 0; b < batchCount; ++b)
        {
            if (hInfoProj[size_t(b)] != 0)
            {
                hDet[size_t(b)] = make_cuDoubleComplex(NAN, NAN);
                continue;
            }

            cuDoubleComplex det = make_cuDoubleComplex(1.0, 0.0);
            int sign = 1;

            const cuDoubleComplex* LU = hProjLU.data() + size_t(b) * size_t(vdim) * size_t(vdim);
            const int* piv = hPivProj.data() + size_t(b) * size_t(vdim);

            for (int k = 0; k < vdim; ++k)
            {
                det = cuCmul(det, LU[k + k * vdim]);
                if (piv[k] != k + 1)
                {
                    sign = -sign;
                }
            }

            if (sign < 0)
            {
                det = make_cuDoubleComplex(-cuCreal(det), -cuCimag(det));
            }

            hDet[size_t(b)] = det;
        }
    }

    std::vector<cuDoubleComplex> hProj(vv_elems);

    {
        F3ScopedTimer timer("GPU D2H projected", debug);
        F3_CUDA_CHECK(cudaMemcpy(hProj.data(), cache.dProj, vv_elems * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost), "D2H projected matrix");
    }

    std::vector<cuDoubleComplex> hX1;
    std::vector<cuDoubleComplex> hX2;
    std::vector<cuDoubleComplex> hF3;

    if (save_large_matrices)
    {
        hX1.resize(nn_elems);
        hX2.resize(nv_elems);
        hF3.resize(nn_elems);

        F3ScopedTimer timer("GPU D2H X1/X2/F3", debug);
        F3_CUDA_CHECK(cudaMemcpy(hX1.data(), cache.dX1, nn_elems * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost), "D2H X1");
        F3_CUDA_CHECK(cudaMemcpy(hX2.data(), cache.dX2, nv_elems * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost), "D2H X2");

        F3_CUDA_CHECK(cudaMemcpy(hF3.data(), cache.dF3, nn_elems * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost), "D2H F3");
    }

    group_results.resize(size_t(batchCount));

    for (int b = 0; b < batchCount; ++b)
    {
        const auto& item = *group[size_t(b)];

        EnergyGpuResult out;
        out.i = item.i;
        out.Ecm = item.Ecm;
        out.En = item.En;
        out.total_dim = item.total_dim;
        out.vdim = item.vdim;
        out.gpu_info_H = hInfoH[size_t(b)];
        out.gpu_info_F3 = hInfoF3[size_t(b)];
        out.gpu_info_proj = hInfoProj[size_t(b)];

        out.det_F3inv_projected = comp(cuCreal(hDet[size_t(b)]), cuCimag(hDet[size_t(b)]));

        out.F3inv_projected.resize(vdim, vdim);
        f3_unpackHostCublasToEigen(hProj.data() + size_t(b) * size_t(vdim) * size_t(vdim), out.F3inv_projected);

        if (out.F3inv_projected.rows() > 0 && out.F3inv_projected.cols() > 0)
        {
            out.eig_F3inv_projected = smallest_eigenvalue(out.F3inv_projected);
        }

        if (save_large_matrices)
        {
            out.X1.resize(n, n);
            out.X2.resize(n, vdim);
            out.F3.resize(n, n);

            f3_unpackHostCublasToEigen(hX1.data() + size_t(b) * size_t(n) * size_t(n), out.X1);
            f3_unpackHostCublasToEigen(hX2.data() + size_t(b) * size_t(n) * size_t(vdim), out.X2);
            f3_unpackHostCublasToEigen(hF3.data() + size_t(b) * size_t(n) * size_t(n), out.F3);
        }

        out.success =
            (out.gpu_info_H == 0) &&
            (out.gpu_info_F3 == 0) &&
            (out.gpu_info_proj == 0) &&
            std::isfinite(out.det_F3inv_projected.real()) &&
            std::isfinite(out.det_F3inv_projected.imag());

        group_results[size_t(b)] = std::move(out);
    }
}

//==============================================================================
// GPU chunk processor: split same total_dim chunk by vdim, then call cuBLAS group processor
//==============================================================================

static void gpu_process_same_dim_chunk_cublas(
    cublasHandle_t handle,
    PipelineGpuCache& cache,
    const CpuGpuChunk& chunk,
    std::vector<EnergyGpuResult>& results,
    bool save_large_matrices,
    char debug)
{
    F3ScopedTimer timer("GPU total chunk " + std::to_string(chunk.chunk_id), debug);

    if (debug == 'y')
    {
        std::cout << "[GPU] received chunk_id = " << chunk.chunk_id
                  << ", i range = [" << chunk.i_start << ", " << chunk.i_end << "]"
                  << ", Ecm range = [" << chunk.Ecm_start << ", " << chunk.Ecm_end << "]"
                  << ", total_dim = " << chunk.total_dim
                  << ", count = " << chunk.count
                  << '\n';
    }

    results.clear();
    results.reserve(chunk.items.size());

    std::map<int, std::vector<const EnergyMatrixPack*>> groups_by_vdim;

    for (const auto& item : chunk.items)
    {
        if (item.total_dim > 0 && item.vdim > 0)
        {
            groups_by_vdim[item.vdim].push_back(&item);
        }
    }

    for (const auto& kv : groups_by_vdim)
    {
        int vdim = kv.first;
        const auto& group = kv.second;

        if (debug == 'y')
        {
            std::cout << "[GPU] chunk_id = " << chunk.chunk_id
                      << ", subgroup total_dim = " << chunk.total_dim
                      << ", vdim = " << vdim
                      << ", count = " << group.size()
                      << '\n';
        }

        std::vector<EnergyGpuResult> group_results;

        gpu_process_same_dim_same_vdim_group_cublas(
            handle,
            cache,
            chunk.total_dim,
            vdim,
            group,
            group_results,
            save_large_matrices,
            debug
        );

        for (auto& r : group_results)
        {
            results.push_back(std::move(r));
        }
    }
}

//==============================================================================
// GPU consumer thread
//==============================================================================

static void gpu_consumer_thread_func(
    ThreadSafeQueue<CpuGpuChunk>& gpu_queue,
    std::vector<EnergyGpuResult>& all_results,
    std::mutex& result_mutex,
    bool save_large_matrices,
    char debug)
{
    cublasHandle_t handle{};
    F3_CUBLAS_CHECK(cublasCreate(&handle), "cublasCreate consumer");

    PipelineGpuCache cache;

    CpuGpuChunk chunk;

    while (gpu_queue.pop(chunk))
    {
        std::vector<EnergyGpuResult> chunk_results;

        try
        {
            gpu_process_same_dim_chunk_cublas(
                handle,
                cache,
                chunk,
                chunk_results,
                save_large_matrices,
                debug
            );
        }
        catch (const std::exception& e)
        {
            std::cerr << "[GPU] ERROR in chunk_id = " << chunk.chunk_id
                      << ": " << e.what() << '\n';

            chunk_results.clear();
            chunk_results.reserve(chunk.items.size());

            for (const auto& item : chunk.items)
            {
                EnergyGpuResult out;
                out.i = item.i;
                out.Ecm = item.Ecm;
                out.En = item.En;
                out.total_dim = item.total_dim;
                out.vdim = item.vdim;
                out.success = false;
                chunk_results.push_back(std::move(out));
            }
        }

        {
            std::lock_guard<std::mutex> lock(result_mutex);

            for (auto& r : chunk_results)
            {
                if (r.i >= 0 && r.i < int(all_results.size()))
                {
                    all_results[size_t(r.i)] = std::move(r);
                }
            }
        }

        if (debug == 'y')
        {
            std::cout << "[GPU] finished chunk_id = " << chunk.chunk_id
                      << ", result count = " << chunk_results.size()
                      << '\n';
        }
    }

    cache.release();
    cublasDestroy(handle);

    if (debug == 'y')
    {
        std::cout << "[GPU] consumer finished.\n";
    }
}

//==============================================================================
// CPU builder for one energy
//==============================================================================

static EnergyMatrixPack build_one_energy_pack(
    int i,
    comp Ecm_c,
    comp En_c,
    const std::vector<int>& waves_vec_1,
    const std::vector<int>& waves_vec_2,
    const std::vector<comp>& total_P,
    const std::vector<comp>& nnP_config,
    const std::string& I,
    double atmK,
    double atmpi,
    double L,
    double alpha,
    double epsilon_h,
    double max_shell_num,
    double tolerance,
    double eta_1,
    double eta_2,
    const std::vector<std::vector<comp>>& scatter_params_1,
    const std::vector<std::vector<comp>>& scatter_params_2,
    bool Q0norm,
    bool sort_orbit_flag,
    int parity,
    double eig_tol,
    double norm_tol,
    double proj_tol,
    char debug)
{
    EnergyMatrixPack pack;

    pack.i = i;
    pack.Ecm = Ecm_c;
    pack.En = En_c;

    std::vector<std::vector<comp>> plm_config(5);
    std::vector<std::vector<comp>> klm_config(5);

    std::vector<std::vector<int>> np_config(5);
    std::vector<std::vector<int>> nk_config(5);

    config_maker_4(
        plm_config,
        np_config,
        waves_vec_1,
        En_c,
        total_P,
        atmK,
        atmK,
        atmpi,
        L,
        epsilon_h,
        max_shell_num,
        tolerance
    );

    config_maker_4(
        klm_config,
        nk_config,
        waves_vec_2,
        En_c,
        total_P,
        atmpi,
        atmK,
        atmK,
        L,
        epsilon_h,
        max_shell_num,
        tolerance
    );

    int dim1 = int(plm_config[0].size());
    int dim2 = int(klm_config[0].size());
    int total_dim = dim1 + dim2;

    pack.total_dim = total_dim;

    if (total_dim <= 0)
    {
        return pack;
    }

    pack.F2.resize(total_dim, total_dim);
    pack.G.resize(total_dim, total_dim);
    pack.K2inv.resize(total_dim, total_dim);
    pack.H.resize(total_dim, total_dim);

    F2_2plus1_mat(
        pack.F2,
        En_c,
        plm_config,
        klm_config,
        total_P,
        atmK,
        atmpi,
        L,
        alpha,
        epsilon_h,
        max_shell_num,
        Q0norm
    );

    K2inv_EREord2_2plus1_mat(
        pack.K2inv,
        eta_1,
        eta_2,
        scatter_params_1,
        scatter_params_2,
        En_c,
        plm_config,
        klm_config,
        total_P,
        atmK,
        atmpi,
        epsilon_h,
        L
    );

    G_2plus1_mat(
        pack.G,
        En_c,
        plm_config,
        klm_config,
        total_P,
        atmK,
        atmpi,
        L,
        alpha,
        epsilon_h,
        max_shell_num,
        Q0norm
    );

    pack.H.noalias() = pack.K2inv + pack.F2 + pack.G;

    Eigen::MatrixXcd P_I(total_dim, total_dim);

    std::string I_mut = I;
    std::vector<comp> total_P_mut = total_P;
    std::vector<comp> nnP_config_mut = nnP_config;

    P_irrep_projection_2plus1(
        P_I,
        plm_config,
        np_config,
        klm_config,
        nk_config,
        I_mut,
        total_P_mut,
        nnP_config_mut,
        sort_orbit_flag,
        parity
    );

    Eigen::MatrixXcd Pproj;

    build_projector_from_eigenvectors_near_one(
        P_I,
        pack.Vsel,
        Pproj,
        eig_tol,
        norm_tol,
        proj_tol,
        'n'
    );

    pack.vdim = int(pack.Vsel.cols());

    if (debug == 'y')
    {
        std::cout << "[CPU] i = " << i
                  << ", Ecm = " << Ecm_c
                  << ", En = " << En_c
                  << ", total_dim = " << total_dim
                  << ", vdim = " << pack.vdim
                  << '\n';
    }

    return pack;
}

//==============================================================================
// Flush dimension-homogeneous CPU buffer into VRAM-safe GPU queue chunks
//==============================================================================

static void flush_cpu_buffer_to_gpu_queue(
    std::vector<EnergyMatrixPack>& buffer,
    ThreadSafeQueue<CpuGpuChunk>& gpu_queue,
    int& chunk_counter,
    size_t gpu_budget_bytes,
    char debug)
{
    if (buffer.empty())
    {
        return;
    }

    int total_dim = buffer.front().total_dim;

    std::vector<EnergyMatrixPack> current;
    size_t current_bytes = 64ull * 1024ull * 1024ull;

    auto push_current = [&]()
    {
        if (current.empty())
        {
            return;
        }

        CpuGpuChunk chunk;
        chunk.chunk_id = chunk_counter++;
        chunk.count = int(current.size());
        chunk.total_dim = current.front().total_dim;
        chunk.i_start = current.front().i;
        chunk.i_end = current.back().i;
        chunk.Ecm_start = current.front().Ecm;
        chunk.Ecm_end = current.back().Ecm;
        chunk.items = std::move(current);

        if (debug == 'y')
        {
            std::cout << "[CPU -> GPU] sending chunk_id = " << chunk.chunk_id
                      << ", i range = [" << chunk.i_start << ", " << chunk.i_end << "]"
                      << ", Ecm range = [" << chunk.Ecm_start << ", " << chunk.Ecm_end << "]"
                      << ", total_dim = " << chunk.total_dim
                      << ", count = " << chunk.count
                      << '\n';
        }

        gpu_queue.push(std::move(chunk));

        current.clear();
        current_bytes = 64ull * 1024ull * 1024ull;
    };

    for (auto& item : buffer)
    {
        if (item.total_dim <= 0 || item.vdim <= 0)
        {
            continue;
        }

        if (item.total_dim != total_dim)
        {
            throw std::runtime_error("flush_cpu_buffer_to_gpu_queue received mixed total_dim buffer.");
        }

        size_t item_bytes = estimate_gpu_bytes_one_energy(item.total_dim, item.vdim);

        if (item_bytes > gpu_budget_bytes)
        {
            std::ostringstream os;
            os << "Single energy matrix does not fit GPU budget. "
               << "i = " << item.i
               << ", total_dim = " << item.total_dim
               << ", vdim = " << item.vdim
               << ", item_bytes = " << item_bytes
               << ", gpu_budget_bytes = " << gpu_budget_bytes;
            throw std::runtime_error(os.str());
        }

        if (!current.empty() && current_bytes + item_bytes > gpu_budget_bytes)
        {
            push_current();
        }

        current_bytes += item_bytes;
        current.push_back(std::move(item));
    }

    push_current();
    buffer.clear();
}

//==============================================================================
// Main rewritten pipeline function
//==============================================================================

void test_F3_with_pwave_all_energy_gpu_omp_cublas_pipeline_v3(
    std::vector<int>& nnP_vec,
    std::string irrep,
    std::string irrep_tag)
{
    char debug = 'y';

    comp pi = std::acos(-1.0);

    double atmpi = 0.06906;
    double atmK  = 0.09698;

    double eta_1 = 1.0;
    double eta_2 = 0.5;

    double alpha         = 0.5;
    double max_shell_num = 20.0;

    std::vector<int> waves_vec_1 = {0, 1};
    std::vector<int> waves_vec_2 = {0};

    double epsilon_h = 0.0;
    bool Q0norm = true;

    double xi    = 3.444;
    double Lbyas = 20.0;
    double L     = xi * Lbyas;

    comp twopibyL = ((comp)2.0) * pi / ((comp)L);

    Vec3 nnP = {nnP_vec[0], nnP_vec[1], nnP_vec[2]};

    std::vector<comp> total_P(3);
    total_P[0] = twopibyL * double(nnP[0]);
    total_P[1] = twopibyL * double(nnP[1]);
    total_P[2] = twopibyL * double(nnP[2]);

    std::vector<comp> nnP_config(3);
    nnP_config[0] = comp(nnP[0], 0.0);
    nnP_config[1] = comp(nnP[1], 0.0);
    nnP_config[2] = comp(nnP[2], 0.0);

    double tolerance = 1.0e-12;

    std::vector<std::vector<comp>> scatter_params_1(4, std::vector<comp>(3));
    scatter_params_1[0][0] = 4.04;
    scatter_params_1[0][1] = 0.0;
    scatter_params_1[0][2] = 0.0;

    scatter_params_1[1][0] = -43.2;
    scatter_params_1[1][1] = 0.0;
    scatter_params_1[1][2] = 0.0;

    std::vector<std::vector<comp>> scatter_params_2(4, std::vector<comp>(3));
    scatter_params_2[0][0] = 4.12;
    scatter_params_2[0][1] = 0.0;
    scatter_params_2[0][2] = 0.0;

    double Ecm_initial = 0.26310;
    double Ecm_final   = 0.36;
    int    Ecm_points  = 10000;

    double del_Ecm = std::abs(Ecm_initial - Ecm_final) / double(Ecm_points);

    std::string I = irrep;
    std::string irrep_tag_for_file = irrep_tag;

    bool sort_orbit_flag = false;
    int parity = -1;

    double eig_tol  = 0.05;
    double norm_tol = 1.0e-12;
    double proj_tol = 1.0e-10;

    double max_norm_value = 2.0;

    int omp_threads = 18;
    omp_set_num_threads(omp_threads);

    bool save_large_matrices = false;

    if (debug == 'y')
    {
        std::cout << std::setprecision(16);
        std::cout << "total_nP = "
                  << nnP_config[0] << ", "
                  << nnP_config[1] << ", "
                  << nnP_config[2] << '\n';

        std::cout << "L = " << L << '\n';
        std::cout << "Ecm range = [" << Ecm_initial << ", " << Ecm_final << "]\n";
        std::cout << "Ecm_points = " << Ecm_points << '\n';
        std::cout << "irrep = " << I << '\n';
    }

    std::vector<comp> Ecm_vec(Ecm_points);
    std::vector<comp> En_vec(Ecm_points);

    for (int i = 0; i < Ecm_points; ++i)
    {
        double Ecm = Ecm_initial + double(i) * del_Ecm;
        comp Ecm_c(Ecm, 0.0);
        comp En_c = Ecm_to_E(Ecm, total_P);

        Ecm_vec[size_t(i)] = Ecm_c;
        En_vec[size_t(i)] = En_c;
    }

    size_t freeB = 0;
    size_t totalB = 0;
    F3_CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB), "cudaMemGetInfo");

    double vram_safety_fraction = 0.80;
    size_t gpu_budget_bytes = size_t(double(freeB) * vram_safety_fraction);

    if (debug == 'y')
    {
        std::cout << "GPU free bytes = " << freeB << '\n';
        std::cout << "GPU total bytes = " << totalB << '\n';
        std::cout << "GPU budget bytes = " << gpu_budget_bytes << '\n';
    }

    ThreadSafeQueue<CpuGpuChunk> gpu_queue;

    std::vector<EnergyGpuResult> all_results(Ecm_points);
    std::mutex result_mutex;

    std::thread gpu_thread(
        gpu_consumer_thread_func,
        std::ref(gpu_queue),
        std::ref(all_results),
        std::ref(result_mutex),
        save_large_matrices,
        debug
    );

    int chunk_counter = 0;
    int cpu_block_size = 64;

    std::vector<EnergyMatrixPack> dim_buffer;
    int current_dim = -1;

    {
        F3ScopedTimer total_timer("TOTAL CPU producer sweep", debug);

        for (int block_start = 0; block_start < Ecm_points; block_start += cpu_block_size)
        {
            int block_end = std::min(Ecm_points, block_start + cpu_block_size);
            int block_count = block_end - block_start;

            std::vector<EnergyMatrixPack> block_packs(static_cast<size_t>(block_count));

            {
                F3ScopedTimer cpu_block_timer(
                    "CPU build block i = [" +
                    std::to_string(block_start) + ", " +
                    std::to_string(block_end - 1) + "]",
                    debug
                );

                #pragma omp parallel for schedule(dynamic)
                for (int k = 0; k < block_count; ++k)
                {
                    int i = block_start + k;

                    block_packs[size_t(k)] =
                        build_one_energy_pack(
                            i,
                            Ecm_vec[size_t(i)],
                            En_vec[size_t(i)],
                            waves_vec_1,
                            waves_vec_2,
                            total_P,
                            nnP_config,
                            I,
                            atmK,
                            atmpi,
                            L,
                            alpha,
                            epsilon_h,
                            max_shell_num,
                            tolerance,
                            eta_1,
                            eta_2,
                            scatter_params_1,
                            scatter_params_2,
                            Q0norm,
                            sort_orbit_flag,
                            parity,
                            eig_tol,
                            norm_tol,
                            proj_tol,
                            'n'
                        );
                }
            }

            if (debug == 'y')
            {
                int first_dim = -1;
                int last_dim = -1;

                if (!block_packs.empty())
                {
                    first_dim = block_packs.front().total_dim;
                    last_dim = block_packs.back().total_dim;
                }

                std::cout << "[CPU] finished block i = ["
                          << block_start << ", " << block_end - 1 << "]"
                          << ", Ecm = [" << Ecm_vec[size_t(block_start)]
                          << ", " << Ecm_vec[size_t(block_end - 1)] << "]"
                          << ", first_dim = " << first_dim
                          << ", last_dim = " << last_dim
                          << '\n';
            }

            for (auto& pack : block_packs)
            {
                if (pack.total_dim <= 0 || pack.vdim <= 0)
                {
                    continue;
                }

                if (current_dim < 0)
                {
                    current_dim = pack.total_dim;
                }

                if (pack.total_dim != current_dim)
                {
                    flush_cpu_buffer_to_gpu_queue(
                        dim_buffer,
                        gpu_queue,
                        chunk_counter,
                        gpu_budget_bytes,
                        debug
                    );

                    current_dim = pack.total_dim;
                }

                dim_buffer.push_back(std::move(pack));
            }
        }

        flush_cpu_buffer_to_gpu_queue(
            dim_buffer,
            gpu_queue,
            chunk_counter,
            gpu_budget_bytes,
            debug
        );
    }

    gpu_queue.close();

    if (gpu_thread.joinable())
    {
        gpu_thread.join();
    }

    if (debug == 'y')
    {
        std::cout << "All CPU/GPU chunks finished.\n";
    }

    //=========================================================================
    // Save raw output
    //=========================================================================

    std::vector<comp> det_proj_F3i_vec;
    std::vector<comp> Ecm_valid_vec;

    det_proj_F3i_vec.reserve(Ecm_points);
    Ecm_valid_vec.reserve(Ecm_points);

    std::string raw_output_filename =
        "det_proj_F3i_raw_" +
        std::to_string(nnP[0]) +
        std::to_string(nnP[1]) +
        std::to_string(nnP[2]) +
        "_" +
        irrep_tag_for_file +
        "_L20.dat";

    std::ofstream fout_raw(raw_output_filename.c_str());

    if (!fout_raw.is_open())
    {
        throw std::runtime_error("Could not open output file: " + raw_output_filename);
    }

    fout_raw << std::setprecision(40);

    fout_raw << "# i"
             << '\t' << "En_real"
             << '\t' << "En_imag"
             << '\t' << "Ecm_real"
             << '\t' << "Ecm_imag"
             << '\t' << "total_dim"
             << '\t' << "vdim"
             << '\t' << "success"
             << '\t' << "gpu_info_H"
             << '\t' << "gpu_info_F3"
             << '\t' << "gpu_info_proj"
             << '\t' << "eig_projF3i_real"
             << '\t' << "eig_projF3i_imag"
             << '\t' << "det_projF3i_real"
             << '\t' << "det_projF3i_imag"
             << '\n';

    for (int i = 0; i < Ecm_points; ++i)
    {
        const auto& r = all_results[size_t(i)];

        if (!r.success)
        {
            continue;
        }

        if (!std::isfinite(r.det_F3inv_projected.real()) ||
            !std::isfinite(r.det_F3inv_projected.imag()))
        {
            continue;
        }

        Ecm_valid_vec.push_back(r.Ecm);
        det_proj_F3i_vec.push_back(r.det_F3inv_projected);

        fout_raw << i
                 << '\t' << r.En.real()
                 << '\t' << r.En.imag()
                 << '\t' << r.Ecm.real()
                 << '\t' << r.Ecm.imag()
                 << '\t' << r.total_dim
                 << '\t' << r.vdim
                 << '\t' << r.success
                 << '\t' << r.gpu_info_H
                 << '\t' << r.gpu_info_F3
                 << '\t' << r.gpu_info_proj
                 << '\t' << r.eig_F3inv_projected.real()
                 << '\t' << r.eig_F3inv_projected.imag()
                 << '\t' << r.det_F3inv_projected.real()
                 << '\t' << r.det_F3inv_projected.imag()
                 << '\n';

        if (debug == 'y' && i % 100 == 0)
        {
            std::cout << "i = " << i
                      << '\t' << "En = " << r.En
                      << '\t' << "Ecm = " << r.Ecm
                      << '\t' << "total_dim = " << r.total_dim
                      << '\t' << "vdim = " << r.vdim
                      << '\t' << "detF3i = " << r.det_F3inv_projected
                      << '\n';
        }
    }

    fout_raw.close();

    if (debug == 'y')
    {
        std::cout << "Saved raw determinant file to: "
                  << raw_output_filename << '\n';
    }

    //=========================================================================
    // Normalize determinant output
    //=========================================================================

    char norm_debug = debug;

    std::vector<comp> det_proj_F3i_vec_normalized =
        normalize_det_vector_by_max(
            det_proj_F3i_vec,
            Ecm_valid_vec,
            max_norm_value,
            norm_debug
        );

    std::string output_filename =
        "det_proj_F3i_normalized_" +
        std::to_string(nnP[0]) +
        std::to_string(nnP[1]) +
        std::to_string(nnP[2]) +
        "_" +
        irrep_tag_for_file +
        "_L20.dat";

    std::ofstream fout(output_filename.c_str());

    if (!fout.is_open())
    {
        throw std::runtime_error("Could not open output file: " + output_filename);
    }

    fout << std::setprecision(16);

    fout << "# i"
         << '\t' << "Ecm_real"
         << '\t' << "Ecm_imag"
         << '\t' << "det_norm_real"
         << '\t' << "det_norm_imag"
         << '\t' << "abs_det_norm"
         << '\n';

    for (std::size_t i = 0; i < det_proj_F3i_vec_normalized.size(); ++i)
    {
        fout << i
             << '\t' << Ecm_valid_vec[i].real()
             << '\t' << Ecm_valid_vec[i].imag()
             << '\t' << det_proj_F3i_vec_normalized[i].real()
             << '\t' << det_proj_F3i_vec_normalized[i].imag()
             << '\t' << std::abs(det_proj_F3i_vec_normalized[i])
             << '\n';
    }

    fout.close();

    if (debug == 'y')
    {
        std::cout << "Saved normalized determinant vector to: "
                  << output_filename << '\n';
    }
}

//==============================================================================
// Optional demo main wrapper.
// Define BUILD_F3_PIPELINE_MAIN if you want this file to produce an executable.
//==============================================================================

#ifdef BUILD_F3_PIPELINE_MAIN
int main()
{
    try
    {
        std::vector<int> nnP_vec = {1, 1, 0};
        std::string irrep = "A2";
        std::string irrep_tag = "A2";

        test_F3_with_pwave_all_energy_gpu_omp_cublas_pipeline_v3(
            nnP_vec,
            irrep,
            irrep_tag
        );
    }
    catch (const std::exception& e)
    {
        std::cerr << "FAILED: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
#endif
