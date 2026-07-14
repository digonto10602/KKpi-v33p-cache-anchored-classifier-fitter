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
#include <optional>
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
// HDF5 disabled in this CPU/OpenMP-reference v21 file.
// #include "F3_matrix_hdf5_saver.hpp"
enum class F3Hdf5OpenMode { Truncate, AppendOrCreate };

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

    // Optional full-matrix payload. These are filled only when matrix
    // HDF5 saving is enabled, because keeping them for every energy can
    // use a lot of host RAM.
    Eigen::MatrixXcd F2;
    Eigen::MatrixXcd G;
    Eigen::MatrixXcd K2inv;
    Eigen::MatrixXcd Vsel;

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

    // Estimated GPU memory usage for this chunk, including chunk-level overhead.
    // This is for logging/chunking only; actual cuBLAS cache allocation may differ.
    size_t estimated_gpu_bytes = 0;

    std::vector<EnergyMatrixPack> items;
};

struct F3MatrixSaveOptions
{
    // Master switch. When false, the pipeline behaves like v7.
    bool enabled = false;

    // Empty filename means: build an automatic filename using nnP, irrep_tag,
    // scan_tag, and Lbyas.
    std::string filename;

    // Truncate is safest for one scan. AppendOrCreate is useful if you are
    // intentionally adding groups to an existing file.
    F3Hdf5OpenMode open_mode = F3Hdf5OpenMode::Truncate;

    int gzip_level = 4;
    char debug = 'n';

    bool save_F2 = true;
    bool save_G = true;
    bool save_K2inv = true;
    bool save_F3 = true;
    bool save_Vsel = true;

    // Optional filters. Leave as defaults to save every successful energy.
    int save_every_n = 1;        // save every N accepted points
    int index_min = 0;
    int index_max = std::numeric_limits<int>::max();
    double Ecm_min = -std::numeric_limits<double>::infinity();
    double Ecm_max =  std::numeric_limits<double>::infinity();

    bool overwrite_existing_epoint = true;
};

static inline bool f3_should_save_matrix_point(
    const F3MatrixSaveOptions& opt,
    int local_index,
    const EnergyGpuResult& r)
{
    if (!opt.enabled || !r.success) return false;
    if (local_index < opt.index_min || local_index > opt.index_max) return false;
    if (opt.save_every_n > 1 && (local_index % opt.save_every_n) != 0) return false;

    const double Ecm = r.Ecm.real();
    if (Ecm < opt.Ecm_min || Ecm > opt.Ecm_max) return false;

    return true;
}

static inline std::string f3_make_auto_hdf5_matrix_filename(
    const Vec3& nnP,
    const std::string& irrep_tag_for_file,
    const std::string& scan_tag,
    double Lbyas)
{
    std::ostringstream os;
    os << "F3_matrix_dump_"
       << nnP[0] << nnP[1] << nnP[2]
       << "_" << irrep_tag_for_file;

    if (!scan_tag.empty())
    {
        os << scan_tag;
    }

    os << "_L" << std::setprecision(12) << Lbyas << ".h5";
    return os.str();
}

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
    // Per-energy memory only.
    // IMPORTANT: do not put a large safety/workspace pad here, otherwise tiny
    // n=40 matrices get artificially limited to a few hundred matrices/chunk.
    // Padding belongs at the chunk level in flush_cpu_buffer_to_gpu_queue().
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
    B += 3 * sizeof(int);                  // rough per-energy info contribution

    return B;
}

static inline double bytes_to_mib(size_t bytes)
{
    return double(bytes) / 1024.0 / 1024.0;
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
        // Free all cached device buffers.  We intentionally synchronize before
        // and after release so that an OOM retry really gets the memory back
        // before the next smaller allocation attempt.
        cudaDeviceSynchronize();
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
        cudaDeviceSynchronize();
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

        // Avoid uncontrolled cache growth across changing matrix sizes.
        // The old doubling rule could allocate for far more matrices than the
        // chunk actually contains after n changes, which can cause artificial
        // OOM.  If n or vdim changes, allocate almost exactly what is needed.
        // If the same shape grows, allow only a small growth margin.
        const bool same_shape = (n == n_in && max_vdim >= max_vdim_in);
        int newCap = 0;
        if (same_shape && cap > 0)
        {
            int grown = cap + std::max(16, cap / 4);
            newCap = std::max(batchCount, grown);
        }
        else
        {
            newCap = std::max(batchCount, 1);
        }

        int newV = std::max(max_vdim_in, 1);

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

        if (save_large_matrices)
        {
            // Preserve CPU-built matrices so they can be written to HDF5 later.
            // F3 is copied back from GPU below.
            out.F2 = item.F2;
            out.G = item.G;
            out.K2inv = item.K2inv;
            out.Vsel = item.Vsel;
        }

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
                  << ", est_gpu_mem = " << bytes_to_mib(chunk.estimated_gpu_bytes) << " MiB"
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
// OOM-safe GPU chunk retry helpers
//==============================================================================

static inline bool f3_exception_looks_like_cuda_oom(const std::exception& e)
{
    std::string msg = e.what();
    return msg.find("out of memory") != std::string::npos ||
           msg.find("cudaErrorMemoryAllocation") != std::string::npos ||
           msg.find("cudaMalloc") != std::string::npos;
}

static inline size_t estimate_gpu_bytes_for_items(
    const std::vector<EnergyMatrixPack>& items,
    size_t begin,
    size_t end,
    size_t chunk_overhead_bytes)
{
    size_t B = chunk_overhead_bytes;
    for (size_t k = begin; k < end; ++k)
    {
        const auto& item = items[k];
        if (item.total_dim > 0 && item.vdim > 0)
        {
            B += estimate_gpu_bytes_one_energy(item.total_dim, item.vdim);
        }
    }
    return B;
}

static CpuGpuChunk make_subchunk_copy(
    const CpuGpuChunk& parent,
    size_t begin,
    size_t end,
    int sub_id,
    size_t chunk_overhead_bytes)
{
    CpuGpuChunk out;
    out.chunk_id = sub_id;
    out.total_dim = parent.total_dim;

    if (begin >= end || begin >= parent.items.size())
    {
        out.count = 0;
        return out;
    }

    end = std::min(end, parent.items.size());
    out.items.reserve(end - begin);
    for (size_t k = begin; k < end; ++k)
    {
        out.items.push_back(parent.items[k]);
    }

    out.count = int(out.items.size());
    out.i_start = out.items.front().i;
    out.i_end = out.items.back().i;
    out.Ecm_start = out.items.front().Ecm;
    out.Ecm_end = out.items.back().Ecm;
    out.estimated_gpu_bytes = estimate_gpu_bytes_for_items(parent.items, begin, end, chunk_overhead_bytes);
    return out;
}

static void append_failed_results_for_chunk(
    const CpuGpuChunk& chunk,
    std::vector<EnergyGpuResult>& results)
{
    for (const auto& item : chunk.items)
    {
        EnergyGpuResult out;
        out.i = item.i;
        out.Ecm = item.Ecm;
        out.En = item.En;
        out.total_dim = item.total_dim;
        out.vdim = item.vdim;
        out.success = false;
        results.push_back(std::move(out));
    }
}

static void gpu_process_chunk_oom_safe_recursive(
    cublasHandle_t handle,
    PipelineGpuCache& cache,
    const CpuGpuChunk& chunk,
    std::vector<EnergyGpuResult>& results,
    bool save_large_matrices,
    char debug,
    int depth = 0)
{
    if (chunk.items.empty())
    {
        return;
    }

    // Before asking the cache to allocate, compare the chunk estimate against
    // *current* free VRAM.  This matters because other allocations, CUDA context
    // overhead, display usage, and previous cached buffers can change available
    // memory during the run.
    size_t freeB = 0;
    size_t totalB = 0;
    cudaMemGetInfo(&freeB, &totalB);

    const double runtime_safety_fraction = 0.65; // intentionally conservative
    const size_t runtime_budget = size_t(double(freeB) * runtime_safety_fraction);

    if (chunk.count > 1 && chunk.estimated_gpu_bytes > runtime_budget)
    {
        size_t mid = chunk.items.size() / 2;
        const size_t overhead = 512ull * 1024ull * 1024ull;
        CpuGpuChunk left  = make_subchunk_copy(chunk, 0, mid, chunk.chunk_id * 10 + 1, overhead);
        CpuGpuChunk right = make_subchunk_copy(chunk, mid, chunk.items.size(), chunk.chunk_id * 10 + 2, overhead);

        if (debug == 'y')
        {
            std::cout << "[GPU] pre-splitting chunk_id = " << chunk.chunk_id
                      << " because est_gpu_mem = " << bytes_to_mib(chunk.estimated_gpu_bytes)
                      << " MiB > runtime_budget = " << bytes_to_mib(runtime_budget)
                      << " MiB, count = " << chunk.count
                      << " -> " << left.count << " + " << right.count << '\n';
        }

        gpu_process_chunk_oom_safe_recursive(handle, cache, left, results, save_large_matrices, debug, depth + 1);
        gpu_process_chunk_oom_safe_recursive(handle, cache, right, results, save_large_matrices, debug, depth + 1);
        return;
    }

    try
    {
        std::vector<EnergyGpuResult> local_results;
        gpu_process_same_dim_chunk_cublas(
            handle,
            cache,
            chunk,
            local_results,
            save_large_matrices,
            debug
        );

        for (auto& r : local_results)
        {
            results.push_back(std::move(r));
        }
    }
    catch (const std::exception& e)
    {
        const bool oom = f3_exception_looks_like_cuda_oom(e);

        // Release cache before retrying.  This is critical because an OOM can
        // occur halfway through PipelineGpuCache::ensure(), leaving partial
        // allocations alive until release() is called.
        cache.release();
        cudaGetLastError();
        cudaDeviceSynchronize();

        if (oom && chunk.count > 1)
        {
            size_t mid = chunk.items.size() / 2;
            const size_t overhead = 512ull * 1024ull * 1024ull;
            CpuGpuChunk left  = make_subchunk_copy(chunk, 0, mid, chunk.chunk_id * 10 + 1, overhead);
            CpuGpuChunk right = make_subchunk_copy(chunk, mid, chunk.items.size(), chunk.chunk_id * 10 + 2, overhead);

            std::cerr << "[GPU] OOM in chunk_id = " << chunk.chunk_id
                      << ": " << e.what() << '\n'
                      << "[GPU] retrying as two smaller chunks: "
                      << left.count << " + " << right.count << '\n';

            gpu_process_chunk_oom_safe_recursive(handle, cache, left, results, save_large_matrices, debug, depth + 1);
            gpu_process_chunk_oom_safe_recursive(handle, cache, right, results, save_large_matrices, debug, depth + 1);
            return;
        }

        std::cerr << "[GPU] ERROR in chunk_id = " << chunk.chunk_id
                  << ": " << e.what() << '\n';
        append_failed_results_for_chunk(chunk, results);
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

        gpu_process_chunk_oom_safe_recursive(
            handle,
            cache,
            chunk,
            chunk_results,
            save_large_matrices,
            debug
        );

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
    int max_mats_per_gpu_chunk,
    char debug)
{
    if (buffer.empty())
    {
        return;
    }

    int total_dim = buffer.front().total_dim;

    std::vector<EnergyMatrixPack> current;

    // Chunk-level overhead only. This covers pointer arrays, allocator/cache slack,
    // and conservative safety margin. The old version effectively added large
    // padding per energy, which is why n=40 was limited to about 341 matrices.
    const size_t chunk_overhead_bytes = 512ull * 1024ull * 1024ull;
    size_t current_bytes = chunk_overhead_bytes;

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
        chunk.estimated_gpu_bytes = current_bytes;
        chunk.items = std::move(current);

        if (debug == 'y')
        {
            std::cout << "[CPU -> GPU] sending chunk_id = " << chunk.chunk_id
                      << ", i range = [" << chunk.i_start << ", " << chunk.i_end << "]"
                      << ", Ecm range = [" << chunk.Ecm_start << ", " << chunk.Ecm_end << "]"
                      << ", total_dim = " << chunk.total_dim
                      << ", count = " << chunk.count
                      << ", est_gpu_mem = " << bytes_to_mib(chunk.estimated_gpu_bytes) << " MiB"
                      << '\n';
        }

        gpu_queue.push(std::move(chunk));

        current.clear();
        current_bytes = chunk_overhead_bytes;
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

        if (!current.empty() &&
            (current_bytes + item_bytes > gpu_budget_bytes ||
             int(current.size()) >= max_mats_per_gpu_chunk))
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


//==============================================================================
// Adaptive scan / zero-pole classification helpers
//==============================================================================

struct F3ScanRunFiles
{
    std::string raw_output_filename;
    std::string normalized_output_filename;
    double Ecm_initial = 0.0;
    double Ecm_final = 0.0;
    int Ecm_points = 0;
    std::string scan_tag;
};

struct SignFlipCandidate
{
    int left_index = -1;
    int right_index = -1;
    int window_left_index = -1;
    int window_right_index = -1;

    double Ecm_left = 0.0;
    double Ecm_right = 0.0;
    double Ecm_mid = 0.0;
    double Ecm_zero_linear = 0.0;
    double Ecm_window_left = 0.0;
    double Ecm_window_right = 0.0;

    double y_left = 0.0;
    double y_right = 0.0;
    double min_abs_y_window = 0.0;
    double max_abs_y_window = 0.0;
    double spike_ratio = 0.0;
    double slope_ratio = 0.0;

    std::string classification;
};

struct DetNormRow
{
    int i = -1;
    double Ecm = 0.0;
    double y = 0.0;
    double abs_y = 0.0;
};

static inline int f3_sign_double(double x)
{
    if (x > 0.0) return 1;
    if (x < 0.0) return -1;
    return 0;
}

static std::vector<DetNormRow> read_normalized_det_file_for_sign_flips(
    const std::string& input_filename)
{
    std::ifstream fin(input_filename.c_str());
    if (!fin.is_open())
    {
        throw std::runtime_error("Could not open normalized determinant file: " + input_filename);
    }

    std::vector<DetNormRow> rows;
    std::string line;

    while (std::getline(fin, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }

        std::istringstream iss(line);
        DetNormRow r;
        double Ecm_imag = 0.0;
        double y_imag = 0.0;

        // Expected normalized-file columns:
        // i Ecm_real Ecm_imag det_norm_real det_norm_imag abs_det_norm
        if (!(iss >> r.i >> r.Ecm >> Ecm_imag >> r.y >> y_imag >> r.abs_y))
        {
            continue;
        }

        if (std::isfinite(r.Ecm) && std::isfinite(r.y) && std::isfinite(r.abs_y))
        {
            rows.push_back(r);
        }
    }

    return rows;
}

static double linear_zero_estimate(double x1, double y1, double x2, double y2)
{
    double denom = y2 - y1;
    if (std::abs(denom) <= 1.0e-300)
    {
        return 0.5 * (x1 + x2);
    }
    return x1 - y1 * (x2 - x1) / denom;
}

std::vector<SignFlipCandidate> classify_sign_flips_zero_vs_pole(
    const std::string& input_filename,
    const std::string& output_filename = "classified_zero_pole_candidates.dat",
    int window = 5,
    double small_y_threshold = 1.0e-3,
    double spike_ratio_threshold = 1.0e2,
    double slope_ratio_threshold = 1.0e2,
    char debug = 'n')
{
    std::vector<DetNormRow> rows = read_normalized_det_file_for_sign_flips(input_filename);
    std::vector<SignFlipCandidate> candidates;

    if (rows.size() < 2)
    {
        return candidates;
    }

    const double tiny = 1.0e-300;

    for (int j = 0; j + 1 < int(rows.size()); ++j)
    {
        double y1 = rows[size_t(j)].y;
        double y2 = rows[size_t(j + 1)].y;

        int s1 = f3_sign_double(y1);
        int s2 = f3_sign_double(y2);

        if (s1 == 0 || s2 == 0 || s1 == s2)
        {
            continue;
        }

        int a = std::max(0, j - window);
        int b = std::min(int(rows.size()) - 1, j + 1 + window);

        double min_abs_y = std::numeric_limits<double>::infinity();
        double max_abs_y = 0.0;

        for (int k = a; k <= b; ++k)
        {
            double ay = std::abs(rows[size_t(k)].y);
            min_abs_y = std::min(min_abs_y, ay);
            max_abs_y = std::max(max_abs_y, ay);
        }

        double spike_ratio = max_abs_y / std::max(min_abs_y, tiny);

        std::vector<double> slopes;
        for (int k = a; k < b; ++k)
        {
            double dx = rows[size_t(k + 1)].Ecm - rows[size_t(k)].Ecm;
            if (std::abs(dx) > 0.0)
            {
                slopes.push_back(std::abs((rows[size_t(k + 1)].y - rows[size_t(k)].y) / dx));
            }
        }

        double min_slope = std::numeric_limits<double>::infinity();
        double max_slope = 0.0;
        for (double sl : slopes)
        {
            min_slope = std::min(min_slope, sl);
            max_slope = std::max(max_slope, sl);
        }

        double slope_ratio = slopes.empty() ? 1.0 : max_slope / std::max(min_slope, tiny);

        SignFlipCandidate c;
        c.left_index = rows[size_t(j)].i;
        c.right_index = rows[size_t(j + 1)].i;
        c.window_left_index = rows[size_t(a)].i;
        c.window_right_index = rows[size_t(b)].i;
        c.Ecm_left = rows[size_t(j)].Ecm;
        c.Ecm_right = rows[size_t(j + 1)].Ecm;
        c.Ecm_mid = 0.5 * (c.Ecm_left + c.Ecm_right);
        c.Ecm_zero_linear = linear_zero_estimate(c.Ecm_left, y1, c.Ecm_right, y2);
        c.Ecm_window_left = rows[size_t(a)].Ecm;
        c.Ecm_window_right = rows[size_t(b)].Ecm;
        c.y_left = y1;
        c.y_right = y2;
        c.min_abs_y_window = min_abs_y;
        c.max_abs_y_window = max_abs_y;
        c.spike_ratio = spike_ratio;
        c.slope_ratio = slope_ratio;

        const bool small_near_zero =
            (std::abs(y1) < small_y_threshold) ||
            (std::abs(y2) < small_y_threshold) ||
            (min_abs_y < small_y_threshold);

        const bool spike_like =
            (spike_ratio > spike_ratio_threshold) &&
            (slope_ratio > slope_ratio_threshold) &&
            !small_near_zero;

        const bool smooth_zero_like =
            small_near_zero &&
            (spike_ratio < spike_ratio_threshold || min_abs_y < small_y_threshold * 1.0e-2);

        if (smooth_zero_like)
        {
            c.classification = "likely_zero";
        }
        else if (spike_like)
        {
            c.classification = "likely_pole";
        }
        else
        {
            c.classification = "ambiguous";
        }

        candidates.push_back(c);
    }

    std::ofstream fout(output_filename.c_str());
    if (!fout.is_open())
    {
        throw std::runtime_error("Could not open classification output file: " + output_filename);
    }

    fout << std::setprecision(17);
    fout << "# input_file = " << input_filename << '\n';
    fout << "# window = " << window
         << " small_y_threshold = " << small_y_threshold
         << " spike_ratio_threshold = " << spike_ratio_threshold
         << " slope_ratio_threshold = " << slope_ratio_threshold << '\n';
    fout << "# left_index\tright_index\tEcm_left\tEcm_right\tEcm_zero_linear\tEcm_window_left\tEcm_window_right\t"
         << "y_left\ty_right\tmin_abs_y_window\tmax_abs_y_window\tspike_ratio\tslope_ratio\tclassification\n";

    for (const auto& c : candidates)
    {
        fout << c.left_index << '\t'
             << c.right_index << '\t'
             << c.Ecm_left << '\t'
             << c.Ecm_right << '\t'
             << c.Ecm_zero_linear << '\t'
             << c.Ecm_window_left << '\t'
             << c.Ecm_window_right << '\t'
             << c.y_left << '\t'
             << c.y_right << '\t'
             << c.min_abs_y_window << '\t'
             << c.max_abs_y_window << '\t'
             << c.spike_ratio << '\t'
             << c.slope_ratio << '\t'
             << c.classification << '\n';

        if (debug == 'y')
        {
            std::cout << "[classify] sign flip: Ecm = ["
                      << c.Ecm_left << ", " << c.Ecm_right << "]"
                      << ", zero_est = " << c.Ecm_zero_linear
                      << ", min_abs = " << c.min_abs_y_window
                      << ", spike_ratio = " << c.spike_ratio
                      << ", slope_ratio = " << c.slope_ratio
                      << ", class = " << c.classification
                      << '\n';
        }
    }

    fout.close();

    if (debug == 'y')
    {
        std::cout << "Saved sign-flip classification file to: "
                  << output_filename << '\n';
    }

    return candidates;
}

static std::string make_scan_tag(const std::string& base, int id)
{
    std::ostringstream os;
    os << "_" << base << "_" << id;
    return os.str();
}

static void add_unique_zero(std::vector<double>& zeros, double z, double tol)
{
    if (!std::isfinite(z))
    {
        return;
    }

    for (double old : zeros)
    {
        if (std::abs(old - z) < tol)
        {
            return;
        }
    }

    zeros.push_back(z);
}

F3ScanRunFiles run_F3_energy_scan_gpu_omp_cublas_pipeline_v4(
    std::vector<int>& nnP_vec,
    std::string irrep,
    std::string irrep_tag,
    double Ecm_initial_input,
    double Ecm_final_input,
    int Ecm_points_input,
    const std::string& scan_tag,
    double Lbyas_input = 20.0,
    char debug_input = 'y',
    F3MatrixSaveOptions matrix_save_options = F3MatrixSaveOptions())
{
    char debug = debug_input;

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
    double Lbyas = Lbyas_input;
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

    double Ecm_initial = Ecm_initial_input;
    double Ecm_final   = Ecm_final_input;
    int    Ecm_points  = Ecm_points_input;

    double del_Ecm = std::abs(Ecm_initial - Ecm_final) / double(Ecm_points);

    std::string I = irrep;
    std::string irrep_tag_for_file = irrep_tag;

    if (matrix_save_options.enabled && matrix_save_options.filename.empty())
    {
        matrix_save_options.filename =
            f3_make_auto_hdf5_matrix_filename(nnP, irrep_tag_for_file, scan_tag, Lbyas);
    }

    bool sort_orbit_flag = false;
    int parity = -1;

    double eig_tol  = 0.05;
    double norm_tol = 1.0e-12;
    double proj_tol = 1.0e-10;

    double max_norm_value = 2.0;

    int omp_threads = 18;
    omp_set_num_threads(omp_threads);

    // Copy F3 back from GPU and preserve F2/G/K2inv/Vsel only when HDF5
    // matrix saving is enabled. This avoids extra D2H transfer and host RAM
    // use in normal determinant-only scans.
    bool save_large_matrices = matrix_save_options.enabled;

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

    // Cap the number of matrices per GPU chunk independently of VRAM.
    // Start conservative. The GPU consumer now also auto-splits on OOM, so
    // you can raise this after confirming stable memory behavior.
    int max_mats_per_gpu_chunk = 2048;

    if (debug == 'y')
    {
        std::cout << "GPU free bytes = " << freeB << '\n';
        std::cout << "GPU total bytes = " << totalB << '\n';
        std::cout << "GPU budget bytes = " << gpu_budget_bytes
                  << " (" << bytes_to_mib(gpu_budget_bytes) << " MiB)" << '\n';
        std::cout << "max_mats_per_gpu_chunk = " << max_mats_per_gpu_chunk << '\n';
        if (matrix_save_options.enabled)
        {
            std::cout << "HDF5 matrix saving enabled: "
                      << matrix_save_options.filename << '\n'
                      << "  save_F2=" << matrix_save_options.save_F2
                      << ", save_G=" << matrix_save_options.save_G
                      << ", save_K2inv=" << matrix_save_options.save_K2inv
                      << ", save_F3=" << matrix_save_options.save_F3
                      << ", save_Vsel=" << matrix_save_options.save_Vsel
                      << ", save_every_n=" << matrix_save_options.save_every_n
                      << '\n';
        }
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
    int cpu_block_size = 512;

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
                        max_mats_per_gpu_chunk,
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
            max_mats_per_gpu_chunk,
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
    // HDF5 matrix dump disabled in this CPU/OpenMP-reference v21 file.
    //=========================================================================

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
        scan_tag +
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
        scan_tag +
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

    F3ScanRunFiles scan_files;
    scan_files.raw_output_filename = raw_output_filename;
    scan_files.normalized_output_filename = output_filename;
    scan_files.Ecm_initial = Ecm_initial;
    scan_files.Ecm_final = Ecm_final;
    scan_files.Ecm_points = Ecm_points;
    scan_files.scan_tag = scan_tag;
    return scan_files;

}


// Backward-compatible wrapper with the old name and old default 10000-point behavior.
void test_F3_with_pwave_all_energy_gpu_omp_cublas_pipeline_v3(
    std::vector<int>& nnP_vec,
    std::string irrep,
    std::string irrep_tag)
{
    run_F3_energy_scan_gpu_omp_cublas_pipeline_v4(
        nnP_vec,
        irrep,
        irrep_tag,
        0.26310,
        0.36,
        10000,
        "",
        'y'
    );
}

// Convenience wrapper: same 10000-point scan as pipeline_v3, but also saves
// selected matrices to HDF5 groups /epoint_0, /epoint_1, ... .
void test_F3_with_pwave_all_energy_gpu_omp_cublas_pipeline_v8_save_matrices(
    std::vector<int>& nnP_vec,
    std::string irrep,
    std::string irrep_tag,
    F3MatrixSaveOptions matrix_save_options)
{
    matrix_save_options.enabled = true;

    run_F3_energy_scan_gpu_omp_cublas_pipeline_v4(
        nnP_vec,
        irrep,
        irrep_tag,
        0.26310,
        0.36,
        10000,
        "",
        'y',
        matrix_save_options
    );
}



//==============================================================================
// CPU/OpenMP-reference v21 side-absolute-trend refinement helpers
//==============================================================================

struct SideAbsRefineOptions
{
    int initial_refine_points = 50;
    int max_zoom_rounds = 12;

    // Side-wise trend requirements. A zero requires both sides to decrease
    // inward toward the sign flip. A pole requires both sides to increase inward.
    double trend_fraction_required = 0.65;
    double score_margin_required = 0.20;

    double duplicate_Ecm_tolerance = 1.0e-7;
    char debug = 'n';
};

struct SideAbsTrendResult
{
    int signflip_local_left_index = -1;
    int signflip_local_right_index = -1;

    double Ecm_left = std::numeric_limits<double>::quiet_NaN();
    double Ecm_right = std::numeric_limits<double>::quiet_NaN();
    double y_left = std::numeric_limits<double>::quiet_NaN();
    double y_right = std::numeric_limits<double>::quiet_NaN();
    double Ecm_zero_linear = std::numeric_limits<double>::quiet_NaN();

    int left_decreasing_steps = 0;
    int left_increasing_steps = 0;
    int left_flat_steps = 0;
    int right_decreasing_steps = 0;
    int right_increasing_steps = 0;
    int right_flat_steps = 0;

    double left_decreasing_fraction = 0.0;
    double left_increasing_fraction = 0.0;
    double right_decreasing_fraction = 0.0;
    double right_increasing_fraction = 0.0;

    double decreasing_fraction = 0.0;
    double increasing_fraction = 0.0;
    double trend_score = 0.0;

    double min_abs_y = std::numeric_limits<double>::quiet_NaN();
    double max_abs_y = std::numeric_limits<double>::quiet_NaN();

    std::string classification = "ambiguous";
};

static inline bool f3_has_sign_flip(double y1, double y2)
{
    if (!std::isfinite(y1) || !std::isfinite(y2)) return false;
    if (y1 == 0.0 || y2 == 0.0) return true;
    return (y1 > 0.0 && y2 < 0.0) || (y1 < 0.0 && y2 > 0.0);
}

static std::vector<int> find_all_sign_flips_in_rows(const std::vector<DetNormRow>& rows)
{
    std::vector<int> flips;
    for (int j = 0; j + 1 < int(rows.size()); ++j)
    {
        if (f3_has_sign_flip(rows[size_t(j)].y, rows[size_t(j + 1)].y)) flips.push_back(j);
    }
    return flips;
}

static int find_sign_flip_nearest_middle(const std::vector<DetNormRow>& rows)
{
    std::vector<int> flips = find_all_sign_flips_in_rows(rows);
    if (flips.empty()) return -1;

    const double center = 0.5 * double(rows.size() - 1);
    double best_dist = std::numeric_limits<double>::infinity();
    int best = -1;

    for (int j : flips)
    {
        const double mid = double(j) + 0.5;
        const double dist = std::abs(mid - center);
        if (dist < best_dist)
        {
            best_dist = dist;
            best = j;
        }
    }
    return best;
}


static SideAbsTrendResult classify_one_signflip_side_abs_trend_v22(
    const std::vector<DetNormRow>& rows,
    int flip_j,
    int left_outer_index,
    int right_outer_index,
    const SideAbsRefineOptions& opt)
{
    SideAbsTrendResult out;
    const int N = int(rows.size());

    if (N < 4 || flip_j < 0 || flip_j + 1 >= N)
    {
        out.classification = "ambiguous";
        return out;
    }

    left_outer_index = std::max(0, std::min(left_outer_index, flip_j));
    right_outer_index = std::min(N - 1, std::max(right_outer_index, flip_j + 1));

    out.signflip_local_left_index = flip_j;
    out.signflip_local_right_index = flip_j + 1;

    const auto& L = rows[size_t(flip_j)];
    const auto& R = rows[size_t(flip_j + 1)];

    out.Ecm_left = L.Ecm;
    out.Ecm_right = R.Ecm;
    out.y_left = L.y;
    out.y_right = R.y;
    out.Ecm_zero_linear = linear_zero_estimate(L.Ecm, L.y, R.Ecm, R.y);

    auto abs_y = [&](int idx) -> double { return std::abs(rows[size_t(idx)].y); };

    out.min_abs_y = std::numeric_limits<double>::infinity();
    out.max_abs_y = 0.0;
    for (int i = left_outer_index; i <= right_outer_index; ++i)
    {
        if (std::isfinite(rows[size_t(i)].y))
        {
            const double a = std::abs(rows[size_t(i)].y);
            out.min_abs_y = std::min(out.min_abs_y, a);
            out.max_abs_y = std::max(out.max_abs_y, a);
        }
    }
    if (!std::isfinite(out.min_abs_y)) out.min_abs_y = std::numeric_limits<double>::quiet_NaN();

    auto classify_step = [](double outer_val, double inner_val) -> int
    {
        const double eps = 1.0e-12 * std::max({1.0, std::abs(outer_val), std::abs(inner_val)});
        if (inner_val < outer_val - eps) return -1; // decreasing inward toward sign flip => zero-like
        if (inner_val > outer_val + eps) return +1; // increasing inward toward sign flip => pole-like
        return 0;
    };

    // Left side inward direction: left_outer_index -> flip_j.
    for (int i = left_outer_index; i < flip_j; ++i)
    {
        const int t = classify_step(abs_y(i), abs_y(i + 1));
        if (t < 0) out.left_decreasing_steps++;
        else if (t > 0) out.left_increasing_steps++;
        else out.left_flat_steps++;
    }

    // Right side inward direction: right_outer_index -> flip_j+1.
    for (int i = right_outer_index; i > flip_j + 1; --i)
    {
        const int t = classify_step(abs_y(i), abs_y(i - 1));
        if (t < 0) out.right_decreasing_steps++;
        else if (t > 0) out.right_increasing_steps++;
        else out.right_flat_steps++;
    }

    const int left_total = out.left_decreasing_steps + out.left_increasing_steps + out.left_flat_steps;
    const int right_total = out.right_decreasing_steps + out.right_increasing_steps + out.right_flat_steps;

    if (left_total <= 0 || right_total <= 0)
    {
        out.classification = "ambiguous";
        return out;
    }

    out.left_decreasing_fraction = double(out.left_decreasing_steps) / double(left_total);
    out.left_increasing_fraction = double(out.left_increasing_steps) / double(left_total);
    out.right_decreasing_fraction = double(out.right_decreasing_steps) / double(right_total);
    out.right_increasing_fraction = double(out.right_increasing_steps) / double(right_total);

    out.decreasing_fraction = 0.5 * (out.left_decreasing_fraction + out.right_decreasing_fraction);
    out.increasing_fraction = 0.5 * (out.left_increasing_fraction + out.right_increasing_fraction);
    out.trend_score = out.decreasing_fraction - out.increasing_fraction;

    const bool both_decrease_inward =
        (out.left_decreasing_fraction >= opt.trend_fraction_required) &&
        (out.right_decreasing_fraction >= opt.trend_fraction_required);

    const bool both_increase_inward =
        (out.left_increasing_fraction >= opt.trend_fraction_required) &&
        (out.right_increasing_fraction >= opt.trend_fraction_required);

    const double left_margin = std::abs(out.left_decreasing_fraction - out.left_increasing_fraction);
    const double right_margin = std::abs(out.right_decreasing_fraction - out.right_increasing_fraction);
    const bool decisive =
        (left_margin >= opt.score_margin_required) &&
        (right_margin >= opt.score_margin_required);

    if (both_decrease_inward && decisive)
    {
        out.classification = "likely_zero";
    }
    else if (both_increase_inward && decisive)
    {
        out.classification = "likely_pole";
    }
    else
    {
        out.classification = "ambiguous";
    }

    return out;
}

static std::vector<SideAbsTrendResult> classify_refined_file_by_multi_side_abs_trend_v22(
    const std::string& normalized_filename,
    const std::string& classified_output_filename,
    const SideAbsRefineOptions& opt)
{
    std::vector<DetNormRow> rows = read_normalized_det_file_for_sign_flips(normalized_filename);
    std::vector<SideAbsTrendResult> results;

    const int N = int(rows.size());
    std::vector<int> flips = find_all_sign_flips_in_rows(rows);

    if (N < 4 || flips.empty())
    {
        SideAbsTrendResult out;
        out.classification = flips.empty() ? "no_sign_flip" : "ambiguous";
        results.push_back(out);
    }
    else
    {
        for (int k = 0; k < int(flips.size()); ++k)
        {
            const int f = flips[size_t(k)];

            int left_outer = 0;
            int right_outer = N - 1;

            if (k > 0)
            {
                const int prev_f = flips[size_t(k - 1)];
                // Boundary between neighboring sign flips. Previous right point is prev_f+1,
                // current left point is f. Use the midpoint region as the outer edge for this crossing.
                left_outer = (prev_f + 1 + f) / 2;
                left_outer = std::min(left_outer, f);
            }

            if (k + 1 < int(flips.size()))
            {
                const int next_f = flips[size_t(k + 1)];
                // Boundary between current right point f+1 and next left point next_f.
                right_outer = (f + 1 + next_f + 1) / 2;
                right_outer = std::max(right_outer, f + 1);
            }

            SideAbsTrendResult out = classify_one_signflip_side_abs_trend_v22(
                rows,
                f,
                left_outer,
                right_outer,
                opt);
            results.push_back(out);
        }
    }

    std::ofstream fout(classified_output_filename.c_str());
    if (!fout.is_open())
    {
        throw std::runtime_error("Could not open multi-side-abs classification output file: " + classified_output_filename);
    }

    fout << std::setprecision(17);
    fout << "# input_file = " << normalized_filename << '\n';
    fout << "# method = cpu_openmp_v22_multi_signflip_side_abs_inward_trend\n";
    fout << "# trend_fraction_required = " << opt.trend_fraction_required
         << " score_margin_required = " << opt.score_margin_required << '\n';
    fout << "# num_sign_flips = " << (results.size() == 1 && results[0].classification == "no_sign_flip" ? 0 : results.size()) << '\n';
    fout << "# row\tsignflip_local_left_index\tsignflip_local_right_index\tEcm_left\tEcm_right\t"
         << "y_left\ty_right\tEcm_zero_linear\t"
         << "left_decreasing_fraction\tleft_increasing_fraction\t"
         << "right_decreasing_fraction\tright_increasing_fraction\t"
         << "decreasing_fraction\tincreasing_fraction\ttrend_score\t"
         << "left_decreasing_steps\tleft_increasing_steps\tleft_flat_steps\t"
         << "right_decreasing_steps\tright_increasing_steps\tright_flat_steps\t"
         << "min_abs_y\tmax_abs_y\tclassification\n";

    for (int r = 0; r < int(results.size()); ++r)
    {
        const auto& out = results[size_t(r)];
        fout << r << '\t'
             << out.signflip_local_left_index << '\t'
             << out.signflip_local_right_index << '\t'
             << out.Ecm_left << '\t'
             << out.Ecm_right << '\t'
             << out.y_left << '\t'
             << out.y_right << '\t'
             << out.Ecm_zero_linear << '\t'
             << out.left_decreasing_fraction << '\t'
             << out.left_increasing_fraction << '\t'
             << out.right_decreasing_fraction << '\t'
             << out.right_increasing_fraction << '\t'
             << out.decreasing_fraction << '\t'
             << out.increasing_fraction << '\t'
             << out.trend_score << '\t'
             << out.left_decreasing_steps << '\t'
             << out.left_increasing_steps << '\t'
             << out.left_flat_steps << '\t'
             << out.right_decreasing_steps << '\t'
             << out.right_increasing_steps << '\t'
             << out.right_flat_steps << '\t'
             << out.min_abs_y << '\t'
             << out.max_abs_y << '\t'
             << out.classification << '\n';
    }

    return results;
}

static void add_unique_zero_cpu_v21(std::vector<double>& zeros, double z, double tol)
{
    if (!std::isfinite(z)) return;
    for (double old : zeros)
    {
        if (std::abs(old - z) < tol) return;
    }
    zeros.push_back(z);
}


static std::string format_Lbyas_for_filename(double Lbyas)
{
    std::ostringstream os;
    os << std::setprecision(12) << Lbyas;
    std::string x = os.str();
    while (!x.empty() && x.find('.') != std::string::npos && x.back() == '0') x.pop_back();
    if (!x.empty() && x.back() == '.') x.pop_back();
    for (char& c : x) { if (c == '.') c = 'p'; }
    return x;
}

struct CpuV22RefineTask
{
    int root_candidate = -1;
    int branch_id = 0;
    int round = 0;
    double E_left = std::numeric_limits<double>::quiet_NaN();
    double E_right = std::numeric_limits<double>::quiet_NaN();
    std::string parent_note;
};

void test_F3_with_pwave_all_energy_cpu_openmp_v22_multi_signflip_side_abs_refinement(
    std::vector<int>& nnP_vec,
    std::string irrep,
    std::string irrep_tag,
    int coarse_Ecm_points = 1000,
    int refine_Ecm_points = 50,
    double Ecm_initial = 0.263101,
    double Ecm_final = 0.36,
    double Lbyas = 20.0,
    char debug = 'n')
{
    SideAbsRefineOptions opt;
    opt.initial_refine_points = refine_Ecm_points;
    opt.max_zoom_rounds = 12;
    opt.trend_fraction_required = 0.65;
    opt.score_margin_required = 0.20;
    opt.debug = debug;

    F3MatrixSaveOptions no_matrix_save;
    no_matrix_save.enabled = false;

    std::cout << "\n[adaptive-cpu-v22] Starting coarse "
              << coarse_Ecm_points << "-point scan.\n";
    std::cout << "[adaptive-cpu-v22] Matrix ingredients use CPU/OpenMP functions; "
              << "F3 solve uses the old v9 GPU consumer path.\n";
    std::cout << "[adaptive-cpu-v22] classifier = multi_signflip_side_abs_inward, "
              << "refine window = exact sign-flip pair\n";

    F3ScanRunFiles coarse_files = run_F3_energy_scan_gpu_omp_cublas_pipeline_v4(
        nnP_vec,
        irrep,
        irrep_tag,
        Ecm_initial,
        Ecm_final,
        coarse_Ecm_points,
        "_cpu_v22_coarse_N" + std::to_string(coarse_Ecm_points),
        Lbyas,
        debug,
        no_matrix_save
    );

    std::vector<DetNormRow> coarse_rows =
        read_normalized_det_file_for_sign_flips(coarse_files.normalized_output_filename);

    std::vector<int> coarse_flip_indices = find_all_sign_flips_in_rows(coarse_rows);

    std::string coarse_signflip_filename =
        "classified_side_abs_cpu_v22_coarse_signflips_" +
        std::to_string(nnP_vec[0]) +
        std::to_string(nnP_vec[1]) +
        std::to_string(nnP_vec[2]) +
        "_" + irrep_tag +
        "_coarse" + std::to_string(coarse_Ecm_points) +
        "_L20.dat";

    {
        std::ofstream fout(coarse_signflip_filename.c_str());
        if (!fout.is_open())
        {
            throw std::runtime_error("Could not open coarse sign-flip output file: " + coarse_signflip_filename);
        }

        fout << std::setprecision(17);
        fout << "# coarse normalized file = " << coarse_files.normalized_output_filename << '\n';
        fout << "# method = cpu_openmp_v22_multi_signflip_side_abs_exact_signflip_pair\n";
        fout << "# left_index\tright_index\tEcm_left\tEcm_right\tEcm_zero_linear\ty_left\ty_right\n";

        for (int j : coarse_flip_indices)
        {
            const double z = linear_zero_estimate(
                coarse_rows[size_t(j)].Ecm,
                coarse_rows[size_t(j)].y,
                coarse_rows[size_t(j + 1)].Ecm,
                coarse_rows[size_t(j + 1)].y
            );

            fout << coarse_rows[size_t(j)].i << '\t'
                 << coarse_rows[size_t(j + 1)].i << '\t'
                 << coarse_rows[size_t(j)].Ecm << '\t'
                 << coarse_rows[size_t(j + 1)].Ecm << '\t'
                 << z << '\t'
                 << coarse_rows[size_t(j)].y << '\t'
                 << coarse_rows[size_t(j + 1)].y << '\n';
        }
    }

    std::cout << "[adaptive-cpu-v22] Coarse sign flips found = "
              << coarse_flip_indices.size() << '\n';
    std::cout << "[adaptive-cpu-v22] Saved coarse sign-flip list to: "
              << coarse_signflip_filename << '\n';

    std::vector<double> finalized_Ecm_zeros;
    int root_candidate = 0;
    int next_branch_id = 1;

    std::deque<CpuV22RefineTask> tasks;

    for (int coarse_flip_j : coarse_flip_indices)
    {
        double refine_left = coarse_rows[size_t(coarse_flip_j)].Ecm;
        double refine_right = coarse_rows[size_t(coarse_flip_j + 1)].Ecm;
        if (refine_right < refine_left) std::swap(refine_left, refine_right);

        if (refine_right > refine_left)
        {
            CpuV22RefineTask t;
            t.root_candidate = root_candidate;
            t.branch_id = 0;
            t.round = 0;
            t.E_left = refine_left;
            t.E_right = refine_right;
            t.parent_note = "coarse_exact_pair";
            tasks.push_back(t);

            const double coarse_zero_est = linear_zero_estimate(
                coarse_rows[size_t(coarse_flip_j)].Ecm,
                coarse_rows[size_t(coarse_flip_j)].y,
                coarse_rows[size_t(coarse_flip_j + 1)].Ecm,
                coarse_rows[size_t(coarse_flip_j + 1)].y
            );

            std::cout << "\n[adaptive-cpu-v22] Seed candidate "
                      << root_candidate
                      << ": coarse sign flip i = ["
                      << coarse_rows[size_t(coarse_flip_j)].i << ", "
                      << coarse_rows[size_t(coarse_flip_j + 1)].i << "]"
                      << ", exact Ecm window = ["
                      << std::setprecision(17) << refine_left
                      << ", " << refine_right << "]"
                      << ", coarse_zero_est = " << coarse_zero_est << '\n';
        }

        root_candidate++;
    }

    const int Nref = refine_Ecm_points;
    int processed_tasks = 0;
    const int max_total_tasks = std::max(1000, int(coarse_flip_indices.size()) * (opt.max_zoom_rounds + 4) * 4);

    while (!tasks.empty())
    {
        if (++processed_tasks > max_total_tasks)
        {
            std::cout << "[adaptive-cpu-v22] reached max_total_tasks=" << max_total_tasks
                      << "; stopping remaining refinement tasks as ambiguous\n";
            break;
        }

        CpuV22RefineTask task = tasks.front();
        tasks.pop_front();

        if (!(task.E_right > task.E_left))
        {
            std::cout << "[adaptive-cpu-v22] candidate=" << task.root_candidate
                      << " branch=" << task.branch_id
                      << " invalid window; skipping\n";
            continue;
        }

        if (task.round >= opt.max_zoom_rounds)
        {
            std::cout << "[adaptive-cpu-v22] candidate=" << task.root_candidate
                      << " branch=" << task.branch_id
                      << " reached max_zoom_rounds; stopping as ambiguous\n";
            continue;
        }

        std::string refine_tag =
            "_cpu_v22_refine_" + std::to_string(task.root_candidate) +
            "_branch" + std::to_string(task.branch_id) +
            "_round" + std::to_string(task.round) +
            "_N" + std::to_string(Nref);

        std::cout << "[adaptive-cpu-v22] candidate=" << task.root_candidate
                  << " branch=" << task.branch_id
                  << " round=" << task.round
                  << " N=" << Nref
                  << " Ecm_L=" << std::setprecision(17) << task.E_left
                  << " Ecm_R=" << task.E_right
                  << " parent=" << task.parent_note << '\n';

        F3ScanRunFiles refine_files = run_F3_energy_scan_gpu_omp_cublas_pipeline_v4(
            nnP_vec,
            irrep,
            irrep_tag,
            task.E_left,
            task.E_right,
            Nref,
            refine_tag,
            Lbyas,
            debug,
            no_matrix_save
        );

        std::string refined_classified_filename =
            "classified_side_abs_cpu_v22_" +
            std::to_string(nnP_vec[0]) +
            std::to_string(nnP_vec[1]) +
            std::to_string(nnP_vec[2]) +
            "_" + irrep_tag +
            refine_tag +
            "_L20.dat";

        std::vector<SideAbsTrendResult> rr_list = classify_refined_file_by_multi_side_abs_trend_v22(
            refine_files.normalized_output_filename,
            refined_classified_filename,
            opt
        );

        std::vector<DetNormRow> ref_rows =
            read_normalized_det_file_for_sign_flips(refine_files.normalized_output_filename);

        const int n_effective_flips =
            (rr_list.size() == 1 && rr_list[0].classification == "no_sign_flip") ? 0 : int(rr_list.size());

        std::cout << "[adaptive-cpu-v22] candidate=" << task.root_candidate
                  << " branch=" << task.branch_id
                  << " round=" << task.round
                  << " found_sign_flips=" << n_effective_flips
                  << " classified_file=" << refined_classified_filename << '\n';

        for (int sf = 0; sf < int(rr_list.size()); ++sf)
        {
            const SideAbsTrendResult& rr = rr_list[size_t(sf)];
            const std::string& cls = rr.classification;
            const double ez = rr.Ecm_zero_linear;

            std::cout << "[adaptive-cpu-v22] candidate=" << task.root_candidate
                      << " branch=" << task.branch_id
                      << " subflip=" << sf
                      << " round=" << task.round
                      << " N=" << Nref
                      << " class=" << cls
                      << " flip=(" << rr.signflip_local_left_index
                      << "," << rr.signflip_local_right_index << ")"
                      << " left_dec=" << std::setprecision(6) << rr.left_decreasing_fraction
                      << " right_dec=" << rr.right_decreasing_fraction
                      << " left_inc=" << rr.left_increasing_fraction
                      << " right_inc=" << rr.right_increasing_fraction
                      << " dec=" << rr.decreasing_fraction
                      << " inc=" << rr.increasing_fraction
                      << " classifier=multi_signflip_side_abs_cpu_openmp"
                      << " E=" << std::setprecision(17) << ez
                      << " window=[" << task.E_left << "," << task.E_right << "]\n";

            if (cls == "likely_zero")
            {
                add_unique_zero_cpu_v21(finalized_Ecm_zeros, ez, opt.duplicate_Ecm_tolerance);
                continue;
            }

            if (cls == "likely_pole" || cls == "no_sign_flip")
            {
                continue;
            }

            if (rr.signflip_local_left_index < 0 || rr.signflip_local_right_index < 0)
            {
                std::cout << "[adaptive-cpu-v22] candidate=" << task.root_candidate
                          << " branch=" << task.branch_id
                          << " subflip=" << sf
                          << " ambiguous but no usable sign flip; stopping this branch\n";
                continue;
            }

            const int new_il = rr.signflip_local_left_index;
            const int new_ir = rr.signflip_local_right_index;

            if (new_il < 0 || new_ir < 0 || new_il >= int(ref_rows.size()) || new_ir >= int(ref_rows.size()))
            {
                std::cout << "[adaptive-cpu-v22] candidate=" << task.root_candidate
                          << " branch=" << task.branch_id
                          << " subflip=" << sf
                          << " invalid sign-flip indices; stopping this branch\n";
                continue;
            }

            double new_left = ref_rows[size_t(new_il)].Ecm;
            double new_right = ref_rows[size_t(new_ir)].Ecm;
            if (new_right < new_left) std::swap(new_left, new_right);

            const double old_width = std::abs(task.E_right - task.E_left);
            const double new_width = std::abs(new_right - new_left);

            std::cout << "[adaptive-cpu-v22] candidate=" << task.root_candidate
                      << " branch=" << task.branch_id
                      << " subflip=" << sf
                      << " ambiguous -> push exact sign-flip child: "
                      << "old_width=" << std::setprecision(17) << old_width
                      << " new_width=" << new_width
                      << " new_window=[" << new_left << "," << new_right << "]\n";

            if (!(new_width > 0.0) || new_width >= 0.999999 * old_width)
            {
                std::cout << "[adaptive-cpu-v22] candidate=" << task.root_candidate
                          << " branch=" << task.branch_id
                          << " subflip=" << sf
                          << " zoom window did not shrink; stopping this branch as ambiguous\n";
                continue;
            }

            CpuV22RefineTask child;
            child.root_candidate = task.root_candidate;
            child.branch_id = (n_effective_flips > 1) ? next_branch_id++ : task.branch_id;
            child.round = task.round + 1;
            child.E_left = new_left;
            child.E_right = new_right;
            child.parent_note = "from_parent_branch_" + std::to_string(task.branch_id) +
                                "_subflip_" + std::to_string(sf);
            tasks.push_back(child);
        }
    }

    std::sort(finalized_Ecm_zeros.begin(), finalized_Ecm_zeros.end());

    std::string zeros_filename =
        "finalized_Ecm_zeros_" +
        std::to_string(nnP_vec[0]) +
        std::to_string(nnP_vec[1]) +
        std::to_string(nnP_vec[2]) +
        "_" + irrep_tag +
        "_cpu_openmp_v22_multi_side_abs_L" + format_Lbyas_for_filename(Lbyas) + ".dat";

    std::ofstream fout_zero(zeros_filename.c_str());
    if (!fout_zero.is_open())
    {
        throw std::runtime_error("Could not open finalized zero output file: " + zeros_filename);
    }

    fout_zero << std::setprecision(17);
    fout_zero << "# Lbyas\tEcm_zero\n";

    std::cout << "\n[adaptive-cpu-v22] Finalized Ecm zeros:\n";
    std::cout << "# Lbyas\tEcm_zero\n";

    for (double z : finalized_Ecm_zeros)
    {
        fout_zero << Lbyas << '\t' << z << '\n';
        std::cout << Lbyas << '\t' << z << '\n';
    }

    fout_zero.close();
    std::cout << "[adaptive-cpu-v22] Saved finalized zero list to: "
              << zeros_filename << '\n';
}

int main(int argc, char** argv)
{
    try
    {
        int n0 = 0, n1 = 0, n2 = 1;
        std::string irrep = "A2";
        std::string irrep_tag = "A2";
        int coarseN = 1000;
        int refineN = 50;
        double E0 = 0.263101;
        double E1 = 0.36;
        char debug = 'n';

        if (argc >= 4)
        {
            n0 = std::stoi(argv[1]);
            n1 = std::stoi(argv[2]);
            n2 = std::stoi(argv[3]);
        }
        if (argc >= 5) irrep = argv[4];
        if (argc >= 6) irrep_tag = argv[5];
        if (argc >= 7) coarseN = std::stoi(argv[6]);
        if (argc >= 8) refineN = std::stoi(argv[7]);
        if (argc >= 10)
        {
            E0 = std::stod(argv[8]);
            E1 = std::stod(argv[9]);
        }
        if (argc >= 11)
        {
            // Backward compatible behavior:
            //   old arg10 was debug char.
            //   new arg10 may be Lbyas if it starts with a number.
            std::string arg10 = argv[10];
            const bool arg10_is_debug = (arg10.size() == 1 && (arg10[0] == 'y' || arg10[0] == 'n'));
            if (arg10_is_debug)
            {
                debug = arg10[0];
            }
            else
            {
                Lbyas = std::stod(arg10);
                if (argc >= 12) debug = argv[11][0];
            }
        }

        std::vector<int> nnP_vec = {n0, n1, n2};

        std::cout << "# F3 CPU/OpenMP-matrix-builder v22 multi-signflip side-abs refinement\n";
        std::cout << "# nnP=[" << n0 << "," << n1 << "," << n2 << "]"
                  << " irrep=" << irrep
                  << " irrep_tag=" << irrep_tag
                  << " coarseN=" << coarseN
                  << " refineN=" << refineN
                  << " Lbyas=" << std::setprecision(12) << Lbyas
                  << " Ecm=[" << std::setprecision(17) << E0 << "," << E1 << "]\n";

        test_F3_with_pwave_all_energy_cpu_openmp_v22_multi_signflip_side_abs_refinement(
            nnP_vec,
            irrep,
            irrep_tag,
            coarseN,
            refineN,
            E0,
            E1,
            Lbyas,
            debug
        );
    }
    catch (const std::exception& e)
    {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
