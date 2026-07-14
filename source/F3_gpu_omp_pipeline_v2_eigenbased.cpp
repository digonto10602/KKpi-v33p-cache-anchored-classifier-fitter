//=============================================================================
// F3_gpu_omp_pipeline_v2.cpp
//
// Single-file pipeline rewrite for CPU OpenMP matrix construction + GPU/solve
// consumer workflow.
//
// Main replacement function:
//     test_F3_with_pwave_all_energy_gpu_omp_pipeline_v2(...)
//
// Important:
//   1. This file assumes your existing project already defines:
//        comp, Vec3, Ecm_to_E, config_maker_4, F2_2plus1_mat,
//        K2inv_EREord2_2plus1_mat, G_2plus1_mat,
//        P_irrep_projection_2plus1,
//        build_projector_from_eigenvectors_near_one,
//        smallest_eigenvalue, normalize_det_vector_by_max,
//        CUDA_CHECK, etc.
//
//   2. The function gpu_process_same_dim_chunk_cusolver(...) currently contains
//      a CPU Eigen fallback so the new pipeline can be tested immediately.
//      Replace the body of that function with cuSOLVER/cuBLAS solves/GEMMs.
//
//   3. This version removes the CPU FullPivLU invertibility pre-check.
//      Solve/factorization failure should be detected inside the GPU solver by
//      the LU/cusolver info arrays.
//
//   4. The mathematical flow is:
//        H X1 = F2
//        F3 = F2/3 - F2 X1
//        F3 X2 = Vsel
//        F3inv_projected = Vsel.adjoint() X2
//        det = det(F3inv_projected)
//
//=============================================================================

#include <Eigen/Dense>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// CUDA/cuBLAS/cuSOLVER headers. If your project includes these elsewhere,
// keeping them here is still usually fine.
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

//=============================================================================
// If your project already defines comp and Vec3, remove these fallback aliases.
//=============================================================================
#ifndef F3_PIPELINE_EXTERNAL_TYPES
using comp = std::complex<double>;
using Vec3 = std::vector<int>;
#endif

//=============================================================================
// Forward declarations for existing project functions.
// Remove these if your project headers already declare them before including this
// file. The signatures here are based on your uploaded/current driver code.
//=============================================================================
#ifndef F3_PIPELINE_EXTERNAL_DECLARATIONS

comp Ecm_to_E(double Ecm, const std::vector<comp>& total_P);

void config_maker_4(
    std::vector<std::vector<comp>>& plm_config,
    std::vector<std::vector<int>>& np_config,
    const std::vector<int>& waves_vec,
    comp En,
    const std::vector<comp>& total_P,
    double mi,
    double mj,
    double mk,
    double L,
    double epsilon_h,
    double max_shell_num,
    double tolerance
);

void F2_2plus1_mat(
    Eigen::MatrixXcd& F2,
    comp En,
    const std::vector<std::vector<comp>>& plm_config,
    const std::vector<std::vector<comp>>& klm_config,
    const std::vector<comp>& total_P,
    double atmK,
    double atmpi,
    double L,
    double alpha,
    double epsilon_h,
    double max_shell_num,
    bool Q0norm
);

void K2inv_EREord2_2plus1_mat(
    Eigen::MatrixXcd& K2inv,
    double eta_1,
    double eta_2,
    const std::vector<std::vector<comp>>& scatter_params_1,
    const std::vector<std::vector<comp>>& scatter_params_2,
    comp En,
    const std::vector<std::vector<comp>>& plm_config,
    const std::vector<std::vector<comp>>& klm_config,
    const std::vector<comp>& total_P,
    double atmK,
    double atmpi,
    double epsilon_h,
    double L
);

void G_2plus1_mat(
    Eigen::MatrixXcd& G,
    comp En,
    const std::vector<std::vector<comp>>& plm_config,
    const std::vector<std::vector<comp>>& klm_config,
    const std::vector<comp>& total_P,
    double atmK,
    double atmpi,
    double L,
    double alpha,
    double epsilon_h,
    double max_shell_num,
    bool Q0norm
);

void P_irrep_projection_2plus1(
    Eigen::MatrixXcd& P_I,
    const std::vector<std::vector<comp>>& plm_config,
    const std::vector<std::vector<int>>& np_config,
    const std::vector<std::vector<comp>>& klm_config,
    const std::vector<std::vector<int>>& nk_config,
    const std::string& I,
    const std::vector<comp>& total_P,
    const std::vector<comp>& nnP_config,
    bool sort_orbit_flag,
    int parity
);

void build_projector_from_eigenvectors_near_one(
    const Eigen::MatrixXcd& P_I,
    Eigen::MatrixXcd& Vsel,
    Eigen::MatrixXcd& Pproj,
    double eig_tol,
    double norm_tol,
    double proj_tol,
    char debug
);

comp smallest_eigenvalue(const Eigen::MatrixXcd& M);

std::vector<comp> normalize_det_vector_by_max(
    const std::vector<comp>& det_vec,
    const std::vector<comp>& Ecm_vec,
    double max_norm_value,
    char debug
);

#endif // F3_PIPELINE_EXTERNAL_DECLARATIONS

//=============================================================================
// CUDA check helpers. If your project already defines CUDA_CHECK, you can remove
// these or rename them.
//=============================================================================
#ifndef CUDA_CHECK
#define CUDA_CHECK(call, msg)                                                     \
    do                                                                            \
    {                                                                             \
        cudaError_t err__ = (call);                                                \
        if (err__ != cudaSuccess)                                                  \
        {                                                                         \
            std::ostringstream os__;                                               \
            os__ << "CUDA error at " << __FILE__ << ":" << __LINE__              \
                 << " in " << (msg) << " : " << cudaGetErrorString(err__);       \
            throw std::runtime_error(os__.str());                                  \
        }                                                                         \
    } while (0)
#endif

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(call, msg)                                                   \
    do                                                                            \
    {                                                                             \
        cublasStatus_t st__ = (call);                                              \
        if (st__ != CUBLAS_STATUS_SUCCESS)                                         \
        {                                                                         \
            std::ostringstream os__;                                               \
            os__ << "cuBLAS error at " << __FILE__ << ":" << __LINE__             \
                 << " in " << (msg) << " : status = " << int(st__);              \
            throw std::runtime_error(os__.str());                                  \
        }                                                                         \
    } while (0)
#endif

#ifndef CUSOLVER_CHECK
#define CUSOLVER_CHECK(call, msg)                                                 \
    do                                                                            \
    {                                                                             \
        cusolverStatus_t st__ = (call);                                            \
        if (st__ != CUSOLVER_STATUS_SUCCESS)                                       \
        {                                                                         \
            std::ostringstream os__;                                               \
            os__ << "cuSOLVER error at " << __FILE__ << ":" << __LINE__           \
                 << " in " << (msg) << " : status = " << int(st__);              \
            throw std::runtime_error(os__.str());                                  \
        }                                                                         \
    } while (0)
#endif

//=============================================================================
// Timer
//=============================================================================
struct ScopedTimer
{
    std::string name;
    char debug;
    std::chrono::high_resolution_clock::time_point t0;

    ScopedTimer(const std::string& name_, char debug_)
        : name(name_),
          debug(debug_),
          t0(std::chrono::high_resolution_clock::now())
    {}

    ~ScopedTimer()
    {
        if (debug == 'y')
        {
            auto t1 = std::chrono::high_resolution_clock::now();
            double sec = std::chrono::duration<double>(t1 - t0).count();
            std::cout << "[TIMER] " << name << " : " << sec << " s\n";
        }
    }
};

//=============================================================================
// Thread-safe queue
//=============================================================================
template <class T>
class ThreadSafeQueue
{
private:
    std::queue<T> q_;
    std::mutex m_;
    std::condition_variable cv_;
    bool closed_ = false;

public:
    void push(T item)
    {
        {
            std::lock_guard<std::mutex> lock(m_);
            if (closed_)
            {
                throw std::runtime_error("Cannot push to closed queue.");
            }
            q_.push(std::move(item));
        }
        cv_.notify_one();
    }

    bool pop(T& item)
    {
        std::unique_lock<std::mutex> lock(m_);
        cv_.wait(lock, [&]() { return closed_ || !q_.empty(); });

        if (q_.empty())
        {
            return false;
        }

        item = std::move(q_.front());
        q_.pop();
        return true;
    }

    void close()
    {
        {
            std::lock_guard<std::mutex> lock(m_);
            closed_ = true;
        }
        cv_.notify_all();
    }
};

//=============================================================================
// Data containers
//=============================================================================
struct EnergyMatrixPack
{
    int i = -1;

    comp Ecm = comp(0.0, 0.0);
    comp En  = comp(0.0, 0.0);

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

    Eigen::MatrixXcd X1;                 // H X1 = F2
    Eigen::MatrixXcd F3;                 // F2/3 - F2 X1
    Eigen::MatrixXcd X2;                 // F3 X2 = Vsel
    Eigen::MatrixXcd F3inv_projected;    // Vsel.adjoint() X2

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
};

struct CpuGpuChunk
{
    int chunk_id = -1;

    int i_start = -1;
    int i_end = -1;

    comp Ecm_start = comp(0.0, 0.0);
    comp Ecm_end   = comp(0.0, 0.0);

    int total_dim = 0;
    int count = 0;

    std::vector<EnergyMatrixPack> items;
};

//=============================================================================
// VRAM estimator
//=============================================================================
inline size_t bytes_complex_matrix(int rows, int cols)
{
    return size_t(rows) * size_t(cols) * sizeof(cuDoubleComplex);
}

inline size_t estimate_gpu_bytes_one_energy(int n, int vdim)
{
    size_t B = 0;

    // H, F2, X1, F3
    B += bytes_complex_matrix(n, n); // H
    B += bytes_complex_matrix(n, n); // F2
    B += bytes_complex_matrix(n, n); // X1
    B += bytes_complex_matrix(n, n); // F3

    // Vsel, X2
    B += bytes_complex_matrix(n, vdim); // Vsel
    B += bytes_complex_matrix(n, vdim); // X2

    // projected matrix
    B += bytes_complex_matrix(vdim, vdim);

    // LU pivots/info/pointer arrays/workspaces/safety padding
    B += size_t(64) * size_t(n) * size_t(n);
    B += 8ull * 1024ull * 1024ull;

    return B;
}

//=============================================================================
// CPU build one energy
//=============================================================================
EnergyMatrixPack build_one_energy_pack(
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

    P_irrep_projection_2plus1(
        P_I,
        plm_config,
        np_config,
        klm_config,
        nk_config,
        I,
        total_P,
        nnP_config,
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

//=============================================================================
// GPU processing function.
//
// This currently contains a CPU Eigen fallback. Replace the internals with actual
// cuSOLVER/cuBLAS code when ready.
//=============================================================================
void gpu_process_same_dim_chunk_cusolver(
    const CpuGpuChunk& chunk,
    std::vector<EnergyGpuResult>& results,
    char debug)
{
    ScopedTimer timer("GPU total chunk " + std::to_string(chunk.chunk_id), debug);

    if (debug == 'y')
    {
        std::cout << "[GPU] received chunk_id = " << chunk.chunk_id
                  << ", i range = [" << chunk.i_start << ", " << chunk.i_end << "]"
                  << ", Ecm range = [" << chunk.Ecm_start << ", " << chunk.Ecm_end << "]"
                  << ", total_dim = " << chunk.total_dim
                  << ", count = " << chunk.count
                  << '\n';
    }

    //=========================================================================
    // CPU fallback starts here.
    // Replace this with cuSOLVER/cuBLAS:
    //     1. Factor H
    //     2. Solve H X1 = F2
    //     3. GEMM F2 X1
    //     4. Build F3 = F2/3 - F2 X1
    //     5. Factor F3
    //     6. Solve F3 X2 = Vsel
    //     7. GEMM Vsel^H X2
    //=========================================================================

    results.resize(chunk.items.size());

    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < int(chunk.items.size()); ++j)
    {
        const auto& item = chunk.items[size_t(j)];

        EnergyGpuResult out;
        out.i = item.i;
        out.Ecm = item.Ecm;
        out.En = item.En;
        out.total_dim = item.total_dim;
        out.vdim = item.vdim;

        try
        {
            if (item.total_dim <= 0 || item.vdim <= 0)
            {
                results[size_t(j)] = std::move(out);
                continue;
            }

            Eigen::PartialPivLU<Eigen::MatrixXcd> luH(item.H);
            out.X1 = luH.solve(item.F2);

            out.F3.resize(item.total_dim, item.total_dim);
            out.F3.noalias() = item.F2 / 3.0 - item.F2 * out.X1;

            Eigen::PartialPivLU<Eigen::MatrixXcd> luF3(out.F3);
            out.X2 = luF3.solve(item.Vsel);

            out.F3inv_projected.noalias() = item.Vsel.adjoint() * out.X2;

            if (out.F3inv_projected.rows() > 0 && out.F3inv_projected.cols() > 0)
            {
                out.det_F3inv_projected = out.F3inv_projected.determinant();
                out.eig_F3inv_projected = smallest_eigenvalue(out.F3inv_projected);
            }

            out.success =
                std::isfinite(out.det_F3inv_projected.real()) &&
                std::isfinite(out.det_F3inv_projected.imag());

            out.gpu_info_H = 0;
            out.gpu_info_F3 = 0;
        }
        catch (...)
        {
            out.success = false;
            out.gpu_info_H = -999;
            out.gpu_info_F3 = -999;
        }

        results[size_t(j)] = std::move(out);
    }
}

//=============================================================================
// GPU consumer thread
//=============================================================================
void gpu_consumer_thread_func(
    ThreadSafeQueue<CpuGpuChunk>& gpu_queue,
    std::vector<EnergyGpuResult>& all_results,
    std::mutex& result_mutex,
    char debug)
{
    CpuGpuChunk chunk;

    while (gpu_queue.pop(chunk))
    {
        std::vector<EnergyGpuResult> chunk_results;

        {
            ScopedTimer timer(
                "GPU process chunk_id = " + std::to_string(chunk.chunk_id),
                debug
            );

            gpu_process_same_dim_chunk_cusolver(
                chunk,
                chunk_results,
                debug
            );
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

    if (debug == 'y')
    {
        std::cout << "[GPU] consumer finished.\n";
    }
}

//=============================================================================
// Flush a same-dimension CPU buffer to GPU-safe chunks.
//=============================================================================
void flush_cpu_buffer_to_gpu_queue(
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
    size_t current_bytes = 32ull * 1024ull * 1024ull;

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
        current_bytes = 32ull * 1024ull * 1024ull;
    };

    for (auto& item : buffer)
    {
        if (item.total_dim <= 0 || item.vdim <= 0)
        {
            continue;
        }

        if (item.total_dim != total_dim)
        {
            throw std::runtime_error(
                "flush_cpu_buffer_to_gpu_queue received mixed total_dim buffer."
            );
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

//=============================================================================
// Main rewritten pipeline function
//=============================================================================
void test_F3_with_pwave_all_energy_gpu_omp_pipeline_v2(
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
    nnP_config[0] = comp(double(nnP[0]), 0.0);
    nnP_config[1] = comp(double(nnP[1]), 0.0);
    nnP_config[2] = comp(double(nnP[2]), 0.0);

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

    int omp_threads = 16;
#ifdef _OPENMP
    omp_set_num_threads(omp_threads);
#endif

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

    std::vector<comp> Ecm_vec(size_t(Ecm_points));
    std::vector<comp> En_vec(size_t(Ecm_points));

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
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB), "cudaMemGetInfo");

    double vram_safety_fraction = 0.80;
    size_t gpu_budget_bytes = size_t(double(freeB) * vram_safety_fraction);

    if (debug == 'y')
    {
        std::cout << "GPU free bytes = " << freeB << '\n';
        std::cout << "GPU total bytes = " << totalB << '\n';
        std::cout << "GPU budget bytes = " << gpu_budget_bytes << '\n';
    }

    ThreadSafeQueue<CpuGpuChunk> gpu_queue;

    std::vector<EnergyGpuResult> all_results(size_t(Ecm_points));
    std::mutex result_mutex;

    std::thread gpu_thread(
        gpu_consumer_thread_func,
        std::ref(gpu_queue),
        std::ref(all_results),
        std::ref(result_mutex),
        debug
    );

    int chunk_counter = 0;

    int cpu_block_size = 64;

    std::vector<EnergyMatrixPack> dim_buffer;
    int current_dim = -1;

    {
        ScopedTimer total_timer("TOTAL CPU producer sweep", debug);

        for (int block_start = 0; block_start < Ecm_points; block_start += cpu_block_size)
        {
            int block_end = std::min(Ecm_points, block_start + cpu_block_size);
            int block_count = block_end - block_start;

            std::vector<EnergyMatrixPack> block_packs(size_t(block_count));

            {
                ScopedTimer cpu_block_timer(
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

    det_proj_F3i_vec.reserve(size_t(Ecm_points));
    Ecm_valid_vec.reserve(size_t(Ecm_points));

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
                      << '\t' << "eigval = " << r.eig_F3inv_projected
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

//=============================================================================
// Optional mini-driver. Enable only if you want this file to produce a binary
// directly and your project links all required definitions.
//=============================================================================
#ifdef F3_PIPELINE_STANDALONE_MAIN
int main()
{
    std::vector<int> nnP_vec = {1, 1, 0};
    test_F3_with_pwave_all_energy_gpu_omp_pipeline_v2(nnP_vec, "A2", "A2");
    return 0;
}
#endif

