#ifndef FUNCTIONS_GPU_CONFIG_MAKER4_CUH
#define FUNCTIONS_GPU_CONFIG_MAKER4_CUH

// ============================================================================
// functions_gpu_config_maker4.cuh
//
// CUDA-safe companion for the config_maker_4 part of functions.h.
//
// Goal:
//   Build the same 5-column configuration produced by config_maker_4 for many
//   energies in parallel on the GPU.
//
// Important:
//   This does NOT replace the full original functions.h. It only ports the
//   scalar pieces needed by config_maker_4 and provides host wrappers that return
//   the usual CPU std::vector<std::vector<...>> containers.
//
// Current config_maker_4 selection in your functions.h is effectively:
//
//   if (En.real() > En_min_plus_for_config(pvec,total_P,mi,mj,mk,0.02,epsilon_h).real())
//       keep state
//
// The cutoff is computed in the CPU code but the final active condition uses
// En_min_plus. This GPU port follows that active condition.
//
// Build requirements:
//   Include this file from a .cu file or from a file compiled by nvcc.
//
// Example:
//   #include "functions_gpu_config_maker4.cuh"
//
//   std::vector<std::vector<std::complex<double>>> plm_config(5);
//   std::vector<std::vector<int>> n_config(5);
//
//   gpu_config_maker_4_single_to_cpu_vectors(
//       plm_config,
//       n_config,
//       waves_vec,
//       En.real(),
//       total_P,
//       mi, mj, mk,
//       L,
//       epsilon_h,
//       max_shell_num,
//       tolerance,
//       'n'
//   );
//
// ============================================================================

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef F4_GPU_CUDA_CHECK
#define F4_GPU_CUDA_CHECK(call)                                                        \
    do {                                                                               \
        cudaError_t f4_err__ = (call);                                                  \
        if (f4_err__ != cudaSuccess) {                                                  \
            std::ostringstream f4_os__;                                                 \
            f4_os__ << "CUDA error at " << __FILE__ << ":" << __LINE__               \
                    << " in " << #call << " : " << cudaGetErrorString(f4_err__);      \
            throw std::runtime_error(f4_os__.str());                                   \
        }                                                                              \
    } while (0)
#endif

namespace f4gpu
{

struct ConfigEntry
{
    double px = 0.0;
    double py = 0.0;
    double pz = 0.0;

    int nx = 0;
    int ny = 0;
    int nz = 0;

    int ell = 0;
    int m = 0;
};

struct ConfigMaker4Params
{
    double Px = 0.0;
    double Py = 0.0;
    double Pz = 0.0;

    double mi = 0.0;
    double mj = 0.0;
    double mk = 0.0;

    double L = 1.0;
    double epsilon_h = 0.0;
    double max_num_shell = 20.0;
    double tolerance = 1.0e-12;

    // Must match your config_maker_4 call:
    // En_min_plus_for_config(..., zval = 0.02, epsilon_h)
    double zval_plus = 0.02;
};

struct Options
{
    double vram_safety_fraction = 0.35;
    int max_energies_per_chunk = 4096;
    int cuda_threads_per_block = 256;
    char debug = 'n';
};

// ---------------------------------------------------------------------------
// Device scalar helpers.
// These are real-valued versions of the active config_maker_4 logic.
// ---------------------------------------------------------------------------

__host__ __device__ inline double f4_pi()
{
    return 3.141592653589793238462643383279502884;
}

__host__ __device__ inline double omega_real(double p_abs, double m)
{
    return sqrt(p_abs * p_abs + m * m);
}

__host__ __device__ inline double norm3(double x, double y, double z)
{
    return sqrt(x * x + y * y + z * z);
}

__host__ __device__ inline double en_min_plus_real(
    double kx,
    double ky,
    double kz,
    double Px,
    double Py,
    double Pz,
    double mi,
    double mj,
    double mk,
    double zval,
    double epsilon_h)
{
    const double k_abs = norm3(kx, ky, kz);
    const double omega_k = omega_real(k_abs, mi);

    const double dx = Px - kx;
    const double dy = Py - ky;
    const double dz = Pz - kz;

    const double Pminusk_sq = dx * dx + dy * dy + dz * dz;

    const double sigma_min_i = fabs(mj * mj - mk * mk)
        + (2.0 * zval * (mj + mk) * fmin(mj, mk)) / (1.0 + epsilon_h);

    return omega_k + sqrt(sigma_min_i + Pminusk_sq);
}

__host__ __device__ inline bool selected_by_config_maker_4_condition(
    double En,
    int nx,
    int ny,
    int nz,
    int ell,
    int m,
    const ConfigMaker4Params& p)
{
    (void)ell;
    (void)m;

    const double twopibyL = 2.0 * f4_pi() / p.L;

    const double px = twopibyL * double(nx);
    const double py = twopibyL * double(ny);
    const double pz = twopibyL * double(nz);

    const double Emin_plus = en_min_plus_real(
        px,
        py,
        pz,
        p.Px,
        p.Py,
        p.Pz,
        p.mi,
        p.mj,
        p.mk,
        p.zval_plus,
        p.epsilon_h
    );

    return En > Emin_plus;
}

// ---------------------------------------------------------------------------
// Host helper: convert waves_vec -> explicit (ell,m) list in the same order as
// the CPU ell_m_vector function.
// ---------------------------------------------------------------------------

inline void build_lm_list_from_waves(
    const std::vector<int>& waves_vec,
    std::vector<int>& ell_list,
    std::vector<int>& m_list)
{
    ell_list.clear();
    m_list.clear();

    for (int ell : waves_vec)
    {
        if (ell == 0)
        {
            ell_list.push_back(0);
            m_list.push_back(0);
        }
        else
        {
            const int ini = -std::abs(ell);
            const int fin =  std::abs(ell);

            for (int m = ini; m <= fin; ++m)
            {
                ell_list.push_back(ell);
                m_list.push_back(m);
            }
        }
    }
}

inline std::size_t candidate_count_per_energy(int nmax, int lm_count)
{
    const std::size_t gridN = std::size_t(2 * nmax + 1);
    return std::size_t(lm_count) * gridN * gridN * gridN;
}

// ---------------------------------------------------------------------------
// Kernel: mark accepted candidates for many energies.
//
// flags layout:
//   flags[e * candidates_per_energy + candidate_id]
//
// candidate_id order exactly follows CPU config_maker_4 loop order:
//   lm loop, then nx, then ny, then nz.
// ---------------------------------------------------------------------------

__global__ void mark_config_maker_4_candidates_kernel(
    const double* __restrict__ En_values,
    int nE,
    ConfigMaker4Params params,
    const int* __restrict__ ell_list,
    const int* __restrict__ m_list,
    int lm_count,
    int nmax,
    std::size_t candidates_per_energy,
    unsigned char* __restrict__ flags)
{
    const std::size_t global_id =
        std::size_t(blockIdx.x) * std::size_t(blockDim.x) + std::size_t(threadIdx.x);

    const std::size_t total = std::size_t(nE) * candidates_per_energy;

    if (global_id >= total)
    {
        return;
    }

    const int e = int(global_id / candidates_per_energy);
    const std::size_t c = global_id - std::size_t(e) * candidates_per_energy;

    const int gridN = 2 * nmax + 1;
    const std::size_t grid3 = std::size_t(gridN) * std::size_t(gridN) * std::size_t(gridN);

    const int lm = int(c / grid3);
    const std::size_t rem0 = c - std::size_t(lm) * grid3;

    const int ix = int(rem0 / (std::size_t(gridN) * std::size_t(gridN)));
    const std::size_t rem1 = rem0 - std::size_t(ix) * std::size_t(gridN) * std::size_t(gridN);
    const int iy = int(rem1 / std::size_t(gridN));
    const int iz = int(rem1 - std::size_t(iy) * std::size_t(gridN));

    const int nx = ix - nmax;
    const int ny = iy - nmax;
    const int nz = iz - nmax;

    const int ell = ell_list[lm];
    const int m   = m_list[lm];

    const double En = En_values[e];

    const bool keep = selected_by_config_maker_4_condition(
        En,
        nx,
        ny,
        nz,
        ell,
        m,
        params
    );

    flags[global_id] = keep ? static_cast<unsigned char>(1) : static_cast<unsigned char>(0);
}

// ---------------------------------------------------------------------------
// Host compact flags into ConfigEntry vectors, preserving exact CPU order.
// ---------------------------------------------------------------------------

inline void compact_flags_to_entries_cpu_order(
    const unsigned char* flags,
    int nE,
    const std::vector<double>& En_values_chunk,
    const ConfigMaker4Params& params,
    const std::vector<int>& ell_list,
    const std::vector<int>& m_list,
    int nmax,
    std::vector<std::vector<ConfigEntry>>& out)
{
    (void)En_values_chunk;

    const int lm_count = int(ell_list.size());
    const int gridN = 2 * nmax + 1;
    const std::size_t grid3 = std::size_t(gridN) * std::size_t(gridN) * std::size_t(gridN);
    const std::size_t candidates = candidate_count_per_energy(nmax, lm_count);
    const double twopibyL = 2.0 * f4_pi() / params.L;

    out.assign(std::size_t(nE), std::vector<ConfigEntry>{});

    for (int e = 0; e < nE; ++e)
    {
        auto& vec = out[std::size_t(e)];
        vec.reserve(256);

        for (int lm = 0; lm < lm_count; ++lm)
        {
            const int ell = ell_list[std::size_t(lm)];
            const int m   = m_list[std::size_t(lm)];

            for (int ix = 0; ix < gridN; ++ix)
            {
                const int nx = ix - nmax;

                for (int iy = 0; iy < gridN; ++iy)
                {
                    const int ny = iy - nmax;

                    for (int iz = 0; iz < gridN; ++iz)
                    {
                        const int nz = iz - nmax;

                        const std::size_t rem0 =
                            std::size_t(ix) * std::size_t(gridN) * std::size_t(gridN)
                            + std::size_t(iy) * std::size_t(gridN)
                            + std::size_t(iz);

                        const std::size_t c = std::size_t(lm) * grid3 + rem0;
                        const std::size_t idx = std::size_t(e) * candidates + c;

                        if (flags[idx] == 0)
                        {
                            continue;
                        }

                        ConfigEntry entry;
                        entry.nx = nx;
                        entry.ny = ny;
                        entry.nz = nz;
                        entry.ell = ell;
                        entry.m = m;
                        entry.px = twopibyL * double(nx);
                        entry.py = twopibyL * double(ny);
                        entry.pz = twopibyL * double(nz);

                        vec.push_back(entry);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Main multi-energy GPU config maker.
//
// Returns vector over energies, each containing ordered ConfigEntry states.
// ---------------------------------------------------------------------------

inline std::vector<std::vector<ConfigEntry>> gpu_config_maker_4_many_entries(
    const std::vector<double>& En_values,
    const std::vector<int>& waves_vec,
    const ConfigMaker4Params& params,
    const Options& options = Options{})
{
    if (En_values.empty())
    {
        return {};
    }

    const int nE_total = int(En_values.size());
    const int nmax = int(std::llround(params.max_num_shell));

    if (nmax < 0)
    {
        throw std::runtime_error("gpu_config_maker_4_many_entries: nmax < 0");
    }

    std::vector<int> ell_list;
    std::vector<int> m_list;
    build_lm_list_from_waves(waves_vec, ell_list, m_list);

    if (ell_list.empty())
    {
        throw std::runtime_error("gpu_config_maker_4_many_entries: empty ell/m list");
    }

    const int lm_count = int(ell_list.size());
    const std::size_t candidates = candidate_count_per_energy(nmax, lm_count);

    if (candidates == 0)
    {
        throw std::runtime_error("gpu_config_maker_4_many_entries: zero candidates");
    }

    std::size_t freeB = 0;
    std::size_t totalB = 0;
    F4_GPU_CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));

    const std::size_t budget = std::size_t(double(freeB) * options.vram_safety_fraction);

    const std::size_t bytes_per_energy_flags = candidates * sizeof(unsigned char);
    const std::size_t bytes_per_energy_En    = sizeof(double);
    const std::size_t overhead = 64ull * 1024ull * 1024ull;

    std::size_t chunk_by_mem = 1;

    if (budget > overhead && bytes_per_energy_flags + bytes_per_energy_En > 0)
    {
        chunk_by_mem = (budget - overhead) / (bytes_per_energy_flags + bytes_per_energy_En);
        if (chunk_by_mem == 0) chunk_by_mem = 1;
    }

    int chunk_size = int(std::min<std::size_t>(
        std::size_t(std::max(1, options.max_energies_per_chunk)),
        chunk_by_mem
    ));

    chunk_size = std::max(1, chunk_size);

    if (options.debug == 'y')
    {
        std::cout << "[gpu_config_maker_4] nE_total = " << nE_total << '\n';
        std::cout << "[gpu_config_maker_4] nmax = " << nmax << '\n';
        std::cout << "[gpu_config_maker_4] lm_count = " << lm_count << '\n';
        std::cout << "[gpu_config_maker_4] candidates/energy = " << candidates << '\n';
        std::cout << "[gpu_config_maker_4] freeB = " << freeB << '\n';
        std::cout << "[gpu_config_maker_4] budget = " << budget << '\n';
        std::cout << "[gpu_config_maker_4] chunk_size = " << chunk_size << '\n';
    }

    int* d_ell = nullptr;
    int* d_m = nullptr;

    F4_GPU_CUDA_CHECK(cudaMalloc(&d_ell, std::size_t(lm_count) * sizeof(int)));
    F4_GPU_CUDA_CHECK(cudaMalloc(&d_m,   std::size_t(lm_count) * sizeof(int)));

    try
    {
        F4_GPU_CUDA_CHECK(cudaMemcpy(
            d_ell,
            ell_list.data(),
            std::size_t(lm_count) * sizeof(int),
            cudaMemcpyHostToDevice
        ));

        F4_GPU_CUDA_CHECK(cudaMemcpy(
            d_m,
            m_list.data(),
            std::size_t(lm_count) * sizeof(int),
            cudaMemcpyHostToDevice
        ));

        std::vector<std::vector<ConfigEntry>> all_out(static_cast<std::size_t>(nE_total));

        for (int start = 0; start < nE_total; start += chunk_size)
        {
            const int end = std::min(nE_total, start + chunk_size);
            const int nE_chunk = end - start;

            std::vector<double> En_chunk(
                En_values.begin() + start,
                En_values.begin() + end
            );

            const std::size_t nflags = std::size_t(nE_chunk) * candidates;

            double* d_En = nullptr;
            unsigned char* d_flags = nullptr;

            F4_GPU_CUDA_CHECK(cudaMalloc(&d_En, std::size_t(nE_chunk) * sizeof(double)));
            F4_GPU_CUDA_CHECK(cudaMalloc(&d_flags, nflags * sizeof(unsigned char)));

            try
            {
                F4_GPU_CUDA_CHECK(cudaMemcpy(
                    d_En,
                    En_chunk.data(),
                    std::size_t(nE_chunk) * sizeof(double),
                    cudaMemcpyHostToDevice
                ));

                F4_GPU_CUDA_CHECK(cudaMemset(d_flags, 0, nflags * sizeof(unsigned char)));

                const std::size_t total_threads = std::size_t(nE_chunk) * candidates;
                const int threads = std::max(32, options.cuda_threads_per_block);
                const int blocks = int((total_threads + std::size_t(threads) - 1) / std::size_t(threads));

                mark_config_maker_4_candidates_kernel<<<blocks, threads>>>(
                    d_En,
                    nE_chunk,
                    params,
                    d_ell,
                    d_m,
                    lm_count,
                    nmax,
                    candidates,
                    d_flags
                );

                F4_GPU_CUDA_CHECK(cudaGetLastError());
                F4_GPU_CUDA_CHECK(cudaDeviceSynchronize());

                std::vector<unsigned char> h_flags(nflags);
                F4_GPU_CUDA_CHECK(cudaMemcpy(
                    h_flags.data(),
                    d_flags,
                    nflags * sizeof(unsigned char),
                    cudaMemcpyDeviceToHost
                ));

                std::vector<std::vector<ConfigEntry>> chunk_out;

                compact_flags_to_entries_cpu_order(
                    h_flags.data(),
                    nE_chunk,
                    En_chunk,
                    params,
                    ell_list,
                    m_list,
                    nmax,
                    chunk_out
                );

                for (int i = 0; i < nE_chunk; ++i)
                {
                    all_out[std::size_t(start + i)] = std::move(chunk_out[std::size_t(i)]);
                }

                if (options.debug == 'y')
                {
                    std::cout << "[gpu_config_maker_4] finished chunk ["
                              << start << ", " << end - 1 << "]";
                    if (!chunk_out.empty())
                    {
                        std::cout << ", first_count = " << chunk_out.front().size()
                                  << ", last_count = " << chunk_out.back().size();
                    }
                    std::cout << '\n';
                }
            }
            catch (...)
            {
                if (d_En) cudaFree(d_En);
                if (d_flags) cudaFree(d_flags);
                throw;
            }

            F4_GPU_CUDA_CHECK(cudaFree(d_En));
            F4_GPU_CUDA_CHECK(cudaFree(d_flags));
        }

        F4_GPU_CUDA_CHECK(cudaFree(d_ell));
        F4_GPU_CUDA_CHECK(cudaFree(d_m));

        return all_out;
    }
    catch (...)
    {
        if (d_ell) cudaFree(d_ell);
        if (d_m) cudaFree(d_m);
        throw;
    }
}

// ---------------------------------------------------------------------------
// Convert ConfigEntry vector to the original CPU containers:
//
//   plm_config[0] = px
//   plm_config[1] = py
//   plm_config[2] = pz
//   plm_config[3] = ell
//   plm_config[4] = m
//
//   n_config[0] = nx
//   n_config[1] = ny
//   n_config[2] = nz
//   n_config[3] = ell
//   n_config[4] = m
// ---------------------------------------------------------------------------

inline void entries_to_cpu_vectors(
    const std::vector<ConfigEntry>& entries,
    std::vector<std::vector<std::complex<double>>>& plm_config,
    std::vector<std::vector<int>>& n_config)
{
    plm_config.assign(5, std::vector<std::complex<double>>{});
    n_config.assign(5, std::vector<int>{});

    for (int r = 0; r < 5; ++r)
    {
        plm_config[std::size_t(r)].reserve(entries.size());
        n_config[std::size_t(r)].reserve(entries.size());
    }

    for (const auto& e : entries)
    {
        plm_config[0].push_back({e.px, 0.0});
        plm_config[1].push_back({e.py, 0.0});
        plm_config[2].push_back({e.pz, 0.0});
        plm_config[3].push_back({double(e.ell), 0.0});
        plm_config[4].push_back({double(e.m), 0.0});

        n_config[0].push_back(e.nx);
        n_config[1].push_back(e.ny);
        n_config[2].push_back(e.nz);
        n_config[3].push_back(e.ell);
        n_config[4].push_back(e.m);
    }
}

inline void gpu_config_maker_4_single_to_cpu_vectors(
    std::vector<std::vector<std::complex<double>>>& plm_config,
    std::vector<std::vector<int>>& n_config,
    const std::vector<int>& waves_vec,
    double En_real,
    const std::vector<std::complex<double>>& total_P,
    double mi,
    double mj,
    double mk,
    double L,
    double epsilon_h,
    double max_num_shell,
    double tolerance,
    char debug = 'n')
{
    if (total_P.size() < 3)
    {
        throw std::runtime_error("gpu_config_maker_4_single_to_cpu_vectors: total_P.size() < 3");
    }

    ConfigMaker4Params params;
    params.Px = total_P[0].real();
    params.Py = total_P[1].real();
    params.Pz = total_P[2].real();
    params.mi = mi;
    params.mj = mj;
    params.mk = mk;
    params.L = L;
    params.epsilon_h = epsilon_h;
    params.max_num_shell = max_num_shell;
    params.tolerance = tolerance;

    Options opt;
    opt.debug = debug;

    std::vector<double> En_values = {En_real};

    auto all_entries = gpu_config_maker_4_many_entries(
        En_values,
        waves_vec,
        params,
        opt
    );

    entries_to_cpu_vectors(all_entries.front(), plm_config, n_config);
}

inline void gpu_config_maker_4_many_to_cpu_vectors(
    std::vector<std::vector<std::vector<std::complex<double>>>>& plm_vec,
    std::vector<std::vector<std::vector<int>>>& n_vec,
    const std::vector<int>& waves_vec,
    const std::vector<double>& En_values,
    const std::vector<std::complex<double>>& total_P,
    double mi,
    double mj,
    double mk,
    double L,
    double epsilon_h,
    double max_num_shell,
    double tolerance,
    const Options& opt = Options{})
{
    if (total_P.size() < 3)
    {
        throw std::runtime_error("gpu_config_maker_4_many_to_cpu_vectors: total_P.size() < 3");
    }

    ConfigMaker4Params params;
    params.Px = total_P[0].real();
    params.Py = total_P[1].real();
    params.Pz = total_P[2].real();
    params.mi = mi;
    params.mj = mj;
    params.mk = mk;
    params.L = L;
    params.epsilon_h = epsilon_h;
    params.max_num_shell = max_num_shell;
    params.tolerance = tolerance;

    auto all_entries = gpu_config_maker_4_many_entries(
        En_values,
        waves_vec,
        params,
        opt
    );

    const std::size_t nE = En_values.size();
    plm_vec.assign(nE, std::vector<std::vector<std::complex<double>>>(5));
    n_vec.assign(nE, std::vector<std::vector<int>>(5));

    for (std::size_t i = 0; i < nE; ++i)
    {
        entries_to_cpu_vectors(all_entries[i], plm_vec[i], n_vec[i]);
    }
}

// Convenience helper matching your common Ecm-grid setup.
inline void gpu_config_maker_4_many_Ecm_to_cpu_vectors(
    std::vector<std::vector<std::vector<std::complex<double>>>>& plm_vec,
    std::vector<std::vector<std::vector<int>>>& n_vec,
    const std::vector<int>& waves_vec,
    double Ecm_initial,
    double Ecm_final,
    int Ecm_points,
    const std::vector<std::complex<double>>& total_P,
    double mi,
    double mj,
    double mk,
    double L,
    double epsilon_h,
    double max_num_shell,
    double tolerance,
    const Options& opt = Options{})
{
    if (Ecm_points <= 0)
    {
        throw std::runtime_error("gpu_config_maker_4_many_Ecm_to_cpu_vectors: Ecm_points <= 0");
    }

    if (total_P.size() < 3)
    {
        throw std::runtime_error("gpu_config_maker_4_many_Ecm_to_cpu_vectors: total_P.size() < 3");
    }

    const double Px = total_P[0].real();
    const double Py = total_P[1].real();
    const double Pz = total_P[2].real();
    const double P2 = Px * Px + Py * Py + Pz * Pz;

    const double del_Ecm = (Ecm_points == 1)
        ? 0.0
        : std::abs(Ecm_final - Ecm_initial) / double(Ecm_points);

    std::vector<double> En_values(static_cast<std::size_t>(Ecm_points));

    for (int i = 0; i < Ecm_points; ++i)
    {
        const double Ecm = Ecm_initial + double(i) * del_Ecm;
        En_values[std::size_t(i)] = std::sqrt(Ecm * Ecm + P2);
    }

    gpu_config_maker_4_many_to_cpu_vectors(
        plm_vec,
        n_vec,
        waves_vec,
        En_values,
        total_P,
        mi,
        mj,
        mk,
        L,
        epsilon_h,
        max_num_shell,
        tolerance,
        opt
    );
}

} // namespace f4gpu

// ---------------------------------------------------------------------------
// Optional global wrapper with a name close to your CPU function.
// This avoids replacing config_maker_4 accidentally. Call it explicitly.
// ---------------------------------------------------------------------------

inline void config_maker_4_gpu_safe(
    std::vector<std::vector<std::complex<double>>>& plm_config,
    std::vector<std::vector<int>>& n_config,
    const std::vector<int>& waves_vec,
    double En_real,
    const std::vector<std::complex<double>>& total_P,
    double mi,
    double mj,
    double mk,
    double L,
    double epsilon_h,
    double max_num_shell,
    double tolerance,
    char debug = 'n')
{
    f4gpu::gpu_config_maker_4_single_to_cpu_vectors(
        plm_config,
        n_config,
        waves_vec,
        En_real,
        total_P,
        mi,
        mj,
        mk,
        L,
        epsilon_h,
        max_num_shell,
        tolerance,
        debug
    );
}

#endif // FUNCTIONS_GPU_CONFIG_MAKER4_CUH
