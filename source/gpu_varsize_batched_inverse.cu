// gpu_varsize_batched_inverse.cu
//
// Build (demo):
//   nvcc -O3 -std=c++17 -DBUILD_DEMO_MAIN -DEIGEN_NO_CUDA --expt-relaxed-constexpr \
//        -Xcompiler -fopenmp -lineinfo -I/usr/include/eigen3 \
//        gpu_varsize_batched_inverse.cu -lcublas -lcudart -o test_varsize_inv
//
// Build (object):
//   nvcc -O3 -std=c++17 -DEIGEN_NO_CUDA --expt-relaxed-constexpr \
//        -Xcompiler -fopenmp -lineinfo -I/usr/include/eigen3 \
//        -c gpu_varsize_batched_inverse.cu -o gpu_varsize_batched_inverse.o

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <sys/sysinfo.h>


#include <Eigen/Dense>

#include <omp.h>
#include <vector>
#include <map>
#include <functional>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <algorithm>

#include <unordered_map>
#include <vector>
#include "dig_tools.hpp"

#include <type_traits>
#include <utility>
#include <stdexcept>

#include <complex>
#include <cstdint>
#include <cstdio>
#include "functions.h"

using comp = std::complex<double>;

// --------------------------- error checks ---------------------------

static inline void throwOnCuda(cudaError_t st, const char* msg, const char* file, int line) {
  if (st != cudaSuccess) {
    std::ostringstream os;
    os << "CUDA error: " << msg << " | " << cudaGetErrorString(st)
       << " @ " << file << ":" << line;
    throw std::runtime_error(os.str());
  }
}
static inline void throwOnCublas(cublasStatus_t st, const char* msg, const char* file, int line) {
  if (st != CUBLAS_STATUS_SUCCESS) {
    std::ostringstream os;
    os << "cuBLAS error: " << msg << " | status=" << int(st)
       << " @ " << file << ":" << line;
    throw std::runtime_error(os.str());
  }
}

#define CUDA_CHECK(x, msg)   throwOnCuda((x), (msg), __FILE__, __LINE__)
#define CUBLAS_CHECK(x, msg) throwOnCublas((x), (msg), __FILE__, __LINE__)

static inline void cudaSyncCheck(const char* where) {
  CUDA_CHECK(cudaDeviceSynchronize(), where);
  CUDA_CHECK(cudaGetLastError(), where);
}

// --------------------------- helpers ---------------------------



static inline size_t vram_bytes_per_mat(int n) {
  const size_t cplx = sizeof(cuDoubleComplex); // 16
  return 2ull * size_t(n) * size_t(n) * cplx
       + size_t(n) * sizeof(int)
       + sizeof(int);
}

static inline size_t ram_bytes_per_energy(int n, int num_mats_host = 6) {
  const size_t cplx = sizeof(comp); // 16
  return size_t(num_mats_host) * size_t(n) * size_t(n) * cplx;
}

static inline size_t get_free_vram_bytes() {
  size_t freeB = 0, totalB = 0;
  cudaError_t st = cudaMemGetInfo(&freeB, &totalB);
  if (st != cudaSuccess) return 0;
  return freeB;
}

static inline size_t get_avail_ram_bytes_linux() {
  struct sysinfo info;
  if (sysinfo(&info) != 0) return 0;
  // freeram includes free pages; bufferram is cache/buffers (often reclaimable).
  // A decent "available" approximation is freeram + bufferram.
  unsigned long long unit = (unsigned long long)info.mem_unit;
  unsigned long long avail = (unsigned long long)info.freeram + (unsigned long long)info.bufferram;
  return (size_t)(avail * unit);
}

// You provide these (already in your codebase):
// void config_maker_3(std::vector<std::vector<comp>>& cfg, ...);

// Returns suggested chunk size based on En_final totsize.
int suggest_chunk_size_from_En_final(
    double En_final,
    // --- your ingredients needed by config_maker_3 ---
    const std::vector<int>& waves_vec_1,
    const std::vector<int>& waves_vec_2,
    const std::vector<comp>& total_P,
    double atmK, double atmpi,
    double L, double epsilon_h, double max_shell_num, double tolerance,
    // --- safety knobs ---
    double vram_safety_fraction = 0.90,
    double ram_safety_fraction  = 0.70,
    int    host_mats_per_energy = 6)
{
  // 1) build configs at En_final
  double mi, mj, mk;

  mi = atmK; mj = atmK; mk = atmpi;
  std::vector<std::vector<comp>> plm_config(5);
  config_maker_3(plm_config, waves_vec_1, En_final, total_P, mi, mj, mk,
                 L, epsilon_h, max_shell_num, tolerance);

  mi = atmpi; mj = atmK; mk = atmK;
  std::vector<std::vector<comp>> klm_config(5);
  config_maker_3(klm_config, waves_vec_2, En_final, total_P, mi, mj, mk,
                 L, epsilon_h, max_shell_num, tolerance);

  const int dim1 = (int)plm_config[0].size();
  const int dim2 = (int)klm_config[0].size();
  const int n    = dim1 + dim2;

  if (n <= 0) return 1;

  // 2) budgets
  const size_t free_vram = get_free_vram_bytes();
  const size_t avail_ram = get_avail_ram_bytes_linux();

  const size_t vram_budget = (size_t)(double(free_vram) * vram_safety_fraction);
  const size_t ram_budget  = (size_t)(double(avail_ram) * ram_safety_fraction);

  // 3) per-item costs
  const size_t per_mat_vram = vram_bytes_per_mat(n);
  const size_t per_E_ram    = ram_bytes_per_energy(n, host_mats_per_energy);

  // Add overhead pads (configs, pointer arrays, misc)
  const size_t vram_pad = 128ull * 1024ull * 1024ull; // 128 MiB pad
  const size_t ram_pad  = 512ull * 1024ull * 1024ull; // 512 MiB pad

  const size_t vram_budget_eff = (vram_budget > vram_pad) ? (vram_budget - vram_pad) : 0;
  const size_t ram_budget_eff  = (ram_budget  > ram_pad)  ? (ram_budget  - ram_pad)  : 0;

  // 4) how many fit
  const int max_by_vram = (per_mat_vram > 0 && vram_budget_eff > 0)
                        ? (int)(vram_budget_eff / per_mat_vram)
                        : 1;

  const int max_by_ram  = (per_E_ram > 0 && ram_budget_eff > 0)
                        ? (int)(ram_budget_eff / per_E_ram)
                        : 1;

  int chunk = std::max(1, std::min(max_by_vram, max_by_ram));

  std::fprintf(stderr,
    "[chunk-suggest] En_final=%.6f dim1=%d dim2=%d totsize=%d\n"
    "  free_vram=%.2f MiB budget_vram=%.2f MiB per_mat_vram=%.2f MiB -> max_by_vram=%d\n"
    "  avail_ram=%.2f GiB budget_ram=%.2f GiB per_E_ram=%.2f MiB (x%d mats) -> max_by_ram=%d\n"
    "  => chunkSize=%d\n",
    En_final, dim1, dim2, n,
    free_vram / 1024.0 / 1024.0,
    vram_budget_eff / 1024.0 / 1024.0,
    per_mat_vram / 1024.0 / 1024.0,
    max_by_vram,
    avail_ram / 1024.0 / 1024.0 / 1024.0,
    ram_budget_eff / 1024.0 / 1024.0 / 1024.0,
    per_E_ram / 1024.0 / 1024.0,
    host_mats_per_energy,
    max_by_ram,
    chunk
  );

  return chunk;
}

// One cached allocation set per matrix size n.
// ==========================
// Drop-in GPU buffer cache (per matrix size n) for batched inversion
// Copy-paste into your .cu (e.g., gpu_varsize_batched_inverse.cu)
// Requires: CUDA_CHECK / CUBLAS_CHECK / cudaSyncCheck helpers already defined
// ==========================



// ---- Cache: one buffer set per size n ----
struct CublasBatchedInvCache {
  struct Buffers {
    int n   = 0;
    int cap = 0; // max batch supported

    cuDoubleComplex* dA = nullptr;        // [cap * n * n]
    cuDoubleComplex* dB = nullptr;        // [cap * n * n]
    int* dPivots = nullptr;               // [cap * n]
    int* dInfo_getrf = nullptr;           // [cap]
    cuDoubleComplex** dA_array = nullptr; // [cap]
    cuDoubleComplex** dB_array = nullptr; // [cap]

    void release() {
      if (dA_array) cudaFree(dA_array), dA_array = nullptr;
      if (dB_array) cudaFree(dB_array), dB_array = nullptr;
      if (dA) cudaFree(dA), dA = nullptr;
      if (dB) cudaFree(dB), dB = nullptr;
      if (dPivots) cudaFree(dPivots), dPivots = nullptr;
      if (dInfo_getrf) cudaFree(dInfo_getrf), dInfo_getrf = nullptr;
      n = 0;
      cap = 0;
    }

    ~Buffers() { release(); }
  };

  std::unordered_map<int, Buffers> by_n;

  Buffers& get(int n) { return by_n[n]; }
  void clear() { by_n.clear(); } // frees via destructors
};

// Grow cached buffers for a given size n to support batchCount
static inline void ensureCapacity(CublasBatchedInvCache::Buffers& buf, int n, int batchCount) {
  if (batchCount <= 0) return;
  if (buf.n == n && buf.cap >= batchCount) return;

  // Grow policy: reduce realloc frequency
  int newCap = std::max(batchCount, std::max(64, buf.cap > 0 ? buf.cap * 2 : 64));

  // Release old buffers (if any)
  buf.release();

  buf.n = n;
  buf.cap = newCap;

  const size_t matElems = size_t(newCap) * size_t(n) * size_t(n);

  CUDA_CHECK(cudaMalloc((void**)&buf.dA, matElems * sizeof(cuDoubleComplex)), "cudaMalloc cache dA");
  CUDA_CHECK(cudaMalloc((void**)&buf.dB, matElems * sizeof(cuDoubleComplex)), "cudaMalloc cache dB");
  CUDA_CHECK(cudaMalloc((void**)&buf.dPivots, size_t(newCap) * size_t(n) * sizeof(int)), "cudaMalloc cache dPivots");
  CUDA_CHECK(cudaMalloc((void**)&buf.dInfo_getrf, size_t(newCap) * sizeof(int)), "cudaMalloc cache dInfo_getrf");
  CUDA_CHECK(cudaMalloc((void**)&buf.dA_array, size_t(newCap) * sizeof(cuDoubleComplex*)), "cudaMalloc cache dA_array");
  CUDA_CHECK(cudaMalloc((void**)&buf.dB_array, size_t(newCap) * sizeof(cuDoubleComplex*)), "cudaMalloc cache dB_array");
}

static inline size_t bytesForOneMatWorstCase(int n) {
  const size_t cplxBytes = sizeof(cuDoubleComplex);
  size_t perMat = 2ull * size_t(n) * size_t(n) * cplxBytes
                + size_t(n) * sizeof(int)
                + sizeof(int);
  return perMat;
}

static inline void packEigenToHostCublas(const Eigen::MatrixXcd& A, cuDoubleComplex* dst) {
  const std::complex<double>* src = A.data();
  const int elems = int(A.size());
  for (int i = 0; i < elems; ++i) {
    dst[i] = make_cuDoubleComplex(src[i].real(), src[i].imag());
  }
}

static inline void unpackHostCublasToEigen(const cuDoubleComplex* src, Eigen::MatrixXcd& A) {
  std::complex<double>* dst = A.data();
  const int elems = int(A.size());
  for (int i = 0; i < elems; ++i) {
    dst[i] = std::complex<double>(cuCreal(src[i]), cuCimag(src[i]));
  }
}

static inline void makeIdentityBatchHost(int n, int batchCount, std::vector<cuDoubleComplex>& Bhost) {
  Bhost.assign(size_t(batchCount) * size_t(n) * size_t(n), make_cuDoubleComplex(0.0, 0.0));
  for (int b = 0; b < batchCount; ++b) {
    cuDoubleComplex* Bb = Bhost.data() + size_t(b) * size_t(n) * size_t(n);
    for (int i = 0; i < n; ++i) Bb[i + i * n] = make_cuDoubleComplex(1.0, 0.0);
  }
}

// --------------------------- GPU core: same-size group inversion ---------------------------
//
// getrfBatched: infoArray is DEVICE -> dInfo_getrf
// getrsBatched: info is HOST        -> hInfo_getrs
//
static void invertSameSizeGroup_cublasBatched(
    cublasHandle_t handle,
    int n,
    const std::vector<Eigen::MatrixXcd>& A_group,
    std::vector<Eigen::MatrixXcd>& Ainv_group_out)
{
  const int batchCount = int(A_group.size());
  if (batchCount == 0) return;

  // Host packed buffers
  std::vector<cuDoubleComplex> Ahost(size_t(batchCount) * size_t(n) * size_t(n));
  std::vector<cuDoubleComplex> Bhost;
  makeIdentityBatchHost(n, batchCount, Bhost);

  for (int b = 0; b < batchCount; ++b) {
    const auto& A = A_group[b];
    if (A.rows() != n || A.cols() != n) throw std::runtime_error("Size-group contains wrong-sized matrix.");
    packEigenToHostCublas(A, Ahost.data() + size_t(b) * size_t(n) * size_t(n));
  }

  // Device allocations
  cuDoubleComplex* dA = nullptr;
  cuDoubleComplex* dB = nullptr;
  int* dPivots = nullptr;       // device
  int* dInfo_getrf = nullptr;   // device
  std::vector<int> hInfo_getrf(batchCount, 0);
  std::vector<int> hInfo_getrs(batchCount, 0); // host

  CUDA_CHECK(cudaMalloc((void**)&dA, Ahost.size() * sizeof(cuDoubleComplex)), "cudaMalloc dA");
  CUDA_CHECK(cudaMalloc((void**)&dB, Bhost.size() * sizeof(cuDoubleComplex)), "cudaMalloc dB");
  CUDA_CHECK(cudaMalloc((void**)&dPivots, size_t(batchCount) * size_t(n) * sizeof(int)), "cudaMalloc dPivots");
  CUDA_CHECK(cudaMalloc((void**)&dInfo_getrf, size_t(batchCount) * sizeof(int)), "cudaMalloc dInfo_getrf");

  CUDA_CHECK(cudaMemcpy(dA, Ahost.data(), Ahost.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), "H2D A");
  CUDA_CHECK(cudaMemcpy(dB, Bhost.data(), Bhost.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), "H2D B");
  CUDA_CHECK(cudaMemset(dInfo_getrf, 0, size_t(batchCount) * sizeof(int)), "memset dInfo_getrf");

  // Device pointer arrays
  cuDoubleComplex** dA_array = nullptr;
  cuDoubleComplex** dB_array = nullptr;
  CUDA_CHECK(cudaMalloc((void**)&dA_array, size_t(batchCount) * sizeof(cuDoubleComplex*)), "cudaMalloc dA_array");
  CUDA_CHECK(cudaMalloc((void**)&dB_array, size_t(batchCount) * sizeof(cuDoubleComplex*)), "cudaMalloc dB_array");

  std::vector<cuDoubleComplex*> hAptr(batchCount), hBptr(batchCount);
  for (int b = 0; b < batchCount; ++b) {
    hAptr[b] = dA + size_t(b) * size_t(n) * size_t(n);
    hBptr[b] = dB + size_t(b) * size_t(n) * size_t(n);
  }
  CUDA_CHECK(cudaMemcpy(dA_array, hAptr.data(), size_t(batchCount) * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice), "H2D A ptrs");
  CUDA_CHECK(cudaMemcpy(dB_array, hBptr.data(), size_t(batchCount) * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice), "H2D B ptrs");

  // LU
  CUBLAS_CHECK(cublasZgetrfBatched(handle, n, dA_array, n, dPivots, dInfo_getrf, batchCount),
               "cublasZgetrfBatched");
  cudaSyncCheck("after cublasZgetrfBatched");

  CUDA_CHECK(cudaMemcpy(hInfo_getrf.data(), dInfo_getrf, size_t(batchCount) * sizeof(int),
                        cudaMemcpyDeviceToHost),
             "D2H info_getrf");
  for (int b = 0; b < batchCount; ++b) {
    if (hInfo_getrf[b] != 0) {
      std::ostringstream os;
      os << "LU failed: info[" << b << "]=" << hInfo_getrf[b]
         << " (singular if >0, illegal param if <0).";
      throw std::runtime_error(os.str());
    }
  }

  // Solve A*X=I (B overwritten with X)
  std::fill(hInfo_getrs.begin(), hInfo_getrs.end(), 0);
  CUBLAS_CHECK(cublasZgetrsBatched(handle, CUBLAS_OP_N, n, n,
                                   (const cuDoubleComplex**)dA_array, n,
                                   dPivots,
                                   dB_array, n,
                                   hInfo_getrs.data(), batchCount),
               "cublasZgetrsBatched");
  cudaSyncCheck("after cublasZgetrsBatched");

  for (int b = 0; b < batchCount; ++b) {
    if (hInfo_getrs[b] != 0) {
      std::ostringstream os;
      os << "Solve failed: info[" << b << "]=" << hInfo_getrs[b]
         << " (illegal param if <0).";
      throw std::runtime_error(os.str());
    }
  }

  // Copy back
  std::vector<cuDoubleComplex> Xhost(Bhost.size());
  CUDA_CHECK(cudaMemcpy(Xhost.data(), dB, Xhost.size() * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost), "D2H X");

  Ainv_group_out.resize(batchCount);
  for (int b = 0; b < batchCount; ++b) {
    Ainv_group_out[b].resize(n, n);
    unpackHostCublasToEigen(Xhost.data() + size_t(b) * size_t(n) * size_t(n), Ainv_group_out[b]);
  }

  // Cleanup
  cudaFree(dA_array);
  cudaFree(dB_array);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dPivots);
  cudaFree(dInfo_getrf);
}

/*
static void invertSameSizeGroup_cublasBatched_cached(
    cublasHandle_t handle,
    int n,
    const std::vector<Eigen::MatrixXcd>& A_group,
    std::vector<Eigen::MatrixXcd>& Ainv_group_out,
    CublasBatchedInvCache& cache)
{
  const int batchCount = int(A_group.size());
  if (batchCount == 0) return;

  // Host packed buffers (still std::vector; later you can upgrade to pinned)
  std::vector<cuDoubleComplex> Ahost(size_t(batchCount) * size_t(n) * size_t(n));
  std::vector<cuDoubleComplex> Bhost;
  makeIdentityBatchHost(n, batchCount, Bhost);

  for (int b = 0; b < batchCount; ++b) {
    const auto& A = A_group[b];
    if (A.rows() != n || A.cols() != n) throw std::runtime_error("Size-group contains wrong-sized matrix.");
    packEigenToHostCublas(A, Ahost.data() + size_t(b) * size_t(n) * size_t(n));
  }

  // Get cached buffers for this n and ensure enough capacity
  auto& buf = cache.get(n);
  ensureCapacity(buf, n, batchCount);

  // Copy only the portion we use (batchCount)
  const size_t usedElems = size_t(batchCount) * size_t(n) * size_t(n);

  CUDA_CHECK(cudaMemcpy(buf.dA, Ahost.data(), usedElems * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), "H2D A");
  CUDA_CHECK(cudaMemcpy(buf.dB, Bhost.data(), usedElems * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), "H2D B");
  CUDA_CHECK(cudaMemset(buf.dInfo_getrf, 0, size_t(batchCount) * sizeof(int)), "memset dInfo_getrf");

  // Build pointer arrays on host then copy into cached device pointer arrays
  std::vector<cuDoubleComplex*> hAptr(batchCount), hBptr(batchCount);
  for (int b = 0; b < batchCount; ++b) {
    hAptr[b] = buf.dA + size_t(b) * size_t(n) * size_t(n);
    hBptr[b] = buf.dB + size_t(b) * size_t(n) * size_t(n);
  }
  CUDA_CHECK(cudaMemcpy(buf.dA_array, hAptr.data(), size_t(batchCount) * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice), "H2D A ptrs");
  CUDA_CHECK(cudaMemcpy(buf.dB_array, hBptr.data(), size_t(batchCount) * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice), "H2D B ptrs");

  // LU (infoArray on DEVICE)
  CUBLAS_CHECK(cublasZgetrfBatched(handle, n, buf.dA_array, n, buf.dPivots, buf.dInfo_getrf, batchCount),
               "cublasZgetrfBatched");
  cudaSyncCheck("after cublasZgetrfBatched");

  // Check LU info
  std::vector<int> hInfo_getrf(batchCount, 0);
  CUDA_CHECK(cudaMemcpy(hInfo_getrf.data(), buf.dInfo_getrf, size_t(batchCount) * sizeof(int), cudaMemcpyDeviceToHost),
             "D2H info_getrf");
  for (int b = 0; b < batchCount; ++b) {
    if (hInfo_getrf[b] != 0) {
      std::ostringstream os;
      os << "LU failed: info[" << b << "]=" << hInfo_getrf[b]
         << " (singular if >0, illegal param if <0).";
      throw std::runtime_error(os.str());
    }
  }

  // Solve A*X = I (info on HOST)
  std::vector<int> hInfo_getrs(batchCount, 0);
  CUBLAS_CHECK(cublasZgetrsBatched(handle, CUBLAS_OP_N, n, n,
                                   (const cuDoubleComplex**)buf.dA_array, n,
                                   buf.dPivots,
                                   buf.dB_array, n,
                                   hInfo_getrs.data(), batchCount),
               "cublasZgetrsBatched");
  cudaSyncCheck("after cublasZgetrsBatched");

  for (int b = 0; b < batchCount; ++b) {
    if (hInfo_getrs[b] != 0) {
      std::ostringstream os;
      os << "Solve failed: info[" << b << "]=" << hInfo_getrs[b]
         << " (illegal param if <0).";
      throw std::runtime_error(os.str());
    }
  }

  // Copy back result
  std::vector<cuDoubleComplex> Xhost(usedElems);
  CUDA_CHECK(cudaMemcpy(Xhost.data(), buf.dB, usedElems * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost), "D2H X");

  // Unpack into Eigen output
  Ainv_group_out.resize(batchCount);
  for (int b = 0; b < batchCount; ++b) {
    Ainv_group_out[b].resize(n, n);
    unpackHostCublasToEigen(Xhost.data() + size_t(b) * size_t(n) * size_t(n), Ainv_group_out[b]);
  }
}
*/




// ---- Cached inversion for one same-size group ----
// - Same math as your invertSameSizeGroup_cublasBatched()
// - No cudaMalloc/cudaFree in the hot loop (reuses cached buffers)
// - Still uses host packing vectors (upgrade to pinned/async later if desired)
static void invertSameSizeGroup_cublasBatched_cached(
    cublasHandle_t handle,
    int n,
    const std::vector<Eigen::MatrixXcd>& A_group,
    std::vector<Eigen::MatrixXcd>& Ainv_group_out,
    CublasBatchedInvCache& cache)
{
  const int batchCount = int(A_group.size());
  if (batchCount == 0) return;

  // Pack A and build B=I on host
  const size_t elems = size_t(batchCount) * size_t(n) * size_t(n);

  std::vector<cuDoubleComplex> Ahost(elems);
  std::vector<cuDoubleComplex> Bhost;
  makeIdentityBatchHost(n, batchCount, Bhost);

  for (int b = 0; b < batchCount; ++b) {
    const auto& A = A_group[b];
    if (A.rows() != n || A.cols() != n) throw std::runtime_error("Size-group contains wrong-sized matrix.");
    packEigenToHostCublas(A, Ahost.data() + size_t(b) * size_t(n) * size_t(n));
  }

  // Get cached buffers and ensure capacity
  auto& buf = cache.get(n);
  ensureCapacity(buf, n, batchCount);

  // Copy packed A, B to device (only used prefix)
  CUDA_CHECK(cudaMemcpy(buf.dA, Ahost.data(), elems * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), "H2D A");
  CUDA_CHECK(cudaMemcpy(buf.dB, Bhost.data(), elems * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), "H2D B");
  CUDA_CHECK(cudaMemset(buf.dInfo_getrf, 0, size_t(batchCount) * sizeof(int)), "memset info_getrf");

  // Build pointer arrays for batched API
  std::vector<cuDoubleComplex*> hAptr(batchCount), hBptr(batchCount);
  for (int b = 0; b < batchCount; ++b) {
    hAptr[b] = buf.dA + size_t(b) * size_t(n) * size_t(n);
    hBptr[b] = buf.dB + size_t(b) * size_t(n) * size_t(n);
  }
  CUDA_CHECK(cudaMemcpy(buf.dA_array, hAptr.data(), size_t(batchCount) * sizeof(cuDoubleComplex*),
                        cudaMemcpyHostToDevice),
             "H2D A ptrs");
  CUDA_CHECK(cudaMemcpy(buf.dB_array, hBptr.data(), size_t(batchCount) * sizeof(cuDoubleComplex*),
                        cudaMemcpyHostToDevice),
             "H2D B ptrs");

  // LU factorization (info array is DEVICE)
  CUBLAS_CHECK(cublasZgetrfBatched(handle, n, buf.dA_array, n, buf.dPivots, buf.dInfo_getrf, batchCount),
               "cublasZgetrfBatched");
  cudaSyncCheck("after cublasZgetrfBatched");

  // Check LU info
  std::vector<int> hInfo_getrf(batchCount, 0);
  CUDA_CHECK(cudaMemcpy(hInfo_getrf.data(), buf.dInfo_getrf, size_t(batchCount) * sizeof(int),
                        cudaMemcpyDeviceToHost),
             "D2H info_getrf");
  for (int b = 0; b < batchCount; ++b) {
    if (hInfo_getrf[b] != 0) {
      std::ostringstream os;
      os << "LU failed: info[" << b << "]=" << hInfo_getrf[b]
         << " (singular if >0, illegal param if <0).";
      throw std::runtime_error(os.str());
    }
  }

  // Solve A*X = I (B overwritten with X). For cublasZgetrsBatched, info is HOST array.
  std::vector<int> hInfo_getrs(batchCount, 0);
  CUBLAS_CHECK(cublasZgetrsBatched(handle, CUBLAS_OP_N, n, n,
                                   (const cuDoubleComplex**)buf.dA_array, n,
                                   buf.dPivots,
                                   buf.dB_array, n,
                                   hInfo_getrs.data(), batchCount),
               "cublasZgetrsBatched");
  cudaSyncCheck("after cublasZgetrsBatched");

  for (int b = 0; b < batchCount; ++b) {
    if (hInfo_getrs[b] != 0) {
      std::ostringstream os;
      os << "Solve failed: info[" << b << "]=" << hInfo_getrs[b]
         << " (illegal param if <0).";
      throw std::runtime_error(os.str());
    }
  }

  // Copy result back from device B (now X)
  std::vector<cuDoubleComplex> Xhost(elems);
  CUDA_CHECK(cudaMemcpy(Xhost.data(), buf.dB, elems * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost), "D2H X");

  // Unpack into Eigen output
  Ainv_group_out.resize(batchCount);
  for (int b = 0; b < batchCount; ++b) {
    Ainv_group_out[b].resize(n, n);
    unpackHostCublasToEigen(Xhost.data() + size_t(b) * size_t(n) * size_t(n), Ainv_group_out[b]);
  }
}

// ==========================
// Integration: inside your sweep function
// ==========================
//
// 1) Create the cache ONCE per sweep (near cublasCreate):
//
//   CublasBatchedInvCache inv_cache;
//
// 2) Replace the call in your group loop:
//
//   invertSameSizeGroup_cublasBatched(handle, n, A_group, Ainv_group);
//
// with:
//
//   invertSameSizeGroup_cublasBatched_cached(handle, n, A_group, Ainv_group, inv_cache);
//
// That’s all.
// ==========================

// --------------------------- High-level pipeline ---------------------------

// Drop-in change: replace the std::function-based API with a templated Builder API.
// Everything else (chunking, grouping, cuBLAS inversion core) remains identical.
//
// Usage stays the same for lambdas/functions, but now it inlines and avoids std::function overhead.
//
// Example call:
//   auto buildA = [&](int i, double En, Eigen::MatrixXcd& A){ ... };
//   build_and_invert_energy_sweep_varsize_batched_gpu(En_i, En_f, N, buildA, Ainv, 0.8, 0);

#include <type_traits>
#include <utility>

// ---- replace ONLY this function signature+body in your file ----

// NOTE: You can keep the same function name.
// We template on Builder and forward it.
// Requirement: build_A(i, En, A) must be a valid expression returning void (or convertible to void).
template <class Builder>
void build_and_invert_energy_sweep_varsize_batched_gpu(
    double En_initial,
    double En_final,
    int total_problem_size,
    Builder&& build_A,                         // <--- templated callable
    std::vector<Eigen::MatrixXcd>& Ainv_vec,
    double vram_safety_fraction = 0.80,
    int omp_threads = 0)
{
  // Optional compile-time check (gives clearer errors if signature mismatches)
  static_assert(
      std::is_invocable_v<Builder&, int, double, Eigen::MatrixXcd&>,
      "Builder must be callable as: build_A(int i, double En, Eigen::MatrixXcd& A)");
  // If you want to enforce void return:
  // static_assert(std::is_void_v<std::invoke_result_t<Builder&, int, double, Eigen::MatrixXcd&>>,
  //              "Builder must return void");

  if (total_problem_size <= 0) throw std::runtime_error("total_problem_size must be > 0");
  if (!(vram_safety_fraction > 0.0 && vram_safety_fraction <= 0.95))
    throw std::runtime_error("vram_safety_fraction should be in (0, 0.95]");

  if (omp_threads > 0) omp_set_num_threads(omp_threads);

  const double del_En = (total_problem_size == 1)
                        ? 0.0
                        : (En_final - En_initial) / double(total_problem_size - 1);

  Ainv_vec.assign(size_t(total_problem_size), Eigen::MatrixXcd());

  cublasHandle_t handle{};
  CUBLAS_CHECK(cublasCreate(&handle), "cublasCreate");

  CublasBatchedInvCache cache; // add near cublasCreate

  size_t freeB = 0, totalB = 0;
  CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB), "cudaMemGetInfo");
  const size_t budget = size_t(double(freeB) * vram_safety_fraction);

  const int blockGrow = 256;

  int start = 0;
  while (start < total_problem_size) {
    int end = start;

    std::vector<Eigen::MatrixXcd> Avec_chunk;
    std::vector<int> n_chunk;

    while (end < total_problem_size) {
      const int trialEnd = std::min(total_problem_size, end + blockGrow);
      const int trialCount = trialEnd - start;

      printer("trialEnd:",trialEnd);
      printer("end:", end);
      printer("trialCount:",trialCount);

      // NVCC-safe sizing
      const size_t trialCountSz = static_cast<size_t>(trialCount);

      std::vector<Eigen::MatrixXcd> Avec_trial(trialCountSz);
      std::vector<int>              n_trial(trialCountSz, -1);

      // IMPORTANT: capture build_A by reference safely, even though Builder is a forwarding ref.
      // We create a reference alias to avoid accidental copies in OpenMP region.
      auto& buildA_ref = build_A;

      int omp_counter = 0;

      #pragma omp parallel for schedule(dynamic)
      for (int k = 0; k < trialCount; ++k) {
        const int i = start + k;
        const double En = En_initial + double(i) * del_En;

        Eigen::MatrixXcd A;
        buildA_ref(i, En, A);

        if (A.rows() == A.cols() && A.rows() > 0) {
          n_trial[size_t(k)] = int(A.rows());
          Avec_trial[size_t(k)] = std::move(A);
        }

        #pragma omp critial 
        {
          omp_counter = omp_counter + 1; 
        }
      }

      printer("omp_counter:", omp_counter); 

      for (int k = 0; k < trialCount; ++k) {
        if (n_trial[size_t(k)] <= 0) {
          cublasDestroy(handle);
          throw std::runtime_error("build_A produced an invalid (non-square or empty) matrix.");
        }
      }

      size_t est = 0;
      for (int k = 0; k < trialCount; ++k) est += bytesForOneMatWorstCase(n_trial[size_t(k)]);
      est += 32ull * 1024ull * 1024ull;

      printer("est: ", est); 
      printer("budget: ", budget); 
      std::cout<<std::setprecision(2); 
      printer("free: ", (float(budget) - float(est))/float(budget) *100.0 ); 
      std::cout<<std::setprecision(3); 

      if (est <= budget || trialCount == 1) {
        Avec_chunk.swap(Avec_trial);
        n_chunk.swap(n_trial);
        end = trialEnd;
        if (est > budget && trialCount == 1)
        { 
          std::cout << "breaking condition 1" << std::endl;
          break;
        }
      } else {
        
        std::cout << "breaking condition 2" << std::endl;
          
        break;
      }
    }

    const int chunkCount = end - start;

    printer("chunkCount: ", chunkCount); 

    if (chunkCount <= 0) {
      cublasDestroy(handle);
      throw std::runtime_error("Chunking failure: chunkCount <= 0");
    }

    std::map<int, std::vector<int>> groups;
    for (int k = 0; k < chunkCount; ++k) groups[n_chunk[size_t(k)]].push_back(k);

    for (const auto& kv : groups) {
      const int n = kv.first;
      const auto& locs = kv.second;

      std::vector<Eigen::MatrixXcd> A_group;
      A_group.reserve(locs.size());
      for (int local : locs) A_group.push_back(Avec_chunk[size_t(local)]);

      std::vector<Eigen::MatrixXcd> Ainv_group;
      //invertSameSizeGroup_cublasBatched(handle, n, A_group, Ainv_group);
      invertSameSizeGroup_cublasBatched_cached(handle, n, A_group, Ainv_group, cache);

      int mat_counter = 0; 
      for (size_t t = 0; t < locs.size(); ++t) {
        const int global_i = start + locs[t];
        Ainv_vec[size_t(global_i)] = std::move(Ainv_group[t]);
        mat_counter += 1; 
      }
      printer("finished inverting matrices, n: ", mat_counter); 
    }

    start = end;
  }

  cublasDestroy(handle);
}

/* New Version of the above */

// Helper: invert one chunk and scatter into Ainv_vec
static inline void invert_and_scatter_chunk(
    cublasHandle_t handle,
    CublasBatchedInvCache& cache,
    int start_index,                              // global start index for this chunk
    const std::vector<Eigen::MatrixXcd>& Avec_chunk,
    const std::vector<int>& n_chunk,
    std::vector<Eigen::MatrixXcd>& Ainv_vec)
{
  const int chunkCount = int(Avec_chunk.size());
  if (chunkCount == 0) return;

  // Group by n
  std::map<int, std::vector<int>> groups;
  for (int k = 0; k < chunkCount; ++k) {
    groups[n_chunk[size_t(k)]].push_back(k);
  }

  int tot_val = 0; 
  // Invert per group
  for (const auto& kv : groups) {
    const int n = kv.first;
    const auto& locs = kv.second;

    TIMER("invert chunk");

    std::vector<Eigen::MatrixXcd> A_group;
    A_group.reserve(locs.size());
    for (int local : locs) A_group.push_back(Avec_chunk[size_t(local)]);

    std::vector<Eigen::MatrixXcd> Ainv_group;
    invertSameSizeGroup_cublasBatched_cached(handle, n, A_group, Ainv_group, cache);

    int scatter_size = 0; 
    // Scatter
    for (size_t t = 0; t < locs.size(); ++t) {
      const int global_i = start_index + locs[t];
      Ainv_vec[size_t(global_i)] = std::move(Ainv_group[t]);
      scatter_size += 1; 
    }
    printer("matrix size, n : ",n); 
    printer("total matrices inverted : ", scatter_size); 
    tot_val = tot_val + scatter_size; 
    std::cout << "====================================" << std::endl; 
  }

  printer("total done : ", tot_val); 

}

template <class Builder>
void build_and_invert_energy_sweep_varsize_batched_gpu_v2(
    double En_initial,
    double En_final,
    int total_problem_size,
    Builder&& build_A,
    std::vector<Eigen::MatrixXcd>& Ainv_vec,
    double vram_safety_fraction = 0.80,
    int omp_threads = 0)
{
  static_assert(std::is_invocable_v<Builder&, int, double, Eigen::MatrixXcd&>,
                "Builder must be callable as: build_A(int i, double En, Eigen::MatrixXcd& A)");

  if (total_problem_size <= 0) throw std::runtime_error("total_problem_size must be > 0");
  if (!(vram_safety_fraction > 0.0 && vram_safety_fraction <= 0.95))
    throw std::runtime_error("vram_safety_fraction should be in (0, 0.95]");

  if (omp_threads > 0) omp_set_num_threads(omp_threads);

  const double del_En = (total_problem_size == 1)
                        ? 0.0
                        : (En_final - En_initial) / double(total_problem_size - 1);

  Ainv_vec.assign(size_t(total_problem_size), Eigen::MatrixXcd());

  cublasHandle_t handle{};
  CUBLAS_CHECK(cublasCreate(&handle), "cublasCreate");

  // Cache must live for the full sweep to avoid cudaFree in hot path
  CublasBatchedInvCache cache;

  size_t freeB = 0, totalB = 0;
  CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB), "cudaMemGetInfo");
  const size_t budget = size_t(double(freeB) * vram_safety_fraction);

  const int blockGrow = 384;

  // IMPORTANT: make a reference alias so OpenMP doesn’t accidentally copy
  auto& buildA_ref = build_A;

  // Current chunk state
  std::vector<Eigen::MatrixXcd> Avec_chunk;
  std::vector<int> n_chunk;
  int chunk_start_global = 0;     // global index corresponding to Avec_chunk[0]
  size_t est_chunk = 0;           // estimated bytes currently in chunk (your model)
  bool chunk_initialized = false;

  int i = 0;
  while (i < total_problem_size) {

    // If chunk is empty, mark its start
    if (!chunk_initialized) {
      chunk_start_global = i;
      est_chunk = 32ull * 1024ull * 1024ull; // your overhead pad
      Avec_chunk.clear();
      n_chunk.clear();
      chunk_initialized = true;
    }

    // Build next block [i, blockEnd)
    const int blockEnd   = std::min(total_problem_size, i + blockGrow);
    const int blockCount = blockEnd - i;

    const size_t blockCountSz = static_cast<size_t>(blockCount);

    // FIX: avoid "std::vector<T> x(size_t(y))" vexing-parse
    std::vector<Eigen::MatrixXcd> Avec_block(blockCountSz);
    std::vector<int> n_block(blockCountSz, -1);

    int omp_counter = 0; 

    #pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < blockCount; ++k) {
      const int global_i = i + k;
      const double En = En_initial + double(global_i) * del_En;

      Eigen::MatrixXcd A;
      buildA_ref(global_i, En, A);

      if (A.rows() == A.cols() && A.rows() > 0) {
        n_block[static_cast<size_t>(k)] = int(A.rows());
        Avec_block[static_cast<size_t>(k)] = std::move(A);
        #pragma omp critical 
        {
          omp_counter += 1; 
        }
      }
    }

    printer("omp counter : ", omp_counter); 

    // Validate + estimate incremental bytes for this block
    size_t est_block = 0;
    for (int k = 0; k < blockCount; ++k) {
      if (n_block[size_t(k)] <= 0) {
        cublasDestroy(handle);
        throw std::runtime_error("build_A produced an invalid (non-square or empty) matrix.");
      }
      est_block += bytesForOneMatWorstCase(n_block[size_t(k)]);
    }

    printer("est_chunk : ", est_chunk); 
    printer("est_block : ", est_block); 
    printer("budget : ", budget); 

    // If adding this block would exceed budget, invert current chunk NOW.
    // (But if chunk is empty, we must accept at least one block even if it exceeds budget.)
    if (!Avec_chunk.empty() && (est_chunk + est_block > budget)) {
      invert_and_scatter_chunk(handle, cache, chunk_start_global, Avec_chunk, n_chunk, Ainv_vec);

      // Reset chunk and DO NOT advance i yet; we still need to place this block
      chunk_initialized = false;
      continue;
    }

    // Otherwise, append this block into current chunk
    const size_t oldSz = Avec_chunk.size();
    Avec_chunk.resize(oldSz + size_t(blockCount));
    n_chunk.resize(oldSz + size_t(blockCount));

    for (int k = 0; k < blockCount; ++k) {
      Avec_chunk[oldSz + size_t(k)] = std::move(Avec_block[size_t(k)]);
      n_chunk[oldSz + size_t(k)] = n_block[size_t(k)];
    }

    est_chunk += est_block;
    i = blockEnd;

    // If we reached the end, invert whatever remains
    if (i >= total_problem_size) {
      invert_and_scatter_chunk(handle, cache, chunk_start_global, Avec_chunk, n_chunk, Ainv_vec);
      chunk_initialized = false;
    }
  }

  cublasDestroy(handle);
}


//THis is the one we are currently using 
template <class Builder>
void build_and_invert_energy_sweep_varsize_batched_gpu_v3(
    int blocksize, 
    double En_initial,
    double En_final,
    int total_problem_size,
    Builder&& build_A,
    std::vector<Eigen::MatrixXcd>& Ainv_vec,
    double vram_safety_fraction = 0.80,
    int omp_threads = 0)
{
  static_assert(std::is_invocable_v<Builder&, int, double, Eigen::MatrixXcd&>,
                "Builder must be callable as: build_A(int i, double En, Eigen::MatrixXcd& A)");

  if (total_problem_size <= 0) throw std::runtime_error("total_problem_size must be > 0");
  if (!(vram_safety_fraction > 0.0 && vram_safety_fraction <= 0.95))
    throw std::runtime_error("vram_safety_fraction should be in (0, 0.95]");

  if (omp_threads > 0) omp_set_num_threads(omp_threads);

  const double del_En = (total_problem_size == 1)
                        ? 0.0
                        : (En_final - En_initial) / double(total_problem_size - 1);

  Ainv_vec.assign(size_t(total_problem_size), Eigen::MatrixXcd());

  cublasHandle_t handle{};
  CUBLAS_CHECK(cublasCreate(&handle), "cublasCreate");

  // Cache must live for the full sweep to avoid cudaFree in hot path
  CublasBatchedInvCache cache;

  size_t freeB = 0, totalB = 0;
  CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB), "cudaMemGetInfo");
  const size_t budget = size_t(double(freeB) * vram_safety_fraction);

  
  const int blockGrow = blocksize;

  // IMPORTANT: make a reference alias so OpenMP doesn’t accidentally copy
  auto& buildA_ref = build_A;

  // Current chunk state
  std::vector<Eigen::MatrixXcd> Avec_chunk;
  std::vector<int> n_chunk;
  int chunk_start_global = 0;     // global index corresponding to Avec_chunk[0]
  size_t est_chunk = 0;           // estimated bytes currently in chunk (your model)
  bool chunk_initialized = false;

  int i = 0;
  while (i < total_problem_size) {

    // If chunk is empty, mark its start
    if (!chunk_initialized) {
      chunk_start_global = i;
      est_chunk = 32ull * 1024ull * 1024ull; // your overhead pad
      Avec_chunk.clear();
      n_chunk.clear();
      chunk_initialized = true;
    }

    // Build next block [i, blockEnd)
    const int blockEnd   = std::min(total_problem_size, i + blockGrow);
    const int blockCount = blockEnd - i;

    const size_t blockCountSz = static_cast<size_t>(blockCount);

    // FIX: avoid "std::vector<T> x(size_t(y))" vexing-parse
    std::vector<Eigen::MatrixXcd> Avec_block(blockCountSz);
    std::vector<int> n_block(blockCountSz, -1);

    int omp_counter = 0; 

    #pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < blockCount; ++k) {
      const int global_i = i + k;
      const double En = En_initial + double(global_i) * del_En;

      Eigen::MatrixXcd A;
      buildA_ref(global_i, En, A);

      if (A.rows() == A.cols() && A.rows() > 0) {
        n_block[static_cast<size_t>(k)] = int(A.rows());
        Avec_block[static_cast<size_t>(k)] = std::move(A);
        
      }
    }

    
    printer("est_chunk : ", est_chunk); 
    //printer("est_block : ", est_block); 
    printer("budget : ", budget); 

    // Validate + estimate incremental bytes for this block
    size_t est_block = 0;
    for (int k = 0; k < blockCount; ++k) {
      if (n_block[size_t(k)] <= 0) {
        cublasDestroy(handle);
        throw std::runtime_error("build_A produced an invalid (non-square or empty) matrix.");
      }
      est_block += bytesForOneMatWorstCase(n_block[size_t(k)]);
    }

    if(est_block < budget)
    {
      invert_and_scatter_chunk(handle, cache, chunk_start_global, Avec_block, n_block, Ainv_vec);

      i = blockEnd;

      // Reset chunk and DO NOT advance i yet; we still need to place this block
      chunk_initialized = false;
    }
    else
    {
      std::string errorstr = "chunk size: " + std::to_string(blockGrow) + " is too small, select bigger.\n";
      throw std::runtime_error( errorstr );
    }

    
  }

  cublasDestroy(handle);
}

// --------------------------- Demo main ---------------------------
#ifdef BUILD_DEMO_MAIN
int main() {
  try {
    const double En_i = 0.1;
    const double En_f = 1.0;
    const int totalN = 20000;

    auto buildA = [](int i, double En, Eigen::MatrixXcd& A) {
      int n = 64 + (i % 3) * 16; // 64, 80, 96
      A.resize(n, n);
      A.setZero();
      for (int r = 0; r < n; ++r) {
        A(r, r) = std::complex<double>(2.0 + 0.1 * En, 0.0);
        if (r + 1 < n) A(r, r + 1) = std::complex<double>(0.01, 0.02);
        if (r - 1 >= 0) A(r, r - 1) = std::complex<double>(0.01, -0.02);
      }
    };

    std::vector<Eigen::MatrixXcd> Ainv;
    build_and_invert_energy_sweep_varsize_batched_gpu(
      En_i, En_f, totalN, buildA, Ainv, 0.80, 0
    );

    std::cout << "Done. Ainv size = " << Ainv.size() << "\n";
    for( int i=0; i<Ainv.size(); ++i)
    {
      std::cout << "Ainv[" << i << "] dims = " << Ainv[i].rows() << "x" << Ainv[i].cols() << "\n";

    }
    std::cout<<std::endl; 
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
#endif