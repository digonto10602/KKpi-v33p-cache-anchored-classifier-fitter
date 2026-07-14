// varsize_grouped_batched_inverse_lib.cu
//
// Provide a callable function:
//   void invert_varsize_mats_batched_gpu(const std::vector<Eigen::MatrixXcd>& A,
//                                       std::vector<Eigen::MatrixXcd>& Ainv,
//                                       double vramSafety=0.80,
//                                       int maxChunkOverride=0);
//
// Behavior:
//   - A can contain different n x n (square) sizes.
//   - Groups matrices by size, runs cuBLAS batched LU+inverse per-size.
//   - Chunks per-size batches to fit GPU VRAM.
//   - Writes results into Ainv (resizes each Ainv[k] to match A[k]).
//
// Build example:
//   nvcc -O3 -std=c++17 -Xcompiler -fopenmp -I/path/to/eigen \
//        -lcublas -lcudart varsize_grouped_batched_inverse_lib.cu -c -o invlib.o
//
// Link into your project with nvcc (or g++ + cuda libs appropriately).

#include <Eigen/Dense>
#include <omp.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>

static inline void checkCuda(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(e));
    throw std::runtime_error(std::string("CUDA error: ") + msg);
  }
}
static inline void checkCublas(cublasStatus_t s, const char* msg) {
  if (s != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuBLAS error (%s): status=%d\n", msg, (int)s);
    throw std::runtime_error(std::string("cuBLAS error: ") + msg);
  }
}

static inline void packEigenColMajor(const Eigen::MatrixXcd& A, cuDoubleComplex* out) {
  const int n = (int)A.rows();
  // Eigen::MatrixXcd is column-major by default, but we still pack explicitly.
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      out[j*n + i] = make_cuDoubleComplex(A(i,j).real(), A(i,j).imag());
}

static inline void unpackEigenColMajor(const cuDoubleComplex* in, int n, Eigen::MatrixXcd& Ainv) {
  Ainv.resize(n, n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const cuDoubleComplex v = in[j*n + i];
      Ainv(i,j) = std::complex<double>(cuCreal(v), cuCimag(v));
    }
}

// Public API
void invert_varsize_mats_batched_gpu(const std::vector<Eigen::MatrixXcd>& A,
                                    std::vector<Eigen::MatrixXcd>& Ainv,
                                    double vramSafety,
                                    int maxChunkOverride)
{
  const int K = (int)A.size();
  Ainv.clear();
  Ainv.resize(K);

  if (K == 0) return;

  // Validate inputs + build grouping by n
  std::unordered_map<int, std::vector<int>> groups;
  groups.reserve(128);

  for (int k = 0; k < K; ++k) {
    if (A[k].rows() != A[k].cols())
      throw std::runtime_error("invert_varsize_mats_batched_gpu: A contains a non-square matrix at index " + std::to_string(k));
    const int n = (int)A[k].rows();
    if (n <= 0)
      throw std::runtime_error("invert_varsize_mats_batched_gpu: A contains an empty matrix at index " + std::to_string(k));
    groups[n].push_back(k);
  }

  // Create cuBLAS + stream
  cublasHandle_t handle;
  checkCublas(cublasCreate(&handle), "cublasCreate");
  cudaStream_t stream;
  checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
  checkCublas(cublasSetStream(handle, stream), "cublasSetStream");

  // Query VRAM (free at the time of call)
  size_t freeB = 0, totalB = 0;
  checkCuda(cudaMemGetInfo(&freeB, &totalB), "cudaMemGetInfo");
  if (vramSafety <= 0.0 || vramSafety > 1.0) vramSafety = 0.80;
  const size_t usableB = size_t(double(freeB) * vramSafety);

  // For each size group, do batched inverse in VRAM-fitting chunks
  for (auto& kv : groups) {
    const int n = kv.first;
    std::vector<int>& idx = kv.second;
    const int B = (int)idx.size();
    const size_t elemsPerMat = size_t(n) * size_t(n);

    // Rough per-matrix GPU bytes:
    // dA + dInv + pivots + info (pointer arrays are small)
    const size_t bytesPerMat =
      2ULL * elemsPerMat * sizeof(cuDoubleComplex) +
      1ULL * size_t(n) * sizeof(int) +
      1ULL * sizeof(int);

    int chunk = (usableB > 0) ? int(usableB / bytesPerMat) : 1;
    chunk = std::max(1, std::min(chunk, B));

    if (maxChunkOverride > 0) chunk = std::max(1, std::min(chunk, std::min(B, maxChunkOverride)));

    // Allocate pinned host staging for one chunk
    cuDoubleComplex* hA = nullptr;
    cuDoubleComplex* hInv = nullptr;

    checkCuda(cudaHostAlloc((void**)&hA,   size_t(chunk) * elemsPerMat * sizeof(cuDoubleComplex), cudaHostAllocDefault),
              "cudaHostAlloc hA");
    checkCuda(cudaHostAlloc((void**)&hInv, size_t(chunk) * elemsPerMat * sizeof(cuDoubleComplex), cudaHostAllocDefault),
              "cudaHostAlloc hInv");

    // Allocate device buffers for one chunk
    cuDoubleComplex *dA = nullptr, *dInv = nullptr;
    int *dPiv = nullptr, *dInfo = nullptr;

    checkCuda(cudaMalloc((void**)&dA,   size_t(chunk) * elemsPerMat * sizeof(cuDoubleComplex)), "cudaMalloc dA");
    checkCuda(cudaMalloc((void**)&dInv, size_t(chunk) * elemsPerMat * sizeof(cuDoubleComplex)), "cudaMalloc dInv");
    checkCuda(cudaMalloc((void**)&dPiv, size_t(chunk) * size_t(n) * sizeof(int)), "cudaMalloc dPiv");
    checkCuda(cudaMalloc((void**)&dInfo, size_t(chunk) * sizeof(int)), "cudaMalloc dInfo");

    // Pointer arrays (device)
    std::vector<cuDoubleComplex*> hAptr(chunk), hInvPtr(chunk);
    for (int i = 0; i < chunk; ++i) {
      hAptr[i]   = dA   + size_t(i) * elemsPerMat;
      hInvPtr[i] = dInv + size_t(i) * elemsPerMat;
    }

    cuDoubleComplex **dAptr = nullptr, **dInvPtr = nullptr;
    checkCuda(cudaMalloc((void**)&dAptr,   size_t(chunk) * sizeof(cuDoubleComplex*)), "cudaMalloc dAptr");
    checkCuda(cudaMalloc((void**)&dInvPtr, size_t(chunk) * sizeof(cuDoubleComplex*)), "cudaMalloc dInvPtr");
    checkCuda(cudaMemcpy(dAptr, hAptr.data(), size_t(chunk) * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice),
              "cudaMemcpy dAptr");
    checkCuda(cudaMemcpy(dInvPtr, hInvPtr.data(), size_t(chunk) * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice),
              "cudaMemcpy dInvPtr");

    // Process in chunks
    for (int base = 0; base < B; base += chunk) {
      const int thisB = std::min(chunk, B - base);
      const size_t thisElems = size_t(thisB) * elemsPerMat;

      // Tail chunk: update pointer arrays for thisB
      if (thisB != chunk) {
        for (int i = 0; i < thisB; ++i) {
          hAptr[i]   = dA   + size_t(i) * elemsPerMat;
          hInvPtr[i] = dInv + size_t(i) * elemsPerMat;
        }
        checkCuda(cudaMemcpyAsync(dAptr, hAptr.data(), size_t(thisB) * sizeof(cuDoubleComplex*),
                                  cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync dAptr tail");
        checkCuda(cudaMemcpyAsync(dInvPtr, hInvPtr.data(), size_t(thisB) * sizeof(cuDoubleComplex*),
                                  cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync dInvPtr tail");
      }

      // Pack chunk (OpenMP parallel over matrices)
      #pragma omp parallel for schedule(static)
      for (int i = 0; i < thisB; ++i) {
        const int k = idx[base + i];
        packEigenColMajor(A[k], hA + size_t(i) * elemsPerMat);
      }

      // H2D
      checkCuda(cudaMemcpyAsync(dA, hA, thisElems * sizeof(cuDoubleComplex),
                                cudaMemcpyHostToDevice, stream),
                "cudaMemcpyAsync H2D dA");

      // Batched LU + Inverse (same n)
      checkCublas(cublasZgetrfBatched(handle, n, dAptr, n, dPiv, dInfo, thisB), "cublasZgetrfBatched");
      checkCublas(cublasZgetriBatched(handle, n,
                                      (const cuDoubleComplex**)dAptr, n,
                                      dPiv,
                                      dInvPtr, n,
                                      dInfo, thisB),
                  "cublasZgetriBatched");

      // D2H
      checkCuda(cudaMemcpyAsync(hInv, dInv, thisElems * sizeof(cuDoubleComplex),
                                cudaMemcpyDeviceToHost, stream),
                "cudaMemcpyAsync D2H hInv");
      checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

      // Unpack back into output vector
      #pragma omp parallel for schedule(static)
      for (int i = 0; i < thisB; ++i) {
        const int k = idx[base + i];
        unpackEigenColMajor(hInv + size_t(i) * elemsPerMat, n, Ainv[k]);
      }
    }

    // Cleanup per-size
    checkCuda(cudaFree(dA), "cudaFree dA");
    checkCuda(cudaFree(dInv), "cudaFree dInv");
    checkCuda(cudaFree(dPiv), "cudaFree dPiv");
    checkCuda(cudaFree(dInfo), "cudaFree dInfo");
    checkCuda(cudaFree(dAptr), "cudaFree dAptr");
    checkCuda(cudaFree(dInvPtr), "cudaFree dInvPtr");
    checkCuda(cudaFreeHost(hA), "cudaFreeHost hA");
    checkCuda(cudaFreeHost(hInv), "cudaFreeHost hInv");
  }

  // Destroy cuBLAS/stream
  checkCublas(cublasDestroy(handle), "cublasDestroy");
  checkCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
}