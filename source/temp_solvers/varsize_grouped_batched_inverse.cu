// varsize_grouped_batched_inverse.cu
//
// Build:
//   nvcc -O3 -std=c++17 -Xcompiler -fopenmp -I/usr/include/eigen3 \
//  -lcublas -lcudart varsize_grouped_batched_inverse.cu -o varsize_inv \
//  > build.log 2>&1
//
// Key idea:
//   - Different sizes cannot be mixed in a single batched call.
//   - Group by size, then do batched inversion per-size in VRAM-fitting chunks.

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
#include <random>

static inline void checkCuda(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) { std::fprintf(stderr, "CUDA (%s): %s\n", msg, cudaGetErrorString(e)); std::exit(1); }
}
static inline void checkCublas(cublasStatus_t s, const char* msg) {
  if (s != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "cuBLAS (%s): %d\n", msg, (int)s); std::exit(1); }
}

static inline void packEigenColMajor(const Eigen::MatrixXcd& A, cuDoubleComplex* out) {
  const int n = (int)A.rows();
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      out[j*n + i] = make_cuDoubleComplex(A(i,j).real(), A(i,j).imag());
}
static inline void unpackEigenColMajor(const cuDoubleComplex* in, int n, Eigen::MatrixXcd& Ainv) {
  Ainv.resize(n,n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      auto v = in[j*n + i];
      Ainv(i,j) = std::complex<double>(cuCreal(v), cuCimag(v));
    }
}

int main() {
  // -----------------------
  // 1) Generate variable-size matrices with OpenMP
  // -----------------------
  const int K = 50000;                 // number of matrices
  const int nMin = 35, nMax = 156;     // variable sizes
  const double vramSafety = 0.80;     // use 80% of reported free VRAM

  std::vector<Eigen::MatrixXcd> A(K), Ainv(K);
  std::vector<int> nOf(K);

  #pragma omp parallel
  {
    std::mt19937_64 rng(1234ULL + 1337ULL*(unsigned)omp_get_thread_num());
    std::uniform_int_distribution<int> ndist(nMin, nMax);
    std::uniform_real_distribution<double> rdist(-1.0, 1.0);

    #pragma omp for schedule(static)
    for (int k = 0; k < K; ++k) {
      int n = ndist(rng);
      nOf[k] = n;

      A[k].resize(n,n);
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
          A[k](i,j) = {rdist(rng), rdist(rng)};

      // diagonal shift for better conditioning
      A[k].diagonal().array() += std::complex<double>(double(n), 0.0);
    }
  }

  // -----------------------
  // 2) Group indices by size
  // -----------------------
  std::unordered_map<int, std::vector<int>> groups;
  groups.reserve(128);
  for (int k = 0; k < K; ++k) groups[nOf[k]].push_back(k);

  // -----------------------
  // 3) Setup cuBLAS
  // -----------------------
  cublasHandle_t handle;
  checkCublas(cublasCreate(&handle), "cublasCreate");
  cudaStream_t stream;
  checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
  checkCublas(cublasSetStream(handle, stream), "cublasSetStream");

  // Query VRAM once (we’ll conservatively chunk within this)
  size_t freeB=0, totalB=0;
  checkCuda(cudaMemGetInfo(&freeB, &totalB), "cudaMemGetInfo");
  const size_t usableB = size_t(double(freeB) * vramSafety);

  std::printf("GPU free VRAM: %.2f GiB (total %.2f GiB), using %.0f%% safety\n",
              double(freeB)/(1024.0*1024.0*1024.0),
              double(totalB)/(1024.0*1024.0*1024.0),
              100.0*vramSafety);

  // -----------------------
  // 4) For each size, run same-size batched inversions in VRAM-fitting chunks
  // -----------------------
  // We allocate per-size buffers (simple + clear). If you have many sizes,
  // you can optimize by reusing a "max-size" buffer and repointing.
  for (auto& kv : groups) {
    const int n = kv.first;
    auto& idx = kv.second;
    const int B = (int)idx.size();
    const size_t elemsPerMat = size_t(n) * size_t(n);

    // memory estimate per matrix for this size
    const size_t bytesPerMat =
      2ULL * elemsPerMat * sizeof(cuDoubleComplex) +  // dA + dInv
      1ULL * size_t(n) * sizeof(int) +                // pivots
      1ULL * sizeof(int);                             // info

    int chunk = int(usableB / bytesPerMat);
    chunk = std::max(1, std::min(chunk, B));

    std::printf("\nSize n=%d: count=%d, est bytes/mat=%.2f KiB, chunk=%d\n",
                n, B, double(bytesPerMat)/1024.0, chunk);

    // Allocate pinned host staging for one chunk
    cuDoubleComplex* hA = nullptr;
    cuDoubleComplex* hInv = nullptr;
    checkCuda(cudaHostAlloc((void**)&hA,   size_t(chunk)*elemsPerMat*sizeof(cuDoubleComplex), cudaHostAllocDefault),
              "cudaHostAlloc hA");
    checkCuda(cudaHostAlloc((void**)&hInv, size_t(chunk)*elemsPerMat*sizeof(cuDoubleComplex), cudaHostAllocDefault),
              "cudaHostAlloc hInv");

    // Allocate device buffers for one chunk
    cuDoubleComplex *dA=nullptr, *dInv=nullptr;
    int *dPiv=nullptr, *dInfo=nullptr;
    checkCuda(cudaMalloc((void**)&dA,   size_t(chunk)*elemsPerMat*sizeof(cuDoubleComplex)), "cudaMalloc dA");
    checkCuda(cudaMalloc((void**)&dInv, size_t(chunk)*elemsPerMat*sizeof(cuDoubleComplex)), "cudaMalloc dInv");
    checkCuda(cudaMalloc((void**)&dPiv, size_t(chunk)*size_t(n)*sizeof(int)), "cudaMalloc dPiv");
    checkCuda(cudaMalloc((void**)&dInfo, size_t(chunk)*sizeof(int)), "cudaMalloc dInfo");

    // Pointer arrays
    std::vector<cuDoubleComplex*> hAptr(chunk), hInvPtr(chunk);
    for (int i=0;i<chunk;++i) { hAptr[i]=dA + size_t(i)*elemsPerMat; hInvPtr[i]=dInv + size_t(i)*elemsPerMat; }
    cuDoubleComplex **dAptr=nullptr, **dInvPtr=nullptr;
    checkCuda(cudaMalloc((void**)&dAptr,   size_t(chunk)*sizeof(cuDoubleComplex*)), "cudaMalloc dAptr");
    checkCuda(cudaMalloc((void**)&dInvPtr, size_t(chunk)*sizeof(cuDoubleComplex*)), "cudaMalloc dInvPtr");
    checkCuda(cudaMemcpy(dAptr, hAptr.data(), size_t(chunk)*sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice),
              "cudaMemcpy dAptr");
    checkCuda(cudaMemcpy(dInvPtr, hInvPtr.data(), size_t(chunk)*sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice),
              "cudaMemcpy dInvPtr");

    // Process this size-group in chunks
    for (int base=0; base<B; base+=chunk) {
      const int thisB = std::min(chunk, B-base);
      const size_t thisElems = size_t(thisB)*elemsPerMat;

      // Tail chunk: rebuild pointer arrays for thisB
      if (thisB != chunk) {
        for (int i=0;i<thisB;++i) { hAptr[i]=dA + size_t(i)*elemsPerMat; hInvPtr[i]=dInv + size_t(i)*elemsPerMat; }
        checkCuda(cudaMemcpyAsync(dAptr,   hAptr.data(),   size_t(thisB)*sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync dAptr tail");
        checkCuda(cudaMemcpyAsync(dInvPtr, hInvPtr.data(), size_t(thisB)*sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync dInvPtr tail");
      }

      // Pack (OpenMP over matrices inside this chunk)
      #pragma omp parallel for schedule(static)
      for (int i=0;i<thisB;++i) {
        const int k = idx[base + i];
        packEigenColMajor(A[k], hA + size_t(i)*elemsPerMat);
      }

      // H2D
      checkCuda(cudaMemcpyAsync(dA, hA, thisElems*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream),
                "cudaMemcpyAsync H2D dA");

      // Batched LU + Inverse (same n for all in this chunk)
      checkCublas(cublasZgetrfBatched(handle, n, dAptr, n, dPiv, dInfo, thisB), "cublasZgetrfBatched");
      checkCublas(cublasZgetriBatched(handle, n,
                                      (const cuDoubleComplex**)dAptr, n,
                                      dPiv,
                                      dInvPtr, n,
                                      dInfo, thisB),
                  "cublasZgetriBatched");

      // D2H
      checkCuda(cudaMemcpyAsync(hInv, dInv, thisElems*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, stream),
                "cudaMemcpyAsync D2H hInv");
      checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

      // Unpack back into Eigen
      #pragma omp parallel for schedule(static)
      for (int i=0;i<thisB;++i) {
        const int k = idx[base + i];
        unpackEigenColMajor(hInv + size_t(i)*elemsPerMat, n, Ainv[k]);
      }

      std::printf("  inverted n=%d chunk [%d,%d)\n", n, base, base+thisB);
    }

    // Cleanup per-size allocations
    checkCuda(cudaFree(dA), "cudaFree dA");
    checkCuda(cudaFree(dInv), "cudaFree dInv");
    checkCuda(cudaFree(dPiv), "cudaFree dPiv");
    checkCuda(cudaFree(dInfo), "cudaFree dInfo");
    checkCuda(cudaFree(dAptr), "cudaFree dAptr");
    checkCuda(cudaFree(dInvPtr), "cudaFree dInvPtr");
    checkCuda(cudaFreeHost(hA), "cudaFreeHost hA");
    checkCuda(cudaFreeHost(hInv), "cudaFreeHost hInv");
  }

  // -----------------------
  // Done
  // -----------------------
  checkCublas(cublasDestroy(handle), "cublasDestroy");
  checkCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");

  // Example sanity check for one random k
  {
    int k = 0;
    int n = nOf[k];
    Eigen::MatrixXcd I = A[k] * Ainv[k];
    double err = (I - Eigen::MatrixXcd::Identity(n,n)).norm();
    std::printf("\nSanity k=%d (n=%d): ||A*Ainv - I||_F = %.3e\n", k, n, err);
  }

  return 0;
}