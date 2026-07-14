// cusolver_varsize_solve_AXeqI.cu
//
// Robust path: cuSOLVER per-matrix LU + solve for B=I (so A*X=I).
// Variable sizes handled by grouping. VRAM batching handled by conservative estimator.
//
// Build:
//   nvcc -O3 -std=c++17 -Xcompiler -fopenmp cusolver_varsize_solve_AXeqI.cu -lcusolver -lcublas -o solve_AXeqI

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

using Complex = cuDoubleComplex;

// -------------------- Error checks --------------------
static inline void CHECK_CUDA(cudaError_t st, const char* msg) {
  if (st != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s | %s\n", msg, cudaGetErrorString(st));
    std::exit(1);
  }
}
static inline void CHECK_SOLVER(cusolverStatus_t st, const char* msg) {
  if (st != CUSOLVER_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuSOLVER error: %s | status=%d\n", msg, (int)st);
    std::exit(1);
  }
}

// -------------------- Complex helpers (CPU check) --------------------
static inline double cre(Complex z) { return cuCreal(z); }
static inline double cim(Complex z) { return cuCimag(z); }

static inline Complex cadd(Complex a, Complex b) {
  return make_cuDoubleComplex(cre(a) + cre(b), cim(a) + cim(b));
}
static inline Complex csub(Complex a, Complex b) {
  return make_cuDoubleComplex(cre(a) - cre(b), cim(a) - cim(b));
}
static inline Complex cmul(Complex a, Complex b) {
  double ar = cre(a), ai = cim(a);
  double br = cre(b), bi = cim(b);
  return make_cuDoubleComplex(ar * br - ai * bi, ar * bi + ai * br);
}
static inline double cabs2(Complex a) {
  double ar = cre(a), ai = cim(a);
  return ar * ar + ai * ai;
}

// -------------------- Data structures --------------------
struct HostMatrix {
  int n = 0;
  std::vector<Complex> A; // column-major
};

struct SolveResult {
  int n = 0;
  std::vector<Complex> X; // solution of A*X=I
  int info = 0;           // 0 success; >0 singular pivot; <0 invalid param
};

// -------------------- GPU kernel: set batched identities --------------------
__global__ void set_identity_batched(Complex* B, int n, int batchCount) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int nn = n * n;
  int total = nn * batchCount;
  if (idx >= total) return;

  int local = idx % nn;
  int row = local % n;
  int col = local / n;
  double v = (row == col) ? 1.0 : 0.0;
  B[idx] = make_cuDoubleComplex(v, 0.0);
}

// -------------------- Generator (customize) --------------------
static inline int size_fn(int i) {
  // Example: 64,128,192,256 repeating
  return 64 + (i % 4) * 64;
}

static inline void fill_fn(int i, int n, Complex* outA) {
  std::mt19937_64 rng(1234ULL + (uint64_t)i);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < n; ++row) {
      double re = dist(rng);
      double im = dist(rng);
      outA[size_t(col) * n + row] = make_cuDoubleComplex(re, im);
    }
  }
  // diagonal dominance to reduce singularity likelihood
  for (int d = 0; d < n; ++d) {
    Complex z = outA[size_t(d) * n + d];
    outA[size_t(d) * n + d] = make_cuDoubleComplex(cre(z) + double(n), cim(z));
  }
}

// -------------------- Conservative VRAM estimate --------------------
static inline size_t bytes_per_matrix_estimate(int n) {
  // We store A and B (B holds identity then X), pivots, info.
  // We do NOT know cuSOLVER workspace yet; keep budget conservative via vram_fraction.
  size_t nn = size_t(n) * size_t(n);
  size_t A = nn * sizeof(Complex);
  size_t B = nn * sizeof(Complex);
  size_t piv = size_t(n) * sizeof(int);
  size_t inf = sizeof(int);
  return size_t(double(A + B + piv + inf) * 1.25) + 16384;
}

// -------------------- CPU residual check: ||A*X - I||_F / ||I||_F --------------------
static double residual_AX_minus_I(const HostMatrix& hm, const SolveResult& res) {
  int n = hm.n;
  assert(res.n == n);
  const Complex* A = hm.A.data();
  const Complex* X = res.X.data();

  std::vector<Complex> C(size_t(n) * n, make_cuDoubleComplex(0.0, 0.0));
  for (int j = 0; j < n; ++j) {        // col of C
    for (int k = 0; k < n; ++k) {
      Complex xkj = X[size_t(j) * n + k]; // X(k,j)
      for (int i = 0; i < n; ++i) {
        Complex aik = A[size_t(k) * n + i]; // A(i,k)
        C[size_t(j) * n + i] = cadd(C[size_t(j) * n + i], cmul(aik, xkj));
      }
    }
  }

  double num = 0.0;
  double den = double(n); // ||I||_F^2 = n
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < n; ++row) {
      Complex cij = C[size_t(col) * n + row];
      Complex iij = (row == col) ? make_cuDoubleComplex(1.0, 0.0)
                                 : make_cuDoubleComplex(0.0, 0.0);
      Complex diff = csub(cij, iij);
      num += cabs2(diff);
    }
  }
  return std::sqrt(num) / std::sqrt(den);
}

// -------------------- Solve a same-n group using cuSOLVER getrf/getrs --------------------
static void solve_group_same_n_cusolver(
    cusolverDnHandle_t solver,
    cudaStream_t stream,
    int n,
    const std::vector<int>& localIdx,
    const std::vector<HostMatrix>& hostBatch,
    const std::vector<int>& absIds,
    std::vector<SolveResult>& out
) {
  const int m = (int)localIdx.size();
  if (m == 0) return;

  const size_t nn = size_t(n) * size_t(n);
  const size_t bytesA = nn * m * sizeof(Complex);
  const size_t bytesB = nn * m * sizeof(Complex);

  Complex* dA = nullptr;
  Complex* dB = nullptr;
  int* dIpiv = nullptr;   // pivot arrays packed: n * m
  int* dInfo = nullptr;   // info per matrix: m

  CHECK_CUDA(cudaMalloc((void**)&dA, bytesA), "malloc dA");
  CHECK_CUDA(cudaMalloc((void**)&dB, bytesB), "malloc dB");
  CHECK_CUDA(cudaMalloc((void**)&dIpiv, size_t(n) * m * sizeof(int)), "malloc dIpiv");
  CHECK_CUDA(cudaMalloc((void**)&dInfo, size_t(m) * sizeof(int)), "malloc dInfo");

  // Pack A from host
  std::vector<Complex> packedA(nn * m);
  for (int k = 0; k < m; ++k) {
    int j = localIdx[k];
    std::memcpy(packedA.data() + nn * k, hostBatch[j].A.data(), nn * sizeof(Complex));
  }
  CHECK_CUDA(cudaMemcpyAsync(dA, packedA.data(), bytesA, cudaMemcpyHostToDevice, stream),
             "H2D A");

  // Set B = I for all matrices in this group (batched identity fill)
  int threads = 256;
  int total = int(nn * m);
  int blocks = (total + threads - 1) / threads;
  set_identity_batched<<<blocks, threads, 0, stream>>>(dB, n, m);
  CHECK_CUDA(cudaGetLastError(), "identity kernel");

  CHECK_SOLVER(cusolverDnSetStream(solver, stream), "set solver stream");

  // Workspace query for one matrix (same n for all)
  int lwork = 0;
  CHECK_SOLVER(cusolverDnZgetrf_bufferSize(solver, n, n, dA, n, &lwork), "getrf_bufferSize");
  Complex* dWork = nullptr;
  CHECK_CUDA(cudaMalloc((void**)&dWork, size_t(lwork) * sizeof(Complex)), "malloc dWork");

  // For each matrix: LU then solve with nrhs=n (B=I). B overwritten with X.
  for (int k = 0; k < m; ++k) {
    Complex* A_k = dA + nn * k;
    Complex* B_k = dB + nn * k;
    int* ipiv_k = dIpiv + n * k;
    int* info_k = dInfo + k;

    CHECK_SOLVER(cusolverDnZgetrf(solver, n, n, A_k, n, dWork, ipiv_k, info_k), "Zgetrf");
    CHECK_SOLVER(cusolverDnZgetrs(solver, CUBLAS_OP_N, n, n, A_k, n, ipiv_k, B_k, n, info_k), "Zgetrs");
  }

  // Copy back B (solutions) and info
  std::vector<Complex> packedX(nn * m);
  std::vector<int> hInfo(m, -999);
  CHECK_CUDA(cudaMemcpyAsync(packedX.data(), dB, bytesB, cudaMemcpyDeviceToHost, stream), "D2H X");
  CHECK_CUDA(cudaMemcpyAsync(hInfo.data(), dInfo, size_t(m) * sizeof(int), cudaMemcpyDeviceToHost, stream),
             "D2H info");
  CHECK_CUDA(cudaStreamSynchronize(stream), "sync");

  // Scatter results
  for (int k = 0; k < m; ++k) {
    int j = localIdx[k];
    int abs = absIds[j];
    out[abs].n = n;
    out[abs].info = hInfo[k];
    out[abs].X.resize(nn);
    std::memcpy(out[abs].X.data(), packedX.data() + nn * k, nn * sizeof(Complex));
  }

  cudaFree(dWork);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dIpiv);
  cudaFree(dInfo);
}

// -------------------- Main --------------------
int main() {
  const int N = 10000;
  const double vram_fraction = 0.55; // keep conservative because cuSOLVER workspace is extra
  const int check_first_k = 5;

  CHECK_CUDA(cudaSetDevice(0), "set device");

  cusolverDnHandle_t solver;
  CHECK_SOLVER(cusolverDnCreate(&solver), "create solver");

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream), "create stream");

  std::vector<SolveResult> results(N);

  int nextId = 0;
  int batchNo = 0;

  while (nextId < N) {
    size_t freeB = 0, totalB = 0;
    CHECK_CUDA(cudaMemGetInfo(&freeB, &totalB), "memgetinfo");
    size_t budget = size_t(double(freeB) * vram_fraction);

    std::vector<int> ids;
    ids.reserve(512);
    size_t used = 0;

    while (nextId < N) {
      int id = nextId;
      int n = size_fn(id);
      size_t need = bytes_per_matrix_estimate(n);
      if (!ids.empty() && used + need > budget) break;
      used += need;
      ids.push_back(id);
      nextId++;
    }
    if (ids.empty()) { ids.push_back(nextId); nextId++; }

    // Generate host matrices (OpenMP)
    std::vector<HostMatrix> hostBatch(ids.size());
    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < (int)ids.size(); ++j) {
      int id = ids[j];
      int n = size_fn(id);
      hostBatch[j].n = n;
      hostBatch[j].A.resize(size_t(n) * n);
      fill_fn(id, n, hostBatch[j].A.data());
    }

    // Group by n
    std::unordered_map<int, std::vector<int>> groups;
    groups.reserve(hostBatch.size());
    for (int j = 0; j < (int)hostBatch.size(); ++j) {
      groups[hostBatch[j].n].push_back(j);
    }

    // Solve each size-group
    for (auto& kv : groups) {
      solve_group_same_n_cusolver(solver, stream, kv.first, kv.second, hostBatch, ids, results);
    }

    std::cout << "Batch " << batchNo++
              << " processed: " << ids.size()
              << " matrices | free=" << (freeB / (1024.0 * 1024.0)) << " MiB"
              << " | budget=" << (budget / (1024.0 * 1024.0)) << " MiB\n";

    // Quick check
    int checked = 0;
    for (int j = 0; j < (int)ids.size() && checked < check_first_k; ++j) {
      int abs = ids[j];
      if (results[abs].info != 0) continue;
      double rel = residual_AX_minus_I(hostBatch[j], results[abs]);
      std::cout << "  check id=" << abs << " n=" << hostBatch[j].n
                << " relFrob(A*X-I)=" << rel << "\n";
      checked++;
    }
  }

  int ok = 0, bad = 0;
  for (int i = 0; i < N; ++i) (results[i].info == 0) ? ok++ : bad++;
  std::cout << "Done: success=" << ok << " fail=" << bad << "\n";

  cudaStreamDestroy(stream);
  cusolverDnDestroy(solver);
  return 0;
}