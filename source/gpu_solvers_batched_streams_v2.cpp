#include <vector>
#include <functional>
#include <unordered_map>
#include <algorithm>
#include <complex>
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuComplex.h>

// your checkCudaError / checkCusolverError / checkCublasError assumed available

#include <cstdio>
#include <cstdlib>
#include <string>


static inline void checkCudaError(cudaError_t status, const char* msg) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "[CUDA] %s failed: %s\n", msg, cudaGetErrorString(status));
        std::fflush(stderr);
        std::exit(EXIT_FAILURE);
    }
}

static inline const char* cusolverStatusToString(cusolverStatus_t status) {
    switch (status) {
        case CUSOLVER_STATUS_SUCCESS: return "CUSOLVER_STATUS_SUCCESS";
        case CUSOLVER_STATUS_NOT_INITIALIZED: return "CUSOLVER_STATUS_NOT_INITIALIZED";
        case CUSOLVER_STATUS_ALLOC_FAILED: return "CUSOLVER_STATUS_ALLOC_FAILED";
        case CUSOLVER_STATUS_INVALID_VALUE: return "CUSOLVER_STATUS_INVALID_VALUE";
        case CUSOLVER_STATUS_ARCH_MISMATCH: return "CUSOLVER_STATUS_ARCH_MISMATCH";
        case CUSOLVER_STATUS_MAPPING_ERROR: return "CUSOLVER_STATUS_MAPPING_ERROR";
        case CUSOLVER_STATUS_EXECUTION_FAILED: return "CUSOLVER_STATUS_EXECUTION_FAILED";
        case CUSOLVER_STATUS_INTERNAL_ERROR: return "CUSOLVER_STATUS_INTERNAL_ERROR";
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
#if defined(CUSOLVER_STATUS_NOT_SUPPORTED)
        case CUSOLVER_STATUS_NOT_SUPPORTED: return "CUSOLVER_STATUS_NOT_SUPPORTED";
#endif
        default: return "CUSOLVER_STATUS_<unknown>";
    }
}

static inline void checkCusolverError(cusolverStatus_t status, const char* msg) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::fprintf(stderr, "[cuSOLVER] %s failed: %s (%d)\n",
                     msg, cusolverStatusToString(status), (int)status);
        std::fflush(stderr);
        std::exit(EXIT_FAILURE);
    }
}

static inline const char* cublasStatusToString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "CUBLAS_STATUS_<unknown>";
    }
}

static inline void checkCublasError(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "[cuBLAS] %s failed: %s (%d)\n",
                     msg, cublasStatusToString(status), (int)status);
        std::fflush(stderr);
        std::exit(EXIT_FAILURE);
    }
}

static inline cuDoubleComplex to_cu(const std::complex<double>& z) {
    return make_cuDoubleComplex(z.real(), z.imag());
}
static inline std::complex<double> from_cu(const cuDoubleComplex& z) {
    return {cuCreal(z), cuCimag(z)};
}

struct QRStreamBuffers {
    cuDoubleComplex* d_A   = nullptr;
    cuDoubleComplex* d_B   = nullptr;
    cuDoubleComplex* d_tau = nullptr;
    cuDoubleComplex* d_work= nullptr;
    int* d_info            = nullptr;

    // pinned host staging (optional but helps)
    cuDoubleComplex* h_A   = nullptr;
    cuDoubleComplex* h_B   = nullptr;
    cuDoubleComplex* h_X   = nullptr;

    int lwork = 0;
};

void cusolverBatchedQR_pipeline_fixed_m(
    const std::vector<int>& indices,     // global indices solved in this call
    int batch_streams,                   // number of concurrent streams
    int m,                               // fixed matrix size for this group
    int nrhs,
    std::function<void(int /*global_idx*/, int /*m_fixed*/, Eigen::MatrixXcd&, Eigen::MatrixXcd&)> matrix_generator,
    std::vector<Eigen::MatrixXcd>& X_vec // output, sized to total_problems outside
){
    if (indices.empty()) return;

    const int lda = m;
    const int ldb = m;
    const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);

    // Persistent CUDA resources
    std::vector<cudaStream_t> streams(batch_streams);
    std::vector<cusolverDnHandle_t> cusolverHandles(batch_streams);
    std::vector<cublasHandle_t> cublasHandles(batch_streams);
    std::vector<QRStreamBuffers> bufs(batch_streams);

    for (int s = 0; s < batch_streams; ++s) {
        checkCudaError(cudaStreamCreate(&streams[s]), "cudaStreamCreate");
        checkCusolverError(cusolverDnCreate(&cusolverHandles[s]), "cusolverDnCreate");
        checkCublasError(cublasCreate(&cublasHandles[s]), "cublasCreate");
        checkCusolverError(cusolverDnSetStream(cusolverHandles[s], streams[s]), "cusolverDnSetStream");
        checkCublasError(cublasSetStream(cublasHandles[s], streams[s]), "cublasSetStream");

        // allocate device buffers once per stream
        checkCudaError(cudaMalloc(&bufs[s].d_A,   sizeof(cuDoubleComplex) * lda * m), "cudaMalloc d_A");
        checkCudaError(cudaMalloc(&bufs[s].d_B,   sizeof(cuDoubleComplex) * ldb * nrhs), "cudaMalloc d_B");
        checkCudaError(cudaMalloc(&bufs[s].d_tau, sizeof(cuDoubleComplex) * m), "cudaMalloc d_tau");
        checkCudaError(cudaMalloc(&bufs[s].d_info,sizeof(int)), "cudaMalloc d_info");

        // workspace query ONCE (geqrf + unmqr)
        int lwork_geqrf = 0, lwork_unmqr = 0;
        checkCusolverError(
            cusolverDnZgeqrf_bufferSize(cusolverHandles[s], m, m, bufs[s].d_A, lda, &lwork_geqrf),
            "cusolverDnZgeqrf_bufferSize"
        );
        checkCusolverError(
            cusolverDnZunmqr_bufferSize(cusolverHandles[s],
                                        CUBLAS_SIDE_LEFT, CUBLAS_OP_C,
                                        m, nrhs, m, bufs[s].d_A, lda,
                                        bufs[s].d_tau, bufs[s].d_B, ldb,
                                        &lwork_unmqr),
            "cusolverDnZunmqr_bufferSize"
        );
        bufs[s].lwork = std::max(lwork_geqrf, lwork_unmqr);
        checkCudaError(cudaMalloc(&bufs[s].d_work, sizeof(cuDoubleComplex) * bufs[s].lwork), "cudaMalloc d_work");

        // pinned host staging (optional but recommended)
        checkCudaError(cudaMallocHost(&bufs[s].h_A, sizeof(cuDoubleComplex) * lda * m), "cudaMallocHost h_A");
        checkCudaError(cudaMallocHost(&bufs[s].h_B, sizeof(cuDoubleComplex) * ldb * nrhs), "cudaMallocHost h_B");
        checkCudaError(cudaMallocHost(&bufs[s].h_X, sizeof(cuDoubleComplex) * ldb * nrhs), "cudaMallocHost h_X");
    }

    // Process indices in chunks of batch_streams (each stream handles one system at a time)
    for (size_t k0 = 0; k0 < indices.size(); k0 += (size_t)batch_streams) {
        const int chunk = (int)std::min((size_t)batch_streams, indices.size() - k0);

        // CPU generate + H2D + launch QR for each stream in this chunk
        for (int s = 0; s < chunk; ++s) {
            const int global_idx = indices[k0 + s];

            Eigen::MatrixXcd A(m, m);
            Eigen::MatrixXcd B(m, nrhs);
            matrix_generator(global_idx, m, A, B); // must fill A,B as m x m and m x nrhs (padded if needed)

            // pack Eigen -> column-major cuDoubleComplex buffers (cuSOLVER expects column-major)
            // A is (m x m)
            int idx = 0;
            for (int col = 0; col < m; ++col) {
                for (int row = 0; row < m; ++row) {
                    bufs[s].h_A[idx++] = to_cu(A(row, col));
                }
            }
            // B is (m x nrhs)
            idx = 0;
            for (int col = 0; col < nrhs; ++col) {
                for (int row = 0; row < m; ++row) {
                    bufs[s].h_B[idx++] = to_cu(B(row, col));
                }
            }

            checkCudaError(cudaMemcpyAsync(bufs[s].d_A, bufs[s].h_A,
                                           sizeof(cuDoubleComplex) * lda * m,
                                           cudaMemcpyHostToDevice, streams[s]),
                           "cudaMemcpyAsync A");
            checkCudaError(cudaMemcpyAsync(bufs[s].d_B, bufs[s].h_B,
                                           sizeof(cuDoubleComplex) * ldb * nrhs,
                                           cudaMemcpyHostToDevice, streams[s]),
                           "cudaMemcpyAsync B");

            // QR factorization
            checkCusolverError(
                cusolverDnZgeqrf(cusolverHandles[s], m, m,
                                 bufs[s].d_A, lda,
                                 bufs[s].d_tau,
                                 bufs[s].d_work, bufs[s].lwork,
                                 bufs[s].d_info),
                "cusolverDnZgeqrf"
            );

            // Apply Q^H to B
            checkCusolverError(
                cusolverDnZunmqr(cusolverHandles[s],
                                 CUBLAS_SIDE_LEFT, CUBLAS_OP_C,
                                 m, nrhs, m,
                                 bufs[s].d_A, lda,
                                 bufs[s].d_tau,
                                 bufs[s].d_B, ldb,
                                 bufs[s].d_work, bufs[s].lwork,
                                 bufs[s].d_info),
                "cusolverDnZunmqr"
            );

            // Solve R X = (Q^H B)
            checkCublasError(
                cublasZtrsm(cublasHandles[s],
                            CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                            m, nrhs, &one,
                            bufs[s].d_A, lda,
                            bufs[s].d_B, ldb),
                "cublasZtrsm"
            );

            // D2H async into pinned
            checkCudaError(cudaMemcpyAsync(bufs[s].h_X, bufs[s].d_B,
                                           sizeof(cuDoubleComplex) * ldb * nrhs,
                                           cudaMemcpyDeviceToHost, streams[s]),
                           "cudaMemcpyAsync X");
        }

        // sync chunk streams and write results
        for (int s = 0; s < chunk; ++s) {
            checkCudaError(cudaStreamSynchronize(streams[s]), "cudaStreamSynchronize");

            Eigen::MatrixXcd X(m, nrhs);
            int idx = 0;
            for (int col = 0; col < nrhs; ++col) {
                for (int row = 0; row < m; ++row) {
                    X(row, col) = from_cu(bufs[s].h_X[idx++]);
                }
            }
            const int global_idx = indices[k0 + s];
            X_vec[global_idx] = std::move(X);
        }
    }

    // cleanup
    for (int s = 0; s < batch_streams; ++s) {
        if (bufs[s].d_A)    cudaFree(bufs[s].d_A);
        if (bufs[s].d_B)    cudaFree(bufs[s].d_B);
        if (bufs[s].d_tau)  cudaFree(bufs[s].d_tau);
        if (bufs[s].d_work) cudaFree(bufs[s].d_work);
        if (bufs[s].d_info) cudaFree(bufs[s].d_info);

        if (bufs[s].h_A) cudaFreeHost(bufs[s].h_A);
        if (bufs[s].h_B) cudaFreeHost(bufs[s].h_B);
        if (bufs[s].h_X) cudaFreeHost(bufs[s].h_X);

        cusolverDnDestroy(cusolverHandles[s]);
        cublasDestroy(cublasHandles[s]);
        cudaStreamDestroy(streams[s]);
    }
}

void cusolverBatchedQR_pipeline_variable_m(
    int total_problems,
    int batch_streams,
    int nrhs,
    std::function<int(int /*idx*/)> size_generator, // returns m_i for idx
    std::function<void(int /*idx*/, int /*m_fixed*/, Eigen::MatrixXcd&, Eigen::MatrixXcd&)> matrix_generator_padded,
    std::vector<Eigen::MatrixXcd>& X_vec,
    int bucket = 0 // 0 = exact sizes, else bucket to multiple (8/16/32)
){
    X_vec.resize(total_problems);

    auto round_up = [](int x, int a){ return ((x + a - 1) / a) * a; };

    // 1) compute sizes and group indices
    std::unordered_map<int, std::vector<int>> groups;
    groups.reserve(64);

    std::vector<int> true_sizes(total_problems, 0);
    for (int i = 0; i < total_problems; ++i) {
        int m_i = size_generator(i);
        true_sizes[i] = m_i;
        int key = (bucket > 0) ? round_up(m_i, bucket) : m_i;
        groups[key].push_back(i);
    }

    // 2) solve each group (fixed m per group)
    for (auto& kv : groups) {
        const int m_fixed = kv.first;
        const std::vector<int>& idxs = kv.second;

        cusolverBatchedQR_pipeline_fixed_m(
            idxs, batch_streams, m_fixed, nrhs,
            [&](int idx, int m, Eigen::MatrixXcd& A, Eigen::MatrixXcd& B){
                matrix_generator_padded(idx, m, A, B); // must fill as m x m and m x nrhs (pad if needed)
            },
            X_vec
        );

        // If you bucketed and padded, you can optionally shrink outputs back to true size
        if (bucket > 0) {
            for (int idx : idxs) {
                int m_true = true_sizes[idx];
                if (m_true != m_fixed) {
                    X_vec[idx] = X_vec[idx].topRows(m_true); // keep only physical rows
                }
            }
        }
    }
}

void cusolverBatchedQR_pipeline_variable_inverse(
    int total_problems,
    int batch_streams,
    std::function<int(int)> size_generator,
    std::function<void(int, int, Eigen::MatrixXcd&, Eigen::MatrixXcd&)> matrix_generator_padded,
    std::vector<Eigen::MatrixXcd>& X_vec,
    int bucket = 0
){
    X_vec.resize(total_problems);

    auto round_up = [](int x, int a){ return ((x + a - 1) / a) * a; };

    std::unordered_map<int, std::vector<int>> groups;
    std::vector<int> true_sizes(total_problems, 0);

    for (int i = 0; i < total_problems; ++i) {
        int m_i = size_generator(i);
        true_sizes[i] = m_i;
        int key = (bucket > 0) ? round_up(m_i, bucket) : m_i;
        groups[key].push_back(i);
    }

    for (auto& kv : groups) {
        int m_fixed = kv.first;
        auto& idxs = kv.second;

        // inverse needs B = I => nrhs = m_fixed
        cusolverBatchedQR_pipeline_fixed_m(
            idxs, batch_streams, m_fixed, /*nrhs=*/m_fixed,
            matrix_generator_padded,
            X_vec
        );

        // if padded, shrink back to true (optional)
        if (bucket > 0) {
            for (int idx : idxs) {
                int mt = true_sizes[idx];
                if (mt != m_fixed) {
                    X_vec[idx] = X_vec[idx].block(0, 0, mt, mt);
                }
            }
        }
    }
}