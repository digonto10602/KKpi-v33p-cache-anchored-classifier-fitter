#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <Eigen/Dense>



inline void checkCudaError(cudaError_t status, const char* msg) {
    if (status != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " - " << cudaGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

inline void checkCusolverError(cusolverStatus_t status, const char* msg) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cuSOLVER error: " << msg << " - code " << status << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

inline void checkCublasError(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS error: " << msg << " - code " << status << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void cusolverComplex_mat(
    Eigen::MatrixXcd &A1,
    Eigen::MatrixXcd &B1,
    Eigen::MatrixXcd &X1,
    int mat_length,
    int mat_width)
{
    cusolverDnHandle_t cusolverH = nullptr;
    cublasHandle_t cublasH = nullptr;

    const int m = mat_length;
    const int lda = m;
    const int ldb = m;
    const int nrhs = mat_width;

    int somenum = 0;
    cuDoubleComplex* A = new cuDoubleComplex[lda * m];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            A[somenum++] = make_cuDoubleComplex(A1(j, i).real(), A1(j, i).imag());
        }
    }
    somenum = 0;

    cuDoubleComplex* B = new cuDoubleComplex[ldb * nrhs];
    for (int i = 0; i < nrhs; i++) {
        for (int j = 0; j < ldb; j++) {
            B[somenum++] = make_cuDoubleComplex(B1(j, i).real(), B1(j, i).imag());
        }
    }
    somenum = 0;

    cuDoubleComplex* XC = new cuDoubleComplex[ldb * nrhs];

    cuDoubleComplex *d_A = nullptr;
    cuDoubleComplex *d_tau = nullptr;
    cuDoubleComplex *d_B = nullptr;
    int *devInfo = nullptr;
    cuDoubleComplex *d_work = nullptr;
    int lwork = 0;
    int info_gpu = 0;

    const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);

    // Create handles
    cusolverStatus_t cusolver_status = cusolverDnCreate(&cusolverH);
    checkCusolverError(cusolver_status, "cusolverDnCreate");

    cublasStatus_t cublas_status = cublasCreate(&cublasH);
    checkCublasError(cublas_status, "cublasCreate");

    // Allocate device memory
    checkCudaError(cudaMalloc((void**)&d_A, sizeof(cuDoubleComplex) * lda * m), "cudaMalloc d_A");
    checkCudaError(cudaMalloc((void**)&d_tau, sizeof(cuDoubleComplex) * m), "cudaMalloc d_tau");
    checkCudaError(cudaMalloc((void**)&d_B, sizeof(cuDoubleComplex) * ldb * nrhs), "cudaMalloc d_B");
    checkCudaError(cudaMalloc((void**)&devInfo, sizeof(int)), "cudaMalloc devInfo");

    // Copy data to device
    checkCudaError(cudaMemcpy(d_A, A, sizeof(cuDoubleComplex) * lda * m, cudaMemcpyHostToDevice), "cudaMemcpy d_A");
    checkCudaError(cudaMemcpy(d_B, B, sizeof(cuDoubleComplex) * ldb * nrhs, cudaMemcpyHostToDevice), "cudaMemcpy d_B");

    // Query workspace size for geqrf
    cusolver_status = cusolverDnZgeqrf_bufferSize(cusolverH, m, m, d_A, lda, &lwork);
    checkCusolverError(cusolver_status, "cusolverDnZgeqrf_bufferSize");
    ////std::cout << "Workspace size for geqrf: " << lwork << std::endl;

    // Allocate workspace
    checkCudaError(cudaMalloc((void**)&d_work, sizeof(cuDoubleComplex) * lwork), "cudaMalloc d_work");

    // QR factorization geqrf
    cusolver_status = cusolverDnZgeqrf(cusolverH, m, m, d_A, lda, d_tau, d_work, lwork, devInfo);
    checkCusolverError(cusolver_status, "cusolverDnZgeqrf");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after geqrf");

    // Check info_gpu
    checkCudaError(cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy info_gpu after geqrf");
    if (info_gpu != 0) {
        std::cerr << "QR factorization failed with info = " << info_gpu << std::endl;
        std::exit(EXIT_FAILURE);
    }
    ////std::cout << "QR factorization successful" << std::endl;

    // Query workspace for unmqr
    cusolver_status = cusolverDnZunmqr_bufferSize(
        cusolverH,
        CUBLAS_SIDE_LEFT,
        CUBLAS_OP_C,
        m,
        nrhs,
        m,
        d_A,
        lda,
        d_tau,
        d_B,
        ldb,
        &lwork);
    checkCusolverError(cusolver_status, "cusolverDnZunmqr_bufferSize");
    ////std::cout << "Workspace size for unmqr: " << lwork << std::endl;

    // Re-allocate workspace if needed
    checkCudaError(cudaFree(d_work), "cudaFree old d_work");
    checkCudaError(cudaMalloc((void**)&d_work, sizeof(cuDoubleComplex) * lwork), "cudaMalloc d_work (unmqr)");

    // Compute Q^T * B
    cusolver_status = cusolverDnZunmqr(
        cusolverH,
        CUBLAS_SIDE_LEFT,
        CUBLAS_OP_C,
        m,
        nrhs,
        m,
        d_A,
        lda,
        d_tau,
        d_B,
        ldb,
        d_work,
        lwork,
        devInfo);
    checkCusolverError(cusolver_status, "cusolverDnZunmqr");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after unmqr");

    checkCudaError(cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy info_gpu after unmqr");
    if (info_gpu != 0) {
        std::cerr << "Q^T*B computation failed with info = " << info_gpu << std::endl;
        std::exit(EXIT_FAILURE);
    }
    ////std::cout << "Q^T*B computation successful" << std::endl;

    // Solve R \ (Q^T * B) via triangular solver
    cublas_status = cublasZtrsm(
        cublasH,
        CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT,
        m,
        nrhs,
        &one,
        d_A,
        lda,
        d_B,
        ldb);
    checkCublasError(cublas_status, "cublasZtrsm");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after cublasZtrsm");

    // Copy solution back to host
    checkCudaError(cudaMemcpy(XC, d_B, sizeof(cuDoubleComplex) * ldb * nrhs, cudaMemcpyDeviceToHost), "cudaMemcpy solution");

    // Convert result to Eigen matrix
    std::complex<double> ii = {0.0, 1.0};
    somenum = 0;
    for (int i = 0; i < nrhs; i++) {
        for (int j = 0; j < ldb; j++) {
            X1(j, i) = std::complex<double>(cuCreal(XC[somenum]), cuCimag(XC[somenum]));
            somenum++;
        }
    }

    // Free resources
    cudaFree(d_A);
    cudaFree(d_tau);
    cudaFree(d_B);
    cudaFree(devInfo);
    cudaFree(d_work);

    cublasDestroy(cublasH);
    cusolverDnDestroy(cusolverH);

    //cudaDeviceReset(); // Comment this out if running multiple times

    delete[] A;
    delete[] B;
    delete[] XC;
}


void cusolverBatchedQR_withStreams(
    std::vector<Eigen::MatrixXcd> &A_vec,
    std::vector<Eigen::MatrixXcd> &B_vec,
    std::vector<Eigen::MatrixXcd> &X_vec,
    int m,
    int nrhs)
{
    int batchSize = static_cast<int>(A_vec.size());
    
    //std::cout << "Starting batched QR solve for " << batchSize << " systems of size " 
    //          << m << "x" << m << std::endl;
    
    // Resize output vector
    X_vec.resize(batchSize);
    for (int b = 0; b < batchSize; b++) {
        X_vec[b].resize(m, nrhs);
    }

    // Create CUDA streams for parallel execution
    std::vector<cudaStream_t> streams(batchSize);
    std::vector<cusolverDnHandle_t> cusolverHandles(batchSize);
    std::vector<cublasHandle_t> cublasHandles(batchSize);

    for (int b = 0; b < batchSize; b++) {
        checkCudaError(cudaStreamCreate(&streams[b]), "cudaStreamCreate");
        checkCusolverError(cusolverDnCreate(&cusolverHandles[b]), "cusolverDnCreate");
        checkCublasError(cublasCreate(&cublasHandles[b]), "cublasCreate");
        
        checkCusolverError(cusolverDnSetStream(cusolverHandles[b], streams[b]), "cusolverDnSetStream");
        checkCublasError(cublasSetStream(cublasHandles[b], streams[b]), "cublasSetStream");
    }

    const int lda = m;
    const int ldb = m;
    
    // Allocate device memory for each batch element
    std::vector<cuDoubleComplex*> d_A_vec(batchSize, nullptr);
    std::vector<cuDoubleComplex*> d_B_vec(batchSize, nullptr);
    std::vector<cuDoubleComplex*> d_tau_vec(batchSize, nullptr);
    std::vector<cuDoubleComplex*> d_work_vec(batchSize, nullptr);
    std::vector<int*> d_info_vec(batchSize, nullptr);
    std::vector<int> lwork_geqrf_vec(batchSize, 0);
    std::vector<int> lwork_unmqr_vec(batchSize, 0);

    const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);

    // Setup and launch solves for each batch
    for (int b = 0; b < batchSize; b++) {
        //std::cout << "Processing batch " << b << std::endl;
        
        // Allocate device memory
        checkCudaError(cudaMalloc(&d_A_vec[b], sizeof(cuDoubleComplex) * lda * m), "cudaMalloc d_A");
        checkCudaError(cudaMalloc(&d_B_vec[b], sizeof(cuDoubleComplex) * ldb * nrhs), "cudaMalloc d_B");
        checkCudaError(cudaMalloc(&d_tau_vec[b], sizeof(cuDoubleComplex) * m), "cudaMalloc d_tau");
        checkCudaError(cudaMalloc(&d_info_vec[b], sizeof(int)), "cudaMalloc d_info");

        // Prepare host data
        std::vector<cuDoubleComplex> h_A(lda * m);
        std::vector<cuDoubleComplex> h_B(ldb * nrhs);

        int idx = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                h_A[idx++] = make_cuDoubleComplex(A_vec[b](j, i).real(), A_vec[b](j, i).imag());
            }
        }

        idx = 0;
        for (int i = 0; i < nrhs; i++) {
            for (int j = 0; j < ldb; j++) {
                h_B[idx++] = make_cuDoubleComplex(B_vec[b](j, i).real(), B_vec[b](j, i).imag());
            }
        }

        // Copy to device asynchronously
        checkCudaError(cudaMemcpyAsync(d_A_vec[b], h_A.data(), sizeof(cuDoubleComplex) * lda * m, 
                                       cudaMemcpyHostToDevice, streams[b]), "cudaMemcpyAsync A");
        checkCudaError(cudaMemcpyAsync(d_B_vec[b], h_B.data(), sizeof(cuDoubleComplex) * ldb * nrhs, 
                                       cudaMemcpyHostToDevice, streams[b]), "cudaMemcpyAsync B");

        // Query workspace size for geqrf
        checkCusolverError(cusolverDnZgeqrf_bufferSize(cusolverHandles[b], m, m, d_A_vec[b], lda, 
                                                       &lwork_geqrf_vec[b]),
                          "cusolverDnZgeqrf_bufferSize");
        
        //std::cout << "  Workspace size for geqrf: " << lwork_geqrf_vec[b] << std::endl;

        // Allocate workspace for geqrf
        checkCudaError(cudaMalloc(&d_work_vec[b], sizeof(cuDoubleComplex) * lwork_geqrf_vec[b]), 
                       "cudaMalloc d_work geqrf");

        // QR factorization
        checkCusolverError(cusolverDnZgeqrf(cusolverHandles[b], m, m, d_A_vec[b], lda, 
                                            d_tau_vec[b], d_work_vec[b], lwork_geqrf_vec[b], 
                                            d_info_vec[b]),
                          "cusolverDnZgeqrf");
        
        // Synchronize and check info
        checkCudaError(cudaStreamSynchronize(streams[b]), "cudaStreamSynchronize after geqrf");
        
        int h_info = 0;
        checkCudaError(cudaMemcpy(&h_info, d_info_vec[b], sizeof(int), cudaMemcpyDeviceToHost), 
                       "cudaMemcpy info after geqrf");
        if (h_info != 0) {
            std::cerr << "Batch " << b << ": QR factorization failed with info = " << h_info << std::endl;
            std::exit(EXIT_FAILURE);
        }
        //std::cout << "  QR factorization successful" << std::endl;

        // Query workspace for unmqr
        checkCusolverError(cusolverDnZunmqr_bufferSize(cusolverHandles[b], CUBLAS_SIDE_LEFT, CUBLAS_OP_C,
                                                       m, nrhs, m, d_A_vec[b], lda, d_tau_vec[b],
                                                       d_B_vec[b], ldb, &lwork_unmqr_vec[b]),
                          "cusolverDnZunmqr_bufferSize");
        
        //std::cout << "  Workspace size for unmqr: " << lwork_unmqr_vec[b] << std::endl;

        // Reallocate workspace if unmqr needs more space
        if (lwork_unmqr_vec[b] > lwork_geqrf_vec[b]) {
            //std::cout << "  Reallocating workspace for unmqr" << std::endl;
            cudaFree(d_work_vec[b]);
            checkCudaError(cudaMalloc(&d_work_vec[b], sizeof(cuDoubleComplex) * lwork_unmqr_vec[b]), 
                           "cudaMalloc d_work unmqr");
        }

        // Compute Q^T * B
        checkCusolverError(cusolverDnZunmqr(cusolverHandles[b], CUBLAS_SIDE_LEFT, CUBLAS_OP_C,
                                            m, nrhs, m, d_A_vec[b], lda, d_tau_vec[b],
                                            d_B_vec[b], ldb, d_work_vec[b], 
                                            std::max(lwork_geqrf_vec[b], lwork_unmqr_vec[b]), 
                                            d_info_vec[b]),
                          "cusolverDnZunmqr");
        
        // Synchronize and check info
        checkCudaError(cudaStreamSynchronize(streams[b]), "cudaStreamSynchronize after unmqr");
        
        checkCudaError(cudaMemcpy(&h_info, d_info_vec[b], sizeof(int), cudaMemcpyDeviceToHost), 
                       "cudaMemcpy info after unmqr");
        if (h_info != 0) {
            std::cerr << "Batch " << b << ": Q^T*B computation failed with info = " << h_info << std::endl;
            std::exit(EXIT_FAILURE);
        }
        //std::cout << "  Q^T*B computation successful" << std::endl;

        // Check for any prior CUDA errors before trsm
        cudaError_t prior_err = cudaGetLastError();
        if (prior_err != cudaSuccess) {
            std::cerr << "Prior CUDA error before trsm: " << cudaGetErrorString(prior_err) << std::endl;
            std::exit(EXIT_FAILURE);
        }

        // Solve R \ (Q^T * B)
        checkCublasError(cublasZtrsm(cublasHandles[b], CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                                     CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, nrhs, &one,
                                     d_A_vec[b], lda, d_B_vec[b], ldb),
                        "cublasZtrsm");
        
        //std::cout << "  Triangular solve successful" << std::endl;
    }

    // Synchronize all streams and copy results back
    //std::cout << "Copying results back to host..." << std::endl;
    for (int b = 0; b < batchSize; b++) {
        checkCudaError(cudaStreamSynchronize(streams[b]), "cudaStreamSynchronize final");

        // Copy solution back
        std::vector<cuDoubleComplex> h_X(ldb * nrhs);
        checkCudaError(cudaMemcpy(h_X.data(), d_B_vec[b], sizeof(cuDoubleComplex) * ldb * nrhs,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy solution");

        // Convert to Eigen
        int idx = 0;
        for (int i = 0; i < nrhs; i++) {
            for (int j = 0; j < ldb; j++) {
                X_vec[b](j, i) = std::complex<double>(cuCreal(h_X[idx]), cuCimag(h_X[idx]));
                idx++;
            }
        }

        // Free device memory
        if (d_A_vec[b]) cudaFree(d_A_vec[b]);
        if (d_B_vec[b]) cudaFree(d_B_vec[b]);
        if (d_tau_vec[b]) cudaFree(d_tau_vec[b]);
        if (d_work_vec[b]) cudaFree(d_work_vec[b]);
        if (d_info_vec[b]) cudaFree(d_info_vec[b]);

        // Destroy handles
        cusolverDnDestroy(cusolverHandles[b]);
        cublasDestroy(cublasHandles[b]);
        cudaStreamDestroy(streams[b]);
    }

    //std::cout << "Batched QR solve completed successfully for " << batchSize << " systems" << std::endl;
}

void cusolverBatchedQR_pipeline(
    int total_problems,           // Total number of problems (e.g., 100)
    int batch_size,               // Number of concurrent CUDA streams (e.g., 10-20)
    int m,                        // Matrix size
    int nrhs,                     // Number of right-hand sides
    std::function<void(int, Eigen::MatrixXcd&, Eigen::MatrixXcd&)> matrix_generator,  // Function to generate A and B
    std::vector<Eigen::MatrixXcd> &X_vec)  // Output solutions
{
    X_vec.resize(total_problems);
    
    //std::cout << "Processing " << total_problems << " systems in batches of " 
    //          << batch_size << std::endl;

    // Create persistent CUDA resources
    std::vector<cudaStream_t> streams(batch_size);
    std::vector<cusolverDnHandle_t> cusolverHandles(batch_size);
    std::vector<cublasHandle_t> cublasHandles(batch_size);

    for (int s = 0; s < batch_size; s++) {
        checkCudaError(cudaStreamCreate(&streams[s]), "cudaStreamCreate");
        checkCusolverError(cusolverDnCreate(&cusolverHandles[s]), "cusolverDnCreate");
        checkCublasError(cublasCreate(&cublasHandles[s]), "cublasCreate");
        
        checkCusolverError(cusolverDnSetStream(cusolverHandles[s], streams[s]), "cusolverDnSetStream");
        checkCublasError(cublasSetStream(cublasHandles[s], streams[s]), "cublasSetStream");
    }

    const int lda = m;
    const int ldb = m;
    const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);

    // Process in batches
    for (int batch_start = 0; batch_start < total_problems; batch_start += batch_size) {
        int current_batch_size = std::min(batch_size, total_problems - batch_start);
        
        //std::cout << "\nProcessing batch starting at " << batch_start 
        //          << " (size: " << current_batch_size << ")" << std::endl;

        // Allocate temporary storage for this batch
        std::vector<Eigen::MatrixXcd> A_batch(current_batch_size);
        std::vector<Eigen::MatrixXcd> B_batch(current_batch_size);
        std::vector<Eigen::MatrixXcd> X_batch(current_batch_size);

        // Generate matrices in parallel with OpenMP
        //std::cout << "Generating matrices with OpenMP..." << std::endl;
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < current_batch_size; i++) {
            int global_idx = batch_start + i;
            A_batch[i].resize(m, m);
            B_batch[i].resize(m, nrhs);
            
            // Call user-provided matrix generator
            matrix_generator(global_idx, A_batch[i], B_batch[i]);
            
            #pragma omp critical
            {
                //std::cout << "  Generated matrix " << global_idx << " (thread " 
                //          << omp_get_thread_num() << ")" << std::endl;
            }
        }

        //std::cout << "Solving on GPU..." << std::endl;
        
        // Device memory for this batch
        std::vector<cuDoubleComplex*> d_A_vec(current_batch_size, nullptr);
        std::vector<cuDoubleComplex*> d_B_vec(current_batch_size, nullptr);
        std::vector<cuDoubleComplex*> d_tau_vec(current_batch_size, nullptr);
        std::vector<cuDoubleComplex*> d_work_vec(current_batch_size, nullptr);
        std::vector<int*> d_info_vec(current_batch_size, nullptr);
        std::vector<int> lwork_geqrf(current_batch_size, 0);
        std::vector<int> lwork_unmqr(current_batch_size, 0);

        // Launch GPU work for this batch
        for (int b = 0; b < current_batch_size; b++) {
            int stream_idx = b % batch_size;  // Reuse streams if needed
            
            // Allocate device memory
            checkCudaError(cudaMalloc(&d_A_vec[b], sizeof(cuDoubleComplex) * lda * m), "cudaMalloc d_A");
            checkCudaError(cudaMalloc(&d_B_vec[b], sizeof(cuDoubleComplex) * ldb * nrhs), "cudaMalloc d_B");
            checkCudaError(cudaMalloc(&d_tau_vec[b], sizeof(cuDoubleComplex) * m), "cudaMalloc d_tau");
            checkCudaError(cudaMalloc(&d_info_vec[b], sizeof(int)), "cudaMalloc d_info");

            // Prepare and copy data
            std::vector<cuDoubleComplex> h_A(lda * m);
            std::vector<cuDoubleComplex> h_B(ldb * nrhs);

            int idx = 0;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    h_A[idx++] = make_cuDoubleComplex(A_batch[b](j, i).real(), 
                                                      A_batch[b](j, i).imag());
                }
            }

            idx = 0;
            for (int i = 0; i < nrhs; i++) {
                for (int j = 0; j < ldb; j++) {
                    h_B[idx++] = make_cuDoubleComplex(B_batch[b](j, i).real(), 
                                                      B_batch[b](j, i).imag());
                }
            }

            checkCudaError(cudaMemcpyAsync(d_A_vec[b], h_A.data(), 
                                          sizeof(cuDoubleComplex) * lda * m, 
                                          cudaMemcpyHostToDevice, streams[stream_idx]), 
                          "cudaMemcpyAsync A");
            checkCudaError(cudaMemcpyAsync(d_B_vec[b], h_B.data(), 
                                          sizeof(cuDoubleComplex) * ldb * nrhs, 
                                          cudaMemcpyHostToDevice, streams[stream_idx]), 
                          "cudaMemcpyAsync B");

            // Query and allocate workspace for geqrf
            checkCusolverError(cusolverDnZgeqrf_bufferSize(cusolverHandles[stream_idx], 
                                                          m, m, d_A_vec[b], lda, 
                                                          &lwork_geqrf[b]),
                              "cusolverDnZgeqrf_bufferSize");
            checkCudaError(cudaMalloc(&d_work_vec[b], sizeof(cuDoubleComplex) * lwork_geqrf[b]), 
                          "cudaMalloc d_work");

            // QR factorization
            checkCusolverError(cusolverDnZgeqrf(cusolverHandles[stream_idx], m, m, 
                                               d_A_vec[b], lda, d_tau_vec[b], 
                                               d_work_vec[b], lwork_geqrf[b], 
                                               d_info_vec[b]),
                              "cusolverDnZgeqrf");

            // Query workspace for unmqr and reallocate if needed
            checkCusolverError(cusolverDnZunmqr_bufferSize(cusolverHandles[stream_idx], 
                                                          CUBLAS_SIDE_LEFT, CUBLAS_OP_C,
                                                          m, nrhs, m, d_A_vec[b], lda, 
                                                          d_tau_vec[b], d_B_vec[b], ldb, 
                                                          &lwork_unmqr[b]),
                              "cusolverDnZunmqr_bufferSize");

            if (lwork_unmqr[b] > lwork_geqrf[b]) {
                cudaFree(d_work_vec[b]);
                checkCudaError(cudaMalloc(&d_work_vec[b], 
                                         sizeof(cuDoubleComplex) * lwork_unmqr[b]), 
                              "cudaMalloc d_work unmqr");
            }

            // Compute Q^T * B
            checkCusolverError(cusolverDnZunmqr(cusolverHandles[stream_idx], 
                                               CUBLAS_SIDE_LEFT, CUBLAS_OP_C,
                                               m, nrhs, m, d_A_vec[b], lda, 
                                               d_tau_vec[b], d_B_vec[b], ldb, 
                                               d_work_vec[b], 
                                               std::max(lwork_geqrf[b], lwork_unmqr[b]), 
                                               d_info_vec[b]),
                              "cusolverDnZunmqr");

            // Triangular solve
            checkCublasError(cublasZtrsm(cublasHandles[stream_idx], 
                                        CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                                        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 
                                        m, nrhs, &one, d_A_vec[b], lda, 
                                        d_B_vec[b], ldb),
                            "cublasZtrsm");
        }

        // Synchronize all streams and copy results
        //std::cout << "Waiting for GPU to finish..." << std::endl;
        for (int b = 0; b < current_batch_size; b++) {
            int stream_idx = b % batch_size;
            checkCudaError(cudaStreamSynchronize(streams[stream_idx]), 
                          "cudaStreamSynchronize");

            // Copy solution back
            std::vector<cuDoubleComplex> h_X(ldb * nrhs);
            checkCudaError(cudaMemcpy(h_X.data(), d_B_vec[b], 
                                     sizeof(cuDoubleComplex) * ldb * nrhs,
                                     cudaMemcpyDeviceToHost), 
                          "cudaMemcpy solution");

            // Convert to Eigen and store in final output
            X_batch[b].resize(m, nrhs);
            int idx = 0;
            for (int i = 0; i < nrhs; i++) {
                for (int j = 0; j < ldb; j++) {
                    X_batch[b](j, i) = std::complex<double>(cuCreal(h_X[idx]), 
                                                            cuCimag(h_X[idx]));
                    idx++;
                }
            }
            
            int global_idx = batch_start + b;
            X_vec[global_idx] = X_batch[b];

            // Free device memory immediately
            cudaFree(d_A_vec[b]);
            cudaFree(d_B_vec[b]);
            cudaFree(d_tau_vec[b]);
            cudaFree(d_work_vec[b]);
            cudaFree(d_info_vec[b]);
        }

        //std::cout << "Batch completed. Memory freed." << std::endl;
        // A_batch, B_batch, X_batch go out of scope and are freed here
    }

    // Cleanup persistent resources
    for (int s = 0; s < batch_size; s++) {
        cusolverDnDestroy(cusolverHandles[s]);
        cublasDestroy(cublasHandles[s]);
        cudaStreamDestroy(streams[s]);
    }

    //std::cout << "\nAll " << total_problems << " systems solved successfully!" << std::endl;
}
