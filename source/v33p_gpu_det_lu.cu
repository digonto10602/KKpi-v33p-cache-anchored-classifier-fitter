#include "v33p_gpu_det_lu.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <complex>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace {

void set_error(char* out, std::size_t cap, const std::string& message) {
    if(!out || cap == 0) return;
    std::strncpy(out, message.c_str(), cap - 1);
    out[cap - 1] = '\0';
}

std::string cuda_error(cudaError_t e, const char* where) {
    std::ostringstream os;
    os << where << ": " << cudaGetErrorString(e);
    return os.str();
}

std::string cublas_error(cublasStatus_t e, const char* where) {
    std::ostringstream os;
    os << where << ": cublas status " << static_cast<int>(e);
    return os.str();
}

bool cuda_ok(cudaError_t e, char* out, std::size_t cap, const char* where) {
    if(e == cudaSuccess) return true;
    set_error(out, cap, cuda_error(e, where));
    return false;
}

bool cublas_ok(cublasStatus_t e, char* out, std::size_t cap, const char* where) {
    if(e == CUBLAS_STATUS_SUCCESS) return true;
    set_error(out, cap, cublas_error(e, where));
    return false;
}

} // namespace

int v33p_gpu_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0 ? 1 : 0;
}

int v33p_gpu_batched_determinants(int n,
                                  int batch,
                                  const V33pComplex* host_matrices,
                                  V33pComplex* host_determinants,
                                  std::uint64_t* device_bytes,
                                  char* error_message,
                                  std::size_t error_capacity) {
    if(device_bytes) *device_bytes = 0;
    if(!host_matrices || !host_determinants || n <= 0 || batch <= 0) {
        set_error(error_message, error_capacity, "invalid determinant batch arguments");
        return 0;
    }

    int device_count = 0;
    if(!cuda_ok(cudaGetDeviceCount(&device_count), error_message, error_capacity, "cudaGetDeviceCount") || device_count <= 0) {
        if(device_count <= 0) set_error(error_message, error_capacity, "no CUDA device available");
        return 0;
    }

    cublasHandle_t handle = nullptr;
    cuDoubleComplex* d_data = nullptr;
    cuDoubleComplex** d_ptrs = nullptr;
    int* d_pivots = nullptr;
    int* d_info = nullptr;
    std::vector<cuDoubleComplex> h_data(static_cast<std::size_t>(n) * n * batch);
    std::vector<cuDoubleComplex*> h_ptrs(static_cast<std::size_t>(batch));
    std::vector<int> h_pivots(static_cast<std::size_t>(n) * batch);
    std::vector<int> h_info(static_cast<std::size_t>(batch));

    const std::size_t matrix_elems = static_cast<std::size_t>(n) * n;
    for(int b = 0; b < batch; ++b) {
        h_ptrs[static_cast<std::size_t>(b)] = nullptr;
        for(std::size_t k = 0; k < matrix_elems; ++k) {
            const auto& z = host_matrices[static_cast<std::size_t>(b) * matrix_elems + k];
            h_data[static_cast<std::size_t>(b) * matrix_elems + k] = make_cuDoubleComplex(z.real, z.imag);
        }
    }

    const std::uint64_t bytes = static_cast<std::uint64_t>(h_data.size() * sizeof(cuDoubleComplex)
        + h_ptrs.size() * sizeof(cuDoubleComplex*)
        + h_pivots.size() * sizeof(int)
        + h_info.size() * sizeof(int));
    if(device_bytes) *device_bytes = bytes;

    auto cleanup = [&] {
        if(d_info) cudaFree(d_info);
        if(d_pivots) cudaFree(d_pivots);
        if(d_ptrs) cudaFree(d_ptrs);
        if(d_data) cudaFree(d_data);
        if(handle) cublasDestroy(handle);
    };

    if(!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&d_data), h_data.size() * sizeof(cuDoubleComplex)), error_message, error_capacity, "cudaMalloc matrix")) { cleanup(); return 0; }
    if(!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&d_ptrs), h_ptrs.size() * sizeof(cuDoubleComplex*)), error_message, error_capacity, "cudaMalloc pointers")) { cleanup(); return 0; }
    if(!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&d_pivots), h_pivots.size() * sizeof(int)), error_message, error_capacity, "cudaMalloc pivots")) { cleanup(); return 0; }
    if(!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&d_info), h_info.size() * sizeof(int)), error_message, error_capacity, "cudaMalloc info")) { cleanup(); return 0; }
    if(!cublas_ok(cublasCreate(&handle), error_message, error_capacity, "cublasCreate")) { cleanup(); return 0; }

    for(int b = 0; b < batch; ++b) h_ptrs[static_cast<std::size_t>(b)] = d_data + static_cast<std::size_t>(b) * matrix_elems;
    if(!cuda_ok(cudaMemcpy(d_data, h_data.data(), h_data.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), error_message, error_capacity, "cudaMemcpy matrices")) { cleanup(); return 0; }
    if(!cuda_ok(cudaMemcpy(d_ptrs, h_ptrs.data(), h_ptrs.size() * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice), error_message, error_capacity, "cudaMemcpy pointers")) { cleanup(); return 0; }

    if(!cublas_ok(cublasZgetrfBatched(handle, n, d_ptrs, n, d_pivots, d_info, batch), error_message, error_capacity, "cublasZgetrfBatched")) { cleanup(); return 0; }
    if(!cuda_ok(cudaDeviceSynchronize(), error_message, error_capacity, "cudaDeviceSynchronize")) { cleanup(); return 0; }
    if(!cuda_ok(cudaMemcpy(h_data.data(), d_data, h_data.size() * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost), error_message, error_capacity, "cudaMemcpy LU")) { cleanup(); return 0; }
    if(!cuda_ok(cudaMemcpy(h_pivots.data(), d_pivots, h_pivots.size() * sizeof(int), cudaMemcpyDeviceToHost), error_message, error_capacity, "cudaMemcpy pivots")) { cleanup(); return 0; }
    if(!cuda_ok(cudaMemcpy(h_info.data(), d_info, h_info.size() * sizeof(int), cudaMemcpyDeviceToHost), error_message, error_capacity, "cudaMemcpy info")) { cleanup(); return 0; }

    for(int b = 0; b < batch; ++b) {
        if(h_info[static_cast<std::size_t>(b)] != 0) {
            std::ostringstream os;
            os << "LU factorization reported info=" << h_info[static_cast<std::size_t>(b)] << " for batch " << b;
            set_error(error_message, error_capacity, os.str());
            cleanup();
            return 0;
        }
        cuDoubleComplex det = make_cuDoubleComplex(1.0, 0.0);
        for(int i = 0; i < n; ++i) {
            const auto& u = h_data[static_cast<std::size_t>(b) * matrix_elems + static_cast<std::size_t>(i) * n + i];
            det = cuCmul(det, u);
            if(h_pivots[static_cast<std::size_t>(b) * n + i] != i + 1) det = make_cuDoubleComplex(-cuCreal(det), -cuCimag(det));
        }
        host_determinants[b] = {cuCreal(det), cuCimag(det)};
    }
    cleanup();
    return 1;
}
