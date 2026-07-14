#include "v32x_fast_backends.hpp"

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <math_constants.h>

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace v32x_fast {

namespace {

#define V32X_CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error in ") + #call + " : " + cudaGetErrorString(err__)); \
    } \
} while(0)

#define V32X_CUBLAS_CHECK(call) do { \
    cublasStatus_t st__ = (call); \
    if (st__ != CUBLAS_STATUS_SUCCESS) { \
        std::ostringstream os__; \
        os__ << "cuBLAS error in " << #call << " status=" << int(st__); \
        throw std::runtime_error(os__.str()); \
    } \
} while(0)

template <typename T>
struct DeviceBuffer {
    T* p = nullptr;
    std::size_t n = 0;
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t n_) { alloc(n_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; o.n = 0; }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            free();
            p = o.p;
            n = o.n;
            o.p = nullptr;
            o.n = 0;
        }
        return *this;
    }
    ~DeviceBuffer() { free(); }
    void alloc(std::size_t n_) {
        free();
        n = n_;
        if (n) V32X_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)));
    }
    void free() {
        if (p) {
            cudaFree(p);
            p = nullptr;
            n = 0;
        }
    }
};

__global__ void assemble_qc_kernel(
    cuDoubleComplex* qc,
    const cuDoubleComplex* f3inv,
    const cuDoubleComplex* m0,
    const cuDoubleComplex* m1,
    const cuDoubleComplex* mB,
    const cuDoubleComplex* mE,
    double k0,
    double k1,
    double kB,
    double kE,
    int vdim,
    int batch_count)
{
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long total = static_cast<long long>(batch_count) * vdim * vdim;
    if (tid >= total) return;
    const int mat = static_cast<int>(tid / (static_cast<long long>(vdim) * vdim));
    const int loc = static_cast<int>(tid - static_cast<long long>(mat) * vdim * vdim);
    const int row = loc % vdim;
    const int col = loc / vdim;
    const std::size_t base = static_cast<std::size_t>(mat) * static_cast<std::size_t>(vdim) * static_cast<std::size_t>(vdim);
    const std::size_t off = base + static_cast<std::size_t>(col) * static_cast<std::size_t>(vdim) + static_cast<std::size_t>(row);
    cuDoubleComplex v = f3inv[off];
    v = cuCadd(v, cuCmul(make_cuDoubleComplex(k0, 0.0), m0[off]));
    v = cuCadd(v, cuCmul(make_cuDoubleComplex(k1, 0.0), m1[off]));
    v = cuCadd(v, cuCmul(make_cuDoubleComplex(kB, 0.0), mB[off]));
    v = cuCadd(v, cuCmul(make_cuDoubleComplex(kE, 0.0), mE[off]));
    qc[off] = v;
}

__global__ void det_real_from_lu_kernel(
    const cuDoubleComplex* lu,
    const int* pivots,
    const int* info,
    int vdim,
    int batch_count,
    double* det_re)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_count) return;
    if (info[b] != 0) {
        det_re[b] = CUDART_NAN;
        return;
    }
    const std::size_t base = static_cast<std::size_t>(b) * static_cast<std::size_t>(vdim) * static_cast<std::size_t>(vdim);
    const cuDoubleComplex* m = lu + base;
    const int* piv = pivots + static_cast<std::size_t>(b) * static_cast<std::size_t>(vdim);
    cuDoubleComplex det = make_cuDoubleComplex(1.0, 0.0);
    int parity = 0;
    for (int i = 0; i < vdim; ++i) {
        det = cuCmul(det, m[static_cast<std::size_t>(i) * static_cast<std::size_t>(vdim) + static_cast<std::size_t>(i)]);
        if (piv[i] != i + 1) parity ^= 1;
    }
    if (parity) det = make_cuDoubleComplex(-cuCreal(det), -cuCimag(det));
    det_re[b] = cuCreal(det);
}

static std::string device_name_or_unknown(int device_id)
{
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device_id) == cudaSuccess) {
        return prop.name;
    }
    return "unknown";
}

struct GroupBuffers {
    int vdim = 0;
    std::size_t row_count = 0;
    std::vector<GpuDetRowResult> meta;
    DeviceBuffer<cuDoubleComplex> d_f3inv;
    DeviceBuffer<cuDoubleComplex> d_k0;
    DeviceBuffer<cuDoubleComplex> d_k1;
    DeviceBuffer<cuDoubleComplex> d_kB;
    DeviceBuffer<cuDoubleComplex> d_kE;
    DeviceBuffer<cuDoubleComplex> d_qc;
    DeviceBuffer<cuDoubleComplex*> d_ptrs;
    DeviceBuffer<int> d_pivots;
    DeviceBuffer<int> d_info;
    DeviceBuffer<double> d_det_re;
    std::size_t resident_bytes = 0;
};

} // namespace

struct GpuBatchedDetBackend::Impl {
    int device_id = 0;
    bool use_double = true;
    std::string device_name;
    cublasHandle_t handle = nullptr;
    std::vector<GroupBuffers> groups;
    std::size_t total_rows = 0;
    std::size_t resident_bytes = 0;
};

GpuBatchedDetBackend::GpuBatchedDetBackend(const std::vector<GpuDetGroupData>& groups, int device_id, bool use_double)
    : impl_(std::make_unique<Impl>())
{
    if (!use_double) {
        throw std::runtime_error("gpu_batched_det backend requires double precision");
    }
    impl_->device_id = device_id;
    impl_->use_double = use_double;

    V32X_CUDA_CHECK(cudaSetDevice(device_id));
    impl_->device_name = device_name_or_unknown(device_id);
    V32X_CUBLAS_CHECK(cublasCreate(&impl_->handle));
    V32X_CUBLAS_CHECK(cublasSetPointerMode(impl_->handle, CUBLAS_POINTER_MODE_HOST));

    impl_->groups.reserve(groups.size());
    for (const auto& g : groups) {
        GroupBuffers gb;
        gb.vdim = g.vdim;
        gb.row_count = g.rows.size();
        gb.meta.reserve(g.rows.size());
        if (gb.row_count == 0 || gb.vdim <= 0) {
            impl_->groups.push_back(std::move(gb));
            continue;
        }
        const std::size_t elems_per = static_cast<std::size_t>(gb.vdim) * static_cast<std::size_t>(gb.vdim);
        const std::size_t total_elems = elems_per * gb.row_count;
        std::vector<cuDoubleComplex> h_f3inv(total_elems);
        std::vector<cuDoubleComplex> h_k0(total_elems);
        std::vector<cuDoubleComplex> h_k1(total_elems);
        std::vector<cuDoubleComplex> h_kB(total_elems);
        std::vector<cuDoubleComplex> h_kE(total_elems);
        gb.meta.reserve(g.rows.size());
        for (std::size_t r = 0; r < g.rows.size(); ++r) {
            const auto& row = g.rows[r];
            if (row.proj_dim != gb.vdim) {
                throw std::runtime_error("gpu_batched_det backend group dimension mismatch");
            }
            if (row.f3inv_proj.size() != elems_per || row.k0_proj.size() != elems_per || row.k1_proj.size() != elems_per ||
                row.kB_proj.size() != elems_per || row.kE_proj.size() != elems_per) {
                throw std::runtime_error("gpu_batched_det backend row basis size mismatch");
            }
            gb.meta.push_back({row.block_index, row.row_index, row.proj_dim, std::numeric_limits<double>::quiet_NaN(), 0});
            const std::size_t base = r * elems_per;
            for (std::size_t i = 0; i < elems_per; ++i) {
                h_f3inv[base + i] = make_cuDoubleComplex(row.f3inv_proj[i].real(), row.f3inv_proj[i].imag());
                h_k0[base + i] = make_cuDoubleComplex(row.k0_proj[i].real(), row.k0_proj[i].imag());
                h_k1[base + i] = make_cuDoubleComplex(row.k1_proj[i].real(), row.k1_proj[i].imag());
                h_kB[base + i] = make_cuDoubleComplex(row.kB_proj[i].real(), row.kB_proj[i].imag());
                h_kE[base + i] = make_cuDoubleComplex(row.kE_proj[i].real(), row.kE_proj[i].imag());
            }
        }
        gb.d_f3inv.alloc(total_elems);
        gb.d_k0.alloc(total_elems);
        gb.d_k1.alloc(total_elems);
        gb.d_kB.alloc(total_elems);
        gb.d_kE.alloc(total_elems);
        gb.d_qc.alloc(total_elems);
        gb.d_pivots.alloc(gb.row_count * static_cast<std::size_t>(gb.vdim));
        gb.d_info.alloc(gb.row_count);
        gb.d_det_re.alloc(gb.row_count);
        gb.d_ptrs.alloc(gb.row_count);
        V32X_CUDA_CHECK(cudaMemcpy(gb.d_f3inv.p, h_f3inv.data(), total_elems * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        V32X_CUDA_CHECK(cudaMemcpy(gb.d_k0.p, h_k0.data(), total_elems * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        V32X_CUDA_CHECK(cudaMemcpy(gb.d_k1.p, h_k1.data(), total_elems * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        V32X_CUDA_CHECK(cudaMemcpy(gb.d_kB.p, h_kB.data(), total_elems * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        V32X_CUDA_CHECK(cudaMemcpy(gb.d_kE.p, h_kE.data(), total_elems * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

        std::vector<cuDoubleComplex*> h_ptrs(gb.row_count);
        for (std::size_t r = 0; r < gb.row_count; ++r) {
            h_ptrs[r] = gb.d_qc.p + r * elems_per;
        }
        V32X_CUDA_CHECK(cudaMemcpy(gb.d_ptrs.p, h_ptrs.data(), gb.row_count * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice));

        gb.resident_bytes = 5 * total_elems * sizeof(cuDoubleComplex)
                          + total_elems * sizeof(cuDoubleComplex)
                          + gb.row_count * sizeof(cuDoubleComplex*)
                          + gb.row_count * static_cast<std::size_t>(gb.vdim) * sizeof(int)
                          + gb.row_count * sizeof(int)
                          + gb.row_count * sizeof(double);
        impl_->resident_bytes += gb.resident_bytes;
        impl_->total_rows += gb.row_count;
        impl_->groups.push_back(std::move(gb));
    }
}

GpuBatchedDetBackend::~GpuBatchedDetBackend() = default;
GpuBatchedDetBackend::GpuBatchedDetBackend(GpuBatchedDetBackend&&) noexcept = default;
GpuBatchedDetBackend& GpuBatchedDetBackend::operator=(GpuBatchedDetBackend&&) noexcept = default;

std::vector<GpuDetGroupResult> GpuBatchedDetBackend::evaluate(const K3Param4& p) const
{
    if (!impl_) return {};
    V32X_CUDA_CHECK(cudaSetDevice(impl_->device_id));

    std::vector<GpuDetGroupResult> out;
    out.reserve(impl_->groups.size());
    for (const auto& gb : impl_->groups) {
        GpuDetGroupResult gr;
        gr.vdim = gb.vdim;
        gr.rows.reserve(gb.row_count);
        if (gb.row_count == 0 || gb.vdim <= 0) {
            out.push_back(std::move(gr));
            continue;
        }

        const int threads = 256;
        const int blocks = static_cast<int>((gb.row_count * static_cast<std::size_t>(gb.vdim) * static_cast<std::size_t>(gb.vdim) + threads - 1) / threads);
        assemble_qc_kernel<<<blocks, threads>>>(
            gb.d_qc.p,
            gb.d_f3inv.p,
            gb.d_k0.p,
            gb.d_k1.p,
            gb.d_kB.p,
            gb.d_kE.p,
            p.K3iso0,
            p.K3iso1,
            p.K3B,
            p.K3E,
            gb.vdim,
            static_cast<int>(gb.row_count));
        V32X_CUDA_CHECK(cudaGetLastError());

        V32X_CUBLAS_CHECK(cublasZgetrfBatched(
            impl_->handle,
            gb.vdim,
            const_cast<cuDoubleComplex**>(gb.d_ptrs.p),
            gb.vdim,
            gb.d_pivots.p,
            gb.d_info.p,
            static_cast<int>(gb.row_count)));
        V32X_CUDA_CHECK(cudaGetLastError());

        const int det_threads = 256;
        const int det_blocks = static_cast<int>((gb.row_count + det_threads - 1) / det_threads);
        det_real_from_lu_kernel<<<det_blocks, det_threads>>>(
            gb.d_qc.p,
            gb.d_pivots.p,
            gb.d_info.p,
            gb.vdim,
            static_cast<int>(gb.row_count),
            gb.d_det_re.p);
        V32X_CUDA_CHECK(cudaGetLastError());
        V32X_CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> h_det(gb.row_count);
        std::vector<int> h_info(gb.row_count);
        V32X_CUDA_CHECK(cudaMemcpy(h_det.data(), gb.d_det_re.p, gb.row_count * sizeof(double), cudaMemcpyDeviceToHost));
        V32X_CUDA_CHECK(cudaMemcpy(h_info.data(), gb.d_info.p, gb.row_count * sizeof(int), cudaMemcpyDeviceToHost));

        for (std::size_t r = 0; r < gb.row_count; ++r) {
            GpuDetRowResult rr = gb.meta[r];
            rr.det_re = h_det[r];
            rr.info = h_info[r];
            gr.rows.push_back(rr);
        }
        out.push_back(std::move(gr));
    }
    return out;
}

std::size_t GpuBatchedDetBackend::resident_bytes() const { return impl_ ? impl_->resident_bytes : 0; }
std::size_t GpuBatchedDetBackend::total_rows() const { return impl_ ? impl_->total_rows : 0; }
std::size_t GpuBatchedDetBackend::group_count() const { return impl_ ? impl_->groups.size() : 0; }
int GpuBatchedDetBackend::device_id() const { return impl_ ? impl_->device_id : -1; }
std::string GpuBatchedDetBackend::device_name() const { return impl_ ? impl_->device_name : std::string(); }

} // namespace v32x_fast
