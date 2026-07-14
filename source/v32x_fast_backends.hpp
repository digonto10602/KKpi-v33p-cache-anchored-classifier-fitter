#ifndef V32X_FAST_BACKENDS_HPP
#define V32X_FAST_BACKENDS_HPP

#include <cstddef>
#include <complex>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace v32x_fast {

using comp = std::complex<double>;

struct K3Param4 {
    double K3iso0 = 0.0;
    double K3iso1 = 0.0;
    double K3B = 0.0;
    double K3E = 0.0;
};

struct GpuDetRowData {
    int block_index = -1;
    int row_index = -1;
    int proj_dim = 0;
    std::vector<comp> f3inv_proj;
    std::vector<comp> k0_proj;
    std::vector<comp> k1_proj;
    std::vector<comp> kB_proj;
    std::vector<comp> kE_proj;
};

struct GpuDetGroupData {
    int vdim = 0;
    std::vector<GpuDetRowData> rows;
};

struct GpuDetRowResult {
    int block_index = -1;
    int row_index = -1;
    int proj_dim = 0;
    double det_re = std::numeric_limits<double>::quiet_NaN();
    int info = -1;
};

struct GpuDetGroupResult {
    int vdim = 0;
    std::vector<GpuDetRowResult> rows;
};

class GpuBatchedDetBackend {
public:
    GpuBatchedDetBackend(const std::vector<GpuDetGroupData>& groups, int device_id, bool use_double);
    ~GpuBatchedDetBackend();

    GpuBatchedDetBackend(const GpuBatchedDetBackend&) = delete;
    GpuBatchedDetBackend& operator=(const GpuBatchedDetBackend&) = delete;

    GpuBatchedDetBackend(GpuBatchedDetBackend&&) noexcept;
    GpuBatchedDetBackend& operator=(GpuBatchedDetBackend&&) noexcept;

    std::vector<GpuDetGroupResult> evaluate(const K3Param4& p) const;

    std::size_t resident_bytes() const;
    std::size_t total_rows() const;
    std::size_t group_count() const;
    int device_id() const;
    std::string device_name() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace v32x_fast

#endif
