#pragma once

#include <Eigen/Dense>

#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <istream>
#include <stdexcept>
#include <string>

namespace gpu_cache_reader_convention {

using comp = std::complex<double>;

enum class GpuComplexReadConvention {
    RealImagSwappedV33dV32zu,
};

enum class GpuCacheFileFormat {
    CombinedRawF3invVselGpu,
    ComponentMatrixRecord,
};

inline const char* name(GpuComplexReadConvention) {
    return "variant_04_real_imag_swapped";
}

inline const char* name(GpuCacheFileFormat fmt) {
    switch(fmt) {
        case GpuCacheFileFormat::CombinedRawF3invVselGpu: return "combined_raw_F3inv_Vsel_gpu";
        case GpuCacheFileFormat::ComponentMatrixRecord: return "component_matrix_record";
    }
    return "unknown";
}

inline constexpr const char* kCombinedRawGpuCacheHeader = "KKPI_F3INV_VSEL_GPU_V32ZT_EXACT_V32Y 1";
inline constexpr const char* kComponentMatrixCacheHeader = "KKPI_COMPONENT_CACHE_V001";
inline constexpr std::size_t kComponentMatrixCacheHeaderLen = sizeof("KKPI_COMPONENT_CACHE_V001") - 1;

struct CombinedRawGpuCacheRowHeader {
    std::int32_t grid_i = 0;
    std::int32_t dim1 = 0;
    std::int32_t dim2 = 0;
    std::int32_t n = 0;
    std::int32_t vdim = 0;
    double Ecm = 0.0;
    double En = 0.0;
};

struct ComponentMatrixCacheHeader {
    std::string object_name;
    std::string created_by;
    double Lbyas = 0.0;
    double xi = 0.0;
    std::string irrep;
    std::int32_t coarseN = 0;
    std::int32_t row_count = 0;
};

struct ComponentMatrixRecord {
    std::int32_t row = 0;
    double Ecm = 0.0;
    Eigen::MatrixXcd matrix;
};

inline std::uint64_t read_u64(std::istream& is) {
    std::uint64_t v{};
    is.read(reinterpret_cast<char*>(&v), sizeof(v));
    if(!is) throw std::runtime_error("truncated GPU cache while reading u64");
    return v;
}

inline std::int32_t read_i32(std::istream& is) {
    std::int32_t v{};
    is.read(reinterpret_cast<char*>(&v), sizeof(v));
    if(!is) throw std::runtime_error("truncated GPU cache while reading i32");
    return v;
}

inline double read_d(std::istream& is) {
    double v{};
    is.read(reinterpret_cast<char*>(&v), sizeof(v));
    if(!is) throw std::runtime_error("truncated GPU cache while reading double");
    return v;
}

inline std::string read_string(std::istream& is) {
    const std::uint32_t n = static_cast<std::uint32_t>(read_i32(is));
    std::string s(n, '\0');
    if(n) {
        is.read(s.data(), static_cast<std::streamsize>(n));
        if(!is) throw std::runtime_error("truncated GPU cache while reading string");
    }
    return s;
}

inline std::string read_header_line(std::istream& is) {
    std::string header;
    std::getline(is, header);
    if(!is && !is.eof()) throw std::runtime_error("failed to read GPU cache header line");
    return header;
}

inline GpuCacheFileFormat format_from_header(const std::string& header) {
    if(header == kCombinedRawGpuCacheHeader) return GpuCacheFileFormat::CombinedRawF3invVselGpu;
    if(header == kComponentMatrixCacheHeader) return GpuCacheFileFormat::ComponentMatrixRecord;
    throw std::runtime_error("unrecognized GPU cache header: " + header);
}

inline GpuCacheFileFormat require_gpu_cache_format_header(std::istream& is, GpuCacheFileFormat expected, const std::string& cache_path = {}) {
    const std::string header = read_header_line(is);
    const GpuCacheFileFormat got = format_from_header(header);
    if(got != expected) {
        const std::string where = cache_path.empty() ? std::string() : (" in " + cache_path);
        throw std::runtime_error(std::string("unexpected GPU cache format") + where + ": expected " + name(expected) + " got " + name(got));
    }
    return got;
}

inline ComponentMatrixCacheHeader read_component_cache_header(std::istream& is, const std::string& cache_path = {}, const std::string* expected_object = nullptr) {
    char header[kComponentMatrixCacheHeaderLen] = {};
    is.read(header, static_cast<std::streamsize>(kComponentMatrixCacheHeaderLen));
    if(!is) throw std::runtime_error("truncated component cache header" + (cache_path.empty() ? std::string() : (" in " + cache_path)));
    if(std::string(header, kComponentMatrixCacheHeaderLen) != kComponentMatrixCacheHeader) {
        throw std::runtime_error("unexpected component cache header" + (cache_path.empty() ? std::string() : (" in " + cache_path)));
    }
    ComponentMatrixCacheHeader h;
    h.object_name = read_string(is);
    h.created_by = read_string(is);
    h.Lbyas = read_d(is);
    h.xi = read_d(is);
    h.irrep = read_string(is);
    h.coarseN = read_i32(is);
    h.row_count = read_i32(is);
    if(expected_object && h.object_name != *expected_object) {
        throw std::runtime_error("unexpected component cache object in " + cache_path + ": expected " + *expected_object + " got " + h.object_name);
    }
    return h;
}

inline CombinedRawGpuCacheRowHeader read_combined_raw_cache_row_header(std::istream& is, const std::string& cache_path = {}) {
    require_gpu_cache_format_header(is, GpuCacheFileFormat::CombinedRawF3invVselGpu, cache_path);
    const std::uint64_t magic = read_u64(is);
    const std::uint64_t expected_magic = 0x5653325a4f524543ULL;
    if(magic != expected_magic) {
        throw std::runtime_error("bad combined raw GPU cache record magic" + (cache_path.empty() ? std::string() : (" in " + cache_path)));
    }
    CombinedRawGpuCacheRowHeader h;
    h.grid_i = read_i32(is);
    h.dim1 = read_i32(is);
    h.dim2 = read_i32(is);
    h.n = read_i32(is);
    h.vdim = read_i32(is);
    h.Ecm = read_d(is);
    h.En = read_d(is);
    return h;
}

inline ComponentMatrixRecord read_component_matrix_record(std::istream& is) {
    ComponentMatrixRecord r;
    r.row = read_i32(is);
    r.Ecm = read_d(is);
    const std::int32_t rows = read_i32(is);
    const std::int32_t cols = read_i32(is);
    if(rows < 0 || cols < 0) throw std::runtime_error("negative component cache matrix dimensions");
    r.matrix.resize(rows, cols);
    for(int rr = 0; rr < rows; ++rr) {
        for(int cc = 0; cc < cols; ++cc) {
            const double re = read_d(is);
            const double im = read_d(is);
            r.matrix(rr, cc) = comp(re, im);
        }
    }
    return r;
}

inline comp read_complex(std::istream& is, GpuComplexReadConvention = GpuComplexReadConvention::RealImagSwappedV33dV32zu) {
    double a = 0.0;
    double b = 0.0;
    is.read(reinterpret_cast<char*>(&a), sizeof(a));
    is.read(reinterpret_cast<char*>(&b), sizeof(b));
    if(!is) throw std::runtime_error("truncated GPU cache while reading complex entry");
    return comp(b, a);
}

inline Eigen::MatrixXcd read_matrix_col_major(std::istream& is, int rows, int cols, GpuComplexReadConvention conv = GpuComplexReadConvention::RealImagSwappedV33dV32zu) {
    if(rows < 0 || cols < 0) throw std::runtime_error("negative GPU cache matrix dimensions");
    Eigen::MatrixXcd M(rows, cols);
    for(int c = 0; c < cols; ++c) {
        for(int r = 0; r < rows; ++r) {
            M(r, c) = read_complex(is, conv);
        }
    }
    return M;
}

} // namespace gpu_cache_reader_convention
