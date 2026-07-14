#pragma once

#include "K3df_minuit_fit_v32f_fullF3inv_QCfull_cached_classifier.hpp"

#include <cstdint>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>

namespace v33g_runtime_k3basis {

using namespace k3df_fit_v32f;
namespace fs = std::filesystem;

struct RuntimeCacheMeta {
    std::string version = "v33g_runtime_k3basis_cache";
    double Lbyas = 0.0;
    double xi = 0.0;
    std::string irrep;
    int coarseN = 0;
    double Ecm_min = 0.0;
    double Ecm_max = 0.0;
    std::size_t rows = 0;
    double atmK = 0.0;
    double atmpi = 0.0;
    double eta_1 = 0.0;
    double eta_2 = 0.0;
    double alpha = 0.0;
    double epsilon_h = 0.0;
    double max_shell_num = 0.0;
    double tolerance = 0.0;
    int parity = 0;
    double eig_tol = 0.0;
    double norm_tol = 0.0;
    double proj_tol = 0.0;
    std::vector<int> waves_vec_1;
    std::vector<int> waves_vec_2;
    double scatter1_00 = 0.0;
    double scatter1_10 = 0.0;
    double scatter2_00 = 0.0;
    std::string source_raw_cache;
    std::uint64_t source_raw_cache_file_size = 0;
    double source_raw_cache_mtime = 0.0;
    std::string created_by;
};

inline void write_raw_u64(std::ofstream& os, std::uint64_t v) { os.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
inline void write_raw_i32(std::ofstream& os, std::int32_t v) { os.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
inline void write_raw_d(std::ofstream& os, double v) { os.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
inline std::uint64_t read_raw_u64(std::ifstream& is) { std::uint64_t v{}; is.read(reinterpret_cast<char*>(&v), sizeof(v)); return v; }
inline std::int32_t read_raw_i32(std::ifstream& is) { std::int32_t v{}; is.read(reinterpret_cast<char*>(&v), sizeof(v)); return v; }
inline double read_raw_d(std::ifstream& is) { double v{}; is.read(reinterpret_cast<char*>(&v), sizeof(v)); return v; }

inline void write_compact_matrix(std::ofstream& os, const Eigen::MatrixXcd& M) {
    write_raw_u64(os, static_cast<std::uint64_t>(M.rows()));
    write_raw_u64(os, static_cast<std::uint64_t>(M.cols()));
    for(int c = 0; c < M.cols(); ++c) {
        for(int r = 0; r < M.rows(); ++r) {
            const comp z = M(r, c);
            write_raw_d(os, z.real());
            write_raw_d(os, z.imag());
        }
    }
}

inline Eigen::MatrixXcd read_compact_matrix(std::ifstream& is) {
    const std::uint64_t rows = read_raw_u64(is);
    const std::uint64_t cols = read_raw_u64(is);
    Eigen::MatrixXcd M(static_cast<int>(rows), static_cast<int>(cols));
    for(int c = 0; c < M.cols(); ++c) {
        for(int r = 0; r < M.rows(); ++r) {
            const double re = read_raw_d(is);
            const double im = read_raw_d(is);
            M(r, c) = comp(re, im);
        }
    }
    return M;
}

inline void write_json_vec(std::ofstream& os, const std::vector<int>& v) {
    os << "[";
    for(std::size_t i = 0; i < v.size(); ++i) {
        if(i) os << ", ";
        os << v[i];
    }
    os << "]";
}

inline std::string runtime_meta_path(const std::string& cache_path) { return cache_path + ".meta.json"; }

inline void write_runtime_meta_json(const std::string& cache_path, const RuntimeCacheMeta& m) {
    fs::create_directories(fs::path(cache_path).parent_path());
    std::ofstream os(runtime_meta_path(cache_path));
    if(!os) throw std::runtime_error("Could not open runtime cache meta for writing: " + runtime_meta_path(cache_path));
    os << std::setprecision(17);
    os << "{\n";
    os << "  \"version\": \"" << m.version << "\",\n";
    os << "  \"Lbyas\": " << m.Lbyas << ",\n";
    os << "  \"xi\": " << m.xi << ",\n";
    os << "  \"irrep\": \"" << m.irrep << "\",\n";
    os << "  \"coarseN\": " << m.coarseN << ",\n";
    os << "  \"Ecm_min\": " << m.Ecm_min << ",\n";
    os << "  \"Ecm_max\": " << m.Ecm_max << ",\n";
    os << "  \"rows\": " << m.rows << ",\n";
    os << "  \"atmK\": " << m.atmK << ",\n";
    os << "  \"atmpi\": " << m.atmpi << ",\n";
    os << "  \"eta_1\": " << m.eta_1 << ",\n";
    os << "  \"eta_2\": " << m.eta_2 << ",\n";
    os << "  \"alpha\": " << m.alpha << ",\n";
    os << "  \"epsilon_h\": " << m.epsilon_h << ",\n";
    os << "  \"max_shell_num\": " << m.max_shell_num << ",\n";
    os << "  \"tolerance\": " << m.tolerance << ",\n";
    os << "  \"parity\": " << m.parity << ",\n";
    os << "  \"eig_tol\": " << m.eig_tol << ",\n";
    os << "  \"norm_tol\": " << m.norm_tol << ",\n";
    os << "  \"proj_tol\": " << m.proj_tol << ",\n";
    os << "  \"waves_vec_1\": "; write_json_vec(os, m.waves_vec_1); os << ",\n";
    os << "  \"waves_vec_2\": "; write_json_vec(os, m.waves_vec_2); os << ",\n";
    os << "  \"scatter1_00\": " << m.scatter1_00 << ",\n";
    os << "  \"scatter1_10\": " << m.scatter1_10 << ",\n";
    os << "  \"scatter2_00\": " << m.scatter2_00 << ",\n";
    os << "  \"source_raw_cache\": \"" << m.source_raw_cache << "\",\n";
    os << "  \"source_raw_cache_file_size\": " << m.source_raw_cache_file_size << ",\n";
    os << "  \"source_raw_cache_mtime\": " << m.source_raw_cache_mtime << ",\n";
    os << "  \"created_by\": \"" << m.created_by << "\"\n";
    os << "}\n";
}

inline std::map<std::string, std::string> read_runtime_meta_kv(const std::string& cache_path) {
    std::ifstream is(runtime_meta_path(cache_path));
    if(!is) throw std::runtime_error("Could not open runtime cache meta for reading: " + runtime_meta_path(cache_path));
    std::map<std::string, std::string> kv;
    std::string line;
    while(std::getline(is, line)) {
        auto q1 = line.find('"');
        if(q1 == std::string::npos) continue;
        auto q2 = line.find('"', q1 + 1);
        if(q2 == std::string::npos) continue;
        auto colon = line.find(':', q2 + 1);
        if(colon == std::string::npos) continue;
        std::string key = line.substr(q1 + 1, q2 - q1 - 1);
        std::string val = line.substr(colon + 1);
        auto h = val.find('#');
        if(h != std::string::npos) val = val.substr(0, h);
        auto trim = [](std::string s) {
            while(!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
            while(!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
            return s;
        };
        val = trim(val);
        if(!val.empty() && val.back() == ',') val.pop_back();
        kv[key] = trim(val);
    }
    return kv;
}

inline double meta_double(const std::map<std::string, std::string>& kv, const std::string& k, double d) {
    auto it = kv.find(k);
    return it == kv.end() ? d : std::stod(it->second);
}
inline int meta_int(const std::map<std::string, std::string>& kv, const std::string& k, int d) {
    auto it = kv.find(k);
    return it == kv.end() ? d : std::stoi(it->second);
}
inline std::string meta_string(const std::map<std::string, std::string>& kv, const std::string& k, const std::string& d) {
    auto it = kv.find(k);
    if(it == kv.end()) return d;
    std::string v = it->second;
    if(v.size() >= 2 && v.front() == '"' && v.back() == '"') v = v.substr(1, v.size() - 2);
    return v;
}

inline ProjectedQCCacheEntry to_projected_entry_v33g(int grid_i,
                                                     double Ecm,
                                                     double En,
                                                     int success,
                                                     int total_dim,
                                                     int proj_dim,
                                                     const Eigen::MatrixXcd& A0,
                                                     const Eigen::MatrixXcd& B0,
                                                     const Eigen::MatrixXcd& B1,
                                                     const Eigen::MatrixXcd& BB,
                                                     const Eigen::MatrixXcd& BE,
                                                     const std::string& irrep) {
    ProjectedQCCacheEntry e;
    e.label = irrep;
    e.spec = parse_label(irrep);
    e.i = grid_i;
    e.Ecm = Ecm;
    e.En = En;
    e.success = success;
    e.total_dim = total_dim;
    e.proj_dim = proj_dim;
    e.F3inv_proj = A0;
    e.K3_proj_basis[0] = B0;
    e.K3_proj_basis[1] = B1;
    e.K3_proj_basis[2] = BB;
    e.K3_proj_basis[3] = BE;
    e.has_precomputed_k3_basis = true;
    e.error = success ? "OK_V33G_RUNTIME" : "BAD_V33G_RUNTIME_ROW";
    return e;
}

inline void write_runtime_cache(const std::string& cache_path, const IrrepCache& cache, const RuntimeCacheMeta& meta) {
    fs::create_directories(fs::path(cache_path).parent_path());
    std::ofstream os(cache_path, std::ios::binary);
    if(!os) throw std::runtime_error("Could not open runtime cache for writing: " + cache_path);
    os << "KKPI_V33G_RUNTIME_K3BASIS_CACHE 1\n";
    const std::uint64_t record_magic = 0x5633334752544B33ULL;
    for(const auto& e : cache.grid) {
        write_raw_u64(os, record_magic);
        write_raw_i32(os, static_cast<std::int32_t>(e.i));
        write_raw_i32(os, static_cast<std::int32_t>(e.total_dim));
        write_raw_i32(os, static_cast<std::int32_t>(e.proj_dim));
        write_raw_i32(os, static_cast<std::int32_t>(e.success));
        write_raw_i32(os, 0);
        write_raw_d(os, e.Ecm);
        write_raw_d(os, e.En);
        write_compact_matrix(os, e.F3inv_proj);
        write_compact_matrix(os, e.K3_proj_basis[0]);
        write_compact_matrix(os, e.K3_proj_basis[1]);
        write_compact_matrix(os, e.K3_proj_basis[2]);
        write_compact_matrix(os, e.K3_proj_basis[3]);
    }
    write_runtime_meta_json(cache_path, meta);
}

inline IrrepCache load_runtime_cache(const std::string& cache_path, const std::string& irrep) {
    std::ifstream is(cache_path, std::ios::binary);
    if(!is) throw std::runtime_error("Could not open runtime cache for reading: " + cache_path);
    std::string header;
    std::getline(is, header);
    if(header != "KKPI_V33G_RUNTIME_K3BASIS_CACHE 1") {
        throw std::runtime_error("Bad runtime cache header in " + cache_path + ": " + header);
    }
    IrrepCache ic;
    ic.label = irrep;
    ic.spec = parse_label(irrep);
    const std::uint64_t record_magic = 0x5633334752544B33ULL;
    while(true) {
        int c = is.peek();
        if(c == EOF) break;
        const std::uint64_t magic = read_raw_u64(is);
        if(!is) break;
        if(magic != record_magic) throw std::runtime_error("Bad runtime cache record magic in " + cache_path);
        const int grid_i = read_raw_i32(is);
        const int total_dim = read_raw_i32(is);
        const int proj_dim = read_raw_i32(is);
        const int success = read_raw_i32(is);
        (void)read_raw_i32(is);
        const double Ecm = read_raw_d(is);
        const double En = read_raw_d(is);
        const Eigen::MatrixXcd A0 = read_compact_matrix(is);
        const Eigen::MatrixXcd B0 = read_compact_matrix(is);
        const Eigen::MatrixXcd B1 = read_compact_matrix(is);
        const Eigen::MatrixXcd BB = read_compact_matrix(is);
        const Eigen::MatrixXcd BE = read_compact_matrix(is);
        ic.grid.push_back(to_projected_entry_v33g(grid_i, Ecm, En, success, total_dim, proj_dim, A0, B0, B1, BB, BE, irrep));
    }
    std::sort(ic.grid.begin(), ic.grid.end(), [](const auto& a, const auto& b) { return a.Ecm < b.Ecm; });
    return ic;
}

} // namespace v33g_runtime_k3basis
