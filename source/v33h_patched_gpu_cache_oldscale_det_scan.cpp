#define DISABLE_OLDSCALE_COMPARE_MAIN
#include "compare_gpu_cache_vs_cpu_openmp_det_oldscale.cpp"
#undef DISABLE_OLDSCALE_COMPARE_MAIN

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace k3df_fit_v32f;
using comp = std::complex<double>;

namespace {

constexpr double kHermiticityFail = 1.0e-6;
constexpr double kDetRelWarn = 1.0e-6;
constexpr double kTiny = 1.0e-300;
constexpr double kDetZeroThreshold = 1.0e-14;

struct ScanArgs {
    fs::path gpu_cache_root = "/media/digonto/Data/F3inv_cache";
    fs::path outdir = "output/v33h_patched_gpu_reader_oldscale_det_coarse20000";
    std::vector<double> Lbyas = {20.0};
    std::vector<std::string> irreps = {"100_A2"};
    double Emin = 0.34;
    double Emax = 0.36;
    double xi = 3.444;
    int coarseN = 20000;
    int threads = 16;
    bool old_scaling = true;
    std::string complex_read_convention = "variant_04_real_imag_swapped";
    bool hard_hermiticity_check = false;
    bool all_default_blocks = false;
    K3dfParameters k3{73735.840894011912, -972421.14060757787, 347174.05548116949, -1226756.7068845264};
};

struct RowOut {
    std::size_t row_global_index = 0;
    std::size_t row_sorted_index = 0;
    double Ecm = std::numeric_limits<double>::quiet_NaN();
    double Lbyas = std::numeric_limits<double>::quiet_NaN();
    std::string irrep;
    double xi = std::numeric_limits<double>::quiet_NaN();
    double const_norm_scale = std::numeric_limits<double>::quiet_NaN();
    int Nfull = 0;
    int Nproj = 0;
    int F3inv_rows = 0;
    int F3inv_cols = 0;
    int Vsel_rows = 0;
    int Vsel_cols = 0;
    int K3df_rows = 0;
    int K3df_cols = 0;
    int QC_rows = 0;
    int QC_cols = 0;
    int projected_QC_rows = 0;
    int projected_QC_cols = 0;
    comp det = {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    double det_real = std::numeric_limits<double>::quiet_NaN();
    double det_imag = std::numeric_limits<double>::quiet_NaN();
    double det_abs = std::numeric_limits<double>::quiet_NaN();
    double slogdet_logabs = std::numeric_limits<double>::quiet_NaN();
    comp slogdet_phase = {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    double F3inv_frob_norm = std::numeric_limits<double>::quiet_NaN();
    double K3df_frob_norm = std::numeric_limits<double>::quiet_NaN();
    double QC_frob_norm = std::numeric_limits<double>::quiet_NaN();
    double projected_QC_frob_norm = std::numeric_limits<double>::quiet_NaN();
    double scaled_projected_QC_frob_norm = std::numeric_limits<double>::quiet_NaN();
    double F3inv_herm_rel = std::numeric_limits<double>::quiet_NaN();
    double F3inv_max_diag_imag = std::numeric_limits<double>::quiet_NaN();
    double F3inv_trace_imag = std::numeric_limits<double>::quiet_NaN();
    double projected_QC_herm_rel = std::numeric_limits<double>::quiet_NaN();
    double scaled_projected_QC_herm_rel = std::numeric_limits<double>::quiet_NaN();
    double Vsel_orthonormality_error = std::numeric_limits<double>::quiet_NaN();
    double Vsel_projector_error = std::numeric_limits<double>::quiet_NaN();
    int Vsel_rank = 0;
    int success = 0;
    std::string failure_reason = "UNSET";
    double det_imag_abs = std::numeric_limits<double>::quiet_NaN();
    double det_imag_rel = std::numeric_limits<double>::quiet_NaN();
    double critical_score = std::numeric_limits<double>::quiet_NaN();
};

struct JumpRow {
    std::size_t jump_id = 0;
    std::size_t row_global_index = 0;
    std::size_t row_sorted_index = 0;
    double Ecm = std::numeric_limits<double>::quiet_NaN();
    int Nfull = 0;
    int Nproj = 0;
    int prev_Nfull = 0;
    int prev_Nproj = 0;
    int delta_Nfull = 0;
    int delta_Nproj = 0;
};

static bool parse_bool(const std::string& s) {
    if(s == "1" || s == "true" || s == "TRUE" || s == "yes" || s == "YES") return true;
    if(s == "0" || s == "false" || s == "FALSE" || s == "no" || s == "NO") return false;
    throw std::runtime_error("could not parse bool: " + s);
}

static v33g_runtime_k3basis::GpuComplexReadConvention parse_complex_read_convention(const std::string& s) {
    if(s == "variant_04_real_imag_swapped") {
        return v33g_runtime_k3basis::GpuComplexReadConvention::RealImagSwappedV33dV32zu;
    }
    throw std::runtime_error("unsupported complex read convention: " + s + " (only variant_04_real_imag_swapped is enabled)");
}

static std::string tag_window(double Emin, double Emax) {
    const int lo = static_cast<int>(std::llround(Emin * 100.0));
    const int hi = static_cast<int>(std::llround(Emax * 100.0));
    std::ostringstream os;
    os << "E" << std::setw(3) << std::setfill('0') << lo << "_" << std::setw(3) << std::setfill('0') << hi;
    return os.str();
}

static std::string scan_prefix(double L, const std::string& irrep, int coarseN, double Emin, double Emax) {
    return std::string("L") + ltag(L) + "_" + irrep + "_oldscale_det_coarse" + std::to_string(coarseN) + "_" + tag_window(Emin, Emax);
}

static ScanArgs parse_scan_args(int argc, char** argv) {
    ScanArgs a;
    for(int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const char* name) -> std::string {
            if(i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if(arg == "--help" || arg == "-h") {
            std::cerr
                << "Usage: " << argv[0] << " [options]\n"
                << "  --gpu-cache-root PATH\n"
                << "  --outdir PATH\n"
                << "  --Lbyas 20,24\n"
                << "  --irrep 000_A1m,100_A2,...\n"
                << "  --Emin 0.34 --Emax 0.36\n"
                << "  --xi 3.444\n"
                << "  --coarseN 20000\n"
                << "  --K3iso0 V --K3iso1 V --K3B V --K3E V\n"
                << "  --old-scaling true|false\n"
                << "  --complex-read-convention variant_04_real_imag_swapped\n"
                << "  --hard-hermiticity-check true|false\n"
                << "  --all-default-blocks true|false\n";
            std::exit(0);
        } else if(arg == "--gpu-cache-root") {
            a.gpu_cache_root = need_value("--gpu-cache-root");
        } else if(arg == "--outdir") {
            a.outdir = need_value("--outdir");
        } else if(arg == "--Lbyas") {
            a.Lbyas.clear();
            for(const auto& s : split_csv(need_value("--Lbyas"))) if(!s.empty()) a.Lbyas.push_back(std::stod(s));
        } else if(arg == "--irrep") {
            a.irreps.clear();
            for(const auto& s : split_csv(need_value("--irrep"))) if(!s.empty()) a.irreps.push_back(s);
        } else if(arg == "--Emin") {
            a.Emin = std::stod(need_value("--Emin"));
        } else if(arg == "--Emax") {
            a.Emax = std::stod(need_value("--Emax"));
        } else if(arg == "--xi") {
            a.xi = std::stod(need_value("--xi"));
        } else if(arg == "--coarseN") {
            a.coarseN = std::max(1, std::stoi(need_value("--coarseN")));
        } else if(arg == "--threads") {
            a.threads = std::max(1, std::stoi(need_value("--threads")));
        } else if(arg == "--old-scaling") {
            a.old_scaling = parse_bool(need_value("--old-scaling"));
        } else if(arg == "--complex-read-convention") {
            a.complex_read_convention = need_value("--complex-read-convention");
        } else if(arg == "--hard-hermiticity-check") {
            a.hard_hermiticity_check = parse_bool(need_value("--hard-hermiticity-check"));
        } else if(arg == "--all-default-blocks") {
            a.all_default_blocks = parse_bool(need_value("--all-default-blocks"));
        } else if(arg == "--k3iso0" || arg == "--K3iso0") {
            a.k3.K3iso0 = std::stod(need_value("--k3iso0"));
        } else if(arg == "--k3iso1" || arg == "--K3iso1") {
            a.k3.K3iso1 = std::stod(need_value("--k3iso1"));
        } else if(arg == "--k3B" || arg == "--K3B") {
            a.k3.K3B = std::stod(need_value("--k3B"));
        } else if(arg == "--k3E" || arg == "--K3E") {
            a.k3.K3E = std::stod(need_value("--k3E"));
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if(a.all_default_blocks) {
        a.Lbyas = {20.0, 24.0};
        a.irreps = {"000_A1m", "100_A2", "110_A2", "111_A2", "200_A2"};
        if(!(a.Emin < a.Emax)) {
            a.Emin = 0.2631;
            a.Emax = 0.36;
        }
    } else {
        if(a.Lbyas.empty()) a.Lbyas = {20.0};
        if(a.irreps.empty()) a.irreps = {"100_A2"};
    }
    return a;
}

static double hermiticity_rel(const Eigen::MatrixXcd& M) {
    if(M.rows() != M.cols() || M.rows() == 0) return std::numeric_limits<double>::quiet_NaN();
    const double n = M.norm();
    if(!(n > 0.0) || !std::isfinite(n)) return std::numeric_limits<double>::quiet_NaN();
    return (M - M.adjoint()).norm() / std::max(1.0, n);
}

static double max_diag_imag(const Eigen::MatrixXcd& M) {
    double out = 0.0;
    const int n = std::min(M.rows(), M.cols());
    for(int i = 0; i < n; ++i) out = std::max(out, std::abs(M(i, i).imag()));
    return out;
}

static int rank_from_svd(const Eigen::MatrixXcd& M, double rel_tol = 1.0e-10) {
    if(M.rows() == 0 || M.cols() == 0) return 0;
    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto sv = svd.singularValues();
    if(sv.size() == 0) return 0;
    const double max_sv = sv(0);
    const double thresh = rel_tol * std::max(1.0, max_sv);
    int rank = 0;
    for(int i = 0; i < sv.size(); ++i) if(sv(i) > thresh) ++rank;
    return rank;
}

static double projector_error_p2(const Eigen::MatrixXcd& V) {
    if(V.rows() <= 0 || V.cols() <= 0) return std::numeric_limits<double>::quiet_NaN();
    const Eigen::MatrixXcd P = V * V.adjoint();
    return (P * P - P).norm();
}

static comp swap_real_imag(const comp& z) {
    return comp(z.imag(), z.real());
}

static std::pair<double, comp> stable_slogdet_local(const Eigen::MatrixXcd& M) {
    if(M.rows() <= 0 || M.rows() != M.cols()) {
        return {std::numeric_limits<double>::quiet_NaN(), {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()}};
    }
    Eigen::PartialPivLU<Eigen::MatrixXcd> lu(M);
    const comp det = lu.determinant();
    double logabs = 0.0;
    bool ok = true;
    const auto& LU = lu.matrixLU();
    for(int i = 0; i < LU.rows() && i < LU.cols(); ++i) {
        const double a = std::abs(LU(i, i));
        if(!(a > 0.0) || !std::isfinite(a)) {
            ok = false;
            break;
        }
        logabs += std::log(a);
    }
    if(!ok) {
        const double a = std::abs(det);
        logabs = (a > 0.0 && std::isfinite(a)) ? std::log(a) : -std::numeric_limits<double>::infinity();
    }
    comp phase = {0.0, 0.0};
    const double a = std::abs(det);
    if(a > 0.0 && std::isfinite(a)) phase = det / a;
    return {logabs, phase};
}

static double orthonormality_error_local(const Eigen::MatrixXcd& V) {
    if(V.rows() <= 0 || V.cols() <= 0) return std::numeric_limits<double>::quiet_NaN();
    return (V.adjoint() * V - Eigen::MatrixXcd::Identity(V.cols(), V.cols())).norm();
}

static std::string quote_csv_local(const std::string& s) {
    if(s.find_first_of(",\"\n") == std::string::npos) return s;
    std::string out = "\"";
    for(char ch : s) {
        if(ch == '"') out += "\"\"";
        else out.push_back(ch);
    }
    out.push_back('"');
    return out;
}

static bool validate_metadata(const fs::path& cache_path, double L, const std::string& irrep, double xi, int coarseN, double Emin, double Emax) {
    const fs::path meta_path = cache_path.string() + ".meta.json";
    if(!fs::exists(meta_path)) throw std::runtime_error("missing cache sidecar: " + meta_path.string());
    const auto meta = v33g_runtime_k3basis::read_runtime_meta_kv(cache_path.string());
    const std::string version = v33g_runtime_k3basis::meta_string(meta, "version", "");
    const std::string cache_kind = v33g_runtime_k3basis::meta_string(meta, "cache_kind", "");
    const std::string cache_file = v33g_runtime_k3basis::meta_string(meta, "cache_file", "");
    const double meta_L = v33g_runtime_k3basis::meta_double(meta, "Lbyas", -1.0);
    const double meta_xi = v33g_runtime_k3basis::meta_double(meta, "xi", -1.0);
    const std::string meta_irrep_label = v33g_runtime_k3basis::meta_string(meta, "irrep_label", "");
    const std::string meta_irrep_tag = v33g_runtime_k3basis::meta_string(meta, "irrep_tag", "");
    const int meta_coarseN = v33g_runtime_k3basis::meta_int(meta, "coarseN", -1);
    const int meta_rows = v33g_runtime_k3basis::meta_int(meta, "rows_written", v33g_runtime_k3basis::meta_int(meta, "rows", -1));
    const double meta_ecm_min = v33g_runtime_k3basis::meta_double(meta, "Ecm_min", -1.0);
    const double meta_ecm_max = v33g_runtime_k3basis::meta_double(meta, "Ecm_max", -1.0);
    // The corrected combined-raw reader is format-aware for both the v33d
    // component writer and the v32zu exact F3inv/Vsel writer.  Provenance is
    // still checked explicitly; this is not a classifier or physics change.
    const bool version_ok = (version == "v33d_gpu_component_cachegen_all_ingredients" ||
                             version == "v32zu_gpu_cachegen_F3inv_Vsel_exact_v32y_opt9_concurrent_streams");
    const bool kind_ok = (cache_kind == "F3inv_plus_Vsel_gpu_full");
    const bool path_ok = cache_file.empty() || cache_file == cache_path.string();
    const bool irrep_ok = (meta_irrep_label == irrep) || (meta_irrep_tag == irrep);
    const bool coarse_ok = (meta_coarseN == coarseN);
    const bool rows_ok = (meta_rows == coarseN);
    const bool window_ok = (meta_ecm_min <= Emin + 1.0e-12) && (meta_ecm_max >= Emax - 1.0e-12);
    if(!version_ok || !kind_ok || !path_ok || std::abs(meta_L - L) > 1e-12 || std::abs(meta_xi - xi) > 1e-12 || !irrep_ok || !coarse_ok || !rows_ok || !window_ok) {
        std::ostringstream os;
        os << "cache metadata mismatch for " << cache_path
           << " version=" << version
           << " cache_kind=" << cache_kind
           << " meta_L=" << meta_L
           << " meta_xi=" << meta_xi
           << " meta_irrep_label=" << meta_irrep_label
           << " meta_irrep_tag=" << meta_irrep_tag
           << " meta_coarseN=" << meta_coarseN
           << " meta_rows=" << meta_rows
           << " meta_ecm_min=" << meta_ecm_min
           << " meta_ecm_max=" << meta_ecm_max;
        throw std::runtime_error(os.str());
    }
    return true;
}

static std::vector<GpuRowMeta> select_window_rows(const std::vector<GpuRowMeta>& raw, double Emin, double Emax) {
    std::vector<GpuRowMeta> rows;
    rows.reserve(raw.size());
    for(const auto& r : raw) {
        if(r.Ecm >= Emin - 1.0e-12 && r.Ecm <= Emax + 1.0e-12) rows.push_back(r);
    }
    std::sort(rows.begin(), rows.end(), [](const GpuRowMeta& a, const GpuRowMeta& b) {
        if(std::abs(a.Ecm - b.Ecm) > 1.0e-13) return a.Ecm < b.Ecm;
        return a.raw_index < b.raw_index;
    });
    std::vector<GpuRowMeta> unique_rows;
    unique_rows.reserve(rows.size());
    for(const auto& row : rows) {
        if(unique_rows.empty() || std::abs(row.Ecm - unique_rows.back().Ecm) > 1.0e-13) unique_rows.push_back(row);
    }
    return unique_rows;
}

static std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd> load_gpu_row_direct_variant04(const fs::path& cache_path, std::streamoff offset) {
    std::ifstream is(cache_path, std::ios::binary);
    if(!is) throw std::runtime_error("could not open GPU cache: " + cache_path.string());
    is.seekg(offset);
    if(!is) throw std::runtime_error("could not seek to GPU row offset in " + cache_path.string());
    auto read_i32 = [&]() -> std::int32_t {
        std::int32_t x{};
        is.read(reinterpret_cast<char*>(&x), sizeof(x));
        if(!is) throw std::runtime_error("truncated GPU cache row in " + cache_path.string());
        return x;
    };
    auto read_f64 = [&]() -> double {
        double x{};
        is.read(reinterpret_cast<char*>(&x), sizeof(x));
        if(!is) throw std::runtime_error("truncated GPU cache row in " + cache_path.string());
        return x;
    };
    std::uint64_t magic = 0;
    is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    const std::uint64_t rec_magic = 0x5653325a4f524543ULL;
    if(magic != rec_magic) throw std::runtime_error("bad GPU cache record magic at seeked row in " + cache_path.string());
    const std::int32_t grid_i = read_i32();
    const std::int32_t dim1 = read_i32();
    const std::int32_t dim2 = read_i32();
    const std::int32_t n = read_i32();
    const std::int32_t vdim = read_i32();
    const double Ecm = read_f64();
    const double En = read_f64();
    (void)grid_i;
    (void)dim1;
    (void)dim2;
    (void)Ecm;
    (void)En;
    const auto conv = v33g_runtime_k3basis::GpuComplexReadConvention::RealImagSwappedV33dV32zu;
    Eigen::MatrixXcd F3inv = v33g_runtime_k3basis::read_gpu_cache_matrix_col_major(is, n, n, conv);
    Eigen::MatrixXcd Vsel = v33g_runtime_k3basis::read_gpu_cache_matrix_col_major(is, n, vdim, conv);
    for(int r = 0; r < F3inv.rows(); ++r) {
        for(int c = 0; c < F3inv.cols(); ++c) {
            F3inv(r, c) = swap_real_imag(F3inv(r, c));
        }
    }
    for(int r = 0; r < Vsel.rows(); ++r) {
        for(int c = 0; c < Vsel.cols(); ++c) {
            Vsel(r, c) = swap_real_imag(Vsel(r, c));
        }
    }
    return {std::move(F3inv), std::move(Vsel)};
}

static RowOut compute_row(const fs::path& cache_path,
                          const GpuRowMeta& row,
                          std::size_t sorted_index,
                          double L,
                          double xi,
                          const std::string& irrep,
                          const ScanArgs& args,
                          const PhysicsParams& par,
                          const FitSettings& settings,
                          double const_norm_scale,
                          v33g_runtime_k3basis::GpuComplexReadConvention conv) {
    RowOut out;
    out.row_global_index = row.raw_index;
    out.row_sorted_index = sorted_index;
    out.Ecm = row.Ecm;
    out.Lbyas = L;
    out.irrep = irrep;
    out.xi = xi;
    out.const_norm_scale = const_norm_scale;
    out.F3inv_rows = row.total_dim;
    out.F3inv_cols = row.total_dim;
    out.Vsel_rows = row.total_dim;
    out.Vsel_cols = row.proj_dim;
    out.Nfull = row.total_dim;
    out.Nproj = row.proj_dim;

    const auto [F3inv, Vsel] = load_gpu_row_direct_variant04(cache_path, row.offset);
    ProjectedQCCacheEntry cpu_entry = build_cache_entry(static_cast<int>(sorted_index), row.Ecm, parse_label(irrep), settings, par, 'n');
    out.Nfull = F3inv.rows();
    out.Nproj = Vsel.cols();
    out.K3df_rows = cpu_entry.total_dim;
    out.K3df_cols = cpu_entry.total_dim;
    out.QC_rows = F3inv.rows();
    out.QC_cols = F3inv.cols();

    if(!cpu_entry.success) {
        out.failure_reason = "CPU_BUILD_FAIL";
        return out;
    }

    Eigen::MatrixXcd K3_full(cpu_entry.total_dim, cpu_entry.total_dim);
    std::vector<comp> Kiso = {comp(args.k3.K3iso0, 0.0), comp(args.k3.K3iso1, 0.0)};
    k3_2plus1::K3mat_2plus1(
        K3_full,
        cpu_entry.En_c,
        cpu_entry.plm_config,
        cpu_entry.klm_config,
        cpu_entry.total_P,
        par.atmK,
        par.atmpi,
        Kiso,
        comp(args.k3.K3B, 0.0),
        comp(args.k3.K3E, 0.0),
        'n'
    );
    out.K3df_rows = K3_full.rows();
    out.K3df_cols = K3_full.cols();

    Eigen::MatrixXcd QC = F3inv + K3_full;
    Eigen::MatrixXcd projected_QC = Vsel.adjoint() * QC * Vsel;
    Eigen::MatrixXcd scaled_projected_QC = projected_QC / comp(const_norm_scale, 0.0);

    out.projected_QC_rows = projected_QC.rows();
    out.projected_QC_cols = projected_QC.cols();
    out.det = det_complex(scaled_projected_QC);
    out.det_real = out.det.real();
    out.det_imag = out.det.imag();
    out.det_abs = std::abs(out.det);
    const auto slog = stable_slogdet_local(scaled_projected_QC);
    out.slogdet_logabs = slog.first;
    out.slogdet_phase = slog.second;
    out.F3inv_frob_norm = F3inv.norm();
    out.K3df_frob_norm = K3_full.norm();
    out.QC_frob_norm = QC.norm();
    out.projected_QC_frob_norm = projected_QC.norm();
    out.scaled_projected_QC_frob_norm = scaled_projected_QC.norm();
    out.F3inv_herm_rel = hermiticity_rel(F3inv);
    out.F3inv_max_diag_imag = max_diag_imag(F3inv);
    out.F3inv_trace_imag = F3inv.trace().imag();
    out.projected_QC_herm_rel = hermiticity_rel(projected_QC);
    out.scaled_projected_QC_herm_rel = hermiticity_rel(scaled_projected_QC);
    out.Vsel_orthonormality_error = orthonormality_error_local(Vsel);
    out.Vsel_projector_error = projector_error_p2(Vsel);
    out.Vsel_rank = rank_from_svd(Vsel, 1.0e-10);
    out.det_imag_abs = std::abs(out.det_imag);
    out.det_imag_rel = out.det_imag_abs / std::max(std::abs(out.det), kTiny);

    const bool nonfinite = !std::isfinite(out.F3inv_herm_rel) ||
                           !std::isfinite(out.F3inv_max_diag_imag) ||
                           !std::isfinite(out.Vsel_orthonormality_error) ||
                           !std::isfinite(out.Vsel_projector_error) ||
                           !std::isfinite(out.det_imag_abs) ||
                           !std::isfinite(out.det_imag_rel);
    out.success = 0;
    if(nonfinite) {
        out.failure_reason = "NONFINITE_ROW";
    } else if(out.F3inv_herm_rel > kHermiticityFail || out.F3inv_max_diag_imag > kHermiticityFail || out.Vsel_orthonormality_error > kHermiticityFail || out.Vsel_projector_error > kHermiticityFail) {
        out.failure_reason = "WARN_CONVENTION_OR_SUBSPACE";
    } else if(out.det_imag_abs > 1.0e-10 && out.det_imag_rel > kDetRelWarn) {
        out.failure_reason = "DET_IMAG_WARN";
    } else {
        out.failure_reason = "OK";
        out.success = 1;
    }
    const double crit1 = std::max({out.F3inv_herm_rel, out.F3inv_max_diag_imag, out.Vsel_orthonormality_error, out.Vsel_projector_error});
    const double crit2 = std::max(out.det_imag_abs, out.det_imag_rel);
    out.critical_score = std::max(crit1, crit2);
    return out;
}

static void write_row_csv_header(std::ofstream& os) {
    os << "row_global_index,row_sorted_index,Ecm,Lbyas,irrep,xi,const_norm_scale,Nfull,Nproj,F3inv_rows,F3inv_cols,Vsel_rows,Vsel_cols,K3df_rows,K3df_cols,QC_rows,QC_cols,projected_QC_rows,projected_QC_cols,det_real,det_imag,det_abs,slogdet_logabs,slogdet_phase_real,slogdet_phase_imag,F3inv_frob_norm,K3df_frob_norm,QC_frob_norm,projected_QC_frob_norm,scaled_projected_QC_frob_norm,F3inv_herm_rel,F3inv_max_diag_imag,F3inv_trace_imag,projected_QC_herm_rel,scaled_projected_QC_herm_rel,Vsel_orthonormality_error,Vsel_projector_error,Vsel_rank,det_imag_abs,det_imag_rel,success,failure_reason\n";
}

static void write_row_csv(std::ofstream& os, const RowOut& r) {
    os << r.row_global_index << ',' << r.row_sorted_index << ',' << std::scientific << std::setprecision(17)
       << r.Ecm << ',' << r.Lbyas << ',' << r.irrep << ',' << r.xi << ',' << r.const_norm_scale << ','
       << r.Nfull << ',' << r.Nproj << ',' << r.F3inv_rows << ',' << r.F3inv_cols << ','
       << r.Vsel_rows << ',' << r.Vsel_cols << ',' << r.K3df_rows << ',' << r.K3df_cols << ','
       << r.QC_rows << ',' << r.QC_cols << ',' << r.projected_QC_rows << ',' << r.projected_QC_cols << ','
       << r.det_real << ',' << r.det_imag << ',' << r.det_abs << ','
       << r.slogdet_logabs << ',' << r.slogdet_phase.real() << ',' << r.slogdet_phase.imag() << ','
       << r.F3inv_frob_norm << ',' << r.K3df_frob_norm << ',' << r.QC_frob_norm << ','
       << r.projected_QC_frob_norm << ',' << r.scaled_projected_QC_frob_norm << ','
       << r.F3inv_herm_rel << ',' << r.F3inv_max_diag_imag << ',' << r.F3inv_trace_imag << ','
       << r.projected_QC_herm_rel << ',' << r.scaled_projected_QC_herm_rel << ','
       << r.Vsel_orthonormality_error << ',' << r.Vsel_projector_error << ',' << r.Vsel_rank << ','
       << r.det_imag_abs << ',' << r.det_imag_rel << ',' << r.success << ',' << quote_csv_local(r.failure_reason) << '\n';
}

static void write_jump_csv_header(std::ofstream& os) {
    os << "jump_id,row_global_index,row_sorted_index,Ecm,Nfull,Nproj,prev_Nfull,prev_Nproj,delta_Nfull,delta_Nproj\n";
}

static void write_jump_csv(std::ofstream& os, const JumpRow& j) {
    os << j.jump_id << ',' << j.row_global_index << ',' << j.row_sorted_index << ',' << std::scientific << std::setprecision(17)
       << j.Ecm << ',' << j.Nfull << ',' << j.Nproj << ',' << j.prev_Nfull << ',' << j.prev_Nproj << ','
       << j.delta_Nfull << ',' << j.delta_Nproj << '\n';
}

static std::string format_double(double x) {
    std::ostringstream os;
    os << std::scientific << std::setprecision(17) << x;
    return os.str();
}

static void process_block(const ScanArgs& args, double L, const std::string& irrep, bool& any_critical_fail) {
    const auto cache_opt = resolve_gpu_cache(args.gpu_cache_root, L, irrep);
    if(!cache_opt) throw std::runtime_error("could not resolve cache for L=" + std::to_string(L) + " irrep=" + irrep);
    const fs::path cache_path = *cache_opt;
    validate_metadata(cache_path, L, irrep, args.xi, args.coarseN, args.Emin, args.Emax);

    const auto raw = scan_gpu_cache(cache_path);
    const auto rows = select_window_rows(raw, args.Emin, args.Emax);
    if(rows.empty()) throw std::runtime_error("no rows in requested window for " + cache_path.string());

    const double const_norm_scale = std::pow(L * args.xi, 6.0);
    const PhysicsParams par = make_physics(L, args.xi, 1);
    const auto conv = parse_complex_read_convention(args.complex_read_convention);
    const FitSettings settings = [&] {
        FitSettings s = make_settings(L, args.xi, args.threads, irrep);
        s.scan_E0 = args.Emin;
        s.scan_E1 = args.Emax;
        s.coarseN = args.coarseN;
        s.omp_threads = 1;
        return s;
    }();

    const std::string prefix = scan_prefix(L, irrep, args.coarseN, args.Emin, args.Emax);
    const fs::path csv_path = args.outdir / (prefix + ".csv");
    const fs::path summary_path = args.outdir / (prefix + "_summary.md");
    const fs::path jump_csv_path = args.outdir / (prefix + "_dimension_jumps.csv");
    const fs::path herm_csv_path = args.outdir / (prefix + "_hermiticity.csv");
    const fs::path worst_csv_path = args.outdir / (prefix + "_worst_rows.csv");
    fs::create_directories(args.outdir);

    #ifdef _OPENMP
    omp_set_num_threads(std::max(1, args.threads));
    #endif

    std::vector<RowOut> out_rows(rows.size());
    #pragma omp parallel for schedule(dynamic, 1)
    for(int i = 0; i < static_cast<int>(rows.size()); ++i) {
        out_rows[static_cast<std::size_t>(i)] = compute_row(cache_path, rows[static_cast<std::size_t>(i)], static_cast<std::size_t>(i), L, args.xi, irrep, args, par, settings, const_norm_scale, conv);
    }
    std::sort(out_rows.begin(), out_rows.end(), [](const RowOut& a, const RowOut& b) { return a.row_sorted_index < b.row_sorted_index; });

    std::vector<JumpRow> jumps;
    for(std::size_t i = 1; i < out_rows.size(); ++i) {
        if(out_rows[i].Nfull != out_rows[i - 1].Nfull || out_rows[i].Nproj != out_rows[i - 1].Nproj) {
            jumps.push_back(JumpRow{
                jumps.size(),
                out_rows[i].row_global_index,
                out_rows[i].row_sorted_index,
                out_rows[i].Ecm,
                out_rows[i].Nfull,
                out_rows[i].Nproj,
                out_rows[i - 1].Nfull,
                out_rows[i - 1].Nproj,
                out_rows[i].Nfull - out_rows[i - 1].Nfull,
                out_rows[i].Nproj - out_rows[i - 1].Nproj,
            });
        }
    }

    std::vector<RowOut> worst = out_rows;
    std::sort(worst.begin(), worst.end(), [](const RowOut& a, const RowOut& b) {
        if(a.critical_score != b.critical_score) return a.critical_score > b.critical_score;
        return a.row_sorted_index < b.row_sorted_index;
    });
    if(worst.size() > 50) worst.resize(50);

    std::ofstream csv(csv_path);
    if(!csv) throw std::runtime_error("could not open " + csv_path.string());
    write_row_csv_header(csv);
    for(const auto& r : out_rows) write_row_csv(csv, r);

    std::ofstream jump_csv(jump_csv_path);
    if(!jump_csv) throw std::runtime_error("could not open " + jump_csv_path.string());
    write_jump_csv_header(jump_csv);
    for(const auto& j : jumps) write_jump_csv(jump_csv, j);

    std::ofstream herm_csv(herm_csv_path);
    if(!herm_csv) throw std::runtime_error("could not open " + herm_csv_path.string());
    herm_csv << "row_global_index,row_sorted_index,Ecm,F3inv_herm_rel,F3inv_max_diag_imag,F3inv_trace_imag,Vsel_orthonormality_error,Vsel_projector_error,projected_QC_herm_rel,scaled_projected_QC_herm_rel,det_real,det_imag,det_abs,det_imag_abs,det_imag_rel,success,failure_reason\n";
    herm_csv << std::scientific << std::setprecision(17);
    for(const auto& r : out_rows) {
        herm_csv << r.row_global_index << ',' << r.row_sorted_index << ',' << r.Ecm << ','
                 << r.F3inv_herm_rel << ',' << r.F3inv_max_diag_imag << ',' << r.F3inv_trace_imag << ','
                 << r.Vsel_orthonormality_error << ',' << r.Vsel_projector_error << ','
                 << r.projected_QC_herm_rel << ',' << r.scaled_projected_QC_herm_rel << ','
                 << r.det_real << ',' << r.det_imag << ',' << r.det_abs << ','
                 << r.det_imag_abs << ',' << r.det_imag_rel << ',' << r.success << ','
                 << quote_csv_local(r.failure_reason) << '\n';
    }

    std::ofstream worst_csv(worst_csv_path);
    if(!worst_csv) throw std::runtime_error("could not open " + worst_csv_path.string());
    write_row_csv_header(worst_csv);
    for(const auto& r : worst) write_row_csv(worst_csv, r);

    std::size_t failed_rows = 0;
    std::size_t warning_rows = 0;
    double max_det_abs = 0.0;
    double max_det_imag = 0.0;
    double max_det_imag_rel = 0.0;
    double max_f3_herm = 0.0;
    double max_proj_herm = 0.0;
    double max_scaled_proj_herm = 0.0;
    double max_vsel_orth = 0.0;
    double max_vsel_proj = 0.0;
    std::vector<double> det_abs_vals;
    std::vector<double> det_real_vals;
    det_abs_vals.reserve(out_rows.size());
    det_real_vals.reserve(out_rows.size());
    std::size_t pos = 0, neg = 0, zero = 0;
    std::size_t pos_near = 0, neg_near = 0, zero_near = 0;
    for(const auto& r : out_rows) {
        failed_rows += (r.success == 0);
        if(r.failure_reason.rfind("WARN", 0) == 0) ++warning_rows;
        max_det_abs = std::max(max_det_abs, r.det_abs);
        max_det_imag = std::max(max_det_imag, r.det_imag_abs);
        max_det_imag_rel = std::max(max_det_imag_rel, r.det_imag_rel);
        max_f3_herm = std::max(max_f3_herm, r.F3inv_herm_rel);
        max_proj_herm = std::max(max_proj_herm, r.projected_QC_herm_rel);
        max_scaled_proj_herm = std::max(max_scaled_proj_herm, r.scaled_projected_QC_herm_rel);
        max_vsel_orth = std::max(max_vsel_orth, r.Vsel_orthonormality_error);
        max_vsel_proj = std::max(max_vsel_proj, r.Vsel_projector_error);
        det_abs_vals.push_back(r.det_abs);
        det_real_vals.push_back(r.det_real);
        if(std::abs(r.det_real) < kDetZeroThreshold) ++zero;
        else if(r.det_real > 0.0) ++pos;
        else ++neg;
        if(std::abs(r.det_real) < kDetZeroThreshold) ++zero_near;
        else if(r.det_real > 0.0) ++pos_near;
        else ++neg_near;
        if(r.success == 0) any_critical_fail = true;
    }
    std::sort(det_abs_vals.begin(), det_abs_vals.end());
    std::sort(det_real_vals.begin(), det_real_vals.end());
    auto median = [](const std::vector<double>& v) {
        if(v.empty()) return std::numeric_limits<double>::quiet_NaN();
        const std::size_t n = v.size();
        if(n % 2 == 1) return v[n / 2];
        return 0.5 * (v[n / 2 - 1] + v[n / 2]);
    };
    auto mean = [](const std::vector<double>& v) {
        if(v.empty()) return std::numeric_limits<double>::quiet_NaN();
        double sum = 0.0;
        for(double x : v) sum += x;
        return sum / double(v.size());
    };

    std::ostringstream md;
    md << "# v33h patched GPU cache reader old-scale determinant scan\n\n";
    md << "## Block\n\n";
    md << "- `Lbyas = " << L << "`\n";
    md << "- `irrep = " << irrep << "`\n";
    md << "- `Ecm window = [" << args.Emin << ", " << args.Emax << "]`\n";
    md << "- `coarseN = " << args.coarseN << "`\n";
    md << "- `cache file = " << cache_path << "`\n";
    md << "- `complex read convention = " << args.complex_read_convention << "`\n";
    md << "- `hard hermiticity check = " << (args.hard_hermiticity_check ? "true" : "false") << "`\n";
    md << "- `old scaling = (Lbyas * xi)^6`\n\n";
    md << "## Parameters\n\n";
    md << "- `K3iso0 = " << format_double(args.k3.K3iso0) << "`\n";
    md << "- `K3iso1 = " << format_double(args.k3.K3iso1) << "`\n";
    md << "- `K3B = " << format_double(args.k3.K3B) << "`\n";
    md << "- `K3E = " << format_double(args.k3.K3E) << "`\n";
    md << "- `atmK = 0.09698`\n";
    md << "- `atmpi = 0.06906`\n";
    md << "- `eta_1 = 1.0`\n";
    md << "- `eta_2 = 0.5`\n";
    md << "- `alpha = 0.5`\n";
    md << "- `epsilon_h = 0.0`\n";
    md << "- `max_shell_num = 20`\n";
    md << "- `tolerance = 1e-12`\n";
    md << "- `parity = -1`\n";
    md << "- `eig_tol = 0.05`\n";
    md << "- `norm_tol = 1e-12`\n";
    md << "- `proj_tol = 1e-10`\n\n";
    md << "## Summary\n\n";
    md << "- rows in window: `" << out_rows.size() << "`\n";
    md << "- failed rows: `" << failed_rows << "`\n";
    md << "- warning rows: `" << warning_rows << "`\n";
    md << "- max |det|: `" << max_det_abs << "`\n";
    md << "- mean |det|: `" << mean(det_abs_vals) << "`\n";
    md << "- median |det|: `" << median(det_abs_vals) << "`\n";
    md << "- max det_real: `" << *std::max_element(det_real_vals.begin(), det_real_vals.end()) << "`\n";
    md << "- mean det_real: `" << mean(det_real_vals) << "`\n";
    md << "- median det_real: `" << median(det_real_vals) << "`\n";
    md << "- max |det_imag|: `" << max_det_imag << "`\n";
    md << "- max det imaginary relative ratio: `" << max_det_imag_rel << "`\n";
    md << "- max F3inv hermiticity error: `" << max_f3_herm << "`\n";
    md << "- max projected_QC hermiticity error: `" << max_proj_herm << "`\n";
    md << "- max scaled projected_QC hermiticity error: `" << max_scaled_proj_herm << "`\n";
    md << "- max Vsel orthonormality error: `" << max_vsel_orth << "`\n";
    md << "- max Vsel projector error: `" << max_vsel_proj << "`\n";
    md << "- det_real sign counts: `+ " << pos << " / - " << neg << " / 0 " << zero << "`\n";
    md << "- det_real near-zero sign threshold: `" << kDetZeroThreshold << "`\n";
    md << "- det_real near-zero sign counts: `+ " << pos_near << " / - " << neg_near << " / 0 " << zero_near << "`\n\n";
    md << "## Dimension Jumps\n\n";
    md << "| jump_id | row | Ecm | Nfull | Nproj | prev_Nfull | prev_Nproj | delta_Nfull | delta_Nproj |\n";
    md << "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n";
    for(const auto& j : jumps) {
        md << "| " << j.jump_id << " | " << j.row_sorted_index << " | " << j.Ecm << " | " << j.Nfull << " | " << j.Nproj
           << " | " << j.prev_Nfull << " | " << j.prev_Nproj << " | " << j.delta_Nfull << " | " << j.delta_Nproj << " |\n";
    }
    md << "\n## Interpretation\n\n";
    if(any_critical_fail) {
        md << "- critical threshold exceeded in at least one row; inspect `*_hermiticity.csv` and `*_worst_rows.csv`.\n";
    } else {
        md << "- all rows cleared the reader sanity thresholds; the complex reconstruction is consistent with `variant_04_real_imag_swapped`.\n";
    }
    md << "- determinant imaginary parts are tracked separately from the Hermiticity checks so tiny near-zero determinants do not mask a reader bug.\n";
    md << "- output files:\n";
    md << "  - `" << csv_path << "`\n";
    md << "  - `" << summary_path << "`\n";
    md << "  - `" << jump_csv_path << "`\n";
    md << "  - `" << herm_csv_path << "`\n";
    md << "  - `" << worst_csv_path << "`\n";

    std::ofstream md_out(summary_path);
    if(!md_out) throw std::runtime_error("could not open " + summary_path.string());
    md_out << md.str();

    std::cout << "[block] Lbyas=" << L << " irrep=" << irrep << " rows=" << out_rows.size() << " failed=" << failed_rows << "\n";
    std::cout << "[block] cache=" << cache_path << "\n";
    std::cout << "[block] csv=" << csv_path << "\n";
    std::cout << "[block] summary=" << summary_path << "\n";
    if(args.hard_hermiticity_check) {
        for(const auto& r : out_rows) {
            if(r.failure_reason.rfind("WARN", 0) == 0 || r.failure_reason == "DET_IMAG_WARN" || r.failure_reason == "NONFINITE_ROW") {
                throw std::runtime_error("hard hermiticity check failed for " + cache_path.string() + " row " + std::to_string(r.row_sorted_index) + " reason=" + r.failure_reason);
            }
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const ScanArgs args = parse_scan_args(argc, argv);
        (void)parse_complex_read_convention(args.complex_read_convention);
        fs::create_directories(args.outdir);
        bool any_critical_fail = false;
        for(double L : args.Lbyas) {
            for(const auto& irrep : args.irreps) {
                process_block(args, L, irrep, any_critical_fail);
            }
        }
        if(any_critical_fail) {
            std::cerr << "[fatal] one or more rows exceeded critical sanity thresholds\n";
            return 2;
        }
        return 0;
    } catch(const std::exception& ex) {
        std::cerr << "[fatal] " << ex.what() << "\n";
        return 1;
    }
}
