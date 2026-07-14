#define V33H_NO_MAIN
#define V33H_BRIDGE_API
#include "v33h_patched_gpu_cache_oldscale_det_scan.cpp"
#include "v33p_gpu_det_lu.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <unordered_map>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct BridgeArgs {
    ScanArgs scan;
    int gpu_batch_rows = 512;
    int progress_every = 500;
    bool cpu_fallback = true;
    bool validate_subset = false;
    fs::path reference_csv;
    fs::path accepted_truezeros_csv;
};

static bool boolean_value(const std::string& s) { return parse_bool(s); }

static BridgeArgs parse_bridge_args(int argc, char** argv) {
    BridgeArgs a;
    a.scan.Emin = 0.26310;
    a.scan.Emax = 0.36;
    a.scan.Lbyas = {20.0};
    a.scan.irreps = {"100_A2"};
    a.scan.outdir = "output/v33p_gpu_det_bridge/L20_100_A2";
    for(int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if(i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
            return argv[++i];
        };
        if(arg == "--gpu-batch-rows") a.gpu_batch_rows = std::max(1, std::stoi(value()));
        else if(arg == "--progress-every") a.progress_every = std::max(1, std::stoi(value()));
        else if(arg == "--cpu-fallback") a.cpu_fallback = boolean_value(value());
        else if(arg == "--validate-subset") a.validate_subset = boolean_value(value());
        else if(arg == "--reference-csv") a.reference_csv = value();
        else if(arg == "--accepted-truezeros") a.accepted_truezeros_csv = value();
        else if(arg == "--help" || arg == "-h") {
            std::cout << "CPU-assembled corrected-reader GPU determinant bridge\n"
                      << "  --gpu-cache-root PATH --outdir PATH --Lbyas N --irrep LABEL\n"
                      << "  --Emin X --Emax X --coarseN N --xi X\n"
                      << "  --K3iso0 X --K3iso1 X --K3B X --K3E X\n"
                      << "  --gpu-batch-rows N --progress-every N --cpu-fallback true|false\n"
                      << "  --validate-subset true|false --reference-csv PATH\n";
            std::exit(0);
        } else if(arg == "--gpu-cache-root") a.scan.gpu_cache_root = value();
        else if(arg == "--outdir") a.scan.outdir = value();
        else if(arg == "--Lbyas") { a.scan.Lbyas.clear(); for(const auto& x : split_csv(value())) if(!x.empty()) a.scan.Lbyas.push_back(std::stod(x)); }
        else if(arg == "--irrep") { a.scan.irreps.clear(); for(const auto& x : split_csv(value())) if(!x.empty()) a.scan.irreps.push_back(x); }
        else if(arg == "--Emin") a.scan.Emin = std::stod(value());
        else if(arg == "--Emax") a.scan.Emax = std::stod(value());
        else if(arg == "--xi") a.scan.xi = std::stod(value());
        else if(arg == "--coarseN") a.scan.coarseN = std::max(1, std::stoi(value()));
        else if(arg == "--threads") a.scan.threads = std::max(1, std::stoi(value()));
        else if(arg == "--old-scaling") a.scan.old_scaling = boolean_value(value());
        else if(arg == "--complex-read-convention") a.scan.complex_read_convention = value();
        else if(arg == "--hard-hermiticity-check") a.scan.hard_hermiticity_check = boolean_value(value());
        else if(arg == "--K3iso0" || arg == "--k3iso0") a.scan.k3.K3iso0 = std::stod(value());
        else if(arg == "--K3iso1" || arg == "--k3iso1") a.scan.k3.K3iso1 = std::stod(value());
        else if(arg == "--K3B" || arg == "--k3B") a.scan.k3.K3B = std::stod(value());
        else if(arg == "--K3E" || arg == "--k3E") a.scan.k3.K3E = std::stod(value());
        else throw std::runtime_error("unknown argument: " + arg);
    }
    if(a.scan.Lbyas.size() != 1 || a.scan.irreps.size() != 1) throw std::runtime_error("bridge requires exactly one Lbyas and one irrep");
    if(a.reference_csv.empty()) a.reference_csv = "output/v33p_classifier_restart_L20_100_A2_E026310_0360/det_grid_L20_100_A2_corrected_E026310_0360.csv";
    if(a.accepted_truezeros_csv.empty()) a.accepted_truezeros_csv = "output/v33p_classifier_restart_L20_100_A2_E026310_0360/accepted_truezeros_L20_100_A2_E026310_0360.csv";
    if(!a.validate_subset && a.scan.outdir.filename() == "L20_100_A2") a.scan.outdir /= "gpu_det_grid";
    return a;
}

struct ReferenceRow {
    double Ecm = NAN;
    double det_real = NAN;
    double det_imag = NAN;
    int Nfull = 0;
    int Nproj = 0;
};

static std::vector<std::string> split_line(const std::string& line) {
    std::vector<std::string> out;
    std::string field;
    bool quoted = false;
    for(char c : line) {
        if(c == '"') quoted = !quoted;
        else if(c == ',' && !quoted) { out.push_back(field); field.clear(); }
        else field.push_back(c);
    }
    out.push_back(field);
    return out;
}

static std::unordered_map<std::size_t, ReferenceRow> read_reference(const fs::path& path) {
    std::ifstream in(path);
    if(!in) throw std::runtime_error("could not open reference CSV: " + path.string());
    std::string line;
    if(!std::getline(in, line)) throw std::runtime_error("empty reference CSV: " + path.string());
    const auto header = split_line(line);
    std::map<std::string, std::size_t> col;
    for(std::size_t i = 0; i < header.size(); ++i) col[header[i]] = i;
    for(const char* name : {"row_global_index", "Ecm", "Nfull", "Nproj", "det_real", "det_imag"}) if(!col.count(name)) throw std::runtime_error("reference CSV missing column " + std::string(name));
    std::unordered_map<std::size_t, ReferenceRow> out;
    while(std::getline(in, line)) {
        if(line.empty()) continue;
        const auto f = split_line(line);
        auto get = [&](const char* name) -> const std::string& { return f.at(col.at(name)); };
        const std::size_t id = static_cast<std::size_t>(std::stoull(get("row_global_index")));
        out[id] = ReferenceRow{std::stod(get("Ecm")), std::stod(get("det_real")), std::stod(get("det_imag")), std::stoi(get("Nfull")), std::stoi(get("Nproj"))};
    }
    return out;
}

static std::vector<double> read_zero_centers(const fs::path& path) {
    std::ifstream in(path);
    if(!in) throw std::runtime_error("could not open accepted-zero CSV: " + path.string());
    std::string line;
    if(!std::getline(in, line)) return {};
    const auto header = split_line(line);
    std::map<std::string, std::size_t> col;
    for(std::size_t i = 0; i < header.size(); ++i) col[header[i]] = i;
    const char* key = col.count("E_zero_linear") ? "E_zero_linear" : (col.count("zero_estimate") ? "zero_estimate" : nullptr);
    if(!key) throw std::runtime_error("accepted-zero CSV has no zero estimate column: " + path.string());
    std::vector<double> out;
    while(std::getline(in, line)) {
        if(line.empty()) continue;
        const auto f = split_line(line);
        if(col.count("user_label") && f.at(col.at("user_label")) != "true") continue;
        out.push_back(std::stod(f.at(col.at(key))));
    }
    return out;
}

static std::set<std::size_t> subset_ids(const std::vector<GpuRowMeta>& rows, const std::unordered_map<std::size_t, ReferenceRow>& ref, const std::vector<double>& accepted_zeros) {
    std::set<std::size_t> ids;
    for(std::size_t i = 0; i < std::min<std::size_t>(20, rows.size()); ++i) ids.insert(rows[i].raw_index);
    std::vector<std::pair<std::size_t, ReferenceRow>> ordered;
    ordered.reserve(ref.size());
    for(const auto& x : ref) ordered.push_back(x);
    std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) { return a.second.Ecm < b.second.Ecm; });
    auto add_neighborhood = [&](std::size_t center) {
        for(int d = -10; d <= 10; ++d) {
            const long j = static_cast<long>(center) + d;
            if(j >= 0 && j < static_cast<long>(ordered.size())) ids.insert(ordered[static_cast<std::size_t>(j)].first);
        }
    };
    for(double target : accepted_zeros) {
        std::size_t nearest = 0;
        for(std::size_t i = 1; i < ordered.size(); ++i) if(std::abs(ordered[i].second.Ecm - target) < std::abs(ordered[nearest].second.Ecm - target)) nearest = i;
        add_neighborhood(nearest);
    }
    int sign_flip_neighborhoods = 0;
    int dimension_jump_neighborhoods = 0;
    for(std::size_t i = 1; i < ordered.size(); ++i) {
        const auto& a = ordered[i - 1].second;
        const auto& b = ordered[i].second;
        if(a.Nfull != b.Nfull || a.Nproj != b.Nproj) {
            if(dimension_jump_neighborhoods++ < 3) add_neighborhood(i);
        } else if(a.det_real * b.det_real < 0.0 && sign_flip_neighborhoods++ < 5) add_neighborhood(i);
    }
    return ids;
}

struct WorkRow {
    V33hBridgeRow assembled;
    comp cpu_det;
    comp gpu_det;
    std::uint64_t gpu_bytes = 0;
    bool gpu_ok = false;
};

static std::vector<V33pComplex> pack_matrices(const std::vector<WorkRow*>& group, int n) {
    const std::size_t elems = static_cast<std::size_t>(n) * n;
    std::vector<V33pComplex> out(group.size() * elems);
    for(std::size_t b = 0; b < group.size(); ++b) {
        const auto& M = group[b]->assembled.scaled_projected_QC;
        for(std::size_t k = 0; k < elems; ++k) out[b * elems + k] = {M.data()[k].real(), M.data()[k].imag()};
    }
    return out;
}

static void write_subset_debug(const fs::path& path, const std::vector<WorkRow>& rows, const std::unordered_map<std::size_t, ReferenceRow>& ref) {
    std::ofstream out(path);
    if(!out) throw std::runtime_error("could not open subset debug CSV: " + path.string());
    out << "row_index,Ecm,Nproj,det_grid_real,det_grid_imag,local_det_real,local_det_imag,gpu_det_real,gpu_det_imag,abs_cpu_assembled_diff,abs_gpu_assembled_diff,sign_grid,sign_local,sign_gpu\n";
    out << std::scientific << std::setprecision(17);
    for(const auto& w : rows) {
        const auto it = ref.find(w.assembled.row.row_global_index);
        if(it == ref.end()) continue;
        const auto& r = it->second;
        const int sg = (r.det_real > 0) - (r.det_real < 0);
        const int sl = (w.cpu_det.real() > 0) - (w.cpu_det.real() < 0);
        const int su = (w.gpu_det.real() > 0) - (w.gpu_det.real() < 0);
        out << w.assembled.row.row_global_index << ',' << w.assembled.row.Ecm << ',' << w.assembled.row.Nproj << ','
            << r.det_real << ',' << r.det_imag << ',' << w.cpu_det.real() << ',' << w.cpu_det.imag() << ','
            << w.gpu_det.real() << ',' << w.gpu_det.imag() << ',' << std::abs(w.cpu_det - comp(r.det_real, r.det_imag)) << ','
            << std::abs(w.gpu_det - w.cpu_det) << ',' << sg << ',' << sl << ',' << su << '\n';
    }
}

static void write_report(const fs::path& path, const BridgeArgs& args, const fs::path& cache, std::size_t rows, std::size_t fallback, std::uint64_t max_gpu_bytes, double assembly_seconds, double gpu_seconds, double max_cpu_diff, double max_gpu_diff, bool subset) {
    std::ofstream out(path);
    if(!out) throw std::runtime_error("could not open report: " + path.string());
    out << "# CPU-assembled corrected-reader GPU determinant bridge\n\n"
        << "Status: " << (subset ? "SUBSET RUN" : "FULL BRIDGE RUN") << "\n\n"
        << "- cache: `" << cache.string() << "`\n"
        << "- sector: Lbyas=" << args.scan.Lbyas.front() << ", irrep=" << args.scan.irreps.front() << "\n"
        << "- reader: corrected `v33h` raw reader, `variant_04_real_imag_swapped`\n"
        << "- assembly: reused `compute_row` with `QC = F3inv + K3df`, `Vsel.adjoint() * QC * Vsel`, and division by `pow(Lbyas * xi, 6)`\n"
        << "- GPU LU API: `cublasZgetrfBatched`; determinant is pivot parity times the LU diagonal product\n"
        << "- rows processed: " << rows << "\n"
        << "- CPU fallback rows: " << fallback << "\n"
        << "- max CPU-assembled vs trusted CSV determinant difference: " << std::setprecision(17) << max_cpu_diff << "\n"
        << "- max GPU vs CPU-assembled determinant difference: " << max_gpu_diff << "\n"
        << "- max reported GPU allocation bytes: " << max_gpu_bytes << "\n"
        << "- CPU assembly seconds: " << assembly_seconds << "\n"
        << "- GPU determinant seconds: " << gpu_seconds << "\n\n"
        << "No cachegen, cache regeneration, cache copying, classifier change, or physics-formula change was performed.\n";
}

static void run_block(const BridgeArgs& args) {
    const double L = args.scan.Lbyas.front();
    const std::string& irrep = args.scan.irreps.front();
    const auto cache_opt = resolve_gpu_cache(args.scan.gpu_cache_root, L, irrep);
    if(!cache_opt) throw std::runtime_error("could not resolve cache");
    const fs::path cache = *cache_opt;
    validate_metadata(cache, L, irrep, args.scan.xi, args.scan.coarseN, args.scan.Emin, args.scan.Emax);
    const auto raw = scan_gpu_cache(cache);
    const auto selected = select_window_rows(raw, args.scan.Emin, args.scan.Emax);
    if(selected.empty()) throw std::runtime_error("no selected rows");
    const auto reference = read_reference(args.reference_csv);
    std::set<std::size_t> ids;
    if(args.validate_subset) ids = subset_ids(selected, reference, read_zero_centers(args.accepted_truezeros_csv));
    std::vector<GpuRowMeta> rows;
    for(const auto& r : selected) if(!args.validate_subset || ids.count(r.raw_index)) rows.push_back(r);
    const auto conv = parse_complex_read_convention(args.scan.complex_read_convention);
    const PhysicsParams par = make_physics(L, args.scan.xi, 1);
    FitSettings settings = make_settings(L, args.scan.xi, 1, irrep);
    settings.scan_E0 = args.scan.Emin;
    settings.scan_E1 = args.scan.Emax;
    settings.coarseN = args.scan.coarseN;
    settings.omp_threads = 1;
    const double scale = std::pow(L * args.scan.xi, 6.0);
    #ifdef _OPENMP
    omp_set_num_threads(std::max(1, args.scan.threads));
    #endif
    std::vector<WorkRow> all;
    all.reserve(rows.size());
    const auto assembly_start = std::chrono::steady_clock::now();
    std::size_t fallback = 0;
    std::uint64_t max_gpu_bytes = 0;
    double max_cpu_diff = 0.0;
    double max_gpu_diff = 0.0;
    double gpu_seconds = 0.0;
    if(!v33p_gpu_available() && !args.cpu_fallback) throw std::runtime_error("CUDA device unavailable and --cpu-fallback false");
    for(std::size_t offset = 0; offset < rows.size(); offset += static_cast<std::size_t>(args.gpu_batch_rows)) {
        const std::size_t end = std::min(rows.size(), offset + static_cast<std::size_t>(args.gpu_batch_rows));
        const std::size_t base = all.size();
        all.resize(base + end - offset);
        #pragma omp parallel for schedule(dynamic, 1)
        for(int local = 0; local < static_cast<int>(end - offset); ++local) {
            const std::size_t i = offset + static_cast<std::size_t>(local);
            all[base + static_cast<std::size_t>(local)].assembled = assemble_scaled_projected_QC_for_row(cache, rows[i], i, L, args.scan.xi, irrep, args.scan, par, settings, scale, conv);
            all[base + static_cast<std::size_t>(local)].cpu_det = all[base + static_cast<std::size_t>(local)].assembled.row.det;
        }
        for(std::size_t i = base; i < all.size(); ++i) {
            const auto ref = reference.find(all[i].assembled.row.row_global_index);
            if(ref != reference.end()) max_cpu_diff = std::max(max_cpu_diff, std::abs(all[i].cpu_det - comp(ref->second.det_real, ref->second.det_imag)));
        }
        std::map<int, std::vector<WorkRow*>> groups;
        for(std::size_t i = base; i < all.size(); ++i) groups[all[i].assembled.row.Nproj].push_back(&all[i]);
        for(const auto& [n, group] : groups) {
            const auto packed = pack_matrices(group, n);
            std::vector<V33pComplex> det(group.size());
            std::uint64_t bytes = 0;
            char error[512] = {};
            const auto t0 = std::chrono::steady_clock::now();
            const int ok = v33p_gpu_batched_determinants(n, static_cast<int>(group.size()), packed.data(), det.data(), &bytes, error, sizeof(error));
            gpu_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            max_gpu_bytes = std::max(max_gpu_bytes, bytes);
            if(!ok) {
                if(!args.cpu_fallback) throw std::runtime_error(std::string("GPU LU failed: ") + error);
                ++fallback;
                continue;
            }
            for(std::size_t j = 0; j < group.size(); ++j) {
                group[j]->gpu_det = comp(det[j].real, det[j].imag);
                group[j]->gpu_ok = true;
                group[j]->assembled.row.det = group[j]->gpu_det;
                group[j]->assembled.row.det_real = det[j].real;
                group[j]->assembled.row.det_imag = det[j].imag;
                group[j]->assembled.row.det_abs = std::abs(group[j]->gpu_det);
                max_gpu_diff = std::max(max_gpu_diff, std::abs(group[j]->gpu_det - group[j]->cpu_det));
            }
        }
        if(((end % static_cast<std::size_t>(args.progress_every)) == 0) || end == rows.size()) std::cout << "[bridge] rows=" << end << "/" << rows.size() << "\n";
    }
    const double assembly_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - assembly_start).count() - gpu_seconds;
    fs::create_directories(args.scan.outdir);
    const fs::path report = args.scan.outdir / (args.validate_subset ? "SUBSET_GPU_VS_CPU_DET_VALIDATION.md" : "GPU_VS_CPU_FULL_VALIDATION_REPORT.md");
    if(args.validate_subset) write_subset_debug(args.scan.outdir / "subset_gpu_det_debug.csv", all, reference);
    else {
        const fs::path tmp = args.scan.outdir / "gpu_det_grid.tmp.csv";
        const fs::path final = args.scan.outdir / (std::string("gpu_det_grid_L") + ltag(L) + "_" + irrep + "_E026310_0360.csv");
        std::ofstream out(tmp);
        if(!out) throw std::runtime_error("could not open temporary GPU CSV");
        write_row_csv_header(out);
        for(const auto& w : all) write_row_csv(out, w.assembled.row);
        out.close();
        fs::rename(tmp, final);
    }
    write_report(report, args, cache, all.size(), fallback, max_gpu_bytes, assembly_seconds, gpu_seconds, max_cpu_diff, max_gpu_diff, args.validate_subset);
    std::cout << "[bridge] cpu_fallback=" << fallback << " max_gpu_bytes=" << max_gpu_bytes << " max_gpu_diff=" << std::setprecision(17) << max_gpu_diff << "\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        run_block(parse_bridge_args(argc, argv));
        return 0;
    } catch(const std::exception& ex) {
        std::cerr << "[fatal] " << ex.what() << "\n";
        return 1;
    }
}
