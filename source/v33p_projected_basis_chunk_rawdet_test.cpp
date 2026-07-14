#define V33H_NO_MAIN
#define V33H_BRIDGE_API
#include "v33h_patched_gpu_cache_oldscale_det_scan.cpp"

#include <chrono>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct BasisArgs {
    fs::path root = "/media/digonto/Data/F3inv_cache";
    fs::path outdir = "output/v33p_projected_basis_validation_start";
    fs::path reference = "output/v33p_classifier_restart_L20_100_A2_E026310_0360/det_grid_L20_100_A2_corrected_E026310_0360.csv";
    fs::path candidates = "output/v33p_classifier_restart_L20_100_A2_E026310_0360/candidate_sign_change_brackets_E026310_0360.csv";
    double L = 20.0, xi = 3.444, Emin = 0.26310, Emax = 0.36;
    int coarseN = 20000, threads = 16;
    K3dfParameters p{73735.840894011912, -972421.14060757787, 347174.05548116949, -1226756.7068845264};
};

static std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while(std::getline(ss, field, ',')) fields.push_back(field);
    return fields;
}

static Eigen::MatrixXcd k3_matrix(const ProjectedQCCacheEntry& e, const PhysicsParams& par, const K3dfParameters& p) {
    Eigen::MatrixXcd K(e.total_dim, e.total_dim);
    std::vector<comp> Kiso = {comp(p.K3iso0, 0.0), comp(p.K3iso1, 0.0)};
    k3_2plus1::K3mat_2plus1(K, e.En_c, e.plm_config, e.klm_config, e.total_P, par.atmK, par.atmpi, Kiso, comp(p.K3B, 0.0), comp(p.K3E, 0.0), 'n');
    return K;
}

static std::vector<std::size_t> read_selected_ids(const fs::path& candidate_path, const std::vector<GpuRowMeta>& rows) {
    std::ifstream in(candidate_path);
    if(!in) throw std::runtime_error("could not open candidate CSV: " + candidate_path.string());
    std::string line;
    std::getline(in, line);
    std::vector<std::pair<std::size_t, int>> centers;
    while(std::getline(in, line)) {
        if(line.empty()) continue;
        auto f = split_csv_line(line);
        if(f.size() < 4) continue;
        const int id = std::stoi(f[0]);
        const std::size_t row_left = static_cast<std::size_t>(std::stoull(f[2]));
        centers.push_back({row_left, id});
    }
    if(centers.size() < 8) throw std::runtime_error("candidate CSV has too few brackets");
    std::set<std::size_t> ids;
    auto add = [&](std::size_t c, int half, const char* label) {
        (void)label;
        const std::size_t lo = c > static_cast<std::size_t>(half) ? c - static_cast<std::size_t>(half) : 0;
        const std::size_t hi = std::min(rows.size(), c + static_cast<std::size_t>(half) + 1);
        for(std::size_t i = lo; i < hi; ++i) ids.insert(rows[i].raw_index);
    };
    // Accepted bracket 7, false/non-true-zero bracket 1, and a 128-row mid-grid chunk.
    add(centers[6].first, 32, "accepted");
    add(centers[0].first, 32, "false_candidate");
    const std::size_t mid = rows.size() / 2;
    for(std::size_t i = mid - 64; i < mid + 64; ++i) ids.insert(rows[i].raw_index);
    return std::vector<std::size_t>(ids.begin(), ids.end());
}

static BasisArgs basis_parse_args(int argc, char** argv) {
    BasisArgs a;
    for(int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto val = [&]() -> std::string { if(i + 1 >= argc) throw std::runtime_error("missing value for " + arg); return argv[++i]; };
        if(arg == "--gpu-cache-root") a.root = val();
        else if(arg == "--outdir") a.outdir = val();
        else if(arg == "--reference-csv") a.reference = val();
        else if(arg == "--candidate-csv") a.candidates = val();
        else if(arg == "--threads") a.threads = std::max(1, std::stoi(val()));
        else if(arg == "--K3iso0") a.p.K3iso0 = std::stod(val());
        else if(arg == "--K3iso1") a.p.K3iso1 = std::stod(val());
        else if(arg == "--K3B") a.p.K3B = std::stod(val());
        else if(arg == "--K3E") a.p.K3E = std::stod(val());
        else if(arg == "--help") { std::cout << "projected-basis chunk raw-det prototype\n"; std::exit(0); }
        else throw std::runtime_error("unknown argument: " + arg);
    }
    return a;
}

struct Result {
    std::size_t row = 0;
    double Ecm = NAN;
    int Nproj = 0;
    double matrix_max_abs = NAN;
    comp direct_det, basis_det;
    std::string section;
};

static void run(const BasisArgs& a) {
    const std::string irrep = "100_A2";
    const auto cache_opt = resolve_gpu_cache(a.root, a.L, irrep);
    if(!cache_opt) throw std::runtime_error("could not resolve cache");
    const fs::path cache = *cache_opt;
    validate_metadata(cache, a.L, irrep, a.xi, a.coarseN, a.Emin, a.Emax);
    const auto raw = select_window_rows(scan_gpu_cache(cache), a.Emin, a.Emax);
    const auto ids = read_selected_ids(a.candidates, raw);
    const auto conv = parse_complex_read_convention("variant_04_real_imag_swapped");
    const PhysicsParams par = make_physics(a.L, a.xi, 1);
    FitSettings settings = make_settings(a.L, a.xi, 1, irrep);
    settings.scan_E0 = a.Emin; settings.scan_E1 = a.Emax; settings.coarseN = a.coarseN; settings.omp_threads = 1;
    const double scale = std::pow(a.L * a.xi, 6.0);
    #ifdef _OPENMP
    omp_set_num_threads(a.threads);
    #endif
    std::vector<Result> results(ids.size());
    const auto t0 = std::chrono::steady_clock::now();
    #pragma omp parallel for schedule(dynamic, 1)
    for(int q = 0; q < static_cast<int>(ids.size()); ++q) {
        const auto raw_it = std::find_if(raw.begin(), raw.end(), [&](const GpuRowMeta& r) { return r.raw_index == ids[static_cast<std::size_t>(q)]; });
        if(raw_it == raw.end()) continue;
        const std::size_t i = static_cast<std::size_t>(q);
        ScanArgs bridge_args;
        bridge_args.gpu_cache_root = a.root;
        bridge_args.outdir = a.outdir;
        bridge_args.Lbyas = {a.L};
        bridge_args.irreps = {irrep};
        bridge_args.Emin = a.Emin;
        bridge_args.Emax = a.Emax;
        bridge_args.xi = a.xi;
        bridge_args.coarseN = a.coarseN;
        bridge_args.threads = 1;
        bridge_args.old_scaling = true;
        bridge_args.complex_read_convention = "variant_04_real_imag_swapped";
        bridge_args.hard_hermiticity_check = false;
        bridge_args.all_default_blocks = false;
        bridge_args.k3 = a.p;
        const auto direct = assemble_scaled_projected_QC_for_row(cache, *raw_it, i, a.L, a.xi, irrep,
                                                                   bridge_args, par, settings, scale, conv);
        const auto [F3inv, Vsel] = load_gpu_row_direct_variant04(cache, raw_it->offset);
        const auto entry = build_cache_entry(static_cast<int>(i), raw_it->Ecm, parse_label(irrep), settings, par, 'n');
        const auto project = [&](const Eigen::MatrixXcd& M) { return (Vsel.adjoint() * M * Vsel) / comp(scale, 0.0); };
        const Eigen::MatrixXcd Fproj = project(F3inv);
        K3dfParameters z0{1.0, 0.0, 0.0, 0.0}, z1{0.0, 1.0, 0.0, 0.0}, zB{0.0, 0.0, 1.0, 0.0}, zE{0.0, 0.0, 0.0, 1.0};
        const Eigen::MatrixXcd basis = Fproj + a.p.K3iso0 * project(k3_matrix(entry, par, z0)) + a.p.K3iso1 * project(k3_matrix(entry, par, z1)) + a.p.K3B * project(k3_matrix(entry, par, zB)) + a.p.K3E * project(k3_matrix(entry, par, zE));
        results[i] = Result{raw_it->raw_index, raw_it->Ecm, Vsel.cols(), (direct.scaled_projected_QC - basis).cwiseAbs().maxCoeff(), direct.row.det, det_complex(basis), i < 64 ? "accepted_bracket" : (i < 128 ? "false_candidate_bracket" : "mid_grid")};
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fs::create_directories(a.outdir);
    std::ofstream csv(a.outdir / "projected_basis_chunk_results.csv");
    csv << "row_index,Ecm,Nproj,matrix_max_abs_diff,det_direct_real,det_direct_imag,det_basis_real,det_basis_imag,det_abs_diff,det_rel_diff,sign_agreement,section\n" << std::scientific << std::setprecision(17);
    double max_matrix = 0.0, max_det = 0.0, max_rel = 0.0; std::size_t sign_ok = 0;
    for(const auto& r : results) {
        const double ad = std::abs(r.direct_det - r.basis_det), rel = ad / std::max(std::abs(r.direct_det), 1e-300);
        const bool sign = (r.direct_det.real() > 0) == (r.basis_det.real() > 0) && (r.direct_det.real() < 0) == (r.basis_det.real() < 0);
        max_matrix = std::max(max_matrix, r.matrix_max_abs); max_det = std::max(max_det, ad); max_rel = std::max(max_rel, rel); sign_ok += sign;
        csv << r.row << ',' << r.Ecm << ',' << r.Nproj << ',' << r.matrix_max_abs << ',' << r.direct_det.real() << ',' << r.direct_det.imag() << ',' << r.basis_det.real() << ',' << r.basis_det.imag() << ',' << ad << ',' << rel << ',' << (sign ? 1 : 0) << ',' << r.section << '\n';
    }
    std::ofstream report(a.outdir / "PROJECTED_BASIS_CHUNK_PROTOTYPE_REPORT.md");
    report << "# Projected-basis chunk prototype\n\nStatus: " << ((max_matrix < 1e-10 && max_det < 1e-15 && sign_ok == results.size()) ? "PROJECTED_BASIS_CHUNK_PROTOTYPE_PASS" : "PROJECTED_BASIS_CHUNK_PROTOTYPE_FAIL") << "\n\n"
           << "- sector: L20/100_A2\n- rows: " << results.size() << " (64 accepted-bracket, 64 false-candidate, 128 mid-grid)\n"
           << "- reader: corrected v33h raw reader, variant_04_real_imag_swapped\n"
           << "- matrix construction: Fproj plus four projected/scaled K3df basis matrices\n"
           << "- max direct-vs-basis matrix absolute difference: " << max_matrix << "\n"
           << "- max direct-vs-basis raw determinant absolute difference: " << max_det << "\n"
           << "- max raw determinant relative difference: " << max_rel << "\n"
           << "- raw determinant sign agreement: " << sign_ok << "/" << results.size() << "\n"
           << "- GPU basis determinant: not used; existing GPU LU raw bridge was already independently revalidated\n"
           << "- elapsed prototype seconds: " << seconds << "\n"
           << "- logdet/logabs: not used as classifier or fitter input\n\n"
           << "The basis path is accepted only if raw matrix, determinant, sign, and bracket behavior agree.\n";
}

} // namespace

int main(int argc, char** argv) {
    try { run(basis_parse_args(argc, argv)); return 0; }
    catch(const std::exception& e) { std::cerr << "[fatal] " << e.what() << "\n"; return 1; }
}
