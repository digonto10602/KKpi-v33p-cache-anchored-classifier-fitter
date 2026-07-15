#define V33H_NO_MAIN
#define V33H_BRIDGE_API
#include "v33h_patched_gpu_cache_oldscale_det_scan.cpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>

namespace v33p_hotwindow_basis {

struct Args {
    fs::path root = "/media/digonto/Data/F3inv_cache";
    fs::path outdir = "output/v33p_projected_basis_cache/hot_windows/sector";
    fs::path windows;
    double L = 20.0, xi = 3.444, Emin = 0.26310, Emax = 0.36;
    int coarseN = 20000;
    std::string irrep = "100_A2";
    std::string windows_sha256;
    std::string git_commit = "unknown";
    K3dfParameters p{73735.840894011912, -972421.14060757787,
                     347174.05548116949, -1226756.7068845264};
};

struct Window { std::size_t bracket = 0, lo = 0, hi = 0; };

static std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string x;
    while (std::getline(ss, x, ',')) out.push_back(x);
    return out;
}

static std::string now_utc() {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
    gmtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

static std::vector<Window> read_windows(const fs::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open accepted windows: " + path.string());
    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("empty accepted windows: " + path.string());
    const auto h = split(line);
    std::map<std::string, std::size_t> col;
    for (std::size_t i = 0; i < h.size(); ++i) col[h[i]] = i;
    for (const char* k : {"bracket_id", "row_left", "row_right"}) {
        if (!col.count(k)) throw std::runtime_error(std::string("accepted windows missing ") + k);
    }
    std::vector<Window> out;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto f = split(line);
        if (f.size() < h.size()) continue;
        out.push_back({
            static_cast<std::size_t>(std::stoull(f[col["bracket_id"]])),
            static_cast<std::size_t>(std::stoull(f[col["row_left"]])),
            static_cast<std::size_t>(std::stoull(f[col["row_right"]]))});
    }
    if (out.empty()) throw std::runtime_error("accepted windows has no rows");
    return out;
}

static Eigen::MatrixXcd k3_matrix(const ProjectedQCCacheEntry& e,
                                  const PhysicsParams& par,
                                  const K3dfParameters& p) {
    Eigen::MatrixXcd K(e.total_dim, e.total_dim);
    std::vector<comp> Kiso = {comp(p.K3iso0, 0.0), comp(p.K3iso1, 0.0)};
    k3_2plus1::K3mat_2plus1(K, e.En_c, e.plm_config, e.klm_config,
                            e.total_P, par.atmK, par.atmpi, Kiso,
                            comp(p.K3B, 0.0), comp(p.K3E, 0.0), 'n');
    return K;
}

static void write_matrix(std::ofstream& out, const Eigen::MatrixXcd& m) {
    for (int c = 0; c < m.cols(); ++c) {
        for (int r = 0; r < m.rows(); ++r) {
            const double re = m(r, c).real();
            const double im = m(r, c).imag();
            out.write(reinterpret_cast<const char*>(&re), sizeof(re));
            out.write(reinterpret_cast<const char*>(&im), sizeof(im));
        }
    }
}

static void json_array(std::ofstream& out, const std::vector<std::size_t>& v) {
    out << '[';
    for (std::size_t i = 0; i < v.size(); ++i) out << (i ? "," : "") << v[i];
    out << ']';
}

static void json_array_d(std::ofstream& out, const std::vector<double>& v) {
    out << '[' << std::setprecision(17);
    for (std::size_t i = 0; i < v.size(); ++i) out << (i ? "," : "") << v[i];
    out << ']';
}

static void json_array_i(std::ofstream& out, const std::vector<int>& v) {
    out << '[';
    for (std::size_t i = 0; i < v.size(); ++i) out << (i ? "," : "") << v[i];
    out << ']';
}

static Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto val = [&]() {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
            return std::string(argv[++i]);
        };
        if (k == "--gpu-cache-root") a.root = val();
        else if (k == "--outdir") a.outdir = val();
        else if (k == "--accepted-windows") a.windows = val();
        else if (k == "--Lbyas") a.L = std::stod(val());
        else if (k == "--irrep") a.irrep = val();
        else if (k == "--Emin") a.Emin = std::stod(val());
        else if (k == "--Emax") a.Emax = std::stod(val());
        else if (k == "--coarseN") a.coarseN = std::stoi(val());
        else if (k == "--xi") a.xi = std::stod(val());
        else if (k == "--accepted-windows-sha256") a.windows_sha256 = val();
        else if (k == "--git-commit") a.git_commit = val();
        else if (k == "--K3iso0") a.p.K3iso0 = std::stod(val());
        else if (k == "--K3iso1") a.p.K3iso1 = std::stod(val());
        else if (k == "--K3B") a.p.K3B = std::stod(val());
        else if (k == "--K3E") a.p.K3E = std::stod(val());
        else throw std::runtime_error("unknown option: " + k);
    }
    if (a.windows.empty()) throw std::runtime_error("--accepted-windows is required");
    return a;
}

static void run(const Args& a) {
    fs::create_directories(a.outdir);
    const auto cache_opt = resolve_gpu_cache(a.root, a.L, a.irrep);
    if (!cache_opt) throw std::runtime_error("cache not found");
    const fs::path cache = *cache_opt;
    validate_metadata(cache, a.L, a.irrep, a.xi, a.coarseN, a.Emin, a.Emax);

    const auto windows = read_windows(a.windows);
    const auto rows = select_window_rows(scan_gpu_cache(cache), a.Emin, a.Emax);
    std::set<std::size_t> selected;
    for (const auto& w : windows) {
        if (w.lo > w.hi || w.hi >= rows.size()) throw std::runtime_error("window row range outside grid");
        for (std::size_t i = w.lo; i <= w.hi; ++i) selected.insert(rows[i].raw_index);
    }

    const auto conv = parse_complex_read_convention("variant_04_real_imag_swapped");
    const PhysicsParams par = make_physics(a.L, a.xi, 1);
    FitSettings settings = make_settings(a.L, a.xi, 1, a.irrep);
    settings.scan_E0 = a.Emin;
    settings.scan_E1 = a.Emax;
    settings.coarseN = a.coarseN;
    settings.omp_threads = 1;
    const double scale = std::pow(a.L * a.xi, 6.0);
    const std::vector<std::size_t> ids(selected.begin(), selected.end());

    const fs::path bin_path = a.outdir / "projected_basis_hotwindows.bin";
    std::ofstream bin(bin_path, std::ios::binary);
    if (!bin) throw std::runtime_error("cannot write " + bin_path.string());
    const char magic[8] = {'V','3','3','P','H','W','B','1'};
    const std::uint32_t version = 1;
    const std::uint32_t count = static_cast<std::uint32_t>(ids.size());
    bin.write(magic, 8);
    bin.write(reinterpret_cast<const char*>(&version), 4);
    bin.write(reinterpret_cast<const char*>(&count), 4);

    std::ofstream csv(a.outdir / "projected_basis_hotwindows_validation.csv");
    csv << "row_index,Ecm,Nfull,Nproj,matrix_max_abs_diff,det_direct_real,det_direct_imag,det_basis_real,det_basis_imag,det_abs_diff,det_rel_diff,sign_agreement\n";
    csv << std::setprecision(17);
    std::vector<double> ecms;
    std::vector<int> nproj, dims;
    double max_matrix = 0.0, max_det = 0.0, max_rel = 0.0;
    std::size_t signs = 0;

    for (const std::size_t id : ids) {
        const auto it = std::find_if(rows.begin(), rows.end(),
                                     [&](const GpuRowMeta& r) { return r.raw_index == id; });
        if (it == rows.end()) throw std::runtime_error("selected row missing");

        ScanArgs bridge;
        bridge.gpu_cache_root = a.root;
        bridge.outdir = a.outdir;
        bridge.Lbyas = {a.L};
        bridge.irreps = {a.irrep};
        bridge.Emin = a.Emin;
        bridge.Emax = a.Emax;
        bridge.xi = a.xi;
        bridge.coarseN = a.coarseN;
        bridge.old_scaling = true;
        bridge.complex_read_convention = "variant_04_real_imag_swapped";
        bridge.k3 = a.p;

        const auto direct = assemble_scaled_projected_QC_for_row(
            cache, *it, id, a.L, a.xi, a.irrep, bridge, par, settings, scale, conv);
        const auto [F3inv, Vsel] = load_gpu_row_direct_variant04(cache, it->offset);
        const auto entry = build_cache_entry(static_cast<int>(id), it->Ecm,
                                             parse_label(a.irrep), settings, par, 'n');
        const auto project = [&](const Eigen::MatrixXcd& M) {
            return (Vsel.adjoint() * M * Vsel) / comp(scale, 0.0);
        };
        const K3dfParameters z0{1,0,0,0}, z1{0,1,0,0}, zB{0,0,1,0}, zE{0,0,0,1};
        const Eigen::MatrixXcd F = project(F3inv);
        Eigen::MatrixXcd B0, B1, BB, BE;
        // The four basis terms are independent at fixed row.  This does not
        // alter formulas or ordering; it only overlaps their CPU assembly.
        #pragma omp parallel sections num_threads(4)
        {
            #pragma omp section
            { B0 = project(k3_matrix(entry, par, z0)); }
            #pragma omp section
            { B1 = project(k3_matrix(entry, par, z1)); }
            #pragma omp section
            { BB = project(k3_matrix(entry, par, zB)); }
            #pragma omp section
            { BE = project(k3_matrix(entry, par, zE)); }
        }
        const Eigen::MatrixXcd A = F + a.p.K3iso0 * B0 + a.p.K3iso1 * B1
                                     + a.p.K3B * BB + a.p.K3E * BE;
        const comp db = det_complex(A);
        const comp dd = direct.row.det;
        const double md = (direct.scaled_projected_QC - A).cwiseAbs().maxCoeff();
        const double ad = std::abs(dd - db);
        const double rd = ad / std::max(std::abs(dd), 1e-300);
        const bool sign = (dd.real() > 0) == (db.real() > 0)
                       && (dd.real() < 0) == (db.real() < 0);

        const std::uint64_t row = id;
        const std::uint32_t dim = static_cast<std::uint32_t>(A.rows());
        const std::int32_t nf = it->total_dim, np = it->proj_dim;
        bin.write(reinterpret_cast<const char*>(&row), 8);
        bin.write(reinterpret_cast<const char*>(&it->Ecm), 8);
        bin.write(reinterpret_cast<const char*>(&nf), 4);
        bin.write(reinterpret_cast<const char*>(&np), 4);
        bin.write(reinterpret_cast<const char*>(&dim), 4);
        write_matrix(bin, F);
        write_matrix(bin, B0);
        write_matrix(bin, B1);
        write_matrix(bin, BB);
        write_matrix(bin, BE);

        csv << id << ',' << it->Ecm << ',' << it->total_dim << ',' << it->proj_dim << ','
            << md << ',' << dd.real() << ',' << dd.imag() << ',' << db.real() << ','
            << db.imag() << ',' << ad << ',' << rd << ',' << (sign ? 1 : 0) << '\n';
        ecms.push_back(it->Ecm);
        nproj.push_back(it->proj_dim);
        dims.push_back(A.rows());
        max_matrix = std::max(max_matrix, md);
        max_det = std::max(max_det, ad);
        max_rel = std::max(max_rel, rd);
        signs += sign;
    }
    bin.close();
    csv.close();

    const auto bin_size = fs::file_size(bin_path);
    const fs::path json_path = a.outdir / "projected_basis_hotwindows.json";
    auto write_json = [&](std::uintmax_t json_size) {
        std::ofstream j(json_path);
        j << "{\n"
          << "  \"schema_version\": \"v33p_projected_basis_hotwindow_v1\",\n"
          << "  \"created_utc\": \"" << now_utc() << "\",\n"
          << "  \"git_commit\": \"" << a.git_commit << "\",\n"
          << "  \"Lbyas\": " << a.L << ", \"irrep\": \"" << a.irrep << "\",\n"
          << "  \"source_F3inv_Vsel_cache_path\": \"" << cache.string() << "\",\n"
          << "  \"source_F3inv_Vsel_cache_size_bytes\": " << fs::file_size(cache) << ",\n"
          << "  \"source_F3inv_Vsel_cache_mtime\": " << fs::last_write_time(cache).time_since_epoch().count() << ",\n"
          << "  \"source_F3inv_Vsel_cache_sha256\": null,\n"
          << "  \"accepted_windows_path\": \"" << a.windows.string() << "\",\n"
          << "  \"accepted_windows_sha256\": \"" << a.windows_sha256 << "\",\n"
          << "  \"n_rows_saved\": " << ids.size() << ",\n  \"row_indices\": ";
        json_array(j, ids);
        j << ",\n  \"Ecm_values\": ";
        json_array_d(j, ecms);
        j << ",\n  \"Ecm_min\": " << *std::min_element(ecms.begin(), ecms.end())
          << ", \"Ecm_max\": " << *std::max_element(ecms.begin(), ecms.end())
          << ",\n  \"Nproj_values\": ";
        json_array_i(j, nproj);
        j << ",\n  \"matrix_dimension_values\": ";
        json_array_i(j, dims);
        j << ",\n"
          << "  \"complex_storage_order\": \"column-major\",\n"
          << "  \"complex_convention\": \"variant_04_real_imag_swapped\",\n"
          << "  \"scaling_convention\": \"divided_by_pow_Lxi_6\",\n"
          << "  \"xi\": 3.444, \"atmK\": 0.09698, \"atmpi\": 0.06906,\n"
          << "  \"eta_1\": 1.0, \"eta_2\": 0.5, \"alpha\": 0.5, \"epsilon_h\": 0.0,\n"
          << "  \"max_shell_num\": 20, \"tolerance\": 1e-12, \"parity\": -1,\n"
          << "  \"eig_tol\": 0.05, \"norm_tol\": 1e-12, \"proj_tol\": 1e-10,\n"
          << "  \"waves_vec_1\": [0,1], \"waves_vec_2\": [0],\n"
          << "  \"scatter1_00\": 4.04, \"scatter1_10\": -43.2, \"scatter2_00\": 4.12,\n"
          << "  \"K3_basis_terms\": [\"Fproj\",\"B_iso0\",\"B_iso1\",\"B_B\",\"B_E\"],\n"
          << "  \"K3df_reference_params\": {\"K3iso0\": 73735.840894011912, \"K3iso1\": -972421.14060757787, \"K3B\": 347174.05548116949, \"K3E\": -1226756.7068845264},\n"
          << "  \"validation\": {\"max_matrix_abs_diff\": " << max_matrix
          << ", \"max_det_abs_diff\": " << max_det
          << ", \"max_det_rel_diff\": " << max_rel
          << ", \"sign_agreement\": \"" << signs << "/" << ids.size()
          << "\", \"rows_checked\": " << ids.size() << "},\n"
          << "  \"file_sizes\": {\"bin_size_bytes\": " << bin_size
          << ", \"json_size_bytes\": " << json_size << "},\n"
          << "  \"notes\": \"Raw determinant remains primary. Source-cache SHA256 omitted because the external cache is large; accepted-window SHA256 is recorded.\"\n"
          << "}\n";
    };
    write_json(0);
    write_json(fs::file_size(json_path));

    std::ofstream report(a.outdir / "PROJECTED_BASIS_HOTWINDOW_CACHE_REPORT.md");
    report << "# Projected-basis hot-window cache\n\n"
           << "Status: " << ((max_matrix < 1e-10 && max_det < 1e-15 && signs == ids.size())
                              ? "PROJECTED_BASIS_HOTWINDOW_CACHE_PASS"
                              : "PROJECTED_BASIS_HOTWINDOW_CACHE_FAIL") << "\n\n"
           << "- Sector: L" << a.L << "/" << a.irrep << "\n"
           << "- Rows saved: " << ids.size() << "\n"
           << "- Max matrix abs diff: " << max_matrix << "\n"
           << "- Max raw determinant abs diff: " << max_det << "\n"
           << "- Max raw determinant relative diff: " << max_rel << "\n"
           << "- Sign agreement: " << signs << "/" << ids.size() << "\n"
           << "- Binary size: " << bin_size << " bytes\n"
           << "- JSON size: " << fs::file_size(json_path) << " bytes\n"
           << "- Raw determinant is primary; logdet/logabs are validation-only.\n"
           << "- No cachegen or original-cache regeneration was run.\n";
}

} // namespace v33p_hotwindow_basis

int main(int argc, char** argv) {
    try {
        v33p_hotwindow_basis::run(v33p_hotwindow_basis::parse(argc, argv));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[fatal] " << e.what() << "\n";
        return 1;
    }
}
