#define V32W_NO_MAIN
#define V33G_NO_MAIN
#include "qc_fitter_norm_refine_v2_multiL.cpp"
#undef V32W_NO_MAIN
#undef V33G_NO_MAIN

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

namespace fs = std::filesystem;
using namespace k3df_fit_v32f;

namespace {

static std::string ltag(double L) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(6) << L;
    std::string s = os.str();
    while(!s.empty() && s.back() == '0') s.pop_back();
    if(!s.empty() && s.back() == '.') s.pop_back();
    for(char& c : s) if(c == '.') c = 'p';
    return s;
}

static std::string block_tag(double L, const std::string& irrep) {
    return std::string("L") + ltag(L) + "_" + irrep;
}

struct MatrixDiff {
    double max_abs = 0.0;
    double max_rel = 0.0;
    int row = -1;
    int col = -1;
};

static double matrix_scale(const Eigen::MatrixXcd& m) {
    double s = 0.0;
    for(int r = 0; r < m.rows(); ++r) {
        for(int c = 0; c < m.cols(); ++c) {
            s = std::max(s, std::abs(m(r, c)));
        }
    }
    return s;
}

static MatrixDiff matrix_diff(const Eigen::MatrixXcd& a, const Eigen::MatrixXcd& b) {
    MatrixDiff d;
    if(a.rows() != b.rows() || a.cols() != b.cols()) {
        d.max_abs = std::numeric_limits<double>::infinity();
        d.max_rel = std::numeric_limits<double>::infinity();
        return d;
    }
    for(int r = 0; r < a.rows(); ++r) {
        for(int c = 0; c < a.cols(); ++c) {
            const double da = std::abs(a(r, c) - b(r, c));
            const double scale = std::max({1.0, std::abs(a(r, c)), std::abs(b(r, c))});
            const double rel = da / scale;
            if(da > d.max_abs) {
                d.max_abs = da;
                d.row = r;
                d.col = c;
            }
            if(rel > d.max_rel) d.max_rel = rel;
        }
    }
    return d;
}

static void write_text_file(const fs::path& path, const std::string& text) {
    fs::create_directories(path.parent_path());
    std::ofstream os(path);
    if(!os) throw std::runtime_error("Could not open " + path.string());
    os << text;
}

static std::vector<v32x_multiL::RuntimeBlock> build_blocks(const v32x_multiL::MultiConfig& cfg,
                                                           const std::vector<v32x_multiL::BlockInfo>& blocks,
                                                           bool load_refined_cache) {
    std::vector<v32x_multiL::RuntimeBlock> out;
    out.reserve(blocks.size());
    for(const auto& b : blocks) {
        out.push_back(v32x_multiL::build_runtime_block(cfg, b, load_refined_cache));
    }
    return out;
}

static std::vector<double> model_from_zero_lists(const std::vector<v32x_multiL::BlockInfo>& blocks,
                                                const std::vector<TargetLevel>& targets,
                                                const std::vector<std::vector<double>>& zero_lists) {
    std::vector<double> model(targets.size(), 0.0);
    for(std::size_t bi = 0; bi < blocks.size(); ++bi) {
        const auto& block = blocks[bi];
        auto zeros = zero_lists[bi];
        std::sort(zeros.begin(), zeros.end());
        for(std::size_t k = 0; k < block.target_indices.size(); ++k) {
            const int gi = block.target_indices[k];
            if(k < zeros.size()) model[std::size_t(gi)] = zeros[k];
        }
    }
    return model;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::string cfgpath = (argc > 1 ? std::string(argv[1]) : std::string("configs/config_v33g_validate_runtime_vs_v33f.in"));
        auto cfg = v32x_multiL::multiconfig_from_config(cfgpath);
        auto cfg_raw = cfg;
        auto cfg_runtime = cfg;
        cfg_raw.cache_backend = "raw_v33e";
        cfg_raw.require_existing_v33g_runtime_cache = false;
        cfg_runtime.cache_backend = "v33g_runtime";
        cfg_runtime.require_existing_v33g_runtime_cache = true;
        cfg_runtime.build_v33g_runtime_if_missing = false;
        fs::create_directories("output_v33g/validation");

        std::cout << "[v33g-validate] config=" << cfgpath << "\n";
        auto [mt, targets, cov, corr, blocks] = v32x_multiL::load_multil_jack_targets(cfg);
        std::cout << "[v33g-validate] targets=" << targets.size() << " blocks=" << blocks.size() << "\n";
        for(const auto& b : blocks) {
            std::cout << "[v33g-validate] " << block_tag(b.L, b.internal_irrep)
                      << " nlevels=" << b.target_indices.size() << "\n";
        }

        auto raw_blocks = build_blocks(cfg_raw, blocks, false);
        auto rt_blocks = build_blocks(cfg_runtime, blocks, false);
        const K3dfParameters kp{cfg.base.guess.K3iso0, cfg.base.guess.K3iso1, cfg.base.guess.K3B, cfg.base.guess.K3E};
        const double coarse_spacing = (cfg.base.coarseN > 1)
            ? std::abs(cfg.base.scan_E1 - cfg.base.scan_E0) / double(cfg.base.coarseN - 1)
            : 0.0;
        const double zero_tol = std::max(2.0 * coarse_spacing, 1e-12);
        std::vector<std::vector<double>> raw_zero_lists(blocks.size());
        std::vector<std::vector<double>> hot_zero_lists(blocks.size());

        bool all_ok = true;
        for(std::size_t bi = 0; bi < blocks.size(); ++bi) {
            const auto& block = blocks[bi];
            auto& raw = raw_blocks[bi];
            auto& hot = rt_blocks[bi];
            const std::string tag = block_tag(block.L, block.internal_irrep);
            bool block_ok = true;
            bool basis_mismatch = false;

            if(raw.coarse.grid.size() != hot.coarse.grid.size()) {
                block_ok = false;
                all_ok = false;
                std::ostringstream os;
                os << "[v33g-validate] " << tag
                   << " row_count raw=" << raw.coarse.grid.size()
                   << " runtime=" << hot.coarse.grid.size() << "\n";
                write_text_file(fs::path("output_v33g/validation") / (tag + "_sign_sequence_compare.dat"), os.str());
                std::cout << os.str();
                continue;
            }

            std::vector<int> raw_signs;
            std::vector<int> hot_signs;
            std::vector<double> raw_zeros;
            std::vector<double> hot_zeros;
            for(std::size_t i = 0; i < raw.coarse.grid.size(); ++i) {
                const auto& re = raw.coarse.grid[i];
                const auto& he = hot.coarse.grid[i];
                if(re.i != he.i || std::abs(re.Ecm - he.Ecm) > 1e-12 || re.total_dim != he.total_dim || re.proj_dim != he.proj_dim || re.success != he.success) {
                    block_ok = false;
                    all_ok = false;
                    std::ostringstream os;
                    os << std::setprecision(17)
                       << "[v33g-validate] " << tag << " row mismatch i=" << i
                       << " raw(i,Ecm,td,pd,s)=" << re.i << "," << re.Ecm << "," << re.total_dim << "," << re.proj_dim << "," << re.success
                       << " runtime(i,Ecm,td,pd,s)=" << he.i << "," << he.Ecm << "," << he.total_dim << "," << he.proj_dim << "," << he.success << "\n";
                    write_text_file(fs::path("output_v33g/validation") / (tag + "_first_matrix_mismatch.dat"), os.str());
                    std::cout << os.str();
                    break;
                }

                if(re.success && he.success) {
                    const auto dA = matrix_diff(re.F3inv_proj, he.F3inv_proj);
                    const auto dB0 = matrix_diff(re.K3_proj_basis[0], he.K3_proj_basis[0]);
                    const auto dB1 = matrix_diff(re.K3_proj_basis[1], he.K3_proj_basis[1]);
                    const auto dBB = matrix_diff(re.K3_proj_basis[2], he.K3_proj_basis[2]);
                    const auto dBE = matrix_diff(re.K3_proj_basis[3], he.K3_proj_basis[3]);
                    const double scale = std::max(1.0, std::max(matrix_scale(re.F3inv_proj), matrix_scale(he.F3inv_proj)));
                    const double tol = 1e-10 * scale;
                    if(dA.max_abs > tol || dB0.max_abs > tol || dB1.max_abs > tol || dBB.max_abs > tol || dBE.max_abs > tol) {
                        basis_mismatch = true;
                        std::ostringstream os;
                        os << std::setprecision(17)
                           << "[v33g-validate] " << tag << " matrix mismatch row=" << i
                           << " A0_abs=" << dA.max_abs << " A0_rel=" << dA.max_rel
                           << " B0_abs=" << dB0.max_abs << " B0_rel=" << dB0.max_rel
                           << " B1_abs=" << dB1.max_abs << " B1_rel=" << dB1.max_rel
                           << " BB_abs=" << dBB.max_abs << " BB_rel=" << dBB.max_rel
                           << " BE_abs=" << dBE.max_abs << " BE_rel=" << dBE.max_rel
                           << " raw_B0_00=" << re.K3_proj_basis[0](0,0)
                           << " hot_B0_00=" << he.K3_proj_basis[0](0,0)
                           << " B0_mismatch_rc=" << dB0.row << "," << dB0.col
                           << " raw_B0_rc=" << re.K3_proj_basis[0](dB0.row,dB0.col)
                           << " hot_B0_rc=" << he.K3_proj_basis[0](dB0.row,dB0.col)
                           << " raw_B1_00=" << re.K3_proj_basis[1](0,0)
                           << " hot_B1_00=" << he.K3_proj_basis[1](0,0)
                           << " raw_BB_00=" << re.K3_proj_basis[2](0,0)
                           << " hot_BB_00=" << he.K3_proj_basis[2](0,0)
                           << " raw_BE_00=" << re.K3_proj_basis[3](0,0)
                           << " hot_BE_00=" << he.K3_proj_basis[3](0,0)
                           << "\n";
                        write_text_file(fs::path("output_v33g/validation") / (tag + "_first_matrix_mismatch.dat"), os.str());
                        std::cout << os.str();
                    }
                }

                const auto raw_eval = v32w::eval_entry_QC(re, kp, raw.par, cfg.base.debug, raw.cscale);
                const auto hot_eval = v32w::eval_entry_QC(he, kp, hot.par, cfg.base.debug, hot.cscale);
                raw_signs.push_back(raw_eval.det.sign);
                hot_signs.push_back(hot_eval.det.sign);
            }

            if(block_ok) {
                std::vector<k3df_fit_v32f::ProjectedQCCacheEntry> new_entries;
                const auto raw_cands = v32w::find_QC_zeros_refined(raw.coarse, raw.refined, raw.settings, raw.par, kp, raw.cscale, cfg.ninside, cfg.maxdepth, cfg.zratio, new_entries);
                new_entries.clear();
                const auto hot_cands = v32w::find_QC_zeros_refined(hot.coarse, hot.refined, hot.settings, hot.par, kp, hot.cscale, cfg.ninside, cfg.maxdepth, cfg.zratio, new_entries);

                for(const auto& c : raw_cands) if(c.kind == "true_zero") raw_zeros.push_back(c.E);
                for(const auto& c : hot_cands) if(c.kind == "true_zero") hot_zeros.push_back(c.E);
                std::sort(raw_zeros.begin(), raw_zeros.end());
                std::sort(hot_zeros.begin(), hot_zeros.end());
                raw_zero_lists[bi] = raw_zeros;
                hot_zero_lists[bi] = hot_zeros;

                if(raw_signs != hot_signs) {
                    block_ok = false;
                    all_ok = false;
                    std::ostringstream os;
                    os << "[v33g-validate] " << tag << " sign sequence mismatch\n";
                    os << "# idx raw hot\n";
                    for(std::size_t i = 0; i < raw_signs.size(); ++i) {
                        if(raw_signs[i] != hot_signs[i]) os << i << " " << raw_signs[i] << " " << hot_signs[i] << "\n";
                    }
                    write_text_file(fs::path("output_v33g/validation") / (tag + "_sign_sequence_compare.dat"), os.str());
                    std::cout << os.str();
                }

                if(raw_zeros.size() != hot_zeros.size()) {
                    block_ok = false;
                    all_ok = false;
                    std::ostringstream os;
                    os << "[v33g-validate] " << tag << " zero count mismatch raw=" << raw_zeros.size() << " runtime=" << hot_zeros.size() << "\n";
                    write_text_file(fs::path("output_v33g/validation") / (tag + "_zero_list_compare.dat"), os.str());
                    std::cout << os.str();
                } else {
                    for(std::size_t i = 0; i < raw_zeros.size(); ++i) {
                        if(std::abs(raw_zeros[i] - hot_zeros[i]) > zero_tol) {
                            block_ok = false;
                            all_ok = false;
                            std::ostringstream os;
                            os << std::setprecision(17)
                               << "[v33g-validate] " << tag << " zero mismatch idx=" << i
                               << " raw=" << raw_zeros[i]
                               << " runtime=" << hot_zeros[i]
                               << " tol=" << zero_tol << "\n";
                            write_text_file(fs::path("output_v33g/validation") / (tag + "_zero_list_compare.dat"), os.str());
                            std::cout << os.str();
                            break;
                        }
                    }
                }

                std::cout << std::setprecision(17)
                          << "[v33g-validate] " << tag
                          << " rows=" << raw.coarse.grid.size()
                          << " raw_true_zeros=" << raw_zeros.size()
                          << " hot_true_zeros=" << hot_zeros.size()
                          << " block_ok=" << (block_ok ? 1 : 0) << "\n";
            }

            if(basis_mismatch) {
                std::cout << "[v33g-validate] " << tag << " basis_mismatch=1\n";
            }
        }

        const auto raw_model = model_from_zero_lists(blocks, targets, raw_zero_lists);
        const auto hot_model = model_from_zero_lists(blocks, targets, hot_zero_lists);
        int raw_found = 0;
        int hot_found = 0;
        for(double m : raw_model) if(std::isfinite(m) && m > 0.0) ++raw_found;
        for(double m : hot_model) if(std::isfinite(m) && m > 0.0) ++hot_found;
        const double chi_raw = chi_square_v32f(targets, raw_model, cov, corr, cfg.base.chi_square_mode, cfg.base.failure_penalty);
        const double chi_hot = chi_square_v32f(targets, hot_model, cov, corr, cfg.base.chi_square_mode, cfg.base.failure_penalty);
        std::cout << std::setprecision(17)
                  << "[v33g-validate] model_levels_found_raw=" << raw_found
                  << " model_levels_found_runtime=" << hot_found
                  << " chi_raw=" << chi_raw
                  << " chi_hot=" << chi_hot
                  << "\n";
        if(raw_found != hot_found || std::abs(chi_raw - chi_hot) > 1e-8 * std::max(1.0, std::abs(chi_raw))) all_ok = false;

        std::cout << "[v33g-validate] overall=" << (all_ok ? 1 : 0) << "\n";
        return all_ok ? 0 : 2;
    } catch(const std::exception& e) {
        std::cerr << "[v33g-validate-error] " << e.what() << "\n";
        return 1;
    }
}
