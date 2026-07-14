#ifndef LATTICE_DATA_COVARIANCE_CPP_HPP
#define LATTICE_DATA_COVARIANCE_CPP_HPP

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// -----------------------------------------------------------------------------
// C++ rewrite of:
//   jackknife.py
//   lattice_data_covariance.py::covariance_between_states_szscl21_based(...)
//
// Intended use:
//   auto [states_avg1, states_err1, nP_list1, state_no1, L_list1,
//         covariance_mat1, correlation_mat1]
//       = covariance_between_states_szscl21_based(
//             ensemble1, Lval1, xival1, energy_cutoff,
//             list_of_mom, max_state);
//
// Requires: C++17 and Eigen.
// Compile example:
//   g++ -std=c++17 -O3 -I/usr/include/eigen3 your_code.cpp -o your_code
// -----------------------------------------------------------------------------

namespace lattice_covariance {

using VecD = std::vector<double>;
using VecI = std::vector<int>;
using MomentumList = std::vector<std::array<int, 3>>;
using MatrixD = Eigen::MatrixXd;

struct LatticeFileAuditRow {
    std::string label;
    int state = -1;
    std::array<int,3> nP{0,0,0};
    double atP = 0.0;
    std::string filename;
    int found = 0;
    int n_samples = 0;
    double jk_avg_read = std::numeric_limits<double>::quiet_NaN();
    double jk_err_read = std::numeric_limits<double>::quiet_NaN();
    double jk_avg_ecm = std::numeric_limits<double>::quiet_NaN();
    double jk_err_ecm = std::numeric_limits<double>::quiet_NaN();
    int keep = 0;
    std::string keep_reason;
};

struct CovarianceResult {
    VecD states_avg;
    VecD states_err;
    MomentumList nP_list;
    VecI state_no;
    VecD L_list;
    MatrixD covariance_mat;
    MatrixD correlation_mat;
    VecD states_read_avg;
    VecD states_read_err;
    VecD P_list;
    std::string lattice_energy_type;
    std::vector<LatticeFileAuditRow> file_audit;
};

// Structured-binding-friendly return type matching the Python output order.
using CovarianceTuple = std::tuple<
    VecD,          // states_avg
    VecD,          // states_err
    MomentumList,  // nP_list
    VecI,          // state_no
    VecD,          // L_list
    MatrixD,       // covariance_mat
    MatrixD        // correlation_mat
>;

inline double E_to_Ecm(const double En, const double P) {
    const double arg = En * En - P * P;
    if (arg < 0.0 && arg > -1.0e-14) {
        return 0.0;
    }
    if (arg < 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::sqrt(arg);
}

inline double Esq_to_Ecmsq(const double En, const double P) {
    return En * En - P * P;
}

inline double Ecmsq_to_Esq(const double Ecm, const double P) {
    return Ecm * Ecm + P * P;
}


inline std::string normalize_lattice_energy_type(std::string t) {
    std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c){ return char(std::tolower(c)); });
    if (t == "ecm" || t == "cm" || t == "center_of_mass" || t == "centre_of_mass") return "Ecm";
    if (t == "en_lab" || t == "lab" || t == "elab" || t == "en" || t == "energy_lab") return "En_lab";
    throw std::runtime_error("Unknown lattice_energy_type='" + t + "'. Use Ecm or En_lab.");
}

inline VecD convert_jackknife_energy_to_ecm(const VecD& energy_jk, const double P, const std::string& lattice_energy_type) {
    const std::string typ = normalize_lattice_energy_type(lattice_energy_type);
    if (typ == "Ecm") return energy_jk;
    VecD out(energy_jk.size(), 0.0);
    for (std::size_t i = 0; i < energy_jk.size(); ++i) {
        out[i] = E_to_Ecm(energy_jk[i], P);
    }
    return out;
}

inline VecD jackknife_resampling(const VecD& data) {
    const std::size_t n = data.size();
    if (n < 2) {
        throw std::runtime_error("jackknife_resampling requires at least 2 data points");
    }

    double total = 0.0;
    for (const double x : data) total += x;

    VecD resampled(n, 0.0);
    const double denom = static_cast<double>(n - 1);
    for (std::size_t i = 0; i < n; ++i) {
        resampled[i] = (total - data[i]) / denom;
    }
    return resampled;
}

inline double jackknife_average(const VecD& resampled_data) {
    if (resampled_data.empty()) {
        throw std::runtime_error("jackknife_average received empty data");
    }
    double sum = 0.0;
    for (const double x : resampled_data) sum += x;
    return sum / static_cast<double>(resampled_data.size());
}

inline double jackknife_error(const VecD& resampled_data) {
    const std::size_t n = resampled_data.size();
    if (n < 2) {
        throw std::runtime_error("jackknife_error requires at least 2 data points");
    }

    const double avg = jackknife_average(resampled_data);
    double sum = 0.0;
    for (const double x : resampled_data) {
        const double dx = x - avg;
        sum += dx * dx;
    }
    return std::sqrt((static_cast<double>(n - 1) / static_cast<double>(n)) * sum);
}

inline std::filesystem::path default_threebody_path() {
#ifdef __APPLE__
    return std::filesystem::path("/Users/digonto/GitHub/3body_quantization");
#else
    return std::filesystem::path("/home/digonto/Codes/Practical_Lattice_v2/3body_quantization");
#endif
}

inline std::filesystem::path default_szscl21_mass_path() {
    return default_threebody_path()
        / "lattice_data/KKpi_interacting_spectrum/twoptvar_analysis/masses";
}

inline std::array<int, 3> nP_from_mom_label(const std::string& moms) {
    // Matches the explicit Python mapping first.
    if (moms == "000_A1m") return {0, 0, 0};
    if (moms == "100_A2")  return {1, 0, 0};
    if (moms == "110_A2")  return {1, 1, 0};
    if (moms == "111_A2")  return {1, 1, 1};
    if (moms == "200_A2")  return {2, 0, 0};

    // Generic fallback for labels like "210_A2", "211_A2", etc.
    // It reads the substring before the first underscore and expects 3 digits.
    const std::size_t us = moms.find('_');
    const std::string p = (us == std::string::npos) ? moms : moms.substr(0, us);
    if (p.size() >= 3 && std::isdigit(p[0]) && std::isdigit(p[1]) && std::isdigit(p[2])) {
        return {p[0] - '0', p[1] - '0', p[2] - '0'};
    }

    throw std::runtime_error("Unknown momentum irrep label: " + moms);
}

inline double momentum_magnitude_from_nP(
    const std::array<int, 3>& nP,
    const double Lval,
    const double xival
) {
    const double pi = std::acos(-1.0);
    const double Lbyas = Lval * xival;
    const double twopibyLbyas = 2.0 * pi / Lbyas;

    const double Px = twopibyLbyas * static_cast<double>(nP[0]);
    const double Py = twopibyLbyas * static_cast<double>(nP[1]);
    const double Pz = twopibyLbyas * static_cast<double>(nP[2]);
    return std::sqrt(Px * Px + Py * Py + Pz * Pz);
}

inline VecD read_second_column_skip_header(const std::filesystem::path& filename) {
    std::ifstream fin(filename);
    if (!fin) {
        throw std::runtime_error("Could not open file: " + filename.string());
    }

    std::string line;
    // Python used np.genfromtxt(..., skip_header=1), so skip one line.
    std::getline(fin, line);

    VecD data;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') continue;

        std::istringstream iss(line);
        double col0 = 0.0;
        double col1 = 0.0;
        if (iss >> col0 >> col1) {
            data.push_back(col1);
        }
    }

    if (data.empty()) {
        throw std::runtime_error("No numeric second-column data found in file: " + filename.string());
    }

    return data;
}

inline bool same_nP(const std::array<int, 3>& a, const std::array<int, 3>& b) {
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
}

inline CovarianceResult covariance_between_states_szscl21_based_result(
    const std::string& ensemble,
    const double Lval,
    const double xival,
    const double energy_cutoff,
    const std::vector<std::string>& list_of_mom,
    const int max_state,
    const std::filesystem::path& path_to_files = default_szscl21_mass_path(),
    const bool verbose = true,
    const std::string& lattice_energy_type_in = "Ecm"
) {
    if (max_state < 0) {
        throw std::runtime_error("max_state must be nonnegative");
    }
    const std::string lattice_energy_type = normalize_lattice_energy_type(lattice_energy_type_in);
    if (verbose) {
        std::cout << "[lattice-covariance] lattice_energy_type=" << lattice_energy_type
                  << " (Ecm: use file values directly; En_lab: convert sqrt(E^2-P^2))\n";
    }

    std::vector<std::filesystem::path> state_file_list;
    MomentumList nP_list;
    std::vector<LatticeFileAuditRow> file_audit;

    // First pass: find files and select states whose jackknife-average Ecm is below cutoff.
    for (const std::string& moms : list_of_mom) {
        const std::array<int, 3> nP = nP_from_mom_label(moms);
        const double P = momentum_magnitude_from_nP(nP, Lval, xival);

        for (int state = 0; state < max_state; ++state) {
            const std::filesystem::path filename =
                path_to_files / (ensemble + "_" + moms + "_state_" + std::to_string(state));

            if (!std::filesystem::exists(filename)) {
                if (verbose) {
                    std::cout << "[lattice-covariance] missing file = " << filename.string() << '\n';
                }
                continue;
            }

            if (verbose) {
                std::cout << "found file = " << filename.string() << '\n';
            }

            const VecD data = read_second_column_skip_header(filename);
            const VecD resampled_data = jackknife_resampling(data);
            const double avgtemp = jackknife_average(resampled_data);
            const double errtemp = jackknife_error(resampled_data);
            const VecD ecm_jk_tmp = convert_jackknife_energy_to_ecm(resampled_data, P, lattice_energy_type);
            const double avgtemp_Ecm = jackknife_average(ecm_jk_tmp);
            const double errtemp_Ecm = jackknife_error(ecm_jk_tmp);

            LatticeFileAuditRow ar;
            ar.label = moms;
            ar.state = state;
            ar.nP = nP;
            ar.atP = P;
            ar.filename = filename.string();
            ar.found = 1;
            ar.n_samples = static_cast<int>(data.size());
            ar.jk_avg_read = avgtemp;
            ar.jk_err_read = errtemp;
            ar.jk_avg_ecm = avgtemp_Ecm;
            ar.jk_err_ecm = errtemp_Ecm;
            ar.keep = (avgtemp_Ecm < energy_cutoff) ? 1 : 0;
            ar.keep_reason = ar.keep ? "KEPT_BELOW_CUTOFF" : "REJECTED_ABOVE_CUTOFF";
            file_audit.push_back(ar);

            if (verbose) {
                std::cout << "[lattice-covariance] audit label=" << moms
                          << " state=" << state
                          << " n=" << data.size()
                          << " E_read_jk_avg=" << std::setprecision(17) << avgtemp
                          << " Ecm_used=" << avgtemp_Ecm
                          << " cutoff=" << energy_cutoff
                          << " keep=" << ar.keep << '\n';
            }

            if (ar.keep) {
                state_file_list.push_back(filename);
                nP_list.push_back(nP);
            }
        }
    }

    const int total_states = static_cast<int>(state_file_list.size());

    VecI state_no;
    state_no.reserve(static_cast<std::size_t>(total_states));
    if (total_states > 0) {
        int temp_state_no = 0;
        std::array<int, 3> temp_nP = nP_list[0];
        for (int i = 0; i < total_states; ++i) {
            if (i == 0) {
                state_no.push_back(temp_state_no);
                temp_state_no += 1;
                temp_nP = nP_list[0];
            } else {
                const std::array<int, 3>& current_nP = nP_list[static_cast<std::size_t>(i)];
                if (same_nP(temp_nP, current_nP)) {
                    state_no.push_back(temp_state_no);
                    temp_state_no += 1;
                } else {
                    temp_state_no = 0;
                    state_no.push_back(temp_state_no);
                    temp_state_no += 1;
                }
                temp_nP = current_nP;
            }
        }
    }

    MatrixD covariance_matrix = MatrixD::Zero(total_states, total_states);
    MatrixD correlation_matrix = MatrixD::Zero(total_states, total_states);
    VecD states_avg(static_cast<std::size_t>(total_states), 0.0);
    VecD states_err(static_cast<std::size_t>(total_states), 0.0);
    VecD states_read_avg(static_cast<std::size_t>(total_states), 0.0);
    VecD states_read_err(static_cast<std::size_t>(total_states), 0.0);
    VecD P_list(static_cast<std::size_t>(total_states), 0.0);
    VecD L_list(static_cast<std::size_t>(total_states), Lval);

    // Cache jackknife Ecm samples, averages, and errors once per selected state.
    std::vector<VecD> jk_ecm(static_cast<std::size_t>(total_states));
    for (int i = 0; i < total_states; ++i) {
        const VecD data = read_second_column_skip_header(state_file_list[static_cast<std::size_t>(i)]);
        const VecD resampled = jackknife_resampling(data);
        const double P = momentum_magnitude_from_nP(nP_list[static_cast<std::size_t>(i)], Lval, xival);
        P_list[static_cast<std::size_t>(i)] = P;
        states_read_avg[static_cast<std::size_t>(i)] = jackknife_average(resampled);
        states_read_err[static_cast<std::size_t>(i)] = jackknife_error(resampled);
        jk_ecm[static_cast<std::size_t>(i)] = convert_jackknife_energy_to_ecm(resampled, P, lattice_energy_type);
        states_avg[static_cast<std::size_t>(i)] = jackknife_average(jk_ecm[static_cast<std::size_t>(i)]);
        states_err[static_cast<std::size_t>(i)] = jackknife_error(jk_ecm[static_cast<std::size_t>(i)]);
    }

    for (int i = 0; i < total_states; ++i) {
        for (int j = 0; j < total_states; ++j) {
            const VecD& x = jk_ecm[static_cast<std::size_t>(i)];
            const VecD& y = jk_ecm[static_cast<std::size_t>(j)];
            if (x.size() != y.size()) {
                throw std::runtime_error(
                    "Selected files do not have the same jackknife sample count: "
                    + state_file_list[static_cast<std::size_t>(i)].string()
                    + " and "
                    + state_file_list[static_cast<std::size_t>(j)].string()
                );
            }

            const std::size_t n = x.size();
            if (n < 2) {
                throw std::runtime_error("Need at least 2 jackknife samples for covariance");
            }

            const double avg_i = states_avg[static_cast<std::size_t>(i)];
            const double avg_j = states_avg[static_cast<std::size_t>(j)];
            const double err_i = states_err[static_cast<std::size_t>(i)];
            const double err_j = states_err[static_cast<std::size_t>(j)];

            double sum_cov = 0.0;
            double sum_corr = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                const double dx = x[k] - avg_i;
                const double dy = y[k] - avg_j;
                sum_cov += dx * dy;

                if (err_i != 0.0 && err_j != 0.0) {
                    sum_corr += (dx / err_i) * (dy / err_j);
                } else {
                    sum_corr = std::numeric_limits<double>::quiet_NaN();
                    break;
                }
            }

            const double prefactor = static_cast<double>(n - 1) / static_cast<double>(n);
            covariance_matrix(i, j) = prefactor * sum_cov;
            correlation_matrix(i, j) = std::isnan(sum_corr)
                ? std::numeric_limits<double>::quiet_NaN()
                : prefactor * sum_corr;
        }
    }

    return CovarianceResult{
        states_avg,
        states_err,
        nP_list,
        state_no,
        L_list,
        covariance_matrix,
        correlation_matrix,
        states_read_avg,
        states_read_err,
        P_list,
        lattice_energy_type,
        file_audit
    };
}

inline CovarianceTuple covariance_between_states_szscl21_based(
    const std::string& ensemble,
    const double Lval,
    const double xival,
    const double energy_cutoff,
    const std::vector<std::string>& list_of_mom,
    const int max_state,
    const std::filesystem::path& path_to_files = default_szscl21_mass_path(),
    const bool verbose = true,
    const std::string& lattice_energy_type = "Ecm"
) {
    CovarianceResult r = covariance_between_states_szscl21_based_result(
        ensemble, Lval, xival, energy_cutoff, list_of_mom, max_state,
        path_to_files, verbose, lattice_energy_type
    );

    return CovarianceTuple{
        std::move(r.states_avg),
        std::move(r.states_err),
        std::move(r.nP_list),
        std::move(r.state_no),
        std::move(r.L_list),
        std::move(r.covariance_mat),
        std::move(r.correlation_mat)
    };
}

inline void print_covariance_summary(
    const VecD& states_avg,
    const VecD& states_err,
    const MomentumList& nP_list,
    const VecI& state_no,
    const VecD& L_list,
    const MatrixD& covariance_mat,
    const MatrixD& correlation_mat,
    std::ostream& os = std::cout
) {
    os << "total states = " << states_avg.size() << '\n';
    os << "i  avg_Ecm  err_Ecm  nPx nPy nPz  state_no  L\n";
    for (std::size_t i = 0; i < states_avg.size(); ++i) {
        os << i << ' '
           << states_avg[i] << ' '
           << states_err[i] << ' '
           << nP_list[i][0] << ' '
           << nP_list[i][1] << ' '
           << nP_list[i][2] << ' '
           << state_no[i] << ' '
           << L_list[i] << '\n';
    }

    os << "\ncovariance matrix:\n" << covariance_mat << "\n";
    os << "\ncorrelation matrix:\n" << correlation_mat << "\n";
}

} // namespace lattice_covariance

// Optional global wrappers so you can call the function with the exact Python-like name.
using lattice_covariance::CovarianceTuple;
using lattice_covariance::MatrixD;
using lattice_covariance::MomentumList;
using lattice_covariance::VecD;
using lattice_covariance::VecI;
using lattice_covariance::covariance_between_states_szscl21_based;
using lattice_covariance::covariance_between_states_szscl21_based_result;
using lattice_covariance::print_covariance_summary;

#endif // LATTICE_DATA_COVARIANCE_CPP_HPP
