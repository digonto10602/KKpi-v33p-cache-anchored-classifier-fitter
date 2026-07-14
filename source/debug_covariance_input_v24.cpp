#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "lattice_data_covariance_cpp.hpp"

namespace fs = std::filesystem;

static std::array<int,3> mom_label_to_nP_python_covariance_convention(const std::string& mom)
{
    // This reproduces the convention used in the original Python covariance code,
    // not the canonical QC representative used in the v24 fitter.
    if (mom == "000_A1m") return {0,0,0};
    if (mom == "100_A2")  return {1,0,0};
    if (mom == "110_A2")  return {1,1,0};
    if (mom == "111_A2")  return {1,1,1};
    if (mom == "200_A2")  return {2,0,0};
    throw std::runtime_error("Unknown momentum label in debug file: " + mom);
}

static std::vector<double> read_second_column_skip_header(const std::string& filename)
{
    std::ifstream fin(filename);
    if (!fin)
    {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::vector<double> data;
    std::string line;

    // Python used np.genfromtxt(..., skip_header=1), so skip exactly one line.
    std::getline(fin, line);

    while (std::getline(fin, line))
    {
        if (line.empty()) continue;
        if (line[0] == '#') continue;

        std::istringstream iss(line);
        double col0 = 0.0;
        double col1 = 0.0;
        if (iss >> col0 >> col1)
        {
            data.push_back(col1);
        }
    }

    return data;
}

static double jackknife_average_local(const std::vector<double>& x)
{
    if (x.empty()) throw std::runtime_error("jackknife_average_local called with empty data");
    double s = 0.0;
    for (double v : x) s += v;
    return s / static_cast<double>(x.size());
}

static std::vector<double> jackknife_resampling_local(const std::vector<double>& data)
{
    const int n = static_cast<int>(data.size());
    if (n < 2) throw std::runtime_error("Need at least two jackknife samples");

    double total = 0.0;
    for (double v : data) total += v;

    std::vector<double> jk(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        jk[static_cast<std::size_t>(i)] = (total - data[static_cast<std::size_t>(i)]) / double(n - 1);
    }
    return jk;
}

static double E_to_Ecm_local(double En, double P)
{
    const double arg = En*En - P*P;
    if (arg < 0.0)
    {
        std::cerr << "WARNING: En^2 - P^2 < 0: En=" << En << " P=" << P << " arg=" << arg << "\n";
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::sqrt(arg);
}

int main(int argc, char** argv)
{
    std::cout << std::setprecision(17);

    std::string ensemble = "szscl21_24_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265";
    double Lval = 20.0;
    double xival = 3.444;
    double energy_cutoff = 1.0;
    int max_state = 10;
    std::string masses_path = "";

    std::vector<std::string> list_of_mom = {"000_A1m","100_A2","110_A2","111_A2","200_A2"};

    if (argc >= 2) ensemble = argv[1];
    if (argc >= 3) Lval = std::stod(argv[2]);
    if (argc >= 4) xival = std::stod(argv[3]);
    if (argc >= 5) energy_cutoff = std::stod(argv[4]);
    if (argc >= 6) max_state = std::stoi(argv[5]);
    if (argc >= 7) masses_path = argv[6];
    if (argc >= 8)
    {
        list_of_mom.clear();
        for (int i = 7; i < argc; ++i) list_of_mom.push_back(argv[i]);
    }

    if (masses_path.empty())
    {
        masses_path = "/home/digonto/Codes/Practical_Lattice_v2/3body_quantization/"
                      "lattice_data/KKpi_interacting_spectrum/twoptvar_analysis/masses/";
    }
    if (!masses_path.empty() && masses_path.back() != '/') masses_path.push_back('/');

    const double pi = std::acos(-1.0);
    const double Lbyas = Lval * xival;
    const double twopibyLbyas = 2.0 * pi / Lbyas;

    std::cout << "# Debugging covariance input selection\n";
    std::cout << "ensemble      = " << ensemble << "\n";
    std::cout << "Lval          = " << Lval << "\n";
    std::cout << "xival         = " << xival << "\n";
    std::cout << "Lbyas         = " << Lbyas << "\n";
    std::cout << "energy_cutoff = " << energy_cutoff << "\n";
    std::cout << "max_state     = " << max_state << "\n";
    std::cout << "masses_path   = " << masses_path << "\n";
    std::cout << "path_exists   = " << fs::exists(masses_path) << "\n";
    std::cout << "# list_of_mom =";
    for (const auto& m : list_of_mom) std::cout << ' ' << m;
    std::cout << "\n\n";

    int found_files = 0;
    int selected_states = 0;

    for (const std::string& mom : list_of_mom)
    {
        const auto nP = mom_label_to_nP_python_covariance_convention(mom);
        const double Px = twopibyLbyas * nP[0];
        const double Py = twopibyLbyas * nP[1];
        const double Pz = twopibyLbyas * nP[2];
        const double P = std::sqrt(Px*Px + Py*Py + Pz*Pz);

        std::cout << "# MOM " << mom << " covariance-nP={"
                  << nP[0] << ',' << nP[1] << ',' << nP[2]
                  << "} |P|=" << P << "\n";

        for (int state = 0; state < max_state; ++state)
        {
            const std::string filename = masses_path + ensemble + "_" + mom + "_state_" + std::to_string(state);
            const bool exists = fs::exists(filename);

            std::cout << "checking state=" << state << " exists=" << int(exists)
                      << " file=" << filename;

            if (!exists)
            {
                std::cout << "\n";
                continue;
            }

            ++found_files;

            try
            {
                std::vector<double> data = read_second_column_skip_header(filename);
                std::cout << " nrows=" << data.size();

                if (data.size() < 2)
                {
                    std::cout << " SELECT=NO reason='too few rows'\n";
                    continue;
                }

                std::vector<double> jk = jackknife_resampling_local(data);
                const double avg_En = jackknife_average_local(jk);
                const double avg_Ecm = E_to_Ecm_local(avg_En, P);
                const bool pass = std::isfinite(avg_Ecm) && (avg_Ecm < energy_cutoff);

                if (pass) ++selected_states;

                std::cout << " avg_En=" << avg_En
                          << " avg_Ecm=" << avg_Ecm
                          << " cutoff=" << energy_cutoff
                          << " SELECT=" << (pass ? "YES" : "NO") << "\n";
            }
            catch (const std::exception& e)
            {
                std::cout << " ERROR='" << e.what() << "'\n";
            }
        }
        std::cout << "\n";
    }

    std::cout << "# summary found_files=" << found_files
              << " selected_states=" << selected_states << "\n\n";

    std::cout << "# Now calling covariance_between_states_szscl21_based(...) directly\n";
    try
    {
        auto [states_avg, states_err, nP_list, state_no, L_list, covariance_mat, correlation_mat]
            = covariance_between_states_szscl21_based(
                ensemble,
                Lval,
                xival,
                energy_cutoff,
                list_of_mom,
                max_state,
                masses_path,
                true
            );

        std::cout << "# covariance function returned N=" << states_avg.size() << "\n";
        for (std::size_t i = 0; i < states_avg.size(); ++i)
        {
            std::cout << i
                      << " Ecm=" << states_avg[i]
                      << " err=" << states_err[i]
                      << " nP={" << nP_list[i][0] << ',' << nP_list[i][1] << ',' << nP_list[i][2] << '}'
                      << " state_no=" << state_no[i]
                      << " L=" << L_list[i]
                      << "\n";
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "# covariance_between_states_szscl21_based threw: " << e.what() << "\n";
        return 2;
    }

    return 0;
}
