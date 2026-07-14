#include "K3df_minuit_fit_v26_reuse_cache.hpp"

#include <array>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

/*
    generate_likely_zeros_K3df_v26.cpp

    Standalone driver to generate likely-zero energies from the v26/v25 cached
    K3df QC solver for a user-given list_of_mom.

    Output rows:

        Lval nPx nPy nPz irrep irrep_tag zero_index likely_zero_Ecm

    Example:

        ./generate_likely_zeros_K3df_v26 \
            --Lval 20 \
            --xival 3.444 \
            --E0 0.2631 \
            --E1 0.38 \
            --coarseN 300 \
            --refineN 50 \
            --K3iso0 200 \
            --K3iso1 400 \
            --K3B 0 \
            --K3E 0 \
            --out likely_zeros_K3df_v26.dat \
            000_A1m 100_A2 110_A2 111_A2 200_A2
*/

namespace kz = k3df_fit_v26;

static void print_usage(const char* prog)
{
    std::cerr
        << "Usage:\n"
        << "  " << prog << " [options] MOM_LABEL [MOM_LABEL ...]\n\n"
        << "Options:\n"
        << "  --Lval X              default 20\n"
        << "  --xival X             default 3.444\n"
        << "  --E0 X                default 0.2631\n"
        << "  --E1 X                default 0.38\n"
        << "  --coarseN N           default 300\n"
        << "  --refineN N           default 50\n"
        << "  --threads N           default 18\n"
        << "  --debug y|n           default n\n"
        << "  --K3iso0 X            default 200\n"
        << "  --K3iso1 X            default 400\n"
        << "  --K3B X               default 0\n"
        << "  --K3E X               default 0\n"
        << "  --atmpi X             default 0.06906\n"
        << "  --atmK X              default 0.09698\n"
        << "  --parity N            default -1\n"
        << "  --eig_tol X           default 0.05\n"
        << "  --out FILE            default likely_zeros_K3df_v26.dat\n"
        << "  --help\n\n"
        << "Supported MOM_LABEL values from the v26 parser:\n"
        << "  000_A1m 100_A2 110_A2 111_A2 200_A2\n";
}

static double parse_double(const std::string& s, const std::string& name)
{
    try
    {
        std::size_t pos = 0;
        double v = std::stod(s, &pos);
        if (pos != s.size()) throw std::invalid_argument("trailing characters");
        return v;
    }
    catch (...)
    {
        throw std::runtime_error("Could not parse double for " + name + ": " + s);
    }
}

static int parse_int(const std::string& s, const std::string& name)
{
    try
    {
        std::size_t pos = 0;
        int v = std::stoi(s, &pos);
        if (pos != s.size()) throw std::invalid_argument("trailing characters");
        return v;
    }
    catch (...)
    {
        throw std::runtime_error("Could not parse int for " + name + ": " + s);
    }
}

int main(int argc, char** argv)
{
    std::cout << std::setprecision(17);

    kz::K3dfFitSettings s;

    s.Lval = 20.0;
    s.xival = 3.444;
    s.scan_E0 = 0.2631;
    s.scan_E1 = 0.38;
    s.energy_cutoff = s.scan_E1;
    s.coarseN = 300;
    s.refineN = 50;
    s.omp_threads = 18;
    s.debug = 'n';

    kz::K3dfParameters k3;
    k3.K3iso0 = 200.0;
    k3.K3iso1 = 400.0;
    k3.K3B = 0.0;
    k3.K3E = 0.0;

    std::string out_file = "likely_zeros_K3df_v26.dat";
    std::vector<std::string> list_of_mom;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        auto need_value = [&](const std::string& opt) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value after " + opt);
            return std::string(argv[++i]);
        };

        if (arg == "--help" || arg == "-h")
        {
            print_usage(argv[0]);
            return 0;
        }
        else if (arg == "--Lval")      { s.Lval = parse_double(need_value(arg), arg); }
        else if (arg == "--xival")     { s.xival = parse_double(need_value(arg), arg); }
        else if (arg == "--E0")        { s.scan_E0 = parse_double(need_value(arg), arg); }
        else if (arg == "--E1")        { s.scan_E1 = parse_double(need_value(arg), arg); s.energy_cutoff = s.scan_E1; }
        else if (arg == "--coarseN")   { s.coarseN = parse_int(need_value(arg), arg); }
        else if (arg == "--refineN")   { s.refineN = parse_int(need_value(arg), arg); }
        else if (arg == "--threads")   { s.omp_threads = parse_int(need_value(arg), arg); }
        else if (arg == "--debug")
        {
            std::string v = need_value(arg);
            if (v.empty() || (v[0] != 'y' && v[0] != 'n')) throw std::runtime_error("--debug must be y or n");
            s.debug = v[0];
        }
        else if (arg == "--K3iso0")    { k3.K3iso0 = parse_double(need_value(arg), arg); }
        else if (arg == "--K3iso1")    { k3.K3iso1 = parse_double(need_value(arg), arg); }
        else if (arg == "--K3B")       { k3.K3B = parse_double(need_value(arg), arg); }
        else if (arg == "--K3E")       { k3.K3E = parse_double(need_value(arg), arg); }
        else if (arg == "--atmpi")     { s.atmpi = parse_double(need_value(arg), arg); }
        else if (arg == "--atmK")      { s.atmK = parse_double(need_value(arg), arg); }
        else if (arg == "--parity")    { s.parity = parse_int(need_value(arg), arg); }
        else if (arg == "--eig_tol")   { s.eig_tol = parse_double(need_value(arg), arg); }
        else if (arg == "--out")       { out_file = need_value(arg); }
        else if (!arg.empty() && arg[0] == '-')
        {
            throw std::runtime_error("Unknown option: " + arg);
        }
        else
        {
            list_of_mom.push_back(arg);
        }
    }

    if (list_of_mom.empty()) list_of_mom = {"000_A1m"};

    if (!(s.scan_E1 > s.scan_E0)) throw std::runtime_error("Need E1 > E0");

    s.list_of_mom = list_of_mom;
    s.use_f3i_projected_cache = true;

    PhysicsParams par = kz::make_physics_params_from_settings(s, k3);
    F3iProjectedCache cache;

    std::ofstream fout(out_file);
    if (!fout) throw std::runtime_error("Could not open output file: " + out_file);

    auto write_header = [&](std::ostream& os)
    {
        os << std::setprecision(17);
        os << "# likely zeros from projected QC = Vsel^dagger (F3^{-1} + Kdf3) Vsel\n";
        os << "# Lval = " << s.Lval << "\n";
        os << "# xival = " << s.xival << "\n";
        os << "# L = Lval*xival = " << s.Lval * s.xival << "\n";
        os << "# E0 = " << s.scan_E0 << "\n";
        os << "# E1 = " << s.scan_E1 << "\n";
        os << "# coarseN = " << s.coarseN << "\n";
        os << "# refineN = " << s.refineN << "\n";
        os << "# K3iso0 = " << k3.K3iso0 << "\n";
        os << "# K3iso1 = " << k3.K3iso1 << "\n";
        os << "# K3B = " << k3.K3B << "\n";
        os << "# K3E = " << k3.K3E << "\n";
        os << "# columns: Lval nPx nPy nPz irrep irrep_tag zero_index likely_zero_Ecm\n";
    };

    write_header(fout);
    write_header(std::cout);

    for (const std::string& label : list_of_mom)
    {
        kz::MomentumIrrepSpec spec = kz::parse_momentum_irrep_label(label);
        std::vector<int> nnP_vec = {spec.nnP[0], spec.nnP[1], spec.nnP[2]};

        std::vector<double> zeros = find_likely_zeros_v25_K3QC_cached(
            nnP_vec,
            spec.irrep,
            s.coarseN,
            s.refineN,
            s.scan_E0,
            s.scan_E1,
            par,
            &cache,
            s.debug
        );

        std::sort(zeros.begin(), zeros.end());

        if (zeros.empty())
        {
            std::cerr << "[warning] no likely zeros found for label=" << label
                      << " nnP={" << spec.nnP[0] << "," << spec.nnP[1] << "," << spec.nnP[2] << "}"
                      << " irrep=" << spec.irrep
                      << " in Ecm range [" << s.scan_E0 << "," << s.scan_E1 << "]\n";
        }

        for (std::size_t iz = 0; iz < zeros.size(); ++iz)
        {
            auto write_row = [&](std::ostream& os)
            {
                os << std::setprecision(17)
                   << s.Lval << ' '
                   << spec.nnP[0] << ' '
                   << spec.nnP[1] << ' '
                   << spec.nnP[2] << ' '
                   << spec.irrep << ' '
                   << spec.irrep_tag << ' '
                   << iz << ' '
                   << zeros[iz] << '\n';
            };

            write_row(fout);
            write_row(std::cout);
        }
    }

    std::cerr << "[done] wrote " << out_file
              << " cache_size=" << cache.size()
              << " cache_hits=" << cache.hits
              << " cache_misses=" << cache.misses
              << "\n";

    return 0;
}
