#include "K3df_minuit_fit_v24.hpp"

#include <iostream>
#include <iomanip>
#include <exception>

int main()
{
    using namespace k3df_fit_v24;

    try
    {
        K3dfFitSettings s;

        // These are copied from your successful debug_covariance_input_v24 log.
        s.ensemble = "szscl21_24_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265";
        s.Lval = 20.0;
        s.xival = 3.444;
        s.energy_cutoff = 0.34;   // high cutoff for first debug; lower later
        s.max_state = 10;

        s.masses_path =
            "/home/digonto/Codes/Practical_Lattice_v2/3body_quantization/"
            "lattice_data/KKpi_interacting_spectrum/twoptvar_analysis/masses/";

        // First do only one irrep/momentum to keep the QC solve cheap.
        // After this runs, expand to all five momenta.
        s.list_of_mom = {"000_A1m"};

        // Small QC scan for smoke test. Increase after the pipeline works.
        s.coarseN = 50;
        s.refineN = 10;
        s.scan_E0 = 0.26311;
        s.scan_E1 = 0.36;

        s.omp_threads = 18;
        s.debug = 'y';
        s.print_found_files = true;

        // Turn this off for the first run because it finite-differences
        // the full QC solve multiple times and is expensive.
        s.compute_model_energy_covariance = false;

        // Initial Kdf3 parameters.
        s.K3iso0_guess = 200.0;
        s.K3iso1_guess = 400.0;
        s.K3B_guess    = 0.0;
        s.K3E_guess    = 0.0;

        s.K3iso0_step = 10.0;
        s.K3iso1_step = 10.0;
        s.K3B_step    = 10.0;
        s.K3E_step    = 10.0;

        std::cout << std::setprecision(17);
        std::cout << "# Starting small debug fit with one momentum: 000_A1m\n";
        std::cout << "# ensemble    = " << s.ensemble << "\n";
        std::cout << "# masses_path = " << s.masses_path << "\n";
        std::cout << "# cutoff      = " << s.energy_cutoff << "\n";

        K3dfFitResult r = fit_K3df_parameters_minuit_v24(s);
        print_fit_result_summary(r);

        std::cout << "\n# zeros by momentum/irrep\n";
        for (const auto& kv : r.zeros_by_momentum_irrep)
        {
            std::cout << kv.first << " :";
            for (double z : kv.second) std::cout << ' ' << z;
            std::cout << "\n";
        }

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "FATAL in test_K3df_minuit_fit_v24_DEBUG_SMALL: " << e.what() << "\n";
        return 1;
    }
}
