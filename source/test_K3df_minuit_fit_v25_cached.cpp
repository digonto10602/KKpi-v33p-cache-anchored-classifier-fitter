#include "K3df_minuit_fit_v25_cached.hpp"

#include <iostream>

int main()
{
    using namespace k3df_fit_v24;

    K3dfFitSettings s;

    s.ensemble = "szscl21_24_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265";
    s.masses_path =
        "/home/digonto/Codes/Practical_Lattice_v2/3body_quantization/"
        "lattice_data/KKpi_interacting_spectrum/twoptvar_analysis/masses/";

    s.Lval = 20.0;
    s.xival = 3.444;

    // Start small. Increase list_of_mom and cutoff after the fixed run behaves well.
    s.list_of_mom = {"000_A1m"};
    s.max_state = 2;
    s.energy_cutoff = 0.34;

    // Scan slightly wider than the selected data range.
    s.scan_E0 = 0.26312;
    s.scan_E1 = 0.36;

    s.coarseN = 1000;
    s.refineN = 10;
    s.omp_threads = 18;

    // Keep low-level determinant/K3 logs off. The v25 FCN summary still prints.
    s.debug = 'y';
    s.print_found_files = true;
    s.print_each_fcn_eval = true;
    s.print_every_fcn_eval = 1;
    s.use_f3i_projected_cache = true;

    // Turn this off for first tests. It performs extra QC solves after the fit.
    s.compute_model_energy_covariance = false;

    s.K3iso0_guess = 200.0;
    s.K3iso1_guess = 400.0;
    s.K3B_guess = 0.0;
    s.K3E_guess = 0.0;

    s.K3iso0_step = 10.0;
    s.K3iso1_step = 10.0;
    s.K3B_step = 10.0;
    s.K3E_step = 10.0;

    try
    {
        auto result = fit_K3df_parameters_minuit_v24(s);
        print_fit_result_summary(result);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Fitter failed: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
