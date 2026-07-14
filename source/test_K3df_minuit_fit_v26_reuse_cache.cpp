#include "K3df_minuit_fit_v26_reuse_cache.hpp"

#include <iostream>

int main()
{
    using namespace k3df_fit_v26;

    K3dfFitSettings s;

    s.ensemble = "szscl21_24_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265";
    s.masses_path =
        "/home/digonto/Codes/Practical_Lattice_v2/3body_quantization/"
        "lattice_data/KKpi_interacting_spectrum/twoptvar_analysis/masses/";

    s.Lval = 20.0;
    s.xival = 3.444;

    s.list_of_mom = {"000_A1m","100_A2","110_A2","111_A2","200_A2"};
    s.max_state = 8;
    s.energy_cutoff = 0.335;

    s.scan_E0 = 0.26310;
    s.scan_E1 = 0.36;

    s.coarseN = 1000;
    s.refineN = 10;
    s.omp_threads = 18;

    s.debug = 'y';
    s.print_found_files = true;
    s.print_each_fcn_eval = true;
    s.print_every_fcn_eval = 1;
    s.use_f3i_projected_cache = true;

    s.compute_model_energy_covariance = true;

    s.K3iso0_guess = 183464;
    s.K3iso1_guess = -786421.0;
    s.K3B_guess = 100.0;
    s.K3E_guess = -100.0;

    s.K3iso0_step = 10.0;
    s.K3iso1_step = 10.0;
    s.K3B_step = 10.0;
    s.K3E_step = 10.0;

    s.write_output_files = true;
    s.output_dir = ".";
    s.output_tag = "v26"; // generates fit_summary_v25.dat, etc.

    try
    {
        auto result = fit_K3df_parameters_minuit_v26(s);
        print_fit_result_summary(result);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Fitter failed: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
