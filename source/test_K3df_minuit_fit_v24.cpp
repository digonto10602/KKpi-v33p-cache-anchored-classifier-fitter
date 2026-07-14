#include "K3df_minuit_fit_v24.hpp"

int main()
{
    using namespace k3df_fit_v24;

    K3dfFitSettings s;
    s.ensemble = "szscl21_24_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265";  // Change this to your actual ensemble prefix.
    s.Lval = 20.0;
    s.xival = 3.444;
    s.energy_cutoff = 0.38;
    s.list_of_mom = {"000_A1m","100_A2","110_A2","111_A2","200_A2"};
    s.max_state = 10;

    // Optional. Leave empty to use the default path from lattice_data_covariance_cpp.hpp.
    // s.masses_path = "/home/digonto/Codes/Practical_Lattice_v2/3body_quantization/lattice_data/KKpi_interacting_spectrum/twoptvar_analysis/masses/";

    // For a first smoke test, keep these small. Increase for production.
    s.coarseN = 1000;
    s.refineN = 10;
    s.scan_E0 = 0.263101;
    s.scan_E1 = 0.36;//s.energy_cutoff;
    s.omp_threads = 18;
    s.debug = 'n';
    s.print_found_files = true;

    // Initial guesses and Minuit steps.
    s.K3iso0_guess = 200.0;
    s.K3iso1_guess = 400.0;
    s.K3B_guess = 10.0;
    s.K3E_guess = -10.0;

    s.K3iso0_step = 20.0;
    s.K3iso1_step = 20.0;
    s.K3B_step = 20.0;
    s.K3E_step = 20.0;

    // Set false for faster first tests. Turn on for final error propagation to model energies.
    s.compute_model_energy_covariance = false;

    K3dfFitResult result = fit_K3df_parameters_minuit_v24(s);
    print_fit_result_summary(result);

    std::cout << "# parameter covariance matrix\n" << result.parameter_covariance << "\n";
    std::cout << "# parameter correlation matrix\n" << result.parameter_correlation << "\n";

    return 0;
}
