#include "lattice_data_covariance_cpp.hpp"
#include <iostream>
#include <string>
#include <vector>

int main() {
    std::string ensemble1 = "szscl21_24_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265";  // change to your ensemble basename
    double Lval1 = 20.0;
    double xival1 = 3.444;
    double energy_cutoff = 0.38;
    int max_state = 10;

    std::vector<std::string> list_of_mom = {
        "000_A1m", "100_A2", "110_A2", "111_A2", "200_A2"
    };

    auto [states_avg1, states_err1, nP_list1, state_no1, L_list1,
          covariance_mat1, correlation_mat1]
        = covariance_between_states_szscl21_based(
            ensemble1, Lval1, xival1, energy_cutoff, list_of_mom, max_state);

    print_covariance_summary(states_avg1, states_err1, nP_list1, state_no1,
                             L_list1, covariance_mat1, correlation_mat1);

    return 0;
}
