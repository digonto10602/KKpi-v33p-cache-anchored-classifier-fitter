#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

struct SignFlipCandidate
{
    int index_left;
    int index_right;

    double Ecm1;
    double Ecm2;
    double Ecm_zero_linear;

    double y1;
    double y2;

    double abs_gap;
    double endpoint_max_abs;

    double local_max_abs;
    double local_median_abs;
    double spike_ratio;

    double local_max_slope;
    double local_median_slope;
    double slope_ratio;

    std::string classification;
};

double median_of_vector(std::vector<double> vals)
{
    if (vals.empty())
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    std::sort(vals.begin(), vals.end());

    const std::size_t n = vals.size();

    if (n % 2 == 1)
    {
        return vals[n / 2];
    }
    else
    {
        return 0.5 * (vals[n / 2 - 1] + vals[n / 2]);
    }
}

std::vector<SignFlipCandidate> classify_sign_flips_zero_vs_pole(
    const std::string& input_filename,
    const std::string& output_filename = "classified_zero_pole_candidates.dat",
    int window = 5,
    double small_y_threshold = 1.0e-3,
    double spike_ratio_threshold = 1.0e2,
    double slope_ratio_threshold = 1.0e2,
    char debug = 'n'
)
{
    std::ifstream fin(input_filename.c_str());

    if (!fin.is_open())
    {
        throw std::runtime_error("Could not open input file: " + input_filename);
    }

    std::vector<double> Ecm_vec;
    std::vector<double> y_vec;

    std::string line;

    while (std::getline(fin, line))
    {
        if (line.empty())
        {
            continue;
        }

        if (line[0] == '#')
        {
            continue;
        }

        std::istringstream iss(line);

        int i_file;
        double Ecm_real;
        double Ecm_imag;
        double det_norm_real;
        double det_norm_imag;
        double abs_det_norm;

        if (!(iss >> i_file
                  >> Ecm_real
                  >> Ecm_imag
                  >> det_norm_real
                  >> det_norm_imag
                  >> abs_det_norm))
        {
            continue;
        }

        if (!std::isfinite(Ecm_real) || !std::isfinite(det_norm_real))
        {
            continue;
        }

        Ecm_vec.push_back(Ecm_real);
        y_vec.push_back(det_norm_real);
    }

    fin.close();

    if (Ecm_vec.size() != y_vec.size())
    {
        throw std::runtime_error("Ecm_vec and y_vec size mismatch.");
    }

    if (Ecm_vec.size() < 2)
    {
        throw std::runtime_error("Need at least two points to search for sign flips.");
    }

    const int N = (int)y_vec.size();

    std::vector<double> slope_vec(N - 1, std::numeric_limits<double>::quiet_NaN());

    for (int i = 0; i < N - 1; ++i)
    {
        double dx = Ecm_vec[i + 1] - Ecm_vec[i];
        double dy = y_vec[i + 1] - y_vec[i];

        if (std::abs(dx) > 0.0)
        {
            slope_vec[i] = std::abs(dy / dx);
        }
    }

    std::vector<SignFlipCandidate> candidates;

    int num_likely_zero = 0;
    int num_likely_pole = 0;
    int num_ambiguous = 0;
    int num_zero_but_check = 0;

    for (int i = 0; i < N - 1; ++i)
    {
        double x1 = Ecm_vec[i];
        double x2 = Ecm_vec[i + 1];

        double y1 = y_vec[i];
        double y2 = y_vec[i + 1];

        if (!std::isfinite(x1) || !std::isfinite(x2) ||
            !std::isfinite(y1) || !std::isfinite(y2))
        {
            continue;
        }

        bool exact_zero_1 = (y1 == 0.0);
        bool exact_zero_2 = (y2 == 0.0);
        bool sign_flip = (y1 * y2 < 0.0);

        if (!sign_flip && !exact_zero_1 && !exact_zero_2)
        {
            continue;
        }

        double Ecm_zero_linear;

        if (exact_zero_1)
        {
            Ecm_zero_linear = x1;
        }
        else if (exact_zero_2)
        {
            Ecm_zero_linear = x2;
        }
        else
        {
            Ecm_zero_linear = x1 - y1 * (x2 - x1) / (y2 - y1);
        }

        double abs_gap = std::abs(y2 - y1);
        double endpoint_max_abs = std::max(std::abs(y1), std::abs(y2));

        int lo = std::max(0, i - window);
        int hi = std::min(N - 1, i + window + 1);

        std::vector<double> local_abs_vals;

        for (int j = lo; j <= hi; ++j)
        {
            if (std::isfinite(y_vec[j]))
            {
                local_abs_vals.push_back(std::abs(y_vec[j]));
            }
        }

        double local_max_abs = 0.0;

        for (double val : local_abs_vals)
        {
            if (val > local_max_abs)
            {
                local_max_abs = val;
            }
        }

        double local_median_abs = median_of_vector(local_abs_vals);

        if (!std::isfinite(local_median_abs) || local_median_abs == 0.0)
        {
            local_median_abs = 1.0e-300;
        }

        double spike_ratio = local_max_abs / local_median_abs;

        int slo = std::max(0, i - window);
        int shi = std::min((int)slope_vec.size() - 1, i + window);

        std::vector<double> local_slopes;

        for (int j = slo; j <= shi; ++j)
        {
            if (j >= 0 && j < (int)slope_vec.size() && std::isfinite(slope_vec[j]))
            {
                local_slopes.push_back(slope_vec[j]);
            }
        }

        double local_max_slope = 0.0;

        for (double val : local_slopes)
        {
            if (val > local_max_slope)
            {
                local_max_slope = val;
            }
        }

        double local_median_slope = median_of_vector(local_slopes);

        if (!std::isfinite(local_median_slope) || local_median_slope == 0.0)
        {
            local_median_slope = 1.0e-300;
        }

        double slope_ratio = local_max_slope / local_median_slope;

        bool both_endpoints_small = endpoint_max_abs < small_y_threshold;
        bool local_region_small = local_max_abs < small_y_threshold;

        bool has_spike = spike_ratio > spike_ratio_threshold;
        bool has_slope_spike = slope_ratio > slope_ratio_threshold;

        std::string classification;

        if (both_endpoints_small && local_region_small && !has_spike && !has_slope_spike)
        {
            classification = "likely_zero";
            num_likely_zero += 1;
        }
        else if (has_spike || has_slope_spike)
        {
            classification = "likely_pole";
            num_likely_pole += 1;
        }
        else if (endpoint_max_abs < small_y_threshold)
        {
            classification = "likely_zero_but_check";
            num_zero_but_check += 1;
        }
        else
        {
            classification = "ambiguous_refine_grid";
            num_ambiguous += 1;
        }

        SignFlipCandidate cand;

        cand.index_left = i;
        cand.index_right = i + 1;

        cand.Ecm1 = x1;
        cand.Ecm2 = x2;
        cand.Ecm_zero_linear = Ecm_zero_linear;

        cand.y1 = y1;
        cand.y2 = y2;

        cand.abs_gap = abs_gap;
        cand.endpoint_max_abs = endpoint_max_abs;

        cand.local_max_abs = local_max_abs;
        cand.local_median_abs = local_median_abs;
        cand.spike_ratio = spike_ratio;

        cand.local_max_slope = local_max_slope;
        cand.local_median_slope = local_median_slope;
        cand.slope_ratio = slope_ratio;

        cand.classification = classification;

        candidates.push_back(cand);

        if (debug == 'y')
        {
            std::cout << std::setprecision(17);
            std::cout << "candidate found: "
                      << "i = " << i
                      << ", Ecm1 = " << x1
                      << ", Ecm2 = " << x2
                      << ", Ecm_zero = " << Ecm_zero_linear
                      << ", y1 = " << y1
                      << ", y2 = " << y2
                      << ", abs_gap = " << abs_gap
                      << ", spike_ratio = " << spike_ratio
                      << ", slope_ratio = " << slope_ratio
                      << ", classification = " << classification
                      << std::endl;
        }
    }

    auto class_rank = [](const std::string& s)
    {
        if (s == "likely_zero") return 0;
        if (s == "likely_zero_but_check") return 1;
        if (s == "ambiguous_refine_grid") return 2;
        if (s == "likely_pole") return 3;
        return 99;
    };

    std::sort(
        candidates.begin(),
        candidates.end(),
        [&](const SignFlipCandidate& a, const SignFlipCandidate& b)
        {
            int ra = class_rank(a.classification);
            int rb = class_rank(b.classification);

            if (ra != rb)
            {
                return ra < rb;
            }

            return a.abs_gap < b.abs_gap;
        }
    );

    std::ofstream fout(output_filename.c_str());

    if (!fout.is_open())
    {
        throw std::runtime_error("Could not open output file: " + output_filename);
    }

    fout << std::setprecision(17);

    fout << "# index_left"
         << '\t' << "index_right"
         << '\t' << "Ecm1"
         << '\t' << "Ecm2"
         << '\t' << "Ecm_zero_linear"
         << '\t' << "y1"
         << '\t' << "y2"
         << '\t' << "abs_gap"
         << '\t' << "endpoint_max_abs"
         << '\t' << "local_max_abs"
         << '\t' << "local_median_abs"
         << '\t' << "spike_ratio"
         << '\t' << "local_max_slope"
         << '\t' << "local_median_slope"
         << '\t' << "slope_ratio"
         << '\t' << "classification"
         << '\n';

    for (const auto& c : candidates)
    {
        fout << c.index_left
             << '\t' << c.index_right
             << '\t' << c.Ecm1
             << '\t' << c.Ecm2
             << '\t' << c.Ecm_zero_linear
             << '\t' << c.y1
             << '\t' << c.y2
             << '\t' << c.abs_gap
             << '\t' << c.endpoint_max_abs
             << '\t' << c.local_max_abs
             << '\t' << c.local_median_abs
             << '\t' << c.spike_ratio
             << '\t' << c.local_max_slope
             << '\t' << c.local_median_slope
             << '\t' << c.slope_ratio
             << '\t' << c.classification
             << '\n';
    }

    fout.close();

    if (debug == 'y')
    {
        std::cout << "__________" << std::endl;
        std::cout << "Sign-flip classification summary" << std::endl;
        std::cout << "input file        = " << input_filename << std::endl;
        std::cout << "output file       = " << output_filename << std::endl;
        std::cout << "total candidates  = " << candidates.size() << std::endl;
        std::cout << "likely zeros      = " << num_likely_zero << std::endl;
        std::cout << "zero but check    = " << num_zero_but_check << std::endl;
        std::cout << "likely poles      = " << num_likely_pole << std::endl;
        std::cout << "ambiguous         = " << num_ambiguous << std::endl;
        std::cout << "__________" << std::endl;
    }

    return candidates;
}