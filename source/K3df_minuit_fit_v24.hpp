#ifndef K3DF_MINUIT_FIT_V24_HPP
#define K3DF_MINUIT_FIT_V24_HPP

#include <Eigen/Dense>

#include <Minuit2/FCNBase.h>
#include <Minuit2/FunctionMinimum.h>
#include <Minuit2/MnHesse.h>
#include <Minuit2/MnMigrad.h>
#include <Minuit2/MnPrint.h>
#include <Minuit2/MnUserParameters.h>
#include <Minuit2/MnUserParameterState.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "lattice_data_covariance_cpp.hpp"
#include "F3_cpu_openmp_v24_K3QC_core.hpp"

namespace k3df_fit_v24 {

using MatrixD = Eigen::MatrixXd;
using VecD = std::vector<double>;
using Momentum = std::array<int, 3>;

struct MomentumIrrepSpec
{
    std::string label;       // e.g. "000_A1m"
    Momentum nnP{{0, 0, 0}};
    std::string irrep;       // e.g. "A1u"
    std::string irrep_tag;   // e.g. "A1m"
};

enum class ChiSquareMode
{
    // Recommended when covariance_between_states_szscl21_based returns both covariance and correlation.
    // Uses r_i = (E_data_i - E_model_i)/sigma_i and chi2 = r^T Corr^{-1} r.
    NormalizedResidualInverseCorrelation,

    // Uses unnormalized residuals and chi2 = (E_data - E_model)^T Cov^{-1} (E_data - E_model).
    RawResidualInverseCovariance,

    // Literal form close to the expression in the prompt, but not statistically standard:
    // r_i = (E_data_i - E_model_i)/sigma_i and chi2 = r^T Corr r.
    NormalizedResidualCorrelationNoInverse
};

struct K3dfFitSettings
{
    std::string ensemble;
    double Lval = 20.0;
    double xival = 3.444;
    double energy_cutoff = 0.38;
    std::vector<std::string> list_of_mom = {"000_A1m"};
    int max_state = 10;
    std::string masses_path = ""; // Empty means lattice_data_covariance_cpp.hpp default.
    bool print_found_files = false;

    // v24 QC zero search settings.
    int coarseN = 1000;
    int refineN = 50;
    double scan_E0 = 0.0;
    double scan_E1 = -1.0; // If <= scan_E0, replaced by energy_cutoff.
    int omp_threads = 18;
    char debug = 'n';

    // Physics / v24 parameters. The defaults match the v24 core header.
    double atmpi = 0.06906;
    double atmK = 0.09698;
    double eta_1 = 1.0;
    double eta_2 = 0.5;
    double alpha = 0.5;
    double epsilon_h = 0.0;
    double max_shell_num = 20.0;
    double tolerance = 1.0e-12;
    int parity = -1;
    double eig_tol  = 0.05;
    double norm_tol = 1.0e-12;
    double proj_tol = 1.0e-10;
    bool Q0norm = true;
    bool sort_orbit_flag = false;

    std::vector<int> waves_vec_1 = {0, 1};
    std::vector<int> waves_vec_2 = {0};

    // Minuit initial guesses and steps.
    double K3iso0_guess = 200.0;
    double K3iso1_guess = 400.0;
    double K3B_guess = 0.0;
    double K3E_guess = 0.0;

    double K3iso0_step = 10.0;
    double K3iso1_step = 10.0;
    double K3B_step = 10.0;
    double K3E_step = 10.0;

    bool use_parameter_limits = false;
    double param_lower = -1.0e8;
    double param_upper =  1.0e8;

    ChiSquareMode chi_square_mode = ChiSquareMode::NormalizedResidualInverseCorrelation;

    // If Minuit finds a trial point where the QC solve fails catastrophically, return this.
    double failure_penalty = 1.0e100;

    // Expensive but useful: finite-difference propagation of Minuit parameter covariance to QC energies.
    bool compute_model_energy_covariance = true;
    double finite_difference_relative_step = 1.0e-4;
    double finite_difference_absolute_step = 1.0e-5;
};

struct K3dfParameters
{
    double K3iso0 = 0.0;
    double K3iso1 = 0.0;
    double K3B = 0.0;
    double K3E = 0.0;
};

struct ModelEnergyRow
{
    std::string mom_label;
    Momentum nnP{{0, 0, 0}};
    std::string irrep;
    std::string irrep_tag;
    int state_no = -1;
    double data_Ecm = std::numeric_limits<double>::quiet_NaN();
    double data_err = std::numeric_limits<double>::quiet_NaN();
    double model_Ecm = 0.0;
};

struct K3dfFitResult
{
    K3dfParameters best;
    K3dfParameters error;
    MatrixD parameter_covariance;
    MatrixD parameter_correlation;

    VecD data_energies;
    VecD data_errors;
    VecD model_energies;
    VecD model_energy_errors;
    MatrixD model_energy_covariance;
    MatrixD model_energy_correlation;

    std::vector<Momentum> nP_list;
    std::vector<int> state_no;
    VecD L_list;
    MatrixD data_covariance;
    MatrixD data_correlation;

    std::vector<ModelEnergyRow> rows;
    std::map<std::string, VecD> zeros_by_momentum_irrep;

    double chi2 = std::numeric_limits<double>::quiet_NaN();
    int ndata = 0;
    int npar = 4;
    int ndof = 0;
    double chi2_dof = std::numeric_limits<double>::quiet_NaN();
    bool is_valid = false;
};

inline bool same_momentum(const Momentum& a, const Momentum& b)
{
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
}

inline MomentumIrrepSpec parse_momentum_irrep_label(const std::string& label)
{
    MomentumIrrepSpec spec;
    spec.label = label;

    // Explicit map requested for the KKpi fits.
    // The lattice filenames use labels such as "100_A2", but for the v24
    // projected-QC solve we use the canonical momentum representatives below.
    if (label == "000_A1m")
    {
        spec.nnP = Momentum{{0, 0, 0}};
        spec.irrep = "A1u";
        spec.irrep_tag = "A1m";
        return spec;
    }

    if (label == "100_A2")
    {
        spec.nnP = Momentum{{0, 0, 1}};
        spec.irrep = "A2";
        spec.irrep_tag = "A2";
        return spec;
    }

    if (label == "110_A2")
    {
        spec.nnP = Momentum{{1, 1, 0}};
        spec.irrep = "A2";
        spec.irrep_tag = "A2";
        return spec;
    }

    if (label == "111_A2")
    {
        spec.nnP = Momentum{{1, 1, 1}};
        spec.irrep = "A2";
        spec.irrep_tag = "A2";
        return spec;
    }

    if (label == "200_A2")
    {
        spec.nnP = Momentum{{0, 0, 2}};
        spec.irrep = "A2";
        spec.irrep_tag = "A2";
        return spec;
    }

    throw std::runtime_error(
        "Unsupported momentum/irrep label in parse_momentum_irrep_label: " + label +
        ". Add an explicit mapping for this label."
    );
}

inline Momentum canonical_shell_momentum(Momentum p)
{
    for (int& x : p) x = std::abs(x);
    std::sort(p.begin(), p.end());
    return p;
}

inline bool same_momentum_shell(const Momentum& a, const Momentum& b)
{
    return canonical_shell_momentum(a) == canonical_shell_momentum(b);
}

inline std::string spec_key(const MomentumIrrepSpec& s)
{
    // Use the original user/lattice label as the key so that output remains
    // tied to list_of_mom entries, even when the QC momentum representative is
    // canonicalized, e.g. "100_A2" -> nnP={0,0,1}.
    return s.label;
}

inline MatrixD symmetric_pseudoinverse(const MatrixD& A, const double rcond = 1.0e-12)
{
    if (A.rows() != A.cols())
    {
        throw std::runtime_error("symmetric_pseudoinverse requires a square matrix");
    }
    if (A.rows() == 0)
    {
        return MatrixD(0, 0);
    }

    MatrixD S = 0.5 * (A + A.transpose());
    Eigen::SelfAdjointEigenSolver<MatrixD> es(S);
    if (es.info() != Eigen::Success)
    {
        throw std::runtime_error("Eigen decomposition failed in symmetric_pseudoinverse");
    }

    const auto evals = es.eigenvalues();
    const MatrixD evecs = es.eigenvectors();
    const double max_abs_eval = evals.cwiseAbs().maxCoeff();
    const double cutoff = std::max(rcond * max_abs_eval, 1.0e-300);

    MatrixD Dinv = MatrixD::Zero(A.rows(), A.cols());
    for (int i = 0; i < evals.size(); ++i)
    {
        if (std::abs(evals(i)) > cutoff)
        {
            Dinv(i, i) = 1.0 / evals(i);
        }
    }

    return evecs * Dinv * evecs.transpose();
}

inline MatrixD covariance_to_correlation(const MatrixD& cov)
{
    MatrixD corr = MatrixD::Zero(cov.rows(), cov.cols());
    for (int i = 0; i < cov.rows(); ++i)
    {
        for (int j = 0; j < cov.cols(); ++j)
        {
            const double di = cov(i, i);
            const double dj = cov(j, j);
            if (di > 0.0 && dj > 0.0)
            {
                corr(i, j) = cov(i, j) / std::sqrt(di * dj);
            }
        }
    }
    return corr;
}

inline PhysicsParams make_physics_params_from_settings(
    const K3dfFitSettings& s,
    const K3dfParameters& p)
{
    PhysicsParams par;
    par.atmpi = s.atmpi;
    par.atmK = s.atmK;
    par.eta_1 = s.eta_1;
    par.eta_2 = s.eta_2;
    par.alpha = s.alpha;
    par.epsilon_h = s.epsilon_h;
    par.max_shell_num = s.max_shell_num;
    par.tolerance = s.tolerance;
    par.xi = s.xival;
    par.Lbyas = s.Lval;
    par.Q0norm = s.Q0norm;
    par.sort_orbit_flag = s.sort_orbit_flag;
    par.parity = s.parity;
    par.eig_tol = s.eig_tol;
    par.norm_tol = s.norm_tol;
    par.proj_tol = s.proj_tol;
    par.omp_threads = s.omp_threads;
    par.waves_vec_1 = s.waves_vec_1;
    par.waves_vec_2 = s.waves_vec_2;

    par.K3iso = {comp(p.K3iso0, 0.0), comp(p.K3iso1, 0.0)};
    par.K3B_par = comp(p.K3B, 0.0);
    par.K3E_par = comp(p.K3E, 0.0);

    return par;
}

inline std::vector<int> infer_row_spec_index_from_covariance_order(
    const std::vector<std::string>& list_of_mom,
    const std::vector<Momentum>& nP_list)
{
    std::vector<MomentumIrrepSpec> specs;
    specs.reserve(list_of_mom.size());
    for (const auto& label : list_of_mom)
    {
        specs.push_back(parse_momentum_irrep_label(label));
    }

    std::vector<int> row_spec_index(nP_list.size(), -1);
    std::size_t row = 0;

    for (std::size_t s = 0; s < specs.size() && row < nP_list.size(); ++s)
    {
        while (row < nP_list.size() && same_momentum_shell(nP_list[row], specs[s].nnP))
        {
            row_spec_index[row] = static_cast<int>(s);
            ++row;
        }
    }

    if (row != nP_list.size())
    {
        std::ostringstream os;
        os << "Could not map all covariance rows to list_of_mom labels. "
           << "Mapped " << row << " of " << nP_list.size()
           << ". Check list_of_mom ordering or duplicate nP irreps.";
        throw std::runtime_error(os.str());
    }

    return row_spec_index;
}

inline VecD solve_model_energies_for_rows(
    const K3dfFitSettings& settings,
    const K3dfParameters& params,
    const std::vector<ModelEnergyRow>& row_template,
    std::map<std::string, VecD>* zeros_by_momentum_irrep_out = nullptr)
{
    VecD model(row_template.size(), 0.0);
    if (zeros_by_momentum_irrep_out) zeros_by_momentum_irrep_out->clear();

    std::vector<MomentumIrrepSpec> specs;
    specs.reserve(settings.list_of_mom.size());
    for (const auto& label : settings.list_of_mom)
    {
        specs.push_back(parse_momentum_irrep_label(label));
    }

    const double E0 = settings.scan_E0;
    const double E1 = (settings.scan_E1 > settings.scan_E0) ? settings.scan_E1 : settings.energy_cutoff;
    PhysicsParams par = make_physics_params_from_settings(settings, params);

    for (const MomentumIrrepSpec& spec : specs)
    {
        const std::vector<int> nnP_vec = {spec.nnP[0], spec.nnP[1], spec.nnP[2]};
        VecD zeros = find_likely_zeros_v24_K3QC_silent(
            nnP_vec,
            spec.irrep,
            settings.coarseN,
            settings.refineN,
            E0,
            E1,
            par,
            settings.debug
        );
        std::sort(zeros.begin(), zeros.end());

        const std::string key = spec_key(spec);
        if (zeros_by_momentum_irrep_out) (*zeros_by_momentum_irrep_out)[key] = zeros;

        int local_data_level = 0;
        for (std::size_t r = 0; r < row_template.size(); ++r)
        {
            const ModelEnergyRow& row = row_template[r];
            if (row.mom_label == spec.label)
            {
                if (local_data_level < static_cast<int>(zeros.size()))
                {
                    model[r] = zeros[static_cast<std::size_t>(local_data_level)];
                }
                else
                {
                    // User-requested convention: if QC gives fewer energies than data, pad missing QC levels by 0.
                    model[r] = 0.0;
                }
                ++local_data_level;
            }
        }
    }

    return model;
}

inline double chi_square_from_model(
    const VecD& data,
    const VecD& err,
    const VecD& model,
    const MatrixD& covariance_mat,
    const MatrixD& correlation_mat,
    const ChiSquareMode mode)
{
    const int n = static_cast<int>(data.size());
    if (static_cast<int>(model.size()) != n || static_cast<int>(err.size()) != n)
    {
        throw std::runtime_error("chi_square_from_model size mismatch");
    }

    Eigen::VectorXd r(n);

    if (mode == ChiSquareMode::RawResidualInverseCovariance)
    {
        for (int i = 0; i < n; ++i) r(i) = data[static_cast<std::size_t>(i)] - model[static_cast<std::size_t>(i)];
        MatrixD W = symmetric_pseudoinverse(covariance_mat);
        return (r.transpose() * W * r)(0, 0);
    }

    for (int i = 0; i < n; ++i)
    {
        const double sigma = err[static_cast<std::size_t>(i)];
        if (!(sigma > 0.0) || !std::isfinite(sigma))
        {
            throw std::runtime_error("Non-positive or non-finite data error in chi_square_from_model");
        }
        r(i) = (data[static_cast<std::size_t>(i)] - model[static_cast<std::size_t>(i)]) / sigma;
    }

    if (mode == ChiSquareMode::NormalizedResidualCorrelationNoInverse)
    {
        return (r.transpose() * correlation_mat * r)(0, 0);
    }

    MatrixD W = symmetric_pseudoinverse(correlation_mat);
    return (r.transpose() * W * r)(0, 0);
}

class K3dfFCN final : public ROOT::Minuit2::FCNBase
{
public:
    K3dfFCN(
        K3dfFitSettings settings_,
        VecD data_,
        VecD err_,
        MatrixD covariance_mat_,
        MatrixD correlation_mat_,
        std::vector<ModelEnergyRow> rows_)
        : settings(std::move(settings_)),
          data(std::move(data_)),
          err(std::move(err_)),
          covariance_mat(std::move(covariance_mat_)),
          correlation_mat(std::move(correlation_mat_)),
          rows(std::move(rows_))
    {}

    double Up() const override { return 1.0; }

    double operator()(const std::vector<double>& x) const override
    {
        if (x.size() < 4) return settings.failure_penalty;

        K3dfParameters p;
        p.K3iso0 = x[0];
        p.K3iso1 = x[1];
        p.K3B = x[2];
        p.K3E = x[3];

        try
        {
            VecD model = solve_model_energies_for_rows(settings, p, rows, nullptr);
            const double chi2 = chi_square_from_model(
                data, err, model, covariance_mat, correlation_mat, settings.chi_square_mode
            );

            if (!std::isfinite(chi2)) return settings.failure_penalty;
            return chi2;
        }
        catch (const std::exception& e)
        {
            if (settings.debug == 'y')
            {
                std::cerr << "[K3dfFCN] failed at K3=("
                          << p.K3iso0 << ", " << p.K3iso1 << ", "
                          << p.K3B << ", " << p.K3E << "): "
                          << e.what() << "\n";
            }
            return settings.failure_penalty;
        }
    }

private:
    K3dfFitSettings settings;
    VecD data;
    VecD err;
    MatrixD covariance_mat;
    MatrixD correlation_mat;
    std::vector<ModelEnergyRow> rows;
};

inline MatrixD minuit_covariance_to_eigen(const ROOT::Minuit2::FunctionMinimum& min, const int npar)
{
    MatrixD cov = MatrixD::Zero(npar, npar);
    try
    {
        const auto& mcov = min.UserState().Covariance();
        for (int i = 0; i < npar; ++i)
        {
            for (int j = 0; j < npar; ++j)
            {
                cov(i, j) = mcov(i, j);
            }
        }
    }
    catch (...)
    {
        cov.setConstant(std::numeric_limits<double>::quiet_NaN());
    }
    return cov;
}

inline VecD finite_difference_energy_errors_and_covariance(
    const K3dfFitSettings& settings,
    const K3dfParameters& best,
    const MatrixD& param_cov,
    const std::vector<ModelEnergyRow>& rows,
    MatrixD& model_cov_out,
    MatrixD& model_corr_out)
{
    const int n = static_cast<int>(rows.size());
    const int pnum = 4;
    model_cov_out = MatrixD::Zero(n, n);
    model_corr_out = MatrixD::Zero(n, n);

    if (n == 0 || param_cov.rows() != pnum || param_cov.cols() != pnum)
    {
        return VecD(static_cast<std::size_t>(n), std::numeric_limits<double>::quiet_NaN());
    }

    const VecD best_vec = {best.K3iso0, best.K3iso1, best.K3B, best.K3E};
    MatrixD J = MatrixD::Zero(n, pnum);

    for (int a = 0; a < pnum; ++a)
    {
        const double sigma_a = (param_cov(a, a) > 0.0 && std::isfinite(param_cov(a, a)))
                               ? std::sqrt(param_cov(a, a)) : 0.0;
        double h = settings.finite_difference_relative_step * std::max({1.0, std::abs(best_vec[static_cast<std::size_t>(a)]), sigma_a});
        h = std::max(h, settings.finite_difference_absolute_step);

        K3dfParameters pp = best;
        K3dfParameters pm = best;
        if (a == 0) { pp.K3iso0 += h; pm.K3iso0 -= h; }
        if (a == 1) { pp.K3iso1 += h; pm.K3iso1 -= h; }
        if (a == 2) { pp.K3B    += h; pm.K3B    -= h; }
        if (a == 3) { pp.K3E    += h; pm.K3E    -= h; }

        VecD Ep = solve_model_energies_for_rows(settings, pp, rows, nullptr);
        VecD Em = solve_model_energies_for_rows(settings, pm, rows, nullptr);

        for (int i = 0; i < n; ++i)
        {
            J(i, a) = (Ep[static_cast<std::size_t>(i)] - Em[static_cast<std::size_t>(i)]) / (2.0 * h);
        }
    }

    model_cov_out = J * param_cov * J.transpose();
    model_corr_out = covariance_to_correlation(model_cov_out);

    VecD model_err(static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i)
    {
        model_err[static_cast<std::size_t>(i)] =
            (model_cov_out(i, i) > 0.0) ? std::sqrt(model_cov_out(i, i)) : 0.0;
    }
    return model_err;
}

inline K3dfFitResult fit_K3df_parameters_minuit_v24(const K3dfFitSettings& settings)
{
    auto [states_avg, states_err, nP_list, state_no, L_list, covariance_mat, correlation_mat]
        = covariance_between_states_szscl21_based(
            settings.ensemble,
            settings.Lval,
            settings.xival,
            settings.energy_cutoff,
            settings.list_of_mom,
            settings.max_state,
            settings.masses_path,
            settings.print_found_files
        );

    if (states_avg.empty())
    {
        throw std::runtime_error("No lattice states passed the energy_cutoff / file-existence selection.");
    }

    std::vector<int> row_spec_index = infer_row_spec_index_from_covariance_order(settings.list_of_mom, nP_list);

    std::vector<MomentumIrrepSpec> specs;
    for (const auto& label : settings.list_of_mom) specs.push_back(parse_momentum_irrep_label(label));

    std::vector<ModelEnergyRow> rows(states_avg.size());
    for (std::size_t i = 0; i < states_avg.size(); ++i)
    {
        const int si = row_spec_index[i];
        const MomentumIrrepSpec& spec = specs[static_cast<std::size_t>(si)];
        rows[i].mom_label = spec.label;
        rows[i].nnP = spec.nnP;
        rows[i].irrep = spec.irrep;
        rows[i].irrep_tag = spec.irrep_tag;
        rows[i].state_no = state_no[i];
        rows[i].data_Ecm = states_avg[i];
        rows[i].data_err = states_err[i];
    }

    K3dfFCN fcn(settings, states_avg, states_err, covariance_mat, correlation_mat, rows);

    ROOT::Minuit2::MnUserParameters upar;
    if (settings.use_parameter_limits)
    {
        upar.Add("K3iso0", settings.K3iso0_guess, settings.K3iso0_step, settings.param_lower, settings.param_upper);
        upar.Add("K3iso1", settings.K3iso1_guess, settings.K3iso1_step, settings.param_lower, settings.param_upper);
        upar.Add("K3B",    settings.K3B_guess,    settings.K3B_step,    settings.param_lower, settings.param_upper);
        upar.Add("K3E",    settings.K3E_guess,    settings.K3E_step,    settings.param_lower, settings.param_upper);
    }
    else
    {
        upar.Add("K3iso0", settings.K3iso0_guess, settings.K3iso0_step);
        upar.Add("K3iso1", settings.K3iso1_guess, settings.K3iso1_step);
        upar.Add("K3B",    settings.K3B_guess,    settings.K3B_step);
        upar.Add("K3E",    settings.K3E_guess,    settings.K3E_step);
    }

    ROOT::Minuit2::MnMigrad migrad(fcn, upar);
    ROOT::Minuit2::FunctionMinimum min = migrad();

    if (!min.IsValid())
    {
        if (settings.debug == 'y')
        {
            std::cerr << "[fit_K3df_parameters_minuit_v24] MIGRAD minimum is not valid; running HESSE anyway.\n";
        }
    }

    ROOT::Minuit2::MnHesse hesse;
    hesse(fcn, min);

    K3dfFitResult result;
    result.is_valid = min.IsValid();
    result.ndata = static_cast<int>(states_avg.size());
    result.npar = 4;
    result.ndof = result.ndata - result.npar;

    const auto& pars = min.UserParameters();
    result.best.K3iso0 = pars.Value("K3iso0");
    result.best.K3iso1 = pars.Value("K3iso1");
    result.best.K3B    = pars.Value("K3B");
    result.best.K3E    = pars.Value("K3E");

    result.error.K3iso0 = pars.Error("K3iso0");
    result.error.K3iso1 = pars.Error("K3iso1");
    result.error.K3B    = pars.Error("K3B");
    result.error.K3E    = pars.Error("K3E");

    result.parameter_covariance = minuit_covariance_to_eigen(min, 4);
    result.parameter_correlation = covariance_to_correlation(result.parameter_covariance);

    result.data_energies = states_avg;
    result.data_errors = states_err;
    result.nP_list = nP_list;
    result.state_no = state_no;
    result.L_list = L_list;
    result.data_covariance = covariance_mat;
    result.data_correlation = correlation_mat;

    result.model_energies = solve_model_energies_for_rows(
        settings, result.best, rows, &result.zeros_by_momentum_irrep
    );

    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        rows[i].model_Ecm = result.model_energies[i];
    }
    result.rows = rows;

    result.chi2 = chi_square_from_model(
        result.data_energies,
        result.data_errors,
        result.model_energies,
        result.data_covariance,
        result.data_correlation,
        settings.chi_square_mode
    );

    result.chi2_dof = (result.ndof > 0) ? result.chi2 / double(result.ndof)
                                        : std::numeric_limits<double>::quiet_NaN();

    if (settings.compute_model_energy_covariance)
    {
        result.model_energy_errors = finite_difference_energy_errors_and_covariance(
            settings,
            result.best,
            result.parameter_covariance,
            result.rows,
            result.model_energy_covariance,
            result.model_energy_correlation
        );
    }
    else
    {
        result.model_energy_errors.assign(result.model_energies.size(), 0.0);
        result.model_energy_covariance = MatrixD::Zero(result.ndata, result.ndata);
        result.model_energy_correlation = MatrixD::Zero(result.ndata, result.ndata);
    }

    return result;
}

inline void print_fit_result_summary(const K3dfFitResult& r, std::ostream& os = std::cout)
{
    os << std::setprecision(17);
    os << "# valid = " << int(r.is_valid) << "\n";
    os << "# chi2 = " << r.chi2 << "\n";
    os << "# ndata = " << r.ndata << " npar = " << r.npar
       << " ndof = " << r.ndof << " chi2/ndof = " << r.chi2_dof << "\n";
    os << "# best-fit Kdf3 parameters\n";
    os << "K3iso0 " << r.best.K3iso0 << " +/- " << r.error.K3iso0 << "\n";
    os << "K3iso1 " << r.best.K3iso1 << " +/- " << r.error.K3iso1 << "\n";
    os << "K3B    " << r.best.K3B    << " +/- " << r.error.K3B    << "\n";
    os << "K3E    " << r.best.K3E    << " +/- " << r.error.K3E    << "\n";

    os << "# levels: i mom irrep state_no data_Ecm data_err model_Ecm model_err residual_sigma\n";
    for (std::size_t i = 0; i < r.rows.size(); ++i)
    {
        const double sigma = r.data_errors[i];
        const double pull = (sigma > 0.0) ? (r.data_energies[i] - r.model_energies[i]) / sigma
                                          : std::numeric_limits<double>::quiet_NaN();
        const double model_err = (i < r.model_energy_errors.size()) ? r.model_energy_errors[i] : 0.0;
        os << i << ' '
           << r.rows[i].mom_label << ' '
           << r.rows[i].irrep << ' '
           << r.rows[i].state_no << ' '
           << r.data_energies[i] << ' '
           << r.data_errors[i] << ' '
           << r.model_energies[i] << ' '
           << model_err << ' '
           << pull << "\n";
    }
}

} // namespace k3df_fit_v24

#endif // K3DF_MINUIT_FIT_V24_HPP
