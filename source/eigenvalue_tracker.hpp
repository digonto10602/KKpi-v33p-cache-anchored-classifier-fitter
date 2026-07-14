// eigenvalue_tracker.hpp
#pragma once
#include "projections_from_config.hpp"
#include <vector>
#include <algorithm>
#include <iomanip>

//==============================================================================
// Single energy step: returns eigenvalues + eigenvectors for one irrep
//==============================================================================
struct EigResult {
    double              E;
    std::vector<double> eigenvalues;  // sorted by abs value
    MatC                eigenvectors; // columns = eigenvectors, same order
};

EigResult compute_eigs_at_E(
    const MatC&         M_I,
    double              E)
{
    EigResult res;
    res.E = E;

    if (M_I.cols() == 0) return res;  // empty irrep

    Eigen::SelfAdjointEigenSolver<MatC> eig(M_I);

    // eig.eigenvalues() already sorted ascending (real)
    // eig.eigenvectors() columns correspond to eigenvalues
    int n = (int)eig.eigenvalues().size();

    // Sort by absolute value (matching Python convention)
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){
        return std::abs(eig.eigenvalues()(a)) < std::abs(eig.eigenvalues()(b));
    });

    res.eigenvalues.resize(n);
    res.eigenvectors.resize(M_I.rows(), n);
    for (int i = 0; i < n; i++) {
        res.eigenvalues[i]      = eig.eigenvalues()(idx[i]);
        res.eigenvectors.col(i) = eig.eigenvectors().col(idx[i]);
    }
    return res;
}

//==============================================================================
// Build overlap matrix between two sets of eigenvectors
// overlap(i,j) = |V_prev[:,i]^dag * V_curr[:,j]|
//==============================================================================
Eigen::MatrixXd overlap_matrix(const MatC& V_prev, const MatC& V_curr) {
    // Both must have same number of rows (basis size)
    // May have different number of columns if subspace dim changed
    int n_prev = (int)V_prev.cols();
    int n_curr = (int)V_curr.cols();
    Eigen::MatrixXd ovlp(n_prev, n_curr);
    for (int i = 0; i < n_prev; i++)
        for (int j = 0; j < n_curr; j++)
            ovlp(i,j) = std::abs(V_prev.col(i).dot(V_curr.col(j)));
    return ovlp;
}

//==============================================================================
// Greedy matching: for each previous level i, find best matching current level j
// Returns perm: perm[i] = j means previous level i matched to current level j
//==============================================================================
std::vector<int> match_levels(const Eigen::MatrixXd& ovlp) {
    int n_prev = (int)ovlp.rows();
    int n_curr = (int)ovlp.cols();
    int n      = std::min(n_prev, n_curr);

    std::vector<int> perm(n_prev, -1);
    std::vector<bool> used(n_curr, false);

    // Sort (i,j) pairs by descending overlap
    std::vector<std::tuple<double,int,int>> pairs;
    for (int i = 0; i < n_prev; i++)
        for (int j = 0; j < n_curr; j++)
            pairs.emplace_back(ovlp(i,j), i, j);
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b){ return std::get<0>(a) > std::get<0>(b); });

    int matched = 0;
    for (auto& [val, i, j] : pairs) {
        if (perm[i] == -1 && !used[j]) {
            perm[i] = j;
            used[j] = true;
            if (++matched == n) break;
        }
    }
    return perm;
}

//==============================================================================
// MAIN TRACKER
// Scans Ecm range, builds QC3_mat at each step, tracks eigenvalues by
// eigenvector overlap across energy steps
//==============================================================================
struct TrackedSpectrum {
    std::vector<double>              E_list;
    // tracks[i][n] = eigenvalue of level i at energy step n
    std::vector<std::vector<double>> tracks;
    // sign_changes[i] = list of Ecm values where level i crossed zero
    std::vector<std::vector<double>> sign_changes;
};

// Callback type: builds and returns the projected M_I at given En
// You provide this lambda — it calls config_maker_4 + F3mat + irrep_proj
using MatrixBuilder = std::function<MatC(double Ecm)>;

TrackedSpectrum track_eigenvalues(
    MatrixBuilder  build_MI,
    double         Ecm_start,
    double         Ecm_end,
    int            Npts)
{
    TrackedSpectrum result;
    double dE = (Ecm_end - Ecm_start) / (Npts - 1);

    EigResult prev;

    for (int n = 0; n < Npts; n++) {
        double Ecm = Ecm_start + n * dE;
        result.E_list.push_back(Ecm);

        MatC      M_I  = build_MI(Ecm);
        EigResult curr = compute_eigs_at_E(M_I, Ecm);

        int n_levels = (int)curr.eigenvalues.size();

        if (n_levels == 0) {
            for (auto& track : result.tracks)
                track.push_back(std::numeric_limits<double>::quiet_NaN());
            prev = curr;
            continue;
        }

        if (n == 0 || result.tracks.empty()) {
            // Initialize tracks
            result.tracks.resize(n_levels);
            result.sign_changes.resize(n_levels);
            for (int i = 0; i < n_levels; i++)
                result.tracks[i].push_back(curr.eigenvalues[i]);
            prev = curr;
            continue;
        }

        // Resize tracks if number of levels changed
        if (n_levels != (int)result.tracks.size()) {
            std::cout << "  Warning: number of levels changed from "
                      << result.tracks.size() << " to " << n_levels
                      << " at Ecm=" << Ecm << "\n";
            result.tracks.resize(n_levels);
            result.sign_changes.resize(n_levels);
        }

        // Choose matching strategy based on whether basis size changed
        std::vector<double> reordered_eigs(n_levels,
            std::numeric_limits<double>::quiet_NaN());
        MatC reordered_vecs = MatC::Zero(curr.eigenvectors.rows(), n_levels);

        bool basis_changed = (prev.eigenvectors.rows() != curr.eigenvectors.rows());

        if (!basis_changed && prev.eigenvectors.cols() > 0) {
            //----------------------------------------------------------
            // STRATEGY 1: eigenvector overlap matching (normal case)
            //----------------------------------------------------------
            Eigen::MatrixXd ovlp = overlap_matrix(prev.eigenvectors,
                                                   curr.eigenvectors);
            std::vector<int> perm = match_levels(ovlp);

            for (int i = 0; i < (int)perm.size(); i++) {
                if (perm[i] >= 0 && perm[i] < n_levels) {
                    reordered_eigs[i]     = curr.eigenvalues[perm[i]];
                    reordered_vecs.col(i) = curr.eigenvectors.col(perm[i]);
                }
            }
        } else {
            //----------------------------------------------------------
            // STRATEGY 2: eigenvalue proximity matching (basis changed)
            //----------------------------------------------------------
            std::cout << "  Basis size changed ("
                      << prev.eigenvectors.rows() << " → "
                      << curr.eigenvectors.rows()
                      << ") at Ecm=" << Ecm
                      << " — using eigenvalue proximity matching\n";

            std::vector<bool> used(n_levels, false);
            int n_prev_levels = (int)result.tracks.size();

            for (int i = 0; i < n_prev_levels; i++) {
                // Get last valid eigenvalue for level i
                double prev_eig = std::numeric_limits<double>::quiet_NaN();
                for (int back = (int)result.tracks[i].size()-1; back >= 0; back--) {
                    if (!std::isnan(result.tracks[i][back])) {
                        prev_eig = result.tracks[i][back];
                        break;
                    }
                }
                if (std::isnan(prev_eig)) continue;

                // Find closest current eigenvalue
                int    best_j   = -1;
                double best_dist = std::numeric_limits<double>::max();
                for (int j = 0; j < n_levels; j++) {
                    if (!used[j]) {
                        double dist = std::abs(curr.eigenvalues[j] - prev_eig);
                        if (dist < best_dist) {
                            best_dist = dist;
                            best_j    = j;
                        }
                    }
                }
                if (best_j >= 0) {
                    reordered_eigs[i]     = curr.eigenvalues[best_j];
                    reordered_vecs.col(i) = curr.eigenvectors.col(best_j);
                    used[best_j]          = true;
                }
            }
        }

        // Append to tracks and detect zero crossings
        for (int i = 0; i < n_levels; i++) {
            double v_curr = reordered_eigs[i];
            result.tracks[i].push_back(v_curr);

            // Zero crossing detection
            if (!result.tracks[i].empty() && n > 0) {
                double v_prev = result.tracks[i].size() >= 2
                    ? result.tracks[i][result.tracks[i].size()-2]
                    : std::numeric_limits<double>::quiet_NaN();

                if (!std::isnan(v_prev) && !std::isnan(v_curr)
                    && v_prev * v_curr < 0.0) {
                    double E_prev  = result.E_list[result.E_list.size()-2];
                    double E_cross = E_prev + (Ecm - E_prev)
                                   * std::abs(v_prev)
                                   / (std::abs(v_prev) + std::abs(v_curr));
                    result.sign_changes[i].push_back(E_cross);
                    std::cout << "  *** Zero crossing: level " << i
                              << " at Ecm=" << E_cross << " ***\n";
                }
            }
        }

        // Update prev with reordered result
        curr.eigenvalues  = reordered_eigs;
        curr.eigenvectors = reordered_vecs;
        prev = curr;
    }
    return result;
}


//==============================================================================
// Print results
//==============================================================================
void print_spectrum(const TrackedSpectrum& spec, const std::string& irrep) {
    std::cout << "\n=== Spectrum in irrep " << irrep << " ===\n";
    std::cout << std::fixed << std::setprecision(6);

    std::cout << "Energy levels (zero crossings):\n";
    for (int i = 0; i < (int)spec.sign_changes.size(); i++) {
        for (double E : spec.sign_changes[i])
            std::cout << "  Level " << i << ":  Ecm = " << E << "\n";
    }

    // Print eigenvalue table: rows=energy, cols=levels
    std::cout << "\nEigenvalue tracks (Ecm | level_0 | level_1 | ...):\n";
    std::cout << std::setw(12) << "Ecm";
    for (int i = 0; i < (int)spec.tracks.size(); i++)
        std::cout << std::setw(14) << ("level_"+std::to_string(i));
    std::cout << "\n";

    for (int n = 0; n < (int)spec.E_list.size(); n++) {
        std::cout << std::setw(12) << spec.E_list[n];
        for (int i = 0; i < (int)spec.tracks.size(); i++) {
            if (n < (int)spec.tracks[i].size())
                std::cout << std::setw(14) << spec.tracks[i][n];
        }
        std::cout << "\n";
    }
}

// Print eigenvalue tracks: each row is one energy, each column is one level
void print_eigenvalue_tracks(const TrackedSpectrum& spec, const std::string& irrep) {
    
    if (spec.tracks.empty()) {
        std::cout << "No tracks to print for irrep " << irrep << "\n";
        return;
    }

    int Npts    = (int)spec.E_list.size();
    int Nlevels = (int)spec.tracks.size();

    std::cout << "\n=== Eigenvalue tracks for irrep " << irrep << " ===\n";
    std::cout << std::fixed << std::setprecision(8);

    // Header
    std::cout << std::setw(14) << "Ecm";
    for (int i = 0; i < Nlevels; i++)
        std::cout << std::setw(16) << ("eig_" + std::to_string(i));
    std::cout << "\n";

    // Separator
    std::cout << std::string(14 + 16*Nlevels, '-') << "\n";

    // Data rows
    for (int n = 0; n < Npts; n++) {
        std::cout << std::setw(14) << spec.E_list[n];
        for (int i = 0; i < Nlevels; i++) {
            if (n < (int)spec.tracks[i].size() &&
                !std::isnan(spec.tracks[i][n]))
                std::cout << std::setw(16) << spec.tracks[i][n];
            else
                std::cout << std::setw(16) << "NaN";
        }
        std::cout << "\n";
    }

    // Zero crossings
    std::cout << "\n--- Zero crossings (energy levels) ---\n";
    bool found = false;
    for (int i = 0; i < (int)spec.sign_changes.size(); i++) {
        for (double E : spec.sign_changes[i]) {
            std::cout << "  eig_" << i << " crosses zero at Ecm = "
                      << E << "\n";
            found = true;
        }
    }
    if (!found)
        std::cout << "  No zero crossings found in scanned range\n";
}

double smallest_eigenvalue(
    const MatC&         F3mat,
    const Vec3&         nnP,
    const std::string&  irrep,
    const FlavorConfig& cfg1,
    const FlavorConfig& cfg2,
    int parity = -1)
{
    MatC M_I = irrep_proj_2plus1_from_config(F3mat, nnP, irrep,
                                              cfg1, cfg2, parity);

    if (M_I.cols() == 0) {
        std::cout << "Empty subspace for irrep " << irrep << "\n";
        return std::numeric_limits<double>::quiet_NaN();
    }

    Eigen::SelfAdjointEigenSolver<MatC> eig(M_I);

    // Return eigenvalue with smallest absolute value
    int    min_idx = 0;
    double min_abs = std::abs(eig.eigenvalues()(0));
    for (int i = 1; i < (int)eig.eigenvalues().size(); i++) {
        double a = std::abs(eig.eigenvalues()(i));
        if (a < min_abs) { min_abs = a; min_idx = i; }
    }
    return eig.eigenvalues()(min_idx);
}

