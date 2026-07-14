// projections_from_config.hpp
// Adapts irrep_proj_2plus1 to work directly with config_maker_4 output
// Requires: projections.hpp above

#pragma once
#include "projections.hpp"
#include <map>
#include <tuple>

//==============================================================================
// PARSING config_maker_4 OUTPUT
//==============================================================================

struct BasisEntry {
    Vec3 nnk;   // integer momentum [i,j,k]
    int  ell;
    int  m;
};

// Parse plm_config + n_config as produced by config_maker_4
// plm_config[3]=ell (stored as comp), plm_config[4]=m
// n_config[0..2] = i,j,k
std::vector<BasisEntry> parse_config(
    const std::vector<std::vector<comp>>& plm_config,
    const std::vector<std::vector<int>>&  n_config)
{
    int N = (int)n_config[0].size();
    std::vector<BasisEntry> basis(N);
    for (int idx = 0; idx < N; idx++) {
        basis[idx].nnk = { n_config[0][idx], n_config[1][idx], n_config[2][idx] };
        basis[idx].ell = (int)std::round(plm_config[3][idx].real());
        basis[idx].m   = (int)std::round(plm_config[4][idx].real());
    }
    return basis;
}

// Get ordered unique nnk list (first appearance order preserved)
std::vector<Vec3> unique_nnk_list(const std::vector<BasisEntry>& basis) {
    std::vector<Vec3> out;
    for (const auto& e : basis)
        if (std::find(out.begin(), out.end(), e.nnk) == out.end())
            out.push_back(e.nnk);
    return out;
}

// Get ordered unique (ell,m) pairs
std::vector<std::pair<int,int>> unique_lm_list(const std::vector<BasisEntry>& basis) {
    std::vector<std::pair<int,int>> out;
    for (const auto& e : basis) {
        auto p = std::make_pair(e.ell, e.m);
        if (std::find(out.begin(), out.end(), p) == out.end())
            out.push_back(p);
    }
    return out;
}

//==============================================================================
// REORDERING
// config_maker_4 layout: lm outer loop, nnk inner loop
//   → [all nnk for (ell=0,m=0)], [all nnk for (ell=1,m=-1)], ...
// projector layout: nnk outer, lm inner
//   → [all lm for nnk_0], [all lm for nnk_1], ...
//==============================================================================

// Build permutation: perm[new_idx] = old_idx
// new = nnk-outer/lm-inner,  old = lm-outer/nnk-inner (your config layout)
std::vector<int> build_reorder_perm(const std::vector<BasisEntry>& basis) {
    auto nnk_list = unique_nnk_list(basis);
    auto lm_list  = unique_lm_list(basis);
    int  N        = (int)basis.size();

    // Build reverse lookup: (nnk, ell, m) -> old flat index
    std::map<std::tuple<Vec3,int,int>, int> old_idx_map;
    for (int i = 0; i < N; i++)
        old_idx_map[{basis[i].nnk, basis[i].ell, basis[i].m}] = i;

    std::vector<int> perm;
    perm.reserve(N);
    for (const auto& nnk : nnk_list) {
        for (const auto& [ell, m] : lm_list) {
            auto key = std::make_tuple(nnk, ell, m);
            auto it  = old_idx_map.find(key);
            if (it != old_idx_map.end())
                perm.push_back(it->second);
        }
    }
    return perm;
}

// Apply permutation to rows and columns of a complex matrix
MatC permute_matrix(const MatC& M, const std::vector<int>& perm) {
    int n = (int)perm.size();
    MatC out(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            out(i, j) = M(perm[i], perm[j]);
    return out;
}

//==============================================================================
// FLAVOR CONFIG CONTAINER
//==============================================================================

struct FlavorConfig {
    std::vector<std::vector<comp>> plm_config;  // [px,py,pz,ell,m] each size N
    std::vector<std::vector<int>>  n_config;    // [i,j,k]  each size N
};

//==============================================================================
// TOP-LEVEL: irrep_proj_2plus1_from_config
// Accepts your raw config_maker_4 output, handles reordering internally
//==============================================================================

MatC irrep_proj_2plus1_from_config(
    const MatC&         M,         // full QC3_mat (complex, n x n)
    const Vec3&         nnP,       // total momentum integer vector e.g. {0,0,1}
    const std::string&  I,         // irrep label e.g. "A1u"
    const FlavorConfig& cfg1,      // flavor-1 config (s+p waves)
    const FlavorConfig& cfg2,      // flavor-2 config (s wave only)
    int parity = -1)
{
    //------------------------------------------------------------------
    // Parse configs into structured basis lists
    //------------------------------------------------------------------
    auto basis1  = parse_config(cfg1.plm_config, cfg1.n_config);
    auto basis2  = parse_config(cfg2.plm_config, cfg2.n_config);

    auto nnk1    = unique_nnk_list(basis1);
    auto nnk2    = unique_nnk_list(basis2);
    auto lm1     = unique_lm_list(basis1);   // e.g. {(0,0),(1,-1),(1,0),(1,1)}
    auto lm2     = unique_lm_list(basis2);   // e.g. {(0,0)}

    int N1 = (int)(nnk1.size() * lm1.size());
    int N2 = (int)(nnk2.size() * lm2.size());
    int N  = N1 + N2;

    if (M.rows() != N || M.cols() != N) {
        std::cerr << "irrep_proj_2plus1_from_config: M is "
                  << M.rows() << "x" << M.cols()
                  << " but config implies " << N << "x" << N << "\n";
        throw std::runtime_error("Matrix / config size mismatch");
    }

    //------------------------------------------------------------------
    // Reorder M: your layout (lm-outer, nnk-inner) → projector layout (nnk-outer, lm-inner)
    //------------------------------------------------------------------
    auto perm1 = build_reorder_perm(basis1);
    auto perm2 = build_reorder_perm(basis2);

    // Build combined permutation over full matrix
    std::vector<int> full_perm;
    full_perm.reserve(N);
    for (int idx : perm1) full_perm.push_back(idx);        // flavor-1 block
    for (int idx : perm2) full_perm.push_back(N1 + idx);   // flavor-2 block (offset)

    MatC M_reordered = permute_matrix(M, full_perm);

    //------------------------------------------------------------------
    // Build full projectors in projector layout (nnk-outer, lm-inner)
    //------------------------------------------------------------------
    MatC P_full1 = build_full_projector(nnP, I, nnk1, lm1, parity);
    MatC P_full2 = build_full_projector(nnP, I, nnk2, lm2, parity);

    //------------------------------------------------------------------
    // Extract eigenvalue-1 subspaces
    //------------------------------------------------------------------
    MatC Psub1 = subspace_from_projector(P_full1);
    MatC Psub2 = subspace_from_projector(P_full2);

    // Block-diagonal subspace projector: (N x N_irrep)
    MatC P_I = block_diag({Psub1, Psub2});

    if (P_I.cols() == 0) return MatC::Zero(0, 0);  // empty irrep

    //------------------------------------------------------------------
    // Project M_reordered → irrep subspace
    //------------------------------------------------------------------
    return chop(P_I.adjoint() * M_reordered * P_I);
}

//==============================================================================
// CONVENIENCE: loop over all irreps and print eigenvalues
//==============================================================================

void print_irrep_eigenvalues(
    const MatC&         M,
    const Vec3&         nnP,
    const FlavorConfig& cfg1,
    const FlavorConfig& cfg2,
    int parity = -1)
{
    for (const auto& I : irrep_list(nnP)) {
        MatC M_I = irrep_proj_2plus1_from_config(M, nnP, I, cfg1, cfg2, parity);

        if (M_I.cols() == 0) {
            std::cout << "Irrep " << I << ": empty (wrong parity)\n";
            continue;
        }

        Eigen::SelfAdjointEigenSolver<MatC> eig(M_I);
        Eigen::VectorXd evals = eig.eigenvalues();

        // Sort by absolute value (matching Python's sorted(..., key=abs))
        std::vector<double> ev(evals.data(), evals.data() + evals.size());
        std::sort(ev.begin(), ev.end(), [](double a, double b){
            return std::abs(a) < std::abs(b);
        });

        std::cout << "Irrep = " << I << "\n";
        std::cout << "Eigenvalues = ";
        for (double v : ev) std::cout << v << "  ";
        std::cout << "\n";
    }
}
