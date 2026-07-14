#ifndef QCFUNCTIONS_V2_H
#define QCFUNCTIONS_V2_H
#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <limits>
#include <string>
#include <complex>
#include <stdexcept>
#include "functions.h"
#include "F2_functions_v2.h"
#include "K2_functions_v2.h"
#include "G_functions_v2.h"
#include <vector>
#include <cmath>


typedef std::complex<double> comp;

//Basic Solvers that we will need in this program
void LinearSolver_3(	Eigen::MatrixXcd &A,
					Eigen::MatrixXcd &X,
					Eigen::MatrixXcd &B,
					double &relerr 			)
{
	X = A.partialPivLu().solve(B);

	relerr = (A*X - B).norm()/B.norm();
}

void LinearSolver_4(	Eigen::MatrixXcd &A,
					Eigen::MatrixXcd &X,
					Eigen::MatrixXcd &B,
					double &relerr 			)
{
	X = A.colPivHouseholderQr().solve(B);

	relerr = (A*X - B).norm()/B.norm();
}



Eigen::MatrixXcd solve_Hmat_X_eq_F2mat(const Eigen::MatrixXcd& Hmat,
                                       const Eigen::MatrixXcd& F2mat,
                                       double imag_tol = 1e-14,
                                       bool verbose = false)
{
    if (Hmat.rows() != Hmat.cols()) {
        throw std::runtime_error("solve_Hmat_X_eq_F2mat: Hmat must be square");
    }

    if (Hmat.rows() != F2mat.rows()) {
        throw std::runtime_error("solve_Hmat_X_eq_F2mat: dimension mismatch, Hmat.rows() must equal F2mat.rows()");
    }

    const int n = Hmat.rows();

    double max_imag_H = Hmat.imag().cwiseAbs().maxCoeff();
    double max_imag_F = F2mat.imag().cwiseAbs().maxCoeff();

    bool H_real_like  = (max_imag_H < imag_tol);
    bool F_real_like  = (max_imag_F < imag_tol);
    bool use_real_solver = H_real_like && F_real_like;

    if (verbose) {
        std::cout << "max |Im(Hmat)| = " << max_imag_H << "\n";
        std::cout << "max |Im(F2mat)| = " << max_imag_F << "\n";
        std::cout << "use_real_solver = " << (use_real_solver ? "true" : "false") << "\n";
    }

    if (use_real_solver) {
        Eigen::MatrixXd Hr = Hmat.real();
        Eigen::MatrixXd Fr = F2mat.real();

        Eigen::FullPivLU<Eigen::MatrixXd> lu_check(Hr);
        if (!lu_check.isInvertible()) {
            throw std::runtime_error("solve_Hmat_X_eq_F2mat: Hmat is singular in real solve branch");
        }

        Eigen::PartialPivLU<Eigen::MatrixXd> lu(Hr);
        Eigen::MatrixXd Xr = lu.solve(Fr);

        return Xr.cast<std::complex<double>>();
    } else {
        Eigen::FullPivLU<Eigen::MatrixXcd> lu_check(Hmat);
        if (!lu_check.isInvertible()) {
            throw std::runtime_error("solve_Hmat_X_eq_F2mat: Hmat is singular in complex solve branch");
        }

        Eigen::PartialPivLU<Eigen::MatrixXcd> lu(Hmat);
        return lu.solve(F2mat);
    }
}



void check_invertibility(const Eigen::MatrixXcd& A,
                         double svd_tol = 1e-12,
                         const std::string& name = "A")
{
    std::cout << "----------------------------------------\n";
    std::cout << "Matrix: " << name << "\n";
    std::cout << "shape  = " << A.rows() << " x " << A.cols() << "\n";

    if (A.rows() != A.cols()) {
        std::cout << "Matrix is not square, so it is NOT invertible.\n";
        std::cout << "----------------------------------------\n";
        return;
    }

    const int dim = A.rows();

    // FullPivLU checks
    Eigen::FullPivLU<Eigen::MatrixXcd> lu(A);
    const auto rank = lu.rank();
    const bool lu_invertible = lu.isInvertible();

    // SVD checks
    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd svals = svd.singularValues();

    double sigma_max = 0.0;
    double sigma_min = 0.0;
    double cond_num  = std::numeric_limits<double>::infinity();
    int svd_rank = 0;

    if (svals.size() > 0) {
        sigma_max = svals.maxCoeff();
        sigma_min = svals.minCoeff();

        for (int i = 0; i < svals.size(); ++i) {
            if (svals(i) > svd_tol) {
                ++svd_rank;
            }
        }

        if (sigma_min > 0.0) {
            cond_num = sigma_max / sigma_min;
        }
    }

    const bool svd_invertible = (svd_rank == dim) && (sigma_min > svd_tol);

    // Print summary
    std::cout << std::setprecision(16);
    std::cout << "dim                  = " << dim << "\n";
    std::cout << "FullPivLU rank       = " << rank << "\n";
    std::cout << "FullPivLU invertible = " << (lu_invertible ? "true" : "false") << "\n";

    std::cout << "SVD tol              = " << svd_tol << "\n";
    std::cout << "SVD rank             = " << svd_rank << "\n";
    std::cout << "sigma_max            = " << sigma_max << "\n";
    std::cout << "sigma_min            = " << sigma_min << "\n";
    std::cout << "condition number     = " << cond_num << "\n";
    std::cout << "SVD invertible       = " << (svd_invertible ? "true" : "false") << "\n";

    /*
    std::cout << "Singular values:\n";
    for (int i = 0; i < svals.size(); ++i) {
        std::cout << "  s[" << i << "] = " << svals(i) << "\n";
    }
    */

    if (lu_invertible && svd_invertible) {
        std::cout << "Conclusion: matrix is invertible and numerically looks full rank.\n";
    } else if (lu_invertible && !svd_invertible) {
        std::cout << "Conclusion: LU says invertible, but SVD says numerically near-singular.\n";
    } else {
        std::cout << "Conclusion: matrix is NOT invertible or is numerically singular.\n";
    }

    std::cout << "----------------------------------------\n";
}


void test_F3iso_ND_2plus1_mat_with_normalization_single_En(
							Eigen::MatrixXcd &F3mat,
							comp &F3iso,
                            Eigen::VectorXcd &state_vec, 
                            Eigen::MatrixXcd &F2mat,
                            Eigen::MatrixXcd &K2imat,
                            Eigen::MatrixXcd &Gmat, 
                            Eigen::MatrixXcd &Hmatinv, 
                            comp En, 
                            std::vector< std::vector<comp> > plm_config,
                            std::vector< std::vector<comp> > klm_config, 
                            std::vector<comp> total_P, 
                            double eta_i_1,
                            double eta_i_2, 
                            std::vector<std::vector<comp> > scatter_params_1,
                            std::vector<std::vector<comp> > scatter_params_2,
                            double m1,
                            double m2,  
                            double alpha, 
                            double epsilon_h, 
                            double L, 
                            int max_shell_num,
							bool Q0norm  

)
{
    char debug = 'y'; 
	int size1 = plm_config[0].size();
	int size2 = klm_config[0].size(); 

	F2_2plus1_mat( F2mat, En, plm_config, klm_config, total_P, m1, m2, L, alpha, epsilon_h, max_shell_num, Q0norm);

	K2inv_EREord2_2plus1_mat(K2imat, eta_i_1, eta_i_2, scatter_params_1, scatter_params_2, En, plm_config, klm_config, total_P, m1, m2, epsilon_h, L);

	G_2plus1_mat(Gmat, En, plm_config, klm_config, total_P, m1, m2, L, alpha, epsilon_h, max_shell_num, Q0norm);
            
	Eigen::MatrixXcd Hmat = K2imat + F2mat + Gmat; 
    
    if(debug=='y')
    {
        check_invertibility(K2imat, 1e-12, "K2imat");
        check_invertibility(F2mat, 1e-12, "F2mat");
        
        check_invertibility(Gmat, 1e-12, "Gmat");
        check_invertibility(Hmat, 1e-12, "Hmat");
    }
    
    Eigen::MatrixXcd X = solve_Hmat_X_eq_F2mat(Hmat, F2mat, 1e-14, true);

    Hmatinv = X; 
    
	int eigvec_counter = 0; 
    Eigen::VectorXcd EigVec(size1 + size2); 
    for(int i=0; i<size1; ++i)
    {
        EigVec(i) = 1.0;
        eigvec_counter += 1; 
    }
    for(int i=eigvec_counter; i<(size1+size2); ++i)
    {
        EigVec(i) = 1.0/std::sqrt(2.0); 
    }

	//Eigen::MatrixXcd temp_identity_mat(size1 + size2,size1 + size2);
    //temp_identity_mat.setIdentity();


	//Eigen::MatrixXcd H_mat_inv(size1 + size2,size1 + size2);
    //double relerror = 0.0;
	//LinearSolver_4(Hmat, H_mat_inv, temp_identity_mat, relerror);

	Eigen::MatrixXcd F3_mat = (F2mat/3.0 - F2mat*X);

	F3mat = F3_mat; 
	F3iso = EigVec.transpose()*F3mat*EigVec; 


	//std::cout<<"matrix size = "<<size1+size2 << "x "<<size1+size2<< std::endl;
                        


}




std::vector<comp> normalize_det_vector_by_max(
    const std::vector<comp>& det_proj_F3i_vec,
    const std::vector<comp>& Ecm_vec,
    double max_norm_value = 2.0,
    char debug = 'n'
)
{
    if (det_proj_F3i_vec.size() != Ecm_vec.size())
    {
        throw std::runtime_error("det_proj_F3i_vec and Ecm_vec must have the same size.");
    }

    if (det_proj_F3i_vec.empty())
    {
        throw std::runtime_error("det_proj_F3i_vec is empty.");
    }

    double max_abs_det = 0.0;
    std::size_t max_index = 0;

    for (std::size_t i = 0; i < det_proj_F3i_vec.size(); ++i)
    {
        double abs_val = std::abs(det_proj_F3i_vec[i]);

        if (abs_val > max_abs_det)
        {
            max_abs_det = abs_val;
            max_index = i;
        }
    }

    if (max_abs_det == 0.0)
    {
        throw std::runtime_error("Maximum determinant magnitude is zero. Cannot normalize.");
    }

    double scale = max_norm_value / max_abs_det;

    std::vector<comp> normalized_vec(det_proj_F3i_vec.size());

    for (std::size_t i = 0; i < det_proj_F3i_vec.size(); ++i)
    {
        normalized_vec[i] = det_proj_F3i_vec[i] * scale;
    }

    if (debug == 'y')
    {
        std::cout << std::setprecision(16);
        std::cout << "max_index = " << max_index << "\n";
        std::cout << "Ecm at max = " << Ecm_vec[max_index] << "\n";
        std::cout << "max det value = " << det_proj_F3i_vec[max_index] << "\n";
        std::cout << "max |det| = " << max_abs_det << "\n";
        std::cout << "scale = " << scale << "\n";
        std::cout << "normalized max value = " << normalized_vec[max_index] << "\n";
        std::cout << "|normalized max value| = " 
                  << std::abs(normalized_vec[max_index]) << "\n";
    }

    return normalized_vec;
}

#endif