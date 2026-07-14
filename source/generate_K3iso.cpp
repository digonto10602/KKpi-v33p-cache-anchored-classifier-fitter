/* This code takes in the frame momentum shell number and the lab energy
and returns the real(F3inv) to constrain K3iso
This uses  the following input */
/* ./K3iso nPx nPy nPz energy */

/* The above is changed for using it for both L20 and L24 dataset, now
the whole data is generated using c++ code, we also pass the filename, data
reading is still done using python */
/* ./K3iso nPx nPy nPz L energy_range_A energy_range_B num_of_energy_points filename */

#include<bits/stdc++.h> 
#include<random>
#include "functions.h"
#include "F2_functions.h"
#include "K2_functions.h"
#include "G_functions.h"
#include "QC_functions.h"
#include "pole_searching.h"
#include "omp.h"
#include "splines.h"

void generate_K3iso_L20_from_F3inv( int nPx, 
                                    int nPy, 
                                    int nPz, 
                                    comp En )
{

    /*  Inputs  */
    
    double L = 20;
    double Lbyas = L;
    double xi = 1; 
    double xi1 = 3.444; /* found from lattice */
    L = L*xi1; 
    Lbyas = L; 

    double scattering_length_1_piK = 4.04;
    double scattering_length_2_KK = 4.07;
    double eta_1 = 1.0;
    double eta_2 = 0.5; 
    double atmpi = 0.06906;
    double atmK = 0.09698;

    //atmpi = atmpi/atmK; 
    //atmK = 1.0;
    

    double alpha = 0.5;
    double epsilon_h = 0.0;
    int max_shell_num = 20;

    double pi = std::acos(-1.0); 
    comp twopibyL = 2.0*pi/L;
    comp twopibyxiLbyas = 2.0*pi/(xi*Lbyas);

    comp Px = ((comp)nPx)*twopibyxiLbyas;//twopibyL;
    comp Py = ((comp)nPy)*twopibyxiLbyas;//twopibyL;
    comp Pz = ((comp)nPz)*twopibyxiLbyas;//twopibyL;
    std::vector<comp> total_P(3);
    total_P[0] = Px; 
    total_P[1] = Py; 
    total_P[2] = Pz; 
    comp total_P_val = std::sqrt(Px*Px + Py*Py + Pz*Pz);

    //comp lab_energy = En; 

    Eigen::MatrixXcd F3_mat;
    Eigen::MatrixXcd F2_mat;
    Eigen::MatrixXcd K2i_mat; 
    Eigen::MatrixXcd G_mat; 

    //these are dummy p_config and k_config 
    std::vector<std::vector<comp> > p_config(3, std::vector<comp>());
    std::vector<std::vector<comp> > k_config(3, std::vector<comp>());

    comp K3iso = -function_F3inv_ND_2plus1_mat(  F3_mat, F2_mat, K2i_mat, G_mat, En, p_config, k_config, total_P, eta_1, eta_2, scattering_length_1_piK, scattering_length_2_KK, atmpi, atmK, alpha, epsilon_h, L, xi, max_shell_num); 
            
    std::cout<<std::setprecision(20); 
    std::cout<<real(K3iso)<<std::endl; 
}


void generate_K3iso_L20_from_F3(    int nPx, 
                                    int nPy, 
                                    int nPz, 
                                    comp En )
{

    /*  Inputs  */
    
    double L = 20;
    double Lbyas = L;
    double xi = 1; 
    double xi1 = 3.444; /* found from lattice */
    L = L*xi1; 
    Lbyas = L; 

    double scattering_length_1_piK = 4.04;
    double scattering_length_2_KK = 4.07;
    double eta_1 = 1.0;
    double eta_2 = 0.5; 
    double atmpi = 0.06906;
    double atmK = 0.09698;

    //atmpi = atmpi/atmK; 
    //atmK = 1.0;
    

    double alpha = 0.5;
    double epsilon_h = 0.0;
    int max_shell_num = 20;

    double pi = std::acos(-1.0); 
    comp twopibyL = 2.0*pi/L;
    comp twopibyxiLbyas = 2.0*pi/(xi*Lbyas);

    comp Px = ((comp)nPx)*twopibyxiLbyas;//twopibyL;
    comp Py = ((comp)nPy)*twopibyxiLbyas;//twopibyL;
    comp Pz = ((comp)nPz)*twopibyxiLbyas;//twopibyL;
    std::vector<comp> total_P(3);
    total_P[0] = Px; 
    total_P[1] = Py; 
    total_P[2] = Pz; 
    comp total_P_val = std::sqrt(Px*Px + Py*Py + Pz*Pz);

    //comp lab_energy = En; 

    Eigen::MatrixXcd F3_mat;
    Eigen::MatrixXcd F2_mat;
    Eigen::MatrixXcd K2i_mat; 
    Eigen::MatrixXcd G_mat; 

    //these are dummy p_config and k_config 
    std::vector<std::vector<comp> > p_config(3, std::vector<comp>());
    std::vector<std::vector<comp> > k_config(3, std::vector<comp>());

    comp F3iso = function_F3_ND_2plus1_mat(  F3_mat, F2_mat, K2i_mat, G_mat, En, p_config, k_config, total_P, eta_1, eta_2, scattering_length_1_piK, scattering_length_2_KK, atmpi, atmK, alpha, epsilon_h, L, xi, max_shell_num); 

    comp K3iso = -1.0/F3iso; 
            
    std::cout<<std::setprecision(20); 
    std::cout<<real(K3iso)<<std::endl; 
}



//We use this function to create F3 data for a given irrep 
//at particular intervals, this is used to generate the K3iso = -1/F3iso 
//data using the lattice spectrum for L20 and L24 
void generate_K3iso_using_F3_vs_En_KKpi_omp_single_irrep_with_bounds(   int nPx, 
                                                                        int nPy, 
                                                                        int nPz,
                                                                        double L,
                                                                        double En_range_A, 
                                                                        double En_range_B, 
                                                                        double En_points_val, 
                                                                        std::string filename_string  )
{

    /*  Inputs  */
    
    //double L = 24;
    int Ltostring = L; 
    double Lbyas = L;
    double xi = 1.0; 
    double xi1 = 3.444;/* found from lattice */
    L = L*xi1; // This is done to set the spatial 
                // unit in terms of a_t everywhere 
    Lbyas = L; 
    double scattering_length_1_piK = 4.04;// - 0.2; //total uncertainty 0.05 stat 0.15 systematic 
    double scattering_length_2_KK = 4.07;// - 0.07; //total uncertainty 0.07 stat 
    double eta_1 = 1.0;
    double eta_2 = 0.5; 
    double atmpi = 0.06906;
    double atmK = 0.09698;

    //atmpi = atmpi/atmK; 
    //atmK = 1.0;
    

    double alpha = 0.5;
    double epsilon_h = 0.0;
    int max_shell_num = 20;

    double pi = std::acos(-1.0); 
    comp twopibyL = 2.0*pi/L;
    comp twopibyxiLbyas = 2.0*pi/(xi*Lbyas);

    /*---------------------------------------------------*/

    /*---------------------P config----------------------*/
    int nPmax = 20;
    std::vector<std::vector<int> > nP_config(3,std::vector<int>());

    nP_config[0].push_back(nPx);
    nP_config[1].push_back(nPy); 
    nP_config[2].push_back(nPz); 


    int P_config_size = nP_config[0].size();

    /*-----------------------------------------------------*/

    for(int ind1=0;ind1<P_config_size;++ind1)
    {
        int nPx = nP_config[0][ind1];
        int nPy = nP_config[1][ind1];
        int nPz = nP_config[2][ind1];
    
        std::string filename =  filename_string;
                                /*  "ultraHQ_F3_for_pole_KKpi_L" + std::to_string((int)Ltostring) + "_nP_"//"F3_for_pole_KKpi_scatlength_--_L20_nP_"//
                                + std::to_string((int)nPx)
                                + std::to_string((int)nPy)
                                + std::to_string((int)nPz)
                                + "_"
                                + filenum 
                                + ".dat";
                                */ 

        //std::string filename = "temp";
        comp Px = ((comp)nPx)*twopibyxiLbyas;//twopibyL;
        comp Py = ((comp)nPy)*twopibyxiLbyas;//twopibyL;
        comp Pz = ((comp)nPz)*twopibyxiLbyas;//twopibyL;
        std::vector<comp> total_P(3);
        total_P[0] = Px; 
        total_P[1] = Py; 
        total_P[2] = Pz; 
        comp total_P_val = std::sqrt(Px*Px + Py*Py + Pz*Pz);


        double mi = atmK;
        double mj = atmK;
        double mk = atmpi; 
        //for nP 100 the first run starts 0.4184939100000000245
        double KKpi_threshold = atmK + atmK + atmpi; 
        double KKpipi_threshold = 2.0*atmK + 2.0*atmpi; 
        double KKKK_threshold = 5.0*atmK; 

        double En_initial = std::sqrt(En_range_A*En_range_A + abs(total_P_val*total_P_val));//.27;//0.4184939100000000245;//0.26302;
        double En_final = std::sqrt(En_range_B*En_range_B + abs(total_P_val*total_P_val));;
        double En_points = En_points_val;

        double delE = abs(En_initial - En_final)/En_points;

        std::ofstream fout; 
        fout.open(filename.c_str());

        double* norm_vec = NULL; 
        norm_vec = new double[(int)En_points+1];

        double* En_vec = NULL; 
        En_vec = new double[(int)En_points+1];

        double* Ecm_vec = NULL;
	    Ecm_vec = new double[(int)En_points+1];

        comp* result_F3 = NULL;
	    result_F3 = new comp[(int)En_points+1];

        comp* result_F2 = NULL; 
        result_F2 = new comp[(int)En_points+1]; 

        comp* result_K2inv = NULL; 
        result_K2inv = new comp[(int)En_points+1];

        comp* result_G = NULL; 
        result_G = new comp[(int)En_points+1];

        comp* result_Hinv = NULL; 
        result_Hinv = new comp[(int)En_points+1];

	    for(int i=0;i<En_points+1;++i)
        {
            norm_vec[i] = 0.0; 
            En_vec[i] = 0.0;
            Ecm_vec[i] = 0.0;
            result_F3[i] = 0.0;
            result_F2[i] = 0.0;
            result_K2inv[i] = 0.0;
            result_G[i] = 0.0;
            result_Hinv[i] = 0.0; 
        } 

        std::cout<<"filename:"<<filename<<" is running."<<std::endl; 
        //#pragma acc data copy(Ecm_vec[0:En_points],result_F3[0:En_points]) 
	    //#pragma acc parallel loop independent
        int loopcounter = 0; 
        int i=0; 
        #pragma omp parallel for 
        for(i=0; i<(int)En_points + 1; ++i)
        {
            double En = En_initial + i*delE; 

            std::vector< std::vector<comp> > p_config(3,std::vector<comp> ());
            double config_tolerance = 1.0e-5;
            //config_maker_1(p_config, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );

            std::vector< std::vector<comp> > k_config = p_config; 


            int size = p_config[0].size();
            //std::cout<<"size = "<<size<<std::endl;
            Eigen::VectorXcd state_vec;   
            Eigen::MatrixXcd F3_mat;//(Eigen::Dynamic,Eigen::Dynamic);
            Eigen::MatrixXcd F2_mat;
            Eigen::MatrixXcd K2i_mat; 
            Eigen::MatrixXcd G_mat; 
            Eigen::MatrixXcd Hmatinv; 

            test_F3iso_ND_2plus1_mat_with_normalization(  F3_mat, state_vec, F2_mat, K2i_mat, G_mat, Hmatinv, En, p_config, k_config, total_P, eta_1, eta_2, scattering_length_1_piK, scattering_length_2_KK, atmpi, atmK, alpha, epsilon_h, L, xi, max_shell_num); 

            comp Ecm_calculated = E_to_Ecm(En, total_P);

            comp F3iso = state_vec.transpose()*F3_mat*state_vec; 
            comp F2iso = state_vec.transpose()*F2_mat*state_vec;
            comp K2inv_iso = state_vec.transpose()*K2i_mat*state_vec;
            comp Giso = state_vec.transpose()*G_mat*state_vec;
            comp Hmatinv_iso = state_vec.transpose()*Hmatinv*state_vec;

            comp norm1 = state_vec.transpose()*state_vec; 
            double norm2 = real(norm1); 
            norm_vec[i] = 1.0/std::sqrt(norm2); 
            En_vec[i] = En; 
            Ecm_vec[i] = real(Ecm_calculated); 
            result_F3[i] = F3iso;
            result_F2[i] = F2iso; 
            result_G[i] = Giso; 
            result_K2inv[i] = K2inv_iso; 
            result_Hinv[i] = Hmatinv_iso; 

            //std::cout<<"running = "<<i<<std::endl; 
            double looppercent = ((loopcounter+1)/(En_points))*100.0;

            loopcounter = loopcounter + 1; 
            int divisor = (En_points)/10; 
            double ddivisor = (En_points)/10.0;
            if(loopcounter%divisor==0)
            //if(std::fmod(loopcounter,ddivisor)==0)
            {
                //if((int)looppercent==templooppercent)
                //{
                //    continue; 
                //}
                //else 
                {
                    std::cout<<"P="<<nPx<<nPy<<nPz<<" run completion: "<<looppercent<<"%"<<std::endl; 
                    //templooppercent = (int) looppercent; 
                }
            }
            
        }

        for(int i=0;i<En_points;++i)
        {
            //std::cout<<std::setprecision(20)<<i<<'\t'<<Ecm_vec[i]<<'\t'<<result_F3[i]<<std::endl; 
            fout<<std::setprecision(20)
                <<En_vec[i]<<'\t'
                <<Ecm_vec[i]<<'\t'
                <<norm_vec[i]<<'\t'
                <<real(result_F3[i])<<'\t'
                <<real(-1.0/result_F3[i])<<'\t' //This is K3iso = -1/F3iso 
                <<real(result_F2[i])<<'\t'
                <<real(result_G[i])<<'\t'
                <<real(result_K2inv[i])<<'\t'
                <<real(result_Hinv[i])<<std::endl; 
        }
        fout.close();

        //std::cout<<"P = "<<nPx<<nPy<<nPz<<" file generated!"<<std::endl; 

        std::cout<<filename<<" file generated!"<<std::endl; 

        delete [] norm_vec;
        norm_vec = NULL; 
        delete [] En_vec; 
        En_vec = NULL; 
        delete [] Ecm_vec;
        Ecm_vec = NULL;
        delete [] result_F3; 
        result_F3 = NULL;
        delete [] result_F2; 
        result_F2 = NULL; 
        delete [] result_K2inv;
        result_K2inv = NULL; 
        delete [] result_G; 
        result_G = NULL; 
        delete [] result_Hinv; 
        result_Hinv = NULL; 

    }               
}

void generate_K3iso_using_F3_vs_En_KKpi_omp_single_irrep_centralvalue(  int nPx, 
                                                                        int nPy, 
                                                                        int nPz,
                                                                        double L,
                                                                        double Ecm_val   )
{

    /*  Inputs  */
    
    //double L = 24;
    int Ltostring = L; 
    double Lbyas = L;
    double xi = 1.0; 
    double xi1 = 3.444;/* found from lattice */
    L = L*xi1; // This is done to set the spatial 
                // unit in terms of a_t everywhere 
    Lbyas = L; 
    double scattering_length_1_piK = 4.04;// - 0.2; //total uncertainty 0.05 stat 0.15 systematic 
    double scattering_length_2_KK = 4.07;// - 0.07; //total uncertainty 0.07 stat 
    double eta_1 = 1.0;
    double eta_2 = 0.5; 
    double atmpi = 0.06906;
    double atmK = 0.09698;

    //atmpi = atmpi/atmK; 
    //atmK = 1.0;
    

    double alpha = 0.5;
    double epsilon_h = 0.0;
    int max_shell_num = 20;

    double pi = std::acos(-1.0); 
    comp twopibyL = 2.0*pi/L;
    comp twopibyxiLbyas = 2.0*pi/(xi*Lbyas);

    /*---------------------------------------------------*/

    /*---------------------P config----------------------*/
    int nPmax = 20;
    std::vector<std::vector<int> > nP_config(3,std::vector<int>());

    nP_config[0].push_back(nPx);
    nP_config[1].push_back(nPy); 
    nP_config[2].push_back(nPz); 


    int P_config_size = nP_config[0].size();

    /*-----------------------------------------------------*/

    
        
    comp Px = ((comp)nPx)*twopibyxiLbyas;//twopibyL;
    comp Py = ((comp)nPy)*twopibyxiLbyas;//twopibyL;
    comp Pz = ((comp)nPz)*twopibyxiLbyas;//twopibyL;
    std::vector<comp> total_P(3);
    total_P[0] = Px; 
    total_P[1] = Py; 
    total_P[2] = Pz; 
    comp total_P_val = std::sqrt(Px*Px + Py*Py + Pz*Pz);


    double mi = atmK;
    double mj = atmK;
    double mk = atmpi; 
    //for nP 100 the first run starts 0.4184939100000000245
    double KKpi_threshold = atmK + atmK + atmpi; 
    double KKpipi_threshold = 2.0*atmK + 2.0*atmpi; 
    double KKKK_threshold = 5.0*atmK; 

    double En = std::sqrt(Ecm_val*Ecm_val + abs(total_P_val*total_P_val));
        

        
    std::vector< std::vector<comp> > p_config(3,std::vector<comp> ());
    double config_tolerance = 1.0e-5;

    std::vector< std::vector<comp> > k_config = p_config; 

    int size = p_config[0].size();
    //std::cout<<"size = "<<size<<std::endl;
    Eigen::VectorXcd state_vec;   
    Eigen::MatrixXcd F3_mat;//(Eigen::Dynamic,Eigen::Dynamic);
    Eigen::MatrixXcd F2_mat;
    Eigen::MatrixXcd K2i_mat; 
    Eigen::MatrixXcd G_mat; 
    Eigen::MatrixXcd Hmatinv; 

    test_F3iso_ND_2plus1_mat_with_normalization(  F3_mat, state_vec, F2_mat, K2i_mat, G_mat, Hmatinv, En, p_config, k_config, total_P, eta_1, eta_2, scattering_length_1_piK, scattering_length_2_KK, atmpi, atmK, alpha, epsilon_h, L, xi, max_shell_num); 

    comp Ecm_calculated = E_to_Ecm(En, total_P);

    comp F3iso = state_vec.transpose()*F3_mat*state_vec; 
    comp F2iso = state_vec.transpose()*F2_mat*state_vec;
    comp K2inv_iso = state_vec.transpose()*K2i_mat*state_vec;
    comp Giso = state_vec.transpose()*G_mat*state_vec;
    comp Hmatinv_iso = state_vec.transpose()*Hmatinv*state_vec;
    comp K3iso = -1.0/F3iso; 
       

    std::cout<<std::setprecision(20); 
    std::cout<<real(K3iso)<<std::endl;             
}


int main(int argc, char *argv[])
{
    int nPx, nPy, nPz;
    double L; 
    double Ecm_range_A, Ecm_range_B, Ecm_points_val;
    double Ecm_val; 
    std::string filename; 
    
    if(argc==9)
    {
        nPx = std::stoi(argv[1]);
        nPy = std::stoi(argv[2]);
        nPz = std::stoi(argv[3]);

        L = std::stof(argv[4]);
        Ecm_range_A = std::stof(argv[5]);
        Ecm_range_B = std::stof(argv[6]);
        Ecm_points_val = std::stof(argv[7]);
        filename = argv[8];
    }
    else if(argc==6)
    {
        nPx = std::stoi(argv[1]);
        nPy = std::stoi(argv[2]);
        nPz = std::stoi(argv[3]);

        L = std::stof(argv[4]);
        Ecm_val = std::stof(argv[5]); 
    }

    //double energy = std::stod(argv[4]);

    //generate_K3iso_L20_from_F3(nPx, nPy, nPz, energy);
    
    if(argc==9)
    generate_K3iso_using_F3_vs_En_KKpi_omp_single_irrep_with_bounds( nPx, nPy, nPz, L, Ecm_range_A, Ecm_range_B, Ecm_points_val, filename);
    else if(argc==6)
    generate_K3iso_using_F3_vs_En_KKpi_omp_single_irrep_centralvalue( nPx, nPy, nPz, L, Ecm_val); 
    else 
    std::cout<<"Error"<<std::endl;  


    return 0; 
}