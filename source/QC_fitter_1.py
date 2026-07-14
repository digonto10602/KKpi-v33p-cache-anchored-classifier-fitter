#This function fits the
import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
from scipy.linalg import block_diag
import scipy.interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PyPDF2 import PdfMerger

import os 
import subprocess 
import math 
import sys 

from timeit import default_timer as timer

from iminuit import minimize 

import random 

threebody_path_ubuntu = '/home/digonto/Codes/Practical_Lattice_v2/3body_quantization/'
threebody_path_macos = '/Users/digonto/GitHub/3body_quantization/'
macos_path2 = '/Users/digonto/GitHub/jackknife_codes/'
ubuntu_path2 = '/home/digonto/Codes/Practical_Lattice_v2/jackknife_codes/'

from sys import platform 

#print(platform)

if platform=="linux" or platform=="linux2":
    print("platform = ",platform)
    jackpath = ubuntu_path2
    threebody_path = threebody_path_ubuntu
elif platform=="darwin":
    print("platform = ",platform)
    jackpath = macos_path2
    threebody_path = threebody_path_macos

sys.path.insert(1, jackpath)

import jackknife 
from jackknife import jackknife_resampling, jackknife_average, jackknife_error 
from lattice_data_covariance import covariance_between_states_szscl21_based

def E_to_Ecm(En, P):
    return np.sqrt(En**2 - P**2)

def Esq_to_Ecmsq(En, P):
    return En**2 - P**2

def Ecmsq_to_Esq(Ecm, P):
    return Ecm**2 + P**2

def QC3(K3iso, F3inv):
    return F3inv + K3iso 

def sign_func(val):
    if(val>0.0):
        return 1.0; 
    else:
        return -1.0; 

def QC3_bissection_interp1d_based_multiL(pointA_ind, pointB_ind, Ecm, F3inv, pointA, pointB, K3iso1, K3iso2, nPx, nPy, nPz, nmax, tol, threeparticle_threshold_in_s):
    
    F3inv_for_interp = np.zeros((abs(pointA_ind - pointB_ind)))
    Ecm_for_interp = np.zeros((abs(pointA_ind - pointB_ind)))
    interp_counter = 0
    for i in range(pointA_ind,pointB_ind,1):
        Ecm_for_interp[interp_counter] = Ecm[i]
        F3inv_for_interp[interp_counter] = F3inv[i]
        interp_counter = interp_counter + 1 
    
    F3inv_interp = scipy.interpolate.interp1d(Ecm_for_interp,F3inv_for_interp, kind='linear')

    A = pointA 
    B = pointB 

    #new_Ecm = np.linspace(A,B,1000)
    #fig, ax = plt.subplots(figsize=(12,5))

    #ax.set_ylim(-1E8,1E8)
    #ax.set_xlim(0.26,0.37)

    #ax.plot(Ecm,F3inv, color='blue', zorder=4)
    #ax.plot(new_Ecm,F3inv_interp(new_Ecm), color='red', zorder=5)

    #plt.show()
    #exit()
    
    
    #F3inv_A = subprocess.check_output(['./eigen_F3inv',str(nPx),str(nPy),str(nPz),str(A)],shell=False)
    #F3inv_result_A = F3inv_A.decode('utf-8')
    #F3inv_fin_result_A = float(F3inv_result_A)

    F3inv_fin_result_A = F3inv_interp(A)
    K3iso_A = K3iso1 + K3iso2*(A*A - threeparticle_threshold_in_s)/threeparticle_threshold_in_s
    QC_A = QC3(K3iso_A, F3inv_fin_result_A)
    
    #F3inv_B = subprocess.check_output(['./eigen_F3inv',str(nPx),str(nPy),str(nPz),str(B)],shell=False)
    #F3inv_result_B = F3inv_B.decode('utf-8')
    #F3inv_fin_result_B = float(F3inv_result_B)

    F3inv_fin_result_B = F3inv_interp(B)
    K3iso_B = K3iso1 + K3iso2*(B*B - threeparticle_threshold_in_s)/threeparticle_threshold_in_s
    QC_B = QC3(K3iso_B, F3inv_fin_result_B)

    print("QC_A = ",QC_A)
    print("QC_B = ",QC_B)
    
    if(QC_A==0.0):
        return A 
    elif(QC_B==0.0):
        return B 
    else:
        fin_result = 0.0
        for i in range(0,nmax,1):
            C = (A + B)/2.0 
            #F3inv_C = subprocess.check_output(['./eigen_F3inv',str(nPx),str(nPy),str(nPz),str(C)],shell=False)
            #F3inv_result_C = F3inv_C.decode('utf-8')
            #F3inv_fin_result_C = float(F3inv_result_C)

            F3inv_fin_result_C = F3inv_interp(C)
            K3iso_C = K3iso1 + K3iso2*(C*C - threeparticle_threshold_in_s)/threeparticle_threshold_in_s
            QC_C = QC3(K3iso_C, F3inv_fin_result_C)
            #print("QC_C = ",QC_C)
            if( abs(QC_C) < tol or abs(B-A)/2.0 < tol/1000000 ):
                fin_result = C 
                print("QC val = ",abs(QC_C)," tol = ",tol)
                print("B-A/2 = ",abs(B-A)/2.0," tol = ", tol/1000000)
                print("entered breaking condition for bissection with C = ",C)
                break 
            
            if(sign_func(QC_C)==sign_func(QC_A)):
                A = C 
                QC_A = QC_C 
            elif(sign_func(QC_C)==sign_func(QC_B)):
                B = C 
                QC_B = QC_C

            #fin_result = C  
        return fin_result 



def K3iso_fitting_function_multiL_oneparameter_interp1d_based(x0, nmax, states_avg, states_err, nP_list, state_no, L_list, covariance_matrix_inv, tol, threeparticle_threshold_in_s, energy_cutoff):
    energy_eps = 1.0E-5
    K3iso_1 = x0[0]

    fit_parameter_no = float(len(x0))

    QC_states = []

    #for i in range(len(state_no)):
    #    print("state nums = ",i,state_no[i])
    
    for ind in range(0,len(states_avg),1):
        state_ecm = states_avg[ind]
        nPx = nP_list[ind][0]
        nPy = nP_list[ind][1]
        nPz = nP_list[ind][2]

        Lval = L_list[ind]

        #print(state_no)
        #print("state num size = ",len(state_no))
        #print("i = ",ind)
        #print(state_no[ind+1])
        
        state_num_val = state_no[ind]

        

        if(state_num_val==0):    
            F3_drive = threebody_path + "/test_files/F3_for_pole_KKpi_L" + str(int(Lval)) + "/"
            F3_file = F3_drive + "ultraHQ_F3_for_pole_KKpi_L" + str(int(Lval)) + "_nP_" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    
            (En1, Ecm1, norm1, F3, F2, G, K2inv, Hinv) = np.genfromtxt(F3_file,unpack=True)
            F3inv = np.zeros((len(F3)))
            for i in range(0,len(F3),1):
                F3inv[i] = 1.0/F3[i]

            F3inv_poles_drive = threebody_path + "/test_files/F3inv_poles_L" + str(int(Lval)) + "/"    
            F3inv_poles_file = F3inv_poles_drive + "F3inv_poles_region_nP_" + str(nPx) + str(nPy) + str(nPz) + "_L" + str(int(Lval)) + ".dat"
    
            (L1, F3inv_poles, F3inv_poles_region_start, F3inv_poles_region_end) = np.genfromtxt(F3inv_poles_file, unpack=True)


        Energy_A_CM = F3inv_poles_region_start[state_num_val] #+ energy_eps #F3inv_poles[state_num_val] + energy_eps
        Energy_B_CM = F3inv_poles_region_end[state_num_val] #- energy_eps #F3inv_poles[state_num_val + 1] - energy_eps

        print("Energy_A_Cm = ",Energy_A_CM)
        print("Energy_B_CM = ",Energy_B_CM)
        for i in range(0,len(Ecm1)-1,1):
            if(Energy_A_CM>=Ecm1[i] and Energy_A_CM<=Ecm1[i+1]):
                ind1 = i
            if(Energy_B_CM>=Ecm1[i] and Energy_B_CM<=Ecm1[i+1]):
                ind2 = i
        print("ind1 = ",ind1)
        print("ind2 = ",ind2)
        Energy_A_CM = Ecm1[ind1 + 5]
        Energy_B_CM = Ecm1[ind2 - 5]

        #This is extra and we are putting this by hand
        if(int(Lval)==24):
            Energy_A_CM = Ecm1[ind1 + 1]
            Energy_B_CM = Ecm1[ind2 - 1]


        actual_ind1 = ind1 + 5
        actual_ind2 = ind2 - 5 

        print("-----------------K3isoFit---------------------")
        print("P = ",nPx, nPy, nPz, " L = ",Lval)
        print("cutoff = ",energy_cutoff)
        print("state no = ",state_no[ind])
        print("ECM A = ",Energy_A_CM)
        print("ECM B = ",Energy_B_CM)
        print("K3iso = ",K3iso_1)
        QC_spectrum = QC3_bissection_interp1d_based_multiL(ind1, ind2, Ecm1, F3inv, Energy_A_CM, Energy_B_CM, K3iso_1, 0.0, nPx, nPy, nPz, nmax, tol, threeparticle_threshold_in_s)#QC3_bissection_spline_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, spline_size)
        #QC_spectrum = QC3_bissection_spline_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, spline_size, energy_eps)
        print("bissection result = ",QC_spectrum)
        Diff = abs((states_avg[ind] - QC_spectrum)/states_avg[ind])*100.0
        print("Ecm_latt = ",states_avg[ind]," Ecm_QC = ",QC_spectrum, " Diff = ",Diff,"%")
        QC_states.append(QC_spectrum)
        print("---------------------------------------------")

    #energy_cutoff = 0.37
    np_QC_states = np.array(QC_states)

    chisquare = 0.0 
    
    '''
    for i in range(E_size):
        iterm = (np_Elatt_CM_selected[i] - np_E_QC_CM[i])
        chisquare_val = iterm*iterm 
        chisquare = chisquare + chisquare_val

    '''
    E_size = len(states_avg)
    for i in range(0,E_size,1):
        for j in range(0,E_size,1):
            #iterm = (states_avg[i] - np_QC_states[i])/states_err[i]
            #jterm = (states_avg[j] - np_QC_states[j])/states_err[j]
            iterm = (states_avg[i] - np_QC_states[i]) #/states_err[i]
            jterm = (states_avg[j] - np_QC_states[j]) #/states_err[j]
            
            
            chisquare_val = iterm*covariance_matrix_inv[i][j]*jterm
            chisquare = chisquare + chisquare_val  

    print("cutoff = ",energy_cutoff) 
    print("ndof = ",E_size,"-",int(fit_parameter_no),"=",abs(E_size - fit_parameter_no))
    print("chisquare = ",chisquare)
    print("chisq per dof = ",chisquare/(E_size - fit_parameter_no))
    print("--------------------------")
    print("\n")
    return chisquare 



def K3iso_fitting_function_multiL_twoparameter_interp1d_based(x0, nmax, states_avg, states_err, nP_list, state_no, L_list, covariance_matrix_inv, tol, threeparticle_threshold_in_s, energy_cutoff):
    energy_eps = 1.0E-5
    K3iso_1 = x0[0]
    K3iso_2 = x0[1]

    fit_parameter_no = float(len(x0))

    QC_states = []

    #for i in range(len(state_no)):
    #    print("state nums = ",i,state_no[i])
    
    for ind in range(0,len(states_avg),1):
        state_ecm = states_avg[ind]
        nPx = nP_list[ind][0]
        nPy = nP_list[ind][1]
        nPz = nP_list[ind][2]

        Lval = L_list[ind]

        #print(state_no)
        #print("state num size = ",len(state_no))
        #print("i = ",ind)
        #print(state_no[ind+1])
        
        state_num_val = state_no[ind]

        

        if(state_num_val==0):    
            F3_drive = threebody_path + "/test_files/F3_for_pole_KKpi_L" + str(int(Lval)) + "/"
            F3_file = F3_drive + "ultraHQ_F3_for_pole_KKpi_L" + str(int(Lval)) + "_nP_" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    
            (En1, Ecm1, norm1, F3, F2, G, K2inv, Hinv) = np.genfromtxt(F3_file,unpack=True)
            F3inv = np.zeros((len(F3)))
            for i in range(0,len(F3),1):
                F3inv[i] = 1.0/F3[i]

            F3inv_poles_drive = threebody_path + "/test_files/F3inv_poles_L" + str(int(Lval)) + "/"    
            F3inv_poles_file = F3inv_poles_drive + "F3inv_poles_region_nP_" + str(nPx) + str(nPy) + str(nPz) + "_L" + str(int(Lval)) + ".dat"
    
            (L1, F3inv_poles, F3inv_poles_region_start, F3inv_poles_region_end) = np.genfromtxt(F3inv_poles_file, unpack=True)


        Energy_A_CM = F3inv_poles_region_start[state_num_val] #+ energy_eps #F3inv_poles[state_num_val] + energy_eps
        Energy_B_CM = F3inv_poles_region_end[state_num_val] #- energy_eps #F3inv_poles[state_num_val + 1] - energy_eps

        print("Energy_A_Cm = ",Energy_A_CM)
        print("Energy_B_CM = ",Energy_B_CM)
        for i in range(0,len(Ecm1)-1,1):
            if(Energy_A_CM>=Ecm1[i] and Energy_A_CM<=Ecm1[i+1]):
                ind1 = i
            if(Energy_B_CM>=Ecm1[i] and Energy_B_CM<=Ecm1[i+1]):
                ind2 = i
        print("ind1 = ",ind1)
        print("ind2 = ",ind2)
        Energy_A_CM = Ecm1[ind1 + 5]
        Energy_B_CM = Ecm1[ind2 - 5]

        actual_ind1 = ind1 + 5
        actual_ind2 = ind2 - 5 

        print("-----------------K3isoFit---------------------")
        print("P = ",nPx, nPy, nPz, " L = ",Lval)
        print("cutoff = ",energy_cutoff)
        print("state no = ",state_no[ind])
        print("ECM A = ",Energy_A_CM)
        print("ECM B = ",Energy_B_CM)
        print("K3iso = ",K3iso_1)
        print("K3iso2 = ",K3iso_2)             
        QC_spectrum = QC3_bissection_interp1d_based_multiL(ind1, ind2, Ecm1, F3inv, Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, threeparticle_threshold_in_s)#QC3_bissection_spline_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, spline_size)
        #QC_spectrum = QC3_bissection_spline_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, spline_size, energy_eps)
        print("bissection result = ",QC_spectrum)
        Diff = abs((states_avg[ind] - QC_spectrum)/states_avg[ind])*100.0
        print("Ecm_latt = ",states_avg[ind]," Ecm_QC = ",QC_spectrum, " Diff = ",Diff,"%")
        print("sigma diff = ",abs((states_avg[ind] - QC_spectrum)/states_err[ind]))
        QC_states.append(QC_spectrum)
        print("---------------------------------------------")

    #energy_cutoff = 0.37
    np_QC_states = np.array(QC_states)

    chisquare = 0.0 
    
    '''
    for i in range(E_size):
        iterm = (np_Elatt_CM_selected[i] - np_E_QC_CM[i])
        chisquare_val = iterm*iterm 
        chisquare = chisquare + chisquare_val

    '''
    E_size = len(states_avg)
    for i in range(0,E_size,1):
        for j in range(0,E_size,1):
            #iterm = (states_avg[i] - np_QC_states[i])/states_err[i]
            #jterm = (states_avg[j] - np_QC_states[j])/states_err[j]
            iterm = (states_avg[i] - np_QC_states[i]) #/states_err[i]
            jterm = (states_avg[j] - np_QC_states[j]) #/states_err[j]
            
            
            chisquare_val = iterm*covariance_matrix_inv[i][j]*jterm
            chisquare = chisquare + chisquare_val  

    print("cutoff = ",energy_cutoff) 
    print("ndof = ",E_size,"-",int(fit_parameter_no),"=",abs(E_size - fit_parameter_no))
    print("chisquare = ",chisquare)
    print("chisq per dof = ",chisquare/(E_size - fit_parameter_no))
    print("--------------------------")
    print("\n")
    return chisquare 

def K3iso_fitting_function_multiL_twoparameter_interp1d_based_K3iso0_fixed(x0, K3iso0, nmax, states_avg, states_err, nP_list, state_no, L_list, covariance_matrix_inv, tol, threeparticle_threshold_in_s, energy_cutoff):
    energy_eps = 1.0E-5
    K3iso_1 = K3iso0 
    K3iso_2 = x0[0]

    fit_parameter_no = float(len(x0))

    QC_states = []

    #for i in range(len(state_no)):
    #    print("state nums = ",i,state_no[i])
    
    for ind in range(0,len(states_avg),1):
        state_ecm = states_avg[ind]
        nPx = nP_list[ind][0]
        nPy = nP_list[ind][1]
        nPz = nP_list[ind][2]

        Lval = L_list[ind]

        #print(state_no)
        #print("state num size = ",len(state_no))
        #print("i = ",ind)
        #print(state_no[ind+1])
        
        state_num_val = state_no[ind]

        

        if(state_num_val==0):    
            F3_drive = threebody_path + "/test_files/F3_for_pole_KKpi_L" + str(int(Lval)) + "/"
            F3_file = F3_drive + "ultraHQ_F3_for_pole_KKpi_L" + str(int(Lval)) + "_nP_" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    
            (En1, Ecm1, norm1, F3, F2, G, K2inv, Hinv) = np.genfromtxt(F3_file,unpack=True)
            F3inv = np.zeros((len(F3)))
            for i in range(0,len(F3),1):
                F3inv[i] = 1.0/F3[i]

            F3inv_poles_drive = threebody_path + "/test_files/F3inv_poles_L" + str(int(Lval)) + "/"    
            F3inv_poles_file = F3inv_poles_drive + "F3inv_poles_region_nP_" + str(nPx) + str(nPy) + str(nPz) + "_L" + str(int(Lval)) + ".dat"
    
            (L1, F3inv_poles, F3inv_poles_region_start, F3inv_poles_region_end) = np.genfromtxt(F3inv_poles_file, unpack=True)


        Energy_A_CM = F3inv_poles_region_start[state_num_val] #+ energy_eps #F3inv_poles[state_num_val] + energy_eps
        Energy_B_CM = F3inv_poles_region_end[state_num_val] #- energy_eps #F3inv_poles[state_num_val + 1] - energy_eps

        print("Energy_A_Cm = ",Energy_A_CM)
        print("Energy_B_CM = ",Energy_B_CM)
        for i in range(0,len(Ecm1)-1,1):
            if(Energy_A_CM>=Ecm1[i] and Energy_A_CM<=Ecm1[i+1]):
                ind1 = i
            if(Energy_B_CM>=Ecm1[i] and Energy_B_CM<=Ecm1[i+1]):
                ind2 = i
        print("ind1 = ",ind1)
        print("ind2 = ",ind2)
        Energy_A_CM = Ecm1[ind1 + 5]
        Energy_B_CM = Ecm1[ind2 - 5]

        actual_ind1 = ind1 + 5
        actual_ind2 = ind2 - 5 

        print("-----------------K3isoFit---------------------")
        print("P = ",nPx, nPy, nPz, " L = ",Lval)
        print("cutoff = ",energy_cutoff)
        print("state no = ",state_no[ind])
        print("ECM A = ",Energy_A_CM)
        print("ECM B = ",Energy_B_CM)
        print("K3iso = ",K3iso_1)
        print("K3iso2 = ",K3iso_2)             
        QC_spectrum = QC3_bissection_interp1d_based_multiL(ind1, ind2, Ecm1, F3inv, Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, threeparticle_threshold_in_s)#QC3_bissection_spline_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, spline_size)
        #QC_spectrum = QC3_bissection_spline_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, spline_size, energy_eps)
        print("bissection result = ",QC_spectrum)
        Diff = abs((states_avg[ind] - QC_spectrum)/states_avg[ind])*100.0
        print("Ecm_latt = ",states_avg[ind]," Ecm_QC = ",QC_spectrum, " Diff = ",Diff,"%")
        print("sigma diff = ",abs((states_avg[ind] - QC_spectrum)/states_err[ind]))
        QC_states.append(QC_spectrum)
        print("---------------------------------------------")

    #energy_cutoff = 0.37
    np_QC_states = np.array(QC_states)

    chisquare = 0.0 
    
    '''
    for i in range(E_size):
        iterm = (np_Elatt_CM_selected[i] - np_E_QC_CM[i])
        chisquare_val = iterm*iterm 
        chisquare = chisquare + chisquare_val

    '''
    E_size = len(states_avg)
    for i in range(0,E_size,1):
        for j in range(0,E_size,1):
            #iterm = (states_avg[i] - np_QC_states[i])/states_err[i]
            #jterm = (states_avg[j] - np_QC_states[j])/states_err[j]
            iterm = (states_avg[i] - np_QC_states[i]) #/states_err[i]
            jterm = (states_avg[j] - np_QC_states[j]) #/states_err[j]
            
            
            chisquare_val = iterm*covariance_matrix_inv[i][j]*jterm
            chisquare = chisquare + chisquare_val  

    print("cutoff = ",energy_cutoff) 
    print("ndof = ",E_size,"-",int(fit_parameter_no),"=",abs(E_size - fit_parameter_no))
    print("chisquare = ",chisquare)
    print("chisq per dof = ",chisquare/(E_size - fit_parameter_no))
    print("--------------------------")
    print("\n")
    return chisquare 


def K3iso_fitting_function_all_moms_two_parameter_secant(x0, nmax, states_avg, states_err, nP_list, state_no, covariance_matrix_inv, tol, spline_size):
    energy_eps = 1.0E-5
    K3iso_1 = x0[0]
    K3iso_2 = x0[1]

    QC_states = []

    #for i in range(len(state_no)):
    #    print("state nums = ",i,state_no[i])
    
    for ind in range(0,len(states_avg),1):
        state_ecm = states_avg[ind]
        nPx = nP_list[ind][0]
        nPy = nP_list[ind][1]
        nPz = nP_list[ind][2]

        #print(state_no)
        #print("state num size = ",len(state_no))
        #print("i = ",ind)
        #print(state_no[ind+1])
        
        state_num_val = state_no[ind]

        

        if(state_num_val==0):    
            F3_drive = threebody_path + "/test_files/F3_for_pole_KKpi_L20/"
            F3_file = F3_drive + "ultraHQ_F3_for_pole_KKpi_L20_nP_" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    
            (En1, Ecm1, norm1, F3, F2, G, K2inv, Hinv) = np.genfromtxt(F3_file,unpack=True)
            F3inv = np.zeros((len(F3)))
            for i in range(0,len(F3),1):
                F3inv[i] = 1.0/F3[i]

            F3inv_poles_drive = threebody_path + "/test_files/F3inv_poles_L20/"    
            F3inv_poles_file = F3inv_poles_drive + "F3inv_poles_region_nP_" + str(nPx) + str(nPy) + str(nPz) + "_L20.dat"
    
            (L1, F3inv_poles, F3inv_poles_region_start, F3inv_poles_region_end) = np.genfromtxt(F3inv_poles_file, unpack=True)


        Energy_A_CM = F3inv_poles_region_start[state_num_val] #+ energy_eps #F3inv_poles[state_num_val] + energy_eps
        Energy_B_CM = F3inv_poles_region_end[state_num_val] #- energy_eps #F3inv_poles[state_num_val + 1] - energy_eps

        print("Energy_A_Cm = ",Energy_A_CM)
        print("Energy_B_CM = ",Energy_B_CM)
        for i in range(0,len(Ecm1)-1,1):
            if(Energy_A_CM>=Ecm1[i] and Energy_A_CM<=Ecm1[i+1]):
                ind1 = i
            if(Energy_B_CM>=Ecm1[i] and Energy_B_CM<=Ecm1[i+1]):
                ind2 = i
        print("ind1 = ",ind1)
        print("ind2 = ",ind2)
        Energy_A_CM = Ecm1[ind1 + 5]
        Energy_B_CM = Ecm1[ind2 - 5]

        print("-----------------K3isoFit---------------------")
        print("P = ",nPx, nPy, nPz)
        print("state no = ",state_no[ind])
        print("ECM A = ",Energy_A_CM)
        print("ECM B = ",Energy_B_CM)
        print("K3iso = ",K3iso_1)
        print("K3iso2 = ",K3iso_2)             
        QC_spectrum = QC3_secant_eigen_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol)#QC3_bissection_spline_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, spline_size)
        #QC_spectrum = QC3_bissection_spline_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, spline_size, energy_eps)
        print("bissection result = ",QC_spectrum)
        Diff = abs((states_avg[ind] - QC_spectrum)/states_avg[ind])*100.0
        print("Ecm_latt = ",states_avg[ind]," Ecm_QC = ",QC_spectrum, " Diff = ",Diff,"%")
        QC_states.append(QC_spectrum)
        print("---------------------------------------------")

    #energy_cutoff = 0.37
    np_QC_states = np.array(QC_states)

    chisquare = 0.0 
    
    '''
    for i in range(E_size):
        iterm = (np_Elatt_CM_selected[i] - np_E_QC_CM[i])
        chisquare_val = iterm*iterm 
        chisquare = chisquare + chisquare_val

    '''
    E_size = len(states_avg)
    for i in range(0,E_size,1):
        for j in range(0,E_size,1):
            iterm = (states_avg[i] - np_QC_states[i])/states_err[i]
            jterm = (states_avg[j] - np_QC_states[j])/states_err[j]
            chisquare_val = iterm*covariance_matrix_inv[i][j]*jterm
            chisquare = chisquare + chisquare_val  
     

    print("chisquare = ",chisquare)
    print("--------------------------")
    print("\n")
    return chisquare 


#This is spline based 
def K3iso_fitting_function_all_moms_one_parameter(x0, nmax, states_avg, states_err, nP_list, state_no, covariance_matrix_inv, tol, spline_size):
    energy_eps = 1.0E-5
    K3iso_1 = x0[0]
    K3iso_2 = 0.0 #x0[1]

    QC_states = []

    #for i in range(len(state_no)):
    #    print("state nums = ",i,state_no[i])
    
    for ind in range(0,len(states_avg),1):
        state_ecm = states_avg[ind]
        nPx = nP_list[ind][0]
        nPy = nP_list[ind][1]
        nPz = nP_list[ind][2]

        #print(state_no)
        #print("state num size = ",len(state_no))
        #print("i = ",ind)
        #print(state_no[ind+1])
        
        state_num_val = state_no[ind]

        

        if(state_num_val==0):    
            F3_drive = threebody_path + "/test_files/F3_for_pole_KKpi_L20/"
            F3_file = F3_drive + "ultraHQ_F3_for_pole_KKpi_L20_nP_" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    
            (En1, Ecm1, norm1, F3, F2, G, K2inv, Hinv) = np.genfromtxt(F3_file,unpack=True)
            F3inv = np.zeros((len(F3)))
            for i in range(0,len(F3),1):
                F3inv[i] = 1.0/F3[i]

            F3inv_poles_drive = threebody_path + "/test_files/F3inv_poles_L20/"    
            F3inv_poles_file = F3inv_poles_drive + "F3inv_poles_region_nP_" + str(nPx) + str(nPy) + str(nPz) + "_L20.dat"
    
            (L1, F3inv_poles, F3inv_poles_region_start, F3inv_poles_region_end) = np.genfromtxt(F3inv_poles_file, unpack=True)


        Energy_A_CM = F3inv_poles_region_start[state_num_val] #+ energy_eps #F3inv_poles[state_num_val] + energy_eps
        Energy_B_CM = F3inv_poles_region_end[state_num_val] #- energy_eps #F3inv_poles[state_num_val + 1] - energy_eps

        print("Energy_A_Cm = ",Energy_A_CM)
        print("Energy_B_CM = ",Energy_B_CM)
        for i in range(0,len(Ecm1)-1,1):
            if(Energy_A_CM>=Ecm1[i] and Energy_A_CM<=Ecm1[i+1]):
                ind1 = i
            if(Energy_B_CM>=Ecm1[i] and Energy_B_CM<=Ecm1[i+1]):
                ind2 = i
        print("ind1 = ",ind1)
        print("ind2 = ",ind2)
        Energy_A_CM = Ecm1[ind1 + 5]
        Energy_B_CM = Ecm1[ind2 - 5]

        print("-----------------K3isoFit---------------------")
        print("P = ",nPx, nPy, nPz)
        print("state no = ",state_no[ind])
        print("ECM A = ",Energy_A_CM)
        print("ECM B = ",Energy_B_CM)
        print("K3iso = ",K3iso_1)
        print("K3iso2 = ",K3iso_2)             
        QC_spectrum = QC3_bissection_eigen_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol)#QC3_bissection_spline_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, spline_size)
        #QC_spectrum = QC3_bissection_spline_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, spline_size, energy_eps)
        print("bissection result = ",QC_spectrum)
        Diff = abs((states_avg[ind] - QC_spectrum)/states_avg[ind])*100.0
        print("Ecm_latt = ",states_avg[ind]," Ecm_QC = ",QC_spectrum, " Diff = ",Diff,"%")
        QC_states.append(QC_spectrum)
        print("---------------------------------------------")

    #energy_cutoff = 0.37
    np_QC_states = np.array(QC_states)

    chisquare = 0.0 
    
    '''
    for i in range(E_size):
        iterm = (np_Elatt_CM_selected[i] - np_E_QC_CM[i])
        chisquare_val = iterm*iterm 
        chisquare = chisquare + chisquare_val

    '''
    E_size = len(states_avg)
    for i in range(0,E_size,1):
        for j in range(0,E_size,1):
            iterm = (states_avg[i] - np_QC_states[i])/states_err[i]
            jterm = (states_avg[j] - np_QC_states[j])/states_err[j]
            chisquare_val = iterm*covariance_matrix_inv[i][j]*jterm
            chisquare = chisquare + chisquare_val  
     

    print("chisquare = ",chisquare)
    print("--------------------------")
    print("\n")
    return chisquare 

def QC_spectrum_one_parameter(x0, nmax, states_avg, states_err, nP_list, state_no, tol):
    energy_eps = 1.0E-3
    K3iso_1 = x0[0]
    K3iso_2 = 0.0 #x0[1]

    QC_states = []

    #for i in range(len(state_no)):
    #    print("state nums = ",i,state_no[i])
    
    for ind in range(0,len(states_avg),1):
        state_ecm = states_avg[ind]
        nPx = nP_list[ind][0]
        nPy = nP_list[ind][1]
        nPz = nP_list[ind][2]

        #print(state_no)
        #print("state num size = ",len(state_no))
        #print("i = ",ind)
        #print(state_no[ind+1])
        
        state_num_val = state_no[ind]

        

        if(state_num_val==0):    
            F3_drive = threebody_path + "/test_files/F3_for_pole_KKpi_L20/"
            F3_file = F3_drive + "ultraHQ_F3_for_pole_KKpi_L20_nP_" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    
            (En1, Ecm1, norm1, F3, F2, G, K2inv, Hinv) = np.genfromtxt(F3_file,unpack=True)
            F3inv = np.zeros((len(F3)))
            for i in range(0,len(F3),1):
                F3inv[i] = 1.0/F3[i]

            F3inv_poles_drive = threebody_path + "/test_files/F3inv_poles_L20/"    
            F3inv_poles_file = F3inv_poles_drive + "F3inv_poles_region_nP_" + str(nPx) + str(nPy) + str(nPz) + "_L20.dat"
    
            (L1, F3inv_poles, F3inv_poles_region_start, F3inv_poles_region_end) = np.genfromtxt(F3inv_poles_file, unpack=True)


        Energy_A_CM = F3inv_poles_region_start[state_num_val] #+ energy_eps #F3inv_poles[state_num_val] + energy_eps
        Energy_B_CM = F3inv_poles_region_end[state_num_val] #- energy_eps #F3inv_poles[state_num_val + 1] - energy_eps

        print("Energy_A_Cm = ",Energy_A_CM)
        print("Energy_B_CM = ",Energy_B_CM)
        for i in range(0,len(Ecm1)-1,1):
            if(Energy_A_CM>=Ecm1[i] and Energy_A_CM<=Ecm1[i+1]):
                ind1 = i
            if(Energy_B_CM>=Ecm1[i] and Energy_B_CM<=Ecm1[i+1]):
                ind2 = i
        print("ind1 = ",ind1)
        print("ind2 = ",ind2)
        Energy_A_CM = Ecm1[ind1 + 5]
        Energy_B_CM = Ecm1[ind2 - 5]

        print("-----------------K3isoFit---------------------")
        print("P = ",nPx, nPy, nPz)
        print("state no = ",state_no[ind])
        print("ECM A = ",Energy_A_CM)
        print("ECM B = ",Energy_B_CM)
        print("K3iso = ",K3iso_1)
        print("K3iso2 = ",K3iso_2)             
        QC_spectrum = QC3_bissection_eigen_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol)#QC3_bissection_spline_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, spline_size)
        #QC_spectrum = QC3_bissection_spline_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, spline_size, energy_eps)
        print("bissection result = ",QC_spectrum)
        Diff = abs((states_avg[ind] - QC_spectrum)/states_avg[ind])*100.0
        print("Ecm_latt = ",states_avg[ind]," Ecm_QC = ",QC_spectrum, " Diff = ",Diff,"%")
        QC_states.append(QC_spectrum)
        print("---------------------------------------------")

    #energy_cutoff = 0.37
    np_QC_states = np.array(QC_states)

    state_counter = 0
    filename = "QC_states_with_K3iso_" + str(K3iso_1) + "_nP_" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    f = open(filename,'w' )
    for i in np_QC_states:
        print("QC state ",state_counter," = ",i)
        state_counter = state_counter + 1 
        f.write("20" + '\t' + str(i) + '\n')

    f.close()     
 
def QC_spectrum_two_parameter(x0, nmax, states_avg, states_err, nP_list, state_no, tol):
    energy_eps = 1.0E-3
    K3iso_1 = x0[0]
    K3iso_2 = x0[1]

    QC_states = []

    #for i in range(len(state_no)):
    #    print("state nums = ",i,state_no[i])
    
    for ind in range(0,len(states_avg),1):
        state_ecm = states_avg[ind]
        nPx = nP_list[ind][0]
        nPy = nP_list[ind][1]
        nPz = nP_list[ind][2]

        #print(state_no)
        #print("state num size = ",len(state_no))
        #print("i = ",ind)
        #print(state_no[ind+1])
        
        state_num_val = state_no[ind]

        

        if(state_num_val==0):    
            F3_drive = threebody_path + "/test_files/F3_for_pole_KKpi_L20/"
            F3_file = F3_drive + "ultraHQ_F3_for_pole_KKpi_L20_nP_" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    
            (En1, Ecm1, norm1, F3, F2, G, K2inv, Hinv) = np.genfromtxt(F3_file,unpack=True)
            F3inv = np.zeros((len(F3)))
            for i in range(0,len(F3),1):
                F3inv[i] = 1.0/F3[i]

            F3inv_poles_drive = threebody_path + "/test_files/F3inv_poles_L20/"    
            F3inv_poles_file = F3inv_poles_drive + "F3inv_poles_region_nP_" + str(nPx) + str(nPy) + str(nPz) + "_L20.dat"
    
            (L1, F3inv_poles, F3inv_poles_region_start, F3inv_poles_region_end) = np.genfromtxt(F3inv_poles_file, unpack=True)


        Energy_A_CM = F3inv_poles_region_start[state_num_val] #+ energy_eps #F3inv_poles[state_num_val] + energy_eps
        Energy_B_CM = F3inv_poles_region_end[state_num_val] #- energy_eps #F3inv_poles[state_num_val + 1] - energy_eps

        print("Energy_A_Cm = ",Energy_A_CM)
        print("Energy_B_CM = ",Energy_B_CM)
        for i in range(0,len(Ecm1)-1,1):
            if(Energy_A_CM>=Ecm1[i] and Energy_A_CM<=Ecm1[i+1]):
                ind1 = i
            if(Energy_B_CM>=Ecm1[i] and Energy_B_CM<=Ecm1[i+1]):
                ind2 = i
        print("ind1 = ",ind1)
        print("ind2 = ",ind2)
        Energy_A_CM = Ecm1[ind1 + 5]
        Energy_B_CM = Ecm1[ind2 - 5]

        print("-----------------K3isoFit---------------------")
        print("P = ",nPx, nPy, nPz)
        print("state no = ",state_no[ind])
        print("ECM A = ",Energy_A_CM)
        print("ECM B = ",Energy_B_CM)
        print("K3iso1 = ",K3iso_1)
        print("K3iso2 = ",K3iso_2)             
        QC_spectrum = QC3_bissection_eigen_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol)#QC3_bissection_spline_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, spline_size)
        #QC_spectrum = QC3_bissection_spline_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, spline_size, energy_eps)
        print("bissection result = ",QC_spectrum)
        Diff = abs((states_avg[ind] - QC_spectrum)/states_avg[ind])*100.0
        print("Ecm_latt = ",states_avg[ind]," Ecm_QC = ",QC_spectrum, " Diff = ",Diff,"%")
        QC_states.append(QC_spectrum)
        print("---------------------------------------------")

    #energy_cutoff = 0.37
    np_QC_states = np.array(QC_states)

    state_counter = 0
    filename = "QC_states_with_2params_K3iso_" + str(K3iso_1) + "_" + str(K3iso_2) + "_nP_" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    f = open(filename,'w' )
    for i in np_QC_states:
        print("QC state ",state_counter," = ",i)
        state_counter = state_counter + 1 
        f.write("20" + '\t' + str(i) + '\n')

    f.close()     
 
def QC_spectrum_two_parameter_multiLs(x0, nmax, states_avg, states_err, nP_list, state_no, L_list, tol, threeparticle_threshold_in_s):
    energy_eps = 1.0E-3
    K3iso_1 = x0[0]
    K3iso_2 = x0[1]

    QC_states = []
    QC_L = []

     
    #for i in range(len(state_no)):
    #    print("state nums = ",i,state_no[i])
    
    for ind in range(0,len(states_avg),1):
        state_ecm = states_avg[ind]
        nPx = nP_list[ind][0]
        nPy = nP_list[ind][1]
        nPz = nP_list[ind][2]

        #print(state_no)
        #print("state num size = ",len(state_no))
        #print("i = ",ind)
        #print(state_no[ind+1])
        
        state_num_val = state_no[ind]

        Lval = L_list[ind]

        if(state_num_val==0):    
            F3_drive = threebody_path + "/test_files/F3_for_pole_KKpi_L" + str(int(Lval)) + "/"
            F3_file = F3_drive + "ultraHQ_F3_for_pole_KKpi_L" + str(int(Lval)) + "_nP_" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    
            (En1, Ecm1, norm1, F3, F2, G, K2inv, Hinv) = np.genfromtxt(F3_file,unpack=True)
            F3inv = np.zeros((len(F3)))
            for i in range(0,len(F3),1):
                F3inv[i] = 1.0/F3[i]

            F3inv_poles_drive = threebody_path + "/test_files/F3inv_poles_L" + str(int(Lval)) + "/"    
            F3inv_poles_file = F3inv_poles_drive + "F3inv_poles_region_nP_" + str(nPx) + str(nPy) + str(nPz) + "_L" + str(int(Lval)) + ".dat"
    
            (L1, F3inv_poles, F3inv_poles_region_start, F3inv_poles_region_end) = np.genfromtxt(F3inv_poles_file, unpack=True)


        Energy_A_CM = F3inv_poles_region_start[state_num_val] #+ energy_eps #F3inv_poles[state_num_val] + energy_eps
        Energy_B_CM = F3inv_poles_region_end[state_num_val] #- energy_eps #F3inv_poles[state_num_val + 1] - energy_eps

        print("Energy_A_Cm = ",Energy_A_CM)
        print("Energy_B_CM = ",Energy_B_CM)
        for i in range(0,len(Ecm1)-1,1):
            if(Energy_A_CM>=Ecm1[i] and Energy_A_CM<=Ecm1[i+1]):
                ind1 = i
            if(Energy_B_CM>=Ecm1[i] and Energy_B_CM<=Ecm1[i+1]):
                ind2 = i
        print("ind1 = ",ind1)
        print("ind2 = ",ind2)
        Energy_A_CM = Ecm1[ind1 + 5]
        Energy_B_CM = Ecm1[ind2 - 5]

        print("-----------------K3isoFit---------------------")
        print("P = ",nPx, nPy, nPz)
        print("state no = ",state_no[ind])
        print("ECM A = ",Energy_A_CM)
        print("ECM B = ",Energy_B_CM)
        print("K3iso1 = ",K3iso_1)
        print("K3iso2 = ",K3iso_2) 
        QC_spectrum = QC3_bissection_interp1d_based_multiL(ind1, ind2, Ecm1, F3inv, Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, threeparticle_threshold_in_s)#QC3_bissection_spline_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, spline_size)
                    
        #QC_spectrum = QC3_bissection_eigen_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol)#QC3_bissection_spline_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, spline_size)
        #QC_spectrum = QC3_bissection_spline_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, spline_size, energy_eps)
        print("bissection result = ",QC_spectrum)
        Diff = abs((states_avg[ind] - QC_spectrum)/states_avg[ind])*100.0
        print("Ecm_latt = ",states_avg[ind]," Ecm_QC = ",QC_spectrum, " Diff = ",Diff,"%")
        QC_states.append(QC_spectrum)
        QC_L.append(Lval) 
        print("---------------------------------------------")

    #energy_cutoff = 0.37
    np_QC_states = np.array(QC_states)

    state_counter = 0
    #filename = "QC_states_with_2params_K3iso_" + str(K3iso_1) + "_" + str(K3iso_2) + "_nP_" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    
    filename = "QC_states_with_2params_multiLs_L" + str(int(Lval)) + "_nP_" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    
    f = open(filename,'w' )
    for i in range(0,len(np_QC_states),1):
        QC_spec_val = np_QC_states[i]
        print("QC state ",state_counter," = ",QC_spec_val)
        state_counter = state_counter + 1 
        f.write(str(int(QC_L[i])) + '\t' + str(QC_spec_val) + '\n')

    f.close()     


#This returns the QC states along with the momentum configuration
#and L values, each of the returns are based on particular K3dfiso0
#and K3dfiso1 values.
def QC_spectrum_one_parameter_multiLs_multiK3df(x0, nmax, states_avg, states_err, nP_list, state_no, L_list, tol, threeparticle_threshold_in_s):
    energy_eps = 1.0E-3
    K3iso_1 = x0[0]

    QC_states = []
    QC_L = []
    QC_diff = [] 
    #QC_nP_list = []   
    #for i in range(len(state_no)):
    #    print("state nums = ",i,state_no[i])
    
    for ind in range(0,len(states_avg),1):
        state_ecm = states_avg[ind]
        nPx = nP_list[ind][0]
        nPy = nP_list[ind][1]
        nPz = nP_list[ind][2]

        #print(state_no)
        #print("state num size = ",len(state_no))
        #print("i = ",ind)
        #print(state_no[ind+1])
        
        state_num_val = state_no[ind]

        Lval = L_list[ind]

        if(state_num_val==0):    
            F3_drive = threebody_path + "/test_files/F3_for_pole_KKpi_L" + str(int(Lval)) + "/"
            F3_file = F3_drive + "ultraHQ_F3_for_pole_KKpi_L" + str(int(Lval)) + "_nP_" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    
            (En1, Ecm1, norm1, F3, F2, G, K2inv, Hinv) = np.genfromtxt(F3_file,unpack=True)
            F3inv = np.zeros((len(F3)))
            for i in range(0,len(F3),1):
                F3inv[i] = 1.0/F3[i]

            F3inv_poles_drive = threebody_path + "/test_files/F3inv_poles_L" + str(int(Lval)) + "/"    
            F3inv_poles_file = F3inv_poles_drive + "F3inv_poles_region_nP_" + str(nPx) + str(nPy) + str(nPz) + "_L" + str(int(Lval)) + ".dat"
    
            (L1, F3inv_poles, F3inv_poles_region_start, F3inv_poles_region_end) = np.genfromtxt(F3inv_poles_file, unpack=True)


        Energy_A_CM = F3inv_poles_region_start[state_num_val] #+ energy_eps #F3inv_poles[state_num_val] + energy_eps
        Energy_B_CM = F3inv_poles_region_end[state_num_val] #- energy_eps #F3inv_poles[state_num_val + 1] - energy_eps

        print("Energy_A_Cm = ",Energy_A_CM)
        print("Energy_B_CM = ",Energy_B_CM)
        for i in range(0,len(Ecm1)-1,1):
            if(Energy_A_CM>=Ecm1[i] and Energy_A_CM<=Ecm1[i+1]):
                ind1 = i
            if(Energy_B_CM>=Ecm1[i] and Energy_B_CM<=Ecm1[i+1]):
                ind2 = i
        print("ind1 = ",ind1)
        print("ind2 = ",ind2)
        Energy_A_CM = Ecm1[ind1 + 5]
        Energy_B_CM = Ecm1[ind2 - 5]

        print("-----------------K3isoFit---------------------")
        print("P = ",nPx, nPy, nPz)
        print("state no = ",state_no[ind])
        print("ECM A = ",Energy_A_CM)
        print("ECM B = ",Energy_B_CM)
        print("K3iso1 = ",K3iso_1)
        QC_spectrum = QC3_bissection_interp1d_based_multiL(ind1, ind2, Ecm1, F3inv, Energy_A_CM, Energy_B_CM, K3iso_1, 0.0, nPx, nPy, nPz, nmax, tol, threeparticle_threshold_in_s)#QC3_bissection_spline_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, spline_size)
                    
        #QC_spectrum = QC3_bissection_eigen_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol)#QC3_bissection_spline_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, spline_size)
        #QC_spectrum = QC3_bissection_spline_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, spline_size, energy_eps)
        print("bissection result = ",QC_spectrum)
        Diff = abs((states_avg[ind] - QC_spectrum)/states_avg[ind])*100.0
        Diff1 = abs((states_avg[ind] - QC_spectrum))/states_err[ind] 
        print("Ecm_latt = ",states_avg[ind]," Ecm_QC = ",QC_spectrum, " Diff = ",Diff,"%")
        QC_states.append(QC_spectrum)
        QC_L.append(Lval) 
        QC_diff.append(Diff1)
        print("---------------------------------------------")

    #energy_cutoff = 0.37
    np_QC_states = np.array(QC_states)
    np_QC_diff = np.array(QC_diff)

    state_counter = 0
    #filename = "QC_states_with_2params_K3iso_" + str(K3iso_1) + "_" + str(K3iso_2) + "_nP_" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    
    return L_list, np_QC_states, nP_list, state_no, np_QC_diff  
    
    '''
    filename = "QC_states_with_2params_multiLs_L" + str(int(Lval)) + "_nP_" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    
    f = open(filename,'w' )
    for i in range(0,len(np_QC_states),1):
        QC_spec_val = np_QC_states[i]
        print("QC state ",state_counter," = ",QC_spec_val)
        state_counter = state_counter + 1 
        f.write(str(int(QC_L[i])) + '\t' + str(QC_spec_val) + '\n')

    f.close()     
    '''



def QC_spectrum_two_parameter_multiLs_multiK3df(x0, nmax, states_avg, states_err, nP_list, state_no, L_list, tol, threeparticle_threshold_in_s):
    energy_eps = 1.0E-3
    K3iso_1 = x0[0]
    K3iso_2 = x0[1]

    QC_states = []
    QC_L = []
    QC_diff = [] 
    #QC_nP_list = []   
    #for i in range(len(state_no)):
    #    print("state nums = ",i,state_no[i])
    
    for ind in range(0,len(states_avg),1):
        state_ecm = states_avg[ind]
        nPx = nP_list[ind][0]
        nPy = nP_list[ind][1]
        nPz = nP_list[ind][2]

        #print(state_no)
        #print("state num size = ",len(state_no))
        #print("i = ",ind)
        #print(state_no[ind+1])
        
        state_num_val = state_no[ind]

        Lval = L_list[ind]

        if(state_num_val==0):    
            F3_drive = threebody_path + "/test_files/F3_for_pole_KKpi_L" + str(int(Lval)) + "/"
            F3_file = F3_drive + "ultraHQ_F3_for_pole_KKpi_L" + str(int(Lval)) + "_nP_" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    
            (En1, Ecm1, norm1, F3, F2, G, K2inv, Hinv) = np.genfromtxt(F3_file,unpack=True)
            F3inv = np.zeros((len(F3)))
            for i in range(0,len(F3),1):
                F3inv[i] = 1.0/F3[i]

            F3inv_poles_drive = threebody_path + "/test_files/F3inv_poles_L" + str(int(Lval)) + "/"    
            F3inv_poles_file = F3inv_poles_drive + "F3inv_poles_region_nP_" + str(nPx) + str(nPy) + str(nPz) + "_L" + str(int(Lval)) + ".dat"
    
            (L1, F3inv_poles, F3inv_poles_region_start, F3inv_poles_region_end) = np.genfromtxt(F3inv_poles_file, unpack=True)


        Energy_A_CM = F3inv_poles_region_start[state_num_val] #+ energy_eps #F3inv_poles[state_num_val] + energy_eps
        Energy_B_CM = F3inv_poles_region_end[state_num_val] #- energy_eps #F3inv_poles[state_num_val + 1] - energy_eps

        print("Energy_A_Cm = ",Energy_A_CM)
        print("Energy_B_CM = ",Energy_B_CM)
        for i in range(0,len(Ecm1)-1,1):
            if(Energy_A_CM>=Ecm1[i] and Energy_A_CM<=Ecm1[i+1]):
                ind1 = i
            if(Energy_B_CM>=Ecm1[i] and Energy_B_CM<=Ecm1[i+1]):
                ind2 = i
        print("ind1 = ",ind1)
        print("ind2 = ",ind2)
        Energy_A_CM = Ecm1[ind1 + 5]
        Energy_B_CM = Ecm1[ind2 - 5]

        print("-----------------Spectrum Generation---------------------")
        print("P = ",nPx, nPy, nPz)
        print("state no = ",state_no[ind])
        print("ECM A = ",Energy_A_CM)
        print("ECM B = ",Energy_B_CM)
        print("K3iso1 = ",K3iso_1)
        print("K3iso2 = ",K3iso_2) 
        QC_spectrum = QC3_bissection_interp1d_based_multiL(ind1, ind2, Ecm1, F3inv, Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, threeparticle_threshold_in_s)#QC3_bissection_spline_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, spline_size)
                    
        #QC_spectrum = QC3_bissection_eigen_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol)#QC3_bissection_spline_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, spline_size)
        #QC_spectrum = QC3_bissection_spline_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, spline_size, energy_eps)
        print("bissection result = ",QC_spectrum)
        Diff = abs((states_avg[ind] - QC_spectrum)/states_avg[ind])*100.0
        Diff1 = abs((states_avg[ind] - QC_spectrum))/states_err[ind] 
        print("Ecm_latt = ",states_avg[ind]," Ecm_QC = ",QC_spectrum, " Diff = ",Diff,"%")
        QC_states.append(QC_spectrum)
        QC_L.append(Lval) 
        QC_diff.append(Diff1)
        print("---------------------------------------------")

    #energy_cutoff = 0.37
    np_QC_states = np.array(QC_states)
    np_QC_diff = np.array(QC_diff)

    state_counter = 0
    #filename = "QC_states_with_2params_K3iso_" + str(K3iso_1) + "_" + str(K3iso_2) + "_nP_" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    
    return L_list, np_QC_states, nP_list, state_no, np_QC_diff  
    
    '''
    filename = "QC_states_with_2params_multiLs_L" + str(int(Lval)) + "_nP_" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    
    f = open(filename,'w' )
    for i in range(0,len(np_QC_states),1):
        QC_spec_val = np_QC_states[i]
        print("QC state ",state_counter," = ",QC_spec_val)
        state_counter = state_counter + 1 
        f.write(str(int(QC_L[i])) + '\t' + str(QC_spec_val) + '\n')

    f.close()     
    '''



#Here we fit K3df with L=20 and L=24
def test1_K3df_fitting_twoLs_one_param_state0_000_A1m_only(energy_cutoff_val):
    K3iso1 = 1141654.34618343
    initial_K3iso1 = K3iso1 
    x0 = [K3iso1]
    nmax = 1500
    tol = 1E-10 
    atmpi = 0.06906
    atmK = 0.09698
    threeparticle_threshold_in_Ecm = atmpi + 2.0*atmK 
    threeparticle_threshold_in_s = threeparticle_threshold_in_Ecm*threeparticle_threshold_in_Ecm
    #list_of_mom = ['000_A1m','100_A2','110_A2','111_A2','200_A2']
    list_of_mom = ['000_A1m']

    energy_cutoff = energy_cutoff_val#0.29 #175 #0.28 #0.30 #0.31 #0.32 #0.33208 #0.34
    max_state = 1 # the highest number of states it is going to load files for 
    # This is because some states have very noisy masses found from GEVP 

    ensemble1 = 'szscl21_20_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265'
    Lval1 = 20
    xival1 = 3.444
    ensemble2 = 'szscl21_24_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265'
    Lval2 = 24
    xival2 = 3.444
    states_avg1, states_err1, nP_list1, state_no1, L_list1, covariance_mat1, correlation_mat1 = covariance_between_states_szscl21_based(ensemble1, Lval1, xival1, energy_cutoff, list_of_mom, max_state)

    states_avg2, states_err2, nP_list2, state_no2, L_list2, covariance_mat2, correlation_mat2 = covariance_between_states_szscl21_based(ensemble2, Lval2, xival2, energy_cutoff, list_of_mom, max_state)

    states_avg = np.concatenate([states_avg1 , states_avg2]) 
    states_err = np.concatenate([states_err1 , states_err2])
    state_no = np.concatenate([state_no1, state_no2]) 
    nP_list = np.concatenate([nP_list1 , nP_list2]) 
    L_list = np.concatenate([L_list1, L_list2]) 
    covariance_mat = block_diag(covariance_mat1,covariance_mat2)

    #print(covariance_mat)

    np_cov_mat = np.array(covariance_mat)

    cov_mat_inv = np.linalg.inv(np_cov_mat)

    #print(cov_mat_inv)

    #print(cov_mat_inv @ np_cov_mat)
    print("covariance matrix inverted")

    for i in range(0,len(states_avg),1):
        print(states_avg[i],states_err[i],nP_list[i],L_list[i],state_no[i])
    
    start = timer()

    res = minimize(K3iso_fitting_function_multiL_oneparameter_interp1d_based,x0=x0,args=(nmax, states_avg, states_err, nP_list, state_no, L_list, cov_mat_inv, tol, threeparticle_threshold_in_s, energy_cutoff))
    
    end = timer()

    print("GUESS K3iso1 = ",initial_K3iso1)
    print("Energy cutoff = ",energy_cutoff)
    print(res) 
    print("time = ",end - start)

    xval_K3df_parameter = res.x 

    
    cov_mat_for_fit_parameters = res.hess_inv 
    fit_errors = np.sqrt(np.diag(cov_mat_for_fit_parameters))

    #print("")
    print("K3iso0 = ",xval_K3df_parameter[0],"+/-",fit_errors[0])
    #print("K3iso1 = ",xval_K3df_parameter[1],"+/-",fit_errors[1])

    ndof_val = float(abs(len(states_avg) - len(xval_K3df_parameter)))

    log_file = "fitting_log"

    f = open(log_file,"a")
    
    f.write("\n")
    f.write("===============================START=============================\n")
    f.write("K3iso one param fit")
    f.write("energy cutoff = " + str(energy_cutoff) + '\n')
    f.write("ndof = " + str(len(states_avg)) + " - " + str(len(xval_K3df_parameter)) + " = " + str(int(ndof_val)) + '\n' )
    f.write("chisquare = " + str(res.fun) + '\n')
    f.write("chisquare per ndof = " + str(res.fun/ndof_val) + '\n')
    f.write("\n")
    f.write("res = " + "\n")
    f.write(str(res))
    f.write("\n")
    f.write("================================END==============================\n")

    f.close() 

    return xval_K3df_parameter, fit_errors 


def test1_K3df_fitting_twoLs_one_param_state0_only(energy_cutoff_val):
    K3iso1 = 1141654.34618343
    initial_K3iso1 = K3iso1 
    x0 = [K3iso1]
    nmax = 1500
    tol = 1E-10 
    atmpi = 0.06906
    atmK = 0.09698
    threeparticle_threshold_in_Ecm = atmpi + 2.0*atmK 
    threeparticle_threshold_in_s = threeparticle_threshold_in_Ecm*threeparticle_threshold_in_Ecm
    list_of_mom = ['000_A1m','100_A2','110_A2','111_A2','200_A2']
    #list_of_mom = ['000_A1m']

    energy_cutoff = energy_cutoff_val#0.29 #175 #0.28 #0.30 #0.31 #0.32 #0.33208 #0.34
    max_state = 1 # the highest number of states it is going to load files for 
    # This is because some states have very noisy masses found from GEVP 

    ensemble1 = 'szscl21_20_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265'
    Lval1 = 20
    xival1 = 3.444
    ensemble2 = 'szscl21_24_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265'
    Lval2 = 24
    xival2 = 3.444
    states_avg1, states_err1, nP_list1, state_no1, L_list1, covariance_mat1, correlation_mat1 = covariance_between_states_szscl21_based(ensemble1, Lval1, xival1, energy_cutoff, list_of_mom, max_state)

    states_avg2, states_err2, nP_list2, state_no2, L_list2, covariance_mat2, correlation_mat2 = covariance_between_states_szscl21_based(ensemble2, Lval2, xival2, energy_cutoff, list_of_mom, max_state)

    states_avg = np.concatenate([states_avg1 , states_avg2]) 
    states_err = np.concatenate([states_err1 , states_err2])
    state_no = np.concatenate([state_no1, state_no2]) 
    nP_list = np.concatenate([nP_list1 , nP_list2]) 
    L_list = np.concatenate([L_list1, L_list2]) 
    covariance_mat = block_diag(covariance_mat1,covariance_mat2)

    #print(covariance_mat)

    np_cov_mat = np.array(covariance_mat)

    cov_mat_inv = np.linalg.inv(np_cov_mat)

    #print(cov_mat_inv)

    #print(cov_mat_inv @ np_cov_mat)
    print("covariance matrix inverted")

    for i in range(0,len(states_avg),1):
        print(states_avg[i],states_err[i],nP_list[i],L_list[i],state_no[i])
    
    start = timer()

    res = minimize(K3iso_fitting_function_multiL_oneparameter_interp1d_based,x0=x0,args=(nmax, states_avg, states_err, nP_list, state_no, L_list, cov_mat_inv, tol, threeparticle_threshold_in_s, energy_cutoff))
    
    end = timer()

    print("GUESS K3iso1 = ",initial_K3iso1)
    print("Energy cutoff = ",energy_cutoff)
    print(res) 
    print("time = ",end - start)

    xval_K3df_parameter = res.x 

    
    cov_mat_for_fit_parameters = res.hess_inv 
    fit_errors = np.sqrt(np.diag(cov_mat_for_fit_parameters))

    #print("")
    print("K3iso0 = ",xval_K3df_parameter[0],"+/-",fit_errors[0])
    #print("K3iso1 = ",xval_K3df_parameter[1],"+/-",fit_errors[1])

    ndof_val = float(abs(len(states_avg) - len(xval_K3df_parameter)))

    log_file = "fitting_log"

    f = open(log_file,"a")
    
    f.write("\n")
    f.write("===============================START=============================\n")
    f.write("K3iso one param fit")
    f.write("energy cutoff = " + str(energy_cutoff) + '\n')
    f.write("ndof = " + str(len(states_avg)) + " - " + str(len(xval_K3df_parameter)) + " = " + str(int(ndof_val)) + '\n' )
    f.write("chisquare = " + str(res.fun) + '\n')
    f.write("chisquare per ndof = " + str(res.fun/ndof_val) + '\n')
    f.write("\n")
    f.write("res = " + "\n")
    f.write(str(res))
    f.write("\n")
    f.write("================================END==============================\n")

    f.close() 

    return xval_K3df_parameter, fit_errors 


def test1_K3df_fitting_twoLs_one_param(energy_cutoff_val):
    K3iso1 = -1141654.34618343
    initial_K3iso1 = K3iso1 
    x0 = [K3iso1]
    nmax = 1500
    tol = 1E-10 
    atmpi = 0.06906
    atmK = 0.09698
    threeparticle_threshold_in_Ecm = atmpi + 2.0*atmK 
    threeparticle_threshold_in_s = threeparticle_threshold_in_Ecm*threeparticle_threshold_in_Ecm
    list_of_mom = ['000_A1m','100_A2','110_A2','111_A2','200_A2']
    #list_of_mom = ['000_A1m']

    energy_cutoff = energy_cutoff_val#0.29 #175 #0.28 #0.30 #0.31 #0.32 #0.33208 #0.34
    max_state = 10 # the highest number of states it is going to load files for 
    # This is because some states have very noisy masses found from GEVP 

    ensemble1 = 'szscl21_20_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265'
    Lval1 = 20
    xival1 = 3.444
    ensemble2 = 'szscl21_24_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265'
    Lval2 = 24
    xival2 = 3.444
    states_avg1, states_err1, nP_list1, state_no1, L_list1, covariance_mat1, correlation_mat1 = covariance_between_states_szscl21_based(ensemble1, Lval1, xival1, energy_cutoff, list_of_mom, max_state)

    states_avg2, states_err2, nP_list2, state_no2, L_list2, covariance_mat2, correlation_mat2 = covariance_between_states_szscl21_based(ensemble2, Lval2, xival2, energy_cutoff, list_of_mom, max_state)

    states_avg = np.concatenate([states_avg1 , states_avg2]) 
    states_err = np.concatenate([states_err1 , states_err2])
    state_no = np.concatenate([state_no1, state_no2]) 
    nP_list = np.concatenate([nP_list1 , nP_list2]) 
    L_list = np.concatenate([L_list1, L_list2]) 
    covariance_mat = block_diag(covariance_mat1,covariance_mat2)

    #print(covariance_mat)

    np_cov_mat = np.array(covariance_mat)

    cov_mat_inv = np.linalg.inv(np_cov_mat)

    #print(cov_mat_inv)

    #print(cov_mat_inv @ np_cov_mat)
    print("covariance matrix inverted")

    for i in range(0,len(states_avg),1):
        print(states_avg[i],states_err[i],nP_list[i],L_list[i],state_no[i])
    
    start = timer()

    res = minimize(K3iso_fitting_function_multiL_oneparameter_interp1d_based,x0=x0,args=(nmax, states_avg, states_err, nP_list, state_no, L_list, cov_mat_inv, tol, threeparticle_threshold_in_s, energy_cutoff))
    
    end = timer()

    print("GUESS K3iso1 = ",initial_K3iso1)
    print("Energy cutoff = ",energy_cutoff)
    print(res) 
    print("time = ",end - start)

    xval_K3df_parameter = res.x 

    
    cov_mat_for_fit_parameters = res.hess_inv 
    fit_errors = np.sqrt(np.diag(cov_mat_for_fit_parameters))

    #print("")
    print("K3iso0 = ",xval_K3df_parameter[0],"+/-",fit_errors[0])
    #print("K3iso1 = ",xval_K3df_parameter[1],"+/-",fit_errors[1])

    ndof_val = float(abs(len(states_avg) - len(xval_K3df_parameter)))

    log_file = "fitting_log"

    f = open(log_file,"a")
    
    f.write("\n")
    f.write("===============================START=============================\n")
    f.write("K3iso one param fit")
    f.write("energy cutoff = " + str(energy_cutoff) + '\n')
    f.write("ndof = " + str(len(states_avg)) + " - " + str(len(xval_K3df_parameter)) + " = " + str(int(ndof_val)) + '\n' )
    f.write("chisquare = " + str(res.fun) + '\n')
    f.write("chisquare per ndof = " + str(res.fun/ndof_val) + '\n')
    f.write("\n")
    f.write("res = " + "\n")
    f.write(str(res))
    f.write("\n")
    f.write("================================END==============================\n")

    f.close() 

    return xval_K3df_parameter, fit_errors 


def test1_K3df_fitting_twoLs_two_params(energy_cutoff_val):
    K3iso1 = 179534.12 #146380.78001935#-1141654.34618343
    K3iso2 = -994345.3919089 #14543829.93897527
    initial_K3iso1 = K3iso1 
    initial_K3iso2 = K3iso2
    x0 = [K3iso1, K3iso2]
    nmax = 1500
    tol = 1E-10 
    atmpi = 0.06906
    atmK = 0.09698
    threeparticle_threshold_in_Ecm = atmpi + 2.0*atmK 
    threeparticle_threshold_in_s = threeparticle_threshold_in_Ecm*threeparticle_threshold_in_Ecm
    list_of_mom = ['000_A1m','100_A2','110_A2','111_A2','200_A2']
    #list_of_mom = ['000_A1m']

    energy_cutoff = energy_cutoff_val#0.29 #175 #0.28 #0.30 #0.31 #0.32 #0.33208 #0.34
    max_state = 10 # the highest number of states it is going to load files for 
    # This is because some states have very noisy masses found from GEVP 

    ensemble1 = 'szscl21_20_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265'
    Lval1 = 20
    xival1 = 3.444
    ensemble2 = 'szscl21_24_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265'
    Lval2 = 24
    xival2 = 3.444
    states_avg1, states_err1, nP_list1, state_no1, L_list1, covariance_mat1, correlation_mat1 = covariance_between_states_szscl21_based(ensemble1, Lval1, xival1, energy_cutoff, list_of_mom, max_state)

    states_avg2, states_err2, nP_list2, state_no2, L_list2, covariance_mat2, correlation_mat2 = covariance_between_states_szscl21_based(ensemble2, Lval2, xival2, energy_cutoff, list_of_mom, max_state)

    states_avg = np.concatenate([states_avg1 , states_avg2]) 
    states_err = np.concatenate([states_err1 , states_err2])
    state_no = np.concatenate([state_no1, state_no2]) 
    nP_list = np.concatenate([nP_list1 , nP_list2]) 
    L_list = np.concatenate([L_list1, L_list2]) 
    covariance_mat = block_diag(covariance_mat1,covariance_mat2)
    correlation_mat = block_diag(correlation_mat1,correlation_mat2)

    #print(covariance_mat)

    np_cov_mat = np.array(covariance_mat)

    np_corr_mat = np.array(correlation_mat)

    cov_mat_inv = np.linalg.inv(np_cov_mat)
    corr_mat_inv = np.linalg.inv(np_corr_mat)

    print(covariance_mat)

    print("----------------------------")

    print(correlation_mat)

    #exit() 
    #print(cov_mat_inv @ np_cov_mat)
    print("covariance matrix inverted")

    for i in range(0,len(states_avg),1):
        print(states_avg[i],states_err[i],nP_list[i],L_list[i],state_no[i])
    
    start = timer()

    res = minimize(K3iso_fitting_function_multiL_twoparameter_interp1d_based,x0=x0,args=(nmax, states_avg, states_err, nP_list, state_no, L_list, cov_mat_inv, tol, threeparticle_threshold_in_s, energy_cutoff))
    
    end = timer()

    print("GUESS K3iso1 = ",initial_K3iso1)
    print("GUESS K3iso2 = ",initial_K3iso2)
    print("Energy cutoff = ",energy_cutoff)
    print(res) 
    print("time = ",end - start)

    xval_K3df_parameter = res.x 

    
    cov_mat_for_fit_parameters = res.hess_inv 
    fit_errors = np.sqrt(np.diag(cov_mat_for_fit_parameters))

    #print("")
    print("K3iso0 = ",xval_K3df_parameter[0],"+/-",fit_errors[0])
    print("K3iso1 = ",xval_K3df_parameter[1],"+/-",fit_errors[1])

    ndof_val = float(abs(len(states_avg) - len(xval_K3df_parameter)))

    log_file = "fitting_log"

    f = open(log_file,"a")
    
    f.write("\n")
    f.write("===============================START=============================\n")
    f.write("K3iso two param fit")
    f.write("energy cutoff = " + str(energy_cutoff) + '\n')
    f.write("ndof = " + str(len(states_avg)) + " - " + str(len(xval_K3df_parameter)) + " = " + str(int(ndof_val)) + '\n' )
    f.write("chisquare = " + str(res.fun) + '\n')
    f.write("chisquare per ndof = " + str(res.fun/ndof_val) + '\n')
    f.write("\n")
    f.write("res = " + "\n")
    f.write(str(res))
    f.write("\n")
    f.write("================================END==============================\n")

    f.close() 

    return xval_K3df_parameter, fit_errors 

def test1_K3df_fitting_twoLs_two_params_K3iso0_fixed(K3iso0, K3iso0err, energy_cutoff_val):
    K3iso1 = K3iso0 #-1141654.34618343
    K3iso2 = -61677465.59824166#37728004.09428708#14543829.93897527
    initial_K3iso1 = K3iso1 
    initial_K3iso2 = K3iso2
    x0 = [K3iso2]
    nmax = 1500
    tol = 1E-10 
    atmpi = 0.06906
    atmK = 0.09698
    threeparticle_threshold_in_Ecm = atmpi + 2.0*atmK 
    threeparticle_threshold_in_s = threeparticle_threshold_in_Ecm*threeparticle_threshold_in_Ecm
    list_of_mom = ['000_A1m','100_A2','110_A2','111_A2','200_A2']
    #list_of_mom = ['000_A1m']

    energy_cutoff = energy_cutoff_val#0.29 #175 #0.28 #0.30 #0.31 #0.32 #0.33208 #0.34
    max_state = 10 # the highest number of states it is going to load files for 
    # This is because some states have very noisy masses found from GEVP 

    ensemble1 = 'szscl21_20_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265'
    Lval1 = 20
    xival1 = 3.444
    ensemble2 = 'szscl21_24_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265'
    Lval2 = 24
    xival2 = 3.444
    states_avg1, states_err1, nP_list1, state_no1, L_list1, covariance_mat1, correlation_mat1 = covariance_between_states_szscl21_based(ensemble1, Lval1, xival1, energy_cutoff, list_of_mom, max_state)

    states_avg2, states_err2, nP_list2, state_no2, L_list2, covariance_mat2, correlation_mat2 = covariance_between_states_szscl21_based(ensemble2, Lval2, xival2, energy_cutoff, list_of_mom, max_state)

    states_avg = np.concatenate([states_avg1 , states_avg2]) 
    states_err = np.concatenate([states_err1 , states_err2])
    state_no = np.concatenate([state_no1, state_no2]) 
    nP_list = np.concatenate([nP_list1 , nP_list2]) 
    L_list = np.concatenate([L_list1, L_list2]) 
    covariance_mat = block_diag(covariance_mat1,covariance_mat2)
    correlation_mat = block_diag(correlation_mat1,correlation_mat2)

    #print(covariance_mat)

    np_cov_mat = np.array(covariance_mat)

    np_corr_mat = np.array(correlation_mat)

    cov_mat_inv = np.linalg.inv(np_cov_mat)
    #corr_mat_inv = np.linalg.inv(np_corr_mat)

    print(covariance_mat)

    print("----------------------------")

    print(correlation_mat)

    print("----------------------------")
    
    #exit() 
    #print(cov_mat_inv @ np_cov_mat)
    print("covariance matrix inverted")

    for i in range(0,len(states_avg),1):
        print(states_avg[i],states_err[i],nP_list[i],L_list[i],state_no[i])
    
    start = timer()

    res = minimize(K3iso_fitting_function_multiL_twoparameter_interp1d_based_K3iso0_fixed,x0=x0,args=(K3iso0, nmax, states_avg, states_err, nP_list, state_no, L_list, cov_mat_inv, tol, threeparticle_threshold_in_s, energy_cutoff))
    
    end = timer()

    print("GUESS K3iso1 = ",initial_K3iso1)
    print("GUESS K3iso2 = ",initial_K3iso2)
    print("Energy cutoff = ",energy_cutoff)
    print(res) 
    print("time = ",end - start)

    xval_K3df_parameter = res.x 

    
    cov_mat_for_fit_parameters = res.hess_inv 
    fit_errors = np.sqrt(np.diag(cov_mat_for_fit_parameters))

    returned_K3iso = [K3iso0, xval_K3df_parameter[0]]
    return_K3iso_err = [K3iso0err, fit_errors[0]]

    #print("")
    print("K3iso0 = ",K3iso0,"+/-",K3iso0err)
    print("K3iso1 = ",xval_K3df_parameter[0],"+/-",fit_errors[0])

    ndof_val = float(abs(len(states_avg) - len(xval_K3df_parameter)))

    log_file = "fitting_log"

    f = open(log_file,"a")
    
    f.write("\n")
    f.write("===============================START=============================\n")
    f.write("K3iso two param fit K3iso0 Fixed\n")
    f.write("energy cutoff = " + str(energy_cutoff) + '\n')
    f.write("ndof = " + str(len(states_avg)) + " - " + str(len(xval_K3df_parameter)) + " = " + str(int(ndof_val)) + '\n' )
    f.write("chisquare = " + str(res.fun) + '\n')
    f.write("chisquare per ndof = " + str(res.fun/ndof_val) + '\n')
    f.write("\n")
    f.write("res = " + "\n")
    f.write(str(res))
    f.write("\n")
    f.write("================================END==============================\n")

    f.close() 

    #return xval_K3df_parameter, fit_errors 
    return returned_K3iso, return_K3iso_err

#Here we determine the QC spectrum using 
#the fit parameters found from above code

def test1_QC_spectroscopy_twoLs_one_param(energy_cutoff_val, K3iso1_val):
    K3iso1 = K3iso1_val
    initial_K3iso1 = K3iso1 
    x0 = [K3iso1]
    nmax = 1500
    tol = 1E-10 
    atmpi = 0.06906
    atmK = 0.09698
    threeparticle_threshold_in_Ecm = atmpi + 2.0*atmK 
    threeparticle_threshold_in_s = threeparticle_threshold_in_Ecm*threeparticle_threshold_in_Ecm
    list_of_mom = ['000_A1m','100_A2','110_A2','111_A2','200_A2']
    #list_of_mom = ['000_A1m']

    energy_cutoff = energy_cutoff_val #0.28 #0.30 #0.31 #0.32 #0.33208 #0.34
    max_state = 15 # the highest number of states it is going to load files for 
    # This is because some states have very noisy masses found from GEVP 

    ensemble1 = 'szscl21_20_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265'
    Lval1 = 20
    xival1 = 3.444
    ensemble2 = 'szscl21_24_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265'
    Lval2 = 24
    xival2 = 3.444
    states_avg1, states_err1, nP_list1, state_no1, L_list1, covariance_mat1, correlation_mat1 = covariance_between_states_szscl21_based(ensemble1, Lval1, xival1, energy_cutoff, list_of_mom, max_state)

    states_avg2, states_err2, nP_list2, state_no2, L_list2, covariance_mat2, correlation_mat2 = covariance_between_states_szscl21_based(ensemble2, Lval2, xival2, energy_cutoff, list_of_mom, max_state)

    states_avg = np.concatenate([states_avg1 , states_avg2]) 
    states_err = np.concatenate([states_err1 , states_err2])
    state_no = np.concatenate([state_no1, state_no2]) 
    nP_list = np.concatenate([nP_list1 , nP_list2]) 
    L_list = np.concatenate([L_list1, L_list2]) 
    
    
    start = timer()

    QC_L, np_QC_states, nP_list, state_no, QC_diff = QC_spectrum_one_parameter_multiLs_multiK3df(x0, nmax, states_avg, states_err, nP_list, state_no, L_list, tol, threeparticle_threshold_in_s)
    
    end = timer()

    return QC_L, np_QC_states, nP_list, state_no, QC_diff 

    '''
    for i in range(0,len(QC_L),1):
        print(QC_L[i], np_QC_states[i], nP_list[i], state_no[i])


    print("time = ",end - start)
    '''
    


def test1_QC_spectroscopy_twoLs_two_params(energy_cutoff_val, K3iso1_val, K3iso2_val):
    K3iso1 = K3iso1_val
    K3iso2 = K3iso2_val
    initial_K3iso1 = K3iso1 
    initial_K3iso2 = K3iso2
    x0 = [K3iso1, K3iso2]
    nmax = 1500
    tol = 1E-10 
    atmpi = 0.06906
    atmK = 0.09698
    threeparticle_threshold_in_Ecm = atmpi + 2.0*atmK 
    threeparticle_threshold_in_s = threeparticle_threshold_in_Ecm*threeparticle_threshold_in_Ecm
    list_of_mom = ['000_A1m','100_A2','110_A2','111_A2','200_A2']
    #list_of_mom = ['000_A1m']

    energy_cutoff = energy_cutoff_val #0.28 #0.30 #0.31 #0.32 #0.33208 #0.34
    max_state = 15 # the highest number of states it is going to load files for 
    # This is because some states have very noisy masses found from GEVP 

    ensemble1 = 'szscl21_20_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265'
    Lval1 = 20
    xival1 = 3.444
    ensemble2 = 'szscl21_24_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265'
    Lval2 = 24
    xival2 = 3.444
    states_avg1, states_err1, nP_list1, state_no1, L_list1, covariance_mat1, correlation_mat1 = covariance_between_states_szscl21_based(ensemble1, Lval1, xival1, energy_cutoff, list_of_mom, max_state)

    states_avg2, states_err2, nP_list2, state_no2, L_list2, covariance_mat2, correlation_mat2 = covariance_between_states_szscl21_based(ensemble2, Lval2, xival2, energy_cutoff, list_of_mom, max_state)

    states_avg = np.concatenate([states_avg1 , states_avg2]) 
    states_err = np.concatenate([states_err1 , states_err2])
    state_no = np.concatenate([state_no1, state_no2]) 
    nP_list = np.concatenate([nP_list1 , nP_list2]) 
    L_list = np.concatenate([L_list1, L_list2]) 
    
    
    start = timer()

    QC_L, np_QC_states, nP_list, state_no, QC_diff = QC_spectrum_two_parameter_multiLs_multiK3df(x0, nmax, states_avg, states_err, nP_list, state_no, L_list, tol, threeparticle_threshold_in_s)
    
    end = timer()

    return QC_L, np_QC_states, nP_list, state_no, QC_diff 

    '''
    for i in range(0,len(QC_L),1):
        print(QC_L[i], np_QC_states[i], nP_list[i], state_no[i])


    print("time = ",end - start)
    '''
    






def QC_states_jackknife_resampler_oneparam(energy_cutoff_val):
    energy_cutoff = energy_cutoff_val
    list_of_moms = ["000","100","110","111","200"]
    list_of_Ls = ["20","24"]

    state_num = 20 


    for i in range(0,len(list_of_Ls),1):
        for j in range(0,len(list_of_moms),1):
            filename = "QC_states_for_jackknife_L_" + str(int(list_of_Ls[i])) + "_nP_" + list_of_moms[j] + "_state_"

            spec_vec = []
            spec_err_vec = []

            diff_vec = []
            diff_err_vec = []
            for k in range(0,state_num,1):
                filename1 = filename + str(int(k)) + "_energycutoff_" + str(energy_cutoff) + "_one_param.dat" 
                print("looking for = ",filename1)
                if(os.path.exists(filename1)):
                    print("file found = ",filename1)
                    (Lval, spec_val, K3val0, QC_diff) = np.genfromtxt(filename1,unpack=True)

                    jackknife_resampled_spec_val = jackknife_resampling(spec_val)
                    jackknife_avg_spec_val = jackknife_average(jackknife_resampled_spec_val)
                    jackknife_err_spec_val = jackknife_error(jackknife_resampled_spec_val)

                    jackknife_resampled_diff_val = jackknife_resampling(QC_diff)
                    jackknife_avg_diff_val = jackknife_average(jackknife_resampled_diff_val)
                    jackknife_err_diff_val = jackknife_error(jackknife_resampled_diff_val)

                    spec_vec.append(jackknife_avg_spec_val)
                    spec_err_vec.append(jackknife_err_spec_val) 
                    diff_vec.append(jackknife_avg_diff_val)
                    diff_err_vec.append(jackknife_err_diff_val)


            
            filename2 = "QC_states_jackknifed_L_" + str((int(list_of_Ls[i]))) + "_nP_" + list_of_moms[j] + "_energycutoff_" + str(energy_cutoff) + "_one_param.dat"

            f = open(filename2,"w")

            for l in range(0,len(spec_vec),1):
                f.write(    str((int(list_of_Ls[i]))) + '\t'
                        +   str(spec_vec[l]) + '\t'
                        +   str(spec_err_vec[l]) + '\t'
                        +   str(diff_vec[l]) + '\t'
                        +   str(diff_err_vec[l]) + '\n'
                    )
            f.close() 



def QC_states_jackknife_resampler_twoparams(energy_cutoff_val):
    energy_cutoff = energy_cutoff_val
    list_of_moms = ["000","100","110","111","200"]
    list_of_Ls = ["20","24"]

    state_num = 20 


    for i in range(0,len(list_of_Ls),1):
        for j in range(0,len(list_of_moms),1):
            filename = "QC_states_for_jackknife_L_" + str(int(list_of_Ls[i])) + "_nP_" + list_of_moms[j] + "_state_"

            spec_vec = []
            spec_err_vec = []

            diff_vec = []
            diff_err_vec = []
            for k in range(0,state_num,1):
                filename1 = filename + str(int(k)) + "_energycutoff_" + str(energy_cutoff) + "_two_params.dat" 
                print("looking for = ",filename1)
                if(os.path.exists(filename1)):
                    print("file found = ",filename1)
                    (Lval, spec_val, K3val0, K3val1, QC_diff) = np.genfromtxt(filename1,unpack=True)

                    jackknife_resampled_spec_val = jackknife_resampling(spec_val)
                    jackknife_avg_spec_val = jackknife_average(jackknife_resampled_spec_val)
                    jackknife_err_spec_val = jackknife_error(jackknife_resampled_spec_val)

                    jackknife_resampled_diff_val = jackknife_resampling(QC_diff)
                    jackknife_avg_diff_val = jackknife_average(jackknife_resampled_diff_val)
                    jackknife_err_diff_val = jackknife_error(jackknife_resampled_diff_val)

                    spec_vec.append(jackknife_avg_spec_val)
                    spec_err_vec.append(jackknife_err_spec_val) 
                    diff_vec.append(jackknife_avg_diff_val)
                    diff_err_vec.append(jackknife_err_diff_val)


            
            filename2 = "QC_states_jackknifed_L_" + str((int(list_of_Ls[i]))) + "_nP_" + list_of_moms[j] + "_energycutoff_" + str(energy_cutoff) + "_two_params.dat"

            f = open(filename2,"w")

            for l in range(0,len(spec_vec),1):
                f.write(    str((int(list_of_Ls[i]))) + '\t'
                        +   str(spec_vec[l]) + '\t'
                        +   str(spec_err_vec[l]) + '\t'
                        +   str(diff_vec[l]) + '\t'
                        +   str(diff_err_vec[l]) + '\n'
                    )
            f.close() 



#There are two cutoffs that goes in, energy_cutoff is for naming purposes
#over which the fit has been done, state_energy_cutoff is for choosing how 
#many values of lattice energy levels it will find the QC states for
def test1_K3df_fitting_and_QC_state_generator_one_param(state_energy_cutoff_val, energy_cutoff_val, K3isoval, K3isoerr):
    #K3isoval, K3isoerr = test1_K3df_fitting_twoLs_two_params()

    #K3iso0 =  1621631.7610182909 +/- 1.4696901579155754
    #K3iso1 =  1061905164.7687186 +/- 1.4696901579156867

    K3iso0 = K3isoval[0]

    K3iso0err = K3isoerr[0]

    K3iso0_initial = K3iso0 - K3iso0err 
    K3iso0_final = K3iso0 + K3iso0err 


    K3iso0vec = np.linspace(K3iso0_initial,K3iso0_final,10)

    QC_L_diffK3df = []
    QC_states_diffK3df = []
    QC_nP_list_diffK3df = []
    QC_stateno_diffK3df = [] 
    QC_K3iso_vec_diffK3df = []
    QC_state_diff = []

    print("central val = ",K3iso0)
    for i in range(0,len(K3iso0vec),1):
        K3iso0 = K3iso0vec[i]
        QC_L, np_QC_states, nP_list, state_no, QC_diff = test1_QC_spectroscopy_twoLs_one_param(state_energy_cutoff_val,K3iso0)

        QC_K3iso_vec_diffK3df.append([K3iso0])
        QC_L_diffK3df.append(QC_L)
        QC_states_diffK3df.append(np_QC_states)
        QC_nP_list_diffK3df.append(nP_list)
        QC_stateno_diffK3df.append(state_no)
        QC_state_diff.append(QC_diff)
        print("==========================================")


    print("QC_K3iso_vec_diffK3df = ",QC_K3iso_vec_diffK3df)
    print("QC_L_diffK3df = ",QC_L_diffK3df)
    print("QC_states_diffK3df = ",QC_states_diffK3df)
    print("QC_nP_list_diffK3df = ",QC_nP_list_diffK3df)
    print("QC_stateno_diffK3df = ",QC_stateno_diffK3df)
    print("QC state diffs = ",QC_state_diff)

    #Here we are going to write down the files of different QC states 
    #according to their state numbers and box sizes Ls, each of them shall 
    #have n number of data points fixed before. We can then perform jackknife 
    #resampling to get the error of the spectrum.

    for j in range(0, len(QC_states_diffK3df[0]), 1):

        selected_L_value = QC_L_diffK3df[0][j]
        nPx, nPy, nPz = QC_nP_list_diffK3df[0][j]
        state_val = QC_stateno_diffK3df[0][j]
        filename = "QC_states_for_jackknife_L_" + str(int(selected_L_value)) + "_nP_" + str(int(nPx)) + str(int(nPy)) + str(int(nPz)) + "_state_" + str(int(state_val)) + "_energycutoff_" + str(energy_cutoff_val) + "_one_param.dat"
        f = open(filename,"w")

        for i in range(0, len(QC_states_diffK3df), 1):
            K3iso0_val = QC_K3iso_vec_diffK3df[i][0]
            spec_val = QC_states_diffK3df[i][j]
            state_diff = QC_state_diff[i][j]
            f.write(        str(int(selected_L_value)) + '\t'
                    +   str(spec_val) + '\t'
                    +   str(K3iso0_val) + '\t'
                    +   str(state_diff) + '\n' 
                )

        f.close() 


#There are two cutoffs that goes in, energy_cutoff is for naming purposes
#over which the fit has been done, state_energy_cutoff is for choosing how 
#many values of lattice energy levels it will find the QC states for
def test1_K3df_fitting_and_QC_state_generator_two_params(state_energy_cutoff_val, energy_cutoff_val, K3isoval, K3isoerr):
    #K3isoval, K3isoerr = test1_K3df_fitting_twoLs_two_params()

    #K3iso0 =  1621631.7610182909 +/- 1.4696901579155754
    #K3iso1 =  1061905164.7687186 +/- 1.4696901579156867

    K3iso0 = K3isoval[0]
    K3iso1 = K3isoval[1]

    K3iso0err = K3isoerr[0]
    K3iso1err = K3isoerr[1]

    K3iso0_initial = K3iso0 - K3iso0err 
    K3iso0_final = K3iso0 + K3iso0err 

    K3iso1_initial = K3iso1 - K3iso1err 
    K3iso1_final = K3iso1 + K3iso1err 

    K3iso0vec = np.linspace(K3iso0_initial,K3iso0_final,10)
    K3iso1vec = np.linspace(K3iso1_initial,K3iso1_final,10)

    QC_L_diffK3df = []
    QC_states_diffK3df = []
    QC_nP_list_diffK3df = []
    QC_stateno_diffK3df = [] 
    QC_K3iso_vec_diffK3df = []
    QC_state_diff = []

    print("central val = ",K3iso0,",",K3iso1)
    for i in range(0,len(K3iso0vec),1):
        K3iso0 = K3iso0vec[i]
        K3iso1 = K3iso1vec[i]
        print("============Spectrum Generation============")

        print("K3iso0 = ",K3iso0,"K3iso1 = ",K3iso1)
        QC_L, np_QC_states, nP_list, state_no, QC_diff = test1_QC_spectroscopy_twoLs_two_params(state_energy_cutoff_val,K3iso0,K3iso1)

        QC_K3iso_vec_diffK3df.append([K3iso0,K3iso1])
        QC_L_diffK3df.append(QC_L)
        QC_states_diffK3df.append(np_QC_states)
        QC_nP_list_diffK3df.append(nP_list)
        QC_stateno_diffK3df.append(state_no)
        QC_state_diff.append(QC_diff)
        print("==========================================")


    print("QC_K3iso_vec_diffK3df = ",QC_K3iso_vec_diffK3df)
    print("QC_L_diffK3df = ",QC_L_diffK3df)
    print("QC_states_diffK3df = ",QC_states_diffK3df)
    print("QC_nP_list_diffK3df = ",QC_nP_list_diffK3df)
    print("QC_stateno_diffK3df = ",QC_stateno_diffK3df)
    print("QC state diffs = ",QC_state_diff)

    #Here we are going to write down the files of different QC states 
    #according to their state numbers and box sizes Ls, each of them shall 
    #have n number of data points fixed before. We can then perform jackknife 
    #resampling to get the error of the spectrum.

    for j in range(0, len(QC_states_diffK3df[0]), 1):

        selected_L_value = QC_L_diffK3df[0][j]
        nPx, nPy, nPz = QC_nP_list_diffK3df[0][j]
        state_val = QC_stateno_diffK3df[0][j]
        filename = "QC_states_for_jackknife_L_" + str(int(selected_L_value)) + "_nP_" + str(int(nPx)) + str(int(nPy)) + str(int(nPz)) + "_state_" + str(int(state_val)) + "_energycutoff_" + str(energy_cutoff_val) + "_two_params.dat"
        f = open(filename,"w")

        for i in range(0, len(QC_states_diffK3df), 1):
            K3iso0_val = QC_K3iso_vec_diffK3df[i][0]
            K3iso1_val = QC_K3iso_vec_diffK3df[i][1]
            spec_val = QC_states_diffK3df[i][j]
            state_diff = QC_state_diff[i][j]
            f.write(        str(int(selected_L_value)) + '\t'
                    +   str(spec_val) + '\t'
                    +   str(K3iso0_val) + '\t'
                    +   str(K3iso1_val) + '\t'
                    +   str(state_diff) + '\n' 
                )

        f.close() 


#This only uses the ground state of each irrep for fitting
def one_param_fitting_state0_000_A1m(energy_cutoff_val):
    K3isoval, K3isoerr = test1_K3df_fitting_twoLs_one_param_state0_000_A1m_only(energy_cutoff_val)

    state_cutoff_val = 0.42
    test1_K3df_fitting_and_QC_state_generator_one_param(state_cutoff_val, energy_cutoff_val, K3isoval, K3isoerr)

    QC_states_jackknife_resampler_oneparam(energy_cutoff_val)



def one_param_fitting_state0(energy_cutoff_val):
    K3isoval, K3isoerr = test1_K3df_fitting_twoLs_one_param_state0_only(energy_cutoff_val)

    state_cutoff_val = 0.42
    test1_K3df_fitting_and_QC_state_generator_one_param(state_cutoff_val, energy_cutoff_val, K3isoval, K3isoerr)

    QC_states_jackknife_resampler_oneparam(energy_cutoff_val)




def one_param_fitting(energy_cutoff_val):
    K3isoval, K3isoerr = test1_K3df_fitting_twoLs_one_param(energy_cutoff_val)

    state_cutoff_val = 0.42
    test1_K3df_fitting_and_QC_state_generator_one_param(state_cutoff_val, energy_cutoff_val, K3isoval, K3isoerr)

    QC_states_jackknife_resampler_oneparam(energy_cutoff_val)




def two_params_fitting(energy_cutoff_val):
    K3isoval, K3isoerr = test1_K3df_fitting_twoLs_two_params(energy_cutoff_val)

    state_cutoff_val = 0.42
    test1_K3df_fitting_and_QC_state_generator_two_params(state_cutoff_val, energy_cutoff_val, K3isoval, K3isoerr)

    QC_states_jackknife_resampler_twoparams(energy_cutoff_val)

def two_params_fitting_K3iso0_fixed(energy_cutoff_val):
    K3iso0 = 2788251.35396196
    K3iso0err = 3065027.045413032
    K3isoval, K3isoerr = test1_K3df_fitting_twoLs_two_params_K3iso0_fixed(K3iso0, K3iso0err, energy_cutoff_val)

    state_cutoff_val = 0.42
    test1_K3df_fitting_and_QC_state_generator_two_params(state_cutoff_val, energy_cutoff_val, K3isoval, K3isoerr)

    QC_states_jackknife_resampler_twoparams(energy_cutoff_val)





def main():
    energy_cutoff_val = 0.334
    #one_param_fitting_state0_000_A1m(energy_cutoff_val)
    #one_param_fitting_state0(energy_cutoff_val)
    #two_params_fitting_K3iso0_fixed(energy_cutoff_val)
    #two_params_fitting(energy_cutoff_val)
    #energy_cutoff_list = [0.29, 0.31, 0.33, 0.334, 0.335, 0.345, 0.35, 0.37]
    energy_cutoff_list = [0.29, 0.31, 0.33, 0.334, 0.335, 0.345]

    for i in range(0,len(energy_cutoff_list),1):
        energy_cutoff_val = energy_cutoff_list[i]
    #    one_param_fitting(energy_cutoff_val)
        two_params_fitting(energy_cutoff_val)
    #    two_params_fitting_K3iso0_fixed(energy_cutoff_val)

main() 


























####################################################################
##############              Temp Codes          ####################
####################################################################


def QC3_bissection_spline_based(pointA, pointB, K3iso1, K3iso2, nPx, nPy, nPz, nmax, tol, spline_size, energy_eps):
    A = pointA + energy_eps 
    B = pointB - energy_eps 
    
    F3inv_A = subprocess.check_output(['./spline_F3inv',str(nPx),str(nPy),str(nPz),str(spline_size),str(pointA),str(pointB),str(A)],shell=False)
    F3inv_result_A = F3inv_A.decode('utf-8')
    F3inv_fin_result_A = float(F3inv_result_A)
    K3iso_A = K3iso1 + K3iso2*(A*A)
    QC_A = QC3(K3iso_A, F3inv_fin_result_A)
    
    F3inv_B = subprocess.check_output(['./spline_F3inv',str(nPx),str(nPy),str(nPz),str(spline_size),str(pointA),str(pointB),str(B)],shell=False)
    F3inv_result_B = F3inv_B.decode('utf-8')
    F3inv_fin_result_B = float(F3inv_result_B)
    K3iso_B = K3iso1 + K3iso2*(B*B)
    QC_B = QC3(K3iso_B, F3inv_fin_result_B)

    print("QC_A = ",QC_A)
    print("QC_B = ",QC_B)
    
    if(QC_A==0.0):
        return A 
    elif(QC_B==0.0):
        return B 
    else:
        fin_result = 0.0
        for i in range(nmax):
            C = (A + B)/2.0 
            F3inv_C = subprocess.check_output(['./spline_F3inv',str(nPx),str(nPy),str(nPz),str(spline_size),str(pointA),str(pointB),str(C)],shell=False)
            F3inv_result_C = F3inv_C.decode('utf-8')
            F3inv_fin_result_C = float(F3inv_result_C)
            K3iso_C = K3iso1 + K3iso2*(C*C)
            QC_C = QC3(K3iso_C, F3inv_fin_result_C)
            print("QC_C = ",QC_C)
            if(QC_C==0.0 or (B-A)/2.0 < tol):
                fin_result = C 
                print("entered breaking condition for bissection with C = ",C)
                break 
            
            if(sign_func(QC_C)==sign_func(QC_A)):
                A = C 
                QC_A = QC_C 
            elif(sign_func(QC_C)==sign_func(QC_B)):
                B = C 
                QC_B = QC_C

            #fin_result = C  
        return fin_result 

def QC3_bissection_eigen_based(pointA, pointB, K3iso1, K3iso2, nPx, nPy, nPz, nmax, tol):
    A = pointA 
    B = pointB 
    
    F3inv_A = subprocess.check_output(['./eigen_F3inv',str(nPx),str(nPy),str(nPz),str(A)],shell=False)
    F3inv_result_A = F3inv_A.decode('utf-8')
    F3inv_fin_result_A = float(F3inv_result_A)
    K3iso_A = K3iso1 + K3iso2*(A*A)
    QC_A = QC3(K3iso_A, F3inv_fin_result_A)
    
    F3inv_B = subprocess.check_output(['./eigen_F3inv',str(nPx),str(nPy),str(nPz),str(B)],shell=False)
    F3inv_result_B = F3inv_B.decode('utf-8')
    F3inv_fin_result_B = float(F3inv_result_B)
    K3iso_B = K3iso1 + K3iso2*(B*B)
    QC_B = QC3(K3iso_B, F3inv_fin_result_B)

    print("QC_A = ",QC_A)
    print("QC_B = ",QC_B)
    
    if(QC_A==0.0):
        return A 
    elif(QC_B==0.0):
        return B 
    else:
        fin_result = 0.0
        for i in range(nmax):
            C = (A + B)/2.0 
            F3inv_C = subprocess.check_output(['./eigen_F3inv',str(nPx),str(nPy),str(nPz),str(C)],shell=False)
            F3inv_result_C = F3inv_C.decode('utf-8')
            F3inv_fin_result_C = float(F3inv_result_C)
            K3iso_C = K3iso1 + K3iso2*(C*C)
            QC_C = QC3(K3iso_C, F3inv_fin_result_C)
            print("QC_C = ",QC_C)
            if(abs(QC_C)<tol or abs(B-A)/2.0 < tol):
                fin_result = C 
                print("QC_C val = ",abs(QC_C), "tol = ",tol)
                print("B-A/2 = ",abs(B-A)/2.0," tol = ", tol)
                print("entered breaking condition for bissection with C = ",C)
                break 
            
            if(sign_func(QC_C)==sign_func(QC_A)):
                A = C 
                QC_A = QC_C 
            elif(sign_func(QC_C)==sign_func(QC_B)):
                B = C 
                QC_B = QC_C

            #fin_result = C  
        return fin_result 

def QC3_bissection_interp1d_based(pointA_ind, pointB_ind, Ecm, F3inv, pointA, pointB, K3iso1, K3iso2, nPx, nPy, nPz, nmax, tol):
    
    F3inv_for_interp = np.zeros((abs(pointA_ind - pointB_ind)))
    Ecm_for_interp = np.zeros((abs(pointA_ind - pointB_ind)))
    interp_counter = 0
    for i in range(pointA_ind,pointB_ind,1):
        Ecm_for_interp[interp_counter] = Ecm[i]
        F3inv_for_interp[interp_counter] = F3inv[i]
        interp_counter = interp_counter + 1 
    
    F3inv_interp = scipy.interpolate.interp1d(Ecm_for_interp,F3inv_for_interp, kind='linear')

    A = pointA 
    B = pointB 

    #new_Ecm = np.linspace(A,B,1000)
    #fig, ax = plt.subplots(figsize=(12,5))

    #ax.set_ylim(-1E8,1E8)
    #ax.set_xlim(0.26,0.37)

    #ax.plot(Ecm,F3inv, color='blue', zorder=4)
    #ax.plot(new_Ecm,F3inv_interp(new_Ecm), color='red', zorder=5)

    #plt.show()
    #exit()
    
    
    #F3inv_A = subprocess.check_output(['./eigen_F3inv',str(nPx),str(nPy),str(nPz),str(A)],shell=False)
    #F3inv_result_A = F3inv_A.decode('utf-8')
    #F3inv_fin_result_A = float(F3inv_result_A)

    F3inv_fin_result_A = F3inv_interp(A)
    K3iso_A = K3iso1 + K3iso2*(A*A)
    QC_A = QC3(K3iso_A, F3inv_fin_result_A)
    
    #F3inv_B = subprocess.check_output(['./eigen_F3inv',str(nPx),str(nPy),str(nPz),str(B)],shell=False)
    #F3inv_result_B = F3inv_B.decode('utf-8')
    #F3inv_fin_result_B = float(F3inv_result_B)

    F3inv_fin_result_B = F3inv_interp(B)
    K3iso_B = K3iso1 + K3iso2*(B*B)
    QC_B = QC3(K3iso_B, F3inv_fin_result_B)

    print("QC_A = ",QC_A)
    print("QC_B = ",QC_B)
    
    if(QC_A==0.0):
        return A 
    elif(QC_B==0.0):
        return B 
    else:
        fin_result = 0.0
        for i in range(0,nmax,1):
            C = (A + B)/2.0 
            #F3inv_C = subprocess.check_output(['./eigen_F3inv',str(nPx),str(nPy),str(nPz),str(C)],shell=False)
            #F3inv_result_C = F3inv_C.decode('utf-8')
            #F3inv_fin_result_C = float(F3inv_result_C)

            F3inv_fin_result_C = F3inv_interp(C)
            K3iso_C = K3iso1 + K3iso2*(C*C)
            QC_C = QC3(K3iso_C, F3inv_fin_result_C)
            #print("QC_C = ",QC_C)
            if( abs(QC_C) < tol or abs(B-A)/2.0 < tol/1000000 ):
                fin_result = C 
                print("QC val = ",abs(QC_C)," tol = ",tol)
                print("B-A/2 = ",abs(B-A)/2.0," tol = ", tol/1000000)
                print("entered breaking condition for bissection with C = ",C)
                break 
            
            if(sign_func(QC_C)==sign_func(QC_A)):
                A = C 
                QC_A = QC_C 
            elif(sign_func(QC_C)==sign_func(QC_B)):
                B = C 
                QC_B = QC_C

            #fin_result = C  
        return fin_result 

def QC3_secant_eigen_based(pointA, pointB, K3iso1, K3iso2, nPx, nPy, nPz, nmax, tol):
    #A = pointA 
    #B = pointB 
    
    mid = (pointA+pointB)/2.0 
    A = pointB - 2.0*tol 
    B = pointB - tol 

    print("mid = ",mid)
    print("A = ",A)
    print("B = ",B)

    for i in range(0,nmax,1):
        itemp = i
        F3inv_A = subprocess.check_output(['./eigen_F3inv',str(nPx),str(nPy),str(nPz),str(A)],shell=False)
        F3inv_result_A = F3inv_A.decode('utf-8')
        F3inv_fin_result_A = float(F3inv_result_A)
        K3iso_A = K3iso1 + K3iso2*(A*A)
        QC_A = QC3(K3iso_A, F3inv_fin_result_A)
        print("QC_A = ",QC_A)
    
        F3inv_B = subprocess.check_output(['./eigen_F3inv',str(nPx),str(nPy),str(nPz),str(B)],shell=False)
        F3inv_result_B = F3inv_B.decode('utf-8')
        F3inv_fin_result_B = float(F3inv_result_B)
        K3iso_B = K3iso1 + K3iso2*(B*B)
        QC_B = QC3(K3iso_B, F3inv_fin_result_B)
        print("QC_B = ",QC_B)

        C = A - QC_A*(A - B)/(QC_A - QC_B)

        print("C = ",C)

        if(C<=pointA or C>=pointB):
            C = random.uniform(pointA,pointB)
            print("random C chosen = ",C)
            if(C>A):
                i = itemp
                B = C 
                continue 
            elif(C<A):
                i = itemp 
                A = C 
                continue 

        F3inv_C = subprocess.check_output(['./eigen_F3inv',str(nPx),str(nPy),str(nPz),str(C)],shell=False)
        F3inv_result_C = F3inv_C.decode('utf-8')
        F3inv_fin_result_C = float(F3inv_result_C)
        K3iso_C = K3iso1 + K3iso2*(C*C)
        QC_C = QC3(K3iso_C, F3inv_fin_result_C)

        print("QC_C = ",QC_C)

        
        if(abs(QC_C)<=tol):
             pole = C 
             break 
        elif(abs(C - A)<=tol):
            pole = C 
            break 
        #point0 = point1 - f1*(point1 - point2)/(f1 - f2);

        if(i==(nmax - 1)):
            print("Secant didn't converge, choose different guess, or no pole present!!")
            pole = 0 

        B = A 
        A = C 
    
    print("pole found = ",pole)
    return pole 


#nmax is the max iteration number 
#tol is the toleration for the bissection procedure 
#spline_size is the max number of splines we set
def K3iso_fitting_function(x0, nPx, nPy, nPz, nmax, tol, spline_size, corr_mat_inv):
    energy_eps = 1.0E-5 
    K3iso_1 = x0[0]
    K3iso_2 = x0[1]

    F3_drive = threebody_path + "/test_files/F3_for_pole_KKpi_L20/"
    F3_file = F3_drive + "ultraHQ_F3_for_pole_KKpi_L20_nP_" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    
    (En1, Ecm1, norm1, F3, F2, G, K2inv, Hinv) = np.genfromtxt(F3_file,unpack=True)
    F3inv = np.zeros((len(F3)))
    for i in range(len(F3)):
        F3inv[i] = 1.0/F3[i]

    F3inv_poles_drive = threebody_path + "/test_files/F3inv_poles_L20/"    
    F3inv_poles_file = F3inv_poles_drive + "F3inv_poles_nP_" + str(nPx) + str(nPy) + str(nPz) + "_L20.dat"
    
    (L1, F3inv_poles) = np.genfromtxt(F3inv_poles_file, unpack=True)

    spectrum_drive = threebody_path + "/lattice_data/KKpi_interacting_spectrum/Three_body/L_20_only/"
    spectrum_filename = spectrum_drive + "KKpi_spectrum.P_" + str(nPx) + str(nPy) + str(nPz) + "_usethisfile"

    (L2, Elatt_CM, Elatt_CM_stat, Elatt_CM_sys) = np.genfromtxt(spectrum_filename, unpack=True)

    energy_cutoff = 0.37

    Elatt_CM_selected = []
    Elatt_CM_SE_selected = []
    for i in range(len(Elatt_CM)):
        if(Elatt_CM[i]<energy_cutoff):
            Elatt_CM_selected.append(Elatt_CM[i])
            Elatt_CM_SE_selected.append(Elatt_CM_sys[i])

    np_Elatt_CM_selected = np.array(Elatt_CM_selected) 
    np_Elatt_CM_SE_selected = np.array(Elatt_CM_SE_selected)

    E_size = len(Elatt_CM_selected)
    print("E_size = ",E_size)

    E_QC_CM = [] 

    for i in range(E_size):
        Energy_A_CM = F3inv_poles[i] + energy_eps
        Energy_B_CM = F3inv_poles[i+1] - energy_eps 

        print("ECM A = ",Energy_A_CM)
        print("ECM B = ",Energy_B_CM)
        print("K3iso1 = ",K3iso_1)
        print("K3iso2 = ",K3iso_2)
        QC_spectrum = QC3_bissection_eigen_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol)
        print("bissection result = ",QC_spectrum)
        E_QC_CM.append(QC_spectrum)

    np_E_QC_CM = np.array(E_QC_CM)

    chisquare = 0.0 
    
    '''
    for i in range(E_size):
        iterm = (np_Elatt_CM_selected[i] - np_E_QC_CM[i])
        chisquare_val = iterm*iterm 
        chisquare = chisquare + chisquare_val

    '''
    
    for i in range(E_size):
        for j in range(E_size):
            iterm = (np_Elatt_CM_selected[i] - np_E_QC_CM[i])/np_Elatt_CM_SE_selected[i]
            jterm = (np_Elatt_CM_selected[j] - np_E_QC_CM[j])/np_Elatt_CM_SE_selected[j]
            chisquare_val = iterm*corr_mat_inv[i][j]*jterm
            chisquare = chisquare + chisquare_val  
     

    print("chisquare = ",chisquare)
    print("--------------------------")
    print("\n")
    return chisquare 


#this is based on the jackknife and lattice_data_covariance code in the 
#jackknife_codes repository, this will be supplied a list of spectrum 
#with frame momenta P as [nPx, nPy, nPz], state number [0,1,2,0,1,..] along with their corresponding 
#covariance matrices to perform the fitting, this is much more robust than 
#the previous fitting function which was built for checking a single frame spectrum 
def K3iso_fitting_function_all_moms_two_parameter(x0, nmax, states_avg, states_err, nP_list, state_no, covariance_matrix_inv, tol, spline_size):
    energy_eps = 1.0E-5
    K3iso_1 = x0[0]
    K3iso_2 = x0[1]

    QC_states = []

    #for i in range(len(state_no)):
    #    print("state nums = ",i,state_no[i])
    
    for ind in range(0,len(states_avg),1):
        state_ecm = states_avg[ind]
        nPx = nP_list[ind][0]
        nPy = nP_list[ind][1]
        nPz = nP_list[ind][2]

        #print(state_no)
        #print("state num size = ",len(state_no))
        #print("i = ",ind)
        #print(state_no[ind+1])
        
        state_num_val = state_no[ind]

        

        if(state_num_val==0):    
            F3_drive = threebody_path + "/test_files/F3_for_pole_KKpi_L20/"
            F3_file = F3_drive + "ultraHQ_F3_for_pole_KKpi_L20_nP_" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    
            (En1, Ecm1, norm1, F3, F2, G, K2inv, Hinv) = np.genfromtxt(F3_file,unpack=True)
            F3inv = np.zeros((len(F3)))
            for i in range(0,len(F3),1):
                F3inv[i] = 1.0/F3[i]

            F3inv_poles_drive = threebody_path + "/test_files/F3inv_poles_L20/"    
            F3inv_poles_file = F3inv_poles_drive + "F3inv_poles_region_nP_" + str(nPx) + str(nPy) + str(nPz) + "_L20.dat"
    
            (L1, F3inv_poles, F3inv_poles_region_start, F3inv_poles_region_end) = np.genfromtxt(F3inv_poles_file, unpack=True)


        Energy_A_CM = F3inv_poles_region_start[state_num_val] #+ energy_eps #F3inv_poles[state_num_val] + energy_eps
        Energy_B_CM = F3inv_poles_region_end[state_num_val] #- energy_eps #F3inv_poles[state_num_val + 1] - energy_eps

        print("Energy_A_Cm = ",Energy_A_CM)
        print("Energy_B_CM = ",Energy_B_CM)
        for i in range(0,len(Ecm1)-1,1):
            if(Energy_A_CM>=Ecm1[i] and Energy_A_CM<=Ecm1[i+1]):
                ind1 = i
            if(Energy_B_CM>=Ecm1[i] and Energy_B_CM<=Ecm1[i+1]):
                ind2 = i
        print("ind1 = ",ind1)
        print("ind2 = ",ind2)
        Energy_A_CM = Ecm1[ind1 + 5]
        Energy_B_CM = Ecm1[ind2 - 5]

        print("-----------------K3isoFit---------------------")
        print("P = ",nPx, nPy, nPz)
        print("state no = ",state_no[ind])
        print("ECM A = ",Energy_A_CM)
        print("ECM B = ",Energy_B_CM)
        print("K3iso = ",K3iso_1)
        print("K3iso2 = ",K3iso_2)             
        QC_spectrum = QC3_bissection_eigen_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol)#QC3_bissection_spline_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, spline_size)
        #QC_spectrum = QC3_bissection_spline_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, spline_size, energy_eps)
        print("bissection result = ",QC_spectrum)
        Diff = abs((states_avg[ind] - QC_spectrum)/states_avg[ind])*100.0
        print("Ecm_latt = ",states_avg[ind]," Ecm_QC = ",QC_spectrum, " Diff = ",Diff,"%")
        QC_states.append(QC_spectrum)
        print("---------------------------------------------")

    #energy_cutoff = 0.37
    np_QC_states = np.array(QC_states)

    chisquare = 0.0 
    
    '''
    for i in range(E_size):
        iterm = (np_Elatt_CM_selected[i] - np_E_QC_CM[i])
        chisquare_val = iterm*iterm 
        chisquare = chisquare + chisquare_val

    '''
    E_size = len(states_avg)
    for i in range(0,E_size,1):
        for j in range(0,E_size,1):
            iterm = (states_avg[i] - np_QC_states[i])/states_err[i]
            jterm = (states_avg[j] - np_QC_states[j])/states_err[j]
            chisquare_val = iterm*covariance_matrix_inv[i][j]*jterm
            chisquare = chisquare + chisquare_val  
     
    print("total number of states used = ",E_size)
    print("chisquare = ",chisquare)
    print("--------------------------")
    print("\n")
    return chisquare 


#This one is based on the interpolator function 
def K3iso_fitting_function_all_moms_two_parameter_interp1d_based(x0, nmax, states_avg, states_err, nP_list, state_no, covariance_matrix_inv, tol, spline_size):
    energy_eps = 1.0E-5
    K3iso_1 = x0[0]
    K3iso_2 = x0[1]

    QC_states = []

    #for i in range(len(state_no)):
    #    print("state nums = ",i,state_no[i])
    
    for ind in range(0,len(states_avg),1):
        state_ecm = states_avg[ind]
        nPx = nP_list[ind][0]
        nPy = nP_list[ind][1]
        nPz = nP_list[ind][2]

        #print(state_no)
        #print("state num size = ",len(state_no))
        #print("i = ",ind)
        #print(state_no[ind+1])
        
        state_num_val = state_no[ind]

        

        if(state_num_val==0):    
            F3_drive = threebody_path + "/test_files/F3_for_pole_KKpi_L20/"
            F3_file = F3_drive + "ultraHQ_F3_for_pole_KKpi_L20_nP_" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    
            (En1, Ecm1, norm1, F3, F2, G, K2inv, Hinv) = np.genfromtxt(F3_file,unpack=True)
            F3inv = np.zeros((len(F3)))
            for i in range(0,len(F3),1):
                F3inv[i] = 1.0/F3[i]

            F3inv_poles_drive = threebody_path + "/test_files/F3inv_poles_L20/"    
            F3inv_poles_file = F3inv_poles_drive + "F3inv_poles_region_nP_" + str(nPx) + str(nPy) + str(nPz) + "_L20.dat"
    
            (L1, F3inv_poles, F3inv_poles_region_start, F3inv_poles_region_end) = np.genfromtxt(F3inv_poles_file, unpack=True)


        Energy_A_CM = F3inv_poles_region_start[state_num_val] #+ energy_eps #F3inv_poles[state_num_val] + energy_eps
        Energy_B_CM = F3inv_poles_region_end[state_num_val] #- energy_eps #F3inv_poles[state_num_val + 1] - energy_eps

        print("Energy_A_Cm = ",Energy_A_CM)
        print("Energy_B_CM = ",Energy_B_CM)
        for i in range(0,len(Ecm1)-1,1):
            if(Energy_A_CM>=Ecm1[i] and Energy_A_CM<=Ecm1[i+1]):
                ind1 = i
            if(Energy_B_CM>=Ecm1[i] and Energy_B_CM<=Ecm1[i+1]):
                ind2 = i
        print("ind1 = ",ind1)
        print("ind2 = ",ind2)
        Energy_A_CM = Ecm1[ind1 + 5]
        Energy_B_CM = Ecm1[ind2 - 5]

        actual_ind1 = ind1 + 3
        actual_ind2 = ind2 - 3 

        print("-----------------K3isoFit---------------------")
        print("P = ",nPx, nPy, nPz)
        print("state no = ",state_no[ind])
        print("ECM A = ",Energy_A_CM)
        print("ECM B = ",Energy_B_CM)
        print("K3iso = ",K3iso_1)
        print("K3iso2 = ",K3iso_2)             
        QC_spectrum = QC3_bissection_interp1d_based(ind1, ind2, Ecm1, F3inv, Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol)#QC3_bissection_spline_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, spline_size)
        #QC_spectrum = QC3_bissection_spline_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, spline_size, energy_eps)
        print("bissection result = ",QC_spectrum)
        Diff = abs((states_avg[ind] - QC_spectrum)/states_avg[ind])*100.0
        print("Ecm_latt = ",states_avg[ind]," Ecm_QC = ",QC_spectrum, " Diff = ",Diff,"%")
        QC_states.append(QC_spectrum)
        print("---------------------------------------------")

    #energy_cutoff = 0.37
    np_QC_states = np.array(QC_states)

    chisquare = 0.0 
    
    '''
    for i in range(E_size):
        iterm = (np_Elatt_CM_selected[i] - np_E_QC_CM[i])
        chisquare_val = iterm*iterm 
        chisquare = chisquare + chisquare_val

    '''
    E_size = len(states_avg)
    for i in range(0,E_size,1):
        for j in range(0,E_size,1):
            iterm = (states_avg[i] - np_QC_states[i])/states_err[i]
            jterm = (states_avg[j] - np_QC_states[j])/states_err[j]
            chisquare_val = iterm*covariance_matrix_inv[i][j]*jterm
            chisquare = chisquare + chisquare_val  
     
    print("total number of states used = ",E_size)
    print("chisquare = ",chisquare)
    print("chisq per dof = ",chisquare/E_size)
    print("--------------------------")
    print("\n")
    return chisquare 


def test():
    nPx = 0
    nPy = 0
    nPz = 0 
    K3iso1 = 1000000.0
    K3iso2 = 10000000.0 

    x0 = [K3iso1, K3iso2]
    nmax = 100
    tol = 1E-10 
    spline_size = 50 

    rows, cols = (3, 3)
    arr = [[0 for i in range(cols)] for j in range(rows)]
    corr_mat = arr 

    corr_mat[0][0] = 1.0
    corr_mat[0][1] = 0.91
    corr_mat[0][2] = 0.90
    corr_mat[1][0] = corr_mat[0][1]
    corr_mat[1][1] = 1.0 
    corr_mat[2][2] = 1.0 
    corr_mat[2][0] = corr_mat[0][2] 
    corr_mat[1][2] = 0.85
    corr_mat[2][1] = corr_mat[1][2]

    print("we have started running")
    np_corr_mat = np.array(corr_mat)
    corr_mat_inv = np.linalg.inv(np_corr_mat)
    print("we inverted the corr mat")
    print(corr_mat_inv)


    res = scipy.optimize.minimize(K3iso_fitting_function,x0=x0,args=(nPx, nPy, nPz, nmax, tol, spline_size, corr_mat_inv),method='Nelder-Mead')
    
    print(res) 

'''    1.00  0.91    0.90  
             1.00    0.85  
                     1.00
'''


#This one is with multiple P frames 
def test1():
    nPx = 0
    nPy = 0
    nPz = 0 
    K3iso1 = -10000#0.000127
    K3iso2 = 0.0#10000000.0 

    x0 = [K3iso1] #[K3iso1, K3iso2]
    nmax = 500
    tol = 1E-16 
    spline_size = 500 

    #list_of_mom = ['000_A1m','100_A2','110_A2','111_A2','200_A2']
    list_of_mom = ['000_A1m']
    
    states_avg, states_err, nP_list, state_no, covariance_mat = covariance_between_states_L20(0.38, list_of_mom)

    for i in range(len(states_avg)):
        print(states_avg[i],states_err[i],nP_list[i],state_no[i])
    print("we have started running")
    np_cov_mat = np.array(covariance_mat)
    cov_mat_inv = np.linalg.inv(np_cov_mat)
    print("we inverted the corr mat")
    print(np_cov_mat)
    print("------------------------")
    print(cov_mat_inv)
    print("------------------------")

    start = timer()
    #res = scipy.optimize.minimize(K3iso_fitting_function_all_moms_one_parameter,x0=x0,args=(nmax, states_avg, states_err, nP_list, state_no, cov_mat_inv, tol, spline_size),method='Nelder-Mead')
    
    #This is done using iminuit
    res = minimize(K3iso_fitting_function_all_moms_one_parameter,x0=x0,args=(nmax, states_avg, states_err, nP_list, state_no, cov_mat_inv, tol, spline_size))
    
    end = timer()

    print(res) 
    print("time = ",end - start)

def test1_two_params():
    nPx = 0
    nPy = 0
    nPz = 0 
    K3iso1 = -1141654.34618343#100000#0.000127
    K3iso2 = 14543829.93897527#-1000.0#10000000.0 
    initial_K3iso1 = K3iso1 
    initial_K3iso2 = K3iso2 

    x0 = [K3iso1, K3iso2]
    nmax = 1500
    tol = 1E-10 
    spline_size = 500 

    list_of_mom = ['000_A1m','100_A2','110_A2','111_A2','200_A2']
    #list_of_mom = ['000_A1m','100_A2']
    
    states_avg, states_err, nP_list, state_no, covariance_mat = covariance_between_states_L20(0.35, list_of_mom)

    
    for i in range(len(states_avg)):
        print(states_avg[i],states_err[i],nP_list[i],state_no[i])
    print("we have started running")
    np_cov_mat = np.array(covariance_mat)
    print("correlation matrix:")
    print(np_cov_mat)
    print("------------------------")
    
    
    cov_mat_inv = np.linalg.inv(np_cov_mat)
    
    
    print("we inverted the corr mat")
    print(cov_mat_inv)
    print("------------------------")
    #exit() 
    start = timer()
    #res = scipy.optimize.minimize(K3iso_fitting_function_all_moms_one_parameter,x0=x0,args=(nmax, states_avg, states_err, nP_list, state_no, cov_mat_inv, tol, spline_size),method='Nelder-Mead')
    
    #This is done using iminuit
    res = minimize(K3iso_fitting_function_all_moms_two_parameter_interp1d_based,x0=x0,args=(nmax, states_avg, states_err, nP_list, state_no, cov_mat_inv, tol, spline_size))
    
    end = timer()

    print("GUESS K3iso1 = ",initial_K3iso1)
    print("GUESS K3iso2 = ",initial_K3iso2)
    print(res) 
    print("time = ",end - start)


#We plot the chisquare based on K3iso1 
def test2():
    nPx = 0
    nPy = 0
    nPz = 0 
    #K3iso1 = 0.1#0.000127
    K3iso2 = 0.0#10000000.0 

    #x0 = [K3iso1] #[K3iso1, K3iso2]
    nmax = 100
    tol = 1E-10 
    spline_size = 500 

    #list_of_mom = ['000_A1m','100_A2','110_A2','111_A2','200_A2']
    list_of_mom = ['000_A1m']
    
    states_avg, states_err, nP_list, state_no, covariance_mat = covariance_between_states_L20(0.38, list_of_mom)

    for i in range(len(states_avg)):
        print(states_avg[i],states_err[i],nP_list[i],state_no[i])
    print("we have started running")
    np_cov_mat = np.array(covariance_mat)
    cov_mat_inv = np.linalg.inv(np_cov_mat)
    print("we inverted the corr mat")
    print(np_cov_mat)
    print("------------------------")
    print(cov_mat_inv)
    print("------------------------")

    start = timer()
    #res = scipy.optimize.minimize(K3iso_fitting_function_all_moms_one_parameter,x0=x0,args=(nmax, states_avg, states_err, nP_list, state_no, cov_mat_inv, tol, spline_size),method='Nelder-Mead')
    
    K3iso_space = np.linspace(2000,200000,5000)

    f = open("chisquare_vs_K3iso1_nP_000_1.dat",'w')
    for i in range(0,len(K3iso_space),1):
        K3iso1 = K3iso_space[i]
        x0 = [K3iso1]
        chisquare = K3iso_fitting_function_all_moms_one_parameter(x0, nmax, states_avg, states_err, nP_list, state_no, cov_mat_inv, tol, spline_size)
        f.write(str(K3iso1) + '\t' + str(chisquare) + '\n')
        print(K3iso1, chisquare)
    end = timer()
    f.close() 
    #print(res) 
    print("time = ",end - start)

#QC_spectrum with one parameter K3iso1 = constant 
def test3():
    nPx = 0
    nPy = 0
    nPz = 0 
    K3iso1 = 299670.6383505#1359193.91087736#-189765.8705129288#0.000127
    K3iso2 = -4146398.0878418#-11037973.63450998#0.0#10000000.0 

    x0 = [K3iso1, K3iso2]
    nmax = 500
    tol = 1E-16 
    spline_size = 500 

    list_of_mom = ['000_A1m','100_A2','110_A2','111_A2','200_A2']
    #list_of_mom = ['200_A2']
    #list_of_mom = ['000_A1m']
    for irreps in list_of_mom:
        newlist_val = [irreps]

        states_avg, states_err, nP_list, state_no, covariance_mat = covariance_between_states_L20(0.40, newlist_val)

        print("irrep = ",irreps)
        for i in range(len(states_avg)):
            print(states_avg[i],states_err[i],nP_list[i],state_no[i])
        print("we have started running")
        np_cov_mat = np.array(covariance_mat)
        cov_mat_inv = np.linalg.inv(np_cov_mat)
        print("we inverted the corr mat")
        print(np_cov_mat)
        print("------------------------")
        print(cov_mat_inv)
        print("------------------------")

        start = timer()
        #res = scipy.optimize.minimize(K3iso_fitting_function_all_moms_one_parameter,x0=x0,args=(nmax, states_avg, states_err, nP_list, state_no, cov_mat_inv, tol, spline_size),method='Nelder-Mead')
        QC_spectrum_two_parameter(x0, nmax, states_avg, states_err, nP_list, state_no, tol)
    
        end = timer() 
        #print(res) 
        print("time = ",end - start)

def test5():
    list_of_mom = ['000_A1m','100_A2','110_A2','111_A2','200_A2']
    #list_of_mom = ['200_A2']
    #list_of_mom = ['000_A1m']
    for irreps in list_of_mom:
        newlist_val = [irreps]

        states_avg, states_err, nP_list, state_no, covariance_mat = covariance_between_states_L20(0.9, newlist_val)

        print("irrep = ",irreps)
        for i in range(len(states_avg)):
            print(states_avg[i],states_err[i],nP_list[i],state_no[i])

#This was done to test the spline based code
#Check spectrum for given K3df values 
def spectrum_checker_for_QC():
    plt.rcParams.update({'font.size': 22})
    plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
    plt.rc('text', usetex=True)
    
    nPx = 0
    nPy = 0
    nPz = 0 

    F3_drive = threebody_path + "test_files/F3_for_pole_KKpi_L20/"
    F3_file = F3_drive + "ultraHQ_F3_for_pole_KKpi_L20_nP_" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    
    (En1, Ecm1, norm1, F3, F2, G, K2inv, Hinv) = np.genfromtxt(F3_file,unpack=True)
    F3inv = np.zeros((len(F3)))
    for i in range(len(F3)):
        F3inv[i] = 1.0/F3[i]

    F3inv_poles_drive = threebody_path + "test_files/F3inv_poles_L20/"    
    F3inv_poles_file = F3inv_poles_drive + "F3inv_poles_nP_" + str(nPx) + str(nPy) + str(nPz) + "_L20.dat"
    
    (L1, F3inv_poles) = np.genfromtxt(F3inv_poles_file, unpack=True)

    F3_poles_drive = threebody_path + "test_files/F3_for_pole_KKpi_L20/"
    F3_poles_file = F3_poles_drive + "F3_poles_nP_" + str(nPx) + str(nPy) + str(nPz) + "_L20.dat"
    
    (LF3, F3poles) = np.genfromtxt(F3_poles_file, unpack=True)
    
    spectrum_drive = threebody_path + "lattice_data/KKpi_interacting_spectrum/Three_body/L_20_only/"
    spectrum_filename = spectrum_drive + "KKpi_spectrum.P_" + str(nPx) + str(nPy) + str(nPz) + "_usethisfile"

    (L2, Elatt_CM, Elatt_CM_stat, Elatt_CM_sys) = np.genfromtxt(spectrum_filename, unpack=True)

    KDFnonzero_spectrum_drive = threebody_path + "test_files/QC_states_L20/"
    KDFnonzero_spectrum_file = KDFnonzero_spectrum_drive + "QC_states_with_K3iso_-189765.8705129288_nP_" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    
    (Lkdf, Kdf_spec) = np.genfromtxt(KDFnonzero_spectrum_file, unpack=True) 
    
    K3iso1 = -189765.8705129288 #-4.45893908e+07  #1336082.36755021
    K3iso2 =  0.0 #4.94547308e+08 #-10866109.92757

    QC_val = []
    for i in range(len(F3inv)):
        K3iso = K3iso1 + K3iso2*Ecm1[i]**2
        QC_temp = QC3(K3iso,F3inv[i])
        QC_val.append(QC_temp)
    
    np_y_val = np.zeros((len(F3inv_poles)))
    
    np_QC_val = np.array(QC_val)


    spline_size = 500
    pointA = F3inv_poles[0] + 1.0E-5
    pointB = F3inv_poles[1] #- 1.0E-5

    Ecm_space = np.linspace(pointA, pointB, 100)
    QC_val_spline = []

    '''
    for i in range(len(Ecm_space)):
        F3inv_C = subprocess.check_output(['./spline_F3inv',str(nPx),str(nPy),str(nPz),str(spline_size),str(pointA),str(pointB),str(Ecm_space[i])],shell=False)
        F3inv_result_C = F3inv_C.decode('utf-8')
        F3inv_fin_result_C = float(F3inv_result_C)
        print("Ecm = ",Ecm_space[i],"F3inv = ",F3inv_fin_result_C)
        K3iso = K3iso1 + K3iso2*Ecm_space[i]**2
        QC_temp = QC3(K3iso,F3inv_fin_result_C)
        QC_val_spline.append(QC_temp)

    np_QC_val_spline = np.array(QC_val_spline)            
    '''

    fig, ax = plt.subplots(figsize=(12,5))

    #ax.set_title("K3iso = "+str(K3iso1))
    ax.set_ylim(-5E6,5E6)
    ax.set_xlim(0.26,0.37)
    ax.set_ylabel("$F_{3,iso}^{-1}$")
    ax.set_xlabel("$E_{cm}$")
    ax.plot(Ecm1,F3inv)
    #ax.plot(Ecm1,np_QC_val)
    ax.axhline(y=-K3iso1,color='red',label="$\\mathcal{K}_{3,iso} $= " + str(K3iso1))
    #ax.plot(Ecm1,np_QC_val)
    #ax.plot(Ecm_space,np_QC_val_spline)
    ax.axhline(y=0,color='black')
    #ax.scatter(F3inv_poles,np_y_val,s=100,facecolor='white',edgecolor='red')
    for i in range(0,len(Elatt_CM),1):
        ax.axvline(x=Elatt_CM[i],color='darkorange')

    for i in range(0, len(F3poles), 1):
        ax.axvline(x=F3poles[i],color='black')

    for i in range(0, len(Kdf_spec), 1):
        ax.axvline(x=Kdf_spec[i],color='red')

    ax.legend() 
    fig.tight_layout() 
    plt.draw() 
    outfile = "temp.png"
    plt.savefig(outfile)
    plt.show()
    plt.close()


def spectrum_checker_for_splines():
    plt.rcParams.update({'font.size': 22})
    plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
    plt.rc('text', usetex=True)
    
    nPx = 0
    nPy = 0
    nPz = 0 

    F3_drive = threebody_path + "test_files/F3_for_pole_KKpi_L20/"
    F3_file = F3_drive + "ultraHQ_F3_for_pole_KKpi_L20_nP_" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    
    (En1, Ecm1, norm1, F3, F2, G, K2inv, Hinv) = np.genfromtxt(F3_file,unpack=True)
    F3inv = np.zeros((len(F3)))
    for i in range(len(F3)):
        F3inv[i] = 1.0/F3[i]

    F3inv_poles_drive = threebody_path + "test_files/F3inv_poles_L20/"    
    F3inv_poles_file = F3inv_poles_drive + "F3inv_poles_nP_" + str(nPx) + str(nPy) + str(nPz) + "_L20.dat"
    
    (L1, F3inv_poles) = np.genfromtxt(F3inv_poles_file, unpack=True)

    
    spectrum_drive = threebody_path + "lattice_data/KKpi_interacting_spectrum/Three_body/L_20_only/"
    spectrum_filename = spectrum_drive + "KKpi_spectrum.P_" + str(nPx) + str(nPy) + str(nPz) + "_usethisfile"

    (L2, Elatt_CM, Elatt_CM_stat, Elatt_CM_sys) = np.genfromtxt(spectrum_filename, unpack=True)


    K3iso1 =  1E-7#-4.45893908e+07  #1336082.36755021
    K3iso2 =  0.0#4.94547308e+08 #-10866109.92757

    QC_val = []
    for i in range(len(F3inv)):
        K3iso = K3iso1 + K3iso2*Ecm1[i]**2
        QC_temp = QC3(K3iso,F3inv[i])
        QC_val.append(QC_temp)
    
    np_y_val = np.zeros((len(F3inv_poles)))
    
    np_QC_val = np.array(QC_val)


    spline_size = 500
    
    QC_val_spline = []

    fig, ax = plt.subplots(figsize=(12,5))

    ax.set_ylim(-1E8,1E8)
    ax.set_xlim(0.26,0.37)

    start = timer()

    for i in range(0,len(F3inv_poles)-1,1):
        pointA = F3inv_poles[i] #+ 1.0E-7
        pointB = F3inv_poles[i+1] #- 1.0E-7
        pointC = (pointA + pointB)/2.0 

        Ecm_space1 = np.linspace(pointA, pointB, 100)
        Ecm_space2 = np.linspace(pointC, pointB, 100)

        Ecm_space = []
        F3inv_spline = []
        '''
        for j in range(0,len(Ecm_space1)-1,1):
            F3inv_C = subprocess.check_output(['./spline_F3inv',str(nPx),str(nPy),str(nPz),str(spline_size),str(pointA),str(pointB),str(Ecm_space1[j])],shell=False)
            F3inv_result_C = F3inv_C.decode('utf-8')
            F3inv_fin_result_C = float(F3inv_result_C)
            print("pointA = ",pointA,"pointB = ", pointB, "Ecm = ",Ecm_space1[j],"F3inv = ",F3inv_fin_result_C)
            #K3iso = K3iso1 + K3iso2*Ecm_space[i]**2
            #QC_temp = QC3(K3iso,F3inv_fin_result_C)
            #QC_val_spline.append(QC_temp)
            F3inv_spline.append(F3inv_fin_result_C) 
            Ecm_space.append(Ecm_space1[j])
        '''
        '''    
        for j in range(0,len(Ecm_space2)-1,1):
            F3inv_C = subprocess.check_output(['./spline_F3inv',str(nPx),str(nPy),str(nPz),str(spline_size),str(pointC),str(pointB),str(Ecm_space2[j])],shell=False)
            F3inv_result_C = F3inv_C.decode('utf-8')
            F3inv_fin_result_C = float(F3inv_result_C)
            print("pointC = ",pointC,"pointB = ", pointB, "Ecm = ",Ecm_space2[j],"F3inv = ",F3inv_fin_result_C)
            #K3iso = K3iso1 + K3iso2*Ecm_space[i]**2
            #QC_temp = QC3(K3iso,F3inv_fin_result_C)
            #QC_val_spline.append(QC_temp)
            F3inv_spline.append(F3inv_fin_result_C)
            Ecm_space.append(Ecm_space2[j])
        '''

        np_F3inv_spline = np.array(F3inv_spline)
        np_Ecm_space = np.array(Ecm_space)

        #ax.plot(np_Ecm_space, np_F3inv_spline, color='darkorange',zorder=3) 
        #ax.scatter(np_Ecm_space, np_F3inv_spline, s=100, color='darkorange',zorder=3) 

    np_QC_val_spline = np.array(QC_val_spline)            

    end = timer() 
    print("time = ",end - start)

    #ax.plot(Ecm1,F3inv, color='blue', zorder=4)
    ax.plot(Ecm1,np_QC_val, color='blue', zorder=4)
    
    #ax.plot(Ecm_space,np_QC_val_spline)
    ax.axhline(y=0,color='black')
    for i in range(0,len(F3inv_poles),1):
        ax.axvline(x=F3inv_poles[i],color='darkorange',zorder=6)
    #ax.scatter(F3inv_poles,np_y_val,s=100,facecolor='white',edgecolor='red')
    #for i in range(len(Elatt_CM)):
    #    ax.axvline(x=Elatt_CM[i],color='darkorange')
    plt.show()

#test() 
#test1()
#test1_two_params()


#spectrum_checker_for_QC()

#test2()

#test3()

#test5()

#spectrum_checker_for_splines()

####################################################################