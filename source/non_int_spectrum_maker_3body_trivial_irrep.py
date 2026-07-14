#This code makes the non-interacting spectrum for given particle with mass and
#lattice anisotropy xi and lattice volume L/a_s (Lbyas)

import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import scipy.interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os.path
from scipy.optimize import curve_fit
import scipy.interpolate
import pandas as pd 
from IPython.display import display

def config_maker( max_nsq ):
    nx = []
    ny = []
    nz = []

    count = 1
    for i in range(-max_nsq,max_nsq+1,1):
        for j in range(-max_nsq,max_nsq+1,1):
            for k in range(-max_nsq,max_nsq+1,1):
                config_sq = i*i + j*j + k*k 
                if(config_sq<=max_nsq):
                    #print("config ",count,": ",i,j,k)
                    count = count + 1 
                    nx.append(i)
                    ny.append(j)
                    nz.append(k)
                else:
                    continue
        
    return nx,ny,nz 

def config_maker_positive_only( max_nsq ):
    nx = []
    ny = []
    nz = []

    count = 1
    for i in range(0,max_nsq+1,1):
        for j in range(0,max_nsq+1,1):
            for k in range(0,max_nsq+1,1):
                config_sq = i*i + j*j + k*k 
                if(config_sq<=max_nsq):
                    print("config ",count,": ",i,j,k)
                    count = count + 1 
                    nx.append(i)
                    ny.append(j)
                    nz.append(k)
                else:
                    continue
        
    return nx,ny,nz 


def energy( atm, 
            xi, 
            Lbyas,
            nx,
            ny,
            nz  ):
    onebyxi = 1.0/xi 
    pi = np.pi
    twopibyLbyas = 2.0*pi/Lbyas 
    nsquare = nx*nx + ny*ny + nz*nz 

    return np.sqrt(atm*atm + onebyxi*onebyxi*twopibyLbyas*twopibyLbyas*nsquare)


def irrep_list_maker(energy_file):
    df = pd.read_csv(energy_file,delim_whitespace=True, header=None)
    dflist = df.values.tolist()
    irrep_list = []
    for i in range(len(dflist)):
        irrep = dflist[i][1]
    
        if len(irrep_list)==0:
            irrep_list.append(irrep)
        else:
            check_irrep_list_flag = 0
            for j in range(len(irrep_list)):
                temp_irrep = irrep_list[j]
            
                if(temp_irrep == irrep):
                    check_irrep_list_flag = 1
                    break 
            if(check_irrep_list_flag==0):
                irrep_list.append(irrep)

    return dflist, irrep_list 

def irrep_energy_list_maker(full_energy_list, fixed_irrep):
    Ecm_list = []
    Elat_list = []

    for i in range(len(full_energy_list)):
        if(full_energy_list[i][1]==fixed_irrep):
            Ecm_list.append(float(full_energy_list[i][2]))
            Elat_list.append(float(full_energy_list[i][3]))

    return Ecm_list, Elat_list 

def canonical_mom_maker(nx, ny, nz):
    temp_list = []
    temp_list.append(abs(nx))
    temp_list.append(abs(ny))
    temp_list.append(abs(nz))
    temp_list.sort(reverse=True)
    
    return temp_list

#This will be used from now on to create non-int spectrum 
#Works only for the trivial irreps 
def full_nonint_spectrum_maker_final(Lbyas_val, atmk, atmpi, xi, max_nsq, irrep, Lmin, Lmax, Lpoints):
    
    #tolerance for energy matching 
    tolerance = 1.0e-4
    tolerance2 = 1.0e-9
    #we get the full channel.energies files and the unique irrep list 
    #full_energy_list, irrep_list = irrep_list_maker(energy_file)
    Lbyas = Lbyas_val #int(full_energy_list[0][0])
    print("L = ",Lbyas)
    #irrep = irrep_list[5]
    print("irrep = ",irrep)

    #we create the momentum configurations possible for max n^2
    nx, ny, nz = config_maker(max_nsq)

    print("config num = ",len(nx))

    #we take the frame momentum from the irrep string 
    frame_mom_str = irrep.split("_")
    frame_n = []
    for i in frame_mom_str[0]:
        frame_n.append(int(i))
    config_num = len(nx)
    frame_nx = frame_n[0]
    frame_ny = frame_n[1]
    frame_nz = frame_n[2]

    #we calculate the momentum square of the frame 
    onebyxi = 1.0/xi 
    pi = np.pi
    twopibyLbyas = 2.0*pi/Lbyas 

    #print(nx,ny,nz)
    print("frame mom = ",frame_nx,frame_ny,frame_nz)
    #frame_nsquare = frame_nx*frame_nx + frame_ny*frame_ny + frame_nz*frame_nz 
    #frame_momsq = onebyxi*onebyxi*twopibyLbyas*twopibyLbyas*frame_nsquare

    #This dataframe is for L = 20 to check with redstar generated non-int spectrum 
    Ecm_list = []
    Elat_list = []
    particle1_config_list = []
    particle2_config_list = []
    particle3_config_list = []

    for i in range(config_num):
        for j in range(config_num):
            onebyxi = 1.0/xi 
            pi = np.pi
            twopibyLbyas = 2.0*pi/Lbyas 
            frame_nsquare = frame_nx*frame_nx + frame_ny*frame_ny + frame_nz*frame_nz 
            frame_momsq = onebyxi*onebyxi*twopibyLbyas*twopibyLbyas*frame_nsquare

            nx1 = nx[i]
            ny1 = ny[i]
            nz1 = nz[i]

            nx2 = nx[j]
            ny2 = ny[j]
            nz2 = nz[j]

            nx3 = frame_nx - (nx[i] + nx[j])    
            ny3 = frame_ny - (ny[i] + ny[j])
            nz3 = frame_nz - (nz[i] + nz[j])

            particle1_energy = energy(atmk,xi,Lbyas,nx[i],ny[i],nz[i])
            particle2_energy = energy(atmk,xi,Lbyas,nx[j],ny[j],nz[j])
            particle3_energy = energy(atmpi,xi,Lbyas,nx3,ny3,nz3)

            E_lat = particle1_energy + particle2_energy + particle3_energy
            Ecm = np.sqrt(E_lat*E_lat - frame_momsq)
            Ecm_list.append(Ecm)
            Elat_list.append(E_lat)
            particle1_config_list.append([nx1,ny1,nz1])
            particle2_config_list.append([nx2,ny2,nz2])
            particle3_config_list.append([nx3,ny3,nz3])


    Energy_df = pd.DataFrame(list(zip(Elat_list, Ecm_list, particle1_config_list, particle2_config_list, particle3_config_list)))

    #display(Energy_df[4])        

    sorted_energy_df = Energy_df.sort_values(1)
    sorted_energy_df = sorted_energy_df.reset_index(drop=True)
    #print(sorted_energy_df)

    
    unique_energy_df = sorted_energy_df[1].unique()

    non_int_points_file = "non_int_L" + str(int(Lbyas)) + "_points_" + str(irrep) + ".dat"
    
    f = open(non_int_points_file,"w")

    for i in range(len(unique_energy_df)):
        f.write(    str(Lbyas) + '\t' 
                +   str(unique_energy_df[i]) + '\n')
        
    f.close()
        
    selected_non_int_Ecm = []
    selected_non_int_Elat = []
    selected_particle1_config = []
    selected_particle2_config = []
    selected_particle3_config = []
    completed_ecm_list = []

    selection_flag = 0
    selection_counter = 0

    for i in range(len(sorted_energy_df[0])):
        val1 = sorted_energy_df[0][i]
        val2 = sorted_energy_df[1][i]
        val3 = sorted_energy_df[2][i]
        val4 = sorted_energy_df[3][i]
        val5 = sorted_energy_df[4][i]
        
        if(selection_counter>=len(unique_energy_df)):
            break 
        
        present_Ecm = unique_energy_df[selection_counter]
        selected = 1
        if(val2==present_Ecm):
            selection_counter = selection_counter + 1
            selected_non_int_Elat.append(val1) 
            selected_non_int_Ecm.append(val2) 
            selected_particle1_config.append(val3)
            selected_particle2_config.append(val4)
            selected_particle3_config.append(val5)
            selected = 0
        if(selected == 0):
            print(val1,val2,val3,val4,val5," selected")
        else:
            print(val1,val2,val3,val4,val5," not selected")
    
    print(unique_energy_df)
    
    out_file_list = np.zeros((config_num*config_num+1,Lpoints)) 
    L = np.linspace(Lmin,Lmax,Lpoints)

    count = 0
    for i in range(len(selected_particle1_config)):
        nx1 = selected_particle1_config[i][0]
        ny1 = selected_particle1_config[i][1]
        nz1 = selected_particle1_config[i][2]

        nx2 = selected_particle2_config[i][0]
        ny2 = selected_particle2_config[i][1]
        nz2 = selected_particle2_config[i][2]

        nx3 = selected_particle3_config[i][0]
        ny3 = selected_particle3_config[i][1]
        nz3 = selected_particle3_config[i][2]


        for l_ind in range(Lpoints):
            onebyxi = 1.0/xi 
            pi = np.pi
            twopibyLbyas = 2.0*pi/L[l_ind] 
            frame_nsquare = frame_nx*frame_nx + frame_ny*frame_ny + frame_nz*frame_nz 
            frame_momsq = onebyxi*onebyxi*twopibyLbyas*twopibyLbyas*frame_nsquare

            particle1_energy = energy(atmk,xi,L[l_ind],nx1,ny1,nz1)
            particle2_energy = energy(atmk,xi,L[l_ind],nx2,ny2,nz2)
            particle3_energy = energy(atmpi,xi,L[l_ind],nx3,ny3,nz3)

            E_lat = particle1_energy + particle2_energy + particle3_energy
            Ecm = np.sqrt(E_lat*E_lat - frame_momsq)
            out_file_list[0][l_ind] = L[l_ind] 
            out_file_list[count+1][l_ind] = Ecm 
        count = count + 1
    


    out_file = "non_int." + str(irrep) 
    f = open(out_file,'w')

    for i in range(len(out_file_list[0])):
        for j in range(len(out_file_list)):
            f.write(str(out_file_list[j][i]) + '\t')
            #print(out_file_list[i][j], sep=",")
        f.write('\n')
        
    f.close() 
    print("file generated = ",out_file)




energy_file = "S2I2_energies"

#full_list, irrep_list = irrep_list_maker(energy_file)
#nx,ny,nz = config_maker(4)
#print(irrep_list)
atmk = 0.09698
atmpi = 0.06906
xi = 3.444
max_nsq = 10
Lmin = 19
Lmax = 25
Lpoints = 2000 
#full_energy_list, irrep_list = irrep_list_maker(energy_file)

Lbyas_val = 24 

irrep = "000_A1m"

irrep_list = ["000_A1m","100_A2","110_A2","111_A2","200_A2"]


for i in range(len(irrep_list)):
    irrep = irrep_list[i]
    full_nonint_spectrum_maker_final(Lbyas_val, atmk,atmpi, xi, max_nsq, irrep, Lmin, Lmax, Lpoints)
    #nonint_spectrum_maker(atmk,atmpi, xi, max_nsq, irrep, Lmin, Lmax, Lpoints, energy_file)
