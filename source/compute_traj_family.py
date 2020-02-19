import glob
import os
import os.path
import numpy as np
import pickle as pickle

from trajectory_compute import *

dn = 5
#dir = 'C:/Users/paclk/OneDrive - University of Reading/traj_data/r{:02d}/'.format(dn)
dir = '/storage/silver/wxproc/xm904103/traj/BOMEX/r6n/'
#   Set to True to calculate trajectory family,False to read pre-calculated from pickle file.
#get_traj = False
get_traj = True
    
files = glob.glob(dir+"diagnostics_3d_ts_*.nc")
files.sort(key=file_key)

def main():
    '''
    Top level code, a bit of a mess.
    '''

    ref_prof_file = glob.glob(dir+'diagnostics_ts_*.nc')[0]
    tr_back_len_min = 40
    tr_forward_len_min = 30

    order_labs =[ \
                'linear', \
                'quadratic', \
                'cubic', \
                'quartic', \
                'quintic', \
                ] 

    interp_order = 1
    iorder = '_'+order_labs[interp_order-1]
    fn = os.path.basename(files[0])[:-3]
    fn = ''.join(fn)+iorder
    fn = dir + fn
    var_list = { \
      "u":r"$u$ m s$^{-1}$", \
      "v":r"$v$ m s$^{-1}$", \
      "w":r"$w$ m s$^{-1}$", \
      "th":r"$\theta$ K", \
      "p":r"Pa", \
      "q_vapour":r"$q_{v}$ kg/kg", \
      "q_cloud_liquid_mass":r"$q_{cl}$ kg/kg", \
      "tracer_rad1":r"Tracer 1 kg/kg", \
      "tracer_rad2":r"Tracer 2 kg/kg", \
      }
    kwa={'thresh':1.0E-5}  

    for hh in range(3,24) :

        first_ref_min = hh*60
        last_ref_min = first_ref_min + 59
        pickle_file = 'traj_family_{:03d}_{:03d}_{:03d}_{:03d}_v2'.\
            format(first_ref_min ,last_ref_min, tr_back_len_min, tr_forward_len_min)
        dt = 60
        
        first_ref_time = first_ref_min * dt
        last_ref_time =  last_ref_min * dt
        tr_back_len = tr_back_len_min * dt
        tr_forward_len = tr_forward_len_min * dt
        ref = 30
            
        if get_traj :
            tfm = Trajectory_Family(files, ref_prof_file, \
                     first_ref_time, last_ref_time, \
                     tr_back_len, tr_forward_len, \
                     100.0, 100.0, 40.0, trajectory_cloud_ref, in_cloud, \
                     kwargs=kwa, variable_list=var_list)
            outfile = open(dir+pickle_file,'wb')
            print('Pickling ',dir+pickle_file)
            pickle.dump(tfm, outfile)
            outfile.close()
        else :
            if os.path.isfile(dir+pickle_file) : 
                infile = open(dir+pickle_file,'rb')
                print('Un-pickling ',dir+pickle_file)
                tfm = pickle.load(infile)
    #                print(tfm)
                infile.close()
            else :
                print("File not found: ",dir+pickle_file)
                

    
if __name__ == "__main__":
    main() 
