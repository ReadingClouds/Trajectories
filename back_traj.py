import glob
import os
import numpy as np
import matplotlib.pyplot as plt

import pickle as pickle
#from datetime import datetime, timedelta
#from netCDF4 import num2date, date2num

from trajectory_compute import *
from trajectory_plot import *

def file_key(file):
    f1 = file.split('_')[-1]
    f2 = f1.split('.')[0]
    return float(f2)

#dir = '/projects/paracon/toweb/r4677_Circle-A/diagnostic_files/'
#dir = '/projects/paracon/toweb/r4677_Circle-A/diagnostic_files/S_ReI_1200_A/'
#dir = '/projects/paracon/toweb/r4677_Circle-A/diagnostic_files/traj_Pete/'

#dir = '/storage/shared/metcloud/wxproc/xm904103/traj/'
#dir = '/projects/paracon/appc/MONC/r4664_CA_traj/diagnostic_files/r5/'
#dir = '/projects/paracon/appc/MONC/prev/r4664_CA_traj/diagnostic_files/'
#dir = '/projects/paracon/appc/MONC/r4664_CA_traj/diagnostic_files/r6/'
dir = 'C:/Users/paclk/OneDrive - University of Reading/traj_data/r6/'
#dir = 'C:/Users/xm904103/OneDrive - University of Reading/traj_data/r6/'
#dir = '/storage/silver/wxproc/xm904103/traj/BOMEX/r6/'
dn = 11
#dir = 'C:/Users/paclk/OneDrive - University of Reading/traj_data/r11/'
dir = 'C:/Users/paclk/OneDrive - University of Reading/traj_data/r{:02d}/'.format(dn)
files = glob.glob(dir+"diagnostics_3d_ts_*.nc")
files.sort(key=file_key)

#print(files)
print(dir)

if dn == 11 :
    first_ref_file = 49
#   first_ref_file = 88
    last_ref_file =  89
    tr_back_len = 40
    tr_forward_len = 30
    ref = 40
else:
    first_ref_file = 24
    last_ref_file =  44
    tr_back_len = 20
    tr_forward_len = 15
    ref = 20


#debug_unsplit = True
debug_unsplit = False
debug_label = False  
#debug_label = True  
#debug_mean = True 
debug_mean = False 
#debug = True
debug = False

order_labs =[ \
             'linear', \
             'quadratic', \
             'cubic', \
             'quartic', \
             'quintic', \
             ] 
#interp_order = 5
#interp_order = 3
interp_order = 1
iorder = '_'+order_labs[interp_order-1]
fn = os.path.basename(files[last_ref_file])[:-3]
fn = ''.join(fn)+iorder
fn = dir + fn

ref_prof_file = glob.glob(dir+'diagnostics_ts_*.nc')[0]

get_traj = False
#get_traj = True

test_pickle = 'traj_family_{:03d}_{:03d}_{:03d}_{:03d}'.format(first_ref_file ,\
                           last_ref_file, tr_back_len, tr_forward_len)
if get_traj :
    tfm = trajectory_family(files, ref_prof_file, \
                 first_ref_file, last_ref_file, \
                 tr_back_len, tr_forward_len, \
                 100.0, 100.0, 40.0)
    outfile = open(dir+test_pickle,'wb')
    print('Pickling ',dir+test_pickle)
    pickle.dump(tfm, outfile)
    outfile.close()
else :
    infile = open(dir+test_pickle,'rb')
    print('Un-pickling ',dir+test_pickle)
    tfm = pickle.load(infile)
    infile.close()
    
    
traj_list = tfm.family

#tfm.print_matching_object_list()
#tfm.print_matching_object_list_summary(overlap_thresh=0.1)

#tfm.print_linked_objects(overlap_thresh=0.1)
#sel = np.array([72])
#tfm.print_matching_object_list(ref=ref,select = sel)
#tfm.print_matching_object_list_summary(ref=ref, select = sel, overlap_thresh=0.1)
#tfm.print_linked_objects(ref=ref, select = sel, overlap_thresh=0.1)

#input("Press Enter to continue...")

#matching_object_list = tfm.matching_object_list()
#iobj = 85
#t_off = 0
#time = 1

#for match_obj in matching_object_list[t_off][time][iobj]  : 
#    inter = tfm.refine_object_overlap(t_off, time, iobj, match_obj)
#    print("Object refinement object {} overlap: {}".format(match_obj,inter))
#
#for th in np.arange(0.3,1.0,0.1) :
#    sup, len_sup = tfm.find_super_objects(overlap_thresh = th) 
#    #print(sup)
#    plt.hist(len_sup)
#    plt.title('Threshold = {:2.0f}'.format(th*100))
#    plt.show()
  

mem_list = [(85,40,40),(0,39,41),(92,39,41),(0,38,42),(1,38,42)]
#mem_list = [(85,40,40),(92,39,41)]
if False :
    plot_traj_family_members(tfm, mem_list, galilean = np.array([-8.5,0]), \
                         with_boxes=True, asint = True, )
#plot_traj_family_members(tfm, mem_list, with_boxes=True )
#plot_traj_family_members(tfm,[(85,40,40),(92,39,41)]) 

#input("Press Enter to continue...")

traj_m = traj_list[-1]
traj_r = traj_list[0]
# Appropriate for r6 test data
#sel_list   = np.array([0, 62, 85])
#sel_list   = np.array([0])
#sel_list   = np.array([0, 62, 70, 85])
sel_list_r = np.array([0, 5, 72, 78, 92])


# Appropriate for r11 test data
sel_list   = np.array([14, 44, 72, 74, 79, 85, 92])
#sel_list   = np.array([72])


if True :
    th = 0.5
    sup, len_sup = tfm.find_super_objects(overlap_thresh = th)

#plot_traj_pos(traj_m, ref_file-start_file, save=False) 
#plot_traj_pos(traj_m, ref_file-start_file+1, save=False) 

#data_mean, traj_m_centroid = compute_traj_centroids(traj_m)
input("Press Enter to continue...")
# Plot all clouds
if True :
    plot_traj_animation(traj_m, save_anim=False, with_boxes=False, \
                        title = 'Reference Time {}'.format(last_ref_file))

#input("Press Enter to continue...")
# Plot all clouds with galilean transform
if True :
    plot_traj_animation(traj_m, save_anim=False, \
        title = 'Reference Time {} Galilean Tranformed'.format(last_ref_file), \
        galilean = np.array([-8.5,0]))

if False :
    plot_traj_animation(traj_r, save_anim=False, \
        title = 'Reference Time {} Galilean Tranformed'.format(first_ref_file), \
        galilean = np.array([-8.5,0]))

# Plot max_list clouds with galilean transform
if True :
    plot_traj_animation(traj_m, save_anim=False, select = sel_list, \
        no_cloud_size = 0.2, cloud_size = 2.0, legend = True, \
        title = 'Reference Time {} Galilean Tranformed'.format(last_ref_file), \
        with_boxes = False, galilean = np.array([-8.5,0]) )

max_list = traj_m.max_at_ref
print(max_list)
#sel_list = max_list
if False :
#    for iobj in range(0,traj_m.nobjects):
#    for iobj in range(1,7):    
    for iobj in sel_list:
        plot_trajectory_history(traj_m, iobj, fn) 

    plt.show()    

# print_boxes(traj_m)


#print_centroids(traj_centroid, data_mean, traj_m.nobjects, traj_m.times, \
#                list(traj_m.variable_list))

input("Press Enter then continue Powerpoint...")


if False :
    plot_traj_animation(traj_r, save_anim=False, select = sel_list_r, \
        no_cloud_size = 0.2, cloud_size = 2.0, legend = True, \
        title = 'Reference Time {} Galilean Tranformed'.format(last_ref_file-1), \
        with_boxes = False, galilean = np.array([-8.5,0]) )
    
if False :
    mean_prop = traj_m.cloud_properties(version = 1)
      
if False :
    for cloud in sel_list :
        plot_traj_animation(traj_m, save_anim=False, \
                    select = np.array([cloud]), fps = 2,  \
                    no_cloud_size = 0.2, cloud_size = 2.0, legend = True, \
                    title = 'Reference Time {0} Cloud {1} Galilean Trans'.\
                    format(last_ref_file, cloud), with_boxes = False, 
                    galilean = np.array([-8.5,0]), plot_class = mean_prop['class'],\
                    version = mean_prop['version'])
if False :
    plot_trajectory_mean_history(traj_m, mean_prop, fn, select = sel_list) 

if True :
    mean_prop2 = traj_m.cloud_properties(version = 3)
#    print(mean_prop2["cloud_trigger_time"])
#    print(mean_prop2["cloud_dissipate_time"])
    
#    print(mean_prop2['cloud_trigger_time'])
#    print(mean_prop2['cloud_dissipate_time'])
if False :
    cloud_lifetime = mean_prop2['cloud_dissipate_time'] - \
                     mean_prop2['cloud_trigger_time']
    
    plt.hist(cloud_lifetime, bins = np.arange(0,75,10, dtype=int), density=True)
    plt.xlabel('Lagrangian lifetime (min)')
    plt.ylabel('Fraction of clouds')
    plt.savefig(dir+'Cloud_lifetime.png')
    plt.show()
    
input("Press Enter to continue...")
      
if True :
    for cloud in sel_list :
        plot_traj_animation(traj_m, save_anim=False, \
                    select = np.array([cloud]), fps = 10,  \
                    no_cloud_size = 0.2, cloud_size = 2.0, legend = True, \
                    title = 'Reference Time {0} Cloud {1} Galilean Trans'.\
                    format(last_ref_file, cloud), with_boxes = False, 
                    galilean = np.array([-8.5,0]), \
                    plot_class = mean_prop2['class'],\
                    version = mean_prop2['version'])
        
input("Press Enter then continue Powerpoint...")        
# Plot max_list clouds mean history
     
      
if False :
    plot_trajectory_mean_history(traj_m, mean_prop2, fn, select = sel_list) 


#max_list=np.array([9,18,21,22,24,36,38,43,49,52,63,69,70,77,83,87,88,96])
#max_list=np.array([20,23,34,35,37,42,48,51,69,76,82,83,86])


#max_list=np.array([2,13,26,29,37,42])
#max_list=np.array([0,62,70,85,95])
#max_list_r=np.array([0,3,7,67,71,86])


# Plot subset max_list clouds with galilean transform
# Not needed   
if False :
    plot_traj_animation(traj_m, save_anim=False, select = max_list, \
                    no_cloud_size = 0.2, cloud_size = 2.0, legend = True, \
                    with_boxes = False, galilean = np.array([-8.5,0]))

# Plot subset max_list clouds with galilean transform
if False :
    plot_traj_animation(traj_r, save_anim=False, select = sel_list_r, \
                    no_cloud_size = 0.2, cloud_size = 2.0, legend = True, \
                    with_boxes = False, galilean = np.array([-8.5,0]))

input("Press Enter to continue...")
 
if True :
    plot_traj_animation(traj_m, save_anim=False, select = sel_list, \
                    legend = True, plot_field = True, \
                    dir_override=dir, \
                    no_cloud_size = 0.2, cloud_size = 2.0, field_size = 0.5, \
                    title = 'Reference Time {0} Galilean Trans with clouds'.\
                    format(last_ref_file), with_boxes = False, \
                    galilean = np.array([-8.5,0]))
    
    
if False :
    plt.hist(len_sup, bins = np.arange(0.5,16.5), density=True)
    plt.title('Threshold = {:2.0f}%'.format(th*100))
    plt.xlabel('Super-object length (min)')
    plt.ylabel('Fraction of super objects')
    plt.savefig(dir+'Super_object_length.png')
    plt.show()  
    
    

sel_list   = np.array([72])
th=0.1
if True :
    for cloud in sel_list :
        plot_traj_animation(traj_m, save_anim=False, select = np.array([cloud]), \
                    fps = 10, legend = True, plot_field = True, \
                    dir_override=dir, \
                    no_cloud_size = 0.2, cloud_size = 2.0, field_size = 0.5, \
                    title = 'Reference Time {0} Cloud {1} Eulerian Gal. Trans'.\
                    format(last_ref_file, cloud), \
                    galilean = np.array([-8.5,0]))
        input("Press Enter then continue Powerpoint...")
        input("Press Enter to continue...")
        for tback in [10, 20] :
            plot_traj_family_animation(tfm, tback, overlap_thresh = th, \
                    save_anim=False, \
                    select = np.array([cloud]), \
                    fps = 10, legend = True, plot_field = False, \
                    dir_override=dir, \
                    title = 'Reference Times {0},{1} Cloud {2} Gal. Trans'.\
                    format(last_ref_file, last_ref_file-tback,  cloud), \
                    no_cloud_size = 0.2, cloud_size = 2.0, field_size = 0.5, \
                    with_boxes = False, galilean = np.array([-8.5,0]))

        for tback in [-1] :
            plot_traj_family_animation(tfm, tback, overlap_thresh = th, \
                    save_anim=False, \
                    select = np.array([cloud]), super_obj = sup, \
                    fps = 10, legend = True, plot_field = False, \
                    dir_override=dir, \
                    title = 'Reference Time {0} Cloud {1} Super Object Gal. Trans '.\
                    format(last_ref_file,  cloud), \
                    no_cloud_size = 0.2, cloud_size = 2.0, field_size = 0.5, \
                    with_boxes=False, galilean = np.array([-8.5,0]))

        for tback in [-1] :
            plot_traj_family_animation(tfm, tback, overlap_thresh = th, \
                    save_anim=False, \
                    select = np.array([cloud]), \
                    fps = 10, legend = True, plot_field = False, \
                    dir_override=dir, \
                    title = 'Reference Time {0} Cloud {1} Linked Objects Gal. Trans'.\
                    format(last_ref_file,  cloud), \
                    no_cloud_size = 0.2, cloud_size = 2.0, field_size = 0.5, \
                    with_boxes=False, galilean = np.array([-8.5,0]))


#print(min_cloud_base)
    
#print(np.where(traj_centroid != trs))

#print(trs[traj_centroid != trs])
#print(traj_centroid[traj_centroid != trs])

#########################################################
