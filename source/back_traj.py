import glob
import os
import numpy as np
import matplotlib.pyplot as plt

import pickle as pickle

from trajectory_compute import *
from trajectory_plot import *

dn = 16
dir = 'C:/Users/paclk/OneDrive - University of Reading/traj_data/r{:02d}/'.format(dn)
files = glob.glob(dir+"diagnostics_3d_ts_*.nc")
files.sort(key=file_key)

def main():
    '''
    Top level code, a bit of a mess.
    '''

    if dn in (11,16) :
        dt = 60
        first_ref_time = 50*dt
    #    first_ref_time = 89*dt
    #   f irst_ref_file = 88
        last_ref_time =  90*dt
        tr_back_len = 40*dt
        tr_forward_len = 30*dt
    #    tr_back_len = 4*dt
    #    tr_forward_len = 3*dt
        ref = 40
    else:
        first_ref_time = 24*dt
        last_ref_time =  44*dt
        tr_back_len = 20*dt
        tr_forward_len = 15*dt
        ref = 20
    
    get_traj = False
    #get_traj = True

    debug_unsplit = False
    debug_label = False  
    debug_mean = False 
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
    fn = os.path.basename(files[0])[:-3]
    fn = ''.join(fn)+iorder
    fn = dir + fn
    
    ref_prof_file = glob.glob(dir+'diagnostics_ts_*.nc')[0]
    test_pickle = 'traj_family_{:03d}_{:03d}_{:03d}_{:03d}'.\
        format(first_ref_time//dt-1 ,last_ref_time//dt-1, \
               tr_back_len//dt, tr_forward_len//dt)
    print(test_pickle)
    if get_traj :
        tfm = trajectory_family(files, ref_prof_file, \
                     first_ref_time, last_ref_time, \
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
        print(tfm)
        infile.close()
        
        
    traj_list = tfm.family
    
    tfm.print_matching_object_list()
    tfm.print_matching_object_list_summary(overlap_thresh=0.1)
    
    tfm.print_linked_objects(overlap_thresh=0.1)
    sel = np.array([72])
    tfm.print_matching_object_list(ref=ref,select = sel)
    tfm.print_matching_object_list_summary(ref=ref, select = sel, overlap_thresh=0.1)
    tfm.print_linked_objects(ref=ref, select = sel, overlap_thresh=0.1)
        
    mem_list = [(85,40,40),(0,39,41),(92,39,41),(0,38,42),(1,38,42)]
    #mem_list = [(85,40,40),(92,39,41)]
    if False :
        plot_traj_family_members(tfm, mem_list, galilean = np.array([-8.5,0]), \
                             with_boxes=True, asint = True, )
    
    #input("Press Enter to continue...")
    
    traj_m = traj_list[-1]
    traj_r = traj_list[0]
    
    # Appropriate for r11 and r16 test data
    sel_list   = np.array([14, 44, 72, 74, 79, 85, 92])
    #sel_list   = np.array([72])
    
    
    if True :
        th = 0.5
        sup, len_sup = tfm.find_super_objects(overlap_thresh = th)
    
    # Plot all clouds
    if True :
        plot_traj_animation(traj_m, save_anim=False, anim_name='traj_all_clouds', \
                            with_boxes=False, \
                            title = 'Reference Time {}'.format(last_ref_time))
    
    #input("Press Enter to continue...")
    # Plot all clouds with galilean transform
    if True :
        plot_traj_animation(traj_m, save_anim=False, anim_name='traj_all_clouds_gal', \
            title = 'Reference Time {} Galilean Tranformed'.format(last_ref_time), \
            galilean = np.array([-8.5,0]))
    
    if False :
        plot_traj_animation(traj_r, save_anim=False, \
            title = 'Reference Time {} Galilean Tranformed'.format(first_ref_time), \
            galilean = np.array([-8.5,0]))
    
    # Plot max_list clouds with galilean transform
    if True :
        plot_traj_animation(traj_m, save_anim=False,  anim_name='traj_sel_clouds_gal', \
            select = sel_list, \
            no_cloud_size = 0.2, cloud_size = 2.0, legend = True, \
            title = 'Reference Time {} Galilean Tranformed'.format(last_ref_time), \
            with_boxes = False, galilean = np.array([-8.5,0]) )
    
    max_list = traj_m.max_at_ref
    print(max_list)
    #sel_list = max_list
    if False :
        for iobj in sel_list:
            plot_trajectory_history(traj_m, iobj, fn) 
    
        plt.show()    

    #input("Press Enter then continue Powerpoint...")
    
    
    if True :
        mean_prop = traj_m.cloud_properties(version = 1)
    
        print(mean_prop2['cloud_trigger_time'])
        print(mean_prop2['cloud_dissipate_time'])

    if True :
        cloud_lifetime = mean_prop['cloud_dissipate_time'] - \
                         mean_prop['cloud_trigger_time']
        
        plt.hist(cloud_lifetime, bins = np.arange(0,75,10, dtype=int), density=True)
        plt.xlabel('Lagrangian lifetime (min)')
        plt.ylabel('Fraction of clouds')
        plt.savefig(dir+'Cloud_lifetime.png')
        plt.show()
        
    #input("Press Enter to continue...")
          
    if True :
        for cloud in sel_list :
            plot_traj_animation(traj_m, save_anim=False, \
                        anim_name='traj_cloud_{:03d}_class'.format(cloud), \
                        select = np.array([cloud]), fps = 10,  \
                        no_cloud_size = 0.2, cloud_size = 2.0, legend = True, \
                        title = 'Reference Time {0} Cloud {1} Galilean Trans'.\
                        format(last_ref_time, cloud), with_boxes = False, 
                        galilean = np.array([-8.5,0]), \
                        plot_class = mean_prop['class'],\
                        version = mean_prop['version'])
            
    #input("Press Enter then continue Powerpoint...")        
    # Plot max_list clouds mean history         
    if True :
        plot_trajectory_mean_history(traj_m, mean_prop, fn, select = sel_list) 
    
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
    
    #input("Press Enter to continue...")
     
    if True :
        plot_traj_animation(traj_m, save_anim=False, \
                        anim_name='traj_cloud_sel_field', \
                        select = sel_list, \
                        fps=10, legend = True, plot_field = True, \
                        dir_override=dir, \
                        no_cloud_size = 0.2, cloud_size = 2.0, field_size = 0.5, \
                        title = 'Reference Time {0} Galilean Trans with clouds'.\
                        format(last_ref_time), with_boxes = False, \
                        galilean = np.array([-8.5,0]))
        
        
    if True :
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
            plot_traj_animation(traj_m, save_anim=False, \
                        anim_name='traj_cloud_{:03d}_field'.format(cloud), \
                        select = np.array([cloud]), \
                        fps = 10, legend = True, plot_field = True, \
                        dir_override=dir, \
                        no_cloud_size = 0.2, cloud_size = 2.0, field_size = 0.5, \
                        title = 'Reference Time {0} Cloud {1} Eulerian Gal. Trans'.\
                        format(last_ref_time, cloud), \
                        galilean = np.array([-8.5,0]))
            #input("Press Enter then continue Powerpoint...")
            #input("Press Enter to continue...")
            for tback in [10, 20] :
                plot_traj_family_animation(tfm, tback, overlap_thresh = th, \
                        save_anim=False, \
                        anim_name='traj_cloud_{:03d}_linked_{:02d}'.format(cloud,tback), \
                        select = np.array([cloud]), \
                        fps = 10, legend = True, plot_field = False, \
                        dir_override=dir, \
                        title = 'Reference Times {0},{1} Cloud {2} Gal. Trans'.\
                        format(last_ref_time, last_ref_time-tback,  cloud), \
                        no_cloud_size = 0.2, cloud_size = 2.0, field_size = 0.5, \
                        with_boxes = False, galilean = np.array([-8.5,0]))
    
            for tback in [-1] :
                plot_traj_family_animation(tfm, tback, overlap_thresh = th, \
                        save_anim=False, \
                        anim_name='traj_cloud_{:03d}_super'.format(cloud), \
                        select = np.array([cloud]), super_obj = sup, \
                        fps = 10, legend = True, plot_field = False, \
                        dir_override=dir, \
                        title = 'Reference Time {0} Cloud {1} Super Object Gal. Trans '.\
                        format(last_ref_time,  cloud), \
                        no_cloud_size = 0.2, cloud_size = 2.0, field_size = 0.5, \
                        with_boxes=False, galilean = np.array([-8.5,0]))
    
            for tback in [-1] :
                plot_traj_family_animation(tfm, tback, overlap_thresh = th, \
                        save_anim=False, \
                        anim_name='traj_cloud_{:03d}_super_linked'.format(cloud), \
                        select = np.array([cloud]), \
                        fps = 10, legend = True, plot_field = False, \
                        dir_override=dir, \
                        title = 'Reference Time {0} Cloud {1} Linked Objects Gal. Trans'.\
                        format(last_ref_time,  cloud), \
                        no_cloud_size = 0.2, cloud_size = 2.0, field_size = 0.5, \
                        with_boxes=False, galilean = np.array([-8.5,0]))
    
if __name__ == "__main__":
    main() 
