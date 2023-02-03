import glob
import os
import numpy as np
import matplotlib.pyplot as plt

import pickle as pickle

from trajectories.trajectory_compute import *
from trajectories.trajectory_family import *
from trajectories.trajectory_plot import *
from trajectories.cloud_properties import *
from trajectories.cloud_selection import *
# Needs
# pip install  git+https://github.com/dbstein/fast_interp
#

dn = 11
#runtest=True
runtest=False

# root_dir = 'E:/'
# root_dir = '/storage/silver/wxproc/xm904103/traj/BOMEX/'
root_dir = 'C:/Users/paclk/OneDrive - University of Reading/'

indir = root_dir + f'traj_data/r{dn:02d}/'
#   Set to True to calculate trajectory family,False to read pre-calculated from pickle file.
get_traj = False
# get_traj = True

debug_unsplit = False
debug_label = False
debug_mean = False
debug = False

files = glob.glob(indir+"diagnostics_3d_ts_*.nc")
files.sort(key=file_key)

#def main():
'''
Top level code, a bit of a mess.
'''

if runtest :
    dt = 60
#        first_ref_file = 88
    first_ref_time = 89*dt
    last_ref_time =  90*dt
    tr_back_len = 5*dt
    tr_forward_len = 4*dt
    ref = 1
    selind = 29
#        sel_list   = np.array([0, 24, 38, 41, 43, 56, 57, 61])
    sel_list   = np.array([21, 29, 32, 34, 38, 40, 47, 48, 61, 64])


elif dn in (5,) :
    dt = 60
    first_ref_time = 50*dt
    last_ref_time =  90*dt
    tr_back_len = 40*dt
    tr_forward_len = 30*dt
    ref = 40
    selind = 61
#        sel_list   = np.array([21, 29, 32, 34, 38, 40, 47, 48, 61, 64])
    sel_list   = np.array([0, 24, 41, 43, 56, 57])

elif dn in (6,) :
    dt = 60
    first_ref_time = 60*dt
    last_ref_time =  119*dt
    tr_back_len = 40*dt
    tr_forward_len = 30*dt
    ref = 30
    selind = 61
#        sel_list   = np.array([21, 29, 32, 34, 38, 40, 47, 48, 61, 64])
    sel_list   = np.array([0, 24, 41, 43, 56, 57])

elif dn in (11,16) :
    dt = 60
    first_ref_time = 50*dt
    last_ref_time =  90*dt
    tr_back_len = 40*dt
    tr_forward_len = 30*dt
    ref = 40
    selind = 44
#        first_ref_time = 50*dt
#        last_ref_time =  55*dt
#        tr_back_len = 2*dt
#        tr_forward_len = 2*dt
#        ref = 40
    selind = 72
    sel_list_nice   = np.array([14, 44, 72, 74, 79, 85, 92])

else:
    dt = 60
    first_ref_time = 24*dt
    last_ref_time =  44*dt
    tr_back_len = 20*dt
    tr_forward_len = 15*dt
    ref = 20

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
fn = indir + fn

ref_prof_file = glob.glob(indir+'diagnostics_ts_*.nc')[0]
test_pickle = 'traj_family_{:03d}_{:03d}_{:03d}_{:03d}_v3'.\
    format(first_ref_time//dt-1 ,last_ref_time//dt-1, \
           tr_back_len//dt, tr_forward_len//dt)
print(test_pickle)
var_list = {
  "u":r"$u$ m s$^{-1}$",
  "v":r"$v$ m s$^{-1}$",
  "w":r"$w$ m s$^{-1}$",
  "th":r"$\theta$ K",
  "p":r"Pa",
  "dp_dx":r"Pa m^{-1}",
  "dp_dy":r"Pa m^{-1}",
  "dp_dz":r"Pa m^{-1}",
  "q_vapour":r"$q_{v}$ kg/kg",
  "q_cloud_liquid_mass":r"$q_{cl}$ kg/kg",
  # "tracer_rad1":r"Tracer 1 kg/kg",
  # "tracer_rad2":r"Tracer 2 kg/kg",
  }
kwa={'thresh':1.0E-5}
if get_traj :

    tfm = Trajectory_Family(files, ref_prof_file, \
                 first_ref_time, last_ref_time, \
                 tr_back_len, tr_forward_len, \
                 100.0, 100.0, 40.0, trajectory_cloud_ref, in_cloud, \
                 kwargs=kwa, variable_list=var_list)
    outfile = open(indir+test_pickle,'wb')
    print('Pickling ',indir+test_pickle)
    pickle.dump(tfm, outfile)
    outfile.close()
else :
    infile = open(indir+test_pickle,'rb')
    print('Un-pickling ',indir+test_pickle)
    tfm = pickle.load(infile)
#        print(tfm)
    infile.close()

traj_list = tfm.family

sel = np.array([selind])
tfm.print_matching_object_list(select = sel)

tfm.print_matching_object_list_summary(select = sel, overlap_thresh=0.1)

tfm.print_linked_objects(master_ref=ref, select = sel, overlap_thresh=0.1)

if False :
    plot_traj_family_members(tfm, mem_list, galilean = np.array([-8.5,0]), \
                         with_boxes=True, asint = True, )

traj_m = traj_list[-1]
traj_r = traj_list[0]

if True :
    traj_m_class = set_cloud_class(traj_m, version = 1)
    print_cloud_class(traj_m, traj_m_class, 61)

if True :
    mean_prop = cloud_properties(traj_m, traj_m_class)

if True :
    th = 0.5
    sup, len_sup = tfm.find_super_objects(overlap_thresh = th)
    print("Super objects",sup)

# Plot all clouds
if True :
    
    anim_all_m = plot_traj_animation(traj_m, save_anim=False, anim_name='traj_all_clouds', \
                        with_boxes=False, \
                        title = 'Ref Time {}'.format(last_ref_time))
# Plot all clouds with galilean transform
if True :
    anim_all_gal_m = plot_traj_animation(traj_m, save_anim=False, anim_name='traj_all_clouds_gal', \
        title = 'Ref Time {} Galilean Transformed'.format(last_ref_time), \
        galilean = np.array([-8.5,0]))

if False :
    anim_all_gal_r = plot_traj_animation(traj_r, save_anim=False, \
        title = 'Ref Time {} Galilean Transformed'.format(first_ref_time), \
        galilean = np.array([-8.5,0]))

# Plot max_list clouds with galilean transform
if True :
    max_list = traj_m.max_at_ref
    print(max_list)
    sel_list = max_list
    anim_max_gal_m = plot_traj_animation(traj_m, save_anim=False,  anim_name='traj_sel_clouds_gal', \
        select = sel_list, \
        no_cloud_size = 0.2, cloud_size = 2.0, legend = True, \
        title = 'Ref Time {} Galilean Transformed Max Clouds'.format(last_ref_time), \
        with_boxes = False, galilean = np.array([-8.5,0]) )

sel_list=np.array([selind])
if False :
    for iobj in sel_list:
        plot_trajectory_history(traj_m, iobj, fn)

    plt.show()

if False :
    cloud_lifetime = traj_m_class['cloud_dissipate_time'] - \
                     traj_m_class['cloud_trigger_time']

    plt.hist(cloud_lifetime, bins = np.arange(0,75,10, dtype=int), density=True)
    plt.xlabel('Lagrangian lifetime (min)')
    plt.ylabel('Fraction of clouds')
    plt.savefig(indir+'Cloud_lifetime.png')
    plt.show()

if True :
    for cloud in sel_list :
        anim_sel = plot_traj_animation(traj_m, save_anim=False, \
                    anim_name='traj_cloud_{:03d}_class'.format(cloud), \
                    select = np.array([cloud]), fps = 10,  \
                    no_cloud_size = 0.2, cloud_size = 2.0, legend = True, \
                    title = 'Reference Time {0} Cloud {1} Galilean Trans'.\
                    format(last_ref_time, cloud), with_boxes = False,
                    galilean = np.array([-8.5,0]), \
                    plot_class = traj_m_class,\
                    )

#input("Press Enter then continue Powerpoint...")
# Plot max_list clouds mean history
if True :
    plot_trajectory_mean_history(traj_m, traj_m_class, mean_prop, fn, select = sel_list)

# Plot subset max_list clouds with galilean transform
if False:
    anim = plot_traj_animation(traj_r, save_anim=False, select = sel_list, \
                    no_cloud_size = 0.2, cloud_size = 2.0, legend = True, \
                    with_boxes = False, galilean = np.array([-8.5,0]))

if True :
    anim_sel_field = plot_traj_animation(traj_m, save_anim=False, \
                    anim_name='traj_cloud_sel_field', \
                    select = sel_list, \
                    fps=10, legend = True, plot_field = True, \
                    dir_override=indir, \
                    no_cloud_size = 0.2, cloud_size = 2.0, field_size = 0.5, \
                    title = 'Reference Time {0} Galilean Trans with clouds'.\
                    format(last_ref_time), with_boxes = False, \
                    galilean = np.array([-8.5,0]))


if False :
    plt.hist(len_sup, bins = np.arange(0.5,16.5), density=True)
    plt.title('Threshold = {:2.0f}%'.format(th*100))
    plt.xlabel('Super-object length (min)')
    plt.ylabel('Fraction of super objects')
    plt.savefig(indir+'Super_object_length.png')
    plt.show()

sel_list   = np.array([selind])
#sel_list = max_list
th=0.1
if True :
    for cloud in sel_list :
        anim_cloud_field = plot_traj_animation(traj_m, 
                    save_anim=False,
                    anim_name=f'traj_cloud_{cloud:03d}_field',
                    select = np.array([cloud]),
                    fps = 10, legend = True, plot_field = True,
                    dir_override=indir,
                    no_cloud_size = 0.2, cloud_size = 2.0, field_size = 0.5,
                    title = f'Reference Time {last_ref_time} Cloud {cloud} Eulerian Gal. Trans',
                    galilean = np.array([-8.5,0]))

        for tback in [10] :
            ttle = f'Cloud {cloud} at ref_time {last_ref_time}, linked cloud at ref_time {last_ref_time-tback} Gal. Trans'
            anim_cloud_linked = plot_traj_family_animation(tfm, tback, 
                    overlap_thresh = th,
                    save_anim=False,
                    anim_name=f'traj_cloud_{cloud:03d}_linked_{tback:02d}',
                    select = np.array([cloud]),
                    fps = 10, legend = True, plot_field = False,
                    dir_override=indir,
                    title = ttle,
                    no_cloud_size = 0.2, cloud_size = 2.0, field_size = 0.5,
                    with_boxes = False, galilean = np.array([-8.5,0]))

        for tback in [-1] :
            anim_cloud_super = plot_traj_family_animation(tfm, tback, 
                    overlap_thresh = th,
                    save_anim=False,
                    anim_name=f'traj_cloud_{cloud:03d}_super',
                    select = np.array([cloud]), super_obj = sup,
                    fps = 10, legend = True, plot_field = False,
                    dir_override=indir,
                    title = f'Reference Time {last_ref_time} Cloud {cloud} Super Object Gal. Trans ',
                    no_cloud_size = 0.2, cloud_size = 2.0, field_size = 0.5,
                    with_boxes=False, galilean = np.array([-8.5,0]))

        for tback in [-1] :
            anim_cloud_super_linked = plot_traj_family_animation(tfm, tback, 
                    overlap_thresh = th,
                    save_anim=False, \
                    anim_name=f'traj_cloud_{cloud:03d}_super_linked'.format(cloud),
                    select = np.array([cloud]),
                    fps = 10, legend = True, plot_field = False,
                    dir_override=indir,
                    title = f'Reference Time {last_ref_time} Cloud {cloud} Linked Objects Gal. Trans',
                    no_cloud_size = 0.2, cloud_size = 2.0, field_size = 0.5,
                    with_boxes=False, galilean = np.array([-8.5,0]))

#if __name__ == "__main__":
#    main()
