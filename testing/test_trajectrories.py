"""
Created on Wed Feb  9 14:58:36 2022

@author: Peter Clark
"""

import glob

import numpy as np
import matplotlib.pyplot as plt

import xarray as xr

import trajectories.trajectory_compute as tc
import trajectories.trajectory_plot as tp
import trajectories.cloud_selection as cl
import trajectories.cloud_properties as cp

import time

from subfilter.utils.string_utils import get_string_index
from subfilter.io.MONC_utils import options_database

#from trajectory_plot import *

#%%
test_case = 0

if test_case == 0:
#    config_file = 'config_test_case_0.yaml'
    indir = 'C:/Users/paclk/OneDrive - University of Reading/ug_project_data/Data/'
    odir = 'C:/Users/paclk/OneDrive - University of Reading/ug_project_data/Data/'
    filespec = 'diagnostics_3d_ts_'
    ref_filespec = 'diagnostics_ts_'
    dx = dy = 50.0
    dz = 25.0
    variable_list = {
                      "u":r"$u$ m s$^{-1}$",
                      "v":r"$v$ m s$^{-1}$",
                      "w":r"$w$ m s$^{-1}$",
                      "th":r"$\theta$ K",
                      "p":r"Pa",
                      "q_vapour":r"$q_{v}$ kg/kg",
                      "q_cloud_liquid_mass":r"$q_{cl}$ kg/kg",
                      "tracer_rad1":r"tracer 1 kg/kg",
                      "tracer_rad2":r"tracer 2 kg/kg",
                      "tracer_rad3":r"tracer 3 kg/kg",
                    }

elif test_case == 1:
#    config_file = 'config_test_case_1.yaml'
    indir = 'C:/Users/paclk/OneDrive - University of Reading/traj_data/CBL/'
    odir = 'C:/Users/paclk/OneDrive - University of Reading/traj_data/CBL/'
    file = 'diagnostics_3d_ts_13200.nc'
    ref_file = None

elif test_case == 2:
#    config_file = 'config_test_case_2.yaml'
    indir = 'C:/Users/paclk/OneDrive - University of Reading/traj_data/r11/'
    filespec = 'diagnostics_3d_ts_'
    ref_filespec = 'diagnostics_ts_'
    dx = dy = 100.0
    dz = 40.0
    variable_list = {
                      "u":r"$u$ m s$^{-1}$",
                      "v":r"$v$ m s$^{-1}$",
                      "w":r"$w$ m s$^{-1}$",
                      "th":r"$\theta$ K",
                      "p":r"Pa",
                      "q_vapour":r"$q_{v}$ kg/kg",
                      "q_cloud_liquid_mass":r"$q_{cl}$ kg/kg",
#                      "tracer_rad1":r"tracer 1 kg/kg",
#                      "tracer_rad2":r"tracer 2 kg/kg",
                    }



ref_prof_file = glob.glob(indir + ref_filespec + '*.nc')[0]
files = glob.glob(indir + filespec + '*.nc')
files.sort(key=tc.file_key)
with xr.open_dataset(files[len(files)//2]) as dataset:
    print(dataset)
    od=options_database(dataset)
    print(od)
    [iix, iiy, iiz, iit] = get_string_index(dataset.dims, ['x', 'y', 'z', 'time'])
    [xvar, yvar, zvar, tvar] = [list(dataset.dims)[i] for i in [iix, iiy, iiz, iit]]

    times = dataset.coords[tvar].values
    nx = len(dataset.coords[xvar].values)
    ny = len(dataset.coords[yvar].values)
    nz = len(dataset.coords[zvar].values)

if test_case == 0:
    ref_time = times[len(times)//2]

    duration = 30
    start_time = ref_time - duration * 60
    end_time = ref_time + duration * 60
elif test_case == 2:
    ref_time = 90* 60

    duration = 30
    start_time = ref_time - duration * 60
    end_time = ref_time + duration * 60

print(start_time, ref_time, end_time)

kwa={'thresh':1.0E-5}

#%%
interp_method = "tri_lin"
interp_order = 1

time1 = time.perf_counter()

traj_ref = tc.Trajectories(files, ref_prof_file,
                           start_time, ref_time, end_time,
                           dx, dy, dz,
                           cl.trajectory_cloud_ref, cl.in_cloud, kwargs=kwa,
                           interp_method = "tri_lin",
                           interp_order = 1,
                           variable_list = variable_list,
                           unsplit = True)

time2 = time.perf_counter()

delta_t = time2 - time1

print(f'Elapsed time = {delta_t}')

#%%
# for it in range(np.shape(traj_ref.trajectory)[0]):
#     tp.plot_traj_pos(traj_ref, it, 'ref')
# plt.show()

#%%

#%%
traj_class = cp.set_cloud_class(traj_ref, version = 1)
trsz = [np.size(np.where(traj_ref.labels == i)[0]) for i in range(traj_ref.nobjects)]

sel = np.argmax(trsz)
sel = 9

anim = tp.plot_traj_animation(traj_ref,
                              fps = 10,
                              select = [sel],
                              galilean = np.array([-7.6,-1.5]),
                              plot_class = traj_class,
                              legend = True,
#                              plot_field = True,
#                              with_boxes = True,
                              var = "tracer_rad1",
                              )

plt.show()

#%%
tp.plot_trajectory_history(traj_ref, sel, 'ref')
plt.show()

#%%
mean_prop = cp.cloud_properties(traj_ref, traj_class)
tp.plot_trajectory_mean_history(traj_ref, traj_class, mean_prop, 'ref')
plt.show()
cp.print_cloud_class(traj_ref, traj_class, sel, list_classes=True)

#%%
mask = (traj_ref.labels == sel)

tracer1 = traj_ref.data[:,mask,traj_ref.var("tracer_rad1")]
tracer2 = traj_ref.data[:,mask,traj_ref.var("tracer_rad2")]
dtrbydt1 = tracer1[1:,:] - tracer1[:-1,:]
dtrbydt2 = tracer2[1:,:] - tracer2[:-1,:]
#plt.plot(dtrbydt2/dtrbydt1)

#plt.show()
#%%

if True:
    time1 = time.perf_counter()

    traj_test = tc.Trajectories(files, ref_prof_file,
                               start_time, ref_time, end_time,
                               dx, dy, dz,
                               cl.trajectory_cloud_ref, cl.in_cloud, kwargs=kwa,
                               interp_method = "grid_interp",
                               interp_order = 5,
                               variable_list = variable_list,
                               unsplit = True)

    time2 = time.perf_counter()

    delta_t = time2 - time1

    print(f'Elapsed time = {delta_t}')
    #%%

    anim2 = tp.plot_traj_animation(traj_test,
                                  fps = 10,
    #                              select = [sel],
                                  galilean = np.array([-7.6,-1.5]),
    #                              plot_field = True,
    #                              with_boxes = True,
                                  var = "tracer_rad1",
                                  )

    plt.show()
#%%
    tp.plot_trajectory_history(traj_ref, sel, 'ref')
    plt.show()

#%%
    traj_class_test = cp.set_cloud_class(traj_test, version = 1)
    mean_prop_test = cp.cloud_properties(traj_test, traj_class_test)
    tp.plot_trajectory_mean_history(traj_test, traj_class_test, mean_prop_test, 'test')
    plt.show()
    #%%

    diff = traj_test.trajectory-traj_ref.trajectory

    for i, n in enumerate([nx, ny]):
    #    d = np.round(diff[..., i], 8) % n
    #    print(np.max(d))
    #    diff[..., i] = d
        k = diff [..., i] > n /2
        diff [..., i][k] -= n
        k = diff [..., i] < -n /2
        diff [..., i][k] += n
#        print(np.max(diff[..., i], axis = 1))

    #%%

    fig, ax = plt.subplots(1,1, figsize=(5,5))
    for i, dim in enumerate(["x", "y", "z"]):
    #    ax.plot(np.std(diff,axis=1)[:, i], label = dim)
        d = np.abs(diff[..., i])
        ax.plot(np.percentile(d, 95, axis=1), label = dim)
    plt.legend()
    plt.show()
