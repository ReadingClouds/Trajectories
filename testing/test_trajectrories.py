# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 14:58:36 2022

@author: paclk
"""

import glob

import numpy as np
import matplotlib.pyplot as plt

import xarray as xr

import trajectories.trajectory_compute as tc
import trajectories.trajectory_plot as tp
import trajectories.cloud_selection as cl
import trajectories.compute_cloud_properties as cp

import time

from subfilter.utils.string_utils import get_string_index
from subfilter.io.MONC_utils import options_database

#from trajectory_plot import *

#%%
test_case = 2

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

time1 = time.perf_counter()

traj = tc.Trajectories(files, ref_prof_file, start_time, ref_time, end_time,
                       dx, dy, dz,
                       cl.trajectory_cloud_ref, cl.in_cloud, kwargs=kwa,
                       variable_list=variable_list)

time2 = time.perf_counter()

delta_t = time2 - time1

print(f'Elapsed time = {delta_t}')

#%%

anim = tp.plot_traj_animation(traj,
                              fps = 10,
#                              select = [19],
                              galilean = np.array([-7.6,-1.5]),
                              plot_field = False,
#                              with_boxes = True,
                              )

plt.show()

#%%
traj_class = cp.set_cloud_class(traj, version = 1)
mean_prop = cp.cloud_properties(traj, traj_class)
tp.plot_trajectory_mean_history(traj, traj_class, mean_prop, 'test')
