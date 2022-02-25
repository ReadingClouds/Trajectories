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
import trajectories.w_selection as ws
import trajectories.cloud_properties as cp

import time

from subfilter.utils.string_utils import get_string_index
from subfilter.io.MONC_utils import options_database

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


dX = [dx, dy, dz]

ref_prof_file = glob.glob(indir + ref_filespec + '*.nc')[0]
files = glob.glob(indir + filespec + '*.nc')
files.sort(key=tc.file_key)
with xr.open_dataset(files[len(files)//2]) as dataset:
    print(dataset)
    od=options_database(dataset)
#    print(od)
    [iix, iiy, iiz, iit] = get_string_index(dataset.dims, ['x', 'y', 'z', 'time'])
    [xvar, yvar, zvar, tvar] = [list(dataset.dims)[i]
                                for i in [iix, iiy, iiz, iit]]

    times = dataset.coords[tvar].values
    nx = len(dataset.coords[xvar].values)
    ny = len(dataset.coords[yvar].values)
    nz = len(dataset.coords[zvar].values)

if test_case == 0:
    ref_time = times[len(times)//2]

    # duration = 30
    # start_time = ref_time - duration * 60
    # end_time = ref_time + duration * 60
    with xr.open_dataset(files[0]) as dataset:
        times = dataset.coords[tvar].values
        start_time = times[0]
    with xr.open_dataset(files[-1]) as dataset:
        times = dataset.coords[tvar].values
        end_time = times[-1]
    duration = ref_time - start_time
elif test_case == 2:
    ref_time = 90* 60

    duration = 30
    start_time = ref_time - duration * 60
    end_time = ref_time + duration * 60

print(start_time, ref_time, end_time)
with xr.open_dataset(files[len(files)//2]) as dataset:
    wmax=dataset['w'].sel({tvar:ref_time}).max().values

print(wmax)
kwa={'thresh': 0.9*wmax}

#%%
#interp_method = "tri_lin"
interp_order = 1

time1 = time.perf_counter()

traj_ref = tc.Trajectories(files, ref_prof_file,
                           start_time, ref_time, end_time,
                           dx, dy, dz,
                           ws.trajectory_w_ref, ws.in_obj, kwargs=kwa,
#                           interp_method = "tri_lin",
                           interp_method = "fast_interp",
                           interp_order = 1,
                           variable_list = variable_list,
                           unsplit = False)

time2 = time.perf_counter()

delta_t = time2 - time1

print(f'Elapsed time = {delta_t}')

#%%
fig, ax = plt.subplots(3,1,sharex=True)
plt.suptitle(f'Trajectories elapsed time = {delta_t:6.2f}')

for idim in range(3):
    c = ax[idim].plot(traj_ref.times, traj_ref.trajectory[:,:,idim]*dX[idim])
ax[0].set_ylim([0,7000])
ax[1].set_ylim([0,7000])
ax[2].set_ylim([0,2000])
for i,c in enumerate('xyz'):
    ax[i].set_ylabel(c)
ax[2].set_xlabel('time')
#    ax[idim].legend(handles=c)
fig.tight_layout()
fig.savefig("C:/Users/paclk\OneDrive - University of Reading/Git/python/advtraj/advtraj/cli/Trajectories.png")

plt.show()

#%%
ds = xr.open_dataset("C:/Users/paclk/OneDrive - University of Reading/ug_project_data/Data/trajectories/diagnostics_3d_ts_trajectories.nc")
x_adv = ds['x']
y_adv = ds['y']
z_adv = ds['z']

trref = traj_ref.trajectory[traj_ref.ref,:,:]
reftime = traj_ref.times[traj_ref.ref]
xref = x_adv.sel(time=reftime)
yref = y_adv.sel(time=reftime)
zref = z_adv.sel(time=reftime)

match=[]
for j, (x1, y1, z1) in enumerate(zip(xref.values, yref.values, zref.values)):
    delta_x = x1 -  trref[:,0] * dx
    delta_y = y1 -  trref[:,1] * dy
    delta_z = z1 -  (trref[:,2]-0.5) * dz
    r2 = delta_x*delta_x + delta_y*delta_y + delta_z*delta_z
#    print(r2)
    i = np.argmin(r2)
#    if delta_z[i] < 0 : i -=1
    if r2[i] == 0:
        match.append((j,i))
    print(j, i, r2[i], delta_x[i], delta_y[i], delta_z[i], )
#%%
fig, ax =plt.subplots(3,1,sharex=True)
for (j,i) in match:
    print(i, j)
    x1 = x_adv.isel(dim_0=j)
    diffx = x1-traj_ref.trajectory[:,i,0]*dx
    diffx[diffx >(6400/2)] -= 6400
    diffx[diffx <-(6400/2)] += 6400
    y1 = y_adv.isel(dim_0=j)
    diffy = y1-traj_ref.trajectory[:,i,1]*dx
    diffy[diffy >(6400/2)] -= 6400
    diffy[diffy <-(6400/2)] += 6400
    z1 = z_adv.isel(dim_0=j)
    diffz = z1-(traj_ref.trajectory[:,i,2]-0.5)*dz
    # print(x1)
    # print(diffx)
    ax[0].plot(traj_ref.times,diffx)
    ax[1].plot(traj_ref.times,diffy)
    ax[2].plot(traj_ref.times,diffz)
    for i,c in enumerate('xyz'):
        ax[i].set_ylabel(c)
    ax[2].set_xlabel('time')
fig.tight_layout()
fig.savefig("C:/Users/paclk\OneDrive - University of Reading/Git/python/advtraj/advtraj/cli/Traj_error.png")

plt.show()
