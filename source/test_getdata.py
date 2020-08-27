# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 16:32:25 2020

@author: paclk
"""

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

from trajectory_compute import *

dn = 11
dir = 'C:/Users/paclk/OneDrive - University of Reading/traj_data/r{:02d}/'.format(dn)
deltax = 100.0
deltay = 100.0

files = glob.glob(dir+"diagnostics_3d_ts_*.nc")

var_list = { \
  "u":r"$u$ m s$^{-1}$", \
  "v":r"$v$ m s$^{-1}$", \
  "w":r"$w$ m s$^{-1}$", \
  "th":r"$\theta$ K", \
  "p":r"Pa", \
  "q_vapour":r"$q_{v}$ kg/kg", \
  "q_cloud_liquid_mass":r"$q_{cl}$ kg/kg", \
#  "tracer_rad1":r"Tracer 1 kg/kg", \
#  "tracer_rad2":r"Tracer 2 kg/kg", \
  "u_prime":r"$u$ m s$^{-1}$", \
  "dp_dx":r"Pa m^{-1}", \
  "dp_dz":r"Pa m^{-1}", \
  "d(d(q_total_prime)_dx)_dz":r"kg/kg m^{-1}", \
  }

dataset = Dataset(files[100])
time_index = 0
refprof=None

if 'zn' in dataset.variables:
    zn = dataset.variables['zn'][...]
    if 'z' in dataset.variables:
        z = dataset.variables['z'][...]
    else:
        z = np.zeros_like(zn)
        z[:-1] = 0.5 * ( zn[:-1] + zn[1:])
        z[-1]  = 2 * zn[-1] - z[-2]
else:
    if 'z' in dataset.variables:
        z = dataset.variables['z'][...]
    else:
        print('No z-grid info available in file. Using supplied deltaz')
        z = zcoord * deltaz
    zn = np.zeros_like(z)
    zn[-1] = 0.5 * (z[-1] + z[-2])
    for i in range(len(z)-2,  -1, -1 ):
        zn[i] = 2 * z[i] - zn[i+1]

coords = {
#    'xcoord': xcoord,
#    'ycoord': ycoord,
#    'zcoord': zcoord,
    'deltax': deltax,
    'deltay': deltay,
    'z'     : z,
    'zn'    : zn,
    }


data_list, varlist, varp_list, time = load_traj_step_data(dataset,
                                                   time_index, var_list,
                                                   refprof, coords)
