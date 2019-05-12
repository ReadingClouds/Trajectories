# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:12:17 2019

@author: paclk
"""

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import math

import pickle as pickle

from trajectory_compute import *
from trajectory_plot import *

def file_key(file):
    f1 = file.split('_')[-1]
    f2 = f1.split('.')[0]
    return float(f2)

def get_err(dirbase, tm, nback, nforward) :
    errlist = list([])
    for n in range(0,4) :
        dn = n + 11
        ts = 2**n
        dirfull = dirbase + 'r{:02d}/'.format(dn)
        ref_prof_file = glob.glob(dirfull+'diagnostics_ts_'+tm+'*.0.nc')[0]
        files = glob.glob(dirfull+'diagnostics_3d_ts_*.nc')
        files.sort(key=file_key)
        reffile = glob.glob(dirfull+'diagnostics_3d_ts_'+tm+'*.0.nc')[0]
        ref = files.index(reffile)
        start_file = ref - nback//ts
        end_file = ref + nforward//ts
        print(ref, start_file,end_file)
        print(files[ref])
        print(files[start_file])
        print(files[end_file])
        traj = trajectories(files, ref_prof_file, start_file, ref, end_file, \
                           100.0, 100.0, 40.0)
        print(np.shape(traj.trajectory))
        if n > 0 :
            err = traj.trajectory - reftraj.trajectory[::ts,...]
            err[:,:,0][err[:,:,0]>(traj.nx//2)] -= traj.nx
            err[:,:,0][err[:,:,0]<(-traj.nx//2)] += traj.nx
            err[:,:,1][err[:,:,1]>(traj.ny//2)] -= traj.nx
            err[:,:,1][err[:,:,1]<(-traj.ny//2)] += traj.nx
            print(np.max(err),np.min(err))
            errlist.append(err)
        else :
            reftraj = traj
    return  errlist

#dirbase = '/storage/silver/wxproc/xm904103/traj/BOMEX/'
dirbase = 'C:/Users/paclk/OneDrive - University of Reading/traj_data/'

#tm = '5280'
tm = '5760'
nback = 8
nforward = 8

dt = 60*8
errlist=None
for t in np.arange(3360,5760+dt,dt):
#for t in np.arange(5760,5760+dt,dt):
    tm = '{:4d}'.format(t)
    errl = get_err(dirbase, tm, nback, nforward)
    if errlist is None : 
        errlist = errl
    else :
        for i,err in enumerate(errl) :
            errlist[i]= np.concatenate((errlist[i],err),axis=1)
        

r=75
pcrl=list([])
for n,err in enumerate(errlist):
    ts = 2**(n+1)    
    pcr = np.percentile(err,r,axis=(1,2))
    plt.plot(np.arange(-nback,nforward+ts,ts),pcr,'*-',label='{:2d}'.format(ts))
    pcrl.append(pcr)
plt.legend() 
plt.title('{:2d} percentile'.format(r))
plt.xlabel('Time (min)')
plt.ylabel('Grid boxes')
plt.savefig('traj_errplot_{:2d}.png'.format(r))
plt.show()

#fig, ax = plt.subplots()
#ax.boxplot(errlist[0][:,:,1].T,sym='')
#plt.show()

ratio1 = pcrl[1][0]/pcrl[0][0]
two_to_alpha1 = (ratio1 + math.sqrt(ratio1*ratio1-4*(ratio1-1)))/2          
alpha1 = math.log(two_to_alpha1,2)
a1 = pcrl[0][0]/2.0**alpha1/nback

ratio2 = pcrl[1][-1]/pcrl[0][-1]
two_to_alpha2 = (ratio2 + math.sqrt(ratio2*ratio2-4*(ratio2-1)))/2          
alpha2 = math.log(two_to_alpha2,2)
a2 = pcrl[0][-1]/2.0**alpha2/nforward

print(alpha1, ratio1, a1)
print(alpha2, ratio2, a2)
        