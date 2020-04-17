import glob
import os
import os.path
import numpy as np
import matplotlib.pyplot as plt

import pickle as pickle

from trajectory_compute import *
from trajectory_plot import *

def heat_map(ax, xd, yd, bins, cmap=plt.cm.Reds) :
    hist, xc, yc, im = ax.hist2d(xd, yd, bins, cmap=cmap)
    xp = (xc[0:-1]+xc[1:])*0.5
    zp = (yc[0:-1]+yc[1:])*0.5
    mn = np.mean(hist*xp[:,np.newaxis], axis=0)/np.mean(hist, axis=0)
    mn2 = np.mean(hist*xp[:,np.newaxis]**2, axis=0)/np.mean(hist, axis=0)
    std = np.sqrt(mn2-mn*mn)
    ax.errorbar(mn,zp,xerr=std,fmt='-k',capsize=5.0)
    return zp, mn, std


CLOUD_HEIGHT = 0
CLOUD_POINTS = 1
CLOUD_VOLUME = 2

TOT_ENTR = 0
TOT_ENTR_Z = 1
SIDE_ENTR = 2
SIDE_ENTR_Z = 3
CB_ENTR = 4
CB_ENTR_Z = 5
DETR = 6
DETR_Z = 7
n_entr_vars = 8


dn = 5
#dir = 'C:/Users/paclk/OneDrive - University of Reading/traj_data/r{:02d}/'.format(dn)
dir = '/storage/silver/wxproc/xm904103/traj/BOMEX/r6n/'
   
dx = 100.0
dy = 100.0
dz = 40.0

#   Set to True to calculate trajectory family,False to read pre-calculated from pickle file.
get_traj = False
#get_traj = True
   
files = glob.glob(dir+"diagnostics_3d_ts_*.nc")
files.sort(key=file_key)

ref_prof_file = glob.glob(dir+'diagnostics_ts_*.nc')[0]
tr_back_len_min = 40
tr_forward_len_min = 30

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



'''
Top level code, a bit of a mess.
This uses computed families of trajectories from files in directory dir.
Current setup is back 40 min, forward 30 min from reference times every minute. 
Trajectories are calculated with reference times from 1 h in to 22 h 59 min - 
each hour's family is pickled in a separate pickle file.
If get_traj==True, these are computed first, otherwise they must already exist.
Various cloud parameter distributions are calculated.
'''

# It is easier to concatentate lists - at the end, the lists are consolidated into numpy arrays.
cloud_time = list([])
cloud_lifetime = list([])
cloud_base = list([])
cloud_top = list([])
cloud_base_area = list([])
cloud_base_variables = list([])
entrainment = list([])

for hh in range(1,23) :
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
                 dx, dy, dz, trajectory_cloud_ref, in_cloud, \
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
            
# Loop over trajectory sets for each reference time in trajectory family.
    for traj_m in tfm.family :
        
        # Classify trajectory points according to in cloud, being entrained from boundary-layer,
        # being entrained from above boundary-layer etc.
        # Note, this dictionary includes some additional cloud statistics.
        traj_m_class = set_cloud_class(traj_m, version = 1)
        
        # Total number of standard variables associated with each trajectory point.
        nvars = np.shape(traj_m.data)[2]
        
        # Save absolute times for trajectory points.
        cloud_time.append(traj_m.times[traj_m.ref])
        print('Trajectory time: ',traj_m.times[traj_m.ref])
        
        # After the classification, most of the work is done here.        
        # Compute means for each cloud over each classification of points.
        # This includes additional derived variables according to the 
        # cloud-properties function.
        # This also calculates entrainment rates etc.. 
        # See documentation
        mean_prop = cloud_properties(traj_m, traj_m_class)

        # Total number of derived variables associated with each trajectory point.
        ndvars = len(mean_prop["derived_variable_list"])
        # Total number of position variables associated with each trajectory point.
        nposvars = 3

        # Set up pointers into the traj_m.data and mean_prop arrays.
        
        # Index of variable in mean_prop which is height
        z_ptr = nvars + ndvars + 2 
        # Index of variable in mean_prop which is number of points in object.
        npts_ptr = nvars + ndvars + nposvars 
        # Index of Moist Static Energy
        mse_ptr = nvars + list(mean_prop["derived_variable_list"].keys()).index("MSE") 
        # Index of total water mixing ratio.
        qt_ptr = nvars + list(mean_prop["derived_variable_list"].keys()).index("q_total")
        
        # Objects (clouds) in traj_m to include in analysis.
        # traj_m.max_at_ref is those clouds with maximim total q_cl at reference time.        
        select = traj_m.max_at_ref
        
        # Extract cloud statistics from traj_m_class for selected objects.
        cloud_lifetime.append(traj_m_class['cloud_dissipate_time'][select] - \
                              traj_m_class['cloud_trigger_time'][select])
        cloud_base.append(traj_m_class['min_cloud_base'][select])
        cloud_top.append(traj_m_class['cloud_top'][select])      

        # Extract some mean data from mean_prop for selected objects.
        # See cloud_properties function for how these are computed.        
        cloud_base_area.append(mean_prop['max_cloud_base_area'][select])
        cloud_base_variables.append(mean_prop['cloud_base_variables'][select, :])         
        
        # Now we want to compute various numbers cloud by cloud.         
        entr = list([])                 
        for iobj in select : 
            # Create an index array for the object        
            index_points = np.arange(len(mean_prop['cloud'][:,iobj,npts_ptr]), \
                                        dtype=int)
                                        
            # Create mask for times between trigger time and dissipation time.                            
            incloud = np.logical_and( \
                      index_points >=  traj_m_class['cloud_trigger_time'][iobj],\
                      index_points  < traj_m_class['cloud_dissipate_time'][iobj])
            cloud_gt_0 = (mean_prop['cloud'][:,iobj,npts_ptr] > 0)
            incloud = np.logical_and(cloud_gt_0, incloud)
            # Rates are computed by finite difference so have 1 fewer items.
            incloud_rates = incloud[1:]
            
            # Extract timeseries of cloud volume and mean height
            vol = mean_prop['cloud_properties'][:,iobj,CLOUD_VOLUME]
            z = mean_prop['cloud_properties'][:,iobj,CLOUD_HEIGHT]
            # Find where cloud volume maximised.
            max_cloud_index = np.where(vol == np.max(vol))[0][0]
            
            # Creat mask for times in growing cloud.
            growing_cloud = np.logical_and(index_points <= max_cloud_index, incloud)
            growing_cloud_rates = growing_cloud[1:]
            
            # Create mask for times inside boundary layer 
            # First, must be before cloud dissipates
            precloud = np.arange(len(mean_prop['cloud'][:,iobj,npts_ptr]), \
                                        dtype=int)
            precloud = (precloud < traj_m_class['cloud_dissipate_time'][iobj])
            
            # Extract mean heights (m) for points in bl prior to entrainment.
            # This needs sorting for variable vertical grid.             
            zbl = (mean_prop['pre_cloud_bl'][:,iobj,z_ptr]-0.5)*traj_m.deltaz
            
            # Now we can make sure pre-cloud points are below cloud base.
            in_bl = (mean_prop['pre_cloud_bl'][:,iobj,npts_ptr] > 0)
            in_bl = np.logical_and(in_bl, precloud)
            in_bl = np.logical_and(in_bl, zbl<= traj_m_class["min_cloud_base"][iobj])

            # The above (in_bl) does not appear to be used but I've left it because it might be useful.
            
            # Extract various variables from mean prop just for the times where cloud exists and is growing.
            z_entr = mean_prop['cloud_properties'][:,iobj,CLOUD_HEIGHT][1:][growing_cloud_rates]        
            entr_rate = mean_prop["entrainment"][:,iobj,CB_ENTR][growing_cloud_rates]
            entr_rate_z = mean_prop["entrainment"][:,iobj,CB_ENTR_Z][growing_cloud_rates]
            side_entr_rate = mean_prop["entrainment"][:,iobj,SIDE_ENTR][growing_cloud_rates]
            side_entr_rate_z = mean_prop["entrainment"][:,iobj,SIDE_ENTR_Z][growing_cloud_rates]
            
            # This is just a check 
            if len(z_entr) != len(entr_rate) :
                print("Object ", iobj, len(z_entr), len(entr_rate), \
                len(mean_prop['cloud_properties'][:,iobj,CLOUD_HEIGHT]), \
                len(mean_prop["entrainment"][:,iobj,CB_ENTR]))
                
            # Add list of results to entr list. 
            entr.append([z_entr, entr_rate, entr_rate_z, side_entr_rate, \
                                side_entr_rate_z])
            # End of extraction for cloud.
        # Add list of results to entrainment list. 
        entrainment.append(entr)
    # End of reference time loop.
    
# Create single dictionary out of various variables extracted above.
cloud_dict={\
"cloud_time": cloud_time, \
"cloud_lifetime": cloud_lifetime, \
"cloud_base": cloud_base, \
"cloud_top": cloud_top, \
"cloud_base_area": cloud_base_area, \
"cloud_base_variables": cloud_base_variables, \
"entrainment": entrainment, \
}

# Pickle this for further procesing.
outf = open(dir+'Cloud_props','wb')
print('Pickling ',dir+'Cloud_props')
pickle.dump(cloud_dict, outf)
outf.close()

# Some variables are just lists of arrays - one entry per time, array of cloud ids.
# Convert to simple arrays.
c_life=np.hstack(cloud_lifetime)
c_top=np.hstack(cloud_top)
c_base=np.hstack(cloud_base)
c_base_area=np.hstack(cloud_base_area)
c_depth=c_top-c_base

# Consolidate cloud-base variables into an array over clouds.
c_base_variables = np.zeros((len(cloud_base_variables[0][0]),0))
for c in cloud_base_variables :
    for cbv in c :
#        print(np.shape(c_base_variables), np.shape(cbv))
        c_base_variables = np.concatenate((c_base_variables, cbv[:,np.newaxis]),axis=1)

# Consolidate z, entrainment rates (4 variables) into arrays and count number of clouds included.        
z=np.array([])                                                                      
ent=np.zeros((4,0))
n_clouds =0
for e in entrainment:
    for c in e :
        # Don't include cloud if no entrainment data.
        if len(c[0]>0) : 
            n_clouds += 1
            if len(c[0]) == len(c[4]) :
                z = np.concatenate((z,c[0])) 
                arr = np.array(c[1:])
                ent = np.concatenate((ent, arr),axis=1)
print(n_clouds)

# Time to do some plotting!
plt.hist(c_life, bins = np.arange(0,75,10, dtype=int), density=True)
plt.xlabel('Lagrangian lifetime (min)')
plt.ylabel('Fraction of clouds')
plt.title('{} Clouds'.format(len(c_life)))
plt.savefig(dir+'Cloud_lifetime.png')
plt.show()



c_sel = np.logical_and(c_base_area>0, c_depth>0)

plt.plot(np.sqrt(c_base_area[c_sel]/np.pi), c_life[c_sel],'.k')
plt.xlim([0,500])
plt.ylim([0,70])
plt.ylabel('Cloud Lifetime (min)')
plt.xlabel('Cloud base radius (m)')
plt.title('{} Clouds'.format(len(c_base_area[c_sel])))
plt.savefig(dir+'Cloud_life_vs_area.png')
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(8,8))
heat_map(ax,np.sqrt(c_base_area[c_sel]/np.pi), c_life[c_sel], \
    [np.arange(50,500,50), np.arange(00,75,5)])
ax.set_ylim([00,70])
ax.set_xlim([0,500])
ax.set_ylabel('Cloud Lifetime (min)')
ax.set_xlabel('Cloud base radius (m)')
ax.set_title('{} Clouds'.format(len(c_base_area[c_sel])))
#ax=plt.gca()
#ax.plot(np.sqrt(c_base_area[c_sel]/np.pi), c_depth[c_sel],'.k')
plt.savefig(dir+'Cloud_base_life_heat_map.png')
plt.show()


plt.plot(np.sqrt(c_base_area[c_sel]/np.pi), c_depth[c_sel],'.k')
plt.ylim([00,900])
plt.xlim([0,500])
plt.ylabel('Cloud Depth (m)')
plt.xlabel('Cloud base radius (m)')
plt.title('{} Clouds'.format(len(c_base_area[c_sel])))
plt.savefig(dir+'Cloud_depth.png')
plt.show()



fig0, ax = plt.subplots(1, 1, figsize=(8,8))
heat_map(ax,np.sqrt(c_base_area[c_sel]/np.pi), c_depth[c_sel], \
    [np.arange(50,500,50), np.arange(00,950,50)])
ax.set_ylim([00,900])
ax.set_xlim([0,500])
ax.set_ylabel('Cloud Depth (m)')
ax.set_xlabel('Cloud base radius (m)')
ax.set_title('{} Clouds'.format(len(c_base_area[c_sel])))
#ax=plt.gca()
#ax.plot(np.sqrt(c_base_area[c_sel]/np.pi), c_depth[c_sel],'.k')
plt.savefig(dir+'Cloud_base_heat_map.png')
plt.show()

fig1, axa = plt.subplots(3, 3, figsize=(10,10), sharex=True)
fntsz = 8    
yr = [ \
       [298.8,299.1], \
       [301.8,302.1], \
       [298.7,299.0], \
       [0.016, 0.017], \
       [0.0,   0.00006], \
       [0.016, 0.017], \
       [0.0, 1.5], \
       [1.0, 2.0], \
       [341000,343000], \
      ]        
for j,v in enumerate(["th","th_v","th_L",\
                      "q_vapour","q_cloud_liquid_mass","q_total",\
                      "w","tracer_rad1","MSE"]):

    if v in tfm.family[0].variable_list :
        lab = tfm.family[0].variable_list[v]
        vptr = tfm.family[0].var(v)
    elif v in mean_prop["derived_variable_list"] :
        lab = mean_prop["derived_variable_list"][v]
        vptr = nvars+list(mean_prop["derived_variable_list"].keys()).index(v)
    else :
        print("Variable {} not found.".format(v))
        lab = ""
        
    ax = axa[(j)%3,(j)//3]
    ybins = np.linspace(yr[j][0],yr[j][1],20)
    bins=[np.arange(0,550,50),ybins] 
#    zp, mn, std = heat_map(ax, np.sqrt(c_base_area[c_sel]/np.pi), \
#                               c_base_variables[vptr,c_sel], bins)
    ax.hist2d(np.sqrt(c_base_area[c_sel]/np.pi), \
                               c_base_variables[vptr,c_sel], bins, cmap=plt.cm.Reds)
    ax.set_xlabel('Cloud base radius (m)',fontsize=fntsz)
    ax.set_ylabel(lab,fontsize=fntsz)
    ax.set_xlim([0,500])
    ax.set_ylim(yr[j])
        
    
#    line = ax.plot(np.sqrt(c_base_area[c_sel]/np.pi), c_base_variables[vptr,c_sel], '.k')
plt.tight_layout()
plt.savefig(dir+'Cloud_base_variables.png')
plt.show()

labs = [r'Cloud Base Entrainment Rate (s$^{-1}$)', \
        r'Cloud Base Entrainment Rate (m$^{-1}$)', \
        r'Cloud Side Entrainment Rate (s$^{-1}$)', \
        r'Cloud Side Entrainment Rate (m$^{-1}$)', \
        ]
        
for i in range(4) :
    fig2, ax = plt.subplots(1, 1, figsize=(8,8))
    zp, mn, std = heat_map(ax, ent[i,:], z, \
      [np.arange(0,0.0055,0.0005),np.arange(600,1550,50)],cmap=plt.cm.Reds)
    
    ax.set_ylim([600,1500])
    ax.set_xlim([0,0.005])
    ax.set_ylabel('Cloud Height (m)')
    ax.set_xlabel(labs[i])
    ax.set_title('{} Clouds'.format(n_clouds))
    ax.annotate(r'1 $\sigma$ errorbar',(0.004,700))
#    plt.plot(zp,mn,'-k')
    plt.savefig(dir+'entr_heat_{:1d}.png'.format(i))
    plt.show()
    
    plt.errorbar(zp,mn,yerr=std,fmt='-k',capsize=5.0)
    plt.xlim([600,1500])
    plt.ylim([0,0.005])
    plt.xlabel('Cloud Height (m)')
    plt.ylabel(labs[i])
    plt.title('{} Clouds'.format(n_clouds))
    plt.annotate(r'1 $\sigma$ errorbar',(700,0.004))
    plt.savefig(dir+'entr_line_{:1d}.png'.format(i))
    plt.show()

