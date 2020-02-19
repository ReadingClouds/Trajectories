import glob
import os
import os.path
import numpy as np
import matplotlib.pyplot as plt

import pickle as pickle

from trajectory_compute import *
from trajectory_plot import *

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
#   Set to True to calculate trajectory family,False to read pre-calculated from pickle file.
get_traj = False
#get_traj = True
   
files = glob.glob(dir+"diagnostics_3d_ts_*.nc")
files.sort(key=file_key)

def heat_map(ax, xd, yd, bins, cmap=plt.cm.Reds) :
    hist, xc, yc, im = ax.hist2d(xd, yd, bins, cmap=cmap)
    xp = (xc[0:-1]+xc[1:])*0.5
    zp = (yc[0:-1]+yc[1:])*0.5
    mn = np.mean(hist*xp[:,np.newaxis], axis=0)/np.mean(hist, axis=0)
    mn2 = np.mean(hist*xp[:,np.newaxis]**2, axis=0)/np.mean(hist, axis=0)
    std = np.sqrt(mn2-mn*mn)
    ax.errorbar(mn,zp,xerr=std,fmt='-k',capsize=5.0)
    return zp, mn, std

#def main():
'''
Top level code, a bit of a mess.
'''

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
            
 
    for traj_m in tfm.family :
        

        traj_m_class = set_cloud_class(traj_m, version = 1)
        nvars = np.shape(traj_m.data)[2]

        cloud_time.append(traj_m.times[traj_m.ref])
        print('Trajectory time: ',traj_m.times[traj_m.ref])
        
        mean_prop = cloud_properties(traj_m, traj_m_class)

        ndvars = len(mean_prop["derived_variable_list"])
        nposvars = 3

        z_ptr = nvars + ndvars + 2 # Index of variable in mean_prop which is height
        npts_ptr = nvars + ndvars + nposvars # Index of variable in mean_prop which is 
        mse_ptr = nvars + list(mean_prop["derived_variable_list"].keys()).index("MSE") 
        qt_ptr = nvars + list(mean_prop["derived_variable_list"].keys()).index("q_total")
        
        select = traj_m.max_at_ref
        
        cloud_lifetime.append(traj_m_class['cloud_dissipate_time'][select] - \
                         traj_m_class['cloud_trigger_time'][select])
        cloud_base.append(traj_m_class['min_cloud_base'][select])
        cloud_top.append(traj_m_class['cloud_top'][select])
        
        cloud_base_area.append(mean_prop['max_cloud_base_area'][select])
        cloud_base_variables.append(mean_prop['cloud_base_variables'][select, :]) 
        
         
        entr = list([])                 
        for iobj in select :             
            index_points = np.arange(len(mean_prop['cloud'][:,iobj,npts_ptr]), \
                                        dtype=int)
            incloud = np.logical_and( \
                      index_points >=  traj_m_class['cloud_trigger_time'][iobj],\
                      index_points  < traj_m_class['cloud_dissipate_time'][iobj])
            precloud = np.arange(len(mean_prop['cloud'][:,iobj,npts_ptr]), \
                                        dtype=int)
            precloud = (precloud < traj_m_class['cloud_dissipate_time'][iobj])
            cloud_gt_0 = (mean_prop['cloud'][:,iobj,npts_ptr] > 0)
            incloud = np.logical_and(cloud_gt_0, incloud)
            incloud_rates = incloud[1:]
            
            vol = mean_prop['cloud_properties'][:,iobj,CLOUD_VOLUME]
            max_cloud_index = np.where(vol == np.max(vol))[0][0]
            
            growing_cloud = np.logical_and(index_points <= max_cloud_index, incloud)
            growing_cloud_rates = growing_cloud[1:]
            
            z = mean_prop['cloud_properties'][:,iobj,CLOUD_HEIGHT]
            zbl = (mean_prop['pre_cloud_bl'][:,iobj,z_ptr]-0.5)*traj_m.deltaz
            in_bl = (mean_prop['pre_cloud_bl'][:,iobj,npts_ptr] > 0)
            in_bl = np.logical_and(in_bl, precloud)
            in_bl = np.logical_and(in_bl, zbl<= traj_m_class["min_cloud_base"][iobj])

            z_entr = mean_prop['cloud_properties'][:,iobj,CLOUD_HEIGHT][1:][growing_cloud_rates]        
            entr_rate = mean_prop["entrainment"][:,iobj,CB_ENTR][growing_cloud_rates]
            entr_rate_z = mean_prop["entrainment"][:,iobj,CB_ENTR_Z][growing_cloud_rates]
            side_entr_rate = mean_prop["entrainment"][:,iobj,SIDE_ENTR][growing_cloud_rates]
            side_entr_rate_z = mean_prop["entrainment"][:,iobj,SIDE_ENTR_Z][growing_cloud_rates]
            
            if len(z_entr) != len(entr_rate) :
                print("Object ", iobj, len(z_entr), len(entr_rate), \
                len(mean_prop['cloud_properties'][:,iobj,CLOUD_HEIGHT]), \
                len(mean_prop["entrainment"][:,iobj,CB_ENTR]))
                

            entr.append([z_entr, entr_rate, entr_rate_z, side_entr_rate, \
                                side_entr_rate_z])
        entrainment.append(entr)

cloud_dict={\
"cloud_time": cloud_time, \
"cloud_lifetime": cloud_lifetime, \
"cloud_base": cloud_base, \
"cloud_top": cloud_top, \
"cloud_base_area": cloud_base_area, \
"cloud_base_variables": cloud_base_variables, \
"entrainment": entrainment, \
}

outf = open(dir+'Cloud_props','wb')
print('Pickling ',dir+'Cloud_props')
pickle.dump(tfm, outf)
outf.close()


c_life=np.hstack(cloud_lifetime)
c_top=np.hstack(cloud_top)
c_base=np.hstack(cloud_base)
c_base_area=np.hstack(cloud_base_area)
c_depth=c_top-c_base

c_base_variables = np.zeros((len(cloud_base_variables[0][0]),0))
for c in cloud_base_variables :
    for cbv in c :
#        print(np.shape(c_base_variables), np.shape(cbv))
        c_base_variables = np.concatenate((c_base_variables, cbv[:,np.newaxis]),axis=1)
        
z=np.array([])                                                                      
ent=np.zeros((4,0))
n_clouds =0
for e in entrainment:
    for c in e :
        if len(c[0]>0) : 
            n_clouds += 1
            if len(c[0]) == len(c[4]) :
                z = np.concatenate((z,c[0])) 
                arr = np.array(c[1:])
                ent = np.concatenate((ent, arr),axis=1)
print(n_clouds)


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

sel_list = traj_m.max_at_ref
 # Plot max_list clouds mean history         
if False :
    plot_trajectory_mean_history(traj_m, traj_m_class, mean_prop, fn, select = sel_list) 

# Plot subset max_list clouds with galilean transform
# Not needed   
if False :
    plot_traj_animation(traj_m, save_anim=False, select = sel_list, \
                    no_cloud_size = 0.2, cloud_size = 2.0, legend = True, \
                    with_boxes = False, galilean = np.array([-8.5,0]))


#input("Press Enter to continue...")
 
if False :
    plot_traj_animation(traj_m, save_anim=False, \
                    anim_name='traj_cloud_sel_field', \
                    select = sel_list, \
                    fps=10, legend = True, plot_field = True, \
                    dir_override=dir, \
                    no_cloud_size = 0.2, cloud_size = 2.0, field_size = 0.5, \
                    title = 'Reference Time {0} Galilean Trans with clouds'.\
                    format(last_ref_time), with_boxes = False, \
                    galilean = np.array([-8.5,0]))
    
    
if False :
    plt.hist(len_sup, bins = np.arange(0.5,16.5), density=True)
    plt.title('Threshold = {:2.0f}%'.format(th*100))
    plt.xlabel('Super-object length (min)')
    plt.ylabel('Fraction of super objects')
    plt.savefig(dir+'Super_object_length.png')
    plt.show()  
        
    
#if __name__ == "__main__":
#    main() 
