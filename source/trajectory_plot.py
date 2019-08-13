import os
from netCDF4 import Dataset
#from scipy.io import netcdf
import numpy as np
import matplotlib.pyplot as plt
#from datetime import datetime, timedelta
#from netCDF4 import num2date, date2num
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from trajectory_compute import file_key
L_vap = 2.501E6
Cp = 1005.0
L_over_cp = L_vap / Cp
grav = 9.81

def plot_trajectory_history(tr, select_obj, fn) :
    """
	Function to plot variables history of all trajectories in an object.
    
	Args: 
		tr                   : trajectory set object.
		select_obj (integer) : ID of object to plot.
		fn                   : root file name for save figures.
    
	Returns:
		Nothing
        
        Plots:
            "w","th","q_vapour","q_cloud_liquid_mass","theta_L","q_t" plus 3D
            history of cloudy points.
        	
	@author: Peter Clark
	
    """

    mask = (tr.labels == select_obj)
    
    fig, axa = plt.subplots(3,2,figsize=(8,10))
#    fig.clf
    traj = tr.trajectory[:,mask,:]
    data = tr.data[:,mask,:]
          
    z = (traj[:,:,2]-0.5)*tr.deltaz
    zn = (np.arange(0,np.size(tr.piref))-0.5)*tr.deltaz
#    print np.shape(z)
    
    #print np.shape(z)
    for j,v in enumerate(["w","th","q_vapour","q_cloud_liquid_mass"]):
#        print (j,v,var(v))        
        ax = axa[(j)%2,(j)//2]
        for i in range(np.shape(z)[1]-1) :
            ax.plot(data[:,i,tr.var(v)],z[:,i])
        ax.set_xlabel(tr.variable_list[v],fontsize=16)
        ax.set_ylabel(r"$z$ m",fontsize=16)
        ax.set_title('Cloud %2.2d'%select_obj)

    ax = axa[2,0]
    for i in range(np.shape(z)[1]-1) :
        piref_z = np.interp(z[:,i],zn,tr.piref)
#        print piref_z
        thl = data[:,i,tr.var("th")] - \
              L_over_cp*data[:,i,tr.var("q_cloud_liquid_mass")]/piref_z
#        print thl, data[:,var("th"),i],data[:,var("q_vapour"),i]
        ax.plot(thl,z[:,i])
    ax.set_xlabel(r"$\theta_L$ K",fontsize=16)
    ax.set_ylabel(r"$z$ m",fontsize=16)
    ax.set_title('Cloud %2.2d'%select_obj)
    
    ax = axa[2,1]
    for i in range(np.shape(z)[1]-1) :
        qt = data[:,i,tr.var("q_vapour")] + \
             data[:,i,tr.var("q_cloud_liquid_mass")]
#        print qt,data[:,var("q_vapour"),i],data[:,var("q_cloud_liquid_mass"),i]
        ax.plot( qt,z[:,i])
    ax.set_xlabel(r"$q_t$ kg/kg",fontsize=16)
    ax.set_ylabel(r"$z$ m",fontsize=16)
    ax.set_title('Cloud %2.2d'%select_obj)

    plt.tight_layout()
    plt.savefig(fn+'_Cloud_traj_%3.3d'%select_obj+'.png')
    
    fig1 = plt.figure(figsize=(10,6))

    ax1 = fig1.add_subplot(111, projection='3d')
    
    ax1.set_xlim(tr.xcoord[0]-10, tr.xcoord[-1]+10)
    ax1.set_ylim(tr.ycoord[0]-10, tr.ycoord[-1]+10)
    ax1.set_zlim(0, tr.zcoord[-1])
    for it in range(len(traj)):
        ax1.plot(traj[it,:,0],traj[it,:,1],zs=traj[it,:,2], \
                linestyle='',marker='.')
    ax1.set_title('Cloud %2.2d'%select_obj)
    
    plt.savefig(fn+'_Cloud_traj_pos_%3.3d'%select_obj+'.png')

    plt.close(fig1)
    
    return
        
def plot_trajectory_mean_history(tr, mean_prop, fn, \
                                 select=None, obj_per_plt=10) :  
    """
	Function to plot variables mean history of cloudy points.
    
	Args: 
		tr                   : trajectory set object.
		mean_prop (dict)     : Mean properties provided by tr.mean_properties().
		fn                   : root file name for save figures.
		select (integer)     : ID of object to plot. Default plots all objects.
		obj_per_plt(integer) : Number of objects to plot before starting new frame.
    
	Returns:
		Nothing
        
        Plots:
            "w","th","q_vapour","q_cloud_liquid_mass","theta_L","q_t","mse" 
            plus mse loss and various entrainment and detrainment parameters.
        	
	@author: Peter Clark
	
    """
    
    nvars = np.shape(tr.data)[2]
    z_ptr = nvars+3 # Index of variable in mean_prop which is height
    npts_ptr = nvars+4 # Index of variable in mean_prop which is 
                       # number of points averaged over
    nobj = np.shape(mean_prop['cloud'])[1]
    
    # Default is plot all objects.
    if select is None : select = np.arange(0,nobj)  
    
    # True heights.
    zn = (np.arange(0,np.size(tr.piref))-0.5)*tr.deltaz
    
    # Start plotting!
    new_fig = True
    obj_plotted = 0
    iobj = 0
    figs = 0
    while obj_plotted < np.size(select) :
        
        ymax = np.ceil(np.max(mean_prop['cloud top'])/100)*100
        if new_fig :
            fig1, axa = plt.subplots(3, 3, figsize=(10,10), sharey=True)
                        
            for j,v in enumerate(["w","th","q_vapour","q_cloud_liquid_mass"]):
    
                ax = axa[(j)%2,(j)//2]
                ax.set_xlabel(tr.variable_list[v],fontsize=16)
                ax.set_ylabel(r"$z$ m",fontsize=16)
                ax.set_ylim(0, ymax)
                
            ax = axa[2,0]
            ax.set_xlabel(r"$\theta_L$ K",fontsize=16)
            ax.set_ylabel(r"$z$ m",fontsize=16)
            ax.set_ylim(0,ymax)

            ax = axa[2,1]
            ax.set_xlabel(r"$q_t$ kg/kg",fontsize=16)
            ax.set_ylabel(r"$z$ m",fontsize=16)
            ax.set_ylim(0,ymax)

            ax = axa[0,2]
            ax.set_xlabel(r"Moist static energy kJ kg$^{-1}$",fontsize=16)
            ax.set_ylabel(r"$z$ m",fontsize=16)
            ax.set_ylim(0,ymax)
            
            ax = axa[1,2]
            ax.set_xlabel(r"Moist static energy change kJ kg$^{-1}$",\
                          fontsize=16)
            ax.set_ylabel(r"$z$ m",fontsize=16)
            ax.set_ylim(0,ymax)
            
            fig2, axb = plt.subplots(3, 2, figsize=(8,10), sharey=True)
            entrmax=0.01
            ax = axb[0,0]
            ax.set_xlabel(r"Volume km$^3$",fontsize=16)
            ax.set_ylabel(r"$z$ m",fontsize=16)
            ax.set_ylim(0,ymax)

            ax = axb[0,1]
            ax.set_xlabel(r"Detrainment rate s$^{-1}$",fontsize=16)
            ax.set_ylabel(r"$z$ m",fontsize=16)
            ax.set_ylim(0,ymax)

            ax = axb[1,0]
            ax.set_xlabel(r"Entrainment rate s$^{-1}$",fontsize=16)
            ax.set_ylabel(r"$z$ m",fontsize=16)
            ax.set_xlim(0,entrmax)
            ax.set_ylim(0,ymax)

            ax = axb[1,1]
            ax.set_xlabel(r"Side Entrainment rate s$^{-1}$",fontsize=16)
            ax.set_ylabel(r"$z$ m",fontsize=16)
            ax.set_xlim(0,entrmax)
            ax.set_ylim(0,ymax)

            ax = axb[2,0]
            ax.set_xlabel(r"Entrainment rate m$^{-1}$",fontsize=16)
            ax.set_ylabel(r"$z$ m",fontsize=16)
            ax.set_xlim(0,entrmax)
            ax.set_ylim(0,ymax)

            ax = axb[2,1]
            ax.set_xlabel(r"Side Entrainment rate m$^{-1}$",fontsize=16)
            ax.set_ylabel(r"$z$ m",fontsize=16)
            ax.set_xlim(0,entrmax)
            ax.set_ylim(0,ymax)

            new_fig = False
            figs +=1
            
        volume = tr.deltax*tr.deltay*tr.deltaz
            
        if np.isin(iobj,select) :
            incloud = np.arange(len(mean_prop['cloud'][:,iobj,npts_ptr]), \
                                dtype=int)
            incloud = np.logical_and( \
                            incloud >= mean_prop['cloud_trigger_time'][iobj],\
                            incloud <  mean_prop['cloud_dissipate_time'][iobj])
            precloud = np.arange(len(mean_prop['cloud'][:,iobj,npts_ptr]), \
                                dtype=int)
            precloud = (precloud < mean_prop['cloud_dissipate_time'][iobj])
            m = (mean_prop['cloud'][:,iobj,npts_ptr] > 0)
            m1 = np.logical_and(m, incloud)
            nplt = 72
            if iobj == nplt :
                print(mean_prop["cloud_trigger_time"][iobj])
                print(mean_prop["cloud_dissipate_time"][iobj])
                print(m1)

            z = (mean_prop['cloud'][:,iobj,z_ptr]-0.5)*tr.deltaz
            
            mbl = (mean_prop['pre_cloud_bl'][:,iobj,npts_ptr] > 0)
            mbl = np.logical_and(mbl, precloud)
            zbl = (mean_prop['pre_cloud_bl'][:,iobj,z_ptr]-0.5)*tr.deltaz
            mbl = np.logical_and(mbl, zbl<= mean_prop["min cloud base"][iobj])
            
            if iobj == nplt :
                print(mean_prop["first cloud base"][iobj])
                print(mean_prop["min cloud base"][iobj])
                print(mbl)
                print(zbl[mbl])
# w   qv
# th  qcl         
            for j,v in enumerate(["w","th","q_vapour","q_cloud_liquid_mass"]):    
                ax = axa[(j)%2,(j)//2]
                line = ax.plot(mean_prop['pre_cloud_bl']\
                               [:,iobj,tr.var(v)][mbl], zbl[mbl])
                ax.plot(mean_prop['cloud'][:,iobj,tr.var(v)][m1], z[m1], \
                        color = line[0].get_color(), linewidth=4, \
                         label='{}'.format(iobj))

# theta_l
            ax = axa[2,0]
            piref_z = np.interp(z,zn,tr.piref)
            thl = mean_prop['cloud'][:,iobj,tr.var("th")] - \
              L_over_cp * \
              mean_prop['cloud'][:,iobj,tr.var("q_cloud_liquid_mass")] \
              / piref_z
              
            thl_bl = mean_prop['pre_cloud_bl'][:,iobj,tr.var("th")] - \
              L_over_cp * \
              mean_prop['pre_cloud_bl'][:,iobj,tr.var("q_cloud_liquid_mass")] \
              / piref_z
                           
    #        print thl, data[:,var("th"),i],data[:,var("q_vapour"),i]
            line = ax.plot(thl_bl[mbl],zbl[mbl])
            ax.plot(thl[m1],z[m1], label='{}'.format(iobj), \
                    color = line[0].get_color(), linewidth=4)
# qt
            ax = axa[2,1]
            qt = mean_prop['cloud'][:,iobj,tr.var("q_vapour")] + \
                 mean_prop['cloud'][:,iobj,tr.var("q_cloud_liquid_mass")]
                 
            qt_bl = mean_prop['pre_cloud_bl'][:,iobj,tr.var("q_vapour")] + \
                 mean_prop['pre_cloud_bl'][:,iobj,tr.var("q_cloud_liquid_mass")]
    #        print qt,data[:,var("q_vapour"),i],data[:,var("q_cloud_liquid_mass"),i]
            line = ax.plot( qt_bl[mbl],zbl[mbl])
            ax.plot( qt[m1],z[m1], label='{}'.format(iobj), \
                    color = line[0].get_color(), linewidth=4)
# mse
            ax = axa[0,2]
            mse = mean_prop['cloud'][:,iobj,nvars] / 1000.0
            mse_bl = mean_prop['pre_cloud_bl'][:,iobj,nvars] / 1000.0
            line = ax.plot(mse_bl[mbl], zbl[mbl])
            ax.plot(mse[m1], z[m1], label='{}'.format(iobj), \
                    color = line[0].get_color(), linewidth=4)
# mse loss
            ax = axa[1,2]
            m2 = (mean_prop['cloud'][1:,iobj,npts_ptr] >10)
            z1 = (mean_prop['cloud'][1:,iobj,z_ptr][m2]-0.5)*tr.deltaz
#           now = cloud + entr + entr_bot + detr ( + bl + above bl)
#           pre = cloud_pre + entr_pre + entr_bot_pre + bl + above_bl
            
            mse_now  = mean_prop['cloud'][1:,iobj,nvars][m2] * \
                       mean_prop['cloud'][1:,iobj,npts_ptr][m2]
                       
            if iobj == nplt : print('now',mse_now)
            
            mse_entr = mean_prop['entr'][1:,iobj,nvars][m2] * \
                       mean_prop['entr'][1:,iobj,npts_ptr][m2] 
                       
            if iobj == nplt : print('entr',mse_entr)
            
            mse_entr_bot = mean_prop['entr_bot'][1:,iobj,nvars][m2] * \
                           mean_prop['entr_bot'][1:,iobj,npts_ptr][m2]
                       
            if iobj == nplt : print('entr_bot', mse_entr_bot)
            
            mse_detr = mean_prop['detr'][1:,iobj,nvars][m2] * \
                       mean_prop['detr'][1:,iobj,npts_ptr][m2] - \
                       mean_prop['detr'][:-1,iobj,nvars][m2] * \
                       mean_prop['detr'][:-1,iobj,npts_ptr][m2]
                       
            if iobj == nplt : print('detr', mse_detr)
            
            mse_prev = mean_prop['cloud'][:-1,iobj,nvars][m2] * \
                       mean_prop['cloud'][:-1,iobj,npts_ptr][m2] +\
                       mean_prop['entr'][:-1,iobj,nvars][m2] * \
                       mean_prop['entr'][:-1,iobj,npts_ptr][m2] + \
                       mean_prop['entr_bot'][:-1,iobj,nvars][m2] * \
                       mean_prop['entr_bot'][:-1,iobj,npts_ptr][m2]
                       
            if iobj == nplt : print('prev',mse_prev)
                       
            mse_entr_pre = mean_prop['pre_cloud_above_bl'][:-1,iobj,nvars][m2] * \
                           mean_prop['pre_cloud_above_bl'][:-1,iobj,npts_ptr][m2]- \
                           mean_prop['pre_cloud_above_bl'][1:,iobj,nvars][m2] * \
                           mean_prop['pre_cloud_above_bl'][1:,iobj,npts_ptr][m2]
                           
            if iobj == nplt : print('entre_pre', mse_entr_pre)
                           
                                   
            mse_entr_pre_bot = mean_prop['pre_cloud_bl'][:-1,iobj,nvars][m2] * \
                               mean_prop['pre_cloud_bl'][:-1,iobj,npts_ptr][m2] - \
                               mean_prop['pre_cloud_bl'][1:,iobj,nvars][m2] * \
                               mean_prop['pre_cloud_bl'][1:,iobj,npts_ptr][m2]
                               
            if iobj == nplt : print('entr_bot_pre', mse_entr_pre_bot)
            
            mse_total_now = mse_now + mse_entr + mse_entr_bot + mse_detr
            
            if iobj == nplt : print('mse_total_now', mse_total_now)
            
            mse_total_pre = mse_prev + mse_entr_pre + mse_entr_pre_bot
            
            if iobj == nplt : print('mse_total_pre', mse_total_pre)
                                                                          
            mse_loss = mse_total_now - mse_total_pre 
            
            if iobj == nplt : print('loss', mse_loss)
            
            n_cloud_points = mean_prop['cloud'][1:,iobj,npts_ptr][m2] + \
                             mean_prop['entr'][1:,iobj,npts_ptr][m2] + \
                             mean_prop['entr_bot'][1:,iobj,npts_ptr][m2] + \
                             mean_prop['detr'][1:,iobj,npts_ptr][m2] - \
                             mean_prop['detr'][:-1,iobj,npts_ptr][m2]
                             
#            print(len(m2),len(incloud[1:]))
            m3 = np.logical_and(m2, incloud[1:])[m2]
                              
            mse_loss = mse_loss / n_cloud_points / 1000.0
            
#            line = ax.plot(mse_loss, z1)
            ax.plot(mse_loss[m3], z1[m3],\
                    linewidth=4, \
                    label='{}'.format(iobj))

############################################################################
# Cloud volume                      
            ax = axb[0,0]
            mass = mean_prop['cloud'][:,iobj,npts_ptr]*volume/1E9
            line = ax.plot(mass[m], z[m])
            ax.plot(mass[m1], z[m1], label='{}'.format(iobj), \
                    color = line[0].get_color(), linewidth=4)
# Detrainment rate
            ax = axb[0,1]
            n_cloud_points = mean_prop['cloud'][1:,iobj,npts_ptr]
            n_new_not_cloud_points = mean_prop['detr'][1:,iobj,npts_ptr] - \
                                     mean_prop['detr'][:-1,iobj,npts_ptr]
            m2 = np.logical_and(n_cloud_points > 0 , n_new_not_cloud_points > 0 )
            m2 = np.logical_and(m2, incloud[1:])
            
            z1 = (mean_prop['cloud'][1:,iobj,z_ptr][m2]-0.5)*tr.deltaz 
            detr_rate = n_new_not_cloud_points[m2] / \
               ( n_cloud_points[m2] + n_new_not_cloud_points[m2] / 2.0) / \
                (tr.times[1:][m2]-tr.times[0:-1][m2])
            ax.plot(detr_rate[detr_rate>0], z1[detr_rate>0], \
                         linestyle='' ,marker='.', \
                         label='{}'.format(iobj))

# Entrainment rate (time)
            ax = axb[1,0]
            n_cloud_points = mean_prop['cloud'][1:,iobj,npts_ptr]
            n_new_cloud_points = mean_prop['entr'][1:,iobj,npts_ptr] + \
                                 mean_prop['entr_bot'][1:,iobj,npts_ptr]
            m2 = np.logical_and(n_cloud_points > 0 , n_new_cloud_points > 0 )            
            m2 = np.logical_and(m2, incloud[1:])
                     
            z1 = (mean_prop['cloud'][1:,iobj,z_ptr][m2]-0.5)*tr.deltaz 
            
#            print(n_cloud_points[m2],n_new_cloud_points[m2], \
#                  n_cloud_points[m2] - n_new_cloud_points[m2] / 2.0)
#            print(tr.times[1:][m2]-tr.times[:-1][m2])
            entr_rate = n_new_cloud_points[m2] / \
               ( n_cloud_points[m2] + n_new_cloud_points[m2] / 2.0) / \
                (tr.times[1:][m2]-tr.times[:-1][m2])
                
            ax.plot(entr_rate[entr_rate>0], z1[entr_rate>0], \
                         linestyle='' ,marker='.', \
                         label='{}'.format(iobj))
            
# Entrainment rate (space)
            ax = axb[2,0]
            entr_rate_z = entr_rate / mean_prop['cloud'][1:,iobj,tr.var('w')][m2]
            ax.plot(entr_rate_z[entr_rate_z>0], z1[entr_rate_z>0], \
                         linestyle='' ,marker='.', \
                         label='{}'.format(iobj))
            
# Side Entrainment rate 
            ax = axb[1,1]
            n_cloud_points = mean_prop['cloud'][1:,iobj,npts_ptr]
            n_new_cloud_points = mean_prop['entr'][1:,iobj,npts_ptr]
            m2 = np.logical_and(n_cloud_points > 0 , n_new_cloud_points > 0 )
            m2 = np.logical_and(m2, incloud[1:])
            
            z1 = (mean_prop['cloud'][1:,iobj,z_ptr][m2]-0.5)*tr.deltaz 
            entr_rate = n_new_cloud_points[m2] / \
               ( n_cloud_points[m2] + n_new_cloud_points[m2] / 2.0) / \
                (tr.times[1:][m2]-tr.times[0:-1][m2])
            ax.plot(entr_rate[entr_rate>0], z1[entr_rate>0], \
                         linestyle='' ,marker='.', \
                         label='{}'.format(iobj))
# Side Entrainment rate  (space)
            ax = axb[2,1]
            entr_rate_z = entr_rate / mean_prop['cloud'][1:,iobj,tr.var('w')][m2]
            ax.plot(entr_rate_z[entr_rate_z>0], z1[entr_rate_z>0], \
                         linestyle='' ,marker='.', \
                         label='{}'.format(iobj))

           
            obj_plotted +=1
            if ((obj_plotted % obj_per_plt) == 0) or \
               ( obj_plotted == np.size(select) ) :
                new_fig = True

                plt.figure(fig1.number)
                axa[0,0].legend()
                plt.tight_layout()
                plt.savefig(fn+\
                            '_Cloud_mean_traj_p1_{:02d}_v{:01d}.png'.\
                            format(figs, mean_prop['version']))

                plt.figure(fig2.number)
                axb[0,0].legend()
                plt.tight_layout()
                plt.savefig(fn+\
                            '_Cloud_mean_traj_p2_{:02d}_v{:01d}.png'.\
                            format(figs, mean_prop['version']))
    
                plt.show()
                plt.close(fig1)
                plt.close(fig2)
        iobj +=1
    
    return
   
def plot_traj_pos(traj, index, fn, save=False) :
    """
	Function to plot a single 3D plot of a set of trajectories at a given time.
    
    Args: 
        traj                 : trajectory set object.
        index (integer)      : Time index.
        fn                   : Root file name for save figures.
        save (bool)          : Save figure.
    
    Returns:
        Nothing
    
    @author: Peter Clark
	
    """

    # First set up the figure, the axis, and the plot element we want to animate

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(traj.xcoord[0]-10, traj.xcoord[-1]+10)
    ax.set_ylim(traj.ycoord[0]-10, traj.ycoord[-1]+10)
    ax.set_zlim(0, traj.zcoord[-1])
    
    line_list = list([])
    #ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
    for iobj in range(0,traj.nobjects):
        line, = ax.plot(traj.trajectory[index,traj.labels == iobj,0], \
                        traj.trajectory[index,traj.labels == iobj,1], \
                   zs = traj.trajectory[index,traj.labels == iobj,2], \
                   linestyle='' ,marker='.')
        line_list.append(line)
        
    plt.title('Time {:6.0f} index {:03d}'.format(traj.times[index]/60,index))
        
    if save : plt.savefig(fn+'_pos_{:03d}.png'.format(index))
    plt.show()
    plt.close(fig)
    return
 
def gal_trans(x, y, galilean, j, timestep, traj) :  
    if galilean[0] != 0 :
        x = ( x - galilean[0]*j*timestep/traj.deltax )%traj.nx
    if galilean[1] != 0 :
        y = ( y - galilean[1]*j*timestep/traj.deltay )%traj.ny
    return x, y

def box_xyz(b):
    x = np.array([b[0,0],b[0,0],b[1,0],b[1,0],b[0,0], \
                  b[0,0],b[0,0],b[1,0],b[1,0],b[0,0]])
    y = np.array([b[0,1],b[1,1],b[1,1],b[0,1],b[0,1], \
                  b[0,1],b[1,1],b[1,1],b[0,1],b[0,1]])
    z = np.array([b[0,2],b[0,2],b[0,2],b[0,2],b[0,2], \
                  b[1,2],b[1,2],b[1,2],b[1,2],b[1,2]])
    return x, y, z

def get_file_times(infiles, dir_override=None) :
    
    if dir_override is None :
        files = infiles
    else :
        files = list([])
        for file in infiles :
            filename = os.path.basename(file).split('\\')[-1] 
            files.append(dir_override+filename)
    
    file_times = np.zeros(len(files))
    for i, file in enumerate(files) : file_times[i] = file_key(file)
    return files, file_times

def plot_traj_animation(traj, save_anim=False, anim_name='traj_anim', \
                        legend = False, select = None, \
                        galilean = None, plot_field = False, \
                        dir_override = None, \
                        title = None, \
                        plot_class = None, \
                        no_cloud_size = 0.2, cloud_size = 2.0, \
                        field_size = 0.5, fps = 10, with_boxes = False, \
                        version = 1) :
    """
    Function to plot animation of trajectories.
    
    Args: 
        traj                 : Trajectory set object.
        save_anim (bool)     : If True, create mpeg animation file.
        anim_name (string)   : Name of animation file. Default is 'traj_anim'.
        legend (bool)        : If True, include legend.
        select (int)         : ID of object to plot.
        galilean (Array[2])  : Array with u and v components of system 
            velocity to apply Galilean Transformation to plot. Default in None.
        plot_field (bool)    : If True also plot Eulerian field of cloud.
        dir_override (string): Override original directory for file names 
            contained in traj.
        title (string)       : Title for plot. Default is None.
        plot_class (int array) : Classifications of trajectory points 
            provided by traj.mean_properties().
        no_cloud_size        : Size of symbols used to plot non-cloudy points.
            Default is 0.2.
        cloud_size           : Size of symbols used to plot cloudy points.
            Default is 2.0.
        field_size           : Size of symbols used to plot Eulerian cloud 
            points. Default is 0.5.
        fps (int)            : Frames per second in animation. Default is 10.
        with_boxes (bool)    : If True, include cloud box in plots.
        version (int)        : Version of classification scheme. Currently only
            version=1 supported.
        
    
    Returns:
        Nothing
       	
	@author: Peter Clark
	
    """

    ntraj = traj.ntimes
    nobj = traj.nobjects
    
    files, file_times = get_file_times(traj.files, dir_override=dir_override)
#                print(filename)
#    print(files)
    if select is None : select = np.arange(0, nobj)
    if version == 1 :
        class_key = list([\
            ["Not set", "0.3"] , \
            ["PRE_CLOUD_ENTR_FROM_BL","r"], \
            ["PRE_CLOUD_ENTR_FROM_ABOVE_BL","g"], \
            ["PREVIOUS_CLOUD","b"], \
            ["CLOUD","k"], \
            ["ENTRAINED_FROM_BL","c"], \
            ["ENTRAINED_FROM_ABOVE_BL","m"], \
            ["DETR_CLOUD","y"], \
            ["SUBS_CLOUD","0.6"], \
            ])
#    print(select)
    #input("Press any key...")
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(2,figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    
    if np.size(select) > 1 :
        x_min = traj.xcoord[0]
        x_max = traj.xcoord[-1]
        y_min = traj.ycoord[0]
        y_max = traj.ycoord[-1]
        
    else :
        iobj = select[0]
        x = traj.trajectory[0,traj.labels == iobj,0]
        y = traj.trajectory[0,traj.labels == iobj,1]

        xm = np.mean(x)
        xr = np.max(x)- np.min(x)
#        print(np.min(x),np.max(x))
        ym = np.mean(y)
        yr = np.max(y)- np.min(y)
        xr = np.min([xr,yr])/2
        x_min = xm-xr
        x_max = xm+xr
        y_min = ym-xr
        y_max = ym+xr
#        print(xm,xr,ym,yr)
        
#    print(x_min-10,x_max+10,y_min-10,y_max+10)

    ax.set_xlim(x_min-10,x_max+10)
    ax.set_ylim(y_min-10,y_max+10)       
    ax.set_zlim(0, traj.zcoord[-1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title is not None :
        ax.set_title(title)
    
    line_list = list([])

    if with_boxes :
        box_list = list([])
        
    if plot_field :
        line_field, = ax.plot([], [], linestyle='' ,marker='o', \
                            markersize = field_size, color = 'k')
        xg, yg, zg = np.meshgrid(traj.xcoord,traj.ycoord,traj.zcoord, \
                                 indexing = 'ij')
    nplt = 0
    timestep = traj.times[1]-traj.times[0]
    for iobj in range(0,traj.nobjects):
        if np.isin(iobj,select) :
            if plot_class is None : 
                line, = ax.plot([], [], linestyle='' ,marker='o', \
                                   markersize = no_cloud_size)
                line_cl, = ax.plot([], [], linestyle='' ,marker='o', \
                                   markersize = cloud_size, \
                                   color = line.get_color(),
                                   label='{}'.format(iobj))
                line_list.append([line, line_cl])
            else:
                line_for_class_list = list([])
                for iclass in range(0,8) :
                    line, = ax.plot([], [], linestyle='' ,marker='o', \
                                   markersize = cloud_size, \
                                   color = class_key[iclass][1],
                                   label = class_key[iclass][0])
                    line_for_class_list.append(line)
                line_list.append(line_for_class_list)
            
            if with_boxes :
                box, = ax.plot([],[],color = line.get_color())
                box_list.append(box)
                
            nplt +=1
        
    if legend : plt.legend()

    # initialization function: plot the background of each frame
    def init():
        if plot_field :
            line_field.set_data([], [])
        nplt = 0
        for iobj in range(0,traj.nobjects):
            if np.isin(iobj,select) :
#                if plot_class is None : 
#                    line_list[nplt][0].set_data([], [])
#                    line_list[nplt][1].set_data([], [])
#                else :
                for line in line_list[nplt]:
                    line.set_data([], [])
               
                if with_boxes :
                    box_list[nplt].set_data([], [])
                    
                nplt +=1
        return
    
    # animation function.  This is called sequentially
    def animate(i):
    #    j = traj.ntimes-i-1
        j = i 
    #    print 'Frame %d Time %d'%(i,j)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        if plot_field :
#            print('Plotting {}'.format(traj.times[j]))
            file_number = np.where(file_times >= traj.times[j])[0][0]
#            print(files[file_number])
            dataset=Dataset(files[file_number])
            qcl_field = dataset.variables["q_cloud_liquid_mass"]
            qcl_times = dataset.variables[qcl_field.dimensions[0]][...]
#            print(qcl_times)
            if len(qcl_times) == 1 :
                it = 0
            else :
                it = np.where(qcl_times == traj.times[j])[0][0]
#            print(file_number,it)
            in_cl = (qcl_field[it,...] > traj.ref_func_kwargs["thresh"])
            dataset.close()
            x = xg[in_cl]
            y = yg[in_cl]
            z = zg[in_cl]
            
            if galilean is not None :
                x, y = gal_trans(x, y,  galilean, j, timestep, traj) 
                       
            clip_arr = (x >= (x_min-10)) & (x <= (x_max+10)) \
                     & (y >= (y_min-10)) & (y <= (y_max+10))
            x = x[clip_arr]
            y = y[clip_arr]
            z = z[clip_arr]

            line_field.set_data(x, y)
            line_field.set_3d_properties(z) 
            
        nplt = 0
        for iobj in range(0,traj.nobjects):
            
            if np.isin(iobj,select) :

                x = traj.trajectory[j,traj.labels == iobj,0]
                y = traj.trajectory[j,traj.labels == iobj,1]
                z = traj.trajectory[j,traj.labels == iobj,2]
                if galilean is not None :
                    x, y = gal_trans(x, y,  galilean, j, timestep, traj) 

                x = conform_plot(x, traj.nx, xlim)
                y = conform_plot(y, traj.ny, ylim)
                        
                if plot_class is None : 
                    qcl = traj.data[j,traj.labels == iobj, \
                                    traj.var("q_cloud_liquid_mass")]
                    in_cl = (qcl > traj.ref_func_kwargs["thresh"]) 
                    not_in_cl = ~in_cl 
                    [line, line_cl] = line_list[nplt]
                    line.set_data(x[not_in_cl], y[not_in_cl])
                    line.set_3d_properties(z[not_in_cl])
                    line_cl.set_data(x[in_cl], y[in_cl])
                    line_cl.set_3d_properties(z[in_cl])
                else :
                    tr_class = plot_class[j,traj.labels == iobj]
                    for (iclass, line) in enumerate(line_list[nplt]) :
                        in_cl = (tr_class == iclass)
                        line.set_data(x[in_cl], y[in_cl])
                        line.set_3d_properties(z[in_cl])
                    
                if with_boxes :
                    b = traj.in_obj_box[j,iobj,:,:]
                    x, y, z = box_xyz(b)
                    if galilean is not None :
                        x, y = gal_trans(x, y, galilean, j, timestep, traj)                        

                    x = conform_plot(x, traj.nx, xlim)
                    y = conform_plot(y, traj.ny, ylim)

                    box = box_list[nplt]
                    box.set_data(x, y)
                    box.set_3d_properties(z)
                    
                nplt +=1
#        plt.title('Time index {:03d}'.format(ntraj-j-1))

        return 
    
#    Writer = animation.writers['ffmpeg']
#    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    
    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=ntraj, interval=1000./fps, blit=False)
    if save_anim : anim.save(anim_name+'.mp4', fps=fps)#, extra_args=['-vcodec', 'libx264'])
    plt.show()
    return

def plot_traj_family_members(traj_family, selection_list, galilean = None, \
                             with_boxes = False, asint = True, \
                             no_cloud_size = 0.2, cloud_size = 2.0) :
    """
    Function to plot animation of members of trajectory family.
    
    Args: 
        traj_family          : Trajectory_family object.
        selection_list (int) : IDs of objects to plot.
        galilean (Array[2])  : Array with u and v components of system 
            velocity to apply Galilean Transformation to plot. Default in None.
        with_boxes (bool)    : If True, include cloud box in plots.
        asint (bool)         : Round x,y,z to nearest integer (grid point).
        no_cloud_size        : Size of symbols used to plot non-cloudy points.
            Default is 0.2.
        cloud_size           : Size of symbols used to plot cloudy points.
            Default is 2.0.
        field_size           : Size of symbols used to plot Eulerian cloud 
            points. Default is 0.5.
        
    
    Returns:
        Nothing
       	
	@author: Peter Clark
	
    """

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    for selection in selection_list :
        iobj, ref, time = selection
        abs_time = ref+time-80
        tr_time = time
        tr = traj_family.family[ref]
        osel = (tr.labels == iobj)
        timestep = tr.times[1]-tr.times[0]
        x = tr.trajectory[tr_time,osel,0]
        y = tr.trajectory[tr_time,osel,1]
        z = tr.trajectory[tr_time,osel,2]
        ax.set_zlim(0, tr.zcoord[-1])
        if galilean is not None :
            x, y = gal_trans(x, y,  galilean, abs_time, timestep, tr)  

#        print(np.min(x),np.max(x))                      
#        print(np.min(y),np.max(y))                      
#        print(np.min(z),np.max(z))                      

        if asint :
            x = (x + 0.5).astype(int)
            y = (y + 0.5).astype(int)
            z = (z + 0.5).astype(int)
        qcl = tr.data[tr_time, osel, tr.var("q_cloud_liquid_mass")]
        in_cl = (qcl > tr.ref_func_kwargs["thresh"]) 
        not_in_cl = ~in_cl 
        
#        print(np.shape(x),np.shape(y),np.shape(y))
#        print(np.shape(x[not_in_cl]),np.shape(y[not_in_cl]),np.shape(y[not_in_cl]))
        
        line, = ax.plot(x[not_in_cl], y[not_in_cl], linestyle='' ,marker='o', \
                               markersize = no_cloud_size)
        line.set_3d_properties(z[not_in_cl])
        line_cl, = ax.plot(x[in_cl], y[in_cl], linestyle='' ,marker='o', \
                               markersize = cloud_size, \
                               color = line.get_color(),
                               label='{}'.format(selection))
        line_cl.set_3d_properties(z[in_cl])
        if with_boxes :
            b = tr.in_obj_box[tr_time,iobj,:,:]
            x, y, z = box_xyz(b)
            if galilean is not None :
                x, y = gal_trans(x, y, galilean, abs_time, timestep, tr)                        

            box, = ax.plot(x,y,color = line.get_color())
            box.set_3d_properties(z)
    plt.legend()
    plt.show()
    return

def plot_traj_family_animation(traj_family, match_index, \
                        overlap_thresh = 0.02, \
                        save_anim=False,  anim_name='traj_anim', \
                        legend = False, \
                        title = None, \
                        select = None,  super_obj = None, \
                        galilean = None, plot_field = False,
                        dir_override = None, \
                        no_cloud_size = 0.2, cloud_size = 2.0, \
                        field_size = 0.5, fps = 10, with_boxes = False) :
    """
    Function to plot animation of members of trajectory family.
    
    Args: 
        traj_family          : Trajectory_family object.
        match_index          : Index of time to match selected objects.
            If positive, matching object comes from 
            traj_family.matching_object_list_summary
            If negative and super_obj is None, matching object comes from 
            traj_family.find_linked_objects
            If negative and super_obj is not None, matching object comes from 
            super_obj
        save_anim (bool)     : If True, create mpeg animation file.
        anim_name (string)   : Name of animation file. Default is 'traj_anim'.
        legend (bool)        : If True, include legend.
        title (string)       : Title for plot. Default is None.
        select (int)         : IDs of objects to plot. None gives all objects.
        super_obj            : Super objects provided by
            traj_family.find_super_objects
        galilean (Array[2])  : Array with u and v components of system 
            velocity to apply Galilean Transformation to plot. Default in None.
        plot_field (bool)    : If True also plot Eulerian field of cloud.
        dir_override (string): Override original directory for file names 
            contained in traj.
        no_cloud_size        : Size of symbols used to plot non-cloudy points.
            Default is 0.2.
        cloud_size           : Size of symbols used to plot cloudy points.
            Default is 2.0.
        field_size           : Size of symbols used to plot Eulerian cloud 
            points. Default is 0.5.
        fps (int)            : Frames per second in animation. Default is 10.
        with_boxes (bool)    : If True, include cloud box in plots.        
    
    Returns:
        Nothing
       	
	@author: Peter Clark
	
    """
    
    traj = traj_family.family[-1]
    ref = len(traj_family.family) - 1
    nobj = traj.nobjects
    files, file_times = get_file_times(traj.files, dir_override=dir_override)

#    print(traj)
    if match_index >= 0 :
        
        if select is None : select = np.arange(0, nobj, dtype = int)
        match_traj = traj_family.family[-(1+match_index)]
        match_objs = traj_family.matching_object_list_summary( \
                select = select, overlap_thresh = overlap_thresh)
#        print(match_objs)
        plot_linked = False
        max_t = match_index -1
        nframes = traj.ntimes+match_index
        
    else:
        
        if select is None : 
            ref_obj = traj.max_at_ref
        else :
            ref_obj = select
        plot_linked = True
        max_t = 0
        if super_obj is None :
            linked_objs = traj_family.find_linked_objects(ref=ref, \
                                    select = ref_obj , \
                                    overlap_thresh = overlap_thresh)
#            print(linked_objs)
            for obj in linked_objs :
                for t,o,mint in obj :
                    max_t = np.max([max_t,ref-t])
        else :
            linked_objs = list([])
            for r in ref_obj :
                for s in super_obj :
                    if r in s[s[:,0]==ref,1] : 
#                            print(s)
                            linked_objs.append(s)
                            max_t = np.max(ref-s[:,0])
        nframes = traj.ntimes+max_t+1
#        print('linked_objs\n',linked_objs)
#        print(max_t, nframes)
#    print(match_traj)
#    print("Match index {}".format(match_index))
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    
    if np.size(select) > 1 :
        x_min = traj.xcoord[0]
        x_max = traj.xcoord[-1]
        y_min = traj.ycoord[0]
        y_max = traj.ycoord[-1]
    else :
        iobj = select[0]
        x = traj.trajectory[0,traj.labels == iobj,0]
        y = traj.trajectory[0,traj.labels == iobj,1]
        xm = np.mean(x)
        xr = np.max(x)- np.min(x)
#        print(np.min(x),np.max(x))
        ym = np.mean(y)
        yr = np.max(y)- np.min(y)
        xr = np.min([xr,yr])/2
        x_min = xm-xr
        x_max = xm+xr
        y_min = ym-xr
        y_max = ym+xr
#        print(xm,xr,ym,yr)
        
# For speed, create lists containing only data to be plotted.
        
# Contains just jrajectory positions, data and box coords for objects 
# in selection list.
    traj_list = list([])
    match_traj_list_list = list([])
    
    nplt = 0
#    for iobj in range(0,traj.nobjects):
    for iobj in select:
#        if np.isin(iobj,select) :
#        print("Adding {} to traj_list".format(iobj))
        traj_list.append((traj.trajectory[:,traj.labels == iobj,...], \
                                traj.data[:,traj.labels == iobj,...], 
                           traj.in_obj_box[:,iobj,...]) )
    
        match_list = list([])

        if plot_linked :
                
#                print(ref_obj, iobj)
                
            if np.isin(iobj,ref_obj) :
                    
#                    print(np.where(ref_obj  == iobj))
                mobj_ptr=np.where(ref_obj == iobj)[0][0]
#                print(mobj_ptr)
#                input("Press enter")
                
                linked_obj_list = linked_objs[mobj_ptr][1:,:]
#                    if super_obj is not None :
#                    linked_obj_list = linked_obj_list
#                        print('Super')
                for i in range(np.shape(linked_obj_list)[0]) :
                    match_obj = linked_obj_list[i,:]
#                        print("Linked object {}".format(match_obj))
                    match_traj = traj_family.family[match_obj[0]]
#                        print("Match traj", match_traj)
                    mobj = match_obj[1]
                    match_list.append((match_traj.trajectory\
                      [:, match_traj.labels == mobj, ...], \
                                       match_traj.data\
                      [:, match_traj.labels == mobj, ...], \
                                       match_traj.in_obj_box \
                      [:, mobj,...]) )                    
                    
        else :
                
            mobj_ptr=np.where(select == iobj)[0][0]
#            print(iobj, mobj_ptr)
            mob = match_objs[match_index-1][mobj_ptr]
                
#            print(mob)
#            input("Press enter")
#            for match_obj in mob :
#                print(match_obj)
#            input("Press enter")

            for match_obj in mob :
                mobj = match_obj[0]
#                print("Matching object {} {}".format(match_obj, mobj))
                match_list.append((match_traj.trajectory\
                      [:, match_traj.labels == mobj, ...], \
                                       match_traj.data\
                      [:, match_traj.labels == mobj, ...], \
                                       match_traj.in_obj_box \
                      [:, mobj,...]) )
    
        match_traj_list_list.append(match_list)
            
        nplt += 1
#    print(len(match_traj_list_list[0]))
#    print(match_traj_list_list)
#    input("Press enter")

    ax.set_xlim(x_min-10,x_max+10)
    ax.set_ylim(y_min-10,y_max+10)       
    ax.set_zlim(0, traj.zcoord[-1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title is not None :
        ax.set_title(title)

    line_list = list([])   
    match_line_list_list = list([])
       
    if with_boxes :
        box_list = list([])
        match_box_list_list = list([])
        
    if plot_field :
        line_field, = ax.plot([], [], linestyle='' ,marker='o', \
                            markersize = field_size, color = 'k')
        xg, yg, zg = np.meshgrid(traj.xcoord,traj.ycoord,traj.zcoord, \
                                 indexing = 'ij')
    
    nplt = 0
    timestep = traj.times[1]-traj.times[0]
    for iobj in select:
#    for iobj in range(0,traj.nobjects):
#        if np.isin(iobj,select) :
        
        line, = ax.plot([], [], linestyle='' ,marker='o', \
                               markersize = no_cloud_size)
        line_cl, = ax.plot([], [], linestyle='' ,marker='o', \
                               markersize = cloud_size, \
                               color = line.get_color(),
                               label='{}'.format([ref,iobj]))
        line_list.append([line,line_cl])
        if with_boxes :
            box, = ax.plot([],[],color = line.get_color())
            box_list.append(box)
                        
        match_line_list = list([])
        match_box_list = list([])
#            print(iobj,line_list)
#            input("Press enter - Here")
            
        if plot_linked :
                
            if np.isin(iobj, ref_obj) :
                    
                mobj_ptr=np.where(ref_obj == iobj)[0][0]
                linked_obj_list = linked_objs[mobj_ptr][1:,:]
                
                for i in range(np.shape(linked_obj_list)[0]) :
                    match_obj = linked_obj_list[i,:].copy()
#                    print('{}'.format(match_obj))
                    line, = ax.plot([], [], linestyle='' ,marker='o', \
                                           markersize = no_cloud_size)
                    line_cl, = ax.plot([], [], linestyle='' ,marker='o', \
                                           markersize = cloud_size, \
                                           color = line.get_color(), \
                                           label='{}'.format(match_obj))
                    match_line_list.append([line,line_cl])
                    if with_boxes :
                        box, = ax.plot([],[],color = line.get_color())
                        match_box_list.append(box)
                    
        else :
                
#                print(match_objs[match_index-1][iobj])
#                input("Press enter")
#                for match_obj in match_objs[match_index-1][iobj] :
#                    print(match_obj)
#                input("Press enter")
            mobj_ptr=np.where(select == iobj)[0][0]
            for match_obj in match_objs[match_index-1][mobj_ptr] :
#                    print("Matching object {} ho ho".format(match_obj))
                line, = ax.plot([], [], linestyle='' ,marker='o', \
                                       markersize = no_cloud_size)
                line_cl, = ax.plot([], [], linestyle='' ,marker='o', \
                                       markersize = cloud_size, \
                                       color = line.get_color(), \
                                       label='{}'.format(match_obj))
#                    print("Matching lines created")
                match_line_list.append([line,line_cl])
                if with_boxes :
                    box, = ax.plot([],[],color = line.get_color())
                    match_box_list.append(box)
#                    print(match_line_list)
            
        match_line_list_list.append(match_line_list)
        if with_boxes :
            match_box_list_list.append(match_box_list)
                
        nplt +=1
    if legend : plt.legend()
    
#    print(line_list)            
#    print(match_line_list_list)
#    print(len(match_line_list_list[0]))
#    input("Press enter")
    
    # initialization function: plot the background of each frame
    def init() :
        if plot_field :
            line_field.set_data([], [])
        nplt = 0
        for iobj in select:
#        for iobj in range(0,traj.nobjects):
#            if np.isin(iobj,select) :
#                print("Initializing line for object {}".format(iobj))
#                input("Press enter")
            for line in line_list[nplt] :
                line.set_data([], [])
                    
            for match_line_list in match_line_list_list[nplt] :
#                    print("Initialising matching line data",match_line_list)
#                    input("Press enter")
                for line in match_line_list :
                    line.set_data([], [])
                    
            if with_boxes :
                box_list[nplt].set_data([], [])
                for box in match_box_list_list[nplt] :
                    box.set_data([], [])
                
            nplt +=1
#            input("Press enter")
        return
    
    def set_line_data(tr, it, t_off, ln, ax) :
#        print("Setting line data")
#        print(tr,it,ln,ln_cl)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        tr_time = it + t_off
        if (tr_time >= 0 ) & (tr_time < np.shape(tr[0])[0]) :
            x = tr[0][tr_time,:,0]
            y = tr[0][tr_time,:,1]
            z = tr[0][tr_time,:,2]
            if galilean is not None :
                x, y = gal_trans(x, y,  galilean, it, timestep, traj) 
                
            x = conform_plot(x, traj.nx, xlim)
            y = conform_plot(y, traj.ny, ylim)

            qcl = tr[1][tr_time, :, traj.var("q_cloud_liquid_mass")]
            in_cl = (qcl > traj.ref_func_kwargs["thresh"]) 
            not_in_cl = ~in_cl 
            ln[0].set_data(x[not_in_cl], y[not_in_cl])
            ln[0].set_3d_properties(z[not_in_cl])
            ln[1].set_data(x[in_cl], y[in_cl])
            ln[1].set_3d_properties(z[in_cl])
        else :
            ln[0].set_data([], [])
            ln[0].set_3d_properties([])
            ln[1].set_data([], [])
            ln[1].set_3d_properties([])
        return

    def set_box_data(tr, it, t_off, box) :
#        print("Setting line data")
#        print(tr,it,ln,ln_cl)
        tr_time = it + t_off
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if (tr_time >= 0 ) & (tr_time < np.shape(tr[0])[0]) :
            b = tr[2][tr_time,:,:]
            x, y, z = box_xyz(b)
            if galilean is not None :
                x, y = gal_trans(x, y,  galilean, it, timestep, traj)  
                                      
            x = conform_plot(x, traj.nx, xlim)
            y = conform_plot(y, traj.ny, ylim)

            box.set_data(x, y)
            box.set_3d_properties(z)
        else :
            box.set_data([], [])
            box.set_3d_properties([])
        return
    # animation function.  This is called sequentially
    def animate(i):
        # i is frame no.
        # i == 0 at start of ref-match_index trajectories
        j = i - max_t - 1
#        else :
#           j = i - match_index 
        match_index = max_t + 1
#        input("Press enter")
#        print("Frame {0} {1}".format(i,j))
#        input("Press enter")
        if plot_field :

            if j >= 0 :
                dataset, file_number, it, delta_t = find_time_in_files(\
                                                    files, traj.times[j])
#                filename = match_traj.files[j]
#            else :                
#                filename = match_traj.files[i]
            qcl_field = dataset.variables["q_cloud_liquid_mass"]
            in_cl = (qcl_field[it,...] > traj.ref_func_kwargs["thresh"])
            x = xg[in_cl]
            y = yg[in_cl]
            z = zg[in_cl]
               
            if galilean is not None :
                x, y = gal_trans(x, y,  galilean, j, timestep, traj)                        
            
            clip_arr = (x >= (x_min-10)) & (x <= (x_max+10)) \
                     & (y >= (y_min-10)) & (y <= (y_max+10))
            x = x[clip_arr]
            y = y[clip_arr]
            z = z[clip_arr]

            line_field.set_data(x, y)
            line_field.set_3d_properties(z) 
            
        nplt = 0
        for iobj in select:
#        for iobj in range(0,traj.nobjects):
#            if np.isin(iobj,select) :
#                print("Setting line data", j, nplt, line_list[nplt])
#                input("Press enter")

            set_line_data(traj_list[nplt], j, 0, line_list[nplt], ax)
#                input("Press enter")
              
            if plot_linked :
                
                if np.isin(iobj,ref_obj) :
                    
                    mobj_ptr=np.where(ref_obj == iobj)[0][0] 
                    
                    linked_obj_list = linked_objs[mobj_ptr][1:,:]
                                            
#                        print(len(match_line_list_list[nplt]))
#                        print(len(match_traj_list_list[nplt]))
#                        print(len(linked_obj_list[:,0]))
                    for (match_line_list, m_traj, match_obj ) in \
                        zip(match_line_list_list[nplt], \
                            match_traj_list_list[nplt], \
                            linked_obj_list[:,0]) :
                        match_index = ref-match_obj
#                            print("match_index",match_index)
                        set_line_data(m_traj, j, match_index, \
                                      match_line_list, ax)
                               
                    if with_boxes :
                        set_box_data(traj_list[nplt], j, 0, \
                                     box_list[nplt])
                        for (box, m_traj, match_obj) in \
                            zip(match_box_list_list[nplt], \
                                match_traj_list_list[nplt], \
                                linked_obj_list[:,0]) :
        #                        print(box, m_traj)
                            match_index = ref-match_obj
                            set_box_data(m_traj, j, match_index, box)
                            
            else :
                    
#                    print(len(match_line_list_list[nplt]), \
#                            len(match_traj_list_list[nplt]))
                for (match_line_list, m_traj) in \
                    zip(match_line_list_list[nplt], \
                        match_traj_list_list[nplt]) :
#                        print(m_traj)
#                        print("Match line list", match_line_list)
    
                    set_line_data(m_traj, j, match_index, match_line_list, ax)
#                        input("Press enter")
                           
                if with_boxes :
                    set_box_data(traj_list[nplt], j, 0, box_list[nplt])
                    for (box, m_traj) in zip(match_box_list_list[nplt], \
                                             match_traj_list_list[nplt]) :
    #                        print(box, m_traj)
                        set_box_data(m_traj, j, match_index, box)
                        
            nplt +=1
#        plt.title('Time index {:03d}'.format(ntraj-j-1))

        return 
    
#    Writer = animation.writers['ffmpeg']
#    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

#    input("Press enter")
   
    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=nframes, interval=1000./fps, blit=False)
    if save_anim : anim.save(anim_name+'.mp4', fps=fps)#, extra_args=['-vcodec', 'libx264'])
    plt.show()
    return
    
def conform_plot(x, nx, xlim ) :
    if xlim[0] < 0 and xlim[1] < nx:
        x[x >= xlim[1]] -= nx
    if xlim[0] > 0 and xlim[1] > nx:
        x[x <= xlim[0]] += nx
    return x      

