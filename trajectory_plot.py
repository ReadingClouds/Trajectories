import os
from netCDF4 import Dataset
#from scipy.io import netcdf
import numpy as np
import matplotlib.pyplot as plt
#from datetime import datetime, timedelta
#from netCDF4 import num2date, date2num
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

L_vap = 2.501E6
Cp = 1005.0
L_over_cp = L_vap / Cp
grav = 9.81

def plot_trajectory_history(tr, select_obj, fn) :
    
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
    
    #
    plt.tight_layout()
    plt.savefig(fn+'_Cloud_traj_%3.3d'%select_obj+'.png')
    
    fig1 = plt.figure(figsize=(10,6))
#    fig1.clf
    
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
                                 select = None, obj_per_plt = 10) :  
    nvars = np.shape(tr.data)[2]
    nobj = np.shape(mean_prop['cloud'])[1]
    if select is None : select = np.arange(0,nobj)    
    zn = (np.arange(0,np.size(tr.piref))-0.5)*tr.deltaz
    new_fig = True
    obj_plotted = 0
    iobj = 0
    figs = 0
    while obj_plotted < np.size(select) :
        
        ymax = np.ceil(np.max(mean_prop['cloud top'])/100)*100
        if new_fig :
            fig1, axa = plt.subplots(3, 2, figsize=(8,10), sharey=True)
                        
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
            
            fig2, axb = plt.subplots(3, 2, figsize=(8,10), sharey=True)
            
            ax = axb[0,0]
            ax.set_xlabel(r"Volume km$^3$",fontsize=16)
            ax.set_ylabel(r"$z$ m",fontsize=16)
            ax.set_ylim(0,ymax)

            ax = axb[0,1]
            ax.set_xlabel(r"Entrainment rate s$^{-1}$",fontsize=16)
            ax.set_ylabel(r"$z$ m",fontsize=16)
            ax.set_ylim(0,ymax)

            ax = axb[1,0]
            ax.set_xlabel(r"Detrainment rate s$^{-1}$",fontsize=16)
            ax.set_ylabel(r"$z$ m",fontsize=16)
            ax.set_ylim(0,ymax)

            ax = axb[1,1]
            ax.set_xlabel(r"Side Entrainment rate s$^{-1}$",fontsize=16)
            ax.set_ylabel(r"$z$ m",fontsize=16)
            ax.set_ylim(0,ymax)

            ax = axb[2,0]
            ax.set_xlabel(r"Moist static energy kJ kg$^{-1}$",fontsize=16)
            ax.set_ylabel(r"$z$ m",fontsize=16)
            ax.set_ylim(0,ymax)
            
            ax = axb[2,1]
            ax.set_xlabel(r"Moist static energy change kJ kg$^{-1}$",\
                          fontsize=16)
            ax.set_ylabel(r"$z$ m",fontsize=16)
            ax.set_ylim(0,ymax)

            new_fig = False
            figs +=1
            
        volume = tr.deltax*tr.deltay*tr.deltaz
            
        if np.isin(iobj,select) :
            m = (mean_prop['cloud'][:,iobj,nvars+4] >0)
            z = (mean_prop['cloud'][:,iobj,nvars+3][m]-0.5)*tr.deltaz
            
            for j,v in enumerate(["w","th","q_vapour","q_cloud_liquid_mass"]):    
                ax = axa[(j)%2,(j)//2]
                ax.plot(mean_prop['cloud'][:,iobj,tr.var(v)][m], z)

            ax = axa[2,0]
            piref_z = np.interp(z,zn,tr.piref)
            thl = mean_prop['cloud'][:,iobj,tr.var("th")][m] - \
              L_over_cp * \
              mean_prop['cloud'][:,iobj,tr.var("q_cloud_liquid_mass")][m] \
              / piref_z
    #        print thl, data[:,var("th"),i],data[:,var("q_vapour"),i]
            ax.plot(thl,z)
            ax = axa[2,1]
            qt = mean_prop['cloud'][:,iobj,tr.var("q_vapour")][m] + \
                 mean_prop['cloud'][:,iobj,tr.var("q_cloud_liquid_mass")][m]
    #        print qt,data[:,var("q_vapour"),i],data[:,var("q_cloud_liquid_mass"),i]
            ax.plot( qt,z, label='{}'.format(iobj))

            ax = axb[0,0]
            mass = mean_prop['cloud'][:,iobj,nvars+4][m]*volume/1E9
            ax.plot(mass, z, label='{}'.format(iobj))

            ax = axb[0,1]
            n_cloud_points = mean_prop['cloud'][1:,iobj,nvars+4]
            n_new_cloud_points = mean_prop['entr'][1:,iobj,nvars+4] + \
                                 mean_prop['entr_bot'][1:,iobj,nvars+4]
            m1 = np.logical_and(n_cloud_points > 0 , n_new_cloud_points > 0 )
            
            z1 = (mean_prop['cloud'][1:,iobj,nvars+3][m1]-0.5)*tr.deltaz 
            entr_rate = n_new_cloud_points[m1] / \
               ( n_cloud_points[m1] - n_new_cloud_points[m1] / 2.0) / \
                (tr.times[1:][m1]-tr.times[0:-1][m1])
                
            ax.plot(entr_rate[entr_rate>0], z1[entr_rate>0], \
                         linestyle='' ,marker='.', \
                         label='{}'.format(iobj))

            ax = axb[1,0]
            n_cloud_points = mean_prop['cloud'][1:,iobj,nvars+4]
            n_new_not_cloud_points = mean_prop['detr'][1:,iobj,nvars+4]
            m1 = np.logical_and(n_cloud_points > 0 , n_new_not_cloud_points > 0 )
            
            z1 = (mean_prop['cloud'][1:,iobj,nvars+3][m1]-0.5)*tr.deltaz 
            entr_rate = n_new_not_cloud_points[m1] / \
               ( n_cloud_points[m1] + n_new_not_cloud_points[m1] / 2.0) / \
                (tr.times[1:][m1]-tr.times[0:-1][m1])
            ax.plot(entr_rate[entr_rate>0], z1[entr_rate>0], \
                         linestyle='' ,marker='.', \
                         label='{}'.format(iobj))

            
            ax = axb[1,1]
            n_cloud_points = mean_prop['cloud'][1:,iobj,nvars+4]
            n_new_cloud_points = mean_prop['entr_bot'][1:,iobj,nvars+4]
            m1 = np.logical_and(n_cloud_points > 0 , n_new_cloud_points > 0 )
            
            z1 = (mean_prop['cloud'][1:,iobj,nvars+3][m1]-0.5)*tr.deltaz 
            entr_rate = n_new_cloud_points[m1] / \
               ( n_cloud_points[m1] - n_new_cloud_points[m1] / 2.0) / \
                (tr.times[1:][m1]-tr.times[0:-1][m1])
            ax.plot(entr_rate[entr_rate>0], z1[entr_rate>0], \
                         linestyle='' ,marker='.', \
                         label='{}'.format(iobj))

            ax = axb[2,0]
            mse = mean_prop['cloud'][:,iobj,nvars][m] / 1000.0
            ax.plot(mse, z, label='{}'.format(iobj))

            ax = axb[2,1]
            m1 = (mean_prop['cloud'][1:,iobj,nvars+4] >0)
            z1 = (mean_prop['cloud'][1:,iobj,nvars+3][m1]-0.5)*tr.deltaz
            
            mse_now  = mean_prop['cloud'][1:,iobj,nvars][m1] * \
                       mean_prop['cloud'][1:,iobj,nvars+4][m1]
                       
            mse_prev = mean_prop['cloud'][0:-1,iobj,nvars][m1] * \
                       mean_prop['cloud'][0:-1,iobj,nvars+4][m1]
                       
            mse_entr = mean_prop['entr'][1:,iobj,nvars][m1] * \
                       mean_prop['entr'][1:,iobj,nvars+4][m1] + \
                       mean_prop['entr_bot'][1:,iobj,nvars][m1] * \
                       mean_prop['entr_bot'][1:,iobj,nvars+4][m1]
                       
            mse_detr = mean_prop['detr'][1:,iobj,nvars][m1] * \
                       mean_prop['detr'][1:,iobj,nvars+4][m1] 
                       
            mse_loss = mse_now + mse_detr - mse_entr - mse_prev  
            
            n_cloud_points = mean_prop['cloud'][1:,iobj,nvars+4][m1] + \
                             mean_prop['detr'][1:,iobj,nvars+4][m1]
                             
            mse_loss = mse_loss / n_cloud_points / 1000.0
            
#            print("MSE budget ",iobj,n_cloud_points)
#            print("MSE now ",mse_now/n_cloud_points/1000.0)
#            print("MSE detr ",mse_detr/n_cloud_points/1000.0)
#            print("MSE entr ",mse_entr/n_cloud_points/1000.0)
#            print("MSE prev ",mse_prev/n_cloud_points/1000.0)
            
            ax.plot(mse_loss, z1, label='{}'.format(iobj))
            
            obj_plotted +=1
            if ((obj_plotted % obj_per_plt) == 0) or \
               ( obj_plotted == np.size(select) ) :
                new_fig = True

                plt.figure(fig1.number)
                plt.legend()
                plt.tight_layout()
                plt.savefig(fn+\
                            '_Cloud_mean_traj_p1_{:02d}_v{:01d}.png'.\
                            format(figs, mean_prop['version']))

                plt.figure(fig2.number)
                plt.legend()
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
 
def gal_trans(x, y, galilean, j, timestep, traj, ax) :  
    if galilean[0] != 0 :
        x = ( x - galilean[0]*j*timestep/traj.deltax )%traj.nx
        xlim = ax.get_xlim()
        if xlim[0] <= 0 :
            x[x >= xlim[1]] -= traj.nx
        if xlim[1] >=  traj.nx:
            x[x <= xlim[0]] += traj.nx
    if galilean[1] != 0 :
        y = ( y - galilean[1]*j*timestep/traj.deltax )%traj.ny
        ylim = ax.get_ylim()
        if ylim[0] <= 0 :
            y[y >= ylim[1]] -= traj.ny
        if ylim[1] >=  traj.ny:
            y[y <= ylim[0]] += traj.ny
    return x, y

def box_xyz(b):
    x = np.array([b[0,0],b[0,0],b[1,0],b[1,0],b[0,0], \
                  b[0,0],b[0,0],b[1,0],b[1,0],b[0,0]])
    y = np.array([b[0,1],b[1,1],b[1,1],b[0,1],b[0,1], \
                  b[0,1],b[1,1],b[1,1],b[0,1],b[0,1]])
    z = np.array([b[0,2],b[0,2],b[0,2],b[0,2],b[0,2], \
                  b[1,2],b[1,2],b[1,2],b[1,2],b[1,2]])
    return x, y, z


def plot_traj_animation(traj, save_anim=False, legend = False, select = None, \
                        galilean = None, plot_field = False, \
                        dir_override = None, \
                        title = None, \
                        plot_class = None, \
                        no_cloud_size = 0.2, cloud_size = 2.0, \
                        field_size = 0.5, fps = 10, with_boxes = False, \
                        version = 1) :

    ntraj = traj.ntimes
    nobj = traj.nobjects
    if select is None : select = np.arange(0, nobj)
    if version == 1 :
        class_key = list([\
            ["Not set", "0.3"] , \
            ["PRE_CLOUD_IN_BL","r"], \
            ["PRE_CLOUD_ABOVE_BL","g"], \
            ["IN_CLOUD_AT_START","b"], \
            ["FIRST_CLOUD","k"], \
            ["NEW_CLOUD_FROM_BOTTOM","c"], \
            ["NEW_CLOUD_FROM_SIDE","m"], \
            ["DETR_CLOUD","y"], \
            ])
    elif version == 2 :
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
        if plot_field :
            if dir_override is None :
                dataset = Dataset(traj.files[j])
            else :
                filename = os.path.basename(traj.files[j]).split('\\')[-1] 
#                print(filename)
                dataset = Dataset(dir_override+filename)
            qcl_field = dataset.variables["q_cloud_liquid_mass"]
            in_cl = (qcl_field[0,...] > traj.thresh)
            x = xg[in_cl]
            y = yg[in_cl]
            z = zg[in_cl]
            
            if galilean is not None :
                x, y = gal_trans(x, y,  galilean, j, timestep, traj, ax)                        
            
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
                    x, y = gal_trans(x, y,  galilean, j, timestep, traj, ax)                        
                        
                if plot_class is None : 
                    qcl = traj.data[j,traj.labels == iobj, \
                                    traj.var("q_cloud_liquid_mass")]
                    in_cl = (qcl > traj.thresh) 
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
                    b = traj.cloud_box[j,iobj,:,:]
                    x, y, z = box_xyz(b)
                    if galilean is not None :
                        x, y = gal_trans(x, y, galilean, j, timestep, traj, ax)                        

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
    if save_anim : anim.save('traj_anim.mp4', fps=fps)#, extra_args=['-vcodec', 'libx264'])
    plt.show()
    return

def plot_traj_family_members(traj_family,selection_list, galilean = None, \
                             with_boxes = False, asint = True, \
                             no_cloud_size = 0.2, cloud_size = 2.0) :
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
            x, y = gal_trans(x, y,  galilean, abs_time, timestep, tr, ax)  

#        print(np.min(x),np.max(x))                      
#        print(np.min(y),np.max(y))                      
#        print(np.min(z),np.max(z))                      

        if asint :
            x = (x + 0.5).astype(int)
            y = (y + 0.5).astype(int)
            z = (z + 0.5).astype(int)
        qcl = tr.data[tr_time, osel, tr.var("q_cloud_liquid_mass")]
        in_cl = (qcl > tr.thresh) 
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
            b = tr.cloud_box[tr_time,iobj,:,:]
            x, y, z = box_xyz(b)
            if galilean is not None :
                x, y = gal_trans(x, y, galilean, abs_time, timestep, tr, ax)                        

            box, = ax.plot(x,y,color = line.get_color())
            box.set_3d_properties(z)
    plt.legend()
    plt.show()
    return

def plot_traj_family_animation(traj_family, match_index, \
                        overlap_thresh = 0.02, \
                        save_anim=False, legend = False, \
                        title = None, \
                        select = None,  super_obj = None, \
                        galilean = None, plot_field = False,
                        dir_override = None, \
                        no_cloud_size = 0.2, cloud_size = 2.0, \
                        field_size = 0.5, fps = 10, with_boxes = False) :
    
    traj = traj_family.family[-1]
    ref = len(traj_family.family) - 1
    nobj = traj.nobjects
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
                           traj.cloud_box[:,iobj,...]) )
    
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
                                       match_traj.cloud_box \
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
                                       match_traj.cloud_box \
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
                               label='{}'.format(iobj))
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
    
    def set_line_data(tr, it, t_off, ln) :
#        print("Setting line data")
#        print(tr,it,ln,ln_cl)
        tr_time = it + t_off
        if (tr_time >= 0 ) & (tr_time < np.shape(tr[0])[0]) :
            x = tr[0][tr_time,:,0]
            y = tr[0][tr_time,:,1]
            z = tr[0][tr_time,:,2]
            if galilean is not None :
                x, y = gal_trans(x, y,  galilean, it, timestep, traj, ax)                        

            qcl = tr[1][tr_time, :, traj.var("q_cloud_liquid_mass")]
            in_cl = (qcl > traj.thresh) 
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
        if (tr_time >= 0 ) & (tr_time < np.shape(tr[0])[0]) :
            b = tr[2][tr_time,:,:]
            x, y, z = box_xyz(b)
            if galilean is not None :
                x, y = gal_trans(x, y,  galilean, it, timestep, traj, ax)                        

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
                filename = match_traj.files[j]
            else :                
                filename = match_traj.files[i]
            if dir_override is not None :
                filename = os.path.basename(filename).split('\\')[-1] 
#                print(filename)
                filename = dir_override + filename
            dataset = Dataset(filename)    
            qcl_field = dataset.variables["q_cloud_liquid_mass"]
            in_cl = (qcl_field[0,...] > traj.thresh)
            x = xg[in_cl]
            y = yg[in_cl]
            z = zg[in_cl]
               
            if galilean is not None :
                x, y = gal_trans(x, y,  galilean, j, timestep, traj, ax)                        
            
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

            set_line_data(traj_list[nplt], j, 0, line_list[nplt])
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
                                      match_line_list)
                               
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
    
                    set_line_data(m_traj, j, match_index, match_line_list)
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
    if save_anim : anim.save('traj_anim.mp4', fps=fps)#, extra_args=['-vcodec', 'libx264'])
    plt.show()
    return

