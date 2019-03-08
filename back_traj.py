import glob
import os
from netCDF4 import Dataset
#from scipy.io import netcdf
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
#from datetime import datetime, timedelta
#from netCDF4 import num2date, date2num
from sklearn.cluster import KMeans

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from scipy.optimize import minimize

L_vap = 2.501E6
Cp = 1005.0
L_over_cp = L_vap / Cp
grav = 9.81

class trajectory_family : 
    def __init__(self, files, ref_prof_file, \
                 first_ref_file, last_ref_file, \
                 back_len, forward_len, \
                 deltax, deltay, deltaz, variable_list=None, \
                 thresh=1.0E-5) : 
        self.family = list([])
        for ref in range(first_ref_file, last_ref_file+1):
            start_file = ref - back_len
            end_file = ref + forward_len            
            traj = trajectories(files, ref_prof_file, start_file, ref, end_file, \
                        deltax, deltay, deltaz, thresh=thresh, \
                        variable_list=variable_list) 
            self.family.append(traj)
#            input("Press a key")
        return
    
    def matching_object_list(self, ref = None ):
        mol = list([])
        if ref == None : ref = len(self.family) - 1
        traj = self.family[ref]
        for t_off in range(0, ref) :
#            print("Matching at reference time offset {}".format(t_off+1))
            match_traj = self.family[ref-(t_off+1)]
            matching_objects = list([])
            for it_back in range(0,traj.ref_file+1) :
#                print("Matching at reference trajectory time {}".format(it_back))
                matching_object_at_time = list([])
                for iobj in range(0, traj.nobjects) :
                    corr_box = np.array([],dtype=int)
                    if traj.num_cloud[traj.ref_file-it_back ,iobj] > 0 :
                        b_test = traj.cloud_box[traj.ref_file-it_back, iobj,...]
                        match_time = match_traj.ref_file + t_off - it_back +1
#                        if iobj == 0 : print("Matching time {}".format(match_time))
                        if (match_time >= 0) & \
                          (match_time < np.shape(match_traj.cloud_box)[0]) :
                            b_set  = match_traj.cloud_box[match_time,...]
                            corr_box = box_overlap_with_wrap(b_test, b_set, \
                                traj.nx, traj.ny)
                            valid = (match_traj.num_cloud[match_time, corr_box]>0)
                            corr_box = corr_box[valid]
#                            if iobj == 0 :
#                                print(iobj,b_test, corr_box, b_set[corr_box,...])
#                                input("Press enter")
                    matching_object_at_time.append(corr_box)
                matching_objects.append(matching_object_at_time)
            mol.append(matching_objects)
        return mol
   
    def print_matching_object_list(self, ref = None) :    
        if ref == None : ref = len(self.family) - 1
        mol = self.matching_object_list(ref = ref)
        for t_off in range(0, len(mol)) :
            matching_objects = mol[t_off]
            for iobj in range(0, len(matching_objects[0])) :
                for it_back in range(0, len(matching_objects)) :
                    for obj in matching_objects[it_back][iobj] :
                        if np.size(obj) > 0 :
                            print("t_off: {0} iobj: {1} it_back: {2} obj: {3}".\
                                  format(t_off+1, iobj, it_back, obj))
        return
    
    def matching_object_list_summary(self, ref = None) :
        if ref == None : ref = len(self.family) - 1
        mol = self.matching_object_list(ref = ref)
        mols = list([])
        for t_off in range(0, len(mol)) :
            matching_objects = mol[t_off]
            match_list = list([])
            for iobj in range(0, len(matching_objects[0])) :
                objlist = list([])
                otypelist = list([])
                for it_back in range(0, len(matching_objects)) :
                    for obj in matching_objects[it_back][iobj] :
                        if np.size(obj) > 0 :
                            if obj not in objlist : 
                                otype = 'Same'
#                                print(self.family)
#                                print(ref,t_off,ref-(t_off+1))
#                                print(self.family[ref-(t_off+1)].max_at_ref)
                                if (ref-(t_off+1)) >= 0 & \
                                   (ref-(t_off+1)) < len(self.family) :
                                    if obj in \
                                      self.family[ref-(t_off+1)].max_at_ref :
                                        otype = 'Linked'
                                objlist.append(obj)
                                otypelist.append(otype)
                match_list.append(list(zip(objlist,otypelist)))
            mols.append(match_list)
        return mols
    
    def print_matching_object_list_summary(self, ref = None) :    
        if ref == None : ref = len(self.family) - 1
        mols = self.matching_object_list_summary(ref = ref)
        for t_off in range(0, len(mols)) :
            matching_objects = mols[t_off]
            for iobj in range(0, len(matching_objects)) :
                rep =""
                for mo, mot in matching_objects[iobj] : 
                    rep += "({}, {})".format(mo,mot)
                print("t_off: {0} iobj: {1} obj: {2} ".\
                                  format(t_off, iobj, rep))
                           
        return

    def find_linked_objects(self, ref = None) :   
        if ref == None : ref = len(self.family) - 1
        mols = self.matching_object_list_summary(ref = ref)
        linked_objects = list([])
        for iobj in self.family[ref].max_at_ref :
            linked_obj = list([])
            for t_off in range(0, len(mols)) :
                matching_objects = mols[t_off][iobj]
                for mo, mot in matching_objects : 
                    if mot == 'Linked' :
                        linked_obj.append([t_off, mo])
            linked_objects.append(linked_obj)
        return linked_objects
    
    def print_linked_objects(self, ref = None) :
        if ref == None : ref = len(self.family) - 1
        linked_objects = self.find_linked_objects(ref = ref)
        for iobj, linked_obj in zip(self.family[ref].max_at_ref, \
                                    linked_objects) :
            rep =""
            for t_off, mo in linked_obj : 
                rep += "({}, {})".format(t_off, mo)
            print("iobj: {0} objects: {1} ".format(iobj, rep))
            
    def __str__(self):
        rep = "Trajectories family\n"
        for tr in self.family :
            rep += tr.__str__()
        return rep
           
    def __repr__(self):
        rep = "Trajectories family\n"
        for tr in self.family :
            rep += tr.__repr__()
        return rep

class trajectories :

    def __init__(self, files, ref_prof_file, start_file_no, ref_file_no, \
                 end_file_no, deltax, deltay, deltaz, variable_list=None, \
                 thresh=1.0E-5) : 
        
        if variable_list == None : 
            variable_list = { \
                  "u":r"$u$ m s$^{-1}$", \
                  "v":r"$v$ m s$^{-1}$", \
                  "w":r"$w$ m s$^{-1}$", \
                  "th":r"$\theta$ K", \
                  "q_vapour":r"$q_{v}$ kg/kg", \
                  "q_cloud_liquid_mass":r"$q_{cl}$ kg/kg", \
                  }
                  
        dataset_ref = Dataset(ref_prof_file)

        self.rhoref = dataset_ref.variables['rhon'][-1,...]
        self.pref = dataset_ref.variables['prefn'][-1,...]
        self.thref = dataset_ref.variables['thref'][-1,...]
        self.piref = (self.pref[:]/1.0E5)**(287.058/1005.0)

        self.data, trajectory, self.traj_error, self.times, self.labels, \
        self.nobjects, self.xcoord, self.ycoord, self.zcoord, self.thresh = \
        compute_trajectories(files, start_file_no, ref_file_no, end_file_no, \
                             variable_list.keys(), self.thref, thresh=thresh) 
         
        self.files = files[start_file_no:end_file_no+1]
        self.ref_file = ref_file_no-start_file_no
        self.end_file = end_file_no-start_file_no
        self.ntimes = np.shape(trajectory)[0]
        self.npoints = np.shape(trajectory)[1]
        self.deltax = deltax
        self.deltay = deltay
        self.deltaz = deltaz
        self.nx = np.size(self.xcoord)
        self.ny = np.size(self.ycoord)
        self.nz = np.size(self.zcoord)
        self.variable_list = variable_list
        self.trajectory = unsplit_objects(trajectory, self.labels, \
                                          self.nobjects, self.nx, self.ny)        
        self.data_mean, self.num_cloud, self.centroid, self.bounding_box, \
            self.cloud_box = compute_traj_boxes(self)

        max_qcl = (self.data_mean[:,:,self.var("q_cloud_liquid_mass")] \
         == np.max(self.data_mean[:,:,self.var("q_cloud_liquid_mass")],\
                   axis=0))
        when_max_qcl = np.where(max_qcl)
        self.max_at_ref = when_max_qcl[1][when_max_qcl[0] == self.ref_file]
        return
    
    def select_object(self, iobj) :
        in_object = (self.labels == iobj)
        obj = self.trajectory[:, in_object, ...]
        dat = self.data[:, in_object, ...]
        return obj, dat
    
    def cloud_properties(self, thresh = None) :

        if thresh == None : thresh = self.thresh
        
        PRE_CLOUD_IN_BL = 1
        PRE_CLOUD_ABOVE_BL = 2
        IN_CLOUD_AT_START = 3
        FIRST_CLOUD = 4
        NEW_CLOUD_FROM_BOTTOM = 5
        NEW_CLOUD_FROM_SIDE = 6
        DETR_CLOUD = 7
                
        nvars = np.shape(self.data)[2]
        total_nvars = nvars + 5
        r1 = slice(0,nvars)
        # Moist static energy
        r2 = nvars
        # Position
        r3 = slice(nvars+1,nvars+4)
        # Number of points
        r4 = nvars+4
        
        mean_cloud_prop = np.zeros([self.ntimes, self.nobjects, total_nvars])
        mean_entr_prop = np.zeros([self.ntimes, self.nobjects, total_nvars])
        mean_entr_bot_prop = np.zeros([self.ntimes, self.nobjects, total_nvars])
        mean_detr_prop = np.zeros([self.ntimes, self.nobjects, total_nvars])
        traj_class = np.zeros_like(self.data[:,:,0],dtype=int)
        
        min_cloud_base = np.zeros(self.nobjects)
        first_cloud_base = np.zeros(self.nobjects)
        cloud_top = np.zeros(self.nobjects)
        
        tr_z = (self.trajectory[:,:,2]-0.5)*self.deltaz
        zn = (np.arange(0,np.size(self.piref))-0.5)*self.deltaz
        piref_z = np.interp(tr_z,zn,self.piref)
      
        moist_static_energy = Cp * self.data[:, :, self.var("th")] * piref_z + \
                              grav * tr_z + \
                              L_vap * self.data[:, :, self.var("q_vapour")]
                                      
        def set_props(prop, time, obj, mask, where_mask) :
            prop[time, obj, r1] = np.sum(data[time, mask,:], axis=0)
            prop[time, obj, r2] = np.sum(mse[time, mask], axis=0)             
            prop[time, obj, r3] = np.sum(tr[time, mask,:], axis=0)
            prop[time, obj, r4] = np.size(where_mask)
            return prop
                   
        for iobj in range(0,self.nobjects) :
            if debug_mean : print('Processing object {}'.format(iobj))
            obj_ptrs = (self.labels == iobj)
            where_obj_ptrs = np.where(obj_ptrs)[0]
            tr = self.trajectory[:, obj_ptrs, :]
            data = self.data[:, obj_ptrs, :]
            if debug_mean : print(np.shape(data))
            mse = moist_static_energy[:, obj_ptrs]
            qcl = data[:,:,self.var("q_cloud_liquid_mass")]
            mask = (qcl >= thresh)
     
    # These possible slices are set for ease of maintenance.
            # Original extracted variables
            min_cloud_base[iobj] = (np.min(tr[mask,2])-0.5)*self.deltaz
    #        data_class = traj_class[:,self.labels == iobj]
    
            bl = ((tr[0,:,2]-0.5)*self.deltaz < min_cloud_base[iobj])
            class_set = where_obj_ptrs[np.where(bl)[0]]
            traj_class[:, class_set] = PRE_CLOUD_IN_BL 
            class_set = where_obj_ptrs[np.where(~bl)[0]]
            traj_class[:, class_set] = PRE_CLOUD_ABOVE_BL 
            
            after_cloud_formed = False        
            after_cloud_dissipated = False  
            
            for it in range(self.ntimes) :
                cloud_bottom = self.cloud_box[it, iobj, 0, 2] 
                cloud = mask[it,:]
                where_cloud = np.where(cloud)[0]
                
                if debug_mean : print('it', it, np.shape(where_cloud))
                if debug_mean : print(qcl[it,mask[it,:]])
                if debug_mean : print('Mean cloud properties at this level.')
                
                if np.size(where_cloud) > 0 :
    
    # Compute mean properties of cloudy points at this time.            
                    mean_cloud_prop = set_props(mean_cloud_prop, it, iobj, \
                                            cloud, where_cloud)

                    if debug_mean : print(mean_cloud_prop[it, iobj, :])
    
    # Set traj_class flag for those points in cloud at this time.              
                    class_set = where_obj_ptrs[np.where(cloud)[0]]
                    
#                    print("Shape class_set ",np.shape(class_set))
                    
                    if it == 0 :
    #                    print("Setting in cloud at start for iobj={0} at time it={1}".format(iobj,it))
                        traj_class[it:, class_set] = IN_CLOUD_AT_START
                        first_cloud_base[iobj] = \
                          (np.min(tr[it,cloud,2])-0.5)*self.deltaz
                        if debug_mean : print('Cloud base',first_cloud_base[iobj])
#                        input("Press Enter to continue...")
                    else : # it > 0
    #                    traj_class[it, class_set] = IN_CLOUD 
                
#                        input("Press Enter to continue...")
    # Detect first points to be in cloud.                
                        if not after_cloud_formed :
    #                        print("Setting first cloud for iobj={0} at time it={1}".format(iobj,it))
                            first_cloud_base[iobj] = \
                              (np.min(tr[it,cloud,2])-0.5)*self.deltaz
                            if debug_mean : print('Cloud base',\
                                                  first_cloud_base[iobj])
                            
                            traj_class[it:, class_set] = FIRST_CLOUD
#                            input("Press Enter to continue...")
                        else : # above cloud base
    
    # Find new cloudy points                    
                            new_cloud = np.logical_and(cloud, \
                                            np.logical_not(mask[it-1,:]) )
    #                    if debug_mean : print(new_cloud)
                            if debug_mean : 
                                print(qcl[it-1,new_cloud],qcl[it,new_cloud])
                                
                            where_newcloud = np.where(new_cloud)[0]
                            if np.size(where_newcloud) > 0 : # Entrainment
                                if debug_mean : print('Entraining air')
                                
                                mean_entr_prop = set_props(mean_entr_prop, \
                                            it, iobj, new_cloud, \
                                            where_newcloud)
                                
                                class_set = where_obj_ptrs[where_newcloud] 
#                                print("Shape class_set ", np.shape(class_set))
                                from_bl = (traj_class[it, class_set] == \
                                           PRE_CLOUD_IN_BL)
#                                print("Shape from_bl",np.shape(from_bl))
                                from_above_bl = (traj_class[it, class_set] == \
                                                 PRE_CLOUD_ABOVE_BL)
                                traj_class[it:, class_set[from_bl]] = \
                                    NEW_CLOUD_FROM_BOTTOM 
                                traj_class[it:, class_set[from_above_bl]] = \
                                    NEW_CLOUD_FROM_SIDE
#                                input("Press Enter to continue...") 
                                
    # Find points that have ceased to be cloudy                     
                            detr_cloud =np.logical_and(np.logical_not(cloud), \
                                                   mask[it-1,:])
    #                       if debug_mean : print(detr_cloud)
                            if debug_mean : 
                                print(qcl[it-1,detr_cloud],qcl[it,detr_cloud])
                                
                            where_detrained = np.where(detr_cloud)[0]
                            if np.size(where_detrained) > 0 : # Detrainment
                                if debug_mean : print('Detraining air')
                                
                                mean_detr_prop = set_props(mean_detr_prop, \
                                            it, iobj, detr_cloud, \
                                            where_detrained)
                                  
                                class_set = where_obj_ptrs[where_detrained] 
                                traj_class[it:, class_set] = DETR_CLOUD 
#                            input("Press Enter to continue...")             
                    after_cloud_formed = True
                else : # No cloud at this time
                    if after_cloud_formed and not after_cloud_dissipated :
    # At cloud top.
    #                    print('At cloud top it={}'.format(it))
                        cloud_top[iobj] = (np.max(tr[it-1,mask[it-1,:],2])-0.5) \
                                            *self.deltaz
                        if debug_mean : print('Cloud top',cloud_top[iobj])
                        
                        detr_cloud = mask[it-1,:]
                        
                        where_detrained = np.where(detr_cloud)[0]
    # Should not need to test if any points
    #                    if np.size(where_detrained) > 0 : # Detrainment
                        if debug_mean : print('Detraining remaining air')
                        mean_detr_prop = set_props(mean_detr_prop, \
                                            it, iobj, detr_cloud, \
                                            where_detrained)

                        class_set = where_obj_ptrs[where_detrained] 
    #                    traj_class[it:, class_set] = DETR_CLOUD 
                        after_cloud_dissipated = True
#                        input("Press Enter to continue...")
    
        m = (mean_cloud_prop[:,:,r4]>0)    
        for ii in range(nvars+4) : 
            mean_cloud_prop[:,:,ii][m] = mean_cloud_prop[:,:,ii][m] \
              /mean_cloud_prop[:,:,r4][m]
        m = (mean_entr_prop[:,:,r4]>0)    
        for ii in range(nvars+4) : 
            mean_entr_prop[:,:,ii][m] = mean_entr_prop[:,:,ii][m] \
              /mean_entr_prop[:,:,r4][m]
        m = (mean_detr_prop[:,:,r4]>0)    
        for ii in range(nvars+4) : 
            mean_detr_prop[:,:,ii][m] = mean_detr_prop[:,:,ii][m] \
              /mean_detr_prop[:,:,r4][m]
    
        return mean_cloud_prop, mean_entr_prop, mean_detr_prop, \
          traj_class, first_cloud_base, min_cloud_base, cloud_top 

        
    def var(self, v) :
        for ii, vr in enumerate(list(self.variable_list)) :
            if vr==v : break
        return ii
                      
    def __str__(self):
        rep = "Trajectories centred on {0}\n".format(self.files[self.ref_file])
        rep += "Reference Time : {}\n".format(self.times[self.ref_file])
        rep += "Times : {}\n".format(self.ntimes)
        rep += "Points : {}\n".format(self.npoints)
        rep += "Objects : {}\n".format(self.nobjects)
        return rep
           
    def __repr__(self):
        rep = "Trajectory Reference time: {0}, Times:{1}, Points:{2}, Objects:{3}\n".format(\
          self.times[self.ref_file],self.ntimes,self.npoints, self.nobjects)
        return rep
    
def compute_trajectories(files, start_file, ref_file, end_file, \
                         variable_list, thref, thresh=1.0E-5) :
    
    trajectory, data_val, traj_error, traj_times, labels, nobjects, \
      xcoord, ycoord, zcoord, thresh \
      = back_trajectory_init(files[ref_file], variable_list, thref, \
                             thresh=thresh)
      
    for ifile in range(ref_file-1,start_file-1,-1) :
        print('File {}'.format(ifile))
        trajectory, data_val, traj_error, traj_times = \
          back_trajectory_step(files[ifile], variable_list, trajectory, \
                               data_val, traj_error, traj_times, \
                               xcoord, ycoord, zcoord, thref)
    for ifile in range(ref_file+1,end_file+1) :
        print('File {}'.format(ifile))
        trajectory, data_val, traj_error, traj_times = \
          forward_trajectory_step(files[ifile], variable_list, trajectory, \
                               data_val,  traj_error, traj_times, \
                               xcoord, ycoord, zcoord, thref)
          
    print('data_val: {} {} {}'.format( len(data_val), len(data_val[0]), \
          np.size(data_val[0][0]) ) )
          
    data_val = np.reshape(np.vstack(data_val), \
               (len(data_val), len(data_val[0]), np.size(data_val[0][0]) ) )
    
    print('trajectory: {} {} {}'.format(len(trajectory[1:]), len(trajectory[0]), \
                               np.size(trajectory[0][0])))
    print('traj_error: {} {} {}'.format(len(traj_error[1:]), len(traj_error[0]), \
                               np.size(traj_error[0][0])))
#    print np.shape()

    trajectory = np.reshape(np.vstack(trajectory[1:]), \
               ( len(trajectory[1:]), len(trajectory[0]), \
                 np.size(trajectory[0][0]) ) ) 
    
    traj_error = np.reshape(np.vstack(traj_error[1:]), \
               ( len(traj_error[1:]), len(traj_error[0]), \
                 np.size(traj_error[0][0]) ) ) 
    
    traj_times = np.reshape(np.vstack(traj_times),-1)
    
    return data_val, trajectory, traj_error, traj_times, labels, nobjects, \
      xcoord, ycoord, zcoord, thresh

def file_key(file):
    f1 = file.split('_')[-1]
    f2 = f1.split('.')[0]
    return float(f2)

def whichbox(xvec, x ) :
    '''
    Find the (vector of) indices ix such that, for each x, 
    xvec[ix]<=x<xvec[ix+1]
    '''
    ix = np.searchsorted(xvec, x, side='left')-1    
    ix[ix > (len(xvec)-1)] = len(xvec)-1
    ix[ix < 0] = 0
    return ix 

def tri_lin_interp(data, pos, xcoord, ycoord, zcoord) :
    '''
    xcoord, ycoord and z coord are 1D coordinate vectors.
    x, y, z are arrays of coordinates, each the same length.
    data is a list of 3D arrays [xcoord. ycoord, zcoord]
    return list of 1D array of values of data interpolated to x,y,z.
    '''
    nx = len(xcoord)
    ny = len(ycoord)
    nz = len(zcoord)

    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]
    ix = whichbox(xcoord, x)
    iy = whichbox(ycoord, y)
    iz = whichbox(zcoord, z)
    dx = 1.0
    dy = 1.0
    iz[iz>(nz-2)] -= 1 
    xp = (x-xcoord[ix])/dx
    yp = (y-ycoord[iy])/dy
    zp = (z-zcoord[iz])/(zcoord[iz+1]-zcoord[iz])
    wx = [1.0 - xp, xp]
    wy = [1.0 - yp, yp]
    wz = [1.0 - zp, zp]
    output= list([])
    for l in range(len(data)) :
        output.append(np.zeros_like(x))
    t = 0
    for i in range(0,2):
        for j in range(0,2):
            for k in range(0,2):
                w = wx[i]*wy[j]*wz[k]
                t += w
                for l in range(len(data)) :
                    output[l] = output[l] + data[l][(ix+i)%nx,(iy+j)%ny,iz+k]*w
#                print 'Adding', ix+i,iy+j,iz+k,w,data[ix+i,iy+j,iz+k]
#    print xp, yp, zp
#    print t
    return output

def value_at_reset_time(data, traj_pos, xcoord, ycoord, zcoord):
    if use_bilin :
        output = tri_lin_interp(data, traj_pos, xcoord, ycoord, zcoord )
    else:
        output= list([])
        for l in range(len(data)) :
#            print 'Calling map_coordinates'
#            print np.shape(data[l]), np.shape(traj_pos)
            out = ndimage.map_coordinates(data[l], traj_pos, mode='wrap', \
                                          order=interp_order)
            output.append(out)
    return output
    
def load_traj_step_data(file, variable_list, thref) :
    dataset = Dataset(file)  
    
    if cyclic_xy :
        xr = dataset.variables['CA_xrtraj'][0,...]
        xi = dataset.variables['CA_xitraj'][0,...]

        yr = dataset.variables['CA_yrtraj'][0,...]
        yi = dataset.variables['CA_yitraj'][0,...]
    
        zpos = dataset.variables['CA_ztraj'][0,...]
        data_list = [xr, xi, yr, yi, zpos]      
        
    else :
        xpos = dataset.variables['CA_xtraj'][0,...]
        ypos = dataset.variables['CA_ytraj'][0,...]
        zpos = dataset.variables['CA_ztraj'][0,...]  
        data_list = [xpos, ypos, zpos]
#    print np.min(zpos),np.max(zpos)
        
    for variable in variable_list :
#        print 'Reading ', variable
        data = dataset.variables[variable]
        if variable == "q_cloud_liquid_mass" :
            times  = dataset.variables[data.dimensions[0]]
        data = data[0,...]
        if variable == 'th' :
            data = data+thref[...]

        data_list.append(data)   
        
    return data_list, times  
    
def phase(vr, vi, n) :
    vr = np.asarray(vr)       
    vi = np.asarray(vi)       
    vpos = np.asarray(((np.arctan2(vi,vr))/(2.0*np.pi)) * n )
    vpos[vpos<0] = vpos[vpos<0] + n
    return vpos
    
def back_trajectory_step(file, variable_list, trajectory, data_val, \
                         traj_error, traj_times, xcoord, ycoord, \
                         zcoord, thref):
    data_list, times =load_traj_step_data(file, variable_list, thref)
    print("Processing file {} at time {}".format(file,times[0]))
    traj_pos = trajectory[0]
#    print "traj_pos ", np.shape(traj_pos), traj_pos[:,0:5]
    out = value_at_reset_time(data_list, traj_pos, xcoord, ycoord, zcoord)
#    print np.shape(out)
    if cyclic_xy :
        n_pvar = 5
    else:
        n_pvar = 3
    if cyclic_xy :
        (nx, ny, nz) = np.shape(data_list[0])
        xpos = phase(out[0],out[1],nx)
        ypos = phase(out[2],out[3],ny)     
        traj_pos = np.array([xpos,ypos,out[4]]).T
    else :
        traj_pos = np.array([out[0],out[1],out[2]]).T

    data_val.insert(0, np.vstack(out[n_pvar:]).T)       
    trajectory.insert(0, traj_pos)  
    traj_error.insert(0, np.zeros_like(traj_pos))
    traj_times.insert(0, times[0])

    return trajectory, data_val, traj_error, traj_times
    
def forward_trajectory_step(file, variable_list, trajectory, data_val, \
                            traj_error, traj_times, xcoord, ycoord, \
                            zcoord, thref):    
    if cyclic_xy :
        n_pvar = 5
    else:
        n_pvar = 3
    data_list, times = load_traj_step_data(file, variable_list, thref)
    
    (nx, ny, nz) = np.shape(data_list[0])
    print("Processing file {} at time {}".format(file,times[0]))
    traj_pos = trajectory[-1]
    traj_pos_next_est = 2*trajectory[0]-trajectory[1]
  
#    traj_pos_next_est = 60.0*np.array([1/100., 1/100., 1/40.])[:,None]*traj_pos*data_val[0][:,2]
#    traj_pos_est = trajectory[1]
    
#    print "traj_pos ", np.shape(traj_pos), traj_pos#[:,0:5]
#    print "traj_pos_prev ",np.shape(trajectory[1]), trajectory[1]#[:,0:5]
    def print_info(kk):
        print('kk = {} Error norm :{} Error :{}'.format(kk, mag_diff[kk], \
              diff[:,kk]))
        print('Looking for at this index ', \
              [traj_pos[:,m][kk] for m in range(3)])
        print('Nearest solution for the index', \
              [traj_pos_at_est[kk,m] for m in range(3)])
        print('Located at ', \
              [traj_pos_next_est[kk,m] for m in range(3)])
        return
    
    def int3d_dist(pos) :
#    global dist_data
        if np.size(np.shape(pos)) == 1 :
           pos = np.atleast_2d(pos).T

        out = np.max([0,ndimage.map_coordinates(dist_data, pos, \
                                                mode='nearest', order=2)])
#   out = ndimage.map_coordinates(dist_data, pos, mode='nearest', order=2)
#   print ('out = {}'.format(out))
        return out
    
    err = 1.0
    niter = 0 
    max_iter = 30
    errtol_iter = 1E-3
    errtol = 5E-2
    relax_param = 0.5
    not_converged = True
    correction_cycle = False 
    while not_converged : 
        out = value_at_reset_time(data_list, traj_pos_next_est, \
                                  xcoord, ycoord, zcoord)

        if cyclic_xy :
            xpos = phase(out[0],out[1],nx)
            ypos = phase(out[2],out[3],ny)     
            traj_pos_at_est = np.array([xpos,ypos,out[4]]).T
        else :
            traj_pos_at_est = np.array([out[0],out[1],out[2]]).T
#        print "traj_pos_at_est ", np.shape(traj_pos_new), traj_pos_new#[:,0:5]
        diff = traj_pos_at_est - traj_pos
        
# Deal with wrap around.
        
        traj_pos_at_est[:,0][diff[:,0]<(-nx/2)] += nx
        diff[:,0][diff[:,0]<(-nx/2)] += nx
        traj_pos_at_est[:,0][diff[:,0]>=(nx/2)] -= nx
        diff[:,0][diff[:,0]>=(nx/2)] -= nx
        traj_pos_at_est[:,1][diff[:,1]<-(ny/2)] += ny
        diff[:,1][diff[:,1]<-(ny/2)] += ny
        traj_pos_at_est[:,1][diff[:,1]>=(ny/2)] -= ny
        diff[:,1][diff[:,1]>= (ny/2)] -= ny
        
        mag_diff = 0
        for i in range(3):
            mag_diff += diff[i,:]**2
            
        err = np.max(mag_diff)
        
        if correction_cycle :
            print('After correction cycle {}'.format(err))
            if debug :
                try: 
                    k
                    for kk in k :
                        print_info(kk)

                except NameError:
                    print('No k')
            break
        
        if err <= errtol_iter :
            not_converged = False
            print(niter, err)

           
        if niter <= max_iter :
#            traj_pos_prev_est = traj_pos_next_est
#            traj_pos_at_prev_est = traj_pos_at_est
            traj_pos_next_est = traj_pos_next_est - diff * relax_param
            niter +=1
        else :   
            print('Iterations exceeding {} {}'.format(max_iter, err))
            if err > errtol :
                bigerr = (mag_diff > errtol)    
                if np.size(diff[:,bigerr]) > 0 :
                    k = np.where(bigerr)[0]
                    if debug :
                        print('Index list into traj_pos_at_est is {}'.format(k))
                    for kk in k :
                        if debug :
                            print_info(kk)            
                        lx = np.int(np.round(traj_pos_next_est[kk,0]))
                        ly = np.int(np.round(traj_pos_next_est[kk,1]))
                        lz = np.int(np.round(traj_pos_next_est[kk,2]))
                        if debug :
                            print('Index in source array is {},{},{}'.\
                                  format(lx,ly,lz))
                        
                        nd = 5
                        
                        xr = np.arange(lx-nd,lx+nd+1, dtype=np.intp) % nx
                        yr = np.arange(ly-nd,ly+nd+1, dtype=np.intp) % ny
                        zr = np.arange(lz-nd,lz+nd+1, dtype=np.intp)
                        
                        zr[zr >= nz] = nz-1
                        zr[zr <  0 ] = 0
                        
                        x_real = data_list[0]
                        x_imag = data_list[1]
                        y_real = data_list[2]
                        y_imag = data_list[3]
                        z_dat   = data_list[4]
        
                        xpos = phase(x_real[np.ix_(xr,yr,zr)],\
                                     x_imag[np.ix_(xr,yr,zr)],nx)
                        ypos = phase(y_real[np.ix_(xr,yr,zr)],\
                                     y_imag[np.ix_(xr,yr,zr)],ny)
                        zpos = z_dat[np.ix_(xr,yr,zr)]
                        
                        dx = xpos - traj_pos[:,0][kk]
                        dy = ypos - traj_pos[:,1][kk]
                        dz = zpos - traj_pos[:,2][kk]
                        dx[dx >= (nx/2)] -= nx
                        dx[dx < (-nx/2)] += nx
                        dy[dy >= (ny/2)] -= ny
                        dy[dy < (-ny/2)] += ny
                                               
                        dist = dx**2 + dy**2 +dz**2
                                                        
                        lp = np.where(dist == np.min(dist))
                        
                        lxm = lp[0][0]
                        lym = lp[1][0]
                        lzm = lp[2][0]

                        if debug :
                            print('Nearest point is ({},{},{},{})'.\
                                  format(lxm,lym,lzm, dist[lxm,lym,lzm]))
                            print(xpos[lxm,lym,lzm],ypos[lxm,lym,lzm],zpos[lxm,lym,lzm])
#                            ix = (lx+lxm-nd)%nx
#                            iy = (ly+lym-nd)%ny
#                            print(phase(x_real[ix, iy, lz+lzm-nd],\
#                                        x_imag[ix, iy, lz+lzm-nd],nx))
#                            print(phase(y_real[ix, iy, lz+lzm-nd],\
#                                        y_imag[ix, iy, lz+lzm-nd],ny))
#                            print(z_dat[ix, iy, lz+lzm-nd]) 
                            
#                            ndimage.map_coordinates(data[l], traj_pos, mode='wrap', \
#                                          order=interp_order)
                        envsize = 2
                        nx1=lxm-envsize
                        nx2=lxm+envsize+1
                        ny1=lym-envsize
                        ny2=lym+envsize+1
                        nz1=lzm-envsize
                        nz2=lzm+envsize+1
                        
                        dist_data = dist[nx1:nx2,ny1:ny2,nz1:nz2]
                        x0 = np.array([envsize,envsize,envsize],dtype='float')
                        res = minimize(int3d_dist, x0, method='BFGS')
                        #,options={'xtol': 1e-8, 'disp': True})
                        if debug :
                            print('New estimate, relative = {} {} '.\
                                  format(res.x, res.fun))
                        newx = lx-nd+lxm-envsize+res.x[0]
                        newy = ly-nd+lym-envsize+res.x[1]
                        newz = lz-nd+lzm-envsize+res.x[2]
                        if debug :
                            print('New estimate, absolute = {}'.\
                                  format(np.array([newx,newy,newz])))                
                        traj_pos_next_est[kk,:] = np.array([newx,newy,newz])
                    # end kk loop
                #  correction 
            correction_cycle = True

        traj_pos_next_est[:,0][ traj_pos_next_est[:,0] <   0 ] += nx
        traj_pos_next_est[:,0][ traj_pos_next_est[:,0] >= nx ] -= nx
        traj_pos_next_est[:,1][ traj_pos_next_est[:,1] <   0 ] += ny
        traj_pos_next_est[:,1][ traj_pos_next_est[:,1] >= ny ] -= ny
        traj_pos_next_est[:,2][ traj_pos_next_est[:,2] <   0 ]  = 0
        traj_pos_next_est[:,2][ traj_pos_next_est[:,2] >= nz ]  = nz
        
    data_val.append(np.vstack(out[n_pvar:]).T)       
    trajectory.append(traj_pos_next_est) 
    traj_error.append(diff)
    traj_times.append(times[0])
#    print 'trajectory:',len(trajectory[:-1]), len(trajectory[0]), np.size(trajectory[0][0])
#    print 'traj_error:',len(traj_error[:-1]), len(traj_error[0]), np.size(traj_error[0][0])
    return trajectory, data_val, traj_error, traj_times
       
def label_3D_cyclic(mask) :
#    print np.shape(mask)        
    (nx, ny, nz) = np.shape(mask)
    labels, nobjects = ndimage.label(mask)
    labels -=1
#    print 'labels', np.shape(labels)     
    def relabel(labs, nobjs, i,j) :
#        if debug_label : 
#            print('Setting label {:3d} to {:3d}'.format(j,i)) 
        lj = (labs == j)
        labs[lj] = i
        for k in range(j+1,nobjs) :
            lk = (labs == k)
            labs[lk] = k-1
#            if debug_label : 
#                print('Setting label {:3d} to {:3d}'.format(k,k-1)) 
        nobjs -= 1
#        if debug_label : print('nobjects = {:d}'.format(nobjects))
        return labs, nobjs
    
    def find_objects_at_edge(minflag, x_or_y, n, labs, nobjs) :
#        for minflag in (True,False) :
        i = 0
        while i < (nobjs-2) :
            posi = np.where(labs == i)
#        print(np.shape(posi))
            if minflag :
                test1 = (np.min(posi[x_or_y][:]) == 0)
                border = '0'
            else:
                test1 = (np.max(posi[x_or_y][:]) == (n-1))
                border = 'n{}-1'.format(['x','y'][x_or_y])
            if test1 :
                if debug_label : 
                    print('Object {:03d} on {}={} border?'.\
                          format(i,['x','y'][x_or_y],border))
                j = i+1
                while j < (nobjs-1) :
                    posj = np.where(labs == j)
                    
                    if minflag :
                        test2 = (np.max(posj[x_or_y][:]) == (n-1))
                        border = 'n{}-1'.format(['x','y'][x_or_y])
                    else:
                        test2 = (np.min(posj[x_or_y][:]) == 0)
                        border = '0'
                        
                    if test2 :
                        if debug_label : 
                            print('Match Object {:03d} on {}={} border?'\
                                  .format(j,['x','y'][x_or_y],border))
                            
                        if minflag :
                            ilist = np.where(posi[x_or_y][:] == 0)
                            jlist = np.where(posj[x_or_y][:] == (n-1))
                        else :
                            ilist = np.where(posi[x_or_y][:] == (n-1))
                            jlist = np.where(posj[x_or_y][:] == 0)
                            
                        if np.size( np.intersect1d(posi[1-x_or_y][ilist], \
                                           posj[1-x_or_y][jlist]) ) :
                            if np.size( np.intersect1d(posi[2][ilist], \
                                           posj[2][jlist]) ) :
                            
                                if debug_label : 
                                    print('Yes!',i,j)
#                                    for ii in range(3) : 
#                                        print(ii, posi[ii][posi[x_or_y][:] \
#                                                       == 0])
#                                    for ii in range(3) : 
#                                        print(ii, posj[ii][posj[x_or_y][:] \
#                                                       == (n-1)])
                                labs, nobjs = relabel(labs, nobjs, i, j)
                    j += 1
            i += 1
        return labs, nobjs

    labels, nobjects = find_objects_at_edge(True,  0, nx, labels, nobjects)
    labels, nobjects = find_objects_at_edge(False, 0, nx, labels, nobjects)
    labels, nobjects = find_objects_at_edge(True,  1, ny, labels, nobjects)
    labels, nobjects = find_objects_at_edge(False, 1, ny, labels, nobjects)
       
    return labels, nobjects

def back_trajectory_init(file, variable_list, thref, thresh = 0.00001) :
    data_list, times =load_traj_step_data(file, variable_list, thref)
    print("Starting at file {} at time {}".format(file,times[0]))

#    print data_list
    xcoord = np.arange(np.shape(data_list[-1])[0],dtype='float')
    ycoord = np.arange(np.shape(data_list[-1])[1],dtype='float')
    zcoord = np.arange(np.shape(data_list[-1])[2],dtype='float')
    
    for lv, variable in enumerate(variable_list):
        if variable == "q_cloud_liquid_mass" :
            break
    if cyclic_xy :
        n_pvar = 5
    else:
        n_pvar = 3

    lv = lv+n_pvar

    data = data_list[lv]
    
#    logical_pos = data[...]>(np.max(data[...])*0.6)
    logical_pos = data[...] >= thresh
    
#    print('q_cl threshold {:10.6f}'.format(np.max(data[...])*0.8))
    
    mask = data.copy()
    mask[...] = 0
    mask[logical_pos] = 1

    print('Setting labels.')
    labels, nobjects = label_3D_cyclic(mask)
    
    labels = labels[logical_pos]
    
    pos = np.where(logical_pos)
    
#    print(np.shape(pos))
    traj_pos = np.array( [xcoord[pos[0][:]], \
                          ycoord[pos[1][:]], \
                          zcoord[pos[2][:]]],ndmin=2 ).T

    if cyclic_xy :
        (nx, ny, nz) = np.shape(data_list[0])
        xpos = phase(data_list[0][logical_pos],data_list[1][logical_pos],nx)
        ypos = phase(data_list[2][logical_pos],data_list[3][logical_pos],ny)
        traj_pos_new = np.array( [xpos, \
                                  ypos, \
                                  data_list[4][logical_pos]]).T
    else :
        traj_pos_new = np.array( [data_list[0][logical_pos], \
                                  data_list[1][logical_pos], \
                                  data_list[2][logical_pos]]).T
    data_val = list([])
    for data in data_list[n_pvar:]:
        data_val.append(data[logical_pos])
    data_val=[np.vstack(data_val).T]
    if debug :
        print('Value of {} at trajectory position.'.format(variable))
        print(np.shape(data_val))
        print(np.shape(traj_pos))
        print('xorg',traj_pos[:,0])
        print('yorg',traj_pos[:,1])
        print('zorg',traj_pos[:,2])
        print('x',traj_pos_new[:,0])
        print('y',traj_pos_new[:,1])
        print('z',traj_pos_new[:,2])
    trajectory = list([traj_pos])
    traj_error = list([np.zeros_like(traj_pos)])
    trajectory.insert(0,traj_pos_new)
    traj_error.insert(0,np.zeros_like(traj_pos_new))
    traj_times = list([times[0]])
#    print 'init'
#    print 'trajectory:',len(trajectory[:-1]), len(trajectory[0]), np.size(trajectory[0][0])
#    print 'traj_error:',len(traj_error[:-1]), len(traj_error[0]), np.size(traj_error[0][0])
    
    return trajectory, data_val, traj_error, traj_times, labels, nobjects, \
      xcoord, ycoord, zcoord, thresh

def unsplit_object( pos, nx, ny ) :
    global debug_unsplit
    if debug_unsplit : print('pos:', pos)
    n_clust = np.min([4,np.shape(pos)[0]])
    if debug_unsplit : print('Shape(pos):',np.shape(pos), \
                             'Number of clutsters:', n_clust)
    kmeans = KMeans(n_clusters=n_clust)
#    print(kmeans)
    kmeans.fit(pos)
#    print(kmeans)
    
    if debug_unsplit : print('Shape(cluster centres):', \
                            np.shape(kmeans.cluster_centers_))
    if debug_unsplit : print('Cluster centres: ',kmeans.cluster_centers_)
    counts = np.zeros(n_clust,dtype=int)
    for i in range(n_clust):
        counts[i] = np.count_nonzero(kmeans.labels_ == i)
    if debug_unsplit : print(counts)
    main_cluster = np.where(counts == np.max(counts))[0]
    def debug_print(j) :
        print('Main cluster:', main_cluster, 'cluster number: ', i, \
              'dist:', dist)
        print('Cluster centres:', kmeans.cluster_centers_)           
        print('Changing ', pos[kmeans.labels_ == i,j])
        return
    
    for i in range(n_clust):     
        dist = kmeans.cluster_centers_[i] - kmeans.cluster_centers_[main_cluster]
#        print 'dist', dist
        dist = dist[0]
        if (dist[0] < -nx/2) :
            if debug_unsplit : debug_print(0)
            pos[kmeans.labels_ == i,0] = pos[kmeans.labels_ == i,0] + nx
        if (dist[0] >  nx/2) :
            if debug_unsplit : debug_print(0)
            pos[kmeans.labels_ == i,0] = pos[kmeans.labels_ == i,0] - nx
        if (dist[1] < -ny/2) :
            if debug_unsplit : debug_print(1)
            pos[kmeans.labels_ == i,1] = pos[kmeans.labels_ == i,1] + ny
        if (dist[1] >  ny/2) :
            if debug_unsplit : debug_print(1)
            pos[kmeans.labels_ == i,1] = pos[kmeans.labels_ == i,1] - ny
    
    return pos
    
def unsplit_objects(trajectory, labels, nobjects, nx, ny) :
    global debug_unsplit
#    print np.shape(trajectory)
    print('Unsplitting Objects:')

    for iobj in range(0,nobjects):
        if debug_unsplit : print('Unsplitting Object: {:03d}'.format(iobj))
#        if iobj == 15 : 
#            debug_unsplit = True
#        else :
#            debug_unsplit = False

        for it in range(0,np.shape(trajectory)[0]) :
            if debug_unsplit : print('Time: {:03d}'.format(it))
            tr = trajectory[it,labels == (iobj),:]
            if ((np.max(tr[:,0])-np.min(tr[:,0])) > nx/2 ) or \
               ((np.max(tr[:,1])-np.min(tr[:,1])) > ny/2 ) :
                trajectory[it, labels == iobj,:] = \
                unsplit_object(trajectory[it,labels == iobj,:], \
                                               nx, ny)
                if debug_unsplit : print('New object:',\
                    trajectory[it,labels == iobj,:])
    return trajectory
    
def compute_traj_boxes(traj) :
#    print('Boxes', traj)
#    print(np.shape(traj.trajectory))
#    print(traj.nobjects)
    data_mean = np.zeros((np.shape(traj.data)[0], \
                          traj.nobjects, np.shape(traj.data)[2]))
    num_cloud = np.zeros((np.shape(traj.data)[0], \
                          traj.nobjects),dtype=int)
    traj_centroid = np.zeros((np.shape(traj.trajectory)[0], traj.nobjects, \
                              np.shape(traj.trajectory)[2]))
    traj_box = np.zeros((np.shape(traj.trajectory)[0], traj.nobjects, \
                              2, np.shape(traj.trajectory)[2]))
    cloud_box = np.zeros((np.shape(traj.trajectory)[0], traj.nobjects, \
                              2, np.shape(traj.trajectory)[2]))
    for iobj in range(traj.nobjects):
        data = traj.data[:,traj.labels == iobj,:]
        qcl = data[...,traj.var("q_cloud_liquid_mass")]
#        print(np.shape(qcl))
        data_mean[:,iobj,:] = np.mean(data, axis=1) 
        obj = traj.trajectory[:, traj.labels == iobj, :]
#        print(np.shape(obj))
        traj_centroid[:,iobj, :] = np.mean(obj,axis=1) 
        traj_box[:,iobj, 0, :] = np.amin(obj, axis=1)
        traj_box[:,iobj, 1, :] = np.amax(obj,axis=1)
        for it in np.arange(0,np.shape(obj)[0]) : 
            mask = (qcl[it,:] >= traj.thresh)
#            print(np.shape(mask))
            num_cloud[it, iobj] = np.size(np.where(mask))
            if num_cloud[it, iobj] > 0 :
                cloud_box[it, iobj, 0, :] = np.amin(obj[it, mask, :], axis=0)
                cloud_box[it, iobj, 1, :] = np.amax(obj[it, mask, :], axis=0)
    return data_mean, num_cloud, traj_centroid, traj_box, cloud_box
   
    
def print_boxes(traj) : 
    varns = list(traj.variable_list)
    for iobj in range(traj.nobjects) :
        print('Object {:03d}'.format(iobj))
        print('  Time    x      y      z   '+\
              ''.join([' {:^10s}'.format(varns[i][:9]) for i in range(6)]))
        for it in range(np.shape(traj.centroid)[0]) :
            strf = '{:6.0f}'.format(traj.times[it]/60)
            strf = strf + ''.join([' {:6.2f}'.\
                    format(traj.centroid[it,iobj,i]) for i in range(3)]) 
            strf = strf + ''.join([' {:10.6f}'.\
                    format(traj.data_mean[it,iobj,i]) for i in range(6)])
            print(strf)
    return 

def box_overlap_with_wrap(b_test, b_set, nx, ny) :
    x_overlap = np.logical_or( np.logical_and( \
        b_test[0,0] >= b_set[...,0,0] , b_test[0,0] <= b_set[...,1,0]), \
                               np.logical_and( \
        b_test[1,0] >= b_set[...,0,0] , b_test[1,0] <= b_set[...,1,0]) ) 
    x_ind = np.where(x_overlap)[0]
    
    y_overlap = np.logical_or( np.logical_and( \
        b_test[0,1] >= b_set[x_ind,0,1] , b_test[0,1] <= b_set[x_ind,1,1]), \
                               np.logical_and( \
        b_test[1,1] >= b_set[x_ind,0,1] , b_test[1,1] <= b_set[x_ind,1,1]) )
    
    y_ind = np.where(y_overlap)[0]
    
    return x_ind[y_ind]
    

def plot_trajectory_history(tr, select_obj) :
    
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
        
def plot_trajectory_mean_history(tr, mean_cloud_properties, \
                                 mean_entr_properties, mean_detr_properties, \
                                 traj_class, cloud_base, cloud_top, \
                                 select = None, obj_per_plt = 10) :  
    nvars = np.shape(tr.data)[2]
    nobj = np.shape(mean_cloud_properties)[1]
    if select is None : select = np.arange(0,nobj)    
    zn = (np.arange(0,np.size(tr.piref))-0.5)*tr.deltaz
    new_fig = True
    obj_plotted = 0
    iobj = 0
    figs = 0
    while obj_plotted < np.size(select) :
        if new_fig :
            fig1, axa = plt.subplots(3, 2, figsize=(8,10), sharey=True)
                        
            for j,v in enumerate(["w","th","q_vapour","q_cloud_liquid_mass"]):
    
                ax = axa[(j)%2,(j)//2]
                ax.set_xlabel(tr.variable_list[v],fontsize=16)
                ax.set_ylabel(r"$z$ m",fontsize=16)
                ax.set_ylim(0,np.ceil(np.max(cloud_top)/100)*100)
                
            ax = axa[2,0]
            ax.set_xlabel(r"$\theta_L$ K",fontsize=16)
            ax.set_ylabel(r"$z$ m",fontsize=16)
            ax.set_ylim(0,np.ceil(np.max(cloud_top)/100)*100)

            ax = axa[2,1]
            ax.set_xlabel(r"$q_t$ kg/kg",fontsize=16)
            ax.set_ylabel(r"$z$ m",fontsize=16)
            ax.set_ylim(0,np.ceil(np.max(cloud_top)/100)*100)
            
            fig2, axb = plt.subplots(2, 2, figsize=(8,10), sharey=True)
            
            ax = axb[0,0]
            ax.set_xlabel(r"Volume km$^3$",fontsize=16)
            ax.set_ylabel(r"$z$ m",fontsize=16)
            ax.set_ylim(0,np.ceil(np.max(cloud_top)/100)*100)

            ax = axb[0,1]
            ax.set_xlabel(r"Entrainment rate s$^{-1}$",fontsize=16)
            ax.set_ylabel(r"$z$ m",fontsize=16)
            ax.set_ylim(0,np.ceil(np.max(cloud_top)/100)*100)

            ax = axb[1,0]
            ax.set_xlabel(r"Moist static energy kJ kg$^{-1}$",fontsize=16)
            ax.set_ylabel(r"$z$ m",fontsize=16)
            ax.set_ylim(0,np.ceil(np.max(cloud_top)/100)*100)
            
            ax = axb[1,1]
            ax.set_xlabel(r"Moist static energy change kJ kg$^{-1}$",\
                          fontsize=16)
            ax.set_ylabel(r"$z$ m",fontsize=16)
            ax.set_ylim(0,np.ceil(np.max(cloud_top)/100)*100)

            new_fig = False
            figs +=1
            
        volume = tr.deltax*tr.deltay*tr.deltaz
            
        if np.isin(iobj,select) :
            m = (mean_cloud_properties[:,iobj,nvars+4] >0)
            z = (mean_cloud_properties[:,iobj,nvars+3][m]-0.5)*tr.deltaz
            
            for j,v in enumerate(["w","th","q_vapour","q_cloud_liquid_mass"]):    
                ax = axa[(j)%2,(j)//2]
                ax.plot(mean_cloud_properties[:,iobj,tr.var(v)][m], z)

            ax = axa[2,0]
            piref_z = np.interp(z,zn,tr.piref)
            thl = mean_cloud_properties[:,iobj,tr.var("th")][m] - \
              L_over_cp * \
              mean_cloud_properties[:,iobj,tr.var("q_cloud_liquid_mass")][m] \
              / piref_z
    #        print thl, data[:,var("th"),i],data[:,var("q_vapour"),i]
            ax.plot(thl,z)
            ax = axa[2,1]
            qt = mean_cloud_properties[:,iobj,tr.var("q_vapour")][m] + \
                 mean_cloud_properties[:,iobj,tr.var("q_cloud_liquid_mass")][m]
    #        print qt,data[:,var("q_vapour"),i],data[:,var("q_cloud_liquid_mass"),i]
            ax.plot( qt,z, label='{}'.format(iobj))

            ax = axb[0,0]
            mass = mean_cloud_properties[:,iobj,nvars+4][m]*volume/1E9
            ax.plot(mass, z, label='{}'.format(iobj))
            
            ax = axb[0,1]
            m1 = np.logical_and((mean_cloud_properties[1:,iobj,nvars+4] > 0) , \
                                (mean_entr_properties[1:,iobj,nvars+4]  > 0 ))
            z1 = (mean_cloud_properties[1:,iobj,nvars+3][m1]-0.5)*tr.deltaz 
            entr_rate = mean_entr_properties[1:,iobj,nvars+4][m1] / \
               (mean_cloud_properties[1:,iobj,nvars+4][m1] + \
                -mean_entr_properties[1:,iobj,nvars+4][m1] / 2.0) / \
                (tr.times[1:][m1]-tr.times[0:-1][m1])
            ax.plot(entr_rate[entr_rate>0], z1[entr_rate>0], \
                         linestyle='' ,marker='.', \
                         label='{}'.format(iobj))

            ax = axb[1,0]
            mse = mean_cloud_properties[:,iobj,nvars][m]/1000.0
            ax.plot(mse, z, label='{}'.format(iobj))

            ax = axb[1,1]
            m1 = (mean_cloud_properties[1:,iobj,nvars+4] >0)
            z1 = (mean_cloud_properties[1:,iobj,nvars+3][m1]-0.5)*tr.deltaz
            
            mse_now  = mean_cloud_properties[1:,iobj,nvars][m1] * \
                       mean_cloud_properties[1:,iobj,nvars+4][m1]
            mse_prev = mean_cloud_properties[0:-1,iobj,nvars][m1] * \
                       mean_cloud_properties[0:-1,iobj,nvars+4][m1]
            mse_entr = mean_entr_properties[1:,iobj,nvars][m1] * \
                       mean_entr_properties[1:,iobj,nvars+4][m1]
            mse_detr = mean_detr_properties[1:,iobj,nvars][m1] * \
                       mean_detr_properties[1:,iobj,nvars+4][m1]           
            mse_loss = mse_now - mse_prev - mse_entr + mse_detr            
            mse_loss = mse_loss / mean_cloud_properties[1:,iobj,nvars+4][m1] \
                       /1000.0
            ax.plot(mse_loss, z1, label='{}'.format(iobj))
            
            obj_plotted +=1
            if ((obj_plotted % obj_per_plt) == 0) or \
               ( obj_plotted == np.size(select) ) :
                new_fig = True

                plt.figure(fig1.number)
                plt.legend()
                plt.tight_layout()
                plt.savefig(fn+\
                            '_Cloud_mean_traj_p1_{:02d}.png'.format(figs))

                plt.figure(fig2.number)
                plt.legend()
                plt.tight_layout()
                plt.savefig(fn+\
                            '_Cloud_mean_traj_p2_{:02d}.png'.format(figs))
    
                plt.show()
                plt.close(fig1)
                plt.close(fig2)
        iobj +=1
    
    return
    
def plot_traj_pos(traj, index, save=False) :
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
                        title = None, \
                        plot_class = None, \
                        no_cloud_size = 0.2, cloud_size = 2.0, \
                        field_size = 0.5, fps = 10, with_boxes = False) :

    ntraj = traj.ntimes
    nobj = traj.nobjects
    if select is None : select = np.arange(0, nobj)
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
            dataset = Dataset(traj.files[j])
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

def plot_traj_family_animation(traj_family, match_index, \
                        save_anim=False, legend = False, \
                        title = None, \
                        select = None, \
                        galilean = None, plot_field = False,
                        no_cloud_size = 0.2, cloud_size = 2.0, \
                        field_size = 0.5, fps = 10, with_boxes = False) :
    
    traj = traj_family.family[-1]
    nobj = traj.nobjects
#    print(traj)
    if match_index >= 0 :
        
        if select is None : select = np.arange(0, nobj, dtype = int)
        match_traj = traj_family.family[-(1+match_index)]
        match_objs = traj_family.matching_object_list_summary()
        plot_linked = False
        max_t = match_index -1
        nframes = traj.ntimes+match_index
        
    else:
        
        ref_obj = traj_family.family[-1].max_at_ref
        if select is None : select = ref_obj
        plot_linked = True
        linked_objs = traj_family.find_linked_objects()
        max_t = 0
        for iobj in range(0,traj.nobjects):
            if np.isin(iobj,ref_obj) :
                mobj_ptr=np.where(ref_obj == iobj)[0][0]
#        for mo in linked_objs :
                for t,o in linked_objs[mobj_ptr] :
#                    print(iobj,t,o)
                    max_t = np.max([max_t,t])
                    nframes = traj.ntimes+max_t+1
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
    for iobj in range(0,traj.nobjects):
        if np.isin(iobj,select) :
            
            traj_list.append((traj.trajectory[:,traj.labels == iobj,...], \
                              traj.data[:,traj.labels == iobj,...], 
                              traj.cloud_box[:,iobj,...]) )
    
            match_list = list([])

            if plot_linked :
                
#                print(ref_obj, iobj)
                
                if np.isin(iobj,ref_obj) :
                    
#                    print(np.where(ref_obj  == iobj))
                    mobj_ptr=np.where(ref_obj == iobj)[0][0]
#                    print(mobj_ptr)
#                    input("Press enter")
        
                    for match_obj in linked_objs[mobj_ptr] :
#                        print("Linked object {}".format(match_obj))
                        match_traj = traj_family.family[-(1+match_obj[0]+1)]
                        mobj = match_obj[1]
                        match_list.append((match_traj.trajectory\
                          [:, match_traj.labels == mobj, ...], \
                                           match_traj.data\
                          [:, match_traj.labels == mobj, ...], \
                                           match_traj.cloud_box \
                          [:, mobj,...]) )
                    
            else :
                
                mob = match_objs[match_index-1][iobj]
#                print(mob)
#                input("Press enter")
#                for match_obj in mob :
#                    print(match_obj)
#                input("Press enter")

                for match_obj in mob :
                    mobj = match_obj[0]
#                    print("Matching object {} {}".format(match_obj, mobj))
                    match_list.append((match_traj.trajectory\
                      [:, match_traj.labels == mobj, ...], \
                                       match_traj.data\
                      [:, match_traj.labels == mobj, ...], \
                                       match_traj.cloud_box \
                      [:, mobj,...]) )
    
            match_traj_list_list.append(match_list)
            
            nplt += 1
            
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
    for iobj in range(0,traj.nobjects):
        if np.isin(iobj,select) :
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
                   
                    for match_obj in linked_objs[mobj_ptr] :
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
                for match_obj in match_objs[match_index-1][iobj] :
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
#    input("Press enter")
    
    # initialization function: plot the background of each frame
    def init() :
        if plot_field :
            line_field.set_data([], [])
        nplt = 0
        for iobj in range(0,traj.nobjects):
            if np.isin(iobj,select) :
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
#        if plot_linked :
        j = i - max_t - 1
#        else :
#           j = i - match_index 
        match_index = max_t + 1
#        input("Press enter")
#        print("Frame {0} {1}".format(i,j))
#        input("Press enter")
        if plot_field :
            if j >= 0 :
                dataset = Dataset(match_traj.files[j])
            else :                
                dataset = Dataset(match_traj.files[i])
                
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
#                print("Setting line data", j, nplt, line_list[nplt])
#                input("Press enter")

                set_line_data(traj_list[nplt], j, 0, line_list[nplt])
#                input("Press enter")
              
                if plot_linked :
                    if np.isin(iobj,ref_obj) :
                        mobj_ptr=np.where(ref_obj == iobj)[0][0]
                        for (match_line_list, m_traj, match_obj ) in \
                            zip(match_line_list_list[nplt], \
                                match_traj_list_list[nplt], \
                                linked_objs[mobj_ptr]) :
#                            input("Press enter")
                            match_index = match_obj[0]+1
                            set_line_data(m_traj, j, match_index, match_line_list)
#                            input("Press enter")
                               
                        if with_boxes :
                            set_box_data(traj_list[nplt], j, 0, box_list[nplt])
                            for (box, m_traj, match_obj) in \
                                zip(match_box_list_list[nplt], \
                                    match_traj_list_list[nplt], \
                                    linked_objs[iobj]) :
        #                        print(box, m_traj)
                                match_index = match_obj[0]+1
                                set_box_data(m_traj, j, match_index, box)
                else :
                    
#                    print(len(match_line_list_list[nplt]), \
#                            len(match_traj_list_list[nplt]))
                    for (match_line_list, m_traj) in \
                        zip(match_line_list_list[nplt], \
                            match_traj_list_list[nplt]) :
#                        print(m_traj)
#                        print("Match line list", match_line_list)
#                        input("Press enter")
    
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

#dir = '/projects/paracon/toweb/r4677_Circle-A/diagnostic_files/'
#dir = '/projects/paracon/toweb/r4677_Circle-A/diagnostic_files/S_ReI_1200_A/'
#dir = '/projects/paracon/toweb/r4677_Circle-A/diagnostic_files/traj_Pete/'
dir = 'C:/Users/paclk/OneDrive - University of Reading/traj_data/r6/'
#dir = 'C:/Users/xm904103/OneDrive - University of Reading/traj_data/r6/'

#dir = '/storage/shared/metcloud/wxproc/xm904103/traj/'
#dir = '/projects/paracon/appc/MONC/r4664_CA_traj/diagnostic_files/r5/'
#dir = '/projects/paracon/appc/MONC/prev/r4664_CA_traj/diagnostic_files/'
#dir = '/projects/paracon/appc/MONC/r4664_CA_traj/diagnostic_files/r6/'
files = glob.glob(dir+"diagnostics_3d_ts_*.nc")
files.sort(key=file_key)

#print thref, piref, thref*piref

#end_file = len(files)-1
#start_file = end_file-1

#ref_file = 39
first_ref_file = 49
last_ref_file =  89
tr_back_len = 40
tr_forward_len = 30


#debug_unsplit = True
debug_unsplit = False
debug_label = False  
#debug_label = True  
#debug_mean = True 
debug_mean = False 
#debug = True
debug = False
cyclic_xy = True
#use_bilin = False
use_bilin = True
order_labs =[ \
             'linear', \
             'quadratic', \
             'cubic', \
             'quartic', \
             'quintic', \
             ] 
#interp_order = 5
#interp_order = 3
interp_order = 1
iorder = '_'+order_labs[interp_order-1]
fn = os.path.basename(files[last_ref_file])[:-3]
fn = ''.join(fn)+iorder
fn = dir + fn

ref_prof_file = glob.glob(dir+'diagnostics_ts_*.nc')[0]

#get_traj = False
get_traj = True

if get_traj :
    tfm = trajectory_family(files, ref_prof_file, \
                 first_ref_file, last_ref_file, \
                 tr_back_len, tr_forward_len, \
                 100.0, 100.0, 40.0)

traj_list = tfm.family
tfm.print_matching_object_list()
tfm.print_matching_object_list_summary()

tfm.print_linked_objects()

matching_object_list = tfm.matching_object_list()

traj_m = traj_list[-1]
traj_r = traj_list[0]
sel_list   = np.array([0, 62, 70, 85])
#sel_list_r = np.array([0, 1, 2, 6, 8, 11, 67, 71])
#sel_list_r = np.array([2,12,73])
sel_list_r = np.array([0, 5, 72, 78, 92])


#plot_traj_pos(traj_m, ref_file-start_file, save=False) 
#plot_traj_pos(traj_m, ref_file-start_file+1, save=False) 

#data_mean, traj_m_centroid = compute_traj_centroids(traj_m)
input("Press Enter to continue...")
# Plot all clouds
if True :
    plot_traj_animation(traj_m, save_anim=False, with_boxes=True, \
                        title = 'Reference Time {}'.format(last_ref_file))

#input("Press Enter to continue...")
# Plot all clouds with galilean transform
if True :
    plot_traj_animation(traj_m, save_anim=False, \
        title = 'Reference Time {} Galilean Tranformed'.format(last_ref_file), \
        galilean = np.array([-8.5,0]))

if False :
    plot_traj_animation(traj_r, save_anim=False, \
        title = 'Reference Time {} Galilean Tranformed'.format(first_ref_file), \
        galilean = np.array([-8.5,0]))

max_list = traj_m.max_at_ref
print(max_list)
if False :
#    for iobj in range(0,traj_m.nobjects):
#    for iobj in range(1,7):    
    for iobj in sel_list:
        plot_trajectory_history(traj_m, iobj) 

    plt.show()    

# print_boxes(traj_m)


#print_centroids(traj_centroid, data_mean, traj_m.nobjects, traj_m.times, \
#                list(traj_m.variable_list))

input("Press Enter to continue...")

# Plot max_list clouds with galilean transform
if True :
    plot_traj_animation(traj_m, save_anim=False, select = sel_list, \
        no_cloud_size = 0.2, cloud_size = 2.0, legend = True, \
        title = 'Reference Time {} Galilean Tranformed'.format(last_ref_file), \
        with_boxes = False, galilean = np.array([-8.5,0]) )

if True :
    plot_traj_animation(traj_r, save_anim=False, select = sel_list_r, \
        no_cloud_size = 0.2, cloud_size = 2.0, legend = True, \
        title = 'Reference Time {} Galilean Tranformed'.format(last_ref_file-1), \
        with_boxes = False, galilean = np.array([-8.5,0]) )

# Plot max_list clouds mean history
if True :
    mean_cloud_properties, mean_entr_properties, mean_detr_properties, \
      traj_class, first_cloud_base, min_cloud_base, cloud_top \
      = traj_m.cloud_properties()
      
if True :
    for cloud in sel_list :
        plot_traj_animation(traj_m, save_anim=False, \
                    select = np.array([cloud]), fps = 10,  \
                    no_cloud_size = 0.2, cloud_size = 2.0, legend = True, \
                    title = 'Reference Time {0} Cloud {1} Galilean Trans'.\
                    format(last_ref_file, cloud), with_boxes = False, 
                    galilean = np.array([-8.5,0]), plot_class = traj_class)
     
      
if True :
    plot_trajectory_mean_history(traj_m, mean_cloud_properties, \
                                 mean_entr_properties, mean_detr_properties, \
                                 traj_class, first_cloud_base, cloud_top, \
                                 select = sel_list) 

#max_list=np.array([9,18,21,22,24,36,38,43,49,52,63,69,70,77,83,87,88,96])
#max_list=np.array([20,23,34,35,37,42,48,51,69,76,82,83,86])


#max_list=np.array([2,13,26,29,37,42])
#max_list=np.array([0,62,70,85,95])
#max_list_r=np.array([0,3,7,67,71,86])

#if False :
#    plot_traj_animation(traj_r, save_anim=False, select=sel_list_r,legend=True)
    

# Plot subset max_list clouds with galilean transform
# Not needed   
if False :
    plot_traj_animation(traj_m, save_anim=False, select = max_list, \
                    no_cloud_size = 0.2, cloud_size = 2.0, legend = True, \
                    with_boxes = False, galilean = np.array([-8.5,0]))

# Plot subset max_list clouds with galilean transform
if False :
    plot_traj_animation(traj_r, save_anim=False, select = sel_list_r, \
                    no_cloud_size = 0.2, cloud_size = 2.0, legend = True, \
                    with_boxes = False, galilean = np.array([-8.5,0]))


# Plot subset max_list clouds mean history
if False :
    plot_trajectory_mean_history(traj_m, mean_cloud_properties, \
                                 mean_entr_properties, mean_detr_properties, \
                                 traj_class, first_cloud_base, cloud_top, \
                                 select = max_list) 
if True :
    plot_traj_animation(traj_m, save_anim=False, select = sel_list, \
                    legend = True, plot_field = True, \
                    no_cloud_size = 0.2, cloud_size = 2.0, field_size = 0.5, \
                    title = 'Reference Time {0} Galilean Trans with clouds'.\
                    format(last_ref_file), with_boxes = False, \
                    galilean = np.array([-8.5,0]))

if True :
    for cloud in sel_list :
        plot_traj_animation(traj_m, save_anim=False, select = np.array([cloud]), \
                    fps = 10, legend = True, plot_field = True, \
                    no_cloud_size = 0.2, cloud_size = 2.0, field_size = 0.5, \
                    title = 'Reference Time {0} Cloud {1} Galilean Trans'.\
                    format(last_ref_file, cloud), \
                    galilean = np.array([-8.5,0]))
        for tback in [30] :
            plot_traj_family_animation(tfm, tback, save_anim=False, \
                    select = np.array([cloud]), \
                    fps = 10, legend = True, plot_field = False, \
                    title = 'Reference Times {0},{1} Cloud {2} Galilean Trans'.\
                    format(last_ref_file, last_ref_file-tback,  cloud), \
                    no_cloud_size = 0.2, cloud_size = 2.0, field_size = 0.5, \
                    with_boxes = False, galilean = np.array([-8.5,0]))

        for tback in [-1] :
            plot_traj_family_animation(tfm, tback, save_anim=False, \
                    select = np.array([cloud]), \
                    fps = 10, legend = True, plot_field = False, \
                    title = 'Reference Times {0} Cloud {1} Galilean Trans'.\
                    format(last_ref_file,  cloud), \
                    no_cloud_size = 0.2, cloud_size = 2.0, field_size = 0.5, \
                    with_boxes = False, galilean = np.array([-8.5,0]))

#if True :
#    for cloud in sel_list :
        

#print(min_cloud_base)
    
#print(np.where(traj_centroid != trs))

#print(trs[traj_centroid != trs])
#print(traj_centroid[traj_centroid != trs])

#########################################################