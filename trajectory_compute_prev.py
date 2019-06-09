# -*- coding: utf-8 -*-
from netCDF4 import Dataset
import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans
from scipy.optimize import minimize

L_vap = 2.501E6
Cp = 1005.0
L_over_cp = L_vap / Cp
grav = 9.81
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

class trajectory_family : 
    """
    Class defining a family of back trajectories.
    
	This is an ordered list of trajectories with sequential reference times.

    @author: Peter Clark
	
    """
	
    def __init__(self, files, ref_prof_file, \
                 first_ref_file, last_ref_file, \
                 back_len, forward_len, \
                 deltax, deltay, deltaz, variable_list=None, \
                 thresh=1.0E-5) : 
        """
        Create an instance of a familiy of back trajectories.
		
        This is an ordered list of trajectories with sequential reference times.

        Args:
            files              : ordered list of files used to generate 
                                 trajectories
            ref_prof_file      : name of file containing reference profile.
            first_ref_file     : Index of first file to use as reference.
            last_ref_file      : Index of last file to use as reference.
            back_len           : How many output timesteps to go back from ref.
            forward_len        : How many output timesteps to go forward.
            deltax             : Model x grid spacing in m.
            deltay             : Model y grid spacing in m.
            deltaz             : Model z grid spacing in m. 
            variable_list=None : List of variable names for data to interpolate
                                 to trajectory.
            thresh=1.0E-5      : Threshold for cloud definition.
			
        @author: Peter Clark
			
        """
		
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
    
    def matching_object_list(self, ref = None, select = None ):
        """
        Method to generate a list of matching objects at all times they match.
		
        Matching is done using overlap of cloud boxes.
        
        Args : 
            ref = None  : Which trajectory set in family to use to find 
                          matching objects.
                          Default is the last set.
                          
        Returns :		
            List with one member for each trajectory set with an earlier 
            reference time than ref. 
            
            By default, this is all the sets apart 
            from the last. Let us say t_off is the backwards offset from the
            reference time, with t_off=0 for the time immediately before the
            reference time.
            
            Each member is a list with one member for each back trajectory time 
            in the ref trajectories, containing a list with one member for 
            every object in ref,comprising an array of integer indices of the 
            matching objects in the earlier set that match the reference object
            at the given time. This may be empty if no matches have been found.
            Let us say the it_back measures the time backwards from ref, with
            it_back=0 at th ereference time.
            
            Thus, if mol is the matching object list,
            mol[t_off][it_back][iobj] is an array of object ids belonging to
            trajectory set ref-(t_off+1), at ref trajectory time it_back before 
            the ref reference time and matching iobj a the reference time.
			
        @author: Peter Clark
		
        """
		
        mol = list([])
        if ref is None : ref = len(self.family) - 1
        traj = self.family[ref]
        if select is None : select = np.arange(0, traj.nobjects, dtype = int)
        # Iterate backwards from first set before ref.
        for t_off in range(0, ref) :
#            print("Matching at reference time offset {}".format(t_off+1))
            # We are looking for objects in match_traj matching those in traj.
            match_traj = self.family[ref-(t_off+1)]
            # matching_objects[t_off] will be those objects in match_traj 
            # which match traj at any time.
            matching_objects = list([])
#            print('ref, t_off',ref, t_off)
            # Iterate backwards over all set times from ref to start. 
            for it_back in range(0,traj.ref_file+1) :
                # Time to make comparison
                ref_time = traj.ref_file-it_back
                # Note match_traj.ref_file in general will equal traj.ref_file 
                # Time in match_traj that matches ref_time
                 #       so match_time = ref - it_back
                match_time = match_traj.ref_file + (t_off + 1)- it_back 
#                print("Matching at reference trajectory time {}".format(it_back))
                matching_object_at_time = list([])
                # Iterate over objects in ref.
 #               print('ref_time, match_time',ref_time, match_time)
#                input("Press enter")
                for iobj in select :
#                    corr_box = np.array([],dtype=int)
                    # Only look at clouds
#                    print('iobj',iobj)
                    if traj.num_cloud[ref_time ,iobj] > 0 :
                        b_test = traj.cloud_box[ref_time, iobj,...]
#                        print("b_test",b_test)
#                        if iobj == 0 : print("Matching time {}".format(match_time))
                        # Find boxes in match_traj at match_time that overlap
                        # cloud box for iobj at the same time.
                        if (match_time >= 0) & \
                          (match_time < np.shape(match_traj.cloud_box)[0]) :
                            b_set  = match_traj.cloud_box[match_time,...]
#                            print(b_set[0:4])
                            corr_box = box_overlap_with_wrap(b_test, b_set, \
                                traj.nx, traj.ny)
#                            print("corr_box",corr_box,b_set[corr_box,...])
                            valid = (match_traj.num_cloud[match_time, corr_box]>0)
#                            print("valid", valid)
#                            print('box 1',match_traj.cloud_box[match_time,1])
                            corr_box = corr_box[valid]
#                            if iobj == 85 :
#                                print(iobj,match_time,b_test, corr_box, b_set[corr_box,...])
#                                input("Press enter")
                    matching_object_at_time.append(corr_box)
                matching_objects.append(matching_object_at_time)
            mol.append(matching_objects)
        return mol
   
    def print_matching_object_list(self, ref = None, select = None) :    
        """
        Method to print matching object list.
		
        See method matching_object_list.
        
        @author: Peter Clark
		
        """
		
        if ref == None : ref = len(self.family) - 1
        if select is None : select = np.arange(0, self.family[ref].nobjects, \
                                           dtype = int)
        mol = self.matching_object_list(ref = ref, select = select)
#        for t_off in range(0, 4) :
        for t_off in range(0, len(mol)) :
            matching_objects = mol[t_off]
            for i in range(0, len(matching_objects[0])) :
                iobj = select[i]
                for it_back in range(0, len(matching_objects)) :
#                for it_back in range(0, 4) :
                    for obj in matching_objects[it_back][i] :
                        if np.size(obj) > 0 :
                            print("t_off: {0} iobj: {1} it_back: {2} obj: {3}".\
                                  format(t_off+1, iobj, it_back, obj))
        return
    
    def matching_object_list_summary(self, ref = None, select = None, \
                                     overlap_thresh=0.02) :
        """
        Method to classify matching objects.
        
        Args : 
            ref=None  : Which trajectory set in family to use to find 
                        matching objects.
                        Default is the last set.
            overlap_thresh=0.02 Threshold for overlap to be sufficient 
            for inclusion.
                          
        Returns :
            List with one member for each trajectory set with an earlier 
            reference than ref. 
            
            By default, this is all the sets apart from the last.
            
            Each member is a list with one member for each object in ref,
            containing a list of objects in the earlier set that match the
            object in ref AT ANY TIME.
            
            Each of these is classified 'Linked' if the object is in the 
            max_at_ref list at the earler reference time, otherwise 'Same'.
			
        @author: Peter Clark
		
        """
        
        if ref is None : ref = len(self.family) - 1
        traj = self.family[ref]
        if select is None : select = np.arange(0, traj.nobjects, dtype = int )
        
        mol = self.matching_object_list(ref = ref, select=select)
        mols = list([])
        # Iterate backwards from first set before ref.
        for t_off, matching_objects in enumerate(mol) :
            match_list = list([])
#            match_time = ref-(t_off+1)
                # Note match_traj.ref_file in general will equal traj.ref_file 
                # Time in match_traj that matches ref_time
                 #       so match_time = ref - it_back
            for i, iobj in enumerate(select) :
                objlist = list([])
                otypelist = list([])
                interlist = list([])
                for it_back in range(0, len(matching_objects)) :
                    match_traj = self.family[ref-(t_off+1)]
                    ref_time = traj.ref_file-it_back
                    match_time = match_traj.ref_file + (t_off + 1)- it_back 
#                    print('times',ref_time,match_time)
                    for obj in matching_objects[it_back][i] :
                        if np.size(obj) > 0 :
                            if obj not in objlist : 
                                
                                if (match_time >= 0) & \
                                        (match_time < len(self.family)) :
                                    inter = self.refine_object_overlap(t_off, \
                                            it_back, iobj, obj, ref = ref)
                                
                                    if inter > overlap_thresh :
                                
                                        otype = 'Same'
                                        if obj in \
                                            match_traj.max_at_ref :
                                                otype = 'Linked'
                                        objlist.append(obj)
                                        otypelist.append(otype)
                                        interlist.append(int(inter*100+0.5))
                match_list.append(list(zip(objlist,otypelist,interlist)))
            mols.append(match_list)
        return mols
    
    def print_matching_object_list_summary(self, ref = None,  select = None, \
                                           overlap_thresh = 0.02) :    
        """
        Method to print matching object list summary.
		
        See method matching_object_list_summary.
        
        @author: Peter Clark
		
        """
		
        if ref == None : ref = len(self.family) - 1
        if select is None : select = np.arange(0, self.family[ref].nobjects, \
                                               dtype = int)
        mols = self.matching_object_list_summary(ref = ref, select = select, \
                                                 overlap_thresh = \
                                                 overlap_thresh)
        for t_off in range(0, len(mols)) :
            matching_objects = mols[t_off]
            for i in range(0, len(matching_objects)) :
                iobj = select[i]
                rep =""
                for mo, mot, mint in matching_objects[i] : 
                    rep += "({}, {}, {:02d})".format(mo,mot,mint)
                print("t_off: {0} iobj: {1} obj: {2} ".\
                                  format(t_off, iobj, rep))
                           
        return

    def find_linked_objects(self, ref = None, select = None, \
        overlap_thresh = 0.02) :   
        """
        Method to find all objects linked to objects in max_at_ref list in ref.
        
        Args : 
            ref = None  : Which trajectory set in family to use to find 
                          matching objects.
                          Default is the last set.
            overlap_thresh=0.02 Threshold for overlap to be sufficient 
            for inclusion.
                          
        Returns :
            List with one member for each object in max_at_ref list in ref.
            Each member is a an array of triplets containing the time,
            object id and percentage ovelap with next of objects in the 
            max_at_ref list of the family at time classified as 'Linked'.
			
        @author: Peter Clark
		
        """
		
        if ref is None : ref = len(self.family) - 1
        if select is None : select = self.family[ref].max_at_ref
        mols = self.matching_object_list_summary(ref = ref, select=select,
                                                 overlap_thresh = \
                                                 overlap_thresh)
        linked_objects = list([])
        for i in range(len(select)) :
            linked_obj = list([])
            linked_obj.append([ref, select[i], 100])
            for t_off in range(0, len(mols)) :
                matching_objects = mols[t_off][i]
                for mo, mot, mint in matching_objects : 
                    if mot == 'Linked' :
                        linked_obj.append([ref-t_off-1, mo, mint])
            linked_objects.append(np.array(linked_obj))
        return linked_objects
    
    def print_linked_objects(self, ref = None, select = None, \
                             overlap_thresh = 0.02) :
        """
        Method to print linked object list.
		
        See method find_linked_objects.
        
        @author: Peter Clark
		
        """
		
        if ref == None : ref = len(self.family) - 1
        if select is None : select = self.family[ref].max_at_ref
        linked_objects = self.find_linked_objects(ref = ref, select=select, \
                                                  overlap_thresh = \
                                                  overlap_thresh)
        for iobj, linked_obj in zip(select, \
                                    linked_objects) :
            rep =""
 #           for t_off, mo in linked_obj : 
            for i in range(np.shape(linked_obj)[0]) :
                t_off, mo, mint = linked_obj[i,:]  
                rep += "[{}, {}, {}]".format(t_off, mo, mint)
            print("iobj: {0} objects: {1} ".format(iobj, rep))
        return
    
    def find_super_objects(self, ref = None, \
        overlap_thresh = 0.02) :
        """
        Method to find all objects linked contiguously to objects in max_at_ref list in ref.
        
        Args : 
            ref = None  : Which trajectory set in family to use to find 
                          matching objects.
                          Default is the last set.
            overlap_thresh=0.02 Threshold for overlap to be sufficient 
            for inclusion.
                          
        Returns :
            List with one member for each object in max_at_ref list in ref.
            Each member is an array of triplets containing the time,
            object id and percentage ovelap with next of objects in the 
            max_at_ref list of the family at time classified as 'Linked'.
			
        @author: Peter Clark
		
        """
		
        
        def step_obj_back(ref, objs) :
            found_super_objs = list([])
            mol = self.matching_object_list(ref = ref, select=objs)
            t_off = 0
            matching_objects = mol[t_off]
            match_time = ref-(t_off+1)
            for i,iobj in enumerate(objs) :
                objlist = list([(ref,iobj,100)])
                it_back = 0
#                print(iobj)
#                print(ref,match_time, matching_objects[it_back][i])
                for obj in matching_objects[it_back][i] :
                    if np.size(obj) > 0 :
                                
                        inter = self.refine_object_overlap(t_off, \
                                    it_back, iobj, obj, ref = ref)
#                        print(obj,inter)        
                        if inter > overlap_thresh :
                            if obj in self.family[ref-(t_off+1)].max_at_ref :
                                objlist.append((match_time,obj,\
                                                int(inter*100+0.5)))
#                                print(objlist)
            
                found_super_objs.append(objlist)
            return found_super_objs
			
        print("Finding super-objects")
        if ref is None : ref = len(self.family) - 1
        select = self.family[ref].max_at_ref
        super_objects = list([])
        incomplete_objects = list([])
        incomplete_object_ids = np.array([],dtype=int)
        for newref in range(ref,0,-1) :
            sup_obj = step_obj_back(newref, select)
            next_level_objs = self.family[newref-1].max_at_ref
            new_incomplete_objects = list([])
            new_incomplete_object_ids = list([])
            for i, obj in enumerate(sup_obj) :
                # Four options
                # 1. New object length 1.
                # 2. Termination of previous object.
                # 3. New object continued at next level.
                # 4. Continuation of previous object.
                if len(obj) == 1 :
                    if obj[0][1] in incomplete_object_ids :
                        # Terminate incomplete object.
                        j = np.where(obj[0][1] == incomplete_object_ids)[0][0]
                        super_objects.append(np.array(incomplete_objects[j]))
                    else :
                        super_objects.append(np.array(obj))
                else :
                    if obj[0][1] in incomplete_object_ids :
                        j = np.where(obj[0][1] == incomplete_object_ids)[0][0]
                        incomplete_objects[j].append(obj[1])
                        new_incomplete_objects.append(incomplete_objects[j]) 
                    else :
                        new_incomplete_objects.append(obj) 
                        
                    new_incomplete_object_ids.append(\
                        new_incomplete_objects[-1][-1][1])
            incomplete_object_ids = np.array(new_incomplete_object_ids)           
            incomplete_objects = new_incomplete_objects
            select = next_level_objs  
			
        for obj in incomplete_objects : super_objects.append(np.array(obj))  
        len_sup=[]
        for s in super_objects : len_sup.append(len(s))
        len_sup = np.array(len_sup)
                         
        return super_objects, len_sup
           
    def refine_object_overlap(self, t_off, time, obj, mobj, ref = None) :
        """
        Method to estimate degree of overlap between two trajectory objects.
		
        
        Args : 
            Reference object is self.family trajectory object obj 
			at time ref-time
			Comparison object is self.family trajectory object mobj 
			at same true time, reference time ref-(t_off+1).
			ref=None - default is last set in family.
                           
        Returns :
            fractional overlap
			
        @author: Peter Clark
		
        """
		
        if ref == None : ref = len(self.family) - 1
#        print(t_off, time, obj, mobj)
#        print("Comparing trajectories {} and {}".format(ref, ref-(t_off+1)))
        
        def extract_obj_as1Dint(fam, ref, time, obj) :
            traj = fam[ref]
            tr_time = traj.ref_file-time
#            print("Time in {} is {}, {}".format(ref, tr_time, traj.times[tr_time]))
            obj_ptrs = (traj.labels == obj)
#            print('extr',ref, time, tr_time)
            qcl = traj.data[tr_time, obj_ptrs, traj.var("q_cloud_liquid_mass")]
#            print(np.shape(qcl))
            mask = (qcl >= traj.thresh)
#            print(np.shape(mask))
            tr = (traj.trajectory[tr_time, obj_ptrs, ... ] + 0.5).astype(int)
#            print(np.shape(tr))
            tr = tr[mask,:]
#            for i in range(2) : print(np.min(tr[:,i]),np.max(tr[:,i]))
            tr1D = np.unique(tr[:,0] + traj.nx * (tr[:,1] + traj.ny * tr[:,2]))
            return tr1D
#        traj = self.family[ref]
#        match_traj = self.family[ref-(t_off+1)]
#        ref_time = traj.ref_file-it_back
#        match_time = match_traj.ref_file + (t_off + 1)- it_back 
    
        tr1D  = extract_obj_as1Dint(self.family, ref,           \
                                    time, obj)
        trm1D = extract_obj_as1Dint(self.family, ref-(t_off+1), \
                                    time-(t_off+1) , mobj)
                
        max_size = np.max([np.size(tr1D),np.size(trm1D)])
        if max_size > 0 :
            intersection = np.size(np.intersect1d(tr1D,trm1D))/max_size
        else :
            intersection = 0
        return intersection
            
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
    """
    Class defining a set of back trajectories with a given reference.
    
    @author: Peter Clark
	
    """

    def __init__(self, files, ref_prof_file, start_file_no, ref_file_no, \
                 end_file_no, deltax, deltay, deltaz, variable_list=None, \
                 thresh=1.0E-5) : 
        """
        Create an instance of a set of trajectories with a given reference.
		
        This is an ordered list of trajectories with sequential reference times.

        Args:
            files              : ordered list of files used to generate 
                                 trajectories
            ref_prof_file      : name of file containing reference profile.
            start_file_no      : Index of file for origin of back trajectories.
            end_file_no        : Index of file for end of forward trajectories.
            deltax             : Model x grid spacing in m.
            deltay             : Model y grid spacing in m.
            deltaz             : Model z grid spacing in m. 
            variable_list=None : List of variable names for data to interpolate
                                 to trajectory.
            thresh=1.0E-5      : Threshold for cloud definition.
			
        @author: Peter Clark
			
        """
        
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
        
    def var(self, v) :
        """
        Method to convert variable name to numerical pointer. 
		
        Args:
			v : string variable name.
			
		Returns:
		    Numerical pointer to data in data array.
			
        @author: Peter Clark
			
        """

        for ii, vr in enumerate(list(self.variable_list)) :
            if vr==v : break
        return ii
    
    def select_object(self, iobj) :
        """
        Method to find trajectory and associated data corresponding to iobj.
		
        Args:
			iobj : object id (integer).
			
		Returns:
		    trajectory_array, associated _data.
			
        @author: Peter Clark
			
        """
		
        in_object = (self.labels == iobj)
        obj = self.trajectory[:, in_object, ...]
        dat = self.data[:, in_object, ...]
        return obj, dat
    
    def cloud_properties(self, thresh = None, version = 1) :
        """
        Method to compute trajectory class and mean cloud properties.
		
        Args:
			thresh = None, : Threshold if LWC to define cloud.
			                 default is self.thresh.
			version = 1
			
		Returns:
            mean_properties = dictionary pointing to arrays of mean properties
			                  and meta data.
							  Dictionary keys:
							   "unclassified"
                               "pre_cloud_bl"
                               "pre_cloud_above_bl"
                               "previous_cloud"
                               "cloud"
                               "entr"
                               "entr_bot"
                               "detr"
                               "subsequent_cloud"
                               "class"
                               "first cloud base"
                               "min cloud base"
                               "cloud top"
                               "cloud_trigger_time"
                               "cloud_dissipate_time"
                               "version"
			
        @author: Peter Clark
			
        """

        if thresh == None : thresh = self.thresh
                        
        nvars = np.shape(self.data)[2]
        total_nvars = nvars + 5
        # These possible slices are set for ease of maintenance.
        r1 = slice(0,nvars)
        # Moist static energy
        r2 = nvars
        # Position
        r3 = slice(nvars+1,nvars+4)
        # Number of points
        r4 = nvars+4
        
        if version < 3 :
            mean_cloud_prop = np.zeros([self.ntimes, self.nobjects, total_nvars])
            mean_entr_prop = np.zeros([self.ntimes, self.nobjects, total_nvars])
            mean_entr_bot_prop = np.zeros([self.ntimes, self.nobjects, total_nvars])
            mean_detr_prop = np.zeros([self.ntimes, self.nobjects, total_nvars])
            mean_bl_prop = np.zeros([self.ntimes, self.nobjects, total_nvars])
        elif version == 3 :
            n_class = 9
            mean_prop = np.zeros([self.ntimes, self.nobjects, total_nvars, 10])
        
        traj_class = np.zeros_like(self.data[:,:,0], dtype=int)
        
        min_cloud_base = np.zeros(self.nobjects)
        first_cloud_base = np.zeros(self.nobjects)
        cloud_top = np.zeros(self.nobjects)
        cloud_trigger_time = np.zeros(self.nobjects, dtype=int)
        cloud_dissipate_time = np.ones(self.nobjects, dtype=int)*self.ntimes
        
        tr_z = (self.trajectory[:,:,2]-0.5)*self.deltaz
        zn = (np.arange(0,np.size(self.piref))-0.5)*self.deltaz
        piref_z = np.interp(tr_z,zn,self.piref)
      
        moist_static_energy = Cp * self.data[:, :, self.var("th")] * piref_z + \
                              grav * tr_z + \
                              L_vap * self.data[:, :, self.var("q_vapour")]
                             
#        print("Computed MSE",np.min(moist_static_energy),np.max(moist_static_energy))
                                      
        def set_props(prop, time, obj, mask, where_mask) :
            prop[time, obj, r1] = np.sum(data[time, mask,:], axis=0)
            prop[time, obj, r2] = np.sum(mse[time, mask], axis=0)             
            prop[time, obj, r3] = np.sum(tr[time, mask,:], axis=0)
            prop[time, obj, r4] = np.size(where_mask)
            return prop
        
        if version == 1 :
            PRE_CLOUD_IN_BL = 1
            PRE_CLOUD_ABOVE_BL = 2
            IN_CLOUD_AT_START = 3
            FIRST_CLOUD = 4
            NEW_CLOUD_FROM_BOTTOM = 5
            NEW_CLOUD_FROM_SIDE = 6
            DETR_CLOUD = 7        
        elif version == 2 or version == 3:
            PRE_CLOUD_ENTR_FROM_BL = 1
            PRE_CLOUD_ENTR_FROM_ABOVE_BL = 2
            PREVIOUS_CLOUD = 3
            CLOUD = 4
            ENTR_FROM_BL = 5
            ENTR_FROM_ABOVE_BL = 6
#            ENTR_PREV_CLOUD = 7
            DETR_CLOUD = 7 
            SUBSEQUENT_CLOUD = 8
        else :
            print("Illegal Version")
            return 
                        
        for iobj in range(0,self.nobjects) :
#            debug_mean = (iobj == 85)
            if debug_mean : print('Processing object {}'.format(iobj))
            obj_ptrs = (self.labels == iobj)
            where_obj_ptrs = np.where(obj_ptrs)[0]
            tr = self.trajectory[:, obj_ptrs, :]
            data = self.data[:, obj_ptrs, :]
            obj_z = tr_z[:,obj_ptrs]
            if debug_mean : print(np.shape(data))
            mse = moist_static_energy[:, obj_ptrs]
#            print("MSE:", iobj,np.min(mse),np.max(mse))
            qcl = data[:,:,self.var("q_cloud_liquid_mass")]
            mask = (qcl >= thresh)
     
            # Original extracted variables
            min_cloud_base[iobj] = (np.min(tr[mask,2])-0.5)*self.deltaz
    #        data_class = traj_class[:,self.labels == iobj]
            if debug_mean : print('Version ',version)

            if version == 1 :
                bl = ((tr[0,:,2]-0.5)*self.deltaz < min_cloud_base[iobj])
                class_set = where_obj_ptrs[np.where(bl)[0]]
                traj_class[:, class_set] = PRE_CLOUD_IN_BL 
                
                class_set = where_obj_ptrs[np.where(~bl)[0]]
                traj_class[:, class_set] = PRE_CLOUD_ABOVE_BL 
                
                after_cloud_formed = False        
                after_cloud_dissipated = False  
                
                for it in range(self.ntimes) :
                    if debug_mean : print("it = {}".format(it))
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
            elif version == 2 :
                
                in_main_cloud = True
                
                for it in range(self.ref_file,-1,-1) :
                    if debug_mean : print("it = {}".format(it))

                    cloud_bottom = self.cloud_box[it, iobj, 0, 2]
                    
                    cloud = mask[it,:]
                    where_cloud = np.where(cloud)[0]
                    
                    not_cloud = np.logical_not(cloud)
                    where_not_cloud = np.where(not_cloud)[0]
#                    if it == self.ref_file :
#                        print(np.size(cloud),np.size(where_cloud))
#                        print(np.size(not_cloud),np.size(where_not_cloud))
                    
                    if debug_mean : 
                        print('it', it, np.shape(where_cloud))
                        print(qcl[it,mask[it,:]])
                    
                    if np.size(where_cloud) > 0 :  
                        # There are cloudy points
                        # Compute mean properties of cloudy points.            
                        mean_cloud_prop = set_props(mean_cloud_prop, it, iobj,\
                                                    cloud, where_cloud)    
                        if debug_mean : 
#                        if True :
                            print('Mean cloud properties at this level.')
                            print(mean_cloud_prop[it, iobj, :])    
                            
                        # Find new cloudy points  
                        if it > 0 :
                            new_cloud = np.logical_and(cloud, \
                                                np.logical_not(mask[it-1,:]) )
                            where_newcloud = np.where(new_cloud)[0]
                            not_new_cloud = np.logical_and(cloud, mask[it-1,:])
                            where_not_newcloud = np.where(not_new_cloud)[0]
#                            print(np.size(where_newcloud),np.size(where_not_newcloud))
                            if debug_mean :
                                print(new_cloud)
                                print(qcl[it-1,new_cloud],qcl[it,new_cloud])
                        else :
                            where_newcloud = np.array([])
#                            not_newcloud = cloud
                            where_not_newcloud = where_cloud
                            
                        if np.size(where_newcloud) > 0 : # Entrainment
                            if debug_mean : print('Entraining air')
                            
                            newcl_from_bl = np.logical_and( \
                                              new_cloud,\
                                              obj_z[0, :] < \
                                              min_cloud_base[iobj])  
                            where_newcl_from_bl = np.where(newcl_from_bl)[0]
                            
                            newcl_from_above_bl = np.logical_and( \
                                                    new_cloud,\
                                                    obj_z[0, :] >= \
                                                    min_cloud_base[iobj])  
                            where_newcl_from_above_bl = np.where( \
                                                        newcl_from_above_bl)[0]

                            class_set_from_bl = where_obj_ptrs[ \
                                              where_newcl_from_bl]
                            traj_class[0:it, class_set_from_bl] = \
                                        PRE_CLOUD_ENTR_FROM_BL 
                            traj_class[it, class_set_from_bl] = \
                                        ENTR_FROM_BL 
                            class_set_from_above_bl = where_obj_ptrs[ \
                                                    where_newcl_from_above_bl]
                            traj_class[0:it, class_set_from_above_bl] = \
                                        PRE_CLOUD_ENTR_FROM_ABOVE_BL
                            traj_class[it, class_set_from_above_bl] = \
                                        ENTR_FROM_ABOVE_BL

                            if debug_mean : 
                                print("From BL",class_set_from_bl)
                                print("From above BL",class_set_from_above_bl)
                                input("Press Enter to continue...")
                            if np.size(where_newcl_from_bl) > 0 :
                                mean_entr_bot_prop = set_props( \
                                        mean_entr_bot_prop, \
                                        it, iobj, newcl_from_bl, \
                                        where_newcl_from_bl)
                            
                            if np.size(where_newcl_from_above_bl) > 0 :
                                mean_entr_prop = set_props( \
                                        mean_entr_prop, \
                                        it, iobj, newcl_from_above_bl, \
                                        where_newcl_from_above_bl)

                            
                        # Set traj_class flag for those points in cloud.              
                        if np.size(where_not_newcloud) > 0 : # Entrainment
                            if debug_mean : print("Setting Cloud",it,np.size(where_not_newcloud))
                            class_set = where_obj_ptrs[where_not_newcloud]
                        
                            if in_main_cloud :
                                # Still in cloud contiguous with reference time.
                                traj_class[it:, class_set] = CLOUD
                            else :
                                # Must be cloud points that were present before.
                                traj_class[it:, class_set] = PREVIOUS_CLOUD
                    else :
                        if in_main_cloud : cloud_trigger_time[iobj] = it
                        in_main_cloud = False
                        
                    # Now what about entraining air?
                    if np.size(where_not_cloud) > 0 : 
                        
                        if it == self.ref_file :
                            print("Problem - reference not all cloud",iobj)
                        
                        # Find point that will be cloud at next step.
                        
#                        new_cloud = np.logical_and(not_cloud, mask[it+1,:] )
#                        where_newcloud = np.where(new_cloud)[0]
                        
                        if debug_mean : 
                            print(qcl[it,new_cloud],qcl[it+1,new_cloud])
                            
#                        if np.size(where_newcloud) > 0 : # Entrainment
                            
                                    
                        # Find points that have ceased to be cloudy     
                        if it > 0 :
                            detr_cloud =np.logical_and(not_cloud, mask[it-1,:])
                            where_detrained = np.where(detr_cloud)[0]
                            if debug_mean : 
                                print(detr_cloud)
                                print(qcl[it-1,detr_cloud],qcl[it,detr_cloud])
                                    
                            if np.size(where_detrained) > 0 : # Detrainment
                                if debug_mean : print('Detraining air')
                                    
                                mean_detr_prop = set_props(mean_detr_prop, \
                                                it, iobj, detr_cloud, \
                                                where_detrained)
                                      
                                class_set = where_obj_ptrs[where_detrained] 
                                traj_class[it, class_set] = DETR_CLOUD 
                                if debug_mean : 
                                    input("Press Enter to continue...")     
                                    
                in_main_cloud = True
                after_cloud_dissipated = False  

                for it in range(self.ref_file+1,self.end_file+1) :
                    if debug_mean : print("it = {}".format(it))
                    cloud_bottom = self.cloud_box[it, iobj, 0, 2]
                    
                    cloud = mask[it,:]
                    where_cloud = np.where(cloud)[0]
                    
                    not_cloud = np.logical_not(cloud)
                    where_not_cloud = np.where(not_cloud)[0]
                    
                    if debug_mean : 
                        print('it', it, np.shape(where_cloud))
                        print(qcl[it,mask[it,:]])
                    
                    if np.size(where_cloud) > 0 :  
                        # There are cloudy points
                        # Compute mean properties of cloudy points.            
                        mean_cloud_prop = set_props(mean_cloud_prop, it, iobj,\
                                                    cloud, where_cloud)    
                        if debug_mean : 
                            print('Mean cloud properties at this level.')
                            print(mean_cloud_prop[it, iobj, :])    
                        # Find new cloudy points  
                        new_cloud = np.logical_and(cloud, \
                                            np.logical_not(mask[it-1,:]) )
                        where_newcloud = np.where(new_cloud)[0]
                        not_new_cloud = np.logical_and(cloud, mask[it-1,:])
                        where_not_newcloud = np.where(not_new_cloud)[0]
#                            print(np.size(where_newcloud),np.size(where_not_newcloud))
                        if debug_mean :
                            print(new_cloud)
                            print(qcl[it-1,new_cloud],qcl[it,new_cloud])
                            
                        if np.size(where_newcloud) > 0 : # Entrainment
                            if debug_mean : print('Entraining air')
                            
                            newcl_from_bl = np.logical_and( \
                                              new_cloud,\
                                              obj_z[0, :] < \
                                              min_cloud_base[iobj])  
                            where_newcl_from_bl = np.where(newcl_from_bl)[0]
                            newcl_from_above_bl = np.logical_and( \
                                                    new_cloud,\
                                                    obj_z[0, :] >= \
                                                    min_cloud_base[iobj])  
                            where_newcl_from_above_bl = np.where( \
                                                        newcl_from_above_bl)[0]

                            class_set_from_bl = where_obj_ptrs[where_newcl_from_bl]
                            traj_class[it-1, class_set_from_bl] = \
                                        PRE_CLOUD_ENTR_FROM_BL 
                            traj_class[it, class_set_from_bl] = \
                                        ENTR_FROM_BL 
                            class_set_from_above_bl = where_obj_ptrs[where_newcl_from_above_bl]
                            traj_class[it-1, class_set_from_above_bl] = \
                                        PRE_CLOUD_ENTR_FROM_ABOVE_BL
                            traj_class[it, class_set_from_above_bl] = \
                                        ENTR_FROM_ABOVE_BL

                            if debug_mean : 
                                print("From BL",class_set_from_bl)
                                print("From above BL",class_set_from_above_bl)
                                input("Press Enter to continue...") 
                            if np.size(where_newcl_from_bl) > 0 :
                                mean_entr_bot_prop = set_props( \
                                        mean_entr_bot_prop, \
                                        it, iobj, newcl_from_bl, \
                                        where_newcl_from_bl)
                            
                            if np.size(where_newcl_from_above_bl) > 0 :
                                mean_entr_prop = set_props( \
                                        mean_entr_prop, \
                                        it, iobj, newcl_from_above_bl, \
                                        where_newcl_from_above_bl)

                            
                        # Set traj_class flag for those points in cloud.              
                        if np.size(where_not_newcloud) > 0 : # Entrainment
                            if debug_mean : print("Setting Cloud",it,np.size(where_not_newcloud))
                            class_set = where_obj_ptrs[where_not_newcloud]
                        
                            if in_main_cloud :
                                # Still in cloud contiguous with reference time.
                                traj_class[it:, class_set] = CLOUD
                            else :
                                # Must be cloud points that were present before.
                                traj_class[it:, class_set] = SUBSEQUENT_CLOUD                           
                        # Set traj_class flag for those points in cloud.              
#                        class_set = where_obj_ptrs[where_cloud]
#                        
#                        if in_main_cloud :
#                            # Still in cloud contiguous with reference time.
#                            traj_class[it:, class_set] = CLOUD
#                        else :
#                            # Must be cloud points that were present before.
#                            traj_class[it:, class_set] =
                    else :
                        if in_main_cloud and not after_cloud_dissipated :
                            # At cloud top.
                            cloud_dissipate_time[iobj] = it
                            cloud_top[iobj] = (np.max(tr[it-1,mask[it-1,:],2])-0.5) \
                                                *self.deltaz
                        in_main_cloud = False
                        after_cloud_dissipated = True                    
                    # Now what about detraining and detraining air?
                    if np.size(where_not_cloud) > 0 : 
                        # Find points that have ceased to be cloudy                     
                        detr_cloud =np.logical_and(not_cloud, mask[it-1,:])
                        
                        if debug_mean : 
                            print(detr_cloud)
                            print(qcl[it-1,detr_cloud],qcl[it,detr_cloud])
                                    
                        where_detrained = np.where(detr_cloud)[0]
                        if np.size(where_detrained) > 0 : # Detrainment
                            if debug_mean : print('Detraining air')
                                
                            mean_detr_prop = set_props(mean_detr_prop, \
                                            it, iobj, detr_cloud, \
                                            where_detrained)
                                  
                            class_set = where_obj_ptrs[where_detrained] 
                            traj_class[it:, class_set] = DETR_CLOUD 
                        
                        # Find point that will be cloud at next step.
                        
            elif version == 3 :
                
                in_main_cloud = True
                
                for it in range(self.ref_file,-1,-1) :
                    if debug_mean : print("it = {}".format(it))

                    cloud_bottom = self.cloud_box[it, iobj, 0, 2]
                    
                    cloud = mask[it,:]
                    where_cloud = np.where(cloud)[0]
                    
                    not_cloud = np.logical_not(cloud)
                    where_not_cloud = np.where(not_cloud)[0]
                    
                    if debug_mean : 
                        print('it', it, np.shape(where_cloud))
                        print(qcl[it,mask[it,:]])
                    
                    if np.size(where_cloud) > 0 :  
                        # There are cloudy points
                        if debug_mean : 
#                        if True :
                            print('Mean cloud properties at this level.')
                            print(mean_cloud_prop[it, iobj, :])    
                            
                        # Find new cloudy points  
                        if it > 0 :
                            new_cloud = np.logical_and(cloud, \
                                                np.logical_not(mask[it-1,:]) )
                            where_newcloud = np.where(new_cloud)[0]
                            not_new_cloud = np.logical_and(cloud, mask[it-1,:])
                            where_not_newcloud = np.where(not_new_cloud)[0]
#                            print(np.size(where_newcloud),np.size(where_not_newcloud))
                            if debug_mean :
                                print(new_cloud)
                                print(qcl[it-1,new_cloud],qcl[it,new_cloud])
                        else :
                            where_newcloud = np.array([])
#                            not_newcloud = cloud
                            not_new_cloud = cloud
                            where_not_newcloud = where_cloud
                            
                        if np.size(where_newcloud) > 0 : # Entrainment
                            if debug_mean : print('Entraining air')
                            
                            newcl_from_bl = np.logical_and( \
                                              new_cloud,\
                                              obj_z[0, :] < \
                                              min_cloud_base[iobj])  
                            where_newcl_from_bl = np.where(newcl_from_bl)[0]
                            class_set_from_bl = where_obj_ptrs[ \
                                              where_newcl_from_bl]
                            
                            newcl_from_above_bl = np.logical_and( \
                                                    new_cloud,\
                                                    obj_z[0, :] >= \
                                                    min_cloud_base[iobj])  
                            where_newcl_from_above_bl = np.where( \
                                                        newcl_from_above_bl)[0]

                            class_set_from_above_bl = where_obj_ptrs[ \
                                                    where_newcl_from_above_bl]
                            traj_class[0:it, class_set_from_bl] = \
                                        PRE_CLOUD_ENTR_FROM_BL 
                            traj_class[it, class_set_from_bl] = \
                                        ENTR_FROM_BL 
                                        
                            traj_class[0:it, class_set_from_above_bl] = \
                                        PRE_CLOUD_ENTR_FROM_ABOVE_BL
                            traj_class[it, class_set_from_above_bl] = \
                                        ENTR_FROM_ABOVE_BL

                            if debug_mean : 
                                print("From BL",class_set_from_bl)
                                print("From above BL",class_set_from_above_bl)
                                input("Press Enter to continue...")
                            
                        # Set traj_class flag for those points in cloud.              
                        if np.size(where_not_newcloud) > 0 : # Entrainment
                            if debug_mean : 
                                print("Setting Cloud",it,\
                                      np.size(where_not_newcloud))
                            class_set = where_obj_ptrs[where_not_newcloud]
                        
                            if in_main_cloud :
                                # Still in cloud contiguous with reference time.
                                traj_class[it, class_set] = CLOUD
                            else :
                                # Must be cloud points that were present before.
                                traj_class[it, class_set] = PREVIOUS_CLOUD
                    else :
                        if in_main_cloud : cloud_trigger_time[iobj] = it+1
                        in_main_cloud = False
                        
                    # Now what about entraining air?
                    if np.size(where_not_cloud) > 0 : 
                        
                        if it == self.ref_file :
                            print("Problem - reference not all cloud",iobj)
                        
                        # Find point that will be cloud at next step.
                                                
                        if debug_mean : 
                            print(qcl[it,new_cloud],qcl[it+1,new_cloud])
                                                                
                        # Find points that have ceased to be cloudy     
                        if it > 0 :
                            detr_cloud =np.logical_and(not_cloud, mask[it-1,:])
                            where_detrained = np.where(detr_cloud)[0]
                            if debug_mean : 
                                print(detr_cloud)
                                print(qcl[it-1,detr_cloud],qcl[it,detr_cloud])
                                    
                            if np.size(where_detrained) > 0 : # Detrainment
                                if debug_mean : print('Detraining air')
                                    
                                class_set = where_obj_ptrs[where_detrained] 
                                traj_class[it, class_set] = DETR_CLOUD 
                                if debug_mean : 
                                    input("Press Enter to continue...")     
                                    
                in_main_cloud = True
                after_cloud_dissipated = False  

                for it in range(self.ref_file+1,self.end_file+1) :
                    if debug_mean : print("it = {}".format(it))
                    cloud_bottom = self.cloud_box[it, iobj, 0, 2]
                    
                    cloud = mask[it,:]
                    where_cloud = np.where(cloud)[0]
                    
                    not_cloud = np.logical_not(cloud)
                    where_not_cloud = np.where(not_cloud)[0]
                    
                    if debug_mean : 
                        print('it', it, np.shape(where_cloud))
                        print(qcl[it,mask[it,:]])
                    
                    if np.size(where_cloud) > 0 :  
                        # There are cloudy points
                        if debug_mean : 
                            print('Mean cloud properties at this level.')
                            print(mean_cloud_prop[it, iobj, :])    
                        # Find new cloudy points  
                        new_cloud = np.logical_and(cloud, \
                                            np.logical_not(mask[it-1,:]) )
                        where_newcloud = np.where(new_cloud)[0]
                        
                        not_new_cloud = np.logical_and(cloud, mask[it-1,:])
                        where_not_newcloud = np.where(not_new_cloud)[0]
#                            print(np.size(where_newcloud),np.size(where_not_newcloud))
                        if debug_mean :
                            print(new_cloud)
                            print(qcl[it-1,new_cloud],qcl[it,new_cloud])
                            
                        if np.size(where_newcloud) > 0 : # Entrainment
                            if debug_mean : print('Entraining air')
                            
                            class_set = where_obj_ptrs[where_newcloud]
                                       
                            traj_class[it-1, class_set] = \
                                        PRE_CLOUD_ENTR_FROM_ABOVE_BL
                            traj_class[it, class_set] = \
                                        ENTR_FROM_ABOVE_BL

                            if debug_mean : 
#                                print("From BL",class_set_from_bl)
                                print("From above BL",class_set)
                                input("Press Enter to continue...") 

                            
                        # Set traj_class flag for those points in cloud.              
                        if np.size(where_not_newcloud) > 0 : # Entrainment
                            if debug_mean : print("Setting Cloud",it,np.size(where_not_newcloud))
                            class_set = where_obj_ptrs[where_not_newcloud]
                        
                            if in_main_cloud :
                                # Still in cloud contiguous with reference time.
                                traj_class[it, class_set] = CLOUD
                            else :
                                # Must be cloud points that were present before.
                                traj_class[it, class_set] = SUBSEQUENT_CLOUD                           

                    else :
                        if in_main_cloud and not after_cloud_dissipated :
                            # At cloud top.
                            cloud_dissipate_time[iobj] = it
                            cloud_top[iobj] = (np.max(tr[it-1,mask[it-1,:],2])-0.5) \
                                                *self.deltaz
                        class_set = where_obj_ptrs[where_not_cloud]                       
                        traj_class[it, class_set] = DETR_CLOUD                           
                        in_main_cloud = False
                        after_cloud_dissipated = True  
                        
                    # Now what about detraining and detraining air?
                    if np.size(where_not_cloud) > 0 : 
                        class_set = where_obj_ptrs[where_not_cloud]
                        traj_class[it, class_set] = DETR_CLOUD

                        # Find points that have ceased to be cloudy                     
#                        detr_cloud =np.logical_and(not_cloud, mask[it-1,:])
                        
#                        if debug_mean : 
#                            print(detr_cloud)
#                            print(qcl[it-1,detr_cloud],qcl[it,detr_cloud])
                                    
#                        where_detrained = np.where(detr_cloud)[0]
#                        if np.size(where_detrained) > 0 : # Detrainment
#                            if debug_mean : print('Detraining air')
                                
#                            class_set = where_obj_ptrs[where_detrained] 
#                            traj_class[it:, class_set] = DETR_CLOUD 
                        
                        # Find point that will be cloud at next step.
############################################################################
                        # Compute mean properties of cloudy points. 

#                obj_ptrs = (self.labels == iobj)
#                where_obj_ptrs = np.where(obj_ptrs)[0]
                for it in range(0, self.end_file+1) :
                    for iclass in range(0,n_class) :
                        
                        trcl = traj_class[it, obj_ptrs]
                        lmask = (trcl == iclass)  
#                        print(np.shape(lmask), \
#                              np.shape(data[it,...]))
                        where_mask = np.where(lmask)[0]
                        if np.size(where_mask) > 0 :
                            mean_prop[it, iobj, r1, iclass] = \
                                np.sum(data[it, lmask, :], axis=0)  
                            msem = mse[it, lmask]                        
#                            print(np.shape(msem), np.max(msem), np.min(msem))
                            mean_prop[it, iobj, r2, iclass] = \
                                np.sum(mse[it, lmask], axis=0)    
                            mean_prop[it, iobj, r3, iclass] = \
                                np.sum(tr[it, lmask,:], axis=0)
                            mean_prop[it, iobj, r4, iclass] = \
                                np.size(where_mask)
                    
            else :
                print("Illegal Version")
                return 
            
        if version < 3 :
            m = (mean_cloud_prop[:,:,r4]>0)    
            for ii in range(nvars+4) : 
                mean_cloud_prop[:,:,ii][m] = mean_cloud_prop[:,:,ii][m] \
                  /mean_cloud_prop[:,:,r4][m]
                  
            m = (mean_entr_prop[:,:,r4]>0)    
            for ii in range(nvars+4) : 
                mean_entr_prop[:,:,ii][m] = mean_entr_prop[:,:,ii][m] \
                  /mean_entr_prop[:,:,r4][m]
                  
            m = (mean_entr_bot_prop[:,:,r4]>0)    
            for ii in range(nvars+4) : 
                mean_entr_bot_prop[:,:,ii][m] = mean_entr_bot_prop[:,:,ii][m] \
                  /mean_entr_bot_prop[:,:,r4][m]
                  
            m = (mean_detr_prop[:,:,r4]>0)    
            for ii in range(nvars+4) : 
                mean_detr_prop[:,:,ii][m] = mean_detr_prop[:,:,ii][m] \
                  /mean_detr_prop[:,:,r4][m]
                  
            mean_properties = {"cloud":mean_cloud_prop, \
                               "entr":mean_entr_prop, \
                               "entr_bot":mean_entr_bot_prop, \
                               "detr":mean_detr_prop, \
                               "class":traj_class, \
                               "first cloud base": first_cloud_base, \
                               "min cloud base":min_cloud_base, \
                               "cloud top":cloud_top, \
                               "cloud_trigger_time":cloud_trigger_time, \
                               "cloud_dissipate_time":cloud_dissipate_time, \
                               "version":version, \
                               }
        elif version == 3:
            
            nplt = 72
            for it in range(np.shape(mean_prop)[0]) :
                s = '{:3d}'.format(it)
                for iclass in range(0,n_class) :
                    s = s+'{:4d} '.format(mean_prop[it, nplt, r4, iclass].astype(int))
                s = s+'{:6d} '.format(np.sum(mean_prop[it, nplt, r4, :].astype(int)))
                print(s)
                
            for it in range(np.shape(mean_prop)[0]) :
                s = '{:3d}'.format(it)
                for iclass in range(0,n_class) :
                    s = s+'{:10f} '.format(mean_prop[it, nplt, nvars, iclass]/1E6)
                s = s+'{:12f} '.format(np.sum(mean_prop[it, nplt, nvars, :])/1E6)
                print(s)
            
            for iclass in range(0,n_class) :
                m = (mean_prop[:, :, r4, iclass]>0)   
                for ii in range(nvars+4) : 
                    mean_prop[:, :, ii, iclass][m] /= \
                        mean_prop[:, :, r4, iclass][m]
                        
#            PRE_CLOUD_ENTR_FROM_BL = 1
#            PRE_CLOUD_ENTR_FROM_ABOVE_BL = 2
#            PREVIOUS_CLOUD = 3
#            CLOUD = 4
#            ENTR_FROM_BL = 5
#            ENTR_FROM_ABOVE_BL = 6
#            DETR_CLOUD = 7 
#            SUBSEQUENT_CLOUD = 8
            mean_properties = {"unclassified":mean_prop[:,:,:, 0], \
                               "pre_cloud_bl":mean_prop[:,:,:, PRE_CLOUD_ENTR_FROM_BL], \
                               "pre_cloud_above_bl":mean_prop[:,:,:, PRE_CLOUD_ENTR_FROM_ABOVE_BL], \
                               "previous_cloud":mean_prop[:,:,:, PREVIOUS_CLOUD], \
                               "cloud":mean_prop[:,:,:, CLOUD], \
                               "entr":mean_prop[:,:,:, ENTR_FROM_ABOVE_BL], \
                               "entr_bot":mean_prop[:,:,:, ENTR_FROM_BL], \
                               "detr":mean_prop[:,:,:, DETR_CLOUD], \
                               "subsequent_cloud":mean_prop[:,:,:, SUBSEQUENT_CLOUD], \
                               "class":traj_class, \
                               "first cloud base": first_cloud_base, \
                               "min cloud base":min_cloud_base, \
                               "cloud top":cloud_top, \
                               "cloud_trigger_time":cloud_trigger_time, \
                               "cloud_dissipate_time":cloud_dissipate_time, \
                               "version":version, \
                               }

        return mean_properties
                      
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

def get_sup_obj(sup, t, o) :
    slist = list()
    for s in sup :
        l = (s[:,0] == t)
        st = s[l,:]
        if len(st) >0 :
            st1 = st[st[:,1]==o]
            if len(st1) > 0 : 
#                print(s,st,st1)
                slist.append(s)
    return slist
    
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
    errtol_iter = 1E-4
    errtol = 5E-3
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
    # Wrap not yet implemented
    
    t1 = np.logical_and( \
        b_test[0,0] >= b_set[...,0,0] , b_test[0,0] <= b_set[...,1,0])
    t2 = np.logical_and( \
        b_test[1,0] >= b_set[...,0,0] , b_test[1,0] <= b_set[...,1,0])
    t3 = np.logical_and( \
        b_test[0,0] <= b_set[...,0,0] , b_test[1,0] >= b_set[...,1,0])
    t4 = np.logical_and( \
        b_test[0,0] >= b_set[...,0,0] , b_test[1,0] <= b_set[...,1,0])
    x_overlap =np.logical_or(np.logical_or( t1, t2), np.logical_or( t3, t4) )

#    print(x_overlap)
    x_ind = np.where(x_overlap)[0]
#    print(x_ind)
    t1 = np.logical_and( \
        b_test[0,1] >= b_set[x_ind,0,1] , b_test[0,1] <= b_set[x_ind,1,1])
    t2 = np.logical_and( \
        b_test[1,1] >= b_set[x_ind,0,1] , b_test[1,1] <= b_set[x_ind,1,1])
    t3 = np.logical_and( \
        b_test[0,1] <= b_set[x_ind,0,1] , b_test[1,1] >= b_set[x_ind,1,1])
    t4 = np.logical_and( \
        b_test[0,1] >= b_set[x_ind,0,1] , b_test[1,1] <= b_set[x_ind,1,1])
    y_overlap = np.logical_or(np.logical_or( t1, t2), np.logical_or( t3, t4) )
    
    y_ind = np.where(y_overlap)[0]
    
    return x_ind[y_ind]
    
