# -*- coding: utf-8 -*-
import os

from netCDF4 import Dataset
import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans
from scipy.optimize import minimize
import random as rnd
#import matplotlib.pyplot as plt

L_vap = 2.501E6
Cp = 1005.0
L_over_cp = L_vap / Cp
R_air = 287.058
r_over_cp = R_air/Cp
grav = 9.81
mol_wt_air        =           28.9640
mol_wt_water      =           18.0150
epsilon           = mol_wt_water/mol_wt_air
c_virtual         = 1.0/epsilon-1.0

var_properties = {"u":[True,False,False],\
                  "v":[False,True,False],\
                  "w":[False,False,True],\
                  "th":[False,False,False],\
                  "p":[False,False,False],\
                  "q_vapour":[False,False,False],\
                  "q_cloud_liquid_mass":[False,False,False],\
                  "tracer_rad1":[False,False,False],\
                  "tracer_rad2":[False,False,False],\
                  }


#debug_unsplit = True
debug_unsplit = False
#debug_label = True
debug_label = False
#debug_mean = True
debug_mean = False
#debug = True
debug = False

cyclic_xy = True
#use_bilin = False
use_bilin = True
interp_order = 1
#use_corrected_positions = False
use_corrected_positions = True

ind=""

class Trajectory_Family :
    """
    Class defining a family of back trajectories.

    This is an ordered list of trajectories with sequential reference times.

    Args:
        files             : ordered list of files used to generate trajectories
        ref_prof_file     : name of file containing reference profile.
        first_ref_time    : Time of reference.
        last_ref_time     : Time of reference.
        back_len          : Time to go back from ref.
        forward_len       : Time to go forward from ref.
        deltax            : Model x grid spacing in m.
        deltay            : Model y grid spacing in m.
        deltaz            : Model z grid spacing in m.
        variable_list=None: List of variable names for data to interpolate to trajectory.
        ref_func          : function to return reference trajectory positions and labels.
        in_obj_func       : function to determine which points are inside an object.
        kwargs            : any additional keyword arguments to ref_func (dict).

    Attributes:
        family(list): List of trajectory objects with required reference times.

    @author: Peter Clark

    """

    def __init__(self, files, ref_prof_file, \
                 first_ref_time, last_ref_time, \
                 back_len, forward_len, \
                 deltax, deltay, deltaz, \
                 ref_func, in_obj_func, kwargs={}, variable_list=None) :
        """
        Create an instance of a family of back trajectories.

        @author: Peter Clark

        """

        self.family = list([])

        first_ref_file, it, delta_t = find_time_in_files(files, \
                                                    first_ref_time)
        dataset=Dataset(files[first_ref_file])
        print("Dataset opened ", files[first_ref_file])
#        print(first_ref_file, it, delta_t)
        if delta_t > 0.0 :
            dataset.close()
            print("dataset closed")
            print(\
            'Starting trajectory family calculation at time {} in file {}'.\
            format(first_ref_time,files[first_ref_file]))
            print('Time step is {}'.format(delta_t))
        else :
            return

        for ref in np.arange(first_ref_time, last_ref_time+delta_t, delta_t):
            print('Trajectories for reference time {}'.format(ref))
            start_time = ref - back_len
            end_time = ref + forward_len

            traj = Trajectories(files, ref_prof_file, \
                                start_time, ref, end_time, \
                                deltax, deltay, deltaz, \
                                ref_func, in_obj_func, kwargs=kwargs, \
                                variable_list=variable_list)
            self.family.append(traj)
#            input("Press a key")
        return

    def matching_object_list(self, master_ref = None, select = None ):
        """
        Method to generate a list of matching objects at all times they match.
        Matching is done using overlap of in_obj boxes.

        Args:
            master_ref=None  : Which trajectory set in family to use to find
                               matching objects. Default is the last set.
            select=None      : Which objects to find matches for.

        Returns:
            dictionary pointing to arrays of mean properties and meta data::

                Dictionary keys:
                  "master_ref"       : master_ref actually used.
                  "objects"          : object numbers selected.
                  "matching_objects" : List described below.

            List with one member for each trajectory set with an earlier
            reference time than master_ref.

            By default, this is all the sets apart
            from the last. Let us say t_off is the backwards offset from the
            reference time, with t_off=0 for the time immediately before the
            reference time.

            Each member is a list with one member for each back trajectory time
            in the master_ref trajectories, containing a list with one member for
            every object in master_ref,comprising an array of integer indices of the
            matching objects in the earlier set that match the reference object
            at the given time. This may be empty if no matches have been found.
            Let us say the it_back measures the time backwards from master_ref, with
            it_back=0 at the reference time.

            Thus, if mol is the matching object list,
            mol[t_off][it_back][iobj] is an array of object ids belonging to
            trajectory set master_ref-(t_off+1), at master_ref trajectory time it_back before
            the master_ref reference time and matching object index iobj a the reference time.

        @author: Peter Clark

        """

        mol = list([])
        if master_ref is None : master_ref = len(self.family) - 1
        traj = self.family[master_ref]
        if select is None : select = np.arange(0, traj.nobjects, dtype = int)
        # Iterate backwards from first set before master_ref.
        for t_off in range(0, master_ref) :
            # We are looking for objects in match_traj matching those in traj.
            match_traj = self.family[master_ref-(t_off+1)]
            # matching_objects[t_off] will be those objects in match_traj
            # which match traj at any time.
            matching_objects = list([])

            # Iterate backwards over all set times from master_ref to start.
            for it_back in range(0,traj.ref+1) :
                # Time to make comparison
                ref_time = traj.ref-it_back
                # Note match_traj.ref in general will equal traj.ref
                # Time in match_traj that matches ref_time
                #       so match_time = master_ref - it_back
                match_time = match_traj.ref + (t_off + 1)- it_back
                matching_object_at_time = list([])

                # Iterate over objects in master_ref.
                for iobj in select :

                    if traj.num_in_obj[ref_time ,iobj] > 0 :
                        b_test = traj.in_obj_box[ref_time, iobj,...]

                        # Find boxes in match_traj at match_time that overlap
                        # in_obj box for iobj at the same time.
                        if (match_time >= 0) & \
                          (match_time < np.shape(match_traj.in_obj_box)[0]) :

                            b_set  = match_traj.in_obj_box[match_time,...]
                            corr_box = box_overlap_with_wrap(b_test, b_set, \
                                traj.nx, traj.ny)
                            valid = (match_traj.num_in_obj[match_time, corr_box]>0)
                            corr_box = corr_box[valid]
                    matching_object_at_time.append(corr_box)
                matching_objects.append(matching_object_at_time)
            mol.append(matching_objects)
        ret_dict = {"master_ref": master_ref, "objects": select, "matching_objects": mol}
        return ret_dict

    def print_matching_object_list(self, master_ref = None, select = None) :
        """
        Method to print matching object list.

        See method matching_object_list.

        @author: Peter Clark

        """

        if master_ref == None : master_ref = len(self.family) - 1
        if select is None : select = np.arange(0, self.family[master_ref].nobjects, \
                                           dtype = int)
        mol_dict = self.matching_object_list(master_ref = master_ref, select = select)
        mol = mol_dict["matching_objects"]
#        for t_off in range(0, 4) :
        for t_off in range(0, len(mol)) :
            matching_objects = mol[t_off]
            for i in range(0, len(select)) :
                iobj = select[i]
                for it_back in range(0, len(matching_objects)) :
#                for it_back in range(0, 4) :
                    rep =""
                    for obj in matching_objects[it_back][i] :
                        rep += "{} ".format(obj)
                    if len(rep) > 0 :
                        print("t_off: {0} iobj: {1} it_back: {2} obj: {3}".\
                                  format(t_off, iobj, it_back, rep))
        return

    def matching_object_list_summary(self, master_ref = None, select = None, \
                                     overlap_thresh=0.02) :
        """
        Method to classify matching objects.

        Args:
            master_ref=None  : Which trajectory set in family to use to find \
             matching objects. Default is the last set.
            overlap_thresh=0.02 Threshold for overlap to be sufficient for inclusion.


        Returns:
            Dictionary pointing to arrays of mean properties and meta data::

                Dictionary keys::
                  "master_ref"              : master_ref actually used.
                  "objects"                 : object numbers selected.
                  "matching_object_summary" : List described below.

            List with one member for each trajectory set with an earlier reference than master_ref.

            By default, this is all the sets apart from the last.

            Each member is a list with one member for each object in master_ref,
            containing a list of objects in the earlier set that match the
            object in master_ref AT ANY TIME.

            Each of these is classified 'Linked' if the object is in the
            max_at_ref list at the earler reference time, otherwise 'Same'.

        @author: Peter Clark

        """

        if master_ref is None : master_ref = len(self.family) - 1
        traj = self.family[master_ref]
        if select is None : select = np.arange(0, traj.nobjects, dtype = int )
        mol_dict = self.matching_object_list(master_ref = master_ref, select=select)
        mol = mol_dict["matching_objects"]
        mols = list([])
        # Iterate backwards from first set before master_ref.
        for t_off, matching_objects in enumerate(mol) :
            match_list = list([])
#            match_time = master_ref-(t_off+1)
                # Note match_traj.ref in general will equal traj.ref
                # Time in match_traj that matches ref_time
                 #       so match_time = master_ref - it_back
            for i, iobj in enumerate(select) :
                objlist = list([])
                otypelist = list([])
                interlist = list([])
                for it_back in range(0, len(matching_objects)) :
                    match_traj = self.family[master_ref-(t_off+1)]
                    ref_time = traj.ref-it_back
                    match_time = match_traj.ref + (t_off + 1)- it_back
#                    print('times',ref_time,match_time)
                    if (match_time >= 0) & (match_time < len(self.family)) :
                        for obj in matching_objects[it_back][i] :
                            if np.size(obj) > 0 :

                                if obj not in objlist :

                                    inter = self.refine_object_overlap(t_off,
                                    it_back, iobj, obj, master_ref = master_ref)

                                    if inter > overlap_thresh :

                                        otype = 'Same'
                                        if obj in match_traj.max_at_ref :
                                            otype = 'Linked'
                                        objlist.append(obj)
                                        otypelist.append(otype)
                                        interlist.append(int(round(inter*100,0)))
                match_list.append(list(zip(objlist,otypelist,interlist)))
            mols.append(match_list)
        ret_dict = {"master_ref": master_ref, "objects": select, "matching_object_summary": mols}
        return ret_dict

    def print_matching_object_list_summary(self, master_ref = None,  select = None, \
                                           overlap_thresh = 0.02) :
        """
        Method to print matching object list summary.

        See method matching_object_list_summary.

        @author: Peter Clark

        """

        if master_ref is None : master_ref = len(self.family) - 1
        if select is None : select = np.arange(0, self.family[master_ref].nobjects, \
                                               dtype = int)
        mols_dict = self.matching_object_list_summary(master_ref = master_ref, select = select, \
                                                 overlap_thresh = overlap_thresh)
        mols = mols_dict["matching_object_summary"]
        for t_off in range(0, len(mols)) :
            matching_objects = mols[t_off]
            for i in range(0, len(select)) :
                iobj = select[i]
                rep =""
                for mo, mot, mint in matching_objects[i] :
                    rep += "({}, {}, {:02d})".format(mo,mot,mint)
                if len(rep) > 0 :
                    print("t_off: {0} iobj: {1} obj: {2} ".\
                                  format(t_off, iobj, rep))

        return

    def find_linked_objects(self, master_ref = None, select = None, \
        overlap_thresh = 0.02) :
        """
        Method to find all objects linked to objects in max_at_ref list in master_ref.

        Args:
            master_ref=None      : Which trajectory set in family to use to find matching objects. Default is the last set.
            overlap_thresh=0.02  : Threshold for overlap to be sufficient for inclusion.

        Returns:
            List with one member for each object in max_at_ref list in master_ref.
            Each member is a an array of triplets containing the time,
            object id and percentage ovelap with next of objects in the
            max_at_ref list of the family at time classified as 'Linked'.

        @author: Peter Clark

        """

        if master_ref is None : master_ref = len(self.family) - 1
        if select is None : select = self.family[master_ref].max_at_ref
        mols_dict = self.matching_object_list_summary(master_ref = master_ref, select = select, \
                                                 overlap_thresh = overlap_thresh)
        mols = mols_dict["matching_object_summary"]
        linked_objects = list([])
        for i in range(len(select)) :
            linked_obj = list([])
            linked_obj.append([master_ref, select[i], 100])
            for t_off in range(0, len(mols)) :
                matching_objects = mols[t_off][i]
                for mo, mot, mint in matching_objects :
                    if mot == 'Linked' :
                        linked_obj.append([master_ref-t_off-1, mo, mint])
            linked_objects.append(np.array(linked_obj))
        return linked_objects

    def print_linked_objects(self, master_ref = None, select = None, \
                             overlap_thresh = 0.02) :
        """
        Method to print linked object list.

        See method find_linked_objects.

        @author: Peter Clark

        """

        if master_ref == None : master_ref = len(self.family) - 1
        if select is None : select = self.family[master_ref].max_at_ref
        linked_objects = self.find_linked_objects(master_ref = master_ref, select=select, \
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

    def find_super_objects(self, master_ref = None, \
        overlap_thresh = 0.02) :
        """
        Method to find all objects linked contiguously to objects in max_at_ref list in master_ref.

        Args:
            master_ref=None           : Which trajectory set in family to use to find matching objects. Default is the last set.
            overlap_thresh=0.02: Threshold for overlap to be sufficient for inclusion.

        Returns:
            super_objects, len_sup

            super_objects is a list with one member for each object in max_at_ref list in master_ref.
            Each member is an array of triplets containing the time,
            object id and percentage ovelap with next of objects in the
            max_at_ref list of the family at time classified as 'Linked'.

            len_sup is an array of the lengths of each super_objects array

        @author: Peter Clark

        """

        def step_obj_back(master_ref, objs) :
            found_super_objs = list([])
            mol_dict = self.matching_object_list(master_ref = master_ref, select=objs)
            mol = mol_dict["matching_objects"]
            t_off = 0
            matching_objects = mol[t_off]
            match_time = master_ref-(t_off+1)
            for i,iobj in enumerate(objs) :
                objlist = list([(master_ref,iobj,100)])
                it_back = 0
#                print(iobj)
#                print(master_ref,match_time, matching_objects[it_back][i])
                for obj in matching_objects[it_back][i] :
                    if np.size(obj) > 0 :

                        inter = self.refine_object_overlap(t_off, \
                                    it_back, iobj, obj, master_ref = master_ref)
#                        print(obj,inter)
                        if inter > overlap_thresh :
                            if obj in self.family[master_ref-(t_off+1)].max_at_ref :
                                objlist.append((match_time,obj,\
                                                int(inter*100+0.5)))
#                                print(objlist)

                found_super_objs.append(objlist)
            return found_super_objs

        print("Finding super-objects")
        if master_ref is None : master_ref = len(self.family) - 1
        select = self.family[master_ref].max_at_ref
        super_objects = list([])
        incomplete_objects = list([])
        incomplete_object_ids = np.array([],dtype=int)
        for newref in range(master_ref,0,-1) :
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

    def refine_object_overlap(self, t_off, time, obj, mobj, master_ref=None) :
        """
        Method to estimate degree of overlap between two trajectory objects.
            Reference object is self.family trajectory object obj at time master_ref-time
            Comparison object is self.family trajectory object mobj
            at same true time, reference time master_ref-(t_off+1).

        Args:
            t_off(integer)    : Reference object is at time index master_ref-time
            time(integer)     : Comparison object is at time index master_ref-(t_off+1)
            obj(integer)      : Reference object id
            mobj(integer)     : Comparison object if.
            master_ref=None(integer) : default is last set in family.

        Returns:
            Fractional overlap

        @author: Peter Clark

        """

        if master_ref == None : master_ref = len(self.family) - 1
#        print(t_off, time, obj, mobj)
#        print("Comparing trajectories {} and {}".format(master_ref, master_ref-(t_off+1)))

        def extract_obj_as1Dint(fam, master_ref, time, obj) :
            traj = fam[master_ref]
            tr_time = traj.ref-time
#            print("Time in {} is {}, {}".format(master_ref, tr_time, traj.times[tr_time]))
            obj_ptrs = (traj.labels == obj)
#            print('extr',master_ref, time, tr_time)
            mask, objvar = traj.in_obj_func(traj, tr_time, obj_ptrs, \
                                            **traj.ref_func_kwargs)
#            mask = mask[tr_time, obj_ptrs]
#            print(np.shape(mask))
            tr = (traj.trajectory[tr_time, obj_ptrs, ... ] + 0.5).astype(int)
#            print(np.shape(tr))
            tr = tr[mask,:]
#            for i in range(2) : print(np.min(tr[:,i]),np.max(tr[:,i]))
            tr1D = np.unique(tr[:,0] + traj.nx * (tr[:,1] + traj.ny * tr[:,2]))
            return tr1D
#        traj = self.family[master_ref]
#        match_traj = self.family[master_ref-(t_off+1)]
#        ref_time = traj.ref-it_back
#        match_time = match_traj.ref + (t_off + 1)- it_back


        tr1D  = extract_obj_as1Dint(self.family, master_ref,           \
                                    time, obj)
        trm1D = extract_obj_as1Dint(self.family, master_ref-(t_off+1), \
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

class Trajectories :
    """
    Class defining a set of back trajectories with a given reference time.

    This is an ordered list of trajectories with sequential reference times.

    Parameters
    ----------
    files              : ordered list of strings.
        Files used to generate trajectories
    ref_prof_file      : string
        name of file containing reference profile.
    start_time         : int
        Time for origin of back trajectories.
    ref                : int
        Reference time of trajectories.
    end_time           : int
        Time for end of forward trajectories.
    deltax             : float
        Model x grid spacing in m.
    deltay             : float
        Model y grid spacing in m.
    deltaz             : float
        Model z grid spacing in m.
    variable_list=None : List of strings.
        Variable names for data to interpolate to trajectory.
    ref_func           : function 
        Return reference trajectory positions and labels.
    in_obj_func        : function 
        Determine which points are inside an object.
    kwargs             : dict
        Any additional keyword arguments to ref_func.

    Attributes
    ----------
    refprof            : Dict 
        Reference profile::   
        
            'rho' (array)  : Reference profile of density.
            'p' (array)    : Reference profile of pressure.
            'th' (array)   : Reference profile of potential temperature.
            'pi' (array)   : Reference profile of Exner pressure.
    data               : float array [nt, m, n] 
        Data associated with trajectory::
        
            nt is total number of times,
            m the number of trajectory points at a given time and
            n is the number of variables in variable_list.
    trajectory: Array [nt, m, 3] where the last index gives x,y,z
    traj_error: Array [nt, m, 3] with estimated error in trajectory.
    traj_times: Array [nt] with times corresponding to trajectory.
    labels: Array [m] labelling points with labels 0 to nobjects-1.
    nobjects: Number of objects.
    coords             : Dict
        Model grid info::   
        
            'xcoord': grid xcoordinate of model space.
            'ycoord': grid ycoordinate of model space.
            'zcoord': grid zcoordinate of model space.
            'ycoord': grid ycoordinate of model space.
            'zcoord': grid zcoordinate of model space.
            'deltax': Model x grid spacing in m.
            'deltay': Model y grid spacing in m.
            'z'     : Model z grid m.
            'zn'    : Model z grid m.        
    nx: int
        length of xcoord
    ny: int
        length of ycoord
    nz: int
        length of zcoord

    deltat: float
        time spacing in trajectories.
    ref_func           : function to return reference trajectory positions and labels.
    in_obj_func        : function to determine which points are inside an object.
    ref_func_kwargs: dict
        Any additional keyword arguments to ref_func (dict).
    files: Input file list.
    ref: Index of reference time in trajectory array.
    end: Index of end time in trajectory array. (start is always 0)
    ntimes: Number of times in trajectory array.
    npoints: Number of points in trajectory array.
    variable_list: variable_list corresponding to data.
    data_mean: mean of in_obj points data.
    num_in_obj: number of in_obj points.
    centroid: centroid of in_objy points
    bounding_box: box containing all trajectory points.
    in_obj_box: box containing all in_obj trajectory points.
    max_at_ref: list of objects which reach maximum LWC at reference time.
    
    """

    def __init__(self, files, ref_prof_file, start_time, ref, end_time,
                 deltax, deltay, deltaz, ref_func, in_obj_func,
                 kwargs={}, variable_list=None ) :

        """
        Create an instance of a set of trajectories with a given reference.

        This is an ordered list of trajectories with sequential reference times.

        @author: Peter Clark

        """

        if variable_list == None :
            variable_list = { \
                  "u":r"$u$ m s$^{-1}$", \
                  "v":r"$v$ m s$^{-1}$", \
                  "w":r"$w$ m s$^{-1}$", \
                  "th":r"$\theta$ K", \
                  "p":r"Pa", \
                  "q_vapour":r"$q_{v}$ kg/kg", \
                  "q_cloud_liquid_mass":r"$q_{cl}$ kg/kg", \
                  }

        dataset_ref = Dataset(ref_prof_file)

        rhoref = dataset_ref.variables['rhon'][-1,...]
        pref = dataset_ref.variables['prefn'][-1,...]
        thref = dataset_ref.variables['thref'][-1,...]
        piref = (pref[:]/1.0E5)**r_over_cp

        self.refprof = {'rho': rhoref,
                        'p'  : pref,
                        'pi'  : piref,
                        'th': thref}

        self.data, trajectory, self.traj_error, self.times, self.ref, \
        self.labels, self.nobjects, self.coords, self.deltat = \
        compute_trajectories(files, start_time, ref, end_time,
                             deltax, deltay, deltaz, variable_list.keys(),
                             self.refprof,
                             ref_func, kwargs=kwargs)

        self.ref_func=ref_func
        self.in_obj_func=in_obj_func
        self.ref_func_kwargs=kwargs
        self.files = files
#        self.ref   = (ref-start_time)//self.deltat
        self.end = len(self.times)-1
        self.ntimes = np.shape(trajectory)[0]
        self.npoints = np.shape(trajectory)[1]
        self.nx = np.size(self.coords['xcoord'])
        self.ny = np.size(self.coords['ycoord'])
        self.nz = np.size(self.coords['zcoord'])
        self.variable_list = variable_list

        self.trajectory = unsplit_objects(trajectory, self.labels, \
                                          self.nobjects, self.nx, self.ny)

        self.data_mean, self.in_obj_data_mean, self.objvar_mean, \
            self.num_in_obj, \
            self.centroid, self.in_obj_centroid, self.bounding_box, \
            self.in_obj_box = compute_traj_boxes(self, in_obj_func, \
                                                kwargs=kwargs)

        max_objvar = (self.objvar_mean == np.max(self.objvar_mean, axis=0))
        when_max_objvar = np.where(max_objvar)
        self.max_at_ref = when_max_objvar[1][when_max_objvar[0] == self.ref]
        return

    def var(self, v) :
        """
        Method to convert variable name to numerical pointer.

        Args:
            v (string):  variable name.

        Returns:
            Numerical pointer to data in data array.

        @author: Peter Clark

        """

        i = list(self.variable_list.keys()).index(v)

        for ii, vr in enumerate(list(self.variable_list)) :
            if vr==v : break
        if ii != i : print("var issue: ", self.variable_list.keys(), v, i, ii)
        return ii

    def select_object(self, iobj) :
        """
        Method to find trajectory and associated data corresponding to iobj.

        Args:
            iobj(integer) : object id .

        Returns:
            trajectory_array, associated _data.

        @author: Peter Clark

        """

        in_object = (self.labels == iobj)
        obj = self.trajectory[:, in_object, ...]
        dat = self.data[:, in_object, ...]
        return obj, dat

    def __str__(self):
        rep = "Trajectories centred on reference Time : {}\n".\
        format(self.times[self.ref])
        rep += "Times : {}\n".format(self.ntimes)
        rep += "Points : {}\n".format(self.npoints)
        rep += "Objects : {}\n".format(self.nobjects)
        return rep

    def __repr__(self):
        rep = "Trajectory Reference time: {0}, Times:{1}, Points:{2}, Objects:{3}\n".format(\
          self.times[self.ref],self.ntimes,self.npoints, self.nobjects)
        return rep


def dict_to_index(variable_list, v) :
    """
    Method to convert variable name to numerical pointer.

    Args:
        v (string):  variable name.

    Returns:
        Numerical pointer to v in variable list.

    @author: Peter Clark

    """

    ii = list(variable_list.keys()).index(v)

#    for ii, vr in enumerate(list(self.variable_list)) :
#        if vr==v : break
    return ii

def compute_trajectories(files, start_time, ref_time, end_time,
                         deltax, deltay, deltaz,
                         variable_list, refprof, ref_func, kwargs={}) :
    """
    Function to compute forward and back trajectories plus associated data.

    Args:
        files         : Ordered list of netcdf files containing 3D MONC output.
        start_time    : Time corresponding to end of back trajectory.
        ref_time      : Time at which reference objects are defined.
        end_time      : Time corresponding to end of forward trajectory.
        variable_list : List of variables to interpolate to trajectory points.
        refprof        : ref profile.

    Returns:
        Set of variables defining trajectories::

            data_val: Array [nt, m, n] where nt is total number of times,
                m the number of trajectory points at a given time and
                n is the number of variables in variable_list.
            trajectory: Array [nt, m, 3] where the last index gives x,y,z
            traj_error: Array [nt, m, 3] with estimated error in trajectory.
            traj_times: Array [nt] with times corresponding to trajectory.
            labels: Array [m] labelling points with labels 0 to nobjects-1.
            nobjects: Number of objects.
            coords             : Dict containing model grid info.
                'xcoord': grid xcoordinate of model space.
                'ycoord': grid ycoordinate of model space.
                'zcoord': grid zcoordinate of model space.
                'ycoord': grid ycoordinate of model space.
                'zcoord': grid zcoordinate of model space.
                'deltax': Model x grid spacing in m.
                'deltay': Model y grid spacing in m.
                'z'     : Model z grid m.
                'zn'    : Model z grid m.
            deltat: time spacing in trajectories.

    @author: Peter Clark

    """

    print('Computing trajectories from {} to {} with reference {}.'.\
          format(start_time, end_time, ref_time))

    ref_file_number, ref_time_index, delta_t = find_time_in_files(\
                                                        files, ref_time)

    dataset=Dataset(files[ref_file_number])
    print("Dataset opened ", files[ref_file_number])
    theta = dataset.variables["th"]
    ref_times = dataset.variables[theta.dimensions[0]][...]
    print('Starting in file number {}, name {}, index {} at time {}.'.\
          format(ref_file_number, os.path.basename(files[ref_file_number]), \
                 ref_time_index, ref_times[ ref_time_index] ))
    file_number = ref_file_number
    time_index = ref_time_index

    # Find initial positions and labels using user-defined function.
    traj_pos, labels, nobjects = ref_func(dataset, time_index, **kwargs)

    times = ref_times
#    print(time_index)
#    input("Press enter")
    trajectory, data_val, traj_error, traj_times, coords \
      = trajectory_init(dataset, time_index, variable_list,
                        deltax, deltay, deltaz, refprof, traj_pos)
#    input("Press enter")
    ref_index = 0

    print("Computing backward trajectories.")

    while (traj_times[0] > start_time) and (file_number >= 0) :
        time_index -= 1
        if time_index >= 0 :
            print('Time index: {} File: {}'.format(time_index, \
                   os.path.basename(files[file_number])))
            trajectory, data_val, traj_error, traj_times = \
            back_trajectory_step(dataset, time_index, variable_list, refprof,
                  coords, trajectory, data_val, traj_error, traj_times)
            ref_index += 1
        else :
            file_number -= 1
            if file_number < 0 :
                print('Ran out of data.')
            else :
                dataset.close()
                print("dataset closed")
                print('File {} {}'.format(file_number, \
                      os.path.basename(files[file_number])))
                dataset = Dataset(files[file_number])
                print("Dataset opened ", files[file_number])
                theta = dataset.variables["th"]
                times = dataset.variables[theta.dimensions[0]][...]
                time_index = len(times)
    dataset.close()
    print("dataset closed")
# Back to reference time for forward trajectories.
    file_number = ref_file_number
    time_index = ref_time_index
    times = ref_times
    dataset = Dataset(files[ref_file_number])
    print("Dataset opened ", files[ref_file_number])
    print("Computing forward trajectories.")

    while (traj_times[-1] < end_time) and (file_number >= 0) :
        time_index += 1
        if time_index < len(times) :
            print('Time index: {} File: {}'.format(time_index, \
                   os.path.basename(files[file_number])))
            trajectory, data_val, traj_error, traj_times = \
                forward_trajectory_step(dataset, time_index,
                                 variable_list, refprof,
                                 coords, trajectory, data_val, traj_error,
                                 traj_times, vertical_boundary_option=2)
        else :
            file_number += 1
            if file_number == len(files) :
                print('Ran out of data.')
            else :
                dataset.close()
                print("dataset closed")
                print('File {} {}'.format(file_number, \
                      os.path.basename(files[file_number])))
                dataset = Dataset(files[file_number])
                print("Dataset opened ", files[file_number])
                theta = dataset.variables["th"]
                times = dataset.variables[theta.dimensions[0]][...]
                time_index = -1
    dataset.close()
    print("dataset closed")

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

    return data_val, trajectory, traj_error, traj_times, ref_index, \
      labels, nobjects, coords, delta_t

def extract_pos(nx, ny, dat) :
    """
    Function to extract 3D position from data array.

    Args:
        nx        : Number of points in x direction.
        ny        : Number of points in y direction.
        dat       : Array[m,n] where n>=5 if cyclic_xy or 3 if not.

    Returns:
        pos       : Array[m,3]
        n_pvar    : Number of dimensions in input data used for pos.
    """

    global cyclic_xy
    if cyclic_xy :
        n_pvar = 5
        xpos = phase(dat[0],dat[1],nx)
        ypos = phase(dat[2],dat[3],ny)
        pos = np.array([xpos, ypos, dat[4]]).T
    else :
        n_pvar = 3
        pos = np.array([dat[0], dat[1], dat[2]]).T

    return pos, n_pvar

def get_z_zn(dataset, default_z):
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
            z = default_z
        zn = np.zeros_like(z)
        zn[-1] = 0.5 * (z[-1] + z[-2])
        for i in range(len(z)-2,  -1, -1 ):
            zn[i] = 2 * z[i] - zn[i+1]
    return z, zn

def trajectory_init(dataset, time_index, variable_list,
                    deltax, deltay, deltaz,
                    refprof, traj_pos) :
    """
    Function to set up origin of back and forward trajectories.

    Args:
        dataset       : Netcdf file handle.
        time_index    : Index of required time in file.
        variable_list : List of variable names.
        refprof       : Dict with reference theta profile arrays.
        traj_pos      : array[n,3] of initial 3D positions.

    Returns:
        Trajectory variables::

            trajectory     : position of origin point.
            data_val       : associated data so far.
            traj_error     : estimated trajectory errors so far.
            traj_times     : trajectory times so far.
            coords             : Dict containing model grid info.
                'xcoord': grid xcoordinate of model space.
                'ycoord': grid ycoordinate of model space.
                'zcoord': grid zcoordinate of model space.
                'deltax': Model x grid spacing in m.
                'deltay': Model y grid spacing in m.
                'z'     : Model z grid m.
                'zn'    : Model z grid m.

    @author: Peter Clark

    """


    th = dataset.variables['th'][0,...]
    (nx, ny, nz) = np.shape(th)

    xcoord = np.arange(nx ,dtype='float')
    ycoord = np.arange(ny, dtype='float')
    zcoord = np.arange(nz, dtype='float')

    z, zn = get_z_zn(dataset, deltaz * zcoord)

    coords = {
        'xcoord': xcoord,
        'ycoord': ycoord,
        'zcoord': zcoord,
        'deltax': deltax,
        'deltay': deltay,
        'z'     : z,
        'zn'    : zn,
        }

    data_list, varlist, varp_list, time = load_traj_step_data(dataset,
                                                   time_index, variable_list,
                                                   refprof, coords )
    print("Starting at time {}".format(time))

    out = data_to_pos(data_list, varp_list, traj_pos,
                      xcoord, ycoord, zcoord)
    traj_pos_new, n_pvar = extract_pos(nx, ny, out)
#    print data_list

#    data_val = list([])
#    for data in data_list[n_pvar:]:
#        data_val.append(data[logical_pos])
#    data_val=[np.vstack(data_val).T]

    data_val = list([np.vstack(out[n_pvar:]).T])

    if debug :

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
    traj_times = list([time])

#    print 'init'
#    print 'trajectory:',len(trajectory[:-1]), len(trajectory[0]), np.size(trajectory[0][0])
#    print 'traj_error:',len(traj_error[:-1]), len(traj_error[0]), np.size(traj_error[0][0])

    return trajectory, data_val, traj_error, traj_times, coords

def back_trajectory_step(dataset, time_index, variable_list, refprof, \
                         coords, trajectory, data_val, traj_error, traj_times) :
    """
    Function to execute backward timestep of set of trajectories.

    Args:
        dataset        : netcdf file handle.
        time_index     : time index in netcdf file.
        variable_list  : list of variable names.
        refprof        : Dict with reference theta profile arrays.
        coords         : Dict containing 1D arrays giving coordinate spaces of data.
        trajectory     : trajectories so far. trajectory[0] is position of earliest point.
        data_val       : associated data so far.
        traj_error     : estimated trajectory errors to far.
        traj_times     : trajectory times so far.

    Returns:
        Inputs updated to new location::

            trajectory, data_val, traj_error, traj_times

    @author: Peter Clark

    """

    data_list, varlist, varp_list, time = load_traj_step_data(dataset,
                                          time_index, variable_list, refprof,
                                          coords)
    print("Processing data at time {}".format(time))

    (nx, ny, nz) = np.shape(data_list[0])

    traj_pos = trajectory[0]
#    print "traj_pos ", np.shape(traj_pos), traj_pos[:,0:5]

    out = data_to_pos(data_list, varp_list, traj_pos,
                      coords['xcoord'], coords['ycoord'], coords['zcoord'])
    traj_pos_new, n_pvar = extract_pos(nx, ny, out)

    data_val.insert(0, np.vstack(out[n_pvar:]).T)
    trajectory.insert(0, traj_pos_new)
    traj_error.insert(0, np.zeros_like(traj_pos_new))
    traj_times.insert(0, time)

    return trajectory, data_val, traj_error, traj_times

def confine_traj_bounds(pos, nx, ny, nz, vertical_boundary_option=1):

    pos[:,0][ pos[:,0] <   0 ] += nx
    pos[:,0][ pos[:,0] >= nx ] -= nx
    pos[:,1][ pos[:,1] <   0 ] += ny
    pos[:,1][ pos[:,1] >= ny ] -= ny
    if vertical_boundary_option == 1:
        pos[:,2][ pos[:,2] <   0 ]  = 0
        pos[:,2][ pos[:,2] >= nz ]  = nz
    elif vertical_boundary_option == 2:
        lam = 1.0 / 0.5
        pos[:,2][ pos[:,2] <=   1.0 ]  = 1.0 + rnd.expovariate(lam)
        pos[:,2][ pos[:,2] >= (nz-1) ]  = nz-1-rnd.expovariate(lam)

    return pos

def forward_trajectory_step(dataset, time_index, variable_list, refprof,
                            coords,
                            trajectory, data_val, traj_error, traj_times,
                            vertical_boundary_option=1) :
    """
    Function to execute forward timestep of set of trajectories.

    Args:
        dataset        : netcdf file handle.
        time_index     : time index in netcdf file.
        variable_list  : list of variable names.
        refprof        : Dict with reference theta profile arrays.
        coords         : Dict containing 1D arrays giving coordinate spaces of data.
        trajectory     : trajectories so far. trajectory[-1] is position of latest point.
        data_val       : associated data so far.
        traj_error     : estimated trajectory errors to far.
        traj_times     : trajectory times so far.

    Returns:
        Inputs updated to new location::

            trajectory, data_val, traj_error, traj_times

    @author: Peter Clark

    """

    global cyclic_xy
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
    if cyclic_xy :
        n_pvar = 5
    else :
        n_pvar = 3

    data_list, varlist, varp_list, time = load_traj_step_data(dataset,
                                          time_index, variable_list, refprof,
                                          coords)
    print("Processing data at time {}".format(time))

    (nx, ny, nz) = np.shape(data_list[0])

    # traj_pos_next_est is our estimate of the trajectory positions at
    # the next time step.
    # We want the 'where from' at traj_pos_next_est to match
    # the current trajectory positions, traj_pos

    # So, we are solving f(x) = 0
    # where
    # x = traj_pos_next_est
    # f = traj_pos_at_est - traj_pos

    traj_pos = trajectory[-1]

    # First guess - extrapolate from last two positions.

    traj_pos_next_est = 2*trajectory[-1]-trajectory[-2]

    confine_traj_bounds(traj_pos_next_est, nx, ny, nz,
                            vertical_boundary_option=vertical_boundary_option)


#    traj_pos_next_est = 60.0*np.array([1/100., 1/100., 1/40.])[:,None]*traj_pos*data_val[0][:,2]
#    traj_pos_est = trajectory[1]

#    print "traj_pos ", np.shape(traj_pos), traj_pos#[:,0:5]
#    print "traj_pos_prev ",np.shape(trajectory[1]), trajectory[1]#[:,0:5]
    use_mean_abs_error = True
    use_point_iteration = True
    limit_gradient = True
    max_iter = 20
    errtol_iter = 0.05
    errtol = 0.05
    relax_param = 1

    err = 1000.0
    niter = 0
    not_converged = True
    correction_cycle = False
    while not_converged :

        out = data_to_pos(data_list, varp_list, traj_pos_next_est,
                          coords['xcoord'], coords['ycoord'], coords['zcoord'],
                          maxindex = n_pvar)

        traj_pos_at_est, n_pvar = extract_pos(nx, ny, out)

#        print("traj_pos_at_est ", np.shape(traj_pos_at_est), traj_pos_at_est)#[:,0:5]
        diff = traj_pos_at_est - traj_pos

# Deal with wrap around.
        diff_lt_minus_nx = diff[:,0]<(-nx/2)
        traj_pos_at_est[:,0][diff_lt_minus_nx] += nx
        diff[:,0][diff_lt_minus_nx] += nx

        diff_gt_plus_nx = diff[:,0]>=(nx/2)
        traj_pos_at_est[:,0][diff_gt_plus_nx] -= nx
        diff[:,0][diff_gt_plus_nx] -= nx


        diff_lt_minus_ny = diff[:,1]<(-ny/2)
        traj_pos_at_est[:,1][diff_lt_minus_ny] += ny
        diff[:,1][diff_lt_minus_ny] += ny

        diff_gt_plus_ny = diff[:,1]>=(ny/2)
        traj_pos_at_est[:,1][diff_gt_plus_ny] -= ny
        diff[:,1][diff_gt_plus_ny] -= ny


        mag_diff = 0
        for i in range(3):
#            print(np.histogram(diff[:,i]))
#            print(niter, np.max(np.abs(diff[:,i])))
            mag_diff += diff[:,i]**2

#        err = np.amax(mag_diff)
        err_prev = err
        if use_mean_abs_error:
            err = np.mean(np.abs(diff))
        else:
            err = np.sqrt(np.mean(diff*diff))
#        print(niter, err, np.sum(np.abs(diff)>errtol_iter/10.), \
#                          np.sum(np.abs(diff)>errtol_iter))

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

        if err <= errtol_iter or err >= err_prev :
            not_converged = False
            print("Converged in {} iterations with error {}."\
                  .format(niter, err))

        if niter <= max_iter :
#            traj_pos_prev_est = traj_pos_next_est
#            traj_pos_at_prev_est = traj_pos_at_est

            if use_point_iteration or niter == 0 :
                if niter == 0:
                    traj_pos_next_est_prev = traj_pos_next_est.copy()
                    diff_prev = diff.copy()

                increment = np.clip(diff * relax_param, -1, 1)
                traj_pos_next_est = traj_pos_next_est - increment

            else :

                # Attempt at 'Newton-Raphson' wil numerical gradients, or
                # Secant method.
                # At present, this does not converge.

                df = (diff - diff_prev)
                inv_gradient = np.ones_like(df) * relax_param
                use_grad = np.abs(df) > 0.001
                dx = traj_pos_next_est - traj_pos_next_est_prev
                # Deal with wrap around.
                dx_lt_minus_nx = dx[:,0]<(-nx/2)
                dx[:,0][dx_lt_minus_nx] += nx

                dx_gt_plus_nx = dx[:,0]>=(nx/2)
                dx[:,0][dx_gt_plus_nx] -= nx

                dx_lt_minus_ny = dx[:,1]<(-ny/2)
                dx[:,1][dx_lt_minus_ny] += ny

                dx_gt_plus_ny = dx[:,1]>=(ny/2)
                dx[:,1][dx_gt_plus_ny] -= ny

                inv_gradient[use_grad] = np.clip(dx[use_grad] / df[use_grad],
                                         -1, 1)
                print("inv_grad :", np.max(np.abs(inv_gradient)))
                traj_pos_next_est_prev = traj_pos_next_est.copy()
                diff_prev = diff.copy()
                increment = np.clip(diff * inv_gradient, -1, 1)
                traj_pos_next_est = traj_pos_next_est - increment


            niter +=1
        else :
            print('Iterations exceeding {} {}'.format(max_iter, err))
            if err > errtol :
                bigerr = (mag_diff > errtol)
                if np.size(diff[bigerr,:]) > 0 :
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
        confine_traj_bounds(traj_pos_next_est, nx, ny, nz,
                            vertical_boundary_option=vertical_boundary_option)
#        traj_pos_next_est[:,0][ traj_pos_next_est[:,0] <   0 ] += nx
#        traj_pos_next_est[:,0][ traj_pos_next_est[:,0] >= nx ] -= nx
#        traj_pos_next_est[:,1][ traj_pos_next_est[:,1] <   0 ] += ny
#        traj_pos_next_est[:,1][ traj_pos_next_est[:,1] >= ny ] -= ny
#        if vertical_boundary_option == 1 :
#            traj_pos_next_est[:,2][ traj_pos_next_est[:,2] <   0 ]  = 0
#            traj_pos_next_est[:,2][ traj_pos_next_est[:,2] >= nz ]  = nz
#        else :
#            traj_pos_next_est[:,2][ traj_pos_next_est[:,2] <   0 ]  = 0.5
#            traj_pos_next_est[:,2][ traj_pos_next_est[:,2] >= nz ]  = nz-0.5

#    print("No. with z<=1.01 ", np.sum(traj_pos_next_est[:,2] <= 1.01))
    out = data_to_pos(data_list, varp_list, traj_pos_next_est,
                      coords['xcoord'], coords['ycoord'], coords['zcoord'])
    data_val.append(np.vstack(out[n_pvar:]).T)
    trajectory.append(traj_pos_next_est)
    traj_error.append(diff)
    traj_times.append(time)
#    print 'trajectory:',len(trajectory[:-1]), len(trajectory[0]), np.size(trajectory[0][0])
#    print 'traj_error:',len(traj_error[:-1]), len(traj_error[0]), np.size(traj_error[0][0])
    return trajectory, data_val, traj_error, traj_times



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

def whichbox(xvec, x ) :
    """
    Find ix such that xvec[ix]<x<=xvec[ix+1]. Uses numpy.searchsorted.

    Args:
        xvec: Ordered array.
        x: Values to locate in xvec.

    Returns:
        Array of indices.

    @author : Peter Clark
    """

    ix = np.searchsorted(xvec, x, side='left')-1
    ix[ix > (len(xvec)-1)] = len(xvec)-1
    ix[ix < 0] = 0
    return ix

def coord_wrap(c, cmax):
    c[c<0] += cmax
    c[c>=cmax] -= cmax
    return c

def tri_lin_interp(data, varp_list, pos,
                   xcoord, ycoord, zcoord, maxindex=None) :

    """
    Tri-linear interpolation with cyclic wrapround in x and y.

    Args:
        data: list of Input data arrays on 3D grid.
        varp_list: list of grid info lists.
        pos(Array[n,3]): Positions to interpolate to (in grid units) .
        xcoord: 1D coordinate vector.
        ycoord: 1D coordinate vector.
        zcoord: 1D coordinate vector.

    Returns:
        list of 1D arrays of data interpolated to pos.

    @author : Peter Clark
    """

    nx = len(xcoord)
    ny = len(ycoord)
    nz = len(zcoord)

    x = coord_wrap(pos[:,0], nx)
    y = coord_wrap(pos[:,1], ny)
    z = pos[:,2]

    ix = np.floor(x).astype(int)
    iy = np.floor(y).astype(int)
    iz = np.floor(z).astype(int)

    iz[iz>(nz-2)] -= 1

    xp = (x-ix)
    yp = (y-iy)
    zp = (z-iz)

    wx = [1.0 - xp, xp]
    wy = [1.0 - yp, yp]
    wz = [1.0 - zp, zp]

    if use_corrected_positions :

        x_u = coord_wrap(pos[:,0] - 0.5, nx)
        y_v = coord_wrap(pos[:,1] - 0.5, ny)
        z_w = pos[:,2] - 0.5

        ix_u = np.floor(x_u).astype(int)
        iy_v = np.floor(y_v).astype(int)
        iz_w = np.floor(z_w).astype(int)
        iz_w[iz_w>(nz-2)] -= 1

        xp_u = (x_u-ix_u)
        yp_v = (y_v-iy_v)
        zp_w = (z_w-iz_w)

        wx_u = [1.0 - xp_u, xp_u]
        wy_v = [1.0 - yp_v, yp_v]
        wz_w = [1.0 - zp_w, zp_w]

    else:

        x_u = x
        y_v = y
        z_w = z

        ix_u = ix
        iy_v = iy
        iz_w = iz

        xp_u = xp
        yp_v = yp
        zp_w = zp

        wx_u = wx
        wy_v = wy
        wz_w = wz

#    ix = whichbox(xcoord, x)
#    iy = whichbox(ycoord, y)
#    iz = whichbox(zcoord, z)
#    dx = 1.0
#    dy = 1.0
#    ix = np.floor(x / dx).astype(int)
#    iy = np.floor(y / dy).astype(int)
#    iz = np.floor(z).astype(int)


#    xp = (x-xcoord[ix])/dx
#    yp = (y-ycoord[iy])/dy
#    zp = (z-zcoord[iz])/(zcoord[iz+1]-zcoord[iz])

    if maxindex is None :
        maxindex = len(data)

    output= list([])
    for l in range(maxindex) :
        output.append(np.zeros_like(x))
#    t = 0

    for i in range(0,2):
#        wi = wx[i]
        ii = (ix + i) % nx
        if  use_corrected_positions :
            ii_u = (ix_u + i) % nx
        else :
            ii_u = ii

        for j in range(0,2):

#            wi_wj = wx[i] * wy[j]
            jj = (iy + j) % ny

            if  use_corrected_positions :
                jj_v = (iy_v + j) % ny
#                wi_u_wj = wx_u[i] * wy[j]
#                wi_wj_v = wx[i] * wy_v[j]
            else :
                jj_v = jj
#                wi_u_wj = wi_wj
#                wi_wj_v = wi_wj

            for k in range(0,2):

#                w = wi_wj * wz[k]
                kk = iz + k

                if  use_corrected_positions :
                    kk_w = iz_w + k
#                    w_u = wi_u_wj * wz[k]
#                    w_v = wi_wj_v * wz[k]
#                    w_w = wi_wj * wz_w[k]
                else :
                    kk_w = kk
#                    w_u = w
#                    w_v = w
#                    w_w = w

#                t += w
                for l in range(maxindex) :
                    if varp_list[l][0]:
                        II = ii_u
                        WX = wx_u
                    else:
                        II = ii
                        WX = wx

                    if varp_list[l][1]:
                        JJ = jj_v
                        WY = wy_v
                    else:
                        JJ = jj
                        WY = wy

                    if varp_list[l][2]:
                        KK = kk_w
                        WZ = wz_w
                    else:
                        KK = kk
                        WZ = wz

#                    print(l, variable_list[l])
                    output[l] +=  data[l][II, JJ, KK] * WX[i] * WY[j] * WZ[k]

    return output

def data_to_pos(data, varp_list, pos,
                xcoord, ycoord, zcoord, maxindex=None):

    """
    Function to interpolate data to pos.

    Args:
        data      : list of data array.
        varp_list: list of grid info lists.
        pos       : array[n,3] of n 3D positions.
        xcoord,ycoord,zcoord: 1D arrays giving coordinate spaces of data.

    Returns:
        list of arrays containing interpolated data.

    @author: Peter Clark

    """

    global interp_order
    if use_bilin :
        output = tri_lin_interp(data, varp_list, pos,
                                xcoord, ycoord, zcoord,
                                maxindex=maxindex )
    else:
        output= list([])
        for l in range(len(data)) :
#            print 'Calling map_coordinates'
#            print np.shape(data[l]), np.shape(traj_pos)
            out = ndimage.map_coordinates(data[l], pos, mode='wrap', \
                                          order=interp_order)
            output.append(out)
    return output

def load_traj_pos_data(dataset, it) :
    """
    Function to read trajectory position variables from file.
    Args:
        dataset        : netcdf file handle.
        it             : time index in netcdf file.

    Returns:
        List of arrays containing interpolated data.

    @author: Peter Clark

    """


    if 'CA_xrtraj' in dataset.variables.keys() :
        # Circle-A Version
        trv = {'xr':'CA_xrtraj', \
               'xi':'CA_xitraj', \
               'yr':'CA_yrtraj', \
               'yi':'CA_yitraj', \
               'zpos':'CA_ztraj' }
        trv_noncyc = {'xpos':'CA_xtraj', \
                      'ypos':'CA_ytraj', \
                      'zpos':'CA_ztraj' }
    else :
        # Stand-alone Version
        trv = {'xr':'tracer_traj_xr', \
               'xi':'tracer_traj_xi', \
               'yr':'tracer_traj_yr', \
               'yi':'tracer_traj_yi', \
               'zpos':'tracer_traj_zr' }
        trv_noncyc = {'xpos':'tracer_traj_xr', \
                      'ypos':'tracer_traj_yr', \
                      'zpos':'tracer_traj_zr' }


    if cyclic_xy :
        xr = dataset.variables[trv['xr']][it,...]
        xi = dataset.variables[trv['xi']][it,...]

        yr = dataset.variables[trv['yr']][it,...]
        yi = dataset.variables[trv['yi']][it,...]

        zpos = dataset.variables[trv['zpos']]
        zposd = zpos[it,...]
        data_list = [xr, xi, yr, yi, zposd]
        varlist = ['xr', 'xi', 'yr', 'yi', 'zpos']
        varp_list = [var_properties['th']]*5

    else :
        # Non-cyclic option may well not work anymore!
        xpos = dataset.variables[trv_noncyc['xpos']][it,...]
        ypos = dataset.variables[trv_noncyc['ypos']][it,...]
        zpos = dataset.variables[trv_noncyc['zpos']]
        zposd = zpos[it,...]
        data_list = [xpos, ypos, zposd]
        varlist = ['xpos', 'ypos', 'zpos']
        varp_list = [var_properties['th']]*3

# Needed as zpos above is numpy array not NetCDF variable.
#    zpos = dataset.variables['CA_ztraj']
    times  = dataset.variables[zpos.dimensions[0]]

    return data_list, varlist, varp_list, times[it]


def d_by_dx_field(field, dx, varp, xaxis=0):
    """
    Numerically differentiate field in x direction assuming cyclic,
    mofiying grid descriptor.

    Parameters
    ----------
    field : array
        Data array, usualy 3D (x, y, z) but should be general.
    dx : float
        Grid spacing
    varp : List of logicals
        Describes grid offset. True means offset from p on grid in that axis.
    xaxis: int
        Pointer to x dimension - usually 0 but can be set to, e.g., 1 if time-
        dimension included in data.
    Returns
    -------
    None.

    """

    d = field[...]
    newfield = (d - np.roll(d,1,axis=xaxis)) / dx
    varp[0] = not varp[0]
    return newfield, varp

def d_by_dy_field(field, dy, varp, yaxis=1):
    """
    Numerically differentiate field in y direction assuming cyclic,
    mofiying grid descriptor.

    Parameters
    ----------
    field : array
        Data array, usualy 3D (x, y, z) but should be general.
    dx : float
        Grid spacing
    varp : List of logicals
        Describes grid offset. True means offset from p on grid in that axis.
    xaxis: int
        Pointer to y dimension - usually 1 but can be set to, e.g., 2 if time-
        dimension included in data.
    Returns
    -------
    None.

    """

    d = field[...]
    newfield = (d - np.roll(d,1,axis=yaxis)) / dy
    varp[1] = not varp[1]
    return newfield, varp


def d_by_dz_field(field, z, zn, varp):
    """
    Numerically differentiate field in z direction,
    mofiying grid descriptor.

    Parameters
    ----------
    field : array
        Data array, usualy 3D (x, y, z) but should be general.
        Assumes z axis is last dimension.
    dx : float
        Grid spacing
    varp : List of logicals
        Describes grid offset. True means offset from p on grid in that axis.
    xaxis: int
        Pointer to y dimension - usually 1 but can be set to, e.g., 2 if time-
        dimension included in data.
    Returns
    -------
    None.

    """

    zaxis = field.ndim - 1

    d = field[...]
    if varp[zaxis]: # Field is on zn points
        new = (d[..., 1:] - d[...,:-1])/ (zn[1:] - zn[:-1])
        zt = 0.5 * (zn[1:] + zn[:-1])
        newfield, newz = padright(new, zt, axis=zaxis)
    else:  # Field is on z points
        new = (d[..., 1:] - d[...,:-1])/ (z[1:] - z[:-1])
        zt = 0.5 * (z[1:] + z[:-1])
        newfield, newz = padleft(new, zt, axis=zaxis)

    varp[-1] = not varp[-1]
    return newfield, varp

def padleft(f, zt, axis=0) :
    """
    Add dummy field at bottom of nD array

    Args:
        f : nD field
        zt: 1D zcoordinates
        axis=0: Specify axis to extend

    Returns:
        extended field, extended coord
    @author: Peter Clark
    """

    s = list(np.shape(f))
    s[axis] += 1
#    print(zt)
    newfield = np.zeros(s)
    newfield[...,1:]=f
    newz = np.zeros(np.size(zt)+1)
    newz[1:] = zt
    newz[0] = 2*zt[0]-zt[1]
#    print(newz)
    return newfield, newz

def padright(f, zt, axis=0) :
    """
    Add dummy field at top of nD array

    Args:
        f : nD field
        zt: 1D zcoordinates
        axis=0: Specify axis to extend

    Returns:
        extended field, extended coord
    @author: Peter Clark
    """

    s = list(np.shape(f))
    s[axis] += 1
#    print(zt)
    newfield = np.zeros(s)
    newfield[...,:-1] = f
    newz = np.zeros(np.size(zt)+1)
    newz[:-1] = zt
    newz[-1] = 2*zt[-1]-zt[-2]
#    print(newz)
    return newfield, newz

def get_data(source_dataset, var_name, it, refprof, coords) :
    """
    Extract data from source NetCDF dataset, derived and/or processed data.

        Currently supported derived data are::
        
            'th_L'    : Liquid water potential temperature.
            'th_v'    : Virtual potential temperature.
            'q_total' : Total water
            'buoyancy': Bouyancy based on layer-mean theta_v.

        Currently supported operators are::
        
            'v_prime' : Deviation from level mean for any variable v.
            'v_crit'  : Level critical value for any variable v.
            'dv_dx'   : Derivative of v in x direction.
            'dv_dy'   : Derivative of v in y direction.
            'dv_dz'   : Derivative of v in z direction.
        
        Operators must be combined using parentheseses, e.g. 
        'd(d(th_prime)_dx)_dy'.
    
    Args:
        source_dataset: handle of NetCDF dataset.
        var_name: String describing data required.
        it: Time index required.
        refprof: Dict containg reference profile.
        coords: Dict containing model coords.

    Returns:
        variable, variable_grid_properties

    @author: Peter Clark and George Efstathiou

    """

    global ind

    # This is the list of supported operators.
    # The offset is how many characters to strip from front of variable name
    # to get raw variable.
    # e.g. dth_dx needs 1 to lose the first d.

    opdict = {'_dx':1,
              '_dy':1,
              '_dz':1,
              '_prime':0,
              '_crit':0,
              }
    vard = None
    varp = None
#    print(ind,var_name)
    try :
        var = source_dataset[var_name]
#        vardim = var.dimensions
        vard = var[it, ...]
        varp = var_properties[var_name]

#        print(vardim)
        if var_name == 'th' and refprof is not None:
#            print(refprof)
            vard[...] += refprof['th']

    except :
#        print("Data not in dataset")
        if var_name == 'th_L':

            theta, varp = get_data(source_dataset, 'th', it, refprof, coords)
            q_cl, vp = get_data(source_dataset, 'q_cloud_liquid_mass', it,
                                refprof, coords)
            vard = theta - L_over_cp * q_cl / refprof['pi']

        elif var_name == 'th_v':
            theta, varp = get_data(source_dataset, 'th', it, refprof, coords)
            q_v, vp = get_data(source_dataset, 'q_vapour', it, refprof, coords)
            q_cl, vp = get_data(source_dataset, 'q_cloud_liquid_mass', it,
                                refprof, coords)
            vard = theta + refprof['th'] * (c_virtual * q_v - q_cl)
 
        elif var_name == 'q_total':

            q_v, varp = get_data(source_dataset, 'q_vapour', it, refprof,
                                 coords)
            q_cl, vp = get_data(source_dataset, 'q_cloud_liquid_mass', it,
                                refprof, coords)
            vard = q_v + q_cl

        elif var_name == 'buoyancy':

            th_v, varp = get_data(source_dataset, 'th_v', it, refprof, coords)
            mean_thv = np.mean(th_v, axis = (0,1))
            vard = grav * (th_v - mean_thv)/mean_thv

        else:
            if ')' in var_name:
#                print(ind,"Found parentheses")
                v = var_name
                i1 = v.find('(')
                i2 = len(v)-v[::-1].find(')')
                source_var = v[i1+1: i2-1]
                v_op = v[i2:]
            else:
                for v_op, offset in opdict.items():
                    if v_op in var_name:
                        source_var = var_name[offset:var_name.find(v_op)]
                        break

#            print(ind,f"Variable: {source_var} v_op: {v_op}")
#            ind+="    "
            vard, varp = get_data(source_dataset, source_var, it, refprof,
                                      coords)
#            ind = ind[:-4]
            if '_dx' == v_op:
#                print(ind,"Exec _dx")
                vard, varp = d_by_dx_field(vard, coords['deltax'], varp)

            elif '_dy' == v_op:
#                print(ind,"Exec _dy")
                vard, varp = d_by_dy_field(vard, coords['deltay'], varp)

            elif '_dz' == v_op:
#                print(ind,"Exec _dz")
                vard, varp = d_by_dz_field(vard, coords['z'], coords['zn'],
                                           varp)

            elif '_prime' == v_op:
#                print(ind,"Exec _prime")
                vard = vard - np.mean(vard, axis = (0,1))

            elif '_crit' == v_op:
#                print(ind,"Exec _crit")
                std = np.std(vard, axis=(0,1))

                std_int=np.ones(len(std))
                c_crit=np.ones(len(std))
                for iz in range(len(std)) :
                    std_int[iz] = 0.05*np.mean(std[:iz+1])
                    c_crit[iz] = np.maximum(std[iz], std_int[iz])

                vard = np.ones_like(vard) * c_crit



        if vard is None:
            print(f'Variable {var_name} not available.')

    return vard, varp

def load_traj_step_data(dataset, it, variable_list, refprof, coords) :
    """
    Function to read trajectory variables and additional data from file
    for interpolation to trajectory.

    Args:
        dataset        : netcdf file handle.
        it             : time index in netcdf file.
        variable_list  : List of variable names.
        refprof        : Dict with reference theta profile arrays.

    Returns:
        List of arrays containing interpolated data.

    @author: Peter Clark

    """

    global ind
    data_list, var_list, varp_list, time = load_traj_pos_data(dataset, it)


    for variable in variable_list :
#        print 'Reading ', variable
        ind=""
        data, varp = get_data(dataset, variable, it, refprof, coords)

        data_list.append(data)
        varp_list.append(varp)
    return data_list, variable_list, varp_list, time

def phase(vr, vi, n) :
    """
    Function to convert real and imaginary points to location on grid
    size n.

    Args:
        vr,vi  : real and imaginary parts of complex location.
        n      : grid size

    Returns:
        Real position in [0,n)

    @author: Peter Clark

    """

    vr = np.asarray(vr)
    vi = np.asarray(vi)
    vpos = np.asarray(((np.arctan2(vi,vr))/(2.0*np.pi)) * n )
    vpos[vpos<0] += n
    return vpos

def label_3D_cyclic(mask) :
    """
    Function to label 3D objects taking account of cyclic boundary
    in x and y. Uses ndimage(label) as primary engine.

    Args:
        mask: 3D logical array with object mask (i.e. objects are
            contiguous True).

    Returns:
        Object identifiers::

            labs  : Integer array[nx,ny,nz] of labels. -1 denotes unlabelled.
            nobjs : number of distinct objects. Labels range from 0 to nobjs-1.

    @author: Peter Clark

    """

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
                while j < nobjs :
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

def unsplit_object( pos, nx, ny ) :
    """
    Function to gather together points in object separated by cyclic boundaries.
        For example, if an object spans the 0/nx boundary, so some
        points are close to zero, some close to nx, they will be adjusted to
        either go from negative to positive, close to 0, or less than nx to
        greater than. The algorithm tries to group on the larges initial set.
        Uses sklearn.cluster.KMeans as main engine.

    Args:
        pos      : grid positions of points in object.
        nx,ny    : number of grid points in x and y directions.
    Returns:
        Adjusted grid positions of points in object.

    @author: Peter Clark

    """

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
    """
    Function to unsplit a set of objects at a set of times using
    unsplit_object on each.

    Args:
        trajectory     : Array[nt, np, 3] of trajectory points, with nt \
                         times and np points.
        labels         : labels of trajectory points.
        nx,ny   : number of grid points in x and y directions.
    Returns:
        Trajectory array with modified positions.

    @author: Peter Clark

    """

    global debug_unsplit
#    print np.shape(trajectory)
    if nobjects < 2:
        return trajectory

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

def compute_traj_boxes(traj, in_obj_func, kwargs={}) :
    """
    Function to compute two rectangular boxes containing all and in_obj
    points for each trajectory object and time, plus some associated data.

    Args:
        traj               : trajectory object.
        in_obj_func        : function to determine which points are inside an object.
        kwargs             : any additional keyword arguments to ref_func (dict).

    Returns:
        Properties of in_obj boxes::

            data_mean        : mean of points (for each data variable) in in_obj.
            in_obj_data_mean : mean of points (for each data variable) in in_obj.
            num_in_obj       : number of in_obj points.
            traj_centroid    : centroid of in_obj box.
            in_obj__centroid : centroid of in_obj box.
            traj_box         : box coordinates of each trajectory set.
            in_obj_box       : box coordinates for in_obj points in each trajectory set.

    @author: Peter Clark

    """

    scalar_shape = (np.shape(traj.data)[0], traj.nobjects)
    centroid_shape = (np.shape(traj.data)[0], traj.nobjects, \
                      3)
    mean_obj_shape = (np.shape(traj.data)[0], traj.nobjects, \
                      np.shape(traj.data)[2])
    box_shape = (np.shape(traj.data)[0], traj.nobjects, 2, 3)

    data_mean = np.zeros(mean_obj_shape)
    in_obj_data_mean = np.zeros(mean_obj_shape)
    objvar_mean = np.zeros(scalar_shape)
    num_in_obj = np.zeros(scalar_shape, dtype=int)
    traj_centroid = np.zeros(centroid_shape)
    in_obj_centroid = np.zeros(centroid_shape)
    traj_box = np.zeros(box_shape)
    in_obj_box = np.zeros(box_shape)

    in_obj_mask, objvar = in_obj_func(traj, **kwargs)

    for iobj in range(traj.nobjects):

        data = traj.data[:,traj.labels == iobj,:]
        data_mean[:,iobj,:] = np.mean(data, axis=1)
        obj = traj.trajectory[:, traj.labels == iobj, :]

        traj_centroid[:,iobj, :] = np.mean(obj,axis=1)
        traj_box[:,iobj, 0, :] = np.amin(obj, axis=1)
        traj_box[:,iobj, 1, :] = np.amax(obj, axis=1)

        objdat = objvar[:,traj.labels == iobj]

        for it in np.arange(0,np.shape(obj)[0]) :
            mask = in_obj_mask[it, traj.labels == iobj]
            num_in_obj[it, iobj] = np.size(np.where(mask))
            if num_in_obj[it, iobj] > 0 :
                in_obj_data_mean[it, iobj, :] = np.mean(data[it, mask, :], axis=0)
                objvar_mean[it, iobj] = np.mean(objdat[it, mask])
                in_obj_centroid[it, iobj, :] = np.mean(obj[it, mask, :],axis=0)
                in_obj_box[it, iobj, 0, :] = np.amin(obj[it, mask, :], axis=0)
                in_obj_box[it, iobj, 1, :] = np.amax(obj[it, mask, :], axis=0)
    return data_mean, in_obj_data_mean, objvar_mean, num_in_obj, \
           traj_centroid, in_obj_centroid, traj_box, in_obj_box


def print_boxes(traj) :
    """
        Function to print information provided by compute_traj_boxes stored
        as attributes of traj.

        Args:
            traj : trajectory object.

        @author: Peter Clark

    """

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
    """
        Function to compute whether rectangular boxes intersect.

        Args:
            b_test: box for testing array[8,3]
            b_set: set of boxes array[n,8,3]
            nx: number of points in x grid.
            ny: number of points in y grid.

        Returns:
            indices of overlapping boxes

        @author: Peter Clark

    """

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

def file_key(file):
    f1 = file.split('_')[-1]
    f2 = f1.split('.')[0]
    return float(f2)

def find_time_in_files(files, ref_time, nodt = False) :
    """
    Function to find file containing data at required time.
        Assumes file names are of form \*_tt.0\* where tt is model output time.

    Args:
        files: ordered list of files
        ref_time: required time.
        nodt: if True do not look for next time to get delta_t

    Returns:
        Variables defining location of data in file list::

            ref_file: Index of file containing required time in files.
            it: Index of time in dataset.
            delta_t: Interval between data.

    @author: Peter Clark

    """

    file_times = np.zeros(len(files))
    for i, file in enumerate(files) : file_times[i] = file_key(file)

    def get_file_times(dataset) :
        theta=dataset.variables["th"]
#        print(theta)
#        print(theta.dimensions[0])
        t=dataset.variables[theta.dimensions[0]][...]
#        print(t)
        return t

    delta_t = 0.0
    it=-1
    ref_file = np.where(file_times >= ref_time)[0][0]
    while True :
        if ref_file >= len(files) or ref_file < 0 :
            ref_file = None
            break
#        print(files[ref_file])
        dataset=Dataset(files[ref_file])
        print("Dataset opened ", files[ref_file])
        times = get_file_times(dataset)
#        dataset.close()
#        print("dataset closed")
#        print(times)

        if len(times) == 1 :
            dataset.close()
            print("dataset closed")
            it = 0
            if times[it] != ref_time :
                print('Could not find exact time {} in file {}'.\
                  format(ref_time,files[ref_file]))
                ref_file = None
            else :
                if nodt :
                    delta_t = 0.0
                else :
                    print('Looking in next file to get dt.')
                    dataset_next=Dataset(files[ref_file+1])
                    print("Dataset_next opened ", files[ref_file+1])
                    times_next = get_file_times(dataset_next)
                    delta_t = times_next[0] - times[0]
                    dataset_next.close()
                    print("dataset_next closed")
            break

        else : # len(times) > 1
            it = np.where(times == ref_time)[0]
            if len(it) == 0 :
                print('Could not find exact time {} in file {}'.\
                  format(ref_time,ref_file))
                it = np.where(times >= ref_time)[0]
#                print("it={}".format(it))
                if len(it) == 0 :
                    print('Could not find time >= {} in file {}, looking in next.'.\
                      format(ref_time,ref_file))
                    ref_file += 1
                    continue
#            else :
            it = it[0]
            if it == (len(times)-1) :
                delta_t = times[it] - times[it-1]
            else :
                delta_t = times[it+1] - times[it]
            break
    print(\
    "Looking for time {}, returning file #{}, index {}, time {}, delta_t {}".\
          format(  ref_time,ref_file, it, times[it], delta_t) )
    return ref_file, it, delta_t.astype(int)

def compute_derived_variables(traj, derived_variable_list=None) :
    if derived_variable_list is None :
        derived_variable_list = { \
                           "q_total":r"$q_{t}$ kg/kg", \
                           "th_L":r"$\theta_L$ K", \
                           "th_v":r"$\theta_v$ K", \
                           "MSE":r"MSE J kg$^{-1}$", \
                           }
    data_list = list([])
    zn = traj.coords['zn']
    tr_z = np.interp(traj.trajectory[...,2], traj.coords['zcoord'], zn)

    piref_z = np.interp(tr_z,zn,traj.refprof['pi'])
    thref_z = np.interp(tr_z,zn,traj.refprof['th'])

    s = list(np.shape(traj.data))
#    print(s)
    s.pop()
    s.append(len(derived_variable_list))
#    print(s)
    out = np.zeros(s)
#    print(np.shape(out))

    for i,variable in enumerate(derived_variable_list.keys()) :

        if variable == "q_total" :
            data = traj.data[..., traj.var("q_vapour")] + \
                   traj.data[..., traj.var("q_cloud_liquid_mass")]

        if variable == "th_L" :
            data = traj.data[..., traj.var("th")] - \
              L_over_cp * \
              traj.data[..., traj.var("q_cloud_liquid_mass")] \
              / piref_z

        if variable == "th_v" :
            data = traj.data[..., traj.var("th")] + \
                   thref_z * (c_virtual * \
                              traj.data[..., traj.var("q_vapour")] -
                              traj.data[..., traj.var("q_cloud_liquid_mass")])

        if variable == "MSE" :
            data = traj.data[:, :, traj.var("th")] * \
                          Cp * piref_z + \
                          grav * tr_z + \
                          L_vap * traj.data[:, :, traj.var("q_vapour")]

#        print(variable, np.min(data),np.max(data),np.shape(data))

        out[...,i] = data
#        data_list.append(data)
#    out = np.vstack(data_list).T
#    out = np.array(data_list).T
#    print(np.shape(out))
    return derived_variable_list, out


def trajectory_cloud_ref(dataset, time_index, thresh=0.00001) :
    """
    Function to set up origin of back and forward trajectories.

    Args:
        dataset        : Netcdf file handle.
        time_index     : Index of required time in file.
        thresh=0.00001 : Cloud liquid water threshold for clouds.

    Returns:
        Trajectory variables::

            traj_pos       : position of origin point.
            labels         : array of point labels.
            nobjects       : number of objects.

    @author: Peter Clark

    """

#    print(dataset)
#    print(time_index)
#    print(thresh)
    data = dataset.variables["q_cloud_liquid_mass"][time_index, ...]

    xcoord = np.arange(np.shape(data)[0],dtype='float')
    ycoord = np.arange(np.shape(data)[1],dtype='float')
    zcoord = np.arange(np.shape(data)[2],dtype='float')

    logical_pos = cloud_select(data, thresh=thresh)

#    print('q_cl threshold {:10.6f}'.format(np.max(data[...])*0.8))

    mask = np.zeros_like(data)
    mask[logical_pos] = 1

    print('Setting labels.')
    labels, nobjects = label_3D_cyclic(mask)

    labels = labels[logical_pos]

    pos = np.where(logical_pos)

#    print(np.shape(pos))
    traj_pos = np.array( [xcoord[pos[0][:]], \
                          ycoord[pos[1][:]], \
                          zcoord[pos[2][:]]],ndmin=2 ).T


    return traj_pos, labels, nobjects

def in_cloud(traj, *argv, thresh=1.0E-5) :
    """
    Function to select trajectory points inside cloud.
    In general, 'in-object' identifier should return a logical mask and a
    quantitative array indicating an 'in-object' scalar for use in deciding
    reference times for the object.

    Args:
        traj           : Trajectory object.
        *argv
        **kwargs

    Returns:
        Variables defining those points in cloud::

            mask           : Logical array like trajectory array.
            qcl            : Cloud liquid water content array.

    @author: Peter Clark

    """
    data = traj.data
    v = traj.var("q_cloud_liquid_mass")

    if len(argv) == 2 :
        (tr_time, obj_ptrs) = argv
        qcl = data[ tr_time, obj_ptrs, v]
    else :
        qcl = data[..., v]

#    mask = data[...]>(np.max(data[...])*0.6)
    mask = cloud_select(qcl, thresh=thresh)
    return mask, qcl

def cloud_select(qcl, thresh=1.0E-5) :
    """
    Simple function to select in-cloud data;
    used in in_cloud and trajectory_cloud_ref for consistency.
    Written as a function to set structure for more complex object identifiers.

    Args:
        qcl            : Cloud liquid water content array.
        thresh=0.00001 : Cloud liquid water threshold for clouds.

    Returns:
        Logical array like trajectory array selecting those points inside cloud.

    @author: Peter Clark

    """

    mask = qcl >= thresh
    return mask


def trajectory_cstruct_ref(dataset, time_index, thresh=0.00001, 
                           find_objects=False) :
    """
    Function to set up origin of back and forward trajectories.

    Args:
        dataset        : Netcdf file handle.
        time_index     : Index of required time in file.
        thresh=0.00001 : Cloud liquid water threshold for clouds.
        find_objects=False : Enable idenfification of distinct objects.

    Returns:
        Trajectory variables::

            traj_pos       : position of origin point.
            labels         : array of point labels.
            nobjects       : number of objects.

    @author: George Efstathiou and Peter Clark

    """

#    print(dataset)
#    print(time_index)
#    print(thresh)
    qcl  = dataset.variables["q_cloud_liquid_mass"][time_index, ...]
    w = dataset.variables["w"][time_index, ...]
    zr = dataset.variables["tracer_traj_zr"][time_index, ...]
    tr1 = dataset.variables["tracer_rad1"][time_index, ...]

    xcoord = np.arange(np.shape(qcl)[0],dtype='float')
    ycoord = np.arange(np.shape(qcl)[1],dtype='float')
    zcoord = np.arange(np.shape(qcl)[2],dtype='float')
    
  
    logical_pos = cstruct_select(qcl, w, zr, tr1, thresh=thresh, ver=6)

    # print('rad threshold {:10.6f}'.format(np.max(data[...])*0.8))

    mask = np.zeros_like(qcl)
    mask[logical_pos] = 1

    if find_objects:
        print('Setting labels.')
        labels, nobjects = label_3D_cyclic(mask)
        print('{} objects found.'.format(nobjects))

    else:
        nobjects = 1
        labels = -1 * np.ones_like(mask)
        labels[logical_pos] = 0

    labels = labels[logical_pos]

    pos = np.where(logical_pos)

#    print(np.shape(pos))
    traj_pos = np.array( [xcoord[pos[0][:]], \
                          ycoord[pos[1][:]], \
                          zcoord[pos[2][:]]], ndmin=2 ).T


    return traj_pos, labels, nobjects

def in_cstruc(traj, *argv, **kwargs) :
    """
    Function to select trajectory points inside coherent structure.
    In general, 'in-object' identifier should return a logical mask and a
    quantitative array indicating an 'in-object' scalar for use in deciding
    reference times for the object.

    Args:
        traj           : Trajectory object.
        *argv
        **kwargs

    Returns:
        Variables defining those points in cloud::

            mask           : Logical array like trajectory array.
            qcl            : Cloud liquid water content array.

    @author: George Efstathiou and Peter Clark

    """
    data = traj.data
    v = traj.var("q_cloud_liquid_mass")
    w_v = traj.var("w")
    z_p = traj.var("tracer_traj_zr") # This seems to be a hack - i.e. wrong.
    tr1_p = traj.var("tracer_rad1")

    if 'thresh' in kwargs:
        thresh = kwargs['thresh']
    else :
        thresh = 1.0E-5

    if len(argv) == 2 :
        (tr_time, obj_ptrs) = argv
        qcl = data[ tr_time, obj_ptrs, v]
        w = data[ tr_time, obj_ptrs, w_v]
        tr1 = data[ tr_time, obj_ptrs, tr1_p]
    else :
        qcl = data[..., v]
        w = data[..., w_v]
        zpos = data[..., z_p]
        tr1 = data[ ..., tr1_p]

    mask = cstruct_select(qcl, w, zpos, tr1, thresh=thresh, ver=6)
    return mask, qcl

def cstruct_select(qcl, w, zpos, tr1, z_lev1=1, z_lev2=3, thresh=1.0E-5, ver=6) :
    """
    Simple function to select in-coherent structure data;
    used in in_cstruct and trajectory_cstruct_ref for consistency.
    Written as a function to set structure for more complex object identifiers.

    Args:
        qcl            : Cloud liquid water content array.
        w              : Vertical velocity array.
        zpos           : Height array.
        tr1            : tracer 1 array.
        tr_i            : tracer i array (if present have to be added).
        thresh=0.00001 : Cloud liquid water threshold for clouds.
        z_lev1         : Height limit (lowest level) (=1)
        z_lev2         : Height limit (highest level) (=3)

    Returns:
        Logical array like trajectory array selecting those points which are:
        
    Version:
        1. Cloudy
        2. Active cloudy
        3. Non-cloudy below level z_lev2
        4. Above level z_lev1 and below z_lev2
        5. Coherent structures
        6. Every other point in domain (Need to set find_objects=False)

    @author: George Efstathiou and Peter Clark

    """
     
    if ver == 1 :
        mask = qcl >= thresh
    elif ver == 2:
        mask = ((qcl >= thresh) and (w > 0.0))
    elif ver == 3 :
        mask = ((qcl < thresh) and (zpos < z_lev2))
    elif ver == 4 :
        mask = np.logical_and( zpos > z_lev1, zpos < z_lev2 )
    elif ver == 5 :
        tr1_pr = tr1 - np.mean(tr1, axis = (0,1))
        std = np.std(tr1, axis=(0,1))
        std_int=np.ones(len(std))
        c_crit=np.ones(len(std))
        for iz in range(len(std)) :
            std_int[iz] = 0.05*np.mean(std[:iz+1])
            c_crit[iz] = np.maximum(std[iz], std_int[iz])
        tr_crit = np.ones_like(tr1) * c_crit
        mask = np.logical_and( tr1_pr > tr_crit, w > 0.0)
    else :
        v_shape=qcl.shape
        mask_pos = np.zeros((v_shape[0],v_shape[1],v_shape[2]))
        mask_pos[::2,::2,::2] = 1
        mask = mask_pos == 1
    return mask
