# -*- coding: utf-8 -*-
import os

from netCDF4 import Dataset
import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans
from scipy.optimize import minimize
import random as rnd

from trajectories.trajectory_compute import (Trajectories,
                                             find_time_in_files,
                                             box_overlap_with_wrap,
                                             )
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
                        rep += f"{obj} "
                    if len(rep) > 0 :
                        print(f"master reference time {master_ref} offset time: {t_off} object: {iobj} it_back: {it_back} matching obj: {rep}")
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
                rep = f" "
                for mo, mot, mint in matching_objects[i] :
                    rep += f"(matching object {mo}, type {mot}, %overlap {mint:02d})"
                if len(rep) > 0 :
                    print(f"Master reference time {master_ref} object {iobj} offset time {t_off} matching obj {rep}")

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
