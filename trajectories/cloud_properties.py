"""
Module cloud_properties.

Created on Thu Aug 27 16:32:10 2020

@author: Peter Clark
"""

import numpy as np
from .trajectory_compute import compute_derived_variables

#debug_mean = True
debug_mean = False


def set_cloud_class(traj, thresh=None, version=1) :
    """
    Compute trajectory class and mean cloud properties.

    Args
    ----
        thresh: Threshold if LWC to define cloud. Default is traj.thresh.
        version: Which version of classification. (Currently only 1).

    Returns
    -------
        Dictionary containing trajectory class points, key and useful derived data::

          Dictionary keys:
          "class"
          "key"
          "first_cloud_base"
          "min_cloud_base"
          "cloud_top"
          "cloud_trigger_time"
          "cloud_dissipate_time"
          "version":version

    """
    if version == 1 :
        PRE_CLOUD_ENTR_FROM_BL = 1
        PRE_CLOUD_ENTR_FROM_ABOVE_BL = 2
        PREVIOUS_CLOUD = 3
        DETR_PREV = 4
        POST_DETR_PREV = 5
        CLOUD = 6
        ENTR_FROM_BL = 7
        ENTR_FROM_ABOVE_BL = 8
#            ENTR_PREV_CLOUD = 7
        DETRAINED = 9
        POST_DETR = 10
        SUBSEQUENT_CLOUD = 11
    else :
        print("Illegal Version")
        return

    class_key = list([\
        "Not set" , \
        "PRE_CLOUD_ENTR_FROM_BL", \
        "PRE_CLOUD_ENTR_FROM_ABOVE_BL", \
        "PREVIOUS_CLOUD", \
        "DETR_PREV", \
        "POST_DETR_PREV", \
        "CLOUD", \
        "ENTRAINED_FROM_BL", \
        "ENTRAINED_FROM_ABOVE_BL", \
        "DETRAINED", \
        "POST_DETR", \
        "SUBS_CLOUD", \
        ])

    if thresh == None : thresh = traj.ref_func_kwargs["thresh"]
    traj_class = np.zeros_like(traj.data[:,:,0], dtype=int)

    min_cloud_base = np.zeros(traj.nobjects)
    first_cloud_base = np.zeros(traj.nobjects)
    cloud_top = np.zeros(traj.nobjects)
    cloud_trigger_time = np.zeros(traj.nobjects, dtype=int)
    cloud_dissipate_time = np.ones(traj.nobjects, dtype=int)*traj.ntimes

    zn = traj.coords['zn']
    tr_z = np.interp(traj.trajectory[:,:,2], np.arange(len(zn)), zn)

    for iobj in range(0,traj.nobjects) :
#        debug_mean = (iobj == 61)
        if debug_mean : print('Processing object {}'.format(iobj))
        obj_ptrs = (traj.labels == iobj)
        where_obj_ptrs = np.where(obj_ptrs)[0]
        tr = traj.trajectory[:, obj_ptrs, :]
        data = traj.data[:, obj_ptrs, :]
        obj_z = tr_z[:,obj_ptrs]

        if debug_mean : print(np.shape(data))

        qcl = data[:,:,traj.var("q_cloud_liquid_mass")]
        mask_qcl = (qcl >= thresh)
#        w = data[:,:,traj.var("w")]
#        mask_w = (w >= 0.1)
#        mask = np.logical_and(mask_qcl, mask_w)
        mask = mask_qcl

        if np.size(np.where(mask[traj.ref,:])[0]) == 0 :
            print("Object {} is not active at reference time.".format(iobj))
            continue

        min_cloud_base[iobj] = np.interp(np.min(tr[mask,2]),
                                         np.arange(len(zn)), zn)

        if debug_mean : print('Version ',version)

        in_main_cloud = True

        for it in range(traj.ref,-1,-1) :

            if debug_mean : print(f"it = {it}")

#            cloud_bottom = traj.in_obj_box[it, iobj, 0, 2]

            cloud = mask[it,:]
            where_cloud = np.where(cloud)[0]

            not_cloud = np.logical_not(cloud)
            where_not_cloud = np.where(not_cloud)[0]

            if it > 0 :

                cloud_at_prev_step = mask[it-1,:]
                not_cloud_at_prev_step = np.logical_not(cloud_at_prev_step)

                new_cloud = np.logical_and(cloud, not_cloud_at_prev_step)
                where_newcloud = np.where(new_cloud)[0]

                not_new_cloud = np.logical_and(cloud, cloud_at_prev_step)
                where_not_newcloud = np.where(not_new_cloud)[0]

            else :
                # First time so no previous data.
                where_newcloud = np.array([])

                not_new_cloud = cloud
                where_not_newcloud = where_cloud

            if np.size(where_cloud) > 0 :
                # There are cloudy points

                if np.size(where_newcloud) > 0 : # Entrainment
                    # New cloudy points
                    if debug_mean : print('Entraining air')

                    # New cloud in air that starts below cloud base.
                    newcl_from_bl = np.logical_and( new_cloud,
                                      obj_z[0, :] < min_cloud_base[iobj])

                    where_newcl_from_bl = np.where(newcl_from_bl)[0]

                    class_set_from_bl = where_obj_ptrs[where_newcl_from_bl]

                    # New cloud in air that starts above cloud base.
                    newcl_from_above_bl = np.logical_and( new_cloud,
                                     obj_z[0, :] >= min_cloud_base[iobj])

                    where_newcl_from_above_bl = np.where(
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

                # Set traj_class flag for those points in cloud.
                if np.size(where_not_newcloud) > 0 : # Not entrainment
                    # Pre-existing cloudy points
                    if debug_mean :
                        print("Setting Cloud", it, np.size(where_not_newcloud))

                    class_set = where_obj_ptrs[where_not_newcloud]

                    if in_main_cloud :
                        # Still in cloud contiguous with reference time.
                        traj_class[it, class_set] = CLOUD
                        if debug_mean :
                            print("CLOUD",class_set)
                    else :
                        # Must be cloud points that were present before.
                        traj_class[it, class_set] = PREVIOUS_CLOUD
                        if debug_mean :
                            print("PREV CLOUD",class_set)

            else : # np.size(where_cloud) == 0

                if in_main_cloud : cloud_trigger_time[iobj] = it+1
                in_main_cloud = False

            # Now what about detraining air?
            if np.size(where_not_cloud) > 0 :

                # Find points that have ceased to be cloudy
                if it > 0 :

                    detr_cloud = np.logical_and(not_cloud,
                                                cloud_at_prev_step)
                    where_detrained = np.where(detr_cloud)[0]

                    if np.size(where_detrained) > 0 : # Detrainment
                        if debug_mean : print('Detraining air')

                        class_set = where_obj_ptrs[where_detrained]

                        if in_main_cloud :

                            traj_class[it, class_set] = DETRAINED

                            if debug_mean :
                                print("Detrained",class_set)

                            for iit in range(it+1,traj.ref) :

                                not_cloud_at_iit = np.logical_not(mask[iit,:])

                                post_detr = np.logical_and(detr_cloud,
                                                           not_cloud_at_iit)

                                where_post_detr = np.where(post_detr)[0]

                                class_set_post_detr = where_obj_ptrs[
                                                      where_post_detr]

                                traj_class[iit, class_set_post_detr] = \
                                    POST_DETR

                                if debug_mean :
                                    print("Post detr",iit, class_set_post_detr)
                        else :

                            traj_class[it, class_set] = DETR_PREV

                            if debug_mean :
                                print("Detrained prev",class_set)

                            for iit in range(it+1,traj.ref) :

                                not_cloud_at_iit = np.logical_not(mask[iit,:])

                                post_detr_prev = np.logical_and(
                                                        detr_cloud,
                                                        not_cloud_at_iit)

                                where_post_detr_prev = np.where(
                                                         post_detr_prev)[0]

                                class_set_post_detr_prev = where_obj_ptrs[
                                                where_post_detr_prev]

                                traj_class[iit, class_set_post_detr_prev] = \
                                    POST_DETR_PREV

                                if debug_mean :
                                    print("Post detr prev", iit,
                                          class_set_post_detr_prev)

        in_main_cloud = True
        after_cloud_dissipated = False

        for it in range(traj.ref+1,traj.end+1) :

            if debug_mean : print("it = {}".format(it))

#            cloud_bottom = traj.in_obj_box[it, iobj, 0, 2]

            cloud = mask[it,:]
            where_cloud = np.where(cloud)[0]

            not_cloud = np.logical_not(cloud)
            where_not_cloud = np.where(not_cloud)[0]

            cloud_at_prev_step = mask[it-1,:]
            not_cloud_at_prev_step = np.logical_not(cloud_at_prev_step)

            new_cloud = np.logical_and(cloud, not_cloud_at_prev_step)
            where_newcloud = np.where(new_cloud)[0]

            not_new_cloud = np.logical_and(cloud, cloud_at_prev_step)
            where_not_newcloud = np.where(not_new_cloud)[0]

            new_not_cloud = np.logical_and(not_cloud, cloud_at_prev_step)
            where_new_not_cloud = np.where(new_not_cloud)[0]

            not_new_not_cloud = np.logical_and(not_cloud, not_cloud_at_prev_step)
            where_not_new_not_cloud = np.where(not_new_not_cloud)[0]

            if np.size(where_newcloud) > 0 : # Entrainment
                if debug_mean : print('Entraining air')
                # New cloudy points

                class_set = where_obj_ptrs[where_newcloud]

                traj_class[it-1, class_set] = \
                            PRE_CLOUD_ENTR_FROM_ABOVE_BL
                traj_class[it, class_set] = \
                            ENTR_FROM_ABOVE_BL

                if debug_mean :
                    print("From above BL",class_set)


            # Set traj_class flag for those points in cloud.
            if np.size(where_not_newcloud) > 0 : # Existing cloud
                if debug_mean : print("Setting Cloud",it,
                                      np.size(where_not_newcloud))
                # Pre-existing cloud

                class_set = where_obj_ptrs[where_not_newcloud]

                if in_main_cloud :
                    # Still in cloud contiguous with reference time.
                    traj_class[it, class_set] = CLOUD
                    if debug_mean : print("Cloud",class_set)
                else :
                    # Must be cloud points that were present before.
                    traj_class[it, class_set] = SUBSEQUENT_CLOUD
                    if debug_mean : print("Subs Cloud",class_set)

            # Now what about detraining and detraining air?
            if np.size(where_new_not_cloud) > 0 :
                class_set = where_obj_ptrs[where_new_not_cloud]
                traj_class[it, class_set] = DETRAINED
                if debug_mean : print("DETRAINED",class_set)

            if np.size(where_not_new_not_cloud) > 0 :
                class_set = where_obj_ptrs[where_not_new_not_cloud]
                traj_class[it, class_set] = POST_DETR
                if debug_mean : print("POST DETRAINED",class_set)

            if np.size(where_cloud) == 0 :
                # There are cloudy points
                if in_main_cloud and not after_cloud_dissipated :
                    # At cloud top.
                    cloud_dissipate_time[iobj] = it
                    cloud_top[iobj] = np.max(obj_z[it-1,cloud_at_prev_step])
                    in_main_cloud = False
                    after_cloud_dissipated = True

    traj_class = {
                   "class": traj_class,
                   "key": class_key,
                   "first_cloud_base": first_cloud_base,
                   "min_cloud_base": min_cloud_base,
                   "cloud_top": cloud_top,
                   "cloud_trigger_time": cloud_trigger_time,
                   "cloud_dissipate_time": cloud_dissipate_time,
                   "version": version,
                 }
    return traj_class



def cloud_properties(traj, traj_cl, thresh=None, use_density = False) :
    """
    Compute trajectory class and mean cloud properties.

    Args
    ----
        traj       : Trajectory object
        traj_cl    : Dict of Classifications of trajectory points
            provided by set_cloud_class function.
        thresh     : Threshold if LWC to define cloud. Default is traj.thresh.

    Returns
    -------
        dictionary pointing to arrays of mean properties and meta data::

            Dictionary keys:
                "overall_mean"
                "unclassified"
                "pre_cloud_bl"
                "pre_cloud_above_bl"
                "previous_cloud"
                "detr_prev"
                "post_detr_prev"
                "cloud"
                "entr_bot"
                "entr"
                "detr"
                "post_detr"
                "subsequent_cloud"
                "cloud_properties"
                "budget_loss"
                "entrainment"
                "derived_variable_list"

    """
    if thresh == None : thresh = traj.ref_func_kwargs["thresh"]
    version = traj_cl["version"]
    if version == 1 :
        n_class = len(traj_cl["key"])
        UNCLASSIFIED = traj_cl["key"].index('Not set')
        PRE_CLOUD_ENTR_FROM_BL = traj_cl["key"].index('PRE_CLOUD_ENTR_FROM_BL')
        PRE_CLOUD_ENTR_FROM_ABOVE_BL = traj_cl["key"].index('PRE_CLOUD_ENTR_FROM_ABOVE_BL')
        PREVIOUS_CLOUD = traj_cl["key"].index('PREVIOUS_CLOUD')
        DETR_PREV = traj_cl["key"].index('DETR_PREV')
        POST_DETR_PREV = traj_cl["key"].index('POST_DETR_PREV')
        CLOUD = traj_cl["key"].index('CLOUD')
        ENTR_FROM_BL = traj_cl["key"].index('ENTRAINED_FROM_BL')
        ENTR_FROM_ABOVE_BL = traj_cl["key"].index('ENTRAINED_FROM_ABOVE_BL')
        DETRAINED = traj_cl["key"].index('DETRAINED')
        POST_DETR = traj_cl["key"].index('POST_DETR')
        SUBSEQUENT_CLOUD = traj_cl["key"].index('SUBS_CLOUD')
    else :
        print("Illegal Version")
        return

    traj_class = traj_cl["class"]

    derived_variable_list, derived_data = compute_derived_variables(traj)

    nvars = np.shape(traj.data)[2]
    ndvars = np.shape(derived_data)[2]
    nposvars = 3
    total_nvars = nvars + ndvars + nposvars + 3

    z_ptr = nvars + ndvars + 2 # Index of variable in mean_prop which is height
    cv_ptr = nvars + ndvars + nposvars + 1
    npts_ptr = cv_ptr + 1 # Index of variable in mean_prop which is
    w_ptr = traj.var('w')
    mse_ptr = nvars + list(derived_variable_list.keys()).index("MSE")
#    qt_ptr = nvars + list(derived_variable_list.keys()).index("q_total")

#    print(nvars, ndvars, nposvars)
    # These possible slices are set for ease of maintenance.
    # Main data variables.
    r_main_var = slice(0,nvars)
    # Derived variables
    r_derv_var = slice(nvars, nvars+ndvars)
    # Position
    r_pos = slice(nvars+ndvars,nvars+ndvars+nposvars)
    # Mass
    r_mass = nvars + ndvars + nposvars
    # Volume
    r_rho = r_mass+ 1
    # Number of points
    r_number = r_rho + 1

    if version == 1 :
        mean_prop = np.zeros([traj.ntimes, traj.nobjects, total_nvars])

        mean_prop_by_class = np.zeros([traj.ntimes, traj.nobjects, \
                                       total_nvars,  n_class+1])
        budget_loss = np.zeros([traj.ntimes-1, traj.nobjects, total_nvars-1])

        # Pointers into cloud_prop array
        CLOUD_HEIGHT = 0
        CLOUD_POINTS = 1
        CLOUD_VOLUME = 2
        n_cloud_prop = 3

        cloud_prop = np.zeros([traj.ntimes, traj.nobjects, n_cloud_prop])

        # Pointers into entrainment array
        TOT_ENTR = 0
        TOT_ENTR_Z = 1
        SIDE_ENTR = 2
        SIDE_ENTR_Z = 3
        CB_ENTR = 4
        CB_ENTR_Z = 5
        DETR = 6
        DETR_Z = 7
        n_entr_vars = 8
        entrainment = -np.ones([traj.ntimes-1, traj.nobjects, n_entr_vars])

        max_cloud_base_area = np.zeros(traj.nobjects)
        max_cloud_base_time = np.zeros(traj.nobjects, dtype = int)
        cloud_base_variables = np.zeros([traj.nobjects, total_nvars-1])

    else :
        print("Illegal Version")
        return


    grid_box_area = traj.coords['deltax'] * traj.coords['deltay']
    grid_box_volume = grid_box_area \
          * np.interp(traj.trajectory[:,:,2],
                      np.arange(len(traj.coords['z'])-1),
                      np.diff(traj.coords['z']))

    rho = np.interp(traj.trajectory[:,:,2],traj.coords['zcoord'],
                    traj.refprof['rho'])

    mass = rho * grid_box_volume

# Compute mean properties of cloudy points.

    for iobj in range(0,traj.nobjects) :
#        debug_mean = (iobj == 61)
        if debug_mean : print('Processing object {}'.format(iobj))

        obj_ptrs = (traj.labels == iobj)

#        where_obj_ptrs = np.where(obj_ptrs)[0]
        tr = traj.trajectory[:, obj_ptrs, :]

# Convert to real space coords.
        tr[:,:,0] = tr[:,:,0] * traj.coords['deltax']
        tr[:,:,1] = tr[:,:,1] * traj.coords['deltay']
        tr[:,:,2] = np.interp(tr[:,:,2], traj.coords['zcoord'],
                            traj.coords['zn'])


        data = traj.data[:, obj_ptrs, :]
        derv_data = derived_data[:, obj_ptrs, :]

        mass_obj = mass[:, obj_ptrs]
        rho_obj = rho[:, obj_ptrs]


#        obj_z = tr_z[:,obj_ptrs]

#        if debug_mean : print(np.shape(data), np.shape(derv_data))

#        qcl = data[:,:,traj.var("q_cloud_liquid_mass")]
#        mask_qcl = (qcl >= thresh)
#        w = data[:,:,traj.var("w")]
#        mask_w = (w >= 0.1)
#        mask = np.logical_and(mask_qcl, mask_w)
#
##        print("Object ",iobj, np.size(np.where(mask[traj.ref,:])[0]))
#
#        if np.size(np.where(mask[traj.ref,:])[0]) == 0 :
#            print("Object {} is not active at reference time.".format(iobj))
#            continue

# Sum properties over classes; division by number of points happens later.
        for it in range(0, traj.end+1) :

            mean_prop[it, iobj, r_main_var] = np.mean(data[it, :, :],
                                                      axis=0)
            mean_prop[it, iobj, r_derv_var] = np.mean(derv_data[it, :, :],
                                                      axis=0)
            mean_prop[it, iobj, r_pos]      = np.mean(tr[it, :, :],
                                                      axis=0)
            mean_prop[it, iobj, r_mass]     = np.mean(mass_obj[it, :],
                                                      axis=0)
            mean_prop[it, iobj, r_rho]      = np.mean(rho_obj[it, :],
                                                      axis=0)
            mean_prop[it, iobj, r_number]   = np.size(data[it, :, 0])

            for iclass in range(0,n_class) :
#                if debug_mean :
#                    print(it, iclass)
                trcl = traj_class[it, obj_ptrs]
                lmask = (trcl == iclass)
                where_mask = np.where(lmask)[0]

                if np.size(where_mask) > 0 :
#                    if debug_mean :
#                        print(data[it, lmask, :])
#                        print(derv_data[it, lmask, :])
                    mean_prop_by_class[it, iobj, r_main_var, iclass] = \
                        np.sum(data[it, lmask, :], axis=0)
                    mean_prop_by_class[it, iobj, r_derv_var, iclass] = \
                        np.sum(derv_data[it, lmask, :], axis=0)
                    mean_prop_by_class[it, iobj, r_pos, iclass] = \
                        np.sum(tr[it, lmask,:], axis=0)
                    mean_prop_by_class[it, iobj, r_mass, iclass] = \
                        np.sum(mass_obj[it, lmask], axis=0)
                    mean_prop_by_class[it, iobj, r_rho, iclass] = \
                        np.sum(rho_obj[it, lmask], axis=0)
                    mean_prop_by_class[it, iobj, r_number, iclass] = \
                        np.size(where_mask)

# Now compute budget and entrainment/detrainment terms.

        delta_t = (traj.times[1:]-traj.times[:-1])

        incloud = np.arange(traj.end + 1, dtype=int)
        incloud = np.logical_and( \
                        incloud >= traj_cl['cloud_trigger_time'][iobj],\
                        incloud <  traj_cl['cloud_dissipate_time'][iobj])

        precloud = np.arange(traj.end + 1, dtype=int)
        precloud = (precloud < traj_cl['cloud_trigger_time'][iobj])
#            print(precloud)
# Cloud volume
        v_main_cloud = mean_prop_by_class[:,iobj,:npts_ptr, CLOUD] + \
                       mean_prop_by_class[:,iobj,:npts_ptr, ENTR_FROM_ABOVE_BL] + \
                       mean_prop_by_class[:,iobj,:npts_ptr, ENTR_FROM_BL]

        n_main_cloud = mean_prop_by_class[:,iobj, npts_ptr, CLOUD] + \
                       mean_prop_by_class[:,iobj, npts_ptr, ENTR_FROM_ABOVE_BL] + \
                       mean_prop_by_class[:,iobj, npts_ptr, ENTR_FROM_BL]

        incl = (n_main_cloud > 0)

#        print(np.shape(z_cloud), np.shape(n_main_cloud),np.shape(incl))
        z_cloud = v_main_cloud[:, z_ptr]
        z_cloud[incl] = z_cloud[incl] / n_main_cloud[incl]

        cloud_volume = v_main_cloud[:, cv_ptr]

        if debug_mean :
            print(traj_cl['cloud_trigger_time'][iobj],traj_cl['cloud_dissipate_time'][iobj])
            print(np.shape(incloud),np.shape(precloud))
            print(mean_prop_by_class[:, iobj, npts_ptr, CLOUD])
            print(precloud)
            print(incloud)
            print(incl)
            print(z_cloud)

        cloud_prop[:,iobj,CLOUD_HEIGHT] = z_cloud
        cloud_prop[:,iobj,CLOUD_POINTS] = n_main_cloud
        cloud_prop[:,iobj,CLOUD_VOLUME] = cloud_volume

##################### Lagrangian budget #####################################

# Budget:
# This timestep
# (In cloud or prev cloud now and not new) +
# (Just entrained at cloud base) +
# (Just entrained above cloud base) +
# (Just detrained)
#
# Where did this come from?
# Previous timestep
# (In cloud or prev cloud now and not new) +
# (Just entrained at cloud base) +
# (Just entrained above cloud base) +
# -(Change in pre cloud entrained at cloud base) +
# -(Change in pre cloud entrained above cloud base) +
# -(Change in post detrained if negative)
# -(Change in post detrained previous cloud if negative)
#
################### This timestep ###########################
#
# In cloud now and not new.
        v_now  = mean_prop_by_class[1:,iobj,:npts_ptr, CLOUD]
        n_now  = mean_prop_by_class[1:,iobj, npts_ptr, CLOUD]

# In prev cloud now and not new.
        v_prev_now  = mean_prop_by_class[1:,iobj,:npts_ptr, PREVIOUS_CLOUD]
        n_prev_now  = mean_prop_by_class[1:,iobj, npts_ptr, PREVIOUS_CLOUD]

# New cloud entrained from cloud base
        v_entr_bot = mean_prop_by_class[1:,iobj,:npts_ptr, ENTR_FROM_BL]
        n_entr_bot = mean_prop_by_class[1:,iobj, npts_ptr, ENTR_FROM_BL]

# New cloud entrained from above cloud base
        v_entr = mean_prop_by_class[1:,iobj,:npts_ptr, ENTR_FROM_ABOVE_BL]
        n_entr = mean_prop_by_class[1:,iobj, npts_ptr, ENTR_FROM_ABOVE_BL]

# Just detrained.
        v_detr = mean_prop_by_class[1:,iobj,:npts_ptr, DETRAINED] + \
                 mean_prop_by_class[1:,iobj,:npts_ptr, DETR_PREV]
        n_detr = mean_prop_by_class[1:,iobj, npts_ptr, DETRAINED] + \
                 mean_prop_by_class[1:,iobj, npts_ptr, DETR_PREV]

################### Previous timestep ###########################
#
# (In cloud or prev cloud now and not new) +
# (Just entrained at cloud base) +
# (Just entrained above cloud base)

        v_prev = mean_prop_by_class[:-1,iobj,:npts_ptr, CLOUD] + \
                 mean_prop_by_class[:-1,iobj,:npts_ptr, PREVIOUS_CLOUD]+ \
                 mean_prop_by_class[:-1,iobj,:npts_ptr, ENTR_FROM_ABOVE_BL] + \
                 mean_prop_by_class[:-1,iobj,:npts_ptr, ENTR_FROM_BL]
        n_prev = mean_prop_by_class[:-1,iobj, npts_ptr, CLOUD] + \
                 mean_prop_by_class[:-1,iobj, npts_ptr, PREVIOUS_CLOUD] + \
                 mean_prop_by_class[:-1,iobj, npts_ptr, ENTR_FROM_ABOVE_BL] + \
                 mean_prop_by_class[:-1,iobj, npts_ptr, ENTR_FROM_BL]

# (Change in pre cloud entrained at cloud base)

        v_entr_pre_bot = mean_prop_by_class[ :-1,iobj,:npts_ptr, PRE_CLOUD_ENTR_FROM_BL] - \
                         mean_prop_by_class[1:,  iobj,:npts_ptr, PRE_CLOUD_ENTR_FROM_BL]
        n_entr_pre_bot = mean_prop_by_class[ :-1,iobj, npts_ptr, PRE_CLOUD_ENTR_FROM_BL] - \
                         mean_prop_by_class[1:,  iobj, npts_ptr, PRE_CLOUD_ENTR_FROM_BL]

# (Change in pre cloud entrained above cloud base)
        v_entr_pre = mean_prop_by_class[ :-1,iobj,:npts_ptr, PRE_CLOUD_ENTR_FROM_ABOVE_BL] - \
                     mean_prop_by_class[1:,  iobj,:npts_ptr, PRE_CLOUD_ENTR_FROM_ABOVE_BL]
        n_entr_pre = mean_prop_by_class[ :-1,iobj, npts_ptr, PRE_CLOUD_ENTR_FROM_ABOVE_BL] - \
                     mean_prop_by_class[1:,  iobj, npts_ptr, PRE_CLOUD_ENTR_FROM_ABOVE_BL]


# Decrease in air detrained from previous cloud.
# (Change in post detrained previous cloud if negative)
        v_entr_prev_cl = mean_prop_by_class[1:  ,iobj,:npts_ptr, POST_DETR_PREV] - \
                         mean_prop_by_class[ :-1,iobj,:npts_ptr, POST_DETR_PREV]
        n_entr_prev_cl = mean_prop_by_class[1:  ,iobj, npts_ptr, POST_DETR_PREV] - \
                         mean_prop_by_class[ :-1,iobj, npts_ptr, POST_DETR_PREV]

        prev_cl_not_entr = (n_entr_prev_cl > 0)[...,0]
        v_entr_prev_cl[prev_cl_not_entr,:] = 0
        n_entr_prev_cl[prev_cl_not_entr,:] = 0

# -(Change in post detrained if negative)
        v_entr_detr_cl = mean_prop_by_class[1:  ,iobj,:npts_ptr, POST_DETR] - \
                         mean_prop_by_class[ :-1,iobj,:npts_ptr, POST_DETR]
        n_entr_detr_cl = mean_prop_by_class[1:  ,iobj, npts_ptr, POST_DETR] - \
                         mean_prop_by_class[ :-1,iobj, npts_ptr, POST_DETR]

        detr_cl_not_entr = (n_entr_detr_cl > 0)[...,0]
        v_entr_detr_cl[detr_cl_not_entr,:] = 0
        n_entr_detr_cl[detr_cl_not_entr,:] = 0

        n_now = n_now[:, np.newaxis]
        n_prev_now = n_prev_now[:, np.newaxis]
        n_entr_bot = n_entr_bot[:, np.newaxis]
        n_entr = n_entr[:, np.newaxis]
        n_detr = n_detr[:, np.newaxis]
        n_prev = n_prev[:, np.newaxis]
        n_entr_pre_bot = n_entr_pre_bot[:, np.newaxis]
        n_entr_pre = n_entr_pre[:, np.newaxis]
        n_entr_prev_cl = n_entr_prev_cl[:, np.newaxis]
        n_entr_detr_cl = n_entr_detr_cl[:, np.newaxis]


#        if debug_mean :
#            print('entr_bot_pre', v_entr_pre_bot[..., mse_ptr ],n_entr_pre_bot)

        v_total_now = v_now + v_prev_now + v_entr + v_entr_bot + v_detr
        n_total_now = n_now + n_prev_now + n_entr + n_entr_bot + n_detr

#        if debug_mean :
#            print('v_total_now', v_total_now[..., mse_ptr ],n_total_now)

        v_total_pre = v_prev + v_entr_pre + v_entr_pre_bot \
                      - v_entr_prev_cl - v_entr_detr_cl
        n_total_pre = n_prev + n_entr_pre + n_entr_pre_bot \
                      - n_entr_prev_cl - n_entr_detr_cl

#        if debug_mean :
#            print('v_total_pre', v_total_pre[..., mse_ptr ],n_total_pre)

        v_loss = v_total_now - v_total_pre
        n_loss = n_total_now - n_total_pre

        if debug_mean :
#            print('now',v_now[..., mse_ptr],n_now )

            print('loss', v_loss[..., mse_ptr])

        some_cl = n_total_now[:,0] > 0
        v_loss_cl = np.zeros_like(v_loss)
        v_loss_cl[some_cl] = v_loss[some_cl] / n_total_now[some_cl]

        if debug_mean :
#            print('now',v_now[..., mse_ptr],n_now )

            print('loss', v_loss_cl[..., mse_ptr])
#        print(np.shape(some_cl), np.shape(v_loss), np.shape(n_total_now))

        if debug_mean :
#            print('loss per point', v_loss[..., mse_ptr ])
            ln = "{:3d} "
            for i in range(11) : ln += "{:5.0f} "
            print(" i    now  prev e_bot   ent   det n_pre  pe_b    pe    pp     ep nloss")
            for i in range(np.shape(v_now)[0]) :
                print(ln.format(i+1, n_now[i,0], \
                                     n_prev_now[i,0], \
                                     n_entr_bot[i,0], \
                                     n_entr[i,0], \
                                     n_detr[i,0], \
                                     n_prev[i,0], \
                                     n_entr_pre_bot[i,0], \
                                     n_entr_pre[i,0], \
                                     n_entr_prev_cl[i,0], \
                                     n_entr_detr_cl[i,0], \
                                     n_loss[i,0]))
# Detrainment rate

        if False :
            print("n_detr",np.shape(n_detr))
            print("n_entr",np.shape(n_entr))
            print("n_entr_bot",np.shape(n_entr_bot))

# Detrainment rate

        n_cloud = n_now + (n_entr + n_entr_bot + n_detr) / 2.0
        some_cl = (n_cloud[:,0] > 0)
        w = mean_prop_by_class[1:, iobj, w_ptr, CLOUD] + \
            (mean_prop_by_class[1:, iobj, w_ptr, ENTR_FROM_ABOVE_BL] + \
             mean_prop_by_class[1:, iobj, w_ptr, ENTR_FROM_BL] + \
             mean_prop_by_class[1:, iobj, w_ptr, DETRAINED]) / 2.0

#        print(np.shape(some_cl), np.shape(w), np.shape(n_cloud))
        w_cloud = np.zeros_like(w)
        w_cloud[some_cl] = w[some_cl] / n_cloud[some_cl,0]

        delta_n_over_n = np.zeros_like(n_cloud[:,0])
        delta_n_over_n[some_cl] = n_detr[some_cl,0] / n_cloud[some_cl,0]
#        print(np.shape(delta_n_over_n),np.shape(delta_t))
        detr_rate = delta_n_over_n / delta_t
# Detrainment rate per m

#        print(np.shape(some_cl), np.shape(w_cloud), np.shape(detr_rate))
        detr_rate_z = np.zeros_like(detr_rate)
        detr_rate_z[some_cl] = detr_rate[some_cl] / w_cloud[some_cl]

# Total Entrainment rate
        n_new_cloud = n_entr + n_entr_bot
        delta_n_over_n = np.zeros_like(n_cloud[:,0])
        delta_n_over_n[some_cl] = n_new_cloud[some_cl,0] / n_cloud[some_cl,0]
        entr_rate_tot = delta_n_over_n / delta_t

# Entrainment rate per m

        entr_rate_tot_z = np.zeros_like(entr_rate_tot)
        entr_rate_tot_z[some_cl] = entr_rate_tot[some_cl] / w_cloud[some_cl]

# Side Entrainment rate
        n_new_cloud = n_entr
        delta_n_over_n = np.zeros_like(n_cloud[:,0])
        delta_n_over_n[some_cl] = n_new_cloud[some_cl,0] / n_cloud[some_cl,0]

#        z1 = (mean_prop_by_class[1:, iobj, z_ptr, CLOUD]-0.5)*traj.deltaz
#        z1 = np.interp(mean_prop_by_class[1:, iobj, z_ptr, CLOUD],
#                       traj.coords['zcoord'],
#                       traj.coords['zn'])

        entr_rate = delta_n_over_n / delta_t
# Side Entrainment rate per m

        entr_rate_z = np.zeros_like(entr_rate)
        entr_rate_z[some_cl] = entr_rate[some_cl] / w_cloud[some_cl]

# Cloud_base Entrainment_rate
        n_new_cloud = n_entr_bot
        delta_n_over_n = np.zeros_like(n_cloud[:,0])
        delta_n_over_n[some_cl] = n_new_cloud[some_cl,0] / n_cloud[some_cl,0]
        entr_rate_cb = delta_n_over_n / delta_t

# Cloud_base Entrainment_rate per m
        entr_rate_cb_z = np.zeros_like(entr_rate)
        entr_rate_cb_z[some_cl] = entr_rate_cb[some_cl] / w_cloud[some_cl]

        if False :
            print('n cloud', n_cloud)
            print('w cloud', mean_prop_by_class[1:, iobj, w_ptr, CLOUD])
            print('w entr', mean_prop_by_class[1:, iobj, w_ptr, ENTR_FROM_ABOVE_BL])
            print('w', w)


        if False :
            print('detr_rate', np.shape(detr_rate))
            print('entr_rate_tot', np.shape(entr_rate_tot))
            print('entr_rate', np.shape(entr_rate))

        if debug_mean :
            print('detr_rate', detr_rate)
            print('detr_rate_z', detr_rate_z)
            print('entr_rate_tot', entr_rate_tot)
            print('entr_rate_tot_z', entr_rate_tot_z)
            print('entr_rate', entr_rate)
            print('entr_rate_z', entr_rate_z)


#        print(np.shape(v_loss), np.shape(budget_loss[:,iobj,:]))
#        budget_loss[:,iobj,:] = v_loss_cl
        budget_loss[:, iobj, :] = mean_prop[1:  , iobj, :-1] - \
                                  mean_prop[ :-1, iobj, :-1]

        entrainment[:, iobj, TOT_ENTR] = entr_rate_tot
        entrainment[:, iobj, TOT_ENTR_Z] = entr_rate_tot_z

        entrainment[:, iobj, SIDE_ENTR] = entr_rate
        entrainment[:, iobj, SIDE_ENTR_Z] = entr_rate_z

        entrainment[:, iobj, CB_ENTR] = entr_rate_cb
        entrainment[:, iobj, CB_ENTR_Z] = entr_rate_cb_z

        entrainment[:, iobj, DETR] = detr_rate
        entrainment[:, iobj, DETR_Z] = detr_rate_z

# Overall cloud properties
# Cloud base
        max_cloud_base_area[iobj] = np.max(n_entr_bot)
#        print(max_cloud_base_area[iobj])
#        print(np.where(n_entr_bot == max_cloud_base_area[iobj])[0][0])
        max_cloud_base_time[iobj] = np.where(n_entr_bot == max_cloud_base_area[iobj])[0][0]
#        print(max_cloud_base_time[iobj])
        if max_cloud_base_area[iobj] > 0 :
            cloud_base_variables[iobj,:] = v_entr_bot[max_cloud_base_time[iobj], :] / max_cloud_base_area[iobj]
        else :
            print('Zero cloud base area for cloud {}'.format(iobj))
        max_cloud_base_area[iobj] = max_cloud_base_area[iobj] * grid_box_area

#        nplt = 72
#        for it in range(np.shape(mean_prop_by_class)[0]) :
#            s = '{:3d}'.format(it)
#            for iclass in range(0,n_class) :
#                s = s+'{:4d} '.format(mean_prop_by_class[it, nplt, r_number, iclass].astype(int))
#            s = s+'{:6d} '.format(np.sum(mean_prop_by_class[it, nplt, r_number, :].astype(int)))
#            print(s)
#
#        for it in range(np.shape(mean_prop_by_class)[0]) :
#            s = '{:3d}'.format(it)
#            for iclass in range(0,n_class) :
#                s = s+'{:10f} '.format(mean_prop_by_class[it, nplt, nvars, iclass]/1E6)
#            s = s+'{:12f} '.format(np.sum(mean_prop_by_class[it, nplt, nvars, :])/1E6)
#            print(s)

    for iclass in range(0,n_class) :
        m = (mean_prop_by_class[:, :, r_number, iclass]>0)
        for ii in range(total_nvars-1) :
            mean_prop_by_class[:, :, ii, iclass][m] /= \
                mean_prop_by_class[:, :, r_number, iclass][m]


    mean_properties = {"overall_mean":mean_prop, \
                       "unclassified":mean_prop_by_class[..., UNCLASSIFIED], \
                       "pre_cloud_bl":mean_prop_by_class[..., PRE_CLOUD_ENTR_FROM_BL], \
                       "pre_cloud_above_bl":mean_prop_by_class[..., PRE_CLOUD_ENTR_FROM_ABOVE_BL], \
                       "previous_cloud":mean_prop_by_class[..., PREVIOUS_CLOUD], \
                       "detr_prev":mean_prop_by_class[..., DETR_PREV], \
                       "post_detr_prev":mean_prop_by_class[..., POST_DETR_PREV], \
                       "cloud":mean_prop_by_class[..., CLOUD], \
                       "entr_bot":mean_prop_by_class[..., ENTR_FROM_BL], \
                       "entr":mean_prop_by_class[..., ENTR_FROM_ABOVE_BL], \
                       "detr":mean_prop_by_class[..., DETRAINED], \
                       "post_detr":mean_prop_by_class[..., DETRAINED], \
                       "subsequent_cloud":mean_prop_by_class[..., SUBSEQUENT_CLOUD], \
                       "cloud_properties":cloud_prop, \
                       "budget_loss":budget_loss, \
                       "entrainment":entrainment, \
                       "max_cloud_base_area":max_cloud_base_area, \
                       "max_cloud_base_time":max_cloud_base_time, \
                       "cloud_base_variables":cloud_base_variables, \
                       "derived_variable_list":derived_variable_list, \
                       }

    return mean_properties

def print_cloud_class(traj, traj_cl, sel_obj, list_classes=True) :
    """
    Print cloud classifiaction properties.

    Args
    ---
        traj       : Trajectory object
        traj_cl    : Dict of Classifications of trajectory points
            provided by set_cloud_class function.

    """
    tr_class = traj_cl["class"][:,traj.labels == sel_obj]
    if list_classes :
        for (iclass,key) in enumerate(traj_cl["key"]) :
            print("{:2d}: {}".format(iclass,key))

    strout = "time "
    for (iclass,key) in enumerate(traj_cl["key"]) :
        strout += "{:5d} ".format(iclass)
    strout += " total"
    print(strout)

    for it in range(0, traj.end+1) :
        strout = "{:4d} ".format(it)
        tot = 0
        for (iclass, key) in enumerate(traj_cl["key"]) :
            in_cl = (tr_class[it,:] == iclass)
            num_in_class = np.size(np.where(in_cl)[0])
            strout += "{:5d} ".format(num_in_class)
            tot += num_in_class
        strout += "{:6d} ".format(tot)
        print(strout)
