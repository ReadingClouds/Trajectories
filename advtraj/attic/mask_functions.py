"""
This module contains routines for computing masks that define where objects are
in 3D for starting trajectory integration
"""

import numpy as np
from scipy import ndimage


def label_3D_cyclic(mask, debug=False):
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
    labels -= 1

    def relabel(labs, nobjs, i, j):
        #        if debug_label :
        #            print('Setting label {:3d} to {:3d}'.format(j,i))
        lj = labs == j
        labs[lj] = i
        for k in range(j + 1, nobjs):
            lk = labs == k
            labs[lk] = k - 1
        #            if debug_label :
        #                print('Setting label {:3d} to {:3d}'.format(k,k-1))
        nobjs -= 1
        #        if debug_label : print('nobjects = {:d}'.format(nobjects))
        return labs, nobjs

    def find_objects_at_edge(minflag, x_or_y, n, labs, nobjs):
        #        for minflag in (True,False) :
        i = 0
        while i < (nobjs - 2):
            posi = np.where(labs == i)
            #        print(np.shape(posi))
            if minflag:
                test1 = np.min(posi[x_or_y][:]) == 0
                border = "0"
            else:
                test1 = np.max(posi[x_or_y][:]) == (n - 1)
                border = "n{}-1".format(["x", "y"][x_or_y])
            if test1:
                if debug:
                    print(
                        "Object {:03d} on {}={} border?".format(
                            i, ["x", "y"][x_or_y], border
                        )
                    )
                j = i + 1
                while j < (nobjs - 1):
                    posj = np.where(labs == j)

                    if minflag:
                        test2 = np.max(posj[x_or_y][:]) == (n - 1)
                        border = "n{}-1".format(["x", "y"][x_or_y])
                    else:
                        test2 = np.min(posj[x_or_y][:]) == 0
                        border = "0"

                    if test2:
                        if debug:
                            print(
                                "Match Object {:03d} on {}={} border?".format(
                                    j, ["x", "y"][x_or_y], border
                                )
                            )

                        if minflag:
                            ilist = np.where(posi[x_or_y][:] == 0)
                            jlist = np.where(posj[x_or_y][:] == (n - 1))
                        else:
                            ilist = np.where(posi[x_or_y][:] == (n - 1))
                            jlist = np.where(posj[x_or_y][:] == 0)

                        if np.size(
                            np.intersect1d(
                                posi[1 - x_or_y][ilist], posj[1 - x_or_y][jlist]
                            )
                        ):
                            if np.size(np.intersect1d(posi[2][ilist], posj[2][jlist])):

                                if debug:
                                    print("Yes!", i, j)
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

    labels, nobjects = find_objects_at_edge(True, 0, nx, labels, nobjects)
    labels, nobjects = find_objects_at_edge(False, 0, nx, labels, nobjects)
    labels, nobjects = find_objects_at_edge(True, 1, ny, labels, nobjects)
    labels, nobjects = find_objects_at_edge(False, 1, ny, labels, nobjects)

    return labels, nobjects


def trajectory_cloud_ref(dataset, time_index, thresh=0.00001):
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

    xcoord = np.arange(np.shape(data)[0], dtype="float")
    ycoord = np.arange(np.shape(data)[1], dtype="float")
    zcoord = np.arange(np.shape(data)[2], dtype="float")

    logical_pos = cloud_select(data, thresh=thresh)

    #    print('q_cl threshold {:10.6f}'.format(np.max(data[...])*0.8))

    mask = np.zeros_like(data)
    mask[logical_pos] = 1

    print("Setting labels.")
    labels, nobjects = label_3D_cyclic(mask)

    labels = labels[logical_pos]

    pos = np.where(logical_pos)

    #    print(np.shape(pos))
    traj_pos = np.array(
        [xcoord[pos[0][:]], ycoord[pos[1][:]], zcoord[pos[2][:]]], ndmin=2
    ).T

    return traj_pos, labels, nobjects


def in_cloud(traj, *argv, thresh=1.0e-5):
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

    if len(argv) == 2:
        (tr_time, obj_ptrs) = argv
        qcl = data[tr_time, obj_ptrs, v]
    else:
        qcl = data[..., v]

    #    mask = data[...]>(np.max(data[...])*0.6)
    mask = cloud_select(qcl, thresh=thresh)
    return mask, qcl


def cloud_select(qcl, thresh=1.0e-5):
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


def trajectory_cstruct_ref(dataset, time_index, thresh=0.00001, find_objects=False):
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
    qcl = dataset.variables["q_cloud_liquid_mass"][time_index, ...]
    w = dataset.variables["w"][time_index, ...]
    zr = dataset.variables["tracer_traj_zr"][time_index, ...]
    tr1 = dataset.variables["tracer_rad1"][time_index, ...]

    xcoord = np.arange(np.shape(qcl)[0], dtype="float")
    ycoord = np.arange(np.shape(qcl)[1], dtype="float")
    zcoord = np.arange(np.shape(qcl)[2], dtype="float")

    logical_pos = cstruct_select(qcl, w, zr, tr1, thresh=thresh, ver=6)

    # print('rad threshold {:10.6f}'.format(np.max(data[...])*0.8))

    mask = np.zeros_like(qcl)
    mask[logical_pos] = 1

    if find_objects:
        print("Setting labels.")
        labels, nobjects = label_3D_cyclic(mask)
        print("{} objects found.".format(nobjects))

    else:
        nobjects = 1
        labels = -1 * np.ones_like(mask)
        labels[logical_pos] = 0

    labels = labels[logical_pos]

    pos = np.where(logical_pos)

    #    print(np.shape(pos))
    traj_pos = np.array(
        [xcoord[pos[0][:]], ycoord[pos[1][:]], zcoord[pos[2][:]]], ndmin=2
    ).T

    return traj_pos, labels, nobjects


def in_cstruc(traj, *argv, **kwargs):
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
    z_p = traj.var("tracer_traj_zr")  # This seems to be a hack - i.e. wrong.
    tr1_p = traj.var("tracer_rad1")

    if "thresh" in kwargs:
        thresh = kwargs["thresh"]
    else:
        thresh = 1.0e-5

    if len(argv) == 2:
        (tr_time, obj_ptrs) = argv
        qcl = data[tr_time, obj_ptrs, v]
        w = data[tr_time, obj_ptrs, w_v]
        tr1 = data[tr_time, obj_ptrs, tr1_p]
    else:
        qcl = data[..., v]
        w = data[..., w_v]
        zpos = data[..., z_p]
        tr1 = data[..., tr1_p]

    mask = cstruct_select(qcl, w, zpos, tr1, thresh=thresh, ver=6)
    return mask, qcl


def cstruct_select(qcl, w, zpos, tr1, z_lev1=1, z_lev2=3, thresh=1.0e-5, ver=6):
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

    if ver == 1:
        mask = qcl >= thresh
    elif ver == 2:
        mask = (qcl >= thresh) and (w > 0.0)
    elif ver == 3:
        mask = (qcl < thresh) and (zpos < z_lev2)
    elif ver == 4:
        mask = np.logical_and(zpos > z_lev1, zpos < z_lev2)
    elif ver == 5:
        tr1_pr = tr1 - np.mean(tr1, axis=(0, 1))
        std = np.std(tr1, axis=(0, 1))
        std_int = np.ones(len(std))
        c_crit = np.ones(len(std))
        for iz in range(len(std)):
            std_int[iz] = 0.05 * np.mean(std[: iz + 1])
            c_crit[iz] = np.maximum(std[iz], std_int[iz])
        tr_crit = np.ones_like(tr1) * c_crit
        mask = np.logical_and(tr1_pr > tr_crit, w > 0.0)
    else:
        v_shape = qcl.shape
        mask_pos = np.zeros((v_shape[0], v_shape[1], v_shape[2]))
        mask_pos[::2, ::2, ::2] = 1
        mask = mask_pos == 1
    return mask
