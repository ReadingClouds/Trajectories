# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 18:57:30 2022

@author: paclk
"""
import numpy as np
from trajectories.object_tools import label_3D_cyclic


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
