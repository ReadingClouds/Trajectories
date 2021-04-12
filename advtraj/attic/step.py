import numpy as np
from scipy import ndimage

from .data import load_traj_step_data
from .utils.interpolation import tri_lin_interp


def _calculate_phase(vr, vi, n):
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
    vpos = np.asarray(((np.arctan2(vi, vr)) / (2.0 * np.pi)) * n)
    vpos[vpos < 0] += n
    return vpos


def _extract_pos(nx, ny, dat, cyclic_xy):
    """
    Function to extract 3D position from data array.

    Args:
        nx        : Number of points in x direction.
        ny        : Number of points in y direction.
        dat       : Array[m,n] where n>=5 if cyclic_xy or 3 if not.
        cyclic_xy : Assume cyclic (x,y)-direction boundary conditions

    Returns:
        pos       : Array[m,3]
        n_pvar    : Number of dimensions in input data used for pos.
    """

    if cyclic_xy:
        n_pvar = 5
        xpos = _calculate_phase(dat[0], dat[1], nx)
        ypos = _calculate_phase(dat[2], dat[3], ny)
        pos = np.array([xpos, ypos, dat[4]]).T
    else:
        n_pvar = 3
        pos = np.array([dat[0], dat[1], dat[2]]).T

    return pos, n_pvar


def data_to_pos(data, varp_list, pos, xcoord, ycoord, zcoord, use_bilin=True, maxindex=None):

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
    # XXX: first-order interpolation hard-coded for now if `use_bilin == False`
    interp_order = 1

    if use_bilin:
        output = tri_lin_interp(
            data, varp_list, pos, xcoord, ycoord, zcoord, maxindex=maxindex
        )
    else:
        output = list([])
        for l in range(len(data)):
            #            print 'Calling map_coordinates'
            #            print np.shape(data[l]), np.shape(traj_pos)
            out = ndimage.map_coordinates(data[l], pos, mode="wrap", order=interp_order)
            output.append(out)
    return output


def back_trajectory_step(
    dataset,
    time_index,
    variable_list,
    refprof,
    coords,
    trajectory,
    data_val,
    traj_error,
    traj_times,
):
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

    data_list, varlist, varp_list, time = load_traj_step_data(
        dataset, time_index, variable_list, refprof, coords
    )
    print("Processing data at time {}".format(time))

    (nx, ny, nz) = np.shape(data_list[0])

    traj_pos = trajectory[0]
    #    print "traj_pos ", np.shape(traj_pos), traj_pos[:,0:5]

    out = data_to_pos(
        data_list,
        varp_list,
        traj_pos,
        coords["xcoord"],
        coords["ycoord"],
        coords["zcoord"],
    )
    traj_pos_new, n_pvar = _extract_pos(nx, ny, out)

    data_val.insert(0, np.vstack(out[n_pvar:]).T)
    trajectory.insert(0, traj_pos_new)
    traj_error.insert(0, np.zeros_like(traj_pos_new))
    traj_times.insert(0, time)

    return trajectory, data_val, traj_error, traj_times
