"""
Calculations for forward trajectory from a point in space and time using the
position scalars
"""
import numpy as np
import scipy.optimize
import xarray as xr
from tqdm import tqdm

from ..utils.grid import wrap_periodic_grid_coords
from .backward import calc_trajectory_previous_position


def _extrapolate_single_timestep(
    ds_position_scalars_origin,
    ds_position_scalars_next,
    ds_traj_posn_prev,
    ds_traj_posn_origin,
    minimization_method=None,
):
    """
    Extrapolate from the trajectory position `ds_traj_posn_origin` to the next time

    The algorithm is as follows:

    1) for a trajectory position `(x,y,z)` at a time `t` extrapolate a first
    guess for the next trajecotory position using the previous point in the
    trajectory
    2) find an optimal value for the estimated next trajectory point by
    minimizing the difference between the true origin and the point found when
    back-tracking from the current "next"-point estimate (using the position
    scalars)
    """
    ds_grid = ds_position_scalars_origin[["x", "y", "z"]]

    def _wrap_coords(ds_posn):
        cyclic_coords = ("x", "y")
        cell_centered_coords = ("x", "y", "z")
        return wrap_periodic_grid_coords(
            ds_grid=ds_grid,
            ds_posn=ds_posn,
            cyclic_coords=cyclic_coords,
            cell_centered_coords=cell_centered_coords,
        )

    # traj_posn_next_est is our estimate of the trajectory positions at
    # the next time step.
    # We want the 'where from' at traj_posn_next_est to match
    # the current trajectory positions, traj_posn_origin

    # Let f(X) be the function which estimates the distance between the actual
    # origin and the point estimated by back-trajactory from the estimated next
    # point, i.e. f(X) -> X_err, we then want to minimize X_err.
    # We will use the Eucledian distance (L2-norm) so that we minimize the
    # magnitude of this error

    # First guess - extrapolate from last two positions.
    # given points A and B the vector spanning from A to B is AB = B-A
    # let C = B + AB, then C = B + B - A = 2B - A

    ds_traj_posn_next_est = 2 * ds_traj_posn_origin - ds_traj_posn_prev

    def _pt_ds_to_arr(ds_pt):
        return np.array([ds_pt[c].data for c in "xyz"])

    def _pt_arr_to_ds(arr_pt):
        ds_pt = xr.Dataset()
        for n, c in enumerate("xyz"):
            ds_pt[c] = arr_pt[n]
        return ds_pt

    # for the minimizsation we will be using just a numpy-array containing the
    # (x,y,z) location
    pt_traj_posn_next_est = _pt_ds_to_arr(ds_traj_posn_next_est)

    def _calc_backtrack_origin_dist(pt_traj_posn_next):
        """
        Compute the distance betweent the true origin and the estimated origin
        (calculated by back-trajectory from `pt_traj_posn_next`)
        """
        ds_traj_posn_next = xr.Dataset()
        for n, c in enumerate("xyz"):
            ds_traj_posn_next[c] = pt_traj_posn_next[n]

        if ds_grid.xy_periodic:
            ds_traj_posn_next = _wrap_coords(ds_traj_posn_next)

        # using this estimate of the next trajectory position use the position
        # scalars at that point to estimate where fluid originated from. If we've
        # made a good guess for the next position this estimated origin position
        # should be close to the true origin
        ds_traj_posn_origin_guess = calc_trajectory_previous_position(
            ds_position_scalars=ds_position_scalars_next,
            ds_traj_posn=ds_traj_posn_next_est,
        )

        ds_p1 = ds_traj_posn_origin
        ds_p2 = ds_traj_posn_origin_guess
        dist_arr = np.array(
            [
                ds_p1.x - ds_p2.x_est,
                ds_p1.y - ds_p2.y_est,
                ds_p1.z - ds_p2.z_est,
            ]
        )
        return dist_arr

    def _calc_backtrack_origin_err(pt_traj_posn_next):
        dist_arr = _calc_backtrack_origin_dist(pt_traj_posn_next=pt_traj_posn_next_est)
        err = np.linalg.norm(dist_arr)
        return err

    sol = scipy.optimize.minimize(
        fun=_calc_backtrack_origin_err,
        x0=pt_traj_posn_next_est,
        method=minimization_method,
    )

    if sol.success:
        ds_traj_posn_next = _pt_arr_to_ds(sol.x)

        if ds_grid.xy_periodic:
            ds_traj_posn_next = _wrap_coords(ds_traj_posn_next)
    else:
        raise Exception("The minimization didn't converge")

    ds_traj_posn_next = ds_traj_posn_next.assign_coords(
        time=ds_position_scalars_next.time
    )

    return ds_traj_posn_next


def forward(ds_position_scalars, ds_back_trajectory, da_times, interp_order=1):
    """
    Using the position scalars `ds_position_scalars` integrate forwards from
    the last point in `ds_back_trajectory` to the times in `da_times`. The
    backward trajectory must contain at least two points in time as these are
    used for the initial guess for the forward extrapolation
    """
    if not ds_back_trajectory.time.count() >= 2:
        raise Exception(
            "The back trajectory must contain at least two points for the forward"
            " extrapolation have an initial guess for the direction"
        )

    # create a copy of the backward trajectory so that we have a dataset into
    # which we will accumulate the full trajectory
    ds_traj = ds_back_trajectory.copy()

    # step forward in time, `t_forward` represents the time we're of the next
    # point (forward) of the trajectory
    for t_next in tqdm(da_times, desc="forward"):
        ds_traj_posn_origin = ds_traj.isel(time=-1)
        t_origin = ds_traj_posn_origin.time
        ds_position_scalars_origin = ds_position_scalars.sel(time=t_origin)
        ds_position_scalars_next = ds_position_scalars.sel(time=t_next)

        # for the original direction estimate we need *previous* position (i.e.
        # where we were before the "origin" point)
        ds_traj_posn_prev = ds_traj.isel(time=-2)

        ds_traj_posn_est = _extrapolate_single_timestep(
            ds_position_scalars_origin=ds_position_scalars_origin,
            ds_position_scalars_next=ds_position_scalars_next,
            ds_traj_posn_prev=ds_traj_posn_prev,
            ds_traj_posn_origin=ds_traj_posn_origin,
        )

        ds_traj = xr.concat([ds_traj, ds_traj_posn_est], dim="time")

    return ds_traj
