"""
Calculations for forward trajectory from a point in space and time using the
position scalars.
"""
import numpy as np
import scipy.optimize
import xarray as xr
from tqdm import tqdm

from ..utils.grid import wrap_periodic_grid_coords
from .backward import calc_trajectory_previous_position
from ..utils.interpolation import gen_interpolator_3d_fields


def _extrapolate_single_timestep(
    ds_position_scalars_origin,
    ds_position_scalars_next,
    ds_traj_posn_prev,
    ds_traj_posn_origin,
    minimization_method=None,
    interp_order=1,
):
    """
    Estimate from the trajectory position `ds_traj_posn_origin` to the next time.

    The algorithm is as follows:

    1) for a trajectory position `(x,y,z)` at a time `t` extrapolate a first
    guess for the next trajecotory position using the previous point in the
    trajectory
    2) find an optimal value for the estimated next trajectory point by
    minimizing the difference between the true origin and the point found when
    back-tracking from the current "next"-point estimate (using the position
    scalars)

    Args
    ----
    ds_position_scalars_origin: xarray DataArray
        3D gridded data at current time.
    ds_position_scalars_next: xarray DataArray
        3D gridded data at current time.
    ds_traj_posn_prev: xarray DataArray
        Trajectory positions at previous time step.
    ds_traj_posn_origin: xarray DataArray
        Trajectory positions at current time step.
    minimization_method: Optional (default=None).
        Method used by scipy.optimize.minimize
    interp_order: int
        Order of interpolation from grid to trajectory.
    """
    ds_grid = ds_position_scalars_origin[["x", "y", "z"]]

    grid_spacing = np.array(
        [ds_position_scalars_origin['x'].attrs['dx'],
         ds_position_scalars_origin['y'].attrs['dy'],
         ds_position_scalars_origin['z'].attrs['dz']])

    if len(ds_traj_posn_origin['x'].dims) > 0:
        grid_spacing = grid_spacing[:,np.newaxis]

    grid_size = np.sqrt(np.mean(grid_spacing*grid_spacing))
    # print(f'grid_size {grid_size}')


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
            ds_pt[c] = xr.DataArray(
                arr_pt.reshape((3,-1))[n])
        return ds_pt

    def _calc_backtrack_origin_dist(ds_traj_posn_next, interpolator=None):
        """
        Compute the distance betweent the true origin and the estimated origin
        (calculated by back-trajectory from `pt_traj_posn_next`)
        """
        if ds_grid.xy_periodic:
            ds_traj_posn_next = _wrap_coords(ds_traj_posn_next)

        # using this estimate of the next trajectory position use the position
        # scalars at that point to estimate where fluid originated from. If we've
        # made a good guess for the next position this estimated origin position
        # should be close to the true origin
        ds_traj_posn_origin_guess = calc_trajectory_previous_position(
            ds_position_scalars=ds_position_scalars_next,
            ds_traj_posn=ds_traj_posn_next,
            interpolator=interpolator,
            interp_order=interp_order,
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

        if ds_grid.xy_periodic:
            Lx = ds_position_scalars_next['x'].attrs['Lx']
            Ly = ds_position_scalars_next['y'].attrs['Ly']
            xerr = np.asarray(dist_arr[0])
            xerr[xerr >  Lx/2] -= Lx
            xerr[xerr < -Lx/2] += Lx
            yerr = np.asarray(dist_arr[1])
            yerr[yerr >  Ly/2] -= Ly
            yerr[yerr < -Ly/2] += Ly
            dist_arr = np.array(
                [
                    xerr,
                    yerr,
                    dist_arr[2],
                ]
            )
        return dist_arr

    def _backtrack_origin_point_iterate(ds_traj_posn_next,
                                        use_mean_abs_error=True,
                                        options=None):
        if options is not None and 'maxiter' in options:
            maxiter = options['maxiter']
        else:
            maxiter = 100
        if options is not None and 'miniter' in options:
            miniter = options['miniter']
        else:
            miniter = 10
        if options is not None and 'relax' in options:
            relax = options['relax']
        else:
            relax = 0.5
        if options is not None and 'tol' in options:
            tol = options['tol']
        else:
            tol=0.1
        if options is not None and 'disp' in options:
            disp = options['disp']
        else:
            disp=False

        if options is not None and 'norm' in options:
            norm='max_abs_error'

        def _norm(d):
            if norm == 'max_abs_error':
                return np.max(np.abs(d))
            if norm == 'mean_abs_error':
                return np.mean(np.abs(d))
            else:
                return np.sqrt(np.mean(d*d))

        interpolator = gen_interpolator_3d_fields(ds_position_scalars_next,
                                                  interp_order=interp_order,
                                                  cyclic_boundaries='xy')

        dist = _calc_backtrack_origin_dist(ds_traj_posn_next,
                                           interpolator=interpolator)

        ndist = dist/grid_spacing
        # err = np.linalg.norm(ndist)
        err = _norm(ndist)
        niter = 0
        not_converged = True
        while not_converged:
            err_prev = err.copy()
            delta = np.clip(dist * relax, -grid_spacing, grid_spacing)
            for i, c in enumerate('xyz'):
                ds_traj_posn_next[c] += delta[i,...]
            dist = _calc_backtrack_origin_dist(ds_traj_posn_next,
                                               interpolator=interpolator)
            ndist = dist/grid_spacing
            # err = np.linalg.norm(ndist)
            err = _norm(ndist)
            niter += 1
            not_converged = (err > tol)
#            print(f"niter: {niter:3d} dist: {np.max(np.abs(dist), axis=1)} err: {err} relax: {relax}")
            if niter >= maxiter:
                break
            if niter%miniter==0: relax *= 0.9

        if disp:
            print(f"Point Iteration converged in {niter:03d} iterations. "
                  f"Final error = {err}")

        return ds_traj_posn_next, err


    if minimization_method == 'point_iter':
        options = {
                    'maxiter': 100,
                    'miniter': 10,
                    'disp': False,
                    'relax': 0.75,
                    'tol': 0.05,
                    'norm': 'max_abs_error',
                  }
        ds_traj_posn_next, err = _backtrack_origin_point_iterate(
                                        ds_traj_posn_next_est,
                                        options=options)

        if ds_grid.xy_periodic:
            ds_traj_posn_next = _wrap_coords(ds_traj_posn_next)
    else:

        interpolator = gen_interpolator_3d_fields(ds_position_scalars_next,
                                                  interp_order=interp_order,
                                                  cyclic_boundaries='xy')
        def _calc_backtrack_origin_err(pt_traj_posn):

            ds_traj_posn = _pt_arr_to_ds(pt_traj_posn)

            dist_arr = _calc_backtrack_origin_dist(ds_traj_posn,
                                                   interpolator=interpolator)

            err = np.linalg.norm(dist_arr)
            # err = np.max(np.abs(dist_arr))
            # print(f"Error = {err}")
            return err

        options = {
                    # 'maxiter': 1000,
                    'maxiter': 10,
                    'disp': False,
                    'gtol': 0.01,
                    # 'fatol': 10.0,
                    # 'ftol': 25.0,
                    # 'eps': 10.0,
                  }
        # for the minimizsation we will be using just a numpy-array
        # containing the (x,y,z) location
        pt_traj_posn_next_est = _pt_ds_to_arr(ds_traj_posn_next_est)
        niter = 0
        while niter < options['maxiter']:
            sol = scipy.optimize.minimize(
                fun=_calc_backtrack_origin_err,
                x0=pt_traj_posn_next_est,
                method=minimization_method,
                options = options,
            )

            if sol.fun <= grid_size * 0.05:
                break
            pt_traj_posn_next_est = sol.x
            niter += 1

        # if sol.success:
        ds_traj_posn_next = _pt_arr_to_ds(sol.x)

        if ds_grid.xy_periodic:
            ds_traj_posn_next = _wrap_coords(ds_traj_posn_next)
        # else:
            # raise Exception("The minimization didn't converge")

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
            minimization_method = 'point_iter',
            # minimization_method = 'CG',
            # minimization_method = 'BFGS',
            # minimization_method = 'Nelder-Mead',
            # minimization_method = 'SLSQP',
        )

        ds_traj = xr.concat([ds_traj, ds_traj_posn_est], dim="time")

    return ds_traj
