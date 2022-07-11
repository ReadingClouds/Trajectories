"""
Calculations for forward trajectory from a point in space and time using the
position scalars.
"""
import numpy as np
import scipy.optimize
import xarray as xr
from tqdm import tqdm

from ..utils.grid import wrap_periodic_grid_coords
from ..utils.interpolation import gen_interpolator_3d_fields
from .backward import calc_trajectory_previous_position


def _wrap_coords(ds_posn, ds_grid):
    """
    Wrapper for wrap_periodic_grid_coords

    Parameters
    ----------
    ds_posn : xarray Dataset
        Trajectory positions 'x', 'y', 'z'.
    ds_grid : xarray Dataset
        grid information.

    Returns
    -------
    xarray Dataset
        Wrapped version of ds_posn.

    """
    cyclic_coords = ("x", "y")
    cell_centered_coords = ("x", "y", "z")
    return wrap_periodic_grid_coords(
        ds_grid=ds_grid,
        ds_posn=ds_posn,
        cyclic_coords=cyclic_coords,
        cell_centered_coords=cell_centered_coords,
    )


def _calc_backtrack_origin_dist(
    ds_position_scalars,
    ds_traj_posn_org,
    ds_traj_posn,
    interpolator=None,
    interp_order=5,
):
    """
    Compute the distance between the true origin and the estimated origin
    (calculated by back-trajectory from `pt_traj_posn_next`)
    """
    ds_grid = ds_position_scalars[["x", "y", "z"]]

    if ds_grid.xy_periodic:
        ds_traj_posn = _wrap_coords(ds_traj_posn, ds_grid)

    # ds_traj_posn_org contains the 'current' (i.e. known) trajectory
    # positions.

    # ds_traj_posn contains a guess of the 'next' trajectory position.
    # Using this estimate of the next trajectory position use the position
    # scalars at that point to estimate where fluid originated from. If we've
    # made a good guess for the next position this estimated origin position
    # should be close to the true origin, i.e. the 'current' trajectory
    # position.
    ds_traj_posn_org_guess = calc_trajectory_previous_position(
        ds_position_scalars=ds_position_scalars,
        ds_traj_posn=ds_traj_posn,
        interpolator=interpolator,
        interp_order=interp_order,
    )

    ds_p1 = ds_traj_posn_org
    ds_p2 = ds_traj_posn_org_guess
    dist_arr = np.array(
        [
            ds_p1.x.values - ds_p2.x_est.values,
            ds_p1.y.values - ds_p2.y_est.values,
            ds_p1.z.values - ds_p2.z_est.values,
        ]
    )

    # Deal with xy wraparound.
    if ds_grid.xy_periodic:
        Lx = ds_position_scalars["x"].attrs["Lx"]
        Ly = ds_position_scalars["y"].attrs["Ly"]
        xerr = np.asarray(dist_arr[0])
        xerr[xerr > Lx / 2] -= Lx
        xerr[xerr < -Lx / 2] += Lx
        yerr = np.asarray(dist_arr[1])
        yerr[yerr > Ly / 2] -= Ly
        yerr[yerr < -Ly / 2] += Ly
        dist_arr = np.array(
            [
                xerr,
                yerr,
                dist_arr[2],
            ]
        )
    # Deal with single trajectory case
    if dist_arr.ndim == 1:
        dist_arr = np.expand_dims(dist_arr, 1)
    return dist_arr


def _pt_ds_to_arr(ds_pt):
    return np.array([ds_pt[c].data for c in "xyz"])


def _pt_arr_to_ds(arr_pt):
    # Ensure arr_pt has 2 dimensions.
    arr_pt = arr_pt.reshape((3, -1))
    ds_pt = xr.Dataset()
    for n, c in enumerate("xyz"):
        ds_pt[c] = xr.DataArray(arr_pt[n, :], dims=("trajectory_number"))

    return ds_pt


def get_error_norm(
    scalars,
    pos_org,
    pos_est,
    grid_spacing,
    norm="max_abs_error",
    interpolator=None,
    interp_order=5,
):
    dist = _calc_backtrack_origin_dist(
        scalars,
        pos_org,
        pos_est,
        interpolator=interpolator,
        interp_order=interp_order,
    )

    ndist = dist / grid_spacing

    if norm == "max_abs_error":
        err = np.linalg.norm(ndist.flatten(), ord=np.inf)
    if norm == "mean_abs_error":
        err = np.mean(np.abs(ndist))
    else:
        err = np.linalg.norm(ndist.flatten()) / np.sqrt(ndist.size)

    return dist, ndist, err


def _backtrack_origin_point_iterate(
    ds_position_scalars,
    ds_traj_posn_org,
    ds_traj_posn_first_guess,
    interpolator,
    solver,
    interp_order=5,
    maxiter=200,
    miniter=10,
    disp=False,
    relax=1.0,
    relax_reduce=0.95,
    tol=0.01,
    opt_one_step=False,
    norm="max_abs_error",
    max_outer_loops=1,
    minimizer="BFGS",
    minim_kwargs=None,
):
    """
    Find the forward trajectory solutions as an Xarray DataSet.

    The solution is found when the the origin of air arriving
    at pt_traj_posn_next equals ds_traj_posn_org, with first guess
    pt_traj_posn_first_guess as a numpy array.

    This uses point iteration;
    optionally (solver='hybrid_hybrid_fixed_point_iterator') uses
    _pt_backtrack_origin_optimize to find solution for points that do not
    converge in the allowed iterations. Note that the optimizer often uses
    many hundreds of error evaluations, so it is worth letting the point
    iteration have lots of iterations before resorting to it.

    Point iteration can be defined as:
        x(n+1) = x(n) + relax * error(n)
    where error(n) is the distance between the required origin of the air and
    the origin at x(n).

    Parameters
    ----------
    ds_position_scalars : Xarray DataSet
        3D fields of trajectory tracer data.
    ds_traj_posn_org : Xarray DataSet
        Current trajectory positions.
    ds_traj_posn_first_guess : Xarray DataSet
        First guess of trajectory endpoints.
    interpolator : fast interp interpolator.
        Pre-calculated interpolator for fields in ds_position_scalars.
        If None, interpolator is calculated on each call to the interpolation.
    solver : str
        "hybrid_fixed_point_iterator" or "hybrid_fixed_point_iterator".
    interp_order : int, optional
        Interpolation order. The default is 5.

    maxiter (int)        : Maximum number of iterations. Delault=500.
    miniter (int)        : Number of iterations to use before adjusting
                           relax. Default=10.
    relax (float)        : Initial relaxation factor. Default=0.8.
    relax_reduce (float) : Factor to mutiply relax after every miniter
                           iterations. Default=0.9.
    tol (float)          : Distance in terms of grid lengths defining
                           convergence. Default=0.01.
    opt_one_step (bool)  : If True, use single call to optimizer for
                           all trajectories. Default=False.
    disp (bool)          : Print information on convergence.
                           Default=False.
    norm (str)           : Norm to use in erro. Options are
                           'mean_abs_error', 'max_abs_error'
                           or RMS distance. Default='max_abs_error'
    max_outer_loops (int): Maximum number of outer loops of minimizer.
    minimizer       (str): Minimizaer to use. Default="BFGS"
    minim_kwargs (dict)  : Options for minimiser.

    Returns
    -------
    Xarray DataSet, numpy array [3, n]
        New trajectory position and residual error vector.

    """
    grid_spacing = np.array(
        [
            ds_position_scalars["x"].attrs["dx"],
            ds_position_scalars["y"].attrs["dy"],
            ds_position_scalars["z"].attrs["dz"],
        ]
    )

    grid_spacing = grid_spacing[:, np.newaxis]

    # First find error in first guess.
    dist, ndist, err = get_error_norm(
        ds_position_scalars,
        ds_traj_posn_org,
        ds_traj_posn_first_guess,
        grid_spacing,
        norm=norm,
        interpolator=interpolator,
        interp_order=interp_order,
    )

    ds_traj_posn_next = ds_traj_posn_first_guess

    # Now setup iteration
    niter = 0
    not_converged = True
    while not_converged:

        # We have found convergence is faster if we move slightly less
        # than the residual distance; this is the relax parameter,
        # generally <1.
        # Restrict change to no more than 1 grid box.
        delta = np.clip(dist * relax, -grid_spacing, grid_spacing)

        for i, c in enumerate("xyz"):
            if delta.shape[1] == 1:
                ds_traj_posn_next[c] += np.squeeze(delta[i])
            else:
                ds_traj_posn_next[c] += delta[i, ...]

        dist, ndist, err = get_error_norm(
            ds_position_scalars,
            ds_traj_posn_org,
            ds_traj_posn_next,
            grid_spacing,
            norm=norm,
            interpolator=interpolator,
            interp_order=interp_order,
        )

        niter += 1
        not_converged = err > tol

        if niter >= maxiter:
            break
        # If we need a lot of iterations, reduce the relaxation factor.
        if niter % miniter == 0:
            relax *= relax_reduce

    # Are we converged?
    if niter >= maxiter:
        # Select out those trajectories that have not converged.
        ncm = np.abs(ndist) > tol
        not_conv_mask = ncm[0, :] | ncm[1, :] | ncm[2, :]
        print(
            f"\nPoint Iteration failed to converge "
            f"in {niter:3d} iterations. "
            f"{np.sum(not_conv_mask)} trajectories did not converge. "
            f"Final error = {err}"
        )

    if solver == "hybrid_fixed_point_iterator" and niter >= maxiter:
        # Use optimization solution for un-converged trajectories.

        # Convert best estimate from point iteration to array.
        pt_traj_posn_next = _pt_ds_to_arr(ds_traj_posn_next)
        # Select unconverged trajectories.
        pt_traj_posn_next_not_conv = pt_traj_posn_next[:, not_conv_mask]

        print(
            f"Optimising {pt_traj_posn_next_not_conv.shape[1]} "
            f"unconverged trajectories."
        )

        # Select out corresponding 'true' value.
        ds_traj_posn_org_subset = ds_traj_posn_org[["x", "y", "z"]].isel(
            trajectory_number=not_conv_mask
        )

        # Choice between optimising all the selected trajectories together
        # or one at a time.
        # First option is likely to be deprecated in future.
        if opt_one_step:
            pt_traj_posn_next_not_conv = pt_traj_posn_next_not_conv.flatten()
            pt_traj_posn_next_not_conv = _pt_backtrack_origin_optimize(
                ds_position_scalars,
                ds_traj_posn_org_subset,
                pt_traj_posn_next_not_conv,
                interpolator,
                minimizer,
                interp_order=interp_order,
                **minim_kwargs,
            )
        else:
            for itraj in range(pt_traj_posn_next_not_conv.shape[1]):

                print(f"Optimising unconverged trajectory {itraj}.")
                pt_traj_posn = pt_traj_posn_next_not_conv[:, itraj].flatten()
                ds_traj_posn_orig = ds_traj_posn_org_subset.isel(
                    trajectory_number=itraj
                )
                pt_traj_posn = _pt_backtrack_origin_optimize(
                    ds_position_scalars,
                    ds_traj_posn_orig,
                    pt_traj_posn,
                    interpolator,
                    minimizer,
                    interp_order=interp_order,
                    **minim_kwargs,
                )

                pt_traj_posn_next_not_conv[:, itraj] = pt_traj_posn

        # Put results back into main array.
        pt_traj_posn_next[:, not_conv_mask] = pt_traj_posn_next_not_conv.reshape(
            (3, -1)
        )
        # Convert to Dataset
        ds_traj_posn_next = _pt_arr_to_ds(pt_traj_posn_next)
        # Calculate final error.
        dist, ndist, err = get_error_norm(
            ds_position_scalars,
            ds_traj_posn_org,
            ds_traj_posn_next,
            grid_spacing,
            norm=norm,
            interpolator=interpolator,
            interp_order=interp_order,
        )
        print(f"After minimization error={err}.")

    if disp:
        print(
            f"Point Iteration finished in {niter:3d} iterations. "
            f"Final error = {err}"
        )

    return ds_traj_posn_next, err, dist


def _pt_backtrack_origin_optimize(
    ds_position_scalars,
    ds_traj_posn_org,
    pt_traj_posn_first_guess,
    interpolator,
    minimization_method,
    interp_order=5,
    max_outer_loops=1,
    tol=0.01,
    minimize_options=None,
):
    """
    Find the forward trajectory solutions as a numpy array of points.

    The solution is found when the the origin of air arriving
    at pt_traj_posn_next equals ds_traj_posn_org, with first guess
    pt_traj_posn_first_guess as a numpy array.

    The minimizer scipy.optimize.minimize is called max_outer_loops times,
    with the previous best solution as the first guess, because, when they
    work, the optimization routines have been found to iterate to
    machine precision, which is much higher than practically needed.

    Parameters
    ----------
    ds_position_scalars : Xarray DataSet
        3D fields of trajectory tracer data.
    ds_traj_posn_org : Xarray DataSet
        Current trajectory positions.
    pt_traj_posn_first_guess : numpy array[3, n]
        First guess of trajectory endpoints.
    interpolator : fast interp interpolator.
        Pre-calculated interpolator for fields in ds_position_scalars.
        If None, interpolator is calculated on each call to the interpolation.
    minimization_method : str
        Methods supported by  scipy.optimize.minimize.
    interp_order : int, optional
        Interpolation order. The default is 5.
    max_outer_loops (int): Max number of calls to optimizer. Default=1.
    tol (float)          : Distance in terms of grid lengths defining
        convergence of outer loops. Default=0.01.
    minimize_options : dict, optional
        Options sent to scipy.optimize.minimize.

    Returns
    -------
    numpy array[3 * n]
        New trajectory position..

    """

    options = {
        "maxiter": 10,
        "disp": False,
    }

    if minimize_options is not None:
        options.update(minimize_options)

    grid_spacing = np.array(
        [
            ds_position_scalars["x"].attrs["dx"],
            ds_position_scalars["y"].attrs["dy"],
            ds_position_scalars["z"].attrs["dz"],
        ]
    )

    grid_size = np.sqrt(np.mean(grid_spacing * grid_spacing))

    def _calc_backtrack_origin_err(pt_traj_posn):

        ds_traj_posn = _pt_arr_to_ds(pt_traj_posn)

        dist_arr = _calc_backtrack_origin_dist(
            ds_position_scalars,
            ds_traj_posn_org,
            ds_traj_posn,
            interpolator=interpolator,
            interp_order=interp_order,
        )

        err = np.linalg.norm(dist_arr.flatten(), ord=2)
        return err

    # for the minimization we will be using just a numpy-array
    # containing the (x,y,z) location

    pt_traj_posn_next = pt_traj_posn_first_guess
    niter = 0
    while niter < max_outer_loops:
        sol = scipy.optimize.minimize(
            fun=_calc_backtrack_origin_err,
            x0=pt_traj_posn_next,
            method=minimization_method,
            options=options,
        )
        pt_traj_posn_next = sol.x
        if sol.fun / np.sqrt(sol.x.size) <= grid_size * tol:
            break
        niter += 1

    return pt_traj_posn_next


def _ds_backtrack_origin_optimize(
    ds_position_scalars,
    ds_traj_posn_org,
    ds_traj_posn_first_guess,
    interpolator,
    minimization_method,
    interp_order=5,
    kwargs=None,
):
    """
    Find the forward trajectory solutions as an Xarray DataSet.

    The solution is found when the the origin of air arriving
    at pt_traj_posn_next equals ds_traj_posn_org, with first guess
    pt_traj_posn_first_guess as a numpy array.

    This is a wrapper for _pt_backtrack_origin_optimize which does the
    work finding the forward trajectory solutions as a numpy array of
    points. This pulls the array out of am xarray dataset then puts
    the result back in one, finally calculating the residual error.

    Parameters
    ----------
    ds_position_scalars : Xarray DataSet
        3D fields of trajectory tracer data.
    ds_traj_posn_org : Xarray DataSet
        Current trajectory positions.
    ds_traj_posn_first_guess : Xarray DataSet
        First guess of trajectory endpoints.
    interpolator : fast interp interpolator.
        If None, interpolator is calculated on each call to the interpolation.
        Pre-calculated interpolator for fields in ds_position_scalars.
    minimization_method : str
        Methods supported by  scipy.optimize.minimize.
    kwargs : dict
        Keyword arguments sent to  _pt_backtrack_origin_optimize.

    Returns
    -------
    Xarray DataSet, numpy array [3, n]
        New trajectory position and residual error vector.

    """
    if kwargs is None:
        kwargs = {}

    pt_traj_posn_next = _pt_ds_to_arr(ds_traj_posn_first_guess).flatten()

    pt_traj_posn_next = _pt_backtrack_origin_optimize(
        ds_position_scalars,
        ds_traj_posn_org,
        pt_traj_posn_next,
        interpolator,
        minimization_method,
        **kwargs,
    )

    ds_traj_posn_next = _pt_arr_to_ds(pt_traj_posn_next)

    ds_grid = ds_position_scalars[["x", "y", "z"]]

    if ds_grid.xy_periodic:
        ds_traj_posn_next = _wrap_coords(ds_traj_posn_next, ds_grid)

    dist = _calc_backtrack_origin_dist(
        ds_position_scalars,
        ds_traj_posn_org,
        ds_traj_posn_next,
        interpolator=interpolator,
        interp_order=interp_order,
    )
    return ds_traj_posn_next, dist


def _extrapolate_single_timestep(
    ds_position_scalars_origin,
    ds_position_scalars_next,
    ds_traj_posn_prev,
    ds_traj_posn_origin,
    solver="hybrid_fixed_point_iterator",
    interp_order=5,
    point_iter_kwargs=None,
    minim_kwargs=None,
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
    solver: Optional (default='hybrid_fixed_point_iterator').
        Method used by scipy.optimize.minimize
    interp_order: int
        Order of interpolation from grid to trajectory.
    """
    if point_iter_kwargs is None:
        point_iter_kwargs = {}
    if minim_kwargs is None:
        minim_kwargs = {}

    ds_grid = ds_position_scalars_origin[["x", "y", "z"]]

    grid_spacing = np.array(
        [
            ds_position_scalars_origin["x"].attrs["dx"],
            ds_position_scalars_origin["y"].attrs["dy"],
            ds_position_scalars_origin["z"].attrs["dz"],
        ]
    )

    if ds_traj_posn_origin["x"].ndim > 0:
        grid_spacing = grid_spacing[:, np.newaxis]

    # Generate interpolator for repeated interpolation of fields during
    # iteration.
    interpolator = gen_interpolator_3d_fields(
        ds_position_scalars_next, interp_order=interp_order, cyclic_boundaries="xy"
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

    ds_traj_posn_next_est = xr.Dataset()
    for c in "xyz":
        ds_traj_posn_next_est[c] = 2.0 * ds_traj_posn_origin[c] - ds_traj_posn_prev[c]

    if "fixed_point_iterator" in solver:

        ds_traj_posn_next, err, dist = _backtrack_origin_point_iterate(
            ds_position_scalars_next,
            ds_traj_posn_origin,
            ds_traj_posn_next_est,
            interpolator,
            solver,
            interp_order=interp_order,
            minim_kwargs=minim_kwargs,
            **point_iter_kwargs,
        )

    else:

        ds_traj_posn_next, dist = _ds_backtrack_origin_optimize(
            ds_position_scalars_next,
            ds_traj_posn_origin,
            ds_traj_posn_next_est,
            interpolator,
            solver,
            interp_order=interp_order,
            kwargs=minim_kwargs,
        )

    if ds_grid.xy_periodic:
        ds_traj_posn_next = _wrap_coords(ds_traj_posn_next, ds_grid)

    # Copy in final error measure for each trajectory.
    for i, c in enumerate("xyz"):
        derr = xr.DataArray(
            dist[i, :], coords={"trajectory_number": np.arange(len(dist[i, :]))}
        )
        ds_traj_posn_next[f"{c}_err"] = derr

    # Set time coordinate.
    ds_traj_posn_next = ds_traj_posn_next.assign_coords(
        time=ds_position_scalars_next.time
    )

    return ds_traj_posn_next


def forward(
    ds_position_scalars,
    ds_back_trajectory,
    da_times,
    interp_order=5,
    solver="fixed_point_iterator",
    point_iter_kwargs=None,
    minim_kwargs=None,
):
    """
    Integrate trajectory forward one timestep.

    Using the position scalars `ds_position_scalars` integrate forwards from
    the last point in `ds_back_trajectory` to the times in `da_times`. The
    backward trajectory must contain at least two points in time as these are
    used for the initial guess for the forward extrapolation.

    The output dataset contains both the trajectory position and the
    residual error.

    Three solvers are available.

    The first uses the minimizer scipy.optimize.minimize to minimise the
    residual error. The keyword 'solver' should be set to one of the options
    for this function. The default (recommended) is 'BFGS'.

    See _ds_backtrack_origin_optimize for available options.

    The second uses point iteration. Selected using keyword 'solver' set to
    "fixed_point_iterator". This is generally much faster than
    scipy.optimize.minimize, but some trajectories do not converge.

    See _backtrack_origin_point_iterate for available options.

    The third is a combination of the first two, selected using keyword
    'solver' set to 'hybrid_fixed_point_iterator'.

    See _backtrack_origin_point_iterate for available options.

    """
    if ds_back_trajectory.time.count() < 2:
        raise Exception(
            "The back trajectory must contain at least two points for the forward"
            " extrapolation have an initial guess for the direction"
        )

    # create a copy of the backward trajectory so that we have a dataset into
    # which we will accumulate the full trajectory
    ds_traj = ds_back_trajectory.copy()

    if da_times.size == 0:
        print("No forward trajectories requested")
        return ds_traj

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
        # ds_traj_posn_prev = ds_traj.isel(time=[-3,-2])

        ds_traj_posn_est = _extrapolate_single_timestep(
            ds_position_scalars_origin=ds_position_scalars_origin,
            ds_position_scalars_next=ds_position_scalars_next,
            ds_traj_posn_prev=ds_traj_posn_prev,
            ds_traj_posn_origin=ds_traj_posn_origin,
            interp_order=interp_order,
            solver=solver,
            point_iter_kwargs=point_iter_kwargs,
            minim_kwargs=minim_kwargs,
        )

        ds_traj = xr.concat([ds_traj, ds_traj_posn_est], dim="time")

    return ds_traj
