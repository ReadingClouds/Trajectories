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

def _calc_backtrack_origin_dist(ds_position_scalars,
                                ds_traj_posn_org, 
                                ds_traj_posn, 
                                interpolator=None,
                                interp_order=1,
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
            ds_p1.x - ds_p2.x_est,
            ds_p1.y - ds_p2.y_est,
            ds_p1.z - ds_p2.z_est,
        ]
    )
    
    # Deal with xy wraparound.
    if ds_grid.xy_periodic:
        Lx = ds_position_scalars['x'].attrs['Lx']
        Ly = ds_position_scalars['y'].attrs['Ly']
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

def _pt_ds_to_arr(ds_pt):
    return np.array([ds_pt[c].data for c in "xyz"])

def _pt_arr_to_ds(arr_pt):
    # Ensure arr_pt has 2 dimensions.
    arr_pt = arr_pt.reshape((3,-1))
    ds_pt = xr.Dataset()
    for n, c in enumerate("xyz"):
        ds_pt[c] = xr.DataArray(arr_pt[n,:], dims=("trajectory_number"))

    return ds_pt

def _extrapolate_single_timestep(
    ds_position_scalars_origin,
    ds_position_scalars_next,
    ds_traj_posn_prev,
    ds_traj_posn_origin,
    solver='PI',
    interp_order=1,
    options=None,
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
    solver: Optional (default='PI').
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

    def _pt_backtrack_origin_optimize(ds_position_scalars,
                                      ds_traj_posn_org, 
                                      pt_traj_posn_first_guess,
                                      interpolator,
                                      minimization_method,
                                      interp_order=interp_order,
                                      minoptions=None):
        
        options={'minimize_options':{}}
        if minoptions is not None:
            options.update(minoptions)
            
        max_outer_loops = options.get('max_outer_loops',1)
        tol = options.get('tol', 0.01)        

        def _calc_backtrack_origin_err(pt_traj_posn):

            ds_traj_posn = _pt_arr_to_ds(pt_traj_posn)

            dist_arr = _calc_backtrack_origin_dist(ds_position_scalars,
                                                   ds_traj_posn_org,
                                                   ds_traj_posn,
                                                   interpolator=interpolator,
                                                   interp_order=interp_order,
                                                   )

            err = np.linalg.norm(dist_arr.flatten(), ord = 2)
            return err

        # for the minimization we will be using just a numpy-array
        # containing the (x,y,z) location
        niter = 0
        while niter < max_outer_loops:
            sol = scipy.optimize.minimize(
                fun=_calc_backtrack_origin_err,
                x0=pt_traj_posn_first_guess,
                method=minimization_method,
                options = options['minimize_options'],
            )
            pt_traj_posn_next = sol.x
            if sol.fun/np.sqrt(sol.x.size) <= grid_size * tol:
                break
            niter += 1

        return pt_traj_posn_next
    
    def _ds_backtrack_origin_optimize(ds_position_scalars,
                                      ds_traj_posn_org,
                                      ds_traj_posn_first_guess,
                                      interpolator,
                                      minimization_method,
                                      minoptions=None):
        
        pt_traj_posn_next = _pt_ds_to_arr(ds_traj_posn_first_guess).flatten()
        
        pt_traj_posn_next = _pt_backtrack_origin_optimize(
                                ds_position_scalars,
                                ds_traj_posn_org,
                                pt_traj_posn_next,
                                interpolator,
                                minimization_method,
                                minoptions=minoptions)        

        # if sol.success:
        ds_traj_posn_next = _pt_arr_to_ds(pt_traj_posn_next)

        if ds_grid.xy_periodic:
            ds_traj_posn_next = _wrap_coords(ds_traj_posn_next, ds_grid)

        dist = _calc_backtrack_origin_dist(ds_position_scalars,
                                           ds_traj_posn_org,
                                           ds_traj_posn_next,
                                           interpolator=interpolator,
                                           interp_order=interp_order,
                                           )
        return ds_traj_posn_next, dist

    def _backtrack_origin_point_iterate(ds_position_scalars,
                                        ds_traj_posn_org,
                                        ds_traj_posn_first_guess,
                                        interpolator,
                                        solver,
                                        pioptions=None):
        
        options={}
        if pioptions is not None:
            options.update(pioptions)

        # defaults
        maxiter = options.get('maxiter', 500)
        miniter = options.get('miniter', 10)
        relax = options.get('relax', 0.8)
        tol = options.get('tol', 0.01)
        alt_delta = options.get('alt_delta', False)
        opt_one_step = options.get('opt_one_step', False)
        use_midpoint = options.get('use_midpoint', False)
        disp = options.get('disp', False)
        norm = options.get('norm', 'max_abs_error')
            # = options.get('', )

        def _norm(d):
            if norm == 'max_abs_error':
                return np.linalg.norm(d.flatten(), ord = np.inf)
            if norm == 'mean_abs_error':
                return np.mean(np.abs(d))
            else:
                return np.linalg.norm(d.flatten())/np.sqrt(d.size)
            
        dist = _calc_backtrack_origin_dist(ds_position_scalars,
                                           ds_traj_posn_org,
                                           ds_traj_posn_first_guess,
                                           interpolator=interpolator,
                                           interp_order=interp_order)

        ndist = dist/grid_spacing
        err = _norm(ndist)
        niter = 0
        not_converged = True
        
        ds_traj_posn_next = ds_traj_posn_first_guess
        
        dist_prev = dist.copy()

        while not_converged:
            
            
            if alt_delta:
                factor = np.clip(relax / (np.max(abs(ndist), axis=0) + tol),
                                 0 , 1)
                delta = dist * factor
            else:                  
                delta = np.clip(dist * relax, -grid_spacing, grid_spacing)
            # print(niter, relax, dist[:,2], delta[:,2])
            
            if use_midpoint:
                ds_traj_posn_prev = ds_traj_posn_next.copy()
                for i, c in enumerate('xyz'):
    
                    ds_traj_posn_next[c] = xr.where(
                                           dist_prev[i,...]*dist[i,...]<0,
                                           (ds_traj_posn_next[c] 
                                          + ds_traj_posn_prev[c]) * 0.5,
                                           ds_traj_posn_prev[c] + delta[i,...]
                                           )
                dist_prev = dist.copy()
            else:
                for i, c in enumerate('xyz'):
                    ds_traj_posn_next[c] += delta[i,...]
                
            
            dist = _calc_backtrack_origin_dist(ds_position_scalars,
                                               ds_traj_posn_org,
                                               ds_traj_posn_next,
                                               interpolator=interpolator,
                                               interp_order=interp_order)
            
            ndist = dist/grid_spacing
            err = _norm(ndist)
            niter += 1
            not_converged = (err > tol)
#            print(f"niter: {niter:3d} dist: {np.max(np.abs(dist), axis=1)} err: {err} relax: {relax}")
            if niter >= maxiter:
                break
            # If we need a lot of iterations, reduce the relaxation factor.
            if niter%miniter==0: relax *= 0.9
        
        if niter >= maxiter:
            # Select out those trajectories that have not converged.
            ncm = (np.abs(ndist) > tol)
            not_conv_mask = ncm[0, :] | ncm[1, :] | ncm[2, :]
            print(f"\nPoint Iteration failed to converge "
                  f"in {niter:3d} iterations. "
                  f"{np.sum(not_conv_mask)} trajectories did not converge. "
                  f"Final error = {err}")
           
            
        if solver == 'PI_hybrid' and niter >= maxiter:
            
            minoptions = {
                        'max_outer_loops': 1,
                        'minimize_options':{
                                            'maxiter': 50,
                                            'disp': True,
                                           }
                         }
            if options is not None and 'minoptions' in options:
                minoptions.update(options['minoptions'])
                            
            pt_traj_posn_next = _pt_ds_to_arr(ds_traj_posn_next)
            
            pt_traj_posn_next_not_conv = pt_traj_posn_next[:, not_conv_mask]            
            
            print(f"Optimising {pt_traj_posn_next_not_conv.shape[1]} "
                  f"unconverged trajectories.")
            
                                         
                                         
            # Select out corresponding 'true' value.
            
            ds_traj_posn_org_subset = ds_traj_posn_org[["x","y","z"]]\
                                        .isel(trajectory_number=not_conv_mask)
                                        
            if opt_one_step:
                pt_traj_posn_next_not_conv = pt_traj_posn_next_not_conv\
                    .flatten()
                pt_traj_posn_next_not_conv = _pt_backtrack_origin_optimize(
                                              ds_position_scalars,
                                              ds_traj_posn_org_subset, 
                                              pt_traj_posn_next_not_conv,
                                              interpolator,
                                              'BFGS',
                                              minoptions=minoptions)
            else:                                        
                for itraj in range(pt_traj_posn_next_not_conv.shape[1]):
                    
                    print(f"Optimising unconverged trajectory {itraj}.")
                    pt_traj_posn = pt_traj_posn_next_not_conv[:,itraj]\
                        .flatten()
                    ds_traj_posn_orig = ds_traj_posn_org_subset.isel(
                        trajectory_number=itraj)
                    pt_traj_posn = _pt_backtrack_origin_optimize(
                                              ds_position_scalars,
                                              ds_traj_posn_orig, 
                                              pt_traj_posn,
                                              interpolator,
                                              'BFGS',
                                              minoptions=minoptions)
                    
                    pt_traj_posn_next_not_conv[:,itraj] = pt_traj_posn
                
                        
                
            pt_traj_posn_next[:, not_conv_mask] = pt_traj_posn_next_not_conv\
                .reshape((3,-1))
                        
            ds_traj_posn_next = _pt_arr_to_ds(pt_traj_posn_next)
            
            dist = _calc_backtrack_origin_dist(ds_position_scalars,
                                               ds_traj_posn_org,
                                               ds_traj_posn_next,
                                               interpolator=interpolator,
                                               interp_order=interp_order)            
            
            ndist = dist/grid_spacing
            err = _norm(ndist)
            print(f"After minimization error={err}.")            
                        
        if disp:
            print(f"Point Iteration finished in {niter:3d} iterations. "
                  f"Final error = {err}")

        return ds_traj_posn_next, err, dist

###############################################################################

    interpolator = gen_interpolator_3d_fields(ds_position_scalars_next,
                                              interp_order=interp_order,
                                              cyclic_boundaries='xy')
    
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
    for c in 'xyz':
        ds_traj_posn_next_est[c] = 2 * ds_traj_posn_origin[c] \
                                      - ds_traj_posn_prev[c]
        # ds_traj_posn_next_est[c] = 2 * ds_traj_posn_origin[c] \
        #                            - ds_traj_posn_prev[c][1] \
        #                            + (2./3.) * (ds_traj_posn_prev[c][0]
        #                                       - ds_traj_posn_prev[c][1])


    if solver == 'PI' or solver == 'PI_hybrid':
        
        pioptions = {
                    'maxiter': 100,
                    'miniter': 10,
                    'disp': False,
                    'relax': 0.8,
                    'tol': 0.01,
                    'norm': 'max_abs_error',
                  }
        
        if options is not None and 'pioptions' in options:
            pioptions.update(options['pioptions'])
            
        ds_traj_posn_next, err, dist = _backtrack_origin_point_iterate(
                                        ds_position_scalars_next,
                                        ds_traj_posn_origin,
                                        ds_traj_posn_next_est,
                                        interpolator,
                                        solver,
                                        pioptions=pioptions)

    else:

        minoptions = {
                    'max_outer_loops': 4,
                    'minimize_options':{
                                        'maxiter': 50,
                                        'disp': False,
                                       }
                     }
        if options is not None and 'minoptions' in options:
            minoptions.update(options['minoptions'])
            
        ds_traj_posn_next, dist = _ds_backtrack_origin_optimize(
                                      ds_position_scalars_next,
                                      ds_traj_posn_origin,
                                      ds_traj_posn_next_est,
                                      interpolator,
                                      solver,
                                      minoptions=minoptions)

    if ds_grid.xy_periodic:
        ds_traj_posn_next = _wrap_coords(ds_traj_posn_next, ds_grid)
        
    # Copy in final error measure for each trajectory.
    for i, c in enumerate('xyz'):
        derr = dist[i, :]
        ds_traj_posn_next[f'{c}_err'] = xr.DataArray(derr)\
              .rename({'dim_0':'trajectory_number'})

    ds_traj_posn_next = ds_traj_posn_next.assign_coords(
        time=ds_position_scalars_next.time
    )

    return ds_traj_posn_next


def forward(ds_position_scalars, ds_back_trajectory, da_times, 
            interp_order=1, solver = 'PI', options=None):
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
        # ds_traj_posn_prev = ds_traj.isel(time=[-3,-2])

        ds_traj_posn_est = _extrapolate_single_timestep(
            ds_position_scalars_origin=ds_position_scalars_origin,
            ds_position_scalars_next=ds_position_scalars_next,
            ds_traj_posn_prev=ds_traj_posn_prev,
            ds_traj_posn_origin=ds_traj_posn_origin,
            interp_order=interp_order,
            solver=solver,
            options=options,
        )

        ds_traj = xr.concat([ds_traj, ds_traj_posn_est], dim="time")

    return ds_traj
