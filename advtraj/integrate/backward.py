"""
Functionality for computing trajectories backward from a set of starting points
at a single point in time using the position scalars.
"""
import xarray as xr
from tqdm import tqdm

from ..utils.grid_mapping import (
    estimate_3d_position_from_grid_indecies,
    estimate_initial_grid_indecies,
)
from ..utils.interpolation import interpolate_3d_fields


def calc_trajectory_previous_position(
    ds_position_scalars, ds_traj_posn, grid_interpolation_order=5
):
    """
    The algorithm is as follows:

    1) for a trajectory position `(x,y,z)` at a time `t` interpolate the
    "position scalars" to find their value at `(x,y,z,t)`
    2) estimate the initial indecies that the fluid at `(x,y,z,t)` came from by
    converting the "position scalars" back to position
    """
    # interpolate the position scalar values at the current trajectory
    # position
    ds_initial_position_scalar_locs = interpolate_3d_fields(
        ds=ds_position_scalars,
        ds_positions=ds_traj_posn,
        interp_order=grid_interpolation_order,
        cyclic_boundaries="xy" if ds_position_scalars.xy_periodic else None,
    )

    # convert these position scalar values to grid positions so we can estimate
    # what grid positions the fluid was advected from
    nx, ny = int(ds_position_scalars.x.count()), int(ds_position_scalars.y.count())
    ds_traj_init_grid_idxs = estimate_initial_grid_indecies(
        ds_position_scalars=ds_initial_position_scalar_locs, N_grid=dict(x=nx, y=ny)
    )

    # interpolate these grid-positions from the position scalars so that we can
    # get an actual xyz-position
    ds_traj_posn_prev = estimate_3d_position_from_grid_indecies(
        ds_grid=ds_position_scalars,
        i=ds_traj_init_grid_idxs.i,
        j=ds_traj_init_grid_idxs.j,
        k=ds_traj_init_grid_idxs.k,
        interp_order=grid_interpolation_order,
    )

    return ds_traj_posn_prev


def backward(ds_position_scalars, ds_starting_point, da_times, interp_order=1):
    """
    Using the position scalars `ds_position_scalars` integrate backwards from
    `ds_starting_point` to the times in `da_times`
    """
    # create a list into which we will accumulate the trajectory points
    # while doing this we turn the time variable into a coordinate
    datasets = [ds_starting_point]

    # step back in time, `t_current` represents the time we're of the next
    # point (backwards) of the trajectory
    for t_current in tqdm(da_times.values[1:][::-1], desc="backward"):
        ds_traj_posn_origin = datasets[-1].drop_vars("time")
        ds_position_scalars_current = ds_position_scalars.sel(time=t_current).drop_vars(
            "time"
        )

        ds_traj_posn_est = calc_trajectory_previous_position(
            ds_position_scalars=ds_position_scalars_current,
            ds_traj_posn=ds_traj_posn_origin,
        )
        # find the previous time so that we can construct a new dataset to contain
        # the trajectory position at the previous time
        try:
            t_previous = ds_position_scalars.time.sel(time=slice(None, t_current)).isel(
                time=-2
            )
        except IndexError:
            # this will happen if we're trying to integrate backwards from the
            # very first timestep, which we can't (and shouldn't). Just check
            # we have as many trajectory points as we're aiminng for
            if len(datasets) == da_times.count():
                break
            else:
                raise

        ds_traj_posn_prev = xr.Dataset()
        for n, c in enumerate(["x", "y", "z"]):
            ds_traj_posn_prev[c] = xr.DataArray(
                ds_traj_posn_est[f"{c}_est"].item(),
                attrs=ds_position_scalars[c].attrs,
                name=c,
            )
        ds_traj_posn_prev["time"] = t_previous
        datasets.append(ds_traj_posn_prev)

    ds_traj = xr.concat(datasets[::-1], dim="time")
    return ds_traj
