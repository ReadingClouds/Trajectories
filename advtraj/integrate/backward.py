from ..utils import estimate_initial_grid_locations

from scipy import ndimage
import numpy as np
import xarray as xr


def grid_location_to_3d_position(ds_grid, ds_grid_locations, interp_order=1):
    """
    Using the 3D grid positions (in real units, not grid indecies) defined in
    `ds_grid` (through coordinates `x`, `y` and `z`) interpolate the "grid
    indecies" in `ds_grid_locations` (these may be fractional, i.e. they are
    not discrete integer grid indecies) to the real x, y and z-positions.
    """

    def _1d_idx_to_posn(idx_grid, dim):
        if dim in ["x", "y"]:
            return ndimage.map_coordinates(
                ds_grid[dim], [[idx_grid]], mode="wrap", order=interp_order
            )
        else:
            return ndimage.map_coordinates(
                ds_grid[dim],
                [[idx_grid]],
                mode="constant",
                cval=np.nan,
                order=interp_order,
            )

    x_pos = _1d_idx_to_posn(ds_grid_locations.x_grid_loc, dim="x")[0]
    y_pos = _1d_idx_to_posn(ds_grid_locations.y_grid_loc, dim="y")[0]
    z_pos = _1d_idx_to_posn(ds_grid_locations.z_grid_loc, dim="z")[0]

    return [x_pos, y_pos, z_pos]


def _extrapolate_single_timestep(ds_position_scalars, ds_traj_posn):
    # the position scalars at the current time will enable us to determine
    # where the fluid started off
    ds_initial_grid_locs = estimate_initial_grid_locations(
        ds_position_scalars=ds_position_scalars
    )

    # interpolate these grid-positions from the position scalars so that we can
    # get an actual xyz-position
    ds_traj_init_grid_locs = ds_initial_grid_locs.interp(ds_traj_posn)
    xyz_traj_posn_new = grid_location_to_3d_position(
        ds_grid=ds_position_scalars, ds_grid_locations=ds_traj_init_grid_locs
    )

    return xyz_traj_posn_new


def main(ds_position_scalars, ds_starting_point, da_times, interp_order=1):
    """
    Using the position scalars `ds_position_scalars` integrate backwards from
    `ds_starting_point` to the times in `da_times`

    The algorithm is as follows:

    1) for a trajectory position `(x,y,z)` at a time `t` interpolate the
    "position scalars" to find their value at `(x,y,z,t)`
    2) estimate the initial location that the fluid at `(x,y,z,t)` came from by
    converting the "position scalars" back to position
    """
    # create a list into which we will accumulate the trajectory points
    # while doing this we turn the time variable into a coordinate
    datasets = [
        ds_starting_point.drop("time").assign_coords(time=ds_starting_point.time)
    ]

    # steo back in time
    for t_current in da_times.values[::-1]:
        xyz_traj_posn_new = _extrapolate_single_timestep(
            ds_position_scalars=ds_position_scalars.sel(time=t_current).drop("time"),
            ds_traj_posn=datasets[-1].drop("time"),
        )
        print(xyz_traj_posn_new)
        # and wrap this up as a new point for the trajectory
        t_current = ds_starting_point.time
        # find the previous time so that we can construct a new dataset to contain
        # the trajectory position at the previous time
        t_previous = ds_position_scalars.time.sel(time=slice(None, t_current)).isel(
            time=-2
        )
        ds_traj_prev = xr.Dataset(coords=dict(time=t_previous))
        for n, c in enumerate(["x", "y", "z"]):
            ds_traj_prev[c] = xr.DataArray(
                xyz_traj_posn_new[n], attrs=ds_position_scalars[c].attrs, name=c
            )
        datasets.append(ds_traj_prev)

    ds_traj = xr.concat(datasets[::-1], dim="time")
    return ds_traj
