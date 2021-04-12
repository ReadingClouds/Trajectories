import numpy as np
import xarray as xr


def _calculate_phase(vr, vi, n_grid):
    """
    Function to convert real and imaginary points to location on grid
    size n_grid.

    Args:
        vr, vi  : real and imaginary parts of complex location.
        n_grid  : grid size

    Returns:
        Real position in [0,n)

    @author: Peter Clark

    """

    vpos = (np.arctan2(vi, vr) / (2.0 * np.pi)) * n_grid
    vpos[vpos < 0] += n_grid
    return vpos


def _estimate_dim_initial_grid_locations(ds_position_scalars, dim, xy_periodic, n_grid):
    """
    Using the position scalars in `ds_position_scalars` estimate the grid
    locations (in indecies) where the fluid was initially located when the
    position scalars were initiated
    """
    if dim in ["x", "y"] and xy_periodic:
        da_vr = ds_position_scalars[f"traj_tracer_{dim}r"]
        da_vi = ds_position_scalars[f"traj_tracer_{dim}i"]
        grid_idx = _calculate_phase(vr=da_vr, vi=da_vi, n_grid=n_grid)
    else:
        grid_idx = ds_position_scalars[f"traj_tracer_{dim}r"]
    return grid_idx


def estimate_initial_grid_locations(ds_position_scalars, xy_periodic, nx, ny):
    """
    Using the position scalars `ds_position_scalars` estimate the original grid
    locations (ijk-indecies) that the position scalars were advected from
    """
    i_idx = _estimate_dim_initial_grid_locations(
        ds_position_scalars=ds_position_scalars,
        dim="x",
        xy_periodic=xy_periodic,
        n_grid=nx,
    )
    j_idx = _estimate_dim_initial_grid_locations(
        ds_position_scalars=ds_position_scalars,
        dim="y",
        xy_periodic=xy_periodic,
        n_grid=ny,
    )
    k_idx = _estimate_dim_initial_grid_locations(
        ds_position_scalars=ds_position_scalars,
        dim="z",
        xy_periodic=None,
        n_grid=ny,
    )
    return (i_idx, j_idx, k_idx)


def grid_locations_to_position_scalars(i, j, k, nx, ny, nz, xy_periodic):
    """
    Based off `reinitialise_trajectories` in
    `components/tracers/src/tracers.F90` in the MONC model source code
    """
    pi = np.pi

    ds = xr.Dataset()
    if xy_periodic:
        ds["traj_tracer_xr"] = np.cos(2.0 * pi * i / nx)
        ds["traj_tracer_xi"] = np.sin(2.0 * pi * i / ny)
        ds["traj_tracer_yr"] = np.cos(2.0 * pi * j / ny)
        ds["traj_tracer_yi"] = np.sin(2.0 * pi * j / ny)
    else:
        ds["traj_tracer_xr"] = i
        ds["traj_tracer_yr"] = j
    ds["traj_tracer_zr"] = k
    return ds