import numpy as np
import xarray as xr

import advtraj.utils.grid_mapping as advtraj_gm_utils


def create_uniform_grid(dL, L):
    Lx, Ly, Lz = L
    dx, dy, dz = dL

    # create cell-center positions
    x_ = np.arange(0, Lx, dx) + dx / 2.0
    y_ = np.arange(0, Ly, dy) + dy / 2.0
    z_ = np.arange(0, Lz, dz) + dz / 2.0

    ds = xr.Dataset(coords=dict(x=x_, y=y_, z=z_))
    ds.x.attrs["units"] = "m"
    ds.x.attrs["dx"] = dx
    ds.y.attrs["units"] = "m"
    ds.y.attrs["dy"] = dy
    ds.z.attrs["units"] = "m"
    ds.z.attrs["dz"] = dz
    ds.x.attrs["long_name"] = "x-horz. posn."
    ds.y.attrs["long_name"] = "y-horz. posn."
    ds.z.attrs["long_name"] = "height"

    return ds


def create_initial_dataset(dL, L, xy_periodic=True):
    """
    Create an initial dataset with a uniform grid and the position scalars
    initiatied to the locations in the grid
    """
    dx, dy, dz = dL
    ds_grid = create_uniform_grid(dL=dL, L=L)

    ds_grid["xy_periodic"] = xy_periodic

    ds = init_position_scalars(ds=ds_grid)
    ds["time"] = np.datetime64("2020-01-01T00:00")

    return ds


def init_position_scalars(ds):
    """
    Add or replace the position scalars in the dataset `ds` using the grid
    defined in there (through the coordinates `x`, `y` and `z`)
    """
    ds_position_scalars = advtraj_gm_utils.grid_locations_to_position_scalars(
        ds_grid=ds, ds_pts=None
    )

    # get rid of the position scalars if they're already in the dataset, since
    # we want to reset them in that case
    for v in ds_position_scalars.data_vars:
        if v in ds.data_vars:
            ds = ds.drop(v)

    xy_periodic = ds.xy_periodic
    ds = xr.merge([ds, ds_position_scalars])
    ds.attrs["xy_periodic"] = xy_periodic

    return ds
