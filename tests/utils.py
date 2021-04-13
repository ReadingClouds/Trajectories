import numpy as np
import xarray as xr

import advtraj.utils as advtraj_utils


def create_uniform_grid(dL, L):
    Lx, Ly, Lz = L
    dx, dy, dz = dL

    x_ = np.arange(0, Lx, dx)
    y_ = np.arange(0, Ly, dy)
    z_ = np.arange(0, Lz, dz)

    ds = xr.Dataset(coords=dict(x=x_, y=y_, z=z_))
    ds.x.attrs["units"] = "m"
    ds.y.attrs["units"] = "m"
    ds.z.attrs["units"] = "m"
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

    # make 3D arrays out of the 1D grid positions
    x, y, z = xr.broadcast(ds_grid.x, ds_grid.y, ds_grid.z)

    i = x / dx
    j = y / dy
    k = z / dz
    nx = ds_grid.x.count()
    ny = ds_grid.y.count()
    nz = ds_grid.z.count()

    ds_position_scalars = advtraj_utils.grid_locations_to_position_scalars(
        i=i, j=j, k=k, nx=nx, ny=ny, nz=nz, xy_periodic=xy_periodic
    )

    ds = xr.merge([ds_grid, ds_position_scalars])

    ds["time"] = np.datetime64("2020-01-01T00:00")
    ds.attrs["xy_periodic"] = xy_periodic

    return ds
