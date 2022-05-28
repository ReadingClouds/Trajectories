import numpy as np
import xarray as xr

import advtraj.utils.grid_mapping as advtraj_gm_utils


def cust_range(*args, rtol=1e-05, atol=1e-08, include=[True, False]):
    """
    Combines numpy.arange and numpy.isclose to mimic
    open, half-open and closed intervals.
    Avoids also floating point rounding errors as with
    >>> numpy.arange(1, 1.3, 0.1)
    array([1. , 1.1, 1.2, 1.3])

    args: [start, ]stop, [step, ]
        as in numpy.arange
    rtol, atol: floats
        floating point tolerance as in numpy.isclose
    include: boolean list-like, length 2
        if start and end point are included

    source: https://stackoverflow.com/a/57321916/271776
    """
    # process arguments
    if len(args) == 1:
        start = 0
        stop = args[0]
        step = 1
    elif len(args) == 2:
        start, stop = args
        step = 1
    else:
        assert len(args) == 3
        start, stop, step = tuple(args)

    # determine number of segments
    n = (stop - start) / step + 1

    # do rounding for n
    if np.isclose(n, np.round(n), rtol=rtol, atol=atol):
        n = np.round(n)

    # correct for start/end is exluded
    if not include[0]:
        n -= 1
        start += step
    if not include[1]:
        n -= 1
        stop -= step

    return np.linspace(start, stop, int(n))


def crange(*args, **kwargs):
    """Create range guaranteed to include max-value of range (if the step-size
    should give an even number of steps)"""
    return cust_range(*args, **kwargs, include=[True, True])


def orange(*args, **kwargs):
    """Create range guaranteed to exclude max-value of range"""
    return cust_range(*args, **kwargs, include=[True, False])


def create_uniform_grid(dL, L, grid_style="cell_centred"):
    """
    Create a grid with uniform resolution in x,y and z with domain spanning
    [0,0,0] to `L` with grid resolution `dL`


    """
    Lx, Ly, Lz = L
    dx, dy, dz = dL

    if grid_style == "cell_centred":
        # create cell-centre positions
        x_ = crange(dx / 2.0, Lx - dx / 2.0, dx)
        y_ = crange(dy / 2.0, Ly - dy / 2.0, dy)
        z_ = crange(dz / 2.0, Lz - dz / 2.0, dz)
    elif grid_style == "monc":
        # create wrapped positions starting at 0.
        x_ = crange(dx / 2.0, Lx - dx / 2.0, dx)
        y_ = crange(dy / 2.0, Ly - dy / 2.0, dy)
        # create cell-centred including virtual point below surface.
        z_ = crange(-dz / 2.0, Lz - dz / 2.0, dz)
    else:
        x_ = crange(0.0, Lx - dx, dx)
        y_ = crange(0.0, Ly - dy, dy)
        z_ = crange(0.0, Lz, dz)

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
    initiated to the locations in the grid.
    """
    dx, dy, dz = dL
    ds_grid = create_uniform_grid(dL=dL, L=L)

    ds_grid.attrs["xy_periodic"] = xy_periodic

    ds = init_position_scalars(ds=ds_grid)
    ds = ds.assign_coords(time=np.datetime64("2020-01-01T00:00"))

    for i, c in enumerate("xyz"):

        ds[c].attrs[f"d{c}"] = dL[i]
        ds[c].attrs[f"L{c}"] = L[i]

    return ds


def init_position_scalars(ds):
    """
    Add or replace the position scalars in the dataset `ds` using the grid
    defined in there (through the coordinates `x`, `y` and `z`).
    """
    ds_position_scalars = advtraj_gm_utils.grid_locations_to_position_scalars(
        ds_grid=ds, ds_pts=None
    )

    # get rid of the position scalars if they're already in the dataset, since
    # we want to reset them in that case
    for v in ds_position_scalars.data_vars:
        if v in ds.data_vars:
            ds = ds.drop(v)

    ds = xr.merge([ds, ds_position_scalars])
    # Add residual errors.
    ds["x_err"] = xr.zeros_like(ds["x"])
    ds["y_err"] = xr.zeros_like(ds["y"])
    ds["z_err"] = xr.zeros_like(ds["z"])

    return ds
