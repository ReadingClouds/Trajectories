"""
Routines for interpolating 3D scalar fields to arbitrary positions in domains
with (optional) cyclic boundary conditions
"""

import numpy as np
import xarray as xr

#from ..lib import fast_interp # this would be better as a separate install.
import fast_interp
from .grid import find_grid_spacing


def map_1d_grid_index_to_position(idx_grid, da_coord):
    """
    Map indecies `idx_grid` to the positios in grid defined by the
    cell-centered positions in `da_coord`.

    We assume that the grid-indecies map to the cell-center positions, so
    that for a grid resolution `dx=25.0m` and a grid with two cells with
    a domain of length 50.0 we have the following:

        i:             0           1
        x:      0.0  12.5  25.0  23.5  50.0
                 |     x     |     x     |

    i_est:     -0.5    0    0.5    1    1.5

    We need to allow for the estimated grid indecies `i_est` to map to grid
    positions up to the domain edges, which is outside of the cell-center
    positions. This is done by making the interpolation extend linearly outside
    the value range (by one index at either end)
    """

    N = int(da_coord.count())
    # use linear interpolation because grid is assumed to be isotropic
    interp_order = 1
    fn_e = fast_interp.interp1d(0, N, 1, da_coord.values, e=1, k=interp_order)
    pos = fn_e(np.array(idx_grid))

    if np.any(np.isnan(pos)):
        raise Exception("Found nan during interpolation")

    return pos


def interpolate_3d_field(da, ds_positions, interp_order=1, cyclic_boundaries=[]):
    """
    Perform interpolation of xr.DataArray `da` at positions given by data
    variables in `ds_posisions` with interpolation order `interp_order`. Cyclic
    boundary conditions are used by providing a `list` of the coordinates which
    have cyclic boundaries, e.g. (`cyclic_boundaries = 'xy'` or
    `cyclic_boundaries = ['x', 'y']`)
    """
#    dx, dy, dz = find_grid_spacing(da, coords="xyz")

    c_min = np.array([da[c].min().values for c in da.dims])
    c_max = np.array([da[c].max().values for c in da.dims])
#    dX = np.array([dx, dy, dz])
    dX = np.array([da[c].attrs[f'd{c}'] for c in da.dims])
    periodicity = [c in cyclic_boundaries for c in da.dims]

    fn = fast_interp.interp3d(
        a=c_min, b=c_max, h=dX, f=da.values, k=interp_order, p=periodicity
    )

    vals = fn(*[ds_positions[c].values for c in da.dims])

    da_interpolated = xr.DataArray(
        vals,
        dims=ds_positions.dims,
        coords=ds_positions.coords,
        attrs=da.attrs,
        name=da.name,
    )

    return da_interpolated

def interpolate_from_interpolator(v, ds_positions, fn):
    """
    Perform interpolation of variable named 'v' at positions given by data
    variables in `ds_positions` using interpolator fn.
    """

    vals = fn(*[ds_positions[c].values for c in 'xyz'])

    da_interpolated = xr.DataArray(
        vals,
        dims=ds_positions.dims,
        coords=ds_positions.coords,
#        attrs=da.attrs, # do we need these?
        name=v,
    )

    return da_interpolated

def interpolate_3d_fields(ds,
                          ds_positions,
                          interpolator=None,
                          interp_order=1,
                          cyclic_boundaries=None):
    """
    Perform interpolation of xr.DataArray `ds` at positions given by data
    variables in `ds_positions`.
    If interpolator provided, look in this for pre-generated fast_inter
    interpolator for each variable.
    Otherwise, interpolate with interpolation order `interp_order`. Cyclic
    boundary conditions are used by providing a `list` of the coordinates which
    have cyclic boundaries, e.g. (`cyclic_boundaries = 'xy'` or
    `cyclic_boundaries = ['x', 'y']`)
    """
    dataarrays = []

    for v in ds.data_vars:
        if interpolator is not None and v in interpolator:

            da_interpolated = interpolate_from_interpolator(
                                v, ds_positions, interpolator[v])

        else:

            da_interpolated = interpolate_3d_field(
                da=ds[v],
                ds_positions=ds_positions,
                interp_order=interp_order,
                cyclic_boundaries=cyclic_boundaries,
            )
        dataarrays.append(da_interpolated)

    ds_interpolated = xr.merge(dataarrays)
    ds_interpolated.attrs.update(ds.attrs)
    return ds_interpolated

def gen_interpolator_3d_field(da, interp_order=1, cyclic_boundaries=None):
    """
    Generate fast_interp interpolators for xr.DataArray `da` at positions with
    interpolation order `interp_order`. Cyclic
    boundary conditions are used by providing a `list` of the coordinates which
    have cyclic boundaries, e.g. (`cyclic_boundaries = 'xy'` or
    `cyclic_boundaries = ['x', 'y']`)
    """
    c_min = np.array([da[c].min().values for c in da.dims])
    c_max = np.array([da[c].max().values for c in da.dims])
    dX = np.array([da[c].attrs[f'd{c}'] for c in da.dims])
    periodicity = [c in cyclic_boundaries for c in da.dims]

    fn = fast_interp.interp3d(
        a=c_min, b=c_max, h=dX, f=da.values, k=interp_order, p=periodicity
    )
    return fn


def gen_interpolator_3d_fields(ds, interp_order=1, cyclic_boundaries=None):
    """
    Generate fast_interp interpolators for xr.DataArray `da` at positions with
    interpolation order `interp_order`. Cyclic
    boundary conditions are used by providing a `list` of the coordinates which
    have cyclic boundaries, e.g. (`cyclic_boundaries = 'xy'` or
    `cyclic_boundaries = ['x', 'y']`)
    """
    interpolators = {}
    for v in ds.data_vars:
        interpolators[v] = gen_interpolator_3d_field(
            da=ds[v],
            interp_order=interp_order,
            cyclic_boundaries=cyclic_boundaries,
        )

    return interpolators
