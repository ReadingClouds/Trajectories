"""
    grid.py

    Utilities to deal with grid wrapping etc.

    Currently, caters for the following grid styles:

    "cell_centred"
    - p-point (cell centre) p[0, 0, 0]] at x = dx/2, y = dy/2, z = dz/2.
    - u-point (cell face) u[0, 0, 0] at p[0, 0, 0] - dx/2, i.e. x = 0.
    - v-point (cell face) v[0, 0, 0] at p[0, 0, 0] - dy/2, i.e. y = 0.
    - w-point (cell face) w[0, 0, 0] at p[0, 0, 0] - dz/2, i.e. z = 0.
    - Virtual p point at z = -dz/2.

    "monc"
    - Virtual p points at z = -dz/2 so:
    - p-point (cell centre) p[0, 0, 0]] at x = dx/2, y = dy/2, z = -dz/2.
    - u-point (cell face) u[0, 0, 0] at p[0, 0, 0] + dx/2, i.e. x = dx.
    - v-point (cell face) v[0, 0, 0] at p[0, 0, 0] + dy/2, i.e. y = dy.
    - w-point (cell face) w[0, 0, 0] at p[0, 0, 0] + dz/2, i.e. z = 0.

    ""
    - All points [0, 0, 0] at x = 0, y = 0, z = 0.
"""
import warnings

import numpy as np


def wrap_posn(x, x_min, x_max):
    """
    Wrap coordinate positions `x` so that they lie inside the range [x_min, x_max[
    """
    lx = x_max - x_min
    x_ = x - x_min

    r = x_ / lx

    N_wrap = np.where(r >= 0.0, r.astype(int), r.astype(int) - 1.0)

    x_wrapped = x - lx * N_wrap

    return x_wrapped


def find_coord_grid_spacing(da_coord, show_warnings=True):
    grid_tol = 0.001
    
    v_name = f"d{da_coord.name}"
    if v_name in da_coord.attrs:
        return da_coord.attrs[v_name]
    else:
        if show_warnings:
            warnings.warn(
                f"The grid spacing isn't currently set for coordinate `{da_coord.name}`"
                f" to speed up calculations and ensure the grid-spacing is set correctly"
                f" set the `{v_name}` attribute of the `{da_coord.name}` coordinate"
                " to the value of the grid-spacing"
            )

    dx_all = np.diff(da_coord.values)

    if (np.max(dx_all) - np.min(dx_all)) / np.mean(dx_all) > grid_tol :
        raise Exception("Non-uniform grid")

    return np.mean(dx_all)


def find_grid_spacing(ds_grid, coords=("x", "y", "z")):
    return [find_coord_grid_spacing(ds_grid[c]) for c in coords]


def wrap_periodic_grid_coords(
    ds_grid, ds_posn, cyclic_coords=("x", "y"), cell_centered_coords=("x", "y")
):
    """
    ensure that positions given by `ds_posn` are inside the coordinates of
    `ds_grid`. Use `cell_centered_coords` to define which coords in `ds_grid`
    are using cell-centered
    """

    ds_posn_copy = ds_posn.copy()

    for c in cyclic_coords:
        da_coord = ds_grid[c]
        dx = find_coord_grid_spacing(da_coord=da_coord)

        x_min, x_max = da_coord.min().data, da_coord.max().data
        if c in cell_centered_coords:
            x_min -= dx / 2.0
            x_max += dx / 2.0
        else:
            x_max += dx

        # Now wrap the position where needed
        wrapped_x = wrap_posn(ds_posn_copy[c].values, x_min=x_min, x_max=x_max)

        # Now update variable c with wrapped values, including dim[0],
        # which should be 'trajectory_number', if available.
        # Do not include dim if it's not in the orriginal.

        d = ds_posn_copy[c].dims
        if len(d) > 0:
            ds_posn_copy = ds_posn_copy.update({c: (d[0], wrapped_x)})
        else:
            ds_posn_copy = ds_posn_copy.update({c: wrapped_x})
    return ds_posn_copy
