import numpy as np
import pytest
import xarray as xr
from utils import create_uniform_grid

from advtraj.utils import grid as grid_utils


def test_wrap_coord_posn():
    Lx = 1000.0
    x = np.array([-100.0, 0.0, Lx / 2.0, Lx, Lx + 100.0])
    x_wrapped_true = np.array([Lx - 100.0, 0.0, Lx / 2.0, 0.0, 100.0])
    x_wrapped = grid_utils.wrap_posn(x=x, x_min=0.0, x_max=Lx)
    np.testing.assert_allclose(x_wrapped_true, x_wrapped)


@pytest.mark.parametrize("cell_centered", [True, False])
def test_cyclic_coord_wrapping(cell_centered):
    dx = 25.0
    dL = (dx, dx, dx)
    L = (1.0e3, 1.0e3, 500.0)
    ds_grid = create_uniform_grid(dL=dL, L=L, cell_centered=cell_centered)

    Lx_c, Ly_c, Lz_c = [L[0] / 2.0, L[1] / 2.0, L[2] / 2.0]
    Lx, Ly, Lz = L

    start_and_wrapped_pt_coords = [
        # a point in the center of the domain should remain the same
        ((Lx_c, Ly_c, Lz_c), (Lx_c, Ly_c, Lz_c)),
        # wrapping in x should map these points to domain center
        ((Lx_c - Lx, Ly_c, Lz_c), (Lx_c, Ly_c, Lz_c)),
        ((Lx_c + Lx, Ly_c, Lz_c), (Lx_c, Ly_c, Lz_c)),
        # same in y
        ((Lx_c, Ly_c - Ly, Lz_c), (Lx_c, Ly_c, Lz_c)),
        ((Lx_c, Ly_c + Ly, Lz_c), (Lx_c, Ly_c, Lz_c)),
        # repeats for two wraps
        ((Lx_c - 2 * Lx, Ly_c, Lz_c), (Lx_c, Ly_c, Lz_c)),
        ((Lx_c + 2 * Lx, Ly_c, Lz_c), (Lx_c, Ly_c, Lz_c)),
        ((Lx_c, Ly_c - 2 * Ly, Lz_c), (Lx_c, Ly_c, Lz_c)),
        ((Lx_c, Ly_c + 2 * Ly, Lz_c), (Lx_c, Ly_c, Lz_c)),
    ]

    def _make_pt_dataset(pt):
        ds_pt = xr.Dataset()
        for n, v in enumerate(["x", "y", "z"]):
            ds_pt[v] = pt[n]
        return ds_pt

    def _pt_from_dataset(ds_pt):
        return np.array([ds_pt[v] for v in ["x", "y", "z"]])

    if cell_centered:
        cell_centered_coords = ["x", "y", "z"]
    else:
        cell_centered_coords = []

    for pt_start, pt_wrapped_correct in start_and_wrapped_pt_coords:
        ds_pt_start = _make_pt_dataset(pt_start)
        ds_pt_wrapped = grid_utils.wrap_periodic_grid_coords(
            ds_grid=ds_grid,
            ds_posn=ds_pt_start,
            cyclic_coords=("x", "y"),
            cell_centered_coords=cell_centered_coords,
        )

        np.testing.assert_allclose(_pt_from_dataset(ds_pt_wrapped), pt_wrapped_correct)
