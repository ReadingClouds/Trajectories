import numpy as np
import xarray as xr
from utils import create_uniform_grid, init_position_scalars

import advtraj.utils.grid_mapping as advtraj_gm_utils


def test_position_scalar_transforms_are_symmetric():
    """
    Test that transforms both to and from the "position scalars" are symmetric
    """
    Lx = Ly = 0.5e3  # [m]
    Lz = 1.0e3  # [m]
    dx = dy = dz = 25.0  # [m]

    ds_grid = create_uniform_grid(dL=(dx, dy, dz), L=(Lx, Ly, Lz))
    nx, ny = int(ds_grid.x.count()), int(ds_grid.y.count())

    N_pts = 5
    ds_pts = xr.Dataset(coords=dict(pt=np.arange(N_pts)))
    ds_pts["x"] = "pt", np.linspace(ds_grid.x.min(), ds_grid.x.max(), N_pts)
    ds_pts["y"] = "pt", np.linspace(ds_grid.y.min(), ds_grid.y.max(), N_pts)
    ds_pts["z"] = "pt", np.linspace(ds_grid.z.min(), ds_grid.z.max(), N_pts)

    for xy_periodic in [True, False]:
        ds_grid["xy_periodic"] = xy_periodic
        ds_position_scalars_pts = advtraj_gm_utils.grid_locations_to_position_scalars(
            ds_grid=ds_grid, ds_pts=ds_pts
        )
        ds_grid_idxs_pts = advtraj_gm_utils.estimate_initial_grid_indecies(
            ds_position_scalars=ds_position_scalars_pts, N_grid=dict(x=nx, y=ny)
        )
        ds_pts_est = advtraj_gm_utils.estimate_3d_position_from_grid_indecies(
            ds_grid=ds_grid,
            i=ds_grid_idxs_pts.i,
            j=ds_grid_idxs_pts.j,
            k=ds_grid_idxs_pts.k,
        )

        np.testing.assert_allclose(ds_pts.x, ds_pts_est.x_est)
        np.testing.assert_allclose(ds_pts.y, ds_pts_est.y_est)
        np.testing.assert_allclose(ds_pts.z, ds_pts_est.z_est)


def test_position_scalars_translation():
    Lx = Ly = 0.5e3  # [m]
    Lz = 1.0e3  # [m]
    dx = dy = dz = 25.0  # [m]

    ds_grid = create_uniform_grid(dL=(dx, dy, dz), L=(Lx, Ly, Lz))
    ds_grid.attrs["xy_periodic"] = True
    nx, ny = int(ds_grid.x.count()), int(ds_grid.y.count())

    ds = init_position_scalars(ds=ds_grid)

    for i_shift in [0, 2, int(Lx / dx)]:
        ds_shifted = ds.roll(x=i_shift, roll_coords=False)

        ds_grid_idxs = advtraj_gm_utils.estimate_initial_grid_indecies(
            ds_position_scalars=ds_shifted, N_grid=dict(x=nx, y=ny)
        )

        ds_grid_idxs_ = ds_grid_idxs.isel(y=0, z=0)

        ds_traj_posn_new = advtraj_gm_utils.estimate_3d_position_from_grid_indecies(
            ds_grid=ds_shifted, i=ds_grid_idxs_.i, j=ds_grid_idxs_.j, k=ds_grid_idxs_.k
        )

        x_ref = ds_shifted.x
        x_est = ds_traj_posn_new.roll(x=-i_shift, roll_coords=False).x_est
        assert np.allclose(x_ref, x_est)
