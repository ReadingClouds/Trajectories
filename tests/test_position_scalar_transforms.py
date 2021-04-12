import numpy as np
from utils import create_uniform_grid

import advtraj.utils as advtraj_utils


def test_position_scalar_transforms_are_symmetric():
    """
    Test that transforms both to and from the "position scalars" are symmetric
    """
    Lx = Ly = 0.5e3  # [m]
    Lz = 1.0e3  # [m]
    dx = dy = dz = 25.0  # [m]

    ds_grid = create_uniform_grid(dL=(dx, dy, dz), L=(Lx, Ly, Lz))

    i = ds_grid.x / dx
    j = ds_grid.y / dy
    k = ds_grid.z / dz
    nx = ds_grid.x.count()
    ny = ds_grid.y.count()
    nz = ds_grid.z.count()

    for xy_periodic in [True, False]:
        ds_position_scalars = advtraj_utils.grid_locations_to_position_scalars(
            i=i, j=j, k=k, nx=nx, ny=ny, nz=nz, xy_periodic=xy_periodic
        )
        i_est, j_est, k_est = advtraj_utils.estimate_initial_grid_locations(
            ds_position_scalars=ds_position_scalars, xy_periodic=xy_periodic, nx=nx, ny=ny
        )

        assert np.allclose(i.values, i_est.values)
        assert np.allclose(j.values, j_est.values)
        assert np.allclose(k.values, k_est.values)
