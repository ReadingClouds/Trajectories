import numpy as np


from utils import create_initial_dataset


def _advect_fields(ds_scalars, dx_grid, dy_grid, u, v, dt):
    """
    Perform poor-mans advection by rolling all variables in `ds_scalars` by a
    finite number of grid positions, requiring that the time-increment `dt`
    causes exactly grid-aligned shifts using grid resolution `dx_grid`,
    `dy_grid` with x- and y-velocity (`u`, `v`)
    """

    def _calc_shift(dx_, vel):
        n_shift = vel * dt / dx_
        if not np.isclose(n_shift, int(n_shift)):
            raise Exception(
                f"Advecting by time {dt} with velocity {vel} along grid with"
                f" spacing {dx_} does not result in a grid-aligned new position"
            )
        return int(n_shift)

    i_shift = _calc_shift(dx_grid, u)
    j_shift = _calc_shift(dy_grid, v)
    ds_new = ds_scalars.roll(dict(x=i_shift, y=j_shift), roll_coords=False)
    ds_new.assign_coords(dict(time=ds_scalars.time + np.timedelta64(int(dt), "s")))
    return ds_new


def test_diagonal_advection():
    """
    Test that the the poor-man's advection implemented above actually works by
    advecting around the whole domain. Domain is twice as long in y-direction
    as in x-direction and velocity is twice as fast.
    """
    Lx = 0.5e3  # [m]
    Ly = 1.0e3
    Lz = 1.0e3  # [m]
    dx = dy = dz = 25.0  # [m]
    u, v = 4.0, 8.0

    dt = Lx / u  # [s]

    ds_initial = create_initial_dataset(dL=(dx, dy, dz), L=(Lx, Ly, Lz))
    ds_advected = _advect_fields(ds_scalars=ds_initial, dx_grid=dx, dy_grid=dy, u=u, v=v, dt=dt)

    for v in ds_initial.data_vars:
        if not v.startswith("traj_tracer_"):
            continue
        assert np.allclose(ds_initial[v], ds_advected[v])
