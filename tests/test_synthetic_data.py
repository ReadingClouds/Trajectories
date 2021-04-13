import xarray as xr
import numpy as np


from advtraj.interpolate import integrate_trajectories
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
    ds_advected = _advect_fields(
        ds_scalars=ds_initial, dx_grid=dx, dy_grid=dy, u=u, v=v, dt=dt
    )

    for v in ds_initial.data_vars:
        if not v.startswith("traj_tracer_"):
            continue
        assert np.allclose(ds_initial[v], ds_advected[v])


def test_trajectory_integration_diagonal_advection():
    Lx = Ly = 0.5e3  # [m]
    Lz = 1.0e3  # [m]
    dx = dy = dz = 25.0  # [m]
    dt = 120.0  # [s]
    t_max = 600.0  # [s]
    u, v = 4.0, -3.0

    ds_initial = create_initial_dataset(dL=(dx, dy, dz), L=(Lx, Ly, Lz))

    dt = 5 * 60.0
    n_timesteps = int(t_max / dt)
    datasets_timesteps = [ds_initial]
    for n in range(n_timesteps):
        ds_new = _advect_fields(
            datasets_timesteps[-1], dx_grid=dx, dy_grid=dy, u=u, v=v, dt=dt
        )
        datasets_timesteps.append(ds_new)

    ds = xr.concat(datasets_timesteps, dim="time")
    # for now we just pick out the position scalars
    position_scalars = [f"traj_tracer_{s}" for s in ["xr", "xi", "yr", "yi", "zr"]]
    ds_position_scalars = ds[position_scalars]

    # make up some starting points for the trajectories, making three trajectories for now
    ds_starting_points = xr.Dataset(coords=dict(trajectory_number=[0, 1, 2]))
    ds_starting_points["x"] = (
        ("trajectory_number"),
        [100.0, 200.0, 50.0],
        dict(units="m"),
    )
    ds_starting_points["y"] = (
        ("trajectory_number"),
        [100.0, 100.0, 100.0],
        dict(units="m"),
    )
    ds_starting_points["z"] = (
        ("trajectory_number"),
        [Lz / 2.0, Lz / 2.0, Lz / 2.0],
        dict(units="m"),
    )
    t0 = ds_initial.time.values
    ds_starting_points["time"] = ("trajectory_number"), [t0, t0, t0]

    integrate_trajectories(
        ds_position_scalars=ds_position_scalars, ds_starting_points=ds_starting_points
    )
