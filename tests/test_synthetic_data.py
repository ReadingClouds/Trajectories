import numpy as np
import pytest
import xarray as xr
from utils import create_initial_dataset, init_position_scalars

from advtraj.integrate import integrate_trajectories


def _advect_fields(ds_scalars, u, v, dt):
    """
    Perform poor-mans advection by rolling all variables in `ds_scalars` by a
    finite number of grid positions, requiring that the time-increment `dt`
    causes exactly grid-aligned shifts using grid resolution `dx_grid`,
    `dy_grid` with x- and y-velocity (`u`, `v`).
    """
    dx_grid = ds_scalars.x.dx
    dy_grid = ds_scalars.y.dy

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

    ds_new = ds_new.assign_coords(time=ds_scalars.time + np.timedelta64(int(dt), "s"))
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
    ds_advected = _advect_fields(ds_scalars=ds_initial, u=u, v=v, dt=dt)

    for v in ds_initial.data_vars:
        if not v.startswith("traj_tracer_"):
            continue
        assert np.allclose(ds_initial[v], ds_advected[v])


def _wrap_add(x, y, a, b):
    """add x+y modulu the answer being in range [a...b[
    https://stackoverflow.com/a/51467186/271776
    """
    y %= b - a
    x = x + y
    return x - (b - a) * (x >= b)


testvars = "forward_solver, n_trajectories"
tests = [
    ("fixed_point_iterator", 1),
    ("fixed_point_iterator", 3),
    ("hybrid_fixed_point_iterator", 1),
    ("hybrid_fixed_point_iterator", 3),
    ("BFGS", 3),
]


@pytest.mark.parametrize(testvars, tests)
def test_stationary_trajectory(forward_solver, n_trajectories):
    L = 5.0e3  # [m]
    dx = 25.0  # [m]
    dt = 120.0  # [s]
    u = 0.0  # [m/s]
    v = 0.0  # [m/s]
    _trajectory_integration(
        forward_solver=forward_solver,
        n_trajectories=n_trajectories,
        u=u,
        v=v,
        dt=dt,
        dx=dx,
        L=L,
    )


@pytest.mark.parametrize(testvars, tests)
def test_linear_trajectory_x_direction(forward_solver, n_trajectories):
    L = 2.0e2  # [m]
    dx = 25.0  # [m]
    dt = 25.0  # [s]
    u = 1.0  # [m/s]
    v = 0.0  # [m/s]
    t_max = L / u * 1.5
    _trajectory_integration(
        forward_solver=forward_solver,
        n_trajectories=n_trajectories,
        u=u,
        v=v,
        dt=dt,
        dx=dx,
        L=L,
        t_max=t_max,
    )


@pytest.mark.parametrize(testvars, tests)
def test_linear_trajectory_y_direction(forward_solver, n_trajectories):
    L = 2.0e2  # [m]
    dx = 25.0  # [m]
    dt = 25.0  # [s]
    u = 0.0  # [m/s]
    v = -1.0  # [m/s]
    t_max = L / abs(v) * 1.5
    _trajectory_integration(
        forward_solver=forward_solver,
        n_trajectories=n_trajectories,
        u=u,
        v=v,
        dt=dt,
        dx=dx,
        L=L,
        t_max=t_max,
    )


@pytest.mark.parametrize(testvars, tests)
def test_linear_trajectory_diagonal(forward_solver, n_trajectories):
    L = 2.0e2  # [m]
    dx = 25.0  # [m]
    dt = 25.0  # [s]
    u = 2.0  # [m/s]
    v = -2.0  # [m/s]
    t_max = L / u * 1.5
    _trajectory_integration(
        forward_solver=forward_solver,
        n_trajectories=n_trajectories,
        u=u,
        v=v,
        dt=dt,
        dx=dx,
        L=L,
        t_max=t_max,
    )


def _make_starting_points(n_trajectories, t0, dx, dy, dz, Lx, Ly, Lz):
    # make up some starting points for the trajectories, making three trajectories for now
    ds_starting_points = xr.Dataset()
    ds_starting_points["x"] = xr.DataArray(
        np.repeat(Lx / 2.0 + 0.25 * dx, n_trajectories),
        dims=["trajectory_number"],
        coords={"trajectory_number": np.arange(n_trajectories)},
    )
    ds_starting_points.x.attrs["units"] = "m"
    ds_starting_points["y"] = xr.DataArray(
        np.repeat(Ly / 2.0 + 0.25 * dy, n_trajectories),
        dims=["trajectory_number"],
        coords={"trajectory_number": np.arange(n_trajectories)},
    )
    ds_starting_points.y.attrs["units"] = "m"
    ds_starting_points["z"] = xr.DataArray(
        np.repeat(Lz / 2.0, n_trajectories),
        dims=["trajectory_number"],
        coords={"trajectory_number": np.arange(n_trajectories)},
    )
    ds_starting_points.z.attrs["units"] = "m"

    ds_starting_points = ds_starting_points.assign_coords(time=t0)

    ds_starting_points["x_err"] = xr.zeros_like(ds_starting_points["x"])
    ds_starting_points["y_err"] = xr.zeros_like(ds_starting_points["y"])
    ds_starting_points["z_err"] = xr.zeros_like(ds_starting_points["z"])

    return ds_starting_points


def _trajectory_integration(
    forward_solver,
    n_trajectories,
    u=4.0,
    v=-3.0,
    dt=5 * 60.0,
    dx=25.0,
    L=5.0e3,
    t_max=5 * 60,
):
    Lx = Ly = L  # [m]
    dy = dz = dx  # [m]

    Lz = 1.0e3  # [m]

    ds_initial = create_initial_dataset(dL=(dx, dy, dz), L=(Lx, Ly, Lz))

    n_timesteps = int(t_max / dt) + 1
    datasets_timesteps = []
    ds_new = None
    for n in range(n_timesteps):
        # ds_prev = datasets_timesteps[-1]
        ds_prev_reset = init_position_scalars(ds=ds_initial.copy())
        if datasets_timesteps:
            ds_prev_reset = ds_prev_reset.assign_coords(time=ds_new["time"])

        # the position scalars are reset every time the 3D output is generated
        ds_new = _advect_fields(ds_prev_reset, u=u, v=v, dt=dt)
        datasets_timesteps.append(ds_new)

    ds = xr.concat(datasets_timesteps, dim="time")
    # for now we just pick out the position scalars
    position_scalars = [f"traj_tracer_{s}" for s in ["xr", "xi", "yr", "yi", "zr"]]
    ds_position_scalars = ds[position_scalars]

    t0 = ds.time.isel(time=int(ds.time.count()) // 2).values
    ds_starting_points = _make_starting_points(
        n_trajectories=n_trajectories, t0=t0, dx=dx, dy=dy, dz=dz, Lx=Lx, Ly=Ly, Lz=Lz
    )

    # work out how far the points should have moved
    def _get_dt64_total_seconds(arr):
        return np.array(
            [dt64.astype("timedelta64[s]").item().total_seconds() for dt64 in arr]
        )

    # Use single trajectory to get 'truth' data.

    ds_starting_points_single = ds_starting_points.isel(
        trajectory_number=0,
        drop=False,
    )

    dt_steps = _get_dt64_total_seconds((ds.time - t0).values)
    x_truth_single = _wrap_add(
        ds_starting_points_single.x.item(), u * dt_steps, 0.0, Lx
    )
    y_truth_single = _wrap_add(
        ds_starting_points_single.y.item(), v * dt_steps, 0.0, Ly
    )

    assert len(x_truth_single) == n_timesteps

    # Test for multiple trajectories.
    if n_trajectories > 1:
        x_truth = np.repeat(x_truth_single[..., np.newaxis], n_trajectories, axis=-1)
        y_truth = np.repeat(y_truth_single[..., np.newaxis], n_trajectories, axis=-1)
    else:
        x_truth = x_truth_single
        y_truth = y_truth_single

    ds_traj = integrate_trajectories(
        ds_position_scalars=ds_position_scalars,
        ds_starting_points=ds_starting_points,
        forward_solver=forward_solver,
    )

    x_est = ds_traj.x.values
    y_est = ds_traj.y.values
    assert len(x_est) == n_timesteps

    # # the estimates for the grid-position aren't perfect, allow for a small
    # # error for now
    atol = 0.1

    np.testing.assert_allclose(x_truth, x_est.squeeze(), atol=atol)
    np.testing.assert_allclose(y_truth, y_est.squeeze(), atol=atol)


def _test_trajectory_integration_diagonal_advection(forward_solver):
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
        ds_new = _advect_fields(datasets_timesteps[-1], u=u, v=v, dt=dt)
        datasets_timesteps.append(ds_new)

    ds = xr.concat(datasets_timesteps, dim="time")
    # for now we just pick out the position scalars
    position_scalars = [f"traj_tracer_{s}" for s in ["xr", "xi", "yr", "yi", "zr"]]
    ds_position_scalars = ds[position_scalars]

    ds_position_scalars.attrs["Lx"] = Lx
    ds_position_scalars.attrs["Ly"] = Ly
    ds_position_scalars.attrs["Lz"] = Lz

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
    t0 = ds.time.isel(time=-1).values
    ds_starting_points["time"] = ("trajectory_number"), [t0, t0, t0]

    ds_starting_points = ds_starting_points.isel(trajectory_number=0)

    integrate_trajectories(
        ds_position_scalars=ds_position_scalars,
        ds_starting_points=ds_starting_points,
        forward_solver=forward_solver,
    )

    raise Exception
