"""
Main routines for integration
"""
from .backward import backward as integrate_backward
from .forward import forward as integrate_forward

POSITION_COORD_NAMES = ["x", "y", "z", "time"]


def _validate_position_scalars(ds, xy_periodic=False):
    """
    Ensure that the required position scalars are available in the provided
    dataset (depending on whether we're using periodic boundaries in the
    xy-direction)
    """
    required_coords = POSITION_COORD_NAMES
    required_vars = ["traj_tracer_xr", "traj_tracer_yr", "traj_tracer_zr"]

    if xy_periodic:
        required_vars += ["traj_tracer_xi", "traj_tracer_yi"]

    missing_vars = list(filter(lambda c: c not in ds, required_vars))

    if missing_vars:
        raise Exception(
            "The position scalars dataset is missing the following requried"
            f" variables: {', '.join(missing_vars)}"
        )

    for v in required_vars:
        missing_dims = list(filter(lambda c: c not in ds[v].coords, required_coords))
        if missing_dims:
            raise Exception(
                f"The position variable `{v}` is missing the coordinates"
                f" {', '.join(missing_dims)}"
            )


def _validate_starting_points(ds):
    """
    Ensure that starting positions dataset contains the necessary variables for
    describe the starting position
    """
    required_vars = POSITION_COORD_NAMES

    missing_vars = list(filter(lambda c: c not in ds.data_vars, required_vars))

    if missing_vars:
        raise Exception(
            "The starting position dataset is missing the following variables:"
            f" {', '.join(missing_vars)}"
        )


def _integrate_single_trajectory(ds_starting_point, ds_position_scalars, da_times):
    """
    Integrate a single trajectory both forward and backward from
    `ds_starting_point` usnig `ds_position_scalars` and times in `da_times`
    """
    da_times_backward = da_times.sel(time=slice(None, ds_starting_point.time))
    da_times_forward = da_times.sel(time=slice(ds_starting_point.time, None)).isel(
        time=slice(1, None)
    )

    ds_traj_backward = integrate_backward(
        ds_position_scalars=ds_position_scalars,
        ds_starting_point=ds_starting_point,
        da_times=da_times_backward,
    )

    interp_order = 1

    ds_traj = integrate_forward(
        ds_position_scalars=ds_position_scalars,
        ds_back_trajectory=ds_traj_backward,
        da_times=da_times_forward,
        interp_order=interp_order,
    )

    return ds_traj


def integrate_trajectories(
    ds_position_scalars, ds_starting_points, times="position_scalars", xy_periodic=True
):
    """
    Using "position scalars" `ds_position_scalars` integrate trajectories from
    starting points in `ds_starting_points` to times as in `times`
    """
    _validate_position_scalars(ds=ds_position_scalars, xy_periodic=xy_periodic)
    _validate_starting_points(ds=ds_starting_points)

    if times == "position_scalars":
        da_times = ds_position_scalars.time
    else:
        raise NotImplementedError(times)

    # all coordinates that are defined for the starting position variables will
    # be treated as if they represent different trajectories
    dims = ds_starting_points[POSITION_COORD_NAMES[0]].dims

    if len(dims) > 0:
        ds_traj = (
            ds_starting_points.stack(trajectory=dims)
            .groupby("trajectory")
            .apply(
                _integrate_single_trajectory,
                ds_position_scalars=ds_position_scalars,
                da_times=da_times,
            )
            .unstack("trajectory")
        )
    else:
        ds_traj = _integrate_single_trajectory(
            ds_position_scalars=ds_position_scalars,
            ds_starting_point=ds_starting_points,
            da_times=da_times,
        )

    return ds_traj
