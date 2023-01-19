"""
Main routines for integration
"""
from .backward import backward as integrate_backward
from .forward import forward as integrate_forward

POSITION_VAR_NAMES = ["x", "y", "z"]
EXPECTED_STARTING_POSITION_COORDS = ["time", "trajectory_number"]


def _validate_position_scalars(ds, xy_periodic=False):
    """
    Ensure that the required position scalars are available in the provided
    dataset (depending on whether we're using periodic boundaries in the
    xy-direction)
    """
    required_coords = POSITION_VAR_NAMES
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
    required_vars = POSITION_VAR_NAMES

    missing_vars = list(filter(lambda c: c not in ds.data_vars, required_vars))

    if missing_vars:
        raise Exception(
            "The starting position dataset is missing the following variables:"
            f" {', '.join(missing_vars)}"
        )

    if "time" not in ds.coords:
        raise Exception("The starting position dataset is missing the time cooord.")

    dims = set(list(ds.dims))
    unexpected_coords = dims.difference(EXPECTED_STARTING_POSITION_COORDS)
    if len(unexpected_coords) > 0:
        raise Exception(
            "The starting position should may only contain dimensions"
            " called `time` and/or `trajectory_number, but the starting points"
            " contain the following unexpected coords:"
            f" {', '.join(unexpected_coords)}"
        )


def _promote_starting_position_vars_to_coords(ds):
    """
    Promote any variables in the starting position dataset to coords for
    variables which have
    """
    for c in EXPECTED_STARTING_POSITION_COORDS:
        if c in ds.data_vars:
            ds = ds.set_coords({c: ds[c]})
    return ds


def integrate_trajectories(
    ds_position_scalars,
    ds_starting_points,
    steps_backward=None,
    steps_forward=None,
    xy_periodic=True,
    interp_order=5,
    forward_solver="fixed_point_iterator",
    point_iter_kwargs=None,
    minim_kwargs=None,
):
    """
    Integrate trajectories forwards and back.

    Using "position scalars" `ds_position_scalars` integrate trajectories from
    starting points in `ds_starting_points` to times as in `times`
    """
    ds_starting_points = _promote_starting_position_vars_to_coords(
        ds=ds_starting_points
    )
    _validate_position_scalars(ds=ds_position_scalars, xy_periodic=xy_periodic)
    _validate_starting_points(ds=ds_starting_points)

    ref_time = ds_starting_points.time
    ds_starting_points = ds_starting_points.assign_coords({"ref_time": ref_time})

    input_times = list(ds_position_scalars["time"].values)
    if ref_time not in input_times:
        raise ValueError(f"Reference time {ref_time} is not in dataset.")

    ref_index = input_times.index(ref_time)

    da_times = ds_position_scalars.time

    # Select start and end time of trajectories.
    if steps_backward is None:
        start_index = 0
    else:
        if steps_backward < 0:
            raise ValueError(
                f"steps_backward ({steps_backward}) must be positive integer."
            )
        start_index = max(0, ref_index - max(1, steps_backward))

    if steps_forward is None:
        end_index = len(input_times)
    else:
        if steps_forward < 0:
            raise ValueError(
                f"steps_forward ({steps_backward}) must be positive integer."
            )
        end_index = min(len(input_times), ref_index + max(0, int(steps_forward)) + 1)

    da_times = ds_position_scalars.time.isel(time=slice(start_index, end_index))

    da_times_backward = da_times.sel(time=slice(None, ref_time))
    da_times_forward = da_times.sel(time=slice(ref_time, None)).isel(
        time=slice(1, None)
    )

    # all coordinates that are defined for the starting position variables will
    # be treated as if they represent different trajectories

    ds_traj_backward = integrate_backward(
        ds_position_scalars=ds_position_scalars,
        ds_starting_point=ds_starting_points,
        da_times=da_times_backward,
        interp_order=interp_order,
    )

    ds_traj = integrate_forward(
        ds_position_scalars=ds_position_scalars,
        ds_back_trajectory=ds_traj_backward,
        da_times=da_times_forward,
        interp_order=interp_order,
        solver=forward_solver,
        point_iter_kwargs=point_iter_kwargs,
        minim_kwargs=minim_kwargs,
    )

    attrs = {
        "interp_order": interp_order,
        "solver": forward_solver,
    }

    for c in "xyz":
        attrs[f"d{c}"] = ds_position_scalars[c].attrs[f"d{c}"]
        attrs[f"L{c}"] = ds_position_scalars[c].attrs[f"L{c}"]

    attrs["trajectory timestep"] = (
        ds_traj.time.values[-1] - ds_traj.time.values[0]
    ) / (ds_traj.time.size - 1)

    if "fixed_point_iterator" in forward_solver and point_iter_kwargs is not None:
        attrs["maxiter"] = point_iter_kwargs["maxiter"]
        attrs["tol"] = point_iter_kwargs["tol"]
    elif minim_kwargs is not None:
        attrs["maxiter"] = minim_kwargs["minimize_options"]["maxiter"]
        attrs["max_outer_loops"] = minim_kwargs["max_outer_loops"]

    if "hybrid_fixed_point_iterator" in forward_solver and minim_kwargs is not None:
        attrs["minimize_maxiter"] = minim_kwargs["minimize_options"]["maxiter"]
        attrs["tol"] = minim_kwargs["tol"]

    ds_traj.attrs = attrs

    return ds_traj
