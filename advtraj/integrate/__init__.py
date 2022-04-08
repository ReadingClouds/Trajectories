"""
Main routines for integration
"""
from advtraj.integrate.backward import backward as integrate_backward
from advtraj.integrate.forward import forward as integrate_forward

POSITION_COORD_NAMES = ["x", "y", "z"]


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

    if 'time' not in ds.coords:
        raise Exception(
            "The starting position dataset is missing the time cooord."
        )

    coords = [ c for c in ds[POSITION_COORD_NAMES[0]].coords
                    if 'time' not in c]

    if len(coords) == 0:
        raise Exception(
            "The starting position dataset is missing the trajectory coord."
        )

def integrate_trajectories(ds_position_scalars,
                           ds_starting_points,
                           times="position_scalars",
                           xy_periodic=True,
                           interp_order = 1
):
    """
    Integrate trajectories forwards and back.

    Using "position scalars" `ds_position_scalars` integrate trajectories from
    starting points in `ds_starting_points` to times as in `times`
    """
    _validate_position_scalars(ds=ds_position_scalars, xy_periodic=xy_periodic)
    _validate_starting_points(ds=ds_starting_points)

    if times == "position_scalars":
        da_times = ds_position_scalars.time
    else:
        raise NotImplementedError(times)

    ref_time = ds_starting_points.time
    ds_starting_points = ds_starting_points.assign_coords(
        {'ref_time':ref_time})
    
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
    )

    return ds_traj
