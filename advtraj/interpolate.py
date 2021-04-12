"""
Main routines for integration
"""

POSITION_COORD_NAMES = ["x", "y", "z", "time"]


def _validate_position_scalars(ds, xy_periodic=False):
    """
    Ensure that the required position scalars are available in the provided
    dataset (depending on whether we're using periodic boundaries in the
    xy-direction)
    """
    required_coords = POSITION_COORD_NAMES
    required_vars = ["tracer_traj_xr", "tracer_traj_yr", "tracer_traj_zr"]

    if xy_periodic:
        required_vars += ["tracer_traj_xi", "tracer_traj_yi"]

    missing_vars = filter(lambda c: c not in ds, required_coords)

    if any(missing_vars):
        raise Exception(
            "The position scalars dataset is missing the following requried"
            f" variables: {', '.join(missing_vars)}"
        )

    for v in required_vars:
        missing_dims = filter(lambda c: c not in ds[v].coords, required_coords)
        if any(missing_dims):
            raise Exception(
                f"The position variable `{v}` is missing the coordinates"
                f"{', '.join(missing_dims)}"
            )


def _validate_starting_points(ds):
    """
    Ensure that starting positions dataset contains the necessary variables for
    describe the starting position
    """
    required_vars = POSITION_COORD_NAMES

    missing_vars = filter(lambda c: c not in ds.data_vars, required_vars)

    if any(missing_vars):
        raise Exception(
            "The starting position dataset is missing the following variables:"
            f" {', '.join(missing_vars)}"
        )


def _integrate_single_trajectory(ds_starting_position, ds_position_scalars):
    pass


def integrate_trajectories(
    ds_position_scalars, ds_starting_points, times="position_scalars", xy_periodic=True
):
    """
    Using "position scalars" `ds_position_scalars` integrate trajectories from
    starting points in `ds_starting_points` to times as in `times`
    """
    _validate_position_scalars(ds=ds_position_scalars, xy_periodic=xy_periodic)
    _validate_starting_points(ds=ds_starting_points)

    # all coordinates that are defined for the starting position variables will
    # be treated as if they represent different trajectories
    dims = ds_starting_points[POSITION_COORD_NAMES[0]].dims

    ds_traj = ds_starting_points.stack(trajectory=dims).apply(
        _integrate_single_trajectory, ds_position_scalars=ds_position_scalars
    )

    return ds_traj
