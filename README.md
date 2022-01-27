# Offline trajectories with advected position scalars

This repository contains the `advtraj` python module for backward- and
forward-integrate trajectories in Eulerian fluid simulations using
position scalars advected with the flow. The utility is based on
https://github.com/ReadingClouds/Trajectories

*NOTE*: currently no mask functions for calculating the starting points for
trajectories (for example inside clouds) are implemented. That will be added
next.

See the [changelog](CHANGELOG.md) for a summary of changes.


## Installation

If you are not going to modify `advtraj` it can be installed directly from
github with `pip`

```bash
$> pip install git+https://github.com/ParaConUK/advtraj
```

Otherwise have a look at the [development notes](DEVELOPING.md) to learn how to
get a local copy and contribute back to the central repository on github

## Usage

You can either call the `integrate_trajectories` function directly by importing
the `advtraj` module or use one of the command-line interfaces (CLI).

### Using the `advtraj` python module directly

```python
import advtraj
import xarray as xr

# load up the position scalars into an xarray.Dataset
# these should be given as `traj_tracer_x`, `traj_tracer_y` and `traj_tracer_z`
# if not using xy-periodic domains, and otherwise as `traj_tracer_xi`,
# `traj_tracer_xr`, `traj_tracer_yi`, `traj_tracer_yr` and `traj_tracer_z`
ds_position_scalars = xr.open_dataset(...)

# define the starting points for the trajectories
ds_starting_points = xr.Dataset()
ds_starting_points["x"] = 0.0
ds_starting_points["y"] = 0.0
ds_starting_points["z"] = 100.0
ds_starting_points["time"] = ds_position_scalars.time.isel(time=4)

ds_traj = advtraj.integrate_trajectories(
    ds_position_scalars=ds_position_scalars,
    ds_starting_points=ds_starting_points,
    xy_periodic=True
)
```

### Using the command-line interface

For now a only a cli for the [UCLA-LES](https://github.com/uclales/uclales)
model has been implemented, but support for MONC will be added soon.

#### UCLA-LES

The UCLA-LES cli just needs to be pointed to source-directory what the model
output and file-prefix

```bash
$> python -m advtraj.cli.uclales --help
usage: uclales.py [-h] [--debug] [--output OUTPUT] data_path file_prefix

positional arguments:
  data_path
  file_prefix

optional arguments:
  -h, --help       show this help message and exit
  --debug
  --output OUTPUT
```

E.g.

```bash
$> python -m advtraj.cli.uclales ~/Desktop/exp_data rico
backward: 100%|███████████████████████████████████████████| 15/15 [00:21<00:00,  1.60it/s]
forward: 100%|████████████████████████████████████████████| 15/15 [00:27<00:00,  1.79s/it]
Trajectories saved to rico.trajectories.nc
```
