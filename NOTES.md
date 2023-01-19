# Notes on this branch.

##  Introduction

This branch contains the following, related, subpackages:
- `plot`
- `classify`
- `family`

The first of these provides some matplotlib-based (i.e. slow) functionality for 3D plotting of trajectories.
We shall use it to illustrate the other two.

## The plot subpackage

The module `plot.plot_trajectory_animation.py` contains two main functions.
The function `plot_traj_animation` is used to plot trajectories from one reference time.
At its most basic, it animates all of the objects in the input xarray (identified by `object_label`).
Here is an example from a MONC simulation of BOMEX, showing each object as a different colour:
![a MONC simulation of BOMEX](animations/Traj_plot_all.gif)
Optionally, a galilean transform can be applied to move the plot with the mean wind:
![move the plot with the mean wind](animations/Traj_plot_all_gal.gif)

Individual objects can be selected, and just a sub-domain plotted.
If the trajectory dataset contains the variable `obj_mask` then points with `obj_mask==True` can be distinguished
with different sized dots.
For example, here the mask was set up using `q_cloud_liquid_mass > 1E-5`.
Here is object 9:
![object 9](animations/Traj_plot_mask.gif)

Sometimes it is useful to compare the Lagrangian view, trajectories with origin at a given time, in cloud, say)
with the Eulerian view, grid points in cloud at different times.
This can be done by supplying a `field_mask` derived from the original gridded data.
For example, here is the same figure as above with the Eulerian cloud field as black dots.
![Eulerian field](animations/Traj_plot_field.gif).
