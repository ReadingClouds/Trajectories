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
Here is an example from
![a MONC simulation of BOMEX](animations/Traj_plot_all.gif).
Optionally, a galilean transform can be applied to
![move the plot with the mean wind](animations/Traj_plot_all_gal.gif).
