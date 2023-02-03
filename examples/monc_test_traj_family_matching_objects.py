# -*- coding: utf-8 -*-
"""
monc_test_traj_family_matching_objects.py
Script to compute matching objects and plot family animations.

"""
import glob
import pickle
import time

# import matplotlib.pyplot as plt
# import networkx as nx
import xarray as xr
from cohobj.object_tools import get_bounding_boxes
from load_data import load_data

from advtraj.family.traj_family import (  # subgraph_overlap_thresh,
    analyse_traj_family,
    draw_object_graph,
    find_family_matching_objects,
    graph_matching_objects,
    matching_obj_to_nodelist,
    print_matching_objects,
    print_matching_objects_graph,
    related_objects,
    start_objects,
    subgraph_ntype,
    summarise_traj_family,
    traj_name_to_data_name,
)
from advtraj.plot.plot_trajectory_animation import plot_family_animation

# import numpy as np


case = "cloud"

minim = "fixed_point_iterator"

interp_order = 5

expt = "std"

root_path = "C:/Users/paclk/OneDrive - University of Reading/"

data_path = root_path + "ug_project_data/Data/"
odir = root_path + "ug_project_data/Data/trajectories/"

file_prefix = "diagnostics_3d_ts_"
ref_file = "diagnostics_ts_"

files = list(glob.glob(f"{data_path}{file_prefix}*.nc"))

ref_files = list(glob.glob(f"{data_path}{ref_file}*.nc"))

if len(ref_files) == 0:
    ref_dataset = None
else:
    ref_dataset = xr.open_dataset(ref_files[0])

output_path = odir + f"{file_prefix}trajectories_{case}_{interp_order}_{minim}_{expt}"

traj_path_list = list(glob.glob(f"{output_path}_[0-9][0-9].nc"))  # [30:50]

family_times = analyse_traj_family(traj_path_list)
summary = summarise_traj_family(family_times)
summary

ds_list = []
time_min = 1e100
time_max = -1e100

traj_fam = []
for path in traj_path_list:
    print(path)
    ds = xr.open_dataset(path)[["x", "y", "z"]]
    print(ds.time.values)
    time_min = min(time_min, ds.time.values.min())
    time_max = max(time_max, ds.time.values.max())

    data_file = traj_name_to_data_name(path, append=False)
    print(data_file)
    ds_data = xr.open_dataset(data_file)

    if case == "w":

        w_max = ds_data.w.sel(time=ds.ref_time).max()

        mask = ds_data.w > 0.9 * w_max
        mask.name = "obj_mask"

    elif case == "cloud":

        # we'll use as starting points for the trajectories all points where
        # the cloud liquid water mass is > 1E-5.
        thresh = 1e-5

        mask = ds_data.q_cloud_liquid_mass > thresh
        mask.name = "obj_mask"

    dsm = xr.merge((ds, mask))
    # dsm = xr.merge((ds, ds_data, mask))

    ds_object_bounds = get_bounding_boxes(dsm, use_mask=True)

    ds_list.append((dsm, ds_object_bounds))

    traj_fam.append(dsm)

timestep = ds.attrs["trajectory timestep"]

ntimes = int(round(time_max - time_min) / timestep) + 1

if case == "w":
    ds_field = load_data(
        files=files, ref_dataset=ref_dataset, traj=False, fields_to_keep=["w"]
    )

    mask_field = ds_field.w > 0.9 * w_max
    mask_field.name = "obj_mask"

elif case == "cloud":
    ds_field = load_data(
        files=files,
        ref_dataset=ref_dataset,
        traj=False,
        fields_to_keep=["q_cloud_liquid_mass"],
    )

    # we'll use as starting points for the trajectories all points where
    # the cloud liquid water mass is > 1E-5.
    thresh = 1e-5

    mask_field = ds_field.q_cloud_liquid_mass > thresh
    mask_field.name = "obj_mask"


time1 = time.perf_counter()

obj_file = "_objects_ref"
pkl_file = output_path + obj_file + ".pkl"

find_mol = False

if find_mol:

    mol_fam = find_family_matching_objects(
        ds_list,
        ref_time_only=True,
        # ref_time_only=False,
        # forward=False,
        # adjacent_only=True,
        adjacent_only=False,
        # fast=True,
        fast=False,
    )

    with open(pkl_file, "wb") as fp:

        pickle.dump(mol_fam, fp)

        print("Done writing matching objects to pickle file.")

else:

    with open(pkl_file, "rb") as fp:

        mol_fam = pickle.load(fp)

        print("Done reading matching objects from pickle file.")

time2 = time.perf_counter()
delta_t = time2 - time1


G = graph_matching_objects(mol_fam, include_types=(1, 2))
print_matching_objects_graph(G)

print(start_objects(G))

draw_object_graph(G, ntypes=(1, 2), save_file=output_path + obj_file + "_all.png")


# node_sel = (23400.0, 17)
node_sel = (23160.0, 9)

mol = mol_fam[node_sel[0]]
nodelist = matching_obj_to_nodelist(
    mol, node_sel[1], ref_times_sel=None, overlap_thresh=None
)

nodelist = related_objects(G, node_sel)  # , overlap_thresh = 0.1)

draw_object_graph(
    G,
    nodelist=nodelist,
    highlight_nodes=[node_sel],
    ntypes=(1, 2),
    save_file=output_path + obj_file + ".png",
)

G1 = subgraph_ntype(G, 1)
# , overlap_thresh = 0.1)
nodelist1 = related_objects(G1, node_sel)  # , overlap_thresh = 0.1)

draw_object_graph(
    G1,
    nodelist=nodelist1,
    highlight_nodes=[node_sel],
    ntypes=(1, 2),
    save_file=output_path + obj_file + ".png",
)

print_matching_objects(
    mol, select=[node_sel[1]], ref_times_sel=[node_sel[0]], full=True
)
print(f"Elapsed time = {delta_t}")
# %%
for vp in [(0, 0), (0, 90), (90, 0), (45, 140)]:
    # for vp in [(0, 0)]:
    anim = plot_family_animation(
        ds_list,
        # field_mask = mask_field,
        nodelist1,
        highlight_obj=[node_sel],
        galilean=(-8, -1.5),
        title="Trajectory Family",
        legend=True,
        figsize=(15, 12),
        not_inobj_size=0.5,
        inobj_size=2.0,
        field_size=4.0,
        fps=5,
        anim_name=f"../animations/Family_plot_mask_related{obj_file}_nt1_{vp[0]:03d}_{vp[1]:03d}.gif",
        with_boxes=False,
        plot_mask=True,
        load_ds=True,
        view_point=vp,
        # x_lim=[3000, 6000],
        x_lim=[2000, 8400],
        # y_lim=[4000, 8000],
        # y_lim=[3000, 8000],
        # z_lim = None,
        # colors=['k', 'r','g','b'],
    )
