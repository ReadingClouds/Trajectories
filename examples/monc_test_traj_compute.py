"""
monc_test_traj_compute.py
Script for producing trajectories from MONC LES model output

"""

import os
import time
from pathlib import Path

import xarray as xr
from cohobj.object_tools import get_object_labels, unsplit_objects
from load_data import load_data
from set_options import set_options

from advtraj.integrate import integrate_trajectories
from advtraj.utils.cli import optional_debugging
from advtraj.utils.point_selection import mask_to_positions


def main(
    data_path,
    file_prefix,
    ref_file,
    odir,
    case="w",
    tref=None,
    steps_backward=None,
    steps_forward=None,
    interp_order=5,
    minim="fixed_point_iterator",
    expt="std",
    selector="*",
    options=None,
):

    files = list(Path(data_path).glob(f"{file_prefix}{selector}.nc"))

    ref_files = list(Path(data_path).glob(f"{ref_file}{selector}.nc"))

    output_path = (
        odir + f"{file_prefix}trajectories_{case}_{interp_order}_{minim}_{expt}.nc"
    )

    if len(ref_files) == 0:
        ref_dataset = None
    else:
        ref_dataset = xr.open_dataset(ref_files[0])

    if case == "w":

        ds = load_data(
            files=files, ref_dataset=ref_dataset, traj=False, fields_to_keep=["w"]
        )

        if tref is None:
            tref = ds.time.values[int(ds.time.count() // 2)]

        ds_subset = ds.sel(time=tref, z=slice(300, None))

        # we'll use as starting points for the trajectories all points where
        # the vertical velocity is 80% of the maximum value
        w_max = ds_subset.w.max()

        mask = ds_subset.w > 0.9 * w_max
        mask.name = "obj_mask"

        get_obj = True

    elif case == "cloud":

        ds = load_data(
            files=files,
            ref_dataset=ref_dataset,
            traj=False,
            fields_to_keep=["q_cloud_liquid_mass"],
        )

        if tref is None:
            tref = ds.time.values[int(ds.time.count() // 2)]

        ds_subset = ds.sel(time=tref, z=slice(300, None))

        # we'll use as starting points for the trajectories all points where
        # the cloud liquid water mass is > 1E-5.
        thresh = 1e-5

        mask = ds_subset.q_cloud_liquid_mass > thresh
        mask.name = "obj_mask"

        get_obj = True

    elif case == "all":

        ds = load_data(
            files=files, ref_dataset=ref_dataset, traj=False, fields_to_keep=["w"]
        )

        # Select middle time in files as reference time
        ds_subset = ds.isel(time=int(ds.time.count()) // 2)

        # Select height >= level 1.
        ds_subset = ds_subset.isel(z=slice(1, None))

        # we'll use as starting points for the trajectories all points where
        # z >= z[1]. w is just a template variable to get the grid etc.

        mask = xr.full_like(
            ds_subset.w,
            True,
            dtype=bool,
        )
        mask.name = "obj_mask"

        get_obj = False

    elif case == "stride":

        stride = 10
        ds = load_data(
            files=files, ref_dataset=ref_dataset, traj=False, fields_to_keep=["w"]
        )

        # Select middle time in files as reference time
        ds_subset = ds.isel(time=int(ds.time.count()) // 2)

        # Select height >= level 1 and every stride points
        ds_subset = ds_subset.isel(
            x=slice(None, None, stride),
            y=slice(None, None, stride),
            z=slice(1, None, stride),
        )

        # we'll use as starting points for the trajectories all points where
        # z >= z[1]. w is just a template variable to get the grid etc.

        mask = xr.full_like(
            ds_subset.w,
            True,
            dtype=bool,
        )
        mask.name = "obj_mask"

        get_obj = False

    print(ds)

    ds_starting_points = mask_to_positions(mask).rename(
        {"pos_number": "trajectory_number"}
    )

    for c in "xyz":
        ds_starting_points[f"{c}_err"] = xr.zeros_like(ds_starting_points[c])

    print(ds_starting_points)

    time1 = time.perf_counter()

    ds = load_data(files=files, ref_dataset=ref_dataset, fields_to_keep=["w", "th"])

    print(ds)

    ds_traj = integrate_trajectories(
        ds_position_scalars=ds,
        ds_starting_points=ds_starting_points,
        steps_backward=steps_backward,
        steps_forward=steps_forward,
        interp_order=interp_order,
        forward_solver=minim,
        point_iter_kwargs=options["pioptions"],
        minim_kwargs=options["minoptions"],
    )

    time2 = time.perf_counter()

    delta_t = time2 - time1

    print(f"Elapsed time = {delta_t}")

    # Add some additional attributes.
    ds_traj.attrs["elapsed_time"] = delta_t
    ds_traj.attrs["source files"] = data_path + file_prefix

    if get_obj:

        Lx = ds.x.Lx
        Ly = ds.y.Ly
        olab = (
            get_object_labels(mask)
            .rename({"pos_number": "trajectory_number"})
            .drop("time")
        )

        ds_traj = ds_traj.assign_coords({"object_label": olab})

        ds_traj = unsplit_objects(ds_traj, Lx, Ly)

    ds_traj.to_netcdf(output_path)
    print(f"Trajectories saved to {output_path}")

    print(ds_traj)


if __name__ == "__main__":

    case = "cloud"

    steps_backward = 30
    steps_forward = 30

    minim = "fixed_point_iterator"

    interp_order_list = [5]

    exptlist = ["std"]

    root_path = "C:/Users/paclk/OneDrive - University of Reading/ug_project_data/Data/"

    data_path = root_path + ""
    odir = root_path + "trajectories/"

    if not os.path.exists(odir):
        os.makedirs(odir)

    file_prefix = "diagnostics_3d_ts_"
    ref_file = "diagnostics_ts_"

    selector = "*"

    for expt in exptlist:

        for interp_order in interp_order_list:

            options = set_options(minim, expt)

            with optional_debugging(False):
                main(
                    data_path,
                    file_prefix,
                    ref_file,
                    odir,
                    case=case,
                    steps_backward=steps_backward,
                    steps_forward=steps_forward,
                    interp_order=interp_order,
                    minim=minim,
                    expt=expt,
                    selector=selector,
                    options=options,
                )
