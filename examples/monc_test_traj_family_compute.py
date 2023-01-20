"""
monc_test_traj_family_compute.py
Script to compute trajectory family from MONC data.

"""

import os
from pathlib import Path

import xarray as xr
from load_data import load_data
from set_options import set_options

from advtraj.family.traj_family import traj_family
from advtraj.utils.cli import optional_debugging


def main(
    data_path,
    file_prefix,
    ref_file,
    output_path_base,
    case="w",
    start_ref_time=None,
    end_ref_time=None,
    steps_backward=None,
    steps_forward=None,
    interp_order=5,
    forward_solver="fixed_point_iterator",
    options=None,
):

    files = list(Path(data_path).glob(f"{file_prefix}*.nc"))

    ref_files = list(Path(data_path).glob(f"{ref_file}*.nc"))

    if len(ref_files) == 0:
        ref_dataset = None
    else:
        ref_dataset = xr.open_dataset(ref_files[0])

    if case == "w":

        ds = load_data(
            files=files, ref_dataset=ref_dataset, traj=True, fields_to_keep=["w"]
        )

        # we'll use as starting points for the trajectories all points where
        # the vertical velocity is 80% of the maximum value
        w_max = ds.w.max()

        mask = ds.w > 0.9 * w_max
        mask.name = "obj_mask"

    elif case == "cloud":

        ds = load_data(
            files=files,
            ref_dataset=ref_dataset,
            traj=True,
            fields_to_keep=["q_cloud_liquid_mass"],
        )

        # we'll use as starting points for the trajectories all points where
        # the cloud liquid water mass is > 1E-5.
        thresh = 1e-5

        mask = ds.q_cloud_liquid_mass > thresh
        mask.name = "obj_mask"

    traj_path_list = traj_family(
        ds,
        mask,
        output_path_base,
        start_ref_time=start_ref_time,
        end_ref_time=end_ref_time,
        steps_backward=steps_backward,
        steps_forward=steps_forward,
        interp_order=interp_order,
        forward_solver=forward_solver,
        options=options,
    )
    if ref_dataset is not None:
        ref_dataset.close()
    print(traj_path_list)


if __name__ == "__main__":

    case = "cloud"

    start_ref_time = 21120.0
    end_ref_time = 25140.0
    steps_backward = 35
    steps_forward = 35

    minim = "fixed_point_iterator"

    interp_order_list = [5]

    exptlist = ["std"]

    root_path = "C:/Users/paclk/OneDrive - University of Reading/"

    data_path = root_path + "ug_project_data/Data/"
    odir = root_path + "ug_project_data/Data/trajectories/"

    if not os.path.exists(odir):
        os.makedirs(odir)

    file_prefix = "diagnostics_3d_ts_"
    ref_file = "diagnostics_ts_"

    for expt in exptlist:

        for interp_order in interp_order_list:

            output_path = (
                odir + f"{file_prefix}trajectories_{case}_{interp_order}_{minim}_{expt}"
            )

            options = set_options(minim, expt)  # , save_iterations_path)

            with optional_debugging(False):
                main(
                    data_path,
                    file_prefix,
                    ref_file,
                    output_path,
                    case=case,
                    start_ref_time=start_ref_time,
                    end_ref_time=end_ref_time,
                    steps_backward=steps_backward,
                    steps_forward=steps_forward,
                    interp_order=interp_order,
                    forward_solver=minim,
                    options=options,
                )
