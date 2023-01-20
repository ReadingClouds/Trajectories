"""
monc_test_data_to_traj.py
Script to interpolate gridded data to trajectories.

"""

import time
from pathlib import Path

import xarray as xr
from load_data import load_data

from advtraj.utils.cli import optional_debugging
from advtraj.utils.data_to_traj import data_to_traj


def main(
    data_path,
    file_prefix,
    ref_file,
    odir,
    case="w",
    interp_order=5,
    minim="fixed_point_iterator",
    expt="std",
    selector="*",
    options=None,
):

    files = list(Path(data_path).glob(f"{file_prefix}{selector}.nc"))

    ref_files = list(Path(data_path).glob(f"{ref_file}{selector}.nc"))

    if len(ref_files) == 0:
        ref_dataset = None
    else:
        ref_dataset = xr.open_dataset(ref_files[0])

    output_path = (
        odir + f"{file_prefix}trajectories_{case}_{interp_order}_{minim}_{expt}.nc"
    )
    output_path_data = (
        odir + f"{file_prefix}data_{case}_{interp_order}_{minim}_{expt}.nc"
    )

    ds_traj = xr.open_dataset(output_path)

    print(ds_traj)

    varlist = [
        "th",
        "u",
        "v",
        "w",
        "q_cloud_liquid_mass",
        "th_L",
    ]

    source_dataset = load_data(
        files=files, ref_dataset=ref_dataset, traj=False, fields_to_keep=varlist
    )

    print(source_dataset)
    time1 = time.perf_counter()

    ds_traj_data = data_to_traj(
        source_dataset,
        ds_traj,
        varlist,
        output_path_data,
        interp_order=5,
    )

    time2 = time.perf_counter()

    delta_t = time2 - time1

    print(f"Elapsed time = {delta_t}")

    print(ds_traj_data)


if __name__ == "__main__":

    case = "cloud"

    minim = "fixed_point_iterator"

    interp_order = 5

    expt_list = ["std"]

    root_path = "C:/Users/paclk/OneDrive - University of Reading/ug_project_data/Data/"

    data_path = root_path + ""
    odir = root_path + "trajectories/"

    file_prefix = "diagnostics_3d_ts_"
    ref_file = "diagnostics_ts_"

    selector = "*"
    for expt in expt_list:

        with optional_debugging(False):
            main(
                data_path,
                file_prefix,
                ref_file,
                odir,
                case=case,
                interp_order=interp_order,
                minim=minim,
                expt=expt,
                selector=selector,
            )
