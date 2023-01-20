"""
monc_test_data_to_traj_family.py
Script to interpolate gridded data to trajectory family.

"""

import glob
import time
from pathlib import Path

import xarray as xr
from load_data import load_data

from advtraj.family.traj_family import data_to_traj_family
from advtraj.utils.cli import optional_debugging


def main(
    data_path,
    file_prefix,
    ref_file,
    odir,
    case="w",
    interp_order=1,
    minim="PI",
    expt="",
    options=None,
):

    files = list(Path(data_path).glob(f"{file_prefix}*.nc"))

    ref_files = list(Path(data_path).glob(f"{ref_file}*.nc"))

    if len(ref_files) == 0:
        ref_dataset = None
    else:
        ref_dataset = xr.open_dataset(ref_files[0])

    varlist = [
        "u",
        "v",
        "w",
        "q_cloud_liquid_mass",
        "th_L",
    ]

    source_dataset = load_data(
        files=files, ref_dataset=ref_dataset, traj=False, fields_to_keep=varlist
    )

    time1 = time.perf_counter()

    output_path = (
        odir + f"{file_prefix}trajectories_{case}_{interp_order}_{minim}_{expt}"
    )

    traj_files = list(glob.glob(f"{output_path}_[0-9][0-9].nc"))

    data_to_traj_family(traj_files, source_dataset, varlist, odir, interp_order=5)

    time2 = time.perf_counter()

    delta_t = time2 - time1

    print(f"Elapsed time = {delta_t}")


if __name__ == "__main__":

    case = "cloud"

    minim = "fixed_point_iterator"

    interp_order_list = [5]

    expt_list = ["std"]

    root_path = "C:/Users/paclk/OneDrive - University of Reading/"

    data_path = root_path + "ug_project_data/Data/"
    odir = root_path + "ug_project_data/Data/trajectories/"

    file_prefix = "diagnostics_3d_ts_"
    ref_file = "diagnostics_ts_"

    for expt in expt_list:

        for interp_order in interp_order_list:

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
                )
