"""
Interface for producing trajectories from MONC LES model output

"""

import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from monc_utils.data_utils.string_utils import get_string_index
from monc_utils.io.datain import get_data_on_grid

from advtraj.integrate import integrate_trajectories
from advtraj.utils.cli import optional_debugging
from advtraj.utils.grid import find_coord_grid_spacing


def load_data(files, ref_dataset=None, fields_to_keep=["w"]):

    tracer_fields = [
        "tracer_traj_xr",
        "tracer_traj_xi",
        "tracer_traj_yr",
        "tracer_traj_yi",
        "tracer_traj_zr",
    ]

    def preprocess(ds):
        return ds[fields_to_keep + tracer_fields + ["options_database", "z", "zn"]]

    def sortkey(p):
        idx_string = p.name.split(".")[-2]
        i, j = int(idx_string[:4]), int(idx_string[4:])
        return j, i

    ds = xr.open_mfdataset(files, preprocess=preprocess)

    ds = ds.rename({"z": "z_w", "zn": "z_p"})

    for f in fields_to_keep + tracer_fields:
        ds[f] = get_data_on_grid(ds, ref_dataset, f)

    ds = ds.rename(dict(x_p="x", y_p="y", z_p="z")).drop_vars("z_w")
    for v in tracer_fields:
        ds = ds.rename({v: "traj_tracer_{}".format(v.split("_")[-1])})

    # simulations with MONC always have periodic boundary conditions
    ds.attrs["xy_periodic"] = True

    # add the grid-spacing as attributes to speed up calculations
    for c in "xyz":

        ds[c].attrs[f"d{c}"] = find_coord_grid_spacing(
            da_coord=ds[c], show_warnings=False
        )
        if c in "xy":
            ds[c].attrs[f"L{c}"] = np.ptp(ds[c].values) + ds[c].attrs[f"d{c}"]
        else:
            ds[c].attrs[f"L{c}"] = np.ptp(ds[c].values)

    if "options_database" in ds.variables:
        [iit] = get_string_index(ds["options_database"].dims, ["time"])
        tv = list(ds["options_database"].dims)[iit]
        ds = ds.drop_vars("options_database")
        ds = ds.drop_vars(tv)

    return ds


def main(
    data_path,
    file_prefix,
    ref_file,
    output_path,
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

    if case == "w":

        ds = load_data(files=files, ref_dataset=ref_dataset, fields_to_keep=["w"])

        ds_subset = ds.isel(time=int(ds.time.count()) // 2).sel(z=slice(300, None))

        # we'll use as starting points for the trajectories all points where
        # the vertical velocity is 80% of the maximum value
        w_max = ds_subset.w.max()
        ds_poi = (
            ds_subset.where(ds_subset.w > 0.9 * w_max, drop=True)
            # ds_subset.where(ds_subset.w == w_max, drop=True)
            .stack(trajectory_number=("x", "y", "z")).dropna(dim="trajectory_number")
        )

    elif case == "cloud":

        ds = load_data(
            files=files, ref_dataset=ref_dataset, fields_to_keep=["q_cloud_liquid_mass"]
        )

        ds_subset = ds.isel(time=int(ds.time.count()) // 2).sel(z=slice(300, None))

        # we'll use as starting points for the trajectories all points where
        # the cloud liquid water mass is > 1E-5.
        thresh = 1e-5
        ds_poi = (
            ds_subset.where(ds_subset.q_cloud_liquid_mass > thresh, drop=True)
            .stack(trajectory_number=("x", "y", "z"))
            .dropna(dim="trajectory_number")
        )

    # now we'll turn this 1D dataset where (x, y, z) are coordinates into
    # one where they are variables instead
    ds_starting_points = (
        ds_poi.reset_index("trajectory_number")
        .assign_coords(trajectory_number=np.arange(ds_poi.trajectory_number.count()))
        .reset_coords(["x", "y", "z"])[["x", "y", "z"]]
    )

    ds_starting_points["x_err"] = xr.zeros_like(ds_starting_points["x"])
    ds_starting_points["y_err"] = xr.zeros_like(ds_starting_points["y"])
    ds_starting_points["z_err"] = xr.zeros_like(ds_starting_points["z"])

    print(ds_starting_points)

    time1 = time.perf_counter()

    ds_traj = integrate_trajectories(
        ds_position_scalars=ds,
        ds_starting_points=ds_starting_points,
        interp_order=interp_order,
        solver=minim,
        options=options,
    )

    time2 = time.perf_counter()

    delta_t = time2 - time1

    print(f"Elapsed time = {delta_t}")

    print(ds_traj["time"] - ds_traj["ref_time"])

    attrs = {
        "interp_order": interp_order,
        "solver": minim,
    }

    if "PI" in minim:
        attrs["maxiter"] = options["pioptions"]["maxiter"]
        attrs["tol"] = options["pioptions"]["tol"]
    else:
        attrs["maxiter"] = options["minoptions"]["minimize_options"]["maxiter"]
        attrs["max_outer_loops"] = options["minoptions"]["max_outer_loops"]

    if "PI_hybrid" in minim:
        attrs["minimize_maxiter"] = options["pioptions"]["minoptions"][
            "minimize_options"
        ]["maxiter"]
        attrs["tol"] = options["pioptions"]["tol"]

    ds_traj.attrs = attrs
    ds_traj.to_netcdf(output_path)
    print(f"Trajectories saved to {output_path}")
    print(ds_traj)

    if case == "w":
        pass
    else:
        ds_traj = ds_traj.isel(trajectory_number=slice(0, 20))

    fig, ax = plt.subplots(3, 2, sharex=True, figsize=[10, 12])
    plt.suptitle(f"advtraj {minim} {interp_order} elapsed time = {delta_t:6.2f}")
    ds_traj["x"].plot.line(x="time", ax=ax[0, 0], add_legend=False)
    ds_traj["y"].plot.line(x="time", ax=ax[1, 0], add_legend=False)
    ds_traj["z"].plot.line(x="time", ax=ax[2, 0], add_legend=False)
    ax[0, 0].set_ylim([0, 7000])
    ax[1, 0].set_ylim([0, 7000])
    ax[2, 0].set_ylim([0, 2000])
    ds_traj["x_err"].plot.line(x="time", ax=ax[0, 1], add_legend=False)
    ds_traj["y_err"].plot.line(x="time", ax=ax[1, 1], add_legend=False)
    ds_traj["z_err"].plot.line(x="time", ax=ax[2, 1], add_legend=False)
    fig.tight_layout()
    fig.savefig(f"advtraj_{case}_{interp_order}_{minim}_{expt}.png")
    plt.show()


if __name__ == "__main__":

    case = "w"
    # case='cloud'

    minim = "PI_hybrid"
    # minim = 'PI'
    # minim = 'BFGS'
    # minim = 'CG'
    # minim = 'Nelder-Mead'
    # minim = 'SLSQP'

    interp_order = 5
    # expt = 'test'
    # expt = "ref"
    expt = "std"

    data_path = "C:/Users/paclk/OneDrive - University of Reading/ug_project_data/Data/"
    odir = "C:/Users/paclk/OneDrive - University of Reading/ug_project_data/Data/trajectories/"
    if not os.path.exists(odir):
        os.makedirs(odir)

    file_prefix = "diagnostics_3d_ts_"
    ref_file = "diagnostics_ts_"
    output_path = (
        odir + f"{file_prefix}trajectories_{case}_{interp_order}_{minim}_{expt}.nc"
    )
    if expt == "ref":
        options = {
            "pioptions": {
                "maxiter": 1000,
                "miniter": 20,
                "disp": False,
                "relax": 0.8,
                "tol": 0.000001,
                "norm": "max_abs_error",
                "minoptions": {
                    "max_outer_loops": 1,
                    "solver": "BFGS",
                    "minimize_options": {
                        "maxiter": 200,
                        "disp": False,
                    },
                },
            },
            "minoptions": {
                "max_outer_loops": 1,
                "tol": 0.000001,
                "minimize_options": {
                    "maxiter": 200,
                    "disp": False,
                },
            },
        }

    elif expt == "std":

        options = {
            "pioptions": {
                "maxiter": 500,
                "miniter": 10,
                "disp": False,
                "relax": 0.8,
                "tol": 0.01,
                "norm": "max_abs_error",
                "minoptions": {
                    "max_outer_loops": 1,
                    "solver": "BFGS",
                    "minimize_options": {
                        "maxiter": 50,
                        "disp": False,
                    },
                },
            },
            "minoptions": {
                "max_outer_loops": 4,
                "tol": 0.01,
                "minimize_options": {
                    "maxiter": 10,
                    "disp": False,
                },
            },
        }

    elif expt == "test":

        options = {
            "pioptions": {
                "maxiter": 100,
                "miniter": 10,
                "disp": False,
                "relax": 1.0,
                "tol": 0.01,
                "alt_delta": False,
                "use_midpoint": False,
                "norm": "max_abs_error",
                "minoptions": {
                    "max_outer_loops": 1,
                    "solver": "BFGS",
                    "minimize_options": {
                        "maxiter": 10,
                        "disp": False,
                    },
                },
            },
            "minoptions": {
                "max_outer_loops": 4,
                "tol": 0.01,
                "minimize_options": {
                    "maxiter": 500,
                    "disp": False,
                },
            },
        }

    else:

        options = {
            "pioptions": {
                "maxiter": 100,
                "miniter": 10,
                "disp": False,
                "relax": 1.0,
                "tol": 0.01,
                "alt_delta": False,
                "use_midpoint": True,
                "norm": "max_abs_error",
                "minoptions": {
                    "max_outer_loops": 1,
                    "solver": "BFGS",
                    "minimize_options": {
                        "maxiter": 10,
                        "disp": False,
                    },
                },
            },
            "minoptions": {
                "max_outer_loops": 4,
                "tol": 0.01,
                "minimize_options": {
                    "maxiter": 500,
                    "disp": False,
                },
            },
        }

    with optional_debugging(False):
        main(
            data_path,
            file_prefix,
            ref_file,
            output_path,
            case=case,
            interp_order=interp_order,
            minim=minim,
            expt=expt,
            options=options,
        )
