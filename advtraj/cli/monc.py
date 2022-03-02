"""
Interface for producing trajectories from MONC LES model output

"""

import os
from pathlib import Path

import xarray as xr
import numpy as np
import time

from monc_utils.io.datain import get_data_on_grid
from monc_utils.data_utils.string_utils import get_string_index

from advtraj.integrate import integrate_trajectories
from advtraj.utils.cli import optional_debugging
from advtraj.utils.grid import find_coord_grid_spacing

import matplotlib.pyplot as plt

def load_data(files, ref_dataset=None, fields_to_keep=["w"]):
    # tracer_fields = ["atrc_xr", "atrc_xi", "atrc_yr", "atrc_yi", "atrc_zr"]
    tracer_fields = ['tracer_traj_xr',
                     'tracer_traj_xi',
                     'tracer_traj_yr',
                     'tracer_traj_yi',
                     'tracer_traj_zr' ]

    def preprocess(ds):
        return ds[fields_to_keep + tracer_fields + ["options_database"]]

    def sortkey(p):
        idx_string = p.name.split(".")[-2]
        i, j = int(idx_string[:4]), int(idx_string[4:])
        return j, i

    # files = sorted(files, key=sortkey)
#    ds = xr.open_mfdataset(files)
#    ds.close()

    ds = xr.open_mfdataset(files, preprocess=preprocess)

    ds = ds.rename({'z':'z_w', 'zn':'z_p'})

    for f in fields_to_keep + tracer_fields:
        ds[f] = get_data_on_grid(ds, ref_dataset, f)

    ds = ds.rename(dict(x_p="x", y_p="y", z_p="z")).drop_vars('z_w')
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
            ds[c].attrs[f"L{c}"]= np.ptp(ds[c].values) + ds[c].attrs[f"d{c}"]
        else:
            ds[c].attrs[f"L{c}"]= np.ptp(ds[c].values)


    if "options_database" in ds.variables:
        [iit] = get_string_index(ds["options_database"].dims, ['time'])
        tv = list(ds["options_database"].dims)[iit]
        ds = ds.drop_vars("options_database")
        ds = ds.drop_vars(tv)

    return ds


def main(data_path, file_prefix, ref_file, output_path, interp_order=1):
    files = list(Path(data_path).glob(f"{file_prefix}*.nc"))

    ref_files = list(Path(data_path).glob(f"{ref_file}*.nc"))

    if len(ref_files) == 0:
        ref_dataset = None
    else:
        ref_dataset = xr.open_dataset(ref_files[0])

    ds = load_data(files=files, ref_dataset=ref_dataset, fields_to_keep=["w"])

    ds_ = ds.isel(time=int(ds.time.count()) // 2).sel(z=slice(300, None))

    X, Y, Z = np.meshgrid(ds_.x, ds_.y, ds_.z, indexing='ij')

#    mask = np.where(ds_.w.values == ds_.w.max().values)
    mask = np.where(ds_.w >= 0.9 * ds_.w.max())

    print(ds_.w.values[mask])

    tv = ds_.coords['time'].values
    x = xr.DataArray(X[mask])
    y = xr.DataArray(Y[mask])
    z = xr.DataArray(Z[mask])

    data = {
        "x": x,
        "y": y,
        "z": z,
    }

#    print(x/ds_.x.attrs['dx'], y/ds_.y.attrs['dy'], (z/ds_.z.attrs['dz']+0.5))

    ds_starting_points = xr.Dataset(data_vars = data, coords={'time':tv})

    print(ds_starting_points)

    time1 = time.perf_counter()

    ds_traj = integrate_trajectories(ds_position_scalars=ds,
                                     ds_starting_points=ds_starting_points,
                                     interp_order=interp_order,
    )

    time2 = time.perf_counter()

    delta_t = time2 - time1

    print(f'Elapsed time = {delta_t}')


    output_path = output_path.format(file_prefix=file_prefix)
    ds_traj.to_netcdf(output_path)
    print(f"Trajectories saved to {output_path}")
    print(ds_traj)
    fig, ax = plt.subplots(3,1,sharex=True)
    plt.suptitle(f'advtraj elapsed time = {delta_t:6.2f}')
    ds_traj["x"].plot.line(x='time', ax = ax[0], add_legend=False)
    ds_traj["y"].plot.line(x='time', ax = ax[1], add_legend=False)
    ds_traj["z"].plot.line(x='time', ax = ax[2], add_legend=False)
    ax[0].set_ylim([0,7000])
    ax[1].set_ylim([0,7000])
    ax[2].set_ylim([0,2000])
    fig.savefig("C:/Users/paclk\OneDrive - University of Reading/Git/python/advtraj/advtraj/cli/advtraj.png")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    interp_order=1
    # data_path  = 'C:/Users/paclk/OneDrive - University of Reading/ug_project_data/Data/'
    # odir = 'C:/Users/paclk/OneDrive - University of Reading/ug_project_data/Data/trajectories/'


    # file_prefix = 'diagnostics_3d_ts_'
    # ref_file = 'diagnostics_ts_'
    # output_path = odir+"{file_prefix}trajectories.nc"
    # main(data_path, file_prefix, ref_file, output_path,
    #      interp_order=interp_order)

    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("data_path", type=Path)
    argparser.add_argument("file_prefix", type=Path)
    argparser.add_argument("ref_file", type=Path)
    argparser.add_argument("odir", type=Path)
    argparser.add_argument("--debug", default=False, action="store_true")
    args = argparser.parse_args()

    if not os.path.exists(args.odir):
        os.makedirs(args.odir)
    output_path=args.odir+f"{args.file_prefix}trajectories.nc"

    with optional_debugging(args.debug):
        main(
            data_path=args.data_path,
            file_prefix=args.file_prefix,
            ref_file=args.ref_file,
            output_path=output_path,
            interp_order=interp_order,
        )
