# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 17:53:20 2022

@author: paclk
"""
import numpy as np
import xarray as xr
from monc_utils.io.datain import get_data_on_grid

from advtraj.utils.grid import find_coord_grid_spacing


def load_data(files, ref_dataset=None, traj=True, fields_to_keep=None):

    if fields_to_keep is None:
        fields_to_keep = []

    tracer_fields = [
        "tracer_traj_xr",
        "tracer_traj_xi",
        "tracer_traj_yr",
        "tracer_traj_yi",
        "tracer_traj_zr",
    ]

    data_fields = fields_to_keep

    if traj:
        data_fields += tracer_fields

    def preprocess(ds):
        if "options_database" in list(ds.data_vars):
            ds = ds.drop_vars(["options_database"])
        return ds

    def sortkey(p):
        idx_string = p.name.split(".")[-2]
        i, j = int(idx_string[:4]), int(idx_string[4:])
        return j, i

    dsod = xr.open_dataset(files[0])

    ds_in = xr.open_mfdataset(files, preprocess=preprocess, combine_attrs="override")
    for var in ["options_database", "z", "zn"]:
        ds_in[var] = dsod[var]

    # print(ds_in)

    ds = xr.Dataset(attrs=ds_in.attrs)

    for f in data_fields:
        ds[f] = get_data_on_grid(ds_in, ref_dataset, f)

    if traj:
        for v in tracer_fields:
            ds = ds.rename({v: "traj_tracer_{}".format(v.split("_")[-1])})

    ds = ds.rename(dict(x_p="x", y_p="y", z_p="z"))

    if "z_w" in ds.coords:
        ds = ds.drop_vars("z_w")

    # simulations with MONC always have periodic boundary conditions
    ds.attrs["xy_periodic"] = True

    # add the grid-spacing as attributes to speed up calculations
    for c in "xy":

        ds[c].attrs[f"d{c}"] = find_coord_grid_spacing(
            da_coord=ds[c], show_warnings=False
        )

        if c in "xy":
            ds[c].attrs[f"L{c}"] = np.ptp(ds[c].values) + ds[c].attrs[f"d{c}"]
        else:
            ds[c].attrs[f"L{c}"] = np.ptp(ds[c].values)

    ds["z"].attrs["dz"] = np.mean(ds["z"].values[1:] - ds["z"].values[:-1])
    ds["z"].attrs["Lz"] = ds["z"].values[-1] - ds["z"].values[0]

    return ds
