"""
Interface for producing trajectories from UCLA-LES model output

Model version with advective tracer trajectories implemented:
https://github.com/leifdenby/uclales/tree/advective-trajectories
"""
from collections import OrderedDict
from pathlib import Path

import numpy as np
import xarray as xr

from .. import integrate_trajectories
from ..utils.cli import optional_debugging
from ..utils.grid import find_coord_grid_spacing


def center_staggered_field(phi_da):
    """
    Create cell-centered values for staggered (velocity) fields
    """
    dim = [d for d in phi_da.dims if d.endswith("m")][0]
    newdim = dim.replace("m", "t")

    s_left, s_right = slice(0, -1), slice(1, None)

    # average vertical velocity to cell centers
    coord_vals = 0.5 * (
        phi_da[dim].isel(**{dim: s_left}).values
        + phi_da[dim].isel(**{dim: s_right}).values
    )
    coord = xr.DataArray(
        coord_vals, coords={newdim: coord_vals}, attrs=dict(units="m"), dims=(newdim,)
    )

    # create new coordinates for cell-centered vertical velocity
    coords = OrderedDict(phi_da.coords)
    del coords[dim]
    coords[newdim] = coord

    phi_cc_vals = 0.5 * (
        phi_da.isel(**{dim: s_left}).values + phi_da.isel(**{dim: s_right}).values
    )

    dims = list(phi_da.dims)
    dims[dims.index(dim)] = newdim

    phi_cc = xr.DataArray(
        phi_cc_vals,
        coords=coords,
        dims=dims,
        attrs=dict(units=phi_da.units, long_name=phi_da.long_name),
    )

    phi_cc.name = phi_da.name

    return phi_cc


def load_data(files, fields_to_keep=["w"]):
    tracer_fields = ["atrc_xr", "atrc_xi", "atrc_yr", "atrc_yi", "atrc_zr"]

    def preprocess(ds):
        return ds[fields_to_keep + tracer_fields]

    def sortkey(p):
        idx_string = p.name.split(".")[-2]
        i, j = int(idx_string[:4]), int(idx_string[4:])
        return j, i

    # files = sorted(files, key=sortkey)

    ds = xr.open_mfdataset(files, preprocess=preprocess)

    ds.w.attrs["long_name"] = ds.w.longname

    for vel_field in ["u", "v", "w"]:
        if vel_field in fields_to_keep:
            ds[vel_field] = center_staggered_field(ds[vel_field])

    ds = ds.rename(dict(xt="x", yt="y", zt="z"))
    for v in tracer_fields:
        ds = ds.rename({v: "traj_tracer_{}".format(v.split("_")[-1])})

    # simulations with UCLA-LES always have periodic boundary conditions
    ds.attrs["xy_periodic"] = True

    # add the grid-spacing and domain extent as attributes
    # dx, dy, dz and Lx, Ly, Lz respectively to speed up calculations
    for c in "xyz":
        ds[c].attrs[f"d{c}"] = find_coord_grid_spacing(
            da_coord=ds[c], show_warnings=False
        )
        if c in "xy":
            ds[c].attrs[f"L{c}"] = np.ptp(ds[c].values) + ds[c].attrs[f"d{c}"]
        else:
            ds[c].attrs[f"L{c}"] = np.ptp(ds[c].values)

    return ds


def main(data_path, file_prefix, output_path):
    files = list(Path(data_path).glob(f"{file_prefix}.????????.nc"))

    ds = load_data(files=files, fields_to_keep=["w"])

    # as an example take the timestep half-way through the available data and
    # from 300m altitude and up
    ds_subset = ds.isel(time=int(ds.time.count()) // 2).sel(z=slice(300, None))

    # we'll use as starting points for the trajectories all points where the
    # vertical velocity is 80% of the maximum value
    w_max = ds_subset.w.max()
    ds_poi = (
        ds_subset.where(ds_subset.w > 0.8 * w_max, drop=True)
        .stack(trajectory_number=("x", "y", "z"))
        .dropna(dim="trajectory_number")
    )

    # now we'll turn this 1D dataset where (x, y, z) are coordinates into one
    # where they are variables instead
    ds_starting_points = (
        ds_poi.reset_index("trajectory_number")
        # .assign_coords(trajectory_number=np.arange(ds_poi.trajectory_number.count()))
        .reset_coords(["x", "y", "z"])[["x", "y", "z"]]
    )

    ds_traj = integrate_trajectories(
        ds_position_scalars=ds, ds_starting_points=ds_starting_points
    )
    output_path = output_path.format(file_prefix=file_prefix)
    ds_traj.to_netcdf(output_path)
    print(f"Trajectories saved to {output_path}")


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("data_path", type=Path)
    argparser.add_argument("file_prefix", type=Path)
    argparser.add_argument("--debug", default=False, action="store_true")
    argparser.add_argument(
        "--output", type=str, default="{file_prefix}.trajectories.nc"
    )
    args = argparser.parse_args()

    with optional_debugging(args.debug):
        main(
            data_path=args.data_path,
            file_prefix=args.file_prefix,
            output_path=args.output,
        )
