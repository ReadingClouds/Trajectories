"""
Interface for producing trajectories from UCLA-LES model output

Model version with advective tracer trajectories implemented:
https://github.com/leifdenby/uclales/tree/advective-trajectories
"""
from collections import OrderedDict
from pathlib import Path

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

    # add the grid-spacing as attributes to speed up calculations
    for c in "xyz":
        ds[c].attrs[f"d{c}"] = find_coord_grid_spacing(
            da_coord=ds[c], show_warnings=False
        )

    return ds


def main(data_path, file_prefix, output_path):
    files = list(Path(data_path).glob(f"{file_prefix}.????????.nc"))

    ds = load_data(files=files, fields_to_keep=["w"])

    ds_ = ds.isel(time=int(ds.time.count()) // 2).sel(z=slice(300, None))
    da_pt = ds_.where(ds_.w == ds_.w.max(), drop=True)

    # n_timesteps = int(ds.time.count())
    ds_starting_points = xr.Dataset()
    ds_starting_points["x"] = da_pt.x.item()
    ds_starting_points["y"] = da_pt.y.item()
    ds_starting_points["z"] = da_pt.z.item()
    ds_starting_points["time"] = da_pt.time

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
