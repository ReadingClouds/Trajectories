"""
Interface for producing trajectories from UCLA-LES model output

Model version with advective tracer trajectories implemented:
https://github.com/leifdenby/uclales/tree/advective-trajectories
"""
from pathlib import Path
import xarray as xr

from .. import interpolate
from ..utils import optional_debugging


def main(data_path, file_prefix, output_path):
    tracer_fields = ["atrc_xr", "atrc_xi", "atrc_yr", "atrc_yi", "atrc_zr"]

    def preprocess(ds):
        return ds[tracer_fields]

    files = list(Path(data_path).glob(f"{file_prefix}.????????.nc"))

    ds = xr.open_mfdataset(files, preprocess=preprocess)
    ds = ds.rename(dict(xt="x", yt="y", zt="z"))
    for v in tracer_fields:
        ds = ds.rename({v: "traj_tracer_{}".format(v.split("_")[-1])})

    # simulations with UCLA-LES always have periodic boundary conditions
    ds.attrs["xy_periodic"] = True

    # n_timesteps = int(ds.time.count())
    ds_starting_points = xr.Dataset()
    ds_starting_points["x"] = ds.x.mean()
    ds_starting_points["y"] = ds.y.mean()
    ds_starting_points["z"] = (ds.z.max() - ds.z.min()) * 0.1
    ds_starting_points["time"] = ds.time.isel(time=-1)

    ds_traj = interpolate.integrate_trajectories(
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
