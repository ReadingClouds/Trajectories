import xarray as xr
import numpy as np
from scipy.constants import pi

from advtraj.compute_trajectories import (compute_trajectories,
                                          trajectory_cloud_ref,
                                          in_cloud)


def _create_synthetic_dataset(dL, L, t_max, dt, U):
    Lx, Ly, Lz = L
    dx, dy, dz = dL

    x_ = np.arange(0, Lx, dx)
    y_ = np.arange(0, Ly, dy)
    z_ = np.arange(0, Lz, dz)
    t_ = np.arange(0, t_max, dt)

    ds = xr.Dataset(coords=dict(x=x_, y=y_, z=z_, t=t_))
    ds.x.attrs['units'] = 'm'
    ds.y.attrs['units'] = 'm'
    ds.z.attrs['units'] = 'm'
    ds.x.attrs['long_name'] = 'x-horz. posn.'
    ds.y.attrs['long_name'] = 'y-horz. posn.'
    ds.z.attrs['long_name'] = 'height'

    x_pos = ds.x + U[0]*ds.t
    y_pos = ds.y + U[1]*ds.t
    z_pos = ds.z + U[2]*ds.t

    ds['tracer_traj_xr'] = np.cos(2.*pi*x_pos/Lx) + 0.*y_pos + 0.*z_pos
    ds['tracer_traj_xi'] = np.sin(2.*pi*x_pos/Ly) + 0.*y_pos + 0.*z_pos
    ds['tracer_traj_yr'] = 0.*x_pos + np.cos(2.*pi*y_pos/Ly) + 0.*z_pos
    ds['tracer_traj_yi'] = 0.*x_pos + np.sin(2.*pi*y_pos/Ly) + 0.*z_pos
    ds['tracer_traj_zr'] = 0.*x_pos + 0.*y_pos + z_pos

    return ds


# def test_single_reference_time():
    # Lx = Ly = 2e3 # [m]
    # Lz = 2e3 # [m]
    # dx = dy = dz = 25.0 # [m]
    # dt = 60. # [s]
    # t_max = 600. # [s]
    # U = [1., 2., 4., ] # [m/s]

    # ds = _create_synthetic_dataset(
        # dL=(dx, dy, dz),
        # L=(Lx, Ly, Lz),
        # t_max=t_max,
        # dt=dt,
        # U=U
    # )

    # # trajectories code needs some reference profiles
    # ds['thref'] = 300.0 + 0.*ds.x + 0.*ds.y + 0.*ds.z
    # ds['th'] = 300.0 + 0.*ds.x + 0.*ds.y + 0.*ds.z
    # ds['q_cloud_liquid_mass'] = 300.0 + 0.*ds.x + 0.*ds.y + 1.0e-3*(ds.z > 600.) + 0.*ds.t

    # fn = "test_0.nc"
    # ds.to_netcdf(fn)

    # compute_trajectories(
        # files=["test_0.nc", "test_1.nc"],
        # start_time=ds.t.max().values,
        # ref_time=0.0,
        # end_time=ds.t.max().values,
        # variable_list=[],
        # thref=0.0,
        # ref_func=trajectory_cloud_ref,
    # )
