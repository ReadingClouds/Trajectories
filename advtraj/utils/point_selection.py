# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 12:04:57 2022

@author: paclk
"""
import numpy as np
import xarray as xr


def mask_to_positions(mask: xr.DataArray) -> xr.Dataset:
    """
    Convert 3D logical mask to coordinate positions.

    Parameters
    ----------
    mask : xr.DataArray
        Evaluates True at required positions.

    Returns
    -------
    positions : xr.Dataset
        Contains data variables "x", "y", "z".
        Coordinates "pos_number" and any others (e.g. "time") in mask.

    """
    poi = (
        mask.where(mask, drop=True)
        .stack(pos_number=("x", "y", "z"))
        .dropna(dim="pos_number")
    )
    # now we'll turn this 1D dataset where (x, y, z) are coordinates into
    # one where they are variables instead
    positions = (
        poi.reset_index("pos_number")
        .assign_coords(pos_number=np.arange(poi.pos_number.size))
        .reset_coords(["x", "y", "z"])[["x", "y", "z"]]
    )

    return positions
