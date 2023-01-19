"""
Module cloud_properties.

Created on Thu Aug 27 16:32:10 2020

@author: Peter Clark
"""

from itertools import product

import numpy as np
import xarray as xr


def set_traj_class(mask):
    """
    Compute trajectory classification.

    Args
    ----
        mask: xarray DataArray contain timeseries of mask data.

    Returns
    -------
        Dictionary containing trajectory class points, key and useful derived data::

          Dictionary keys:
          "class"
          "key"
    """
    traj_class = xr.full_like(mask, 0, dtype=int)

    it_ref = list(mask.time).index(mask.ref_time)

    m = mask.isel(time=it_ref)

    v = xr.where(m, 0, -1)

    traj_class.loc[dict(time=mask.ref_time)] = v.values

    for it in range(it_ref - 1, -1, -1):

        t = mask.time.isel(time=it)

        m = mask.isel(time=it)
        m1 = mask.isel(time=it + 1)

        change = np.logical_xor(m, m1)

        v1 = traj_class.isel(time=it + 1)

        v = xr.where(change, v1 - 1, v1)

        traj_class.loc[dict(time=t)] = v.values

    for it in range(it_ref + 1, mask.time.size, 1):

        t = mask.time.isel(time=it)

        m = mask.isel(time=it)
        m1 = mask.isel(time=it - 1)

        change = np.logical_xor(m, m1)

        v1 = traj_class.isel(time=it - 1)

        v = xr.where(change, v1 + 1, v1)

        traj_class.loc[dict(time=t)] = v.values

    traj_class_key = {}

    for ic in range(traj_class.min().item(), 0):
        if ic % 2 == 0:
            traj_class_key[ic] = ("In", (ic + 1) // 2)
        else:
            traj_class_key[ic] = ("Pre", (ic + 1) // 2)

    traj_class_key[0] = ("In", 0)

    for ic in range(1, traj_class.max().item() + 1):
        if ic % 2 == 0:
            traj_class_key[ic] = ("In", (ic) // 2)
        else:
            traj_class_key[ic] = ("Post", (ic) // 2)

    traj_class.name = "class_no"

    return {"class": traj_class, "key": traj_class_key}


def combine_traj_classes(traj_class: tuple):
    """
    Combine multiple trajectory classifications. Currently only supports 'and'.

    Args
    ----
        traj_class: tuple of matching dicts.
            Tuple of timeseries of trajectory class dictionaries.

    Returns
    -------
        Dictionary containing combined trajectory class points, key and useful derived data::

          Dictionary keys:
          "class"
          "key"
    """

    keylist = [list(tr["key"].keys()) for tr in traj_class]

    new_keys = {}
    new_traj_class = xr.full_like(traj_class[0]["class"], -1, dtype=int)
    v = new_traj_class.values

    key_no = 0
    for keys in product(*keylist):
        mask = xr.full_like(traj_class[0]["class"], True, dtype=bool)
        new_key = []
        for (k, tr) in zip(keys, traj_class):
            mask = np.logical_and(mask, tr["class"] == k)
            new_key.append(tr["key"][k])

        if mask.sum() > 0:
            new_keys[key_no] = new_key
            v[mask] = key_no
            key_no += 1
    new_traj_class.values = v

    return {"class": new_traj_class, "key": new_keys}


def print_class(traj, traj_cl, sel_obj=None, list_classes=True):
    """
    Print cloud classification properties.

    Parameters
    ---
        traj       : xarray dataset
            Trajectory dataset
        traj_cl    : Dict of
            Classifications of trajectory points provided by set_cloud_class
            function.
        sel_obj    : list(int)
            List of object numbers to print.

    """

    trm = xr.merge((traj, traj_cl["class"]))

    keys = traj_cl["key"]

    keys_list = list(keys)
    keys_list.sort()

    if sel_obj is None:
        sel_obj = traj.object_label.values

    if list_classes:
        for iclass, key in enumerate(keys_list):
            val = keys[key]
            print(f"{iclass:2d}: {key:4d} {val}")

    for iobj, traj in trm.groupby("object_label"):

        if np.isin(iobj, sel_obj):

            print(f"Object: {iobj}")

            tr_class = traj["class_no"]

            strout = "time "
            for (iclass, key) in enumerate(traj_cl["key"]):
                strout += f"{iclass:5d} "
            strout += " total"
            print(strout)

            for it in range(0, len(traj.time)):
                tr = tr_class.isel(time=it)
                strout = f"{it:4d} "
                tot = 0
                for iclass, key in enumerate(keys_list):
                    num_in_class = (tr == key).sum().item()
                    strout += f"{num_in_class:5d} "
                    tot += num_in_class
                strout += f"{tot:6d} "
                print(strout)
