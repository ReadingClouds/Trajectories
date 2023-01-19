# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:38:37 2022

@author: xm904103
"""
import xarray as xr

from advtraj.classify.traj_classify import combine_traj_classes, set_traj_class


def redefine_combined_keys(traj_class_cloud_bl):
    """
    Give combined classes more meaningful names

    Parameters
    ----------
    traj_class_cloud_bl : dict
        Combined cloud and bl classification.

    Returns
    -------
    dict
        Combined cloud and bl classification with new class names
        (and numbers).

    """

    keys = traj_class_cloud_bl["key"]

    new_keys = []
    new_traj_class = xr.full_like(traj_class_cloud_bl["class"], -1, dtype=int)
    v = new_traj_class.values

    keys_list = list(keys)
    keys_list.sort()

    for k in keys_list:

        val = keys[k]
        new_key = ""
        cloud_key = val[0]
        bl_key = val[1]
        if cloud_key[0] == "In":
            if cloud_key[1] < 0:
                new_key = f"Prev Cloud {cloud_key[1]}"
            elif cloud_key[1] == 0:
                new_key = "Cloud"
            else:
                new_key = f"Post Cloud {cloud_key[1]}"
        elif cloud_key[0] == "Pre":
            if cloud_key[1] == 0:
                new_key = "Pre-entrainment"
            else:
                new_key = f"Pre-entr Cloud {cloud_key[1]}"
        elif cloud_key[0] == "Post":
            if cloud_key[1] == 0:
                new_key = "Post-detrainment"
            else:
                new_key = f"Post-detr Cloud {cloud_key[1]}"

        if bl_key[0] == "Pre" or (bl_key[0] == "In" and bl_key[1] < 0):
            new_key += " BL"

        mask = traj_class_cloud_bl["class"].values == k

        if mask.sum() > 0:
            if new_key not in new_keys:
                new_keys.append(new_key)
            new_key_no = new_keys.index(new_key)
            v[mask] = new_key_no
    new_traj_class.values = v
    new_keys = {i: v for i, v in enumerate(new_keys)}

    return {"class": new_traj_class, "key": new_keys}


def cloud_bl_classify(ds):
    """
    Classify cloud objects depending origin in BL or not.

    Parameters
    ----------
    ds : xarray dataset
        Combined trajectory and supporting data.
        Must contain q_cloud_liquid_mass

    Returns
    -------
    traj_class_cloud_bl : dict
        'class': New class numbers
        'key'  : Key re-defined using redefine_combined_keys.

    """
    # First generate cloud classification.

    thresh = 1e-5
    mask_cloud = ds["q_cloud_liquid_mass"] >= thresh
    mask_cloud.name = "state mask"

    traj_class_cloud = set_traj_class(mask_cloud)

    cloud_base = ds.z.values[mask_cloud].min()

    # Second generate BL classification; was air below cloud base?

    mask_bl = ds["z"] >= cloud_base
    mask_bl.name = "state mask"

    traj_class_bl = set_traj_class(mask_bl)

    # Combine the wo classifications
    traj_class_cloud_bl = combine_traj_classes((traj_class_cloud, traj_class_bl))

    # Give combined classes more meaningful names.
    traj_class_cloud_bl = redefine_combined_keys(traj_class_cloud_bl)

    return traj_class_cloud_bl
