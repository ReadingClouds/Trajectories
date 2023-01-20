"""
monc_test_traj_classify.py
Script to test trajectory classification.

"""

import xarray as xr
from cloud_classify import cloud_bl_classify

from advtraj.classify.traj_classify import print_class, set_traj_class


def main(
    data_path,
    file_prefix,
    odir,
    case="w",
    interp_order=5,
    minim="fixed_point_iterator",
    expt="std",
    options=None,
    obj_select=None,
):

    output_path = (
        odir + f"{file_prefix}trajectories_{case}_{interp_order}_{minim}_{expt}.nc"
    )
    output_path_data = (
        odir + f"{file_prefix}data_{case}_{interp_order}_{minim}_{expt}.nc"
    )

    ds_traj = xr.open_dataset(output_path)

    ds_traj_data = xr.open_dataset(output_path_data)

    ds_all = xr.merge((ds_traj, ds_traj_data))

    print(ds_all)

    thresh = 1e-5
    mask_cloud = ds_all["q_cloud_liquid_mass"] >= thresh
    mask_cloud.name = "state mask"

    traj_class_cloud = set_traj_class(mask_cloud)

    print("Cloud classes")
    print_class(ds_traj, traj_class_cloud, sel_obj=obj_select, list_classes=True)
    cloud_base = ds_all.z.values[mask_cloud].min()

    print(f"{cloud_base=}")

    mask_bl = ds_all["z"] >= cloud_base
    mask_bl.name = "state mask"

    traj_class_bl = set_traj_class(mask_bl)

    print("BL classes")
    print_class(ds_traj, traj_class_bl, sel_obj=obj_select, list_classes=True)

    traj_class_cloud_bl = cloud_bl_classify(ds_all)

    print(traj_class_cloud_bl)

    print("Combined cloud/BL classes")

    print_class(ds_traj, traj_class_cloud_bl, sel_obj=obj_select, list_classes=True)


if __name__ == "__main__":

    case = "cloud"

    minim = "fixed_point_iterator"

    interp_order = 5

    expt = "std"

    root_path = "C:/Users/paclk/OneDrive - University of Reading/ug_project_data/Data/"

    data_path = root_path + ""
    odir = root_path + "trajectories/"
    file_prefix = "diagnostics_3d_ts_"

    obj_select = [20]

    main(
        data_path,
        file_prefix,
        odir,
        case=case,
        interp_order=interp_order,
        minim=minim,
        expt=expt,
        obj_select=obj_select,
    )
