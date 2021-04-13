import numpy as np

from .constants import L_over_cp, c_virtual, grav, L_vap, Cp


def compute_derived_variables(traj, derived_variable_list=None):
    if derived_variable_list is None:
        derived_variable_list = {
            "q_total": r"$q_{t}$ kg/kg",
            "th_L": r"$\theta_L$ K",
            "th_v": r"$\theta_v$ K",
            "MSE": r"MSE J kg$^{-1}$",
        }
    zn = traj.coords["zn"]
    tr_z = np.interp(traj.trajectory[..., 2], traj.coords["zcoord"], zn)

    piref_z = np.interp(tr_z, zn, traj.refprof["pi"])
    thref_z = np.interp(tr_z, zn, traj.refprof["th"])

    s = list(np.shape(traj.data))
    s.pop()
    s.append(len(derived_variable_list))
    out = np.zeros(s)

    for i, variable in enumerate(derived_variable_list.keys()):

        if variable == "q_total":
            data = (
                traj.data[..., traj.var("q_vapour")]
                + traj.data[..., traj.var("q_cloud_liquid_mass")]
            )

        if variable == "th_L":
            data = (
                traj.data[..., traj.var("th")]
                - L_over_cp * traj.data[..., traj.var("q_cloud_liquid_mass")] / piref_z
            )

        if variable == "th_v":
            data = traj.data[..., traj.var("th")] + thref_z * (
                c_virtual * traj.data[..., traj.var("q_vapour")]
                - traj.data[..., traj.var("q_cloud_liquid_mass")]
            )

        if variable == "MSE":
            data = (
                traj.data[:, :, traj.var("th")] * Cp * piref_z
                + grav * tr_z
                + L_vap * traj.data[:, :, traj.var("q_vapour")]
            )

        out[..., i] = data
    return derived_variable_list, out
