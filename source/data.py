from .constants import L_over_cp, c_virtual, grav
import numpy as np

var_properties = {
    "u": [True, False, False],
    "v": [False, True, False],
    "w": [False, False, True],
    "th": [False, False, False],
    "p": [False, False, False],
    "q_vapour": [False, False, False],
    "q_cloud_liquid_mass": [False, False, False],
    "tracer_rad1": [False, False, False],
    "tracer_rad2": [False, False, False],
}


def padleft(f, zt, axis=0):
    """
    Add dummy field at bottom of nD array

    Args:
        f : nD field
        zt: 1D zcoordinates
        axis=0: Specify axis to extend

    Returns:
        extended field, extended coord
    @author: Peter Clark
    """

    s = list(np.shape(f))
    s[axis] += 1
    #    print(zt)
    newfield = np.zeros(s)
    newfield[..., 1:] = f
    newz = np.zeros(np.size(zt) + 1)
    newz[1:] = zt
    newz[0] = 2 * zt[0] - zt[1]
    #    print(newz)
    return newfield, newz


def padright(f, zt, axis=0):
    """
    Add dummy field at top of nD array

    Args:
        f : nD field
        zt: 1D zcoordinates
        axis=0: Specify axis to extend

    Returns:
        extended field, extended coord
    @author: Peter Clark
    """

    s = list(np.shape(f))
    s[axis] += 1
    #    print(zt)
    newfield = np.zeros(s)
    newfield[..., :-1] = f
    newz = np.zeros(np.size(zt) + 1)
    newz[:-1] = zt
    newz[-1] = 2 * zt[-1] - zt[-2]
    #    print(newz)
    return newfield, newz


def d_by_dx_field(field, dx, varp, xaxis=0):
    """
    Numerically differentiate field in x direction assuming cyclic,
    mofiying grid descriptor.

    Parameters
    ----------
    field : array
        Data array, usualy 3D (x, y, z) but should be general.
    dx : float
        Grid spacing
    varp : List of logicals
        Describes grid offset. True means offset from p on grid in that axis.
    xaxis: int
        Pointer to x dimension - usually 0 but can be set to, e.g., 1 if time-
        dimension included in data.
    Returns
    -------
    None.

    """

    d = field[...]
    newfield = (d - np.roll(d, 1, axis=xaxis)) / dx
    varp[0] = not varp[0]
    return newfield, varp


def d_by_dy_field(field, dy, varp, yaxis=1):
    """
    Numerically differentiate field in y direction assuming cyclic,
    mofiying grid descriptor.

    Parameters
    ----------
    field : array
        Data array, usualy 3D (x, y, z) but should be general.
    dx : float
        Grid spacing
    varp : List of logicals
        Describes grid offset. True means offset from p on grid in that axis.
    xaxis: int
        Pointer to y dimension - usually 1 but can be set to, e.g., 2 if time-
        dimension included in data.
    Returns
    -------
    None.

    """

    d = field[...]
    newfield = (d - np.roll(d, 1, axis=yaxis)) / dy
    varp[1] = not varp[1]
    return newfield, varp


def d_by_dz_field(field, z, zn, varp):
    """
    Numerically differentiate field in z direction,
    mofiying grid descriptor.

    Parameters
    ----------
    field : array
        Data array, usualy 3D (x, y, z) but should be general.
        Assumes z axis is last dimension.
    dx : float
        Grid spacing
    varp : List of logicals
        Describes grid offset. True means offset from p on grid in that axis.
    xaxis: int
        Pointer to y dimension - usually 1 but can be set to, e.g., 2 if time-
        dimension included in data.
    Returns
    -------
    None.

    """

    zaxis = field.ndim - 1

    d = field[...]
    if varp[zaxis]:  # Field is on zn points
        new = (d[..., 1:] - d[..., :-1]) / (zn[1:] - zn[:-1])
        zt = 0.5 * (zn[1:] + zn[:-1])
        newfield, newz = padright(new, zt, axis=zaxis)
    else:  # Field is on z points
        new = (d[..., 1:] - d[..., :-1]) / (z[1:] - z[:-1])
        zt = 0.5 * (z[1:] + z[:-1])
        newfield, newz = padleft(new, zt, axis=zaxis)

    varp[-1] = not varp[-1]
    return newfield, varp


def get_data(source_dataset, var_name, it, refprof, coords):
    """
    Extract data from source NetCDF dataset, derived and/or processed data.

        Currently supported derived data are::

            'th_L'    : Liquid water potential temperature.
            'th_v'    : Virtual potential temperature.
            'q_total' : Total water
            'buoyancy': Bouyancy based on layer-mean theta_v.

        Currently supported operators are::

            'v_prime' : Deviation from level mean for any variable v.
            'v_crit'  : Level critical value for any variable v.
            'dv_dx'   : Derivative of v in x direction.
            'dv_dy'   : Derivative of v in y direction.
            'dv_dz'   : Derivative of v in z direction.

        Operators must be combined using parentheseses, e.g.
        'd(d(th_prime)_dx)_dy'.

    Args:
        source_dataset: handle of NetCDF dataset.
        var_name: String describing data required.
        it: Time index required.
        refprof: Dict containg reference profile.
        coords: Dict containing model coords.

    Returns:
        variable, variable_grid_properties

    @author: Peter Clark and George Efstathiou

    """

    global ind

    # This is the list of supported operators.
    # The offset is how many characters to strip from front of variable name
    # to get raw variable.
    # e.g. dth_dx needs 1 to lose the first d.

    opdict = {
        "_dx": 1,
        "_dy": 1,
        "_dz": 1,
        "_prime": 0,
        "_crit": 0,
    }
    vard = None
    varp = None
    #    print(ind,var_name)
    try:
        var = source_dataset[var_name]
        #        vardim = var.dimensions
        vard = var[it, ...]
        varp = var_properties[var_name]

        #        print(vardim)
        if var_name == "th" and refprof is not None:
            #            print(refprof)
            vard[...] += refprof["th"]

    except Exception:
        #        print("Data not in dataset")
        if var_name == "th_L":

            theta, varp = get_data(source_dataset, "th", it, refprof, coords)
            q_cl, vp = get_data(
                source_dataset, "q_cloud_liquid_mass", it, refprof, coords
            )
            vard = theta - L_over_cp * q_cl / refprof["pi"]

        elif var_name == "th_v":
            theta, varp = get_data(source_dataset, "th", it, refprof, coords)
            q_v, vp = get_data(source_dataset, "q_vapour", it, refprof, coords)
            q_cl, vp = get_data(
                source_dataset, "q_cloud_liquid_mass", it, refprof, coords
            )
            vard = theta + refprof["th"] * (c_virtual * q_v - q_cl)

        elif var_name == "q_total":

            q_v, varp = get_data(source_dataset, "q_vapour", it, refprof, coords)
            q_cl, vp = get_data(
                source_dataset, "q_cloud_liquid_mass", it, refprof, coords
            )
            vard = q_v + q_cl

        elif var_name == "buoyancy":

            th_v, varp = get_data(source_dataset, "th_v", it, refprof, coords)
            mean_thv = np.mean(th_v, axis=(0, 1))
            vard = grav * (th_v - mean_thv) / mean_thv

        else:
            if ")" in var_name:
                #                print(ind,"Found parentheses")
                v = var_name
                i1 = v.find("(")
                i2 = len(v) - v[::-1].find(")")
                source_var = v[i1 + 1 : i2 - 1]
                v_op = v[i2:]
            else:
                for v_op, offset in opdict.items():
                    if v_op in var_name:
                        source_var = var_name[offset : var_name.find(v_op)]
                        break

            #            print(ind,f"Variable: {source_var} v_op: {v_op}")
            #            ind+="    "
            vard, varp = get_data(source_dataset, source_var, it, refprof, coords)
            #            ind = ind[:-4]
            if "_dx" == v_op:
                #                print(ind,"Exec _dx")
                vard, varp = d_by_dx_field(vard, coords["deltax"], varp)

            elif "_dy" == v_op:
                #                print(ind,"Exec _dy")
                vard, varp = d_by_dy_field(vard, coords["deltay"], varp)

            elif "_dz" == v_op:
                #                print(ind,"Exec _dz")
                vard, varp = d_by_dz_field(vard, coords["z"], coords["zn"], varp)

            elif "_prime" == v_op:
                #                print(ind,"Exec _prime")
                vard = vard - np.mean(vard, axis=(0, 1))

            elif "_crit" == v_op:
                #                print(ind,"Exec _crit")
                std = np.std(vard, axis=(0, 1))

                std_int = np.ones(len(std))
                c_crit = np.ones(len(std))
                for iz in range(len(std)):
                    std_int[iz] = 0.05 * np.mean(std[: iz + 1])
                    c_crit[iz] = np.maximum(std[iz], std_int[iz])

                vard = np.ones_like(vard) * c_crit

        if vard is None:
            print(f"Variable {var_name} not available.")

    return vard, varp


def load_traj_step_data(dataset, it, variable_list, refprof, coords):
    """
    Function to read trajectory variables and additional data from file
    for interpolation to trajectory.

    Args:
        dataset        : netcdf file handle.
        it             : time index in netcdf file.
        variable_list  : List of variable names.
        refprof        : Dict with reference theta profile arrays.

    Returns:
        List of arrays containing interpolated data.

    @author: Peter Clark

    """

    global ind
    data_list, var_list, varp_list, time = load_traj_pos_data(dataset, it)

    for variable in variable_list:
        #        print 'Reading ', variable
        ind = ""
        data, varp = get_data(dataset, variable, it, refprof, coords)

        data_list.append(data)
        varp_list.append(varp)
    return data_list, variable_list, varp_list, time


def load_traj_pos_data(dataset, it, cyclic_xy):
    """
    Function to read trajectory position variables from file.
    Args:
        dataset        : netcdf file handle.
        it             : time index in netcdf file.
        cyclic_xy      : using cyclic boundary conditions in x,y-direction

    Returns:
        List of arrays containing interpolated data.

    @author: Peter Clark

    """

    if "CA_xrtraj" in dataset.variables.keys():
        # Circle-A Version
        trv = {
            "xr": "CA_xrtraj",
            "xi": "CA_xitraj",
            "yr": "CA_yrtraj",
            "yi": "CA_yitraj",
            "zpos": "CA_ztraj",
        }
        trv_noncyc = {"xpos": "CA_xtraj", "ypos": "CA_ytraj", "zpos": "CA_ztraj"}
    else:
        # Stand-alone Version
        trv = {
            "xr": "tracer_traj_xr",
            "xi": "tracer_traj_xi",
            "yr": "tracer_traj_yr",
            "yi": "tracer_traj_yi",
            "zpos": "tracer_traj_zr",
        }
        trv_noncyc = {
            "xpos": "tracer_traj_xr",
            "ypos": "tracer_traj_yr",
            "zpos": "tracer_traj_zr",
        }

    if cyclic_xy:
        xr = dataset.variables[trv["xr"]][it, ...]
        xi = dataset.variables[trv["xi"]][it, ...]

        yr = dataset.variables[trv["yr"]][it, ...]
        yi = dataset.variables[trv["yi"]][it, ...]

        zpos = dataset.variables[trv["zpos"]]
        zposd = zpos[it, ...]
        data_list = [xr, xi, yr, yi, zposd]
        varlist = ["xr", "xi", "yr", "yi", "zpos"]
        varp_list = [var_properties["th"]] * 5

    else:
        # Non-cyclic option may well not work anymore!
        xpos = dataset.variables[trv_noncyc["xpos"]][it, ...]
        ypos = dataset.variables[trv_noncyc["ypos"]][it, ...]
        zpos = dataset.variables[trv_noncyc["zpos"]]
        zposd = zpos[it, ...]
        data_list = [xpos, ypos, zposd]
        varlist = ["xpos", "ypos", "zpos"]
        varp_list = [var_properties["th"]] * 3

    # Needed as zpos above is numpy array not NetCDF variable.
    #    zpos = dataset.variables['CA_ztraj']
    times = dataset.variables[zpos.dimensions[0]]

    return data_list, varlist, varp_list, times[it]
