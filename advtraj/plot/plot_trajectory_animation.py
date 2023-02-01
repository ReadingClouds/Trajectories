# -*- coding: utf-8 -*-
"""
Plot Trajectories

@author: Peter Clark
"""

import matplotlib.pyplot as plt
import numpy as np
from cohobj.object_tools import box_xyz, get_bounding_boxes

# import matplotlib.colors as mcolors
from matplotlib import animation

from ..family.traj_family import family_coords

# from mpl_toolkits import mplot3d
from ..utils.point_selection import mask_to_positions

def_colors = [
    "blue",
    "blueviolet",
    "brown",
    "crimson",
    "cyan",
    "darkblue",
    "darkcyan",
    "darkgray",
    "darkgreen",
    "darkorange",
    "darkorchid",
    "darkred",
    "darksalmon",
    "darkseagreen",
    "darkslateblue",
    "darkslategray",
    "hotpink",
    "indianred",
    "indigo",
    "lavender",
    "limegreen",
    "magenta",
    "maroon",
    "mediumblue",
    "mediumorchid",
    "mediumpurple",
    "mediumseagreen",
    "mediumslateblue",
    "mediumspringgreen",
    "mediumturquoise",
    "mediumvioletred",
    "orange",
    "orangered",
    "silver",
    "skyblue",
    "slateblue",
    "slategray",
]


def plot_traj_animation(
    ds_traj,
    class_no=None,
    field_mask=None,
    select=None,
    galilean=None,
    title=None,
    legend=False,
    figsize=(15, 12),
    not_inobj_size=0.2,
    inobj_size=2.0,
    field_size=0.5,
    fps=10,
    anim_name=None,
    plot_mask=False,
    with_boxes=False,
    load_ds=True,
    view_point=(0, 0),
    x_lim=None,
    y_lim=None,
    z_lim=None,
    colors=None,
):
    """
    Plot animated trajectory points in 3D.

    Parameters
    ----------
    ds_traj : xarray Dataset
        Contains trajectory position data, object_label coord .
    class_no : dict, optional
        Trajectory classification. The default is None.
    field_mask : xarray DataArray, optional
        Eulerian field of bool with times matching ds_traj.
        The default is None.
    select : list(int), optional
        list of object_label values to plot. The default is None.
    galilean : tuple (u,v), optional
        Velocity compoinents for Galilean Tranform. The default is None.
    title : str, optional
        Title of plot. Default is None.
    legend : bool, optional
        Plot legend. The default is False.
    figsize : tuple (width, height), optional
        Size of figure. The default is (20,12).
    not_inobj_size : float, optional
        Size of blob when mask = False. The default is 0.2.
    inobj_size : float, optional
        Size of blob when mask = True. The default is 2.0.
    field_size : float, optional
        Size of blob when field_mask = True.. The default is 0.5.
    fps : float or int, optional
        frames per second animation. The default is 10.
    anim_name : str, optional
        File name for animation. None generated if None. The default is None.
    plot_mask : bool, optional
        Plot points with inobj_size where obj_mask is True.
    with_boxes : bool, optional
        Display bounding boxes. The default is False.
    load_ds : bool, optional
        Pre-load data from file. The default is True.
    view_point : tuple (elevation, azimuth), optional
        3D plot view point. The default is (0,0).
    x_lim : list (min, max), optional
        x-axis limits. The default is None.
    y_lim : list (min, max), optional
        y-axis limits. The default is None.
    z_lim : list (min, max), optional
        z-axis limits. The default is None.
    colors: list(colors), optional
        colors for objects or classes. The default is None.
    Returns
    -------
    animation object.

    """

    timestep = ds_traj.attrs["trajectory timestep"]

    ntimes = ds_traj.time.size
    nobj = ds_traj.object_label.nobjects

    plot_class = class_no is not None
    plot_field = field_mask is not None

    if plot_mask and "obj_mask" not in ds_traj.variables:
        print("Data does not contain obj_mask: plot_mask set to False.")
        plot_mask = False

    if plot_class:
        if with_boxes:
            print(
                "Option with_boxes is not compatible with plotting classes; set to False."
            )
            with_boxes = False

    if select is None:
        select = np.arange(0, nobj)

    var = ["x", "y", "z"]

    if plot_mask:
        var.append("obj_mask")

    ds = ds_traj[var]

    if class_no is not None:
        #        class_key = class_no['key']
        ds = ds.merge(class_no["class"])

    if colors is None:
        colors = def_colors

    # Keep only selected objects

    ds = ds.sel(trajectory_number=(np.isin(ds_traj.object_label, select)))

    if load_ds:
        print("Loading data to memory.")
        ds = ds.load()
        print("Loaded.")

    if with_boxes:
        ds_object_bounds = get_bounding_boxes(ds, use_mask=True)

    if title is None:
        title = ""

    x_lim, y_lim, z_lim, Lx, Ly, Lz = domain_limits(
        ds_traj, x_lim=x_lim, y_lim=y_lim, z_lim=z_lim
    )

    fig, ax = init_figure(figsize, view_point, x_lim, y_lim, z_lim)

    if plot_field:

        (line_field,) = ax.plot3D(
            [], [], [], linestyle="", marker="o", markersize=field_size, color="k"
        )

    # Create empty line objects for each trajectory selected.

    nplt = 0
    if plot_class:

        line_list, class_range, nplt = create_class_lines(
            ax, ds.class_no, inobj_size, class_no["key"], colors
        )
        legend_title = "Class"

    else:

        line_list, box_list, nplt = create_obj_lines(
            ax, select, plot_mask, not_inobj_size, inobj_size, with_boxes, colors
        )
        legend_title = "Object Number"

    if legend:
        plt.legend(title=legend_title, loc="upper right", ncol=3)

    # animation function.  This is called sequentially
    def animate_trplt(itime):
        # print(f'Frame {i}')
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()

        ds_time = ds.isel(time=itime)
        ref_time = ds_time.ref_time.item()

        plot_time = ds_time.time.item()

        if plot_field:

            field_mask_at_time = mask_to_positions(field_mask.sel(time=plot_time))

            _update_field_plot(
                field_mask_at_time,
                itime,
                xlim,
                ylim,
                zlim,
                Lx,
                Ly,
                galilean,
                timestep,
                line_field,
            )

        nplt = 0

        if plot_class:

            traj = ds_time

            if not load_ds:
                traj.load()

            _update_class_plot(
                traj, itime, xlim, ylim, Lx, Ly, galilean, timestep, line_list[nplt]
            )

            nplt += 1

        else:

            for iobj, traj in ds_time.groupby("object_label"):

                if not np.isin(iobj, select):
                    continue

                if not load_ds:
                    traj.load()

                _update_obj_plot(
                    traj,
                    itime,
                    plot_mask,
                    xlim,
                    ylim,
                    Lx,
                    Ly,
                    galilean,
                    timestep,
                    line_list[nplt],
                )

                if with_boxes:
                    _update_box_plot(
                        ds_object_bounds,
                        itime,
                        Lx,
                        Ly,
                        iobj,
                        galilean,
                        timestep,
                        box_list[nplt],
                    )

                nplt += 1

        ax.set_title(
            f"{title}\nTime index {itime:03d} Time={plot_time} Ref={ref_time}."
        )

        return

    #    Writer = animation.writers['ffmpeg']
    #    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    anim = animation.FuncAnimation(
        fig, animate_trplt, frames=ntimes, interval=1000.0 / fps, blit=False
    )

    plt.show()

    if anim_name is not None:
        anim_type = anim_name.split(".")[-1]
        if anim_type == "gif":
            anim.save(anim_name, writer="imagemagick", fps=fps)
        elif anim_type == "mp4":
            anim.save(anim_name, fps=fps)

    # if save_anim : #, extra_args=['-vcodec', 'libx264'])
    return anim


def plot_family_animation(
    traj_family_list,
    obj_list,
    highlight_obj=None,
    field_mask=None,
    galilean=None,
    title=None,
    legend=False,
    figsize=(15, 12),
    not_inobj_size=0.2,
    inobj_size=2.0,
    field_size=0.5,
    fps=10,
    anim_name=None,
    plot_mask=False,
    with_boxes=False,
    load_ds=True,
    view_point=(0, 0),
    x_lim=None,
    y_lim=None,
    z_lim=None,
    colors=None,
):
    """
    Plot animated trajectory family points in 3D.

    Parameters
    ----------
    traj_family_list : list(xr.Dataset) or list(tuple(xr.Dataset, xr.Dataset))
        List of trajectory family datasets: if tuple, second item
        contains object box boundaries for box plotting.
    obj_list : list[(ref_time, object)]
        List of objects to plot
    field_mask : xarray DataArray, optional
        Eulerian field of bool with times matching ds_traj.
        The default is None.
    galilean : tuple (u,v), optional
        Velocity compoinents for Galilean Tranform. The default is None.
    title : str, optional
        Title of plot. Default is None.
    legend : bool, optional
        Plot legend. The default is False.
    figsize : tuple (width, height), optional
        Size of figure. The default is (20,12).
    not_inobj_size : float, optional
        Size of blob when obj_mask = False. The default is 0.2.
    inobj_size : float, optional
        Size of blob when obj_mask = True. The default is 2.0.
    field_size : float, optional
        Size of blob when field_mask = True.. The default is 0.5.
    fps : float or int, optional
        frames per second animation. The default is 10.
    anim_name : str, optional
        File name for animation. None generated if None. The default is None.
    plot_mask : bool, optional
        Plot points with inobj_size where obj_mask is True.
    with_boxes : bool, optional
        Display bounding boxes. The default is False.
    load_ds : bool, optional
        Pre-load data from file. The default is True.
    view_point : tuple (elevation, azimuth), optional
        3D plot view point. The default is (0,0).
    x_lim : list (min, max), optional
        x-axis limits. The default is None.
    y_lim : list (min, max), optional
        y-axis limits. The default is None.
    z_lim : list (min, max), optional
        z-axis limits. The default is None.
    colors: list(colors), optional
        colors for objects or classes. The default is None.
    Returns
    -------
    animation object.

    """
    if type(traj_family_list[0]) is tuple:
        traj_family = [t[0] for t in traj_family_list]
        traj_object_bounds = [t[1] for t in traj_family_list]
    else:
        traj_family = traj_family_list
        traj_object_bounds = None

    if load_ds:
        print("Loading data to memory.")
        for ds in traj_family:
            ds = ds.load()
        print("Loaded.")

    ref_times = family_coords(traj_family, "ref_time")

    if highlight_obj is None:
        highlight_obj = []

    if colors is None:
        colors = def_colors

    if title is None:
        title = ""

    master_ref_time = obj_list[0][0]

    master_ref = ref_times.index(master_ref_time)

    plot_field = field_mask is not None

    traj_master = traj_family[master_ref]

    if plot_mask and "obj_mask" not in traj_master.variables:
        print("Data does not contain obj_mask: plot_mask set to False.")
        plot_mask = False

    # Find times for whole plot.
    time_min = 1e100
    time_max = -1e100

    # for ds in traj_family:
    for (obj_time, objnum) in obj_list:
        obj_index = ref_times.index(obj_time)
        ds = traj_family[obj_index]
        time_min = min(time_min, ds.time.values.min())
        time_max = max(time_max, ds.time.values.max())

    timestep = ds.attrs["trajectory timestep"]

    ntimes = int(round((time_max - time_min) / timestep)) + 1

    plot_times = np.linspace(time_min, time_max, ntimes)

    # Set up figure and exes

    x_lim, y_lim, z_lim, Lx, Ly, Lz = domain_limits(
        traj_family[0], x_lim=x_lim, y_lim=y_lim, z_lim=z_lim
    )

    fig, ax = init_figure(figsize, view_point, x_lim, y_lim, z_lim)

    # Create empty line objects for field plot.

    if plot_field:

        (line_field,) = ax.plot3D(
            [], [], [], linestyle="", marker="o", markersize=field_size, color="k"
        )

    # Create empty line objects for each trajectory selected.

    lines, boxes, nplt = create_family_obj_lines(
        ax,
        obj_list,
        highlight_obj,
        plot_mask,
        not_inobj_size,
        inobj_size,
        with_boxes,
        colors,
        # ref_times_sel=ref_times_sel,
        # overlap_thresh=overlap_thresh,
    )

    legend_title = "Object (Ref Time: Number)"

    if legend:
        ncolmax = 6
        plt.legend(title=legend_title, loc="upper right", ncol=min(nplt, ncolmax))

    # animation function.  This is called sequentially
    def animate_trfplt(itime):
        # print(f'Frame {i}')
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()

        plot_time = plot_times[itime]

        if plot_field:

            field_mask_at_time = mask_to_positions(field_mask.sel(time=plot_time))

            _update_field_plot(
                field_mask_at_time,
                itime,
                xlim,
                ylim,
                zlim,
                Lx,
                Ly,
                galilean,
                timestep,
                line_field,
            )

        _update_family_obj_plot(
            traj_family,
            ref_times,
            plot_time,
            itime,
            plot_mask,
            xlim,
            ylim,
            Lx,
            Ly,
            galilean,
            timestep,
            lines,
        )
        if with_boxes:
            _update_family_box_plot(
                traj_object_bounds,
                ref_times,
                plot_time,
                itime,
                xlim,
                ylim,
                Lx,
                Ly,
                galilean,
                timestep,
                boxes,
            )

        ax.set_title(f"{title}\nTime index {itime:03d} Time={plot_time}.")

        return

    #    Writer = animation.writers['ffmpeg']
    #    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # anim = animate_trplt(0)

    anim = animation.FuncAnimation(
        fig, animate_trfplt, frames=ntimes, interval=1000.0 / fps, blit=False
    )

    plt.show()

    if anim_name is not None:
        anim_type = anim_name.split(".")[-1]
        if anim_type == "gif":
            anim.save(anim_name, writer="imagemagick", fps=fps)
        elif anim_type == "mp4":
            anim.save(anim_name, fps=fps)

    # if save_anim : #, extra_args=['-vcodec', 'libx264'])
    return anim


# ########### Functions below are support of overall plotting. ##########


def domain_limits(tr, x_lim=None, y_lim=None, z_lim=None):
    Lx = tr.attrs["Lx"]
    Ly = tr.attrs["Ly"]
    Lz = tr.attrs["Lz"]

    if x_lim is None:
        x_lim = [0, Lx]

    if y_lim is None:
        y_lim = [0, Ly]

    if z_lim is None:
        z_lim = [0, Lz]

    return x_lim, y_lim, z_lim, Lx, Ly, Lz


def init_figure(figsize, view_point, x_lim, y_lim, z_lim):
    fig = plt.figure(figsize=figsize)  # , tight_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    (elev, azim) = view_point
    ax.view_init(elev, azim)

    ax.set_xlim(x_lim[0], x_lim[1])
    rx = x_lim[1] - x_lim[0]

    ax.set_ylim(y_lim[0], y_lim[1])
    ry = y_lim[1] - y_lim[0]

    ax.set_zlim(z_lim[0], z_lim[1])
    rz = z_lim[1] - z_lim[0]

    ax.set_box_aspect((rx, ry, rz))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    return fig, ax


def create_class_lines(ax, class_no, inobj_size, key, colors):
    nplt = 0
    # Store animated lines in lists
    line_list = list([])
    class_min = class_no.min()
    class_max = class_no.max()
    class_range = np.arange(class_min, class_max + 1)

    line_for_class_list = list([])

    for iclass, class_no in enumerate(class_range):
        (line,) = ax.plot3D(
            [],
            [],
            [],
            linestyle="",
            marker="o",
            markersize=inobj_size,
            label=key[class_no],
            color=colors[iclass % len(colors)],
            # color = f'C{iclass:03d}',
        )
        line_for_class_list.append((line, class_no))
    line_list.append(line_for_class_list)
    nplt += 1
    return line_list, class_range, nplt


def create_obj_lines(
    ax, select, plot_mask, not_inobj_size, inobj_size, with_boxes, colors
):
    nplt = 0
    # Store animated lines in lists
    line_list = list([])
    if with_boxes:
        box_list = list([])
    else:
        box_list = None

    for iobj in select:
        if plot_mask:
            (line_out,) = ax.plot3D(
                [],
                [],
                [],
                linestyle="",
                marker="o",
                markersize=not_inobj_size,
                color=colors[nplt % len(colors)],
            )
            (line_in,) = ax.plot3D(
                [],
                [],
                [],
                linestyle="",
                marker="o",
                markersize=inobj_size,
                color=line_out.get_color(),
                label=f"{iobj}",
            )
            line_list.append([line_out, line_in])
        else:
            line = ax.plot3D(
                [],
                [],
                [],
                linestyle="",
                marker="o",
                markersize=not_inobj_size,
                color=colors[nplt % len(colors)],
                label=f"{iobj}",
            )
            line_list.append(line)

        if with_boxes:
            (box,) = ax.plot([], [], color=line_out.get_color())
            box_list.append(box)

        nplt += 1
    return line_list, box_list, nplt


def create_family_obj_lines(
    ax,
    obj_list,
    highlight_obj,
    plot_mask,
    not_inobj_size,
    inobj_size,
    with_boxes,
    colors,
):

    lines = {}
    nplt = 0
    # Store animated lines in lists
    if with_boxes:
        boxes = {}
    else:
        boxes = None

    for obj in obj_list:
        # print(f'Processing {obj=}')
        (match_time, objnum) = obj

        lab = f"{match_time}: {objnum}"
        if obj in highlight_obj:
            lab += "*"

        if plot_mask:
            # print('With mask')

            (line_out,) = ax.plot3D(
                [],
                [],
                [],
                linestyle="",
                marker="o",
                markersize=not_inobj_size,
                color=colors[nplt % len(colors)],
            )

            (line_in,) = ax.plot3D(
                [],
                [],
                [],
                linestyle="",
                marker="o",
                markersize=inobj_size,
                color=line_out.get_color(),
                label=lab,
            )

            lines[obj] = (line_out, line_in)
            if with_boxes:
                (box,) = ax.plot([], [], color=line_out.get_color())
                boxes[obj] = box
        else:
            # print('No mask')
            (line,) = ax.plot3D(
                [],
                [],
                [],
                linestyle="",
                marker="o",
                markersize=not_inobj_size,
                color=colors[nplt % len(colors)],
                label=lab,
            )
            lines[obj] = line
            if with_boxes:
                (box,) = ax.plot([], [], color=line.get_color())
                boxes[obj] = box
        nplt += 1
    # print(lines, boxes)
    return lines, boxes, nplt


def _get_xyz(f, i, xlim, ylim, Lx, Ly, galilean, timestep):
    x = f.x
    y = f.y
    z = f.z

    if galilean is not None:
        # print(f'Gal {i=}')
        x, y = gal_trans(x, y, galilean, i, timestep)

    x = conform_plot(x, Lx, xlim)
    y = conform_plot(y, Ly, ylim)

    return x, y, z


def _update_field_plot(
    field_mask, itime, xlim, ylim, zlim, Lx, Ly, galilean, timestep, line_field
):

    x, y, z = _get_xyz(field_mask, itime, xlim, ylim, Lx, Ly, galilean, timestep)

    x = x.values
    y = y.values
    z = z.values

    inx = np.logical_and(x >= xlim[0], x <= xlim[1])
    iny = np.logical_and(y >= ylim[0], y <= ylim[1])
    inz = np.logical_and(z >= zlim[0], z <= zlim[1])

    inplt = np.logical_and(inx, iny, inz)

    line_field.set_data(x[inplt], y[inplt])
    line_field.set_3d_properties(z[inplt])

    return


def _update_class_plot(traj, itime, xlim, ylim, Lx, Ly, galilean, timestep, line_list):
    x, y, z = _get_xyz(traj, itime, xlim, ylim, Lx, Ly, galilean, timestep)

    for (line, class_no) in line_list:
        in_cl = traj.class_no == class_no
        line.set_data(x[in_cl].values, y[in_cl].values)
        line.set_3d_properties(z[in_cl].values)

        # if list_class_numbers :
        #     print(class_key[iclass][0],len(np.where(in_cl)[0]))
    return


def _update_obj_plot(
    traj, itime, plot_mask, xlim, ylim, Lx, Ly, galilean, timestep, lines
):

    x, y, z = _get_xyz(traj, itime, xlim, ylim, Lx, Ly, galilean, timestep)

    if plot_mask:
        mask = traj.obj_mask
    else:
        mask = None

    _xyz_plot(x, y, z, lines, plot_mask, mask)

    return


def _xyz_plot(x, y, z, lines, plot_mask, mask, reset=False):
    if reset:
        if plot_mask:
            (line, line_cl) = lines
            line.set_data([], [])
            line.set_3d_properties([])
            line_cl.set_data([], [])
            line_cl.set_3d_properties([])
        else:
            (line) = lines
            line.set_data([], [])
            line.set_3d_properties([])
        return

    if plot_mask:
        in_obj = mask
        not_in_obj = ~mask

        (line, line_cl) = lines
        line.set_data(x[not_in_obj], y[not_in_obj])
        line.set_3d_properties(z[not_in_obj])
        line_cl.set_data(x[in_obj], y[in_obj])
        line_cl.set_3d_properties(z[in_obj])
    else:
        (line) = lines
        line.set_data(x, y)
        line.set_3d_properties(z)
    return


def _update_family_obj_plot(
    ds_list,
    # master_ref,
    ref_times,
    plot_time,
    itime,
    plot_mask,
    xlim,
    ylim,
    Lx,
    Ly,
    galilean,
    timestep,
    lines,
):

    # print(f"{itime=}")
    nplt = 0
    for obj, line in lines.items():
        # print(f'Plotting {obj=}')
        (match_time, objnum) = obj
        traj_ref = ds_list[ref_times.index(match_time)]
        if plot_time in traj_ref.time:
            # print(f"{match_time=}")
            traj_ref_at_time = traj_ref.sel(time=plot_time)

            # for obj, lines in matches.items():
            traj = traj_ref_at_time.where(
                traj_ref_at_time.object_label == objnum, drop=True
            )
            x, y, z = _get_xyz(traj, itime, xlim, ylim, Lx, Ly, galilean, timestep)
            if plot_mask:
                mask = traj.obj_mask >= 1
            else:
                mask = None
            # print(f'{len(x)}')
            _xyz_plot(x, y, z, line, plot_mask, mask)
        else:
            # print(f"No {match_time=}")
            # for obj, lines in matches.items():
            _xyz_plot([], [], [], line, plot_mask, [], reset=True)
            #
        nplt += 1

    return


def _box_plot(b, box, itime, Lx, Ly, galilean, timestep, reset=False):
    if reset:
        box.set_data([], [])
        box.set_3d_properties([])
        return

    x, y, z = box_xyz(b)

    if any(np.isnan(x)):
        x = y = z = []
    else:
        if galilean is not None:
            x, y = gal_trans(x, y, galilean, itime, timestep)

        # x = conform_plot(x, Lx, xlim)
        # y = conform_plot(y, Ly, ylim)

        while np.min(x) > Lx:
            x -= Lx
        while np.max(x) < 0:
            x += Lx
        while np.min(y) > Ly:
            y -= Ly
        while np.max(y) < 0:
            y += Ly

    box.set_data(x, y)
    box.set_3d_properties(z)
    return


def _update_box_plot(ds_object_bounds, itime, Lx, Ly, iobj, galilean, timestep, box):

    b = ds_object_bounds.isel(time=itime).sel(object_label=iobj)

    _box_plot(b, box, itime, Lx, Ly, galilean, timestep)

    return


def _update_family_box_plot(
    bb_list,
    # master_ref,
    ref_times,
    plot_time,
    itime,
    xlim,
    ylim,
    Lx,
    Ly,
    galilean,
    timestep,
    boxes,
):

    nplt = 0
    nplt = 0
    for obj, box in boxes.items():
        # print(f'Plotting {obj=}')
        (match_time, objnum) = obj
        # print(bb_list)
        # print(match_time)

        bb_ref = bb_list[ref_times.index(match_time)]
        # print(bb_ref)
        if plot_time in bb_ref.time:
            # print(f"{match_time=}")
            bb_ref_at_time = bb_ref.sel(time=plot_time)

            # for obj, box in matches.items():
            bb = bb_ref_at_time.sel(object_label=objnum)
            _box_plot(bb, box, itime, Lx, Ly, galilean, timestep)
        else:
            # print(f"No {match_time=}")
            _box_plot(None, box, itime, Lx, Ly, galilean, timestep, reset=True)
        nplt += 1
    return


def gal_trans(x, y, galilean, j, timestep):
    if galilean[0] != 0:
        x = x - galilean[0] * j * timestep
    if galilean[1] != 0:
        y = y - galilean[1] * j * timestep
    return x, y


def conform_plot(x, Lx, xlim):

    x = x % Lx
    if xlim[0] == 0 and xlim[1] == Lx:
        return x

    xmin = x.min()
    xmax = x.max()

    if xlim[0] < 0:
        if xlim[0] <= (xmax - Lx):
            x[x > xlim[1]] -= Lx
    if Lx < xlim[1]:
        if (xmin + Lx) <= xlim[1]:
            x[x < xlim[0]] += Lx

    return x
