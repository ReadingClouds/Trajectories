"""
Module trajectory_compute.

@author Peter Clark
"""
import os

from netCDF4 import Dataset
import numpy as np
from scipy import ndimage
from scipy.optimize import minimize
import numpy.random as rnd

from trajectories.grid_interpolation import (tri_lin_interp,
                                             multi_dim_lagrange_interp,
                                             fast_interp_3D)
from trajectories.object_tools import unsplit_objects
#import matplotlib.pyplot as plt

L_vap = 2.501E6
Cp = 1005.0
L_over_cp = L_vap / Cp
R_air = 287.058
r_over_cp = R_air/Cp
grav = 9.81
mol_wt_air        =           28.9640
mol_wt_water      =           18.0150
epsilon           = mol_wt_water/mol_wt_air
c_virtual         = 1.0/epsilon-1.0

var_properties = {"u":[True,False,False],\
                  "v":[False,True,False],\
                  "w":[False,False,True],\
                  "th":[False,False,False],\
                  "p":[False,False,False],\
                  "q_vapour":[False,False,False],\
                  "q_cloud_liquid_mass":[False,False,False],\
                  "tracer_rad1":[False,False,False],\
                  "tracer_rad2":[False,False,False],\
                  "tracer_rad3":[False,False,False],\
                  }


#debug_mean = True
debug_mean = False
#debug = True
debug = False

cyclic_xy = True


use_corrected_positions = False
#use_corrected_positions = True

ind=""

class Trajectories :
    """
    Class defining a set of back trajectories with a given reference time.

    This is an ordered list of trajectories with sequential reference times.

    Parameters
    ----------
    files              : ordered list of strings.
        Files used to generate trajectories
    ref_prof_file      : string
        name of file containing reference profile.
    start_time         : int
        Time for origin of back trajectories.
    ref                : int
        Reference time of trajectories.
    end_time           : int
        Time for end of forward trajectories.
    deltax             : float
        Model x grid spacing in m.
    deltay             : float
        Model y grid spacing in m.
    deltaz             : float
        Model z grid spacing in m.
    variable_list=None : List of strings.
        Variable names for data to interpolate to trajectory.
    ref_func           : function
        Return reference trajectory positions and labels.
    in_obj_func        : function
        Determine which points are inside an object.
    kwargs             : dict
        Any additional keyword arguments to ref_func.

    Attributes
    ----------
    refprof            : Dict
        Reference profile::

            'rho' (array)  : Reference profile of density.
            'p' (array)    : Reference profile of pressure.
            'th' (array)   : Reference profile of potential temperature.
            'pi' (array)   : Reference profile of Exner pressure.
    data               : float array [nt, m, n]
        Data associated with trajectory::

            nt is total number of times,
            m the number of trajectory points at a given time and
            n is the number of variables in variable_list.
    trajectory: Array [nt, m, 3] where the last index gives x,y,z
    traj_error: Array [nt, m, 3] with estimated error in trajectory.
    traj_times: Array [nt] with times corresponding to trajectory.
    labels: Array [m] labelling points with labels 0 to nobjects-1.
    nobjects: Number of objects.
    coords             : Dict
        Model grid info::

            'xcoord': grid xcoordinate of model space.
            'ycoord': grid ycoordinate of model space.
            'zcoord': grid zcoordinate of model space.
            'ycoord': grid ycoordinate of model space.
            'zcoord': grid zcoordinate of model space.
            'deltax': Model x grid spacing in m.
            'deltay': Model y grid spacing in m.
            'z'     : Model z grid m.
            'zn'    : Model z grid m.
    nx: int
        length of xcoord
    ny: int
        length of ycoord
    nz: int
        length of zcoord
    deltat: float
        time spacing in trajectories.
    ref_func: function
        to return reference trajectory positions and labels.
    in_obj_func: function
        to determine which points are inside an object.
    ref_func_kwargs: dict
        Any additional keyword arguments to ref_func (dict).
    files: list
        Input file list.
    ref: int
        Index of reference time in trajectory array.
    end: Index of end time in trajectory array. (start is always 0)
    ntimes: int
        Number of times in trajectory array.
    npoints: int
        Number of points in trajectory array.
    variable_list: dict
        variable_list corresponding to data.
    data_mean: numpy array
        mean of in_obj points data.
    num_in_obj: numpy array
        number of in_obj points.
    centroid: numpy array
        centroid of in_objy points
    bounding_box: numpy array
        box containing all trajectory points.
    in_obj_box: numpy array
        box containing all in_obj trajectory points.
    max_at_ref: list(int)
        list of objects which reach maximum LWC at reference time.

    """

    def __init__(self, files, ref_prof_file, start_time, ref, end_time,
                 deltax, deltay, deltaz, ref_func, in_obj_func,
                 kwargs={}, variable_list=None,  interp_method = "tri_lin",
                 interp_order = 1, unsplit = True) :
        """
        Create an instance of a set of trajectories with a given reference.

        This is an ordered list of trajectories with sequential reference times.

        @author: Peter Clark

        """
        if variable_list == None :
            variable_list = { \
                  "u":r"$u$ m s$^{-1}$", \
                  "v":r"$v$ m s$^{-1}$", \
                  "w":r"$w$ m s$^{-1}$", \
                  "th":r"$\theta$ K", \
                  "p":r"Pa", \
                  "q_vapour":r"$q_{v}$ kg/kg", \
                  "q_cloud_liquid_mass":r"$q_{cl}$ kg/kg", \
                  }

        dataset_ref = Dataset(ref_prof_file)

        rhoref = dataset_ref.variables['rhon'][-1,...]
        pref = dataset_ref.variables['prefn'][-1,...]
        thref = dataset_ref.variables['thref'][-1,...]
        piref = (pref[:]/1.0E5)**r_over_cp

        self.refprof = {'rho': rhoref,
                        'p'  : pref,
                        'pi'  : piref,
                        'th': thref}

        self.data, self.trajectory, self.traj_error, self.times, self.ref, \
        self.labels, self.nobjects, self.coords, self.deltat = \
        compute_trajectories(files, start_time, ref, end_time,
                             deltax, deltay, deltaz, variable_list.keys(),
                             self.refprof, ref_func, kwargs=kwargs,
                             interp_method = interp_method,
                             interp_order = interp_order,
                             )
        self.interp_method = interp_method
        self.interp_order = interp_order
        self.ref_func=ref_func
        self.in_obj_func=in_obj_func
        self.ref_func_kwargs=kwargs
        self.files = files
#        self.ref   = (ref-start_time)//self.deltat
        self.end = len(self.times)-1
        self.ntimes = np.shape(self.trajectory)[0]
        self.npoints = np.shape(self.trajectory)[1]
        self.nx = np.size(self.coords['xcoord'])
        self.ny = np.size(self.coords['ycoord'])
        self.nz = np.size(self.coords['zcoord'])
        self.variable_list = variable_list
        if unsplit:
            self.trajectory = unsplit_objects(self.trajectory, self.labels,
                                          self.nobjects, self.nx, self.ny)

        self.data_mean, self.in_obj_data_mean, self.objvar_mean, \
            self.num_in_obj, \
            self.centroid, self.in_obj_centroid, self.bounding_box, \
            self.in_obj_box = compute_traj_boxes(self, in_obj_func,
                                                kwargs=kwargs)

        max_objvar = (self.objvar_mean == np.max(self.objvar_mean, axis=0))
        when_max_objvar = np.where(max_objvar)
        self.max_at_ref = when_max_objvar[1][when_max_objvar[0] == self.ref]
        return

    def var(self, v) :
        """
        Convert variable name to numerical pointer.

        Args:
            v (string):  variable name.

        Returns
        -------
            Numerical pointer to data in data array.

        @author: Peter Clark

        """
        i = list(self.variable_list.keys()).index(v)

        for ii, vr in enumerate(list(self.variable_list)) :
            if vr==v : break
        if ii != i : print("var issue: ", self.variable_list.keys(), v, i, ii)
        return ii

    def select_object(self, iobj) :
        """
        Find trajectory and associated data corresponding to iobj.

        Args:
            iobj(integer) : object id .

        Returns
        -------
            trajectory_array, associated _data.

        @author: Peter Clark

        """
        in_object = (self.labels == iobj)
        obj = self.trajectory[:, in_object, ...]
        dat = self.data[:, in_object, ...]
        return obj, dat

    def __str__(self):
        """
        Generate string representation.

        Returns
        -------
        rep : str

        """
        rep = "Trajectories centred on reference Time : {}\n".\
        format(self.times[self.ref])
        rep += "Times : {}\n".format(self.ntimes)
        rep += "Points : {}\n".format(self.npoints)
        rep += "Objects : {}\n".format(self.nobjects)
        return rep

    def __repr__(self):
        """
        Generate string retresentation.

        Returns
        -------
        rep : str

        """
        rep = "Trajectory Reference time: {0}, Times:{1}, Points:{2}, Objects:{3}\n".format(\
          self.times[self.ref],self.ntimes,self.npoints, self.nobjects)
        return rep


def dict_to_index(variable_list, v) :
    """
    Convert variable name to numerical pointer.

    Args:
        v (string):  variable name.

    Returns
    -------
        Numerical pointer to v in variable list.

    @author: Peter Clark

    """
    ii = list(variable_list.keys()).index(v)

#    for ii, vr in enumerate(list(self.variable_list)) :
#        if vr==v : break
    return ii

def compute_trajectories(files,
                         start_time, ref_time, end_time,
                         deltax, deltay, deltaz,
                         variable_list,
                         refprof, ref_func, kwargs={},
                         interp_method = "tri_lin",
                         interp_order = 1
                         ) :
    """
    Compute forward and back trajectories plus associated data.

    Args:
        files         : Ordered list of netcdf files containing 3D MONC output.
        start_time    : Time corresponding to end of back trajectory.
        ref_time      : Time at which reference objects are defined.
        end_time      : Time corresponding to end of forward trajectory.
        variable_list : List of variables to interpolate to trajectory points.
        refprof        : ref profile.

    Returns
    -------
        Set of variables defining trajectories::

            data_val: Array [nt, m, n] where nt is total number of times,
                m the number of trajectory points at a given time and
                n is the number of variables in variable_list.
            trajectory: Array [nt, m, 3] where the last index gives x,y,z
            traj_error: Array [nt, m, 3] with estimated error in trajectory.
            traj_times: Array [nt] with times corresponding to trajectory.
            labels: Array [m] labelling points with labels 0 to nobjects-1.
            nobjects: Number of objects.
            coords             : Dict containing model grid info.
                'xcoord': grid xcoordinate of model space.
                'ycoord': grid ycoordinate of model space.
                'zcoord': grid zcoordinate of model space.
                'ycoord': grid ycoordinate of model space.
                'zcoord': grid zcoordinate of model space.
                'deltax': Model x grid spacing in m.
                'deltay': Model y grid spacing in m.
                'z'     : Model z grid m.
                'zn'    : Model z grid m.
            deltat: time spacing in trajectories.

    @author: Peter Clark

    """
    print('Computing trajectories from {} to {} with reference {}.'.\
          format(start_time, end_time, ref_time))

    ref_file_number, ref_time_index, delta_t = find_time_in_files(\
                                                        files, ref_time)

    dataset=Dataset(files[ref_file_number])
    print("Dataset opened ", files[ref_file_number])
#    print(dataset)
    theta = dataset.variables["th"]
    ref_times = dataset.variables[theta.dimensions[0]][...]
    print('Starting in file number {}, name {}, index {} at time {}.'.\
          format(ref_file_number, os.path.basename(files[ref_file_number]), \
                 ref_time_index, ref_times[ ref_time_index] ))
    file_number = ref_file_number
    time_index = ref_time_index

    # Find initial positions and labels using user-defined function.
    traj_pos, labels, nobjects = ref_func(dataset, time_index, **kwargs)

#    print(traj_pos)

    times = ref_times
    trajectory, data_val, traj_error, traj_times, coords \
      = trajectory_init(dataset, time_index, variable_list,
                        deltax, deltay, deltaz, refprof, traj_pos,
                        interp_method = interp_method,
                        interp_order = interp_order,
                        )
    ref_index = 0

    print("Computing backward trajectories.")

    while (traj_times[0] > start_time) and (file_number >= 0) :
        time_index -= 1
        if time_index >= 0 :
            print('Time index: {} File: {}'.format(time_index, \
                   os.path.basename(files[file_number])))
            trajectory, data_val, traj_error, traj_times = \
            back_trajectory_step(dataset, time_index, variable_list, refprof,
                  coords, trajectory, data_val, traj_error, traj_times,
                  interp_method = interp_method,
                  interp_order = interp_order,
                  )
            ref_index += 1
        else :
            file_number -= 1
            if file_number < 0 :
                print('Ran out of data.')
            else :
                dataset.close()
                print("dataset closed")
                print('File {} {}'.format(file_number, \
                      os.path.basename(files[file_number])))
                dataset = Dataset(files[file_number])
                print("Dataset opened ", files[file_number])
                theta = dataset.variables["th"]
                times = dataset.variables[theta.dimensions[0]][...]
                time_index = len(times)
    dataset.close()
    print("dataset closed")
# Back to reference time for forward trajectories.
    file_number = ref_file_number
    time_index = ref_time_index
    times = ref_times
    dataset = Dataset(files[ref_file_number])
    print("Dataset opened ", files[ref_file_number])
    print("Computing forward trajectories.")

    while (traj_times[-1] < end_time) and (file_number >= 0) :
        time_index += 1
        if time_index < len(times) :
            print('Time index: {} File: {}'.format(time_index, \
                   os.path.basename(files[file_number])))
            trajectory, data_val, traj_error, traj_times = \
                forward_trajectory_step(dataset, time_index,
                                 variable_list, refprof,
                                 coords, trajectory, data_val, traj_error,
                                 traj_times,
                                 interp_method = interp_method,
                                 interp_order = interp_order,
                                 vertical_boundary_option=2)
        else :
            file_number += 1
            if file_number == len(files) :
                print('Ran out of data.')
            else :
                dataset.close()
                print("dataset closed")
                print('File {} {}'.format(file_number, \
                      os.path.basename(files[file_number])))
                dataset = Dataset(files[file_number])
                print("Dataset opened ", files[file_number])
                theta = dataset.variables["th"]
                times = dataset.variables[theta.dimensions[0]][...]
                time_index = -1
    dataset.close()
    print("dataset closed")

    print('data_val: {} {} {}'.format( len(data_val), len(data_val[0]), \
          np.size(data_val[0][0]) ) )

    data_val = np.reshape(np.vstack(data_val), \
               (len(data_val), len(data_val[0]), np.size(data_val[0][0]) ) )

    print('trajectory: {} {} {}'.format(len(trajectory[1:]), len(trajectory[0]), \
                               np.size(trajectory[0][0])))
    print('traj_error: {} {} {}'.format(len(traj_error[1:]), len(traj_error[0]), \
                               np.size(traj_error[0][0])))
#    print np.shape()

    trajectory = np.reshape(np.vstack(trajectory[1:]), \
               ( len(trajectory[1:]), len(trajectory[0]), \
                 np.size(trajectory[0][0]) ) )

    traj_error = np.reshape(np.vstack(traj_error[1:]), \
               ( len(traj_error[1:]), len(traj_error[0]), \
                 np.size(traj_error[0][0]) ) )

    traj_times = np.reshape(np.vstack(traj_times),-1)

    return data_val, trajectory, traj_error, traj_times, ref_index, \
      labels, nobjects, coords, delta_t

def extract_pos(nx, ny, dat) :
    """
    Extract 3D position from data array.

    Args:
        nx        : Number of points in x direction.
        ny        : Number of points in y direction.
        dat       : Array[m,n] where n>=5 if cyclic_xy or 3 if not.

    Returns
    -------
        pos       : Array[m,3]
        n_pvar    : Number of dimensions in input data used for pos.
    """
    global cyclic_xy
    if cyclic_xy :
        n_pvar = 5
        xpos = phase(dat[0],dat[1],nx)
        ypos = phase(dat[2],dat[3],ny)
        pos = np.array([xpos, ypos, dat[4]]).T
    else :
        n_pvar = 3
        pos = np.array([dat[0], dat[1], dat[2]]).T

    return pos, n_pvar

def get_z_zn(dataset, default_z):
    if 'zn' in dataset.variables:
        zn = dataset.variables['zn'][...]
        if 'z' in dataset.variables:
            z = dataset.variables['z'][...]
        else:
            z = np.zeros_like(zn)
            z[:-1] = 0.5 * ( zn[:-1] + zn[1:])
            z[-1]  = 2 * zn[-1] - z[-2]
    else:
        if 'z' in dataset.variables:
            z = dataset.variables['z'][...]
        else:
            print('No z-grid info available in file. Using supplied deltaz')
            z = default_z
        zn = np.zeros_like(z)
        zn[-1] = 0.5 * (z[-1] + z[-2])
        for i in range(len(z)-2,  -1, -1 ):
            zn[i] = 2 * z[i] - zn[i+1]
    return z, zn

def trajectory_init(dataset, time_index, variable_list,
                    deltax, deltay, deltaz,
                    refprof, traj_pos,
                    interp_method = "tri_lin",
                    interp_order = 1
                    ) :
    """
    Set up origin of back and forward trajectories.

    Args:
        dataset       : Netcdf file handle.
        time_index    : Index of required time in file.
        variable_list : List of variable names.
        refprof       : Dict with reference theta profile arrays.
        traj_pos      : array[n,3] of initial 3D positions.

    Returns
    -------
        Trajectory variables::

            trajectory     : position of origin point.
            data_val       : associated data so far.
            traj_error     : estimated trajectory errors so far.
            traj_times     : trajectory times so far.
            coords             : Dict containing model grid info.
                'xcoord': grid xcoordinate of model space.
                'ycoord': grid ycoordinate of model space.
                'zcoord': grid zcoordinate of model space.
                'deltax': Model x grid spacing in m.
                'deltay': Model y grid spacing in m.
                'z'     : Model z grid m.
                'zn'    : Model z grid m.

    @author: Peter Clark

    """
    th = dataset.variables['th'][0,...]
    (nx, ny, nz) = np.shape(th)

    xcoord = np.arange(nx ,dtype='float')
    ycoord = np.arange(ny, dtype='float')
    zcoord = np.arange(nz, dtype='float')

    z, zn = get_z_zn(dataset, deltaz * zcoord)

    coords = {
        'xcoord': xcoord,
        'ycoord': ycoord,
        'zcoord': zcoord,
        'deltax': deltax,
        'deltay': deltay,
        'z'     : z,
        'zn'    : zn,
        }

    data_list, varlist, varp_list, time = load_traj_step_data(dataset,
                                                   time_index, variable_list,
                                                   refprof, coords )
    print("Starting at time {}".format(time))

    out = data_to_pos(data_list, varp_list, traj_pos,
                      xcoord, ycoord, zcoord,
                      interp_method = interp_method,
                      interp_order = interp_order,
                      )
    traj_pos_new, n_pvar = extract_pos(nx, ny, out)

    data_val = list([np.vstack(out[n_pvar:]).T])

    if debug :

        print(np.shape(data_val))
        print(np.shape(traj_pos))
        print('xorg',traj_pos[:,0])
        print('yorg',traj_pos[:,1])
        print('zorg',traj_pos[:,2])
        print('x',traj_pos_new[:,0])
        print('y',traj_pos_new[:,1])
        print('z',traj_pos_new[:,2])

    trajectory = list([traj_pos])
    traj_error = list([np.zeros_like(traj_pos)])
    trajectory.insert(0,traj_pos_new)
    traj_error.insert(0,np.zeros_like(traj_pos_new))
    traj_times = list([time])

    return trajectory, data_val, traj_error, traj_times, coords

def back_trajectory_step(dataset, time_index, variable_list, refprof, \
                         coords, trajectory, data_val, traj_error, traj_times,
                         interp_method = "tri_lin",
                         interp_order = 1
                         ) :
    """
    Execute backward timestep of set of trajectories.

    Args:
        dataset        : netcdf file handle.
        time_index     : time index in netcdf file.
        variable_list  : list of variable names.
        refprof        : Dict with reference theta profile arrays.
        coords         : Dict containing 1D arrays giving coordinate spaces of data.
        trajectory     : trajectories so far. trajectory[0] is position of earliest point.
        data_val       : associated data so far.
        traj_error     : estimated trajectory errors to far.
        traj_times     : trajectory times so far.

    Returns
    -------
        Inputs updated to new location::

            trajectory, data_val, traj_error, traj_times

    @author: Peter Clark

    """
    data_list, varlist, varp_list, time = load_traj_step_data(dataset,
                                          time_index, variable_list, refprof,
                                          coords)
    print("Processing data at time {}".format(time))

    (nx, ny, nz) = np.shape(data_list[0])

    traj_pos = trajectory[0]
#    print "traj_pos ", np.shape(traj_pos), traj_pos[:,0:5]

    out = data_to_pos(data_list, varp_list, traj_pos,
                      coords['xcoord'], coords['ycoord'], coords['zcoord'],
                      interp_method = interp_method,
                      interp_order = interp_order,
                      )
    traj_pos_new, n_pvar = extract_pos(nx, ny, out)

    data_val.insert(0, np.vstack(out[n_pvar:]).T)
    trajectory.insert(0, traj_pos_new)
    traj_error.insert(0, np.zeros_like(traj_pos_new))
    traj_times.insert(0, time)

    return trajectory, data_val, traj_error, traj_times

def confine_traj_bounds(pos, nx, ny, nz, vertical_boundary_option=1):
    """
    Confine trajectory position to domain.

    Parameters
    ----------
    pos : numpy array
        trajectory positions (n, 3).
    nx : int or float
        max value of x.
    ny : int or float
        max value of y.
    nz :  int or float
        max value of z.
    vertical_boundary_option : int, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    pos : TYPE
        DESCRIPTION.

    """
    pos[:,0] = pos[:,0] % nx
    pos[:,1] = pos[:,1] % ny

    if vertical_boundary_option == 1:

        pos[:, 2] = np.clip(pos[:, 2], 0, nz)

    elif vertical_boundary_option == 2:

        lam = 1.0 / 0.5
        k1 = np.where(pos[:,2] <=   1.0 )
        k2 = np.where(pos[:,2] >= (nz-1))
        pos[:,2][k1]  = 1.0 + rnd.exponential(scale=lam, size = np.shape(k1))
        pos[:,2][k2]  = nz - (1.0
                            + rnd.exponential(scale=lam, size = np.shape(k2)))

    return pos

def forward_trajectory_step(dataset, time_index, variable_list, refprof,
                            coords,
                            trajectory, data_val, traj_error, traj_times,
                            vertical_boundary_option=1,
                            interp_method = "tri_lin",
                            interp_order = 1
                            ) :
    """
    Execute forward timestep of set of trajectories.

    Args:
        dataset        : netcdf file handle.
        time_index     : time index in netcdf file.
        variable_list  : list of variable names.
        refprof        : Dict with reference theta profile arrays.
        coords         : Dict containing 1D arrays giving coordinate spaces of data.
        trajectory     : trajectories so far. trajectory[-1] is position of latest point.
        data_val       : associated data so far.
        traj_error     : estimated trajectory errors to far.
        traj_times     : trajectory times so far.

    Returns
    -------
        Inputs updated to new location::

            trajectory, data_val, traj_error, traj_times

    @author: Peter Clark

    """
    global cyclic_xy
    def print_info(kk):
        print('kk = {} Error norm :{} Error :{}'.format(kk, mag_diff[kk], \
              diff[:,kk]))
        print('Looking for at this index ', \
              [traj_pos[:,m][kk] for m in range(3)])
        print('Nearest solution for the index', \
              [traj_pos_at_est[kk,m] for m in range(3)])
        print('Located at ', \
              [traj_pos_next_est[kk,m] for m in range(3)])
        return

    def int3d_dist(pos) :
#    global dist_data
        if np.size(np.shape(pos)) == 1 :
            pos = np.atleast_2d(pos).T

        out = np.max([0,ndimage.map_coordinates(dist_data, pos, \
                                                mode='nearest', order=2)])
#   out = ndimage.map_coordinates(dist_data, pos, mode='nearest', order=2)
#   print ('out = {}'.format(out))
        return out
    if cyclic_xy :
        n_pvar = 5
    else :
        n_pvar = 3

    data_list, varlist, varp_list, time = load_traj_step_data(dataset,
                                          time_index, variable_list, refprof,
                                          coords)
    print("Processing data at time {}".format(time))

    (nx, ny, nz) = np.shape(data_list[0])

    # traj_pos_next_est is our estimate of the trajectory positions at
    # the next time step.
    # We want the 'where from' at traj_pos_next_est to match
    # the current trajectory positions, traj_pos

    # So, we are solving f(x) = 0
    # where
    # x = traj_pos_next_est
    # f = traj_pos_at_est - traj_pos

    traj_pos = trajectory[-1]

    # First guess - extrapolate from last two positions.

    traj_pos_next_est = 2*trajectory[-1]-trajectory[-2]

    confine_traj_bounds(traj_pos_next_est, nx, ny, nz,
                            vertical_boundary_option=vertical_boundary_option)


#    norm = 'max_abs_error'
    norm = 'mean_abs_error'
    use_point_iteration = True
    limit_gradient = True
    max_iter = 100
    errtol_iter = 0.05
    errtol = 0.05
    relax_param = 0.75

    err = 1000.0
    niter = 0
    not_converged = True
    correction_cycle = False
    while not_converged :

        out = data_to_pos(data_list, varp_list, traj_pos_next_est,
                          coords['xcoord'], coords['ycoord'], coords['zcoord'],
                          interp_method = interp_method,
                          interp_order = interp_order,
                          maxindex = n_pvar)

        traj_pos_at_est, n_pvar = extract_pos(nx, ny, out)

        diff = traj_pos_at_est - traj_pos

# Deal with wrap around.
        diff_lt_minus_nx = diff[:,0]<(-nx/2)
        traj_pos_at_est[:,0][diff_lt_minus_nx] += nx
        diff[:,0][diff_lt_minus_nx] += nx

        diff_gt_plus_nx = diff[:,0]>=(nx/2)
        traj_pos_at_est[:,0][diff_gt_plus_nx] -= nx
        diff[:,0][diff_gt_plus_nx] -= nx

        diff_lt_minus_ny = diff[:,1]<(-ny/2)
        traj_pos_at_est[:,1][diff_lt_minus_ny] += ny
        diff[:,1][diff_lt_minus_ny] += ny

        diff_gt_plus_ny = diff[:,1]>=(ny/2)
        traj_pos_at_est[:,1][diff_gt_plus_ny] -= ny
        diff[:,1][diff_gt_plus_ny] -= ny

        mag_diff = 0
        for i in range(3):
#            print(np.histogram(diff[:,i]))
#            print(niter, np.max(np.abs(diff[:,i])))
            mag_diff += diff[:,i]**2

#        err = np.amax(mag_diff)
        err_prev = err
        if norm == 'mean_abs_error':
           err = np.mean(np.abs(diff))
        if norm == 'max_abs_error':
            err = np.max(np.abs(diff))
        else:
            err = np.sqrt(np.mean(diff*diff))
#        print(niter, err, np.sum(np.abs(diff)>errtol_iter/10.), \
#                          np.sum(np.abs(diff)>errtol_iter))
#        print(f"niter: {niter:3d} dist: {np.max(np.abs(diff), axis=0)} err: {err} relax_param: {relax_param}")


        if correction_cycle :
            print('After correction cycle {}'.format(err))
            if debug :
                try:
                    k
                    for kk in k :
                        print_info(kk)

                except NameError:
                    print('No k')
            break

#        if err <= errtol_iter or err >= err_prev :
        if err <= errtol_iter :
            not_converged = False
            print("Converged in {} iterations with error {}."\
                  .format(niter, err))
#        elif niter > 10 and err > err_prev : relax_param *= 0.9
        elif niter%10==0: relax_param *= 0.9


        if niter <= max_iter :
#            traj_pos_prev_est = traj_pos_next_est
#            traj_pos_at_prev_est = traj_pos_at_est

            if use_point_iteration or niter == 0 :
                if niter == 0:
                    traj_pos_next_est_prev = traj_pos_next_est.copy()
                    diff_prev = diff.copy()

                increment = np.clip(diff * relax_param, -1, 1)
                traj_pos_next_est = traj_pos_next_est - increment

            else :

                # Attempt at 'Newton-Raphson' with numerical gradients, or
                # Secant method.
                # At present, this does not converge.

                df = (diff - diff_prev)
                inv_gradient = np.ones_like(df) * relax_param
                use_grad = np.abs(df) > 0.001
                dx = traj_pos_next_est - traj_pos_next_est_prev
                # Deal with wrap around.
                dx_lt_minus_nx = dx[:,0]<(-nx/2)
                dx[:,0][dx_lt_minus_nx] += nx

                dx_gt_plus_nx = dx[:,0]>=(nx/2)
                dx[:,0][dx_gt_plus_nx] -= nx

                dx_lt_minus_ny = dx[:,1]<(-ny/2)
                dx[:,1][dx_lt_minus_ny] += ny

                dx_gt_plus_ny = dx[:,1]>=(ny/2)
                dx[:,1][dx_gt_plus_ny] -= ny

                inv_gradient[use_grad] = np.clip(dx[use_grad] / df[use_grad],
                                         -1, 1)
                print("inv_grad :", np.max(np.abs(inv_gradient)))
                traj_pos_next_est_prev = traj_pos_next_est.copy()
                diff_prev = diff.copy()
                increment = np.clip(diff * inv_gradient, -1, 1)
                traj_pos_next_est = traj_pos_next_est - increment


            niter +=1
        else :
            print('Iterations exceeding {} {}'.format(max_iter, err))
            if err <= errtol or not correction_cycle:
                not_converged = False
            else:
                bigerr = (mag_diff > errtol)
                if np.size(diff[bigerr,:]) > 0 :
                    k = np.where(bigerr)[0]
                    if debug :
                        print('Index list into traj_pos_at_est is {}'.format(k))
                    for kk in k :
                        if debug :
                            print_info(kk)
                        lx = np.int(np.round(traj_pos_next_est[kk,0]))
                        ly = np.int(np.round(traj_pos_next_est[kk,1]))
                        lz = np.int(np.round(traj_pos_next_est[kk,2]))
                        if debug :
                            print('Index in source array is {},{},{}'.\
                                  format(lx,ly,lz))

                        nd = 5

                        xr = np.arange(lx-nd,lx+nd+1, dtype=np.intp) % nx
                        yr = np.arange(ly-nd,ly+nd+1, dtype=np.intp) % ny
                        zr = np.arange(lz-nd,lz+nd+1, dtype=np.intp)

                        zr[zr >= nz] = nz-1
                        zr[zr <  0 ] = 0

                        x_real = data_list[0]
                        x_imag = data_list[1]
                        y_real = data_list[2]
                        y_imag = data_list[3]
                        z_dat   = data_list[4]

                        xpos = phase(x_real[np.ix_(xr,yr,zr)],\
                                     x_imag[np.ix_(xr,yr,zr)],nx)
                        ypos = phase(y_real[np.ix_(xr,yr,zr)],\
                                     y_imag[np.ix_(xr,yr,zr)],ny)
                        zpos = z_dat[np.ix_(xr,yr,zr)]

                        dx = xpos - traj_pos[:,0][kk]
                        dy = ypos - traj_pos[:,1][kk]
                        dz = zpos - traj_pos[:,2][kk]
                        dx[dx >= (nx/2)] -= nx
                        dx[dx < (-nx/2)] += nx
                        dy[dy >= (ny/2)] -= ny
                        dy[dy < (-ny/2)] += ny

                        dist = dx**2 + dy**2 +dz**2

                        lp = np.where(dist == np.min(dist))

                        lxm = lp[0][0]
                        lym = lp[1][0]
                        lzm = lp[2][0]

                        if debug :
                            print('Nearest point is ({},{},{},{})'.\
                                  format(lxm,lym,lzm, dist[lxm,lym,lzm]))
                            print(xpos[lxm,lym,lzm],ypos[lxm,lym,lzm],zpos[lxm,lym,lzm])
                        envsize = 2
                        nx1=lxm-envsize
                        nx2=lxm+envsize+1
                        ny1=lym-envsize
                        ny2=lym+envsize+1
                        nz1=lzm-envsize
                        nz2=lzm+envsize+1

                        dist_data = dist[nx1:nx2,ny1:ny2,nz1:nz2]
                        x0 = np.array([envsize,envsize,envsize],dtype='float')
                        res = minimize(int3d_dist, x0, method='BFGS')
                        #,options={'xtol': 1e-8, 'disp': True})
                        if debug :
                            print('New estimate, relative = {} {} '.\
                                  format(res.x, res.fun))
                        newx = lx-nd+lxm-envsize+res.x[0]
                        newy = ly-nd+lym-envsize+res.x[1]
                        newz = lz-nd+lzm-envsize+res.x[2]
                        if debug :
                            print('New estimate, absolute = {}'.\
                                  format(np.array([newx,newy,newz])))
                        traj_pos_next_est[kk,:] = np.array([newx,newy,newz])
                    # end kk loop
                #  correction
            # correction_cycle = True
        confine_traj_bounds(traj_pos_next_est, nx, ny, nz,
                            vertical_boundary_option=vertical_boundary_option)
    out = data_to_pos(data_list, varp_list, traj_pos_next_est,
                      coords['xcoord'], coords['ycoord'], coords['zcoord'],
                      interp_method = interp_method,
                      interp_order = interp_order,
                      )
    confine_traj_bounds(traj_pos_next_est, nx, ny, nz,
                            vertical_boundary_option=vertical_boundary_option)

    data_val.append(np.vstack(out[n_pvar:]).T)
    trajectory.append(traj_pos_next_est)
    traj_error.append(diff)
    traj_times.append(time)
#    print 'trajectory:',len(trajectory[:-1]), len(trajectory[0]), np.size(trajectory[0][0])
#    print 'traj_error:',len(traj_error[:-1]), len(traj_error[0]), np.size(traj_error[0][0])
    return trajectory, data_val, traj_error, traj_times


def data_to_pos(data, varp_list, pos,
                xcoord, ycoord, zcoord,
                interp_method = "tri_lin",
                interp_order = 1,
                maxindex=None):
    """
    Interpolate data to pos.

    Args:
        data      : list of data array.
        varp_list: list of grid info lists.
        pos       : array[n,3] of n 3D positions.
        xcoord,ycoord,zcoord: 1D arrays giving coordinate spaces of data.

    Returns
    -------
        list of arrays containing interpolated data.

    @author: Peter Clark

    """
    if interp_method ==  "tri_lin":
        if interp_order != 1:
            raise ValueError(
              f"Variable interp_order must be set to 1 for tri_lin_interp; "
              f"currently {interp_order}.")
        output = tri_lin_interp(data, varp_list, pos,
                                xcoord, ycoord, zcoord,
                                maxindex=maxindex )
    elif interp_method ==  "grid_interp":
        output = multi_dim_lagrange_interp(data, pos,
                                           order = interp_order,
                                           wrap = [True, True, False])
    elif interp_method ==  "fast_interp":
        output = fast_interp_3D(data, pos,
                                order = interp_order,
                                wrap = [True, True, False])
    else:
        output= list([])
        for l in range(len(data)) :
#            print 'Calling map_coordinates'
#            print np.shape(data[l]), np.shape(traj_pos)
            out = ndimage.map_coordinates(data[l], pos, mode='wrap', \
                                          order=interp_order)
            output.append(out)
    return output

def load_traj_pos_data(dataset, it) :
    """
    Read trajectory position variables from file.

    Args:
        dataset        : netcdf file handle.
        it             : time index in netcdf file.

    Returns
    -------
        List of arrays containing interpolated data.

    @author: Peter Clark

    """
    if 'CA_xrtraj' in dataset.variables.keys() :
        # Circle-A Version
        trv = {'xr':'CA_xrtraj', \
               'xi':'CA_xitraj', \
               'yr':'CA_yrtraj', \
               'yi':'CA_yitraj', \
               'zpos':'CA_ztraj' }
        trv_noncyc = {'xpos':'CA_xtraj', \
                      'ypos':'CA_ytraj', \
                      'zpos':'CA_ztraj' }
    else :
        # Stand-alone Version
        trv = {'xr':'tracer_traj_xr', \
               'xi':'tracer_traj_xi', \
               'yr':'tracer_traj_yr', \
               'yi':'tracer_traj_yi', \
               'zpos':'tracer_traj_zr' }
        trv_noncyc = {'xpos':'tracer_traj_xr', \
                      'ypos':'tracer_traj_yr', \
                      'zpos':'tracer_traj_zr' }


    if cyclic_xy :
        xr = dataset.variables[trv['xr']][it,...]
        xi = dataset.variables[trv['xi']][it,...]

        yr = dataset.variables[trv['yr']][it,...]
        yi = dataset.variables[trv['yi']][it,...]

        zpos = dataset.variables[trv['zpos']]
        zposd = zpos[it,...]
        data_list = [xr, xi, yr, yi, zposd]
        varlist = ['xr', 'xi', 'yr', 'yi', 'zpos']
        varp_list = [var_properties['th']]*5

    else :
        # Non-cyclic option may well not work anymore!
        xpos = dataset.variables[trv_noncyc['xpos']][it,...]
        ypos = dataset.variables[trv_noncyc['ypos']][it,...]
        zpos = dataset.variables[trv_noncyc['zpos']]
        zposd = zpos[it,...]
        data_list = [xpos, ypos, zposd]
        varlist = ['xpos', 'ypos', 'zpos']
        varp_list = [var_properties['th']]*3

# Needed as zpos above is numpy array not NetCDF variable.
#    zpos = dataset.variables['CA_ztraj']
    times  = dataset.variables[zpos.dimensions[0]]

    return data_list, varlist, varp_list, times[it]


def d_by_dx_field(field, dx, varp, xaxis=0):
    """
    Numerically differentiate field in x direction.

    Assuming cyclic, modifying grid descriptor.

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
    differentiated field, grid indicator

    """
    d = field[...]
    newfield = (d - np.roll(d,1,axis=xaxis)) / dx
    varp[0] = not varp[0]
    return newfield, varp

def d_by_dy_field(field, dy, varp, yaxis=1):
    """
    Numerically differentiate field in y direction.

    Assuming cyclic, modifying grid descriptor.

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
    newfield = (d - np.roll(d,1,axis=yaxis)) / dy
    varp[1] = not varp[1]
    return newfield, varp

def d_by_dz_field(field, z, zn, varp):
    """
    Numerically differentiate field in z direction.

    Modifying grid descriptor.

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
    if varp[zaxis]: # Field is on zn points
        new = (d[..., 1:] - d[...,:-1])/ (zn[1:] - zn[:-1])
        zt = 0.5 * (zn[1:] + zn[:-1])
        newfield, newz = padright(new, zt, axis=zaxis)
    else:  # Field is on z points
        new = (d[..., 1:] - d[...,:-1])/ (z[1:] - z[:-1])
        zt = 0.5 * (z[1:] + z[:-1])
        newfield, newz = padleft(new, zt, axis=zaxis)

    varp[-1] = not varp[-1]
    return newfield, varp

def padleft(f, zt, axis=0) :
    """
    Add dummy field at bottom of nD array.

    Args:
        f : nD field
        zt: 1D zcoordinates
        axis=0: Specify axis to extend

    Returns
    -------
        extended field, extended coord
    @author: Peter Clark
    """
    s = list(np.shape(f))
    s[axis] += 1
#    print(zt)
    newfield = np.zeros(s)
    newfield[...,1:]=f
    newz = np.zeros(np.size(zt)+1)
    newz[1:] = zt
    newz[0] = 2*zt[0]-zt[1]
#    print(newz)
    return newfield, newz

def padright(f, zt, axis=0) :
    """
    Add dummy field at top of nD array.

    Args:
        f : nD field
        zt: 1D zcoordinates
        axis=0: Specify axis to extend

    Returns
    -------
        extended field, extended coord
    @author: Peter Clark
    """
    s = list(np.shape(f))
    s[axis] += 1
#    print(zt)
    newfield = np.zeros(s)
    newfield[...,:-1] = f
    newz = np.zeros(np.size(zt)+1)
    newz[:-1] = zt
    newz[-1] = 2*zt[-1]-zt[-2]
#    print(newz)
    return newfield, newz

def get_data(source_dataset, var_name, it, refprof, coords) :
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

    Returns
    -------
        variable, variable_grid_properties

    @author: Peter Clark and George Efstathiou

    """
    global ind

    # This is the list of supported operators.
    # The offset is how many characters to strip from front of variable name
    # to get raw variable.
    # e.g. dth_dx needs 1 to lose the first d.

    opdict = {'_dx':1,
              '_dy':1,
              '_dz':1,
              '_prime':0,
              '_crit':0,
              }
    vard = None
    varp = None
#    print(ind,var_name)
    try :
        var = source_dataset[var_name]
#        vardim = var.dimensions
        vard = var[it, ...]
        varp = var_properties[var_name]

#        print(vardim)
        if var_name == 'th' and refprof is not None:
#            print(refprof)
            vard[...] += refprof['th']

    except :
#        print("Data not in dataset")
        if var_name == 'th_L':

            theta, varp = get_data(source_dataset, 'th', it, refprof, coords)
            q_cl, vp = get_data(source_dataset, 'q_cloud_liquid_mass', it,
                                refprof, coords)
            vard = theta - L_over_cp * q_cl / refprof['pi']

        elif var_name == 'th_v':
            theta, varp = get_data(source_dataset, 'th', it, refprof, coords)
            q_v, vp = get_data(source_dataset, 'q_vapour', it, refprof, coords)
            q_cl, vp = get_data(source_dataset, 'q_cloud_liquid_mass', it,
                                refprof, coords)
            vard = theta + refprof['th'] * (c_virtual * q_v - q_cl)

        elif var_name == 'q_total':

            q_v, varp = get_data(source_dataset, 'q_vapour', it, refprof,
                                 coords)
            q_cl, vp = get_data(source_dataset, 'q_cloud_liquid_mass', it,
                                refprof, coords)
            vard = q_v + q_cl

        elif var_name == 'buoyancy':

            th_v, varp = get_data(source_dataset, 'th_v', it, refprof, coords)
            mean_thv = np.mean(th_v, axis = (0,1))
            vard = grav * (th_v - mean_thv)/mean_thv

        else:
            if ')' in var_name:
#                print(ind,"Found parentheses")
                v = var_name
                i1 = v.find('(')
                i2 = len(v)-v[::-1].find(')')
                source_var = v[i1+1: i2-1]
                v_op = v[i2:]
            else:
                for v_op, offset in opdict.items():
                    if v_op in var_name:
                        source_var = var_name[offset:var_name.find(v_op)]
                        break

#            print(ind,f"Variable: {source_var} v_op: {v_op}")
#            ind+="    "
            vard, varp = get_data(source_dataset, source_var, it, refprof,
                                      coords)
#            ind = ind[:-4]
            if '_dx' == v_op:
#                print(ind,"Exec _dx")
                vard, varp = d_by_dx_field(vard, coords['deltax'], varp)

            elif '_dy' == v_op:
#                print(ind,"Exec _dy")
                vard, varp = d_by_dy_field(vard, coords['deltay'], varp)

            elif '_dz' == v_op:
#                print(ind,"Exec _dz")
                vard, varp = d_by_dz_field(vard, coords['z'], coords['zn'],
                                           varp)

            elif '_prime' == v_op:
#                print(ind,"Exec _prime")
                vard = vard - np.mean(vard, axis = (0,1))

            elif '_crit' == v_op:
#                print(ind,"Exec _crit")
                std = np.std(vard, axis=(0,1))

                std_int=np.ones(len(std))
                c_crit=np.ones(len(std))
                for iz in range(len(std)) :
                    std_int[iz] = 0.05*np.mean(std[:iz+1])
                    c_crit[iz] = np.maximum(std[iz], std_int[iz])

                vard = np.ones_like(vard) * c_crit



        if vard is None:
            print(f'Variable {var_name} not available.')

    return vard, varp

def load_traj_step_data(dataset, it, variable_list, refprof, coords) :
    """
    Read trajectory variables and additional data from file.

    For interpolation to trajectory.

    Args:
        dataset        : netcdf file handle.
        it             : time index in netcdf file.
        variable_list  : List of variable names.
        refprof        : Dict with reference theta profile arrays.

    Returns
    -------
        List of arrays containing interpolated data.

    @author: Peter Clark

    """
    global ind
    data_list, var_list, varp_list, time = load_traj_pos_data(dataset, it)


    for variable in variable_list :
#        print 'Reading ', variable
        ind=""
        data, varp = get_data(dataset, variable, it, refprof, coords)

        data_list.append(data)
        varp_list.append(varp)
    return data_list, variable_list, varp_list, time

def phase(vr, vi, n) :
    """
    Convert real and imaginary points to location on grid size n.

    Args:
        vr,vi  : real and imaginary parts of complex location.
        n      : grid size

    Returns
    -------
        Real position in [0,n)

    @author: Peter Clark

    """
    vr = np.asarray(vr)
    vi = np.asarray(vi)
    vpos = np.asarray((((np.arctan2(vi,vr))/(2.0*np.pi)) * n) % n )
#    vpos[vpos<0] += n
    return vpos

def compute_traj_boxes(traj, in_obj_func, kwargs={}) :
    """
    Compute two rectangular boxes containing all and in_obj points.

    For each trajectory object and time, plus some associated data.

    Args:
        traj               : trajectory object.
        in_obj_func        : function to determine which points are inside an object.
        kwargs             : any additional keyword arguments to ref_func (dict).

    Returns
    -------
        Properties of in_obj boxes::

            data_mean        : mean of points (for each data variable) in in_obj.
            in_obj_data_mean : mean of points (for each data variable) in in_obj.
            num_in_obj       : number of in_obj points.
            traj_centroid    : centroid of in_obj box.
            in_obj__centroid : centroid of in_obj box.
            traj_box         : box coordinates of each trajectory set.
            in_obj_box       : box coordinates for in_obj points in each trajectory set.

    @author: Peter Clark

    """
    scalar_shape = (np.shape(traj.data)[0], traj.nobjects)
    centroid_shape = (np.shape(traj.data)[0], traj.nobjects, \
                      3)
    mean_obj_shape = (np.shape(traj.data)[0], traj.nobjects, \
                      np.shape(traj.data)[2])
    box_shape = (np.shape(traj.data)[0], traj.nobjects, 2, 3)

    data_mean = np.zeros(mean_obj_shape)
    in_obj_data_mean = np.zeros(mean_obj_shape)
    objvar_mean = np.zeros(scalar_shape)
    num_in_obj = np.zeros(scalar_shape, dtype=int)
    traj_centroid = np.zeros(centroid_shape)
    in_obj_centroid = np.zeros(centroid_shape)
    traj_box = np.zeros(box_shape)
    in_obj_box = np.zeros(box_shape)

    in_obj_mask, objvar = in_obj_func(traj, **kwargs)

    for iobj in range(traj.nobjects):

        data = traj.data[:,traj.labels == iobj,:]
        data_mean[:,iobj,:] = np.mean(data, axis=1)
        obj = traj.trajectory[:, traj.labels == iobj, :]

        traj_centroid[:,iobj, :] = np.mean(obj,axis=1)
        traj_box[:,iobj, 0, :] = np.amin(obj, axis=1)
        traj_box[:,iobj, 1, :] = np.amax(obj, axis=1)

        objdat = objvar[:,traj.labels == iobj]

        for it in np.arange(0,np.shape(obj)[0]) :
            mask = in_obj_mask[it, traj.labels == iobj]
            num_in_obj[it, iobj] = np.size(np.where(mask))
            if num_in_obj[it, iobj] > 0 :
                in_obj_data_mean[it, iobj, :] = np.mean(data[it, mask, :], axis=0)
                objvar_mean[it, iobj] = np.mean(objdat[it, mask])
                in_obj_centroid[it, iobj, :] = np.mean(obj[it, mask, :],axis=0)
                in_obj_box[it, iobj, 0, :] = np.amin(obj[it, mask, :], axis=0)
                in_obj_box[it, iobj, 1, :] = np.amax(obj[it, mask, :], axis=0)
    return data_mean, in_obj_data_mean, objvar_mean, num_in_obj, \
           traj_centroid, in_obj_centroid, traj_box, in_obj_box


def print_boxes(traj) :
    """
    Print information provided by compute_traj_boxes.

    Args
    ----
        traj : trajectory object.

    @author: Peter Clark

    """
    varns = list(traj.variable_list)
    for iobj in range(traj.nobjects) :
        print('Object {:03d}'.format(iobj))
        print('  Time    x      y      z   '+\
              ''.join([' {:^10s}'.format(varns[i][:9]) for i in range(6)]))
        for it in range(np.shape(traj.centroid)[0]) :
            strf = '{:6.0f}'.format(traj.times[it]/60)
            strf = strf + ''.join([' {:6.2f}'.\
                    format(traj.centroid[it,iobj,i]) for i in range(3)])
            strf = strf + ''.join([' {:10.6f}'.\
                    format(traj.data_mean[it,iobj,i]) for i in range(6)])
            print(strf)
    return

def box_overlap_with_wrap(b_test, b_set, nx, ny) :
    """
    Compute whether rectangular boxes intersect.

    Args
    ----
        b_test: box for testing array[8,3]
        b_set: set of boxes array[n,8,3]
        nx: number of points in x grid.
        ny: number of points in y grid.

    Returns
    -------
        indices of overlapping boxes

    @author: Peter Clark

    """
    # Wrap not yet implemented

    t1 = np.logical_and( \
        b_test[0,0] >= b_set[...,0,0] , b_test[0,0] <= b_set[...,1,0])
    t2 = np.logical_and( \
        b_test[1,0] >= b_set[...,0,0] , b_test[1,0] <= b_set[...,1,0])
    t3 = np.logical_and( \
        b_test[0,0] <= b_set[...,0,0] , b_test[1,0] >= b_set[...,1,0])
    t4 = np.logical_and( \
        b_test[0,0] >= b_set[...,0,0] , b_test[1,0] <= b_set[...,1,0])
    x_overlap =np.logical_or(np.logical_or( t1, t2), np.logical_or( t3, t4) )

#    print(x_overlap)
    x_ind = np.where(x_overlap)[0]
#    print(x_ind)
    t1 = np.logical_and( \
        b_test[0,1] >= b_set[x_ind,0,1] , b_test[0,1] <= b_set[x_ind,1,1])
    t2 = np.logical_and( \
        b_test[1,1] >= b_set[x_ind,0,1] , b_test[1,1] <= b_set[x_ind,1,1])
    t3 = np.logical_and( \
        b_test[0,1] <= b_set[x_ind,0,1] , b_test[1,1] >= b_set[x_ind,1,1])
    t4 = np.logical_and( \
        b_test[0,1] >= b_set[x_ind,0,1] , b_test[1,1] <= b_set[x_ind,1,1])
    y_overlap = np.logical_or(np.logical_or( t1, t2), np.logical_or( t3, t4) )

    y_ind = np.where(y_overlap)[0]

    return x_ind[y_ind]

def file_key(file):
    f1 = file.split('_')[-1]
    f2 = f1.split('.')[0]
    return float(f2)

def find_time_in_files(files, ref_time, nodt = False) :
    """
    Find file containing data at required time.

        Assumes file names are of form \*_tt.0\* where tt is model output time.

    Args:
        files: ordered list of files
        ref_time: required time.
        nodt: if True do not look for next time to get delta_t

    Returns
    -------
        Variables defining location of data in file list::

            ref_file: Index of file containing required time in files.
            it: Index of time in dataset.
            delta_t: Interval between data.

    @author: Peter Clark

    """
    file_times = np.zeros(len(files))
    for i, file in enumerate(files) : file_times[i] = file_key(file)

    def get_file_times(dataset) :
        theta=dataset.variables["th"]
#        print(theta)
#        print(theta.dimensions[0])
        t=dataset.variables[theta.dimensions[0]][...]
#        print(t)
        return t

    delta_t = 0.0
    it=-1
    ref_file = np.where(file_times >= ref_time)[0][0]
    while True :
        if ref_file >= len(files) or ref_file < 0 :
            ref_file = None
            break
#        print(files[ref_file])
        dataset=Dataset(files[ref_file])
        print("Dataset opened ", files[ref_file])
        times = get_file_times(dataset)
#        dataset.close()
#        print("dataset closed")
#        print(times)

        if len(times) == 1 :
            dataset.close()
            print("dataset closed")
            it = 0
            if times[it] != ref_time :
                print('Could not find exact time {} in file {}'.\
                  format(ref_time,files[ref_file]))
                ref_file = None
            else :
                if nodt :
                    delta_t = 0.0
                else :
                    print('Looking in next file to get dt.')
                    dataset_next=Dataset(files[ref_file+1])
                    print("Dataset_next opened ", files[ref_file+1])
                    times_next = get_file_times(dataset_next)
                    delta_t = times_next[0] - times[0]
                    dataset_next.close()
                    print("dataset_next closed")
            break

        else : # len(times) > 1
            it = np.where(times == ref_time)[0]
            if len(it) == 0 :
                print('Could not find exact time {} in file {}'.\
                  format(ref_time,ref_file))
                it = np.where(times >= ref_time)[0]
#                print("it={}".format(it))
                if len(it) == 0 :
                    print('Could not find time >= {} in file {}, looking in next.'.\
                      format(ref_time,ref_file))
                    ref_file += 1
                    continue
#            else :
            it = it[0]
            if it == (len(times)-1) :
                delta_t = times[it] - times[it-1]
            else :
                delta_t = times[it+1] - times[it]
            break
    print(\
    "Looking for time {}, returning file #{}, index {}, time {}, delta_t {}".\
          format(  ref_time,ref_file, it, times[it], delta_t) )
    return ref_file, it, delta_t.astype(int)

def compute_derived_variables(traj, derived_variable_list=None) :
    """
    Compute required variables from model input.

    Parameters
    ----------
    traj : Trajectory
        DESCRIPTION.
    derived_variable_list : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    dict list of variables, data (numpy array)

    """
    if derived_variable_list is None :
        derived_variable_list = { \
                           "q_total":r"$q_{t}$ kg/kg", \
                           "th_L":r"$\theta_L$ K", \
                           "th_v":r"$\theta_v$ K", \
                           "MSE":r"MSE J kg$^{-1}$", \
                           }
    zn = traj.coords['zn']
    tr_z = np.interp(traj.trajectory[...,2], traj.coords['zcoord'], zn)

    piref_z = np.interp(tr_z,zn,traj.refprof['pi'])
    thref_z = np.interp(tr_z,zn,traj.refprof['th'])

    s = list(np.shape(traj.data))
#    print(s)
    s.pop()
    s.append(len(derived_variable_list))
#    print(s)
    out = np.zeros(s)
#    print(np.shape(out))

    for i,variable in enumerate(derived_variable_list.keys()) :

        if variable == "q_total" :
            data = traj.data[..., traj.var("q_vapour")] + \
                   traj.data[..., traj.var("q_cloud_liquid_mass")]

        if variable == "th_L" :
            data = traj.data[..., traj.var("th")] - \
              L_over_cp * \
              traj.data[..., traj.var("q_cloud_liquid_mass")] \
              / piref_z

        if variable == "th_v" :
            data = traj.data[..., traj.var("th")] + \
                   thref_z * (c_virtual * \
                              traj.data[..., traj.var("q_vapour")] -
                              traj.data[..., traj.var("q_cloud_liquid_mass")])

        if variable == "MSE" :
            data = traj.data[:, :, traj.var("th")] * \
                          Cp * piref_z + \
                          grav * tr_z + \
                          L_vap * traj.data[:, :, traj.var("q_vapour")]

#        print(variable, np.min(data),np.max(data),np.shape(data))

        out[...,i] = data

    return derived_variable_list, out
