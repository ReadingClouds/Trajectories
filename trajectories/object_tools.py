"""
object_tools module.

Created on Tue Feb 15 15:21:50 2022

@author: Peter Clark
"""
import numpy as np
from scipy import ndimage

#debug_unsplit = True
debug_unsplit = False
#debug_label = True
debug_label = False


def label_3D_cyclic(mask) :
    """
    Label 3D objects taking account of cyclic boundary in x and y.

    Uses ndimage(label) as primary engine.

    Args:
        mask: 3D logical array with object mask (i.e. objects are
            contiguous True).

    Returns
    -------
        Object identifiers::

            labs  : Integer array[nx,ny,nz] of labels. -1 denotes unlabelled.
            nobjs : number of distinct objects. Labels range from 0 to nobjs-1.

    @author: Peter Clark

    """
    (nx, ny, nz) = np.shape(mask)
    labels, nobjects = ndimage.label(mask)
    labels -=1
    def relabel(labs, nobjs, i,j) :
        lj = (labs == j)
        labs[lj] = i
        for k in range(j+1,nobjs) :
            lk = (labs == k)
            labs[lk] = k-1
        nobjs -= 1
        return labs, nobjs

    def find_objects_at_edge(minflag, dim, n, labs, nobjs) :
        i = 0
        while i < (nobjs-2) :
            # grid points corresponding to label i
            posi = np.where(labs == i)
            posid = posi[dim]
            if minflag :
                test1 = (np.min(posid) == 0)
                border = '0'
            else:
                test1 = (np.max(posid) == (n-1))
                border = f"n{['x','y'][dim]}-1"
            if test1 :
                if debug_label :
                    print('Object {:03d} on {}={} border?'.\
                          format(i,['x','y'][dim],border))
                j = i+1
                while j < nobjs :
                    # grid points corresponding to label j
                    posj = np.where(labs == j)
                    posjd = posj[dim]

                    if minflag :
                        test2 = (np.max(posjd) == (n-1))
                        border = f"n{['x','y'][dim]}-1"
                    else:
                        test2 = (np.min(posjd) == 0)
                        border = '0'

                    if test2 :
                        if debug_label :
                            print('Match Object {:03d} on {}={} border?'\
                                  .format(j,['x','y'][dim],border))

                        if minflag :
                            ilist = np.where(posid == 0)
                            jlist = np.where(posjd == (n-1))
                        else :
                            ilist = np.where(posid == (n-1))
                            jlist = np.where(posjd == 0)

                        int1 = np.intersect1d(posi[1-dim][ilist],
                                              posj[1-dim][jlist])
                        # z-intersection
                        int2 = np.intersect1d(posi[2][ilist],
                                              posj[2][jlist])
                        if np.size(int1)>0 and np.size(int2)>0 :
                            if debug_label :
                                print('Yes!',i,j)
                            labs, nobjs = relabel(labs, nobjs, i, j)
                    j += 1
            i += 1
        return labs, nobjs

    labels, nobjects = find_objects_at_edge(True,  0, nx, labels, nobjects)
    labels, nobjects = find_objects_at_edge(False, 0, nx, labels, nobjects)
    labels, nobjects = find_objects_at_edge(True,  1, ny, labels, nobjects)
    labels, nobjects = find_objects_at_edge(False, 1, ny, labels, nobjects)

    return labels, nobjects

def unsplit_object( pos, nx, ny ) :
    """
    Gather together points in object separated by cyclic boundaries.

        For example, if an object spans the 0/nx boundary, so some
        points are close to zero, some close to nx, they will be adjusted to
        either go from negative to positive, close to 0, or less than nx to
        greater than. The algorithm tries to group on the larges initial set.

    Args:
        pos      : grid positions of points in object.
        nx,ny    : number of grid points in x and y directions.

    Returns
    -------
        Adjusted grid positions of points in object.

    @author: Peter Clark

    """
    global debug_unsplit
    if debug_unsplit : print('pos:', pos)

    n = (nx, ny)

    for dim in range(2):
        q0 = pos[:, dim] < n[dim] * 0.25
        q3 = pos[:, dim] > n[dim] * 0.75
        if np.sum(q0) < np.sum(q3):
            pos[q0, dim] += n[dim]
        else:
            pos[q3, dim] -= n[dim]

    return pos

def unsplit_objects(trajectory, labels, nobjects, nx, ny) :
    """
    Unsplit a set of objects at a set of times using unsplit_object on each.

    Args:
        trajectory     : Array[nt, np, 3] of trajectory points, with nt \
                         times and np points.
        labels         : labels of trajectory points.
        nx,ny   : number of grid points in x and y directions.

    Returns
    -------
        Trajectory array with modified positions.

    @author: Peter Clark

    """
    global debug_unsplit
#    print np.shape(trajectory)
    if nobjects < 2:
        return trajectory

    print('Unsplitting Objects:')

    for iobj in range(0,nobjects):
        if debug_unsplit : print('Unsplitting Object: {:03d}'.format(iobj))

        for it in range(0,np.shape(trajectory)[0]) :
            if debug_unsplit : print('Time: {:03d}'.format(it))
            tr = trajectory[it,labels == (iobj),:]
            if ((np.max(tr[:,0])-np.min(tr[:,0])) > nx/2 ) or \
               ((np.max(tr[:,1])-np.min(tr[:,1])) > ny/2 ) :
                trajectory[it, labels == iobj,:] = \
                unsplit_object(trajectory[it,labels == iobj,:], \
                                               nx, ny)
                if debug_unsplit : print('New object:',\
                    trajectory[it,labels == iobj,:])

    return trajectory
