# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 10:31:20 2021

@author: paclk
"""
import numpy as np
from typing import List
from fast_interp import interp3d


def coord_wrap(c, cmax):
#    c[c<0] += cmax
#    c[c>=cmax] -= cmax
    return c%cmax

def tri_lin_interp(data, varp_list, pos,
                   xcoord, ycoord, zcoord, maxindex=None,
                   use_corrected_positions=False) :
    """
    Tri-linear interpolation with cyclic wrapround in x and y.

    Args:
        data: list of Input data arrays on 3D grid.
        varp_list: list of grid info lists.
        pos(Array[n,3]): Positions to interpolate to (in grid units) .
        xcoord: 1D coordinate vector.
        ycoord: 1D coordinate vector.
        zcoord: 1D coordinate vector.

    Returns:
        list of 1D arrays of data interpolated to pos.

    @author : Peter Clark
    """

    nx = len(xcoord)
    ny = len(ycoord)
    nz = len(zcoord)

    x = coord_wrap(pos[:,0], nx)
    y = coord_wrap(pos[:,1], ny)
    z = pos[:,2]

    ix = np.floor(x).astype(int)
    iy = np.floor(y).astype(int)
    iz = np.floor(z).astype(int)

    iz[iz>(nz-2)] -= 1

    xp = (x-ix)
    yp = (y-iy)
    zp = (z-iz)

    wx = [1.0 - xp, xp]
    wy = [1.0 - yp, yp]
    wz = [1.0 - zp, zp]

    if use_corrected_positions :

        x_u = coord_wrap(pos[:,0] - 0.5, nx)
        y_v = coord_wrap(pos[:,1] - 0.5, ny)
        z_w = pos[:,2] - 0.5

        ix_u = np.floor(x_u).astype(int)
        iy_v = np.floor(y_v).astype(int)
        iz_w = np.floor(z_w).astype(int)
        iz_w[iz_w>(nz-2)] -= 1

        xp_u = (x_u-ix_u)
        yp_v = (y_v-iy_v)
        zp_w = (z_w-iz_w)

        wx_u = [1.0 - xp_u, xp_u]
        wy_v = [1.0 - yp_v, yp_v]
        wz_w = [1.0 - zp_w, zp_w]

    else:

        x_u = x
        y_v = y
        z_w = z

        ix_u = ix
        iy_v = iy
        iz_w = iz

        xp_u = xp
        yp_v = yp
        zp_w = zp

        wx_u = wx
        wy_v = wy
        wz_w = wz


    if maxindex is None :
        maxindex = len(data)

    output= list([])
    for l in range(maxindex) :
        output.append(np.zeros_like(x))
#    t = 0

    for i in range(0,2):
#        wi = wx[i]
        ii = (ix + i) % nx
        if  use_corrected_positions :
            ii_u = (ix_u + i) % nx
        else :
            ii_u = ii

        for j in range(0,2):

#            wi_wj = wx[i] * wy[j]
            jj = (iy + j) % ny

            if  use_corrected_positions :
                jj_v = (iy_v + j) % ny
#                wi_u_wj = wx_u[i] * wy[j]
#                wi_wj_v = wx[i] * wy_v[j]
            else :
                jj_v = jj
#                wi_u_wj = wi_wj
#                wi_wj_v = wi_wj

            for k in range(0,2):

#                w = wi_wj * wz[k]
                kk = iz + k

                if  use_corrected_positions :
                    kk_w = iz_w + k
#                    w_u = wi_u_wj * wz[k]
#                    w_v = wi_wj_v * wz[k]
#                    w_w = wi_wj * wz_w[k]
                else :
                    kk_w = kk
#                    w_u = w
#                    w_v = w
#                    w_w = w

#                t += w
                for l in range(maxindex) :
                    if varp_list[l][0]:
                        II = ii_u
                        WX = wx_u
                    else:
                        II = ii
                        WX = wx

                    if varp_list[l][1]:
                        JJ = jj_v
                        WY = wy_v
                    else:
                        JJ = jj
                        WY = wy

                    if varp_list[l][2]:
                        KK = kk_w
                        WZ = wz_w
                    else:
                        KK = kk
                        WZ = wz

#                    print(l, variable_list[l])
                    output[l] +=  data[l][II, JJ, KK] * WX[i] * WY[j] * WZ[k]

    return output

def multi_dim_lagrange_interp(data: List[np.ndarray], pos: np.ndarray,
                              order: int = 3,
                              wrap: List[bool] = None) -> List[np.ndarray]:
    """
    Multidimensional arbitrary order Lagrange interpolation.

    Parameters
    ----------
    data : list[np.ndarray]
        List of N-dimensional numpy arrays with data to interpolate.
    pos : np.ndarray [..., N]
        Positions in N-dimensional space to interpolate to in grid units.
    order : int, optional
        Lagrange polynomial order. The default is 3.
    wrap : list[bool], optional
        True for any dimension means periodic wrapping. Otherwise fixed
        values outside boundaries. The default is None, i.e. wrap all dims.

    Returns
    -------
    List of nump arrays containing data interpolated to pos, retaing structure
    of pos apart from last dimension.

    @author : Peter Clark (C) 2021
    """
    if type(data) is not list:
        raise TypeError('Argument data should be a list of numpy arrays.')
    if type(pos) is not np.ndarray:
        raise TypeError('Argument pos should be numpy array.')

# Stencil of points to use assuming x between 0 and 1.
# So for order=1, [0,1], order=3, [-1,0,1,2] etc.

    local_grid = np.arange(order+1)-order//2

# Weights in 1D for each stencil point.
    grid_weight = np.ones(order+1)
    for i in range(order+1):
        for j in range(order+1):
            if i==j:
                continue
            else:
                grid_weight[i] *=  1.0/(i-j)

    npts = np.shape(data[0])
    ndims = len(npts)

# Default is wrap if not supplied.
    if wrap is None:
        wrap = [True for i in range(ndims)]


# Make sure points to interpolate to are in acceptable range.
    for dim in range(ndims):
        if wrap[dim]:
            pos[..., dim] %= npts[dim]
        else:
            pos[..., dim] = np.clip(pos[..., dim], 0, npts[dim]-1)

# Split x into integer and fractional parts.

    idim = np.floor(pos).astype(int)
    xdim = pos - idim

    def compute_interp(weight, inds, off, dim):
        """
        Recursive function to compute Lagrange polynomial interpolation.

        Parameters
        ----------
        weight : float or numpy array.
            Weight for current gridpoint in stencil.
        inds : list of numpy arrays of interger indices.
            Actual gridpoints for this point in stencil.
        off : int
            Position in stencil for current dimension.
        dim : int
            Dimension.

        Returns
        -------
        Either contribution from this gridpoint or final result,
        data list interpolated to pos.

        """
        if dim >= 0:
#            print('Finding weight')
            # Find indices for stencil position in this dimension.
            ii = (idim[..., dim] + local_grid[off])
            if wrap[dim]:
                ii %= npts[dim]
            else:
                ii = np.clip(ii, 0, npts[dim]-1)
            inds.append(ii)

            # Find weight for this stencil position u=in this dimension.
            w = grid_weight[off]
            for woffset in range(order+1):
                if woffset == off:
                    continue
                else:
                    w *= (xdim[..., dim] - local_grid[woffset])

            weight *= w

        if dim == ndims-1:
            # Weight is now weight over all dimensions, so we can find
            # contribution to final result.
            contrib = []
            for d in data:
                o = d[tuple(inds)] * weight
                contrib.append(o)

            return contrib

        else:
            # Find contributions from each dimension and add in.
            interpolated_data = None
            for offset in range(order+1):
                contrib = compute_interp(weight.copy(), inds.copy(),
                                         offset, dim+1)
                if contrib is not None:
                    if interpolated_data is not None:
                        for l, c in enumerate(contrib):
                            interpolated_data[l] += c
                    else:
                        interpolated_data = contrib
            return interpolated_data

    weight = np.ones_like(xdim[..., 0])
    inds = []
    off = -1
    dim = -1
    output = compute_interp(weight, inds, off, dim)

    return output

def fast_interp_3D(data: List[np.ndarray], pos: np.ndarray,
                                  order: int = 3,
                                  wrap: List[bool] = None) -> List[np.ndarray]:
    output = []
    grid_min = (0,0,0)
    grid_max = list(np.shape(data[0]))
    grid_max[2] -= 1
    grid_int = (1.0, 1.0, 1.0)
    for v in data:
        interpolator = interp3d(grid_min, grid_max, grid_int, v, k=order,
                                p=wrap)
        new_val = interpolator(pos[..., 0], pos[..., 1], pos[...,2])
        output.append(new_val)
        del interpolator
    return output
