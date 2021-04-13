"""
Interpolation routines for cyclic boundary conditions
"""
import numpy as np


def _coord_wrap(c, cmax):
    c[c < 0] += cmax
    c[c >= cmax] -= cmax
    return c


def tri_lin_interp(
    data,
    varp_list,
    pos,
    xcoord,
    ycoord,
    zcoord,
    use_corrected_positions=True,
    maxindex=None,
):

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
    # TODO: what does `use_corrected_positions` mean?

    nx = len(xcoord)
    ny = len(ycoord)
    nz = len(zcoord)

    x = _coord_wrap(pos[:, 0], nx)
    y = _coord_wrap(pos[:, 1], ny)
    z = pos[:, 2]

    ix = np.floor(x).astype(int)
    iy = np.floor(y).astype(int)
    iz = np.floor(z).astype(int)

    iz[iz > (nz - 2)] -= 1

    xp = x - ix
    yp = y - iy
    zp = z - iz

    wx = [1.0 - xp, xp]
    wy = [1.0 - yp, yp]
    wz = [1.0 - zp, zp]

    if use_corrected_positions:

        x_u = _coord_wrap(pos[:, 0] - 0.5, nx)
        y_v = _coord_wrap(pos[:, 1] - 0.5, ny)
        z_w = pos[:, 2] - 0.5

        ix_u = np.floor(x_u).astype(int)
        iy_v = np.floor(y_v).astype(int)
        iz_w = np.floor(z_w).astype(int)
        iz_w[iz_w > (nz - 2)] -= 1

        xp_u = x_u - ix_u
        yp_v = y_v - iy_v
        zp_w = z_w - iz_w

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

    #    ix = whichbox(xcoord, x)
    #    iy = whichbox(ycoord, y)
    #    iz = whichbox(zcoord, z)
    #    dx = 1.0
    #    dy = 1.0
    #    ix = np.floor(x / dx).astype(int)
    #    iy = np.floor(y / dy).astype(int)
    #    iz = np.floor(z).astype(int)

    #    xp = (x-xcoord[ix])/dx
    #    yp = (y-ycoord[iy])/dy
    #    zp = (z-zcoord[iz])/(zcoord[iz+1]-zcoord[iz])

    if maxindex is None:
        maxindex = len(data)

    output = list([])
    for l in range(maxindex):
        output.append(np.zeros_like(x))
    #    t = 0

    for i in range(0, 2):
        #        wi = wx[i]
        ii = (ix + i) % nx
        if use_corrected_positions:
            ii_u = (ix_u + i) % nx
        else:
            ii_u = ii

        for j in range(0, 2):

            #            wi_wj = wx[i] * wy[j]
            jj = (iy + j) % ny

            if use_corrected_positions:
                jj_v = (iy_v + j) % ny
            #                wi_u_wj = wx_u[i] * wy[j]
            #                wi_wj_v = wx[i] * wy_v[j]
            else:
                jj_v = jj
            #                wi_u_wj = wi_wj
            #                wi_wj_v = wi_wj

            for k in range(0, 2):

                #                w = wi_wj * wz[k]
                kk = iz + k

                if use_corrected_positions:
                    kk_w = iz_w + k
                #                    w_u = wi_u_wj * wz[k]
                #                    w_v = wi_wj_v * wz[k]
                #                    w_w = wi_wj * wz_w[k]
                else:
                    kk_w = kk
                #                    w_u = w
                #                    w_v = w
                #                    w_w = w

                #                t += w
                for l in range(maxindex):
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
                    output[l] += data[l][II, JJ, KK] * WX[i] * WY[j] * WZ[k]

    return output
