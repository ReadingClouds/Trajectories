import numpy as np


def box_overlap_with_wrap(b_test, b_set, nx, ny):
    """
    Function to compute whether rectangular boxes intersect.

    Args:
        b_test: box for testing array[8,3]
        b_set: set of boxes array[n,8,3]
        nx: number of points in x grid.
        ny: number of points in y grid.

    Returns:
        indices of overlapping boxes

    @author: Peter Clark

    """

    # Wrap not yet implemented
    # TODO: add NotImplementedError exception for when "wrap" is required

    t1 = np.logical_and(
        b_test[0, 0] >= b_set[..., 0, 0], b_test[0, 0] <= b_set[..., 1, 0]
    )
    t2 = np.logical_and(
        b_test[1, 0] >= b_set[..., 0, 0], b_test[1, 0] <= b_set[..., 1, 0]
    )
    t3 = np.logical_and(
        b_test[0, 0] <= b_set[..., 0, 0], b_test[1, 0] >= b_set[..., 1, 0]
    )
    t4 = np.logical_and(
        b_test[0, 0] >= b_set[..., 0, 0], b_test[1, 0] <= b_set[..., 1, 0]
    )
    x_overlap = np.logical_or(np.logical_or(t1, t2), np.logical_or(t3, t4))

    #    print(x_overlap)
    x_ind = np.where(x_overlap)[0]
    #    print(x_ind)
    t1 = np.logical_and(
        b_test[0, 1] >= b_set[x_ind, 0, 1], b_test[0, 1] <= b_set[x_ind, 1, 1]
    )
    t2 = np.logical_and(
        b_test[1, 1] >= b_set[x_ind, 0, 1], b_test[1, 1] <= b_set[x_ind, 1, 1]
    )
    t3 = np.logical_and(
        b_test[0, 1] <= b_set[x_ind, 0, 1], b_test[1, 1] >= b_set[x_ind, 1, 1]
    )
    t4 = np.logical_and(
        b_test[0, 1] >= b_set[x_ind, 0, 1], b_test[1, 1] <= b_set[x_ind, 1, 1]
    )
    y_overlap = np.logical_or(np.logical_or(t1, t2), np.logical_or(t3, t4))

    y_ind = np.where(y_overlap)[0]

    return x_ind[y_ind]
