"""Miscellaneous functions

This private module contains miscellaneous functions.
"""
import numpy as np

__author__ = 'Kenneth (Kip) Hart'

kw_solid = {'crystalline', 'granular', 'solid'}
kw_amorph = {'amorphous', 'glass', 'matrix'}
kw_void = {'void', 'crack', 'hole'}

ori_kws = {'orientation', 'matrix', 'angle', 'angle_deg', 'angle_rad',
           'rot_seq', 'rot_seq_rad', 'rot_seq_deg'}
gen_kws = {'material_type', 'fraction', 'shape', 'name', 'color', 'position'}

demo_needs = {'basalt_circle.xml': ['aphanitic_cdf.csv', 'olivine_cdf.csv'],
              'from_image.py': ['aluminum_micro.png']}

mpl_plural_kwargs = {'edgecolors', 'facecolors', 'linewidths', 'antialiaseds',
                     'offsets'}
plt_3d_adj = {
    'left': 0.4,
    'right': 1,
    'bottom': 0,
    'top': 0.8,
}


# --------------------------------------------------------------------------- #
#                                                                             #
# Convert String to Value (Infer Type)                                        #
#                                                                             #
# --------------------------------------------------------------------------- #
def from_str(string):
    """ Convert string to number

    This function takes a string and converts it into a number or a list.

    Args:
        string (str): The string.

    Returns:
        The value in the string.

    """
    beg_delims = ('(', '[', '{', '<')
    end_delims = (')', ']', '}', '>')

    string = string.strip()
    has_delims = False
    for beg, end in zip(beg_delims, end_delims):
        has_beg = string.startswith(beg)
        has_end = string.endswith(end)
        has_delims |= has_beg and has_end
    if has_delims or ',' in string:
        val = _list_from_str(string, beg_delims, end_delims)
    else:
        val = _single_from_str(string)
    return val


def _list_from_str(string, beg_delims, end_delims):
    if string[0] in beg_delims:
        string = string[1:]
    if string[-1] in end_delims:
        string = string[:-1]
    val = []
    n_beg = 0
    n_end = 0
    elem_str = ''
    for char in string:
        if char in beg_delims:
            n_beg += 1
        elif char in end_delims:
            n_end += 1

        if (char == ',') and n_beg == n_end:
            val.append(from_str(elem_str.strip()))
            elem_str = ''
        else:
            elem_str += char
    if elem_str == string and ',' in string:
        return _single_from_str(string)
    val.append(from_str(elem_str.strip()))
    return val


def _single_from_str(string):
    try:
        val = int(string)
    except ValueError:
        try:
            val = float(string)
        except ValueError:
            if string.lower() in ('true', 'yes'):
                val = True
            elif string.lower() in ('false', 'no'):
                val = False
            else:
                val = str(string)
    return val


# --------------------------------------------------------------------------- #
#                                                                             #
# Tangent Spheres                                                             #
#                                                                             #
# --------------------------------------------------------------------------- #
def tangent_sphere(points, radii=None, simplices=None):
    """Calculate center and radius of tangent sphere(s)

    This function computes the center and radius of an n-dimensional sphere
    that is tangent to (n+1) spheres. For example, in 2D this function
    computes the center and radius of a circle tangent to three other circles.

    The operation of this function can be vectorized using the ``simplices``
    input. The simplices should be an Mx(n+1) list of indices of the points.
    The result is an Mx(n+1) numpy array, where the first n columns are the
    coordinates of the sphere center. The final column is the radius of the
    sphere.

    If no radii are specified, the results are circumspheres of the simplices
    (circumcircles in 2D).

    Radii at each point can be speficied. If no radii are given, then the
    results are circumspheres of the simplices (circumcircles in 2D).

    Args:
        points (list, tuple, numpy.ndarray): List of points.
        radii (list, tuple, numpy.ndarray): List of radii. *(optional)*
        simplices (list, tuple, numpy.ndarray): List of simplices. *(optional)*

    Returns:
        numpy.ndarray: The centers and radii of tangent spheres.

    """
    # set radii
    if radii is None:
        radii = np.full(len(points), 0)

    # extract points
    if simplices is None:
        simplices = np.arange(len(points)).reshape(1, -1)

    pts = np.array(points)[simplices]
    rs = np.array(radii)[simplices]

    # define circle distances
    cs = np.sum(pts * pts, axis=-1) - rs * rs

    # matrix and vector quantities
    pos1 = pts[:, 0]
    r1 = rs[:, 0]
    A = pts[:, 1:] - pos1[:, np.newaxis, :]
    b = -1 * (rs[:, 1:] - r1[:, np.newaxis])
    c = 0.5 * (cs[:, 1:] - cs[:, 0, np.newaxis])

    # linear system coefficients
    alpha = np.linalg.solve(A, b)
    beta = np.linalg.solve(A, c)

    # quadratic equation in rc
    r_beta = beta - pos1
    C1 = np.sum(alpha * alpha, axis=-1) - 1
    C2 = np.sum(r_beta * alpha, axis=-1) - r1
    C3 = np.sum(r_beta * r_beta, axis=-1) - r1 * r1

    # solve for rc
    discr = C2 * C2 - C1 * C3
    rt_discr = np.sqrt(discr)
    rt_discr[discr < 0] = 0

    rc1 = (-C2 + rt_discr) / C1
    rc2 = (-C2 - rt_discr) / C1

    mask = np.abs(rc1) < np.abs(rc2)
    rc = rc2
    rc[mask] = rc1[mask]
    rc[discr < 0] = 0

    # solve for center position
    posc = alpha * rc[:, np.newaxis] + beta

    # return results
    spheres = np.hstack((posc, rc.reshape(-1, 1)))
    return np.squeeze(spheres)


def axisEqual3D(ax):
    '''From stackoverflow: https://stackoverflow.com/a/19248731'''
    extents = np.array([getattr(ax, 'get_{}lim'.format(d))() for d in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def ax_objects(ax):
    n = 0
    for att in ['collections', 'images', 'lines', 'patches', 'texts']:
        n += len(getattr(ax, att))
    return n
