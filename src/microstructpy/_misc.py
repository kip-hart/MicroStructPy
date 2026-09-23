"""Miscellaneous functions

This private module contains miscellaneous functions.
"""

import ast
import re

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

    Booleans are recognized regardless of case (``true``, ``FALSE``, ...),
    on their own or inside a tuple/list such as ``(true, false)``.
    Strings that merely contain these words (``true_cdf.csv``) are returned
    unchanged. Values that Python does not accept as literals but ``float``
    does, such as ``inf``, ``-inf`` and ``nan``, are converted to floats.

    Args:
        string (str): The string.

    Returns:
        The value in the string.

    """
    s = string.strip()
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        pass

    # Booleans, case-insensitive
    if s.lower() in ('true', 'false'):
        return s.lower() == 'true'

    # Booleans inside a literal, e.g. '(true, False)'
    norm_s = _bool_re.sub(lambda m: m.group(0).capitalize(), s)
    if norm_s != s:
        try:
            return ast.literal_eval(norm_s)
        except (ValueError, SyntaxError):
            pass

    # Floats that are not Python literals: inf, -inf, nan
    try:
        return float(s)
    except ValueError:
        pass

    return s


_bool_re = re.compile(r'\b(true|false)\b', flags=re.IGNORECASE)


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
    ax.set_aspect('equal')


def ax_objects(ax):
    n = 0
    for att in ['collections', 'images', 'lines', 'patches', 'texts']:
        n += len(getattr(ax, att))
    return n


# --------------------------------------------------------------------------- #
#                                                                             #
# Periodicity                                                                 #
#                                                                             #
# --------------------------------------------------------------------------- #
def periodic_axes(periodic, n_dim):
    """Per-axis periodicity flags.

    The periodicity of a microstructure can be given as a boolean (all axes
    or none), a list of booleans (one per axis), or a string with the names
    of the periodic axes, such as ``'x'``, ``'xy'`` or ``'xz'``.

    Args:
        periodic (bool, list, or str): The periodicity specification.
        n_dim (int): Number of dimensions of the domain.

    Returns:
        list: ``n_dim`` booleans, True for the periodic axes.

    Raises:
        ValueError: If the specification cannot be interpreted.

    """
    if periodic is None:
        return [False for _ in range(n_dim)]

    if isinstance(periodic, (bool, np.bool_)):
        return [bool(periodic) for _ in range(n_dim)]

    axis_names = 'xyz'[:n_dim]
    if isinstance(periodic, str):
        text = periodic.strip().lower()
        if text in ('true', 'all', 'yes'):
            return [True for _ in range(n_dim)]
        if text in ('false', 'none', 'no', ''):
            return [False for _ in range(n_dim)]
        flags = [False for _ in range(n_dim)]
        for word in text.replace(',', ' ').split():
            for char in word:
                if char not in axis_names:
                    e_str = 'Cannot interpret periodic axes ' + repr(periodic)
                    e_str += '. Use a boolean, a list of ' + str(n_dim)
                    e_str += ' booleans, or axis names such as '
                    e_str += repr(axis_names) + '.'
                    raise ValueError(e_str)
                flags[axis_names.index(char)] = True
        return flags

    flags = [bool(f) for f in periodic]
    if len(flags) != n_dim:
        e_str = 'Expected ' + str(n_dim) + ' periodicity flags, got '
        e_str += str(len(flags)) + ': ' + repr(periodic) + '.'
        raise ValueError(e_str)
    return flags


def periodic_domain_limits(domain):
    """(lower, upper) bounds of a rectangular, axis-aligned domain.

    Periodic microstructures are only supported in such domains.

    Args:
        domain (from :mod:`microstructpy.geometry`): The domain.

    Returns:
        list: One (lower, upper) tuple per axis.

    Raises:
        ValueError: If the domain is not a rectangle, square, box, or cube,
            or if it is rotated.

    """
    name = type(domain).__name__.lower()
    if name not in ('rectangle', 'square', 'box', 'cube'):
        e_str = 'Periodic microstructures require a rectangular domain '
        e_str += '(Rectangle, Square, Box, or Cube), not ' + name + '.'
        raise ValueError(e_str)
    if not np.allclose(np.array(domain.matrix), np.eye(domain.n_dim)):
        e_str = 'Periodic microstructures require an axis-aligned domain.'
        raise ValueError(e_str)
    return [(float(lb), float(ub)) for lb, ub in domain.limits]
