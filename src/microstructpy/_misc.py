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


def periodic_bounds(points, per_axes):
    """(lower, upper) bounds of the domain of a periodic mesh.

    The points of a mesh that fills a rectangular domain span the domain,
    so its bounds are the extents of the points.

    Args:
        points (list or numpy.ndarray): The points of the mesh.
        per_axes (list): Periodicity flag of each axis.

    Returns:
        list: One (lower, upper) tuple per axis.

    """
    pts = np.array(points, dtype='float')
    return [(float(lb), float(ub)) for lb, ub in
            zip(pts.min(axis=0), pts.max(axis=0))]


def pair_periodic_points(points, per_axes, dom_lims, rel_tol=1e-8):
    """Pair the points on opposite periodic faces of a domain.

    For each periodic axis, every point on the lower face is matched with
    its image on the upper face, and the coordinates of the pair are
    snapped so that the image is exactly the point translated by the
    domain length.

    Args:
        points (list or numpy.ndarray): The points.
        per_axes (list): Periodicity flag of each axis.
        dom_lims (list): (lower, upper) bounds of the domain, per axis.
        rel_tol (float): Matching tolerance, relative to the largest
            domain length.

    Returns:
        tuple: The snapped points (numpy.ndarray) and a dictionary that
        maps each periodic axis to a list of (lower, upper) point numbers.

    Raises:
        ValueError: If a point on a periodic face has no image on the
            opposite face.

    """
    pts = np.array(points, dtype='float')
    n_dim = pts.shape[1]
    lengths = [ub - lb for lb, ub in dom_lims]
    tol = rel_tol * max(lengths)

    pairs = {}
    for axis, flag in enumerate(per_axes):
        if not flag:
            continue
        lb, ub = dom_lims[axis]
        shift = np.zeros(n_dim)
        shift[axis] = ub - lb
        others = [i for i in range(n_dim) if i != axis]

        low = np.nonzero(np.abs(pts[:, axis] - lb) <= tol)[0]
        high = np.nonzero(np.abs(pts[:, axis] - ub) <= tol)[0]
        if len(low) != len(high):
            e_str = 'The periodic faces along axis ' + str(axis)
            e_str += ' have different numbers of points ('
            e_str += str(len(low)) + ' and ' + str(len(high)) + ').'
            raise ValueError(e_str)

        axis_pairs = []
        if len(low) > 0:
            rel = pts[low][:, None, :][:, :, others]
            rel = rel - pts[high][None, :, :][:, :, others]
            dists = np.sqrt(np.sum(rel * rel, axis=-1))
            for i_low, kp_low in enumerate(low):
                i_high = int(np.argmin(dists[i_low]))
                if dists[i_low, i_high] > tol:
                    e_str = 'Point ' + str(kp_low) + ' on the lower '
                    e_str += 'periodic face of axis ' + str(axis)
                    e_str += ' has no image on the upper face.'
                    raise ValueError(e_str)
                dists[:, i_high] = np.inf  # one-to-one
                kp_high = int(high[i_high])
                pts[kp_low, axis] = lb
                pts[kp_high] = pts[kp_low] + shift
                axis_pairs.append((int(kp_low), kp_high))
        pairs[axis] = axis_pairs
    return pts, pairs


def pair_periodic_facets(facets, point_pairs):
    """Pair the facets lying on opposite periodic faces.

    Args:
        facets (list): Facets (lists of point numbers).
        point_pairs (dict): Output of :func:`pair_periodic_points`.

    Returns:
        dict: Maps each periodic axis to a list of (lower, upper) facet
        numbers.

    Raises:
        ValueError: If a facet on a periodic face has no image.

    """
    pairs = {}
    for axis, axis_pairs in point_pairs.items():
        kp_map = dict(axis_pairs)
        low_set = set(kp_map)
        high_set = set(kp_map.values())
        high_facets = {}
        for f_num, facet in enumerate(facets):
            if len(facet) > 0 and all([kp in high_set for kp in facet]):
                high_facets[frozenset(facet)] = f_num
        f_pairs = []
        for f_num, facet in enumerate(facets):
            if len(facet) == 0 or not all([kp in low_set for kp in facet]):
                continue
            key = frozenset([kp_map[kp] for kp in facet])
            if key not in high_facets:
                e_str = 'Facet ' + str(f_num) + ' on the lower periodic'
                e_str += ' face of axis ' + str(axis) + ' has no image'
                e_str += ' on the upper face.'
                raise ValueError(e_str)
            f_pairs.append((f_num, high_facets[key]))
        pairs[axis] = f_pairs
    return pairs


def unwrap_points(points, center, per_axes, dom_lims):
    """Translate points by domain lengths to the image nearest a center.

    Used to reassemble a grain that a periodic domain splits into pieces:
    along each periodic axis, every point is moved by a multiple of the
    domain length so that it lies within half a length of the center.

    Args:
        points (list or numpy.ndarray): The points.
        center (list or numpy.ndarray): The reference point (e.g. the seed
            position).
        per_axes (list): Periodicity flag of each axis.
        dom_lims (list): (lower, upper) bounds of the domain, per axis.

    Returns:
        numpy.ndarray: The unwrapped points.

    """
    pts = np.array(points, dtype='float')
    cen = np.array(center, dtype='float')
    for axis, flag in enumerate(per_axes):
        if not flag:
            continue
        length = dom_lims[axis][1] - dom_lims[axis][0]
        n_shift = np.round((cen[axis] - pts[:, axis]) / length)
        pts[:, axis] += n_shift * length
    return pts
