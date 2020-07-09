# --------------------------------------------------------------------------- #
#                                                                             #
# Import Modules                                                              #
#                                                                             #
# --------------------------------------------------------------------------- #
from __future__ import division

import warnings

import numpy as np
import scipy.spatial.distance
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyquaternion import Quaternion

from microstructpy import _misc
from microstructpy.geometry.ellipse import Ellipse

__author__ = 'Kenneth (Kip) Hart'


# --------------------------------------------------------------------------- #
#                                                                             #
# Ellipsoid Class                                                             #
#                                                                             #
# --------------------------------------------------------------------------- #
class Ellipsoid(object):
    """A 3D Ellipsoid

    This class contains the data and functions for a 3D ellispoid.
    It is defined by its center, axes, and orientation.

    If multiple keywords are given for the shape of the ellipsoid, there
    is no guarantee for which keywords are used.

    Args:
        a (float): *(optional)* First semi-axis of ellipsoid. Default is 1.
        b (float): *(optional)* Second semi-axis of ellipsoid. Default is 1.
        c (float): *(optional)* Third semi-axis of ellipsoid. Default is 1.
        center (list): *(optional)* The ellipsoid center.
            Defaults to (0, 0, 0).
        axes (list): *(optional)* List of 3 semi-axes.
            Defaults to (1, 1, 1).
        size (float): *(optional)* The diameter of a sphere with equal volume.
            Defaults to 2.
        ratio_ab (float): *(optional)* The ratio of a to b.
        ratio_ac (float): *(optional)* The ratio of a to c.
        ratio_bc (float): *(optional)* The ratio of b to c.
        ratio_ba (float): *(optional)* The ratio of b to a.
        ratio_ca (float): *(optional)* The ratio of c to a.
        ratio_cb (float): *(optional)* The ratio of c to b.
        rot_seq (list): *(optional)* List of rotations (deg). Each element of
            the list should be an (axis, angle) tuple. The options for the
            axis are: 'x', 'y', 'z', 1, 2, or 3.
            For example::

                rot_seq = [('x', 10), (2, -20), ('z', 85), ('x', 21)]

        rot_seq_deg (list): *(optional)* Alias for ``rot_seq``, with degrees
            stated explicitly.
        rot_seq_rad (list): *(optional)* Same format as ``rot_seq``, except the
            angles are expressed in radians.
        matrix (numpy.ndarray): *(optional)* A 3x3 rotation matrix expressing
            the orientation of the ellipsoid. Defaults to the identity.
        position : *(optional)* Alias for ``center``.
        orientation: *(optional)* Alias for ``matrix``.
    """
    def __init__(self, **kwargs):
        # Position
        if 'center' in kwargs:
            self.center = kwargs['center']

        elif 'position' in kwargs:
            self.center = kwargs['position']

        else:
            self.center = (0, 0, 0)

        # Axes
        self.a = None
        self.b = None
        self.c = None

        ratio_ab = None
        ratio_ac = None
        ratio_bc = None

        size = None

        if 'a' in kwargs:
            self.a = kwargs['a']
        if 'b' in kwargs:
            self.b = kwargs['b']
        if 'c' in kwargs:
            self.c = kwargs['c']
        if 'ratio_ab' in kwargs:
            ratio_ab = kwargs['ratio_ab']
        elif 'ratio_ba' in kwargs:
            ratio_ab = 1 / kwargs['ratio_ba']
        if 'ratio_ac' in kwargs:
            ratio_ac = kwargs['ratio_ac']
        elif 'ratio_ca' in kwargs:
            ratio_ac = 1 / kwargs['ratio_ca']
        if 'ratio_bc' in kwargs:
            ratio_bc = kwargs['ratio_bc']
        elif 'ratio_cb' in kwargs:
            ratio_bc = 1 / kwargs['ratio_cb']
        if 'size' in kwargs:
            size = kwargs['size']
        elif 'volume' in kwargs:
            size = 2 * np.cbrt(3 * kwargs['volume'] / (4 * np.pi))
        if 'axes' in kwargs:
            self.a, self.b, self.c = kwargs['axes']

        if (self.a is not None) and (self.b is not None):
            ratio_ab = self.a / self.b
        if (self.a is not None) and (self.c is not None):
            ratio_ac = self.a / self.c
        if (self.b is not None) and (self.c is not None):
            ratio_bc = self.b / self.c

        if (ratio_ab is not None) and (ratio_bc is not None):
            ratio_ac = ratio_ab * ratio_bc
        if (ratio_ab is not None) and (ratio_ac is not None):
            ratio_bc = ratio_ac / ratio_ab
        if (ratio_ac is not None) and (ratio_bc is not None):
            ratio_ab = ratio_ac / ratio_bc

        for _ in range(2):
            if self.a is None:
                if (ratio_ab is not None) and (self.b is not None):
                    self.a = ratio_ab * self.b
                elif (ratio_ac is not None) and (self.c is not None):
                    self.a = ratio_ac * self.c

            if self.b is None:
                if (ratio_ab is not None) and (self.a is not None):
                    self.b = self.a / ratio_ab
                elif (ratio_bc is not None) and (self.c is not None):
                    self.b = ratio_bc * self.a

            if self.c is None:
                if (ratio_ac is not None) and (self.a is not None):
                    self.c = self.a / ratio_ac
                elif (ratio_bc is not None) and (self.b is not None):
                    self.c = self.b / ratio_bc

        if (self.a is not None) and (self.b is not None):
            ratio_ab = self.a / self.b
        if (self.a is not None) and (self.c is not None):
            ratio_ac = self.a / self.c
        if (self.b is not None) and (self.c is not None):
            ratio_bc = self.b / self.c

        if (ratio_ab is not None) and (ratio_bc is not None):
            ratio_ac = ratio_ab * ratio_bc
        if (ratio_ab is not None) and (ratio_ac is not None):
            ratio_bc = ratio_ac / ratio_ab
        if (ratio_ac is not None) and (ratio_bc is not None):
            ratio_ab = ratio_ac / ratio_bc

        if (size is not None):
            r_eff = 0.5 * size
            r3 = r_eff * r_eff * r_eff
            if (self.a is not None) and (self.b is not None):
                self.c = r3 / (self.a * self.b)
            elif (self.b is not None) and (self.c is not None):
                self.a = r3 / (self.b * self.c)
            elif (self.a is not None) and (self.c is not None):
                self.b = r3 / (self.a * self.c)
            elif (self.a is not None) and (ratio_bc is not None):
                r2 = r3 / self.a
                # r2 = b * c
                # r2 = c * ratio_bc * c = ratio_bc * c^2
                # c = sqrt(r2 / ratio_bc) and b = ratio_bc * c
                self.c = np.sqrt(r2 / ratio_bc)
                self.b = ratio_bc * self.c
            elif (self.b is not None) and (ratio_ac is not None):
                r2 = r3 / self.b
                self.c = np.sqrt(r2 / ratio_ac)
                self.a = ratio_ac * self.c
            elif (self.c is not None) and (ratio_ab is not None):
                r2 = r3 / self.c
                self.b = np.sqrt(r2 / ratio_ab)
                self.a = ratio_ab * self.b
            else:
                # r3 = a * b * c
                # r3 = a * (a / ratio_ab) * (a / ratio_ac)
                # r3 * ratio_ab * ratio_ac = a^3
                # a = r_eff * cbrt(ratio_ab * ratio_ac)

                self.a = r_eff * np.cbrt(ratio_ab * ratio_ac)
                self.b = self.a / ratio_ab
                self.c = self.a / ratio_ac

        if self.a is None:
            self.a = 1
        if self.b is None:
            self.b = 1
        if self.c is None:
            self.c = 1

        # Orientation
        if 'rot_seq' in kwargs:
            self.rot_seq = kwargs['rot_seq']
        elif 'rot_seq_deg' in kwargs:
            self.rot_seq = kwargs['rot_seq_deg']
        elif 'rot_seq_rad' in kwargs:
            rs_rad = kwargs['rot_seq_rad']
            self.rot_seq = [(ax, 180 * ang / np.pi) for ax, ang in rs_rad]
        elif ('matrix' in kwargs) or ('orientation' in kwargs):
            # adapted from:
            # https://www.learnopencv.com/rotation-matrix-to-euler-angles/
            if 'matrix' in kwargs:
                R = kwargs['matrix']
            else:
                R = kwargs['orientation']
            sy = np.sqrt(R[0][0] * R[0][0] + R[1][0] * R[1][0])

            if sy > 1e-6:
                x = np.arctan2(R[2][1], R[2][2])
                y = np.arctan2(-R[2][0], sy)
                z = np.arctan2(R[1][0], R[0][0])
            else:
                x = np.arctan2(-R[1][2], R[1][1])
                y = np.arctan2(-R[2][0], sy)
                z = 0

            axes = ('z', 'y', 'x')
            angs = [180 * ang / np.pi for ang in (z, y, x)]
            self.rot_seq = list(zip(axes, angs))
        else:
            self.rot_seq = []

    # ----------------------------------------------------------------------- #
    # Ellipsoid of Best Fit                                                   #
    # ----------------------------------------------------------------------- #
    def best_fit(self, points):
        """Find ellipsoid of best fit.

        This function takes a list of 3D points and computes the ellipsoid of
        best fit for the points. It uses a published algorithm to fit the
        ellipsoid, then attempts to define the axes in such a way that they
        most align with this ellipsoid's axes. [#turner]_

        Args:
            points (list): Points to fit ellipsoid

        Returns:
            Ellipsoid: The ellipsoid that best fits the points.

        .. [#turner] Turner, D. A., Anderson, I. J., Mason, J. C., and Cox,
            M. G., "An Algorithm for Fitting an Ellipsoid to Data," *National
            Physical Laboratory*, 1999, The United Kingdom.
            (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.36.2773&rep=rep1&type=pdf)
        """  # NOQA: E501
        pts = np.array(points)
        pts_mean = pts.mean(axis=0)
        trans_pts = pts - pts_mean

        x, y, z = trans_pts.T

        L = np.zeros((len(x), 9), dtype='float')
        L[:, 0] = x * x + y * y - 2 * z * z
        L[:, 1] = x * x - 2 * y * y + z * z
        L[:, 2] = 4 * x * y
        L[:, 3] = 2 * x * z
        L[:, 4] = 2 * y * z
        L[:, 5] = x
        L[:, 6] = y
        L[:, 7] = z
        L[:, 8] = 1

        e = x * x + y * y + z * z

        u, v, m, n, p, q, r, s, t = np.linalg.lstsq(L, e, rcond=None)[0]
        axx = 1 - u - v
        ayy = 1 - u + 2 * v
        azz = 1 + 2 * u - v
        axy = -4 * m
        axz = -2 * n
        ayz = -2 * p
        ax = - q
        ay = - r
        az = - s
        ac = - t

        hom_mat = np.array([[axx, 0.5 * axy, 0.5 * axz, 0.5 * ax],
                            [0.5 * axy, ayy, 0.5 * ayz, 0.5 * ay],
                            [0.5 * axz, 0.5 * ayz, azz, 0.5 * az],
                            [0.5 * ax, 0.5 * ay, 0.5 * az, ac]])

        cen = np.linalg.lstsq(-hom_mat[:3, :3], hom_mat[-1, :3], rcond=None)[0]
        glbl_cen = cen + pts_mean

        T = np.eye(4)
        T[-1, :3] = cen
        R = T.dot(hom_mat.dot(T.T))
        evals, evecs = np.linalg.eigh(- R[:3, :3] / R[-1, -1])

        if np.any(evals <= 0):
            w_str = 'Some of the eigenvalues of the quadratic form for'
            w_str += 'the ellipsoid are non-positive. Using absolute value, '
            w_str += 'but fit may be poor.'
            warnings.warn(w_str, RuntimeWarning)

        # extract ellipsoid parameters
        try:
            # reorganize the axes
            opt_axes = 1 / np.sqrt(np.abs(evals))
            axes = np.zeros(3)
            avail_axes = np.copy(opt_axes)
            rearr = np.zeros((3, 3))
            for ax_i, ax in enumerate(self.axes):
                opt_i = np.argmin(np.abs(avail_axes - ax))
                rearr[ax_i, opt_i] = 1
                avail_axes[opt_i] = np.inf
                axes[ax_i] = opt_axes[opt_i]
            new_evecs = evecs.dot(rearr)

            # settle the orientation ambiguity
            ori_dot = self.matrix.T.dot(new_evecs)
            vec_flip = np.sign(np.diagonal(ori_dot))
            ori_matrix = new_evecs.dot(np.diag(vec_flip))

        except AttributeError:
            axes = 1 / np.sqrt(np.abs(evals))
            ori_matrix = evecs

        return type(self)(center=glbl_cen, axes=axes, matrix=ori_matrix)

    # ----------------------------------------------------------------------- #
    # String and Representation Functions                                     #
    # ----------------------------------------------------------------------- #
    def __str__(self):
        str_str = 'center: ' + str(tuple(self.center)) + '\n'
        str_str += 'a: ' + str(self.a) + '\n'
        str_str += 'b: ' + str(self.b) + '\n'
        str_str += 'c: ' + str(self.c)
        if len(self.rot_seq) > 0:
            str_str += '\nrot_seq: ('
            for i, (ax, ang) in enumerate(self.rot_seq):
                str_str += '(' + str(ax) + ', ' + str(ang) + ')'
                if i < len(self.rot_seq) - 1:
                    str_str += ', '
                else:
                    str_str += ')'
        return str_str

    def __repr__(self):
        repr_str = 'Ellipsoid('
        repr_str += 'center=' + repr(tuple(self.center))
        repr_str += ', axes=' + repr(tuple([self.a, self.b, self.c]))
        rot_str = repr(tuple([tuple(r) for r in self.rot_seq]))
        repr_str += ', rot_seq=' + rot_str
        repr_str += ')'
        return repr_str

    # ----------------------------------------------------------------------- #
    # Size and Orientation Getters                                            #
    # ----------------------------------------------------------------------- #
    @property
    def size(self):
        """float: diameter of equivalent volume sphere"""
        return 2 * np.cbrt(self.a * self.b * self.c)

    @property
    def axes(self):
        """tuple: the 3 semi-axes of the ellipsoid"""
        return self.a, self.b, self.c

    @property
    def ratio_ab(self):
        """float: ratio of x-axis length to y-axis length"""
        return self.a / self.b

    @property
    def ratio_ba(self):
        """float: ratio of y-axis length to x-axis length"""
        return self.b / self.a

    @property
    def ratio_ac(self):
        """float: ratio of x-axis length to z-axis length"""
        return self.a / self.c

    @property
    def ratio_ca(self):
        """float: ratio of z-axis length to x-axis length"""
        return self.c / self.a

    @property
    def ratio_bc(self):
        """float: ratio of y-axis length to z-axis length"""
        return self.b / self.c

    @property
    def ratio_cb(self):
        """float: ratio of z-axis length to y-axis length"""
        return self.c / self.b

    @property
    def rot_seq_deg(self):
        """list: rotation sequence, with angles in degrees"""
        return self.rot_seq

    @property
    def rot_seq_rad(self):
        """list: rotation sequence, with angles in radiands"""
        return [(ax, np.pi * ang / 180) for ax, ang in self.rot_seq]

    @property
    def matrix(self):
        """numpy.ndarray: A 3x3 rotation matrix"""
        ax_dict = {1: 0, 2: 1, 3: 2, 'x': 0, 'y': 1, 'z': 2}

        q = Quaternion()
        for ax, ang in self.rot_seq_deg:
            vec = np.eye(3)[ax_dict[ax]]
            q *= Quaternion(axis=vec, degrees=ang)
        return q.rotation_matrix

    @property
    def orientation(self):
        """numpy.ndarray: A 3x3 rotation matrix"""
        return self.matrix

    # ----------------------------------------------------------------------- #
    # Quadratic Form Matrix                                                   #
    # ----------------------------------------------------------------------- #
    @property
    def matrix_quadform(self):
        """numpy.ndarray: Matrix of the quadratic form"""
        R = np.array(self.orientation)
        scl_mat = np.diag(1 / (np.array(self.axes) * np.array(self.axes)))
        return R.dot(scl_mat).dot(R.T)

    @property
    def matrix_quadeq(self):
        """numpy.ndarray: Matrix of the quadratic equation"""
        A33 = self.matrix_quadform
        grad_vec = - A33.dot(self.center)
        cen_vec = np.array(self.center).reshape(-1, 1)
        const = cen_vec.T.dot(A33.dot(cen_vec))[0, 0] - 1

        quad_mat = np.zeros((4, 4))
        quad_mat[:3, :3] = A33
        quad_mat[-1, :3] = grad_vec.T
        quad_mat[:3, -1] = grad_vec
        quad_mat[-1, -1] = const
        return quad_mat

    @property
    def coefficients(self):
        """tuple: coeffificients of equation,
        :math:`(A, B, C, D, E, F, G, H, K, L)` in
        :math:`Ax^2 + Bxy + Cxz + Dy^2 + Eyz + Fz^2 + Gx + Hy + Kz + L = 0`
        """
        quad_eq = self.matrix_quadeq
        A = quad_eq[0, 0]
        B = 2 * quad_eq[0, 1]
        C = 2 * quad_eq[0, 2]
        D = quad_eq[1, 1]
        E = 2 * quad_eq[1, 2]
        F = quad_eq[2, 2]
        G = 2 * quad_eq[3, 0]
        H = 2 * quad_eq[3, 1]
        K = 2 * quad_eq[3, 2]
        L = quad_eq[3, 3]
        return A, B, C, D, E, F, G, H, K, L

    # ----------------------------------------------------------------------- #
    # Number of Dimensions                                                    #
    # ----------------------------------------------------------------------- #
    @property
    def n_dim(self):
        """int: number of dimensions, 3"""
        return 3

    # ----------------------------------------------------------------------- #
    # Volume Property                                                         #
    # ----------------------------------------------------------------------- #
    @property
    def volume(self):
        """ float: volume of ellipsoid, :math:`V = \\frac{4}{3}\\pi a b c`"""
        return 4 * np.pi * self.a * self.b * self.c / 3

    @classmethod
    def volume_expectation(cls, **kwargs):
        r"""Expected value of volume.

        This function computes the expected value for the volume of an
        ellipsoid. The keyword arguments are the same as the input parameters
        for the class, :class:`microstructpy.geometry.Ellipsoid`. The
        values for these keywords can be either constants or distributions from
        the SciPy :mod:`scipy.stats` module.

        The expected value is computed by the following formula:

        .. math::

            \mathbb{E}[V] &= \mathbb{E}[\frac{4}{3}\pi A B C] \\
                            &= \frac{4}{3}\pi \mathbb{E}[A] \mathbb{E}[B] \mathbb{E}[C] \\
                            &= \frac{4}{3}\pi \mu_A \mu_B \mu_C

        If the ellisoid is specified by size and aspect ratios, then the
        expected volume is computed by:

        .. math::

            \mathbb{E}[V] &= \mathbb{E}[\frac{\pi}{6} S^3] \\
                            &= \frac{\pi}{6} (\mu_S^3 + 3 \mu_S \sigma_S^2 + \gamma_{1, S} \sigma_S^3)

        If the ellipsoid is specified using a combination of semi-axes and
        aspect ratios, then the expected volume is the mean of 1000 random
        samples:

        .. math::

            \mathbb{E}[V] \approx \frac{1}{n} \sum_{i=1}^n V_i

        where :math:`n=1000`.

        Args:
            **kwargs: Keyword arguments, see :class:`microstructpy.geometry.Ellipsoid`.

        Returns:
            float: Expected value of the volume of the sphere.

        """  # NOQA: E501
        # Check for size distribution
        if 'size' in kwargs:
            s_dist = kwargs['size']
            if type(s_dist) in (float, int):
                return 0.5 * np.pi * s_dist * s_dist * s_dist / 3
            return 0.5 * np.pi * s_dist.moment(3) / 3

        if 'volume' in kwargs:
            v_dist = kwargs['volume']
            try:
                v_exp = v_dist.moment(1)
            except AttributeError:
                v_exp = v_dist
            return v_exp

        # check for a, b, and c distribution
        try:
            exp_vol = 4 * np.pi / 3
            for kw in ('a', 'b', 'c'):
                dist = kwargs[kw]
                try:
                    mu = dist.moment(1)
                except AttributeError:
                    mu = dist
                exp_vol *= mu
            return exp_vol
        except KeyError:
            pass

        # Use Monte Carlo to determine expected volume
        n_trials = 1000
        kws = set(kwargs.keys()) - set(_misc.ori_kws)

        total_vol = 0
        for i in range(n_trials):
            params = {}
            for kw in kws:
                try:
                    params[kw] = kwargs[kw].rvs()
                except AttributeError:
                    params[kw] = kwargs[kw]
            total_vol += Ellipsoid(**params).volume
        avg_vol = total_vol / n_trials
        return avg_vol

    # ----------------------------------------------------------------------- #
    # Bounding Circles                                                        #
    # ----------------------------------------------------------------------- #
    @property
    def bound_max(self):
        """tuple: maximum bounding sphere, (x, y, z, r)"""
        r = max(self.a, self.b, self.c)
        return tuple(list(self.center) + [r])

    @property
    def bound_min(self):
        """tuple: minimum interior sphere, (x, y, z, r)"""
        r = min(self.a, self.b, self.c)
        return tuple(list(self.center) + [r])

    # ----------------------------------------------------------------------- #
    # Sphere Approximation of Ellipsoid                                       #
    # ----------------------------------------------------------------------- #
    def approximate(self, x1=None):
        """Approximate Ellipsoid with Spheres

        This function approximates the ellipsoid by a set of spheres.
        It does so by approximating the x-z and y-z elliptical cross sections
        with circles, then scaling those circles and promoting them to spheres.

        See the documentation for
        :meth:`microstructpy.geometry.Ellipse.approximate` for more details.

        Args:
            x1 (float): *(optional)* Center position of the first sphere.
                Default is 0.75x the minimum semi-axis.

        Returns:
            numpy.ndarray: An Nx4 list of the (x, y, z, r) data of the spheres
            that approximate the ellipsoid.
        """
        if (self.a == self.b) and (self.a == self.c):
            return np.array([self.bound_max])

        if x1 is None:
            x1 = 0.25 * min(self.axes)

        # Perform approximation such that a > b > c
        if (self.a >= self.b) and (self.b >= self.c):
            a = self.a
            b = self.b
            c = self.c
            inds = [0, 1, 2]
        elif (self.a >= self.c) and (self.c >= self.b):
            a = self.a
            b = self.c
            c = self.b
            inds = [0, 2, 1]
        elif (self.b >= self.a) and (self.a >= self.c):
            a = self.b
            b = self.a
            c = self.c
            inds = [1, 0, 2]
        elif (self.c >= self.a) and (self.a >= self.b):
            a = self.c
            b = self.a
            c = self.b
            inds = [0, 2, 1]
        else:
            a = self.c
            b = self.b
            c = self.a
            inds = [2, 1, 0]

        # Prolate Ellipsoid
        if np.isclose(b, c):
            circs_xz = Ellipse(a=a, b=c).approximate()
            spheres = np.insert(circs_xz, 1, 0, axis=1)[:, np.append(inds, 3)]
            cens = np.array(self.center) + spheres[:, :-1].dot(self.matrix.T)
            spheres[:, :-1] = cens
            return spheres

        # Points in First Octant
        surface_pts = _ellipse_pts(a, b, c)
        plane_mask = np.isclose(surface_pts[:, -1], 0)

        # Circle centers
        circs_xz = Ellipse(a=a, b=c).approximate(x1)
        x_grid = np.append(circs_xz[circs_xz[:, 0] >= 0, 0][:-1], a)

        circs_yz = Ellipse(a=b, b=c).approximate(x1)
        y_grid = np.append(circs_yz[circs_yz[:, 0] >= 0, 0][:-1], b)

        xx, yy = np.meshgrid(x_grid / a, y_grid / b)

        # Conformal mapping
        xx_mesh = a * xx * np.sqrt(1 - 0.5 * yy * yy)
        yy_mesh = b * yy * np.sqrt(1 - 0.5 * xx * xx)

        xx_inner = xx_mesh[:-1, :-1].flatten()
        yy_inner = yy_mesh[:-1, :-1].flatten()
        pts_inner = np.array([xx_inner, yy_inner, 0 * xx_inner]).T

        # Interior spheres
        inner_dists = scipy.spatial.distance.cdist(pts_inner, surface_pts)

        nearest_ind = np.argmin(inner_dists, axis=1)
        nearest_dist = inner_dists[np.arange(len(nearest_ind)), nearest_ind]
        in_plane = plane_mask[nearest_ind]

        cens_inner = pts_inner[~in_plane]
        rads_inner = nearest_dist[~in_plane]

        # Exterior spheres
        xx_outer = np.concatenate((xx_mesh[:, -1], xx_mesh[-1, :-1]))
        yy_outer = np.concatenate((yy_mesh[:, -1], yy_mesh[-1, :-1]))

        u_outer = np.arctan2(yy_outer / b, xx_outer / a)
        n_x = b * np.cos(u_outer)
        n_y = a * np.sin(u_outer)
        norm_vec = np.array([n_x, n_y]).T
        norm_vec /= np.linalg.norm(norm_vec, axis=1).reshape(-1, 1)
        rads_outer = c * c * np.sqrt(n_x * n_x + n_y * n_y) / (a * b)

        rel_pos_outer = rads_outer.reshape(-1, 1) * norm_vec
        cens_outer = np.array([xx_outer, yy_outer]).T - rel_pos_outer
        cens_outer = np.hstack((cens_outer, 0 * xx_outer.reshape(-1, 1)))

        # First octant spheres
        cens = np.vstack((cens_inner, cens_outer))
        rads = np.concatenate((rads_inner, rads_outer))
        spheres = np.hstack((cens, rads.reshape(-1, 1)))

        # All octants
        for dim in range(3):
            refl_spheres = np.copy(spheres)
            mask = spheres[:, dim] > 0
            refl_spheres[mask, dim] *= -1
            spheres = np.vstack((spheres, refl_spheres[mask]))
        spheres = spheres[:, np.append(inds, 3)]
        cens = np.array(self.center) + spheres[:, :-1].dot(self.matrix.T)
        spheres[:, :-1] = cens
        return spheres

    # ----------------------------------------------------------------------- #
    # Plot Function                                                           #
    # ----------------------------------------------------------------------- #
    def plot(self, **kwargs):
        """Plot the ellipsoid.

        This function uses the :meth:`mpl_toolkits.mplot3d.Axes3D.plot_surface`
        method to add an ellipsoid to the current axes. The keyword arguments
        are passes through to the plot_surface function.

        Args:
            **kwargs (dict): Keyword arguments for matplotlib.

        """  # NOQA: E501
        if len(plt.gcf().axes) == 0:
            ax = plt.axes(projection=Axes3D.name)
        else:
            ax = plt.gca()

        u = np.linspace(0, 2 * np.pi, 11)
        cv = np.linspace(-1, 1, 12)
        uu, cvv = np.meshgrid(u, cv)
        svv = np.sin(np.arccos(cvv))
        grid_shape = uu.shape

        a = self.a
        b = self.b
        c = self.c

        xp = a * np.cos(uu) * svv
        yp = b * np.sin(uu) * svv
        zp = c * cvv

        pts = np.array([xp.reshape(1, -1).flatten(),
                        yp.reshape(1, -1).flatten(),
                        zp.reshape(1, -1).flatten()])
        rot_pts = self.orientation.dot(pts)

        xr = rot_pts[0].reshape(grid_shape)
        yr = rot_pts[1].reshape(grid_shape)
        zr = rot_pts[2].reshape(grid_shape)

        xc, yc, zc = self.center
        xx = xc + xr
        yy = yc + yr
        zz = zc + zr

        mod_kwargs = {}
        for key, val in kwargs.items():
            if key == 'facecolors' and type(val) != list:
                mod_kwargs['color'] = val
            else:
                mod_kwargs[key] = val
        ax.plot_surface(xx, yy, zz, **mod_kwargs)

    # ----------------------------------------------------------------------- #
    # Limits                                                                  #
    # ----------------------------------------------------------------------- #
    @property
    def limits(self):
        """list: List of (lower, upper) bounds for the bounding box"""
        if np.all(np.isclose(self.matrix, np.eye(3))):
            ax = np.array(self.axes)
            cen = np.array(self.center)
            return [(x - r, x + r) for x, r in zip(cen, ax)]

        n = 4
        u = np.linspace(0, 2 * np.pi, 1 + 4 * n)
        cv = np.linspace(-1, 1, 1 + 2 * n)
        uu, cvv = np.meshgrid(u, cv)
        svv = np.sin(np.arccos(cvv))

        xp = self.a * np.cos(uu) * svv
        yp = self.b * np.sin(uu) * svv
        zp = self.c * cvv

        pts = np.array([xp.flatten(), yp.flatten(), zp.flatten()])
        r_pts = self.matrix.dot(pts)
        lbs = r_pts.min(axis=-1) + np.array(self.center)
        ubs = r_pts.max(axis=-1) + np.array(self.center)
        return list(zip(lbs, ubs))

    @property
    def sample_limits(self):
        """list: List of (lower, upper) bounds for the sampling region"""
        return self.limits

    # ----------------------------------------------------------------------- #
    # Within Test                                                             #
    # ----------------------------------------------------------------------- #
    def within(self, points):
        """Test if points are within ellipsoid.

        This function tests whether a point or set of points are within the
        ellipsoid. For the set of points, a list of booleans is returned to
        indicate which points are within the ellipsoid.

        Args:
            points (list or numpy.ndarray): Point or list of points.

        Returns:
            bool or numpy.ndarray: Set to True for points in geometry.

        """
        pts = np.array(points)
        single_pt = pts.ndim == 1
        if single_pt:
            pts = pts.reshape(1, -1)

        rel_pos = pts - np.array(self.center)
        rot_pos = rel_pos.dot(self.orientation)
        scl_pos = rot_pos / np.array(self.axes).reshape(1, -1)

        sq_dist = np.sum(scl_pos * scl_pos, axis=-1)

        mask = sq_dist <= 1
        if single_pt:
            return mask[0]
        else:
            return mask

    # ----------------------------------------------------------------------- #
    # Reflect                                                                 #
    # ----------------------------------------------------------------------- #
    def reflect(self, points):
        """Reflect points across surface.

        This function reflects a point or set of points across the surface
        of the ellipsoid. Points at the center of the ellipsoid are not
        reflected.

        Args:
            points (list or numpy.ndarray): Points to reflect.

        Returns:
            numpy.ndarray: Reflected points.

        """
        pts = np.array(points)
        single_pt = pts.ndim == 1
        if single_pt:
            pts = pts.reshape(1, -1)

        rel_pos = pts - np.array(self.center)
        rot_pos = rel_pos.dot(self.orientation)
        scl_pos = rot_pos / np.array(self.axes).reshape(1, -1)

        dist = np.sqrt(np.sum(scl_pos * scl_pos, axis=-1))
        mask = dist > 0
        if not np.any(mask):
            return np.array([])

        new_dist = 2 - dist[mask]
        scl = new_dist / dist[mask]

        new_scl_pos = scl_pos[mask] * scl
        new_rel_pos = new_scl_pos.dot(self.orientation.T)
        new_pos = new_rel_pos + np.array(self.center)

        if single_pt:
            return new_pos[0]
        else:
            return new_pos


def _ellipse_arc(a, b, n):
    horiz_x = a * np.linspace(1, -1, n)
    verti_y = np.linspace(0, 2 * b, n)

    t_denom = (a - horiz_x) * verti_y + 4 * a * b
    t_numer = 4 * a * b
    t = t_numer / t_denom

    x_arc = (2 * t - 1) * a
    y_arc = t * verti_y

    return np.array([x_arc, y_arc]).T


def _ellipse_pts(a, b, c, n=81):
    arc_xy = _ellipse_arc(a, b, n)
    arc_xz = _ellipse_arc(a, c, n)
    arc_yz = _ellipse_arc(b, c, n)

    ii_xy, ii_z = np.meshgrid(np.arange(n - 1), np.arange(n - 1))
    theta_xy = np.arctan2(arc_xy[:, 1] / b, arc_xy[:, 0] / a)
    tt_xy = theta_xy[ii_xy]

    lat_xz = np.arctan2(arc_xz[:, 1] / c, arc_xz[:, 0] / a)
    lat_yz = np.arctan2(arc_yz[:, 1] / c, arc_yz[:, 0] / b)

    ll_f_xz = lat_xz[ii_z] * np.cos(tt_xy)
    ll_f_yz = lat_yz[ii_z] * np.sin(tt_xy)
    latlat = np.sqrt(ll_f_xz * ll_f_xz + ll_f_yz * ll_f_yz)

    xx = arc_xy[:, 0][ii_xy] * np.cos(latlat)
    yy = arc_xy[:, 1][ii_xy] * np.cos(latlat)
    zz = c * np.sin(latlat)

    pts = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    pts = np.append(pts, [[0, 0, c]], axis=0)
    return pts
