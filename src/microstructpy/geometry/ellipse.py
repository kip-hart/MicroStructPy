# --------------------------------------------------------------------------- #
#                                                                             #
# Import Modules                                                              #
#                                                                             #
# --------------------------------------------------------------------------- #
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

__author__ = 'Kenneth (Kip) Hart'


# --------------------------------------------------------------------------- #
#                                                                             #
# Ellipse Class                                                               #
#                                                                             #
# --------------------------------------------------------------------------- #
class Ellipse(object):
    r"""A 2-D ellipse geometry.

    This class contains a 2-D ellipse. It is defined by a center point, axes
    and an orientation.

    Args:
        center (list, optional): The ellipse center.
            Defaults to (0, 0).
        axes (list, optional): A 2-element list of semi-axes.
            Defaults to [1, 1].
        size (float, optional): The diameter of a circle with equivalent area.
            Defaults to 1.
        aspect_ratio (float, optional): The ratio of x-axis to y-axis length
            Defaults to 1.
        angle (float, optional): The rotation angle, in degrees.
        angle_deg (float, optional): The rotation angle, in degrees.
        angle_rad (float, optional): The rotation angle, in radians.
        matrix (2x2 array, optional): The rotation matrix.
        a : Alias for *axes[0]*.
        b : Alias for *axes[1]*.
        angle_deg: Alias for *angle*.
        angel_rad: Alias for *pi \* angle / 180*.
        orientation: Alias for *matrix*.
    """
    # ----------------------------------------------------------------------- #
    # Constructor                                                             #
    # ----------------------------------------------------------------------- #
    def __init__(self, **kwargs):
        # position
        if 'center' in kwargs:
            self.center = kwargs['center']

        elif 'position' in kwargs:
            self.center = kwargs['position']

        else:
            self.center = (0, 0)

        # axes
        if ('a' in kwargs) and ('b' in kwargs):
            assert kwargs['a'] > 0
            assert kwargs['b'] > 0

            self.a = kwargs['a']
            self.b = kwargs['b']

        elif 'axes' in kwargs:
            self.a, self.b = kwargs['axes']

        elif ('size' in kwargs) and ('aspect_ratio' in kwargs):
            assert kwargs['size'] > 0
            assert kwargs['aspect_ratio'] > 0

            # Derivation for converting (r, k) into (a, b)
            # Area formula: pi*a*b = pi*r^2
            # Aspect ratio: k = a/b
            #
            # Plug AR into area: k*b^2 = r^2
            # Solve for b: b = r/sqrt(k)
            # Solve for a: a = k * b

            r = 0.5 * kwargs['size']
            k = kwargs['aspect_ratio']

            self.b = r / np.sqrt(k)
            self.a = k * self.b

        elif ('a' in kwargs) and ('size' in kwargs):
            assert kwargs['a'] > 0
            assert kwargs['size'] > 0

            r = 0.5 * kwargs['size']

            self.a = kwargs['a']
            self.b = (r * r) / self.a

        elif ('b' in kwargs) and ('size' in kwargs):
            assert kwargs['b'] > 0
            assert kwargs['size'] > 0

            r = 0.5 * kwargs['size']

            self.b = kwargs['b']
            self.a = (r * r) / self.b
        elif ('a' in kwargs) and ('aspect_ratio' in kwargs):
            assert kwargs['a'] > 0
            assert kwargs['aspect_ratio'] > 0

            self.a = kwargs['a']
            self.b = self.a / kwargs['aspect_ratio']
        elif ('b' in kwargs) and ('aspect_ratio' in kwargs):
            assert kwargs['b'] > 0
            assert kwargs['aspect_ratio'] > 0

            self.b = kwargs['b']
            self.a = self.b * kwargs['aspect_ratio']

        else:
            self.a = 1
            self.b = 1

        # orientation
        if 'angle_deg' in kwargs:
            self.angle = kwargs['angle_deg']

        elif 'angle_rad' in kwargs:
            self.angle = 180 * kwargs['angle_rad'] / np.pi

        elif 'angle' in kwargs:
            self.angle = kwargs['angle']

        elif 'matrix' in kwargs:
            ct = kwargs['matrix'][0][0]
            st = kwargs['matrix'][1][0]
            self.angle = 180 * np.arctan2(st, ct) / np.pi

        elif 'orientation' in kwargs:
            ct = kwargs['orientation'][0][0]
            st = kwargs['orientation'][1][0]
            self.angle = 180 * np.arctan2(st, ct) / np.pi

        else:
            self.angle = 0

    # ----------------------------------------------------------------------- #
    # Best Fit Function                                                       #
    # ----------------------------------------------------------------------- #
    def best_fit(self, points):
        r"""Find ellipse of best fit for points

        This function computes the ellipse of best fit for a set of points.
        It is heavily adapted from the `least-squares-ellipse-fitting`_
        repository on GitHub. This repository implements a published fitting
        algorithm in Python. [#halir]_

        The current instance of the class is used as an initial guess for
        the ellipse of best fit. Since an ellipse can be expressed multiple
        ways (e.g. rotate 90 degrees and flip the axes), this initial guess
        is used to choose from the multiple parameter sets.

        Args:
            points (list): List of points to fit.

        Returns:
            .Ellipse: An instance of the class that best fits the points.

        .. _`least-squares-ellipse-fitting`: https://github.com/bdhammel/least-squares-ellipse-fitting

        .. [#halir] Halir, R., Flusser, J., "Numerically Stable Direct Least
          Squares Fitting of Ellipses," *6th International Conference in Central
          Europe on Computer Graphics and Visualization*, Vol. 98, 1998.
          (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1.7559&rep=rep1&type=pdf)

        """  # NOQA: E501
        # Unpack the input points
        pts = np.array(points, dtype='float')
        x, y = pts.T

        # Quadratic part of design matrix
        D1 = np.mat(np.vstack([x * x, x * y, y * y])).T

        # Linear part of design matrix
        D2 = np.mat(np.vstack([x, y, np.ones(len(x))])).T

        # Scatter matrix
        S1 = D1.T * D1
        S2 = D1.T * D2
        S3 = D2.T * D2

        # Constraint matrix
        C1inv = np.mat([[0, 0, 0.5], [0, -1, 0], [0.5, 0, 0]])

        # Reduced scatter matrix
        M = C1inv * (S1 - S2 * S3.I * S2.T)

        # Find eigenvalues
        _, evec = np.linalg.eig(M)

        # Mask
        cond = 4 * np.multiply(evec[0, :], evec[2, :])
        cond -= np.multiply(evec[1, :], evec[1, :])
        a1 = evec[:, np.nonzero(cond.A > 0)[1]]

        a2 = -S3.I * S2.T * a1

        # Coefficients
        a = a1[0, 0]
        b = 0.5 * a1[1, 0]
        c = a1[2, 0]

        d = 0.5 * a2[0, 0]
        f = 0.5 * a2[1, 0]
        g = a2[2, 0]

        # Center of ellipse
        k = b * b - a * c
        xc = (c * d - b * f) / k
        yc = (a * f - b * d) / k

        # Semi-axes lengths
        numer = a * f * f
        numer += c * d * d
        numer += g * b * b
        numer -= 2 * b * d * f
        numer -= a * c * g
        numer *= 2

        tan2 = 2 * b / (a - c)
        sq_val = np.sqrt(1 + tan2 * tan2)
        denom1 = k * ((c - a) * sq_val - (c + a))
        denom2 = k * ((a - c) * sq_val - (c + a))
        width = np.sqrt(numer / denom1)
        height = np.sqrt(numer / denom2)

        # Angle of rotation
        phi = 0.5 * np.arctan(tan2)

        # Find pair closest to self
        s = np.sin(phi)
        c = np.cos(phi)
        rot = np.array([[c, -s], [s, c]])

        x_ax_seed = np.array(self.matrix)[:, 0]
        x_dot, y_dot = rot.T.dot(x_ax_seed)

        if np.abs(x_dot) > np.abs(y_dot):
            if x_dot > 0:
                x_ax_fit = rot[:, 0]
            else:
                x_ax_fit = - rot[:, 0]

            a = width
            b = height
        else:
            if y_dot > 0:
                x_ax_fit = rot[:, 1]
            else:
                x_ax_fit = - rot[:, 1]

            a = height
            b = width

        ang_diff = np.arcsin(np.cross(x_ax_seed, x_ax_fit))
        ang_rad = self.angle_rad + ang_diff

        return type(self)(center=(xc, yc), a=a, b=b, angle_rad=ang_rad)

    # ----------------------------------------------------------------------- #
    # String and Representation Functions                                     #
    # ----------------------------------------------------------------------- #
    def __str__(self):
        str_str = 'center: ' + str(tuple(self.center)) + '\n'
        str_str += 'a: ' + str(self.a) + '\n'
        str_str += 'b: ' + str(self.b) + '\n'
        str_str += 'angle: ' + str(self.angle)
        return str_str

    def __repr__(self):
        repr_str = 'Ellipse('
        repr_str += 'center=' + repr(tuple(self.center))
        repr_str += ', a=' + repr(self.a) + ', b=' + repr(self.b)
        repr_str += ', angle=' + repr(self.angle)
        repr_str += ')'
        return repr_str

    # ----------------------------------------------------------------------- #
    # Size and Orientation Getters                                            #
    # ----------------------------------------------------------------------- #
    @property
    def size(self):
        """float: diameter of equivalent area circle"""
        return 2 * np.sqrt(self.a * self.b)

    @property
    def aspect_ratio(self):
        """float: ratio of x-axis length to y-axis length"""
        return self.a / self.b

    @property
    def axes(self):
        """ tuple: list of semi-axes."""
        return self.a, self.b

    @property
    def angle_deg(self):
        """float: rotation angle, in degrees"""
        return self.angle

    @property
    def angle_rad(self):
        """float: rotation angle, in radians"""
        return self.angle * np.pi / 180

    @property
    def matrix(self):
        """numpy.ndarray: rotation matrix"""
        ct = np.cos(self.angle_rad)
        st = np.sin(self.angle_rad)
        return np.array([[ct, -st], [st, ct]])

    @property
    def orientation(self):
        """numpy.ndarray: rotation matrix"""
        return self.matrix

    # ----------------------------------------------------------------------- #
    # Number of Dimensions                                                    #
    # ----------------------------------------------------------------------- #
    @property
    def n_dim(self):
        """int: number of dimensions, 2"""
        return 2

    # ----------------------------------------------------------------------- #
    # Area Property                                                           #
    # ----------------------------------------------------------------------- #
    @property
    def area(self):
        r""" float: area of ellipse, :math:`A = \pi a b`"""
        return np.pi * self.a * self.b

    @property
    def volume(self):
        r""" float: alias for area"""
        return self.area

    # ----------------------------------------------------------------------- #
    # Expected Area                                                           #
    # ----------------------------------------------------------------------- #
    @classmethod
    def area_expectation(cls, **kwargs):
        r"""Expected value of area.

        This function computes the expected value for the area of an ellipse.
        The keyword arguments are the same as the input parameters of the
        class. The keyword values can be either constants (ints or floats) or
        `scipy.stats`_ distributions.

        If an ellipse is specified by size, the expected value is computed
        as follows.

        .. math::

            \begin{align}
            \mathbb{E}[A] &= \frac{\pi}{4} \mathbb[S^2] \\
                          &= \frac{\pi}{4} (\mu_S^2 + \sigma_S^2)
            \end{align}

        If the ellipse is specified by independent distributions for each
        semi-axis, the expected value is computed by:

        .. math::

            \mathbb{E}[A] = \pi \mathbb{E}[A B] = \pi \mu_A \mu_B

        If the ellipse is specified by the second semi-axis and the aspect
        ratio, the expected value is computed by:

        .. math::

            \begin{align}
            \mathbb{E}[A] &= \pi \mathbb{E}[K B^2] \\
                          &= \pi \mu_K (\mu_B^2 + \sigma_B^2)
            \end{align}

        Finally, if the ellipse is specified by the first semi-axis and the
        aspect ratio, the expected value is computed by Monte Carlo:

        .. math::

            \begin{align}
            \mathbb{E}[A] &= \pi \mathbb{E}\left[\frac{A^2}{K}\right] \\
                          &\approx \frac{\pi}{n} \sum_{i=1}^n \frac{A_i}{K_i}
            \end{align}

        where :math:`n=1000`.

        Args:
            **kwargs: Keyword arguments, see
                :class:`microstructpy.geometry.Ellipse`.

        Returns:
            float: Expected value of the area of the ellipse.

        .. _`scipy.stats`: https://docs.scipy.org/doc/scipy/reference/stats.html
        """  # NOQA: E501
        if 'size' in kwargs:
            s_dist = kwargs['size']

            if type(s_dist) in (float, int):
                return 0.25 * np.pi * s_dist * s_dist
            else:
                return 0.25 * np.pi * s_dist.moment(2)

        elif ('a' in kwargs) and ('b' in kwargs):
            exp = np.pi
            for kw in ('a', 'b'):
                dist = kwargs[kw]
                if type(dist) in (float, int):
                    mu = dist
                else:
                    mu = dist.moment(1)
                exp *= mu
            return exp
        elif ('b' in kwargs) and ('aspect_ratio' in kwargs):
            exp = np.pi
            try:
                exp *= kwargs['b'].moment(2)
            except AttributeError:
                exp *= kwargs['b'] * kwargs['b']

            try:
                exp *= kwargs['aspect_ratio'].moment(1)
            except AttributeError:
                exp *= kwargs['aspect_ratio']
            return exp
        elif ('a' in kwargs) and ('aspect_ratio' in kwargs):
            n = 1000
            try:
                a = kwargs['a'].rvs(size=n)
            except AttributeError:
                a = np.full(n, kwargs['a'])

            try:
                k = kwargs['aspect_ratio'].rvs(size=n)
            except AttributeError:
                k = np.full(n, kwargs['aspect_ratio'])
            return np.pi * np.mean((a * a) / k)
        else:
            e_str = 'Could not calculate expected area from keywords '
            e_str += str(kwargs.keys()) + '.'
            raise KeyError(e_str)

    # ----------------------------------------------------------------------- #
    # Bounding Circles                                                        #
    # ----------------------------------------------------------------------- #
    @property
    def bound_max(self):
        """tuple: maximum bounding circle of ellipse, (x, y, r)"""
        r = max(self.a, self.b)
        return tuple(list(self.center) + [r])

    @property
    def bound_min(self):
        """tuple: minimum interior circle of ellipse, (x, y, r)"""
        r = min(self.a, self.b)
        return tuple(list(self.center) + [r])

    # ----------------------------------------------------------------------- #
    # Circle Approximation of Ellipse                                         #
    # ----------------------------------------------------------------------- #
    def approximate(self, x1=None):
        """Approximate ellipse with a set of circles.

        This function converts an ellipse into a set of circles.
        It implements a published algorithm. [#ilin]_

        Args:
            x1 (float): Center position of first circle.

        Returns:
            numpy.ndarray: An Nx3 list of the (x, y, r) data of each circle
            approximating the ellipse.

        Raises:
            AssertionError: Thrown if max(a, b) < x1.

        .. [#ilin] Ilin, D.N., and Bernacki, M., "Advancing Layer Algorithm
            of Dense Ellipse Packing for Generating Statistically Equivalent
            Polygonal Structures," Granular Matter, vol. 18(3), pp. 43, 2016.

        """
        if x1 is None:
            x1 = 0.5 * min(self.a, self.b)

        flip = self.a < self.b
        if flip:
            a = self.b
            b = self.a
        else:
            a = self.a
            b = self.b

        if a == b:
            return np.append(self.center, a).reshape(1, -1)

        a_str = 'Center of first circle, x1=' + str(x1) + ', must be less than'
        a_str += ' semi-major axis, a=' + str(a)
        assert x1 < a, a_str

        R_N = b * b / a  # Eq. 8
        x_N = a - R_N  # Eq 9

        def R_i(x):
            return b * np.sqrt(1 - x * x / (a * a - b * b))  # Eq. 6

        def y_i(x):
            ratio = x / a
            return b * np.sqrt(1 - ratio * ratio)  # Eq. 1

        circles = [(0, 0, b)]
        y_vals = [b]

        adding_circles = x1 < x_N
        if adding_circles:
            circles.append((x1, 0, R_i(x1)))
            y_vals.append(y_i(x1))

        while adding_circles:
            y_ratio = y_vals[-1] / y_vals[-2]
            x_diff = circles[-1][0] - circles[-2][0]
            x_ip1 = y_ratio * x_diff + circles[-1][0]  # Eq. 7

            adding_circles = x_ip1 < x_N
            if adding_circles:
                circle = (x_ip1, 0, R_i(x_ip1))
                circles.append(circle)
                y_vals.append(y_i(x_ip1))
        circles.append((x_N, 0, R_N))
        reflect = [(-x, y, r) for x, y, r in circles[1:]]

        all_circles = np.array(circles + reflect)
        if flip:
            all_circles[:, [0, 1]] = all_circles[:, [1, 0]]

        rot_cens = all_circles[:, :-1].dot(self.matrix.T)
        all_circles[:, :-1] = rot_cens + np.array(self.center)
        return all_circles

    # ----------------------------------------------------------------------- #
    # Plot Function                                                           #
    # ----------------------------------------------------------------------- #
    def plot(self, **kwargs):
        """Plot the ellipse.

        This function adds a `matplotlib.patches.Ellipse`_ patch to the
        current axes using matplotlib. The keyword arguments are passed to
        the patch.

        Args:
            **kwargs (dict): Keyword arguments for matplotlib.

        .. _`matplotlib.patches.Ellipse`: https://matplotlib.org/api/_as_gen/matplotlib.patches.Ellipse.html

        """  # NOQA: E501
        p = patches.Ellipse(self.center, 2 * self.a, 2 * self.b,
                            self.angle_deg, **kwargs)
        plt.gca().add_patch(p)

    # ----------------------------------------------------------------------- #
    # Limits                                                                  #
    # ----------------------------------------------------------------------- #
    @property
    def limits(self):
        """list: list of (lower, upper) bounds for the bounding box"""
        theta = self.angle_rad
        tan_t = np.tan(theta)

        tan_tx = - self.b / self.a * tan_t
        tx_max = np.arctan(tan_tx)
        tx_min = np.pi + tx_max

        if np.isclose(tan_t, 0) and np.cos(theta) > 0:
            ty_max = 0.5 * np.pi
            ty_min = - 0.5 * np.pi
        elif np.isclose(tan_t, 0) and np.cos(theta) < 0:
            ty_max = -0.5 * np.pi
            ty_min = 0.5 * np.pi
        else:
            tan_ty = self.b / (self.a * np.tan(theta))
            ty_max = (np.arctan(tan_ty) + np.pi) % np.pi
            ty_min = np.pi + ty_max

        def xp(t):
            return self.a * np.cos(t)

        def yp(t):
            return self.b * np.sin(t)

        def xr(t):
            return np.cos(theta) * xp(t) - np.sin(theta) * yp(t)

        def yr(t):
            return np.sin(theta) * xp(t) + np.cos(theta) * yp(t)

        x_max = xr(tx_max) + self.center[0]
        x_min = xr(tx_min) + self.center[0]

        y_max = yr(ty_max) + self.center[1]
        y_min = yr(ty_min) + self.center[1]

        return [sorted((x_min, x_max)), sorted((y_min, y_max))]

    @property
    def sample_limits(self):
        """list: list of (lower, upper) bounds for the sampling region"""
        return self.limits

    # ----------------------------------------------------------------------- #
    # Within Test                                                             #
    # ----------------------------------------------------------------------- #
    def within(self, points):
        """Test if points are within ellipse.

        This function tests whether a point or set of points are within the
        ellipse. For the set of points, a list of booleans is returned to
        indicate which points are within the ellipse.

        Args:
            points (list): Point or list of points.

        Returns:
            bool or numpy.ndarray: Set to True for points in ellipse.

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
        of the ellipse. Points at the center of the ellipse are not
        reflected.

        Args:
            points (list): Points to reflect.

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
