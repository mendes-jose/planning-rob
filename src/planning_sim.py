"""
The :mod:`planning_sim` module implements classes and functions to simulate a
navigation scenario consisting of one or more mobile robots that autonomously plan their
motion from an initial state to a final state avoiding static obstacles and
other robots, and respecting kinematic (including nonhonolonomic) constraints.

The motion planner is based on the experimental work developed by Michael Defoort
that seeks a near-optimal solution minimizing the time spend by a robot to
complete its mission.

.. codeauthor:: Jose Magno MENDES FILHO <josemagno.mendes@gmail.com>
"""
# 05/20115
# Copyright 2015 CEA

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import scipy.interpolate as si
import time
import itertools
import multiprocessing as mpc
import sys
import os
import logging
from scipy.optimize import fmin_slsqp
from optparse import OptionParser

###############################################################################
# Obstacle
###############################################################################
class Obstacle(object):
    """ Base class for implementing simulated obstacles.

    Input
        *position*: array-like type representing the obstacle's position (x,y).
    """
    def __init__(self, position):
        """ Class constructor.

        Input
            *position*: array-like type representing the obstacle's position (x,y).
        """
        self.centroid = position
        """ Obstacle's centroid. Considered here the obstacle given position.
        """

    def pt_2_obst(self, pt, offset=0.0):
        """ Return the point-to-obstacle distance less the offset value (as a float).
        """
        return LA.norm(np.array(pt)-np.array(self.centroid))-offset

    def detected_dist(self, pt):
        """ Return the detected distance of this obstacle as seen by a robot at the position pt
        (as a float).
        """
        return LA.norm(np.array(pt) - np.array(self.centroid))

###############################################################################
# RoundObstacle
###############################################################################
class RoundObstacle(Obstacle):
    """ Representation of an obstacle as a circle.
    """
    def __init__(self, position, radius):

        # call ancestor's __init__
        Obstacle.__init__(self, position)

        self.x = self.centroid[0]
        """ x-coordinate of the obstacle centroid.
        """
        self.y = self.centroid[1]
        """ y-coordinate of the obstacle centroid.
        """
        self.radius = radius
        """ Obstacle radius.
        """

    def _plt_circle(self, color='k', linestyle='solid', filled=False, alpha=1.0, offset=0.0):
        """ Return a :py:class:`matplotlib.patches.Circle` object representing the obstacle
        geometry.
        """
        return plt.Circle(
                (self.x, self.y), # position
                self.radius+offset, # radius
                color=color,
                ls = linestyle,
                fill=filled,
                alpha=alpha)

    def plot(self, fig, offset=0.0):
        """ Given a figure this method gets its active axes and plots a grey circle representing
        the obstacle as well as a second, concentric, dashed circle having the original circle
        radius plus the offset value as its own radius.
        """
        ax = fig.gca()
        ax.add_artist(self._plt_circle(filled=True, alpha=0.3))
        ax.add_artist(self._plt_circle(linestyle='dashed', offset=offset))
        ax.plot(self.x, self.y, 'k.')

    def pt_2_obst(self, pt, offset=0.0):
        """ Return the point-to-obstacle distance less the offset value
        and less the obstacle radius (as a float).
        """
        return Obstacle.pt_2_obst(self, pt, offset+self.radius)

###############################################################################
# PolygonObstacle
###############################################################################
class PolygonObstacle(Obstacle):

    @staticmethod
    def _calculate_polygon_area(polygon, signed=False):
        """ Calculate the signed area of non-self-intersecting polygon

        Input
            *polygon*: Numeric array of points (longitude, latitude). It is assumed
                     to be closed, i.e. first and last points are identical

            *signed*: Optional flag deciding whether returned area retains its sign:
                    If points are ordered counter clockwise, the signed area
                    will be positive.

                    If points are ordered clockwise, it will be negative.

                    Default is False which means that the area is always positive.
        Output
            Area of polygon (subject to the value of argument signed)
        """

        # Make sure it is numeric
        P = np.array(polygon)

        # Check input
        msg = ('Polygon is assumed to consist of coordinate pairs. '
               'I got second dimension %i instead of 2' % P.shape[1])
        assert P.shape[1] == 2, msg

        msg = ('Polygon is assumed to be closed. '
               'However first and last coordinates are different: '
               '(%f, %f) and (%f, %f)' % (P[0, 0], P[0, 1], P[-1, 0], P[-1, 1]))
        assert np.allclose(P[0, :], P[-1, :]), msg

        # Extract x and y coordinates
        x = P[:, 0]
        y = P[:, 1]

        # Area calculation
        a = x[:-1] * y[1:]
        b = y[:-1] * x[1:]
        A = np.sum(a - b) / 2.

        # Return signed or unsigned area
        if signed:
            return A
        else:
            return abs(A)

    @staticmethod
    def _calculate_polygon_centroid(polygon):
        """ Calculate the centroid of non-self-intersecting polygon

        Input
            *polygon*: Numeric array of points (longitude, latitude). It is assumed
            to be closed, i.e. first and last points are identical
        Output
            Numeric (1 x 2) array of points representing the centroid
        """

        # Make sure it is numeric
        P = np.array(polygon)

        # Get area - needed to compute centroid
        A = PolygonObstacle._calculate_polygon_area(P, signed=True)

        # Extract x and y coordinates
        x = P[:, 0]
        y = P[:, 1]

        a = x[:-1] * y[1:]
        b = y[:-1] * x[1:]

        cx = x[:-1] + x[1:]
        cy = y[:-1] + y[1:]

        Cx = np.sum(cx * (a - b)) / (6. * A)
        Cy = np.sum(cy * (a - b)) / (6. * A)

        # Create Nx2 array and return
        C = np.array([Cx, Cy])
        return C

    def __init__(self, vertices):

        self._orig_vertices = vertices
        """ Original polygon vertices
        """

        self.centroid = PolygonObstacle._calculate_polygon_centroid(np.vstack([
                vertices, vertices[0]]))
        """ Obstacle's centroid
        """

        self.bounding_circle_radius = max(LA.norm(self.centroid-vertices, axis=1))
        """ Compute the radius of a bounding circle for the obstacle centred on the centroid.

        .. warning:: This is not necessarily the smaller bounding circle. It would be better to
                 get the the midpoint of the greatest line segment formed by two of the
                 polygons vertices as the circle center and half of the that line segment
                 as its radius.
        """

        # call ancestor's __init__
        Obstacle.__init__(self, self.centroid)

        self.x = self.centroid[0]
        """ x-coordinate of the obstacle centroid.
        """
        self.y = self.centroid[1]
        """ y-coordinate of the obstacle centroid.
        """

        self._aug_vertices = np.zeros((vertices.shape[0]*2, vertices.shape[1]))
        """ Represent the auxiliar vertices of the
        convex set resulting from the Minkowski difference between obstacle
        and robot geometry.
        """

        # create linear equations used when computing distance from obstacle
        self._lines_list = []
        for v, vb, va in zip( # v: current vertex, vb: previous vertex, va: following vertex
                self._orig_vertices,
                np.roll(self._orig_vertices, 1, axis=0),
                np.roll(self._orig_vertices, -1, axis=0)):

            # calculate normal vector to the edge difined by v and vb
            vec_b = np.array([+v[1]-vb[1], -v[0]+vb[0]])

            # calculate normal vector to the edge difined by v and va
            vec_a = np.array([-v[1]+va[1], v[0]-va[0]])

            # create 3 line equations:
            # * first is the orthogonal line to the v-vb edge passing thru v
            # * second is the orthogonal line to the v-va edge passing thru v
            # * third is the line coincident to the v-va edge
            self._lines_list += [[(vec_b[1], -vec_b[0], vec_b[0]*v[1]-v[0]*vec_b[1]),
                    (vec_a[1], -vec_a[0], vec_a[0]*v[1]-v[0]*vec_a[1]),
                    (va[1]-v[1], v[0]-va[0], va[0]*v[1]-v[0]*va[1])]]

    def _create_aug_vertices(self, offset):
        """Create augmented vertices from original vertices and offset value

        Input
            *offset*: Real value used as offset from original vertices. Can also
            be seen as the radius of the circle that will be subtracted (in the
            Minkowski difference sense) from the obstacle geometry.
        """
        new_vertices_list = []
        for v, vb, va in zip(
                self._orig_vertices,
                np.roll(self._orig_vertices, 1, axis=0),
                np.roll(self._orig_vertices, -1, axis=0)):
            vec_b = np.array([+v[1]-vb[1], -v[0]+vb[0]])
            unit_vec_b = vec_b/LA.norm(vec_b)
            vec_a = np.array([-v[1]+va[1], v[0]-va[0]])
            unit_vec_a = vec_a/LA.norm(vec_a)

            new_vertices_list += [offset*unit_vec_b+v, offset*unit_vec_a+v]
        self._aug_vertices = np.array(new_vertices_list)

    def plot(self, fig, offset=0.0):
        """ Given a figure this method gets its active axes and plots a grey polygon representing
        the obstacle as well as some dashed lines associated with the Minkowski difference of
        the obstacle representation and a circle of radius equals to offset.
        """
        self._create_aug_vertices(offset)
        ax = fig.gca()
        [ax.add_artist(plt.Circle(
                (v[0], v[1]),
                offset,
                color='k',
                ls='dashed',
                fill=False)) for v in self._orig_vertices]

        ax.add_artist(plt.Polygon(
                self._orig_vertices, color='k', ls='solid', fill=True, alpha=0.3))
        ax.add_artist(plt.Polygon(self._aug_vertices, color='k', ls='dashed', fill=False))
        ax.plot(self.x, self.y, 'k.')

    def pt_2_obst(self, pt, offset=0.0):
        """Calculate distance from point to the obstacle

        Input
            *pt*: cardinal coordinates of the point

            *offset*: offset distance that will be subtracted from the actual
            point-to-obstacle distance

        Output
            Real number representing the distance pt_2_obstacle less offset. Negative value means
            that the point is closer then the offset value from the obstacle, or even inside it.

        .. todo:: Fix error handling
        """
        # update augmented vertices using offset value
        self._create_aug_vertices(offset)

        # compute the signed distances from the point (pt) to each of the...
        # ... 3*card(_orig_vertices) lines
        signed_dists = []
        for ls in self._lines_list:
            s_dist = []
            for l in ls:

                # distance pt-2-line calculate replacing pt in line equation...
                # ... divided by sqrt(a^2+b^2)
                s_dist += [(l[0]*pt[0] + l[1]*pt[1] + l[2])/LA.norm(np.array([l[0], l[1]]))]

            signed_dists += [s_dist]

        # calculate the zone where the point (pt) is
        for idx in range(self._orig_vertices.shape[0]):

            # if the point is "between" the 2 orthogonal lines that passes thru the vertex
            if (signed_dists[idx][0]<=0.0 and \
                    signed_dists[idx][1]>0.0) \
                    == True:

                # return point to point dist less offset
                return LA.norm(pt-self._orig_vertices[idx, ])-offset

            # if the point is "between" the edge v-va, and the 2 lines...
            # ...passing thru v and va that are _|_ to the edge
            elif (signed_dists[idx][2]>0.0 and \
                    signed_dists[idx][1]<=0.0 and \
                    signed_dists[(idx+1)%self._orig_vertices.shape[0]][0]>0.0) \
                    == True:

                # return the distance pt_2_line less offset
                return abs(signed_dists[idx][2])-offset

        # if it reachs this line it probably means that the point (pt) is inside the obstacle
        pt_is_inside = True

        # verifying if the point is realy inside the obstacle
        for idx in range(self._orig_vertices.shape[0]):
            pt_is_inside = pt_is_inside and signed_dists[idx][2]<=0.0

        # if pt_is_inside is still true
        if pt_is_inside == True:
            # compute negative distance from the closest edge less offset
            pt_2_edges = [abs(signed_dists[idx][2]) for idx in range(self._orig_vertices.shape[0])]
            return -min(pt_2_edges)-offset
        else:
            print 'I\'m out of ideas about what happend. WhereTF is the path going?'
            # stop execution TODO error handling etc

###############################################################################
# Boundary
###############################################################################
class Boundary(object):
    def __init__(self, x, y):
        self.x_min = x[0]
        """ Min bound in the x direction.
        """
        self.x_max = x[1]
        """ Max bound in the x direction.
        """
        self.y_min = y[0]
        """ Min bound in the y direction.
        """
        self.y_max = y[1]
        """ Max bound on y direction.
        """

###############################################################################
# Unicycle Kinematic Model
###############################################################################
class UnicycleKineModel(object):
    """ This class defines the kinematic model of an unicycle mobile robot.

    Unicycle kinematic model:

    .. math::
        \\begin{array}{c}
        \dot{q} = \mathrm{f}(q, u) \Rightarrow\\\\
        \left[\\begin{array}{c}
        \dot{x}\\\\
        \dot{y}\\\\
        \dot{\\theta}
        \end{array}\\right]=
        \left[\\begin{array}{c}
        v\cos(\\theta)\\\\
        v\sin(\\theta)\\\\
        w
        \end{array}\\right]
        \end{array}

    Changing variables (:math:`z = [x, y]^T`) we can rewrite the system using the
    flat output :math:`z` as:

    .. math::
        \left[\\begin{array}{c}
        x\\\\
        y\\\\
        \\theta\\\\
        v\\\\
        w
        \end{array}\\right]=
        \left[\\begin{array}{c}
        z_1\\\\
        z_2\\\\
        \\arctan(\dot{z}_2/\dot{z}_1)\\\\
        \\sqrt{\dot{z}_{1}^{2} + \dot{z}_{2}^{2}}\\\\
        \\dfrac{\dot{z}_{1}\ddot{z}_{2} - \dot{z}_{2}\ddot{z}_{1}}{\dot{z}_{1}^{2}+\dot{z}_{2}^{2}}
        \end{array}\\right]

    where the the state and input vector (:math:`q, v`) are function of :math:`z`
    and its derivatives.
    """

    @staticmethod
    def _unsigned_angle(angle):
        """ Map signed angles (:math:`\\theta \in (-\pi, \pi]`) to unsigned
        (:math:`\\theta \in [0, 2\pi)`).

        .. note:: Method not used.
        """
        return 2.*np.pi+angle if angle < 0.0 else angle

    def __init__(
            self,
            q_init,
            q_final,
            u_init=[0.0, 0.0],
            u_final=[0.0, 0.0],
            u_max=[1.0, 5.0],
            a_max=[2.0, 10.0]):
        # Control
        self.u_dim = 2
        """ Input vector order.
        """
        self.u_init = np.matrix(u_init).T
        """ Initial input.
        """
        self.u_final = np.matrix(u_final).T
        """ Final input.
        """
        self.u_max = np.matrix(u_max).T
        """ Absolute max input.
        """
        self.acc_max = np.matrix(a_max).T
        """ Absolute maximum values for the first derivative of the input vector.

        .. note:: Value not used.
        """
        # State
        self.q_dim = 3
        """ State vector order.
        """
        self.q_init = np.matrix(q_init).T #angle in [-pi, pi]
        """ Initial state.
        """
        self.q_final = np.matrix(q_final).T #angle in [-pi, pi]
        """ Final state.
        """
        # Flat output
        self.z_dim = self.u_dim
        """ Flat output vector order.
        """
        self.z_init = self.phi_0(self.q_init)
        """ Initial flat output.
        """
        self.z_final = self.phi_0(self.q_final)
        """ Final flat output.
        """
        self.l = 2
        """ Number of flat output derivatives that are needed to calculate the state
        and input vectors.
        """

    def phi_0(self, q):
        """ Returns :math:`z` given :math:`q`
        """
        return q[0:2, 0]

    def phi_1(self, zl):
        """ Returns :math:`[x\\ y\\ \\theta]^T` given :math:`[z\\ \dot{z}\\ \dotsc\\ z^{(l)}]`
        (only :math:`z` and :math:`\dot{z}` are used). :math:`\\theta` is in the range
        :math:`(-\pi, \pi]`.

        .. math::
            \\begin{array}{l}
            \\varphi_1(z(t_k),\dotsc,z^{(l)}(t_k))=\\
            \left[\\begin{array}{c}
            x\\\\
            y\\\\
            \omega
            \end{array}\\right]
            \left[\\begin{array}{c}
            z_1\\\\
            z_2\\\\
            \\arctan(\dot{z}_2/\dot{z}_1)\\\\
            \end{array}\\right]
            \end{array}
        """
        if zl.shape >= (self.u_dim, self.l+1):
            return np.matrix(np.append(zl[:, 0], \
                    np.asarray(
                    np.arctan2(zl[1, 1], zl[0, 1])))).T
        else:
            logging.warning('Bad zl input. Returning zeros')
            return np.matrix('0.0; 0.0; 0.0')

    def phi_2(self, zl):
        """ Returns :math:`[v\\ \omega]^T` given :math:`[z\\ \dot{z}\\ \dotsc\\ z^{(l)}]`
        (only :math:`\dot{z}` and :math:`\ddot{z}` are used).

        .. math::
            \\begin{array}{l}
            \\varphi_2(z(t_k),\dotsc,z^{(l)}(t_k))=\\
            \left[\\begin{array}{c}
            v\\\\
            \omega
            \end{array}\\right]
            = \left[\\begin{array}{c}
            \\sqrt{\dot{z}_{1}^{2} + \dot{z}_{2}^{2}}\\\\
            \\dfrac{\dot{z}_{1}\ddot{z}_{2} -
            \dot{z}_{2}\ddot{z}_{1}}{\dot{z}_{1}^{2}+\dot{z}_{2}^{2}}
            \end{array}\\right]
            \end{array}
        """
        if zl.shape >= (self.u_dim, self.l+1):
            # Prevent division by zero
            den = zl[0, 1]**2 + zl[1, 1]**2 + np.finfo(float).eps
            return np.matrix([[LA.norm(zl[:, 1])], \
                    [(zl[0, 1]*zl[1, 2]-zl[1, 1]*zl[0, 2] \
                    )/den]])
        else:
            logging.warning('Bad zl input. Returning zeros')
            return np.matrix('0.0; 0.0')

    def phi_3(self, zl):
        """ Returns :math:`[\dot{v}\\ \dot{\omega}]^T` given
        :math:`[z\\ \dot{z}\\ \dotsc\\ z^{(l+1)}]`.
        (only :math:`\dot{z}`, :math:`\ddot{z}` and :math:`z^{(3)}` are used).

        .. math::
            \\begin{array}{l}
            \\varphi_3(z(t_k),\dotsc,z^{(l+1)}(t_k))=\\
            \left[\\begin{array}{c}
            \dot{v}\\\\
            \dot{\omega}
            \end{array}\\right]
            = \left[\\begin{array}{c}
            \\frac{\partial}{\partial t}v\\\\
            \\frac{\partial}{\partial t}\omega
            \end{array}\\right]
            = \left[\\begin{array}{c}
            \\frac{\dot{z}_1\ddot{z}_1 + \dot{z}_2\ddot{z}_2}{\|\dot{z}\|}\\\\
            \\frac{(\ddot{z}_1\ddot{z}_2+ z^{(3)}_2\dot{z}_1 -
            (\ddot{z}_2\ddot{z}_1+z^{(3)}_1\dot{z}_2))\|\dot{z}\|^2-
            2(\dot{z}_1\ddot{z}_2-\dot{z}_2\ddot{z}_1)\|\dot{z}\|\dot{v}}{\|\dot{z}\|^4}
            \end{array}\\right]
            \end{array}
        """
        if zl.shape >= (self.u_dim, self.l+1):
            # Prevent division by zero
            dz_norm = LA.norm(zl[:, 1])
            den = dz_norm + np.finfo(float).eps
            dv = (zl[0, 1]*zl[0, 2]+zl[1, 1]*zl[1, 2])/den
            dw = ((zl[0, 2]*zl[1, 2]+zl[1, 3]*zl[0, 1]- \
                    (zl[1, 2]*zl[0, 2]+zl[0, 3]*zl[1, 1]))*(dz_norm**2) - \
                    (zl[0, 1]*zl[1, 2]-zl[1, 1]*zl[0, 2])*2*dz_norm*dv)/den**4
            return np.matrix([[dv], [dw]])
        else:
            logging.warning('Bad zl input. Returning zeros')
            return np.matrix('0.0; 0.0')

###############################################################################
# Communication Msg
###############################################################################
class RobotMsg(object):
    """ Robot message class for exchanging information about theirs intentios.
    """
    def __init__(self, dp, ip_z1, ip_z2, lz):
        self.done_planning = dp
        """ Flag to indicate that the robot has finished its planning process.
        """
        self.intended_path_z1 = ip_z1
        """ Intended path (z1 coordiante).
        """
        self.intended_path_z2 = ip_z2
        """ Intended path (z2 coordiante).
        """
        self.latest_z = lz
        """ z value calculated on the previews planned section.
        """

###############################################################################
# Robot
###############################################################################
class Robot(object):
    """ Class for creating a robot in the simulation.
    It implements the Defoort's experimental motion planning algorithm
    """
    def __init__(
            self,
            eyed,
            kine_model,
            obstacles,
            phy_boundary,
            neigh,                  # neighbors to whom this robot shall talk...
                                    # ...(used for conflict only, not communic)
            N_s=20,
            n_knots=6,
            t_init=0.0,
            t_sup=1e10,
            Tc=1.0,
            Tp=3.0,
            Td=3.0,
            rho=0.2,
            detec_rho=3.0,
            com_range=15.0,
            def_epsilon=0.5,
            safe_epsilon=0.1,
            ls_time_opt_scale = 1.,
            ls_min_dist = 0.5):

        self.eyed = eyed
        """ Robot ID.
        """
        self.k_mod = kine_model
        """ Robot kinematic model.
        """
        self._obst = obstacles
        self._p_bound = phy_boundary
        self._tc_syncer = None
        self._tc_syncer_cond = None
        self._conflict_syncer = None
        self._conflict_syncer_conds = None
        self._com_link = None
        self.sol = None
        """ Solution, i.e., finded path.
        """
        self.rtime = None
        """ "Race" time. Discrete time vector of the planning process.
        """
        self.ctime = None
        """ List of computation time spend for calculating each planned section.
        """
        self._neigh = neigh
        self._N_s = N_s # no of samples for discretization of time
        self._n_knots = n_knots
        self._t_init = t_init
        self._t_sup = t_sup # superior limit of time
        self._Tc = Tc
        self._Tp = Tp
        self._Td = Td
        self.rho = rho
        """ Robot's radius.
        """
        self._d_rho = detec_rho
        self._com_range = com_range
        self._def_epsilon = def_epsilon
        self._safe_epsilon = safe_epsilon
        self._log_lock = None
        self._ls_time_opt_scale = ls_time_opt_scale
        self._ls_min_dist = ls_min_dist

        # number of robots
        self._n_robots = None

        # index for sliding windows
        td_step = (self._Td-self._t_init)/(self._N_s-1)
        tp_step = (self._Tp-self._t_init)/(self._N_s-1)
        self._Tcd_idx = int(round(self._Tc/td_step))
        self._Tcp_idx = int(round(self._Tc/tp_step))
        self._Tcd = self._Tcd_idx*td_step
        self._Tcp = self._Tcp_idx*tp_step

        # optimization parameters
        self._maxit = 100
        self._fs_maxit = 100
        self._ls_maxit = 100
        self._acc = 1e-6

        # init planning
        self._detected_obst_idxs = range(len(self._obst))

        self._latest_q = self.k_mod.q_init
        self._latest_u = self.k_mod.u_init
        self._latest_z = self.k_mod.z_init
        self._final_z = self.k_mod.z_final

        self._D = self._Tp * self.k_mod.u_max[0, 0]

        self._d = self.k_mod.l+2 # B-spline order (integer | d > l+1)
        self._n_ctrlpts = self._n_knots + self._d - 1 # nb of ctrl points

        self._C = np.zeros((self._n_ctrlpts, self.k_mod.u_dim))

        self._all_dz = []
        self._all_times = []
        self._all_comp_times = []

        # Instantiating the planning process
        self.planning_process = mpc.Process(target=Robot._plan, args=(self, ))
        """ Planning process handler for where the planning routine is called.
        """

    def set_option(self, name, value=None):
        """ Setter for some optimation parameters.
        """
        if value != None:
            if name == 'maxit':
                self._maxit = value
            elif name == 'fs_maxit':
                self._fs_maxit = value
            elif name == 'ls_maxit':
                self._ls_maxit = value
            elif name == 'acc':
                self._acc = value
            elif name == 'com_link':
                self._com_link = value
            elif name == 'sol':
                self.sol = value
            elif name == 'rtime':
                self.rtime = value
            elif name == 'ctime':
                self.ctime = value
            elif name == 'tc_syncer':
                self._tc_syncer = value
            elif name == 'tc_syncer_cond':
                self._tc_syncer_cond = value
            elif name == 'conflict_syncer':
                self._conflict_syncer = value
                self._n_robots = len(value)
            elif name == 'conflict_syncer_conds':
                self._conflict_syncer_conds = value
            elif name == 'log_lock':
                self._log_lock = value
            else:
                self._log('w', 'Unknown parameter '+name+', nothing will be set')

    def _gen_knots(self, t_init, t_final):
        """ Generate b-spline knots given initial and final times.
        """
        gk = lambda x:t_init + (x-(self._d-1.0))*(t_final-t_init)/self._n_knots
        knots = [t_init for _ in range(self._d)]
        knots.extend([gk(i) for i in range(self._d, self._d+self._n_knots-1)])
        knots.extend([t_final for _ in range(self._d)])
        return np.asarray(knots)

    def _comb_bsp(self, t, ctrl_pts, deriv_order):
        """ Combine base b-splines into a Bezier curve given control points and derivate order.

        Input
            *ctrl_pts*: numpy array with dimension :math:`n_{ctrl}\\times z_{dim}`, :math:`n_{ctrl}`
            being the number of
            control points and :math:`z_{dim}` the flat output dimension.

            *deriv_order*: derivative order of the Bezier curve.

            *t*: discrete time array.

        Return
            :math:`z_{dim}\\times N_s` numpy array representing the resulting Bezier curve,
            :math:`z_{dim}` being
            the flat output dimension and :math:`N_{s}` the discrete time array dimension.
        """
        tup = (
                self._knots, # knots
                ctrl_pts[:, 0], # first dim ctrl pts
                self._d-1) # b-spline degree

        # interpolation
        z = si.splev(t, tup, der=deriv_order).reshape(len(t), 1)

        for i in range(self.k_mod.u_dim)[1:]:
            tup = (
                    self._knots,
                    ctrl_pts[:, i],
                    self._d-1)
            z = np.append(
                    z,
                    si.splev(t, tup, der=deriv_order).reshape(len(t), 1),
                    axis=1)
        return z

    def _log(self, logid, strg):
        """ Log writer (multiprocess safe).

        Input
            *logid*:
                === ===============
                'd' for debug
                'i' for information
                'w' for warning
                'e' for error
                'c' for critical
                === ===============

            *strg*: log string.
        """
        if logid == 'd':
            log_call = logging.debug
        elif logid == 'i':
            log_call = logging.info
        elif logid == 'w':
            log_call = logging.warning
        elif logid == 'e':
            log_call = logging.error
        elif logid == 'c':
            log_call = logging.critical
        else:
            log_call = logging.debug

        if self._log_lock != None:
            self._log_lock.acquire()
        log_call(strg)
        if self._log_lock != None:
            self._log_lock.release()

    def _detect_obst(self):
        """ Determinate which obstacles are within the detection radius.
        This method updates the :attr:`_detected_obst_idxs`
        private attribute.
        """
        idx_list = []
        for idx in range(len(self._obst)):
            dist = self._obst[idx].detected_dist(np.squeeze(np.asarray(self._latest_z.T)))
            if dist < self._d_rho:
                idx_list += [idx]
        self._detected_obst_idxs = idx_list

    def _ls_sa_criterion(self, x):
        """ Cost function to be minimized used for optimizing
        the last section of the plan when no conflicts are detected.
        It calculates the square of the total plan time.

        Input
            *x*: optimization argument.

        Return
            Cost value.

        .. warning:: Optimization solver can misbehave for costs too big (:math:`> 10^{6}`).
        """
        return self._ls_time_opt_scale*(x[0]+self._mtime[0])**2

    def _ls_sa_feqcons(self, x):
        """ Calculate the **equations** constraints values for the last section of the plan when
        there are no conflicts.

        The following expressions are evaluated:

        .. math::
            \left\lbrace\\begin{array}{lcl}
            \\varphi_1(z(t_{0,\ sec}),\dotsc,z^{(l)}(t_{0,\ sec}))
            &-& q_{N_s-1,\ sec-1}\\\\
            \\varphi_1(z(t_{N_s-1,\ sec}),\dotsc,z^{(l)}(t_{N_s-1,\ sec}))
            &-& q_{final}\\\\
            \\varphi_2(z(t_{0,\ sec}),\dotsc,z^{(l)}(t_{0,\ sec}))
            &-&u_{N_s-1,\ sec-1}\\\\
            \\varphi_2(z(t_{N_s-1,\ sec}),\dotsc,z^{(l)}(t_{N_s-1,\ sec})) &-& u_{final}\\\\
            \end{array}\\right.
        where:
            =========== ===================================
            :math:`sec` indicates the current plan section.
            =========== ===================================

        Input
            *x*: optimization argument.

        Return
            Array with the equations' values.
        """
        # get time and ctrl pts
        dt_final = x[0]
        t_final = self._mtime[0]+dt_final
        C = x[1:].reshape(self._n_ctrlpts, self.k_mod.u_dim)

        # get new knots and the flat output initial and final values for this plan section.
        self._knots = self._gen_knots(self._mtime[0], t_final)
        dztinit = self._comb_bsp([self._mtime[0]], C, 0).T
        for dev in range(1, self.k_mod.l+1):
            dztinit = np.append(dztinit, self._comb_bsp([self._mtime[0]], C, dev).T, axis=1)

        dztfinal = self._comb_bsp([t_final], C, 0).T
        for dev in range(1, self.k_mod.l+1):
            dztfinal=np.append(dztfinal, self._comb_bsp([t_final], C, dev).T, axis=1)

        # calculate equations
        eq_cons = list(np.squeeze(np.array(self.k_mod.phi_1(dztinit)-self._latest_q)))+\
                list(np.squeeze(np.array(self.k_mod.phi_1(dztfinal)-self.k_mod.q_final)))+\
                list(np.squeeze(np.array(self.k_mod.phi_2(dztinit)-self._latest_u)))+\
                list(np.squeeze(np.array(self.k_mod.phi_2(dztfinal)-self.k_mod.u_final)))

        # Count how many inequations are not respected
        self._unsatisf_eq_values = [ec for ec in eq_cons if ec != 0]

        return np.asarray(eq_cons)

    def _ls_sa_fieqcons(self, x):
        """ Calculate the **inequations** constraints values for the last section of the plan when
        there are no conflicts.

        The following expressions are evaluated:

        .. math::
            \left\lbrace\\begin{array}{lcl}
            u_{max} - |\\varphi_2(z(t_{1,\ sec}),\dotsc,z^{(l)}(t_{1,\ sec}))|\\\\
            u_{max} - |\\varphi_2(z(t_{2,\ sec}),\dotsc,z^{(l)}(t_{2,\ sec}))|\\\\
            \dotsc\\\\
            u_{max} - |\\varphi_2(z(t_{N_s-2,\ sec}),\dotsc,z^{(l)}(t_{N_s-2,\ sec}))|\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{1,\ sec}),\dotsc,z^{(l)}(t_{1,\ sec})), O_0)\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{2,\ sec}),\dotsc,z^{(l)}(t_{2,\ sec})), O_0)\\\\
            \dotsc\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{N_s-2,\ sec}),
            \dotsc,z^{(l)}(t_{N_s-2,\ sec})), O_0)\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{1,\ sec}),\dotsc,z^{(l)}(t_{1,\ sec})), O_1)\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{2,\ sec}),\dotsc,z^{(l)}(t_{2,\ sec})), O_1)\\\\
            \dotsc\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{N_s-2,\ sec}),
            \dotsc,z^{(l)}(t_{N_s-2,\ sec})), O_1)\\\\
            \dotsc\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{1,\ sec}),
            \dotsc,z^{(l)}(t_{1,\ sec})), O_{M-1})\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{2,\ sec}),
            \dotsc,z^{(l)}(t_{2,\ sec})), O_{M-1})\\\\
            \dotsc\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{N_s-2,\ sec}),
            \dotsc,z^{(l)}(t_{N_s-2,\ sec})), O_{M-1})\\\\
            \end{array}\\right.

        where:
            =========== =================================================
            :math:`sec` indicates the current plan section.
            :math:`O_i` is the ith detected obstacle.
            :math:`M`   is the number of detected obstacles.
            :math:`N_s` is the number of time samples for a plan section.
            =========== =================================================

        Input
            *x*: optimization argument.

        Return
            Array with the inequations' values.
        """
        dt_final = x[0]
        t_final = self._mtime[0]+dt_final
        C = x[1:].reshape(self._n_ctrlpts, self.k_mod.u_dim)

        self._knots = self._gen_knots(self._mtime[0], t_final)

        mtime = np.linspace(self._mtime[0], t_final, self._N_s)

        # get a list over time of the matrix [z dz ddz](t) t in [tk, tk+Tp]
        dz = self._comb_bsp(mtime[1:-1], C, 0).T
        for dev in range(1, self.k_mod.l+1):
            dz = np.append(dz, self._comb_bsp(mtime[1:-1], C, dev).T, axis=0)

        dztTp = [dzt.reshape(self.k_mod.l+1, self.k_mod.u_dim).T for dzt in dz.T]

        # get a list over time of command values u(t)
        utTp = map(self.k_mod.phi_2, dztTp)

        # get a list over time of values q(t)
        qtTp = map(self.k_mod.phi_1, dztTp)

        ## Obstacles constraints
        # N_s*nb_obst_detected
        obst_cons = []
        for m in self._detected_obst_idxs:
            obst_cons += [self._obst[m].pt_2_obst(np.squeeze(np.asarray(qt[0:2, 0].T)), self.rho)\
                    for qt in qtTp]

        ## Max speed constraints
        # N_s*u_dim inequations
        max_speed_cons = list(itertools.chain.from_iterable(
                [[self.k_mod.u_max[i, 0]-abs(ut[i, 0]) \
                for i in range(self.k_mod.u_dim)]for ut in utTp]))

        # Create final array
        ieq_cons = obst_cons + max_speed_cons

        # Count how many inequations are not respected
        self._unsatisf_ieq_values = [ieq for ieq in ieq_cons if ieq < 0]

        return np.asarray(ieq_cons)

    def _sa_criterion(self, x):
        """ Cost function to be minimized used for optimizing
        the first and intermediaries sections of the plan when no conflicts are detected.

        It calculates the distance between the final position of the proposed plan and
        the *goal point*.

        The *goal point* is calcualted as follows:

        .. todo:: Add the expression for computing the goal point.

        Input
            *x*: optimization argument.

        Return
            Cost value.

        .. warning:: Optimization solver can misbehave for costs too big (:math:`> 10^{6}`).
        """
        C = x.reshape(self._n_ctrlpts, self.k_mod.u_dim)

        dz = self._comb_bsp([self._mtime[-1]], C, 0).T
        for dev in range(1, self.k_mod.l+1):
            dz = np.append(dz, self._comb_bsp([self._mtime[-1]], C, dev).T, axis=1)
        qTp = self.k_mod.phi_1(dz)

        cte = 1.5 # TODO no magic constants
        pos2target = self.k_mod.q_final[0:-1, :] - self._latest_z
        pos2target_norm = LA.norm(pos2target)
        if pos2target_norm > cte*self._D:
            goal_pt = self._latest_z+pos2target/pos2target_norm*cte*self._D
        elif pos2target_norm < self._D:
            goal_pt = self._latest_z+pos2target/pos2target_norm*self._D
        else:
            goal_pt = self.k_mod.q_final[0:-1, :]
        cost = LA.norm(qTp[0:-1, :] - goal_pt)**2
        # TODO
        if cost > 1e6:
            self._log('d', 'R{}: Big problem {}'.format(self.eyed, cost))
            print ('R{}: Big problem {}'.format(self.eyed, cost))
        return cost

    def _sa_feqcons(self, x):
        """ Calculate the **equations** constraints values for the first and intermediaries
        sections of the plan when there are no conflicts.

        The following expressions are evaluated:

        .. math::
            \left\lbrace\\begin{array}{lcl}
            \\varphi_1(z(t_{0,\ sec}),\dotsc,z^{(l)}(t_{0,\ sec}))
            &-& q_{N_s-1,\ sec-1}\\\\
            \\varphi_2(z(t_{0,\ sec}),\dotsc,z^{(l)}(t_{0,\ sec}))
            &-& u_{N_s-1,\ sec-1}\\\\
            \end{array}\\right.

        where:
            =========== =================================================
            :math:`sec` indicates the current plan section.
            =========== =================================================

        Input
            *x*: optimization argument.

        Return
            Array with the equations' values.
        """
        C = x.reshape(self._n_ctrlpts, self.k_mod.u_dim)

        dztinit = self._comb_bsp([self._mtime[0]], C, 0).T
        for dev in range(1, self.k_mod.l+1):
            dztinit = np.append(dztinit, self._comb_bsp([self._mtime[0]], C, dev).T, axis=1)

        # dimension: q_dim + u_dim
        eq_cons = list(np.squeeze(np.array(self.k_mod.phi_1(dztinit)-self._latest_q)))+\
               list(np.squeeze(np.array(self.k_mod.phi_2(dztinit)-self._latest_u)))

        # Count how many equations are not respected
        unsatisf_list = [eq for eq in eq_cons if eq != 0]
        self._unsatisf_eq_values = unsatisf_list

        return np.asarray(eq_cons)

    def _sa_fieqcons(self, x):
        """ Calculate the **inequations** constraints values for the first and intermediaries
        sections of the plan when there are no conflicts.

        The following expressions are evaluated:

        .. math::
            \left\lbrace\\begin{array}{lcl}
            u_{max} - |\\varphi_2(z(t_{1,\ sec}),\dotsc,z^{(l)}(t_{1,\ sec}))|\\\\
            u_{max} - |\\varphi_2(z(t_{2,\ sec}),\dotsc,z^{(l)}(t_{2,\ sec}))|\\\\
            \dotsc\\\\
            u_{max} - |\\varphi_2(z(t_{N_s-1,\ sec}),\dotsc,z^{(l)}(t_{N_s-1,\ sec}))|\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{1,\ sec}),\dotsc,z^{(l)}(t_{1,\ sec})), O_0)\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{2,\ sec}),\dotsc,z^{(l)}(t_{2,\ sec})), O_0)\\\\
            \dotsc\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{N_s-1,\ sec}),
            \dotsc,z^{(l)}(t_{N_s-1,\ sec})), O_0)\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{1,\ sec}),\dotsc,z^{(l)}(t_{1,\ sec})), O_1)\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{2,\ sec}),\dotsc,z^{(l)}(t_{2,\ sec})), O_1)\\\\
            \dotsc\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{N_s-1,\ sec}),
            \dotsc,z^{(l)}(t_{N_s-1,\ sec})), O_1)\\\\
            \dotsc\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{1,\ sec}),
            \dotsc,z^{(l)}(t_{1,\ sec})), O_{M-1})\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{2,\ sec}),
            \dotsc,z^{(l)}(t_{2,\ sec})), O_{M-1})\\\\
            \dotsc\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{N_s-1,\ sec}),
            \dotsc,z^{(l)}(t_{N_s-1,\ sec})), O_{M-1})\\\\
            \end{array}\\right.

        where:
            =========== =================================================
            :math:`sec` indicates the current plan section.
            :math:`O_i` is the ith detected obstacle.
            :math:`M`   is the number of detected obstacles.
            :math:`N_s` is the number of time samples for a plan section.
            =========== =================================================

        Input
            *x*: optimization argument.

        Return
            Array with the inequations' values.
        """
        C = x.reshape(self._n_ctrlpts, self.k_mod.u_dim)

        # get a list over time of the matrix [z dz ddz](t) t in [t_{k+1}, t_k+Tp]
        dz = self._comb_bsp(self._mtime[1:], C, 0).T
        for dev in range(1, self.k_mod.l+1):
            dz = np.append(dz, self._comb_bsp(self._mtime[1:], C, dev).T, axis=0)

        dztTp = [dzt.reshape(self.k_mod.l+1, self.k_mod.u_dim).T for dzt in dz.T]

        # get a list over time of command values u(t)
        utTp = map(self.k_mod.phi_2, dztTp)

        # get a list over time of values q(t)
        qtTp = map(self.k_mod.phi_1, dztTp)

        ## Obstacles constraints
        # N_s*nb_obst_detected
        obst_cons = []
        for m in self._detected_obst_idxs:
            obst_cons += [self._obst[m].pt_2_obst(np.squeeze(np.asarray(qt[0:2, 0].T)), self.rho)\
                    for qt in qtTp]

        ## Max speed constraints
        # N_s*u_dim inequations
        max_speed_cons = list(itertools.chain.from_iterable(
                [[self.k_mod.u_max[i, 0]-abs(ut[i, 0]) for i in range(self.k_mod.u_dim)]\
                for ut in utTp]))

        # Create final array
        ieq_cons = obst_cons + max_speed_cons

        # Count how many inequations are not respected
        unsatisf_list = [ieq for ieq in ieq_cons if ieq < 0]
        self._unsatisf_ieq_values = unsatisf_list

        # return arrray where each element is an inequation constraint
        return np.asarray(ieq_cons)

    def _ls_co_criterion(self, x):
        """ Cost function to be minimized used for optimizing
        the last section of the plan when conflicts are detected.
        It calculates the square of the total plan time.

        Input
            *x*: optimization argument.

        Return
            Cost value.

        .. warning:: Optimization solver can misbehave for costs too big (:math:`> 10^{6}`).

        .. note:: This method is just a call to the :meth:`_ls_sa_criterion` method.
        """
        return self._ls_sa_criterion(x)

    def _ls_co_feqcons(self, x):
        """ Calculate the **equations** constraints values for the last section of the plan when
        there are conflicts.

        Input
            *x*: optimization argument.

        Return
            Array with the equations' values.

        .. note:: This method is just a call to the :meth:`_ls_co_criterion` method.
        """
        return self._ls_sa_feqcons(x)

    def _ls_co_fieqcons(self, x):
        """ Calculate the **inequations** constraints values for the last section of the plan when
        there are conflicts.

        The following expressions are evaluated:

        .. math::
            \left\lbrace\\begin{array}{lcl}
            u_{max} - |\\varphi_2(z(t_{1,\ sec}),\dotsc,z^{(l)}(t_{1,\ sec}))|\\\\
            u_{max} - |\\varphi_2(z(t_{2,\ sec}),\dotsc,z^{(l)}(t_{2,\ sec}))|\\\\
            \dotsc\\\\
            u_{max} - |\\varphi_2(z(t_{N_s-2,\ sec}),\dotsc,z^{(l)}(t_{N_s-2,\ sec}))|\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{1,\ sec}),\dotsc,z^{(l)}(t_{1,\ sec})), O_0)\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{2,\ sec}),\dotsc,z^{(l)}(t_{2,\ sec})), O_0)\\\\
            \dotsc\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{N_s-2,\ sec}),
            \dotsc,z^{(l)}(t_{N_s-2,\ sec})), O_0)\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{1,\ sec}),\dotsc,z^{(l)}(t_{1,\ sec})), O_1)\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{2,\ sec}),\dotsc,z^{(l)}(t_{2,\ sec})), O_1)\\\\
            \dotsc\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{N_s-2,\ sec}),
            \dotsc,z^{(l)}(t_{N_s-2,\ sec})), O_1)\\\\
            \dotsc\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{1,\ sec}),
            \dotsc,z^{(l)}(t_{1,\ sec})), O_{M-1})\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{2,\ sec}),
            \dotsc,z^{(l)}(t_{2,\ sec})), O_{M-1})\\\\
            \dotsc\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{N_s-2,\ sec}),
            \dotsc,z^{(l)}(t_{N_s-2,\ sec})), O_{M-1})\\\\
            \end{array}\\right.

        where:
            =========== =================================================
            :math:`sec` indicates the current plan section.
            :math:`O_i` is the ith detected obstacle.
            :math:`M`   is the number of detected obstacles.
            :math:`N_s` is the number of time samples for a plan section.
            =========== =================================================

        Input
            *x*: optimization argument.

        Return
            Array with the inequations' values.

        .. todo::
            Take into account the coflict constraints!
            The way it is this method is identical to the :meth:`_ls_sa_fieqcons` method,
            which is pretty bad.
        """
        dt_final = x[0]
        t_final = self._mtime[0]+dt_final
        C = x[1:].reshape(self._n_ctrlpts, self.k_mod.u_dim)

        self._knots = self._gen_knots(self._mtime[0], t_final)

        mtime = np.linspace(self._mtime[0], t_final, self._N_s)

        # get a list over time of the matrix [z dz ddz](t) t in [tk, tk+Tp]
        dz = self._comb_bsp(mtime[1:-1], C, 0).T
        for dev in range(1, self.k_mod.l+1):
            dz = np.append(dz, self._comb_bsp(mtime[1:-1], C, dev).T, axis=0)

        dztTp = [dzt.reshape(self.k_mod.l+1, self.k_mod.u_dim).T for dzt in dz.T]

        # get a list over time of command values u(t)
        utTp = map(self.k_mod.phi_2, dztTp)

        # get a list over time of values q(t)
        qtTp = map(self.k_mod.phi_1, dztTp)

        ## Obstacles constraints
        # N_s*nb_obst_detected
        obst_cons = []
        for m in self._detected_obst_idxs:
            obst_cons += [self._obst[m].pt_2_obst(np.squeeze(np.asarray(qt[0:2, 0].T)), self.rho)\
                    for qt in qtTp]

        ## Max speed constraints
        # N_s*u_dim inequations
        max_speed_cons = list(itertools.chain.from_iterable(
                [[self.k_mod.u_max[i, 0]-abs(ut[i, 0]) for i in range(self.k_mod.u_dim)]\
                for ut in utTp]))

        # Create final array
        ieq_cons = obst_cons + max_speed_cons
        # Count how many inequations are not respected
        self._unsatisf_ieq_values = [ieq for ieq in ieq_cons if ieq < 0]
        return np.asarray(ieq_cons)

    def _co_criterion(self, x):
        """ Cost function to be minimized used for optimizing
        the first and intermediaries sections of the plan when conflicts are detected.

        Input
            *x*: optimization argument.

        Return
            Cost value.

        .. warning:: Optimization solver can misbehave for costs too big (:math:`> 10^{6}`).

        .. note:: This method is just a call to the :meth:`_sa_criterion` method.
        """
        return self._sa_criterion(x)

    def _co_feqcons(self, x):
        """ Calculate the **equations** constraints values for the first and intermadiaries
        section of the plan when there are conflicts.

        Input
            *x*: optimization argument.

        Return
            Array with the equations' values.

        .. note:: This method is just a call to the :meth:`_sa_feqcons` method.
        """
        return self._sa_feqcons(x)

    def _co_fieqcons(self, x):
        """ Calculate the **inequations** constraints values for the first and intermadiaries
        section of the plan when there are conflicts.

        The following expressions are evaluated:

        .. math::
            \left\lbrace\\begin{array}{lcl}
            u_{max} - |\\varphi_2(z(t_{1,\ sec}),\dotsc,z^{(l)}(t_{1,\ sec}))|\\\\
            u_{max} - |\\varphi_2(z(t_{2,\ sec}),\dotsc,z^{(l)}(t_{2,\ sec}))|\\\\
            \dotsc\\\\
            u_{max} - |\\varphi_2(z(t_{N_s-1,\ sec}),\dotsc,z^{(l)}(t_{N_s-1,\ sec}))|\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{1,\ sec}),\dotsc,z^{(l)}(t_{1,\ sec})), O_0)\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{2,\ sec}),\dotsc,z^{(l)}(t_{2,\ sec})), O_0)\\\\
            \dotsc\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{N_s-1,\ sec}),
            \dotsc,z^{(l)}(t_{N_s-1,\ sec})), O_0)\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{1,\ sec}),\dotsc,z^{(l)}(t_{1,\ sec})), O_1)\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{2,\ sec}),\dotsc,z^{(l)}(t_{2,\ sec})), O_1)\\\\
            \dotsc\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{N_s-1,\ sec}),
            \dotsc,z^{(l)}(t_{N_s-1,\ sec})), O_1)\\\\
            \dotsc\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{1,\ sec}),
            \dotsc,z^{(l)}(t_{1,\ sec})), O_{M-1})\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{2,\ sec}),
            \dotsc,z^{(l)}(t_{2,\ sec})), O_{M-1})\\\\
            \dotsc\\\\
            \mathrm{pt2obstacle}(\\varphi_1(z(t_{N_s-1,\ sec}),
            \dotsc,z^{(l)}(t_{N_s-1,\ sec})), O_{M-1})\\\\
            \mathrm{collision()}\\\\
            \mathrm{communication()}\\\\
            \mathrm{deviation()}
            \end{array}\\right.

        where:
            =========== =================================================
            :math:`sec` indicates the current plan section.
            :math:`O_i` is the ith detected obstacle.
            :math:`M`   is the number of detected obstacles.
            :math:`N_s` is the number of time samples for a plan section.
            =========== =================================================

        .. todo:: Add the expression for collision, communication and deviation constraints.

        Input
            *x*: optimization argument.

        Return
            Array with the inequations' values.
        """
        C = x.reshape(self._n_ctrlpts, self.k_mod.u_dim)

        # get a list over time of the matrix [z dz ddz](t) t in [tk, tk+Tp]
        dz = self._comb_bsp(self._mtime[1:], C, 0).T
        for dev in range(1, self.k_mod.l+1):
            dz = np.append(dz, self._comb_bsp(self._mtime[1:], C, dev).T, axis=0)

        dztTp = [dzt.reshape(self.k_mod.l+1, self.k_mod.u_dim).T for dzt in dz.T]

        # get a list over time of command values u(t)
        utTp = map(self.k_mod.phi_2, dztTp)

        # get a list over time of values q(t)
        qtTp = map(self.k_mod.phi_1, dztTp)

        ## Obstacles constraints
        # N_s*nb_obst_detected
        obst_cons = []
        for m in self._detected_obst_idxs:
            obst_cons += [self._obst[m].pt_2_obst(np.squeeze(np.asarray(qt[0:2, 0].T)), self.rho)\
                    for qt in qtTp]

        ## Max speed constraints
        # N_s*u_dim inequations
        max_speed_cons = list(itertools.chain.from_iterable(
                [[self.k_mod.u_max[i, 0]-abs(ut[i, 0]) for i in range(self.k_mod.u_dim)]\
                for ut in utTp]))

        ## Communication constraints
        com_cons = []
        for p in self._com_robots_idx:
            for i in range(1, self._sa_dz.shape[1]):
                if self._com_link.done_planning[p] == 1:
                    d_ip = LA.norm(dz[0:2, i-1] - np.asarray(\
                            [self._com_link.latest_z[p][0], \
                            self._com_link.latest_z[p][1]]))
                else:
                    d_ip = LA.norm(dz[0:2, i-1] - np.asarray(\
                            [self._com_link.intended_path_z1[p][i], \
                            self._com_link.intended_path_z2[p][i]]))
                com_cons.append(self._com_range - self._safe_epsilon - d_ip)

        ## Collision constraints
        collision_cons = []
        for p in self._collision_robots_idx:
            for i in range(1, self._sa_dz.shape[1]):
                if self._com_link.done_planning[p] == 1:
                    d_secu = self.rho
                    d_ip = LA.norm(dz[0:2, i-1] - np.asarray(\
                            [self._com_link.latest_z[p][0], \
                            self._com_link.latest_z[p][1]]))
                else:
                    d_secu = 2*self.rho
                    d_ip = LA.norm(dz[0:2, i-1] - np.asarray(\
                            [self._com_link.intended_path_z1[p][i], \
                            self._com_link.intended_path_z2[p][i]]))
                collision_cons.append(d_ip - d_secu - self._safe_epsilon)

        ## Deformation from intended path constraint
        deform_cons = []
        for i in range(1, self._sa_dz.shape[1]):
            d_ii = LA.norm(self._sa_dz[0:2, i] - dz[0:2, i-1])
            deform_cons.append(self._def_epsilon - d_ii)

        # Create final array
        ieq_cons = obst_cons + max_speed_cons + com_cons + collision_cons + deform_cons

        # Count how many inequations are not respected
        unsatisf_list = [ieq for ieq in ieq_cons if ieq < 0]
        self._unsatisf_ieq_values = unsatisf_list

        # return arrray where each element is an inequation constraint
        return np.asarray(ieq_cons)

    def _compute_conflicts(self):
        """ Determine the list of conflictous robots among all other robots.
        This method updates the :attr:`_conflict_robots_idx` private attribute.
        """
        self._collision_robots_idx = []
        self._com_robots_idx = []

        for i in [j for j in range(self._n_robots) if j != self.eyed]:
            if self._com_link.done_planning[i] == 1:
                d_secu = self.rho
                linspeed_max = self.k_mod.u_max[0, 0]
            else:   # TODO each robot must know the radius of the other robot
                d_secu = 2*self.rho
                linspeed_max = 2*self.k_mod.u_max[0, 0]

            d_ip = LA.norm(self._latest_z - self._com_link.latest_z[i])

            # TODO shouldn't it be Tc instead of Tp?
            if d_ip <= d_secu + linspeed_max*self._Tp:
                self._collision_robots_idx.append(i)

            if i in self._neigh: # if the ith robot is a communication neighbor
                # TODO right side of condition should be min(self._com_range, ...
                # ... self._com_link.com_range[i])
                if d_ip + linspeed_max*self._Tp >= self._com_range:
                    self._com_robots_idx.append(i)

        self._conflict_robots_idx = self._collision_robots_idx + self._com_robots_idx

    def _solve_opt_pbl(self):
        """ Call the optimization solver with the appropriate parameters and parse
        the information returned by it. This method updates the :attr:`_C`, :attr:`_dt_final`
        and :attr:`_t_final` attributes.
        """

        if self._plan_state != 'ls':
            if self._std_alone:
                p_criterion = self._sa_criterion
                p_eqcons = self._sa_feqcons
                p_ieqcons = self._sa_fieqcons
            else:
                p_criterion = self._co_criterion
                p_eqcons = self._co_feqcons
                p_ieqcons = self._co_fieqcons

            init_guess = self._C[0:self._n_ctrlpts,:].reshape(self._n_ctrlpts*self.k_mod.u_dim)
            acc = self._acc
            maxit = self._maxit if self._plan_state == 'ms' else self._fs_maxit

        else:
            if self._std_alone:
                p_criterion = self._ls_sa_criterion
                p_eqcons = self._ls_sa_feqcons
                p_ieqcons = self._ls_sa_fieqcons
            else:
                p_criterion = self._ls_co_criterion
                p_eqcons = self._ls_co_feqcons
                p_ieqcons = self._ls_co_fieqcons

            init_guess = np.append(np.asarray([self._est_dtime]),
                    self._C[0:self._n_ctrlpts,:].reshape(self._n_ctrlpts*self.k_mod.u_dim))
            acc = self._acc
            maxit = self._ls_maxit

        output = fmin_slsqp(
            p_criterion,
            init_guess,
            eqcons=(),
            f_eqcons=p_eqcons,
            ieqcons=(),
            f_ieqcons=p_ieqcons,
            iprint=0,
            iter=maxit,
            acc=acc,
            full_output=True)

            #imode = output[3]
            # TODO handle optimization exit mode
        if self._plan_state == 'ls':
            self._C[0:self._n_ctrlpts,:] = output[0][1:].reshape(self._n_ctrlpts, self.k_mod.u_dim)
            self._dt_final = output[0][0]
            self._t_final = self._mtime[0] + self._dt_final
        else:
            self._C[0:self._n_ctrlpts,:] = output[0].reshape(self._n_ctrlpts, self.k_mod.u_dim)
#            #imode = output[3]
#            # TODO handle optimization exit mode

        self._n_it = output[2]
        self._exit_mode = output[4]

    def _plan_section(self):
        """ This method takes care of planning a section of the final path over a :math:`T_{d/p}`
        time horizon.

        It also performs syncronization and data exchange among the robots.
        """

        btic = time.time()

        # update detected obstacles list
        self._detect_obst()

        def gen_ctrlpts_from_curve(last_ctrl_pt, curve):
            """ Interpolate a given curve by Bezier splines defined by its control points.
            """

            aux_t = np.linspace(self._knots[0], self._knots[-1], self._n_ctrlpts)

            # create b-spline representation of that curve
            tck = [si.splrep(aux_t, c, task=-1, t=self._knots[self._d:-self._d], k=self._d-1)\
                    for c in curve]

            # get the ctrl points
            ctrl = [tck_elem[1][0:-self._d,] for tck_elem in tck]

            # initiate C
            for i in range(self.k_mod.z_dim):
                self._C[0:self._n_ctrlpts, i] = ctrl[i]

        # first guess for ctrl pts
        if self._plan_state == 'ms':
            # get the direction to target state unit vector
            direc = self._final_z - self._latest_z
            direc = direc/LA.norm(direc)

            # estimate position of the last ctrl point
            last_ctrl_pt = self._latest_z+self._D*direc

            # create positions thru time assuming that the speed is constant (thus the linspace)
            curve = []
            for i in range(self.k_mod.z_dim):
                curve += [np.linspace(self._latest_z[i,0], last_ctrl_pt[i,0], self._n_ctrlpts)]

            gen_ctrlpts_from_curve(last_ctrl_pt, curve)

        elif self._plan_state == 'ls':
            # final state
            last_ctrl_pt = self._final_z

            last_displ = (self.k_mod.u_final[0,0] + self.k_mod.u_max[0,0])/2.\
                    *self._est_dtime/(self._n_ctrlpts-1) #+ np.finfo(float).eps

            minus2th_pt = (last_ctrl_pt.T - last_displ*np.array(\
                    [np.cos(self.k_mod.q_final[-1, 0]), np.sin(self.k_mod.q_final[-1, 0])])).T

            # create positions thru time assuming that the speed is constant (thus the linspace)
            curve1 = []
            curve2 = []
            for i in range(self.k_mod.z_dim):
                curve1 += [np.insert(np.linspace(self._latest_z[i,0], minus2th_pt[i,0],
                        self._n_ctrlpts-1), self._n_ctrlpts-1, last_ctrl_pt[i,0])]
                curve2 += [np.linspace(self._latest_z[i,0], last_ctrl_pt[i,0],
                        self._n_ctrlpts)]
            curve = [
                    [(ec1 + ec2)/2. for ec1, ec2 in zip(c1, c2)] for c1, c2 in zip(curve1, curve2)]
#            curve = curve1
            gen_ctrlpts_from_curve(last_ctrl_pt, curve)

#            eps = self.k_mod.u_final[0,0]*self._est_dtime/(self._n_ctrlpts-1) + np.finfo(float).eps

            # correcting the [-2]th ctrl pt so it take in account the final state orientation and speed
#            eps = np.finfo(float).eps
#            minus2th_C = last_ctrl_pt.T - eps*np.array(\
#                    [np.cos(self.k_mod.q_final[-1, 0]), np.sin(self.k_mod.q_final[-1, 0])])
#            self._C[self._n_ctrlpts-2,] = minus2th_C

        else: # 'fs'
            # get the direction to target state unit vector
            direc = self._final_z - self._latest_z
            direc = direc/LA.norm(direc)

            # estimate position of the last ctrl point
            last_ctrl_pt = self._latest_z+self._D*direc

            first_displ = (self.k_mod.u_init[0,0] + self.k_mod.u_max[0,0])/2.\
                    *self._Td/(self._n_ctrlpts-1) #+ np.finfo(float).eps

            _2th_pt = (self._latest_z.T + first_displ*np.array(\
                    [np.cos(self.k_mod.q_init[-1, 0]), np.sin(self.k_mod.q_init[-1, 0])])).T

            # create positions thru time assuming that the speed is constant (thus the linspace)
            curve1 = []
            curve2 = []
            for i in range(self.k_mod.z_dim):
                sec2last = np.linspace(_2th_pt[i,0], last_ctrl_pt[i,0],self._n_ctrlpts-1)
                first = self._latest_z[i,0]
                curve1 += [np.insert(sec2last, 0, first)]
                curve2 += [np.linspace(self._latest_z[i,0], last_ctrl_pt[i,0],
                        self._n_ctrlpts)]
            curve = [[(ec1 + ec2)/2. for ec1, ec2 in zip(c1, c2)] for c1, c2 in zip(curve1, curve2)]
#            curve = curve1
            gen_ctrlpts_from_curve(last_ctrl_pt, curve)

            # correcting the [2]th ctrl pt so it take in account the initial state orientation and speed
#            eps = self.k_mod.u_init[0,0]*self._Td/(self._n_ctrlpts-1) + 1e-6
#            _2th_C = final_ctrl_pt.T + eps*np.array(\
#                    [np.cos(self.k_mod.q_final[-1, 0]), np.sin(self.k_mod.q_final[-1, 0])])
#            self._C[self._n_ctrlpts-2,] = minus2_C

#        print 'First guess :\n', self._C

        self._std_alone = True

        tic = time.time()
        self._solve_opt_pbl()
        toc = time.time()

        # No need to sync process here, the intended path does impact the conflicts computation

        self._log('i', 'R{rid}@tkref={tk:.4f}: Time to solve stand alone optimization '
                'problem: {t}'.format(rid=self.eyed, t=toc-tic, tk=self._mtime[0]))
        self._log('i', 'R{rid}@tkref={tk:.4f}: N of unsatisfied eq: {ne}'\
                .format(rid=self.eyed, t=toc-tic, tk=self._mtime[0], ne=len(self._unsatisf_eq_values)))
        self._log('i', 'R{rid}@tkref={tk:.4f}: N of unsatisfied ieq: {ne}'\
                .format(rid=self.eyed, t=toc-tic, tk=self._mtime[0], ne=len(self._unsatisf_ieq_values)))
        self._log('i', 'R{rid}@tkref={tk:.4f}: Summary: {summ} after {it} it.'\
                .format(rid=self.eyed, t=toc-tic, tk=self._mtime[0], summ=self._exit_mode, it=self._n_it))

#        if self._final_step:
        if self._plan_state == 'ls':
            self._knots = self._gen_knots(self._mtime[0], self._t_final)
            self._mtime = np.linspace(self._mtime[0], self._t_final, self._N_s)

        time_idx = None if self._plan_state == 'ls' else self._Tcd_idx+1

        dz = self._comb_bsp(self._mtime, self._C[0:self._n_ctrlpts,:], 0).T
        for dev in range(1, self.k_mod.l+1):
            dz = np.append(dz, self._comb_bsp(
                    self._mtime, self._C[0:self._n_ctrlpts,:], dev).T, axis=0)

#        TODO verify process safety
        for i in range(dz.shape[1]):
            self._com_link.intended_path_z1[self.eyed][i] = dz[0, i]
            self._com_link.intended_path_z2[self.eyed][i] = dz[1, i]

        self._compute_conflicts()
        self._log('d', 'R{0}@tkref={1:.4f}: $$$$$$$$ CONFLICT LIST $$$$$$$$: {2}'
                .format(self.eyed, self._mtime[0], self._conflict_robots_idx))

        # Sync with every robot on the conflict list
        #  1. notify every robot waiting on this robot that it ready for conflict solving
        with self._conflict_syncer_conds[self.eyed]:
            self._conflict_syncer[self.eyed].value = 1
            self._conflict_syncer_conds[self.eyed].notify_all()
        #  2. check if the robots on this robot conflict list are ready
        for i in self._conflict_robots_idx:
            with self._conflict_syncer_conds[i]:
                if self._conflict_syncer[i].value == 0:
                    self._conflict_syncer_conds[i].wait()
        # Now is safe to read the all robots' in the conflict list intended paths (or are done planning)

#        if self._conflict_robots_idx != [] and False:
        if self._conflict_robots_idx != [] and self._plan_state != 'ls':

            self._std_alone = False

#            self._conflict_dz = [self._read_com_link()]
#            self._read_com_link()

            self._sa_dz = dz

            tic = time.time()
            self._solve_opt_pbl()
            toc = time.time()

            self._log('i', 'R{rid}@tkref={tk:.4f}: Time to solve optimization probl'
                    'em: {t}'.format(rid=self.eyed, t=toc-tic, tk=self._mtime[0]))
            self._log('i', 'R{rid}@tkref={tk:.4f}: N of unsatisfied eq: {ne}'\
                    .format(rid=self.eyed, t=toc-tic, tk=self._mtime[0], ne=len(self._unsatisf_eq_values)))
            self._log('i', 'R{rid}@tkref={tk:.4f}: N of unsatisfied ieq: {ne}'\
                    .format(rid=self.eyed, t=toc-tic, tk=self._mtime[0], ne=len(self._unsatisf_ieq_values)))
            self._log('i', 'R{rid}@tkref={tk:.4f}: Summary: {summ} after {it} it.'\
                    .format(rid=self.eyed, t=toc-tic, tk=self._mtime[0], summ=self._exit_mode, it=self._n_it))

#            if self._final_step:
            if self._plan_state == 'ls':
                self._knots = self._gen_knots(self._mtime[0], self._t_final)
                self._mtime = np.linspace(self._mtime[0], self._t_final, self._N_s)

            time_idx = None if self._plan_state == 'ls' else self._Tcp_idx+1

            dz = self._comb_bsp(self._mtime[0:time_idx], self._C[0:self._n_ctrlpts,:], 0).T
            for dev in range(1, self.k_mod.l+1):
                dz = np.append(dz, self._comb_bsp(
                        self._mtime[0:time_idx], self._C[0:self._n_ctrlpts,:], dev).T, axis=0)

        # Storing
#        self._all_C[0:self._n_ctrlpts,:] += [self._C[0:self._n_ctrlpts,:]]
        self._all_dz.append(dz[:, 0:time_idx])
        self._all_times.extend(self._mtime[0:time_idx])
        # TODO rejected path

        # Updating

        latest_z = self._all_dz[-1][0:self.k_mod.u_dim, -1].reshape(
                self.k_mod.u_dim, 1)

        # Sync robots here so no robot computing conflict get the wrong latest_z of some robot
        with self._tc_syncer_cond:
            self._tc_syncer.value += 1
            if self._tc_syncer.value != self._n_robots:  # if not all robots are read
                self._log('d', 'R{}: I\'m going to sleep!'.format(self.eyed))
                self._tc_syncer_cond.wait()
            else:                                # otherwise wake up everybody
                self._tc_syncer_cond.notify_all()
            self._tc_syncer.value -= 1            # decrement syncer (idem)
#            self._com_link.latest_z[self.eyed] = latest_z
            for i in range(self.k_mod.u_dim):
                self._com_link.latest_z[self.eyed][i] = latest_z[i, 0]

        with self._conflict_syncer_conds[self.eyed]:
            self._conflict_syncer[self.eyed].value = 0


#        if not self._final_step:
        if self._plan_state != 'ls':
            if self._std_alone == False:
                self._knots = self._knots + self._Tcp
                self._mtime = [tk+self._Tcp for tk in self._mtime]
            else:
                self._knots = self._knots + self._Tcd
                self._mtime = [tk+self._Tcd for tk in self._mtime]
            self._latest_z = latest_z
            self._latest_q = self.k_mod.phi_1(self._all_dz[-1][:, -1].reshape(
                    self.k_mod.l+1, self.k_mod.u_dim).T)
            self._latest_u = self.k_mod.phi_2(self._all_dz[-1][:, -1].reshape(
                    self.k_mod.l+1, self.k_mod.u_dim).T)
            if self._plan_state == 'fs':
                self._plan_state = 'ms'
        btoc = time.time()
        self._all_comp_times.append(btoc-btic)

#        print 'Solved C :\n', self._C

    def _plan(self):
        """ Motion/path planner method. At the end of its execution :attr:`rtime`, :attr:`ctime`
        and :attr:`sol` attributes will be updated with the plan for completing the mission.
        """

        self._log('i', 'R{rid}: Init motion planning'.format(rid=self.eyed))

#        self._final_step = False
        self._plan_state = 'fs'

        self._knots = self._gen_knots(self._t_init, self._Td)
        self._mtime = np.linspace(self._t_init, self._Td, self._N_s)

        # while the remaining dist is greater than the max dist during Tp
#        while LA.norm(self._latest_z - self._final_z) > self._D:

        while True:
            remaining_dist = LA.norm(self._latest_z - self._final_z)
#            if remaining_dist < self._D:
#                break
#            elif remaining_dist < self._ls_min_dist + self._Tc*self.k_mod.u_max[0,0] and False:
            if remaining_dist < self._ls_min_dist + self._Tc*self.k_mod.u_max[0,0]:
                self._log('d', 'R{0}: LAST STEP'.format(self.eyed))
                self._log('d', 'R{0}: Approx remaining dist: {1}'.format(self.eyed, remaining_dist))
                self._log('d', 'R{0}: Usual approx plan dist: {1}'.format(self.eyed, self._D))
                self._log('d', 'R{0}: Approx gain in dist: {1}'.format(self.eyed, self._D-remaining_dist))
                self._log('d', 'R{0}: Last step min dist: {1}'.format(self.eyed, self._ls_min_dist))
                scale_factor = min(1., remaining_dist/self.k_mod.u_max[0,0]/self._Td)
                self._n_knots = max(int(round(self._n_knots*scale_factor)), self._d-1)
                self._n_ctrlpts = self._n_knots + self._d - 1 # nb of ctrl points
                self._N_s = max(int(round(self._N_s*scale_factor)), self._n_ctrlpts+1)
                self._log('i', 'R{0}: scale {1} Ns {2:d} Nk {3:d}'.format(self.eyed,
                        scale_factor, self._N_s, self._n_knots))
                break

            self._plan_section()
            self._log('i', 'R{}: --------------------------'.format(self.eyed))

#        self._final_step = True
        self._plan_state = 'ls'
        self._est_dtime = LA.norm(self._latest_z - self._final_z)/self.k_mod.u_max[0, 0]

        self._knots = self._gen_knots(self._mtime[0], self._mtime[0]+self._est_dtime)
        self._mtime = np.linspace(self._mtime[0], self._mtime[0]+self._est_dtime, self._N_s)

        self._plan_section()
        self._log('i', 'R{}: Finished motion planning'.format(self.eyed))
        self._log('i', 'R{}: --------------------------'.format(self.eyed))

        self.sol[self.eyed] = self._all_dz
        self.rtime[self.eyed] = self._all_times
        self.ctime[self.eyed] = self._all_comp_times
        self._com_link.done_planning[self.eyed] = 1

        #  Notify every robot waiting on this robot that it is ready for the conflict solving
        with self._conflict_syncer_conds[self.eyed]:
            self._conflict_syncer[self.eyed].value = 1
            self._conflict_syncer_conds[self.eyed].notify_all()

        # Make sure any robot waiting on this robot awake before returning
        with self._tc_syncer_cond:
            self._tc_syncer.value += 1               # increment synker
            if self._tc_syncer.value == self._n_robots:  # if all robots are read
                self._tc_syncer_cond.notify_all()

###############################################################################
# World
###############################################################################
class WorldSim(object):
    """ This class is a container of all simulation elements and also the
    interface for running the simulation.
    """
    def __init__(self, sim_id_str, path, robots, obstacles, phy_boundary, plot=False):
        self._sn = sim_id_str
        self._robs = robots
        self._obsts = obstacles
        self._Tc = robots[0]._Tc
        self._ph_bound = phy_boundary
        self._plot = plot
        self._direc = path

    def run(self):
        """ Run simulation by first calling the :py:meth:`multiprocessing.Process.start` method on
        the :attr:`Robot.planning_process` to
        initiate the motion planning of each robot. And secondly by parsing theirs solutions and
        prompting the option to plot/save it.
        """

        n_robots = len(self._robs)
        N_s = self._robs[0]._N_s

        ####################################################################
        # Multiprocessing stuff ############################################
        ####################################################################
        # Log Lock
        log_lock = mpc.Lock()

        # Conditions
        tc_syncer_cond = mpc.Condition()
        conflict_syncer_conds = [mpc.Condition() for i in range(n_robots)]

        # Shared memory for sync
        tc_syncer = mpc.Value('I', 0) # unsigned int
        conflict_syncer = [mpc.Value('I', 0) for i in range(n_robots)]

        # Shared memory for communication link
        done_planning = [mpc.Value('I', 0) for i in range(n_robots)]
        intended_path_z1 = [mpc.Array('d', N_s*[0.0]) for i in range(n_robots)]
        intended_path_z2 = [mpc.Array('d', N_s*[0.0]) for i in range(n_robots)]
        latest_z = [mpc.Array('d', [r.k_mod.q_init[0, 0], \
                r.k_mod.q_init[1, 0]]) for r in self._robs]

        # Packing shared memory into a RobotMsg object
        com_link = RobotMsg(done_planning, intended_path_z1, intended_path_z2, latest_z)

        # More complex and expense shared memory thru a server process manager ...
        # ... (because they can support arbitrary object types)
        manager = mpc.Manager()
        solutions = manager.list(range(n_robots))
        race_time = manager.list(range(n_robots))
        comp_time = manager.list(range(n_robots))

        # Setting multiprocessing stuff for every robot
        [r.set_option('log_lock', log_lock) for r in self._robs]
        [r.set_option('tc_syncer', tc_syncer) for r in self._robs]
        [r.set_option('tc_syncer_cond', tc_syncer_cond) for r in self._robs]
        [r.set_option('conflict_syncer', conflict_syncer) for r in self._robs]
        [r.set_option('conflict_syncer_conds', conflict_syncer_conds) for r in self._robs]
        [r.set_option('com_link', com_link) for r in self._robs]
        [r.set_option('sol', solutions) for r in self._robs]
        [r.set_option('rtime', race_time) for r in self._robs]
        [r.set_option('ctime', comp_time) for r in self._robs]
        ####################################################################
        ####################################################################
        ####################################################################

        # Make all robots plan their trajectories
        [r.planning_process.start() for r in self._robs]
        [r.planning_process.join() for r in self._robs]

        # Reshaping the solution
        path = range(len(self._robs))
        seg_pts_idx = [[] for _ in range(len(self._robs))]
        for i in range(len(self._robs)):
            path[i] = self._robs[0].sol[i][0]
            seg_pts_idx[i] += [0]
            for p in self._robs[0].sol[i][1:]:
                c = path[i].shape[1]
                seg_pts_idx[i] += [c]
                path[i] = np.append(path[i], p, axis=1)

        # From [z dz ddz](t) get q(t) and u(t)
        zdzddz = range(len(self._robs))
        for i in range(len(self._robs)):
            zdzddz[i] = [
                    z.reshape(self._robs[i].k_mod.l+1, self._robs[i].k_mod.u_dim).T\
                    for z in path[i].T]

        # get a list over time of command values u(t)
        ut = range(len(self._robs))
        for i in range(len(self._robs)):
            ut[i] = map(self._robs[i].k_mod.phi_2, zdzddz[i])
            # Fixing division by near-zero value when calculating angspeed for plot
            ut[i][0][1,0] = self._robs[i].k_mod.u_init[1,0]
            ut[i][-1][1,0] = self._robs[i].k_mod.u_final[1,0]

        # get a list over time of values q(t)
        qt = range(len(self._robs))
        for i in range(len(self._robs)):
            qt[i] = map(self._robs[i].k_mod.phi_1, zdzddz[i])

        # get "race" times (time spend on each section of the path planning)
        rtime = range(len(self._robs))
        for i in range(len(self._robs)):
            rtime[i] = self._robs[0].rtime[i]

        # get computation times
        ctime = range(len(self._robs))
        for i in range(len(self._robs)):
            ctime[i] = self._robs[0].ctime[i]

        # Logging simulation summary
        for i in range(len(self._robs)):
            ctime_len = len(ctime[i])
            if ctime_len > 1:
                g_max_idx = np.argmax(ctime[i][1:]) + 1
            else:
                g_max_idx = 0
            logging.info('R{rid}: TOT: {d}'.format(rid=i, d=rtime[i][-1]))
            logging.info('R{rid}: NSE: {d}'.format(rid=i, d=ctime_len))
            logging.info('R{rid}: FIR: {d}'.format(rid=i, d=ctime[i][0]))
            logging.info('R{rid}: LAS: {d}'.format(rid=i, d=ctime[i][-1]))
            if g_max_idx == ctime_len-1:
                logging.info('R{rid}: LMA: {d}'.format(rid=i, d=1))
            else:
                logging.info('R{rid}: LMA: {d}'.format(rid=i, d=0))
            if ctime_len > 2:
                logging.info('R{rid}: MAX: {d}'.format(rid=i, d=max(ctime[i][1:-1])))
                logging.info('R{rid}: MIN: {d}'.format(rid=i, d=min(ctime[i][1:-1])))
                logging.info('R{rid}: AVG: {d}'.format(rid=i, d=np.mean(ctime[i][1:-1])))
                logging.info('R{rid}: RMP: {d}'.format(rid=i, d=max(ctime[i][1:-1])/self._Tc))
                logging.info('R{rid}: RMG: {d}'.format(rid=i, d=max(ctime[i][1:])/self._Tc))
            else:
                logging.info('R{rid}: MAX: {d}'.format(rid=i, d=ctime[i][-1]))
                logging.info('R{rid}: MIN: {d}'.format(rid=i, d=ctime[i][-1]))
                logging.info('R{rid}: AVG: {d}'.format(rid=i, d=ctime[i][-1]))
                logging.info('R{rid}: RMP: {d}'.format(rid=i, d=ctime[i][-1]/self._Tc))
                logging.info('R{rid}: RMG: {d}'.format(rid=i, d=ctime[i][-1]/self._Tc))

        # PLOT/SAVE ###########################################################

#        while True:
#            try:
#                x = str(raw_input("Want to see the result plotted? [y/n]: "))
#                if x != 'n' and x != 'N':
#                    if x != 'y' and x != 'Y':
#                        print("I'll take that as an \'n\'.")
#                        x = 'n'
#                    else:
#                        x = 'y'
#                else:
#                    x = 'n'
#                break
#            except ValueError:
#                print("Oops! That was no valid characther. Try again...")
#            except KeyboardInterrupt:
#                print("\nGood bye!")
#                return
#
#        if x == 'n':
#            return

        # Interactive plot
        if self._plot:
            plt.ion()

        try:
            os.mkdir(self._direc+'/images/')
        except OSError:
            print('Probably the output directory '+self._direc+\
                    '/images'+' already exists, going to overwrite content')

        try:
            os.mkdir(self._direc+'/images/'+self._sn)
        except OSError:
            print('Probably the output directory '+self._direc+\
                    '/images/'+self._sn+' already exists, going to overwrite content')

        fig_s, axarray = plt.subplots(2)
        axarray[0].set_ylabel('v(m/s)')
        axarray[0].set_title('Linear speed')
        axarray[1].set_xlabel('time(s)')
        axarray[1].set_ylabel('w(rad/s)')
        axarray[1].set_title('Angular speed')

        fig = plt.figure()
        ax = fig.gca()
        ax.set_xlabel('x(m)')
        ax.set_ylabel('y(m)')
        ax.set_title('Generated trajectory')
        ax.axis('equal')

        aux = np.linspace(0.0, 1.0, 1e2)
        colors = [[i, 1.0-i, np.random.choice(aux)] for i in np.linspace(0.0, 1.0, len(self._robs))]

        while True:
            # Creating obstacles in the plot
            [obst.plot(fig, offset=self._robs[0].rho) for obst in self._obsts]

            plt_paths = range(len(self._robs))
            plt_seg_pts = range(len(self._robs))
            plt_robots_c = range(len(self._robs))
            plt_robots_t = range(len(self._robs))
            for i in range(len(self._robs)):
                plt_paths[i], = ax.plot(path[i][0, 0], path[i][1, 0], color=colors[i])
                plt_seg_pts[i], = ax.plot(path[i][0, seg_pts_idx[i][0]], \
                        path[i][1, seg_pts_idx[i][0]], color=colors[i], ls='None', marker='o', markersize=5)
                plt_robots_c[i] = plt.Circle(
                        (path[i][0, 0], path[i][1, 0]), # position
                        self._robs[i].rho, # radius
                        color='m',
                        ls = 'solid',
                        fill=False)
                rho = self._robs[i].rho
                xy = np.array(
                        [[rho*np.cos(qt[i][0][-1, 0])+path[i][0, 0], \
                        rho*np.sin(qt[i][0][-1, 0])+path[i][1, 0]],
                        [rho*np.cos(qt[i][0][-1, 0]-2.5*np.pi/3.0)+path[i][0, 0], \
                        rho*np.sin(qt[i][0][-1, 0]-2.5*np.pi/3.0)+path[i][1, 0]],
                        [rho*np.cos(qt[i][0][-1, 0]+2.5*np.pi/3.0)+path[i][0, 0], \
                        rho*np.sin(qt[i][0][-1, 0]+2.5*np.pi/3.0)+path[i][1, 0]]])
                plt_robots_t[i] = plt.Polygon(xy, color='m', fill=True, alpha=0.2)

            [ax.add_artist(r) for r in plt_robots_c]
            [ax.add_artist(r) for r in plt_robots_t]
            
            ctr = 0
            while True:
                end = 0

                for i in range(len(self._robs)):
                    if ctr < path[i].shape[1]:
                        plt_paths[i].set_xdata(path[i][0, 0:ctr+1])
                        plt_paths[i].set_ydata(path[i][1, 0:ctr+1])
                        aux = [s for s in seg_pts_idx[i] if  ctr > s]
                        plt_seg_pts[i].set_xdata(path[i][0, aux])
                        plt_seg_pts[i].set_ydata(path[i][1, aux])
                        plt_robots_c[i].center = path[i][0, ctr], path[i][1, ctr]
                        rho = self._robs[i].rho
                        xy = np.array([
                            [rho*np.cos(qt[i][ctr][-1, 0])+path[i][0, ctr],
                            rho*np.sin(qt[i][ctr][-1, 0])+path[i][1, ctr]],
                            [rho*np.cos(qt[i][ctr][-1, 0]-2.5*np.pi/3.0)+path[i][0, ctr],
                            rho*np.sin(qt[i][ctr][-1, 0]-2.5*np.pi/3.0)+path[i][1, ctr]],
                            [rho*np.cos(qt[i][ctr][-1, 0]+2.5*np.pi/3.0)+path[i][0, ctr],
                            rho*np.sin(qt[i][ctr][-1, 0]+2.5*np.pi/3.0)+path[i][1, ctr]]])
                        plt_robots_t[i].set_xy(xy)
                    else:
                        end += 1
                ctr += 1
                if end == len(self._robs):
                    break
#                time.sleep(0.01)
                if self._plot:
                    ax.relim()
                    ax.autoscale_view(True, True, True)
                    fig.canvas.draw()
                    fig.savefig(self._direc+'/images/'+self._sn+'/multirobot-path-'+str(ctr)+'.png',\
                            bbox_inches='tight')
            #end

            ax.relim()
            ax.autoscale_view(True, True, True)
            fig.canvas.draw()
            fig.savefig(self._direc+'/images/'+self._sn+'/multirobot-path.png', bbox_inches='tight')

            for i in range(len(self._robs)):
                linspeed = [x[0, 0] for x in ut[i]]
                angspeed = [x[1, 0] for x in ut[i]]
                axarray[0].plot(rtime[i], linspeed, color=colors[i])
                axarray[1].plot(rtime[i], angspeed, color=colors[i])
            axarray[0].grid()
            axarray[1].grid()
            axarray[0].set_ylim([0.0, 1.1*self._robs[0].k_mod.u_max[0, 0]])
            axarray[1].set_ylim([-5.5, 5.5])
            fig_s.canvas.draw()
            fig_s.savefig(self._direc+'/images/'+self._sn+'/multirobot-vw.png',bbox_inches='tight')

            if self._plot:
                while True:
                    try:
                        x = str(raw_input("Plot again? [y/n]: "))
                        if x != 'n' and x != 'N':
                            if x != 'y' and x != 'Y':
                                print("I'll take that as an \'n\'.")
                                x = 'n'
                            else:
                                x = 'y'
                        else:
                            x = 'n'
                        break
                    except ValueError:
                        print("Oops! That was no valid characther. Try again...")
                    except KeyboardInterrupt:
                        print("Good bye!")
                        return
    
                if x == 'n':
                    return
            else:
                return

            axarray[0].cla()
            axarray[1].cla()
            fig.gca().cla()
        #end
    #end
#end
###############################################################################
# Script
###############################################################################

def _isobstok(obsts, c, r):
    if len(obsts) > 0:
        for obst in obsts:
            if (c[0]-obst[0][0])**2 + (c[1]-obst[0][1])**2 < (r+obst[1])**2:
                return False
    return True

def rand_round_obst(no, boundary, min_radius=0.15, max_radius=0.40):
    """ Generate random values for creating :class:`RoundObstacle` objects.
    Obstacles will have a random radius between *min_radius* and *max_radius*
    meters and a random position such as the whole obstacle will be within the area
    determined by the *boundary* parameter.

    Input
        *no*: number of round obstacles to be generated.

        *boundary*: area where the obstacles shall be placed.

        *min_radius*: minimum radius length.

        *max_radius*: maximum radius length.

    Return
        List containing the information to initialize a :class:`RoundObstacle` object.
    """
    N = 1/0.0001
    radius_range = np.linspace(min_radius, max_radius, N)
    x_range =np.linspace(boundary.x_min+max_radius, boundary.x_max-max_radius, N)
    y_range =np.linspace(boundary.y_min+max_radius, boundary.y_max-max_radius, N)

    obsts = []
    i=0
    while i < no:
        x = np.random.choice(x_range)
        y = np.random.choice(y_range)
        r = np.random.choice(radius_range)
        if _isobstok(obsts, [x, y], r):
            obsts += [([x, y], r)]
            i += 1
    return obsts

# MAIN ########################################################################

if __name__ == '__main__':

    ls_time_opt_scale = 1.0

    def add_cmdline_options(parser):
        parser.add_option("-L", "--savelog", dest='savelog',
                action='store_true', default=False,
                help='create log file')
        parser.add_option("-v", "--veiwplot", dest='plot',
                action='store_true', default=False,
                help='view results in a iteractive plot and save each frame')
        parser.add_option('-P', '--storepath', dest='direc',
                help='path for storing simulation data', metavar='PATH', default='./simoutput')
        parser.add_option('-b', '--robots', dest='no_robots', default=2,
                action='store', type='int', help='number of robots')
        parser.add_option('-o', '--obstacles', dest='no_obsts', default=3,
                action='store', type='int', help='number of obstacles')
        parser.add_option('-c', '--comphorizon', dest='time_c',
                action='store', type='float', help='computation time horizon', default=0.5)
        parser.add_option('-p', '--planhorizon', dest='time_p', default=2.0,
                action='store', type='float', help='planning time horizon')
        parser.add_option('-s', '--timesampling', dest='no_s', default=14,
                action='store', type='int', help='number of time samples')
        parser.add_option('-k', '--knots', dest='no_knots', default=5,
                action='store', type='int', help='number of internal knots')
        parser.add_option('-a', '--accuracy', dest='acc', default=1E-3,
                action='store', type='float', help='optimization accuracy')
        parser.add_option('-m', '--maxiteration', dest='max_it', default=15,
                action='store', type='int',
                help='number of maximum iterations for intermadiaries plan sections')
        parser.add_option('-I', '--lastmaxiteration', dest='l_max_it', default=20,
                action='store', type='int',
                help='number of maximum iterations for last plan sections')
        parser.add_option('-i', '--firstmaxiteration', dest='f_max_it', default=40,
                action='store', type='int',
                help='number of maximum iterations for first plan sections')
        parser.add_option('-d', '--deviation', dest='deps', default=5.0,
                action='store', type='float',
                help='path deviation from initial one when dealing with conflict (in meters)')
        parser.add_option('-f', '--safety', dest='seps', default=0.1,
                action='store', type='float',
                help='minimal allowed distance between two robots (in meters)')
        parser.add_option('-r', '--detectionradius', dest='drho', default=6.0,
                action='store', type='float',
                help='detection radius within which the robot can detect an obstacle (in meters)')
        parser.add_option('-l', '--lastsecmindist', dest='ls_min_dist', default=0.5,
                action='store', type='float',
                help=\
                'minimal distance left for completing the last section of the planning (in meters)')
        return

    scriptname = sys.argv[0]

    parser = OptionParser()
    add_cmdline_options(parser)
    (options, args) = parser.parse_args()

    try:
        os.mkdir(options.direc)
    except OSError:
        print('Probably the output directory '+options.direc+' already exists.')

    sim_id = '_'+str(options.no_robots)+\
            '_'+str(options.no_obsts)+\
            '_'+str(options.time_c)+\
            '_'+str(options.time_p)+\
            '_'+str(options.no_s)+\
            '_'+str(options.no_knots)+\
            '_'+str(options.acc)+\
            '_'+str(options.max_it)+\
            '_'+str(options.f_max_it)+\
            '_'+str(options.l_max_it)+\
            '_'+str(options.deps)+\
            '_'+str(options.seps)+\
            '_'+str(options.drho)+\
            '_'+str(options.ls_min_dist)

    if options.savelog:
        flog = options.direc+'/'+scriptname[0:-3]+sim_id+'.log'
        logging.basicConfig(filename=flog, format='%(levelname)s:%(message)s', \
                filemode='w', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    boundary = Boundary([-12.0, 12.0], [-12.0, 12.0])

# Generate random round obstacles
#    obst_info = rand_round_obst(options.no_obsts, Boundary([-1., 1.], [0.8, 5.2]))

    # 0 obsts
    if options.no_obsts == 0:
        obst_info = []
    # 2 obsts
    elif options.no_obsts == 2:
        obst_info = [#([0.0, 1.6], 0.3),
                ([0.6, 3.0], 0.35), ([-0.6, 3.0], 0.35)]
    # 3 obsts
    elif options.no_obsts == 3:
        obst_info = [([0.55043504350435046, 1.9089108910891091], 0.31361636163616358),
                ([-0.082028202820282003, 3.6489648964896491], 0.32471747174717469),
                ([0.37749774977497741, 4.654905490549055], 0.16462646264626463)]
    # 6 obsts
    elif options.no_obsts == 6:
        obst_info = [([-0.35104510451045101, 1.3555355535553557], 0.38704870487048704),
                ([0.21441144114411448, 2.5279927992799281], 0.32584258425842583),
                ([-0.3232123212321232, 4.8615661566156621], 0.23165816581658166),
                ([0.098239823982398278, 3.975877587758776], 0.31376637663766377),
                ([0.62277227722772288, 1.247884788478848], 0.1802030203020302),
                ([1.16985698569856988, 3.6557155715571559], 0.25223522352235223)]
    else:
        logging.info("Using 3 obstacles configuration")
        obst_info = [([0.0, 1.6], 0.3),
                ([0.6, 3.0], 0.35), ([-0.6, 3.0], 0.35)]

    obstacles = [RoundObstacle(i[0], i[1]) for i in obst_info]

    # Polygon obstacle exemple
#    obstacles += [PolygonObstacle(np.array([[0,1],[1,0],[3,0],[4,2]]))]

    kine_models = [UnicycleKineModel(
            [-0.05, 0., np.pi/2.], # q_initial
            [0.1,  7.0, np.pi/2.], # q_final
            [1.0,  0.0],          # u_initial
            [1.0,  0.0],          # u_final
            [1.0,  5.0]),          # u_max
            UnicycleKineModel(
            [0.4,  0., np.pi/2.], # q_initial
            [-0.4, 5.0, np.pi/2.], # q_final
            [1.0,  0.0],          # u_initial
            [1.0,  0.0],          # u_final
            [1.0,  5.0])]          # u_max

    robots = []
    for i in range(options.no_robots):
        if i-1 >= 0 and i+1 < options.no_robots:
            neigh = [i-1, i+1]
        elif i-1 >= 0:
            neigh = [i-1]
        else:
            neigh = [i+1]
        robots += [Robot(
            i,                      # Robot ID
            kine_models[i],         # kinetic model
            obstacles,              # all obstacles
            boundary,               # planning plane boundary
            neigh,                  # neighbors to whom this robot shall talk ...
                                    #...(used for conflict only, not for real comm between process)
            N_s=options.no_s,                # numbers samplings for each planning interval
            n_knots=options.no_knots,# number of knots for b-spline interpolation
            Tc=options.time_c,       # computation time
            Tp=options.time_p,       # planning horizon
            Td=options.time_p,
            def_epsilon=options.deps,       # in meters
            safe_epsilon=options.seps,      # in meters
            detec_rho=options.drho,
            ls_time_opt_scale = ls_time_opt_scale,
            ls_min_dist = options.ls_min_dist)] 

    [r.set_option('acc', options.acc) for r in robots] 
    [r.set_option('maxit', options.max_it) for r in robots] 
    [r.set_option('ls_maxit', options.l_max_it) for r in robots] 
    [r.set_option('fs_maxit', options.f_max_it) for r in robots]

    world_sim = WorldSim(sim_id, options.direc, robots, obstacles, boundary, plot=options.plot)

    summary_info = world_sim.run() # run simulation

