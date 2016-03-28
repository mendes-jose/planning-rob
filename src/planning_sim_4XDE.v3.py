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
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import random

__version__ = '1.0.2'

def _frange(initial, final, step):
		""" Float point range function with round at the int(round(1./step))+4 decimal position
		"""
		_range = []
		n = 0
		while n*step+initial < final:
			_range += [round(n*step+initial, 4+int(round(1./step)))]
			n+=1
		return _range

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
#		ax.plot(self.x, self.y, 'k.')

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
#		ax.plot(self.x, self.y, 'k.')

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
		""" Map angles (:math:`\\theta \in R`) to unsigned angles
		(:math:`\\theta \in [0, 2\pi)`).

		.. note:: Method not used.
		"""
		while angle < 0.0:
			angle += 2*np.pi
		while angle >= 2*np.pi:
			angle -= 2*np.pi
		return 2.*np.pi+angle if angle < 0.0 else angle

	@staticmethod
	def _signed_angle(angle):
		""" Map angles (:math:`\\theta \in R`) to signed angles
		(:math:`\\theta \in [-pi, +pi)`).
		"""
		while angle < -np.pi:
			angle += 2*np.pi;
		while angle >= np.pi:
			angle -= 2*np.pi;
		return angle;

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
			\\theta
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
			eps = np.finfo(float).eps
			den = zl[0, 1]**2 + zl[1, 1]**2
			den = den if abs(den) > eps else eps
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
			eps = np.finfo(float).eps
			eps2thequarter = eps**(.25)
			# Prevent division by zero
			dz_norm = LA.norm(zl[:, 1])
			# min_den_norm = np.finfo(float).eps**(-4)
			# den = dz_norm if dz_norm >= min_den_norm else min_den_norm
			den = dz_norm if abs(dz_norm) > eps2thequarter else eps2thequarter
			# den = dz_norm
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
	def __init__(self, dp, ip_z1, ip_z2, lz, lq, lu):
		self.done_planning = dp
		""" Flag to indicate that the robot has finished its planning process.
		"""
		self.intended_path_z1 = ip_z1
		""" Intended path (z1 coordiante).
		"""
		self.intended_path_z2 = ip_z2
		""" Intended path (z2 coordiante).
		"""
		self.latest_q = lq
		""" q value calculated on the previews planned section.
		"""
		self.latest_u = lu
		""" u value calculated on the previews planned section.
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
			neigh,				  # neighbors to whom this robot shall talk...
									# ...(used for conflict only, not communic)
			N_s=20,
			N_ssol=100,
			n_knots=6,
			t_init=0.0,
			t_sup=1e10,
			Tc=1.0,
			Tp=3.0,
			Td=3.0,
			rho=0.35,
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
		self.accel = None
		self.p_sol = None
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
		self._N_ssol = N_ssol # no of samples for discretization of time
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
		#td_step = (self._Td)/(self._N_s-1)
		#tp_step = (self._Tp)/(self._N_s-1)
		tp_step_sol = (self._Tp)/(self._N_ssol-1)
		self._Tc_idx_sol = max(1, int(round(self._Tc/tp_step_sol)))

		# optimization solver parameters
		self._maxit = 100
		self._fs_maxit = 100
		self._ls_maxit = 100
		self._acc = 1e-6

		# init planning
		self._detected_obst_idxs = range(len(self._obst))
		self._known_obst_idxs = []

		self._latest_q = self.k_mod.q_init
		self._latest_u = self.k_mod.u_init
		self._latest_z = self.k_mod.z_init
		self._final_z = self.k_mod.z_final
		self._waypoint = self._final_z
		self._latest_rot2rob_mat = np.eye(2)
		self._latest_rot2ref_mat = np.eye(2)

		self._D = self._Tp * self.k_mod.u_max[0, 0]

		self._d = self.k_mod.l+2 # B-spline order (integer | d > l+1)
		self._n_ctrlpts = self._n_knots + self._d - 1 # nb of ctrl points

		self._C = np.zeros((self._n_ctrlpts, self.k_mod.u_dim))

		self._all_dz = []
		self._accel = []
		self._all_C = []
		self._all_times = []
		self._all_comp_times = []

		self._unsatisf_eq_values = []
		self._unsatisf_ieq_values = []
		self._unsatisf_acc_ieq_values = []
		self._unsatisf_collcons_ieq_values = []
		self._exit_mode = 0
		self._n_it = 0
		self._t_final = 0

		self._opttime = np.linspace(self._t_init, self._t_init+self._Td, self._N_s)
		self._soltime = np.linspace(self._t_init, self._t_init + self._Td, self._N_ssol)

		self._conflict_robots_idx = []
		self._collision_robots_idx = []
		self._com_robots_idx = []

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
			elif name == 'xacc':
				self._xacc = value
				self._acc = value
			elif name == 'facc':
				self._facc = value
			elif name == 'eacc':
				self._eacc = value
			elif name == 'iacc':
				self._iacc = value
			elif name == 'com_link':
				self._com_link = value
			elif name == 'sol':
				self.sol = value
			elif name == 'p_sol':
				self.p_sol = value
			elif name == 'accel':
				self.accel = value
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

	def _num_phi_3(self, t, C):
		#np.append(dztinit, self._comb_bsp([self._opttime[0]], C, dev).T, axis=1)
		"""
		"""
		eps = np.finfo(float).eps
		# sqrteps = sqrt(eps)

		# h = eps if t < eps else sqrteps*t
		h = 1e-12

		dt = 2*h

		# ta = t-h
		# tp = t+h
		# for vta, vtp, i in zip(ta, tp, range(len(ta))):
		#	 if vta < t[0]:
		#		 ta[i] = t[0]
		#	 if vta > t[-1]:
		#		 ta[i] = t[-1]
		#	 if vtp < t[0]:
		#		 tp[i] = t[0]
		#	 if vtp > t[-1]:
		#		 tp[i] = t[-1]

		dz = self._comb_bsp(t, C, 0).T + np.asarray(self._latest_z)
		for dev in range(1, self.k_mod.l+1):
			dz = np.append(dz, self._comb_bsp(t, C, dev).T, axis=0)

		dztTp = [dzt.reshape(self.k_mod.l+1, self.k_mod.u_dim).T for dzt in dz.T]

		# get a list over time of command values u(t)
		ut = map(self.k_mod.phi_2, dztTp)

		dz = self._comb_bsp([x-h for x in t], C, 0).T + np.asarray(self._latest_z)
		for dev in range(1, self.k_mod.l+1):
			dz = np.append(dz, self._comb_bsp([x-h for x in t], C, dev).T, axis=0)

		dztTp = [dzt.reshape(self.k_mod.l+1, self.k_mod.u_dim).T for dzt in dz.T]

		# get a list over time of command values u(t)
		uta = map(self.k_mod.phi_2, dztTp)

		dz = self._comb_bsp([x+h for x in t], C, 0).T + np.asarray(self._latest_z)
		for dev in range(1, self.k_mod.l+1):
			dz = np.append(dz, self._comb_bsp([x+h for x in t], C, dev).T, axis=0)

		dztTp = [dzt.reshape(self.k_mod.l+1, self.k_mod.u_dim).T for dzt in dz.T]

		# get a list over time of command values u(t)
		utp = map(self.k_mod.phi_2, dztTp)

		ret = [(utpi - utai)/dt for utpi, utai in zip(utp[1:-1], uta[1:-1])]
		ret = [(utp[0] - ut[0])/dt/2] + ret + [(ut[-1] - uta[-1])/dt/2]
		# print ret
		return ret

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
				idx_list += [[idx, dist-self.rho-self._obst[idx].radius]]
		idx_list.sort(key=lambda x:x[1])
		self._detected_obst_idxs = [i[0] for i in idx_list]
		self._known_obst_idxs += [i[0] for i in idx_list if i[0] not in self._known_obst_idxs]

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
		return self._ls_time_opt_scale*(x[0]+self._opttime[0])**2

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
		t_final = self._opttime[0]+dt_final
		C = x[1:].reshape(self._n_ctrlpts, self.k_mod.u_dim)

		# get new knots and the flat output initial and final values for this plan section.
		self._knots = self._gen_knots(self._opttime[0], t_final)
		dztinit = self._comb_bsp([self._opttime[0]], C, 0).T
		for dev in range(1, self.k_mod.l+1):
			dztinit = np.append(dztinit, self._comb_bsp([self._opttime[0]], C, dev).T, axis=1)

		dztfinal = self._comb_bsp([t_final], C, 0).T
		for dev in range(1, self.k_mod.l+1):
			dztfinal=np.append(dztfinal, self._comb_bsp([t_final], C, dev).T, axis=1)

		# calculate equations
		final_theta_rf = UnicycleKineModel._signed_angle(self.k_mod.q_final[-1] - self._latest_q[-1])
		final_q_rf = np.vstack((self._latest_rot2rob_mat*(self.k_mod.q_final[0:2, 0]-self._latest_z), final_theta_rf))

		eq_cons = list(np.squeeze(np.array(self.k_mod.phi_1(dztinit))))+\
				list(np.squeeze(np.array(self.k_mod.phi_1(dztfinal)-final_q_rf)))+\
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
		t_final = self._opttime[0]+dt_final
		C = x[1:].reshape(self._n_ctrlpts, self.k_mod.u_dim)

		self._knots = self._gen_knots(self._opttime[0], t_final)

		mtime = np.linspace(self._opttime[0], t_final, self._N_s)

		# get a list over time of the matrix [z dz ddz](t) t in [tk, tk+Tp]
		dz = self._comb_bsp(mtime, C, 0).T # + np.asarray(self._latest_z)
		for dev in range(1, self.k_mod.l+2):
			dz = np.append(dz, self._comb_bsp(mtime, C, dev).T, axis=0)

		dztTp = [dzt.reshape(self.k_mod.l+2, self.k_mod.u_dim).T for dzt in dz.T]

		# get a list over time of command values u(t)
		utTp = map(self.k_mod.phi_2, dztTp[1:-1])

		# get a list over time of values q(t)
		qtTp = map(self.k_mod.phi_1, dztTp[1:-1])

		# get a list over time of values a(t)
		# atTp = self._num_phi_3(mtime, C)
		atTp = map(self.k_mod.phi_3, dztTp)

		## Obstacles constraints
		# N_s*nb_obst_detected
		obst_cons = []
		for m in self._detected_obst_idxs:
			obst_cons += [self._obst[m].pt_2_obst(np.squeeze(np.asarray((self._latest_rot2ref_mat*qt[0:2, 0]+np.asarray(self._latest_z)).T)), self.rho)\
					for qt in qtTp]

		## Max speed constraints
		# N_s*u_dim inequations
		max_speed_cons = list(itertools.chain.from_iterable(
				[[self.k_mod.u_max[i, 0]-abs(ut[i, 0]) \
				for i in range(self.k_mod.u_dim)]for ut in utTp]))

		## Max acceleration constraints
		# N_s*u_dim inequations
		max_acc_cons = list(itertools.chain.from_iterable(
				[[self.k_mod.acc_max[i, 0]-abs(at[i, 0]) for i in range(self.k_mod.u_dim)]\
				for at in atTp]))

		# Create final array
		ieq_cons = obst_cons + max_speed_cons + max_acc_cons

		# Count how many inequations are not respected
		self._unsatisf_ieq_values = [ieq for ieq in ieq_cons if ieq < 0]
		self._unsatisf_acc_ieq_values = [ieq for ieq in max_acc_cons if ieq < 0]

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

		dz = self._comb_bsp([self._opttime[-1]], C, 0).T
		for dev in range(1, self.k_mod.l+1):
			dz = np.append(dz, self._comb_bsp([self._opttime[-1]], C, dev).T, axis=1)
		qTp = self.k_mod.phi_1(dz)
		ptTp = self._latest_rot2ref_mat*qTp[0:-1, 0] + np.asarray(self._latest_z)

		cte = 1.1 # TODO no magic constants
		pos2target = self._waypoint - self._latest_z
		pos2target_norm = LA.norm(pos2target)
		goal_pt = self._latest_z+pos2target/pos2target_norm*cte*self._D
		#if pos2target_norm > cte*self._D:
		#	goal_pt = self._latest_z+pos2target/pos2target_norm*cte*self._D
		#elif pos2target_norm < self._D:
		#	goal_pt = self._latest_z+pos2target/pos2target_norm*cte*self._D
		#else:
		#	goal_pt = self.k_mod.q_final[0:-1, :]
		cost = np.sqrt(LA.norm(ptTp - goal_pt))
		#cost = LA.norm(qTp[0:-1, :] - self.k_mod.q_final[0:-1, :])
		# TODO
		if cost > 1e6:
			#cost = self.prev_cost
			self._log('d', 'R{}: Big problem {}'.format(self.eyed, cost))
			print ('R{}: Big problem {}'.format(self.eyed, cost))

		#self.prev_cost = cost
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

		dztinit = self._comb_bsp([self._opttime[0]], C, 0).T # + np.asarray(self._latest_z)
		for dev in range(1, self.k_mod.l+1):
			dztinit = np.append(dztinit, self._comb_bsp([self._opttime[0]], C, dev).T, axis=1)

		# compute equations

		# dimension: q_dim + u_dim
		pose_cons = list(np.squeeze(np.array(self.k_mod.phi_1(dztinit))))
		speed_cons = list(np.squeeze(np.array(self.k_mod.phi_2(dztinit)-self._latest_u)))

		eq_cons = pose_cons + speed_cons

#		self._eq_ratio = [pose_cons[i]/self._eacc for eq, i in zip(pose_cons, range(len(pose_cons)))] +[speed_cons[i]/self._eacc for eq, i in zip(speed_cons, range(len(speed_cons)))]


		# Count how many equations are not respected
		self._unsatisf_eq_values = [eq for eq in eq_cons if eq != 0]

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
		dz = self._comb_bsp(self._opttime, C, 0).T # + np.asarray(self._latest_z)
		for dev in range(1, self.k_mod.l+2):
			dz = np.append(dz, self._comb_bsp(self._opttime, C, dev).T, axis=0)

		dztTp = [dzt.reshape(self.k_mod.l+2, self.k_mod.u_dim).T for dzt in dz.T]

		# get a list over time of command values u(t)
		utTp = map(self.k_mod.phi_2, dztTp[1:])

		# get a list over time of values q(t)
		qtTp = map(self.k_mod.phi_1, dztTp[1:])

		# get a list over time of values a(t)
		# atTp = self._num_phi_3(self._opttime, C)
		atTp = map(self.k_mod.phi_3, dztTp)

		## Obstacles constraints
		# N_s*nb_obst_detected
		obst_cons = []
		for m in self._detected_obst_idxs:
			obst_cons += [self._obst[m].pt_2_obst(np.squeeze(np.asarray((self._latest_rot2ref_mat*qt[0:2, 0]+np.asarray(self._latest_z)).T)), self.rho)\
					for qt in qtTp]

		## Max speed constraints
		# N_s*u_dim inequations
		max_speed_cons = list(itertools.chain.from_iterable(
				[[self.k_mod.u_max[i, 0]-abs(ut[i, 0]) for i in range(self.k_mod.u_dim)]\
				for ut in utTp]))

		## Max acceleration constraints
		# N_s*u_dim inequations
		max_acc_cons = list(itertools.chain.from_iterable(
				[[self.k_mod.acc_max[i, 0]-abs(at[i, 0]) for i in range(self.k_mod.u_dim)]\
				for at in atTp]))

		# Create final array
		ieq_cons = obst_cons + max_speed_cons + max_acc_cons

		# Count how many inequations are not respected
		unsatisf_list = [ieq for ieq in ieq_cons if ieq < 0]
		# print unsatisf_list
		self._unsatisf_ieq_values = unsatisf_list
		self._unsatisf_acc_ieq_values = [ieq for ieq in max_acc_cons if ieq < 0]


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
		t_final = self._opttime[0]+dt_final
		C = x[1:].reshape(self._n_ctrlpts, self.k_mod.u_dim)

		self._knots = self._gen_knots(self._opttime[0], t_final)

		mtime = np.linspace(self._opttime[0], t_final, self._N_s)

		# get a list over time of the matrix [z dz ddz](t) t in [tk, tk+Tp]
		dz = self._comb_bsp(mtime, C, 0).T # + np.asarray(self._latest_z)
		for dev in range(1, self.k_mod.l+2):
			dz = np.append(dz, self._comb_bsp(mtime, C, dev).T, axis=0)

		dztTp = [dzt.reshape(self.k_mod.l+2, self.k_mod.u_dim).T for dzt in dz.T]

		# get a list over time of command values u(t)
		utTp = map(self.k_mod.phi_2, dztTp[1:-1])

		# get a list over time of values q(t)
		qtTp = map(self.k_mod.phi_1, dztTp[1:-1])

		# get a list over time of values a(t)
		# atTp = self._num_phi_3(mtime, C)
		atTp = map(self.k_mod.phi_3, dztTp)

		## Obstacles constraints
		# N_s*nb_obst_detected
		obst_cons = []
		for m in self._detected_obst_idxs:
			obst_cons += [self._obst[m].pt_2_obst(np.squeeze(np.asarray((self._latest_rot2ref_mat*qt[0:2, 0]+np.asarray(self._latest_z)).T)), self.rho)\
					for qt in qtTp]

		## Max speed constraints
		# N_s*u_dim inequations
		max_speed_cons = list(itertools.chain.from_iterable(
				[[self.k_mod.u_max[i, 0]-abs(ut[i, 0]) for i in range(self.k_mod.u_dim)]\
				for ut in utTp]))
			   
		## Max acceleration constraints
		# N_s*u_dim inequations
		max_acc_cons = list(itertools.chain.from_iterable(
				[[self.k_mod.acc_max[i, 0]-abs(at[i, 0]) for i in range(self.k_mod.u_dim)]\
				for at in atTp]))

		# Create final array
		ieq_cons = obst_cons + max_speed_cons + max_acc_cons

		# Count how many inequations are not respected
		self._unsatisf_ieq_values = [ieq for ieq in ieq_cons if ieq < 0]
		self._unsatisf_acc_ieq_values = [ieq for ieq in max_acc_cons if ieq < 0]
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
		# self._log('d', 'R{rid}@tkref={tk:.4f} inside CO IEQ'.format(rid=self.eyed, tk=self._opttime[0]))

		C = x.reshape(self._n_ctrlpts, self.k_mod.u_dim)

		# get a list over time of the matrix [z dz ddz](t) t in [tk, tk+Tp]
		dz = self._comb_bsp(self._opttime, C, 0).T #+ np.asarray(self._latest_z)
		for dev in range(1, self.k_mod.l+2):
			dz = np.append(dz, self._comb_bsp(self._opttime, C, dev).T, axis=0)

		dztTp = [dzt.reshape(self.k_mod.l+2, self.k_mod.u_dim).T for dzt in dz.T]

		# get a list over time of command values u(t)
		utTp = map(self.k_mod.phi_2, dztTp[1:])

		# get a list over time of values q(t)
		qtTp = map(self.k_mod.phi_1, dztTp[1:])

		# get a list over time of values a(t)
		# atTp = self._num_phi_3(self._opttime, C)
		atTp = map(self.k_mod.phi_3, dztTp)

		## Obstacles constraints
		# N_s*nb_obst_detected
		obst_cons = []
		for m in self._detected_obst_idxs:
			obst_cons += [self._obst[m].pt_2_obst(np.squeeze(np.asarray((self._latest_rot2ref_mat*qt[0:2, 0]+np.asarray(self._latest_z)).T)), self.rho)\
					for qt in qtTp]

		## Max speed constraints
		# N_s*u_dim inequations
		max_speed_cons = list(itertools.chain.from_iterable(
				[[self.k_mod.u_max[i, 0]-abs(ut[i, 0]) for i in range(self.k_mod.u_dim)]\
				for ut in utTp]))

		## Max acceleration constraints
		# N_s*u_dim inequations
		max_acc_cons = list(itertools.chain.from_iterable(
				[[self.k_mod.acc_max[i, 0]-abs(at[i, 0]) for i in range(self.k_mod.u_dim)]\
				for at in atTp]))

		## Communication constraints
		com_cons = []
		for p in self._com_robots_idx:
			for i in range(1, self._sa_dz.shape[1]):
				myPos = np.squeeze(np.asarray((self._latest_rot2ref_mat*np.matrix(dz[0:2, i-1]).T+self._latest_z).T))
				if self._com_link.done_planning[p] == 1:
					otherPos = np.asarray([self._com_link.latest_z[p][0], self._com_link.latest_z[p][1]])
					d_ip = LA.norm(myPos - otherPos)
				else:
					otherPos = np.asarray([self._com_link.intended_path_z1[p][i], self._com_link.intended_path_z2[p][i]])
					d_ip = LA.norm(myPos - otherPos)
				com_cons.append(self._com_range - self._safe_epsilon - d_ip)

		## Collision constraints
		collision_cons = []
		for p in self._collision_robots_idx:
			for i in range(1, self._sa_dz.shape[1]):

				myPos = np.squeeze(np.asarray((self._latest_rot2ref_mat*np.matrix(dz[0:2, i-1]).T+self._latest_z).T))


				if self._com_link.done_planning[p] == 1:
					d_secu = 2*self.rho
					
					otherPos = np.asarray([self._com_link.latest_z[p][0], self._com_link.latest_z[p][1]])

					d_ip = LA.norm(myPos - otherPos)

					# d_ip2 = ((float(dz[0, i-1]) - float(self._com_link.latest_z[p][0]))**2 +  (float(dz[1, i-1]) - float(self._com_link.latest_z[p][1]))**2)**(.5)
					# self._log('d', 'R{rid}@tkref={tk:.4f} collision CONS\nDist norm: {dist1}\nDist sqrt{dist2}'.format(rid=self.eyed, tk=self._opttime[0], dist1=d_ip, dist2=d_ip2))
				else:
					d_secu = 2*self.rho

					otherPos = np.asarray([self._com_link.intended_path_z1[p][i], self._com_link.intended_path_z2[p][i]])
					#print 'otherpos[', i, ': ', otherPos
					#print 'mypos[', i, ': ', myPos
					
					#print myPos.shape
					#print otherPos.shape

					d_ip = LA.norm(myPos - otherPos)
					#d_ip2 = ((float(dz[0, i-1]) - float(self._com_link.intended_path_z1[p][i]))**2 +  (float(dz[1, i-1]) - float(self._com_link.intended_path_z2[p][i]))**2)**(.5)
					# self._log('d', 'R{rid}@tkref={tk:.4f} collision CONS\nDist norm: {dist1}\nDist sqrt: {dist2}'.format(rid=self.eyed, tk=self._opttime[0], dist1=d_ip, dist2=d_ip2))
				collision_cons.append(d_ip - d_secu - self._safe_epsilon)

		## Deformation from intended path constraint
		deform_cons = []
		for i in range(1, self._sa_dz.shape[1]):
			d_ii = LA.norm(self._sa_dz[0:2, i] - dz[0:2, i-1])
			deform_cons.append(self._def_epsilon - d_ii)

		# Create final array
		ieq_cons = obst_cons + max_speed_cons + collision_cons + max_acc_cons #+ ocm_cons + collision_cons #+ deform_cons

		# Count how many inequations are not respected
		unsatisf_list = [ieq for ieq in ieq_cons if ieq < 0]
		self._unsatisf_ieq_values = unsatisf_list
		self._unsatisf_acc_ieq_values = [ieq for ieq in max_acc_cons if ieq < 0]
		self._unsatisf_collcons_ieq_values = [ieq for ieq in collision_cons if ieq < 0]

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
			#print 'self._com_link.latest_z', self._com_link.latest_z
			d_ip = LA.norm(self._latest_z.T - np.asarray([self._com_link.latest_z[i][0], self._com_link.latest_z[i][1]]))
			#d_ip = ((float(self._latest_z[0]) - float(self._com_link.latest_z[i][0]))**2 +  (float(self._latest_z[1]) - float(self._com_link.latest_z[i][1]))**2)**(.5)

			# self._log('d', 'R{rid}@tkref={tk:.4f} in Compute Conflicts\n{p1}\n{p2}, {p22}\nDist: {dist}\n{lins}\n{tp}\n{dsecu}'.format(rid=self.eyed, tk=self._opttime[0], p1=self._latest_z, p2=self._com_link.latest_z[i][0], p22=self._com_link.latest_z[i][1], dist=d_ip, lins=linspeed_max, tp=self._Tp, dsecu=d_secu))

			# TODO shouldn't it be Tc instead of Tp?
			if d_ip <= d_secu + linspeed_max*self._Tp:
				self._collision_robots_idx.append(i)

			if i in self._neigh: # if the ith robot is a communication neighbor
				# TODO right side of condition should be min(self._com_range, ...
				# ... self._com_link.com_range[i])
				if d_ip + linspeed_max*self._Tp >= self._com_range:
					self._com_robots_idx.append(i)

		self._conflict_robots_idx = self._collision_robots_idx + self._com_robots_idx

	def _robotDirecControl(self, accelChoice):
		output = [[], 0, 0, 0, 0]

		###################################################
		init_direc = np.vstack((np.cos(self._latest_q[-1]), np.sin(self._latest_q[-1])))

		robot2waypt = self._waypoint - self._latest_z
		robot2waypt_dist = LA.norm(robot2waypt)
		waypt_direc = robot2waypt/robot2waypt_dist
		#waypt_theta = np.arctan2(waypt_direc[1], waypt_direc[0])
		self._latest_rot2rob_mat = np.hstack((np.multiply(init_direc, np.vstack((1,-1))), np.flipud(init_direc)))

		waypt_direc = self._latest_rot2rob_mat*waypt_direc

		# Get planning horizon
		planHor = self._est_dtime if self._plan_state == 'ls' else self._Td

		# Get acceleration
		#accelChoice = random.choice(_frange(.75*self.k_mod.acc_max[0,0], self.k_mod.acc_max[0,0], 0.01))
		#accelChoice = accelChoice if doAccelerate else -accelChoice

		max_displ_var = (planHor/(self._n_ctrlpts-1))*self.k_mod.u_max[0,0]
		
		# Create a sampled trajectory for a "bounded uniformed accelerated motion" in x axis
		# And create a sampled trajectory for a "bounded uniformed accelerated motion" in waypt dir
		curve_latest_theta = [[]]*self.k_mod.z_dim
		curve_waypoint_direc = [[]]*self.k_mod.z_dim

		for i in range(self.k_mod.z_dim):
			curve_latest_theta[i] = np.zeros(self._n_ctrlpts)
			curve_waypoint_direc[i] = np.zeros(self._n_ctrlpts)

		prev_displ = 0.0

		for i in range(self._n_ctrlpts)[1:]:

			delta_t = i*(planHor/(self._n_ctrlpts-1))
			
			displ = self._latest_u[0,0]*delta_t + accelChoice/2.*delta_t**2

			displ = max(displ, prev_displ) if displ-prev_displ < max_displ_var  else prev_displ + max_displ_var

			prev_displ = displ

			curve_latest_theta[0][i] = displ

			for j in range(len(waypt_direc)):
				curve_waypoint_direc[j][i] = displ*waypt_direc[j]

		curve = [[]]*self.k_mod.z_dim
		for i in range(self.k_mod.z_dim):
			n = 1.5 # TODO no magic number, specially hardcoded
			p = [(float(j)/(self._n_ctrlpts-1))**n for j in range(self._n_ctrlpts)]
			curve[i] = np.array([p[j] * curve_waypoint_direc[i][j] + (1-p[j]) * curve_latest_theta[i][j] for j in range(self._n_ctrlpts)])

		self._gen_ctrlpts_from_curve(curve)
		###################################################

		output[0] = self._C[0:self._n_ctrlpts,:].reshape(self._n_ctrlpts*self.k_mod.u_dim)

		if self._plan_state == 'ls':
			output[0] = np.insert(output[0], 0, self._est_dtime)

		return output

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
			print 'Estimate last delta t: ', self._est_dtime
			init_guess = np.append(np.asarray([self._est_dtime]),
					self._C[0:self._n_ctrlpts,:].reshape(self._n_ctrlpts*self.k_mod.u_dim))
			acc = self._acc
			maxit = self._ls_maxit

		# if True: #self._plan_state != 'ls':
		if self._plan_state != 'ls':
		#if False:
		#if True:
			output = fmin_slsqp(p_criterion, init_guess, eqcons=(), f_eqcons=p_eqcons, ieqcons=(), f_ieqcons=p_ieqcons, iprint=0, iter=maxit, acc=acc, full_output=True)
		else:

			output = [[], 0, 0, 0, 0]
			output[0] = self._C[0:self._n_ctrlpts,:].reshape(self._n_ctrlpts*self.k_mod.u_dim)

			if self._plan_state == 'ls':
				#print output[0].shape
				output[0] = np.insert(output[0], 0, self._est_dtime)
				#print output[0].shape

		if any([i < -1e-2 for i in self._unsatisf_acc_ieq_values+self._unsatisf_collcons_ieq_values]): # TODO no magic number, specially hardcoded
		#if False:
			self._unsatisf_acc_ieq_values = []
			self._unsatisf_collcons_ieq_values = []

			random.seed(self.eyed)
			accelChoice = random.choice(np.linspace(-self.k_mod.acc_max[0,0], self.k_mod.acc_max[0,0], 100))
			output = self._robotDirecControl(accelChoice)

			print '______________________________BAD OPT. USING "DirecCtrl" at {acc} m/s2 (ui = {ui} m/s)__________________________'.format(acc=accelChoice, ui=self._latest_u[0,0])
			self._log('d', 'R{rid}@tkref={tk:.4f}: ______________________________BAD OPT. USING "DirecCtrl" at {acc} m/s2 (ui = {ui} m/s)__________________________'.format(rid=self.eyed, tk=self._opttime[0], acc=accelChoice, ui=self._latest_u[0,0]))

		if self._plan_state == 'ls':
			#print output[0].shape
			#print self._C.shape
			self._C[0:self._n_ctrlpts,:] = (self._latest_rot2ref_mat*(output[0][1:].reshape(self._n_ctrlpts, self.k_mod.u_dim)).T).T
			self._dt_final = output[0][0]
			print 'Last delta t: ', self._dt_final
			self._t_final = self._opttime[0] + self._dt_final
		else:
			# print self._latest_rot2ref_mat.shape
			# print (output[0].reshape(self._n_ctrlpts, self.k_mod.u_dim))
			self._C[0:self._n_ctrlpts,:] = (self._latest_rot2ref_mat*(output[0].reshape(self._n_ctrlpts, self.k_mod.u_dim)).T).T
#			#imode = output[3]
#			# TODO handle optimization exit mode

		self._n_it = output[2]
		self._exit_mode = output[4]

		#else:
		# if self._plan_state == 'ls':
		# 	self._dt_final = self._est_dtime
		# 	self._t_final = self._opttime[0] + self._dt_final

		# self._C[0:self._n_ctrlpts,:] = (self._latest_rot2ref_mat*self._C[0:self._n_ctrlpts,:].T).T

	def _gen_ctrlpts_from_curve(self, curve):
			""" Interpolate a given curve by Bezier splines defined by its control points.
			"""

			aux_t = np.linspace(self._knots[0], self._knots[-1], self._n_ctrlpts)

			# create b-spline representation of that curve
			tck = [si.splrep(aux_t, c, task=-1, t=self._knots[self._d:-self._d], k=self._d-1) for c in curve]

			# get the ctrl points
			ctrl = [tck_elem[1][0:-self._d,] for tck_elem in tck]

			# initiate C
			for i in range(self.k_mod.z_dim):
				self._C[0:self._n_ctrlpts, i] = ctrl[i]

	def _plan_section(self):
		""" This method takes care of planning a section of the final path over a :math:`T_{d/p}`
		time horizon.

		It also performs syncronization and data exchange among the robots.
		"""

		btic = time.time()

		# update detected obstacles list
		self._detect_obst()


		def isAnyObstInRobotWay2Goal():
			robot2goal = self._final_z - self._latest_z
			robot2goal_dist = LA.norm(robot2goal)
			goal_direc = robot2goal/robot2goal_dist
			goal_theta = np.arctan2(goal_direc[1], goal_direc[0])

			
			for obstIdx in self._known_obst_idxs:
				cent2robot = self._latest_z - np.matrix(self._obst[obstIdx].centroid).T
				cent2goal = self._final_z - np.matrix(self._obst[obstIdx].centroid).T

				segment_norm_goal = LA.norm(cent2goal - cent2robot)

				determinant_p1p2_goal = cent2robot[0, 0]*cent2goal[1, 0] - cent2goal[0, 0]*cent2robot[1, 0]

				discriminant_goal = (self._obst[obstIdx].radius+self.rho)**2 * segment_norm_goal**2 - determinant_p1p2_goal**2

				if discriminant_goal >= 0: # intersection

					robot2obst = -1.*cent2robot

					robot2obst_norm = LA.norm(robot2obst)

					#signed_r2o_proj_on_wdir = robot2obst_norm * np.cos(np.arctan2(robot2obst[1], robot2obst[0])-waypt_theta)

					signed_r2o_proj_on_gdir = robot2obst_norm * np.cos(np.arctan2(robot2obst[1], robot2obst[0])-goal_theta)


					if signed_r2o_proj_on_gdir <= self.k_mod.u_max[0,0]*self._Tc: #0.0
						#and signed_r2o_proj_on_gdir <= self.k_mod.u_max[0,0]*self._Tc:
						print obstIdx, ': obstacle was left behind'
						# obstacle was left behind
						#return (d_theta, d_theta, LA.norm(robot2obst)-rad)
						#return (0.0, 0.0, robot2goal_dist)
						continue

					return True


			return False


		def isObstInRobotsWay(obstIdx):

			robot2goal = self._final_z - self._latest_z
			robot2goal_dist = LA.norm(robot2goal)
			goal_direc = robot2goal/robot2goal_dist
			goal_theta = np.arctan2(goal_direc[1], goal_direc[0])

			robot2waypt = self._waypoint - self._latest_z
			robot2waypt_dist = LA.norm(robot2waypt)
			waypt_direc = robot2waypt/robot2waypt_dist
			waypt_theta = np.arctan2(waypt_direc[1], waypt_direc[0])

			cent2robot = self._latest_z - np.matrix(self._obst[obstIdx].centroid).T
			robot2obst = -1.*cent2robot
			cent2waypt = self._waypoint - np.matrix(self._obst[obstIdx].centroid).T
			cent2goal = self._final_z - np.matrix(self._obst[obstIdx].centroid).T

			segment_norm = LA.norm(cent2waypt - cent2robot)

			determinant_p1p2 = cent2robot[0, 0]*cent2waypt[1, 0] - cent2waypt[0, 0]*cent2robot[1, 0]

			discriminant_way = (self._obst[obstIdx].radius+self.rho)**2 * segment_norm**2 - determinant_p1p2**2


			# if discriminant < 0: # no intersection in waypt dir, check goal
			# 	print obstIdx, ': no intersection in waypt dir, check goal'
			# 	cent2goal = self._final_z - np.matrix(self._obst[obstIdx].centroid).T
			# 	segment_norm = LA.norm(cent2goal - cent2robot)
			# 	determinant_p1p2 = cent2robot[0, 0]*cent2goal[1, 0] - cent2goal[0, 0]*cent2robot[1, 0]

			# 	discriminant2 = (self._obst[obstIdx].radius+self.rho)**2 * segment_norm**2 - determinant_p1p2**2
			# 	if discriminant2 >= 0.0:
			# 		print obstIdx, ': intersection in goal dir found'
			# 		discriminant = discriminant2

			# if discriminant >= 0: # secant or tangent to waypt dir or goal dir
			
			# 	# check if obstacle was left behind
			# 	print obstIdx, ': check if obstacle was left behind'

			# 	robot2obst_norm = LA.norm(robot2obst)

			# 	signed_r2o_proj_on_wdir = robot2obst_norm * np.cos(np.arctan2(robot2obst[1], robot2obst[0])-waypt_theta)

			# 	signed_r2o_proj_on_gdir = robot2obst_norm * np.cos(np.arctan2(robot2obst[1], robot2obst[0])-goal_theta)


			# 	if signed_r2o_proj_on_wdir <= self.k_mod.u_max[0,0]*self._Tc and signed_r2o_proj_on_gdir <= self.k_mod.u_max[0,0]*self._Tc:
			# 	#if signed_r2o_proj_on_wdir <= 0.0 and signed_r2o_proj_on_gdir <= 0.0:
			# 		print obstIdx, ': obstacle was left behind'
			# 		# obstacle was left behind
			# 		#return (d_theta, d_theta, LA.norm(robot2obst)-rad)
			# 		#return (0.0, 0.0, robot2goal_dist)
			# 		return False

			# else: # no intersection to waypt dir or goal dir
			# 	print obstIdx, ': no intersection in waypt dir nor in goal dir'
			# 	#return (0.0, 0.0, robot2goal_dist)
			# 	return False

			if discriminant_way < 0: # no intersection in waypt dir, check goal
				print obstIdx, ': no intersection in waypt dir'
				return False

			#if discriminant_way >= 0: # secant or tangent to waypt dir or goal dir
			else:
				print obstIdx, ': intersection in waypt dir'
				# check if obstacle was left behind
				print obstIdx, ': check if obstacle was left behind'

				robot2obst_norm = LA.norm(robot2obst)

				signed_r2o_proj_on_wdir = robot2obst_norm * np.cos(np.arctan2(robot2obst[1], robot2obst[0])-waypt_theta)

				#signed_r2o_proj_on_gdir = robot2obst_norm * np.cos(np.arctan2(robot2obst[1], robot2obst[0])-goal_theta)


				if signed_r2o_proj_on_wdir <= self.k_mod.u_max[0,0]*self._Tc: #0.0
					#and signed_r2o_proj_on_gdir <= self.k_mod.u_max[0,0]*self._Tc:
					print obstIdx, ': obstacle was left behind'
					# obstacle was left behind
					#return (d_theta, d_theta, LA.norm(robot2obst)-rad)
					#return (0.0, 0.0, robot2goal_dist)
					return False

			# obstacle would be in the robots way but if it knew it can go directly to the goal so the obstacle is not really "in the way" (the way to the greater objective which is the goal)
			if not isAnyObstInRobotWay2Goal():
				print '{}: No robot in the way to goal'.format(self.eyed)
				#time.sleep(2)
				return False

			return True

		def getThetas(obstIdx):

			robot2goal = self._final_z - self._latest_z
			robot2goal_dist = LA.norm(robot2goal)
			goal_direc = robot2goal/robot2goal_dist
			goal_theta = np.arctan2(goal_direc[1], goal_direc[0])

			robot2waypt = self._waypoint - self._latest_z
			robot2waypt_dist = LA.norm(robot2waypt)
			waypt_direc = robot2waypt/robot2waypt_dist
			waypt_theta = np.arctan2(waypt_direc[1], waypt_direc[0])

			cent2robot = self._latest_z - np.matrix(self._obst[obstIdx].centroid).T
			robot2obst = -1.*cent2robot

			rad = self._obst[obstIdx].radius+self.rho

			# solving second degree eq for finding the tow m's
			a = robot2obst[0]**2 - rad**2
			b = -2*robot2obst[0]*robot2obst[1]
			c = robot2obst[1]**2 - rad**2

			discriminant = b**2 - 4*a*c

			if discriminant < 0:
							
				self._log('d', 'R{rid}@tkref={tk:.4f}: $$$$$ Latest Z inside a obstacle!!! $$$$$'.format(rid=self.eyed, tk=self._opttime[0]))
				print 'R{rid}@tkref={tk:.4f}: $$$$$ Latest Z inside a obstacle!!! $$$$$'.format(rid=self.eyed, tk=self._opttime[0])
				#d_theta = UnicycleKineModel._signed_angle(goal_theta - waypt_theta)
				d_theta = 0.0
				#return (d_theta[0,0], d_theta[0,0], LA.norm(robot2obst)-rad)
				return (d_theta, d_theta, LA.norm(robot2obst)-rad)

			elif discriminant == 0:
				# y/x of unit vector is m
				self._log('d', 'R{rid}@tkref={tk:.4f}: $$$$$ Latest Z in border of a obstacle !!! $$$$$'.format(rid=self.eyed, tk=self._opttime[0]))

				theta1 = np.arctan(-b/2*a) # [-pi, pi)
				theta2 = theta1-np.pi # [-2pi, 0.0]

			else:
				theta1 = np.arctan((-b + np.sqrt(discriminant))/(2*a)) # [-pi/2, pi/2]
				theta2 = np.arctan((-b - np.sqrt(discriminant))/(2*a)) # [-pi/2, pi/2]


			ref_theta = np.arctan2(robot2obst[1], robot2obst[0])

			#print '_______ O [{}] ms ({}, {}), thetaref {}'.format(obstIdx, theta1[0,0], theta2[0,0], ref_theta[0,0])

			quad14_ref_theta = ref_theta if ref_theta <= np.pi/2 and ref_theta >= -np.pi/2 else UnicycleKineModel._signed_angle(ref_theta-np.pi)

			#quad14_goal_theta = goal_theta if goal_theta <= np.pi/2 and goal_theta >= -np.pi/2 else UnicycleKineModel._signed_angle(goal_theta-np.pi)

			#quad14_waypt_theta = waypt_theta if waypt_theta <= np.pi/2 and waypt_theta >= -np.pi/2 else UnicycleKineModel._signed_angle(waypt_theta-np.pi)

			if (quad14_ref_theta < theta1 and quad14_ref_theta < theta2) or (quad14_ref_theta > theta1 and quad14_ref_theta > theta2):
			 	self._log('d', 'obj centroid not in the middle')
			 	# if abs(quad14_ref_theta - theta1) > np.pi/2:
			 	#self._log('d', '{}'.format(quad14_ref_theta - theta1))
			 	#self._log('d', '{}'.format(quad14_ref_theta - theta2))
			 	if abs(quad14_ref_theta - theta1) > abs(quad14_ref_theta - theta2):
			 		theta1 = UnicycleKineModel._signed_angle(theta1 - np.pi)
			 	else:
			 		theta2 = UnicycleKineModel._signed_angle(theta2 - np.pi)

			print '_______ [{}] ms ({}, {}), thetaref {}'.format(obstIdx, theta1[0,0], theta2[0,0], ref_theta[0,0])

			d_theta1 = UnicycleKineModel._signed_angle(quad14_ref_theta - theta1)
			d_theta2 = UnicycleKineModel._signed_angle(quad14_ref_theta - theta2)

			d_theta1 = UnicycleKineModel._signed_angle(waypt_theta + d_theta1 - ref_theta)
			d_theta2 = UnicycleKineModel._signed_angle(waypt_theta + d_theta2 - ref_theta)

			#absd_theta1 = abs(d_theta1)
			#absd_theta2 = abs(d_theta2)
			#theta1 = UnicycleKineModel._signed_angle(goal_theta - d_theta1)
			#theta2 = UnicycleKineModel._signed_angle(goal_theta - d_theta2)

			return (d_theta1[0,0], d_theta2[0,0], LA.norm(robot2obst)-rad)

		# Find direction for init:
		# in: self._D, self._final_z, self._latest_z, otherRobots' pos and speed, obstacles' pos closer than self._D in the 120º (or other angle) cone in front of the robot
		# out:  direc
		def _find_direction():
			"""
			| :math:`\\Delta`   | incidence	   |
			|-------------------|-----------------|
			| :math:`\\Delta<0` | no intersection |
			| :math:`\\Delta=0` | tangent		 |
			| :math:`\\Delta>0` | secant		  |
			"""

			robot2goal = self._final_z - self._latest_z
			robot2goal_dist = LA.norm(robot2goal)
			goal_direc = robot2goal/robot2goal_dist
			goal_theta = np.arctan2(goal_direc[1], goal_direc[0])

			robot2waypt = self._waypoint - self._latest_z
			robot2waypt_dist = LA.norm(robot2waypt)
			waypt_direc = robot2waypt/robot2waypt_dist
			waypt_theta = np.arctan2(waypt_direc[1], waypt_direc[0])

			print 'known obstacles', self._known_obst_idxs

			print 'Z\n', self._latest_z

			if self._known_obst_idxs == [] and self._collision_robots_idx == []:
				return (goal_direc, self._final_z)

			bubbles = self._obst
			bubbles_idx = self._known_obst_idxs
			fake_obstacles_cntr = 0

			if self._collision_robots_idx != []:
			#if False:
				for oth_idx in self._collision_robots_idx:

					#if oth_idx > self.eyed:
					#	continue

					otherPose = np.matrix([[self._com_link.latest_q[oth_idx][0]], [self._com_link.latest_q[oth_idx][1]], [self._com_link.latest_q[oth_idx][2]]])

					#print 'otherPose', otherPose

					otherPos = otherPose[0:-1]

					otherDirec = np.vstack((np.cos(otherPose[-1]), np.sin(otherPose[-1])))

					otherLinVel = self._com_link.latest_u[oth_idx][0]

					myLinVel = self._latest_u[0,0]

					myDirec = np.vstack((np.cos(self._latest_q[-1]), np.sin(self._latest_q[-1])))

					myPos = self._latest_z

					deltaVX = myLinVel*myDirec[0] - otherLinVel*otherDirec[0]
					deltaVY = myLinVel*myDirec[1] - otherLinVel*otherDirec[1]

					#print 'myPos', myPos
					#print 'otherPos', otherPos

					deltaPX = myPos[0]-otherPos[0]
					deltaPY = myPos[1]-otherPos[1]

					a = (deltaVX)**2 + (deltaVY)**2

					b = 2*(deltaPX)*(deltaVX) + 2*(deltaPY)*(deltaVY)

					c = deltaPX**2 + deltaPY**2 - (2*self.rho + self._safe_epsilon) # TODO get other robot radius

					discriminant = b**2 - 4*a*c

#					if discriminant > 0 and random.choice([True, False]): # then add buble
					if discriminant > 0: # then add buble
#					if False:

						print '\t\t\t\t\t COLLISION'

						t1 = float((-b + np.sqrt(discriminant))/(2*a))
						t2 = float((-b - np.sqrt(discriminant))/(2*a))

						if t1 < 0.0 and t2 < 0.0:
							print '\t\t\t\t\t BUT IN THE PAST (t<0)'
							continue

						if t1 < t2 and t1 < 0.0:
							t1 = 0.0

						if t2 < t1 and t2 < 0.0:
							t2 = 0.0

						centroid = (myPos + myLinVel*myDirec*t1 + myPos + myLinVel*myDirec*t2)/2.0

						radius = LA.norm((myPos + myLinVel*myDirec*t1) - (myPos + myLinVel*myDirec*t2))/2.0 /4.0
						#/ random.choice(_frange(1.0, 2.0, .01)) # assuming that the other robot will "take care of half of the problem (update: not assuming that anymore since we're using robIDs for solving symmetry"

						bubbles.append(RoundObstacle([centroid[0,0], centroid[1,0]], radius))
						bubbles_idx.append(len(bubbles)-1)
						fake_obstacles_cntr += 1

			if bubbles == []:
				for _ in range(fake_obstacles_cntr):
					bubbles.pop()
					bubbles_idx.pop()
				return (goal_direc, self._final_z)

			else:

				obstInfo = dict()
				listOfGroups = []

				for i in bubbles_idx:

					if any([i in j for j in listOfGroups]):
						continue

					listOfGroups += [[i]]
					# obstInfo[i] = getThetas(i)

					tree = [i]
					cntr = 0
					while cntr < len(tree):
						root = tree[cntr]

						for j in bubbles_idx:
								if j != root and not any([j in k for k in listOfGroups]) and LA.norm(np.matrix(bubbles[root].centroid).T - np.matrix(bubbles[j].centroid).T) < bubbles[root].radius + bubbles[j].radius + 2*self.rho:

									listOfGroups[-1] += [j]
									# obstInfo[j] = getThetas(j, force=True)
									tree.append(j)
						cntr += 1

				# Create dictionary with each osbtacle avoidance info
				# if any of the obstacles in a group is tested positive then
				# 	compute thetas to all from 2nd degree equation (Getthetas)
				# 
				# else (if none are tested positive) then
				# 	put the following in the dictionary
				#   return (0.0, 0.0, robot2goal_dist)
				print 'listOfGroups', listOfGroups
				i = 0
				while i < len(listOfGroups):
				#for group in listOfGroups:
					group = listOfGroups[i]
					print 'group', group
					if any([isObstInRobotsWay(x) for x in group]):
						j = 0
						while j < len(group):
						#for x in group:
							obstXInfo = getThetas(group[j])
							#print 'obstXInfo', obstXInfo

							maxAngle = 2.8

							if abs(obstXInfo[0]) > maxAngle or abs(obstXInfo[1]) > maxAngle or obstXInfo[2] > robot2goal_dist:
								print 'take {} out'.format(x)
								listOfGroups[i].remove(group[j])
								print listOfGroups
							else:
								obstInfo[group[j]] = obstXInfo
								j += 1
						i += 1
					else:
						print '-------------------------------- group', group, 'can be ignored -----------------------'
						# d_theta = UnicycleKineModel._signed_angle(waypt_theta - goal_theta)
						# d_theta = 0.0
						#for x in group:
						#	obstInfo[x] = (d_theta[0,0], d_theta[0,0], robot2goal_dist)
						listOfGroups.remove(group)

				#print 'obstInfo:', obstInfo

				testList = [item for sublist in listOfGroups for item in sublist]
				#testList = [x ]
				#if listOfGroups == []:
				if testList == []:
					for _ in range(fake_obstacles_cntr):
						bubbles.pop()
						bubbles_idx.pop()
					return (goal_direc, self._final_z)


				minMaxList = []
				#print 'listOfGroups', listOfGroups
				for i in listOfGroups:
					if i == []:
						continue
					for j in i:
						if obstInfo[j][2] < 0.0:
							#theta = UnicycleKineModel._signed_angle(waypt_theta - d_theta)
							#theta = UnicycleKineModel._signed_angle(goal_theta - obstInfo[j][0])
							#direc = np.vstack((np.cos(waypt_theta), np.sin(waypt_theta)))

							direc = np.vstack((np.cos(self._latest_q[-1]), np.sin(self._latest_q[-1])))
							waypoint = direc*robot2waypt_dist + self._latest_z
							for _ in range(fake_obstacles_cntr):
								bubbles.pop()
								bubbles_idx.pop()
							return (direc, waypoint)
					#subdic = [obstInfo[j] for j in i]
					subdic = dict((k, obstInfo[k]) for k in i if k in obstInfo)
					#print 'subdic', subdic


					doubled_keys = [val for val in [x for x in subdic] for _ in (0, 1)]
					doubled_dists = [val[-1] for val in [subdic[x] for x in subdic] for _ in (0, 1)]
					all_d_thetas = [item for sublist in [subdic[x] for x in subdic] for item in sublist[0:-1]]

					#print 'doubled keys:', doubled_keys
					#print 'doubled dists:', doubled_dists
					#print 'all_d_thetas:', all_d_thetas

					X = [(x, y, z) for x, y, z in sorted(zip(all_d_thetas, doubled_keys, doubled_dists))]
					#print 'Ordered index', [i[1] for i in X]
					#print 'Ordered dthetas', [i[0] for i in X]
					#print K
					# x[0] min dtheta
					# x[-1] max dtheta
					# print 'X', X
					if (abs(X[0][0]) <= abs(X[-1][0])):
						absMinVal = X[0][0]
						absMinIdx = X[0][1]
						absMinDist = X[0][2]
					else:
						absMinVal = X[-1][0]
						absMinIdx = X[-1][1]
						absMinDist = X[-1][2]
					#print 'To add to minMaxList', (absMinVal, absMinDist, absMinIdx, X[0][0], X[-1][0], X[0][2], X[-1][2], X[0][1], X[-1][1])
					minMaxList += [ (absMinVal, absMinDist, absMinIdx, X[0][0], X[-1][0], X[0][2], X[-1][2], X[0][1], X[-1][1])]
					#add to list
				# for i in minlist:
					#min in dist

				#print 'minMaxList: ', minMaxList

				#random.shuffle(minMaxList)

				sortedValMinMaxList = [i for _, i in sorted(zip([x[0] for x in minMaxList], minMaxList))]

				print 'sortedValMinMaxList', sortedValMinMaxList
				# print 'i[0]', i[0]
				# sortedDistMinMaxList = [i for _, i in sorted(zip([x[2] for x in minMaxList], minMaxLi1st))]

				for k in sortedValMinMaxList:
					#subSortedValMinMaxList = [sortedValMinMaxList[gidx] for ginfo, gidx in zip(sortedValMinMaxList, range(len(sortedValMinMaxList))) if k[0] > ginfo[3] and k[0] > ginfo[4]]
					subSortedValMinMaxList = [ginfo for ginfo in sortedValMinMaxList if k[0] > ginfo[3] or k[0] < ginfo[4]]
					print 'subSortedValMinMaxList', subSortedValMinMaxList
					if subSortedValMinMaxList != []:
						proximityTests = [k[1] > j[1] for j in subSortedValMinMaxList]
						print 'proximityTests:', proximityTests
					 	if any(proximityTests):
					 		continue
					break


				# dists = [obstInfo[i][-1] for i in minlist]
				# closerAbsMinDirec = [y for x, y in sorted(zip(dists, minlist))][0]

				# obstInfo[closerAbsMinDirec]
				d_theta = k[0]
				#print 'd_theta', d_theta
				#theta = UnicycleKineModel._signed_angle(goal_theta - d_theta)
				eps = -0.03 if d_theta < 0.0 else +0.03
				#eps = 0.0
				theta = UnicycleKineModel._signed_angle(waypt_theta - d_theta + eps)

				direc = np.vstack((np.cos(theta), np.sin(theta)))
				waypoint = direc*robot2waypt_dist + self._latest_z
				for _ in range(fake_obstacles_cntr):
					bubbles.pop()
					bubbles_idx.pop()
				return (direc, waypoint)


		# Get the direction of the robot, the new waypoint and the direction to it
		init_direc = np.vstack((np.cos(self._latest_q[-1]), np.sin(self._latest_q[-1])))
		direc, self._waypoint = _find_direction()
		#direc, self._waypoint = _find_direction()
		self._log('d', 'R{rid}@tkref={tk:.4f}: found wayPoint:\n{wp}'.format(rid=self.eyed, tk=self._opttime[0], wp=self._waypoint))
		print '\n\t\t\t\twayPoint:{wp}\n'.format(wp=self._waypoint.T)
		self._log('d', 'R{rid}@tkref={tk:.4f}: found direction angle:\n{dir}'.format(rid=self.eyed, tk=self._opttime[0], dir=np.arctan2(direc[1], direc[0])*180.0/np.pi))
		self._log('d', 'direction:\n{dir}'.format(dir=direc))

		# Get rotation matrix for absolute to robot's frame of reference
		self._latest_rot2ref_mat = np.hstack((init_direc, np.multiply(np.flipud(init_direc), np.vstack((-1,1)))))
		self._latest_rot2rob_mat = np.hstack((np.multiply(init_direc, np.vstack((1,-1))), np.flipud(init_direc)))
		rotated_direc = self._latest_rot2rob_mat*direc
		self._log('d', 'Rotation2rob\n{}'.format(self._latest_rot2rob_mat))
		self._log('d', 'Rotation2ref\n{}'.format(self._latest_rot2ref_mat))
		# print 'Prod\n', self._latest_rot2rob_mat*self._latest_rot2ref_mat


		# Get planning horizon
		planHor = self._est_dtime if self._plan_state == 'ls' else self._Td

		# Get right acceleration
		accel = -1.*min(0.99*(self.k_mod.u_final[0,0]+self._latest_u[0,0])/(planHor), self.k_mod.acc_max[0,0]) if self._plan_state == 'ls' else self.k_mod.acc_max[0,0]
		
		# Get direction
		#rotated_direc = rotated_direc if self._plan_state != 'ls' else self._latest_rot2rob_mat*np.vstack((np.cos(self.k_mod.q_final[-1, 0]), np.sin(self.k_mod.q_final[-1, 0])))
			#+ np.vstack((1.0, 0.0)))/2.
		rotated_direc = rotated_direc if self._plan_state != 'ls' else np.vstack((1.0, 0.0))

		# Create a sampled trajectory for a "bounded uniformed accelerated motion" in x axis
		curve_init_direc = [[]]*self.k_mod.z_dim

		for i in range(self.k_mod.z_dim):
			curve_init_direc[i] = np.zeros(self._n_ctrlpts)

		max_displ_var = (planHor/(self._n_ctrlpts-1))*self.k_mod.u_max[0,0]
		prev_displ = 0.0

		for i in range(self._n_ctrlpts)[1:]:

			delta_t = i*(planHor/(self._n_ctrlpts-1))
			
			displ = self._latest_u[0,0]*delta_t + accel/2.*delta_t**2


			displ = max(displ, prev_displ) if displ-prev_displ < max_displ_var  else prev_displ + max_displ_var

			#displ = displ if displ-prev_displ > 0 else 0.02

			prev_displ = displ

			curve_init_direc[0][i] = displ
			#print i, self._n_ctrlpts

		# print 'Curve robot dir:\n', curve_init_direc

		# Create a sampled trajectory for a "bounded uniformed accelerated motion" in (direc-init_direc) direction in the xy plane
		curve_waypoint_direc = [[]]*self.k_mod.z_dim

		for i in range(self.k_mod.z_dim):
			curve_waypoint_direc[i] = np.zeros(self._n_ctrlpts)

		max_displ_var = (planHor/(self._n_ctrlpts-1))*self.k_mod.u_max[0,0]
		prev_displ = 0.0

		for i in range(self._n_ctrlpts)[1:]:

			delta_t = i*(planHor/(self._n_ctrlpts-1))
			
			displ = self._latest_u[0,0]*delta_t + accel/2.*delta_t**2

			displ = max(displ, prev_displ) if displ-prev_displ < max_displ_var  else prev_displ + max_displ_var

			#print 'displ:', displ

			#displ = displ if displ-prev_displ > 0 else 0.0

			prev_displ = displ

			for j in range(len(rotated_direc)):
				curve_waypoint_direc[j][i] = displ*rotated_direc[j]

		# print 'Curve way pt:\n', curve_waypoint_direc

		curve = [[]]*self.k_mod.z_dim
		for i in range(self.k_mod.z_dim):
			#print 'Curve', curve2[i]
			n = 1.5
			p = [(float(j)/(self._n_ctrlpts-1))**n for j in range(self._n_ctrlpts)]
			curve[i] = np.array([p[j] * curve_waypoint_direc[i][j] + (1-p[j]) * curve_init_direc[i][j] for j in range(self._n_ctrlpts)])

		#if self._plan_state == 'ls':
		if False:

			#accel = min((self.k_mod.u_final[0,0]+self._latest_u[0,0])/(planHor), self.k_mod.acc_max[0,0])
		# 	#alpha_accel = min((self.k_mod.u_final[1,0]+self._latest_u[1,0])/(planHor), self.k_mod.acc_max[0,0])
		# 	alpha_accel = self.k_mod.acc_max[0,0]


			final_direc = self._latest_rot2rob_mat*np.vstack((np.cos(self.k_mod.q_final[-1, 0]), np.sin(self.k_mod.q_final[-1, 0])))

			max_displ_var = (planHor/(self._n_ctrlpts-1))*self.k_mod.u_max[0,0]
		# 	max_angdispl_var = (planHor/(self._n_ctrlpts-1))*self.k_mod.u_max[1,0]

		# 	prev_displ = 0.0
		# 	prev_angdispl = 0.0

			delta_t = (planHor/(self._n_ctrlpts-1))
			
			displ = self.k_mod.u_final[0,0]*delta_t + self.k_mod.acc_max[0,0]/2.*delta_t**2
			#displ = max(displ, prev_displ) if displ-prev_displ < max_displ_var  else prev_displ + max_displ_var
			
			latest_direction = -1.*final_direc
		# 	print 'direc: ', latest_direction

			#curve[0][-2] = curve[0][-1] + displ*latest_direction[0]
			#curve[1][-2] = curve[1][-1] + displ*latest_direction[1]

			print curve
			curve[0] = np.insert(curve[0], 1, (curve[0][0]+curve[0][1])/2.)
			curve[1] = np.insert(curve[1], 1, (curve[1][0]+curve[1][1])/2.)
			curve[0] = curve[0][0:-1]
			curve[1] = curve[1][0:-1]
			print curve


		# 	print 'FINAL pt: ', curve[0][-1], ', ', curve[1][-1]
		# 	print 'pt: ', curve[0][-2], ', ', curve[1][-2]
		# 	init_pt_y = curve[1][-2]
		# 	pt_y = curve[1][-2]

		# 	prev_displ = displ

		# 	i = self._n_ctrlpts-2
			
		# 	while np.sign(init_pt_y) == np.sign(pt_y) and i > 0:

				
		# 		delta_t = (self._n_ctrlpts - i)*(planHor/(self._n_ctrlpts-1))

		# 		displ = self.k_mod.u_final[0,0]*delta_t + accel/2.*delta_t**2

		# 		angdispl = self.k_mod.u_final[1,0]*delta_t + alpha_accel/2.*delta_t**2
		# 		print 'angdispl: ', angdispl

		# 		displ = max(displ, prev_displ) if displ-prev_displ < max_displ_var  else prev_displ + max_displ_var

		# 		angdispl = max(angdispl, prev_angdispl) if angdispl-prev_angdispl < max_angdispl_var  else prev_angdispl + max_angdispl_var

		# 		dtheta = abs(angdispl-prev_angdispl)*-1. if pt_y < 0 else abs(angdispl-prev_angdispl)

		# 		print 'dtheta: ', dtheta
		# 		print 'maxa:', max_angdispl_var

		# 		rot_mat = np.matrix([[np.cos(dtheta), -1.*np.sin(dtheta)],[np.sin(dtheta), np.cos(dtheta)]])

		# 		latest_direction = rot_mat*latest_direction
		# 		print 'latest_direction: ', latest_direction


		# 		curve[0][i] = curve[0][i+1] + (displ-prev_displ)*latest_direction[0]
		# 		curve[1][i] = curve[1][i+1] + (displ-prev_displ)*latest_direction[1]
		# 		print 'pt: ', curve[0][i], ', ', curve[1][i]

		# 		pt_y = curve[1][i]

		# 		prev_displ = displ
		# 		prev_angdispl = angdispl

		# 		i -= 1


		self._gen_ctrlpts_from_curve(curve)

		self._log('i', 'R{rid}@tkref={tk:.4f}: Ctrlpts: \n{ctrl}'.format(rid=self.eyed, tk=self._opttime[0], ctrl=self._C))

		self._std_alone = True

		tic = time.time()
		#print 'Control pts a:\n', self._C
		self._solve_opt_pbl()
		#print 'Control pts d:\n', self._C
		toc = time.time()

		self._log('i', 'R{rid}@tkref={tk:.4f}: Ctrlpts: \n{ctrl}'.format(rid=self.eyed, tk=self._opttime[0], ctrl=self._C))

		# No need to sync process here, the intended path does impact the conflicts computation

		self._log('i', 'R{rid}@tkref={tk:.4f}: Time to solve stand alone optimization '
				'problem: {t}'.format(rid=self.eyed, t=toc-tic, tk=self._opttime[0]))
		self._log('i', 'R{rid}@tkref={tk:.4f}: N of unsatisfied eq: {ne}'\
				.format(rid=self.eyed, t=toc-tic, tk=self._opttime[0], ne=len(self._unsatisf_eq_values)))
		self._log('i', 'R{rid}@tkref={tk:.4f}: N of unsatisfied ieq: {ne}'\
				.format(rid=self.eyed, t=toc-tic, tk=self._opttime[0], ne=len(self._unsatisf_ieq_values)))
		self._log('i', 'R{rid}@tkref={tk:.4f}: Summary: {summ} after {it} it.'\
				.format(rid=self.eyed, t=toc-tic, tk=self._opttime[0], summ=self._exit_mode, it=self._n_it))

#		if self._final_step:
		if self._plan_state == 'ls':
			self._knots = self._gen_knots(self._opttime[0], self._t_final)
			self._opttime = np.linspace(self._opttime[0], self._t_final, self._N_s)
			self._soltime = np.linspace(self._soltime[0], self._t_final, self._N_ssol)

		#time_idx = None if self._plan_state == 'ls' else self._Tc_idx+1
		#time_idx_sol = None if self._plan_state == 'ls' else self._Tc_idx_sol+1

		dz = self._comb_bsp(self._opttime, self._C[0:self._n_ctrlpts,:], 0).T + np.asarray(self._latest_z)
		for dev in range(1, self.k_mod.l+1):
			dz = np.append(dz, self._comb_bsp(
					self._opttime, self._C[0:self._n_ctrlpts,:], dev).T, axis=0)

#		TODO verify process safety
		for i in range(dz.shape[1]):
			self._com_link.intended_path_z1[self.eyed][i] = dz[0, i]
			self._com_link.intended_path_z2[self.eyed][i] = dz[1, i]

		self._compute_conflicts()
		self._log('d', 'R{0}@tkref={1:.4f}: $$$$$$$$ CONFLICT LIST $$$$$$$$: {2}'
				.format(self.eyed, self._opttime[0], self._conflict_robots_idx))

		# Sync with every robot on the conflict list
		#  1. notify every robot waiting on this robot that it is ready for conflict solving
		with self._conflict_syncer_conds[self.eyed]:
			self._conflict_syncer[self.eyed].value = 1
			self._conflict_syncer_conds[self.eyed].notify_all()
		#  2. check if the robots on this robot conflict list are ready
		for i in self._conflict_robots_idx:
			with self._conflict_syncer_conds[i]:
				if self._conflict_syncer[i].value == 0:
					self._conflict_syncer_conds[i].wait()
		# Now is safe to read the all robots' in the conflict list intended paths (or are done planning)

		if self._conflict_robots_idx != []:
		#if False:
		#if self._conflict_robots_idx != [] and self._plan_state != 'ls':

			self._std_alone = False


#			self._conflict_dz = [self._read_com_link()]
#			self._read_com_link()

			
			self._C[0:self._n_ctrlpts,:] = (self._latest_rot2rob_mat*(self._C[0:self._n_ctrlpts,:].T)).T

			if self._plan_state == 'ls':
				self._est_dtime = self._dt_final
				self._knots = self._gen_knots(self._opttime[0], self._opttime[0]+self._est_dtime)
				self._opttime = np.linspace(self._opttime[0], self._opttime[0]+self._est_dtime, self._N_s)

			self._sa_dz = dz

			tic = time.time()
			self._solve_opt_pbl()
			toc = time.time()

			self._log('i', 'R{rid}@tkref={tk:.4f}: Time to solve optimization probl'
					'em: {t}'.format(rid=self.eyed, t=toc-tic, tk=self._opttime[0]))
			self._log('i', 'R{rid}@tkref={tk:.4f}: N of unsatisfied eq: {ne}'\
					.format(rid=self.eyed, t=toc-tic, tk=self._opttime[0], ne=len(self._unsatisf_eq_values)))
			self._log('i', 'R{rid}@tkref={tk:.4f}: N of unsatisfied ieq: {ne}'\
					.format(rid=self.eyed, t=toc-tic, tk=self._opttime[0], ne=len(self._unsatisf_ieq_values)))
			self._log('i', 'R{rid}@tkref={tk:.4f}: Summary: {summ} after {it} it.'\
					.format(rid=self.eyed, t=toc-tic, tk=self._opttime[0], summ=self._exit_mode, it=self._n_it))

#			if self._final_step:
			if self._plan_state == 'ls':
				self._knots = self._gen_knots(self._opttime[0], self._t_final)
				self._opttime = np.linspace(self._opttime[0], self._t_final, self._N_s)
				self._soltime = np.linspace(self._soltime[0], self._t_final, self._N_ssol)

			#time_idx = None if self._plan_state == 'ls' else self._Tc_idx+1
			#time_idx_sol = None if self._plan_state == 'ls' else self._Tc_idx_sol+1

			#dz = self._comb_bsp(self._opttime[0:time_idx], self._C[0:self._n_ctrlpts,:], 0).T + self._latest_z
			#for dev in range(1, self.k_mod.l+1):
			#	dz = np.append(dz, self._comb_bsp(
			#			self._opttime[0:time_idx], self._C[0:self._n_ctrlpts,:], dev).T, axis=0)

		#self._log('d', 'R{rid}@tkref{tk:.4f}: dz_sol:\n{path}'.format(rid=self.eyed, tk=self._opttime[0], path=dz_sol[0:2,:]))

		time_idx_sol = None if self._plan_state == 'ls' else self._Tc_idx_sol+1
		dz_sol = self._comb_bsp(self._soltime[0:time_idx_sol], self._C[0:self._n_ctrlpts,:], 0).T + self._latest_z
		for dev in range(1, self.k_mod.l+2):
			dz_sol = np.append(dz_sol, self._comb_bsp(
					self._soltime[0:time_idx_sol], self._C[0:self._n_ctrlpts,:], dev).T, axis=0)
		
		dztTp = [dzt.reshape(self.k_mod.l+2, self.k_mod.u_dim).T for dzt in dz_sol.T]
		
		ut = map(self.k_mod.phi_2, dztTp)

		eps = np.finfo(float).eps
		h = 1e-12
		dt = 2*h

		dz = self._comb_bsp([x-h for x in self._soltime[0:time_idx_sol]], self._C[0:self._n_ctrlpts,:], 0).T + np.asarray(self._latest_z)
		for dev in range(1, self.k_mod.l+1):
			dz = np.append(dz, self._comb_bsp([x-h for x in self._soltime[0:time_idx_sol]], self._C[0:self._n_ctrlpts,:], dev).T, axis=0)

		dztTp = [dzt.reshape(self.k_mod.l+1, self.k_mod.u_dim).T for dzt in dz.T]

		# get a list over time of command values u(t)
		uta = map(self.k_mod.phi_2, dztTp)

		dz = self._comb_bsp([x+h for x in self._soltime[0:time_idx_sol]], self._C[0:self._n_ctrlpts,:], 0).T + np.asarray(self._latest_z)
		for dev in range(1, self.k_mod.l+1):
			dz = np.append(dz, self._comb_bsp([x+h for x in self._soltime[0:time_idx_sol]], self._C[0:self._n_ctrlpts,:], dev).T, axis=0)

		dztTp = [dzt.reshape(self.k_mod.l+1, self.k_mod.u_dim).T for dzt in dz.T]

		# get a list over time of command values u(t)
		utp = map(self.k_mod.phi_2, dztTp)

		acc = [(utp[0] - ut[0])/dt/2] + [(utpi - utai)/dt for utpi, utai in zip(utp[1:-1], uta[1:-1])] + [(ut[-1] - uta[-1])/dt/2]
		

		# Storing
		if self._plan_state == 'ls':
			self._all_C.append(
					[self._dt_final]+
					list(self._C[0:self._n_ctrlpts,:].reshape(self._n_ctrlpts*self.k_mod.u_dim)))
		else:
			#print list(self._C[0:self._n_ctrlpts,:].reshape(self._n_ctrlpts*self.k_mod.u_dim))
			self._all_C.append(list(self._C[0:self._n_ctrlpts,:].reshape(self._n_ctrlpts*self.k_mod.u_dim)))

		#self._log('d', 'R{rid}@tkref={tk:.4f}: dz')

		if self._plan_state == 'fs':
			self._all_dz.append(dz_sol[:, 0:time_idx_sol])
			self._accel += acc[0:time_idx_sol]
#			self._all_dz.append(dz[:, 0:time_idx])
			self._all_times.extend(self._soltime[0:time_idx_sol])
		else:
			self._accel += acc[1:time_idx_sol]
			self._all_dz.append(dz_sol[:, 1:time_idx_sol])
			self._all_times.extend(self._soltime[1:time_idx_sol])
		# TODO rejected path

		# Updating
		latest_dz = np.asarray(self._comb_bsp([self._opttime[0]+self._Tc], self._C[0:self._n_ctrlpts,:], 0).T + self._latest_z)
		for dev in range(1, self.k_mod.l+1):
				latest_dz = np.append(latest_dz, self._comb_bsp(
						[self._opttime[0]+self._Tc], self._C[0:self._n_ctrlpts,:], dev).T, axis=0)
		latest_z = latest_dz[0:self.k_mod.u_dim, 0].reshape(
				self.k_mod.u_dim, 1)

		latest_q = self.k_mod.phi_1(latest_dz[:, 0].reshape(self.k_mod.l+1, self.k_mod.u_dim).T)
		latest_u = self.k_mod.phi_2(latest_dz[:, 0].reshape(self.k_mod.l+1, self.k_mod.u_dim).T)

		# Sync robots here so no robot computing conflict get the wrong latest_z of some robot
		with self._tc_syncer_cond:
			self._tc_syncer.value += 1
			if self._tc_syncer.value != self._n_robots:  # if not all robots are read
				self._log('d', 'R{}: I\'m going to sleep!'.format(self.eyed))
				self._tc_syncer_cond.wait()
			else:								# otherwise wake up everybody
				self._tc_syncer_cond.notify_all()
			self._tc_syncer.value -= 1			# decrement syncer (idem)
			for i in range(self.k_mod.u_dim):
				self._com_link.latest_z[self.eyed][i] = latest_z[i, 0]
				self._com_link.latest_u[self.eyed][i] = latest_u[i, 0]
			for i in range(self.k_mod.q_dim):
				self._com_link.latest_q[self.eyed][i] = latest_q[i, 0]

		with self._conflict_syncer_conds[self.eyed]:
			self._conflict_syncer[self.eyed].value = 0


#		print 'mtime[0] antes', self._opttime[0]
		if self._plan_state != 'ls':
			self._knots = self._knots + self._Tc
			self._opttime = [tk+self._Tc for tk in self._opttime]
			self._soltime = [tk+self._Tc for tk in self._soltime]
			self._latest_z = latest_z
			self._latest_q = latest_q
			self._latest_u = latest_u
			#print 'DZ:\n', dz[:, 0].reshape(
			#		self.k_mod.l+1, self.k_mod.u_dim).T, '\nSize:\n', dz[:, 0].reshape(
			#		self.k_mod.l+1, self.k_mod.u_dim).T.shape, '\nType:\n',type( dz[:, 0].reshape(
			#		self.k_mod.l+1, self.k_mod.u_dim).T)
			#print 'latestDZ:\n', latest_dz[:, 0].reshape(
			#		self.k_mod.l+1, self.k_mod.u_dim).T, '\nSize:\n', latest_dz[:, 0].reshape(
			#		self.k_mod.l+1, self.k_mod.u_dim).T.shape, '\nType:\n', type(latest_dz[:, 0].reshape(
			#		self.k_mod.l+1, self.k_mod.u_dim).T)
			if self._plan_state == 'fs':
				self._plan_state = 'ms'
		else:
			latest_dz = np.asarray(self._comb_bsp([self._t_final], self._C[0:self._n_ctrlpts,:], 0).T + self._latest_z)
			for dev in range(1, self.k_mod.l+1):
					latest_dz = np.append(latest_dz, self._comb_bsp(
							[self._t_final], self._C[0:self._n_ctrlpts,:], dev).T, axis=0)
			latest_z = latest_dz[0:self.k_mod.u_dim, 0].reshape(
					self.k_mod.u_dim, 1)
			self._latest_z = latest_z
		btoc = time.time()
		self._all_comp_times.append(btoc-btic)

#		print 'mtime[0] depois', self._opttime[0]
#		print 'Solved C :\n', self._C

	def _plan(self):
		""" Motion/path planner method. At the end of its execution :attr:`rtime`, :attr:`ctime`
		and :attr:`sol` attributes will be updated with the plan for completing the mission.
		"""

		self._log('i', 'R{rid}: Init motion planning'.format(rid=self.eyed))

#		self._final_step = False
		self._plan_state = 'fs'

		self._knots = self._gen_knots(self._t_init, self._Td)
		#self._knots = self._gen_knots(0.0, self._Td)
		#self._opttime = np.linspace(self._t_init, self._t_init+self._Td, self._N_s)
		#self._opttime = np.linspace(0.0, self._Td, self._N_s)
		#self._soltime = np.linspace(self._t_init, self._t_init + self._Td, self._N_ssol)

		# while the remaining dist is greater than the max dist during Tp
#		while LA.norm(self._latest_z - self._final_z) > self._D:

		remaining_dist = -1.0

		while True:
			remaining_dist = LA.norm(self._latest_z - self._final_z)
#			if remaining_dist < self._D:
#				break
#			elif remaining_dist < self._ls_min_dist + self._Tc*self.k_mod.u_max[0,0] and False:
			# if remaining_dist < self._ls_min_dist + self._Tc*self.k_mod.u_max[0,0]:
			#if remaining_dist < self._ls_min_dist + (self.k_mod.u_final[0,0]**2 - self._latest_u[0,0]**2)/(2.*self.k_mod.acc_max[0,0]):
			if remaining_dist < self._ls_min_dist + max(abs((self.k_mod.u_final[0,0]**2 - self._latest_u[0,0]**2)/(2.*self.k_mod.acc_max[0,0])), self._Tc*self.k_mod.u_max[0,0]):
			# if remaining_dist < max(abs((self.k_mod.u_final[0,0]**2 - self._latest_u[0,0]**2)/(2.*self.k_mod.acc_max[0,0])), self._Tc*self.k_mod.u_max[0,0]):
			#if remaining_dist < 14.0:
				print (self.k_mod.u_final[0,0]**2 - self._latest_u[0,0]**2)/(2.*self.k_mod.acc_max[0,0])
				print self._Tc*self.k_mod.u_max[0,0]

				self._log('d', 'R{0}: LAST STEP'.format(self.eyed))
				self._log('d', 'R{0}: Approx remaining dist: {1}'.format(self.eyed, remaining_dist))
				self._log('d', 'R{0}: Usual approx plan dist: {1}'.format(self.eyed, self._D))
				self._log('d', 'R{0}: Approx gain in dist: {1}'.format(self.eyed, self._D-remaining_dist))
				self._log('d', 'R{0}: Last step min dist: {1}'.format(self.eyed, self._ls_min_dist))
				# scale_factor = min(1., remaining_dist/self.k_mod.u_max[0,0]/self._Td)
				# self._n_knots = max(int(round(self._n_knots*scale_factor)), self._d-1)
				# self._n_ctrlpts = self._n_knots + self._d - 1 # nb of ctrl points
				# self._N_s = max(int(round(self._N_s*scale_factor)), self._n_ctrlpts+1)
				self._log('d', 'R{0}: Last step nk: {1}'.format(self.eyed, self._n_knots))
				self._log('d', 'R{0}: Last step nctrl: {1}'.format(self.eyed, self._n_ctrlpts))
				# self._N_ssol = max(int(round(self._N_ssol*scale_factor)), self._n_ctrlpts+1)
				# self._log('i', 'R{0}: scale {1} Ns {2:d} Nk {3:d}'.format(self.eyed, scale_factor, self._N_s, self._n_knots))
				break

			self._plan_section()
			self._log('i', 'R{}: --------------------------'.format(self.eyed))
			self._log('i', 'R{}: Latest Q: {}'.format(self.eyed, self._latest_q))
			self._log('i', 'R{}: --------------------------'.format(self.eyed))

#		self._final_step = True
		self._plan_state = 'ls'
		self._est_dtime = remaining_dist/(self._latest_u[0,0]+self.k_mod.u_final[0,0])*2.
		#print self._est_dtime

		self._knots = self._gen_knots(self._opttime[0], self._opttime[0]+self._est_dtime)
#		print 'LAST mtime[0]', self._opttime[0]
#		print 'LAST mtime[0]+self._Tc', self._opttime[0]+self._Tc
		self._opttime = np.linspace(self._opttime[0], self._opttime[0]+self._est_dtime, self._N_s)

		self._plan_section()
		self._log('i', 'R{}: Finished motion planning'.format(self.eyed))
		self._log('i', 'R{}: --------------------------'.format(self.eyed))
#		print 'LAST mtime[-1]', self._opttime[-1]

		self.sol[self.eyed] = self._all_dz
		self.accel[self.eyed] = self._accel
		self.p_sol[self.eyed] = self._all_C
		self.rtime[self.eyed] = self._all_times
		self.ctime[self.eyed] = self._all_comp_times
		self._com_link.done_planning[self.eyed] = 1

		#  Notify every robot waiting on this robot that it is ready for the conflict solving
		with self._conflict_syncer_conds[self.eyed]:
			self._conflict_syncer[self.eyed].value = 1
			self._conflict_syncer_conds[self.eyed].notify_all()

		# Make sure any robot waiting on this robot awake before returning
		with self._tc_syncer_cond:
			self._tc_syncer.value += 1			   # increment synker
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
		latest_q = [mpc.Array('d', [r.k_mod.q_init[0, 0], \
				r.k_mod.q_init[1, 0], r.k_mod.q_init[2, 0]]) for r in self._robs]
		latest_z = [mpc.Array('d', [r.k_mod.q_init[0, 0], \
				r.k_mod.q_init[1, 0]]) for r in self._robs]
		latest_u = [mpc.Array('d', [r.k_mod.u_init[0, 0], \
				r.k_mod.u_init[1, 0]]) for r in self._robs]

		# Packing shared memory into a RobotMsg object
		com_link = RobotMsg(done_planning, intended_path_z1, intended_path_z2, latest_z, latest_q, latest_u)

		# More complex and expense shared memory thru a server process manager ...
		# ... (because they can support arbitrary object types)
		manager = mpc.Manager()
		solutions = manager.list(range(n_robots))
		acceleration_solution = manager.list([[]]*n_robots)
		race_time = manager.list(range(n_robots))
		comp_time = manager.list(range(n_robots))
		paramzed_solution = manager.list([[]]*n_robots)

		# Setting multiprocessing stuff for every robot
		[r.set_option('log_lock', log_lock) for r in self._robs]
		[r.set_option('tc_syncer', tc_syncer) for r in self._robs]
		[r.set_option('tc_syncer_cond', tc_syncer_cond) for r in self._robs]
		[r.set_option('conflict_syncer', conflict_syncer) for r in self._robs]
		[r.set_option('conflict_syncer_conds', conflict_syncer_conds) for r in self._robs]
		[r.set_option('com_link', com_link) for r in self._robs]
		[r.set_option('sol', solutions) for r in self._robs]
		[r.set_option('accel', acceleration_solution) for r in self._robs]
		[r.set_option('p_sol', paramzed_solution) for r in self._robs]
		[r.set_option('rtime', race_time) for r in self._robs]
		[r.set_option('ctime', comp_time) for r in self._robs]
		####################################################################
		####################################################################
		####################################################################

		# Make all robots plan their trajectories
		[r.planning_process.start() for r in self._robs]
		[r.planning_process.join() for r in self._robs]

		 # Saving parameterized solution to xml file
		#<?xml version="1.0" encoding="utf-8"?>
		xml_root = ET.Element('root')

		for i in range(len(self._robs)):
			#ET.SubElement(doc, "field2", name="asdfasd").text = "some vlaue2"
			agv = ET.SubElement(xml_root, 'AdeptLynx'+str(self._robs[i].eyed))
			for p, ct in zip(paramzed_solution[i], range(0, len(paramzed_solution[i]))):
				ET.SubElement(agv, 'plan'+str(ct)).text = str(p)[1:-1]
		tree = ET.ElementTree(xml_root)
		# tree.write('output.xml')
		f = open('../../xde/xde/xde/xde/src/aiv/output.xml', 'w')
		f.write(minidom.parseString(ET.tostring(xml_root, encoding='utf-8')).toprettyxml(indent="\t"))

		# Reshaping the solution
		path = range(len(self._robs))
		seg_pts_idx = [[] for _ in range(len(self._robs))]
		for i in range(len(self._robs)):
			path[i] = solutions[i][0]
			seg_pts_idx[i] += [0]
			for p in solutions[i][1:]:
				c = path[i].shape[1]
				seg_pts_idx[i] += [c]
				path[i] = np.append(path[i], p, axis=1)

		# From [z dz ddz](t) get q(t) and u(t)
		zdzddz = [[]]*n_robots
		for i in range(n_robots):
			zdzddz[i] = [np.asarray(z.reshape(self._robs[i].k_mod.l+2, self._robs[i].k_mod.u_dim).T) for z in path[i].T]

		# get a list over time of command values u(t)
		ut = range(len(self._robs))
		for i in range(len(self._robs)):
			ut[i] = map(self._robs[i].k_mod.phi_2, zdzddz[i])
			# Fixing division by near-zero value when calculating angspeed for plot
			ut[i][0][1,0] = self._robs[i].k_mod.u_init[1,0]
			ut[i][-1][1,0] = self._robs[i].k_mod.u_final[1,0]

		# get a list over time of accelerations
		at = range(len(self._robs))
		for i in range(len(self._robs)):
			at[i] = map(self._robs[i].k_mod.phi_3, zdzddz[i])
			# Fixing division by near-zero
			# at[i][0][1,0] = self._robs[i].k_mod.u_init[1,0]
			# at[i][-1][1,0] = self._robs[i].k_mod.u_final[1,0]

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

#		# Calculate the obstacles-robots dists (only works while we consider static obstcacles)
#		or_dist = []
#		no_obst = []
#		penetr = []
#		cpenetr = []
#		penetr_v = []
##		aux = np.empty((len(self._robs),len(self._obsts)))
#		for i in range(len(self._robs)):
#			or_dist += [[]]
#			no_obst += [[]]
#			no_obst[i] = len(seg_pts_idx[i])*[0]
#			penetr_v += [[]]
#			penetr += [0]
#			cpenetr += [0]
#			for j, obst in zip(range(len(self._obsts)), self._obsts):
#				or_dist[i] += [[]]
#				cno_obst = 0
#				for k, q, u in zip(range(len(qt[i])), qt[i], ut[i]):
#					v = obst.pt_2_obst(np.squeeze(np.asarray(q[0:2, 0].T)), self._robs[i].rho)
#					#print v
#					or_dist[i][j] += [v]
#					if k in seg_pts_idx[i]:
#						d = obst.detected_dist(np.squeeze(np.asarray(q[0:2, 0].T)))
#						if d < self._robs[i]._d_rho:
#							no_obst[i][cno_obst] += 1
#						cno_obst += 1
#					if v < 0.0:
#
#						penetr[i] += (-1.0*obst.radius*v - v**2/2)/(obst.radius-v)*u[0,0]
#						cpenetr[i] += 1
#						penetr_v[i] += [(-1.0*obst.radius*v - v**2/2)/(obst.radius-v)*u[0,0]]
#		print no_obst[0]
#		print cno_obst

		rr_dist = []
		k = 0
		robs = []
		for i in range(len(self._robs)):
			for j in range(i+1, len(self._robs)):
				rr_dist += [[]]
				robs += [(i,j)]
				for k1, q1, k2, q2 in zip(range(len(qt[i])), qt[i], range(len(qt[j])), qt[j]):
					dist = LA.norm(q1[0:-1]-q2[0:-1]) - 2*self._robs[0].rho
					rr_dist[k] += [dist]
				k += 1

		# Logging simulation summary
		for i in range(len(self._robs)):
			ctime_len = len(ctime[i])
			if ctime_len > 1:
				g_max_idx = np.argmax(ctime[i][1:]) + 1
			else:
				g_max_idx = 0
#			if cpenetr[i] > 0:
#				logging.info('R{rid}: PEN: {d}'.format(rid=i, d=penetr[i]*self._robs[i]._Tp/self._robs[i]._N_ssol))
#			else:
#				logging.info('R{rid}: PEN: {d}'.format(rid=i, d=0.0))
#			logging.info('R{rid}: MOB: {d}'.format(rid=i, d=max(no_obst[i])))
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

#		while True:
#			try:
#				x = str(raw_input("Want to see the result plotted? [y/n]: "))
#				if x != 'n' and x != 'N':
#					if x != 'y' and x != 'Y':
#						print("I'll take that as an \'n\'.")
#						x = 'n'
#					else:
#						x = 'y'
#				else:
#					x = 'n'
#				break
#			except ValueError:
#				print("Oops! That was no valid characther. Try again...")
#			except KeyboardInterrupt:
#				print("\nGood bye!")
#				return
#
#		if x == 'n':
#			return

		# Interactive plot
		#if self._plot:
			#plt.ion()

#		fig_obst_dist = plt.figure()
#		ax_obst_dist= fig_obst_dist.gca()
#		for i in range(len(self._robs)):
#			for j in range(len(self._obsts)):
#				ax_obst_dist.plot(or_dist[i][j])

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

		# PLOT INTERROBOT DIST
		fig_rr_d = plt.figure()
		ax_rr_d = fig_rr_d.gca()
		rrcolors = [[i, 1.0-i, ((2+int(i*10))%10)/10.0] for i in np.linspace(0.0, 1.0, len(rr_dist))]
		for rrd, k in zip(rr_dist, range(len(rr_dist))):
			plttime = rtime[robs[k][0]] if len(rtime[robs[k][0]]) < len(rtime[robs[k][1]]) else rtime[robs[k][1]]
#			print len(rtime[robs[k][0]]), len(rtime[robs[k][1]]), len(plttime), len(rrd)
			ax_rr_d.plot(plttime, rrd, color=rrcolors[k], label = r'$d(R_{0},R_{1})-\rho_{0}-\rho_{1}$'.format(robs[k][0], robs[k][1]))
		ax_rr_d.grid()
		ax_rr_d.set_xlabel('time (s)')
		ax_rr_d.set_ylabel('Inter-robot distance (m)')
		ax_rr_d.set_title('Inter-robot distances throughout simulation')
		#ax_rr_d.set_xlim([0,min([x[-1] for x in rtime])])
		handles, labels = ax_rr_d.get_legend_handles_labels()
		ax_rr_d.legend(handles, labels, loc=1, ncol=4, prop={'size':9})

		fig_rr_d.set_size_inches(1.0*18.5/2.54,1.0*6.5/2.54)
		fig_rr_d.savefig(self._direc+'/images/'+self._sn+'/multirobot-interr.png',\
				bbox_inches='tight', dpi=300)
		fig_rr_d.savefig(self._direc+'/images/'+self._sn+'/multirobot-interr.pdf',\
				bbox_inches='tight', dpi=300)


		fig_s, axarray = plt.subplots(2)
		#fig_s = plt.figure()
		#ax_lin_vel = fig_s.gca()
		#axarray[0] = ax_lin_vel
		axarray[0].set_ylabel(r'$v\,(m/s)$')
		axarray[0].set_title('Linear and angular speeds')
		axarray[1].set_xlabel('time(s)')
		#axarray[0].set_xlabel('time(s)')
		axarray[1].set_ylabel(r'$\omega\,(rad/s)$')
		#axarray[1].set_title('Angular speed')

		fig_a, aaxarray = plt.subplots(2)
		#fig_s = plt.figure()
		#ax_lin_vel = fig_s.gca()
		#axarray[0] = ax_lin_vel
		aaxarray[0].set_ylabel(r'$a\,(m/s^2)$')
		aaxarray[0].set_title('Linear and angular accelerations')
		aaxarray[1].set_xlabel('time(s)')
		#aaxarray[0].set_xlabel('time(s)')
		aaxarray[1].set_ylabel(r'$\alpha\,(rad/s^2)$')
		# aaxarray[1].set_title('Angular speed')

		fig = plt.figure()
		ax = fig.gca()
		ax.set_xlabel(r'$x\,(m)$')
		ax.set_ylabel(r'$y\,(m)$')
		ax.set_title('Generated trajectory')
		ax.axis('equal')

		aux = np.linspace(0.0, 1.0, 1e2)
		colors = [[i, 1.0-i, ((2+int(i*10))%10)/10.0] for i in np.linspace(0.0, 1.0, len(self._robs))]

		while True:
			# Creating obstacles in the plot
			[obst.plot(fig, offset=self._robs[0].rho) for obst in self._obsts]

			plt_paths = range(len(self._robs))
			plt_seg_pts = range(len(self._robs))
			plt_robots_c = range(len(self._robs))
			plt_robots_det = range(len(self._robs))
			plt_robots_t = range(len(self._robs))
			for i in range(len(self._robs)):
				plt_paths[i], = ax.plot(path[i][0, 0], path[i][1, 0], color=colors[i], label=r'$R_{}$'.format(i), linestyle='-')
				plt_seg_pts[i], = ax.plot(path[i][0, seg_pts_idx[i][0]], \
						path[i][1, seg_pts_idx[i][0]], color=colors[i], ls='None', marker='o', markersize=5)
				plt_robots_c[i] = plt.Circle(
						(path[i][0, 0], path[i][1, 0]), # position
						self._robs[i].rho, # radius
						color='m',
						ls = 'solid',
						fill=False)
				#plt_robots_det[i] = plt.Circle( (path[i][0, 0], path[i][1, 0]), self._robs[i]._d_rho, color=colors[i], alpha=0.5, ls = 'solid', fill=False)
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
			#[ax.add_artist(r) for r in plt_robots_det]
			[ax.add_artist(r) for r in plt_robots_t]
#			ax.add_artist(plt.Circle((3.2, 1.8), .5, color = 'r', fill=True, alpha=0.5))
#			ax.add_artist(plt.Circle((6.5, 3.1), .5, color = 'r', fill=True, alpha=0.5))
#			ax.text(2.95, 2.08, 'collision', fontsize=9)
#			ax.text(6.25, 2.82, 'collision', fontsize=9)
			
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
						#plt_robots_det[i].center = path[i][0, ctr], path[i][1, ctr]
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
#				time.sleep(0.01)
				if self._plot:
					ax.relim()
					ax.autoscale_view(True, True, True)
					fig.canvas.draw()

					handles, labels = ax.get_legend_handles_labels()
					ax.legend(handles, labels, loc=1, ncol=3)

					fig.savefig(self._direc+'/images/'+self._sn+'/multirobot-path-'+str(ctr)+'.png', bbox_inches='tight', dpi=300)
					#fig.savefig(self._direc+'/images/'+self._sn+'/multirobot-path-'+str(ctr)+'.pdf', bbox_inches='tight', dpi=300)
			#end

			ax.relim()
			ax.autoscale_view(True, True, True)
			
			fig.canvas.draw()

			handles, labels = ax.get_legend_handles_labels()
			ax.legend(handles, labels, loc=2, ncol=3)

			fig.savefig(self._direc+'/images/'+self._sn+'/multirobot-path.png', bbox_inches='tight', dpi=300)
			fig.savefig(self._direc+'/images/'+self._sn+'/multirobot-path.pdf', bbox_inches='tight', dpi=300)
			
			auxcolors = ['b', 'y']*10
			for i in range(len(self._robs)):
				linspeed = [x[0, 0] for x in ut[i]]
				angspeed = [x[1, 0] for x in ut[i]]
				linacc = [x[0, 0] for x in acceleration_solution[i]]
				angacc = [x[1, 0] for x in acceleration_solution[i]]
				linaccal = [x[0, 0] for x in at[i]]
				angaccal = [x[1, 0] for x in at[i]]
				# print 'LEN: ', len(linacc), len(rtime[i])
				axarray[0].plot(rtime[i], linspeed, color=colors[i], label = r'$R_{}$'.format(i), linestyle='-')
				axarray[1].plot(rtime[i], angspeed, color=colors[i], label = r'$R_{}$'.format(i), linestyle='-')
				# axarray[1].plot(rtime[i], angspeed, color=colors[i])
				# aaxarray[0].plot(rtime[i], linacc, color=auxcolors[i], label = r'$R_{} a$'.format(i))
				# aaxarray[1].plot(rtime[i], angacc, color=auxcolors[i], label = r'$R_{} a$'.format(i))
				aaxarray[0].plot(rtime[i], linaccal, color=colors[i], label = r'$R_{}$'.format(i), linestyle='-')
				# aaxarray[0].plot(rtime[i], linaccal, color=colors[i])
				aaxarray[1].plot(rtime[i], angaccal, color=colors[i], label = r'$R_{}$'.format(i), linestyle='-')
			axarray[0].grid()
			axarray[1].grid()
			aaxarray[0].grid()
			aaxarray[1].grid()
			axarray[0].set_ylim([0.0, 1.1*self._robs[0].k_mod.u_max[0, 0]])
			#axarray[1].set_ylim([-1.1*self._robs[0].k_mod.u_max[1, 0], 1.1*self._robs[0].k_mod.u_max[1, 0]])
			# aaxarray[0].set_ylim([-1.1*self._robs[0].k_mod.a_max[0, 0]], 1.1*self._robs[0].k_mod.a_max[0, 0]])
			aaxarray[1].set_ylim([-15.0, 15.0])
			fig_s.canvas.draw()
			fig_a.canvas.draw()

			handles1, labels1 = axarray[0].get_legend_handles_labels()
			axarray[0].legend(handles1, labels1, ncol=3, loc=3)
			handles2, labels2 = axarray[1].get_legend_handles_labels()
			# axarray[1].legend(handles2, labels2, ncol=3, loc=3)
			handles1, labels1 = aaxarray[0].get_legend_handles_labels()
			# aaxarray[0].legend(handles1, labels1, ncol=3, loc=3)
			handles2, labels2 = aaxarray[1].get_legend_handles_labels()
			aaxarray[1].legend(handles2, labels2, ncol=3, loc=3)

			fig_s.set_size_inches(1.0*18.5/2.54,1.0*13/2.54)
			fig_a.set_size_inches(1.0*18.5/2.54,1.0*13/2.54)
			fig_s.savefig(self._direc+'/images/'+self._sn+'/multirobot-vw.png',bbox_inches='tight', dpi=300)
			fig_s.savefig(self._direc+'/images/'+self._sn+'/multirobot-vw.pdf',bbox_inches='tight', dpi=300)
			fig_a.savefig(self._direc+'/images/'+self._sn+'/multirobot-aalpha.png',bbox_inches='tight', dpi=300)
			fig_a.savefig(self._direc+'/images/'+self._sn+'/multirobot-aalpha.pdf',bbox_inches='tight', dpi=300)

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
			ax.cla()
			ax.set_xlabel(r'$x(m)$')
			ax.set_ylabel(r'$y(m)$')
			ax.set_title('Generated trajectory')
			ax.axis('equal')
			axarray[0].set_ylabel(r'$v(m/s)$')
			axarray[0].set_title('Linear speed')
			axarray[1].set_xlabel('time(s)')
			axarray[1].set_ylabel(r'$\omega(rad/s)$')
			axarray[1].set_title('Angular speed')
		#end
	#end
#end
###############################################################################
# Script
###############################################################################

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
	print boundary.y_min
	print boundary.y_max
	print boundary.x_min
	print boundary.x_max

	arb_max_no = int(round((boundary.x_max-boundary.x_min)*\
			(boundary.y_max-boundary.y_min)/((min_radius*2)**2)/10.))
	if no > arb_max_no:
		logging.info('Too many obstacles for the given boundary.')
		logging.info('Using {:.0f} obstacles instead of {:.0f}.'.format(arb_max_no, no))
		no = arb_max_no

	def _isobstok(obsts, c, r):
		""" Vefify if random generated obstacle is ok (not touching another obstacle)
		"""
		if len(obsts) > 0:
			for obst in obsts:
				if (c[0]-obst[0][0])**2 + (c[1]-obst[0][1])**2 < (r+obst[1])**2:
					return False
		return True

	resol = 0.0001 # meters

	radius_range = _frange(min_radius, max_radius, resol)
	x_range = _frange(boundary.x_min+max_radius, boundary.x_max-max_radius, resol)
	y_range = _frange(boundary.y_min+max_radius, boundary.y_max-max_radius, resol)
	print 'y_range', y_range[0], y_range[-1]

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
	mpc.freeze_support() # windows freeze bug fix TODO verify if it works

	#################################
	# python planning_sim.py --help #
	#################################

	# XML parsing #############################################################
	root = ET.parse('../../xde/xde/xde/xde/src/aiv/config.xml').getroot()

	mpmethod = root.find('mpmethod')

	time_c = float(mpmethod.find('comphorizon').text)
	time_p = float(mpmethod.find('planninghorizon').text)
	no_sopt = int(mpmethod.find('sampling').text)
	no_knots = int(mpmethod.find('interknots').text)
	ls_min_dist = float(mpmethod.find('terminationdist').text)
	seps = float(mpmethod.find('interrobotsafetydist').text)
	deps = float(mpmethod.find('conflictfreepathdeviation').text)


	optimizer = root.find('optimizer')

	f_max_it = int(optimizer.find('maxiteraction').find('first').text)
	max_it = int(optimizer.find('maxiteraction').find('inter').text)
	l_max_it = int(optimizer.find('maxiteraction').find('last').text)
	xaccuracy = float(optimizer.find('xtolerance').text)
	faccuracy = float(optimizer.find('ftolerance').text)
	eaccuracy = float(optimizer.find('eqtolerance').text)
	iaccuracy = float(optimizer.find('ineqtolerance').text)
	

	# iterate over obstacles
	obstacles = []
	#print rand_round_obst(6, Boundary([-6.0, 6.0], [-1.5, 1.5]), 0.3, 1.0)
	obst_info = rand_round_obst(6, Boundary([-6.0, 6.0], [-3.5, 3.5]), 0.3, 1.0)
	#obst_info = [([0.47740000000000027, -1.5644], 0.3785), ([-1.7155999999999998, 2.1472000000000007], 0.96740000000000004), ([1.6221000000000005, 1.5644], 0.7268), ([-3.5967000000000002, 1.0576000000000003], 0.39549999999999996), ([2.8158000000000003, -0.32899999999999974], 0.7802), ([-4.0045999999999999, -1.5223], 0.8407), ([-4.8949999999999996, -0.44329999999999981], 0.40489999999999998), ([0.96060000000000034, -2.4384999999999999], 0.31629999999999997)]
	#obst_info = [([0.49680000000000035, 0.72990000000000022], 0.93759999999999999), ([3.1783999999999999, 1.8445], 0.49429999999999996), ([-4.6891999999999996, -0.58109999999999995], 0.32019999999999998), ([0.35390000000000033, -0.92769999999999997], 0.3125), ([-1.0412999999999997, -1.8829], 0.85339999999999994), ([-1.8340999999999998, 0.90850000000000009], 0.61370000000000002), ([-3.1894, 2.2316000000000003], 0.98540000000000005), ([-2.8441000000000001, -0.039099999999999913], 0.38519999999999999)]

	#good
	#obst_info = [([-0.1452, -0.60349999999999993], 0.52690000000000003), ([-3.0335000000000001, 1.8125999999999998], 0.69430000000000003), ([-2.9085000000000001, -1.9249000000000001], 0.73399999999999999), ([4.1028000000000002, -1.2079], 0.90149999999999997), ([1.0766, -1.7332999999999998], 0.99249999999999994), ([4.2360000000000007, 1.0126000000000004], 0.54249999999999998), ([1.9709000000000003, 0.60519999999999996], 0.48760000000000003)]

	
	# too close robots
	#obst_info = [([-2.7424999999999997, -1.3217999999999999], 0.56369999999999998), ([-4.3575999999999997, -1.8159999999999998], 0.39479999999999998), ([2.3048999999999999, 0.65350000000000019], 0.60289999999999999), ([3.4901999999999997, -1.7970999999999999], 0.77550000000000008), ([-1.0943999999999998, 2.1657999999999999], 0.44309999999999999), ([-1.9043999999999999, 0.076600000000000001], 0.42330000000000001), ([4.7538999999999998, -1.2408999999999999], 0.37909999999999999)]

	# shows problem with using waypoint instead of goal (update: not any more)
	#obst_info = [([-0.88399999999999945, 0.77760000000000007], 0.311), ([4.8483000000000001, 2.0894000000000004], 0.54430000000000001), ([-4.1993999999999998, 1.8029000000000002], 0.88129999999999997), ([-2.234, -1.9874999999999998], 0.46040000000000003), ([2.0665000000000004, -1.3129], 0.3246), ([-4.9908000000000001, -1.0219], 0.42010000000000003), ([-0.020599999999999952, -0.26149999999999984], 0.9205000000000001)]

	# wierd error after solving the previous problem (update: not anymore, there was an error)
	#obst_info = [([4.3548000000000009, 1.5979999999999999], 0.50609999999999999), ([2.9330000000000007, -1.6259999999999999], 0.91749999999999998), ([2.5638000000000005, 2.4053000000000004], 0.91030000000000011), ([-3.6666999999999996, 1.3669000000000002], 0.56240000000000001), ([-2.0231999999999997, -0.21369999999999978], 0.96300000000000008), ([0.30909999999999993, 0.73850000000000016], 0.97680000000000011), ([-4.9151999999999996, -0.51069999999999993], 0.58260000000000001)]

	#obst_info = [([1.8823000000000008, 1.0751000000000004], 0.87090000000000001), ([-2.3323, 1.7221000000000002], 0.37740000000000001), ([2.8101000000000003, -0.3254999999999999], 0.55899999999999994), ([-0.41239999999999988, -1.2206999999999999], 0.3271), ([3.7513000000000005, 0.76020000000000021], 0.55469999999999997), ([-4.5777999999999999, -0.36509999999999998], 0.68520000000000003), ([-0.40120000000000022, 1.335], 0.92430000000000012)]

	# triple collision case
	#obst_info = [([-2.5352999999999999, 1.6351000000000004], 0.43079999999999996), ([0.91250000000000053, 2.0611000000000006], 0.58960000000000001), ([-4.4641000000000002, -2.04], 0.77490000000000003), ([3.3092000000000006, -2.3753000000000002], 0.98550000000000004), ([2.1196000000000002, 1.6164000000000005], 0.3725), ([-0.42790000000000017, 0.69450000000000012], 0.4607), ([-2.1358999999999999, 0.76829999999999998], 0.42999999999999999)]

	#obst_info = [([2.3321000000000005, 2.1857000000000006], 0.73380000000000001), ([-0.64290000000000003, 2.2968000000000002], 0.31379999999999997), ([2.8586, 0.15080000000000027], 0.94829999999999992), ([-2.8447, -0.12719999999999976], 0.33299999999999996), ([-3.6498999999999997, -2.1516000000000002], 0.69579999999999997), ([4.3231999999999999, 2.4077999999999999], 0.65129999999999999), ([-1.3243, -1.8136999999999999], 0.8015000000000001)]

	obstacles = [RoundObstacle(i[0], i[1]) for i in obst_info]
	print obst_info
	# for obstacle in root.find('obstacles'):
	# 	if obstacle.tag == 'circular':
	# 		obstacles.append(RoundObstacle(
	# 				[float(obstacle.find('cmposition').find('x').text), float(obstacle.find('cmposition').find('y').text)],
	# 				float(obstacle.find('radius').text)))
	# 	elif obstacle.tag == 'polygon':
	# 		vertices = []
	# 		for vertex in obstacle.find('vertices'):
	# 			vertices.append([float(vertex.find('x').text), float(vertex.find('y').text)])
	# 		obstacles.append(PolygonObstacle((np.array(vertices))))
	# 	else:
	# 		logging.info("Unknown type of obstacle")

	boundary = Boundary([-12.0, 12.0], [-12.0, 12.0])

	# iterate over robots

	kine_models = []
	for robot in root.find('aivs'):
		if robot.tag == 'aiv':
			kine_models.append(UnicycleKineModel(
				[float(robot.find('initpose').find('x').text),
				float(robot.find('initpose').find('y').text),
				float(robot.find('initpose').find('theta').text)],# q_initial
				[float(robot.find('goalpose').find('x').text),
				float(robot.find('goalpose').find('y').text),
				float(robot.find('goalpose').find('theta').text)],# q_final
				[float(robot.find('initvelo').find('linear').text),
				float(robot.find('initvelo').find('angular').text)],# u_initial
				[float(robot.find('goalvelo').find('linear').text),
				float(robot.find('goalvelo').find('angular').text)],# u_final
				[float(robot.find('maxvelo').find('linear').text),
				float(robot.find('maxvelo').find('angular').text)],# u_max
				[float(robot.find('maxacc').find('linear').text),
				float(robot.find('maxacc').find('angular').text)])) #a_max
		else:
			logging.info("Unknown type of robot")

	no_robots = len(kine_models)
	no_obsts = len(obstacles)

	###########################################################################


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
		parser.add_option('-S', '--timesamplingsol', dest='no_ssol', default=no_sopt, action='store', type='int', help='number of time samples used for creating the solution')
		return

	scriptname = sys.argv[0]

	parser = OptionParser()
	add_cmdline_options(parser)
	(options, args) = parser.parse_args()

	try:
		os.mkdir(options.direc)
	except OSError:
		print('Probably the output directory '+options.direc+' already exists.')

	sim_id = '_'+str(no_robots)+\
			'_'+str(no_obsts)+\
			'_'+str(time_c)+\
			'_'+str(time_p)+\
			'_'+str(no_sopt)+\
			'_'+str(options.no_ssol)+\
			'_'+str(no_knots)+\
			'_'+str(xaccuracy)+\
			'_'+str(max_it)+\
			'_'+str(f_max_it)+\
			'_'+str(l_max_it)+\
			'_'+str(deps)+\
			'_'+str(seps)+\
			'_'+str(6.0)+\
			'_'+str(ls_min_dist)

	if options.savelog:
		flog = options.direc+'/'+scriptname[0:-3]+sim_id+'.log'
		logging.basicConfig(filename=flog, format='%(levelname)s:%(message)s', \
				filemode='w', level=logging.DEBUG)
	else:
		logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


	logging.debug('\nObstInfo\n{}'.format(obst_info))

	boundary = Boundary([-12.0, 12.0], [-12.0, 12.0])

	robots = []
	for i, robot in zip(range(len(kine_models)), root.find('aivs')):
		if i-1 >= 0 and i+1 < len(kine_models):
			neigh = [i-1, i+1]
		elif i-1 >= 0:
			neigh = [i-1]
		else:
			neigh = [i+1]
		robots += [Robot(
			i,					  # Robot ID
			kine_models[i],		 # kinetic model
			obstacles,			  # all obstacles
			boundary,			   # planning plane boundary
			neigh,				  # neighbors to whom this robot shall talk ...
									#...(used for conflict only, not for real comm between process)
			N_s=no_sopt,				# numbers samplings for each planning interval
			N_ssol=options.no_ssol,
			n_knots=no_knots,# number of knots for b-spline interpolation
			Tc=time_c,	   # computation time
			Tp=time_p,	   # planning horizon
			Td=time_p,
			def_epsilon=deps,	   # in meters
			safe_epsilon=seps,	  # in meters
			detec_rho=float(robot.find('detectionradius').text),
			ls_time_opt_scale = ls_time_opt_scale,
			ls_min_dist = ls_min_dist,
			rho = 0.4)]

	[r.set_option('xacc', xaccuracy) for r in robots]
	[r.set_option('facc', faccuracy) for r in robots] 
	[r.set_option('eacc', eaccuracy) for r in robots] 
	[r.set_option('iacc', iaccuracy) for r in robots] 
	[r.set_option('maxit', max_it) for r in robots] 
	[r.set_option('ls_maxit', l_max_it) for r in robots] 
	[r.set_option('fs_maxit', f_max_it) for r in robots]

	world_sim = WorldSim(sim_id, options.direc, robots, obstacles, boundary, plot=options.plot)

	summary_info = world_sim.run() # run simulation