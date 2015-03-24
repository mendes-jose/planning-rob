#!/usr/bin/python
# Planification for mobile robots
# angles domain: [0, 2pi]

import numpy as np
import numpy.linalg as LA
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab
import math as mth
from scipy.optimize import fmin_slsqp
import scipy.interpolate as si
import time
import itertools
import sys
import pyOpt
import logging

###############################################################################
# Unicycle Kinematic Model 
###############################################################################
class UnicycleKineModel(object):
    """ This class defines the kinematic model of an unicycle mobile robot.
        Unicycle kinematic model:
        q' = f(q,u)
        [x', y', theta']^T = [v cos(theta), v sin(theta), w]^T

        Changing variables (z = [x, y]^T) we rewrite the system as:

        |x    |   |z1                                 |
        |y    |   |z2                                 |
        |theta| = |arctan(z2'/z1')                    |
        |v    |   |sqrt(z1'^2 + z2'^2)                |
        |w    |   |(z1'z2'' - z2'z1'')/(z1'^2 + z2'^2)|
    """
    def __init__(
            self,
            q_init,
            q_final,
            u_init=[0.0,0.0],
            u_final=[0.0,0.0],
            u_max=[0.5,5.0]):
        # Control
        self.u_dim = 2
        self.u_init = np.matrix(u_init).T
        self.u_final = np.matrix(u_final).T
        self.u_max = np.matrix(u_max).T
        # State
        self.q_dim = 3
        self.q_init = np.matrix(q_init).T #angle in [0, 2pi]
        self.q_final = np.matrix(q_final).T #angle in [0, 2pi]
        
        self.l = 2 # number of need derivations

    # z here is a list of matrix [z dz ddz]
    def phi1(self, z):
        """ Returns [x, y, theta] given [z dz ddz] (only z and dz are used)
        """
        if z.shape >= (self.u_dim, self.l+1):
            return np.append(z[:,0], \
                    np.matrix(np.arctan2(z[1,1], z[0,1])), axis = 0)
        else:
            logging.warning('Bad z input. Returning zeros')
            return np.matrix('0.0, 0.0, 0.0')

    # z here is a list of matrix [z dz ddz]
    def phi2(self, z):
        """ Returns [v, w] given [z dz ddz] (only dz and ddz are used)
        """
        if z.shape >= (self.u_dim, self.l+1):
            if (z[0,1]**2 + z[1,1]**2 != 0):
                return np.matrix([[LA.norm(z[:,1])], \
                        [(z[0,1]*z[1,2]-z[1,1]*z[0,2] \
                        )/(z[0,1]**2 + z[1,1]**2)]])
            else:
                return np.matrix([[LA.norm(z[:,1])],[0.0]])
        else:
            logging.warning('Bad z input. Returning zeros')
            return np.matrix('0.0, 0.0')

###############################################################################
# Robot
###############################################################################
class Robot(object):
    def __init__(
            self,
            kine_model,
            obstacles,
            phy_boundary,
            N_s=100,
            t_init=0.0,
            t_sup=1e10,
            rho=0.2):

        self.k_mod = kine_model
        self.obst = obstacles
        self.p_bound = phy_boundary
        self.N_s = N_s # no of samples for discretization of time
        self.t_init = t_init
        self.t_sup = t_sup # superior limit of time
        self.rho = rho

        # Generic params
        self.setOption('ALGO')
        self.setOption('IPLOT')

        # Optimal solver params
        self.setOption('NKNOTS')
        self.setOption('OPTMETHOD')
        self.setOption('ACC')
        self.setOption('MAXIT')
        self.setOption('IPRINT')

        self.q_tktk = self.k_mod.q_initial
        self.q_tktik = self.k_mod.q_initial

    # set Options
    def setOption(self, name, value=None, extra_value=None):
        if name is 'ACC':
            if value is None:
                self.optAcc = 1e-6
            else:
                self.optAcc = value
        elif name is 'MAXIT':
            if value is None:
                self.optMaxIt = 100
            else:
                self.optMaxIt = value
        elif name is 'IPRINT':
            if value is None:
                self.optIprint = -1
            else:
                self.optIprint = value
        elif name is 'NKNOTS':
            if value is None:
                self.n_knots = 15
            else:
                self.n_knots = value
        elif name is 'ALGO':
            if value is None:
                self.algo = 'optsol'
            else:
                self.algo = value
        elif name is 'OPTMETHOD':
            if value is None:
                self.optMethod = 'slsqp'
            else:
                self.optMethod = value
        elif name is 'IPLOT':
            if value is None:
                self.interacPlot = False
            else:
                self.interacPlot = value
                self.fig = extra_value
        else:
            logging.warning('Unknown parameter '+name+', nothing will be set')

    # Generate b-spline knots
    def _gen_knots(self, t_final):
        knots = [self.t_init]
        for j in range(1,self.d):
            knots_j = self.t_init
            knots = knots + [knots_j]

        for j in range(self.d,self.d+self.n_knots):
            knots_j = self.t_init + (j-(self.d-1.0))* \
                    (t_final-self.t_init)/self.n_knots
            knots = knots + [knots_j]

        for j in range(self.d+self.n_knots,2*self.d-1+self.n_knots):
            knots_j = t_final
            knots = knots + [knots_j]

        return np.asarray(knots)

    # Combine base b-splines
    def _comb_bsp(self, t, ctrl_pts, deriv_order):
        tup = (
                self.knots, # knots
                np.squeeze(np.asarray(ctrl_pts[:,0].transpose())), # ctrl pts
                self.d-1) # b-spline degree

        # interpolation
        z = np.matrix(si.splev(t, tup, der=deriv_order)).transpose()

        for i in range(self.k_mod.u_dim)[1:]:
            tup = (
                    self.knots,
                    np.squeeze(np.asarray(ctrl_pts[:,i].transpose())),
                    self.d-1)
            z = np.append(
                    z,
                    np.matrix(si.splev(t,tup,der=deriv_order)).transpose(),
                    axis=1)
        return z

    # Give the curve which interpolates the control points (or its derivates)
    def _gen_dtraj(self, x, deriv_order):
        ctrl_pts = np.asmatrix(
                np.asarray(x[1:]).reshape(self.n_ctrlpts, self.k_mod.u_dim))
        mtime = np.linspace(self.t_init, x[0], self.N_s)
        return self._comb_bsp(mtime, ctrl_pts, deriv_order)

    # Object Function (used when finding the path with optsol option)
    def _obj_func(self, x):

        # creating some useful variables
        t_final = x[0]
        ctrl_pts = np.asmatrix(
                np.asarray(x[1:]).reshape(self.n_ctrlpts, self.k_mod.u_dim))

        # updatating knots
        self.knots = self._gen_knots(t_final)

        if self.interacPlot is True:
            self.curve = self._gen_dtraj(x, 0)
            self.ctrl_pts = ctrl_pts
            self._plot_update_optsol()

        # creating time
        mtime = np.linspace(self.t_init, t_final, self.N_s)

        # get a list over time of the matrix [z dz ddz](t)
        all_zl = [np.append(np.append(
                self._comb_bsp(tk, ctrl_pts, 0).transpose(),
                self._comb_bsp(tk, ctrl_pts, 1).transpose(), axis = 1),
                self._comb_bsp(tk, ctrl_pts, 2).transpose(), axis = 1) \
                for tk in mtime]

        # get a list over time of command values u(t)
        all_us = map(self.k_mod.phi2, all_zl)

        #----------------------------------------------------------------------
        # Cost Object (criterium)
        #----------------------------------------------------------------------
        J = (t_final-self.t_init)**2

        #----------------------------------------------------------------------
        # Final and initial values constraints
        #----------------------------------------------------------------------
        econs = np.append(np.append(np.append(
                np.asarray(self.k_mod.phi1(all_zl[0])-self.k_mod.q_init),
                np.asarray(self.k_mod.phi1(all_zl[-1])-self.k_mod.q_final)),
                np.asarray(self.k_mod.phi2(all_zl[0])-self.k_mod.u_init)),
                np.asarray(self.k_mod.phi2(all_zl[-1])-self.k_mod.u_final))

        # Get equations that were not respected
        self.unsatisf_eq_values = [ec for ec in econs if ec is not 0]

        #----------------------------------------------------------------------
        # Obstacles constraints at each time step
        #----------------------------------------------------------------------
        obst_cons = np.array([(self.rho + self.obst[0].radius()) - \
                LA.norm(np.matrix(self.obst[0].pos).T - zl[:,0]) \
                for zl in all_zl])
        for m in range(1,len(self.obst)):
            obst_cons = np.append(
                    obst_cons,
                    np.array([(self.rho + self.obst[m].radius()) - \
                    LA.norm(np.matrix(self.obst[m].pos).T - zl[:,0]) \
                    for zl in all_zl]))

        #----------------------------------------------------------------------
        # Discrete displacement constraints
        #----------------------------------------------------------------------
#        disc_dis = [LA.norm(all_zl[ind][:,0]-all_zl[ind+1][:,0])-0.1 \
#                for ind in range(len(all_zl)-1)]

        #----------------------------------------------------------------------
        # Max speed constraints
        #----------------------------------------------------------------------
        max_speed_cons = np.asarray(list(itertools.chain.from_iterable(
                map(lambda u:[abs(u[0,0]) - self.k_mod.u_max[0,0],
                abs(u[1,0]) - self.k_mod.u_max[1,0]], all_us))))

        icons = np.append(obst_cons, max_speed_cons)
#        icons = np.append(icons, disc_dis)

        # Get inequations that were not respected
        self.unsatisf_ieq_values = [ic for ic in icons if ic > 0]

        cons = np.append(econs, icons)

        return J, cons, 0

    # Object Function (used when finding the path with optinline option)
    def _obj_func_optinline(self, x):

        # creating some useful variables
        ctrl_pts = np.asmatrix(
                np.asarray(x).reshape(self.n_ctrlpts, self.k_mod.u_dim))

        # updatating knots
        self.knots = self._gen_knots(self.Tp+self.t_init)

        if self.interacPlot is True:
            self.curve = self._gen_dtraj(x, 0)
            self.ctrl_pts = ctrl_pts
            self._plot_update_optsol()

        # creating time
        mtime = np.linspace(self.t_init, self.Tp+self.t_init, self.N_s)

        # get a list over time of the matrix [z dz ddz](t)
        all_zl = [np.append(np.append(
                self._comb_bsp(tk, ctrl_pts, 0).transpose(),
                self._comb_bsp(tk, ctrl_pts, 1).transpose(), axis = 1),
                self._comb_bsp(tk, ctrl_pts, 2).transpose(), axis = 1) \
                for tk in mtime]

        # get a list over time of command values u(t)
        all_us = map(self.k_mod.phi2, all_zl)

        #----------------------------------------------------------------------
        # Cost Object (criterium)
        #----------------------------------------------------------------------
        J = (t_final-self.t_init)**2
        q_taukTp = self.k_mod.phi1(all_zl[-1])
        J = LA.norm(q_taukTp-self.q_final)**2

        #----------------------------------------------------------------------
        # Final and initial values constraints
        #----------------------------------------------------------------------
        econs = np.append(np.append(np.append(
                np.asarray(self.k_mod.phi1(all_zl[0])-self.k_mod.q_init),
                np.asarray(self.k_mod.phi1(all_zl[-1])-self.k_mod.q_final)),
                np.asarray(self.k_mod.phi2(all_zl[0])-self.k_mod.u_init)),
                np.asarray(self.k_mod.phi2(all_zl[-1])-self.k_mod.u_final))

        # Get equations that were not respected
        self.unsatisf_eq_values = [ec for ec in econs if ec is not 0]

        #----------------------------------------------------------------------
        # Obstacles constraints at each time step
        #----------------------------------------------------------------------
        obst_cons = np.array([(self.rho + self.obst[0].radius()) - \
                LA.norm(np.matrix(self.obst[0].pos).T - zl[:,0]) \
                for zl in all_zl])
        for m in range(1,len(self.obst)):
            obst_cons = np.append(
                    obst_cons,
                    np.array([(self.rho + self.obst[m].radius()) - \
                    LA.norm(np.matrix(self.obst[m].pos).T - zl[:,0]) \
                    for zl in all_zl]))

        #----------------------------------------------------------------------
        # Discrete displacement constraints
        #----------------------------------------------------------------------
#        disc_dis = [LA.norm(all_zl[ind][:,0]-all_zl[ind+1][:,0])-0.1 \
#                for ind in range(len(all_zl)-1)]

        #----------------------------------------------------------------------
        # Max speed constraints
        #----------------------------------------------------------------------
        max_speed_cons = np.asarray(list(itertools.chain.from_iterable(
                map(lambda u:[abs(u[0,0]) - self.k_mod.u_max[0,0],
                abs(u[1,0]) - self.k_mod.u_max[1,0]], all_us))))

        icons = np.append(obst_cons, max_speed_cons)
#        icons = np.append(icons, disc_dis)

        # Get inequations that were not respected
        self.unsatisf_ieq_values = [ic for ic in icons if ic > 0]

        cons = np.append(econs, icons)

        return J, cons, 0
    def _init_optsol(self):
        # Optimal solution parameters
        self.d = self.k_mod.l+2 # B-spline order (integer | d > l+1)
        self.n_ctrlpts = self.n_knots + self.d - 1 # nb of ctrl points

        # Unknown parameters (defining initial value)
        #   Initiate t_final with the limit inferior of time to go from   
        #   q_init to q_final which is:
        #   dist_between_intial_and_final_positions/linear_speed_max_value
        self.t_inf=LA.norm(
                self.k_mod.q_init[0:-1,0]-self.k_mod.q_final[0:-1,0])/ \
                self.k_mod.u_max[0,0]
        self.t_final = self.t_inf
        logging.debug('t_sup: {}'.format(self.t_sup))
        logging.debug('t_inf: {}'.format(self.t_inf))

        # Initiate control points so the robot do a straight line from
        #   inital to final positions
        self.ctrl_pts=np.matrix(np.zeros((self.n_ctrlpts,self.k_mod.u_dim)))
        self.ctrl_pts[:,0]=np.matrix(np.linspace( # linspace in x
                self.k_mod.q_init[0,0],
                self.k_mod.q_final[0,0],
                self.n_ctrlpts)).T
        self.ctrl_pts[:,1]=np.matrix(np.linspace( # linsapce in y
                self.k_mod.q_init[1,0],
                self.k_mod.q_final[1,0],
                self.n_ctrlpts)).T

        # Generate initial b-spline knots
        self.knots = self._gen_knots(self.t_final)

        # Minimization argument (x)
        x_init = np.append(self.t_inf, np.asarray(self.ctrl_pts)).tolist()

        str_x_init = 'x_init: '
        for i in x_init:
            str_x_init = str_x_init+"%0.2f " % i
        logging.debug(str_x_init)

        x_lower = [self.t_inf] + \
                [self.p_bound.x_min, self.p_bound.y_min]*self.n_ctrlpts
        x_upper = [self.t_sup] + \
                [self.p_bound.x_max, self.p_bound.y_max]*self.n_ctrlpts

        # Initiate the path (interpolation of control points)
        self.curve = self._gen_dtraj(x_init, 0)

        # Define the optimization problem
        self.opt_prob = pyOpt.Optimization(
                'Faster path with obstacles', # name of the problem
                self._obj_func) # object function (criterium, eq. and ineq.)

        self.opt_prob.addObj('J')

        self.opt_prob.addVarGroup( # minimization arguments
                'x',
                self.k_mod.u_dim*self.n_ctrlpts + 1, # dimension
                'c', # continous
                lower=x_lower,
                value=x_init,
                upper=x_upper)

        self.opt_prob.addConGroup( # equations constraints
                'ec',
                2*self.k_mod.q_dim + 2*self.k_mod.u_dim, # dimension
                'e') # equations

        self.opt_prob.addConGroup( # inequations constraints
                'ic',
                self.N_s*self.k_mod.u_dim +
#                        self.N_s-1+
                        self.N_s*len(self.obst), # dimenstion
                'i') # inequations

        # Optimization statistics
        self.unsatisf_eq_values = []
        self.unsatisf_ieq_values = []

        # Init plot
        ax = self.fig.gca()
        # plot curve and its control points
        self.plt_ctrl_pts,self.cont_curve,self.plt_curve = ax.plot(
            self.ctrl_pts[:,0],
            self.ctrl_pts[:,1],
            '*',
            self.curve[:,0],
            self.curve[:,1],
            'b-',
            self.curve[:,0],
            self.curve[:,1],
            '.')

    def _gen_optsol(self):
        if self.optMethod is 'slsqp':
            slsqp = pyOpt.SLSQP(pll_type='POA') # parameter for parallel solv.
            slsqp.setOption('ACC', self.optAcc) # accuracy param (1e-6)
            slsqp.setOption('MAXIT', self.optMaxIt) # max no of iterations (50)
            slsqp.setOption('IPRINT', self.optIprint) # output style (-1)

            # find the near-optimal solution
            [J, x, inform] = slsqp(self.opt_prob)
            logging.info('Optimization result:\n{}'.format(x))

            self.t_final = x[0]
            self.ctrl_pts = np.asmatrix(
                    np.asarray(x[1:]).reshape(self.n_ctrlpts,self.k_mod.u_dim))
            self.knots = self._gen_knots(self.t_final)
            self.curve = self._gen_dtraj(x, 0)

            return ('Optimization summary: {} exit code {}'.format(
                    inform['text'],
                    inform['value']))

        else:
            logging.warning('Unknown optimization method')

    def _plot_update_optsol(self):
        logging.info('plot update')
        self.plt_curve.set_xdata(self.curve[:,0])
        self.plt_curve.set_ydata(self.curve[:,1])
        self.cont_curve.set_xdata(self.curve[:,0])
        self.cont_curve.set_ydata(self.curve[:,1])
        self.plt_ctrl_pts.set_xdata(self.ctrl_pts[:,0])
        self.plt_ctrl_pts.set_ydata(self.ctrl_pts[:,1])
        self.fig.canvas.draw()

    def gen_trajectory(self):

        # Global optimal solution (knowledge of the whole map)
        if self.algo is 'optsol':

            self._init_optsol()
            self._gen_optsol()
            if self.interacPlot is True:
                self._plot_update_optsol()

        
            
        # Unknow method
        else:
            logging.warning('Unknown trajectory generation method')


###############################################################################
# Obstacle/Boundary
###############################################################################
class Obstacle(object):
    def __init__(self, position, dimension):
        self.pos = position
        self.dim = dimension

class RoundObstacle(Obstacle):
    def x(self):
        return self.pos[0]

    def y(self):
        return self.pos[1]

    def radius(self):
        return self.dim

    def pltCircle(self, color='r',linestyle='solid',isFill=False, offset=0.0):
        return plt.Circle(
                (self.x(), self.y()), # position
                self.radius()+offset, # radius
                color=color,
                ls = linestyle,
                fill=isFill)

    def plot(self, fig, offset=0.0):
        ax = fig.gca()
        ax.add_artist(self.pltCircle())
        ax.add_artist(self.pltCircle(linestyle='dashed', offset=offset))

class Boundary(object):
    def __init__(self, x, y):
        self.x_min = x[0]
        self.x_max = x[1]
        self.y_min = y[0]
        self.y_max = y[1]

###############################################################################
# World
###############################################################################
class WorldSim(object):
    """ Where to instatiate the obstacles, the robot, the bonderies
        initial and final conditions
    """
    def __init__(self, mrobot, obstacles, phy_boundary):
        self.mrobot = mrobot
        self.obst = obstacles
        self.p_bound = phy_boundary

    def run(self, interacPlot=False):

        # Interactive plot
        if interacPlot is True:
            plt.ion()

        # Initiating plot
        self.fig = plt.figure()
        ax = self.fig.gca()
        ax.set_xlabel('x(m)')
        ax.set_ylabel('y(m)')
        ax.set_title('Generated trajectory')
        ax.axis('equal')
#        ax.axis([
#                self.p_bound.x_min,
#                self.p_bound.x_max,
#                self.p_bound.y_min,
#                self.p_bound.y_max])

        if interacPlot is True:
            self.mrobot.setOption('IPLOT', True, self.fig)
            # TODO: in the future we may set IPLOT True for obstacles as well

        # Creating obstacles in the plot
        [obst.plot(self.fig, offset=self.mrobot.rho) for obst in self.obst]

        # Creating robot path (and plot)
        self.mrobot.gen_trajectory()

        plt.show(block=True)

#-----------------------------------------------------------------------------

###############################################################################
# MAIN
###############################################################################
if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    obstacles = [RoundObstacle([ 0.25,  2.50], 0.20),
                 RoundObstacle([ 2.30,  2.50], 0.50), 
                 RoundObstacle([ 1.25,  3.00], 0.10),
                 RoundObstacle([ 0.30,  1.00], 0.10),
                 RoundObstacle([-0.50,  1.50], 0.30)]
#                 RoundObstacle([ 0.70,  1.45], 0.25)]

    boundary = Boundary([-5.0,10.0], [-5.0,10.0])

    kine_model = UnicycleKineModel(
            [ 0.0, 0.0, np.pi/2],
            [ 2.0, 5.0, np.pi/2],
            [ 0.0, 0.0],
            [ 0.0, 0.0],
            [ 0.5, 5.0])

    robot = Robot(
            kine_model,
            obstacles,
            boundary,
#            N_s=100,
#            t_init=0.0,
#            t_sup=1e10,
            rho=0.2)

#    robot.setOption('ALGO', 'optsol')
#    robot.setOption('OPTMETHOD', 'slsqp')
#    robot.setOption('NKNOTS', 15)
#    robot.setOption('ACC', 1e-6)
    robot.setOption('MAXIT', 50)
    robot.setOption('IPRINT', -1)

    world_sim = WorldSim(robot,obstacles,boundary)

    world_sim.run(interacPlot=True)
