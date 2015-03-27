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
            u_max=[0.5,5.0],
            a_max=[1.5,10.0]):
        # Control
        self.u_dim = 2
        self.u_init = np.matrix(u_init).T
        self.u_final = np.matrix(u_final).T
        self.u_max = np.matrix(u_max).T
        self.acc_max = np.matrix(a_max).T
        # State
        self.q_dim = 3
        self.q_init = np.matrix(q_init).T #angle in [0, 2pi]
        self.q_final = np.matrix(q_final).T #angle in [0, 2pi]
        
        self.l = 2 # number of need derivations

    # z here is a list of matrix [z dz ddz]
    def phi1(self, z):
        """ Returns [x, y, theta]^T given [z dz ddz] (only z and dz are used)
        """
        if z.shape >= (self.u_dim, self.l+1):
            return np.append(z[:,0], \
                    np.matrix(np.arctan2(z[1,1], z[0,1])), axis = 0)
        else:
            logging.warning('Bad z input. Returning zeros')
            return np.matrix('0.0; 0.0; 0.0')

    # z here is a list of matrix [z dz ddz]
    def phi2(self, z):
        """ Returns [v, w]^T given [z dz ddz] (only dz and ddz are used)
        """
        if z.shape >= (self.u_dim, self.l+1):
            if (z[0,1]**2 + z[1,1]**2 != 0):
                return np.matrix([[LA.norm(z[:,1])], \
                        [(z[0,1]*z[1,2]-z[1,1]*z[0,2] \
                        )/(z[0,1]**2 + z[1,1]**2)]])
            else:
                logging.warning('x\' and y\' are zero! Using angspeed=0')
                return np.matrix([[LA.norm(z[:,1])],[0.0]])
        else:
            logging.warning('Bad z input. Returning zeros')
            return np.matrix('0.0; 0.0')

    def phi3(self, z):
        """
        """
        if z.shape >= (self.u_dim, self.l+1):
            dz_norm = LA.norm(z[:,1])
            if (dz_norm != 0):
                dz_norm = LA.norm(z[:,1])
                dv = (z[0,1]*z[0,2]+z[1,1]*z[1,2])/dz_norm
                dw = ((z[0,2]*z[1,2]+z[1,3]*z[0,1]- \
                        (z[1,2]*z[0,2]+z[0,3]*z[1,1]))*(dz_norm**2) - \
                        (z[0,1]*z[1,2]-z[1,1]*z[0,2])*2*dz_norm*dv)/dz_norm**4
                return np.matrix([[dv],[dw]])
            else:
                return np.matrix('0.0; 0.0')
        else:
            logging.warning('Bad z input. Returning zeros')
            return np.matrix('0.0; 0.0')


###############################################################################
# Planner
###############################################################################
class Planner(object):
    def __init__(
        self,
        robot,
        


###############################################################################
# Robot
###############################################################################
class Robot(object):
    def __init__(
            self,
            kine_model,
            obstacles,
            phy_boundary,
            planner,
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
        self.setOption('LIB')
        self.setOption('IPLOT')

        # Optimal solver params
        self.setOption('NKNOTS')
        self.setOption('OPTMETHOD')
        self.setOption('ACC')
        self.setOption('MAXIT')
        self.setOption('IPRINT')

        # Init plots
        self.plot()
        self.plotSpeeds()

    # set Options
    def setOption(self, name, value=None):
        if name == 'ACC':
            if value == None:
                self.optAcc = 1e-6
            else:
                self.optAcc = value
        elif name == 'MAXIT':
            if value == None:
                self.optMaxIt = 100
            else:
                self.optMaxIt = value
        elif name == 'IPRINT':
            self.optIprint = value
        elif name == 'NKNOTS':
            if value == None:
                self.n_knots = 15
            else:
                self.n_knots = value
        elif name == 'LIB':
            if value == None:
                self.lib = 'pyopt'
            else:
                self.lib = value
        elif name == 'OPTMETHOD':
            if value == None:
                self.optMethod = 'slsqp'
            else:
                self.optMethod = value
        elif name == 'IPLOT':
            if value == None:
                self.interacPlot = False
            else:
                self.interacPlot = value
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

    def _criteria(self, x):
        #----------------------------------------------------------------------
        # Cost Object (criterium)
        #----------------------------------------------------------------------
        return (x[0]-self.t_init)**2

    def _feqcons(self, x):
        # creating some useful variables
        t_final = x[0]
        ctrl_pts = np.asmatrix(
                np.asarray(x[1:]).reshape(self.n_ctrlpts, self.k_mod.u_dim))

        # updatating knots
        self.knots = self._gen_knots(t_final)

        # get matrix [z dz ddz](t_init)
        dz_t_init = self._comb_bsp(self.t_init, ctrl_pts, 0).T
        for dev in range(1,self.k_mod.l+1):
            dz_t_init=np.append(dz_t_init,self._comb_bsp(self.t_init, ctrl_pts, dev).T,axis=1)

        # get matrix [z dz ddz](t_final)
        dz_t_final = self._comb_bsp(t_final, ctrl_pts, 0).T
        for dev in range(1,self.k_mod.l+1):
            dz_t_final=np.append(dz_t_final,self._comb_bsp(t_final, ctrl_pts, dev).T,axis=1)

        #----------------------------------------------------------------------
        # Final and initial values constraints
        #----------------------------------------------------------------------
        econs = np.append(np.append(np.append(
                np.asarray(self.k_mod.phi1(dz_t_init)-self.k_mod.q_init),
                np.asarray(self.k_mod.phi1(dz_t_final)-self.k_mod.q_final)),
                np.asarray(self.k_mod.phi2(dz_t_init)-self.k_mod.u_init)),
                np.asarray(self.k_mod.phi2(dz_t_final)-self.k_mod.u_final))

        # Get equations that were not respected
        self.unsatisf_eq_values = [ec for ec in econs if ec != 0]

        return econs

    def _fieqcons(self, x):
        # creating some useful variables
        t_final = x[0]
        ctrl_pts = np.asmatrix(
                np.asarray(x[1:]).reshape(self.n_ctrlpts, self.k_mod.u_dim))

        # updatating knots
        self.knots = self._gen_knots(t_final)

        # creating time
        mtime = np.linspace(self.t_init, t_final, self.N_s)

        # get a list over time of the matrix [z dz ddz dddz ...](t)
        all_dz = []
        for tk in mtime:
            dz = self._comb_bsp(tk, ctrl_pts, 0).T
            for dev in range(1,self.k_mod.l+1):
                dz = np.append(dz,self._comb_bsp(tk, ctrl_pts, dev).T,axis=1)
            all_dz += [dz]

        # get a list over time of command values u(t)
        all_us = map(self.k_mod.phi2, all_dz)

        #----------------------------------------------------------------------
        # Obstacles constraints at each time step
        #----------------------------------------------------------------------
        obst_cons = np.array([- (self.rho + self.obst[0].radius()) + \
                LA.norm(np.matrix(self.obst[0].pos).T - zl[:,0]) \
                for zl in all_dz])
        for m in range(1,len(self.obst)):
            obst_cons = np.append(
                    obst_cons,
                    np.array([- (self.rho + self.obst[m].radius()) + \
                    LA.norm(np.matrix(self.obst[m].pos).T - zl[:,0]) \
                    for zl in all_dz]))

        #----------------------------------------------------------------------
        # Discrete displacement constraints
        #----------------------------------------------------------------------
#        disc_dis = [-LA.norm(all_dz[ind][:,0]-all_dz[ind+1][:,0])+0.1 \
#                for ind in range(len(all_dz)-1)]

        #----------------------------------------------------------------------
        # Max speed constraints
        #----------------------------------------------------------------------
        max_speed_cons = np.asarray(list(itertools.chain.from_iterable(
                map(lambda u:[-abs(u[0,0]) + self.k_mod.u_max[0,0],
                -abs(u[1,0]) + self.k_mod.u_max[1,0]], all_us))))

        icons = np.append(obst_cons, max_speed_cons)

        #----------------------------------------------------------------------
        # Max acceleration constraints
        #----------------------------------------------------------------------
        if self.k_mod.l > 2:
            all_dv = map(self.k_mod.phi3, all_dz)
            max_acc_cons = np.asarray(list(itertools.chain.from_iterable(
                    map(lambda dv:[-abs(dv[0,0]) + self.k_mod.acc_max[0,0],
                    -abs(dv[1,0]) + self.k_mod.acc_max[1,0]], all_dv))))

            icons = np.append(icons, max_acc_cons)

        # Get inequations that were not respected
        self.unsatisf_ieq_values = [ic for ic in icons if ic > 0]

        return icons

    def _scipy_callback(self,x):
        self._update_opt(x)
        self._update_opt_plot()

    def _init_scipy(self):
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

        # creating time
        self.mtime = np.linspace(self.t_init, self.t_final, self.N_s)

        # get a list over time of the matrix [z dz ddz dddz ...](t)
        all_dz = []
        for tk in self.mtime:
            dz = self._comb_bsp(tk, self.ctrl_pts, 0).T
            for dev in range(1,self.k_mod.l+1):
                dz = np.append(dz,self._comb_bsp(tk, self.ctrl_pts, dev).T,axis=1)
            all_dz += [dz]

        # get a list over time of command values u(t)
        all_us = map(self.k_mod.phi2, all_dz)

        self.linspeed = map(lambda u:u[0,0], all_us)
        self.angspeed = map(lambda u:u[1,0], all_us)

        # Minimization argument (x)
        self.x_init = np.append(self.t_inf, np.asarray(self.ctrl_pts)).tolist()

        str_x_init = 'x_init: '
        for i in self.x_init:
            str_x_init = str_x_init+"%0.2f " % i
        logging.debug(str_x_init)

        self.x_lower = [self.t_inf] + \
                [self.p_bound.x_min, self.p_bound.y_min]*self.n_ctrlpts
        self.x_upper = [self.t_sup] + \
                [self.p_bound.x_max, self.p_bound.y_max]*self.n_ctrlpts

        # Initiate the path (interpolation of control points)
        self.curve = self._gen_dtraj(self.x_init, 0)

        # Optimization statistics
        self.unsatisf_eq_values = []
        self.unsatisf_ieq_values = []

        self._init_opt_plot()

    def _gen_scipy(self):

        logging.info('Optimization_method: {}'.format(self.optMethod))

        if self.optMethod == 'slsqp':

            if self.interacPlot != None:
                f_callback = self._scipy_callback
            else:
                f_callback = None

            x = fmin_slsqp(self._criteria,
                    self.x_init,
                    eqcons=(),
                    f_eqcons=self._feqcons,
                    ieqcons=(),
                    f_ieqcons=self._fieqcons,
                    iter=self.optMaxIt,
                    acc=self.optAcc,
                    iprint=self.optIprint,
                    callback=f_callback)

            if self.interacPlot == None:
                self._update_opt()

            logging.info('Optimization_results:\n{}'.format(x))

        else:
            logging.warning('Unknown optimization method inside lib {}'.format(self.lib))

    # Object Function (used when finding the path with pyopt option)
    def _obj_func_pyopt(self, x):

        # creating some useful variables
        t_final = x[0]
        ctrl_pts = np.asmatrix(
                np.asarray(x[1:]).reshape(self.n_ctrlpts, self.k_mod.u_dim))

        # updatating knots
        self.knots = self._gen_knots(t_final)

        # creating time
        mtime = np.linspace(self.t_init, t_final, self.N_s)

        # get a list over time of the matrix [z dz ddz dddz ...](t)
        all_dz = []
        for tk in mtime:
            dz = self._comb_bsp(tk, ctrl_pts, 0).T
            for dev in range(1,self.k_mod.l+1):
                dz = np.append(dz,self._comb_bsp(tk, ctrl_pts, dev).T,axis=1)
            all_dz += [dz]

        # get a list over time of command values u(t)
        all_us = map(self.k_mod.phi2, all_dz)

        # update plot if interactive plot is true
        if self.interacPlot == True:
            self.curve = self._gen_dtraj(x, 0)
            self.ctrl_pts = ctrl_pts
            self.linspeed = map(lambda x:x[0,0], all_us)
            self.angspeed = map(lambda x:x[1,0], all_us)
            self.mtime = mtime
            self._update_opt_plot()

        #----------------------------------------------------------------------
        # Cost Object (criterium)
        #----------------------------------------------------------------------
        J = (t_final-self.t_init)**2

        #----------------------------------------------------------------------
        # Final and initial values constraints
        #----------------------------------------------------------------------
        econs = np.append(np.append(np.append(
                np.asarray(self.k_mod.phi1(all_dz[0])-self.k_mod.q_init),
                np.asarray(self.k_mod.phi1(all_dz[-1])-self.k_mod.q_final)),
                np.asarray(self.k_mod.phi2(all_dz[0])-self.k_mod.u_init)),
                np.asarray(self.k_mod.phi2(all_dz[-1])-self.k_mod.u_final))

        # Get equations that were not respected
        self.unsatisf_eq_values = [ec for ec in econs if ec != 0]

        #----------------------------------------------------------------------
        # Obstacles constraints at each time step
        #----------------------------------------------------------------------
        obst_cons = np.array([(self.rho + self.obst[0].radius()) - \
                LA.norm(np.matrix(self.obst[0].pos).T - zl[:,0]) \
                for zl in all_dz])
        for m in range(1,len(self.obst)):
            obst_cons = np.append(
                    obst_cons,
                    np.array([(self.rho + self.obst[m].radius()) - \
                    LA.norm(np.matrix(self.obst[m].pos).T - zl[:,0]) \
                    for zl in all_dz]))

        #----------------------------------------------------------------------
        # Discrete displacement constraints
        #----------------------------------------------------------------------
#        disc_dis = [LA.norm(all_dz[ind][:,0]-all_dz[ind+1][:,0])-0.1 \
#                for ind in range(len(all_dz)-1)]

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

    def _init_pyopt(self):
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

        # creating time
        self.mtime = np.linspace(self.t_init, self.t_final, self.N_s)

        # get a list over time of the matrix [z dz ddz dddz ...](t)
        all_dz = []
        for tk in self.mtime:
            dz = self._comb_bsp(tk, self.ctrl_pts, 0).T
            for dev in range(1,self.k_mod.l+1):
                dz = np.append(dz,self._comb_bsp(tk, self.ctrl_pts, dev).T,axis=1)
            all_dz += [dz]

        # get a list over time of command values u(t)
        all_us = map(self.k_mod.phi2, all_dz)

        self.linspeed = map(lambda u:u[0,0], all_us)
        self.angspeed = map(lambda u:u[1,0], all_us)

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
                self._obj_func_pyopt) # object function (criterium, eq. and ineq.)

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

        self._init_opt_plot()

    def _gen_pyopt(self):
        #--------------------------------------------------
        #-----------------free and working-----------------
        #--------------------------------------------------
        #    'slsqp'
        #    'psqp'
        #    'algencan'
        #    'alpso' # works but really bad
        #    'alhso' # works but really bad
        #--------------------------------------------------
        #----------------free but does apply---------------
        #--------------------------------------------------
        #    'filtersd' # cannot handle eq cons
        #    'conmin' # cannot handle eq cons
        #    'mmfd' # didn't find the module
        #    'ksopt' # cannot handle eq cons
        #    'cobyla' # cannot handle eq cons
        #    'sdpen' # cannot handle eq cons
        #    'solvopt' # err constraints has no upper att
        #    'nsga2' # cannot handle eq cons
        #--------------------------------------------------
        #------------------licenced------------------------
        #--------------------------------------------------
        #    'snopt'
        #    'nlpql'
        #    'fsqp'
        #    'mma'
        #    'gcmma'
        #    'midaco'
        #--------------------------------------------------

        logging.info('Optimization_method: {}'.format(self.optMethod))

        if self.optMethod == 'slsqp':
            solver = pyOpt.SLSQP(pll_type='POA') # parameter for parallel solv.
            solver.setOption('ACC', self.optAcc) # accuracy param (1e-6)
            solver.setOption('MAXIT', self.optMaxIt) # max no of iterations
        elif self.optMethod == 'psqp':
            solver = pyOpt.PSQP(pll_type='POA') # parameter for parallel solv.
            solver.setOption('MIT', self.optMaxIt) # max no of iterations
        elif self.optMethod == 'algencan':
            solver = pyOpt.ALGENCAN(pll_type='POA') # parameter for parallel solv
        elif self.optMethod == 'alpso':
            solver = pyOpt.ALPSO(pll_type='POA') # parameter for parallel solv.
        elif self.optMethod == 'alhso':
            solver = pyOpt.ALHSO(pll_type='POA') # parameter for parallel solv.
        else:
            logging.warning('Unknown optimization method inside lib {}'.format(self.lib))
            return

        if self.optIprint != None:
            solver.setOption('IPRINT', self.optIprint) # output style

        # find the near-optimal solution
        [J, x, inform] = solver(self.opt_prob)
        logging.info('Optimization_results:\n{}'.format(x))

        # update values according with optimization result
        self._update_opt(x)

        return ('Optimization summary: {} exit code {}'.format(
                inform['text'],
                inform['value']))

    def _update_opt(self, x):
        self.t_final = x[0]
        self.ctrl_pts = np.asmatrix(
                np.asarray(x[1:]).reshape(self.n_ctrlpts, self.k_mod.u_dim))
        self.knots = self._gen_knots(self.t_final)
        self.curve = self._gen_dtraj(x, 0)
        self.mtime = np.linspace(self.t_init, self.t_final, self.N_s)
        all_dz = []
        for tk in self.mtime:
            dz = self._comb_bsp(tk, self.ctrl_pts, 0).T
            for dev in range(1,self.k_mod.l+1):
                dz = np.append(dz,self._comb_bsp(tk, self.ctrl_pts, dev).T,axis=1)
            all_dz += [dz]
        all_us = map(self.k_mod.phi2, all_dz)
        self.linspeed = map(lambda u:u[0,0], all_us)
        self.angspeed = map(lambda u:u[1,0], all_us)
        if self.k_mod.l > 2:
            all_dv = map(self.k_mod.phi3, all_dz)
            self.linacc = map(lambda dv:dv[0,0], all_dv)
            self.angacc = map(lambda dv:dv[1,0], all_dv)

    def _init_opt_plot(self):
        # PLOTS
        if self.fig != None:
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

        if self.figSpeed != None:
            axarray = self.figSpeed.axes
            self.plt_linspeed, = axarray[0].plot(
                    self.mtime,
                    self.linspeed)
            self.plt_angspeed, = axarray[1].plot(
                    self.mtime,
                    self.angspeed)
            axarray[0].set_ylabel('v(m/s)')
            axarray[0].set_title('Linear speed')
            axarray[1].set_xlabel('time(s)')
            axarray[1].set_ylabel('w(rad/s)')
            axarray[1].set_title('Angular speed')
            axarray[0].grid()
            axarray[1].grid()

    def _update_opt_plot(self):
        # plot
        self.plt_curve.set_xdata(self.curve[:,0])
        self.plt_curve.set_ydata(self.curve[:,1])
        self.cont_curve.set_xdata(self.curve[:,0])
        self.cont_curve.set_ydata(self.curve[:,1])
        self.plt_ctrl_pts.set_xdata(self.ctrl_pts[:,0])
        self.plt_ctrl_pts.set_ydata(self.ctrl_pts[:,1])

        ax = self.fig.gca()
        ax.relim()
        ax.autoscale_view(True,True,True)

        self.fig.canvas.draw()

        if self.figSpeed != None:
            # speed plot
            self.plt_angspeed.set_xdata(self.mtime)
            self.plt_angspeed.set_ydata(self.angspeed)
            self.plt_linspeed.set_xdata(self.mtime)
            self.plt_linspeed.set_ydata(self.linspeed)
    
            axarray = self.figSpeed.axes
            axarray[0].relim()
            axarray[0].autoscale_view(True,True,True)
            axarray[1].relim()
            axarray[1].autoscale_view(True,True,True)
    
            self.figSpeed.canvas.draw()

    def plot(self, figure=None):
        self.fig = figure

    def plotSpeeds(self, figure=None):
        if figure != None:
            self.figSpeed = figure[0]
        else:
            self.figSpeed = None

    def gen_trajectory(self):

        logging.info('Optimization_lib: {}'.format(self.lib))
    
        # Global optimal solution using pyopt (knowledge of the whole map)
        if self.lib == 'pyopt':
            self._init_pyopt()
            self.elapsed_time = time.clock()
            ret = self._gen_pyopt()
            self.elapsed_time = time.clock() - self.elapsed_time
            self._update_opt_plot()
            
        # Global optimal solution using scipy (knowledge of the whole map)
        elif self.lib == 'scipy':
            self._init_scipy()
            self.elapsed_time = time.clock()
            ret = self._gen_scipy()
            self.elapsed_time = time.clock() - self.elapsed_time
            self._update_opt_plot()
            
        # Unknow method
        else:
            logging.warning('Unknown optimization lib')
            ret = None
        return ret

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

    def run(self, interacPlot=False, speedPlot=False):

        # Interactive plot
        if interacPlot == True:
            plt.ion()
            self.mrobot.setOption('IPLOT', True)
            # TODO: in the future we may set IPLOT True for obstacles as well

        # Initiating plot
        self.fig = plt.figure()
        ax = self.fig.gca()
        ax.set_xlabel('x(m)')
        ax.set_ylabel('y(m)')
        ax.set_title('Generated trajectory')
        ax.axis('equal')

        # Creating obstacles in the plot
        [obst.plot(self.fig, offset=self.mrobot.rho) for obst in self.obst]

        # Initiate robot path in plot
        self.mrobot.plot(self.fig)

        # Initiate robot speed plot
        if speedPlot == True:
            self.mrobot.plotSpeeds(plt.subplots(2))

        # Creating robot path (and updating plot if IPLOT is true)
        ret = self.mrobot.gen_trajectory()

        plt.show(block=True)

        return ret

def parse_cmdline():
    # parsing command line eventual optmization method options
    lib = None
    method = None
    if len(sys.argv) > 1:
        lib = str(sys.argv[1])
        if len(sys.argv) > 2:
            method = str(sys.argv[2])
        else:
            method = None
    else:
        lib = 'scipy'

    return lib, method

###############################################################################
#                                   MAIN
###############################################################################
if __name__ == "__main__":

    # set logging level (TODO receve from command line)
    logging.basicConfig(level=logging.DEBUG)

    obstacles = [RoundObstacle([ 0.25,  2.50], 0.20),
                 RoundObstacle([ 2.30,  2.50], 0.50), 
                 RoundObstacle([ 1.25,  3.00], 0.10),
                 RoundObstacle([ 0.30,  1.00], 0.10),
                 RoundObstacle([-0.50,  1.50], 0.30),
                 RoundObstacle([ 0.70,  1.45], 0.25)]

    boundary = Boundary([-5.0,10.0], [-5.0,10.0])

    kine_model = UnicycleKineModel(
            [ 0.0,  0.0, np.pi/2], # q_initial
            [ 2.0,  5.0, np.pi/2], # q_final
            [ 0.0,  0.0],          # u_initial
            [ 0.0,  0.0],          # u_final
            [ 0.5,  5.0],          # u_max
            [ 1.6,  3.0])          # du_max

    robot = Robot(
            kine_model,
            obstacles,
            boundary,
            N_s=200,
#            t_init=0.0,
#            t_sup=1e10,
            rho=0.2)

    lib, method = parse_cmdline()

    robot.setOption('LIB', lib) # only has slsqp method
    robot.setOption('OPTMETHOD', method)
    robot.setOption('ACC', 1e-6)
#    robot.setOption('NKNOTS', 15)
#    robot.setOption('IPRINT', 2)
    robot.setOption('MAXIT', 100)

    world_sim = WorldSim(robot,obstacles,boundary)

    logging.info('Plannification summary {}'.format(world_sim.run(interacPlot=True, speedPlot=True)))

    if robot.k_mod.l > 2:
        f, axarr = plt.subplots(2)
        axarr[0].plot(robot.mtime, robot.linacc)
        axarr[1].plot(robot.mtime, robot.angacc)
        plt.show(block=True)

    logging.info('Elapsed Time for path generation: {}'.format(robot.elapsed_time))
