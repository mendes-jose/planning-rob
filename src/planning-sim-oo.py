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
import copy

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
                    np.matrix(
                    np.arctan2(z[1,1], z[0,1])), axis = 0)
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

    def _unsigned_angle(self, angle):
        return np.pi+angle if angle < 0.0 else angle

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

    def _improve_init_guess(self):

        mag = LA.norm(self.ctrl_pts[0]-self.ctrl_pts[1])
        self.ctrl_pts[1]+np.matrix([1,2])
        dx = mag*np.cos(self.k_mod.q_init[-1,0])
        dy = mag*np.sin(self.k_mod.q_init[-1,0])
        self.ctrl_pts[1] = self.ctrl_pts[0] + np.matrix([dx, dy])

        dx = mag*np.cos(self.k_mod.q_final[-1,0])
        dy = mag*np.sin(self.k_mod.q_final[-1,0])
        self.ctrl_pts[-2] = self.ctrl_pts[-1] - np.matrix([dx, dy])

        no_obst = len(self.obst)

        nrp = [] # narrow paths
        for i in range(no_obst):
            for j in range(i+1, no_obst):
                if (self.obst[i].x()-self.obst[j].x())**2 + \
                        (self.obst[i].y()-self.obst[j].y())**2 < \
                        (self.obst[i].radius()+ \
                        self.obst[j].radius()+2*self.rho)**2:
                    nrp += [(i, j)]

        if nrp == []:
            return

        # Now that we know that there are narrow paths let's check if
        # they represent a problem:

        c = 3 #TODO find a appropriate value
        epsilon = (c*self.k_mod.u_max[0]*(self.mtime[1]-self.mtime[0]))[0,0]

        # interpolate control points using combination of b-splines
        z = [self._comb_bsp(tk, self.ctrl_pts, 0) for tk in self.mtime]

        maxit = len(nrp)*2
        logging.debug('Max iter of improving algo {}'.format(maxit))

        mov_hist = []
        while maxit > 0:
            done = 0
            for i in nrp:
                logging.debug('\n\n/////////////////////NRP////////////////////: {}'.format(i))
    
                dcent2 = (self.obst[i[1]].y()- self.obst[i[0]].y())**2+\
                        (self.obst[i[1]].x()- self.obst[i[0]].x())**2
    
                thir_n_fourth = self.obst[i[1]].x()*self.obst[i[0]].y()-\
                        self.obst[i[1]].y()*self.obst[i[0]].x()
    
                # compute all distances from the Ns points on the path to the line passing thru the 2 centers
                dists = map(lambda x:abs(x[0,0]*(self.obst[i[1]].y()- \
                        self.obst[i[0]].y())-x[0,1]* \
                        (self.obst[i[1]].x()-self.obst[i[0]].x()) +\
                        thir_n_fourth)**2/dcent2, z)
    
                sorteddists = copy.deepcopy(dists)
    
                sorteddists.sort()
    
                # save index of the closest
                j = dists.index(sorteddists[0])
    
                # if distance bigger than epsilon the curve does not cross the line passing thru the 2 centers
                if dists[j] > epsilon**2:
                    done += 1
                    continue
                
                # check if there are multiples intersection points
                closest_dists = [x for x in dists if x <= epsilon**2]
                idx_clst_d = [dists.index(value) for value in closest_dists]
                j = [idx_clst_d[0]]
                for idx in range(1, len(idx_clst_d)):
                    if abs(idx_clst_d[idx-1] - idx_clst_d[idx]) > 1:
                        j += [idx_clst_d[idx]]

                logging.debug('No of intersections: {}'.format(len(j)))
                logging.debug(idx_clst_d)
                logging.debug(j)

                # check if the found intersection point is between the two obstacles' centers
                isecc1_list = []
                isecc2_list = []
                for ij in j:
                    isecc1_list += [(z[ij][0,0]-self.obst[i[0]].x())**2+\
                            (z[ij][0,1]-self.obst[i[0]].y())**2]
                    isecc2_list += [(z[ij][0,0]-self.obst[i[1]].x())**2+\
                            (z[ij][0,1]-self.obst[i[1]].y())**2]
                internal_intersecc1 = []
                internal_intersecc2 = []
                for ij in range(len(isecc1_list)):
                    if isecc1_list[ij] <= dcent2 and isecc2_list[ij] <= dcent2:
                        internal_intersecc1 += [isecc1_list[ij]]
                        internal_intersecc2 += [isecc2_list[ij]]

                if internal_intersecc1 == []:
                    done += 1
                    continue
                elif len(internal_intersecc1) == 1:
                    isecc1 = isecc1_list[0]
                    isecc2 = isecc2_list[0]
                    # TODO Change several ctrl points using something like a exponential curve
                    logging.debug('Attempt to improve first guess by moving control points')
    
                    # choose the closest obstacle to be the one to avoid
                    if (self.obst[i[0]] in mov_hist) == (self.obst[i[1]] in mov_hist): # not xor
                        if isecc1 < isecc2:
                            to_be_avoided = self.obst[i[0]]
                            other = self.obst[i[1]]
                            if not self.obst[i[0]] in mov_hist:
                                mov_hist += [self.obst[i[0]]]
                            logging.debug('1.appeded to hist {}'.format(self.obst[i[0]]))
                        else:
                            to_be_avoided = self.obst[i[1]]
                            other = self.obst[i[0]]
                            if not self.obst[i[1]] in mov_hist:
                                mov_hist += [self.obst[i[1]]]
                            logging.debug('2.appeded to hist {}'.format(self.obst[i[1]]))
                    elif not self.obst[i[0]] in mov_hist:
                        to_be_avoided = self.obst[i[0]]
                        other = self.obst[i[1]]
                        mov_hist += [self.obst[i[0]]]
                        logging.debug('3.appeded to hist {}'.format(self.obst[i[0]]))
                    else:
                        to_be_avoided = self.obst[i[1]]
                        other = self.obst[i[0]]
                        mov_hist += [self.obst[i[1]]]
                        logging.debug('4.appeded to hist {}'.format(self.obst[i[1]]))

                    logging.debug('Its center pos {}'.format(to_be_avoided.cp))

                    # find which 2 control points are the closest to the obst. to be avoided
                    distcp = map(lambda cp:(cp[0,0]-to_be_avoided.x())**2 + \
                            (cp[0,1]-to_be_avoided.y())**2, self.ctrl_pts)

                    distz = map(lambda az:(az[0,0]-to_be_avoided.x())**2 + \
                            (az[0,1]-to_be_avoided.y())**2, z)

                    sorteddistcp = copy.deepcopy(distcp)
                    sorteddistz = copy.deepcopy(distz)
    
                    sorteddistcp.sort()
                    sorteddistz.sort()
    
                    ctrlpt_index1 = distcp.index(sorteddistcp[0])
                    ctrlpt_index2 = distcp.index(sorteddistcp[1])
                    ctrlpt_index3 = distcp.index(sorteddistcp[2])
    
                    z_index1 = distz.index(sorteddistz[0])
                    z_index2 = distz.index(sorteddistz[1])
                    closest_z = z[z_index1]
                    z_aux = z[z_index2]

                    ctrl_pt_2b_moved = self.ctrl_pts[ctrlpt_index1]
    
                    path = closest_z - z_aux
                    path = path/LA.norm(path)
    
                    dp = ((to_be_avoided.cp-z_aux)*path.T)[0,0]
    
                    ctrlaux2x = dp*path
    
                    x2to_be_avoided = (to_be_avoided.cp-z_aux) - ctrlaux2x
    
                    radius_ortho = (self.rho+to_be_avoided.radius())*x2to_be_avoided/LA.norm(x2to_be_avoided)
    
    #                self.ctrl_pts[ctrlpt_index1] = np.matrix('0.02, 1.0') # 0.02, 1.0
                    no_ctrl_to_be_moved = int(
                            (2*to_be_avoided.radius()+2*self.rho)/\
                            LA.norm(self.ctrl_pts[ctrlpt_index1]-\
                            self.ctrl_pts[ctrlpt_index2]))
                    if no_ctrl_to_be_moved < 1:
                        no_ctrl_to_be_moved = 1

                    logging.debug('+++++++++++ no ctrl tb moved ++++++++++++ {}'.format(no_ctrl_to_be_moved))

                    pltmoved = []
                    for k in range(no_ctrl_to_be_moved):
                        idx = distcp.index(sorteddistcp[k])
                        self.ctrl_pts[idx] += x2to_be_avoided+\
                                radius_ortho*(no_ctrl_to_be_moved-k)/no_ctrl_to_be_moved
                        pltmoved += [self.ctrl_pts[idx,0], self.ctrl_pts[idx,1], 'o']

                    # interpolate control points using combination of b-splines
                    z = [self._comb_bsp(tk,self.ctrl_pts,0) for tk in self.mtime]
    
                    # Recalculate t_final approx.
                    self.t_final = sum(LA.norm(z[ind-1]-z[ind])/ \
                            self.k_mod.u_max[0,0] \
                            for ind in range(1, len(z)))

                    logging.debug('New t_final: {}'.format(self.t_final))

                    # Generate initial b-spline knots
                    self.knots = self._gen_knots(self.t_final)
    
                    # creating time
                    self.mtime = np.linspace(self.t_init, self.t_final, self.N_s)
                    f = plt.figure()
                    pltarg = []
                    pltarg += [self.ctrl_pts[:,0], self.ctrl_pts[:,1], '*']
                    pltarg += pltmoved
                    pltarg += [to_be_avoided.x(), to_be_avoided.y(), 'k.',
                            other.x(), other.y(), 'b.',
                            map(lambda x:x[0,0], z), map(lambda x:x[0,1], z), 'g-']
                    pltarg += [[self.k_mod.q_init[0,0], self.k_mod.q_final[0,0]], [self.k_mod.q_init[1,0], self.k_mod.q_final[1,0]], 'y--']
                    plt.plot(*pltarg)
                    ax = f.gca()
                    for o in self.obst:
                        o.plot(f, self.rho)
                    ax.axis('equal')
                #endif
            #endfor
            if done == len(nrp):
                logging.debug('DONE')    
                break
            maxit -= 1
            logging.debug('Max it: {}'.format(maxit))
        #endfor
#        plt.show(block=True)
    #endfunc

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

        # if need be:
        self._improve_init_guess()

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
        self.x_init = np.append(self.t_final, np.asarray(self.ctrl_pts)).tolist()

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

            if self.optIprint != None:
                iprint_option = self.optIprint
            else:
                iprint_option = 1

            x = fmin_slsqp(self._criteria,
                    self.x_init,
                    eqcons=(),
                    f_eqcons=self._feqcons,
                    ieqcons=(),
                    f_ieqcons=self._fieqcons,
                    iter=self.optMaxIt,
                    acc=self.optAcc,
                    iprint=iprint_option,
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
#                0.01,
                self.n_ctrlpts)).T
        self.ctrl_pts[:,1]=np.matrix(np.linspace( # linsapce in y
                self.k_mod.q_init[1,0],
                self.k_mod.q_final[1,0],
#                -0.01,
                self.n_ctrlpts)).T

        # Generate initial b-spline knots
        self.knots = self._gen_knots(self.t_final)

        # creating time
        self.mtime = np.linspace(self.t_init, self.t_final, self.N_s)

        # if need be:
        self._improve_init_guess()

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
        x_init = np.append(self.t_final, np.asarray(self.ctrl_pts)).tolist()

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
#            solver.setOption('epsfeas', 9e-1)
#            solver.setOption('epsopt', 9e-1)
            solver.setOption('iprint', 10)
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
            self.elapsed_time = time.time()
            ret = self._gen_pyopt()
            logging.debug(time.time() - self.elapsed_time)
            self.elapsed_time = time.time() - self.elapsed_time
            self._update_opt_plot()
            
        # Global optimal solution using scipy (knowledge of the whole map)
        elif self.lib == 'scipy':
            self._init_scipy()
            self.elapsed_time = time.time()
            ret = self._gen_scipy()
            logging.debug(time.time() - self.elapsed_time)
            self.elapsed_time = time.time() - self.elapsed_time
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
#    def __init__(self, position, dimension):
#        self = Obstacle(position, dimension)
#        self.cp = np.matrix(position)

    def __init__(self, position, dimension):
        Obstacle.__init__(self, position, dimension)
        self.cp = np.matrix(self.pos)

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

        logging.info('Fineshed planning')

        plt.show(block=True)

        return ret

def _isobstok(obsts, c, r):
    if len(obsts) > 0:
        for obst in obsts:
            if (c[0]-obst[0][0])**2 + (c[1]-obst[0][1])**2 < (r+obst[1])**2:
                return False
    return True

def rand_round_obst(no, boundary):

    N = 1/0.0001
    min_radius = 0.15
    max_radius = 0.6
    radius_range = np.linspace(min_radius,max_radius,N)
    x_range = np.linspace(boundary.x_min+max_radius,boundary.x_max-max_radius,N)
    y_range = np.linspace(boundary.y_min+max_radius,boundary.y_max-max_radius,N)
    
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

def parse_cmdline():
    # parsing command line eventual optmization method options
    scriptname = sys.argv[0]
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

    return scriptname, lib, method

###############################################################################
#                                   MAIN
###############################################################################
if __name__ == "__main__":

    scriptname, lib, method = parse_cmdline()

    # set logging level (TODO receve from command line)
#    logging.basicConfig(level=logging.DEBUG)
    fname = scriptname[0:-3]+'_'+lib+'_'+method+'.log'
    logging.basicConfig(filename=fname,format='%(levelname)s:%(message)s',filemode='w',level=logging.DEBUG)

    boundary = Boundary([-5.0,5.0], [-1.0,6.0])

    n_obsts = 5

    obst_info = rand_round_obst(n_obsts, Boundary([-1.0,4.0],[0.7,4.0]))
    obst_info = [([1.6757675767576758, 1.9096909690969095], 0.27628262826282624),([0.70021002100210017, 1.8473147314731473], 0.29225922592259224),([1.8783278327832784, 2.8722172217221722], 0.44252925292529255),([3.1392939293929389, 1.3825382538253823], 0.38249324932493245),([0.70933093309330919, 2.6199819981998198], 0.43951395139513949)]

#    obst_info = [([1.776097609760976, 3.2933093309330932], 0.44752475247524748), ([0.25746574657465737, 2.1951095109510952], 0.32043204320432039), ([-0.37567756775677569, 3.3804680468046802], 0.23195319531953196), ([0.72035203520352031, 2.9929792979297929], 0.49702970297029703), ([2.7824382438243824, 2.8625562556255622], 0.55828082808280821)]

#    obst_info = [([-0.1077507750775078, 3.2611761176117611], 0.46417641764176421), ([2.2994099409940993, 1.7433543354335432], 0.17848784878487847), ([1.3154915491549155, 2.4402040204020401], 0.59635463546354628), ([0.13053305330533049, 2.0588058805880585], 0.492979297929793), ([1.8213221322132211, 1.7185718571857185], 0.28069306930693072)]

#    obst_info = [([0.25404540454045399, 3.2559255925592558], 0.59972997299729969), ([2.1686768676867687, 2.3488448844884484], 0.57542754275427543), ([3.0028602860286027, 3.3941194119411939], 0.59135913591359135), ([1.0164016401640161, 2.3614461446144612], 0.23717371737173715), ([1.48004800480048, 1.4468046804680466], 0.52038703870387037)]
    logging.debug(obst_info)
    obstacles = []
    for i in obst_info:
        obstacles += [RoundObstacle(i[0], i[1])]

#    obstacles = [RoundObstacle([ 0.25,  2.50], 0.20),
#                 RoundObstacle([ 2.30,  2.50], 0.50), 
#                 RoundObstacle([ 1.25,  3.00], 0.10),
#                 RoundObstacle([ 0.30,  1.00], 0.10),
#                 RoundObstacle([-0.50,  1.50], 0.30),
#                 RoundObstacle([ 0.70,  1.45], 0.25)]
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
            N_s=100,
#            t_init=0.0,
#            t_sup=1e10,
            rho=0.2)


    robot.setOption('LIB', lib) # only has slsqp method
    robot.setOption('OPTMETHOD', method)
    robot.setOption('ACC', 5e-2)
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
