#!/usr/bin/python

import numpy as np
import numpy.linalg as LA
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.interpolate as si
import time
import itertools
import pyOpt
import multiprocessing as mpc
import sys
import logging
from scipy.optimize import fmin_slsqp
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

# Data manipulation:

def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments


# Interface to LineCollection:

def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
           
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
        
    z = np.asarray(z)
    
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    
    ax = plt.gca()
    ax.add_collection(lc)
    
    return lc
        
###############################################################################
# Obstacle
###############################################################################
class Obstacle(object):
    def __init__(self, position, dimension):
        self.pos = position
        self.dim = dimension

class RoundObstacle(Obstacle):
    def __init__(self, position, dimension):
        Obstacle.__init__(self, position, dimension)
        self.cp = np.asarray(self.pos)
        self.x = self.pos[0]
        self.y = self.pos[1]
        self.radius = self.dim

    def _plt_circle(self, color='k',linestyle='solid',isFill=False,offset=0.0):
        return plt.Circle(
                (self.x, self.y), # position
                self.radius+offset, # radius
                color=color,
                ls = linestyle,
                fill=isFill)

    def plot(self, fig, offset=0.0):
        ax = fig.gca()
        ax.add_artist(self._plt_circle())
        ax.add_artist(self._plt_circle(linestyle='dashed', offset=offset))

###############################################################################
# Boundary
###############################################################################
class Boundary(object):
    def __init__(self, x, y):
        self.x_min = x[0]
        self.x_max = x[1]
        self.y_min = y[0]
        self.y_max = y[1]

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
            u_max=[1.0,5.0],
            a_max=[2.0,10.0]):
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
        # Flat output
        self.z_init = self.phi0(self.q_init)
        self.z_final = self.phi0(self.q_final)
        self.l = 2 # number of need derivations

    def phi0(self, q):
        """ Returns z given q
        """
        return q[0:2,0]

    def phi1(self, z):
        """ Returns [x, y, theta]^T given [z dz ddz] (only z and dz are used)
        """
        if z.shape >= (self.u_dim, self.l+1):
            return np.matrix(np.append(z[:,0], \
                    np.asarray(
                    np.arctan2(z[1,1], z[0,1])))).T
        else:
            logging.warning('Bad z input. Returning zeros')
            return np.matrix('0.0; 0.0; 0.0')

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

#    def _unsigned_angle(self, angle):
#        return np.pi+angle if angle < 0.0 else angle

###############################################################################
# Robot
###############################################################################
class Robot(object):
    def __init__(
            self,
            eyed,
            kine_model,
            obstacles,
            phy_boundary,
            com_link,
            process_counter,
            sol,
            N_s=20,
            n_knots=6,
            t_init=0.0,
            t_sup=1e10,
            Tc=1.0,
            Tp=2.0,
            Td=2.0,
            rho=0.2,
            detec_rho=2.0,
            log_lock=None):

        self.eyed = eyed
        self.k_mod = kine_model
        self.obst = obstacles
        self.p_bound = phy_boundary
        self.N_s = N_s # no of samples for discretization of time
        self.n_knots = n_knots
        self.t_init = t_init
        self.t_sup = t_sup # superior limit of time
        self.Tc = Tc
        self.Tp = Tp
        self.Td = Td
        self.rho = rho
        self.d_rho = detec_rho

        self.log_lock = log_lock

        td_step = (self.Td-self.t_init)/(self.N_s-1)
        tp_step = (self.Tp-self.t_init)/(self.N_s-1)
        self.Tcd_idx = int(round(self.Tc/td_step))
        self.Tcp_idx = int(round(self.Tc/tp_step))

        self.set_option('maxit')
        self.set_option('acc')

        self.sol = sol

        # Declaring the planning process
        self.planning_process = mpc.Process(target=Robot._plan, args=(self,))

    def set_option(self, name, value=None):
        if name == 'maxit':
            self.maxit = 50 if value == None else value
        elif name == 'acc':
            self.acc = 1e-6 if value == None else value
        else:
            self._log('w', 'Unknown parameter '+name+', nothing will be set')
        return

    def _gen_knots(self, t_init, t_final):
        """ Generate b-spline knots given initial and final times
        """
        gk = lambda x:t_init + (x-(self.d-1.0))*(t_final-t_init)/self.n_knots
        knots = [t_init for _ in range(self.d)]
        knots.extend([gk(i) for i in range(self.d,self.d+self.n_knots-1)])
        knots.extend([t_final for _ in range(self.d)])
        return np.asarray(knots)

    def _comb_bsp(self, t, ctrl_pts, deriv_order):
        """ Combine base b-splines into a Bezier curve
        """
        tup = (
                self.knots, # knots
                ctrl_pts[:,0], # first dim ctrl pts
                self.d-1) # b-spline degree

        # interpolation
        z = si.splev(t, tup, der=deriv_order).reshape(len(t),1)

        for i in range(self.k_mod.u_dim)[1:]:
            tup = (
                    self.knots,
                    ctrl_pts[:,i],
                    self.d-1)
            z = np.append(
                    z,
                    si.splev(t, tup, der=deriv_order).reshape(len(t),1),
                    axis=1)
        return z

    def _log(self, logid, strg):
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
        elif logid == 'c':
            log_call = logging.critical
        else:
            log_call = logging.debug

        if self.log_lock != None:
            self.log_lock.acquire()
        log_call(strg)
        if self.log_lock != None:
            self.log_lock.release()
        return

    def _linspace_ctrl_pts(self, final_ctrl_pt):
        self.C[:,0] = np.array(np.linspace(self.last_z[0,0],\
                final_ctrl_pt[0,0], self.n_ctrlpts)).T
        self.C[:,1] = np.array(np.linspace(self.last_z[1,0],\
                final_ctrl_pt[1,0], self.n_ctrlpts)).T

    def _detected_obst_idxs(self):
        idx_list = []
        for idx in range(len(self.obst)):
            dist = LA.norm(self.obst[idx].cp - self.last_z.T)
            if dist < self.d_rho:
                idx_list += [idx]
        self.detected_obst_idxs = idx_list

    def _ls_sa_criterion(self, x):
        return (x[0])**2

    def _ls_sa_feqcons(self, x):
        dt_final = x[0]
        t_final = self.mtime[0]+dt_final
        C = x[1:].reshape(self.n_ctrlpts, self.k_mod.u_dim)

        self.knots = self._gen_knots(self.mtime[0], t_final)
        dztinit = self._comb_bsp([self.mtime[0]], C, 0).T
        for dev in range(1,self.k_mod.l+1):
            dztinit = np.append(dztinit,self._comb_bsp([self.mtime[0]], C, dev).T,axis=1)

        # get matrix [z dz ddz](t_final)
        dztfinal = self._comb_bsp([t_final], C, 0).T
        for dev in range(1,self.k_mod.l+1):
            dztfinal=np.append(dztfinal,self._comb_bsp([t_final], C, dev).T,axis=1)
    
        eq_cons = list(np.squeeze(np.array(self.k_mod.phi1(dztinit)-self.last_q)))+\
                list(np.squeeze(np.array(self.k_mod.phi1(dztfinal)-self.k_mod.q_final)))+\
                list(np.squeeze(np.array(self.k_mod.phi2(dztinit)-self.last_u)))+\
                list(np.squeeze(np.array(self.k_mod.phi2(dztfinal)-self.k_mod.u_final)))
        self.unsatisf_eq_values = [ec for ec in eq_cons if ec != 0]
        return np.asarray(eq_cons)

    def _ls_sa_fieqcons(self, x):
        dt_final = x[0]
        t_final = self.mtime[0]+dt_final
        C = x[1:].reshape(self.n_ctrlpts, self.k_mod.u_dim)
        
        self.knots = self._gen_knots(self.mtime[0], t_final)
    
        mtime = np.linspace(self.mtime[0], t_final, self.N_s)
    
        # get a list over time of the matrix [z dz ddz](t) t in [tk, tk+Tp]
        dz = self._comb_bsp(mtime[1:-1], C, 0).T
        for dev in range(1,self.k_mod.l+1):
            dz = np.append(dz,self._comb_bsp(mtime[1:-1], C, dev).T,axis=0)
    
        dztTp = map(lambda dzt:dzt.reshape(self.k_mod.l+1, self.k_mod.u_dim).T, dz.T)
    
        # get a list over time of command values u(t)
        utTp = map(self.k_mod.phi2, dztTp)
    
        # get a list over time of values q(t)
        qtTp = map(self.k_mod.phi1, dztTp)
        
        ## Obstacles constraints
        # N_s*nb_obst_detected
        obst_cons = []
        for m in self.detected_obst_idxs:
            obst_cons += [LA.norm(self.obst[m].cp-qt[0:2,0].T) \
              - (self.rho + self.obst[m].radius) for qt in qtTp]
    
        ## Max speed constraints
        # N_s*u_dim inequations
        max_speed_cons = list(itertools.chain.from_iterable(
            map(lambda ut:[self.k_mod.u_max[i,0]-abs(ut[i,0])\
            for i in range(self.k_mod.u_dim)],utTp)))
    
        # Create final array
        ieq_cons = obst_cons + max_speed_cons
        # Count how many inequations are not respected
        self.unsatisf_ieq_values = [ieq for ieq in ieq_cons if ieq < 0]
        return np.asarray(ieq_cons)
        
    def _sa_criterion(self, x):
        C = x.reshape(self.n_ctrlpts, self.k_mod.u_dim)
        
        dz = self._comb_bsp([self.mtime[-1]], C, 0).T
        for dev in range(1,self.k_mod.l+1):
            dz = np.append(dz,self._comb_bsp([self.mtime[-1]], C, dev).T,axis=1)
        qTp = self.k_mod.phi1(dz)
        return LA.norm(qTp - self.k_mod.q_final)**2

    def _sa_feqcons(self, x):
        C = x.reshape(self.n_ctrlpts, self.k_mod.u_dim)

        dztinit = self._comb_bsp([self.mtime[0]], C, 0).T
        for dev in range(1,self.k_mod.l+1):
            dztinit = np.append(dztinit,self._comb_bsp([self.mtime[0]], C, dev).T,axis=1)
    
        # dimension: q_dim + u_dim (=5 equations)
        eq_cons = list(np.squeeze(np.array(self.k_mod.phi1(dztinit)-self.last_q)))+\
               list(np.squeeze(np.array(self.k_mod.phi2(dztinit)-self.last_u)))
    
        # Count how many equations are not respected
        unsatisf_list = [eq for eq in eq_cons if eq != 0]
        self.unsatisf_eq_values = unsatisf_list
    
        return np.asarray(eq_cons)

    def _sa_fieqcons(self, x):
        C = x.reshape(self.n_ctrlpts, self.k_mod.u_dim)
    
        # get a list over time of the matrix [z dz ddz](t) t in [tk, tk+Tp]
        dz = self._comb_bsp(self.mtime[1:], C, 0).T
        for dev in range(1,self.k_mod.l+1):
            dz = np.append(dz,self._comb_bsp(self.mtime[1:], C, dev).T,axis=0)
    
        dztTp = map(lambda dzt:dzt.reshape(self.k_mod.l+1, self.k_mod.u_dim).T, dz.T)
    
        # get a list over time of command values u(t)
        utTp = map(self.k_mod.phi2, dztTp)
    
        # get a list over time of values q(t)
        qtTp = map(self.k_mod.phi1, dztTp)
    
        ## Obstacles constraints
        # N_s*nb_obst_detected
        obst_cons = []
        for m in self.detected_obst_idxs:
            obst_cons += [LA.norm(self.obst[m].cp-qt[0:2,0].T) \
              - (self.rho + self.obst[m].radius) for qt in qtTp]
    
        ## Max speed constraints
        # N_s*u_dim inequations
        max_speed_cons = list(itertools.chain.from_iterable(
            map(lambda ut:[self.k_mod.u_max[i,0]-abs(ut[i,0])\
            for i in range(self.k_mod.u_dim)],utTp)))
    
        # Create final array
        ieq_cons = obst_cons + max_speed_cons
    
        # Count how many inequations are not respected
        unsatisf_list = [ieq for ieq in ieq_cons if ieq < 0]
        self.unsatisf_ieq_values = unsatisf_list

        # return arrray where each element is an inequation constraint
        return np.asarray(ieq_cons)

    def _ls_co_criterion(self, x):
        return (x[0]+self.mtime[0])**2

    def _ls_co_feqcons(self, x):
        
        return

    def _ls_co_fieqcons(self, x):
        return

    def _co_criterion(self, x):
        return

    def _co_feqcons(self, x):
        return

    def _co_fieqcons(self, x):
        return

    def _scipy_callback(self):
        return

    def _solve_opt_pbl(self):

        if not self.final_step:
            if self.std_alone:
                p_criterion = self._sa_criterion
                p_eqcons = self._sa_feqcons
                p_ieqcons = self._sa_fieqcons
            else:
                p_criterion = self._co_criterion
                p_eqcons = self._co_feqcons
                p_ieqcons = self._co_fieqcons

            init_guess = self.C.reshape(self.n_ctrlpts*self.k_mod.u_dim)

        else:
            if self.std_alone:
                p_criterion = self._ls_sa_criterion
                p_eqcons = self._ls_sa_feqcons
                p_ieqcons = self._ls_sa_fieqcons
            else:
                p_criterion = self._ls_co_criterion
                p_eqcons = self._ls_co_feqcons
                p_ieqcons = self._ls_co_fieqcons

            init_guess = np.append(np.asarray([self.est_dtime]),
                    self.C.reshape(self.n_ctrlpts*self.k_mod.u_dim))

        output = fmin_slsqp(
                p_criterion,
                init_guess,
                eqcons=(),
                f_eqcons=p_eqcons,
                ieqcons=(),
                f_ieqcons=p_ieqcons,
                iprint=0,
                iter=self.maxit,
                acc=self.acc,
                full_output=True)

            #imode = output[3]
            # TODO handle optimization exit mode
        if self.final_step:
            self.C = output[0][1:].reshape(self.n_ctrlpts, self.k_mod.u_dim)
            self.dt_final = output[0][0]
            self.t_final = self.mtime[0] + self.dt_final
        else:
            self.C = output[0].reshape(self.n_ctrlpts, self.k_mod.u_dim)
#            #imode = output[3]
#            # TODO handle optimization exit mode

        self.n_it = output[2]
        self.exit_mode = output[4]
        return

    def _plan_section(self):

        # update obstacles zone
        self._detected_obst_idxs()

#        # first guess for ctrl pts
        if not self.final_step:
            direc = self.final_z - self.last_z
            direc = direc/LA.norm(direc)
            last_ctrl_pt = self.last_z+self.D*direc
        else:
            last_ctrl_pt = self.final_z

        self._linspace_ctrl_pts(last_ctrl_pt)

        self.std_alone = True

        tic = time.time()
        self._solve_opt_pbl()
        toc = time.time()

        self._log('i','R{rid}@tkref={tk}: Time to solve stand alone optimisation '
                'problem: {t}'.format(rid=self.eyed,t=toc-tic,tk=self.mtime[0]))
        self._log('i','R{rid}@tkref={tk}: N of unsatisfied eq: {ne}'\
                .format(rid=self.eyed,t=toc-tic,tk=self.mtime[0],ne=len(self.unsatisf_eq_values)))
        self._log('i','R{rid}@tkref={tk}: N of unsatisfied ieq: {ne}'\
                .format(rid=self.eyed,t=toc-tic,tk=self.mtime[0],ne=len(self.unsatisf_ieq_values)))
        self._log('i','R{rid}@tkref={tk}: Summary: {summ} after {it} it.'\
                .format(rid=self.eyed,t=toc-tic,tk=self.mtime[0],summ=self.exit_mode,it=self.n_it))

        if self.final_step:
            self.knots = self._gen_knots(self.mtime[0], self.t_final)
            self.mtime = np.linspace(self.mtime[0], self.t_final, self.N_s)

        time_idx = None if self.final_step else self.Tcd_idx

        dz = self._comb_bsp(self.mtime[0:time_idx], self.C, 0).T
        for dev in range(1,self.k_mod.l+1):
            dz = np.append(dz,self._comb_bsp(
                    self.mtime[0:time_idx], self.C, dev).T,axis=0)

#        TODO Write on shared memory my intention. Process sync

#        self._compute_conflicts()
#
#        if self.conflicts != []:
        if False:
            # TODO save path for Td 
            self.std_alone = False

#            self._read_com_link()

            tic = time.time()
            self._solve_opt_pbl()
            toc = time.time()

            self._log('i','R{rid}@tkref={tk}: Time to solve optimisation probl'
                    'em: {t}'.format(rid=self.eyed,t=toc-tic,tk=self.mtime[0]))
            self._log('i','R{rid}@tkref={tk}: N of unsatisfied eq: {ne}'\
                    .format(rid=self.eyed,t=toc-tic,tk=self.mtime[0],ne=len(self.unsatisf_eq_values)))
            self._log('i','R{rid}@tkref={tk}: N of unsatisfied ieq: {ne}'\
                    .format(rid=self.eyed,t=toc-tic,tk=self.mtime[0],ne=len(self.unsatisf_ieq_values)))
            self._log('i','R{rid}@tkref={tk}: Summary: {summ} after {it} it.'\
                    .format(rid=self.eyed,t=toc-tic,tk=self.mtime[0],summ=self.exit_mode,it=self.n_it))

            if self.final_step:
                self.knots = self._gen_knots(self.mtime[0], self.t_final)
                self.mtime = np.linspace(self.mtime[0], self.t_final, self.N_s)

            time_idx = None if self.final_step else self.Tcp_idx
                
            dz = self._comb_bsp(self.mtime[0:time_idx], self.C, 0).T
            for dev in range(1,self.k_mod.l+1):
                dz = np.append(dz,self._comb_bsp(
                        self.mtime[0:time_idx], self.C, dev).T,axis=0)
#        else:
#            TODO Consider planning during Td the final plan
            
        # Storing
#        self._update()
        self.all_C += [self.C]
        
        self.all_dz += [dz]
        # TODO rejected path

        # Updating
        if not self.final_step:
            self.knots = self.knots + self.Tc
            self.mtime = [tk+self.Tc for tk in self.mtime]
            self.last_z = self.all_dz[-1][0:self.k_mod.u_dim,-1].reshape(
                    self.k_mod.u_dim,1)
            self.last_q = self.k_mod.phi1(self.all_dz[-1][:,-1].reshape(
                    self.k_mod.l+1, self.k_mod.u_dim).T)
            self.last_u = self.k_mod.phi2(self.all_dz[-1][:,-1].reshape(
                    self.k_mod.l+1, self.k_mod.u_dim).T)
        return

    def _init_planning(self):

        self.detected_obst_idxs = range(len(self.obst))

        self.last_q = self.k_mod.q_init
        self.last_u = self.k_mod.u_init
        self.last_z = self.k_mod.z_init
        self.final_z = self.k_mod.z_final

        self.D = self.Tp * self.k_mod.u_max[0,0]

        self.d = self.k_mod.l+2 # B-spline order (integer | d > l+1)
        self.n_ctrlpts = self.n_knots + self.d - 1 # nb of ctrl points

        self.C = np.zeros((self.n_ctrlpts,self.k_mod.u_dim))

        self.all_C = []
        self.all_dz = []

    def _plan(self):

        self._log('i', 'R{rid}: Init motion planning'.format(rid=self.eyed))

        self._init_planning()

        self.final_step = False

        self.knots = self._gen_knots(self.t_init, self.Td)
        self.mtime = np.linspace(self.t_init, self.Td, self.N_s)

        # while the remaining dist is greater than the max dist during Tp
        while LA.norm(self.last_z - self.final_z) > self.D:
            self._plan_section()

        self.final_step = True

        self.est_dtime = LA.norm(self.last_z - self.final_z)/self.k_mod.u_max[0,0]

        # TODO verify change if this is ok
        self.knots = self._gen_knots(self.mtime[0], self.mtime[0]+self.est_dtime)
        self.mtime = np.linspace(self.mtime[0], self.mtime[0]+self.est_dtime, self.N_s)

        self._plan_section()

        self.sol[self.eyed] = self.all_dz

        self._log('i','R{}: Finished motion planning'.format(self.eyed))

#        fig = plt.figure(self.eyed)
#        ax = fig.gca()
#        ax.set_xlabel('x(m)')
#        ax.set_ylabel('y(m)')
#        ax.set_title('Generated trajectory')
#        ax.axis('equal')
#
#        # Creating obstacles in the plot
#        [obst.plot(fig, offset=self.rho) for obst in self.obst]
#
#        path = self.all_dz[0][0:2,:]
#        for p in self.all_dz[1:]:
#            path = np.append(path, p[0:2,:], axis=1)
#        ax.plot(path[0,:], path[1,:])
#        plt.show()
        
        return

###############################################################################
# World
###############################################################################
class WorldSim(object):
    """ Where to instatiate obstacles, robots, bonderies
        initial and final conditions etc
    """
    def __init__(self, robots, obstacles, phy_boundary):
        self.robs = robots
        self.obsts = obstacles
        self.ph_bound = phy_boundary

    def run(self, interac_plot=False, speed_plot=False):

        [r.planning_process.start() for r in self.robs]
        [r.planning_process.join() for r in self.robs]

        # PLOT

        # Interactive plot
        plt.ion()

        fig = plt.figure()
        ax = fig.gca()
        ax.set_xlabel('x(m)')
        ax.set_ylabel('y(m)')
        ax.set_title('Generated trajectory')
        ax.axis('equal')

        # Creating obstacles in the plot
        [obst.plot(fig, offset=self.robs[0].rho) for obst in self.obsts]

        colors = [[1-i, i, 0.0] for i in np.linspace(0.0, 1.0,len(self.robs))]

        path = range(len(self.robs))
        plt_rob = range(len(self.robs))
        for i in range(len(self.robs)):
            path[i] = self.robs[0].sol[i][0][0:2,:]
            for p in self.robs[0].sol[i][1:]:
                path[i] = np.append(path[i], p[0:2,:], axis=1)

            plt_rob[i], = ax.plot(path[i][0,0], path[i][1,0], color=colors[i])

        counter = 1
        while True:
            end = 0
            for i in range(len(self.robs)):
#                print(path[i].shape)
                if counter < path[i].shape[1]:
                    plt_rob[i].set_xdata(path[i][0,0:counter+1])
                    plt_rob[i].set_ydata(path[i][1,0:counter+1])
                else:
                    end += 1
            if end == len(self.robs):
                break
            time.sleep(0.1)
            ax.relim()
            ax.autoscale_view(True,True,True)
            fig.canvas.draw()
            counter += 1
            
        plt.show(block=True)
        
        logging.info('All robots have finished')

        return

###############################################################################
# Script
###############################################################################

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
    x_range =np.linspace(boundary.x_min+max_radius,boundary.x_max-max_radius,N)
    y_range =np.linspace(boundary.y_min+max_radius,boundary.y_max-max_radius,N)
    
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
    method = None
    if len(sys.argv) > 1:
        method = str(sys.argv[1])
    else:
        method = 'slsqp'

    return scriptname, method


# MAIN ########################################################################

if __name__ == '__main__':

    n_obsts = 5
    n_robots = 3
    N_s = 20

    scriptname, method = parse_cmdline()

    if method != None:
        fname = scriptname[0:-3]+'_'+method+'.log'
    else:
        fname = scriptname[0:-3]+'.log'

#    logging.basicConfig(filename=fname,format='%(levelname)s:%(message)s',\
#            filemode='w',level=logging.DEBUG)
    logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.DEBUG)

    boundary = Boundary([-5.0,5.0], [-1.0,6.0])

#    obst_info = rand_round_obst(n_obsts, Boundary([-1.0,4.0],[0.7,4.0]))

    # these obst info
    obst_info = [([0.25, 2.5], 0.20),([ 2.30,  2.50], 0.50),
            ([ 1.25,  3.00], 0.10),([ 0.30,  1.00], 0.10),
            ([-0.50,  1.50], 0.30)]


    obstacles = [RoundObstacle(i[0], i[1]) for i in obst_info]


    kine_models = [UnicycleKineModel(
            [ float(i),  0.0, np.pi/2], # q_initial
            [ n_robots-i+1.0,  5.0, np.pi/2], # q_final
            [ 0.0,  0.0],          # u_initial
            [ 0.0,  0.0],          # u_final
            [ 1.0,  5.0])          # u_max
            for i in [j-n_robots/2 for j in range(n_robots)]]

    log_lock = mpc.Lock()
    pcc_lock = mpc.Lock()
    com_lock = mpc.Lock()

    process_counter = mpc.Value('I', 0, lock=pcc_lock) # unsigned int
    com_link = mpc.Array('d', N_s, lock=com_lock)
    manager = mpc.Manager()
    sol = manager.list(range(n_robots))
#    com_link = mpc.Array('d', max(robots, key=\
#            lambda rob:rob.N_s).N_s, lock=com_lock)

    robots = [Robot(
            i,
            kine_models[i],
            obstacles,
            boundary,
            com_link,
            process_counter,
            sol,
            N_s=N_s,
            n_knots=6) for i in range(n_robots)]

    [r.set_option('acc', 1e-6) for r in robots]
    [r.set_option('maxit', 50) for r in robots]

    world_sim = WorldSim(robots,obstacles,boundary)

    summary_info = world_sim.run(interac_plot=False, speed_plot=True)

