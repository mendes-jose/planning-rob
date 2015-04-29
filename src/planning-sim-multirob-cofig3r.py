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
import os
import logging
from scipy.optimize import fmin_slsqp

###############################################################################
# Obstacle
###############################################################################
class Obstacle(object):
    def __init__(self, position, dimension):
        self.pos = position
        self.dim = dimension

    def dist(self, pos, offset=0.0):
        return LA.norm(np.array([pos[0],pos[1]])-np.array([self.pos[0],self.pos[1]]))-offset

    def detection_dist(self, pos):
        return LA.norm(np.array([self.pos[0], self.pos[1]])-pos)

###############################################################################
# RoundObstacle
###############################################################################
class RoundObstacle(Obstacle):
    def __init__(self, position, dimension):
        Obstacle.__init__(self, position, dimension)
        self.cp = np.asarray(self.pos)
        self.x = self.pos[0]
        self.y = self.pos[1]
        self.radius = self.dim

    def _plt_circle(self, color='k',linestyle='solid',isFill=False,alpha=1.0,offset=0.0):
        return plt.Circle(
                (self.x, self.y), # position
                self.radius+offset, # radius
                color=color,
                ls = linestyle,
                fill=isFill,
                alpha=alpha)

    def plot(self, fig, offset=0.0):
        ax = fig.gca()
        ax.add_artist(self._plt_circle(isFill=True,alpha=0.3))
        ax.add_artist(self._plt_circle(linestyle='dashed', offset=offset))

    def dist(self, pos, offset=0.0):
        return Obstacle.dist(self, pos, offset+self.radius)

###############################################################################
# PolygonObstacle
###############################################################################
class PolygonObstacle(Obstacle):
    def __init__(self, vertices):
        
        # compute fat(card(vertices)) dists
        dists = []
        for i in range(vertices.shape[0]):
            for j in range(i+1,vertices.shape[0]):
                dists += [(LA.norm(vertices[i,]-vertices[j,]),i,j)]
        # find max
        max_d, iv1, iv2 = max(dists, key=lambda x:x[0])
        # use max/2 as dimension
        dimension = max_d/2.0
        # use (v_a + v_b) / 2 as position
        position = (vertices[iv1,] + vertices[iv2,])*0.5

        Obstacle.__init__(self, position, dimension)

        self.cp = np.asarray(self.pos)
        self.x = self.pos[0]
        self.y = self.pos[1]
        self.radius = self.dim
        self._orig_vertices = vertices
        # init self.vertices
        self.vertices = np.zeros((vertices.shape[0]*2,vertices.shape[1]))
        # create linear equations for each vertices
        self.lines_list = []
        for v,vb,va in zip(
                self._orig_vertices,
                np.roll(self._orig_vertices,1,axis=0),
                np.roll(self._orig_vertices,-1,axis=0)):
            vec_b = np.array([+v[1]-vb[1], -v[0]+vb[0]])
            vec_a = np.array([-v[1]+va[1], v[0]-va[0]])
            self.lines_list += [[(vec_b[1],-vec_b[0],vec_b[0]*v[1]-v[0]*vec_b[1]),
                    (vec_a[1],-vec_a[0],vec_a[0]*v[1]-v[0]*vec_a[1]),
                    (va[1]-v[1],v[0]-va[0],va[0]*v[1]-v[0]*va[1])]]
        return

    def _create_aug_vertices(self, offset):
        new_vertices_list = []
        for v,vb,va in zip(
                self._orig_vertices,
                np.roll(self._orig_vertices,1,axis=0),
                np.roll(self._orig_vertices,-1,axis=0)):
            vec_b = np.array([+v[1]-vb[1], -v[0]+vb[0]])
            unit_vec_b = vec_b/LA.norm(vec_b)
            vec_a = np.array([-v[1]+va[1], v[0]-va[0]])
            unit_vec_a = vec_a/LA.norm(vec_a)
            
            new_vertices_list += [offset*unit_vec_b+v, offset*unit_vec_a+v]
        self.vertices = np.array(new_vertices_list)
        return

    def plot(self, fig, offset=0.0):
        self._create_aug_vertices(offset)
        ax = fig.gca()
        [ax.add_artist(plt.Circle(
                (v[0],v[1]),
                offset,
                color='k',
                ls='dashed',
                fill=False)) for v in self._orig_vertices]
        
        ax.add_artist(plt.Polygon(self._orig_vertices, color='k',ls='solid',fill=True,alpha=0.3))
        ax.add_artist(plt.Polygon(self.vertices, color='k',ls='dashed',fill=False))
        ax.plot(self.x, self.y,'k*')
        return

    def dist(self, pos, offset=0.0):
        self._create_aug_vertices(offset)
        signed_dists = []
        for ls in self.lines_list:
            s_dist = []
            for l in ls:
                s_dist += [(l[0]*pos[0] + l[1]*pos[1] + l[2])/np.sqrt(l[0]**2+l[1]**2)]
            signed_dists += [[s_dist[0], s_dist[1], s_dist[2]]]
#        print signed_dists
#        dist = -1000000.0
        for idx in range(self._orig_vertices.shape[0]):
#            print signed_dists[idx][0]
#            print 'sd 0', signed_dists[idx][0]
#            print 'sd 1',signed_dists[idx][1]
            if (signed_dists[idx][0]<=0.0 and \
                    signed_dists[idx][1]>0.0) \
                    == True:
#                print 'vertex', idx
                return LA.norm(pos-self._orig_vertices[idx,])-offset
                
            elif (signed_dists[idx][2]>0.0 and \
                    signed_dists[idx][1]<=0.0 and \
                    signed_dists[(idx+1)%self._orig_vertices.shape[0]][0]>0.0) \
                    == True:
#                print 'edge', idx, (idx+1)%self._orig_vertices.shape[0]
                return abs(signed_dists[idx][2])-offset
        # if it gets here is probably because the pos is inside the obstacle
        is_inside = True
        for idx in range(self._orig_vertices.shape[0]):
            is_inside = is_inside and signed_dists[idx][2]<=0.0
        if is_inside == True:
#            print 'INSIDE THE OBSTACLE!!!'
#            return -100.0
#            return LA.norm(self.cp - self.)
#            print pos
#            time.sleep(6)
            return -min([abs(signed_dists[idx][2]) for idx in range(self._orig_vertices.shape[0])])-offset
        else:
            print 'I\'m out of ideas about what happend. WhereTF is the path going?'    
            raw_input()
#        print pos, signed_dists, idx
        return
#
#    def detection_dist(self, pos):
#        return LA.norm(self.cp-pos)
           
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
# Communication Msg
###############################################################################
class RobotMsg(object):
    def __init__(self, dp, ip_x, ip_y, lz):
        self.done_planning = dp
        self.intended_path_x = ip_x
        self.intended_path_y = ip_y
        self.last_z = lz
        return

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
            tc_syncer,
            tc_syncer_cond,
            conflict_syncer,        # array used for sync between robots having conflicts
            conflict_syncer_conds,  # array used for sync between robots having conflicts
            com_link,
            sol,
            robots_time,
            robots_comp_time,
            neigh,                  # neighbors to whom this robot shall talk (used for conflict only, not communic)
            N_s=20,
            n_knots=6,
            t_init=0.0,
            t_sup=1e10,
            Tc=1.0,
            Tp=3.0,
            Td=3.0,
            rho=0.2,
            detec_rho=13.0,
            com_range=15.0,
            def_epsilon=0.5,
            safe_epsilon=0.1,
            log_lock=None):

        self.eyed = eyed
        self.k_mod = kine_model
        self.obst = obstacles
        self.p_bound = phy_boundary
        self.tc_syncer = tc_syncer
        self.tc_syncer_cond = tc_syncer_cond
        self.conflict_syncer = conflict_syncer
        self.conflict_syncer_conds = conflict_syncer_conds
        self.com_link = com_link
        self.sol = sol
        self.rtime = robots_time
        self.ctime = robots_comp_time
        self.N_s = N_s # no of samples for discretization of time
        self.n_knots = n_knots
        self.t_init = t_init
        self.t_sup = t_sup # superior limit of time
        self.Tc = Tc
        self.Tp = Tp
        self.Td = Td
        self.rho = rho
        self.d_rho = detec_rho
        self.com_range = com_range
        self.def_epsilon = def_epsilon
        self.safe_epsilon = safe_epsilon
        self.log_lock = log_lock

        # get number of robots      
        self.n_robots = len(conflict_syncer)

        # index for sliding windows
        td_step = (self.Td-self.t_init)/(self.N_s-1)
        tp_step = (self.Tp-self.t_init)/(self.N_s-1)
        self.Tcd_idx = int(round(self.Tc/td_step))
        self.Tcp_idx = int(round(self.Tc/tp_step))
        self.Tcd = self.Tcd_idx*td_step
        self.Tcp = self.Tcp_idx*tp_step

        # optimization parameters
        self.set_option('maxit')
        self.set_option('acc')

        # init planning
        self.detected_obst_idxs = range(len(self.obst))

        self.last_q = self.k_mod.q_init
        self.last_u = self.k_mod.u_init
        self.last_z = self.k_mod.z_init
        self.final_z = self.k_mod.z_final

        self.D = self.Tp * self.k_mod.u_max[0,0]

        self.d = self.k_mod.l+2 # B-spline order (integer | d > l+1)
        self.n_ctrlpts = self.n_knots + self.d - 1 # nb of ctrl points

        self.C = np.zeros((self.n_ctrlpts,self.k_mod.u_dim))

        self.all_dz = []
        self.all_times = []
        self.all_comp_times = []

        # Declaring the planning process
        self.planning_process = mpc.Process(target=Robot._plan, args=(self,))

    def set_option(self, name, value=None):
        if name == 'maxit':
            self.maxit = 100 if value == None else value
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
        self.C[0:self.n_ctrlpts,0] = np.array(np.linspace(self.last_z[0,0],\
                final_ctrl_pt[0,0], self.n_ctrlpts)).T
        self.C[0:self.n_ctrlpts,1] = np.array(np.linspace(self.last_z[1,0],\
                final_ctrl_pt[1,0], self.n_ctrlpts)).T

    def _detected_obst_idxs(self):
        idx_list = []
        for idx in range(len(self.obst)):
            dist = self.obst[idx].detection_dist(np.squeeze(np.asarray(self.last_z.T)))
            if dist < self.d_rho:
                idx_list += [idx]
        self.detected_obst_idxs = idx_list

    def _ls_sa_criterion(self, x):
        # Minimize the total time:
        # * since there is no constraints about the time it self this would be
        # the same as minimizing only x[0]. However, for numeric reasons we
        # keep the cost far from values too small (~0) and too big (>1e6)
        return (x[0]+self.mtime[0])**2

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
            obst_cons += [self.obst[m].dist(np.squeeze(np.asarray(qt[0:2,0].T)), self.rho) for qt in qtTp]
    
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
        # Minimize the remaining distance to reach the final state:
        # * since there is no constraints about the time it self this would be
        # the same as minimizing only x[0]. However, for numeric reasons we
        # keep the cost far from values too small (~0) and too big (>1e6)
        C = x.reshape(self.n_ctrlpts, self.k_mod.u_dim)
        
        dz = self._comb_bsp([self.mtime[-1]], C, 0).T
        for dev in range(1,self.k_mod.l+1):
            dz = np.append(dz,self._comb_bsp([self.mtime[-1]], C, dev).T,axis=1)
        qTp = self.k_mod.phi1(dz)
        cost = LA.norm(qTp[0:-1,:] - self.k_mod.q_final[0:-1,:])**2
#        cost = LA.norm(qTp - self.k_mod.q_final)
        # TODO
        if cost > 1e5:
            self._log('d', 'R{}: Big problem {}'.format(self.eyed, cost))
        return cost

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
            obst_cons += [self.obst[m].dist(np.squeeze(np.asarray(qt[0:2,0].T)), self.rho) for qt in qtTp]
    
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
        # Minimize the total time:
        # * since there is no constraints about the time it self this would be
        # the same as minimizing only x[0]. However, for numeric reasons we
        # keep the cost far from values too small (~0) and too big (>1e6)
        return 0.1*(x[0]+self.mtime[0])**2

    def _ls_co_feqcons(self, x):
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
        
    def _ls_co_fieqcons(self, x):
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
            obst_cons += [self.obst[m].dist(np.squeeze(np.asarray(qt[0:2,0].T)), self.rho) for qt in qtTp]
    
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
        
    def _co_criterion(self, x):
        # Minimize the remaining distance to reach the final state:
        # * since there is no constraints about the time it self this would be
        # the same as minimizing only x[0]. However, for numeric reasons we
        # keep the cost far from values too small (~0) and too big (>1e6)
        C = x.reshape(self.n_ctrlpts, self.k_mod.u_dim)
        
        dz = self._comb_bsp([self.mtime[-1]], C, 0).T
        for dev in range(1,self.k_mod.l+1):
            dz = np.append(dz,self._comb_bsp([self.mtime[-1]], C, dev).T,axis=1)
        qTp = self.k_mod.phi1(dz)
        cost = LA.norm(qTp[0:-1,:] - self.k_mod.q_final[0:-1,:])**2
#        cost = LA.norm(qTp - self.k_mod.q_final)
        # TODO
        if cost > 1e5:
            self._log('d', 'R{}: Big problem {}'.format(self.eyed, cost))
        return cost

    def _co_feqcons(self, x):
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

    def _co_fieqcons(self, x):
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
            obst_cons += [self.obst[m].dist(np.squeeze(np.asarray(qt[0:2,0].T)), self.rho) for qt in qtTp]

        ## Max speed constraints
        # N_s*u_dim inequations
        max_speed_cons = list(itertools.chain.from_iterable(
            map(lambda ut:[self.k_mod.u_max[i,0]-abs(ut[i,0])\
            for i in range(self.k_mod.u_dim)],utTp)))

        ## Communication constraints
        com_cons = []
        for p in self.com_robots_idx:
            for i in range(1,self.sa_dz.shape[1]):
                if self.com_link.done_planning[p] == 1:
                    d_ip = LA.norm(dz[0:2,i-1] - np.asarray(\
                            [self.com_link.last_z[p][0],\
                            self.com_link.last_z[p][1]]))
                else:
                    d_ip = LA.norm(dz[0:2,i-1] - np.asarray(\
                            [self.com_link.intended_path_x[p][i],\
                            self.com_link.intended_path_y[p][i]]))
                com_cons.append(self.com_range - self.safe_epsilon - d_ip)

        ## Collision constraints
        collision_cons = []
        for p in self.collision_robots_idx:
            for i in range(1,self.sa_dz.shape[1]):
                if self.com_link.done_planning[p] == 1:
                    d_secu = self.rho
                    d_ip = LA.norm(dz[0:2,i-1] - np.asarray(\
                            [self.com_link.last_z[p][0],\
                            self.com_link.last_z[p][1]]))
                else:
                    d_secu = 2*self.rho
                    d_ip = LA.norm(dz[0:2,i-1] - np.asarray(\
                            [self.com_link.intended_path_x[p][i],\
                            self.com_link.intended_path_y[p][i]]))
                collision_cons.append(d_ip - d_secu - self.safe_epsilon)

        ## Deformation from intended path constraint
        deform_cons = []
        for i in range(1,self.sa_dz.shape[1]):
            d_ii = LA.norm(self.sa_dz[0:2,i] - dz[0:2,i-1])
            deform_cons.append(self.def_epsilon - d_ii)

        # Create final array
        ieq_cons = obst_cons + max_speed_cons + com_cons + collision_cons + deform_cons

        # Count how many inequations are not respected
        unsatisf_list = [ieq for ieq in ieq_cons if ieq < 0]
        self.unsatisf_ieq_values = unsatisf_list

        # return arrray where each element is an inequation constraint
        return np.asarray(ieq_cons)

    def _compute_conflicts(self):

        self.collision_robots_idx = []
        self.com_robots_idx = []

        for i in [j for j in range(n_robots) if j != self.eyed]:
            if self.com_link.done_planning[i] == 1:
                d_secu = self.rho
                linspeed_max = self.k_mod.u_max[0,0]
            else:   # TODO each robot must know the radius of the other robot
                d_secu = 2*self.rho 
                linspeed_max = 2*self.k_mod.u_max[0,0]

            d_ip = LA.norm(self.last_z - self.com_link.last_z[i])

            # TODO shouldn't it be Tc instead of Tp
            if d_ip <= d_secu + linspeed_max*self.Tp:
                self.collision_robots_idx.append(i)

            if i in neigh: # if the ith robot is a communication neighbor
                # TODO right side of condition should be min(self.com_range,self.com_link.com_range[i])
                if d_ip + linspeed_max*self.Tp >= self.com_range:
                    self.com_robots_idx.append(i)

        self.conflict_robots_idx = self.collision_robots_idx + self.com_robots_idx
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
            acc = self.acc

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
            acc = 1e-2

        output = fmin_slsqp(
                p_criterion,
                init_guess,
                eqcons=(),
                f_eqcons=p_eqcons,
                ieqcons=(),
                f_ieqcons=p_ieqcons,
                iprint=0,
                iter=self.maxit,
                acc=acc,
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

        btic = time.time()

        # update obstacles zone
        self._detected_obst_idxs()

#        # first guess for ctrl pts
        if not self.final_step:
            direc = self.final_z - self.last_z
            direc = direc/LA.norm(direc)
            last_ctrl_pt = self.last_z+self.D*direc
            self._linspace_ctrl_pts(last_ctrl_pt)
        else:
            # TODO eps constant
            eps = 0.0001
            final_ctrl_pt = self.final_z
            self.C[self.n_ctrlpts-1,:] = final_ctrl_pt.T
            aux = final_ctrl_pt.T - eps*np.array(\
                    [np.cos(self.k_mod.q_final[-1,0]), np.sin(self.k_mod.q_final[-1,0])])
            self.C[0:self.n_ctrlpts-1,0] = np.array(np.linspace(self.last_z[0,0],\
                    aux[0,0], self.n_ctrlpts-1, endpoint=False)).T
            self.C[0:self.n_ctrlpts-1,1] = np.array(np.linspace(self.last_z[1,0],\
                    aux[0,1], self.n_ctrlpts-1, endpoint=False)).T

        self.std_alone = True

        tic = time.time()
        self._solve_opt_pbl()
        toc = time.time()

        # No need to sync process here, the intended path does impact the conflicts computation

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

        time_idx = None if self.final_step else self.Tcd_idx+1

        dz = self._comb_bsp(self.mtime, self.C, 0).T
        for dev in range(1,self.k_mod.l+1):
            dz = np.append(dz,self._comb_bsp(
                    self.mtime, self.C, dev).T,axis=0)

#        TODO verify process safety
        for i in range(dz.shape[1]):
            self.com_link.intended_path_x[self.eyed][i] = dz[0,i]
            self.com_link.intended_path_y[self.eyed][i] = dz[1,i]

        self._compute_conflicts()
        self._log('d', 'R{0}@tkref{1}: $$$$$$$$ CONFLICT LIST $$$$$$$$: {2}'
                .format(self.eyed,self.mtime[0],self.conflict_robots_idx))

        # Sync with every robot on the conflict list
        #  1. notify every robot waiting on this robot that it ready for conflict solving
        with self.conflict_syncer_conds[self.eyed]:
            self.conflict_syncer[self.eyed].value = 1
            self.conflict_syncer_conds[self.eyed].notify_all()
        #  2. check if the robots on this robot conflict list are ready
        for i in self.conflict_robots_idx:
            with self.conflict_syncer_conds[i]:
                if self.conflict_syncer[i].value == 0:
                    self.conflict_syncer_conds[i].wait()
        # Now is safe to read the all robots' in the conflict list intended paths (or are done planning)

#        if self.conflict_robots_idx != [] and False:
        if self.conflict_robots_idx != []:

            self.std_alone = False

#            self.conflict_dz = [self._read_com_link()]
#            self._read_com_link()

            self.sa_dz = dz

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

            time_idx = None if self.final_step else self.Tcp_idx+1
                
            dz = self._comb_bsp(self.mtime[0:time_idx], self.C, 0).T
            for dev in range(1,self.k_mod.l+1):
                dz = np.append(dz,self._comb_bsp(
                        self.mtime[0:time_idx], self.C, dev).T,axis=0)
            
        # Storing
#        self.all_C += [self.C]
        self.all_dz.append(dz[:,0:time_idx])
        self.all_times.extend(self.mtime[0:time_idx])
        # TODO rejected path

        # Updating
        
        last_z = self.all_dz[-1][0:self.k_mod.u_dim,-1].reshape(
                self.k_mod.u_dim,1)

        # Sync robots here so no robot computing conflict get the wrong last_z of some robot
        with self.tc_syncer_cond:
            self.tc_syncer.value += 1
            if self.tc_syncer.value != self.n_robots:  # if not all robots are read
                self._log('d', 'R{}: I\'m going to sleep!'.format(self.eyed))
                self.tc_syncer_cond.wait()
            else:                                # otherwise wake up everybody
                self.tc_syncer_cond.notify_all()
            self.tc_syncer.value -= 1            # decrement syncer (idem)
#            self.com_link.last_z[self.eyed] = last_z
            for i in range(self.k_mod.u_dim):
                self.com_link.last_z[self.eyed][i] = last_z[i,0]

        with self.conflict_syncer_conds[self.eyed]:
            self.conflict_syncer[self.eyed].value = 0


        if not self.final_step:
            if self.std_alone == False:
                self.knots = self.knots + self.Tcp
                self.mtime = [tk+self.Tcp for tk in self.mtime]
            else:
                self.knots = self.knots + self.Tcd
                self.mtime = [tk+self.Tcd for tk in self.mtime]
            self.last_z = last_z
            self.last_q = self.k_mod.phi1(self.all_dz[-1][:,-1].reshape(
                    self.k_mod.l+1, self.k_mod.u_dim).T)
            self.last_u = self.k_mod.phi2(self.all_dz[-1][:,-1].reshape(
                    self.k_mod.l+1, self.k_mod.u_dim).T)

        btoc = time.time()
        self.all_comp_times.append(btoc-btic)

        return

    def _plan(self):

        self._log('i', 'R{rid}: Init motion planning'.format(rid=self.eyed))

        self.final_step = False

        self.knots = self._gen_knots(self.t_init, self.Td)
        self.mtime = np.linspace(self.t_init, self.Td, self.N_s)

        # while the remaining dist is greater than the max dist during Tp
        while LA.norm(self.last_z - self.final_z) > self.D:

            self._plan_section()
            self._log('i', 'R{}: --------------------------'.format(self.eyed))

            # SYNC process
#            with self.tc_syncer_cond:
#                self.tc_syncer.value += 1
#                if self.tc_syncer.value != self.n_robots:  # if not all robots are read
#                    self._log('d', 'R{}: I\'m going to sleep!'.format(self.eyed))
#                    self.tc_syncer_cond.wait()
#                else:                                # otherwise wake up everybody
#                    self.tc_syncer_cond.notify_all()
#                self.tc_syncer.value -= 1            # decrement syncer (idem)
#
#            with self.conflict_syncer_conds[self.eyed]:
#                self.conflict_syncer[self.eyed].value = 0

        self.final_step = True
        self.est_dtime = LA.norm(self.last_z - self.final_z)/self.k_mod.u_max[0,0]

        self.knots = self._gen_knots(self.mtime[0], self.mtime[0]+self.est_dtime)
        self.mtime = np.linspace(self.mtime[0], self.mtime[0]+self.est_dtime, self.N_s)

        self._plan_section()
        self._log('i','R{}: Finished motion planning'.format(self.eyed))
        self._log('i', 'R{}: --------------------------'.format(self.eyed))

        self.sol[self.eyed] = self.all_dz
        self.rtime[self.eyed] = self.all_times
        self.ctime[self.eyed] = self.all_comp_times
        self.com_link.done_planning[self.eyed] = 1

        #  Notify every robot waiting on this robot that it is ready for the conflict solving
        with self.conflict_syncer_conds[self.eyed]:
            self.conflict_syncer[self.eyed].value = 1
            self.conflict_syncer_conds[self.eyed].notify_all()

        # Make sure any robot waiting on this robot awake before returning
        with self.tc_syncer_cond:
            self.tc_syncer.value += 1               # increment synker
            if self.tc_syncer.value == self.n_robots:  # if all robots are read
                self.tc_syncer_cond.notify_all()

        return

###############################################################################
# World
###############################################################################
class WorldSim(object):
    """ Where to instatiate obstacles, robots, bonderies
        initial and final conditions etc
    """
    def __init__(self, Tc, robots, obstacles, phy_boundary):
        self.robs = robots
        self.obsts = obstacles
        self.Tc = Tc
        self.ph_bound = phy_boundary

    def run(self, interac_plot=False, speed_plot=False):

        # Make all robots plan their trajectories
        [r.planning_process.start() for r in self.robs]
        [r.planning_process.join() for r in self.robs]

        # Reshaping the solution
        path = range(len(self.robs))
        seg_pts_idx = [[] for _ in range(len(self.robs))]
        for i in range(len(self.robs)):
            path[i] = self.robs[0].sol[i][0]
            seg_pts_idx[i] += [0]
            for p in self.robs[0].sol[i][1:]:
                c = path[i].shape[1]
                seg_pts_idx[i] += [c]
                path[i] = np.append(path[i], p, axis=1)

        # From [z dz ddz](t) get q(t) and u(t)
        zdzddz = range(len(self.robs))
        for i in range(len(self.robs)):
            zdzddz[i] = map(lambda z:z.reshape(self.robs[i].k_mod.l+1, \
                    self.robs[i].k_mod.u_dim).T, path[i].T)

        # get a list over time of command values u(t)
        ut = range(len(self.robs))
        for i in range(len(self.robs)):
            ut[i] = map(self.robs[i].k_mod.phi2, zdzddz[i])

        # get a list over time of values q(t)
        qt = range(len(self.robs))
        for i in range(len(self.robs)):
            qt[i] = map(self.robs[i].k_mod.phi1, zdzddz[i])

        # get times
        rtime = range(len(self.robs))
        for i in range(len(self.robs)):
            rtime[i] = self.robs[0].rtime[i]

        ctime = range(len(self.robs))
        for i in range(len(self.robs)):
            ctime[i] = self.robs[0].ctime[i]

        for i in range(len(self.robs)):
            logging.info('R{rid}: TOT: {d}'.format(rid=i,d=rtime[i][-1]))
            logging.info('R{rid}: NSE: {d}'.format(rid=i,d=len(ctime[i])))
            logging.info('R{rid}: FIR: {d}'.format(rid=i,d=ctime[i][0]))
            logging.info('R{rid}: LAS: {d}'.format(rid=i,d=ctime[i][-1]))
            if len(ctime[i]) > 2:
                logging.info('R{rid}: MAX: {d}'.format(rid=i,d=max(ctime[i][1:-1])))
                logging.info('R{rid}: MIN: {d}'.format(rid=i,d=min(ctime[i][1:-1])))
                logging.info('R{rid}: AVG: {d}'.format(rid=i,d=np.mean(ctime[i][1:-1])))
                logging.info('R{rid}: RAT: {d}'.format(rid=i,d=max(ctime[i][1:-1])/self.Tc))
            else:
                logging.info('R{rid}: MAX: {d}'.format(rid=i,d=max(ctime[i])))
                logging.info('R{rid}: MIN: {d}'.format(rid=i,d=min(ctime[i])))
                logging.info('R{rid}: AVG: {d}'.format(rid=i,d=np.mean(ctime[i])))
                logging.info('R{rid}: RAT: {d}'.format(rid=i,d=max(ctime[i])/self.Tc))

        # PLOT ###############################################################

        raw_input('Press enter to start plot')

        # Interactive plot
        plt.ion()

        fig = plt.figure()
        ax = fig.gca()
        ax.set_xlabel('x(m)')
        ax.set_ylabel('y(m)')
        ax.set_title('Generated trajectory')
        ax.axis('equal')

        fig_s, axarray = plt.subplots(2)
        axarray[0].set_ylabel('v(m/s)')
        axarray[0].set_title('Linear speed')
        axarray[1].set_xlabel('time(s)')
        axarray[1].set_ylabel('w(rad/s)')
        axarray[1].set_title('Angular speed')

        aux = np.linspace(0.0, 1.0, 1e2)
        colors = [[i, 1.0-i, np.random.choice(aux)] for i in np.linspace(0.0, 1.0,len(self.robs))]

        while True:
            # Creating obstacles in the plot
            [obst.plot(fig, offset=self.robs[0].rho) for obst in self.obsts]

            plt_paths = range(len(self.robs))
            plt_seg_pts = range(len(self.robs))
            plt_robots_c = range(len(self.robs))
            plt_robots_t = range(len(self.robs))
            for i in range(len(self.robs)):
                plt_paths[i], = ax.plot(path[i][0,0],path[i][1,0],color=colors[i])
                plt_seg_pts[i], = ax.plot(path[i][0,seg_pts_idx[i][0]],\
                        path[i][1,seg_pts_idx[i][0]],color=colors[i],ls='None',marker='o',markersize=5)
                plt_robots_c[i] = plt.Circle(
                        (path[i][0,0], path[i][1,0]), # position
                        self.robs[i].rho, # radius
                        color='m',
                        ls = 'solid',
                        fill=False)
                rho = self.robs[i].rho
                xy = np.array(
                        [[rho*np.cos(qt[i][0][-1,0])+path[i][0,0],\
                        rho*np.sin(qt[i][0][-1,0])+path[i][1,0]],
                        [rho*np.cos(qt[i][0][-1,0]-2.5*np.pi/3.0)+path[i][0,0],\
                        rho*np.sin(qt[i][0][-1,0]-2.5*np.pi/3.0)+path[i][1,0]],
                        [rho*np.cos(qt[i][0][-1,0]+2.5*np.pi/3.0)+path[i][0,0],\
                        rho*np.sin(qt[i][0][-1,0]+2.5*np.pi/3.0)+path[i][1,0]]])
                plt_robots_t[i] = plt.Polygon(xy, color='m',fill=True,alpha=0.2)
            
            [ax.add_artist(r) for r in plt_robots_c]
            [ax.add_artist(r) for r in plt_robots_t]
            for i in range(1,10):
                fig.savefig('../traces/pngs/multirobot-path-'+str(i)+'.png', bbox_inches='tight')
    
            ctr = 1
            while True:
                end = 0
                for i in range(len(self.robs)):
    #                print(path[i].shape)
                    if ctr < path[i].shape[1]:
                        plt_paths[i].set_xdata(path[i][0,0:ctr+1])
                        plt_paths[i].set_ydata(path[i][1,0:ctr+1])
                        aux = [s for s in seg_pts_idx[i] if  ctr > s ]
                        plt_seg_pts[i].set_xdata(path[i][0,aux])
                        plt_seg_pts[i].set_ydata(path[i][1,aux])
                        plt_robots_c[i].center = path[i][0,ctr],\
                                path[i][1,ctr]
                        rho = self.robs[i].rho
                        xy = np.array(
                                [[rho*np.cos(qt[i][ctr][-1,0])+path[i][0,ctr],
                                rho*np.sin(qt[i][ctr][-1,0])+path[i][1,ctr]],
                                [rho*np.cos(qt[i][ctr][-1,0]-2.5*np.pi/3.0)+path[i][0,ctr],
                                rho*np.sin(qt[i][ctr][-1,0]-2.5*np.pi/3.0)+path[i][1,ctr]],
                                [rho*np.cos(qt[i][ctr][-1,0]+2.5*np.pi/3.0)+path[i][0,ctr],
                                rho*np.sin(qt[i][ctr][-1,0]+2.5*np.pi/3.0)+path[i][1,ctr]]])
                        plt_robots_t[i].set_xy(xy)
                    else:
                        end += 1
                if end == len(self.robs):
                    break
                time.sleep(0.01)
                ax.relim()
                ax.autoscale_view(True,True,True)
                fig.canvas.draw()
                ctr += 1
                fig.savefig('../traces/pngs/multirobot-path-'+str(ctr+8)+'.png', bbox_inches='tight')
            for i in range(1,10):
                fig.savefig('../traces/pngs/multirobot-path-'+str(ctr+8+i)+'.png', bbox_inches='tight')
    
            for i in range(len(self.robs)):
                linspeed = map(lambda x:x[0,0], ut[i])
                angspeed = map(lambda x:x[1,0], ut[i])
                axarray[0].plot(rtime[i], linspeed, color=colors[i])
                axarray[1].plot(rtime[i], angspeed, color=colors[i])
            axarray[0].grid()
            axarray[1].grid()
            axarray[0].set_ylim([0.0,1.1*self.robs[0].k_mod.u_max[0,0]])
            axarray[1].set_ylim([-7.0,7.0])
            fig_s.savefig('../traces/pngs/multirobot-vw.png', bbox_inches='tight')
                
            plt.show()
    
            raw_input('Press enter to see the animation again or Ctrl-c + enter to kill the simulation')
            axarray[0].cla()
            axarray[1].cla()
            fig.gca().cla()
        
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
    max_radius = 0.4
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

    n_obsts = 1
    n_robots = 2
    N_s = 20
    Tc = 1.0

    scriptname, method = parse_cmdline()

    if method != None:
        fname = scriptname[0:-3]+'_'+method+'.log'
    else:
        fname = scriptname[0:-3]+'.log'

#    logging.basicConfig(filename=fname,format='%(levelname)s:%(message)s',\
#            filemode='w',level=logging.DEBUG)
    logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.DEBUG)

    boundary = Boundary([-6.0,6.0], [-6.0,6.0])

    obst_info = rand_round_obst(n_obsts, Boundary([-1.2,1.2], [-2.0,2.0]))

    # these obst info
#    obst_info = [([0.25, 2.5], 0.20),([ 3.0,  2.40], 0.50),
#            ([ 1.25,  3.00], 0.10),([ 0.30,  1.00], 0.10),
#            ([-0.50,  1.50], 0.30)]

#    obst_info = [([0.25, 2.5], 0.20),([ 2.30,  2.50], 0.50),
#            ([ 1.25,  3.00], 0.10),([ 0.30,  1.00], 0.10),
#            ([-0.50,  1.50], 0.30)]
#    obst_info = [([-0.5, 2.5], 0.30),([0.7,  2.50], 0.30), ([0.0, 1.1], 0.3)]
#    obst_info = [([-0.52, -0.552], 0.31),([-0.58,  0.541], 0.298),([1.41, -0.1], 0.35)]
    obst_info = [([-0.26506650665066511, 0.40226022602260203], 0.39504950495049507), \
            ([0.7218821882188216, -0.93849384938494], 0.28383838383838383), \
            ([-0.58077807780778068, 2.4762276227622757], 0.37094709470947096)]

#[([-0.26506650665066511, 0.40226022602260203], 0.39504950495049507), ([0.39979997999799977, 2.5724372437243725], 0.30019001900190018), ([0.28218821882188216, -1.493849384938494], 0.28383838383838383), ([-0.58077807780778068, 2.4762276227622757], 0.37094709470947096)]

    obstacles = [RoundObstacle(i[0], i[1]) for i in obst_info]
    obstacles += [PolygonObstacle(np.array(
            [[1.0, 1.64],[0.8,-0.1],[1.75,0.1],[1.6,1.4]]))]

    kine_models = [UnicycleKineModel(
#            [-2.56, -0.59, 0.0], # q_initial
#            [ 2.56,  0.51, 0.0], # q_final
#            [ 0.0,  0.0],          # u_initial
#            [ 0.0,  0.0],          # u_final
#            [ 1.0,  5.0]),          # u_max
 #           UnicycleKineModel(
            [-2.5,  1.2, 0.0], # q_initial
            [ 2.5,  1.0, 0.0], # q_final
            [ 0.0,  0.0],          # u_initial
            [ 0.0,  0.0],          # u_final
            [ 1.0,  5.0]),          # u_max
            UnicycleKineModel(
            [-2.4,  0.1, 0.0], # q_initial
            [ 2.6,  0.49, 0.0], # q_final
            [ 0.0,  0.0],          # u_initial
            [ 0.0,  0.0],          # u_final
            [ 1.0,  5.0])]          # u_max
            
#    kine_models = [UnicycleKineModel(
#            [ float(i),  0.0, np.pi/2], # q_initial
#            [ float(-i),  5.0, np.pi/2], # q_final
#            [ 0.0,  0.0],          # u_initial
#            [ 0.0,  0.0],          # u_final
#            [ 1.0,  5.0])          # u_max
#            for i in [j-n_robots/2+0.5 for j in range(n_robots)]]

#    kine_models = [UnicycleKineModel(
#            [ float(i)/2.0,  0.0, np.pi/2], # q_initial
#            [ float(i),  0.0, np.pi/2], # q_initial
#            [ 0.0,  0.0, np.pi/2], # q_initial
#            [(n_robots-i+1.0)/2.0,  5.0, np.pi/2], # q_final
#            [ (n_robots-i+1.0),  5.0, np.pi/2], # q_final
#            [ 3.5,  5.0, np.pi/2], # q_final
#            [ 0.0,  0.0],          # u_initial
#            [ 0.0,  0.0],          # u_final
#            [ 1.0,  5.0])          # u_max
#            for i in [j-n_robots for j in range(n_robots)]]


    # Multiprocessing stuff ############################################
    # Locks
    log_lock = mpc.Lock()

    # Conditions
    tc_syncer_cond = mpc.Condition()
    conflict_syncer_conds = [mpc.Condition() for i in range(n_robots)]

    # Shared memory (for simple data)
    tc_syncer = mpc.Value('I', 0) # unsigned int
    conflict_syncer = [mpc.Value('I', 0) for i in range(n_robots)]
    # communication link shared memory
    done_planning = [mpc.Value('I', 0) for i in range(n_robots)]
    intended_path_x = [mpc.Array('d', N_s*[0.0]) for i in range(n_robots)]
    intended_path_y = [mpc.Array('d', N_s*[0.0]) for i in range(n_robots)]
    last_z = [mpc.Array('d', [kine_models[i].q_init[0,0], \
            kine_models[i].q_init[1,0]]) for i in range(n_robots)]
    com_link = RobotMsg(done_planning, intended_path_x, intended_path_y, last_z)

    # shared memory by a server process manager (because they can support arbitrary object types)
    manager = mpc.Manager()
    solutions = manager.list(range(n_robots))
    robots_time = manager.list(range(n_robots))
    robots_comp_time = manager.list(range(n_robots))
    ####################################################################

    robots = []
    for i in range(n_robots):
        if i-1 >= 0 and i+1 < n_robots:
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
                tc_syncer,              # process counter for sync
                tc_syncer_cond,
                conflict_syncer,        # array used for sync between robots having conflicts
                conflict_syncer_conds,
                com_link,               # communication link
                solutions,              # where to store the solutions
                robots_time,
                robots_comp_time,
                neigh,                  # neighbors to whom this robot shall talk (used for conflict only, not communic)
                N_s=N_s,                 # numbers samplings for each planning interval
                n_knots=6,              # number of knots for b-spline interpolation
                Tc=Tc,                 # computation time
                Tp=2.0,                 # planning horizon
                Td=2.0,
                def_epsilon=10.1,       # in meters
                safe_epsilon=0.1,      # in meters
                log_lock=log_lock)]                 # planning horizon (for stand alone plan)

    [r.set_option('acc', 1e-4) for r in robots] # accuracy (hard to understand the physical meaning of this)
    [r.set_option('maxit', 50) for r in robots] # max number of iterations for the opt solver

    world_sim = WorldSim(Tc,robots,obstacles,boundary) # create the world

    summary_info = world_sim.run(interac_plot=False, speed_plot=True) # run simulation (TODO take parameters out)

