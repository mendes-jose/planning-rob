#!/usr/bin/python

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import scipy.interpolate as si
import time
import itertools
import pyOpt
import multiprocessing as mpc
from scipy.optimize import f_min_slsqp

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
        ax.add_artist(self.plt_circle())
        ax.add_artist(self.plt_circle(linestyle='dashed', offset=offset))

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
        
        self.l = 2 # number of need derivations

    def phi0(sefl, q)
        """ Returns z given q
        """
        return q[0:2,0]

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
            kine_model,
            obstacles,
            phy_boundary,
            com_link,
            N_s=20,
            n_knots=6,
            t_init=0.0,
            t_sup=1e10,
            Tc=1.0,
            Tp=2.0,
            Td=2.5,
            rho=0.2,
            detec_rho=2.0,
            log_lock=None):

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

        slelf.log_lock = log_lock

        # Declaring the planning process
        self.planning_process = mpc.Process(target=Robot._plan, args=(self,))

    def _gen_knots(self, t_init, t_final):
        """ Generate b-spline knots given initial and final times
        """
        gk = lambda x:t_init + (x-(self.d-1.0))*(t_final-t_init)/self.n_knot
        knots = [t_init for _ in range(self.d)]
        knots.extend([gk(i) for i in range(self.d,self.d+self.n_knot-1)])
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

    def _linspace_ctrl_pts(self, final_ctrl_pt):
        self.C = np.array(np.linspace(self.last_z[0,0
        self.C[:,0] = np.array(np.linspace(self.last_z[0,0],\
                final_ctrl_pt[0,0], self.n_ctrlpts)).T
        self.C[:,1] = np.array(np.linspace(last_z[1,0],\
                final_ctrl_pt[1,0], self.n_ctrlpts)).T

    def _detected_obst_idx(self):
        idx_list = []
        for idx in range(self.obst_map.shape[0]):
            dist = LA.norm(self.obst[idx].cp - self.last_z)
            if dist < self.d_rho:
                idx_list += [idx]
        return idx_list

    def _ls_sa_criterion(self, x):
        return (x[0]+self.mtime[0])**2

    def _ls_sa_feqcons(self, x):
        return

    def _ls_sa_fieqcons(self, x):
        return

    def _sa_criterion(self, x):
        return

    def _sa_feqcons(self, x):
        return

    def _sa_fieqcons(self, x):
        return

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

    def _scipy_callback(self): #TODO
        return

    def _solve_opt_pbl(self):

        if self.interac_plot:
            f_callback = self._scipy_callback
        else:
            f_callback = None

        if not self.final_step:

            if self.std_alone:
                f_fieqcons = self._fieqcons
            else:
                f_fieqcons = self._c_fieqcons

            output = fmin_slsqp(self._criterion,
                    self.C.reshape(self.n_ctrlpts*self.k_mod.u_dim),
                    eqcons=(),
                    f_eqcons=self._feqcons,
                    ieqcons=(),
                    f_ieqcons=self._fieqcons,
                    iprint=1,
                    iter=self.it,
                    acc=self.acc,
                    full_output=True,
                    callback=f_callback)

            #imode = output[3]
            # TODO handle optimization exit mode
            self.C = output[0].reshape(self.n_ctrlpts, self.k_mod.u_dim)
            
        else:
            output = fmin_slsqp(self._lcriterion,
                    self.C.reshape(self.n_ctrlpts*self.mrob.u_dim),
                    eqcons=(),
                    f_eqcons=self._lfeqcons,
                    ieqcons=(),
                    f_ieqcons=self._lfieqcons,
                    iprint=1,
                    iter=self.it,
                    acc=self.acc,
                    full_output=True,
                    callback=f_callback)

            #imode = output[3]
            # TODO handle optimization exit mode
            self.C = output[0][1:].reshape(self.n_ctrlpts, self.k_mod.u_dim)
            self.dt_final = output[0][0]
            #sefl.t_final = self.mtime[0] + dt_final
            
        return

    def _plan_section(self):

        # update obstacles zone
        self._detect_obst_idx()

        # first guess for ctrl pts
        if not self.final_step:
            direc = self.final_z - self.last_z
            direc = direc/LA.norm(direc)
            last_ctrl_pt = self.last_z+self.D*direc
        else:
            last_ctrl_pt = self.final_z

        self._linspace_ctrl_pts_(last_ctrl_pt)

        self.std_alone = True

        tic = time.time()
        self._solve_opt_pbl()
        toc = time.time()

        if self.log_lock != None:
            self.log_lock.acquire()
        logging.info(
                'R{rid},{tk}: Time to solve stand alone optimisation problem: \
                {t}'.format(rid=self.eyed, t=toc-tic, tk=self.tk))
        if self.log_lock != None:
            self.log_lock.release()

        self._compute_conflicts()

        if self.conflicts != []:

            self.std_alone = False

            self._read_com_link()

            tic = time.time()
            self._solve_opt_pbl()
            toc = time.time()

            if self.log_lock != None:
                self.log_lock.acquire()
            logging.info(
                    'R{rid},{tk}: Time to solve optimisation problem: {t}'.\
                    format(rid=self.eyed, t=toc-tic, tk=self.tk))
            if self.log_lock != None:
                self.log_lock.release()

        return

    def _plan(self):

        self._init_planning()

        self.final_step = False

        # while the remaining dist is greater than the max dist during Tp
        while LA.norm(self.last_z - self.final_z) > self.D:
            self._plan_section()

        self.final_step = True
        self._plan_section()

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

        # Interactive plot
        if interac_plot == True:
            plt.ion()
#            self.mrobot.setOption('IPLOT', True)
            # TODO: in the future we may set IPLOT True for obstacles as well

        # Initiating plot
#        self.fig = plt.figure()
#        ax = self.fig.gca()
#        ax.set_xlabel('x(m)')
#        ax.set_ylabel('y(m)')
#        ax.set_title('Generated trajectory')
#        ax.axis('equal')

        # Creating obstacles in the plot
 #       [obst.plot(self.fig, offset=self.mrobot.rho) for obst in self.obst]

        # Initiate robot path in plot
#        self.mrobot.plot(self.fig)

        # Initiate robot speed plot
#        if speedPlot == True:
#            self.mrobot.plotSpeeds(plt.subplots(2))

        # Creating robot path (and updating plot if IPLOT is true)

        

        

        ret = self.mrobot.gen_trajectory()

        logging.info('Fineshed planning')

        plt.show(block=True)

        return ret


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


# MAIN ########################################################################

if __name__ == '__main__':

    scriptname, method = parse_cmdline()

    if method != None:
        fname = scriptname[0:-3]+'_'+method+'.log'
    else:
        fname = scriptname[0:-3]+'.log'

    logging.basicConfig(filename=fname,format='%(levelname)s:%(message)s',\
            filemode='w',level=logging.DEBUG)
#    logging.basicConfig(level=logging.DEBUG)

    boundary = Boundary([-5.0,5.0], [-1.0,6.0])

    n_obsts = 7
    obst_info = rand_round_obst(n_obsts, Boundary([-1.0,4.0],[0.7,4.0]))

    obstacles = [RoundObstacle(i[0], i[1]) for i in obst_info]

    kine_models = [UnicycleKineModel(
            [ 0.0,  0.0, np.pi/2], # q_initial
            [ 2.0,  5.0, np.pi/2], # q_final
            [ 0.0,  0.0],          # u_initial
            [ 0.0,  0.0],          # u_final
            [ 1.0,  5.0]]          # u_max
#            [ 1.6,  3.0])]          # du_max

    kine_models += [UnicycleKineModel(
            [ 1.0,  0.0, np.pi/2], # q_initial
            [ 1.0,  5.0, np.pi/2], # q_final
            [ 0.0,  0.0],          # u_initial
            [ 0.0,  0.0],          # u_final
            [ 1.0,  5.0]]          # u_max
#            [ 1.6,  3.0])]          # du_max

    kine_models += [UnicycleKineModel(
            [-1.0,  0.0, np.pi/2], # q_initial
            [ 3.0,  5.0, np.pi/2], # q_final
            [ 0.0,  0.0],          # u_initial
            [ 0.0,  0.0],          # u_final
            [ 1.0,  5.0]]          # u_max
#            [ 1.6,  3.0])]          # du_max

    log_lock = mpc.Lock()
    pcc_lock = mpc.Lock()
    com_lock = mpc.Lock()

    process_counter = Value('I',0,pcc_lock) # unsigned int
    com_link = Array('d',max(self.robots,key=lambda rob:rob.N_s).N_s,com_lock)

    robots = [Robot(
            kine_models[i],
            obstacles,
            boundary,
            com_link,
            pcc
            T
            ) for 
    


