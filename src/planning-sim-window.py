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
import pyOpt

# Unicycle Kinematic Model ----------------------------------------------------
class Unicycle_Kine_Model(object):
  def __init__(self):
    self.u_dim = 2
    self.q_dim = 3
    self.rho = 0.2 # m
    self.u_abs_max = np.matrix('1.0; 5')

    self.l = 2 # number of need derivations

  ## Defining functions phi1(z, dz) and phi2(dz, ddz)
  # Unicycle kinematic model:
  # q' = f(q,u)
  # [x', y', theta']^T = [v cos(theta), v sin(theta), w]^T
  #
  # Changing variables to obtain phi1 and phi2 given z = [x, y]^T:
  #
  # |x    |   |z1                                 |
  # |y    |   |z2                                 |
  # |theta| = |arctan(z2'/z1')                    |
  # |v    |   |sqrt(z1'^2 + z2'^2)                |
  # |w    |   |(z1'z2'' - z2'z1'')/(z1'^2 + z2'^2)|
  #

  def phi0(self, q):
    return q[0:2,0]

  # z here is a list of matrix [z dz ddz]
  def phi1(self, z):
    if z.shape >= (self.u_dim, self.l+1):
      return np.append(z[:,0], \
                       np.matrix(np.arctan2(z[1,1], z[0,1])), axis = 0)
    else:
      # TODO ERROR
      print('Bad z input. Returning zeros')
#      return np.matrix('0.0; 0.0; 0.0')
      return coco1

  # z here is a list of matrix [z dz ddz]
  def phi2(self, z):
    if z.shape >= (self.u_dim, self.l+1):
      if (z[0,1]**2 + z[1,1]**2 != 0):
        return np.matrix([[LA.norm(z[:,1])], \
            [(z[0,1]*z[1,2]-z[1,1]*z[0,2] \
            )/(z[0,1]**2 + z[1,1]**2)]])
      else:
        print('x\' and y\' are zero! Unsing angspeed=0')
        return np.matrix([[LA.norm(z[:,1])],[0.0]])
    else:
      # TODO ERROR
      print('Bad z input. Returning zeros')
#      return np.matrix('0.0; 0.0')
      return coco2

#  def angle(x):
#    pi = math.pi
#    twopi = 2*pi
#    return (x+pi)%twopi-pi

# Trajectory Generation -------------------------------------------------------
class Trajectory_Generation(object):
  def __init__(self, mrobot):

  ## "Arbitrary" parameters
    self.N_s = 80 # nb of samples for discretization
    self.n_knot = 12 # nb of non zero lenght intervals of the knot series
    self.t_init = 0.0
    self.Tc = 1.0
    self.Tp = 12.2
    tstep = (self.Tp-self.t_init)/(self.N_s-1)
    Tc_idx = int(round(self.Tc/tstep))
    self.detection_radius = 12.0

  ## Mobile Robot object
    self.mrob = mrobot

  ## Other parameters
    self.d = self.mrob.l+2 # B-spline order (integer | d > l+1)
    self.n_ctrlpts = self.n_knot + self.d - 1 # nb of ctrl points

  ## Constaints values...
  ## ...for equations:
    # initial state
    self.q_init = np.matrix([[0.0], [0.0], [np.pi/2]])
    # final state
    self.q_fin = np.matrix([[2.0], [5.0], [np.pi/2]])
    # initial control input
    self.u_init = np.matrix([[0.0], [0.0]])
    # final control input
    self.u_fin = np.matrix([[0.0], [0.0]])
  ## ...for inequations:
    # Control boundary
    self.u_abs_max = self.mrob.u_abs_max
    # Obstacles (Q_occupied)
    # TODO: Obstacles random generation
    self.obst_map =                          np.matrix([0.25, 2.5, 0.2])
    self.obst_map = np.append(self.obst_map, np.matrix([2.3,  2.5, 0.5]),
        axis = 0)
    self.obst_map = np.append(self.obst_map, np.matrix([1.25, 3,   0.1]),
        axis = 0)
    self.obst_map = np.append(self.obst_map, np.matrix([0.3,  1.0,   0.1]),
        axis = 0)
    self.obst_map = np.append(self.obst_map, np.matrix([-0.5, 1.5, 0.3]),
        axis = 0)
#    self.obst_map = np.append(self.obst_map, np.matrix([1.6, 4.3, 0.2]),
#        axis = 0)
    
    # max distance within Tp
    self.D = self.Tp * self.u_abs_max[0,0]

    # Ctrl pts init
    C = np.array(np.zeros((self.n_ctrlpts,self.mrob.u_dim)))
    C_lower = np.array([-10, -1]*self.n_ctrlpts)
    C_upper = np.array([+10, +6]*self.n_ctrlpts)
    
    self.detected_obst_idxs = []

    # final trajectory
    self.C_ref = []

  ## Generate initial b-spline knots
    self.knots = self._gen_knots(self.t_init, self.Tp)
    self.mtime = np.linspace(self.t_init, self.Tp, self.N_s)

  ## Optimization results
    self.unsatisf_eq_values = []
    self.unsatisf_ieq_values = []

  ## Plot initialization
    plt.ion()
    self.fig = plt.figure()
    self.fig_speed = plt.subplots(2)
    ax = self.fig.gca()

    # creating obstacles' circles
    circ = []
    for r in range(self.obst_map.shape[0]):
      # external dashed circles
      circ = circ + \
          [plt.Circle((self.obst_map[r,0], self.obst_map[r,1]),
          self.obst_map[r,2]+self.mrob.rho,color='k',ls = 'dashed',fill=False)]
      # internal continous circles
      circ = circ + \
          [plt.Circle((self.obst_map[r,0], self.obst_map[r,1]),
          self.obst_map[r,2], color='k', fill=False)]

    # adding circles to axis
    [ax.add_artist(c) for c in circ]

    # plot curve and its control points
    self.rejected_path,self.plt_ctrl_pts,self.plt_curve,self.plt_dot_curve,self.seg_pts = ax.plot(
            0.0, 0.0, 'm:',
            0.0, 0.0, '*',
            0.0, 0.0, 'b-',
            0.0, 0.0, 'g.',
            0.0, 0.0, 'rd')
    
    # formating figure
    plt.xlabel('x(m)')
    plt.ylabel('y(m)')
    plt.title('Generated trajectory')
    ax.axis('equal')

    axarray = self.fig_speed[0].axes
    self.plt_linspeed, = axarray[0].plot(0.0, 0.0)
    self.plt_angspeed, = axarray[1].plot(0.0, 0.0)
    axarray[0].set_ylabel('v(m/s)')
    axarray[0].set_title('Linear speed')
    axarray[1].set_xlabel('time(s)')
    axarray[1].set_ylabel('w(rad/s)')
    axarray[1].set_title('Angular speed')
    axarray[0].grid()
    axarray[1].grid()

    final_z = self.mrob.phi0(self.q_fin)
    self.last_q = self.q_init
    self.last_u = self.u_init
    last_z = self.mrob.phi0(self.last_q)
    self.all_dz = []
    self.all_rejected_z = []
    self.itcount = 0

    usepyopt = False

    while LA.norm(last_z - final_z) > self.D: # while the remaining dist (straight line) is greater than the max dist during Tp

        self.detected_obst_idxs = self._detected_obst_idx(last_z)

        # initiate ctrl points (straight line towards final z)
        direc = final_z - last_z
        self.normalization_dist = LA.norm(direc)
        direc = direc/self.normalization_dist
#        C[:,0] =np.array(np.linspace(last_z[0,0],\
#                last_z[0,0]+self.D*direc[0,0], self.n_ctrlpts)).T
#        C[:,1] =np.array(np.linspace(last_z[1,0],\
#                last_z[1,0]+self.D*direc[1,0], self.n_ctrlpts)).T

        tic = time.time()
        if usepyopt:
            # Define the optimization problem
            self.opt_prob = pyOpt.Optimization(
                    'Faster path with obstacles', # name of the problem
                    self._obj_func) # object function (criterium, eq. and ineq.)
    
            self.opt_prob.addObj('J')
    
            self.opt_prob.addVarGroup( # minimization arguments
                    'C',
                    self.mrob.u_dim*self.n_ctrlpts, # dimension
                    'c', # continous
                    lower=list(C_lower),
                    value=list(np.squeeze(C.reshape(1,self.n_ctrlpts*self.mrob.u_dim))),
                    upper=list(C_upper))
    
            self.opt_prob.addConGroup( # equations constraints
                    'ec',
                    self.mrob.q_dim + self.mrob.u_dim, # dimension
                    'e') # equations
    
            self.opt_prob.addConGroup( # inequations constraints
                    'ic',
                    self.N_s*self.mrob.u_dim +
                            self.N_s*len(self.detected_obst_idxs), # dimenstion
                    'i') # inequations
    
            # solve constrained optmization
#            solver = pyOpt.PSQP(pll_type='POA')
#            solver.setOption('ACC', 1e-6)
#            solver.setOption('MAXIT', 30)
            solver = pyOpt.ALGENCAN(pll_type='POA')
            solver.setOption('epsfeas', 1e-2)
            solver.setOption('epsopt', 8e-1)
    
            [J, C_aux, information] = solver(self.opt_prob) 
            C_aux = np.array(C_aux)
#            if information.value != 0 and iformation.value != 9:
            usepyopt = False 

        else:
            # solve constrained optmization
            outp = fmin_slsqp(self._criteria,
                                np.squeeze(C.reshape(1,self.n_ctrlpts*self.mrob.u_dim)),
                                eqcons=(),
                                f_eqcons=self._feqcons,
                                ieqcons=(),
                                f_ieqcons=self._fieqcons,
                                iprint=1,
                                iter=30,
                                acc=1e-4,
                                full_output=True,
                                callback=self._plot_update)
            C_aux = outp[0]
            imode = outp[3]
            print('Exit mode: {}'.format(imode))
            if imode != 0 and imode != 9:
                usepyopt = True
                continue
        

        print('------------------------\nElapsed time for {} iteraction: {}'.format(self.itcount, time.time()-tic))

        print('No of equations unsatisfied: {}'.format(len(self.unsatisf_eq_values)))
        print('Mean and variance of equations unsatisfied: ({},{})'.format(np.mean(self.unsatisf_eq_values), np.std(self.unsatisf_eq_values)))
        print('No of inequations unsatisfied: {}'.format(len(self.unsatisf_ieq_values)))
        print('Mean and variance of inequations unsatisfied: ({},{})'.format(np.mean(self.unsatisf_ieq_values), np.std(self.unsatisf_ieq_values)))

        # test if opt went well
        # if yes
        C = C_aux
        # if no
        # continue

        C = C.reshape(self.n_ctrlpts, self.mrob.u_dim)
        # store ctrl points and [z dz ddz](t)
        self.C_ref += [C]
        dz = self._comb_bsp(self.mtime[0:Tc_idx], C, 0).T
        for dev in range(1,self.mrob.l+1):
            dz = np.append(dz,self._comb_bsp(
                    self.mtime[0:Tc_idx], C, dev).T,axis=0)
        self.all_dz += [dz]

        rejected_z = self._comb_bsp(self.mtime[Tc_idx:], C, 0).T
        self.all_rejected_z += [np.append(np.append(dz[0:2,:], rejected_z, axis=1), np.fliplr(rejected_z), axis=1)]
        
        # update needed values
        self.knots = self.knots + self.Tc
        self.mtime = [tk+self.Tc for tk in self.mtime]
        last_z = self.all_dz[-1][0:self.mrob.u_dim,-1]
        self.last_q = self.mrob.phi1(self.all_dz[-1][:,-1].reshape(
                self.mrob.l+1, self.mrob.u_dim).T)
        self.last_u = self.mrob.phi2(self.all_dz[-1][:,-1].reshape(
                self.mrob.l+1, self.mrob.u_dim).T)

#        for i in range(len(self.last_u)):
#            if abs(self.last_u[i]) > self.u_abs_max[i]:
#                self.last_u[i] = mth.copysign(self.u_abs_max[i], self.last_u[i])

        self.itcount += 1
    #endwhile
    
    self.detected_obst_idxs = self._detected_obst_idx(last_z)

    # initiate ctrl points (straight line towards final z)
    C[:,0] =np.array(np.linspace(last_z[0,0],\
            final_z[0,0], self.n_ctrlpts)).T
    C[:,1] =np.array(np.linspace(last_z[1,0],\
            final_z[1,0], self.n_ctrlpts)).T
    x_aux = np.append(np.asarray([self.Tp]), np.squeeze(C.reshape(1,self.n_ctrlpts*self.mrob.u_dim)), axis=1)

    self._lstep_plot_update(x_aux)

    tic = time.time()

    while True:
        if False:
            # Define the optimization problem
            self.opt_prob = pyOpt.Optimization(
                    'Faster path with obstacles', # name of the problem
                    self._lstep_obj_func) # object function (criterium, eq. and ineq.)
        
            self.opt_prob.addObj('J')
        
            self.opt_prob.addVarGroup( # minimization arguments
                    'x',
                    self.mrob.u_dim*self.n_ctrlpts+1, # dimension
                    'c', # continous
                    lower=[0.0]+list(C_lower),
                    value=[self.Tp]+list(np.squeeze(C.reshape(1,self.n_ctrlpts*self.mrob.u_dim))),
                    upper=[1e10]+list(C_upper))
        
            self.opt_prob.addConGroup( # equations constraints
                    'ec',
                    2*self.mrob.q_dim + 2*self.mrob.u_dim, # dimension
                    'e') # equations
        
            self.opt_prob.addConGroup( # inequations constraints
                    'ic',
                    (self.N_s-2)*self.mrob.u_dim +
                            (self.N_s-2)*len(self.detected_obst_idxs), # dimenstion
                    'i') # inequations
        
            # solve constrained optmization
#            solver = pyOpt.PSQP(pll_type='POA')
#            solver.setOption('ACC', 1e-6)
#            solver.setOption('MAXIT', 30)
            solver = pyOpt.ALGENCAN(pll_type='POA')
            solver.setOption('epsfeas', 5e-1)
            solver.setOption('epsopt', 9e-1)
        
            [J, x_aux, information] = solver(self.opt_prob) 
            self._lstep_plot_update(x_aux)
            x_aux = np.array(x_aux)
            break

        else:
            # solve constrained optmization
            outp = fmin_slsqp(self._lstep_criteria,
                                x_aux,
                                eqcons=(),
                                f_eqcons=self._lstep_feqcons,
                                ieqcons=(),
                                f_ieqcons=self._lstep_fieqcons,
                                iprint=1,
                                iter=30,
                                acc=1e-4,
                                full_output=True,
                                callback=self._lstep_plot_update)

            x_aux = outp[0]
            imode = outp[3]
            print('Exit mode: {}'.format(imode))
            if imode != 0 and imode != 9:
                usepyopt = True
                continue
            break

    print('------------------------\nElapsed time for {} (last) iteraction: {}'.format(self.itcount, time.time()-tic))
    print('No of equations unsatisfied: {}'.format(len(self.unsatisf_eq_values)))
    print('Mean and variance of equations unsatisfied: ({},{})'.format(np.mean(self.unsatisf_eq_values), np.std(self.unsatisf_eq_values)))
    print('No of inequations unsatisfied: {}'.format(len(self.unsatisf_ieq_values)))
    print('Mean and variance of inequations unsatisfied: ({},{})'.format(np.mean(self.unsatisf_ieq_values), np.std(self.unsatisf_ieq_values)))

    # test if opt went well
    # if yes
    dt_final = x_aux[0]
    self.t_final = self.mtime[0] + dt_final
    C = x_aux[1:].reshape(self.n_ctrlpts, self.mrob.u_dim)
    # if no
    # continue

    print('Final time: {}'.format(self.t_final))

    # store ctrl points and [z dz ddz](t)
    self.C_ref += [C]

    self.mtime = np.linspace(self.mtime[0], self.t_final, self.N_s)

    dz = self._comb_bsp(self.mtime, C, 0).T
    for dev in range(1,self.mrob.l+1):
        dz = np.append(dz,self._comb_bsp(
                self.mtime, C, dev).T,axis=0)
    self.all_dz += [dz]

  def _detected_obst_idx(self, pos):
    idx_list = []
    for idx in range(0, self.obst_map.shape[0]):
        dist = LA.norm(self.obst_map[idx,0:2].T - pos)
        if dist < self.detection_radius:
            idx_list += [idx]
    return idx_list

  ## Generate b-spline knots
  def _gen_knots(self, t_init, t_fin):
    knots = [t_init]
    for j in range(1,self.d):
      knots_j = t_init
      knots = knots + [knots_j]

    for j in range(self.d,self.d+self.n_knot):
      knots_j = t_init + (j-(self.d-1.0))* \
          (t_fin-t_init)/self.n_knot
      knots = knots + [knots_j]

    for j in range(self.d+self.n_knot,2*self.d-1+self.n_knot):
      knots_j = t_fin
      knots = knots + [knots_j]
    return np.asarray(knots)

  ## Combine base b-splines
  def _comb_bsp(self, t, C, deriv_order):
    tup = (self.knots, np.squeeze(C[:,0].T), self.d-1)
    z = np.matrix(si.splev(t, tup, der=deriv_order)).T
    for i in range(1, self.mrob.u_dim):
      tup = (self.knots, np.squeeze(np.asarray(C[:,i].transpose())), self.d-1)
      z = np.append(z, np.matrix(si.splev(t, tup,
          der=deriv_order)).T, axis=1)
    return z

  def _lstep_obj_func(self, x):

    dt_final = x[0]
    t_final = self.mtime[0]+dt_final
    C = x[1:].reshape(self.n_ctrlpts, self.mrob.u_dim)
    
    # updatating knots
    self.knots = self._gen_knots(self.mtime[0], t_final)

    # creating time
    mtime = np.linspace(self.mtime[0], t_final, self.N_s)

#    self._lstep_plot_update(x)

    # get a list over time of the matrix [z dz ddz](t) t in [tk, tk+Tp]
    dz = self._comb_bsp(mtime, C, 0).T
    for dev in range(1,self.mrob.l+1):
        dz = np.append(dz,self._comb_bsp(mtime, C, dev).T,axis=0)

    dztTp = map(lambda dzt:dzt.reshape(self.mrob.l+1, self.mrob.u_dim).T, dz.T)

    # get a list over time of command values u(t)
    utTp = map(self.mrob.phi2, dztTp)

    # get a list over time of values q(t)
    qtTp = map(self.mrob.phi1, dztTp)

    J = dt_final**2

    # return array where each element is an equation constraint
    # dimension: q_dim + u_dim (=5 equations)
#    print('last U: {}'.format(self.last_u))
#    print('last Q: {}'.format(self.last_q))
    eq_cons = list(np.squeeze(np.array(qtTp[0]-self.last_q)))+\
            list(np.squeeze(np.array(qtTp[-1]-self.q_fin)))+\
            list(np.squeeze(np.array(utTp[0]-self.last_u)))+\
            list(np.squeeze(np.array(utTp[-1]-self.u_fin)))

    # Count how many equations are not respected
    self.unsatisf_eq_values = [eq for eq in eq_cons if eq != 0]

    ## Obstacles constraints
    # (N_s-2)*nb_obst_detected. The -2 is to account for initial and final cond.
    obst_cons = []
    for m in self.detected_obst_idxs:
        obst_cons += [-1.0*LA.norm(self.obst_map[m,0:-1].T-qt[0:2,0]) \
          + (self.mrob.rho + self.obst_map[m,-1]) for qt in qtTp[1:-1]]

    ## Max speed constraints
    # (N_s-2)*u_dim inequations. The -2 is to account for initial and final cond.
    max_speed_cons = list(itertools.chain.from_iterable(
        map(lambda ut:[-1.0*self.u_abs_max[i,0]+abs(ut[i,0])\
        for i in range(self.mrob.u_dim)], utTp[1:-1])))

    # Create final array
    ieq_cons = obst_cons + max_speed_cons

    # Count how many equations are not respected
    self.unsatisf_ieq_values = [ieq for ieq in ieq_cons if eq > 0]

    cons = eq_cons + ieq_cons

    return J, np.asarray(cons), 0

  def _obj_func(self, x):

#    self._plot_update(x)

    C = x.reshape(self.n_ctrlpts, self.mrob.u_dim)
    
    # get a list over time of the matrix [z dz ddz](t) t in [tk, tk+Tp]
    dz = self._comb_bsp(self.mtime, C, 0).T
    for dev in range(1,self.mrob.l+1):
        dz = np.append(dz,self._comb_bsp(self.mtime, C, dev).T,axis=0)

    # reshape data
    dztTp = map(lambda dzt:dzt.reshape(self.mrob.l+1, self.mrob.u_dim).T, dz.T)

    # get a list over time of command values u(t)
    utTp = map(self.mrob.phi2, dztTp)

    # get a list over time of values q(t)
    qtTp = map(self.mrob.phi1, dztTp)

    J = LA.norm(qtTp[-1] - self.q_fin)**2

    # return array where each element is an equation constraint
    # dimension: q_dim + u_dim (=5 equations)
#    print('last U: {}'.format(self.last_u))
#    print('last Q: {}'.format(self.last_q))
    eq_cons = list(np.squeeze(np.array(qtTp[0]-self.last_q)))+\
           list(np.squeeze(np.array(utTp[0]-self.last_u)))

    # Count how many equations are not respected
    self.unsatisf_eq_values = [eq for eq in eq_cons if eq != 0]

    ## Obstacles constraints
    # (N_s-1)*nb_obst_detected
    obst_cons = []
    for m in self.detected_obst_idxs:
        obst_cons += [-1.0*LA.norm(self.obst_map[m,0:-1].T-qt[0:2,0]) \
          + (self.mrob.rho + self.obst_map[m,-1]) for qt in qtTp[1:]]

    ## Max speed constraints
    # (N_s-1)*u_dim inequations
    max_speed_cons = list(itertools.chain.from_iterable(
        map(lambda ut:[-1.0*self.u_abs_max[i,0]+abs(ut[i,0])\
        for i in range(self.mrob.u_dim)], utTp[1:])))

    # Create final array
    ieq_cons = obst_cons + max_speed_cons

    # Count how many equations are not respected
    self.unsatisf_ieq_values = [ieq for ieq in ieq_cons if eq > 0]
#    print(len(self.unsatisf_ieq_values))
#    print(ieq_cons)
#    print(len(ieq_cons))

    cons = eq_cons + ieq_cons

    return J, np.asarray(cons), 0

  def _lstep_criteria(self, x):
    dt_final = x[0]
    t_final = self.mtime[0]+dt_final
#    C = x[1:].reshape(self.n_ctrlpts, self.mrob.u_dim)
#
#    self.knots = self._gen_knots(self.mtime[0], t_final)
#    
#    dz = self._comb_bsp(t_final, C, 0).T
#    for dev in range(1,self.mrob.l+1):
#        dz = np.append(dz,self._comb_bsp(t_final, C, dev).T,axis=1)
#    qTp = self.mrob.phi1(dz)
    return \
            (t_final**2)
#            LA.norm(qTp - self.q_fin)**2\
#            +\

#  def _lstep_criteria(self, x):
#    dt_final = x[0]
#    t_final = self.mtime[0]+dt_final
#    C = x[1:].reshape(self.n_ctrlpts, self.mrob.u_dim)
#    
#    return t_final**2 # delta t for the last section of the path

  def _lstep_feqcons(self, x):
    dt_final = x[0]
    t_final = self.mtime[0]+dt_final
    C = x[1:].reshape(self.n_ctrlpts, self.mrob.u_dim)
    
    self.knots = self._gen_knots(self.mtime[0], t_final)

    dztinit = self._comb_bsp(self.mtime[0], C, 0).T
    for dev in range(1,self.mrob.l+1):
        dztinit = np.append(dztinit,self._comb_bsp(self.mtime[0], C, dev).T,axis=1)

    # get matrix [z dz ddz](t_final)
    dztfinal = self._comb_bsp(t_final, C, 0).T
    for dev in range(1,self.mrob.l+1):
        dztfinal=np.append(dztfinal,self._comb_bsp(t_final, C, dev).T,axis=1)

    #----------------------------------------------------------------------
    # Final and initial values constraints
    #----------------------------------------------------------------------
    eq_cons = list(np.squeeze(np.array(self.mrob.phi1(dztinit)-self.last_q)))+\
            list(np.squeeze(np.array(self.mrob.phi1(dztfinal)-self.q_fin)))+\
            list(np.squeeze(np.array(self.mrob.phi2(dztinit)-self.last_u)))+\
            list(np.squeeze(np.array(self.mrob.phi2(dztfinal)-self.u_fin)))

    self.unsatisf_eq_values = [ec for ec in eq_cons if ec != 0]
    return np.asarray(eq_cons)

  def _lstep_fieqcons(self, x):
    dt_final = x[0]
    t_final = self.mtime[0]+dt_final
    C = x[1:].reshape(self.n_ctrlpts, self.mrob.u_dim)
    
    self.knots = self._gen_knots(self.mtime[0], t_final)

    mtime = np.linspace(self.mtime[0], t_final, self.N_s)

    # get a list over time of the matrix [z dz ddz](t) t in [tk, tk+Tp]
    dz = self._comb_bsp(mtime, C, 0).T
    for dev in range(1,self.mrob.l+1):
        dz = np.append(dz,self._comb_bsp(mtime, C, dev).T,axis=0)

    dztTp = map(lambda dzt:dzt.reshape(self.mrob.l+1, self.mrob.u_dim).T, dz.T)

    # get a list over time of command values u(t)
    utTp = map(self.mrob.phi2, dztTp)

    # get a list over time of values q(t)
    qtTp = map(self.mrob.phi1, dztTp)
    
    ## Obstacles constraints
    # (N_s-2)*nb_obst_detected
    obst_cons = []
    for m in self.detected_obst_idxs:
        obst_cons += [LA.norm(self.obst_map[m,0:-1].T-qt[0:2,0]) \
          - (self.mrob.rho + self.obst_map[m,-1]) for qt in qtTp[1:-1]]

    ## Max speed constraints
    # (N_s-2)*u_dim inequations
    max_speed_cons = list(itertools.chain.from_iterable(
        map(lambda ut:[self.u_abs_max[i,0]-abs(ut[i,0])\
        for i in range(self.mrob.u_dim)],utTp[1:-1])))

    # Create final array
    ieq_cons = obst_cons + max_speed_cons
    
    # Count how many inequations are not respected
    self.unsatisf_ieq_values = [ieq for ieq in ieq_cons if ieq < 0]
    return np.asarray(ieq_cons)

  ##------------------------------Cost Function--------------------------------
  #----------------------------------------------------------------------------
  def _criteria(self, x):
    C = x.reshape(self.n_ctrlpts, self.mrob.u_dim)
    
    dz = self._comb_bsp(self.mtime[-1], C, 0).T
    for dev in range(1,self.mrob.l+1):
        dz = np.append(dz,self._comb_bsp(self.mtime[-1], C, dev).T,axis=1)
    qTp = self.mrob.phi1(dz)
    return 1e1*(LA.norm(qTp - self.q_fin))**2

  ##------------------------Constraints Equations------------------------------
  #----------------------------------------------------------------------------
  def _feqcons(self, x):
    C = x.reshape(self.n_ctrlpts, self.mrob.u_dim)

    dzt = self._comb_bsp(self.mtime[0], C, 0).T
    for dev in range(1,self.mrob.l+1):
        dzt = np.append(dzt,self._comb_bsp(self.mtime[0], C, dev).T,axis=1)

    # return array where each element is an equation constraint
    # dimension: q_dim + u_dim (=5 equations)
    eq_cons = list(np.squeeze(np.array(self.mrob.phi1(dzt)-self.last_q)))+\
           list(np.squeeze(np.array(self.mrob.phi2(dzt)-self.last_u)))

    # Count how many equations are not respected
    self.unsatisf_eq_values = [eq for eq in eq_cons if eq != 0]
    return np.asarray(eq_cons)

  ##------------------------Constraints Inequations----------------------------
  #----------------------------------------------------------------------------
  def _fieqcons(self, x):
    # get time and control points from U array
    C = x.reshape(self.n_ctrlpts, self.mrob.u_dim)

    # get a list over time of the matrix [z dz ddz](t) t in [tk, tk+Tp]
    dz = self._comb_bsp(self.mtime, C, 0).T
    for dev in range(1,self.mrob.l+1):
        dz = np.append(dz,self._comb_bsp(self.mtime, C, dev).T,axis=0)

    dztTp = map(lambda dzt:dzt.reshape(self.mrob.l+1, self.mrob.u_dim).T, dz.T)

    # get a list over time of command values u(t)
    utTp = map(self.mrob.phi2, dztTp)

    # get a list over time of values q(t)
    qtTp = map(self.mrob.phi1, dztTp)

#    print('q: {}'.format(qtTp[0][0:2,0]))
#    print('dz: {}'.format(dztTp[0][0:2,0]))
#    print('LA:: {}'.format(LA.norm(self.obst_map[self.detected_obst_idxs[0],0:-1].T-dztTp[0][0:2,0])-self.mrob.rho-self.obst_map[self.detected_obst_idxs[0], -1]))

    ## Obstacles constraints
    # (N_s-1)*nb_obst_detected
    obst_cons = []
    for m in self.detected_obst_idxs:
        obst_cons += [LA.norm(self.obst_map[m,0:-1].T-qt[0:2,0]) \
          - (self.mrob.rho + self.obst_map[m,-1]) for qt in qtTp[1:]]

    ## Max speed constraints
    # (N_s-1)*u_dim inequations
    max_speed_cons = list(itertools.chain.from_iterable(
        map(lambda ut:[self.u_abs_max[i,0]-abs(ut[i,0])\
        for i in range(self.mrob.u_dim)],utTp[1:])))

    # Create final array
    ieq_cons = obst_cons + max_speed_cons

    # Count how many inequations are not respected
    unsatisf_list = [ieq for ieq in ieq_cons if ieq < 0]
    self.unsatisf_ieq_values = unsatisf_list
#    print(obst_cons)
#    print(len(obst_cons))
#    print(self.detected_obst_idxs)
    # return arrray where each element is an inequation constraint
    return np.asarray(ieq_cons)

  def _plot_update(self, x):
    C = x.reshape(self.n_ctrlpts, self.mrob.u_dim)

    seg_pts = [] # segmentation red diamonds for the path

    # reshape data
    dztlist = []
    for dz in self.all_dz:
        seg_pts += [dz[0:2,-1].T]
        dztlist += map(lambda dzt:dzt.reshape(self.mrob.l+1, self.mrob.u_dim).T, dz.T)

    dz = self._comb_bsp(self.mtime, C, 0).T
    for dev in range(1,self.mrob.l+1):
        dz = np.append(dz,self._comb_bsp(self.mtime, C, dev).T,axis=0)

    dztlist += map(lambda dzt:dzt.reshape(self.mrob.l+1, self.mrob.u_dim).T, dz.T)

    # dztlist = [array3x3_t0, array3x3_t1, ..]

    # get a list over time of command values u(t)
    u = map(self.mrob.phi2, dztlist)

    # get a list over time of values q(t)
    q = map(self.mrob.phi1, dztlist)

    curvex = map(lambda x:x[0,0], q)
    curvey = map(lambda y:y[1,0], q)
    linspeed = map(lambda x:x[0,0], u)
    angspeed = map(lambda y:y[1,0], u)

    # transform list of segmentation points in an array
    if seg_pts != []:
        seg_arr = seg_pts[0]
        for i in range(1, len(self.all_dz)):
            seg_arr = np.append(seg_arr, seg_pts[i], axis=0)
        self.seg_pts.set_xdata(seg_arr[:,0])
        self.seg_pts.set_ydata(seg_arr[:,1])

    if self.all_rejected_z != []:
        reject = self.all_rejected_z[0].T
        for prejpath in self.all_rejected_z[1:]:
            reject = np.append(reject, prejpath.T, axis=0)
        self.rejected_path.set_xdata(reject[:,0])
        self.rejected_path.set_ydata(reject[:,1])
    self.plt_curve.set_xdata(curvex)
    self.plt_curve.set_ydata(curvey)
    self.plt_dot_curve.set_xdata(curvex)
    self.plt_dot_curve.set_ydata(curvey)
    self.plt_ctrl_pts.set_xdata(C[:,0])
    self.plt_ctrl_pts.set_ydata(C[:,1])
    ax = self.fig.gca()
    ax.relim()
    ax.autoscale_view(True,True,True)

    self.fig.canvas.draw()

    axarray = self.fig_speed[0].axes
    mtime = np.linspace(self.t_init, self.mtime[-1], len(dztlist))
    self.plt_linspeed.set_xdata(mtime)
    self.plt_angspeed.set_xdata(mtime)
    self.plt_linspeed.set_ydata(linspeed)
    self.plt_angspeed.set_ydata(angspeed)
    axarray[0].relim()
    axarray[1].relim()
    axarray[0].autoscale_view(True,True,True)
    axarray[1].autoscale_view(True,True,True)
    self.fig_speed[0].canvas.draw()

  def _lstep_plot_update(self, x):
    dt_final = x[0]
    C = x[1:].reshape(self.n_ctrlpts, self.mrob.u_dim)
    t_final = self.mtime[0] + dt_final
    self.knots = self._gen_knots(self.mtime[0], t_final)
    mtime = np.linspace(self.mtime[0], t_final, self.N_s)

    seg_pts = [] # segmentation red diamonds for the path

    # reshape data
    dztlist = []
    for dz in self.all_dz:
        seg_pts += [dz[0:2,-1].T]
        dztlist += map(lambda dzt:dzt.reshape(self.mrob.l+1, self.mrob.u_dim).T, dz.T)

    dz = self._comb_bsp(mtime, C, 0).T
    for dev in range(1,self.mrob.l+1):
        dz = np.append(dz,self._comb_bsp(mtime, C, dev).T,axis=0)

    dztlist += map(lambda dzt:dzt.reshape(self.mrob.l+1, self.mrob.u_dim).T, dz.T)

    # dztlist = [array3x3_t0, array3x3_t1, ..]

    # get a list over time of command values u(t)
    u = map(self.mrob.phi2, dztlist)

    # get a list over time of values q(t)
    q = map(self.mrob.phi1, dztlist)

    curvex = map(lambda x:x[0,0], q)
    curvey = map(lambda y:y[1,0], q)
    linspeed = map(lambda x:x[0,0], u)
    angspeed = map(lambda y:y[1,0], u)

    # transform list of segmentation points in an array
    if seg_pts != []:
        seg_arr = seg_pts[0]
        for i in range(1, len(self.all_dz)):
            seg_arr = np.append(seg_arr, seg_pts[i], axis=0)
        self.seg_pts.set_xdata(seg_arr[:,0])
        self.seg_pts.set_ydata(seg_arr[:,1])
    
    if self.all_rejected_z != []:
        reject = self.all_rejected_z[0].T
        for prejpath in self.all_rejected_z[1:]:
            reject = np.append(reject, prejpath.T, axis=0)
        self.rejected_path.set_xdata(reject[:,0])
        self.rejected_path.set_ydata(reject[:,1])

    self.plt_curve.set_xdata(curvex)
    self.plt_curve.set_ydata(curvey)
    self.plt_dot_curve.set_xdata(curvex)
    self.plt_dot_curve.set_ydata(curvey)
    self.plt_ctrl_pts.set_xdata(C[:,0])
    self.plt_ctrl_pts.set_ydata(C[:,1])
    ax = self.fig.gca()
    ax.relim()
    ax.autoscale_view(True,True,True)
    self.fig.canvas.draw()

    axarray = self.fig_speed[0].axes
    wholetime = np.linspace(self.t_init, mtime[-1], len(dztlist))
    self.plt_linspeed.set_xdata(wholetime)
    self.plt_angspeed.set_xdata(wholetime)
    self.plt_linspeed.set_ydata(linspeed)
    self.plt_angspeed.set_ydata(angspeed)
    axarray[0].relim()
    axarray[1].relim()
    axarray[0].autoscale_view(True,True,True)
    axarray[1].autoscale_view(True,True,True)
    self.fig_speed[0].canvas.draw()

  def _improve_init_guess(self):
    mag = LA.norm(self.ctrl_pts[0]-self.ctrl_pts[1])
    self.ctrl_pts[1]+np.matrix([1,2])
    dx = mag*np.cos(self.k_mod.q_init[-1,0])
    dy = mag*np.sin(self.k_mod.q_init[-1,0])
    self.ctrl_pts[1] = self.ctrl_pts[0] + np.matrix([dx, dy])

    dx = mag*np.cos(self.k_mod.q_final[-1,0])
    dy = mag*np.sin(self.k_mod.q_final[-1,0])
    self.ctrl_pts[-2] = self.ctrl_pts[-1] - np.matrix([dx, dy])


##-----------------------------------------------------------------------------
## Initializations

tic = time.clock()
trajc = Trajectory_Generation(Unicycle_Kine_Model())
toc = time.clock()

#mtime = trajc._gen_time(trajc.t_fin)
#curve = trajc._gen_dtraj(trajc.U, 0)

# get a list over time of the matrix [z dz ddz](t)
#all_zl = [np.append(np.append(
#    trajc._comb_bsp(t, trajc.C, 0).transpose(),
#    trajc._comb_bsp(t, trajc.C, 1).transpose(), axis = 1),
#    trajc._comb_bsp(t, trajc.C, 2).transpose(), axis = 1) for t in mtime]

# get a list over time of command values u(t)
#all_us = map(trajc.mrob.phi2, all_zl)
#linspeed = map(lambda x:x[0], all_us)
#angspeed = map(lambda x:x[1], all_us)

#Print('Elapsed time: {}'.format(toc-tic))
#Print('Final t_fin: {}'.format(trajc.t_fin))
#Print('Number of unsatisfied equations: {}'.format(len(trajc.unsatisf_eq_values)))
#Print('Number of unsatisfied inequations: {}'.format(len(trajc.unsatisf_ieq_values)))
#Print('Mean and standard deviation of equations diff: ({},{})'.format(np.mean(trajc.unsatisf_eq_values), np.std(trajc.unsatisf_eq_values)))
#Print('Mean and standard deviation of inequations diff: ({},{})'.format(np.mean(trajc.unsatisf_ieq_values), np.std(trajc.unsatisf_ieq_values)))

## Plot final speeds

#f, axarr = plt.subplots(2)
#axarr[0].plot(mtime, map(lambda x:x[0,0], linspeed))
#axarr[0].set_xlabel('time(s)')
#axarr[0].set_ylabel('v(m/s)')
#axarr[0].set_title('Linear speed')
#
#axarr[1].plot(mtime, map(lambda x:x[0,0], angspeed))
#axarr[1].set_xlabel('time(s)')
#axarr[1].set_ylabel('w(rad/s)')
#axarr[1].set_title('Angular speed')
#axarr[0].grid()
#axarr[1].grid()
#
plt.show(block=True)
