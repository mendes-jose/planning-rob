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

# Unicycle Kinematic Model ----------------------------------------------------
class Unicycle_Kine_Model(object):
  def __init__(self):
    self.u_dim = 2
    self.q_dim = 3
    self.rho = 0.2 # m
    self.u_abs_max = np.matrix('0.5; 5')

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
    self.N_s = 20 # nb of samples for discretization
    self.n_knot = 6 # nb of non zero lenght intervals of the knot series
    self.t_init = 0.0
    self.Tc = 1.0
    self.Tp = 2.5
    Tc_idx = int(round(self.Tc/(self.Tp-self.t_init)*(self.N_s-1)))
    self.detection_radius = 6.0

  ## Mobile Robot object
    self.mrob = mrobot

  ## Other parameters
    self.d = self.mrob.l+2 # B-spline order (integer | d > l+1)
    self.n_ptctrl = self.n_knot + self.d - 1 # nb of ctrl points

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
    self.obst_map = np.append(self.obst_map, np.matrix([0.2,  0.7,   0.1]),
        axis = 0)
    self.obst_map = np.append(self.obst_map, np.matrix([-0.5, 1.5, 0.3]),
        axis = 0)
#    self.obst_map = np.append(self.obst_map, np.matrix([1.6, 4.3, 0.2]),
#        axis = 0)
    
    # max distance within Tp
    self.D = self.Tp * self.u_abs_max[0,0]

    # Ctrl pts init
    C = np.array(np.zeros((self.n_ptctrl,self.mrob.u_dim)))
    
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
#    self.fig_speed = plt.figure()
    ax = self.fig.gca()

    # creating obstacles' circles
    circ = []
    for r in range(self.obst_map.shape[0]):
      # external dashed circles
      circ = circ + \
          [plt.Circle((self.obst_map[r,0], self.obst_map[r,1]),
          self.obst_map[r,2]+self.mrob.rho,color='r',ls = 'dashed',fill=False)]
      # internal continous circles
      circ = circ + \
          [plt.Circle((self.obst_map[r,0], self.obst_map[r,1]),
          self.obst_map[r,2], color='r', fill=False)]

    # adding circles to axis
    [ax.add_artist(c) for c in circ]

    # plot curve and its control points
    self.plt_ctrl_pts,self.plt_curve,self.plt_dot_curve = ax.plot(
            0.0, 0.0, '*',\
            0.0, 0.0, 'b-',\
            0.0, 0.0, 'g.')
    
    # formating figure
    plt.xlabel('x(m)')
    plt.ylabel('y(m)')
    plt.title('Generated trajectory')
    ax.axis('equal')

    final_z = self.mrob.phi0(self.q_fin)
    self.last_q = self.q_init
    self.last_u = self.u_init
    last_z = self.mrob.phi0(self.last_q)
    self.all_dz = []
    while LA.norm(last_z - final_z) > 0.1:

        self.detected_obst_idxs = self._detected_obst_idx(last_z)
        print('No of detected obst: {}'.format(len(self.detected_obst_idxs)))
        print('Detected obst: {}'.format(self.obst_map[self.detected_obst_idxs,:]))

        # initiate ctrl points (straight line towards final z)
        direc = final_z - last_z
        direc = direc/LA.norm(direc)
        C[:,0] =np.array(np.linspace(last_z[0,0],\
                last_z[0,0]+self.D*direc[0,0], self.n_ptctrl)).T
        C[:,1] =np.array(np.linspace(last_z[1,0],\
                last_z[1,0]+self.D*direc[1,0], self.n_ptctrl)).T

        # solve constrained optmization
        C = fmin_slsqp(self._criteria,
                            C.reshape(1,self.n_ptctrl*self.mrob.u_dim),
                            eqcons=(),
                            f_eqcons=self._feqcons,
                            ieqcons=(),
                            f_ieqcons=self._fieqcons,
                            iprint=1,
                            iter=15,
                            acc=1e-2,
                            callback=self._plot_update)
    
        C = C.reshape(self.n_ptctrl, self.mrob.u_dim)
        print(C)
        raw_input('Ola')
        # store ctrl points and [z dz ddz](t)
        self.C_ref += [C]
        dz = self._comb_bsp(self.mtime[0:Tc_idx], C, 0).T
        for dev in range(1,self.mrob.l+1):
            dz = np.append(dz,self._comb_bsp(
                    self.mtime[0:Tc_idx], C, dev).T,axis=0)
        self.all_dz += [dz]

        plt.figure()
        my_dz = [self.all_dz[-1][:,i].reshape(self.mrob.l+1, self.mrob.u_dim).T \
                for i in range(len(self.mtime[0:Tc_idx]))]
        my_u = map(self.mrob.phi2, my_dz)
        linspeed = map(lambda x:x[0,0], my_u)
        angspeed = map(lambda x:x[1,0], my_u)
        plt.plot(self.mtime[0:Tc_idx], linspeed)

        # update needed values
        self.knots = self.knots + self.Tc
        self.mtime = [tk+self.Tc for tk in self.mtime]
        last_z = self.all_dz[-1][0:self.mrob.u_dim,-1]
        print('Last z: {}', last_z)
        self.last_q = self.mrob.phi1(self.all_dz[-1][:,-1].reshape(
                self.mrob.l+1, self.mrob.u_dim).T)
        self.last_u = self.mrob.phi2(self.all_dz[-1][:,-1].reshape(
                self.mrob.l+1, self.mrob.u_dim).T)

    #endwhile

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

  ## Generate the trajectory
#  def _gen_dtraj(self, U, deriv_order):
#    t_fin = U[0]
#    C = np.asmatrix(U[1:].reshape(self.n_ptctrl, self.mrob.u_dim))
#    t = np.asarray(self._gen_time(t_fin))
#    self.knots = self._gen_knots(t_fin)
#    return self._comb_bsp(t, C, deriv_order)

  ##------------------------------Cost Function--------------------------------
  #----------------------------------------------------------------------------
  def _criteria(self, x):
    C = x.reshape(self.n_ptctrl, self.mrob.u_dim)
    
    dz = self._comb_bsp(self.mtime[-1], C, 0).T
    for dev in range(1,self.mrob.l+1):
        dz = np.append(dz,self._comb_bsp(self.mtime[-1], C, dev).T,axis=1)
    qTp = self.mrob.phi1(dz)
    return LA.norm(qTp - self.q_fin)**2

  ##------------------------Constraints Equations------------------------------
  #----------------------------------------------------------------------------
  def _feqcons(self, x):
    C = x.reshape(self.n_ptctrl, self.mrob.u_dim)

    dzt = self._comb_bsp(self.mtime[0], C, 0).T
    for dev in range(1,self.mrob.l+1):
        dzt = np.append(dzt,self._comb_bsp(self.mtime[0], C, dev).T,axis=1)

    # return array where each element is an equation constraint
    # dimension: q_dim + u_dim (=5 equations)
    eq_const = list(np.squeeze(np.array(self.mrob.phi1(dzt)-self.last_q)))+\
           list(np.squeeze(np.array(self.mrob.phi2(dzt)-self.last_u)))

    # Count how many equations are not respected
    unsatisf_list = [eq for eq in eq_const if eq != 0]
    self.unsatisf_eq_values = unsatisf_list

#    raw_input('ola')
    return np.array(eq_const)

  ##------------------------Constraints Inequations----------------------------
  #----------------------------------------------------------------------------
  def _fieqcons(self, x):
    # get time and control points from U array
    C = x.reshape(self.n_ptctrl, self.mrob.u_dim)

    # get a list over time of the matrix [z dz ddz](t) t in [tk, tk+Tp]
    dz = self._comb_bsp(self.mtime, C, 0).T
    for dev in range(1,self.mrob.l+1):
        dz = np.append(dz,self._comb_bsp(self.mtime, C, dev).T,axis=0)

    dztTp = map(lambda dzt:dzt.reshape(self.mrob.l+1, self.mrob.u_dim).T, dz.T)

    # get a list over time of command values u(t)
    utTp = map(self.mrob.phi2, dztTp)

    # get a list over time of values q(t)
    qtTp = map(self.mrob.phi1, dztTp)

    ## Obstacles constraints
    # N_s*nb_obst_detected
    obst_cons = []
#    for m in self.detected_obst_idxs:
#        obst_cons += [LA.norm(self.obst_map[m,0:-1].T-qt[0:2,0]) \
#          - (self.mrob.rho + self.obst_map[m,-1]) for qt in qtTp]

    ## Max speed constraints
    # N_s*u_dim inequations
    max_speed_cons = list(itertools.chain.from_iterable(
        map(lambda ut:[self.u_abs_max[i,0]-abs(ut[i,0])\
        for i in range(self.mrob.u_dim)],utTp)))

    # Create final array
    ieq_const = obst_cons + max_speed_cons

    # Count how many inequations are not respected
    unsatisf_list = [ieq for ieq in ieq_const if ieq < 0]
    self.unsatisf_ieq_values = unsatisf_list

    # return arrray where each element is an inequation constraint
    return np.array(ieq_const)

  def _plot_update(self, x):
    C = x.reshape(self.n_ptctrl, self.mrob.u_dim)
    curve = self._comb_bsp(self.mtime, C, 0)
    for past_path in reversed(self.all_dz):
        curve = np.append(past_path[0:self.mrob.u_dim,:].T, curve, axis=0)
    self.plt_curve.set_xdata(curve[:,0])
    self.plt_curve.set_ydata(curve[:,1])
    self.plt_dot_curve.set_xdata(curve[:,0])
    self.plt_dot_curve.set_ydata(curve[:,1])
    self.plt_ctrl_pts.set_xdata(C[:,0])
    self.plt_ctrl_pts.set_ydata(C[:,1])
    ax = self.fig.gca()
    ax.relim()
    ax.autoscale_view(True,True,True)
    self.fig.canvas.draw()

##-----------------------------------------------------------------------------
## Initializations

tic = time.clock()
trajc = Trajectory_Generation(Unicycle_Kine_Model())
toc = time.clock()

mtime = trajc._gen_time(trajc.t_fin)
#curve = trajc._gen_dtraj(trajc.U, 0)

# get a list over time of the matrix [z dz ddz](t)
all_zl = [np.append(np.append(
    trajc._comb_bsp(t, trajc.C, 0).transpose(),
    trajc._comb_bsp(t, trajc.C, 1).transpose(), axis = 1),
    trajc._comb_bsp(t, trajc.C, 2).transpose(), axis = 1) for t in mtime]

# get a list over time of command values u(t)
all_us = map(trajc.mrob.phi2, all_zl)
linspeed = map(lambda x:x[0], all_us)
angspeed = map(lambda x:x[1], all_us)

print('Elapsed time: {}'.format(toc-tic))
print('Final t_fin: {}'.format(trajc.t_fin))
print('Number of unsatisfied equations: {}'.format(len(trajc.unsatisf_eq_values)))
print('Number of unsatisfied inequations: {}'.format(len(trajc.unsatisf_ieq_values)))
print('Mean and standard deviation of equations diff: ({},{})'.format(np.mean(trajc.unsatisf_eq_values), np.std(trajc.unsatisf_eq_values)))
print('Mean and standard deviation of inequations diff: ({},{})'.format(np.mean(trajc.unsatisf_ieq_values), np.std(trajc.unsatisf_ieq_values)))

## Plot final speeds

f, axarr = plt.subplots(2)
axarr[0].plot(mtime, map(lambda x:x[0,0], linspeed))
axarr[0].set_xlabel('time(s)')
axarr[0].set_ylabel('v(m/s)')
axarr[0].set_title('Linear speed')

axarr[1].plot(mtime, map(lambda x:x[0,0], angspeed))
axarr[1].set_xlabel('time(s)')
axarr[1].set_ylabel('w(rad/s)')
axarr[1].set_title('Angular speed')
axarr[0].grid()
axarr[1].grid()

plt.show(block=True)
