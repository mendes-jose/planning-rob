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

  # z here is a list of matrix [z dz ddz]
  def phi1(self, z):
    if z.shape >= (self.u_dim, self.l+1):
      return np.append(z[:,0], \
                       np.matrix(np.arctan2(z[1,1], z[0,1])), axis = 0)
    else:
      # TODO ERROR
      print('ERROR')
      return -1

  # z here is a list of matrix [z dz ddz]
  def phi2(self, z):
    if z.shape >= (self.u_dim, self.l+1):
      if (z[0,1]**2 + z[1,1]**2 != 0):
        return np.matrix([[LA.norm(z[:,1])], \
            [(z[0,1]*z[1,2]-z[1,1]*z[0,2] \
            )/(z[0,1]**2 + z[1,1]**2)]])
      else:
        return np.matrix([[LA.norm(z[:,1])],[0.0]])
    else:
      # TODO ERROR
      print('ERROR')
      return -1

#  def angle(x):
#    pi = math.pi
#    twopi = 2*pi
#    return (x+pi)%twopi-pi

# Trajectory Generation -------------------------------------------------------
class Trajectory_Generation(object):
  def __init__(self, mrobot):

  ## "Arbitrary" parameters
    self.N_s = 100 # nb of samples for discretization
    self.n_knot = 15 # nb of non zero lenght intervals of the knot series
    self.t_init = 0.0

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
    self.q_fin = np.matrix([[0.0], [5.0], [np.pi/2]])
    # initial control input
    self.u_init = np.matrix([[0.0], [0.0]])
    # final control input
    self.u_fin = np.matrix([[0.0], [0.0]])
  ## ...for inequations:
    # Control boundary
    self.u_abs_max = self.mrob.u_abs_max
    # Obstacles (Q_occupied)
    # TODO: Obstacles random generation
    self.obst_map =                          np.matrix([0.0, 2.5, 0.3])
#    self.obst_map = np.append(self.obst_map, np.matrix([2.3,  2.5, 0.5]),
#        axis = 0)
#    self.obst_map = np.append(self.obst_map, np.matrix([1.25, 3,   0.1]),
#        axis = 0)
#    self.obst_map = np.append(self.obst_map, np.matrix([0.3,  1,   0.1]),
#        axis = 0)
#    self.obst_map = np.append(self.obst_map, np.matrix([-0.5, 1.5, 0.3]),
#        axis = 0)

  ## Unknown parameters (defining initial value)
    # TODO: chose a good way to initiate unknow parameters
    # Initiate t_final with :
    # (dist between intial and final positions) / (linear speed max value)
    self.t_fin = LA.norm(self.q_init[0:-1,0]-self.q_fin[0:-1,0])/ \
        self.u_abs_max[0,0]
    print('First guess for t_fin', self.t_fin)
    #self.t_fin = 20

    # Initiate control points so the robot do a straight line from
    # inital to final positions
    self.C = np.matrix(np.zeros((self.n_ptctrl,self.mrob.u_dim)))
    self.C[:,0] = np.matrix(np.linspace(self.q_init[0,0], self.q_fin[0,0],
        self.n_ptctrl)).transpose()
    self.C[:,1] = np.matrix(np.linspace(self.q_init[1,0], self.q_fin[1,0],
        self.n_ptctrl)).transpose()

  ## Generate initial b-spline knots
    self.knots = self._gen_knots(self.t_fin)
    self.U = np.append(self.t_fin, np.asarray(self.C))
  ## Call SLSQP solver
    # U: argument wich will minimize the criteria given the constraints
    
    self.U = fmin_slsqp(self._criteria,
                        self.U,
                        eqcons=(),
                        f_eqcons=self._feqcons,
                        ieqcons=(),
                        f_ieqcons=self._fieqcons,
                        iprint=2,
                        iter=200)

    print ('Final t_fin', mth.sqrt(self._criteria(self.U)))
    self.t_fin = self.U[0]
    self.C = np.asmatrix(self.U[1:].reshape(self.n_ptctrl, self.mrob.u_dim))

  ## Generate time vector
  def _gen_time(self, t_fin):
    return [j * t_fin/(self.N_s-1) for j in range(0, self.N_s)]

  ## Generate b-spline knots
  def _gen_knots(self, t_fin):
    knots = [self.t_init]
    for j in range(1,self.d):
      knots_j = self.t_init
      knots = knots + [knots_j]

    for j in range(self.d,self.d+self.n_knot):
      knots_j = self.t_init + (j-(self.d-1.0))* \
          (t_fin-self.t_init)/self.n_knot
      knots = knots + [knots_j]

    for j in range(self.d+self.n_knot,2*self.d-1+self.n_knot):
      knots_j = t_fin
      knots = knots + [knots_j]
    return np.asarray(knots)

  ## Combine base b-splines
  def _comb_bsp(self, t, C, deriv_order):
    tup = (self.knots, np.squeeze(np.asarray(C[:,0].transpose())), self.d-1)
    z = np.matrix(si.splev(t, tup, der=deriv_order)).transpose()
    for i in range(self.mrob.u_dim)[1:]:
      tup = (self.knots, np.squeeze(np.asarray(C[:,i].transpose())), self.d-1)
      z = np.append(z, np.matrix(si.splev(t, tup,
          der=deriv_order)).transpose(), axis=1)
    return z

  ## Generate the trajectory
  def _gen_dtraj(self, U, deriv_order):
    t_fin = U[0]
    C = np.asmatrix(U[1:].reshape(self.n_ptctrl, self.mrob.u_dim))
    t = np.asarray(self._gen_time(t_fin))
    self.knots = self._gen_knots(t_fin)
    return self._comb_bsp(t, C, deriv_order)

  ## Calculate the criteria to be minimized
  # We want to minimize the time spent to go from q_init to q_final
  # use a quadratic criterium (temps au carre)
  def _criteria(self, U):
    t_fin = U[0]
    #print('--------------------U---------------', U)
    C = np.asmatrix(U[1:].reshape(self.n_ptctrl, self.mrob.u_dim))
    #print('--------------------C---------------', C)
    S = self._gen_dtraj(U,0)
    #print('--------------------S---------------', S)
    V = self._gen_dtraj(U,1)
    #print('--------------------V---------------', V)
    # Integration to find the time spent to go from q_init to q_final
    # TODO: Improve this integration
    self.t_fin=sum(LA.norm(S[ind-1,:]-S[ind,:])/LA.norm(V[ind-1,:]/2.0+V[ind,:]/2.0) \
        for ind in range(1, S.shape[0]))
    return (self.t_fin-self.t_init)**2
    
#  def _criteria(self, U):
#    t_fin = U[0]
#    # Integration to find the time spent to go from q_init to q_final
#    # TODO: Improve this integration
#    return (t_fin-self.t_init)**2

  ## Constraint Equations
  def _feqcons(self, U):
    t_fin = U[0]
    C = np.asmatrix(U[1:].reshape(self.n_ptctrl, self.mrob.u_dim))

    # updtate knots
    self.knots = self._gen_knots(t_fin)

    # get the matrix [z dz ddz] at t_initial
    zl_t_init = np.append(np.append( \
        self._comb_bsp(self.t_init, C, 0).transpose(),
        self._comb_bsp(self.t_init, C, 1).transpose(), axis = 1),
        self._comb_bsp(self.t_init, C, 2).transpose(), axis = 1)

    # get the matrix [z dz ddz] at t_final
    zl_t_fin = np.append(np.append(
        self._comb_bsp(t_fin, C, 0).transpose(),
        self._comb_bsp(t_fin, C, 1).transpose(), axis = 1),
        self._comb_bsp(t_fin, C, 2).transpose(), axis = 1)

    # return array where each element is an equation constraint
    # dimension: 2*q_dim + 2*u_dim (=10 equations)
    ret = np.append(np.append(np.append(
           np.asarray(self.mrob.phi1(zl_t_init)-self.q_init),
           np.asarray(self.mrob.phi1(zl_t_fin)-self.q_fin)),
           np.asarray(self.mrob.phi2(zl_t_init)-self.u_init)),
           np.asarray(self.mrob.phi2(zl_t_fin)-self.u_fin))
    #print ('EQCONS:', ret)
    #print('ERROR EQUATIONS CONS:', LA.norm(ret))
    return ret

  ## Constraints Inequations
  def _fieqcons(self, U):
    t_fin = U[0]
    C = np.asmatrix(U[1:].reshape(self.n_ptctrl, self.mrob.u_dim))

    # updtate knots
    self.knots = self._gen_knots(t_fin)

    mtime = self._gen_time(t_fin)

    # get a list over time of the matrix [z dz ddz](t)
    all_zl = [np.append(np.append(
        self._comb_bsp(t, C, 0).transpose(),
        self._comb_bsp(t, C, 1).transpose(), axis = 1),
        self._comb_bsp(t, C, 2).transpose(), axis = 1) for t in mtime]

    # get a list over time of command values u(t)
    all_us = map(self.mrob.phi2, all_zl)

    # Obstacles constraints
    # N_s*nb_obst_detected
    #print('RADIUS:', self.mrob.rho + self.obst_map[0,-1])
    #all_dists = map(lambda x:LA.norm(self.obst_map[0,0:-1].transpose()-x[:,0]), all_zl)
    #all_xy = map(lambda x:x[:,0], all_zl)
    #print('all robot positions over time', all_xy)
    #print('1st obstacle position:', self.obst_map[0,0:-1].transpose())
    #print('all dists over time:', all_dists)
    obst_cons = np.array([LA.norm(self.obst_map[0,0:-1].transpose()-zl[:,0]) \
          - (self.mrob.rho + self.obst_map[0,-1]) for zl in all_zl])
    for m in range(1,self.obst_map.shape[0]):
      obst_cons = np.append(obst_cons, np.array([LA.norm(self.obst_map[m,0:-1].transpose()-zl[:,0]) \
             - (self.mrob.rho + self.obst_map[m,-1]) for zl in all_zl]))

    # Max speed constraints
    # N_s*u_dim inequations

    max_speed_cons = np.asarray(list(itertools.chain.from_iterable(
        map(lambda u:[self.u_abs_max[0,0] - abs(u[0,0]),
        self.u_abs_max[1,0] - abs(u[1,0])], all_us))))
    #print('INEQCONS:', ret)
    #raw_input()

    ret = np.append(obst_cons, max_speed_cons)

    tot = 0
    for i in ret:
      if i < -1E-4:
        tot = tot+1
    print('NB CONS < 0:', tot)
    #print('RET size:', ret.shape)
    #print('INEQCONS:', ret)
    # return arrray where each element is an inequation constraint
    return ret

##-----------------------------------------------------------------------------
## Initializations

tic = time.clock()
trajc = Trajectory_Generation(Unicycle_Kine_Model())
toc = time.clock()

mtime = trajc._gen_time(trajc.t_fin)
curve = trajc._gen_dtraj(trajc.U, 0)

# get a list over time of the matrix [z dz ddz](t)
all_zl = [np.append(np.append(
    trajc._comb_bsp(t, trajc.C, 0).transpose(),
    trajc._comb_bsp(t, trajc.C, 1).transpose(), axis = 1),
    trajc._comb_bsp(t, trajc.C, 2).transpose(), axis = 1) for t in mtime]

# get a list over time of command values u(t)
all_us = map(trajc.mrob.phi2, all_zl)
linspeed = map(lambda x:x[0], all_us)
angspeed = map(lambda x:x[1], all_us)

print('Elapsed time: ', toc-tic)

## Plot

fig = plt.figure()

circ = []
for r in range(trajc.obst_map.shape[0]):
  circ = circ + [plt.Circle((trajc.obst_map[r,0], trajc.obst_map[r,1]), \
      trajc.obst_map[r,2]+trajc.mrob.rho,color='g',ls = 'dashed',fill=False)]
  circ = circ + [plt.Circle((trajc.obst_map[r,0], trajc.obst_map[r,1]), \
      trajc.obst_map[r,2], color='g', fill=False)]

ax = fig.gca()
ax.plot(trajc.C[:,0], trajc.C[:,1], '.', curve[:,0], curve[:,1])

[ax.add_artist(c) for c in circ]
plt.xlabel('x(m)')
plt.ylabel('y(m)')
plt.title('Generated trajectory')
ax.axis('equal')
ax.axis([-2, 6, 0, 5])

fig = plt.figure()
plt.plot(mtime, map(lambda x:x[0,0], linspeed))
plt.xlabel('time(s)')
plt.ylabel('v(m/s)')
plt.title('Linear speed')

fig = plt.figure()
plt.plot(mtime, map(lambda x:x[0,0], angspeed))
plt.xlabel('time(s)')
plt.ylabel('w(rad/s)')
plt.title('Angular speed')

plt.show()

