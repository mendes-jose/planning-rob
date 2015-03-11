import numpy as np
from numpy import linalg as LA
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab
import math 
from scipy.optimize import fmin_slsqp

class Tricycle_Kine_Model(object):
  def __init__(self):
    self.u_dim = 2
    self.q_dim = 3
    self.l = 2

    self.u_abs_max = np.matrix('0.5; 0.5')

  ## Construction des fonctions phi1(z, dz) et phi2(dz, ddz)
  def phi1(self, z, dz):
    return np.append(z, np.matrix(np.arctan2(dz[0], dz[1])), axis = 0)

  def phi2(self, dz, ddz):
    return np.matrix([[LA.norm(dz)], [(dz[0]*ddz[1]-dz[1]*ddz[0])/(np.square(dz[0])+np.square(dz[1]))]])


# Trajectory Generation
class Trajectory_Generation(object):
  def __init__(self, mrobot):

  ## Arbitrary parameters
    self.N_s = 100 # nb of samples for discretization
    self.n_knot = 15 # nb of non zero lenght intervals of the knot series (error X comp. speed)
    self.t_init = 0.0

  ## Other parameters
    self.d = mrobot.l+2 # B-spline order (integer | d > l+1)
    n_ptctrl = self.n_knot + self.d - 1 # nb of ctrl points

  ## Unknown
    self.t_fin = 20.0
    self.C = np.matrix(np.zeros((n_ptctrl,mrobot.u_dim)))
    
  ## Defining knots
    self.knots = [self.t_init]
    for j in range(1,self.d):
      knots_j = self.t_init
      self.knots = self.knots + [knots_j]
    
    for j in range(self.d,self.d+self.n_knot):
      knots_j = self.t_init + (j-(self.d-1.0))*(self.t_fin-self.t_init)/self.n_knot
      self.knots = self.knots + [knots_j]
    
    for j in range(self.d+self.n_knot,2*self.d-1+self.n_knot):
      knots_j = self.t_fin
      self.knots = self.knots + [knots_j]

  ## Definition des contraintes pour le probleme d'optimisation
    # etat initial
    self.q_init = np.matrix([[0.0], [0.0], [np.pi/4]])
    # etat final
    self.q_fin = np.matrix([[2.0], [5.0], [np.pi/4]])
    # commande initiale
    self.u_init = np.matrix([[0.0], [0.0]])
    # commande finale
    self.u_fin = np.matrix([[0.0], [0.0]])
    # sortie plate initiale
    self.z_init = self.q_init[0:2]
    # sortie plate finale
    self.z_fin = self.q_fin[0:2]
    # commandes admissibles
    self.u_abs_max = mrobot.u_abs_max

  ## Evitement obstacles
    obst_map =                     np.matrix([[0.25], [2.5], [0.2]])
    obst_map = np.append(obst_map, np.matrix([[2.3],  [2.5], [0.5]]), axis = 0)
    obst_map = np.append(obst_map, np.matrix([[1.25], [3],   [0.1]]), axis = 0)
    obst_map = np.append(obst_map, np.matrix([[0.3],  [1],   [0.1]]), axis = 0)
    obst_map = np.append(obst_map, np.matrix([[-0.5], [1.5], [0.3]]), axis = 0)
    # TODO: Obstacles random generation
    
  ## B-splines recursive algorithm
  def _bspline(self, t, j, d, der_order):
    if der_order == 0:
      if d == 1:
        return 1 if t >= self.knots[j] and t < self.knots[j+1] else 0
      else:
        if self.knots[j+d-1]-self.knots[j] == 0 or self.knots[j+d]-self.knots[j+1] == 0:
          return 0
        else:
          return (t - self.knots[j])/(self.knots[j+d-1]-self.knots[j])*self._bspline(t, j, d-1, der_order) + \
              (self.knots[j+d]-t)/(self.knots[j+d]-self.knots[j+1])*self._bspline(t, j+1, d-1, der_order)
    else:
      if d == 1:
        return 0
      else:
        if self.knots[j+d-1]-self.knots[j] == 0 or self.knots[j+d]-self.knots[j+1] == 0:
          return 0
        else:
          return (d-1)*((self._bspline(t, j, d-1, der_order-1))/(self.knots[j+d-1]-self.knots[j])- \
              (self._bspline(t, j+1, d-1, der_order-1))/(self.knots[j+d]-self.knots[j+1]))

  ## Parametrisation par une fonction spline de la sortie plate z(t)=[x(t);y(t)]^T et ses derivees dz et ddz


  

  ## resolution probleme par optimisation SQP
  #
  # Le probleme est de la forme :
  # min 0.5 * U^T QQ U + pp U
  # avec
  #     CE^T U + ce0 = 0
  #     CCu^T U + DDu >= 0
  
    # U = fmin_slsqp(self.__criteria, self.__U0, eqcons=[], f_eqcons=None, ieqcons=[], 
    	 # f_ieqcons=self.__fieqcons, bounds=[], fprime=self.__criteria_deriv, fprime_eqcons=None, fprime_ieqcons=self.__fieqcons_deriv,args=([X0,PP,QQ,CCu,DDu]),
    	 # iter=100, acc=1e-06, iprint=0, disp=None, full_output=0, epsilon = 0.001)
 
  # def __criteria(self,U,*args):
    # U = np.matrix(np.reshape(U,(-1,1)))
    # X0 = args[0]
    # PP = args[1]
    # QQ = args[2]
    # pp = PP*X0
    # cost = U.transpose()*QQ*U/2+pp.transpose()*U # min 0.5 * U^T QQ U + pp U
    # return cost

  # def __criteria_deriv(self,U,*args):
    # X0 = args[0]
    # PP = args[1]
    # QQ = args[2]
    # pp = PP*X0
    # cost_deriv = U.transpose()*QQ+pp.transpose()
    # return np.asarray(cost_deriv).reshape(-1)
    
  # def __feqcons(self,U,*args):
    # U = np.matrix(np.reshape(U,(-1,1)))
    # CE = args[]
    # ce0 = args[]
    # const = CE*U+ce0 # CE^T U + ce0
    # return np.asarray(const).reshape(-1)
    
  # def __feqcons_deriv(self,U,*args):
    # CE = args[]
    # const_deriv = CE
    # return np.asarray(const_deriv)
    
  # def __fieqcons(self,U,*args):
    # U = np.matrix(np.reshape(U,(-1,1)))
    # CCu = args[3]
    # DDu = args[4]
    # const = CCu*U+DDu # CCu^T U + DDu
    # return np.asarray(const).reshape(-1)
    
  # def __fieqcons_deriv(self,U,*args):
    # CCu = args[3]
    # const_deriv = CCu
    # return np.asarray(const_deriv)
      
    
    
# def angle(x):
  # pi = math.pi
  # twopi = 2*pi
  # return (x+pi)%twopi-pi

# Initializations
Gene = Trajectory_Generation(Tricycle_Kine_Model())
#print Gene.bspline(0.5, 0, 4, 0)
t = [x * 0.1 for x in range(0, 200)]
curve = []
for i in t:
  a = Gene._bspline(i,5,4,0)
  curve = curve+[a]
#Gene.update()
plt.plot(t,curve)
plt.show()
# # Plot
# fig_curve = plt.figure()
# ax_curve = fig_curve.add_subplot(111)
# line_curve = mpl.lines.Line2D(Gene.k[:,0], Gene.k[:,1], c = 'blue', ls = '-',lw = 1)
# ax_curve.add_line(line_curve)
# plt.xlim([-1.0, 11.0])
# plt.ylim([-60.0, 60.0])
# plt.title('Curvature')
# plt.xlabel('s (m)')
# plt.ylabel('k (m^(-1))')
# #plt.legend(('front', 'rear'), loc='lower right')
# plt.grid()
# plt.draw()

# fig_theta = plt.figure()
# ax_theta = fig_theta.add_subplot(111)
# line_theta = mpl.lines.Line2D(Gene.theta[:,0], Gene.theta[:,1], c = 'blue', ls = '-',lw = 1)
# ax_theta.add_line(line_theta)
# plt.xlim([-1.0, 11.0])
# plt.ylim([-10.0, 180.0])
# plt.title('Yaw angle')
# plt.xlabel('s (m)')
# plt.ylabel('theta (rad)')
# plt.grid()
# plt.draw()

# plt.show()
