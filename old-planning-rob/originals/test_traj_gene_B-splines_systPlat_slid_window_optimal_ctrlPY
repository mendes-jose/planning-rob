import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab
import math 
from scipy.optimize import fmin_slsqp

# Trajectory Generation
class Trajectory_Generation(object):
  def __init__(self):
  ## Variables et Parametres Publics
    self.N_ech = 100 # nbre echantillons pour la discretisation
    self.d = 4 # robot unicycle
    self.n_knot = 15 # choix du nbre de points de ctrl
    p = self.n_knot + self.d - 2
    
  ## Inconnues
    self.t_fin = 10000.0
    
    C = np.matrix([[0.0,0.0]])
    for j in range(1,p+1):
      C_j = np.matrix([0.0,0.0])
      C = np.append(C, np.matrix(C_j), axis = 0)
    self.C = C
    #print(np.transpose(self.C[10]))
    
  ## Definition des noeuds
    self.t_init = 0.0

    noeud = np.matrix([[self.t_init]])
    for j in range(1,self.d):
      noeud_j = self.t_init
      noeud = np.append(noeud, np.matrix([[noeud_j]]), axis = 0)
    
    for j in range(self.d,self.d+self.n_knot):
      noeud_j = self.t_init + (j-(self.d-1.0))*(self.t_fin-self.t_init)/self.n_knot
      noeud = np.append(noeud, np.matrix([[noeud_j]]), axis = 0)
    
    for j in range(self.d+self.n_knot,2*self.d-1+self.n_knot):
      noeud_j = self.t_fin
      noeud = np.append(noeud, np.matrix([[noeud_j]]), axis = 0)
    
  ## B-splines
   # construction des B-splines d'ordre d=1
   
   # construction des B-splines d'ordre d=2
   
   # construction des B-splines d'ordre d=3
   
   # construction des B-splines d'ordre d=4

   # construction des derivees premieres des B-splines d'ordre d=3
   
   # construction des derivees premieres des B-splines d'ordre d=4
   
   # construction des derivees secondes des B-splines d'ordre d=4


  ## Parametrisation par une fonction spline de la sortie plate z(t)=[x(t);y(t)]^T et ses derivees dz et ddz
  
  
  ## Construction des fonctions phi1(z, dz) et phi2(dz, ddz)
  
  
  ## Definition des contraintes pour le probleme d'optimisation
   # etat initial
   
   # etat final
   
   # commande initiale
   
   # commande finale
   
   # commandes admissibles
   
   # evitement obstacles
   
  
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
Gene = Trajectory_Generation()
#Gene.update()

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