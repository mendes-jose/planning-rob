import numpy as np
from numpy import linalg as LA
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab
import math 
from scipy.optimize import fmin_slsqp


def bspline(t, j, d, deriv_ord, knots):
  if deriv_ord == 0:
    if d == 1:
      return 1 if (t >= knots[j] and t < knots[j+1]) else 0
    else:
      return (t-knots[j])/(knots[j+d-1]-knots[j])*bspline(t, j, d-1, deriv_ord, knots)+(knots[j+d]-t)/(knots[j+d]-knots[j+1])*bspline(t, j+1, d-1, deriv_ord, knots)
  else:
    return -1

   # if der_order == 0:
   #   if d == 1:
   #     return 1 if t >= self.knots[j] and t < self.knots[j+1] else 0
   #   else:
   #     #if self.knots[j+d+1]-self.knots[j] == 0 or self.knots[j+d]-self.knots[j+1] == 0:
   #     if False :
   #       return 0
   #     else:
   #       return (t - self.knots[j])/(self.knots[j+d+1]-self.knots[j])*self.bspline(t, j, d-1, der_order) + \
   #           (self.knots[j+d]-t)/(self.knots[j+d]-self.knots[j+1])*self.bspline(t, j+1, d-1, der_order)
   # else:
   #   if d == 1:
   #     return 0
   #   else:
   #     #if self.knots[j+d+1]-self.knots[j] == 0 or self.knots[j+d]-self.knots[j+1] == 0:
   #     if False:
   #       return 0
   #     else:
   #       return (d-1)*((self.bspline(t, j, d-1, der_order-1))/(self.knots[j+d-1]-self.knots[j])- \
   #           (self.bspline(t, j+1, d-1, der_order-1))/(self.knots[j+d]-self.knots[j+1]))


  ## Parametrisation par une fonction spline de la sortie plate z(t)=[x(t);y(t)]^T et ses derivees dz et ddz


  ## Construction des fonctions phi1(z, dz) et phi2(dz, ddz)
    # const = CCu*U+DDu # CCu^T U + DDu
    # return np.asarray(const).reshape(-1)

# Initializations
t = [x * 0.1 for x in range(0, 41)]
print t
knots = [x for x in range(0,20)]
print knots
curve = []
for i in t:
  tmp = bspline(i,0,4,0,knots)
  curve = curve+[tmp]

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
