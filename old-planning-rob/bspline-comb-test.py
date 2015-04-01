import numpy as np
from numpy import linalg as LA
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab
import math 
import scipy.interpolate as si
import time as tm

## B-spline recursive algorithm
# TODO: Say NO to losing time with recursive, use iteractive with execution stack instead!
def bspline(t, j, d, deriv_order, knots):
  if deriv_order == 0:
    if d == 1:
      return 1 if (t >= knots[j] and t < knots[j+1]) else 0
    else:
      if (knots[j+d-1]-knots[j] == 0 and \
          knots[j+d]-knots[j+1] == 0):
        return 0

      elif knots[j+d-1]-knots[j] == 0:
        return (knots[j+d]-t)/ \
            (knots[j+d]-knots[j+1])* \
            bspline(t, j+1, d-1, deriv_order, knots)

      elif knots[j+d]-knots[j+1] == 0:
        return (t - knots[j])/(knots[j+d-1]-knots[j])* \
            bspline(t, j, d-1, deriv_order, knots)

      else:
        return (t - knots[j])/(knots[j+d-1]-knots[j])* \
            bspline(t, j, d-1, deriv_order, knots) + (knots[j+d]-t)/ \
            (knots[j+d]-knots[j+1])* \
            bspline(t, j+1, d-1, deriv_order, knots)
  else:
    if d == 1:
      return 0
    else:
      if (knots[j+d-1]-knots[j] == 0 or \
          knots[j+d]-knots[j+1] == 0):
        return 0
      else:
        return (d-1)*((bspline(t, j, d-1, deriv_order-1, knots))/ \
            (knots[j+d-1]-knots[j])- \
            (bspline(t, j+1, d-1, deriv_order-1, knots))/ \
            (knots[j+d]-knots[j+1]))


## Combine b-splines
def comb_bsp(t, n_ptctrl, C, d, deriv_order, knots):
  return sum([C[j]*bspline(t, j, d, deriv_order, knots) \
      for j in range(0, n_ptctrl)])

## Generate b-spline knots
def gen_knots(t_init, t_fin, d, n_knot):
  knots = [t_init]
  for j in range(1,d):
    knots_j = t_init
    knots = knots + [knots_j]
  
  for j in range(d,d+n_knot):
    knots_j = t_init + (j-(d-1.0))* \
        (t_fin-t_init)/n_knot
    knots = knots + [knots_j]
  
  for j in range(d+n_knot,2*d-1+n_knot):
    knots_j = t_fin
    knots = knots + [knots_j]
  return knots

# Initializations
t_fin=4.0
N = 100
n_knot = 15
d = 4
n_ptctrl = n_knot + d - 1 # nb of ctrl points

t = [x * t_fin/(N-1)  for x in range(0, N)]
knots = gen_knots(0.0, t_fin, d, n_knot)

C = np.array(np.zeros(n_ptctrl))
C[0] = 0
C[1] = 0
C[2] = 0
C[3] = 1
C[4] = -1
C[5] = .1
C[6] = .1
C[7] = 1
C[8] = -1
C[9] = 1
C[10] = 1
C[11] = 1
C[12] = 1
C[13] = -1
C[14] = 1
C[15] = -1
C[16] = 1
C[17] = -1

print('TIME',np.asarray(t))
tup = (np.asarray(knots), np.asarray(C), d-1)
print('TUP', tup)

t0 = tm.clock()
curve1 = si.splev(np.asarray(t), (np.asarray(knots), np.asarray(C), d-1),der=0)
print('Tcurve1', tm.clock()-t0)

t0 = tm.clock()
curve2 = []
# time, nptctrl,C,d,deriv,knots
for i in t:
  tmp = comb_bsp(i,n_ptctrl,C,d,0,knots) #
  curve2 = curve2+[tmp]
print('Tcurve2', tm.clock()-t0)

#print [x * t_fin/(n_ptctrl-1) for x in range(0,n_ptctrl)]
plt.plot(t,curve1, t,curve2)
plt.axis('equal')
plt.show()
