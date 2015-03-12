import numpy as np
from numpy import linalg as LA
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab
import math 

## B-spline recursive algorithm
# TODO: Say NO to losing time with recursive, use iteractive with execution stack instead!
def bspline(t, j, d, deriv_order, knots):
  if deriv_order == 0:
    if d == 1:
      return 1 if (t >= knots[j] and t < knots[j+1]) else 0
    else:
      if (knots[j+d-1]-knots[j] == 0 or \
          knots[j+d]-knots[j+1] == 0):
        return 0
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

# Initializations
t_fin=4.0
N = 100
t = [x * t_fin/(N-1)  for x in range(0, N)]
print t
knots = [x for x in range(0,20)]
print knots
curve = []
for i in t:
  tmp = bspline(i,0,2,0,knots) # B_{0,2}(t)
  curve = curve+[tmp]

f_b02 = plt.figure()
plt.plot(t,curve)

curve = []
for i in t:
  tmp = bspline(i,0,4,0,knots) # B_{0,4}(t)
  curve = curve+[tmp]

f_b04 = plt.figure()
plt.plot(t,curve)

curve = []
for i in t:
  tmp = bspline(i,0,3,1,knots) # B'_{0,3}(t)
  curve = curve+[tmp]

f_db03 = plt.figure()
plt.plot(t,curve)

plt.show()

