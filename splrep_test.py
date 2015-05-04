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

d = 4 # b-spline order + 1
n_knots = 4 #number of internal knots + 1 (or non-zero intervals)

#mtime = np.linspace(0.0, 10.0, 20)
mtime = np.linspace(0.0,10.0,d+n_knots-1)


gk = lambda x:mtime[0] + (x-(d-1.0))*(mtime[-1]-mtime[0])/n_knots
knots = [mtime[0] for _ in range(d)]
knots.extend([gk(i) for i in range(d, d+n_knots-1)])
knots.extend([mtime[-1] for _ in range(d)])

x = np.sin(mtime)+mtime+1.0
y = mtime+1.0
#x = mtime+1.0
#y = mtime+1.0

nx = np.array([x,y])

# BUG with splprep???
#tck,u = si.splprep(nx, u=mtime, task=-1, t=knots, k=d-1)
#tck,u = si.splprep(nx, u=mtime, k=d-1)
#t=tck[0]
#c=tck[1]
tckx = si.splrep(mtime, x, task=-1, t=knots[d:-d], k=d-1)
tcky = si.splrep(mtime, y, task=-1, t=knots[d:-d], k=d-1)
#tckx = si.splrep(mtime, x, k=d-1)
#tcky = si.splrep(mtime, y, k=d-1)
t = tckx[0]
c = [tckx[1], tcky[1]]

cx = c[0]
cy = c[1]
#cx = np.roll(cx, 1)[0:-3,]
#cy = np.roll(cy, 1)[0:-3,]
cx = cx[0:-d,]
cy = cy[0:-d,]

print 'Cx Cy ',cx,cy

plt.plot(x,y,'bd--')
plt.hold(True)
hatx = si.splev(mtime, (t, cx, 3))
haty = si.splev(mtime, (t, cy, 3))
plt.plot(hatx, haty,'g--')
plt.plot(cx, cy,'ro')
fig = plt.figure()
ax = fig.gca()
C = np.array([cx, cy]).T
print C.shape
l = []
for i,j in zip(C[0:-1,], C[1:,]):
    l += [LA.norm(i-j)]
ax.plot(l,'o--')
plt.show()

