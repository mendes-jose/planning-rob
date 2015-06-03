import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import csv

direc = "./"

with open(direc+'/interpdata_ts.csv', 'rb') as csvfile:
    treader = csv.reader(csvfile, delimiter=',',quotechar='|',quoting=csv.QUOTE_NONE)
    tlist = list(treader)
    rows = len(tlist)
    cols = len(tlist[0][0:-1]) # ignoring last character
    interp_tab = np.zeros((rows,cols))
    for i in range(len(tlist)):
        interp_tab[i] = np.array([float(t) for t in tlist[i][0:-1]])

with open(direc+'/points_ts.csv', 'rb') as csvfile:
    treader = csv.reader(csvfile, delimiter=',',quotechar='|',quoting=csv.QUOTE_NONE)
    tlist = list(treader)
    rows = len(tlist)
    cols = len(tlist[0][0:-1])
    points_tab = np.zeros((rows,cols))
    for i in range(len(tlist)):
        points_tab[i] = np.array([float(t) for t in tlist[i][0:-1]])

fig = plt.figure()
ax = fig.gca()
plt.hold(True)

ax.plot(points_tab[0,:], points_tab[1,:])

ax.plot(interp_tab[0,:], interp_tab[1,:], 'ko')

plt.show()
