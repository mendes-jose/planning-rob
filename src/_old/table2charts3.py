import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import interp1d

direc = "./"

table = []
header = []
with open('../traces/temp/table_complete2.csv', 'rb') as csvfile:
    treader = csv.reader(csvfile, delimiter=',',quotechar='|',quoting=csv.QUOTE_NONNUMERIC)
    tlist = list(treader)
    header = tlist[0]
    table = np.array(tlist[1:])

nkni = header.index('Nkn')
nobsti = header.index('Nobsts')
nsi = header.index('Ns')
tpi = header.index('Tp')
tci = header.index('Tc')
lmai = header.index('LMA')
rmgi = header.index('RMG')
rmpi = header.index('RMP')
toti = header.index('TOT')
lmai = header.index('LMA')

#table = table[np.where(table[:,rati] < 6.0)]

fig = []
fig_uni = []
plt.hold(True)

# define goodzone
xy = np.array([[0.1, 0.0], [0.5, 0.0], [0.5, 1.0], [0.1, 1.0]])
good_zone = plt.Polygon(xy, color='b',fill=True,alpha=0.2)

# Get how many different scenarios there are and how many obstacles each one has
n_scenarios = 1
all_nobsts = [table[0, nobsti]]
for v in table[1:, nobsti]:
    if v not in all_nobsts:
        all_nobsts += [v]
        n_scenarios += 1

# split table on n_scenarios tables
print table.shape
#print all_nobsts
scenarios_tables = []
for nobst in all_nobsts:
    scenarios_tables += [np.squeeze(table[np.where(table[:,nobsti] == nobst),:])]

for scnt in scenarios_tables:
    tot = scnt[:,toti]
    idxs = tot.argsort()
    tot = tot[idxs]
    print tot
    col = {'Ns': nsi, 'Tp': tpi, 'Tc': tci, 'max(tc/Tc)': rmpi, 'max(tc/Tc\')': rmgi}
    ns = scnt[idxs,nsi]
    tp = scnt[idxs,tpi]
    tc = scnt[idxs,tci]
    rmp = scnt[idxs,rmpi]
    rmg = scnt[idxs,rmgi]

    fig += [plt.figure()]
    fiidx = len(fig)-1
    ax = fig[fiidx].gca()
    [ax.plot(tot[0:-20], scnt[idxs, col[c]][0:-20], label = c, marker='o') for c in col]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
plt.show()

