import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

direc = "./"

table = []
header = []
with open('table.csv', 'rb') as csvfile:
    treader = csv.reader(csvfile, delimiter=',',quotechar='|',quoting=csv.QUOTE_NONNUMERIC)
    tlist = list(treader)
    header = tlist[0]
    table = np.array(tlist[1:])

cnt = 1
v_aux = table[0,1]
biggest_sec = 0
for v in table[1:,1]:
    if v != v_aux:
        if v-v_aux < 0.05:
            print(v,v_aux)
        v_aux = v
        cnt += 1

colors = [[i, 1.0, 0.0] for i in np.linspace(0.0, 1.0, cnt)]

fig = plt.figure()
ax = fig.gca(projection='3d')
plt.hold(True)

p_idx = 0
cnt = 0

tp

tx = np.linspace(-1.0, 1.0, 10)
ty = np.linspace(-1.0, 1.0, 10)
tx, ty = np.meshgrid(tx, ty)
y = np.cos(tx+ty)
print(tx,ty,y)
ax.plot_surface(tx, ty, y)
plt.show()
tci = header.index('Tc')
tpi = header.index('Tp')
rati = header.index('RAT')
x, y = np.meshgrid(table[:,tpi], np.divide(table[:,tci],table[:,tpi]))
#table = np.array(sorted(table, key=lambda x:x[tpi]))
table = np.array(sorted(table, key=lambda x:x[rati]))
ax.plot_surface(table[:,tpi], np.divide(table[:,tci],table[:,tpi]), table[:,rati])
#ax.plot_trisurf(table2[:,tpi], np.divide(table2[:,tci],table2[:,tpi]), table2[:,rati])
#ax.plot(table2[:,tpi], np.divide(table2[:,tci],table2[:,tpi]), table2[:,rati])
#for idx in range(1,table.shape[0]):
#    if table[idx,1] != table[idx-1,1]:
#        tci = header.index('Tc')
#        tpi = header.index('Tp')
#        rati = header.index('RAT')
#        ax.plot(table[p_idx,tpi], np.divide(table[p_idx:idx,tci],table[p_idx:idx,tpi]), table[p_idx:idx,rati], color=colors[cnt])
#        p_idx = idx
#        cnt += 1
plt.show()
