import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import interp1d

direc = "./"

table = []
header = []
with open('../traces/temp/table.csv', 'rb') as csvfile:
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
toti = header.index('TOT')
lmai = header.index('LMA')

#table = table[np.where(table[:,rati] < 6.0)]

fig = []
plt.hold(True)

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
    # Get how many different n_knots there are and their values
    all_nknots = [scnt[0, nkni]]
    for v in scnt[1:, nkni]:
        if v not in all_nknots:
            all_nknots += [v]
            v_aux = v
    n_knots = len(all_nknots)

    # split table on n_knots tables
    knots_tables = []
    for nkn in all_nknots:
        knots_tables += [np.squeeze(scnt[np.where(scnt[:,nkni] == nkn),:])]

    print len(knots_tables)
    print knots_tables[0].shape

    for knt in knots_tables:
        # Get how many different Ns there are and their values
        all_Ns = [knt[0, nsi]]
        for v in knt[1:, nsi]:
            if v not in all_Ns:
                all_Ns += [v]
        n_Ns = len(all_Ns)

        # split table on Ns tables
        ns_tables = []
        for ns in all_Ns:
            ns_tables += [np.squeeze(knt[np.where(knt[:,nsi] == ns),:])]
#        print 'all NS ', all_Ns
#        print 'nstables len ', len(ns_tables)
#        print 'nstables 0 shape ', ns_tables[0].shape

        fig += [plt.figure()]
        fidx = len(fig) - 1
        ax = fig[fidx].gca()

        for nst, nsidx in zip(ns_tables, range(len(ns_tables))):
            colors = plt.get_cmap('jet')([int(round(v)) for v in np.linspace(0, 255, len(ns_tables))])

            print 'nst shape ', nst.shape

            all_tp = [nst[0, tpi]]
            for v in nst[1:, tpi]:
                if v not in all_tp:
                    all_tp += [v]
            n_Tp = len(all_tp)
#            print all_tp
            
            n_Tc = list(nst[:,tpi]).count(nst[-1,tpi]) #final tp is the one that has more tcs

            rmg = np.zeros(n_Tc)
            aux_tp = nst[0,tpi]
            idxs = list(np.where(nst[:,tpi] == aux_tp)[0])
            new_tctp_interv = np.linspace(nst[0,tci]/nst[0,tpi], nst[idxs[-1],tci]/nst[idxs[-1],tpi], n_Tc)

#            print new_tctp_interv

            for v, idx in zip(all_tp, range(len(all_tp))):
#                print 'v ', v
#                print 'nst : tpi ', nst[:,tpi]
                idxs = list(np.where(nst[:,tpi] == v)[0])
#                print 'dims ', nst[idxs,tci].shape, n_Tc
                f = interp1d(np.divide(nst[idxs,tci],nst[idxs,tpi]), nst[idxs,rmgi], kind='cubic')
                new_rmg = f(new_tctp_interv)
                rmg += new_rmg

            rmg /= len(all_tp)
            
            ax.plot(new_tctp_interv, rmg, color=colors[nsidx])
plt.show()

raw_input()

nTp = 1
v_aux = table[0,1]
all_tp = [v_aux]
for v in table[1:,1]:
    if v != v_aux:
        all_tp += [v] 
        v_aux = v
        nTp += 1
nTc = list(table[:,tpi]).count(table[-1,tpi])

fig_2d = plt.figure()
ax2d = fig_2d.gca()
#tpv = [v for v in all_tp if v >1.5 and v<3.5]
tpv = np.linspace(1.4,5.8,int(round((5.8-1.4)/0.3)))
tpv = [round(v,1) for v in tpv]

#print [int(round(v)) for v in np.linspace(0, 255, len(tpv))]
colors = plt.get_cmap('jet')([int(round(v)) for v in np.linspace(0, 255, len(tpv))])

for v, idx in zip(tpv, range(len(tpv))):
    idxs = list(np.where(table[:,tpi] == v)[0])
    line, = ax2d.plot(np.divide(table[idxs,tci],
            table[idxs,tpi]), table[idxs,rati],
            label='Tp = {}'.format(v),linewidth=2,color=colors[idx])
    for i in idxs[::4]:
        ax2d.text(table[i,tci]/table[i,tpi], table[i,rati], '{0:.1f}'.format(table[i,toti],1))
ax2d.plot([0.1, 0.9], [1.0]*2, label='real_max_Tc == Tc',ls='--',color='k')
ax2d.plot([0.5]*2, [0.0,10], label='Tc/Tp == 0.5',ls='--',color='k')

xy = np.ar1ray([[0.1, 0.0], [0.5, 0.0], [0.5, 1.0], [0.1, 1.0]])
good_zone = plt.Polygon(xy, color='b',fill=True,alpha=0.2)
ax2d.add_artist(good_zone)
ax2d.set_ylim(0.0,5.0)
ax2d.set_xlim(0.1,0.9)
ax2d.set_xlabel('Tc/Tp')
ax2d.set_ylabel('real_max_Tc/Tc')
ax2d.set_title('Tc/Tp and real_max_Tc relation')
handles, labels = ax2d.get_legend_handles_labels()
plt.legend(handles, labels)

# PLOT 3D

table = table[np.where(table[:,rati] < 28.0)]
table = table[np.where(table[:,tpi] < 6.0)]
table = table[np.where(np.divide(table[:,tci], table[:,tpi]) <= 0.5)]
nTp = 1
v_aux = table[0,1]
all_tp = [v_aux]
for v in table[1:,1]:
    if v != v_aux:
        all_tp += [v] 
        v_aux = v
        nTp += 1
nTc = list(table[:,tpi]).count(table[-1,tpi])

X = np.zeros((nTc, nTp))
Y = np.zeros((nTc, nTp))
Z = np.zeros((nTc, nTp))

idi = 0
cidx = 0
idx = 1
v_aux = table[0,1]
for v in np.append(table[1:,1], np.array([-1.0])):
    if v != v_aux:
#        print v, v_aux
#        print table[idi,tci]
#        print table[idx-1,tci]
        new_tc_interv = np.linspace(table[idi,tci]/v_aux, table[idx-1,tci]/v_aux, nTc)
#        print new_tc_interv
#        print table[idi:idx,tci]
        X[:,cidx] = new_tc_interv
        f = interp1d(table[idi:idx,tci]/v_aux, table[idi:idx,rati], kind='cubic')
        Z[:,cidx] = f(new_tc_interv)
        Y[:,cidx] = np.zeros(nTc)+v_aux
        idi = idx
        cidx += 1
        v_aux = v
    idx += 1

fig = plt.figure()
ax3d = fig.gca(projection='3d')
ax3d.plot_surface(X, Y, Z,
        cmap=cm.jet, linewidth=1)
ax3d.plot_surface([[0.1,0.5],[0.1,0.5]], [[1.0, 1.0],[6.0,6.0]],[[1.0,1.0],[1.0,1.0]],color='b',alpha=0.0)
ax3d.set_xlabel('Tc/Tp')
ax3d.set_ylabel('Tp')
ax3d.set_zlabel('real_max_Tc/Tc')
plt.show()

#x, y = np.meshgrid(table[:,tpi], np.divide(table[:,tci],table[:,tpi]))
#table = np.array(sorted(table, key=lambda x:x[tpi]))
#table = np.array(sorted(table, key=lambda x:x[rati]))
#ax.plot_surface(table[:,tpi], np.divide(table[:,tci],table[:,tpi]), table[:,rati],
#        rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
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
#plt.show()
