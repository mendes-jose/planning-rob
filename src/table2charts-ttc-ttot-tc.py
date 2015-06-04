import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import interp1d

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 10}

mpl.rc('font', **font)

direc = "../traces/rt-full-table"

table = []
header = []
with open(direc+'/table.csv', 'rb') as csvfile:
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

# Delete aberrant values
#table = table[np.where(table[:,rati] < 6.0)]

# Declare figures array
fig = []
plt.hold(True)

# Get how many different scenarios there are and how many obstacles each one has
n_scenarios = 1
all_nobsts = [table[0, nobsti]]
for v in table[1:, nobsti]:
    if v not in all_nobsts:
        all_nobsts += [v]
        n_scenarios += 1

print('Table shape: {0}'.format(table.shape))
print('N scenarios: {0}'.format(n_scenarios))
print('N obsts: {0}'.format(all_nobsts))

# split table on n_scenarios tables
scenarios_tables = []
for nobst in all_nobsts:
    scenarios_tables += [np.squeeze(table[np.where(table[:,nobsti] == nobst),:])]

# iterate on scenarios tables
for scnt in scenarios_tables:
    # Get how many different n_knots there are and their values
    all_nknots = [scnt[0, nkni]]
    for v in scnt[1:, nkni]:
        if v not in all_nknots:
            all_nknots += [v]
            v_aux = v
    n_knots = len(all_nknots)

    print('N knots for scn {0}: {1}'.format(scnt[0, nobsti], n_knots))
    print('Knots for scn {0}: {1}'.format(scnt[0, nobsti], all_nknots))

    # split table on n_knots tables
    knots_tables = []
    for nkn in all_nknots:
        knots_tables += [np.squeeze(scnt[np.where(scnt[:,nkni] == nkn),:])]

    # get N_s limits
    nsinf = min(knots_tables[0][:, nsi])
    nssup = max(knots_tables[0][:, nsi])
    for knt in knots_tables[1:]:
        nsinf = min(knt[:, nsi]) if nsinf < min(knt[:, nsi]) else nsinf
        nssup = max(knt[:, nsi]) if nssup > max(knt[:, nsi]) else nssup

    # iterate on knots tables
    for knt in knots_tables:

        # add fig to fig array
        fig += [plt.figure()]
        fidx = len(fig) - 1
        ax = fig[fidx].gca()

        # Get how many different Ns there are and their values
        all_Ns = []
        for v in knt[:, nsi]:
            if v not in all_Ns and (v >= nsinf and v <= nssup):
                all_Ns += [v]
        n_Ns = len(all_Ns)

        print('N Ns for scn {0} knot {1}: {2}'.format(scnt[0, nobsti], knt[0, nkni], n_Ns))
        print('Ns for scn {0} knot {1}: {2}'.format(scnt[0, nobsti], knt[0, nkni], all_Ns))

        # split table on Ns tables
        ns_tables = []
        for ns in all_Ns:
            ns_tables += [np.squeeze(knt[np.where(knt[:,nsi] == ns),:])]

        # creating store directories

        direc_charts = direc+'/charts'
        try:
            os.mkdir(direc_charts)
        except OSError:
            print('Probably the output directory already exists, going to overwrite content')

        direc_scen = direc_charts+'/Scenario_{a:.0f}__N_knots_{b:.0f}'.format(b=knt[0,nkni], a=scnt[0,nobsti])
        try:
            os.mkdir(direc_scen)
        except OSError:
            print('Probably the output directory already exists, going to overwrite content')


        # create lines colors
        colors_ns = plt.get_cmap('jet')([int(round(rgb)) for rgb in np.linspace(0, 255, n_Ns)])

        all_y_init = False

        # iterate over ns_tables
        for nst, nsidx in zip(ns_tables, range(len(ns_tables))):

            if nst[0, nsi] < nsinf or nst[0, nsi] > nssup:
                continue

            # get all Tp values
            all_tp = [nst[0, tpi]]
            for v in nst[1:, tpi]:
                if v not in all_tp:
                    all_tp += [v]

            all_tp.sort()
            n_Tp = len(all_tp)

            # get number of Tcs
            aux_tpi_column = sorted(nst[:,tpi])
            n_Tc = list(aux_tpi_column).count(aux_tpi_column[-1]) #greater tp is the one that has more tcs

            rmg = np.zeros(n_Tc)

            aux_tp = nst[0,tpi]
            idxs = list(np.where(nst[:,tpi] == aux_tp)[0])
            new_tctp_interv = np.linspace(nst[0,tci]/nst[0,tpi], nst[idxs[-1],tci]/nst[idxs[-1],tpi], n_Tc)
#            new_tc_interv = np.linspace(nst[0,tci], nst[idxs[-1],tci], n_Tc)

#            colors = plt.get_cmap('jet')([int(round(rgb)) for rgb in np.linspace(0, 255, len(all_tp))])
            for v, idx in zip(all_tp, range(len(all_tp))):
                
                _idxs = list(np.where(nst[:,tpi] == v)[0])
                x = np.divide(nst[mtp_idxs,tci],nst[mtp_idxs,tpi])
                sort_idxs = x.argsort()
                x = x[sort_idxs]
                y = nst[mtp_idxs,rmpi]
                y = y[sort_idxs]
                f = interp1d(x, y, kind='cubic', bounds_error=False)
                new_rmg = f(new_tctp_interv)
                rmg += new_rmg

            rmg /= len(all_tp)

            if not all_y_init:
                all_y = rmg
                all_y_init = True
            else:
                all_y = np.vstack((all_y,rmg))

            ax.plot(new_tctp_interv, rmg, label='N_s = {}'.format(nst[0,nsi]),linewidth=1,\
                    color=colors_ns[nsidx],marker='.')

        print 'VAR:', np.mean(np.var(all_y, 0))

        ax.plot([0.5]*2, [0.0,2.0], ls='--',color='k')
        ax.plot([0.1, 0.9], [1.0]*2, ls='--',color='k')
        if scnt[0, nobsti] == 0.0:
            ax.set_ylim(0.0,.5)
        ax.set_xlim(0.2,0.8)
        ax.set_xlabel('Tc/Tp')
        ax.set_ylabel('max_comp_t/Tc')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        fig[fidx].set_size_inches(1.2*18.5/2.54,1.2*10.5/2.54)
        fig[fidx].savefig(direc_scen+'/mcttc-tctp.eps'.format(nst[0,nsi]), bbox_inches='tight', dpi=100)
        fig[fidx].savefig(direc_scen+'/mcttc-tctp.png'.format(nst[0,nsi]), bbox_inches='tight', dpi=100)

#plt.show()

