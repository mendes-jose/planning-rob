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
with open(direc+'/table_complete2.csv', 'rb') as csvfile:
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

    # iterate on knots tables
    for knt in knots_tables:

        # Get how many different Ns there are and their values
        all_Ns = [knt[0, nsi]]
        for v in knt[1:, nsi]:
            if v not in all_Ns:
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

        # iterate over ns_tables
        for nst, nsidx in zip(ns_tables, range(len(ns_tables))):

            # add fig to fig array
            fig += [plt.figure()]
            fidx = len(fig) - 1
            ax = fig[fidx].gca()

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

            colors_tp = plt.get_cmap('jet')([int(round(rgb)) for rgb in np.linspace(0, 255, len(all_tp))])
            for v, idx in zip(all_tp, range(len(all_tp))):

                mtp_idxs = list(np.where(nst[:,tpi] == v)[0])
                x = nst[mtp_idxs,tci]
                sort_idxs = x.argsort()
                x = x[sort_idxs]
                y = np.multiply(nst[mtp_idxs,rmpi],nst[mtp_idxs,tci])
                y = y[sort_idxs]
                ax.plot(x, y, label='T_p = {}'.format(v), color=colors_tp[idx], marker='.')
#                for xi, yi, islast in zip(x, y, nst[mtp_idxs, lmai]):
#                    if islast == 1.0:
#                        ax.plot(xi, yi, 'ko')

            ax.set_xlabel('Tc')
            ax.set_ylabel('max_comp_t')
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels)
            fig[fidx].set_size_inches(1.2*18.5/2.54,1.2*10.5/2.54)
            fig[fidx].savefig(direc_scen+'/mct-tc-Ns{}.eps'.format(nst[0,nsi]), bbox_inches='tight', dpi=100)
            fig[fidx].savefig(direc_scen+'/mct-tc-Ns{}.png'.format(nst[0,nsi]), bbox_inches='tight', dpi=100)

#plt.show()

