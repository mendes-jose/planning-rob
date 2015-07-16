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
direc = "./Ns_var_pen2"

table = []
header = []
with open(direc+'/table.csv', 'rb') as csvfile:
    treader = csv.reader(csvfile, delimiter=',',quotechar='|',quoting=csv.QUOTE_NONNUMERIC)
    tlist = list(treader)
    header = tlist[0]
    table = np.array(tlist[1:])
with open(direc+'/table2.csv', 'rb') as csvfile:
    treader = csv.reader(csvfile, delimiter=',',quotechar='|',quoting=csv.QUOTE_NONNUMERIC)
    tlist = list(treader)
    header = tlist[0]
    print tlist[0]
    print tlist[-1]
    table = np.append(table, np.array(tlist[1:]), axis=0)

print table[-1]
nkni = header.index('Nkn')
nobsti = header.index('Nobsts')
nsi = header.index('Ns')
nssoli = header.index('Nssol')
tpi = header.index('Tp')
tci = header.index('Tc')
lmai = header.index('LMA')
rmgi = header.index('RMG')
rmpi = header.index('RMP')
toti = header.index('TOT')
lmai = header.index('LMA')
peni = header.index('PEN')
mobi = header.index('MOB')

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

    all_Ns = []
    for v in knots_tables[0][:, nsi]:
        if v not in all_Ns and (v >= nsinf and v <= nssup):
            all_Ns += [v]
    n_Ns = len(all_Ns)
    all_pen = np.zeros(n_Ns)

    # iterate on knots tables
    for knt in knots_tables:

        # add fig to fig array
        fig += [plt.figure()]
        fidx = len(fig) - 1
        ax = fig[fidx].gca()

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

        # create lines colors
        colors_ns = plt.get_cmap('jet')([int(round(rgb)) for rgb in np.linspace(0, 255, n_Ns)])

        all_y_init = False

        # iterate over ns_tables
#        for nst, nsidx in zip(ns_tables, range(len(ns_tables))):

#            if nst[0, nsi] < nsinf or nst[0, nsi] > nssup:
#                continue
            
#            all_pen = nst[0:15, peni]
#            all_ns = nst[:, nssoli]/nst[:, nsi]
#
#            ax.plot(all_nssol, all_pen, label='P(N_ssol) with N_s = 10'.format(nst[0,nsi]),linewidth=1,\
#                    color=colors_ns[nsidx],marker='.')
#
#            ax.plot([0.0, all_nssol[-1]], [0.0031515750838]*2, ls='--',color='k', label='P(N_ssol>>N_s)')

        all_pen += knt[:, peni]

    all_pen /= n_knots

    ax.plot(all_Ns, 10000*all_pen, label='P(N_s)', marker='.')

    ax.grid()
    ax.set_title('Time sampling and obstacle penetration relationship')
    ax.set_xlabel(r'$N_s$')
    ax.set_ylabel(r'$P\ (cm^2)$')
    handles, labels = ax.get_legend_handles_labels()
    #ax.legend(handles, labels, loc=1)
    #ax.set_xlim(0.0,all_Ns[-1])
    fig[fidx].set_size_inches(1.0*18.5/2.54,1.0*8.5/2.54)
    fig[fidx].savefig(direc_charts+'/pen-nsi.eps', bbox_inches='tight')
    fig[fidx].savefig(direc_charts+'/pen-nsi.pdf', bbox_inches='tight')
    fig[fidx].savefig(direc_charts+'/pen-nsi.png', bbox_inches='tight')

#plt.show()

