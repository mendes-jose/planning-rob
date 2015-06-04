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

direc = "../traces/rt-tcfix"

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
toti = header.index('TOT')
lmai = header.index('LMA')

#table = table[np.where(table[:,rati] < 6.0)]

fig = []
figtot = []
fig_tot_1tp = []
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

#    print len(knots_tables)
#    print knots_tables[0].shape

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

        fig_uni += [plt.figure()]
        funiidx = len(fig_uni) - 1
        ax_uni = fig_uni[funiidx].gca()
        colorsa = plt.get_cmap('jet')([int(round(rgb)) for rgb in np.linspace(0, 255, len(ns_tables))])
        for nst, nsidx in zip(ns_tables, range(len(ns_tables))):
#            print 'nst shape ', nst.shape
            fig += [plt.figure()]
            figtot += [plt.figure()]
            fidx = len(fig) - 1
            fig_tot_1tp += [plt.figure()]
            f_tot_1tp_idx = len(fig_tot_1tp) - 1
            ax = fig[fidx].gca()
            axtot = figtot[fidx].gca()
            ax_tot_1tp = fig_tot_1tp[f_tot_1tp_idx].gca()

            all_tp = [nst[0, tpi]]
            for v in nst[1:, tpi]:
                if v not in all_tp:
                    all_tp += [v]
            all_tp.sort()
            n_Tp = len(all_tp)

            aux_tpi_column = sorted(nst[:,tpi])
            n_Tc = list(aux_tpi_column).count(aux_tpi_column[-1]) #greater tp is the one that has more tcs

            rmg = np.zeros(n_Tc)
            aux_tp = nst[0,tpi]
            idxs = list(np.where(nst[:,tpi] == aux_tp)[0])
            new_tctp_interv = np.linspace(nst[0,tci]/nst[0,tpi], nst[idxs[-1],tci]/nst[idxs[-1],tpi], n_Tc)

#            print new_tctp_interv
            colors = plt.get_cmap('jet')([int(round(rgb)) for rgb in np.linspace(0, 255, 7)])
            for v, idx in zip(all_tp[2:9], range(7)):

#                print 'v ', v
#                print 'nst : tpi ', nst[:,tpi]
                idxs = list(np.where(nst[:,tpi] == v)[0])
#                print 'dims ', nst[idxs,tci].shape, n_Tc
                x = np.divide(nst[idxs,tci],nst[idxs,tpi])
                sort_idxs = x.argsort()
                x = x[sort_idxs]
                y = nst[idxs,rmgi]
                y = y[sort_idxs]
                f = interp1d(x, y, kind='cubic', bounds_error=False)
                new_rmg = f(new_tctp_interv)
                rmg += new_rmg
                ax.plot(x, y, label='Tp = {}'.format(v),linewidth=1,color=colors[idx],marker='.')
                #scenarios_tables += [np.squeeze(table[np.where(table[:,nobsti] == nobst),:])]
                tpt = nst[idxs,:]
                a = tpt[:,toti] > 7.0
                b = tpt[:,toti] < 8.2
                c = a * b
                reduced_tpt = np.squeeze(tpt[np.where(c), :])
                xtc = reduced_tpt[:,tci]
                totidx = xtc.argsort()
                xtc = xtc[totidx]
                print xtc
                ytot = reduced_tpt[:,toti]
                ytot = ytot[totidx]
                axtot.plot(xtc, ytot, label='Tp = {}s'.format(v),linewidth=1,color=colors[idx],marker='.')

            # print only one tp
            idxs = list(np.where(nst[:,tpi] == nst[3,tpi])[0])
            #print nst[:,tpi]
            #print idxs
            tpt = nst[idxs,:]
            a = tpt[:,toti] < 8.2
            b = tpt[:,toti] > 7.0
            c = a * b
            reduced_tpt = np.squeeze(tpt[np.where(c), :])
            xtc = reduced_tpt[:,tci]
            totidx = xtc.argsort()
            xtc = xtc[totidx]
            ytot = reduced_tpt[:,toti]
            ytot = ytot[totidx]
            ax_tot_1tp.plot(xtc, ytot, label='Tp = {}s'.format(v),linewidth=1,color=colors[idx],marker='.')

            rmg /= len(all_tp)
            ax_uni.plot(new_tctp_interv, rmg, label='N_s = {}'.format(nst[0,nsi]),linewidth=1,\
                    color=colorsa[nsidx],marker='.')

            ax.plot([0.1, 0.9], [1.0]*2, ls='--',color='k')
#            ax.plot([0.5]*2, [0.0,5.0], ls='--',color='k')
#            ax.set_ylim(0.0,.0)
            ax.set_xlim(0.2,0.9)
#            ax.add_artist(good_zone)
            ax.set_xlabel('Tc/Tp')
            ax.set_ylabel('max_comp_t/Tc')
#            ax.set_title('Tc/Tp and max_comp_t relation for Ns = {:.0f},N_knots = {:.0f}, Scenario = {:.0f}'\
#                    .format(nst[0,nsi], knt[0,nkni], scnt[0,nobsti]))
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels)
            handles, labels = axtot.get_legend_handles_labels()
            axtot.legend(handles, labels)
#            axtot.set_ylim(7.16,8.6)
            axtot.set_xlabel('Tc (s)')
            axtot.set_ylabel('Ttot (s)')
            fig[fidx].set_size_inches(1.2*18.5/2.54,1.2*10.5/2.54)
            fig[fidx].savefig(direc_scen+'/c{:.0f}.eps'.format(nst[0,nsi]), bbox_inches='tight', dpi=100)
            fig[fidx].savefig(direc_scen+'/c{:.0f}.png'.format(nst[0,nsi]), bbox_inches='tight', dpi=100)
            figtot[fidx].set_size_inches(1.2*18.5/2.54,1.2*10.5/2.54)
            figtot[fidx].savefig(direc_scen+'/tot{:.0f}.eps'.format(nst[0,nsi]), bbox_inches='tight', dpi=100)
            figtot[fidx].savefig(direc_scen+'/tot{:.0f}.png'.format(nst[0,nsi]), bbox_inches='tight', dpi=100)
            fig_tot_1tp[f_tot_1tp_idx].set_size_inches(1.2*18.5/2.54,1.2*10.5/2.54)
            fig_tot_1tp[f_tot_1tp_idx].savefig(direc_scen+'/tot1tp{:.0f}.eps'.format(nst[0,nsi]), bbox_inches='tight', dpi=100)
            fig_tot_1tp[f_tot_1tp_idx].savefig(direc_scen+'/tot1tp{:.0f}.png'.format(nst[0,nsi]), bbox_inches='tight', dpi=100)

        ax_uni.plot([0.1, 0.9], [1.0]*2, ls='--',color='k')
#        ax_uni.plot([0.5]*2, [0.0,4.0], ls='--',color='k')
#        ax_uni.set_ylim(0.0,.0)
        ax_uni.set_xlim(0.2,0.8)
#        ax_uni.add_artist(good_zone)
        ax_uni.set_xlabel('Tc/Tp')
        ax_uni.set_ylabel('max_comp_t/Tc')
#        ax_uni.set_title('Tc/Tp and max_comp_t relation for N_knots = {:.0f}, Scenario = {:.0f}'\
#                .format(knt[0,nkni], scnt[0,nobsti]))
        handles, labels = ax_uni.get_legend_handles_labels()
        ax_uni.legend(handles, labels)
        fig_uni[funiidx].set_size_inches(1.2*18.5/2.54,1.2*10.5/2.54)
        fig_uni[funiidx].savefig(direc_scen+'/uni.eps', bbox_inches='tight', dpi=100)
        fig_uni[funiidx].savefig(direc_scen+'/uni.png', bbox_inches='tight', dpi=100)
#plt.show()

