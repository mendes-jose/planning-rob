import numpy as np
import os
import csv
import time
import sys

script_name='planning_sim.py'

def _frange(initial, final, step):
    """ Float point range function with round at the int(round(1./step))+4 decimal position
    """
    _range = []
    n = 0
    while n*step+initial < final:
        _range += [round(n*step+initial, 4+int(round(1./step)))]
        n+=1
    return _range

# output directory
direc = sys.argv[1]

try:
    os.mkdir(direc)
except OSError:
    print('Probably the output directory '+direc+' already exists.')

# output from simulation delimiter
delim = " "

# field's names
input_names = ["Nobsts","Tc","Tp","Ns","Nssol","Nkn","Acc","MaxIt","FS_MaxIt","LS_MaxIt","Deps","Seps","Drho","LS_mdist","LS_topt","Dopt"]
key_words = ["MOB:","PEN:","NSE:","FIR:","LAS:","LMA:","MAX:","MIN:","AVG:","RMP:","RMG:","TOT:"]

# write first line with the fields names 
#with open('../traces/tableData2/table.csv', 'w') as csv_file:
with open(direc+'/table.csv', 'w') as csv_file:
    table_writer = csv.writer(csv_file, delimiter=',',quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
    table_writer.writerow(input_names+[kw[0:-1] for kw in key_words])

init_time = time.strftime("%H:%M:%S")

#raw_input()
# parameters definition
#stepTp = 0.1
#stepTc = 0.1
acc = 1e-3; maxit = 15; ls_maxit = 26; fs_maxit = 40;
drho = 4.0; ls_min_dist = 0.5; ls_time_opt = 1.0;
deps = 5.0; seps = 0.1;
N_ssol = 50;
dist_opt = 1e1 # param deprecated TODO
#for N_ssol in range(10*10, 200*10, 5):
for N_ssol in range(10, 30, 1):
    #for drho in _frange(1.2, 15.0, 0.5):
    for drho in [15.0]:
        for n_obsts in [7]:
            for n_knots in [5]:
        #        for N_s in range(n_knots+3+1, (n_knots+3+1)+3, 1):
                for N_s in [10]:
        #            for tp in np.linspace(2.0, 6.0, 10, endpoint=False):
                    tp_i = .8
                    tp_f = 5.0
                    tp_s = 0.1
                    #for tp in np.linspace(tp_i, tp_f, (tp_f-tp_i)/tp_s+1, endpoint=True):
                    for tp in [2.4]:
                        tc_i = 0.2
                        tc_s = 0.1
        #                for tc in _frange(tc_i, tp-0.3, tc_s):
                        for tc in [0.6]:
        #planning_sim.py -b1 -L -P../traces/rt-tcfix -o3 -c1.0 -p3.8 -s10 -k6 -a0.001 -m15 -i40 -I25 -d5.0 -f0.1 -r4.0 -l0.5
    #                        if n_obsts < 3:
    #                            print '----------------------------'
    #                            continue
    #                        elif n_knots < 6:
    #                            print '----------------------------'
    #                            continue
    #                        elif N_s < 10:
    #                            print '----------------------------'
    #                            continue
    #                        elif tp < 3.8:
    #                            print '----------------------------'
    #                            continue
                            cmmd = "python "+script_name+\
                                    " -b1 -L"\
                                    " -P"+direc+\
                                    " -o"+str(n_obsts)+\
                                    " -c"+str(tc)+\
                                    " -p"+str(tp)+\
                                    " -s"+str(N_s)+\
                                    " -S"+str(N_ssol)+\
                                    " -k"+str(n_knots)+\
                                    " -a"+str(acc)+\
                                    " -m"+str(maxit)+\
                                    " -i"+str(fs_maxit)+\
                                    " -I"+str(ls_maxit)+\
                                    " -d"+str(deps)+\
                                    " -f"+str(seps)+\
                                    " -r"+str(drho)+\
                                    " -l"+str(ls_min_dist)
                            # run simulation
                            print 'Running '+cmmd
                            os.system(cmmd)
                            print cmmd+' is DONE'
                    
                            # open log file
                            with open(direc+"/"+script_name[0:-3]+
                                    "_1_"+str(n_obsts)+\
                                    "_"+str(tc)+\
                                    "_"+str(tp)+\
                                    "_"+str(N_s)+\
                                    "_"+str(N_ssol)+\
                                    "_"+str(n_knots)+\
                                    "_"+str(acc)+\
                                    "_"+str(maxit)+\
                                    "_"+str(fs_maxit)+\
                                    "_"+str(ls_maxit)+\
                                    "_"+str(deps)+\
                                    "_"+str(seps)+\
                                    "_"+str(drho)+\
                                    "_"+str(ls_min_dist)+\
                                    ".log") as f:
                                values = range(len(key_words))
                                for line in f:
                                    for i in range(len(key_words)):
                                        if key_words[i] in line:
                                            s_line = line.strip().split(delim)
                                            svalue = s_line[s_line.index(key_words[i])+1]
                                            values[i] = float(svalue)
                                oi = values
                                ii = [n_obsts, tc, tp, N_s, N_ssol, n_knots, acc, maxit, fs_maxit, ls_maxit, deps, seps, drho, ls_min_dist, ls_time_opt, dist_opt]
                    
                                # csv file
                                with open(direc+'/table.csv', 'a') as csv_file:
                                    table_writer = csv.writer(csv_file, delimiter=',',quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
                                    table_writer.writerow(ii+oi)
                                print('TIME: '+time.strftime("%H:%M:%S"))
print('Init time: '+init_time)
