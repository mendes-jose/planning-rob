import numpy as np
import os
import csv
import time
import sys

# output directory
direc = sys.argv[1]

os.system("mkdir -p "+direc)

# output from simulation delimiter
delim = " "

# field's names
input_names = ["Nobsts","Tc","Tp","Ns","Nkn","Acc","MaxIt","FS_MaxIt","LS_MaxIt","Deps","Seps","Drho","LS_mdist","LS_topt","Dopt"]
key_words = ["NSE:","FIR:","LAS:","LMA:","MAX:","MIN:","AVG:","RMP:","RMG:","TOT:"]

# write first line with the fields names 
#with open('../traces/tableData2/table.csv', 'w') as csv_file:
with open(direc+'table.csv', 'w') as csv_file:
    table_writer = csv.writer(csv_file, delimiter=',',quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
    table_writer.writerow(input_names+[kw[0:-1] for kw in key_words])

init_time = time.strftime("%H:%M:%S")

#raw_input()
# parameters definition
#stepTp = 0.1
#stepTc = 0.1
acc = 1e-3; maxit = 15; ls_maxit = 20; fs_maxit = 40;
deps = 5.0; seps = 0.1; drho = 3.0; ls_min_dist = 0.5; ls_time_opt = 1.0;
dist_opt = 1e1 # param deprecated TODO
for n_knots in [4, 5, 6]:
    for n_obsts in [0, 3, 6]:
        for N_s in range(n_knots+3+1, 2*(n_knots+3+1), 1):
#            for tp in np.linspace(2.0, 6.0, 10, endpoint=False):
            for tp in np.linspace(0.8, 1.6, 3, endpoint=True):
                for tc in np.linspace(tp/5., 4.*tp/5., int(round(3.*tp/5./0.3)), endpoint=False):
                    cmmd = "python planning-sim-multirob-integration-time-analysis.py"+\
                            " "+direc+\
                            " "+str(n_obsts)+\
                            " "+str(tc)+\
                            " "+str(tp)+\
                            " "+str(N_s)+\
                            " "+str(n_knots)+\
                            " "+str(acc)+\
                            " "+str(maxit)+\
                            " "+str(fs_maxit)+\
                            " "+str(ls_maxit)+\
                            " "+str(deps)+\
                            " "+str(seps)+\
                            " "+str(drho)+\
                            " "+str(ls_min_dist)+\
                            " "+str(ls_time_opt)+\
                            " "+str(dist_opt)
                    # run simulation
                    print cmmd
                    os.system(cmmd)
                    print 'Done'
            
                    # open log file
                    with open(direc+"planning-sim-multirob-integration-time-analysis"+
                            "_"+str(n_obsts)+\
                            "_"+str(tc)+\
                            "_"+str(tp)+\
                            "_"+str(N_s)+\
                            "_"+str(n_knots)+\
                            "_"+str(acc)+\
                            "_"+str(maxit)+\
                            "_"+str(fs_maxit)+\
                            "_"+str(ls_maxit)+\
                            "_"+str(deps)+\
                            "_"+str(seps)+\
                            "_"+str(drho)+\
                            "_"+str(ls_min_dist)+\
                            "_"+str(ls_time_opt)+\
                            "_"+str(dist_opt)+\
                            ".log") as f:
                        values = range(len(key_words))
                        for line in f:
                            for i in range(len(key_words)):
                                if key_words[i] in line:
                                    s_line = line.strip().split(delim)
                                    svalue = s_line[s_line.index(key_words[i])+1]
                                    values[i] = float(svalue)
                        oi = values
                        ii = [n_obsts, tc, tp, N_s, n_knots, acc, maxit, fs_maxit, ls_maxit, deps, seps, drho, ls_min_dist, ls_time_opt, dist_opt]
            
                        # csv file
                        with open(direc+'table.csv', 'a') as csv_file:
                            table_writer = csv.writer(csv_file, delimiter=',',quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
                            table_writer.writerow(ii+oi)
                        print('TIME: '+time.strftime("%H:%M:%S"))
print('Init time: '+init_time)
