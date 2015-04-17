import numpy as np
import os
import csv

# csv file
csv_file = open('../../../Dropbox/traces/table.csv', 'w+')
table_writer = csv.writer(csv_file, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)

# output from simulation delimiter
delim = " "

# field's names
input_names = ["Tc","Tp","Ns","Nkn","Acc","MaxIt"]
key_words = ["NSE:","FIR:","LAS:","MAX:","MIN:","AVG:","TOT:"]

# write first line with the fields names
table_writer.writerow(input_names+[kw[0:-1] for kw in key_words])

# parameters definition
nTp = 25
nTc = 25
N_s = 20; n_knots = 6; acc = 1e-4; maxit = 100;

for tp in np.linspace(0.5, 15.0, nTp):
    for tc in np.linspace(tp/10.0, tp, nTc):
        cmmd = "python planning-sim-multirob-time-analysis.py"+\
                " "+str(tc)+\
                " "+str(tp)+\
                " "+str(N_s)+\
                " "+str(n_knots)+\
                " "+str(acc)+\
                " "+str(maxit)
        # run simulation
        os.system(cmmd)

        # open log file
        with open("planning-sim-multirob-time-analysis"+
                "_"+str(tc)+\
                "_"+str(tp)+\
                "_"+str(N_s)+\
                "_"+str(n_knots)+\
                "_"+str(acc)+\
                "_"+str(maxit)+\
                ".log") as f:
            values = range(len(key_words))
            for line in f:
                for i in range(len(key_words)):
                    if key_words[i] in line:
                        s_line = line.strip().split(delim)
                        svalue = s_line[s_line.index(key_words[i])+1]
                        values[i] = float(svalue)
            oi = values
            ii = [tc, tp, N_s, n_knots, acc, maxit]

            table_writer.writerow(ii+oi)
