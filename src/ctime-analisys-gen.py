import numpy as np
import os
import csv

# output from simulation delimiter
delim = " "

# field's names
input_names = ["Tc","Tp","Ns","Nkn","Acc","MaxIt"]
key_words = ["NSE:","FIR:","LAS:","MAX:","MIN:","AVG:","RAT:","TOT:"]

# write first line with the fields names 
#with open('../traces/tableData2/table.csv', 'w') as csv_file:
with open('./fistline.csv', 'w') as csv_file:
    table_writer = csv.writer(csv_file, delimiter=',',quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
    table_writer.writerow(input_names+[kw[0:-1] for kw in key_words])

raw_input()
# parameters definition
stepTp = 0.1
stepTc = 0.1
N_s = 20; n_knots = 6; acc = 1e-4; maxit = 50;

for tp in np.linspace(1.0, 11.0, 100, endpoint=False):
    for tc in np.linspace(tp/10.0, 9.0*tp/10.0, int(round(8.0*tp/10.0/0.1)), endpoint=False):
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
        with open("../traces/tableData2/planning-sim-multirob-time-analysis"+
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

            # csv file
            with open('../traces/tableData2/table.csv', 'a') as csv_file:
                table_writer = csv.writer(csv_file, delimiter=',',quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
                table_writer.writerow(ii+oi)
            print(tc, tp)
