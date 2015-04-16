import math
import numpy as np
import os
tab=open('./table.csv', 'w+')
delim = " "
N_s = 20; n_knots = 6; acc = 1e-4; maxit = 50;
nTp = 15
nTc = 15
Tp = np.linspace(1.0, 6.0, nTp)
for tp in Tp:
    Tc = np.linspace(tp/5.0, 3.0*tp/4.0, nTc)
    for tc in Tc:   
        cmmd = "python planning-sim-multirob-time-analysis.py"+\
                " "+str(tc)+\
                " "+str(tp)+\
                " "+str(N_s)+\
                " "+str(n_knots)+\
                " "+str(acc)+\
                " "+str(maxit)
        os.system(cmmd)

        with open("planning-sim-multirob-time-analysis"+
                "_"+str(tc)+\
                "_"+str(tp)+\
                "_"+str(N_s)+\
                "_"+str(n_knots)+\
                "_"+str(acc)+\
                "_"+str(maxit)+\
                ".log") as f:
            key_words = ["INFO:MAX:","INFO:MIN:","INFO:AVG:","INFO:TOT:"]
            values = range(len(key_words))
            for line in f.readlines()[-6:]:
                for i in range(len(key_words)):
                    if key_words[i] in line:
                        s_line = line.strip().split(delim)
                        svalue = s_line[s_line.index(key_words[i])+1]
                        values[i] = float(svalue)
            oi = values
            ii = [tc, tp, N_s, n_knots, acc, maxit]
            # do something with it
            string = "{},{},{},{},{},{},{},{},{},{}\n".format(
                    ii[0],
                    ii[1],
                    ii[2],
                    ii[3],
                    ii[4],
                    ii[5],
                    oi[0],
                    oi[1],
                    oi[2],
                    oi[3])
            print(string)
            tab.write(string)
