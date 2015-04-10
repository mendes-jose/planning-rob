#!/usr/bin/python

import numpy as np
import time

def gen_knots(d, n, t_init, t_fin):
  knots = [t_init]
  for j in range(1,d):
    knots_j = t_init
    knots = knots + [knots_j]

  for j in range(d,d+n):
    knots_j = t_init + (j-(d-1.0))* \
        (t_fin-t_init)/n
    knots = knots + [knots_j]

  for j in range(d+n,2*d-1+n):
    knots_j = t_fin
    knots = knots + [knots_j]
  return knots

def gen_knots2(d, n, t_init, t_final):
  gk = lambda x:t_init + (x-(d-1.0))*(t_final-t_init)/n
  knots = [t_init for _ in range(d)]
#  knots += list(np.linspace(t_init, t_final, n+1))
  knots.extend([gk(i) for i in range(d,d+n-1)])
#  knots += [t_init for _ in range(n+1)]
  knots.extend([t_final for _ in range(d)])
  return knots

tic = time.time()
l1 = gen_knots(4,int(1e4),0.0,10.0)
print(time.time()-tic)
tic = time.time()
l2 = gen_knots2(4,int(1e4),0.0,10.0)
print(time.time()-tic)
print(l1==l2)
#print(l1)
#print(l2)
print(len(l1),len(l2))
