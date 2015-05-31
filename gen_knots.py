import numpy as np

def _gen_knots(t_init, t_final, d, n_knots):
       """ generate b-spline knots given initial and final times.
       """
       gk = lambda x:t_init + (x-(d-1.0))*(t_final-t_init)/n_knots
       knots = [t_init for _ in range(d)]
       knots.extend([gk(i) for i in range(d, d+n_knots-1)])
       knots.extend([t_final for _ in range(d)])
       return np.asarray(knots)

k=4
spl_d=3

print (len(_gen_knots(0.0, 5.0, spl_d+1, k)))
print (_gen_knots(0.0, 5.0, spl_d+1, k))
print (k+1 + 2*(spl_d))
