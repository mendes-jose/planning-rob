import planning_sim as ps
import numpy as np

n_robots = 2
n_obsts = 2
Tc = 0.5
Tp = 2.0
Td = 2.0
N_s = 14
n_knots = 5
acc = 1e-3
maxit = 20
fs_maxit = 50
ls_maxit = 25
deps = 5.0
seps = 0.1
drho = 3.0
ls_min_dist = 0.5
ls_time_opt_scale = 1.0
dist_opt_offset = 10.0 # unused

scriptname = ps.parse_cmdline()

name_id = '_'+str(n_obsts)+\
        '_'+str(Tc)+\
        '_'+str(Tp)+\
        '_'+str(N_s)+\
        '_'+str(n_knots)+\
        '_'+str(acc)+\
        '_'+str(maxit)+\
        '_'+str(fs_maxit)+\
        '_'+str(ls_maxit)+\
        '_'+str(deps)+\
        '_'+str(seps)+\
        '_'+str(drho)+\
        '_'+str(ls_min_dist)+\
        '_'+str(ls_time_opt_scale)+\
        '_'+str(dist_opt_offset)

fname = scriptname[0:-3]+name_id+'.log'

#logging.basicConfig(filename=fname, format='%(levelname)s:%(message)s', \
#        filemode='w', level=logging.DEBUG)
ps.logging.basicConfig(format='%(levelname)s:%(message)s', level=ps.logging.DEBUG)

boundary = ps.Boundary([-12.0, 12.0], [-12.0, 12.0])

#obst_info = rand_round_obst(n_obsts, Boundary([-1., 1.], [0.8, 5.2]))
#print 'OBSTS\n', obst_info

# 0 obsts
if n_obsts == 0:
    obst_info = []
elif n_obsts == 2:
    obst_info = [#([0.0, 1.6], 0.3),
            ([0.6, 3.0], 0.35), ([-0.6, 3.0], 0.35)]
# 3 obsts
elif n_obsts == 3:
    #obst_info = [([0.55043504350435046, 1.9089108910891091], 0.31361636163616358),
    #        ([-0.082028202820282003, 3.6489648964896491], 0.32471747174717469),
    #        ([0.37749774977497741, 4.654905490549055], 0.16462646264626463)]
    obst_info = [([0.0, 1.6], 0.3),
            ([0.6, 3.0], 0.35), ([-0.6, 3.0], 0.35)]
# 6 obsts
elif n_obsts == 6:
    obst_info = [([-0.35104510451045101, 1.3555355535553557], 0.38704870487048704),
            ([0.21441144114411448, 2.5279927992799281], 0.32584258425842583),
            ([-0.3232123212321232, 4.8615661566156621], 0.23165816581658166),
            ([0.098239823982398278, 3.975877587758776], 0.31376637663766377),
            ([0.62277227722772288, 1.247884788478848], 0.1802030203020302),
            ([1.16985698569856988, 3.6557155715571559], 0.25223522352235223)]
else:
    ps.logging.info("Only 3 or 6 obstacles configurations are permited")
    ps.logging.info("Using 3 obstacles configuration")
    obst_info = [([0.55043504350435046, 1.9089108910891091], 0.31361636163616358),
            ([-0.082028202820282003, 3.6489648964896491], 0.32471747174717469),
            ([0.37749774977497741, 4.654905490549055], 0.16462646264626463)]

obstacles = [ps.RoundObstacle(i[0], i[1]) for i in obst_info]
#obstacles += [PolygonObstacle(np.array([[0,1],[1,0],[3,0],[4,2]]))]

kine_models = [ps.UnicycleKineModel(
        [-0.4, 0., np.pi/2.], # q_initial
        [0.4,  5.0, np.pi/2.], # q_final
        [0.0,  0.0],          # u_initial
        [0.0,  0.0],          # u_final
        [1.0,  5.0]),          # u_max
        ps.UnicycleKineModel(
        [0.4,  0., np.pi/2.], # q_initial
        [-0.4, 5.0, np.pi/2.], # q_final
        [0.0,  0.0],          # u_initial
        [0.0,  0.0],          # u_final
        [1.0,  5.0])]          # u_max

robots = []
for i in range(n_robots):
    if i-1 >= 0 and i+1 < n_robots:
        neigh = [i-1, i+1]
    elif i-1 >= 0:
        neigh = [i-1]
    else:
        neigh = [i+1]
    robots += [ps.Robot(
        i,                      # Robot ID
        kine_models[i],         # kinetic model
        obstacles,              # all obstacles
        boundary,               # planning plane boundary
        neigh,                  # neighbors to whom this robot shall talk ...
                                #...(used for conflict only, not for real comm between process)
        N_s=N_s,                # numbers samplings for each planning interval
        n_knots=n_knots,        # number of knots for b-spline interpolation
        Tc=Tc,                  # computation time
        Tp=Tp,                  # planning horizon
        Td=Tp,
        def_epsilon=deps,       # in meters
        safe_epsilon=seps,      # in meters
        detec_rho=drho,
        ls_time_opt_scale = ls_time_opt_scale,
        dist_opt_offset = dist_opt_offset,          # TODO param deprecated
        ls_min_dist = ls_min_dist)]                 # planning horizon (for stand alone plan)

[r.set_option('acc', acc) for r in robots] # accuracy (hard to understand the physical meaning of this)
[r.set_option('maxit', maxit) for r in robots] # max number of iterations for intermediary steps
[r.set_option('ls_maxit', ls_maxit) for r in robots] # max number of iterations for the last step
[r.set_option('fs_maxit', fs_maxit) for r in robots] # max number of iterations for the first step

world_sim = ps.WorldSim(name_id, robots, obstacles, boundary) # create the world

summary_info = world_sim.run() # run simulation

