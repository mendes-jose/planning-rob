import planning_sim as ps
import numpy as np
import os, sys

ls_time_opt_scale = 1.0

options = {'acc': 0.001, 'no_obsts': 3, 'no_robots': 2, 'time_p': 2.0, 'time_c': 0.5,
        'plot': False, 'no_s': 14, 'max_it': 15, 'ls_min_dist': 0.5, 'drho': 3.0,
        'direc': "./simout", 'savelog': False, 'deps': 5.0, 'f_max_it': 40, 'l_max_it': 25,
        'seps': 0.1, 'no_knots': 5}
scriptname = sys.argv[0]

try:
    os.mkdir(options['direc'])
except OSError:
    print('Probably the output directory '+options['direc']+' already exists.')

name_id = '_'+str(options['no_robots'])+\
        '_'+str(options['no_obsts'])+\
        '_'+str(options['time_c'])+\
        '_'+str(options['time_p'])+\
        '_'+str(options['no_s'])+\
        '_'+str(options['no_knots'])+\
        '_'+str(options['acc'])+\
        '_'+str(options['max_it'])+\
        '_'+str(options['f_max_it'])+\
        '_'+str(options['l_max_it'])+\
        '_'+str(options['deps'])+\
        '_'+str(options['seps'])+\
        '_'+str(options['drho'])+\
        '_'+str(options['ls_min_dist'])

if options['savelog']:
    flog = options.direc+'/'+scriptname[0:-3]+name_id+'.log'
    ps.logging.basicConfig(filename=flog, format='%(levelname)s:%(message)s', \
            filemode='w', level=ps.logging.DEBUG)
else:
    ps.logging.basicConfig(format='%(levelname)s:%(message)s', level=ps.logging.DEBUG)

boundary = ps.Boundary([-12.0, 12.0], [-12.0, 12.0])

# Generate random round obstacles
#obst_info = ps.rand_round_obst(options['no_obsts'], ps.Boundary([-1., 1.], [0.8, 5.2]))

# 0 obsts
if options['no_obsts'] == 0:
    obst_info = []
# 2 obsts
elif options['no_obsts'] == 2:
    obst_info = [#([0.0, 1.6], 0.3),
            ([0.6, 3.0], 0.35), ([-0.6, 3.0], 0.35)]
# 3 obsts
elif options['no_obsts'] == 3:
    obst_info = [([1.16, 0.0], 0.4),
            ([-0.5, 0.52], 0.35), ([-0.51, -.52], 0.35)]
# 6 obsts
elif options['no_obsts'] == 6:
    obst_info = [([-0.35104510451045101, 1.3555355535553557], 0.38704870487048704),
            ([0.21441144114411448, 2.5279927992799281], 0.32584258425842583),
            ([-0.3232123212321232, 4.8615661566156621], 0.23165816581658166),
            ([0.098239823982398278, 3.975877587758776], 0.31376637663766377),
            ([0.62277227722772288, 1.247884788478848], 0.1802030203020302),
            ([1.16985698569856988, 3.6557155715571559], 0.25223522352235223)]
else:
    logging.info("Using 3 obstacles configuration")
    obst_info = [([0.0, 1.6], 0.3),
            ([0.6, 3.0], 0.35), ([-0.6, 3.0], 0.35)]

obstacles = [ps.RoundObstacle(i[0], i[1]) for i in obst_info]

# Polygon obstacle exemple
#obstacles += [ps.PolygonObstacle(np.array([[0,1],[1,0],[3,0],[4,2]]))]

kine_models = [ps.UnicycleKineModel(
        [-2.5, 0.5, .0], # q_initial
        [2.4,  -0.5, .0], # q_final
        [0.0,  0.0],          # u_initial
        [0.0,  0.0],          # u_final
        [1.0,  5.0]),          # u_max
        ps.UnicycleKineModel(
        [-2.46,  -0.53, .0], # q_initial
        [2.39, 0.48, .0], # q_final
        [0.0,  0.0],          # u_initial
        [0.0,  0.0],          # u_final
        [1.0,  5.0])]          # u_max

robots = []
for i in range(options['no_robots']):
    if i-1 >= 0 and i+1 < options['no_robots']:
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
        N_s=options['no_s'],                # numbers samplings for each planning interval
        n_knots=options['no_knots'],# number of knots for b-spline interpolation
        Tc=options['time_c'],       # computation time
        Tp=options['time_p'],       # planning horizon
        Td=options['time_p'],
        def_epsilon=options['deps'],       # in meters
        safe_epsilon=options['seps'],      # in meters
        detec_rho=options['drho'],
        ls_time_opt_scale = ls_time_opt_scale,
        ls_min_dist = options['ls_min_dist'])] 

[r.set_option('acc', options['acc']) for r in robots] 
[r.set_option('maxit', options['max_it']) for r in robots] 
[r.set_option('ls_maxit', options['l_max_it']) for r in robots] 
[r.set_option('fs_maxit', options['f_max_it']) for r in robots]

world_sim = ps.WorldSim(name_id, options['direc'], robots, obstacles, boundary, plot=options['plot'])

summary_info = world_sim.run() # run simulation

