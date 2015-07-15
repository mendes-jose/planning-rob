import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab
import math 

# Trajectory Generation
class Trajectory_Generation(object):
  def __init__(self):
  # Public Variables and Parameters
    self.x_0 = 0.0
    self.y_0 = 0.0
    self.theta_0 = 0.0
    self.s_0 = 0.0
    self.k_0 = 0.0

    self.x_f = 10.0
    self.y_f = 2.0
    self.theta_f = math.pi/2
    self.k_f = 0.0
    #self.X_f = np.matrix([[x_f],[y_f],[theta_f],[k_f]])
    
    self.n_pas = 200 # number of steps for Simpson method integral approximation
    self.n_newt = 20 # number of iterations for the Newton zero approximation method

  # Unknowns
    self.a = 0.0
    self.b = 0.0
    self.c = 0.0
    self.d = 0.0
    self.s_f = 0.0
    #self.Q = np.matrix([[a],[b],[c],[d],[s_f]])

  # Initializations
    self.a = self.k_0
    self.d = 0.000
    self.s_f = np.sqrt(self.x_f**2 + self.y_f**2)*(1 + self.theta_f*self.theta_f / 5)
    
    self.QQ = np.matrix([[self.b],[self.c]])
    
    s_f = self.s_f
    k_f = self.k_f
    k_0 = self.k_0
    d = self.d
    theta_f = self.theta_f
    QQ = self.QQ
    
    A = np.matrix([[s_f,s_f**2],[(s_f**2)/2,(s_f**3)/3]])
    B = np.matrix([[k_f-k_0-d*(s_f**3)],[theta_f-k_0*s_f-(d/4)*s_f**4]])
    QQ = np.linalg.inv(A)*B
    self.b = QQ[0,0]
    self.c = QQ[1,0]
    
    a = self.a
    b = self.b
    c = self.c
    d = self.d
    
    n_pas = self.n_pas
    resolution = s_f/n_pas
    #n_pas = int(s_f/resolution) 
    
    k = np.matrix([[0.0,0.0]])
    k[0,1] = a
    for i in range(1,n_pas+1):
      ds = resolution*i
      k_i = a + b*ds + c*(ds**2) + d*(ds**3)
      k = np.append(k, np.matrix([[ds,k_i]]), axis = 0)
      
    self.k = k
    #print(self.k[:,1])
    
    s = 0
    theta = np.matrix([[0.0,0.0]])
    theta[0,1] = 0.0
    for i in range(1,n_pas+1):
      s = resolution*i
      theta_i = a*s + b*(s**2)/2 + c*(s**3)/3 + d*(s**4)/4
      theta_i = angle(theta_i)
      theta = np.append(theta, np.matrix([[s,theta_i]]), axis = 0)
    self.theta = theta
    
    # w = np.matrix([[1]])
    # #ponderation w_i
    # for i in range(1,n_pas+1):
      # if i%2==0:
        # wi = 2
      # else:
        # wi = 4
      # if i == n_pas:
        # wi = 1
      # w = np.append(w, np.matrix([[wi]]), axis = 0)
    # #print(w)
    
    f = np.matrix([[np.cos(self.theta[0,1])]])
    #function f = cos(theta)
    for i in range(1,n_pas+1):
      f_i = np.cos(self.theta[i,1])
      f = np.append(f, np.matrix([[f_i]]), axis = 0)
    self.f = f

    g = np.matrix([[np.sin(self.theta[0,1])]])
    #function g = sin(theta)
    for i in range(1,n_pas+1):
      g_i = np.sin(self.theta[i,1])
      g = np.append(g, np.matrix([[g_i]]), axis = 0)
    self.g = g
    
    #self.s_f = np.sqrt(self.x_f**2 + self.y_f**2)*(1 + self.theta_f*self.theta_f / 5)    
    #s_f = self.s_f
    #n_pas = 2000
    #resolution = s_f/n_pas
    s_f_i = 0.0
    x = np.matrix([[s_f_i, self.x_0]])
    #function x = (ds/3)*somme des wi*fi
    y = np.matrix([[s_f_i, self.y_0]])
    #function y = (ds/3)*somme des wi*gi
    for i in range(1,n_pas+1):
      s_f_i = resolution * i
      n_pas_i = i
      #ponderations wi
      wi = np.matrix([[1]])
      for j in range(1,n_pas_i+1):
        if j%2==0:
          w_j = 2
        else:
          w_j = 4
        if j == n_pas_i:
          w_j = 1
        wi = np.append(wi, np.matrix([[w_j]]), axis = 0)
      # function fi identique a f = cos(theta) pour la meme resolution
      # x_j = somme des w_j*f_j
      x_j = 0.0
      for j in range(0,n_pas_i+1):
        x_j += wi[j,0]*f[j,0]
      x_i = (resolution/3)*x_j
      x = np.append(x, np.matrix([[s_f_i, x_i]]), axis = 0)
      # function gi identique a g = sin(theta) pour la meme resolution
      # y_j = somme des w_j*g_j
      y_j = 0.0
      for j in range(0,n_pas_i+1):
        y_j += wi[j,0]*g[j,0]
      y_i = (resolution/3)*y_j
      y = np.append(y, np.matrix([[s_f_i, y_i]]), axis = 0)
    self.x = x
    self.y = y
    #print(self.x)
    
    
  def update(self): # Deplacement du point final (x_f;y_f) a la bonne place
  # Satisfaction de contrainte de positions initial et finale avec methode de Newton
    
    #Initializations
    n_pas = self.n_pas # number of steps for Simpson method integral approximation
    n_newt = self.n_newt # number of iterations for the Newton zero approximation method
    X_b = np.matrix([[self.x_f],[self.y_f],[self.theta_f],[self.k_f]])
    a = self.a
    
    b_k = self.b
    c_k = self.c
    d_k = self.d
    s_f_k = self.s_f
    Q_k = np.matrix([[b_k],[c_k],[d_k],[s_f_k]])
    
    for k in range(1,n_newt+1):
      ## x_k and y_k computation
      #curvature k_k
      #angle theta_k
      #function f_k = cos(theta_k)
      #function g_k = sin(theta_k)
      resolution_k = s_f_k/n_pas
      k_k = np.matrix([[0.0,a]])
      theta_k = np.matrix([[0.0,0.0]])
      f_k = np.matrix([[np.cos(theta_k[0,1])]])
      g_k = np.matrix([[np.sin(theta_k[0,1])]])
      for i in range(1,n_pas+1):
        s = resolution_k*i
        k_i = a + b_k*s + c_k*(s**2) + d_k*(s**3)
        k_k = np.append(k_k, np.matrix([[s,k_i]]), axis = 0)
        theta_i = a*s + b_k*(s**2)/2 + c_k*(s**3)/3 + d_k*(s**4)/4
        theta_i = angle(theta_i)
        theta_k = np.append(theta_k, np.matrix([[s,theta_i]]), axis = 0)
        f_i = np.cos(theta_k[i,1])
        f_k = np.append(f_k, np.matrix([[f_i]]), axis = 0)
        g_i = np.sin(theta_k[i,1])
        g_k = np.append(g_k, np.matrix([[g_i]]), axis = 0)
      #function x_k[i,1] = (ds/3)*somme des w_j*f[j,0]
      #function y_k[i,1] = (ds/3)*somme des w_j*g[j,0]
      s_f_i = 0.0
      x_k = np.matrix([[s_f_i, self.x_0]])
      y_k = np.matrix([[s_f_i, self.y_0]])
      for i in range(1,n_pas+1):
        s_f_i = resolution_k * i
        n_pas_i = i
        #ponderations wi
        wi = np.matrix([[1]])
        for j in range(1,n_pas_i+1):
          if j%2==0:
            w_j = 2
          else:
            w_j = 4
          if j == n_pas_i:
            w_j = 1
          wi = np.append(wi, np.matrix([[w_j]]), axis = 0)
        # functions f_k[j,0] de f_k = cos(theta_k)
        # x_j = somme des w_j*f_j
        x_j = 0.0
        for j in range(0,n_pas_i+1):
          x_j += wi[j,0]*f_k[j,0]
        x_i = (resolution_k/3)*x_j
        x_k = np.append(x_k, np.matrix([[s_f_i, x_i]]), axis = 0)
        # functions g_k[j,0] de g_k = sin(theta_k)
        # y_j = somme des w_j*g_j
        y_j = 0.0
        for j in range(0,n_pas_i+1):
          y_j += wi[j,0]*g_k[j,0]
        y_i = (resolution_k/3)*y_j
        y_k = np.append(y_k, np.matrix([[s_f_i, y_i]]), axis = 0)
      
      ## constraint function H_k
      H_k = np.matrix([[x_k[n_pas,1]],[y_k[n_pas,1]],[theta_k[n_pas,1]],[k_k[n_pas,1]]])
      
      ## constraint Jacobian matrix computation H_dq_k
      # w_k
      w_k = np.matrix([[1]])
      for i in range(1,n_pas+1):
        if i%2==0:
          wi = 2
        else:
          wi = 4
        if i == n_pas:
          wi = 1
        w_k = np.append(w_k, np.matrix([[wi]]), axis = 0)
      # x_db_k  = -(ds/6)*somme des (s^2)*w_k*g_k
      # x_dc_k  = -(ds/9)*somme des (s^3)*w_k*g_k
      # x_dd_k  = -(ds/12)*somme des (s^4)*w_k*g_k
      # y_db_k  = (ds/6)*somme des (s^2)*w_k*f_k
      # y_dc_k  = (ds/9)*somme des (s^3)*w_k*f_k
      # y_dd_k  = (ds/12)*somme des (s^4)*w_k*f_k
      x_db_i = 0.0
      x_dc_i = 0.0
      x_dd_i = 0.0
      y_db_i = 0.0
      y_dc_i = 0.0
      y_dd_i = 0.0
      for i in range(0,n_pas+1):
        si = resolution_k*i
        x_db_i += (si**2)*w_k[i,0]*g_k[i,0]
        x_dc_i += (si**3)*w_k[i,0]*g_k[i,0]
        x_dd_i += (si**4)*w_k[i,0]*g_k[i,0]
        y_db_i += (si**2)*w_k[i,0]*f_k[i,0]
        y_dc_i += (si**3)*w_k[i,0]*f_k[i,0]
        y_dd_i += (si**4)*w_k[i,0]*f_k[i,0]
      x_db_k = -(resolution_k/6)*x_db_i
      x_dc_k = -(resolution_k/9)*x_dc_i
      x_dd_k = -(resolution_k/12)*x_dd_i
      y_db_k = (resolution_k/6)*y_db_i
      y_dc_k = (resolution_k/9)*y_dc_i
      y_dd_k = (resolution_k/12)*y_dd_i
      # x_ds_f_k  = cos(theta_k[n_pas,1])
      x_ds_f_k  = f_k[n_pas,0]
      # y_ds_f_k  = sin(theta_k[n_pas,1])
      y_ds_f_k  = g_k[n_pas,0]
      # theta_db_k  = s_f_k^2/2
      theta_db_k = (s_f_k**2)/2
      # theta_dc_k  = s_f_k^3/3
      theta_dc_k = (s_f_k**3)/3
      # theta_dd_k  = s_f_k^4/4
      theta_dd_k = (s_f_k**4)/4
      # theta_ds_f_k  = k_k[n_pas,1]
      theta_ds_f_k = k_k[n_pas,1]
      # k_db_k  = s_f_k
      k_db_k = s_f_k
      # k_dc_k  = s_f_k^2
      k_dc_k = s_f_k**2
      # k_dd_k  = s_f_k^3
      k_dd_k = s_f_k**3
      # k_ds_f_k  = dk_k/ds_f_k
      k_ds_f_k = b_k + 2*c_k*s_f_k + 3*d_k*(s_f_k**2)
      
      H_dq_k = np.matrix([[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]])
      # x
      H_dq_k[0,0] = x_db_k
      H_dq_k[0,1] = x_dc_k
      H_dq_k[0,2] = x_dd_k
      H_dq_k[0,3] = x_ds_f_k
      # y
      H_dq_k[1,0] = y_db_k
      H_dq_k[1,1] = y_dc_k
      H_dq_k[1,2] = y_dd_k
      H_dq_k[1,3] = y_ds_f_k
      # theta
      H_dq_k[2,0] = theta_db_k
      H_dq_k[2,1] = theta_dc_k
      H_dq_k[2,2] = theta_dd_k
      H_dq_k[2,3] = theta_ds_f_k
      # k
      H_dq_k[3,0] = k_db_k
      H_dq_k[3,1] = k_dc_k
      H_dq_k[3,2] = k_dd_k
      H_dq_k[3,3] = k_ds_f_k
      
      # Q_k+1 = H_dq_k^-1 * (X_b-H_k) + Q_k
      Q_new = np.linalg.inv(H_dq_k)*(X_b-H_k) + Q_k
      Q_k = Q_new
      b_k = Q_new[0,0]
      c_k = Q_new[1,0]
      d_k = Q_new[2,0]
      s_f_k = Q_new[3,0]
      
    self.x_k = x_k
    self.y_k = y_k
      
  def __theta__(self, b, c, d, s):
    theta = self.a*s + (b/2)*s*s + (c/3)*s*s*s + (d/4)*s*s*s*s
    return theta
    
  def __k__(self, b, c, d, s):
    k = self.a + b*s + c*s*s + d*s*s*s
    return k
    
    
def angle(x):
  pi = math.pi
  twopi = 2*pi
  return (x+pi)%twopi-pi

# Initializations
Gene = Trajectory_Generation()
Gene.update()

# Plot
fig_curve = plt.figure()
ax_curve = fig_curve.add_subplot(111)
line_curve = mpl.lines.Line2D(Gene.k[:,0], Gene.k[:,1], c = 'blue', ls = '-',lw = 1)
ax_curve.add_line(line_curve)
plt.xlim([-1.0, 11.0])
plt.ylim([-60.0, 60.0])
plt.title('Curvature')
plt.xlabel('s (m)')
plt.ylabel('k (m^(-1))')
#plt.legend(('front', 'rear'), loc='lower right')
plt.grid()
plt.draw()

fig_theta = plt.figure()
ax_theta = fig_theta.add_subplot(111)
line_theta = mpl.lines.Line2D(Gene.theta[:,0], Gene.theta[:,1], c = 'blue', ls = '-',lw = 1)
ax_theta.add_line(line_theta)
plt.xlim([-1.0, 11.0])
plt.ylim([-10.0, 180.0])
plt.title('Yaw angle')
plt.xlabel('s (m)')
plt.ylabel('theta (rad)')
plt.grid()
plt.draw()

fig_xy = plt.figure()
ax_xy = fig_xy.add_subplot(111)
line_x = mpl.lines.Line2D(Gene.x[:,0], Gene.x[:,1], c = 'blue', ls = '-',lw = 1)
line_y = mpl.lines.Line2D(Gene.y[:,0], Gene.y[:,1], c = 'red', ls = '-',lw = 1)
ax_xy.add_line(line_x)
ax_xy.add_line(line_y)
plt.xlim([-1.0, 11.0])
plt.ylim([-10.0, 180.0])
plt.title('pose')
plt.xlabel('s (m)')
plt.ylabel('pose (m)')
plt.legend(('x', 'y'), loc='lower right')
plt.grid()
plt.draw()

fig_pose = plt.figure()
ax_pose = fig_pose.add_subplot(111)
line_pose = mpl.lines.Line2D(Gene.x[:,1], Gene.y[:,1], c = 'blue', ls = '-',lw = 1)
ax_pose.add_line(line_pose)
plt.xlim([-1.0, 11.0])
plt.ylim([-10.0, 180.0])
plt.title('Initial pose')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.grid()
plt.draw()

# fig_pose = plt.figure()
# ax_pose = fig_pose.add_subplot(111)
# ax_pose.plot(Gene.x_k[:,1], Gene.y_k[:,1], 'b^')
# plt.xlim([-1.0, 11.0])
# plt.ylim([-10.0, 180.0])
# plt.title('Pose with position constraint')
# plt.xlabel('x (m)')
# plt.ylabel('y (m)')
# plt.grid()
# plt.draw()

fig_pose = plt.figure()
ax_pose = fig_pose.add_subplot(111)
line_pose = mpl.lines.Line2D(Gene.x_k[:,1], Gene.y_k[:,1], c = 'blue', ls = '-',lw = 1)
ax_pose.add_line(line_pose)
plt.xlim([-1.0, 11.0])
plt.ylim([-10.0, 180.0])
plt.title('Pose with position constraint')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.grid()
plt.draw()

plt.show()