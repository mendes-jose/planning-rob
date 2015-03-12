import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si

d = 4
C = [[0, 0], [0, 2], [2, 3], [4, 0], [6, 3], [8, 2], [8, 0]]
C = np.array(C)
x = C[:,0]
y = C[:,1]

t = range(len(C))
ipl_t = np.linspace(0.0, len(C) - 1, 100)

x_tup = si.splrep(t, C[:,0], k=d)
print '-------------------'
print x_tup
print '-------------------'
y_tup = si.splrep(t, C[:,1], k=d)

x_list = list(x_tup)
print x_list
xl = x.tolist()
print xl
x_list[1] = xl + [0.0, 0.0, 0.0, 0.0]
print '------------------'
print x_list

y_list = list(y_tup)
yl = y.tolist()
y_list[1] = yl + [0.0, 0.0, 0.0, 0.0]

print 'time', ipl_t
print 'xlist', x_list

x_i = si.splev(ipl_t, x_list)
print 'XI', x_i
y_i = si.splev(ipl_t, y_list)

#==============================================================================
# Plot
#==============================================================================

fig = plt.figure()

ax = fig.add_subplot(231)
plt.plot(t, x, '-og')
plt.plot(ipl_t, x_i, 'r')
plt.xlim([0.0, max(t)])
plt.title('Splined x(t)')

ax = fig.add_subplot(232)
plt.plot(t, y, '-og')
plt.plot(ipl_t, y_i, 'r')
plt.xlim([0.0, max(t)])
plt.title('Splined y(t)')

ax = fig.add_subplot(233)
plt.plot(x, y, '-og')
plt.plot(x_i, y_i, 'r')
plt.xlim([min(x) - 0.3, max(x) + 0.3])
plt.ylim([min(y) - 0.3, max(y) + 0.3])
plt.title('Splined f(x(t), y(t))')

ax = fig.add_subplot(234)
for i in range(7):
    vec = np.zeros(11)
    vec[i] = 1.0
    x_list = list(x_tup)
    x_list[1] = vec.tolist()
    x_i = si.splev(ipl_t, x_list)
    plt.plot(ipl_t, x_i)
plt.xlim([0.0, max(t)])
plt.title('Basis splines')
plt.show()
