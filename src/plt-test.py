#import matplotlib as mpl
#import matplotlib.pyplot as plt
#
#fig = plt.figure(1)
#
#circle1=plt.Circle((0,0),2,color='r')
## now make a circle with no fill, which is good for hilighting key results
#circle2=plt.Circle((5,5),.5,color='b',fill=False)
#circle3=plt.Circle((10,10),2,color='g',clip_on=False)
#ax = fig.gca()
##ax.cla() # clear things for fresh plot
## change default range so that new circles will work
##ax.set_xlim((0,10))
##ax.set_ylim((0,10))
## some data
#ax.plot(range(11),'o',color='black')
## key data point that we are encircling
#ax.plot((5),(5),'o',color='y')
#
#ax.add_artist(circle1)
#ax.add_artist(circle2)
#ax.add_artist(circle3)
#plt.show()

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 6*np.pi, 100)
y = np.sin(x)

# You probably won't need this if you're embedding things in a tkinter plot...
#plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)
(line1,line2,) = ax.plot(x, y, 'r-', x, y+1) # Returns a tuple of line objects, thus the comma
print('line info', line1)
for phase in np.linspace(0, 10*np.pi, 500):
    line1.set_ydata(np.sin(x + phase))
    line1.set_ydata(np.sin(x + phase))
    fig.canvas.draw()
plt.show()
