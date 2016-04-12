# All robots opt together

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
#import scipy.interpolate as si
#import time
#import itertools
#import multiprocessing as mpc
#import sys
#import os
#import logging
from scipy.optimize import fmin_slsqp
#from optparse import OptionParser
import xml.etree.ElementTree as ET
import csv


class Robot:

	def __init__(self, myID, direc, u0, w0, x0, y0, theta0):

		self.id = myID

		self._u0 = u0
		self._w0 = w0
		self._x0 = x0
		self._y0 = y0
		self._theta0 = theta0

		with open(direc+'build/out/Release/bin/ctrl_ts_AdeptLynx' + str(myID) + '.csv', 'rb') as csvfile:
			treader = csv.reader(csvfile, delimiter=',',quotechar='|',quoting=csv.QUOTE_NONE)
			tlist = list(treader)
			rows = len(tlist)-1 # last line can be incomplete (e.g. if program stopped with ctrl-c)
			cols = len(tlist[0][0:-1]) # last column has nothing, only \n
			self.ctrlTabl = np.zeros((rows,cols))
			for i in range(rows):
				self.ctrlTabl[i] = np.array([float(t) for t in tlist[i][0:-1]])

		with open(direc+'build/out/Release/bin/real_ts_AdeptLynx' + str(myID) + '.csv', 'rb') as csvfile:
			treader = csv.reader(csvfile, delimiter=',',quotechar='|',quoting=csv.QUOTE_NONE)
			tlist = list(treader)
			rows = len(tlist)-1 # last line can be incomplete (e.g. if program stopped with ctrl-c)
			cols = len(tlist[0][0:-1]) # last column has nothing, only \n
			self.realTabl = np.zeros((rows,cols))
			for i in range(rows):
				self.realTabl[i] = np.array([float(t) for t in tlist[i][0:-1]])

		self.rows = max(self.realTabl.shape[0], self.ctrlTabl.shape[0])

		self.resize()

		# self._dyModelParam = np.array([1, 1, 1, 1, 1, 1])
		#self._dyModelParam = np.array([ 0.07320565,  0.2671858 , -0.00537168,  1.01050981,  0.79198508,  0.1762324 ])
		#self._dyModelParam = np.array([ 1./0.06047654,  1./0.26907369, -0.00282409,  1.00003766,  0.98050515, -0.03475284])
		self._dyModelParam = np.array([ 0.07315401,  0.26722134, -0.00535513,  1.01043586,  0.78976158,  0.17841226])
		self._acc = 1e-6
		self._maxit = 100

		self._simTime = self.realTabl[:,0]

		self._computeApproxLinAccel()
		self._computeApproxAngAccel()


	def resize(self):
		if self.ctrlTabl.shape[0] < self.rows:
			for _ in range(self.rows-self.ctrlTabl.shape[0]):
				self.ctrlTabl = np.append(self.ctrlTabl, [self.ctrlTabl[-1,:]], axis=0)
		elif self.ctrlTabl.shape[0] > self.rows:
			for _ in range(self.ctrlTabl.shape[0]-self.rows):
				self.ctrlTabl = np.delete(self.ctrlTabl, -1, axis=0)
		if self.realTabl.shape[0] < self.rows:
			for _ in range(self.rows-self.realTabl.shape[0]):
				self.realTabl = np.append(self.realTabl, [self.realTabl[-1,:]], axis=0)
		elif self.realTabl.shape[0] > self.rows:
			for _ in range(self.realTabl.shape[0]-self.rows):
				self.realTabl = np.delete(self.realTabl, -1, axis=0)

		self._simTime = self.realTabl[:,0]
		self._computeApproxLinAccel()
		self._computeApproxAngAccel()

	def ctrlLinVelTS(self):
		return self.ctrlTabl[:,1]
	def ctrlAngVelTS(self):
		return self.ctrlTabl[:,2]
	def realLinAccelTS(self):
		return self.realTabl[:,7]
	def realAngAccelTS(self):
		return self.realTabl[:,8]
	def realLinVelTS(self):
		return self.realTabl[:,5]
	def realAngVelTS(self):
		return self.realTabl[:,6]
	def realXTS(self):
		return self.realTabl[:,2]
	def realYTS(self):
		return self.realTabl[:,3]
	def realThetaTS(self):
		return self.realTabl[:,4]
	def xErrTS(self):
		return self.realTabl[:,2] - self.plTabl[:,1]
	def yErrTS(self):
		return self.realTabl[:,3] - self.plTabl[:,2]
	def posErrTS(self):
		return np.array([np.sqrt(ex**2+ey**2) for ex, ey in zip(self.xErrTS(), self.yErrTS())])
	def thetaErrTS(self):
		return self.realTabl[:,4] - self.plTabl[:,3]
	def realRadTS(self):
		return self.realTabl[:,1]

	def _computeApproxLinAccel(self):
		linVel = self.realLinVelTS()
		self._approxLinAccelTS = [(linVel[1] - linVel[0])/(self._simTime[1] - self._simTime[0])] +\
			[(post - ant)/(tp - ta) for post, ant, tp, ta in zip(linVel[2:], linVel[0:-2], self._simTime[2:], self._simTime[0:-2])] +\
			[(linVel[-1] - linVel[-2])/(self._simTime[-1] - self._simTime[-2])]
		return self._approxLinAccelTS

	def _computeApproxAngAccel(self):
		angVel = self.realAngVelTS()
		self._approxAngAccelTS = [(angVel[1] - angVel[0])/(self._simTime[1] - self._simTime[0])] +\
			[(post - ant)/(tp - ta) for post, ant, tp, ta in zip(angVel[2:], angVel[0:-2], self._simTime[2:], self._simTime[0:-2])] +\
			[(angVel[-1] - angVel[-2])/(self._simTime[-1] - self._simTime[-2])]

		return self._approxAngAccelTS


	def _cost(self, x):

		# ukm1 = self._u0
		# wkm1 = self._w0
		u = self._u0
		w = self._w0
		x_ = self._x0
		y = self._y0
		theta = self._theta0

		dt = (self._simTime[1]-self._simTime[0])
		
		aTS = list()
		alfaTS = list()
		uTS = list()
		wTS = list()
		xTS = list()
		yTS = list()
		thetaTS = list()
		for uc, wc in zip(self.ctrlLinVelTS()[30:], self.ctrlAngVelTS()[30:]):

			uTS += [u]
			wTS += [w]
			xTS += [x_]
			yTS += [y]
			thetaTS += [theta]

			a = x[2]/x[0]*w**2 - x[3]/x[0]*u + 1./x[0]*uc
			alfa = -x[4]/x[1]*u*w - x[5]/x[1]*w + 1./x[1]*wc

			u = u + a*dt
			w = w + alfa*dt

			theta = w*dt + theta
			x_ = x_ + u*np.cos(theta)*dt
			y = y + u*np.sin(theta)*dt

			aTS += [a]
			alfaTS += [alfa]

		# return LA.norm(np.append(np.array(aTS)-self._approxLinAccelTS[30:], np.array(alfaTS)-self._approxAngAccelTS[30:]))
		# return LA.norm(np.append(np.array(uTS)-self.realLinVelTS()[30:], np.array(wTS)-self.realAngVelTS()[30:]))
		return LA.norm(np.append(np.append(np.array(xTS)-self.realXTS()[30:], np.array(yTS)-self.realYTS()[30:]), np.array(thetaTS)-self.realThetaTS()[30:]))
		
		# ucTS = self.ctrlLinVelTS()[30:]
		# wcTS = self.ctrlAngVelTS()[30:]
		# urTS = self.realLinVelTS()[30:]
		# wrTS = self.realAngVelTS()[30:]

		# #for uc, wc in zip(self.ctrlLinVelTS(), self.ctrlAngVelTS()):
		# aTS = x[2]*x[0]*np.square(wrTS) - x[3]*x[0]*urTS + x[0]*ucTS
		# alfaTS = -x[4]*x[1]*urTS*wrTS - x[5]*x[1]*wrTS + x[1]*wcTS

		# return np.square(LA.norm(np.append(np.array(aTS)-self._approxLinAccelTS[30:], np.array(alfaTS)-self._approxAngAccelTS[30:])))
		#return np.square(LA.norm(np.array(alfaTS)-self._approxAngAccelTS[30:]))

		# ukm1 = 0.0
		# wkm1 = 0.0
		# u = self._u0
		# w = self._w0
		# dt = (self._simTime[1]-self._simTime[0])
		# aTS = list()
		# alfaTS = list()
		# for uc, wc in zip(self.ctrlLinVelTS()[30:], self.ctrlAngVelTS()[30:]):
		# 	a = x[2]/x[0]*w**2 - x[3]/x[0]*u + 1./x[0]*uc
		# 	alfa = -x[4]/x[1]*u*w - x[5]/x[1]*w + 1./x[1]*wc

		# 	# aux_u = u
		# 	# aux_w = w
		# 	# u = ukm1 + a*2*dt
		# 	# w = wkm1 + alfa*2*dt
		# 	# ukm1 = aux_u
		# 	# wkm1 = aux_w

		# 	u = u + a*dt
		# 	w = w + alfa*dt


		# 	aTS += [a]
		# 	alfaTS += [alfa]

		# return LA.norm(np.append(np.array(aTS)-self._approxLinAccelTS[30:], np.array(alfaTS)-self._approxAngAccelTS[30:]))
		
		# ucTS = self.ctrlLinVelTS()[30:]
		# wcTS = self.ctrlAngVelTS()[30:]
		# urTS = self.realLinVelTS()[30:]
		# wrTS = self.realAngVelTS()[30:]

		# #for uc, wc in zip(self.ctrlLinVelTS(), self.ctrlAngVelTS()):
		# aTS = x[2]*x[0]*np.square(wrTS) - x[3]*x[0]*urTS + x[0]*ucTS
		# alfaTS = -x[4]*x[1]*urTS*wrTS - x[5]*x[1]*wrTS + x[1]*wcTS

		# return np.square(LA.norm(np.append(np.array(aTS)-self._approxLinAccelTS[30:], np.array(alfaTS)-self._approxAngAccelTS[30:])))
		#return np.square(LA.norm(np.array(alfaTS)-self._approxAngAccelTS[30:]))



	def dySysAccel(self):
		x = self._dyModelParam
		u = self._u0
		w = self._w0
		x_ = self._x0
		y = self._y0
		theta = self._theta0
		dt = (self._simTime[1]-self._simTime[0])
		aTS = list()
		alfaTS = list()
		uTS = []
		wTS = []
		xTS = []
		yTS = []
		thetaTS = []
		for uc, wc in zip(self.ctrlLinVelTS(), self.ctrlAngVelTS()):
			uTS += [u]
			wTS += [w]
			xTS += [x_]
			yTS += [y]
			thetaTS += [theta]

			a = x[2]/x[0]*w**2 - x[3]/x[0]*u + 1./x[0]*uc
			alfa = -x[4]/x[1]*u*w - x[5]/x[1]*w + 1./x[1]*wc

			u = u + a*dt
			w = w + alfa*dt

			theta = w*dt + theta
			x_ = x_ + u*np.cos(theta)*dt
			y = y + u*np.sin(theta)*dt

			aTS += [a]
			alfaTS += [alfa]

		return (aTS, alfaTS, uTS, wTS, xTS, yTS, thetaTS)

		# ukm1 = 0.0
		# wkm1 = 0.0
		# u = 0.0
		# w = 0.0
		# dt = (self._simTime[1]-self._simTime[0])
		# aTS = list()
		# alfaTS = list()
		# for uc, wc in zip(self.ctrlLinVelTS(), self.ctrlAngVelTS()):
		# 	a = x[2]/x[0]*w**2 - x[3]/x[0]*u + 1./x[0]*uc
		# 	alfa = -x[4]/x[1]*u*w - x[5]/x[1]*w + 1./x[1]*wc

		# 	aux_u = u
		# 	aux_w = w
		# 	u = ukm1 + a*2*dt
		# 	w = wkm1 + alfa*2*dt
		# 	ukm1 = aux_u
		# 	wkm1 = aux_w


		# 	aTS += [a]
		# 	alfaTS += [alfa]

		# return (aTS, alfaTS)

		# ucTS = self.ctrlLinVelTS()
		# wcTS = self.ctrlAngVelTS()
		# urTS = self.realLinVelTS()
		# wrTS = self.realAngVelTS()

		# #for uc, wc in zip(self.ctrlLinVelTS(), self.ctrlAngVelTS()):
		# aTS = x[2]*x[0]*np.square(wrTS) - x[3]*x[0]*urTS + x[0]*ucTS
		# alfaTS = -x[4]*x[1]*urTS*wrTS - x[5]*x[1]*wrTS + x[1]*wcTS

		# return (aTS, alfaTS)



direc = "C:/Users/JM246044/workspace/dev/xde/xde/xde/xde/"

ridx = 0

root = ET.parse(direc+'src/aiv/output.xml').getroot()
nbOfRobots = len(root)
root = ET.parse(direc+'src/aiv/config.xml').getroot()

robot = Robot(ridx, direc,
	float(root.find('aivs')[ridx].find('initvelo').find('linear').text),
	float(root.find('aivs')[ridx].find('initvelo').find('angular').text),
	float(root.find('aivs')[ridx].find('initpose').find('x').text),
	float(root.find('aivs')[ridx].find('initpose').find('y').text),
	float(root.find('aivs')[ridx].find('initpose').find('theta').text))

print "\nINITIAL DYNAMIC MODEL PARAMETERS\n", robot._dyModelParam

robot.optimize()
print "\nOPTIMIZED DYNAMIC MODEL PARAMETERS\n", robot._dyModelParam
print "\nnb itereations", robot._n_it
print robot._exit_mode


a, alfa, v, w, x, y, theta = robot.dySysAccel()

print "\nRATIOS\n", robot._dyModelParam[2]/robot._dyModelParam[0], robot._dyModelParam[3]/robot._dyModelParam[0], robot._dyModelParam[4]/robot._dyModelParam[1], robot._dyModelParam[5]/robot._dyModelParam[1], 1/robot._dyModelParam[0], 1./robot._dyModelParam[1]

figAccel, axArrayAccel = plt.subplots(2, sharex=True)

axArrayAccel[0].grid()
axArrayAccel[1].grid()

#axArrayAccel[0].plot(robot._simTime, robot.planLinAccelTS(), 'b', label=r'$u_{ref}[0]$ (planner linaccel)')
axArrayAccel[0].plot(robot._simTime[30:], robot._approxLinAccelTS[30:], 'r', label=r'actual linaccel')
axArrayAccel[0].plot(robot._simTime[30:], a[30:], 'm', label=r'sysid linaccel')

#axArrayAccel[1].plot(robot._simTime, robot.planAngAccelTS(), 'b', label=r'planner angaccel')
axArrayAccel[1].plot(robot._simTime[30:], robot._approxAngAccelTS[30:], 'r', label=r'actual angaccel')
axArrayAccel[1].plot(robot._simTime[30:], alfa[30:], 'm', label=r'sysid angaccel')

hand, lab = axArrayAccel[1].get_legend_handles_labels()
axArrayAccel[1].legend(hand, lab, ncol=1, prop={'size':10}, loc=1)

figAccel.savefig(direc+'build/out/Release/bin/images/IDaalpha'+ str(robot.id) +'.pdf', bbox_inches='tight', dpi=300)
figAccel.savefig(direc+'build/out/Release/bin/images/IDaalpha'+ str(robot.id) +'.png', bbox_inches='tight', dpi=300)

figVel, axArrayVel = plt.subplots(2, sharex=True)

axArrayVel[0].grid()
axArrayVel[1].grid()

#axArrayVel[0].plot(robot._simTime, robot.planLinVelTS(), 'b', label=r'$u_{ref}[0]$ (planner linaccel)')
axArrayVel[0].plot(robot._simTime[30:], robot.realLinVelTS()[30:], 'r', label=r'actual linaccel')
axArrayVel[0].plot(robot._simTime[30:], v[30:], 'm', label=r'sysid linaccel')

#axArrayVel[1].plot(robot._simTime, robot.planAngVelTS(), 'b', label=r'planner angaccel')
axArrayVel[1].plot(robot._simTime[30:], robot.realAngVelTS()[30:], 'r', label=r'actual angel')
axArrayVel[1].plot(robot._simTime[30:], w[30:], 'm', label=r'sysid angvel')

hand, lab = axArrayVel[1].get_legend_handles_labels()
axArrayVel[1].legend(hand, lab, ncol=1, prop={'size':10}, loc=1)

figVel.savefig(direc+'build/out/Release/bin/images/IDww'+ str(robot.id) +'.pdf', bbox_inches='tight', dpi=300)
figVel.savefig(direc+'build/out/Release/bin/images/IDvw'+ str(robot.id) +'.png', bbox_inches='tight', dpi=300)


figPos, axArrayPos = plt.subplots(2, sharex=True)

axArrayPos[0].grid()
axArrayPos[1].grid()

#axArrayPos[0].plot(robot._simTime, robot.planLinPosTS(), 'b', label=r'$u_{ref}[0]$ (planner linaccel)')
axArrayPos[0].plot(robot._simTime[30:], np.sqrt(np.square(robot.realXTS()[30:]) + np.square(robot.realYTS()[30:]) ), 'r', label=r'actual linpos')
axArrayPos[0].plot(robot._simTime[30:], np.sqrt(np.square(x[30:]) + np.square(y[30:]) ), 'm', label=r'sysid linpos')

#axArrayPos[1].plot(robot._simTime, robot.planAngPosTS(), 'b', label=r'planner angaccel')
axArrayPos[1].plot(robot._simTime[30:], robot.realThetaTS()[30:], 'r', label=r'actual angpos')
axArrayPos[1].plot(robot._simTime[30:], theta[30:], 'm', label=r'sysid angpos')

hand, lab = axArrayPos[1].get_legend_handles_labels()
axArrayPos[1].legend(hand, lab, ncol=1, prop={'size':10}, loc=1)

figPos.savefig(direc+'build/out/Release/bin/images/IDrtheta'+ str(robot.id) +'.pdf', bbox_inches='tight', dpi=300)
figPos.savefig(direc+'build/out/Release/bin/images/IDrtheta'+ str(robot.id) +'.png', bbox_inches='tight', dpi=300)


figErrPos, axArrayErrPos = plt.subplots(2, sharex=True)

axArrayErrPos[0].grid()
axArrayErrPos[1].grid()

#axArrayErrPos[0].plot(robot._simTime, robot.planLinErrPosTS(), 'b', label=r'$u_{ref}[0]$ (planner linaccel)')
axArrayErrPos[0].plot(robot._simTime[30:], np.sqrt(np.square(robot.realXTS()[30:]) + np.square(robot.realYTS()[30:]) ) - np.sqrt(np.square(x[30:]) + np.square(y[30:]) ), 'r', label=r'err linpos')

#axArrayErrPos[1].plot(robot._simTime, robot.planAngErrPosTS(), 'b', label=r'planner angaccel')
axArrayErrPos[1].plot(robot._simTime[30:], robot.realThetaTS()[30:] - theta[30:], 'r', label=r'err angpos')

hand, lab = axArrayErrPos[1].get_legend_handles_labels()
axArrayErrPos[1].legend(hand, lab, ncol=1, prop={'size':10}, loc=1)

figErrPos.savefig(direc+'build/out/Release/bin/images/IDerrrtheta'+ str(robot.id) +'.pdf', bbox_inches='tight', dpi=300)
figErrPos.savefig(direc+'build/out/Release/bin/images/IDerrrtheta'+ str(robot.id) +'.png', bbox_inches='tight', dpi=300)