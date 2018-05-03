import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats
from matplotlib import rcParams
from scipy.optimize import curve_fit

# Make figs pretty
rcParams['axes.labelsize'] = 18
rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['legend.fontsize'] = 15
rcParams['axes.titlesize'] = 20
rcParams['savefig.dpi'] = 600

# Data paths
path1off = "Data/res_1_off.txt"
path1on = "Data/res_1_on.txt"
path2off = "Data/res_2_off.txt"
path2on = "Data/res_2_on.txt"

#generate data
#path1:85, path2:87
data85off = np.genfromtxt(path1off, dtype='float', skip_header=1)
data85off = np.transpose(data85off)
data85on = np.genfromtxt(path1on, dtype='float', skip_header=1, skip_footer=1)
data85on = np.transpose(data85on)

data87off = np.genfromtxt(path2off, dtype='float', skip_header=1)
data87off = np.transpose(data87off)
data87on = np.genfromtxt(path2on, dtype='float', skip_header=1, skip_footer=1)
data87on = np.transpose(data87on)

I85off = data85off[0]
f85off = data85off[1]
I85on = data85on[0]
f85on = data85on[1]#*-1

I87off = data87off[0]
f87off = data87off[1]
I87on = data87on[0]
f87on = data87on[1]#*-1

I85 = np.append(I85off, I85on)
I87 = np.append(I87off, I87on)
f85 = np.append(f85off, f85on)
f87 = np.append(f87off, f87on)

#get best fits and errors

def bestFits(i,f):
	coeffs, cov = np.polyfit(i,f,1, cov=True)
	m = coeffs[0]
	c = coeffs[1]
	N = f.size
	y = m*i + c
	res = y - f #(fit - measured)
	var = np.var(res)
	merr = np.sqrt(N*var/(N*np.sum(i**2) - np.sum(i)**2 ))
	cerr = np.sqrt(var*np.sum(i**2)/(N*np.sum(i**2) - np.sum(i)**2))
	#merr = np.sqrt(cov[0,0])
	#cerr = np.sqrt(cov[1,1])
	Rsq = 1 - (np.sum(res**2))/(np.sum((y-np.mean(f))**2))

	return m, c, merr, cerr, Rsq

m85off, c85off, merr85off, cerr85off, Rsq85off = bestFits(I85off, f85off)
m85on, c85on, merr85on, cerr85on, Rsq85on = bestFits(I85on, f85on)
m87off, c87off, merr87off, cerr87off, Rsq87off = bestFits(I87off, f87off)
m87on, c87on, merr87on, cerr87on, Rsq87on = bestFits(I87on, f87on)
m85, c85, merr85, cerr85, Rsq85 = bestFits(I85, f85)
m87, c87, merr87, cerr87, Rsq87 = bestFits(I87, f87)

#make best fit line equations
y85off = m85off*I85off+c85off
y85on = m85on*I85on+c85on
y87off = m87off*I87off+c87off
y87on = m87on*I87on+c87on
y85 = m85*I85+c85
y87 = m87*I87+c87


'''
def makeSinglePlots(path):
	data = np.genfromtxt(path, dtype='float', skip_header=1)
	data = np.transpose(data)
	current = data[0]
	freq = data[1]
	plt.plot(current, freq, 'ro')
	plt.xlabel("Current [Amperes]")
	plt.ylabel("Frequency [MHz]")
	plt.xlim(-0.05,1)
	plt.ylim(-0.05, 3.05)
	plt.title("$Rb^{87}$ Coil Polarity turned OFF")
	return current, freq
'''

def makeAlignedPlots():
	
	plt.plot(I85off, f85off,  'go', label = '$Rb^{85}$')
	plt.plot(I87off, f87off, 'r^', label = '$Rb^{87}$')
	plt.plot(I85off, y85off, 'g', label = '$Rb^{85}$ fit')
	plt.plot(I87off, y87off, 'r', label = '$Rb^{87}$ fit')
	plt.xlabel("Current [Amperes]")
	plt.ylabel("Frequency [MHz]")
	plt.title("Frequency vs. Current, $B_{coil}$ Aligned with $B_{earth}$")
	plt.legend(loc='best')
	errbar = 8E-4 #errors in frequency are around 800Hz
	plt.errorbar(I87off, f87off, color='r', yerr = errbar, fmt='.')
	plt.errorbar(I85off, f85off, color='g', yerr = errbar, fmt='.')

def makeUnalignedPlots():
	
	plt.plot(I85on, f85on,  'go', label = '$Rb^{85}$')
	plt.plot(I87on, f87on, 'r^', label = '$Rb^{87}$')
	plt.plot(I85on, y85on, 'g', label = '$Rb^{85}$ fit')
	plt.plot(I87on, y87on, 'r', label = '$Rb^{87}$ fit')
	plt.xlabel("Current [Amperes]")
	plt.ylabel("Frequency [MHz]")
	plt.title("Frequency vs. Current, $B_{coil}$ Unaligned with $B_{earth}$")
	plt.legend(loc='best')
	errbar = 8E-4 #errors in frequency are around 800Hz
	plt.errorbar(I87on, f87on, color='r', yerr = errbar, fmt='.')
	plt.errorbar(I85on, f85on, color='g', yerr = errbar, fmt='.')

def allPlots():
	
	plt.plot(I85off, f85off,  'go', label = '$Rb^{85}$')
	plt.plot(I87off, f87off, 'r^', label = '$Rb^{87}$')
	plt.plot(I85on, f85on,  'go')
	plt.plot(I87on, f87on, 'r^')
	#plt.plot(I85off, y85off, 'g', label = '$Rb^{85}$ fit')
	#plt.plot(I87off, y87off, 'r', label = '$Rb^{87}$ fit')
	#plt.plot(I85on, y85on, 'g')
	#plt.plot(I87on, y87on, 'r')

	plt.xlabel("Current [Amperes]")
	plt.ylabel("Frequency [MHz]")
	plt.title("Frequency vs. Current, all data")
	plt.legend(loc='best')
	errbar = 8E-4 #errors in frequency are around 800Hz
	plt.errorbar(I87off, f87off, color='r', yerr = errbar, fmt='.')
	plt.errorbar(I85off, f85off, color='g', yerr = errbar, fmt='.')
	plt.errorbar(I87on, f87on, color='r', yerr = errbar, fmt='.')
	plt.errorbar(I85on, f85on, color='g', yerr = errbar, fmt='.')

def makeUniPlot():
	
	plt.plot(I85, f85,  'go', label = '$Rb^{85}$')
	plt.plot(I87, f87, 'r^', label = '$Rb^{87}$')
	plt.plot(I85, y85, 'g', label = '$Rb^{85}$ fit')
	plt.plot(I87, y87, 'r', label = '$Rb^{87}$ fit')
	
	plt.xlabel("Current [Amperes]")
	plt.ylabel("Frequency [MHz]")
	plt.title("Frequency vs. Current, all data")
	plt.legend(loc='best')
	errbar = 8E-4 #errors in frequency are around 800Hz
	plt.errorbar(I87, f87, color='r', yerr = errbar, fmt='.')
	plt.errorbar(I85, f85, color='g', yerr = errbar, fmt='.')
	
def solveI(v1,v2,v3,v4):
	I1 = (v3/v4 - v1/v2)/(2*v1/v2 - 2*v3/v4)
	I2 = ( (v1*I1)/v2 - 1)/2
	return I1, I2

def BreitRabi(f, I):
	B = (2*I+1)*f/2.799
	return B
