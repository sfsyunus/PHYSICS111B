import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats
from matplotlib import rcParams
from scipy.optimize import curve_fit
import os

rcParams['axes.labelsize'] = 18
rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['legend.fontsize'] = 15
rcParams['figure.titlesize'] = 20
rcParams['savefig.dpi'] = 600

#load raw data
data = np.genfromtxt('RUT_var.dat', dtype='float', skip_header=5, skip_footer=5)
data = np.transpose(data)
thetaDeg = data[0]
thetaRad = np.deg2rad(thetaDeg)
N = data[1]
I = data[2]
Var = data[3]
t = data[4]

def getSum(day):
	#needs a string input
	path = 'data/%s/' %day
	files = os.listdir(path)
	sumArr = np.array([])
	for i in range(len(files)):
		if float(day[3]) == 1.:
			data = np.genfromtxt(path+files[i], dtype='float', skip_header=242, skip_footer=997).T
		else:
			data = np.genfromtxt(path+files[i], dtype='float', skip_header=127, skip_footer=511).T
		V = data[0]
		N = data[1]
		sumN = np.sum(N)
		sumArr = np.append(sumArr, sumN)
	Sum = np.sum(sumArr)
	return Sum, sumArr, np.mean(sumArr)



def getSpec(day):
	#needs a string input
	path = 'data/%s/' %day
	files = os.listdir(path)
	if float(day[3]) == 1.:
	   data = np.genfromtxt(path+files[0], dtype='float', skip_header=242, skip_footer=997).T
	else:
	   data = np.genfromtxt(path+files[0], dtype='float', skip_header=127, skip_footer=511).T
        x = data[0]

	NArr = np.zeros((data[1].size))
	VArr = x

	
	for i in range(len(files)):
	   if float(day[3]) == 1.:
	       data = np.genfromtxt(path+files[i], dtype='float', skip_header=242, skip_footer=997).T
	   else:
	   		data = np.genfromtxt(path+files[i], dtype='float', skip_header=127, skip_footer=511).T
	   N = data[1]
	   NArr += N
	plt.plot(VArr, NArr, label = r'$\theta=18.5\degree$ with foil')
	plt.xlabel('Energy [V]')
	plt.ylabel('Counts')
	plt.legend(loc='best')



x = np.sin(abs(thetaRad/2))
y = I
x2 = np.sin(np.deg2rad(np.arange(2,26)/2))
y2 = x2**(-4)
x1 = np.log(np.sin(abs(thetaRad/2))**(-4))
y1 = np.log(I)

	

SA = 0.0977
I_0 = 2252.566667
Z1 = 2
E = 6.04*10**(-13)
q = 1.60*10**(-19)
e_0 = 8.85*10**(-12)
N_0 = 1.61E+23
K = (I_0*N_0*SA)*((Z1*q**2.)/(4.*np.pi*e_0*4.*E))**2
#K =1.0104801043009808e-06

def getChisq(y, fit):
	err = np.sqrt(np.mean((y-fit)**2))
	#chisq = scipy.stats.chisquare(y,fit)
	chisq = np.sum(((y-fit)**2)/N)/np.size(y-2)
	return chisq

def x4model(x, a4, a3, a2, a1, a0):
	return a4*x**(-4) +a3*x**(-3) + a2*x**(-2) + a1*x**(-1) + a0

def fit4():
	err = np.sqrt(N)/t
	par, cov = curve_fit(x4model, x, y, sigma = err)
	curve = x4model(x,par[0],par[1], par[2], par[3],par[4])
	plt.plot(x,y,'.', label = 'Data')
	plt.plot(x,curve, label= 'Fit')
	plt.xlabel(r'$sin(\theta/2)$')
	plt.ylabel(r'$I$ $counts/min^{-1}$')
	# plt.title('')
	plt.errorbar(x, y, color='r', yerr = err, fmt=',', label=r'$\sigma_N/t$')
	plt.legend(loc = 'best', numpoints=1)
	chisq = getChisq(y,curve)
	return par, cov, curve, chisq


def linmodel(x,y):
	err = 1/np.sqrt(N)
	coeffs,cov = np.polyfit(x,y,1, cov=True)
	a1 = coeffs[0]
	a0 = coeffs[1]
	y2 = a1*x + a0
	plt.plot(x,y,'.', label = 'Data')
	plt.plot(x,y2, label = 'Best fit')
	plt.xlabel(r'$log(sin(\theta/2)^{-4}$)')
	plt.ylabel(r'$log(I)$')
	#plt.title('')
	plt.errorbar(x, y, color='r', yerr = err, fmt=',', label=r'$\sigma_N/N$')
	plt.legend(loc = 'best', numpoints = 1)
	chisq = getChisq(y,y2)
	return coeffs, cov, y2, chisq

def plotTheory():
	thetaTh = np.deg2rad(np.arange(4,26))
	x = np.log((np.sin(abs(thetaRad/2)))**(-4))
	y = np.log(K*44**2) + x
	plt.plot(x,y, label='Theoretical curve')
	plt.legend(loc = 'best', numpoints = 1)
	coeffs = np.polyfit(x,y,1)
	return coeffs, y

def plotTheory4():
	y = (69.)**2*K*x**(-4)
	plt.plot(x,y,label='Theoretical curve')
	plt.legend(loc = 'best', numpoints = 1)
	coeffs = np.polyfit(x,y,1)
	return coeffs, y


def getZ2(c):
	Z = np.sqrt(np.exp(c)/K)
	err = np.sqrt((np.exp(c)*0.56897492**2/(4*K))+(np.exp(c)*1.51363E-07**2/(4*K**2)))
	return K, Z, err


def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def fitGauss():
	# p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
	p0 = [374, 0, 1]
	err = np.sqrt(N)/t
	coeff, cov = curve_fit(gauss, thetaDeg, I, sigma = err,p0=p0)
	print(coeff)
	# Get the fitted curve
	theta = np.arange(-26,26)
	gauss_line = gauss(theta, *coeff)
	plt.plot(theta, gauss_line, 'r',label='Gaussian Fit')
	plt.plot(thetaDeg, I,'go', label = 'Raw Data')
	plt.show()
	plt.legend(loc = 'best', numpoints=1)