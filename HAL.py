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

# Data paths
pathIvV = "Data/IvV/"
pathIvB = "Data/IvB/"
pathEvB = "Data/EvB/"

def bestFit(x,y):
	coeffs, cov = np.polyfit(x,y,1, cov=True)
	m = coeffs[0]
	c = coeffs[1]
	N = y.size
	fit = m*x + c
	res = fit - y #(fit - measured)
	var = np.var(res)
	#merr = np.sqrt(N*var/(N*np.sum(x**2) - np.sum(X)**2 ))
	#cerr = np.sqrt(var*np.sum(x**2)/(N*np.sum(x**2) - np.sum(x)**2))
	#chisq = scipy.stats.chisquare(y,fit)
	#merr = np.sqrt(cov[0,0])
	#cerr = np.sqrt(cov[1,1])
	return m, c

def bestFitErr(x,y):
	coeffs, cov = np.polyfit(x,y,1, cov=True)
	m = coeffs[0]
	c = coeffs[1]
	N = y.size
	fit = m*x + c
	res = fit - y #(fit - measured)
	var = np.var(res)
	merr = np.sqrt(cov[0,0])
	cerr = np.sqrt(cov[1,1])
	#merr = np.sqrt(N*var/(N*np.sum(x**2) - np.sum(x)**2 ))
	#cerr = np.sqrt(var*np.sum(x**2)/(N*np.sum(x**2) - np.sum(x)**2))

	#### From Measurements and Uncertainties pg 58, 
	se = np.sqrt(np.sum(res**2)/(N-2))	# standard err
	Delta = N*np.sum(x**2)-(np.sum(x))**2
	#merr = se*np.sqrt(np.sum(x**2)/Delta)
	#cerr = se*np.sqrt(N/Delta)
	chisq = scipy.stats.chisquare(y,fit)
	#chisq = np.sum(((y-fit)**2)/N)/np.size(y-2)
	Rsq = 1 - np.sum(res**2)/np.sum((y-(np.sum(y)/N))**2)

	return m,c,merr, cerr,chisq, Rsq

## get data for discharge current vs discharge voltage measurements

def getIvV(path, n):
	files = os.listdir(path)
	
	for j in range(len(files)):
		if j == n:
			data = np.genfromtxt(path+files[j], dtype='float', skip_header=1).T
			hiV = data[0]
			disV = data[3]
			disI = data[2]

	return hiV, disV, disI


hiV_15, disV_15, disI_15 = getIvV(pathIvV, 0)
hiV_18, disV_18, disI_18 = getIvV(pathIvV, 1)
hiV_21, disV_21, disI_21 = getIvV(pathIvV, 2)
hiV_24, disV_24, disI_24 = getIvV(pathIvV, 3)
hiV_27, disV_27, disI_27 = getIvV(pathIvV, 4)
hiV_30, disV_30, disI_30 = getIvV(pathIvV, 5)

def plotIvV():
	plt.errorbar(disI_15, disV_15, marker = '.', color='#F13106', yerr = .5, fmt=',', label = '15 Torr')
	plt.errorbar(disI_18, disV_18, marker = '.', color='#06E3F1', yerr = .5, fmt=',', label = '18 Torr')
	plt.errorbar(disI_21, disV_21, marker = '.', color='#FCF401', yerr = .5, fmt=',', label = '21 Torr')
	plt.errorbar(disI_24, disV_24, marker = '.', color='#3BDA15', yerr = .5, fmt=',', label = '24 Torr')
	plt.errorbar(disI_27, disV_27, marker = '.', color='#0F3FE3', yerr = .5, fmt=',', label = '27 Torr')
	plt.errorbar(disI_30, disV_30, marker = '.', color='#B40AB4', yerr = .5, fmt=',', label = '30 Torr')
	plt.xlabel('Discharge Current [mA]')
	plt.ylabel('Discharge Voltage [V]')
	plt.title('Discharge Current vs Discharge Voltage for Various Pressures')
	plt.ylim(45, 90)
	plt.legend(loc='best', numpoints=1, ncol=2)
	plt.show()

def plotIvHiV():
	plt.errorbar(disI_15, hiV_15, marker = '.', color='#F13106', yerr = 25, fmt=',', label = '15 Torr')
	plt.errorbar(disI_18, hiV_18, marker = '.', color='#06E3F1', yerr = 25, fmt=',', label = '18 Torr')
	plt.errorbar(disI_21, hiV_21, marker = '.', color='#FCF401', yerr = 25, fmt=',', label = '21 Torr')
	plt.errorbar(disI_24, hiV_24, marker = '.', color='#3BDA15', yerr = 25, fmt=',', label = '24 Torr')
	plt.errorbar(disI_27, hiV_27, marker = '.', color='#0F3FE3', yerr = 25, fmt=',', label = '27 Torr')
	plt.errorbar(disI_30, hiV_30, marker = '.', color='#B40AB4', yerr = 25, fmt=',', label = '30 Torr')
	plt.xlabel('Discharge Current [mA]')
	plt.ylabel('High Voltage [V]')
	plt.title('Discharge Current vs High Voltage for Various Pressures')
	plt.legend(loc='best', numpoints=1, ncol=2)
	plt.show()

#m15, c15 = bestFit(disI_15, hiV_15)
#m18, c18 = bestFit(disI_18, hiV_18)
#m21, c21 = bestFit(disI_21, hiV_21)
#m24, c24 = bestFit(disI_24, hiV_24)
#m27, c27 = bestFit(disI_27, hiV_27)
#m30, c30 = bestFit(disI_30, hiV_30)

## get data for magnet current vs magnet field measurements

def getIvB(path, n):
	files = os.listdir(path)
	
	for j in range(len(files)):
		if j == n:
			data = np.genfromtxt(path+files[j], dtype='float', skip_header=1).T
			magI = data[0]
			magB = data[1]

	return magI, magB


magI_FI, magB_FI = getIvB(pathIvB, 0)
magI_FD, magB_FD = getIvB(pathIvB, 1)
magI_RI, magB_RI = getIvB(pathIvB, 2)
magI_RD, magB_RD = getIvB(pathIvB, 3)

mFI, cFI = bestFit(magI_FI, magB_FI)
mFD, cFD = bestFit(magI_FD, magB_FD)
mRI, cRI = bestFit(magI_RI, magB_RI)
mRD, cRD = bestFit(magI_RD, magB_RD)

def plotIvB():
	plt.errorbar(magI_FI, magB_FI, color='red', marker = 'o', linestyle = 'none', yerr = 0.3, fmt=',', label = 'Forward Polarity')
	plt.errorbar(magI_FD, magB_FD, color='red', marker = 'o', linestyle = 'none', yerr = 0.3, fmt=',')
	plt.errorbar(magI_RI, magB_RI, color='blue', marker = 'o', linestyle = 'none', yerr = 0.3, fmt=',', label = 'Reverse Polarity')
	plt.errorbar(magI_RD, magB_RD, color='blue', marker = 'o', linestyle = 'none', yerr = 0.3, fmt=',')
	
	plt.plot(magI_FI, magB_FI, color = 'red', linestyle = '--', label = 'Increasing Current')
	plt.plot(magI_FD, magB_FD, color = 'red', label = 'Decreasing Current')
	plt.plot(magI_RI, magB_RI, color = 'blue', linestyle = '--')
	plt.plot(magI_RD, magB_RD, color = 'blue')
	plt.xlabel('Magnet Current [A]')
	plt.ylabel('Magnetic Field Strength [Gauss]')
	plt.title('Magnet Current vs Field for Forward and Reverse Polarity')
	plt.legend(loc='best', numpoints=1)
	plt.show()
	

magI_I = np.append(-1*magI_RI[::-1], magI_FI)
magI_D = np.append(-1*magI_RD[::-1], magI_FD)
magB_I = np.append(magB_RI[::-1], magB_FI)
magB_D = np.append(magB_RD[::-1], magB_FD)

mI, cI = bestFit(magI_I, magB_I)
mD, cD = bestFit(magI_D, magB_D)

def plotIvB2():
	plt.errorbar(magI_I, magB_I, color='green',  marker = '.', linestyle = 'none', label = 'Increasing Current', yerr = 0.3, fmt=',')
	plt.errorbar(magI_D, magB_D, color='blue',  marker = '.', linestyle = 'none', label = 'Decreasing Current', yerr = 0.3, fmt=',')

	plt.plot(magI_I, magI_I*mI + cI, color = 'green', label = '$B = %iI + %i$' %(mI,cI))
	plt.plot(magI_D, magI_D*mD + cD, color = 'blue', label = '$B = %iI + %i$' %(mD,cD))

	plt.xlabel('Magnet Current [A]')
	plt.ylabel('Magnetic Field Strength [Gauss]')
	plt.title('Magnet Current vs Field for Forward and Reverse Polarity')
	plt.legend(loc='best', numpoints=1)
	plt.show()


def plotIvBlog():
	# plot log of B values
	magI_I = np.array([ 2.5, 2.25, 2., 1.75, 1.5, 1.25, 1., 0.75, 0.5, 0.25, 0., -0., -0.25, -0.5, -0.75, -1., -1.25, -1.5, -1.75, -2., -2.25, -2.5 ])
	magI_D = np.array([ 2.5, 2.25, 2., 1.75, 1.5, 1.25, 1., 0.75, 0.5, 0.25, 0., -0.25, -0.5, -0.75, -1., -1.25, -1.5, -1.75, -2., -2.25, -2.5 ])
	magB_I = np.array([ 6.58340922, 6.49223984, 6.3851944, 6.26149168, 6.1180972, 5.94803499, 5.74939299, 5.4823042, 5.11679549, 4.5685062, 3.32862669, -2.93385687, -4.48300255, -5.06638531, -5.44241771, -5.72031178, -5.9242558 , -6.09582456, -6.24222327, -6.37331979, -6.47850964, -6.57507584])
	magB_D = np.array([ 6.58479139, 6.47543272, 6.35784227, 6.21860012, 6.06145692, 5.87493073, 5.64685933, 5.34520095, 4.91338991, 4.22097721, 2.30258509, -4.05178495, -4.83945148, -5.29229929, -5.61130162, -5.83773045, -6.03068526, -6.19847872, -6.33859408, -6.45676966, -6.57646957])
	plt.plot(magI_I, magB_I, color = 'green', marker = '.', label = 'Increasing Current')
	plt.plot(magI_D, magB_D, color = 'blue', marker = '.', label = 'Decreasing Current')
	plt.xlabel('Magnet Current [A]')
	plt.ylabel('Log (Magnetic Field Strength) [Gauss]')
	plt.title('Hysteresis Effect')
	plt.legend(loc='best', numpoints=1)
	plt.show()


magI = magI_I
magB = (magB_I+magB_D)/2

def plotIvBavg():
	plt.errorbar(magI, magB, color='blue', marker = 'o', linestyle = 'none', label = 'Average field', yerr = 0.3, fmt=',')
	m, c = bestFit(magI, magB)
	plt.plot(magI, magI*m + c, color = 'green', label = '$B = 295I + 4.3$')
	plt.xlabel('Magnet Current [A]')
	plt.ylabel('Magnetic Field Strength [Gauss]')
	plt.title('Average Magnetic Field to find $B(I)$')
	plt.legend(loc='best', numpoints=1)
	plt.show()


def ItoB(I):
	B = 289.767012987012*I+4.55681818181816
	errStat = np.sqrt((I*1.028457705)**2+(0.695364324)**2)
	errSys = 0.3 #Gauss
	errB = np.sqrt(errSys**2 + errStat**2)
	return B, errB


## get data for hall voltage vs magnet current measurements

def getEvB(path, n):
	files = os.listdir(path)
	
	for j in range(len(files)):
		if j == n:
			data = np.genfromtxt(path+files[j], dtype='float', skip_header=1).T
			magI = data[0]
			VH = data[1]

	B, errB = ItoB(magI)
	EH = VH/0.004

	return B, EH, errB


B_F15, EH_F15, errB_F15 = getEvB(pathEvB, 0)
B_F18, EH_F18, errB_F18 = getEvB(pathEvB, 1)
B_F21, EH_F21, errB_F21 = getEvB(pathEvB, 2)
B_F24, EH_F24, errB_F24 = getEvB(pathEvB, 3)
B_F27, EH_F27, errB_F27 = getEvB(pathEvB, 4)
B_F30, EH_F30, errB_F30 = getEvB(pathEvB, 5)

B_R15, EH_R15, errB_R15 = getEvB(pathEvB, 6)
B_R18, EH_R18, errB_R18 = getEvB(pathEvB, 7)
B_R21, EH_R21, errB_R21 = getEvB(pathEvB, 8)
B_R24, EH_R24, errB_R24 = getEvB(pathEvB, 9)
B_R27, EH_R27, errB_R27 = getEvB(pathEvB, 10)
B_R30, EH_R30, errB_R30 = getEvB(pathEvB, 11)


#B_F = B_F15
#B_R = -1*B_R15
#B = np.append(B_R[::-1], B_F)
#errB = np.append(errB_F15[::-1], errB_F15)

B, errB = ItoB(magI)
errEH = 0.2/0.004

EH_15 = np.append(EH_R15[::-1], EH_F15)
EH_18 = np.append(EH_R18[::-1], EH_F18)
EH_21 = np.append(EH_R21[::-1], EH_F21)
EH_24 = np.append(EH_R24[::-1], EH_F24)
EH_27 = np.append(EH_R27[::-1], EH_F27)
EH_30 = np.append(EH_R30[::-1], EH_F30)

m15, c15 = bestFit(B[5:17], EH_15[5:17])
m18, c18 = bestFit(B[5:17], EH_18[5:17])
m21, c21 = bestFit(B[5:17], EH_21[5:17])
m24, c24 = bestFit(B[5:17], EH_24[5:17])
m27, c27 = bestFit(B[5:17], EH_27[5:17])
m30, c30 = bestFit(B[5:17], EH_30[5:17])

def plotHall():
	plt.errorbar(B, EH_15, color='#F13106', marker = '.', xerr=errB, yerr=errEH, fmt=',', label = '15 Torr')
	plt.errorbar(B, EH_18, color='#06E3F1', marker = '.', xerr=errB, yerr=errEH, fmt=',', label = '18 Torr')
	plt.errorbar(B, EH_21, color='#EADB20', marker = '.', xerr=errB, yerr=errEH, fmt=',', label = '21 Torr')
	plt.errorbar(B, EH_24, color='#3BDA15', marker = '.', xerr=errB, yerr=errEH, fmt=',', label = '24 Torr')
	plt.errorbar(B, EH_27, color='#0F3FE3', marker = '.', xerr=errB, yerr=errEH, fmt=',', label = '27 Torr')
	plt.errorbar(B, EH_30, color='#B40AB4', marker = '.', xerr=errB, yerr=errEH, fmt=',', label = '30 Torr')
	
	#plt.plot(B[5:17], m15*B[5:17] + c15, color = '#F13106', linestyle = '-.', label = '$E_H = B %i$' %(c15))
	#plt.plot(B[5:17], m18*B[5:17] + c18, color = '#06E3F1', linestyle = '-.', label = '$E_H = %iB + %i$' %(m18, c18))
	#plt.plot(B[5:17], m21*B[5:17] + c21, color = '#EADB20', linestyle = '-.', label = '$E_H = %iB %i$' %(m21, c21))
	#plt.plot(B[5:17], m24*B[5:17] + c24, color = '#3BDA15', linestyle = '-.', label = '$E_H = %iB %i$' %(m24, c24))
	#plt.plot(B[5:17], m27*B[5:17] + c27, color = '#0F3FE3', linestyle = '-.', label = '$E_H = B %i$' %(c27))
	#plt.plot(B[5:17], m30*B[5:17] + c30, color = '#B40AB4', linestyle = '-.', label = '$E_H = B + %i$' %(c30))

	plt.xlabel('Magnetic Field Strength [Gauss]')
	plt.ylabel('Hall Field [V/m]')
	plt.title('Hall Effect')
	plt.legend(loc='best', numpoints=1, ncol=2)
	plt.show()

def driftVel():
	#E/((10**-4)*B)
	v15,a1,v15err,b1,c1,d1 = bestFitErr(B[5:17], EH_15[5:17])
	v18,a2,v18err,b2,c2,d2 = bestFitErr(B[5:17], EH_18[5:17])
	v21,a3,v21err,b3,c3,d3 = bestFitErr(B[5:17], EH_21[5:17])
	v24,a4,v24err,b4,c4,d4 = bestFitErr(B[5:17], EH_24[5:17])
	v27,a5,v27err,b5,c5,d5 = bestFitErr(B[5:17], EH_27[5:17])
	v30,a6,v30err,b6,c6,d6 = bestFitErr(B[5:17], EH_30[5:17])
	vel = 10**4*np.array([v15,v18,v21,v24,v27,v30])
	velErr = 10**4*np.array([v15err,v18err,v21err,v24err,v27err,v30err])
	return vel, velErr


def getDriftVel():
	v15=EH_15[11:22]/(10**-4*B[11:22])
	v18=EH_18[11:22]/(10**-4*B[11:22])
	v21=EH_21[11:22]/(10**-4*B[11:22])
	v24=EH_24[11:22]/(10**-4*B[11:22])
	v27=EH_27[11:22]/(10**-4*B[11:22])
	v30=EH_30[11:22]/(10**-4*B[11:22])

	nu15=q*673.33/(m*v15)
	nu18=q*740.67/(m*v18)
	nu21=q*808.00/(m*v21)
	nu24=q*875.33/(m*v24)
	nu27=q*942.67/(m*v27)
	nu30=q*1010.0/(m*v30)

	n15=0.00095/(q*A*v15)
	n18=0.00150/(q*A*v18)
	n21=0.00150/(q*A*v21)
	n24=0.00150/(q*A*v24)
	n27=0.00150/(q*A*v27)
	n30=0.00150/(q*A*v30)

	vAvg15=nu15/(3.8E-20*4.90996378e+23)
	vAvg18=nu18/(3.8E-20*5.89195653e+23)
	vAvg21=nu21/(3.8E-20*6.87394929e+23)
	vAvg24=nu24/(3.8E-20*7.85594205e+23)
	vAvg27=nu27/(3.8E-20*8.83793480e+23)
	vAvg30=nu30/(3.8E-20*9.81992756e+23)

	T15=m*(1.085*vAvg15)**2/(3*kb)
	T18=m*(1.085*vAvg18)**2/(3*kb)
	T21=m*(1.085*vAvg21)**2/(3*kb)
	T24=m*(1.085*vAvg24)**2/(3*kb)
	T27=m*(1.085*vAvg27)**2/(3*kb)
	T30=m*(1.085*vAvg30)**2/(3*kb)

	plt.errorbar(EH_15[11:22], v15, color='#F13106', marker = '.', fmt=',', label = '15 Torr')
	plt.errorbar(EH_18[11:22], v18, color='#06E3F1', marker = '.', fmt=',', label = '18 Torr')
	plt.errorbar(EH_21[11:22], v21, color='#EADB20', marker = '.', fmt=',', label = '21 Torr')
	plt.errorbar(EH_24[11:22], v24, color='#3BDA15', marker = '.', fmt=',', label = '24 Torr')
	plt.errorbar(EH_27[11:22], v27, color='#0F3FE3', marker = '.', fmt=',', label = '27 Torr')
	plt.errorbar(EH_30[11:22], v30, color='#B40AB4', marker = '.', fmt=',', label = '30 Torr')
	plt.xlabel('Magnetic Field Strength [Gauss]')
	plt.ylabel('Hall Voltage [V/m]')
	plt.title('Hall Effect')
	plt.legend(loc='best', numpoints=1, ncol=2)
	plt.show()


Vo = np.array([50.5,55.55,60.6,65.65,70.7,75.75])
VoErr = 0.5
Eo = Vo/0.075
EoErr = VoErr/0.075	#V/m
q = 1.602E-19	#C
m = 9.1094E-31 #kg
vel, velErr = driftVel()


def collision():
	nu = (q*Eo)/(m*vel)
	#nu = Eo/vel*2E11
	nuErr = np.sqrt((q*EoErr/(m*vel))**2 + (q*Eo*velErr/(m*vel**2))**2)
	return nu, nuErr

Id = np.array([0.95,1.5,1.5,1.5,1.5,1.5])/1000
Vo = np.array([50.5,55.55,60.6,65.65,70.7,75.75])
A = np.pi*(0.004)**2
P = np.array([15,18,21,24,27,30]) # kgm/s^2
jd = Id/A

def numberDen():
	n = Id/(q*A*vel) 
	nerr = Id*velErr/(q*A*vel**2)
	return n, nerr

kb = 1.380645E-23 #m^2kg/s^2K
T = 295			  #K at room temp
Ag = 6.022E23
R = 8.31445

#get gas density
Ng = 133.322*P*Ag/(R*T)
nu,nuErr = collision()
n, nErr = numberDen()

def rho1():
	rho = (m*nu)/(n*q**2)
	rhoErr = np.sqrt((m*nuErr/(n*q**2))**2 + (m*nu*nErr/(n**2*q**2))**2)
	return rho, rhoErr

#rho = Eo/jd
#rhoErr = EoErr/jd

rho, rhoErr = rho1()

vAvg = nu/(3.8E-20*Ng)
vAvgErr = nuErr/(3.8E-20*Ng)

vRMS = (1.085*vAvg)**2
vRMSerr = 2*1.085**2*vAvg*vAvgErr

T = m*vRMS/(3*kb)
Terr = m*vRMSerr/(3*kb)

mHe = 4*1.6726E-27
vHe = mHe*vel**2/2

ion = n/Ng
ionErr = nErr/Ng

def plotP():
	f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2, sharex=True)
	ax1.errorbar(P, vel, yerr=velErr, color='r', marker = 'o', fmt=',', label = 'Electron drift velocity')
	ax1.set_ylabel('Drift velocity [$ms^{-1}$]')
	ax1.legend(loc='best', numpoints=1)

	ax2.errorbar(P, nu, yerr=nuErr, color='b', marker = 'o', fmt=',', label = 'Electron collision frequency')
	ax2.set_ylabel('Collision frequency [$s^{-1}$]')
	ax2.legend(loc='best', numpoints=1)
	
	ax3.errorbar(P, n, yerr=nErr, color='g', marker = 'o', fmt=',', label = 'Electron number density')
	ax3.set_ylabel('Number density [$m^{-3}$]')
	ax3.legend(loc='best', numpoints=1)
	
	ax4.errorbar(P, ion, yerr=ionErr,color='orange', marker = 'o', fmt=',', label = 'Ionization Factor')
	ax4.set_ylabel('Ionization Factor')
	ax4.legend(loc='best', numpoints=1)
	
	ax5.errorbar(P, rho, yerr=rhoErr, color='purple', marker = 'o', fmt=',', label = 'Resistivity')
	ax5.set_xlabel('Pressure [Torr]')
	ax5.set_ylabel('Resistivity [$\Omega m^{-1}$]')
	ax5.legend(loc='best', numpoints=1)
	
	ax6.errorbar(P, T, yerr=Terr, color='k', marker = 'o', fmt=',', label = 'Temperature')
	ax6.set_xlabel('Pressure [Torr]')
	ax6.set_ylabel('Temperature [$K$]')
	ax6.legend(loc='best', numpoints=1)

i= np.array([0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35])
r = m*20306.28947**2*133.322*30/(q*kb*295*i)

def IonF():
	
	n15 = i/(q*A*vel[0]*1000)
	n18 = i/(q*A*vel[1]*1000)
	n21 = i/(q*A*vel[2]*1000)
	n24 = i/(q*A*vel[3]*1000)
	n27 = i/(q*A*vel[4]*1000)
	n30 = i/(q*A*vel[5]*1000)

	ion15 = n15/Ng[0]
	ion18 = n18/Ng[1]
	ion21 = n21/Ng[2]
	ion24 = n24/Ng[3]
	ion27 = n27/Ng[4]
	ion30 = n30/Ng[5]

	rho15 = m*vel[0]**2*P[0]*1000/(q*kb*295*i)
	rho18 = m*vel[1]**2*P[1]*1000/(q*kb*295*i)
	rho21 = m*vel[2]**2*P[2]*1000/(q*kb*295*i)
	rho24 = m*vel[3]**2*P[3]*1000/(q*kb*295*i)
	rho27 = m*vel[4]**2*P[4]*1000/(q*kb*295*i)
	rho30 = m*vel[5]**2*P[5]*1000/(q*kb*295*i)


	plt.errorbar(i,rho15, color='#F13106', marker = '.', fmt=',', label = '15 Torr')
	plt.errorbar(i,rho18, color='#06E3F1', marker = '.', fmt=',', label = '18 Torr')
	plt.errorbar(i,rho21, color='#EADB20', marker = '.', fmt=',', label = '21 Torr')
	plt.errorbar(i,rho24, color='#3BDA15', marker = '.', fmt=',', label = '24 Torr')
	plt.errorbar(i,rho27, color='#0F3FE3', marker = '.', fmt=',', label = '27 Torr')
	plt.errorbar(i,rho30, color='#B40AB4', marker = '.', fmt=',', label = '30 Torr')
	plt.xlabel('Discharge Current [mA]')
	plt.ylabel('Ionization Factor')
	plt.legend(loc='best', numpoints=1, ncol=2)
	
def printStuff(var):
	print var[0]
	print var[1]
	print var[2]
	print var[3]
	print var[4]
	print var[5]

#plt.errorbar(P, vel, yerr=velErr, color='r', marker = 'o', fmt=',', label = 'Electron drift velocity')
#plt.xlabel('Pressure [Torr]')
#plt.ylabel('Drift velocity [$ms^{-1}$]')
#plt.legend(loc='best', numpoints=1)

#plt.errorbar(P, nu, yerr=nuErr, color='b', marker = 'o', fmt=',', label = 'Electron collision frequency')
#plt.xlabel('Pressure [Torr]')
#plt.ylabel('Collision frequency [$s^{-1}$]')
#plt.legend(loc='best', numpoints=1)
	
#plt.errorbar(P, n, yerr=nErr, color='g', marker = 'o', fmt=',', label = 'Electron number density')
#plt.xlabel('Pressure [Torr]')
#plt.ylabel('Number density [$m^{-3}$]')
#plt.legend(loc='best', numpoints=1)
	
#plt.errorbar(P, ion, yerr=ionErr,color='orange', marker = 'o', fmt=',', label = 'Ionization Factor')
#plt.xlabel('Pressure [Torr]')
#plt.ylabel('Ionization Factor')
#plt.legend(loc='best', numpoints=1)
	
#plt.errorbar(P, rho, yerr=rhoErr, color='purple', marker = 'o', fmt=',', label = 'Resistivity')
#plt.xlabel('Pressure [Torr]')
#plt.ylabel('Resistivity [$\Omega m^{-1}$]')
#plt.legend(loc='best', numpoints=1)
	
#plt.errorbar(P, T, yerr=Terr, color='k', marker = 'o', fmt=',', label = 'Temperature')
#plt.xlabel('Pressure [Torr]')
#plt.ylabel('Temperature [$K$]')
#plt.legend(loc='best', numpoints=1)
