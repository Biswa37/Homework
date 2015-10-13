#!/usr/bin/env python
import numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from StringIO import StringIO
from scipy.optimize import curve_fit

fig = plt.figure(num=None, figsize=(15, 15), dpi=200, facecolor='w', edgecolor='k')

photon_pair_mass = []
num_bins = 60
mass = 126.5
width = 1.66


def poly(x, c1, c2, c3):
    return c1*x*x + c2*x + c3

def gaussian(x, mu, sig, const):
    return const * np.exp(-(x - mu)**2 / 2*sig**2)

def BW(Energy,Mass,Gamma):
    g = ((Mass**2.0 + Gamma**2.0)*Mass**2.0)**(1.0/2.0)
    k = (2.0 * 2.0**(1.0/2.0) * Mass * Gamma * g)/(np.pi * (Mass**(2.0)+g)**(1.0/2.0))
    return (k/((Energy**2.0-Mass**2.0)**2.0 + (Gamma*Mass)**2.0))

def gaus_poly(x, mu, sig, cont, c1, c2, c3):
    return poly(x, c1, c2, c3) + gaussian(x, mu, sig, cont)

def bw_poly(x, Mass, Gamma, c1, c2, c3):
    return poly(x, c1, c2, c3) + BW(x,Mass,Gamma)

def chi_2(ys,yknown):
    total = 0
    for i in range(len(yknown)):
        temp = (ys[i]-yknown[i])**2.0
        if yknown[i] == 0:
            total += 1#temp
        else :
            total += temp/yknown[i]
    return total/len(yknown)

lines = [line.rstrip('\n') for line in open('data2')]

for line in lines:
    a = np.array(np.loadtxt(StringIO(line)))
    photon_pair_mass.append(a)

photon_pair_mass = np.array(photon_pair_mass)
hist, bin_edges = numpy.histogram(photon_pair_mass,bins=num_bins)
xdata = 0.5*(bin_edges[1:]+bin_edges[:-1])
ydata = hist

n, bins, patches = plt.hist(photon_pair_mass, num_bins, histtype=u'stepfilled',facecolor='g' , alpha=0.45)

#first fit to get background parameters
x0 = np.array([mass,width,1,1,1])
popt_1, pcov_1 = curve_fit(poly, xdata, ydata,maxfev = 10000)
#fit with known background to get peak
x0 = np.array([mass,width,popt_1[0],popt_1[1],popt_1[2]])
popt_1, pcov_1 = curve_fit(bw_poly, xdata, ydata, p0=x0,maxfev = 10000)
perr_1 = np.sqrt(np.diag(pcov_1))

plt.plot(xdata,bw_poly(xdata,popt_1[0],popt_1[1],popt_1[2],popt_1[3],popt_1[4]),'b--', lw=4,
    label=r'$\mathrm{Poly\ bkg\ BW\ peak\ :\ Mass=%.4f \pm %.4f \ GeV,}\ \Gamma=%.4f \pm %.4f$' 
         %(popt_1[0], perr_1[0], popt_1[1], perr_1[1]))

poly_params = np.array([popt_1[2],popt_1[3],popt_1[4]])


##Use integrals to find sigma values
bkg_int = quad(poly, 0, 1000, args=(poly_params[0],poly_params[1],poly_params[2]))
sig_int = quad(bw_poly, 0, 1000, args=(popt_1[0],popt_1[1],popt_1[2],popt_1[3],popt_1[4]))

sigma = 100-abs(bkg_int[0]-sig_int[0])
##

c2 = chi_2(bw_poly(xdata,popt_1[0],popt_1[1],popt_1[2],popt_1[3],popt_1[4]),ydata)

plt.plot(xdata,bw_poly(xdata,popt_1[0],popt_1[1],popt_1[2],popt_1[3],popt_1[4]),'b--', lw=4,
    label=r'$\mathrm{Poly\ bkg\ BW\ peak\ : \ \chi^{2} = %.4f \ \ \ \sigma = %.1f}$' %(c2,sigma))

signal = []
for i in range(num_bins):
    temp = ydata[i] - poly(xdata[i],poly_params[0],poly_params[1],poly_params[2])
    if temp <= 0:
        temp = 0
    signal.append(temp)

plt.scatter(xdata, signal,marker='o', color='r', label=r'$\mathrm{Signal}$')
plt.plot(xdata,poly(xdata,poly_params[0],poly_params[1],poly_params[2]),'k-', lw=2, label=r'$\mathrm{Background}$')
plt.plot(xdata,BW(xdata,popt_1[0],popt_1[1]),'r-',lw=3, label=r'$\mathrm{Signal}$')
plt.legend()
plt.xlabel(r'$\mathrm{m_{\gamma \gamma} (GeV)}$', fontsize=20)
plt.ylabel(r'$\mathrm{Counts}$', fontsize=18)
plt.ylim((0,np.max(ydata)))
plt.xlim((np.min(xdata),np.max(xdata)))
#plt.show()
fig.savefig('Problem_1_BW.pdf')
