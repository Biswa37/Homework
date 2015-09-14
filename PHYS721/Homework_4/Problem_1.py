#!/usr/bin/env python
import numpy 
import numpy as np
from StringIO import StringIO
from ROOT import TLorentzVector
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

fig = plt.figure(num=None, figsize=(10, 10), dpi=200, facecolor='w', edgecolor='k')
lines = [line.rstrip('\n') for line in open('data1')]
mass_sum = []
vec1 = TLorentzVector()
vec2 = TLorentzVector()
num_bins = 60

for line in lines:
    a = np.array(np.loadtxt(StringIO(line)))
    vec1.SetPxPyPzE(a[1],a[2],a[3],a[0])
    vec2.SetPxPyPzE(a[5],a[6],a[7],a[4])
    mass_sum.append((vec1+vec2).M())

def myBW(Energy,Mass,Gamma):
    g = ((Mass**2.0 + Gamma**2.0)*Mass**2.0)**(1.0/2.0)
    k = (2.0 * 2.0**(1.0/2.0) * Mass * Gamma * g)/(np.pi * (Mass**(2.0)+g)**(1.0/2.0))
    return (k/((Energy**2.0-Mass**2.0)**2.0 + (Gamma*Mass)**2.0))

def BW_NonR(Energy,Mass,Gamma):
    return (((Gamma/(2.0*np.pi)))/((Energy-Mass)**2.0 + (Gamma/2.0)**2.0))

#This is not right I still need to fix this
def Gamma_2(Energy):
    return popt_2[1]
    
def myBW_2(Energy,Mass):
    g = ((Mass**2.0 + Gamma_2(Energy)**2.0)*Mass**2.0)**(1.0/2.0)
    k = (2.0 * 2.0**(1.0/2.0) * Mass * Gamma_2(Energy) * g)/(np.pi * (Mass**(2.0)+g)**(1.0/2.0))
    return (k/((Energy**2.0-Mass**2.0)**2.0 + (Gamma_2(Energy)*Mass)**2.0))
#Up to here

hist, bin_edges = numpy.histogram(mass_sum,bins=num_bins)
xdata = 0.5*(bin_edges[1:]+bin_edges[:-1])
ydata = hist
n, bins, patches = plt.hist(mass_sum, num_bins, histtype=u'stepfilled',facecolor='green' , alpha=0.5)
x0 = numpy.array([1.02, 0.0043])
    
popt_1, pcov_1 = curve_fit(myBW, xdata, ydata, p0=x0)
plt.plot(xdata,myBW(xdata,popt_1[0],popt_1[1]),'g-',lw=4,label=r'$\mathrm{Relatavistic \ BW:\ Mass=%.7f \ GeV,}\ \Gamma=%.7f$' %(popt_1[0], popt_1[1]))
    
popt_2, pcov_2 = curve_fit(BW_NonR, xdata, ydata, p0=x0)
plt.plot(xdata,BW_NonR(xdata,popt_2[0],popt_2[1]),'r--',lw=2,label=r'$\mathrm{Non-Rel. \ BW:\ Mass=%.7f \ GeV,}\ \Gamma=%.7f$' %(popt_2[0], popt_2[1]))

#This part too
popt_3, pcov_3 = curve_fit(myBW_2, xdata, ydata, p0=x0[0])
plt.plot(xdata,myBW_2(xdata,popt_3[0]),'b-.',lw=2,label=r'$\mathrm{BW:\ Mass=%.7f \ GeV,}\ \Gamma=%.7f$' %(popt_3[0], popt_3[0]))
    
plt.xlabel(r'Mass (GeV)')
plt.ylabel(r'Counts (#)')
plt.legend()

#plt.show()
fig.savefig('Problem_1.pdf')