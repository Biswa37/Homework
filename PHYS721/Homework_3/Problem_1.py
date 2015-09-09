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
    
hist, bin_edges = numpy.histogram(mass_sum,bins=num_bins)
xdata = bin_edges[1:]
ydata = hist

def myBW(Energy,Mass,Gamma):
    g = ((Mass**2.0 + Gamma**2.0)*Mass**2.0)**(1.0/2.0)
    k = (2.0 * 2.0**(1.0/2.0) * Mass * Gamma * g)/(np.pi * (Mass**(2.0)+g)**(1.0/2.0))
    return (k/((Energy**2.0-Mass**2.0)**2.0 + (Gamma*Mass)**2.0))

n, bins, patches = plt.hist(mass_sum, num_bins, histtype=u'stepfilled',facecolor='green' , alpha=0.5)

popt, pcov = curve_fit(myBW, xdata, ydata)
plt.plot(xdata,myBW(xdata,popt[0],popt[1]),'b-')
plt.xlabel(r'Mass (GeV)')
plt.ylabel(r'Counts (#)')
plt.title(r'$\mathrm{Mass\ Histogram:\ Mass=%.4f \ GeV,}\ \Gamma=%.4f$' %(popt[0], popt[1]))

#plt.show()

fig.savefig('Problem_1.pdf')