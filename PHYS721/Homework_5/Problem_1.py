#!/usr/bin/env python
import numpy 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from StringIO import StringIO
from ROOT import TLorentzVector
from scipy.optimize import curve_fit

fig = plt.figure(num=None, figsize=(15, 15), dpi=200, facecolor='w', edgecolor='k')
num_bins = 10000
m_k = 0.493677
m_phi = 1.019461
mass = 1.019461
gamma = 0.00426

mass_sum = []
vec1 = TLorentzVector()
vec2 = TLorentzVector()
num_bins = 60

def BW(Energy,Mass,Gamma):
    g = ((Mass**2.0 + Gamma**2.0)*Mass**2.0)**(1.0/2.0)
    k = (2.0 * 2.0**(1.0/2.0) * Mass * Gamma * g)/(np.pi * (Mass**(2.0)+g)**(1.0/2.0))
    return (k/((Energy**2.0-Mass**2.0)**2.0 + (Gamma*Mass)**2.0))

def BW_NonR(Energy,Mass,Gamma):
    return (((Gamma/(2.0*np.pi)))/((Energy-Mass)**2.0 + (Gamma/2.0)**2.0))
 
def BW_2(Energy,Mass,Gamma_0):
    g = ((Mass**2.0 + Gamma_P(Energy,Gamma_0)**2.0)*Mass**2.0)**(1.0/2.0)
    k = (2.0 * 2.0**(1.0/2.0) * Mass * Gamma_P(Energy,Gamma_0) * g)/(np.pi * (Mass**(2.0)+g)**(1.0/2.0))
    return (k/((Energy**2.0-Mass**2.0)**2.0 + (Gamma_P(Energy,Gamma_0)*Mass)**2.0))


def Gamma_P(Energy,Gamma_0):
    p = ((Energy**2.0/4.0)-m_k**2.0)**(1.0/2.0)
    p0 = ((m_phi**2.0/4.0)-m_k**2.0)**(1.0/2.0)
    return Gamma_0*(p/p0)**3.0


lines = [line.rstrip('\n') for line in open('data1')]

for line in lines:
    a = np.array(np.loadtxt(StringIO(line)))
    vec1.SetPxPyPzE(a[1],a[2],a[3],a[0])
    vec2.SetPxPyPzE(a[5],a[6],a[7],a[4])
    mass_sum.append((vec1+vec2).M())

hist, bin_edges = numpy.histogram(mass_sum,bins=num_bins)
xdata = bin_edges[1:]
ydata = hist

n, bins, patches = plt.hist(mass_sum, num_bins, histtype=u'stepfilled',facecolor='green' , alpha=0.5)

popt, pcov = curve_fit(BW, xdata, ydata)
plt.plot(xdata,BW(xdata,popt[0],popt[1]),'b-')
print popt

popt, pcov = curve_fit(BW_NonR, xdata, ydata)
plt.plot(xdata,BW_NonR(xdata,popt[0],popt[1]),'r-')
print popt

popt, pcov = curve_fit(BW_2, xdata, ydata)
plt.plot(xdata,BW_2(xdata,popt[0],popt[1]),'g-')
print popt

#plt.xlabel(r'Mass (GeV)')
#plt.ylabel(r'Counts (#)')
plt.legend()

#plt.show()
fig.savefig('Problem_1.pdf')