#!/usr/bin/env python
import numpy 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from StringIO import StringIO
from ROOT import TLorentzVector
from scipy.optimize import curve_fit

fig = plt.figure(num=None, figsize=(15, 15), dpi=200, facecolor='w', edgecolor='k')
m_k = 0.493677
m_phi = 1.019461

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

def BW_2(Energy,Mass,Gamma):
    g = abs((Mass**2.0 + Gamma*P_fac(Energy)**2.0)*Mass**2.0)**(1.0/2.0)
    k = (2.0 * 2.0**(1.0/2.0) * Mass * Gamma*P_fac(Energy) * g)/(np.pi * (Mass**(2.0)+g)**(1.0/2.0))
    return (k/((Energy**2.0-Mass**2.0)**2.0 + (Gamma*P_fac(Energy)*Mass)**2.0))

def P_fac(Energy):
    p = abs((Energy**2.0/4.0)-m_k**2.0)**(1.0/2.0)
    p0 = ((m_phi**2.0/4.0)-m_k**2.0)**(1.0/2.0)
    return (p/p0)**3.0

#change coeficiets to tune graph
def bac_poly(x):
    c1 = 1
    c2 = 1
    c3 = 1
    return c1 * x**2.0 + c2 * x + c3


lines = [line.rstrip('\n') for line in open('data1')]

for line in lines:
    a = np.array(np.loadtxt(StringIO(line)))
    vec1.SetPxPyPzE(a[1],a[2],a[3],a[0])
    vec2.SetPxPyPzE(a[5],a[6],a[7],a[4])
    mass_sum.append((vec1+vec2).M())

#new stuff###
bkg = numpy.random.uniform(0.99, 1.09, size=len(mass_sum))
bkg = bac_poly(bkg)

########

hist, bin_edges = numpy.histogram(mass_sum,bins=num_bins)
###
bkg_hist, bkg_bin_edges = numpy.histogram(bkg,bins=num_bins)
xdata = 0.5*(bin_edges[1:]+bin_edges[:-1])
ydata = hist

n, bins, patches = plt.hist(mass_sum, num_bins, histtype=u'stepfilled',facecolor='b' , alpha=0.5)

x0 = np.array([1.02,0.0043])

popt_3, pcov_3 = curve_fit(BW_2, xdata, ydata, p0=x0)
perr_3 = np.sqrt(np.diag(pcov_3))
plt.plot(xdata,BW_2(xdata,popt_3[0],popt_3[1]),'g-', lw=4,
    label=r'$\mathrm{Relatavistic \ BW:\ Mass=%.7f \pm %.7f \ GeV,}\ \Gamma=%.7f \pm %.7f$' %(popt_3[0], perr_3[0], popt_3[1], perr_3[1]))

popt_1, pcov_1 = curve_fit(BW, xdata, ydata, p0=x0)
perr_1 = np.sqrt(np.diag(pcov_1))
plt.plot(xdata,BW(xdata,popt_1[0],popt_1[1]),'b-.', lw=4,
    label=r'$\mathrm{Relatavistic \ BW:\ Mass=%.7f \pm %.7f \ GeV,}\ \Gamma=%.7f \pm %.7f$' %(popt_1[0], perr_1[0], popt_1[1], perr_1[1]))

popt_2, pcov_2 = curve_fit(BW_NonR, xdata, ydata, p0=x0)
perr_2 = np.sqrt(np.diag(pcov_2))
plt.plot(xdata,BW_NonR(xdata,popt_2[0],popt_2[1]),'r--', lw=4,
    label=r'$\mathrm{Relatavistic \ BW:\ Mass=%.7f \pm %.7f \ GeV,}\ \Gamma=%.7f \pm %.7f$' %(popt_2[0], perr_2[0], popt_2[1], perr_2[1]))

plt.xlabel(r'Mass (GeV)')
plt.ylabel(r'Counts (#)')
plt.legend()

#plt.show()
fig.savefig('Problem_3.pdf')