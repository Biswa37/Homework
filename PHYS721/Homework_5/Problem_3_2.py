#!/usr/bin/env python
import numpy 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from StringIO import StringIO
from ROOT import TLorentzVector
from scipy.optimize import curve_fit
from ROOT import TRandom
from ROOT import TF1

fig = plt.figure(num=None, figsize=(15, 15), dpi=200, facecolor='w', edgecolor='k')
m_k = 0.493677
m_phi = 1.019461

mass_sum = []
vec1 = TLorentzVector()
vec2 = TLorentzVector()
num_bins = 60

def poly(x, c1, c2, c3):
    return c1*x*x + c2*x + c3

def BW(Energy,Mass,Gamma, c1, c2, c3):
    g = ((Mass**2.0 + Gamma**2.0)*Mass**2.0)**(1.0/2.0)
    k = (2.0 * 2.0**(1.0/2.0) * Mass * Gamma * g)/(np.pi * (Mass**(2.0)+g)**(1.0/2.0))
    return (k/((Energy**2.0-Mass**2.0)**2.0 + (Gamma*Mass)**2.0)) + poly(Energy, c1, c2, c3)

def BW_NonR(Energy,Mass,Gamma, c1, c2, c3):
    return (((Gamma/(2.0*np.pi)))/((Energy-Mass)**2.0 + (Gamma/2.0)**2.0)) + poly(Energy, c1, c2, c3)

def BW_2(Energy,Mass,Gamma, c1, c2, c3):
    g = abs((Mass**2.0 + Gamma*P_fac(Energy)**2.0)*Mass**2.0)**(1.0/2.0)
    k = (2.0 * 2.0**(1.0/2.0) * Mass * Gamma*P_fac(Energy) * g)/(np.pi * (Mass**(2.0)+g)**(1.0/2.0))
    return (k/((Energy**2.0-Mass**2.0)**2.0 + (Gamma*P_fac(Energy)*Mass)**2.0)) + poly(Energy, c1, c2, c3)

def P_fac(Energy):
    p = abs((Energy**2.0/4.0)-m_k**2.0)**(1.0/2.0)
    p0 = ((m_phi**2.0/4.0)-m_k**2.0)**(1.0/2.0)
    return (p/p0)**3.0

def chi_2(ys,yknown):
    total = 0
    for i in xrange(len(yknown)):
        temp = (ys[i]-yknown[i])**2.0
        if yknown[i] == 0:
            total += 1#temp
        else :
            total += temp/yknown[i]
    return total/len(yknown)

lines = [line.rstrip('\n') for line in open('data1')]

for line in lines:
    a = np.array(np.loadtxt(StringIO(line)))
    vec1.SetPxPyPzE(a[1],a[2],a[3],a[0])
    vec2.SetPxPyPzE(a[5],a[6],a[7],a[4])
    mass_sum.append((vec1+vec2).M())

#new stuff###
f1 = TF1("f1","[0]*x*x+[1]*x+[2]",0.99,1.09)
""" Concave
f1.SetParameter(0,66000.0)
f1.SetParameter(1,-137780.0)
f1.SetParameter(2,71916.0)
"""
""" Convex
f1.SetParameter(0,-8000.0)
f1.SetParameter(1,15240.0)
f1.SetParameter(2,-7096.8)
"""
"""reasonable """
f1.SetParameter(0,-3.0)
f1.SetParameter(1,7.0)
f1.SetParameter(2,-3.0)

r = [] 
for i in range(len(mass_sum)):
    r.append(f1.GetRandom())

bkg = np.array(r)
mass_sum = np.concatenate([mass_sum, bkg])
########

hist, bin_edges = numpy.histogram(mass_sum,bins=num_bins)
xdata = 0.5*(bin_edges[1:]+bin_edges[:-1])
ydata = hist

x0 = np.array([1.02,0.0043,0,0,0])

n, bins, patches = plt.hist(mass_sum, num_bins, histtype=u'stepfilled',facecolor='g' , alpha=0.5)

popt_3, pcov_3 = curve_fit(BW_2, xdata, ydata, p0=x0)
perr_3 = np.sqrt(np.diag(pcov_3))
plt.plot(xdata,BW_2(xdata,popt_3[0],popt_3[1],popt_3[2],popt_3[3],popt_3[4]),'g-', lw=4,
    label=r'$\mathrm{Mass \ dep. \ BW:\ Mass=%.6f \pm %.6f \ GeV,}\ \Gamma=%.6f \pm %.6f$' %(popt_3[0], perr_3[0], popt_3[1], perr_3[1]))
plt.plot(xdata,BW_2(xdata,popt_3[0],popt_3[1],popt_3[2],popt_3[3],popt_3[4]),'g-', lw=4,
    label=r'$\mathrm{Mass \ dep. \ BW \ : \ \chi^{2} = %.6f}$' %(chi_2(BW_2(xdata,popt_3[0],popt_3[1],popt_3[2],popt_3[3],popt_3[4]),ydata)))

popt_1, pcov_1 = curve_fit(BW, xdata, ydata, p0=x0)
perr_1 = np.sqrt(np.diag(pcov_1))
plt.plot(xdata,BW(xdata,popt_1[0],popt_1[1],popt_1[2],popt_1[3],popt_1[4]),'b-.', lw=4,
    label=r'$\mathrm{Relatavistic \ BW:\ Mass=%.6f \pm %.6f \ GeV,}\ \Gamma=%.6f \pm %.6f$' %(popt_1[0], perr_1[0], popt_1[1], perr_1[1]))
plt.plot(xdata,BW(xdata,popt_1[0],popt_1[1],popt_1[2],popt_1[3],popt_1[4]),'b-.', lw=4,
    label=r'$\mathrm{Rel. \ BW \ : \ \chi^{2} = %.6f}$' %(chi_2(BW(xdata,popt_1[0],popt_1[1],popt_1[2],popt_1[3],popt_1[4]),ydata)))

popt_2, pcov_2 = curve_fit(BW_NonR, xdata, ydata, p0=x0)
perr_2 = np.sqrt(np.diag(pcov_2))
plt.plot(xdata,BW_NonR(xdata,popt_2[0],popt_2[1],popt_2[2],popt_2[3],popt_2[4]),'r--', lw=4,
    label=r'$\mathrm{Non-Rel. \ BW:\ Mass=%.6f \pm %.6f \ GeV,}\ \Gamma=%.6f \pm %.6f$' %(popt_2[0], perr_2[0], popt_2[1], perr_2[1]))
plt.plot(xdata,BW_NonR(xdata,popt_2[0],popt_2[1],popt_2[2],popt_2[3],popt_2[4]),'r--', lw=4,
    label=r'$\mathrm{Non-Rel. \ BW  \ : \ \chi^{2} = %.6f}$' %(chi_2(BW_NonR(xdata,popt_2[0],popt_2[1],popt_2[2],popt_2[3],popt_2[4]),ydata)))

poly_params = np.array([(popt_2[2]+popt_3[2]+popt_1[2])/3.0,(popt_2[3]+popt_3[3]+popt_1[3])/3.0,(popt_2[4]+popt_3[4]+popt_1[4])/3.0])

plt.plot(xdata,poly(xdata,poly_params[0],poly_params[1],poly_params[2]),'k-', lw=2)

plt.xlabel(r'Mass (GeV)')
plt.ylabel(r'Counts (#)')
plt.legend()

#plt.show()
fig.savefig('Problem_3_withbackfit.pdf')