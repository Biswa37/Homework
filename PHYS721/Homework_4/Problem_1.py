#!/usr/bin/env python
import numpy 
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad
from scipy.optimize import curve_fit

fig = plt.figure(num=None, figsize=(15, 15), dpi=200, facecolor='w', edgecolor='k')
num_bins = 1000

def myBW(Energy,Mass,Gamma):
    g = ((Mass**2.0 + Gamma**2.0)*Mass**2.0)**(1.0/2.0)
    k = (2.0 * 2.0**(1.0/2.0) * Mass * Gamma * g)/(np.pi * (Mass**(2.0)+g)**(1.0/2.0))
    return (k/((Energy**2.0-Mass**2.0)**2.0 + (Gamma*Mass)**2.0))

def BW_NonR(Energy,Mass,Gamma):
    return (((Gamma/(2.0*np.pi)))/((Energy-Mass)**2.0 + (Gamma/2.0)**2.0))
 
def myBW_2(Energy,Mass,Gamma_0):
    g = ((Mass**2.0 + Gamma_P(Energy,Gamma_0)**2.0)*Mass**2.0)**(1.0/2.0)
    k = (2.0 * 2.0**(1.0/2.0) * Mass * Gamma_P(Energy,Gamma_0) * g)/(np.pi * (Mass**(2.0)+g)**(1.0/2.0))
    return (k/((Energy**2.0-Mass**2.0)**2.0 + (Gamma_P(Energy,Gamma_0)*Mass)**2.0))

def Gamma_P(Energy,Gamma_0):
    p = 1.5
    p0 = 1
    return Gamma_0*(p/p0)**3.0

"""
My homemade algorithm for getting FWHM
Imporve it by chaning sigma to be dynamic to adapt to changing widths 

"""
def FWHM(x,y):
    temp = []
    max_2 = y.max()/2.0
    sigma = 2 #this needs to be dynamic so that it only gets two values
    YMaxIndex = numpy.where(y == y.max())[0][0]

    for value in y:
        if value >= max_2-sigma and value <= max_2+sigma:
            temp.append(numpy.where(y == value)[0][0])
    temp = x[temp]
    return temp, (temp[1] - temp[0])

x = np.linspace(0.99, 1.08, num_bins)
y1 = myBW(x,1.019461,0.00426)
y2 = BW_NonR(x,1.019461,0.00426)
y3 = myBW_2(x,1.019461,0.00426)

temp, val = FWHM(x,y1)
plt.plot(
    [ temp[0] , temp[1]  ],
    [ y1[numpy.where(x == temp[0])[0][0]], y1[numpy.where(x == temp[1])[0][0]] ],
    'g', lw=4
    )

plt.plot(x,y1,'g-',lw=2,
    label=r'$\mathrm{Relatavistic \ BW:\ Peak=%.7f \ GeV,}\ FWHM=%.7f$' %(max(y1), val))

temp, val = FWHM(x,y2)
plt.plot(
    [ temp[0] , temp[1]  ],
    [ y1[numpy.where(x == temp[0])[0][0]], y1[numpy.where(x == temp[1])[0][0]] ],
    'r', lw=4
    )

plt.plot(x,y2,'r--',lw=2,
    label=r'$\mathrm{Non-Rel. \ BW:\ Peak=%.7f \ GeV,}\ FWHM=%.7f$' %(max(y2), val))

temp, val = FWHM(x,y3)
plt.plot(
    [ temp[0] , temp[1]  ],
    [ y1[numpy.where(x == temp[0])[0][0]], y1[numpy.where(x == temp[1])[0][0]] ],
    'b', lw=4
    )

plt.plot(x,y3,'b-.',lw=2,
    label=r'$\mathrm{Relatavistic \ BW:\ Peak=%.7f \ GeV,}\ FWHM=%.7f$' %(max(y3), val))

plt.xlabel(r'Mass (GeV)')
plt.ylabel(r'Counts (#)')
plt.legend()

#plt.show()
fig.savefig('Problem_1.pdf')