#!/usr/bin/env python
import numpy
import numpy as np
import matplotlib.pyplot as plt
"""from scipy.integrate import quad"""

fig = plt.figure(
    num=None,
    figsize=(15, 15),
    dpi=200,
    facecolor='w',
    edgecolor='k'
    )
num_bins = 10000
m_k = 0.493677
m_phi = 1.019461
mass = 1.019461
gamma = 0.00426


def BW(Energy, Mass, Gamma):
    g = ((Mass**2.0 + Gamma**2.0)*Mass**2.0)**(1.0/2.0)
    k = (2.0 * 2.0**(1.0/2.0) * Mass * Gamma * g)/(np.pi * (Mass**(2.0)+g)**(1.0/2.0))
    return (k/((Energy**2.0-Mass**2.0)**2.0 + (Gamma*Mass)**2.0))


def BW_NonR(Energy, Mass, Gamma):
    return (((Gamma/(2.0*np.pi)))/((Energy-Mass)**2.0 + (Gamma/2.0)**2.0))


def BW_2(Energy, Mass, Gamma_0):
    g = ((Mass**2.0 + Gamma_P(Energy, Gamma_0)**2.0)*Mass**2.0)**(1.0/2.0)
    k = (2.0 * 2.0**(1.0/2.0) * Mass * Gamma_P(Energy, Gamma_0) * g)/(np.pi * (Mass**(2.0)+g)**(1.0/2.0))
    return (k/((Energy**2.0-Mass**2.0)**2.0 + (Gamma_P(Energy, Gamma_0)*Mass)**2.0))


def Gamma_P(Energy, Gamma_0):
    p = ((Energy**2.0/4.0)-m_k**2.0)**(1.0/2.0)
    p0 = ((m_phi**2.0/4.0)-m_k**2.0)**(1.0/2.0)
    return Gamma_0*(p/p0)**3.0


"""
My homemade algorithm for getting FWHM
Imporve it by changing sigma to be dynamic to adapt to changing widths

"""
def FWHM(x,y):
    temp = []
    max_2 = y.max()/2.0
    sigma = 0.25 #this should be dynamic so that it only gets two values

    for value in y:
        if value >= max_2-sigma and value <= max_2+sigma:
            temp.append(numpy.where(y == value)[0][0])

    temp = x[temp] #convert bin numbers to x values
    return temp, (temp[-1] - temp[0])

def normalize(values):
    return values

x = np.linspace(0.99, 1.1, num_bins)
y1 = BW(x,mass,gamma)
y2 = BW_NonR(x,mass,gamma)
y3 = BW_2(x,mass,gamma)

temp_1, val_1 = FWHM(x,y1)
temp_2, val_2 = FWHM(x,y2)
temp_3, val_3 = FWHM(x,y3)

plt.plot(x,y1,'g-',lw=4,
    label=r'$\mathrm{Relatavistic \ BW:\ Peak=%.7f \ GeV,}\ FWHM=%.7f \ GeV$' %(max(y1), val_1))

plt.plot(x,y2,'r--',lw=4,
    label=r'$\mathrm{Non-Rel. \ BW:\ Peak=%.7f \ GeV,}\ FWHM=%.7f \ GeV$' %(max(y2), val_2))


plt.plot(x,y3,'b-.',lw=6,
    label=r'$\mathrm{Rel. \ BW \ with \ Mass \ dep. \ \Gamma:\ Peak=%.7f \ GeV,}\ FWHM=%.7f \ GeV$' %(max(y3), val_3))

"""
#These are visual checks of the FWHM. The more horiontal the line the better accuacy of FWHM.
plt.plot(
    [ temp[0] , temp[-1]  ],
    [ normalize(y1)[numpy.where(x == temp[0])[0][0]], normalize(y1)[numpy.where(x == temp[-1])[0][0]] ],
    'g', lw=2
    )

plt.plot(
    [ temp[0] , temp[-1]  ],
    [ normalize(y2)[numpy.where(x == temp[0])[0][0]], normalize(y2)[numpy.where(x == temp[-1])[0][0]] ],
    'r', lw=2
    )

plt.plot(
    [ temp[0] , temp[-1]  ],
    [ normalize(y3)[numpy.where(x == temp[0])[0][0]], normalize(y3)[numpy.where(x == temp[-1])[0][0]] ],
    'b', lw=2
    )
"""
""" 
#check normilizations. 
print quad(BW, 0, 200, args=(mass,gamma))
print quad(BW_2, 0, 200, args=(mass,gamma))
print quad(BW_NonR, 0.99, 200, args=(mass,gamma))
"""

plt.xlabel(r'Mass (GeV)')
plt.ylabel(r'Counts (#)')
plt.legend()

#plt.show()
fig.savefig('Problem_1.pdf')