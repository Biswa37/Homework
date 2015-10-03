#!/usr/bin/env python
from astropy.modeling import models, fitting
import numpy 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from StringIO import StringIO
from scipy.optimize import curve_fit

fig = plt.figure(num=None, figsize=(15, 15), dpi=200, facecolor='w', edgecolor='k')

photon_pair_mass = []
num_bins = 50
mass = 126.5
width = 1.66

lines = [line.rstrip('\n') for line in open('data2')]

for line in lines:
    a = np.array(np.loadtxt(StringIO(line)))
    photon_pair_mass.append(a)

photon_pair_mass = np.array(photon_pair_mass)
hist, bin_edges = numpy.histogram(photon_pair_mass,bins=num_bins)
xdata = 0.5*(bin_edges[1:]+bin_edges[:-1])
ydata = hist

#astropy fitting
degree = 4
bck_init = models.Polynomial1D(degree=degree)
fit_bck = fitting.LinearLSQFitter()
b = fit_bck(bck_init, xdata, ydata)

g_init = models.Gaussian1D(amplitude=1E10, mean=mass, stddev=width) + models.Polynomial1D(degree=degree)
fit_g = fitting.LevMarLSQFitter()
g = fit_g(g_init, xdata, ydata)
c2 = (g(xdata)-ydata)**2 /g(xdata)
c2 = np.sum(c2)/num_bins

# Plot the data with the best-fit model
n, bins, patches = plt.hist(photon_pair_mass, num_bins, histtype=u'stepfilled',facecolor='g' , alpha=0.45)

plt.plot(xdata, g(xdata),'b--', lw=4, 
    label=r'$\mathrm{Poly\ bkg\ gaus\ peak\ :\ Mass=%.4f \pm %.4f \ GeV,}\ \Gamma=%.4f \pm %.4f$' %(g.parameters[1],0,g.parameters[2],0))
plt.plot(xdata, g(xdata),'b--', lw=4, 
    label=r'$\mathrm{Poly\ bkg\ gaus\ peak\ : \ \chi^{2} = %.4f \ \ \ \sigma = %.1f}$' %(c2,0))

plt.plot(xdata, b(xdata), 'k-', label='$\mathrm{Background}$', lw=2)
plt.plot(xdata, g(xdata)-b(xdata), 'r-', label='$\mathrm{Gausian}$', lw=3)

signal = []
for i in xrange(num_bins):
    temp = ydata[i] - b(xdata[i])
    if temp <= 0:
        temp = 0
    signal.append(temp)

plt.scatter(xdata, signal ,marker='o', color='r', label=r'$\mathrm{Signal}$')

bkg_int = quad(b, 100, 160)
sig_int = quad(g, 100, 160)

plt.legend()
plt.xlabel(r'$\mathrm{m_{\gamma \gamma} (GeV)}$', fontsize=20)
plt.ylabel(r'Counts (#)', fontsize=18)
plt.ylim((0,np.max(ydata)))
plt.xlim((np.min(xdata),np.max(xdata)))
#plt.show()
fig.savefig('Problem_1_astro.pdf')