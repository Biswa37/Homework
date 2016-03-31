import numpy as np
import matplotlib.pyplot as plt
from ROOT import TLorentzVector
from scipy.optimize import curve_fit
import pandas as pd

data = pd.DataFrame.from_csv('truth.csv') #Read data from csv file
data = data.apply(pd.to_numeric, errors='coerce') #Make the numbers the right type

fig = plt.figure(num=None, figsize=(11,8.5), dpi=200, facecolor='w', edgecolor='k')

num_bins = 75
def chi_2(ys,yknown):
    total = 0
    for i in xrange(len(yknown)):
        temp = (ys[i]-yknown[i])**2.0
        if yknown[i] == 0:
            total += temp
        else :
            total += temp/yknown[i]
    return total/len(yknown)

def gauss(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

mom = 0
phi_mass = []
temp = TLorentzVector(0,0,0,0)
for index, row in data.iterrows():
    if row.ID == 333:
        if temp.M() != 0 and temp.M() < 1.2:
            phi_mass.append(temp.M())
        temp.SetPxPyPzE(0,0,0,0)
        mom = row['event']
    elif np.abs(row.ID) == 321 and row.mother == mom:
        temp = temp + TLorentzVector(row.Px,row.Py,row.Pz,row.E)

mass_sum = np.array(phi_mass)

hist, bin_edges = np.histogram(mass_sum,bins=num_bins)
xdata = 0.5*(bin_edges[1:]+bin_edges[:-1])
ydata = hist

n, bins, patches = plt.hist(mass_sum, num_bins, histtype=u'stepfilled',facecolor='g' , alpha=0.5)

n = len(xdata)
mean = sum(xdata*ydata)/n 
x0 = np.array([mean,1])

popt, pcov = curve_fit(gauss, xdata, ydata,p0=[1,mean,1])
perr = np.sqrt(np.diag(pcov))
plt.plot(xdata,gauss(xdata,*popt),'g-', lw=4,
    label=r'$\mathrm{Gaus:\ \mu=%.3f \pm %.3f \ GeV,}\ \sigma=%.3f \pm %.3f$' %(popt[1], perr[1], popt[0], perr[0]))
plt.plot(xdata,gauss(xdata,*popt),'g-', lw=4,
    label=r'$\mathrm{Gaus: \ \chi^{2} = %.3f}$' %(chi_2(gauss(xdata,*popt),ydata)))

plt.xlim([0.9,1.15])
plt.ylim([0,40])
plt.xlabel(r'Mass (GeV)')
plt.ylabel(r'Counts (#)')
plt.legend(loc=1)
plt.savefig('non_smear.pdf')