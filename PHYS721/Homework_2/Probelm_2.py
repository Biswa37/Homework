#!/usr/bin/env python
from ROOT import gStyle
from ROOT import gROOT
from ROOT import TStyle,TCanvas
from ROOT import TH1D,TH1F
from ROOT import TH2D
from ROOT import TLorentzVector
from StringIO import StringIO
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ROOT
font = {'size'	:	10}
#####Homework 2 Question 2
mass1 = []
mass2 = []
mass_sum = []
vec1 = TLorentzVector()
vec2 = TLorentzVector()
lines = [line.rstrip('\n') for line in open('data1')]
bins = 75

fig = plt.figure(num=None, figsize=(10, 10), dpi=200, facecolor='w', edgecolor='k')
matplotlib.rc('font', **font)

for line in lines:
    a = np.array(np.loadtxt(StringIO(line)))
    vec1.SetPxPyPzE(a[1],a[2],a[3],a[0])
    vec2.SetPxPyPzE(a[5],a[6],a[7],a[4])
    mass1.append(vec1.M())
    mass2.append(vec2.M())
    mass_sum.append((vec1+vec2).M())

plt.subplot(221)
plt.hist(mass1, bins, histtype=u'stepfilled', alpha=0.5)
plt.xlabel(r'Mass (GeV/$c^2$)')
plt.ylabel(r'Counts (#)')
plt.title(r'Mass found from four vector 1')

plt.subplot(222)
plt.hist(mass2, bins, histtype=u'stepfilled',facecolor='red' , alpha=0.5)
plt.xlabel(r'Mass (GeV/$c^2$)')
plt.ylabel(r'Counts (#)')
plt.title(r'Mass found from four vector 2')

plt.subplot(212)
plt.hist(mass_sum, 4*bins, histtype=u'stepfilled',facecolor='green' , alpha=0.5)
plt.xlabel(r'Mass (GeV/$c^2$)')
plt.ylabel(r'Counts (#)')
plt.title(r'Mass found from the sum of four vectors')
#plt.show()

fig.savefig('Problem_2.pdf')