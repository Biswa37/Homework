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

gStyle.SetPalette(55);

#####Homework 2 Question 1
energy1 = []
energy2 = []
vec1 = TLorentzVector()
vec2 = TLorentzVector()
lines = [line.rstrip('\n') for line in open('data1')]

EnergyHist1 = TH1F('EnergyHist1','Energy',25,0,2.5)
EnergyHist2 = TH1F('EnergyHist2','Energy',25,0,2.5)
fig = plt.figure(num=None, figsize=(8, 8), dpi=800, facecolor='w', edgecolor='k')

for line in lines:
    a = np.array(np.loadtxt(StringIO(line)))
    vec1.SetPxPyPzE(a[1],a[2],a[3],a[0])
    vec2.SetPxPyPzE(a[5],a[6],a[7],a[4])
    energy1.append(vec1.E())
    EnergyHist1.Fill(vec1.E())
    energy2.append(vec2.E())
    EnergyHist2.Fill(vec2.E())

plt.hist(energy1, 25, histtype=u'stepfilled' , alpha=0.5)
plt.xlabel(r'Energy (GeV)')
plt.ylabel(r'Counts (#)')
plt.title(r'Energy four vector 1')
#plt.show()

fig.savefig('Problem_1.pdf')
