import numpy as np
import matplotlib.pyplot as plt
from ROOT import TLorentzVector
from scipy.optimize import curve_fit
import pandas as pd

data = pd.DataFrame.from_csv('truth.csv') #Read data from csv file
data = data.apply(pd.to_numeric, errors='coerce') #Make the numbers the right type
#Drop the data I won't be using in this graph
data.drop(['i1','i2','i3'],inplace=True,axis=1)
data.drop(['X','Y','Z','T'],inplace=True,axis=1)

data['Pt'] = np.sqrt(data.Px**2 + data.Py**2) #calculate Pt = Sqrt(Px^2 + Py^2)

'''
For some reason the phi values don't have the same value all the value
and it's not always that they are off by a minus sign.
'''
data['Phi_x'] = np.arccos(data.Px/data.Pt) #Calculate Phi from Px = Pt Cos(phi)
data['Phi_y'] = np.arcsin(data.Py/data.Pt) #Calculate Phi from Py = Pt Sin(phi)
data['Theta'] = np.arctan(data.Pt/data.Pz) #Calculate Theta from Pz = Pt/Tan(theta)

'''
Can't seem to figure out the correct smearing factors
'''
pt_smear = 115e-4 
phi_smear = 1e-4 
theta_smear = 1e-4 


#Calculate smearing from random normal distrobution
data['Pt_measure'] = np.random.normal(data.Pt, pt_smear*data.Pt, len(data))
data['Phi_x_measure'] = np.random.normal(data.Phi_x, phi_smear, len(data))
data['Phi_y_measure'] = np.sign(data.Py)*data.Phi_x_measure
data['Theta_measure'] = np.random.normal(data.Theta, theta_smear, len(data))

#Calcualte measured values from smeared data
data['Px_measure'] = data.Pt_measure * np.cos(data.Phi_x_measure)
data['Py_measure'] = data.Pt_measure * np.sin(data.Phi_y_measure)
data['Pz_measure'] = data.Pt/np.tan(data.Theta_measure)
data['E_measure'] = np.sqrt(data.Px_measure**2 + data.Py_measure**2 + data.Pz_measure**2)

num_bins = 75
fig = plt.figure(num=None, figsize=(11,8.5), dpi=200, facecolor='w', edgecolor='k')

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

n, bins, patches = plt.hist(mass_sum, num_bins, histtype=u'stepfilled',facecolor='b' , alpha=0.5)

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
        temp = temp + TLorentzVector(row.Px_measure,row.Py_measure,row.Pz_measure,row.E)

mass_sum = np.array(phi_mass)

n, bins, patches = plt.hist(mass_sum, num_bins, histtype=u'stepfilled',facecolor='g' , alpha=0.5)

plt.xlim([0.9,1.15])
plt.ylim([0,40])
plt.xlabel(r'Mass (GeV)')
plt.ylabel(r'Counts (#)')
plt.savefig('both.pdf')