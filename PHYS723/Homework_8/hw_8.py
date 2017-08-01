import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.patches import Ellipse
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

#All information obtained from:
#http://pdg.lbl.gov/2015/reviews/rpp2015-rev-ckm-matrix.pdf
#http://ckmfitter.in2p3.fr 

Vckm = np.matrix([[0.97427,0.22536,0.00355],[0.22522,0.97343,0.0414],[0.00886,0.0405,0.99914]])
Vckm_error = np.matrix([[0.00014,0.00061,0.00015],[0.00061,0.00015,0.0012],[0.00033,0.0012,0.00005]])

rho = 0.124
rho_error = 0.024
eta = 0.354
eta_error = 0.015

e = Ellipse((rho, eta), 2*rho_error, 2*eta_error, 0)

xs = [0,1,rho,0]
ys = [0,0,eta,0]
xerr = [0,0,rho_error,0]
yerr = [0,0,eta_error,0]

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.add_artist(e)
e.set_facecolor([0.97427,0.22536,0.00355])
e.set_clip_box(ax.bbox)

plt.scatter(xs,ys,lw=2)
plt.plot(xs,ys,lw=2)

plt.ylabel(r'$\bar{\eta}$',fontsize=28)
plt.xlabel(r'$\bar{\rho}$',fontsize=28)
plt.ylim([-0.1,0.45])
plt.xlim([-0.1,1.1])
plt.title(r'CKM matrix values [$\bar{\eta},\bar{\rho}$]',fontsize=20)

plt.savefig("CKMTri.pdf")