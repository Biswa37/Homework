import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import numpy as np

#data = pd.DataFrame.from_csv('events/evnt_1.csv')
data = pd.DataFrame.from_csv('truth.csv')
data = data.apply(pd.to_numeric, errors='coerce')

fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(111, projection='3d')

mothers = np.hstack(np.array(data['mother']))
uniq_mom = np.unique(mothers)
values = cm.viridis(np.linspace(0,1,len(uniq_mom)))
#values = cm.Paired(np.linspace(0,1,len(uniq_mom)))
col = dict(zip(uniq_mom, values))


for mom in uniq_mom:
    temp = data[data['mother'] == mom]
    xs = np.hstack(np.array(temp['X']))
    ys = np.hstack(np.array(temp['Y']))
    zs = np.hstack(np.array(temp['Z']))
    ax.scatter(xs, ys, zs,alpha=1,color=col[mom])

plt.show()