#!/usr/local/bin/ipython
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import json

fig_size = (16, 9)
fig = plt.figure(num=None, figsize=fig_size, dpi=200, facecolor='w', edgecolor='k')

with open('acc.json') as data_file:    
    data = json.load(data_file)

df = pd.DataFrame(data["Accelerator"])
df['logE'] = df["Energy_MeV"].apply(np.log)
types = np.hstack(np.array(df['type']))

uniq = np.unique(types)
values = cm.rainbow(np.linspace(0,1,len(uniq)))
col = dict(zip(uniq, values))

for i,acc in zip(xrange(len(uniq)),uniq):
    temp = df[df['type'] == acc]
    year = np.hstack(np.array(temp['year']))
    logE = np.hstack(np.array(temp['logE']))
    c = col[acc]
    plt.scatter(year, logE, alpha=1,color=c,label=r'%s' %(acc))

plt.xlabel(r'Year')
plt.ylabel(r'Log of Energy')
plt.legend(loc=4)
#plt.show()
plt.savefig("acc.pdf")