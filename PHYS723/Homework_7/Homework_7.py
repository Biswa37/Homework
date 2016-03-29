import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure(figsize=(11,8.5))

data = pd.DataFrame.from_csv('truth.csv')
data = data.apply(pd.to_numeric, errors='coerce')

event_end = []
for index, row in data.iterrows():
    if row.mother == -1:
        if index != 1:
            event_end.append(index-1)

num_of_events = np.hstack(data['event'][event_end])
plt.hist(num_of_events,alpha=0.8,histtype='stepfilled')
plt.ylabel('Frequency')
plt.xlabel('Number of particles')
plt.title("Number of particles per event")
plt.savefig('Number_of_particles.pdf')

fig = plt.figure(figsize=(11,8.5))
pions = data[np.abs(data.ID) == 211]
ax = pions.Px.plot.hist(figsize=(11,8.5),bins=50,alpha=0.8,histtype='stepfilled')
ax.set_xlabel('Pion momentum in X direction')
ax.set_title('Pion momentum in X direction')
plt.savefig('Pion_px.pdf')