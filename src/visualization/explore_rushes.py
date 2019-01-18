# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# set plot style
plt.style.use('ggplot')

# import data
data_dir = 'C:/Users/mworley/nfl_tracking/data/'
df = pd.read_csv(data_dir + 'interim/rush_features.csv', index_col=0,
                 low_memory=False)

# %%
a = df['displayName'] == 'Kareem Hunt'
b = (df['xuscrim'] < 10) & (df['xuscrim'] > -10)
c = df['dis_min'] < 2
dfsub = df[(a) & (b) & (c)]
#
fig, axs = plt.subplots(figsize=(10, 8))
for i, g in dfsub.groupby(['gameId', 'playId']):
    x = g['yu']
    y = g['xuscrim']
    c = g['s']
    ax2 = plt.scatter(x = x, y = y, marker="8", c=c, s = 40,
                     cmap='magma', alpha=0.40)
    dc = g[g['time_fromcontact'] == 0]
    xc = dc['yu']
    yc = dc['xuscrim']
    ax3 = plt.scatter(x = xc, y = yc, marker="8", c='red', s = 40,
                     cmap='magma', alpha=0.40)
plt.clim(0,8)
#plt.colorbar()
plt.show()

# %%
# scatterplot to appear as heatmap
fig, axs = plt.subplots(figsize=(10, 8))
for i, d in dfsub.groupby(['gameId', 'playId']):
    bc = d[d['after_contact'] == 1]
    x = bc['yu']
    y = bc['xuscrim']
    c = bc['s']
    ax = plt.scatter(x = x, y = y, marker="8", c=c, s = 200,
                     cmap='magma', alpha=0.25)
plt.clim(0,8)
plt.colorbar()
plt.show()

# %%
# plot all runs
fig, axs = plt.subplots(figsize=(10, 8))
sub = df[df['time_cum'] < 6]
for i, d in sub.groupby(['gameId', 'playId']):
    x = d['time_cum']
    y = d['s']
    c = d['x_fromscrim']
    ax = plt.plot(x = x, y = y) #, marker='8', s = 8, c = c,
                  #cmap='magma', alpha=0.50)
#plt.clim(0,8)
plt.colorbar()
plt.show()


# %%
dfsub = dfsub[['gameId', 'playId', 'dis_cum', 's']]
dfsub.set_index('dis_cum', inplace=True)
dfsub.groupby(['gameId', 'playId'])['s'].plot()
plt.show()
