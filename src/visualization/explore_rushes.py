# import packages
import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import os

# set plot style
plt.style.use('ggplot')

# import data
data_dir = 'C:/Users/mworley/nfl_tracking/data/'
df = pd.read_csv(data_dir + 'interim/rushes_clean.csv', index_col=0,
                 low_memory=False)
dfd = pd.read_csv(data_dir + 'interim/defenders.csv', index_col=0)

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
for i, d in dfsub.groupby(['gameId', 'playId']):
    bc = d[d['dis_min'] >1]
    x = bc['y']
    y = bc['xuscrim']
    c = bc['dis_min']
    #s = bc['dis_min']
    ax = plt.scatter(x = x, y = y, marker='D', c=c, s = 8, cmap='hot', alpha=0.50)
    ac = d[d['dis_min'] <=1]
    x = ac['y']
    y = ac['xuscrim']
    c = ac['dis_min']
    #s = bc['dis_min']
    ax2 = plt.scatter(x = x, y = y, marker='+', c=c, s = 8, cmap='hot', alpha=0.50)
plt.clim(0,8)
plt.colorbar()
plt.show()


# %%
dfsub = dfsub[['gameId', 'playId', 'dis_cum', 's']]
dfsub.set_index('dis_cum', inplace=True)
dfsub.groupby(['gameId', 'playId'])['s'].plot()
plt.show()
