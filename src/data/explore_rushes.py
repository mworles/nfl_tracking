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
df = pd.read_csv(data_dir + 'interim/rushes_clean.csv', index_col=0)

# %%
# subset data
dfsub = df[df['PlayResult'] < 10]
dfsub = df[df['dir_cum'] < 150]
dfsub = dfsub[dfsub['xuscrim'] < 5]
dfsub = dfsub[dfsub['displayName'] == "Ty Montgomery"]
dfsub = dfsub.loc[:, ['gameId', 'playId', 'frame.id', 's', 'xufs', 'yu', 'dir_cum']]
#dfsub.set_index('frame.id', inplace=True)

# plot all runs
for i, d in dfsub.groupby(['gameId', 'playId']):
    x = d['frame.id']
    y = d['dir_cum']
    c = d['s']
    plt.scatter(x = x, y = y, c=c, s = 10, cmap='hot', alpha=0.50)
plt.show()

# %% 3-d plots
dfsub = df.loc[:, ['gameId', 'playId', 'displayName', 'frame.id',
                   'xufs', 'xuscrim', 'yu', 's', 'dis', 'dir_cum', 'dis_cum']]
dfsub = dfsub[dfsub['displayName'] == "Dalvin Cook"]
dfsub = dfsub[dfsub['dir_cum'] < 100]

# create arrays for plot
for i, d in dfsub.groupby(['gameId', 'playId']):
    x = d['dir_cum'].values
    y = d['dis_cum'].values
    s = d['s'].values
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(x, y, s, cmap='viridis', edgecolor='none')
plt.show()


# %%
#chk = df[df['gameId'] == 2017091100]
chk = df.loc[:, ['displayName', 'quarter', 'event', 'frame.id', 'x',
                  'playId', 'xufs', 'yu', 'x_ball', 'x_snap', 'x_ball_cmid', 'homedir']]

chk['xufs_min'] = chk.groupby(['playId'])['xufs'].transform('min')
chk = chk[chk['event'] == 'ball_snap']
chk.sort_values(['xufs_min', 'playId', 'frame.id'], inplace=True)
chk.head(40)


# %%
fig, ax = plt.subplots(figsize=(8,6))
#bp = dfsubgby.plot(x = 'x', y = 'y', kind='line', ax=ax)
for label, df in dfsubgby:
    df.plot(ax=ax, label=label)
plt.show()



# %%
df[df['frame.id'] == 1].groupby('displayName')['frame.id'].count().sort_values(ascending=False).head(30)


# %%
# %% subset data for plot
#dfsub = df[df['displayName'] == "Le'Veon Bell"]
dfsub = df.loc[:, ['displayName', 'gameId', 'playId', 'dis_cum', 's']]

# create groupby and plot
dfsub.set_index('dis_cum', inplace=True)
dfsub.groupby(['displayName', 'gameId', 'playId'])['s'].plot()
plt.show()

# %%
fig, ax = plt.subplots(figsize=(8,6))
#bp = dfsubgby.plot(x = 'x', y = 'y', kind='line', ax=ax)
for label, df in dfsubgby:
    df.plot(ax=ax, label=label)
plt.show()

# %%
dfsub.set_index('Date', inplace=True)
df.groupby('ticker')['adj_close'].plot(legend=True)

# %% create arrays for plot
x = dfsub['x'].values
y = dfsub['y'].values
s = dfsub['s'].values

# %% create plot
ax = plt.axes(projection='3d')
ax.plot_trisurf(x, y, s, cmap='viridis', edgecolor='none')
plt.show()


# %%

plt.close()
