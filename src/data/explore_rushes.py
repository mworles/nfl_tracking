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

dfd.groupby(['gameId', 'playId'])['frame.id'].count().describe()


# %%
dfd['dis_min'] = dfd.groupby(['gameId', 'playId', 'frame.id'])['distance'].transform('min')
c_drop = dfd.columns[-6:-1]
dfd = dfd.drop(columns=c_drop)
dfd = dfd.drop_duplicates()
dfd = dfd[['playId', 'gameId', 'frame.id', 'dis_min']]
df = pd.merge(df, dfd, on=['playId', 'gameId', 'frame.id'], how='inner')

# cumulative minimum defender distance
df['dis_mincum'] = df.groupby(['gameId', 'playId'])['dis_min'].transform('cummin')
df['first_contact'] = np.where(df['event'] == 'first_contact', 1, 0)
df['after_contact'] = df.groupby(['gameId', 'playId'])['first_contact'].transform('cummax')
df[df['after_contact'] == 1][['playId', 'frame.id', 'first_contact', 'xuscrim', 'dis_min', 'event']].head(20)
smax_first5 = df[df['dis_cum'] <=5].groupby(['gameId', 'playId'], as_index=False)['s'].max()
smax_first5.rename(columns={'s': 'smax_first5'}, inplace=True)
df = pd.merge(df, smax_first5, on=['gameId', 'playId'], how='inner')

# compute peak speed before/after contact
dfbc = df[df['after_contact'] == 0].groupby(['gameId', 'playId'], as_index=False)
s_max_bc = dfbc['s'].apply(max)
s_max_bc = s_max_bc.reset_index()
s_max_bc.columns=['gameId', 'playId', 's_max_bc']
df = pd.merge(df, s_max_bc, on=['gameId', 'playId'], how='inner')
df['s_max_bc_loss'] = df['s'] - df['s_max_bc']

# x position at first contact
df['x_atcontact'] = np.where(df['event'] == 'first_contact', df['xu'], np.nan)
df['x_atcontact'] = df.groupby(['gameId', 'playId'])['x_atcontact'].transform('max')
df['xufs_atcontact'] = df['x_atcontact'] - df['x_snap']
df['xuscrim_atcontact'] = df['x_atcontact'] - df['xu_ball_snap']
df['xu_fromcontact'] = df['xu'] - df['x_atcontact']
df['xu_aftercontact'] = df.groupby(['gameId', 'playId'])['xu_fromcontact'].transform('max')

# frame at contact
df['time'] = 0.10
df['time_cum'] = df.groupby(['gameId', 'playId'])['time'].transform('cumsum')
df['time_atcontact'] = np.where(df['event'] == 'first_contact', df['time_cum'], np.nan)
df['time_atcontact'] = df.groupby(['gameId', 'playId'])['time_atcontact'].transform('max')
df['time_fromcontact'] = df['time_cum'] - df['time_atcontact']


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

# attempt a heat map
import seaborn as sns
runs = dfsub.pivot("xufs", 'yu', 's')
dfsub = dfsub.reset_index()



# %%
dfsub = dfsub[['gameId', 'playId', 'dis_cum', 's']]
dfsub.set_index('dis_cum', inplace=True)
dfsub.groupby(['gameId', 'playId'])['s'].plot()
plt.show()

# %%
fig, axs = plt.subplots(figsize=(12, 6))
dfsub = df[df['first_contact'] == 1]
dfsub = dfsub[dfsub['PlayResult'] < 20]
y = dfsub['xuscrim']
x = dfsub['s']
c = dfsub['smax_first5']
plt.scatter(x, y, c=c, cmap='magma', alpha = 0.5)
plt.colorbar()
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
