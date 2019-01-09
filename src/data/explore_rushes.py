# import packages
import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import os

# set plot style
plt.style.use('ggplot')

# import data
data_dir = 'C:/Users/mworley/nflbdb/Data/'
df = pd.read_csv(data_dir + 'interim/rushes_std.csv', index_col=0)

# plot all runs
dfsub = df.loc[:, ['displayName', 'gameId', 'playId', 'x_std', 'y_std']]
dfsub.set_index('x_std', inplace=True)
dfsub.groupby(['displayName', 'gameId', 'playId'])['y_std'].plot()
plt.show()


# %%
fig, ax = plt.subplots(figsize=(8,6))
#bp = dfsubgby.plot(x = 'x', y = 'y', kind='line', ax=ax)
for label, df in dfsubgby:
    df.plot(ax=ax, label=label)
plt.show()



# %%
df[df['frame.id'] == 1].groupby('displayName')['frame.id'].count().sort_values(ascending=False).head(30)


# %%
df['dir_lag'] = df.groupby(['displayName', 'gameId', 'playId'])['dir'].apply(lambda y: y.shift(1))
df['dir_lag'] = np.where(df['dir_lag'].isnull(), df['dir'], df['dir_lag'])
df['dir_dif'] = abs(df['dir'] - df['dir_lag'])
df['dir_cum'] = df.groupby(['displayName', 'gameId', 'playId'])['dir_dif'].apply(lambda x: x.cumsum())

# %%
df['dis_cum'] = df.groupby(['displayName', 'gameId', 'playId'])['dis'].apply(lambda x: x.cumsum())

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

# code to calculate distance between players
dfnd = play[(play['team'] == 'home') & (play['jerseyNumber'] == 30)]
off = play[(play['team'] == 'away') & (play['jerseyNumber'] == 27)]

ctokeep = ['x', 'y', 's', 'dis', 'dir', 'event', 'displayName', 'frame.id']
off = off[ctokeep]
dfnd = dfnd[ctokeep]

newcols = [c + '_off' for c in off.columns if c !='frame.id']
newcols.append('frame.id')
off.columns = newcols
off.head()

newcols = [c + '_def' for c in dfnd.columns if c !='frame.id']
newcols.append('frame.id')
dfnd.columns = newcols
dfnd.head()

# %%
both = pd.merge(off, dfnd, on='frame.id', how='inner')

both.iloc[41:, :]
import math
rw = both.iloc[54, :]
dist = math.sqrt( (rw['x_def'] - rw['x_off'])**2 + (rw['y_def'] - rw['y_off'])**2 )

def calc_dist(row):
    x1 = row['x_off']
    x2 = row['x_def']
    y1 = row['y_off']
    y2 = row['y_def']
    dist = math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
    return dist

both['distance'] = both.apply(lambda x: calc_dist(x), axis=1)
