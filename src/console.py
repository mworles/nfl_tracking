import pandas as pd
import numpy as np
from tsfresh import extract_features
import matplotlib.pyplot as plt

# import data
data_dir = 'C:/Users/mworley/nfl_tracking/data/'
plot_dir = 'C:/Users/mworley/nfl_tracking/plots/'
df = pd.read_csv(data_dir + 'interim/rushes_clean.csv', index_col=0)


us = pd.Series(['_'] * df.shape[0]).astype(str)
df['game_play'] = df["gameId"].map(str) + us + df["playId"].map(str)
ts = df.loc[: , ['game_play', 'frame.id', 'xufs', 'yu', 's', 'dir', 'dis']].copy()
ts = ts[~ts.isnull().any(axis=1)]
ts['one'] = 1
ts['gpnum'] = ts.groupby('game_play')['one'].transform('cumsum')
ts['gpnum_max'] = ts.groupby('game_play')['gpnum'].transform('max')
ts = ts[(ts['gpnum_max'] >= 30) & (ts['gpnum'] < 31)]
ts = ts[['game_play', 'frame.id', 's']]

# %%
features = extract_features(ts, column_id='game_play', column_sort='frame.id')
features.to_csv(data_dir + '/interim/ts_features.csv')

# %%
cols = features.columns
n = 0
for c in cols:
    ss = features.sort_values(c, ascending=False).head(10)
    ss_gp = ss.index.values
    sub = df[df['game_play'].isin(ss_gp)]

    fig, axs = plt.subplots(figsize=(10, 8))

    for i, g in sub.groupby(['gameId', 'playId']):
        x = g['yu']
        y = g['xuscrim']
        col = g['s']
        last = g.iloc[-1, :]
        last = last[['displayName', 'yu', 'xuscrim']].values
        plt.scatter(x = x, y = y, marker="8", c=col, s = 30,
                         cmap='magma', alpha=0.50)
        plt.annotate(last[0], (last[1], last[2]))
    plt.xlim(-5, 60)
    plt.ylim(-10, 30)
    plt.title(c)
    plt.clim(0, 10)
    plt.colorbar()
    plt.savefig(plot_dir + str(n) + '.jpg')
    plt.close()
    print ' ' .join(['plot', str(n), c])
    n += 1

features.shape
# %%
n = 0
for c in cols:
    print ' '.join([str(n), c])
    n +=1
