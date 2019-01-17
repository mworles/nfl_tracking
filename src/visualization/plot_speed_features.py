import pandas as pd
import numpy as np
from tsfresh import extract_features
import matplotlib.pyplot as plt

# import data
data_dir = 'C:/Users/mworley/nfl_tracking/data/'
plot_dir = 'C:/Users/mworley/nfl_tracking/plots/'
features = pd.read_csv(data_dir + 'tmp/speed_features_tmp.csv', index_col=0)
features.columns.tolist().index('s__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_0__w_20')


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
