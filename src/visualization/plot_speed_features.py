import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# import data
data_dir = 'C:/Users/mworley/nfl_tracking/data/'
plot_dir = 'C:/Users/mworley/nfl_tracking/plots/'
rfe = pd.read_csv(data_dir + 'out/speed_rfe.csv', index_col=0)
df = pd.read_csv(data_dir + 'interim/rush_selected.csv', index_col=0)
speed = pd.read_csv(data_dir + 'tmp/speed_features_tmp.csv', index_col=0)

# create df 'game_play' column
us = pd.Series(['_'] * df.shape[0]).astype(str)
df['game_play'] = df["gameId"].map(str) + us + df["playId"].map(str)

# plot speed over frames, colored by feature value
#x_speed = speed[rfe.columns]
x_speed = speed.reset_index().rename(columns={'id': 'game_play'})
mrg = pd.merge(df, x_speed, on='game_play', how='inner')

# %%
# plot speed over frames, colored by feature value
n = 5

comp_feats = ['s__abs_energy',
              's__c3__lag_1',
              's__fft_coefficient__coeff_0__attr_"real"',
              's__fft_coefficient__coeff_0__attr_"abs"']

for feat in comp_feats: #rfe.columns:
    print feat
    minmax = mrg[feat].describe().loc[['min', 'max'], ].values
    p1 = np.percentile(mrg[feat], 10)
    p2 = np.percentile(mrg[feat], 90)

    sub = mrg[(mrg[feat] <= p1) | (mrg[feat] >= p2)]
    norm = mpl.colors.Normalize(vmin=minmax[0], vmax=minmax[1])
    cmap = mpl.cm.get_cmap('magma')

    # %%
    fig, axs = plt.subplots(figsize=(10, 8))

    for i, g in sub.groupby('game_play'):
        if any(g[feat] <= p1):
            x = g['frame_num']
            y = g['s']
            cval = g[feat].iloc[0]
            col = cmap(norm(cval))
            ax1 = plt.scatter(x = x, y = y, marker="8", c=col, s = 40,
                              alpha=0.25)
        else: # elif any(g[feat] >= p2):
            x = g['frame_num']
            y = g['s']
            cval = g[feat].iloc[0]
            col = cmap(norm(cval))
            ax2 = plt.scatter(x = x, y = y, marker="v", c=col, s = 40,
                              alpha=0.25)
    plt.clim(minmax[0], minmax[1])
    plt.title(feat)
    m = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array(sub[feat].values)
    plt.colorbar(m)
    #plt.show()
    #plt.close()
    plt.savefig(plot_dir + str(n) + '_highlow'+ '.jpg')
    n += 1
'''
# %%
runs = mrg.drop_duplicates('game_play')
runs['n_runs'] = runs.groupby(['nflId'])['game_play'].transform('count')
runs = runs[runs['n_runs'] > 20]
dir = ['h', 'h', 'l', 'l', 'l']
gb = runs.groupby('displayName')

for tup in zip(rfe.columns, dir):
    gb_mean = gb[tup[0]].mean()
    if tup[1] == 'h':
        gb_mean = gb_mean.sort_values(ascending=False)
    else:
        gb_mean = gb_mean.sort_values()
    print gb_mean.iloc[0:10]


# %%
cols = rfe.columns
n = 0
for c in cols:
    ss = rfe.sort_values(c, ascending=False).head(20)
    ss_gp = ss.index.values
    sub = df[df['game_play'].isin(ss_gp)]

    fig, axs = plt.subplots(figsize=(10, 8))

    for i, g in sub.groupby(['gameId', 'playId']):
        x = g['y']
        y = g['x_fromscrim']
        col = g['s']
        last = g.iloc[-1, :]
        last = last[['displayName', 'y', 'x_fromscrim']].values
        plt.scatter(x = x, y = y, marker="8", c=col, s = 30,
                         cmap='magma', alpha=0.50)
        plt.annotate(last[0], (last[1], last[2]))
    plt.xlim(-5, 60)
    plt.ylim(-10, 30)
    plt.title(c)
    plt.clim(0, 10)
    plt.colorbar()
    #plt.savefig(plot_dir + str(n) + '.jpg')
    plt.show()
    plt.close()
    print ' ' .join(['plot', str(n), c])
    n += 1
'''
