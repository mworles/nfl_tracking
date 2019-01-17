import pandas as pd
import numpy as np
from tsfresh import extract_features, select_features, calculate_relevance_table
from tsfresh.feature_selection.relevance import calculate_relevance_table

# import data file
data_dir = 'C:/Users/mworley/nfl_tracking/data/'
df = pd.read_csv(data_dir + 'interim/rush_features.csv', index_col=0)
# select runs where rusher crossed line of scrimmage
df = df[df['madeacross'] == 1]

# select frames occurring before crossing line of scrimmage
df = df[df['hascrossed'] == 1]

df = df.reset_index()
# create 'game_play' column as unique identifier
us = pd.Series(['_'] * df.shape[0]).astype(str)
df['game_play'] = df["gameId"].map(str) + us + df["playId"].map(str)

# select data for tsfresh feature extraction
ts = df[['game_play', 'frame_num', 's']].copy()
ts = ts[ts['s'].notnull()]

# import target data
t = pd.read_csv(data_dir + 'interim/epa_rush.csv', index_col=0)
game_plays = df[['gameId', 'playId', 'game_play']].drop_duplicates()
tsy = pd.merge(t, game_plays, on=['gameId', 'playId'], how='inner')
tsy = tsy.set_index('game_play')

features = extract_features(ts, column_id='game_play', column_sort='frame_num')

# save to file
f = 'tmp/speed_features_tmp.csv'
print 'writing %s' % (f)
features.to_csv(data_dir + f)

# %%
mrg = pd.merge(features, tsy, left_index=True, right_index=True, how='inner')
mrg = mrg.sort_index()
features = mrg[features.columns].copy()
y = mrg['success'].values

# %%
# drop features that have missing values, cannot be used in models
features = features.dropna(axis=1)
ff = select_features(features, y, n_jobs=8)
ff = ff.reset_index().rename(columns={'index': 'game_play'})
ff = pd.merge(ff, game_plays, on='game_play', how='inner')

#save to file
f = 'interim/speed_features.csv'
print 'writing %s' % (f)
ff.to_csv(data_dir + f)
