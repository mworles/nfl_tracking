import pandas as pd
import numpy as np
from tsfresh import extract_features

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
features = extract_features(ts, column_id='game_play', column_sort='frame_num')

# import target data
t = pd.read_csv(dat_dir + 'interim/epa_rush.csv')


'''
# save to file
f = 'interim/play_features.csv'
print 'writing %s' % (f)
play_features.to_csv(data_dir + f)
'''
