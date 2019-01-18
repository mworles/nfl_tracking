# import packages
import pandas as pd
import numpy as np

# import data
data_dir = 'C:/Users/mworley/nfl_tracking/data/'
rush_ft = pd.read_csv(data_dir + 'interim/rush_selected.csv', index_col=0)
play_ft = pd.read_csv(data_dir + 'interim/play_features.csv', index_col=0)
speed_ft = pd.read_csv(data_dir + 'interim/speed_features.csv', index_col=0)
tgts = pd.read_csv(data_dir + 'interim/epa_rush.csv', index_col=0)

# keep one frame per rush
rush_ft = rush_ft[rush_ft['event'] == 'ball_snap']

# fill x gain to contact with x gain to end if no contact was made
rush_ft['bf_contact'] = rush_ft['x_fromscrim_atcont'].fillna(rush_ft['x_fromscrim'])

x1cols = ['gameId', 'playId', 'testset'] #'bf_contact']
x1 = rush_ft[x1cols]

x2cols = play_ft.columns.tolist()
x2cols.remove('down')
x2 = play_ft[x2cols]

x = pd.merge(x1, x2, on=['gameId', 'playId'], how='inner')
x = pd.merge(x, speed_ft, on=['gameId', 'playId'], how='inner')
x = x.dropna()
y = tgts[['gameId', 'playId', 'EPA', 'WPA', 'success']]

xy = pd.merge(x, y, on=['gameId', 'playId'], how='inner')
us = pd.Series(['_'] * xy.shape[0]).astype(str)
game_play = xy["gameId"].map(str) + us + xy["playId"].map(str)

y = xy[y.columns].set_index(game_play)
x = xy[x.columns].set_index(game_play)
x = x.drop(columns=['game_play', 'gameId', 'playId'])
y = y.drop(columns=['gameId', 'playId'])
x.shape
# save to file
fx = 'processed/features.csv'
print 'writing %s' % (fx)
x.to_csv(data_dir + fx)

fy = 'processed/targets.csv'
print 'writing %s' % (fy)
y.to_csv(data_dir + fy)
