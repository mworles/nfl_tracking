import pandas as pd
import sys
sys.path.insert(0, 'c:/users/mworley/nfl_tracking/src/features')
from feature_functions import *

# import data file
data_dir = 'C:/Users/mworley/nfl_tracking/data/'
df = pd.read_csv(data_dir + 'interim/rush_features.csv', index_col=0)

# select runs where rusher crossed line of scrimmage
df = df[df['madeacross'] == 1]

# select frames occurring up to first frame crossing line of scrimmage
df['across_sum'] = df.groupby(['gameId', 'playId'])['across'].transform('cumsum')
df = df[df['across_sum'] < 2]


# create column indicating distance to closest defender at LOS
nc = df.groupby(['gameId', 'playId'])['dfndis_min'].transform('last')
df['dfndis_LOS'] = nc

# keep one frame per rush
df = df[df['event'] == 'ball_snap']

game_ids = df['gameId'].unique()
traincut_game = game_ids[int(len(game_ids) * 0.80)]
df['testset'] = (df['gameId'] > traincut_game).astype(int)

# save file
dfout = game_play_index(df)

write_file('interim/rush_selected.csv', dfout)
