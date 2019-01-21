import pandas as pd
import numpy as np
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

# write file for speed feature extraction
write_file('interim/rush_toLOS_frames.csv', df)

# keep one frame per rush
df_one = df[df['event'] == 'ball_snap']

# set first 80% of games as training set
game_ids = df_one['gameId'].unique()
traincut_game = game_ids[int(len(game_ids) * 0.80)]
df_one['testset'] = (df_one['gameId'] > traincut_game).astype(int)

# save file
dfout = game_play_index(df_one)
write_file('interim/rush_toLOS.csv', dfout)

# read in data again
df = pd.read_csv(data_dir + 'interim/rush_features.csv', index_col=0)

# select runs where rusher crossed line of scrimmage
df = df[df['madeacross'] == 1]

# select runs where contact ocurred
df = df[df['x_atcont'].notnull()]

# keep runs where:
# contact occurred 20 yards or less past LOS
df = df[df['x_fromscrim_atcont'] <=20]
# yards after contact was 20 yards or less
df = df[df['x_aftercont'] <=20]

# write file for speed feature extraction
write_file('interim/rush_contact_frames.csv', df)

# keep one frame per rush
df_one = df[df['event'] == 'ball_snap']

# set first 80% of games as training set
game_ids = df_one['gameId'].unique()
traincut_game = game_ids[int(len(game_ids) * 0.80)]
df_one['testset'] = (df_one['gameId'] > traincut_game).astype(int)

# save file
dfout = game_play_index(df_one)
write_file('interim/rush_contact.csv', dfout)

# select frames occuring up to point of first contact
df = df[df['after_cont'] != 1]

# write file for speed feature extraction
write_file('interim/rush_tocontact_frames.csv', df)

# keep one frame per rush
df_one = df_one[df_one['event'] == 'ball_snap']

# set first 80% of games as training set
game_ids = df_one['gameId'].unique()
traincut_game = game_ids[int(len(game_ids) * 0.80)]
df_one['testset'] = (df_one['gameId'] > traincut_game).astype(int)

# save file
dfout = game_play_index(df_one)
write_file('interim/rush_tocontact.csv', dfout)
