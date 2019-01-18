import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'c:/users/mworley/nfl_tracking/src/features')
from feature_functions import *

data_dir = 'C:/Users/mworley/nfl_tracking/data/'

df = pd.read_csv(data_dir + 'interim/rushes_clean.csv', index_col=0)

# verify no frames are duplicated
duplicate_frames = df[df[['gameId', 'playId', 'frame.id']].duplicated()]

print "duplicate frames?"
if duplicate_frames.shape[0] == 0:
    print "No"
else:
    print "Yes"

# create list to group frames by game and play
group = ['gameId', 'playId']
shift = lambda x: x.shift(1)

# create time elapsed per frame and cumulative time elapsed
df['time'] = 0.10
df['time_cum'] = df.groupby(['gameId', 'playId'])['time'].transform('cumsum')

# create a frame counter column
df['frame'] = 1
df['frame_num'] = df.groupby(group)['frame'].transform('cumsum')
df = df.drop(columns=['frame'])

# column for position difference from last frame
df['x_diff'] = df.groupby(group)['x'].diff()
df['y_diff'] = df.groupby(group)['y'].diff()

# create columns for difference between frame value and value at snap
df['x_fromsnap'] = df['x'] - df.groupby(group)['x'].transform('first')
df['y_fromsnap'] = df['y'] - df.groupby(group)['y'].transform('first')
df['x_fromscrim'] = df['x'] - df.groupby(group)['x_ball'].transform('first')

# drop some plays with errors in ball/scrimmage location
df['x_fromscrim_min'] =  df.groupby(group)['x_fromscrim'].transform('min')
df = df[df['x_fromscrim_min'] > - 11.0]
df = df.drop(columns=['x_fromscrim_min'])

# diff from previous row
df['x_fromsnap_diff'] = df.groupby(group)['x_fromsnap'].diff()
df['y_fromsnap_diff'] = df.groupby(group)['y_fromsnap'].diff()
df['x_fromscrim_diff'] = df.groupby(group)['x_fromscrim'].diff()

# compute abolute difference in directional angle
df['dir_lag'] = df.groupby(group)['dir'].apply(shift)
df['dir_diff'] = 180 - abs(abs(df['dir'] - df['dir_lag']) - 180)
df = df.drop(columns=['dir_lag'])

# cumulative columns for distance and absolute direction change
fx_cum = lambda x: x.cumsum()
df['dis_cum'] = df.groupby(group)['dis'].apply(fx_cum)
df['dir_cum'] = df.groupby(group)['dir_diff'].apply(fx_cum)

# for each play, columns for after contact and x position at first contact
df['first_cont'] = np.where(df['event'] == 'first_contact', 1, 0)
df['after_cont'] = df.groupby(group)['first_cont'].shift(1).transform('cummax')
df['x_atcont'] = np.where(df['event'] == 'first_contact', df['x'], np.nan)
df['x_atcont'] = df.groupby(group)['x_atcont'].transform('max')
df['x_fromscrim_atcont'] = df['x_atcont'] - df.groupby(group)['x_ball'].transform('first')
df['x_aftercont'] = df.groupby(group)['x'].transform('last') - df['x_atcont']


# fill x gain to contact with x gain to end if no contact was made
early_con = df['x_fromscrim_atcont'].fillna(df['x_fromscrim'])
df['early_con'] = np.where(early_con < 0, early_con, 0)

# get first frame crossing line of scrimmage
df['across'] = (df['x_fromscrim'] > 0).astype(int)
df['hascrossed'] = df.groupby(group)['across'].transform('cummax')
df['madeacross'] = df.groupby(group)['across'].transform('max')

# get defense features
dfdef = pd.read_csv(data_dir + 'interim/defense_features.csv', index_col=0)

# merge with rusher tracking
mrg = pd.merge(df, dfdef, on=['gameId', 'playId', 'frame.id'], how='inner')

write_file('interim/rush_features.csv', mrg)
