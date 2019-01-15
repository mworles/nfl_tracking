import pandas as pd
import numpy as np

data_dir = 'C:/Users/mworley/nfl_tracking/data/'
df = pd.read_csv(data_dir + 'interim/rushes_clean.csv', index_col=0)
#games = pd.read_csv(data_dir + 'raw/games.csv')

# x of previous row
# verify no frames are duplicated

duplicate_frames = df[df[['gameId', 'playId', 'frame.id']].duplicated()]

print "duplicate frames?"
if duplicate_frames.shape[0] == 0:
    print "No"
else:
    print "Yes"

# set list to group by game play
group = ['gameId', 'playId']
shift = lambda x: x.shift(1)

df['x_diff'] = df.groupby(group)['x'].diff()
df['y_diff'] = df.groupby(group)['y'].diff()

# function to get a play_level value and transform across all play rows
df['x_fromsnap'] = df['x'] - df.groupby(group)['x'].transform('first')
df['y_fromsnap'] = df['y'] - df.groupby(group)['y'].transform('first')
df['x_fromscrim'] = df['x'] - df.groupby(group)['x_ball'].transform('first')

df['x_fromscrim_min'] =  df.groupby(group)['x_fromscrim'].transform('min')

# drop some plays with errors in ball/scrimmage location
df = df[df['x_fromscrim_min'] > - 11.0]
df = df.drop(columns=['x_fromscrim_min'])

# diff from previous row
df['x_fromsnap_diff'] = df.groupby(group)['x_fromsnap'].diff()
df['y_fromsnap_diff'] = df.groupby(group)['y_fromsnap'].diff()
df['x_fromscrim_diff'] = df.groupby(group)['x_fromscrim'].diff()

# dir of previous row
df['dir_lag'] = df.groupby(group)['dir'].apply(shift)
df['dir_ch'] = 180 - abs(abs(df['dir'] - df['dir_lag']) - 180)
df = df.drop(columns=['dir_lag'])

# cumulative columns
fx_cum = lambda x: x.cumsum()
df['dis_cum'] = df.groupby(group)['dis'].apply(fx_cum)
df['dir_cum'] = df.groupby(group)['dir_ch'].apply(fx_cum)

df.to_csv(data_dir + 'interim/rush_features.csv')
