import pandas as pd
import numpy as np
import os

data_dir = 'C:/Users/mworley/nfl_tracking/data/'
df = pd.read_csv(data_dir + 'interim/rushes.csv', index_col=0)
games = pd.read_csv(data_dir + 'raw/games.csv')

# drop some columns unrelated to rushes
cols_todrop = ['isSTPlay', 'SpecialTeamsPlayType', 'KickReturnYardage',
               'PassLength', 'PassResult', 'YardsAfterCatch']
df = df.drop(columns=cols_todrop)

# merge game data
df = pd.merge(df, games, on='gameId', how='inner')

# create 0/1 indicator columns for snap and handoff events
df['snap'] = (df['event'] == 'ball_snap').apply(int)
df['handoff'] = (df['event'] == 'handoff').apply(int)

# create 0/1 column for "end of play" event
end_events = ['tackle', 'touchdown', 'out_of_bounds', 'fumble']
df['end'] = (df['event'].isin(end_events)).apply(int)

# columns for frame.id of snap, handoff, end of play
df['snap_frame'] = np.where(df['snap'] == 1, df['frame.id'], np.nan)
df['handoff_frame'] = np.where(df['handoff'] == 1, df['frame.id'], np.nan)
df['end_frame'] = np.where(df['end'] == 1, df['frame.id'], np.nan)

# within game play, get first frame of each event type
# transform to add to all rows for each play
df['play_snap_frame'] = df.groupby(['gameId', 'playId'])['snap_frame'].transform('min')
df['play_handoff_frame'] = df.groupby(['gameId', 'playId'])['handoff_frame'].transform('min')
df['play_end_frame'] = df.groupby(['gameId', 'playId'])['end_frame'].transform('min')

# drop single-row indicators
df = df.drop(columns=['snap_frame', 'handoff_frame', 'end_frame'])

# drop frames before snap and after end
df = df[df['frame.id'] >= df['play_snap_frame']]
df = df[df['frame.id'] <= df['play_end_frame']]

# create 0/1 column for if frame is after handoff
df['has_ball'] = np.where(df['frame.id'] >= df['play_handoff_frame'], 1, 0)

# create team abbreviation indicator
df['teamcode'] = np.where(df['team'] == 'home', df['homeTeamAbbr'], df['visitorTeamAbbr'])

# create key to indicate direction of team's offense
# remove rows close to midfield to avoid some directional errors
ckey = df[df['frame.id'] == df['play_snap_frame']]
ckey = ckey[~ckey['x_ball'].between(57, 63)]
ckey = ckey.groupby(['gameId', 'quarter'], as_index=False)[['x_ball',
                                                          'yardlineSide',
                                                          'homeTeamAbbr',
                                                          'yardlineNumber']].first()
ckey = ckey[(ckey['quarter'].isin([1, 2, 3, 4, 5]))]
ckey['x_ball_cmid'] = ckey['x_ball'] - 60

# %%
def set_homedir(row):
    if row['yardlineSide'] == row['homeTeamAbbr']:
        if row['x_ball_cmid'] > 0:
            return 'down'
        else:
            return 'up'
    else:
        if row['x_ball_cmid'] > 0:
            return 'up'
        else:
            return 'down'

ckey['homedir'] = ckey.apply(lambda x: set_homedir(x), axis=1)
ckey = ckey.loc[:, ['gameId', 'quarter', 'homedir']]

df = pd.merge(df, ckey, on=['gameId', 'quarter'], how='inner')

def standard_direction(row, col, mval):
    if row['team'] == 'home':
        if row['homedir'] == 'down':
            x = mval - row[col]
        else:
            x = row[col]
    else:
        if row['homedir'] == 'up':
            x = mval - row[col]
        else:
            x = row[col]
    return x

# set min and max of y
df['y'] = np.where(df['y'] < 0, 0, df['y'])
df['y'] = np.where(df['y'] > 53.3, 53.3, df['y'])

# create standard direction for x, y for man, x for ball
df['xu'] = df.apply(lambda x: standard_direction(x, 'x', 120), axis=1)
df['yu'] = df.apply(lambda x: standard_direction(x, 'y', 53.3), axis=1)
df['xu_ball'] = df.apply(lambda x: standard_direction(x, 'x_ball', 120), axis=1)

# x of previous row
group = ['displayName', 'gameId', 'playId']
shift = lambda x: x.shift(1)

df['xu_lag'] = df.groupby(group)['xu'].apply(shift)
df['xu_ch'] = df['xu'] - df['xu']

# y of previous row
df['yu_lag'] = df.groupby(group)['yu'].apply(shift)
df['yu_ch'] = df['yu'] - df['yu_lag']

# function to get a play_level value and transform across all play rows
def transform_play_value(data, event, var, new_var):
    c = data.loc[data[event] == 1, ['gameId', 'playId', var]]
    c = c.rename(columns={var: new_var})
    new_data = pd.merge(data, c, on=['gameId', 'playId'], how='inner')
    return new_data

df = transform_play_value(df, 'snap', 'xu', 'x_snap')
df = transform_play_value(df, 'snap', 'yu', 'y_snap')
df = transform_play_value(df, 'snap', 'xu_ball', 'xu_ball_snap')

df['xufs'] = df['xu'] - df['x_snap']
df['yufs'] = df['yu'] - df['y_snap']
df['xuscrim'] = df['xu'] - df['xu_ball_snap']

# xu_fs of previous row
df['xufs_lag'] = df.groupby(group)['xufs'].apply(shift)
df['xufs_ch'] = df['xufs'] - df['xufs_lag']

# yu_fs of previous row
df['yufs_lag'] = df.groupby(group)['yufs'].apply(shift)
df['yufs_ch'] = df['yufs'] - df['yufs_lag']

# dir of previous row
df['dir_lag'] = df.groupby(group)['dir'].apply(shift)
df['dir_ch'] = 180 - abs(abs(df['dir'] - df['dir_lag']) - 180)

# cumulative columns
fx_cum = lambda x: x.cumsum()
df['dis_cum'] = df.groupby(group)['dis'].apply(fx_cum)
df['dir_cum'] = df.groupby(group)['dir_ch'].apply(fx_cum)

# drop lagged columns
cols_lag = ['xufs_lag', 'yufs_lag', 'dir_lag']
df = df.drop(columns=cols_lag)

'''
df = transform_play_value(df, 'first_contact', 'frame.id', 'fc_frame')
df['after_contact'] = np.where(df['frame.id'] > df['fc_frame'], 1, 0)
'''
# save file
f = 'interim/rushes_clean.csv'
print 'saving %s' % (f)
df.to_csv(data_dir + f)
