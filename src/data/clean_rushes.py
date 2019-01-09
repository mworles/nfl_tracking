import pandas as pd
import numpy as np
import os


data_dir = 'C:/Users/mworley/nflbdb/Data/'
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
# transform to add to all forw for each play
df['play_snap_frame'] = df.groupby(['gameId', 'playId'])['snap_frame'].transform('min')
df['play_handoff_frame'] = df.groupby(['gameId', 'playId'])['handoff_frame'].transform('min')
df['play_end_frame'] = df.groupby(['gameId', 'playId'])['end_frame'].transform('min')

# drop frames before snap and after end
df = df[df['frame.id'] >= df['play_snap_frame']]
df = df[df['frame.id'] <= df['play_end_frame']]

# create 0/1 column for if frame is after handoff
df['has_ball'] = np.where(df['frame.id'] >= df['play_handoff_frame'], 1, 0)

# create team abbreviation indicator
df['teamcode'] = np.where(df['team'] == 'home', df['homeTeamAbbr'], df['visitorTeamAbbr'])

# create key to indicate direction of team's offense
ckey = df.groupby(['gameId', 'quarter'], as_index=False)[['x_ball',
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

df['x_std'] = df.apply(lambda x: standard_direction(x, 'x', 120), axis=1)

# set min and max of y
df['y'] = np.where(df['y'] < 0, 0, df['y'])
df['y'] = np.where(df['y'] > 53.3, 53.3, df['y'])

# create standard direction y column
df['y_std'] = df.apply(lambda x: standard_direction(x, 'y', 53.3), axis=1)

# x of previous row
df['x_std_lag'] = df.groupby(['displayName', 'gameId', 'playId'])['x_std'].apply(lambda y: y.shift(1))
df['x_std_ch'] = df['x_std'] - df['x_std_lag']


df.head()

# save temporary file
df.to_csv(data_dir + 'interim/rushes_std.csv')
