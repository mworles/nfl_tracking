import pandas as pd
import numpy as np

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

# code only first event as positive, some plays coded multiple snaps
df['snapcnt']  = df.groupby(['gameId', 'playId'])['snap'].transform('cumsum')
df['snap'] = np.where(df['snapcnt'] < 2, df['snap'], 0)
df = df.drop(columns=['snapcnt'])

# create 0/1 column for "end of play" event
end_events = ['tackle', 'touchdown', 'out_of_bounds', 'fumble']
df['end'] = (df['event'].isin(end_events)).apply(int)

# columns for frame.id of snap, handoff, end of play
df['snap_frame'] = np.where(df['snap'] == 1, df['frame.id'], np.nan)
df['handoff_frame'] = np.where(df['handoff'] == 1, df['frame.id'], np.nan)
df['end_frame'] = np.where(df['end'] == 1, df['frame.id'], np.nan)

# within game play, get first frame of each event type
# transform to add to all rows for each play
df['snap_frame'] = df.groupby(['gameId', 'playId'])['snap_frame'].transform('min')
df['handoff_frame'] = df.groupby(['gameId', 'playId'])['handoff_frame'].transform('min')
df['end_frame'] = df.groupby(['gameId', 'playId'])['end_frame'].transform('min')

# drop frames before snap and after end
df = df[df['frame.id'] >= df['snap_frame']]
df = df[df['frame.id'] <= df['end_frame']]

# create 0/1 column for if frame is after handoff
df['has_ball'] = np.where(df['frame.id'] >= df['handoff_frame'], 1, 0)

# create team abbreviation indicator
df['teamcode'] = np.where(df['team'] == 'home', df['homeTeamAbbr'], df['visitorTeamAbbr'])

# create key to indicate direction of team's offense
# remove rows close to midfield to avoid some directional errors
ckey = df[df['frame.id'] == df['snap_frame']]
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

# set columns to keep
cols_tokeep = ['gameId', 'playId', 'frame.id', 'event', 'nflId', 'displayName',
               'xu', 'yu', 'dis', 'dir', 'xu_ball', 'y_ball', 's_ball', 'dis_ball', 'dir_ball',
               'team', 'PlayResult', 'snap_frame', 'handoff_frame', 'end_frame',
                'has_ball', 'teamcode', 'playDescription']
df = df[cols_tokeep].copy()
df = df.rename(columns={'xu': 'x', 'yu': 'y', 'xu_ball': 'x_ball'})

# save file
f = 'interim/rushes_clean.csv'
print 'saving %s' % (f)
df.to_csv(data_dir + f)
