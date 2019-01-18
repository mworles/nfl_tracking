import pandas as pd
import sys
sys.path.insert(0, 'c:/users/mworley/nfl_tracking/src/features')
from feature_functions import *


# %%# get plays to use to filter scrapR data
data_dir = 'C:/Users/mworley/nfl_tracking/data/'
rd = pd.read_csv(data_dir + 'interim/rushes_clean.csv', index_col=0,
                 low_memory=False)
plays = rd[['gameId', 'playId']].copy()
plays = plays.drop_duplicates()

# %%# import scrapR data
df = pd.read_csv("C:/Users/mworley/nfl_tracking/data/raw/scrapR2017.csv")
dfs = df[df['RushAttempt'] == 1]
dfs = dfs.rename(columns={'GameID': 'gameId', 'play_id': 'playId'})
mrg = pd.merge(plays, dfs, on=['gameId', 'playId'], how='inner')

# %%
def sucrate(row):
    sucrate_neut = {1: .40, 2: 0.60, 3: 1.0, 4: 1.0}
    sucrate_down = {1: .50, 2: 0.65, 3: 1.0, 4: 1.0}
    sucrate_up = {1: .30, 2: 0.50, 3: 1.0, 4: 1.0}

    if row['qtr'] != 4:
        return sucrate_neut[row['down']]
    elif row['qtr'] < -7:
        return sucrate_down[row['down']]
    elif row['qtr'] > 0:
        return sucrate_up[row['down']]
    else:
        return sucrate_neut[row['down']]

# %%
mrg['sucrate_bar'] = mrg.apply(lambda x: sucrate(x), axis=1) * mrg['ydstogo']
mrg['success'] = (mrg['Yards.Gained'] >= mrg['sucrate_bar']).astype(int)
cols_tokeep = ['gameId', 'playId', 'Rusher', 'Rusher_ID',
               'RunLocation', 'RunGap', 'Tackler1', 'Tackler2',
               'ExpPts', 'EPA', 'Win_Prob', 'WPA', 'success']

mrg_keep = mrg[cols_tokeep]

dfout = game_play_index(mrg_keep)

write_file('interim/epa_rush.csv', dfout)
