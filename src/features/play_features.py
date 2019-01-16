import pandas as pd
import numpy as np

data_dir = 'C:/Users/mworley/nfl_tracking/data/'
df = pd.read_csv(data_dir + 'raw/plays.csv')

# get list of relevant plays
df_plays = pd.read_csv(data_dir + 'interim/rush_features.csv', index_col=0)
game_plays = df_plays.loc[:, ['gameId', 'playId']].drop_duplicates()
mrg = pd.merge(df, game_plays, on=['gameId', 'playId'], how='inner')

# clean and code offense formations
mrg['offenseFormation'].fillna('WILDCAT', inplace=True)
formations = mrg['offenseFormation'].drop_duplicates().values
formations_shot = ['PISTOL', 'SHOTGUN']
formations_single = ['ACE', 'EMPTY', 'SINGLEBACK', 'WILDCAT']
mrg['form_shotgun'] = (mrg['offenseFormation'].isin(formations_shot)).astype(int)
mrg['form_single'] = mrg['offenseFormation'].isin(formations_single).astype(int)

# dummy code down
mrg['down_1st'] = (mrg['down'] == 1).astype(int)
mrg['down_2nd'] = (mrg['down'] == 2).astype(int)

cols_tokeep = ['gameId', 'playId', 'down', 'yardsToGo', 'defendersInTheBox',
               'form_shotgun', 'form_single', 'down_1st', 'down_2nd']
play_features = mrg.loc[:, cols_tokeep]

# save to file
f = 'interim/play_features.csv'
print 'writing %s' % (f)
play_features.to_csv(data_dir + f)
