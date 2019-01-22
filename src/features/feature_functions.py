import pandas as pd
import numpy as np

def combine_game_play(row):
    game_play = str(int(row['gameId'])) + '_' + str(int(row['playId']))
    return game_play

def game_play_index(df):
    """Create a game play index for dataframe."""
    game_play = df.apply(lambda x: combine_game_play(x), axis=1)
    df = df.set_index(game_play)
    df.index.name = 'game_play'
    df = df.drop(columns=['gameId', 'playId'])
    return df

def write_file(file, df):
    data_dir = 'C:/Users/mworley/nfl_tracking/data/'
    path = data_dir + file
    print 'writing %s' % (path)
    df.to_csv(path)

def combine_features(rush_type, target_type):
    data_dir = 'C:/Users/mworley/nfl_tracking/data/'
    rush_file = data_dir + 'interim/rush_' + rush_type
    rush = pd.read_csv(rush_file + '.csv', index_col=0)
    play = pd.read_csv(data_dir + 'interim/play_features.csv', index_col=0)
    speed_file = "".join([data_dir, 'interim/speed_rush_', rush_type, '_',
                          target_type, '.csv'])
    speed = pd.read_csv(speed_file, index_col=0)
    rush_cols = ['testset', 'early_con', 'dfndis_LOS']
    rush = rush[rush_cols]

    x1 = pd.merge(rush, play, left_index=True, right_index=True, how='inner')
    x2 = pd.merge(x1, speed, left_index=True, right_index=True, how='inner')
    x3 = x2.dropna()

    x3 = x3[~x3.index.duplicated(keep='first')]

    return x3
