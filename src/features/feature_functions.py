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
