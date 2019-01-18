import pandas as pd

# import data file
data_dir = 'C:/Users/mworley/nfl_tracking/data/'
df = pd.read_csv(data_dir + 'interim/rush_features.csv', index_col=0)

# select runs where rusher crossed line of scrimmage
df = df[df['madeacross'] == 1]

# select frames occurring before crossing line of scrimmage
df = df[df['hascrossed'] == 0]
df = df.reset_index()

game_ids = df['gameId'].unique()
traincut_game = game_ids[int(len(game_ids) * 0.80)]
df['testset'] = (df['gameId'] > traincut_game).astype(int)

# save file
f = 'interim/rush_selected.csv'
print 'writing %s' % (f)
df.to_csv(data_dir + f)
