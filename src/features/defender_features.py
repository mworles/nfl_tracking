import pandas as pd
import sys

data_dir = 'C:/Users/mworley/nfl_tracking/data/'
df = pd.read_csv(data_dir + 'interim/defenders.csv', index_col=0)

gbc = ['gameId', 'playId', 'frame.id']
df['dfndis_min'] = df.groupby(gbc)['distance'].transform('min')
df = df[['playId', 'gameId', 'frame.id', 'dfndis_min']]
df = df.drop_duplicates()

# cumulative minimum defender distance
df['dfndis_mincum'] = df.groupby(['gameId', 'playId'])['dfndis_min'].transform('cummin')

# indicator of whether defender within one yard or less
df['dfn_in1'] = (df['dfndis_min'] <= 1).astype(int)
df['dfn_in1_yet'] = df.groupby(['gameId', 'playId'])['dfn_in1'].transform('cummax')

# write data file
f = 'interim/defense_features.csv'
print 'saving %s' % (f)
df.to_csv(data_dir + f)
