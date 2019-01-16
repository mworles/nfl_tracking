import pandas as pd

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

df[df['dfndis_min'] < 1.2].head(50)


'''
smax_first5 = df[df['dis_cum'] <=5].groupby(['gameId', 'playId'], as_index=False)['s'].max()
smax_first5.rename(columns={'s': 'smax_first5'}, inplace=True)
df = pd.merge(df, smax_first5, on=['gameId', 'playId'], how='inner')
'''

f = 'interim/defense_features.csv'
print 'saving %s' % (f)
df.to_csv(data_dir + f)
