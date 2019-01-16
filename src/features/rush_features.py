import pandas as pd
import numpy as np

data_dir = 'C:/Users/mworley/nfl_tracking/data/'
df = pd.read_csv(data_dir + 'interim/rushes_clean.csv', index_col=0)

# verify no frames are duplicated
duplicate_frames = df[df[['gameId', 'playId', 'frame.id']].duplicated()]

print "duplicate frames?"
if duplicate_frames.shape[0] == 0:
    print "No"
else:
    print "Yes"

# create list to group frames by game and play
group = ['gameId', 'playId']
shift = lambda x: x.shift(1)

df['x_diff'] = df.groupby(group)['x'].diff()
df['y_diff'] = df.groupby(group)['y'].diff()

# create columns for difference between value and value at snap
df['x_fromsnap'] = df['x'] - df.groupby(group)['x'].transform('first')
df['y_fromsnap'] = df['y'] - df.groupby(group)['y'].transform('first')
df['x_fromscrim'] = df['x'] - df.groupby(group)['x_ball'].transform('first')

df['x_fromscrim_min'] =  df.groupby(group)['x_fromscrim'].transform('min')

# drop some plays with errors in ball/scrimmage location
df = df[df['x_fromscrim_min'] > - 11.0]
df = df.drop(columns=['x_fromscrim_min'])

# diff from previous row
df['x_fromsnap_diff'] = df.groupby(group)['x_fromsnap'].diff()
df['y_fromsnap_diff'] = df.groupby(group)['y_fromsnap'].diff()
df['x_fromscrim_diff'] = df.groupby(group)['x_fromscrim'].diff()

# dir of previous row
df['dir_lag'] = df.groupby(group)['dir'].apply(shift)
df['dir_diff'] = 180 - abs(abs(df['dir'] - df['dir_lag']) - 180)
df = df.drop(columns=['dir_lag'])

# cumulative columns
fx_cum = lambda x: x.cumsum()
df['dis_cum'] = df.groupby(group)['dis'].apply(fx_cum)
df['dir_cum'] = df.groupby(group)['dir_diff'].apply(fx_cum)

# %%
f = 'interim/rush_features.csv'
print 'saving %s' % (f)
df.to_csv(data_dir + f)


'''
# script for additional rush features
#
f = 'tmp/rush_features_tmp.csv'
df = pd.read_csv(data_dir + f, index_col=0)

#
df['first_contact'] = np.where(df['event'] == 'first_contact', 1, 0)
df['after_contact'] = df.groupby(group)['first_contact'].transform('cummax')

# compute peak speed before/after contact
dfbc = df[df['after_contact'] == 0].groupby(['gameId', 'playId'], as_index=False)
s_max_bc = dfbc['s'].apply(max)
s_max_bc = s_max_bc.reset_index()
s_max_bc.columns=['gameId', 'playId', 's_max_bc']
df = pd.merge(df, s_max_bc, on=['gameId', 'playId'], how='inner')
df['s_max_bc_loss'] = df['s'] - df['s_max_bc']

# x position at first contact
df['x_atcontact'] = np.where(df['event'] == 'first_contact', df['xu'], np.nan)
df['x_atcontact'] = df.groupby(['gameId', 'playId'])['x_atcontact'].transform('max')
df['xufs_atcontact'] = df['x_atcontact'] - df['x_snap']
df['xuscrim_atcontact'] = df['x_atcontact'] - df['xu_ball_snap']
df['xu_fromcontact'] = df['xu'] - df['x_atcontact']
df['xu_aftercontact'] = df.groupby(['gameId', 'playId'])['xu_fromcontact'].transform('max')

# frame at contact
df['time'] = 0.10
df['time_cum'] = df.groupby(['gameId', 'playId'])['time'].transform('cumsum')
df['time_atcontact'] = np.where(df['event'] == 'first_contact', df['time_cum'], np.nan)
df['time_atcontact'] = df.groupby(['gameId', 'playId'])['time_atcontact'].transform('max')
df['time_fromcontact'] = df['time_cum'] - df['time_atcontact']
'''
