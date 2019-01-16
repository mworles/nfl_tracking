import pandas as pd
import numpy as np
import math

def calc_dist(row):
    x1 = row['x']
    x2 = row['x_def']
    y1 = row['y']
    y2 = row['y_def']
    dist = math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
    return dist

def get_defenders(gid):

    # import offense play tracking data
    print 'importing offense tracking game %s' % (gid)
    g = df[df['gameId'] == gid]

    # keep necessary columns
    g = g.loc[:, ['playId', 'x', 'y', 'frame.id', 'nflId', 'displayName',
                  'team']]

    # identify and import all tracking data for game
    print 'importing all tracking game %s' % (gid)
    f = data_dir + 'raw/tracking_gameId_' + str(gid) + '.csv'
    t = pd.read_csv(f)

    # remove football tracking
    t = t[t['displayName'] != 'football']

    # get list of relevant plays from game
    plays = g['playId'].drop_duplicates().values
    # use list to subset the tracking data
    t = t[t['playId'].isin(plays)]

    # create df of offensive team for each play
    off = g[['playId', 'team']].drop_duplicates()
    off.columns = ['playId', 'team_off']

    # merge tracking data with offensive team
    d = pd.merge(t, off, on='playId', how='inner')

    # defense is all tracking rows with team != offense team
    d = d[d['team'] != d['team_off']]

    # identify columns to keep for the defense
    dcols = ['gameId', 'x', 'y', 'nflId', 'displayName', 'frame.id', 'playId']
    d = d[dcols]

    # rename defense columns and merge with offense play tracking
    d.columns = ['gameId', 'x_def', 'y_def', 'nflId_def', 'name_def', 'frame.id', 'playId']
    print 'merging defense tracking with offense'
    mrg = pd.merge(g, d, on=['playId', 'frame.id'], how='inner')

    # calculate distances using calc_dist function
    print 'calculating distance'
    mrg['distance'] = mrg.apply(lambda x: calc_dist(x), axis=1)

    print 'distance computation complete'
    return mrg

data_dir = 'C:/Users/mworley/nfl_tracking/data/'
df = pd.read_csv(data_dir + 'interim/rush_features.csv', index_col=0)
gids = df['gameId'].drop_duplicates().values

ds = map(lambda x: get_defenders(x), gids)
df_new = pd.concat(ds)
df_new.to_csv(data_dir + 'interim/defenders.csv')
