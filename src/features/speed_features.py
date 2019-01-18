import pandas as pd
import numpy as np
from tsfresh import extract_features, select_features

def run():

    print 'running speed_features.py'

    print 'importing data'
    # import data file
    data_dir = 'C:/Users/mworley/nfl_tracking/data/'
    df = pd.read_csv(data_dir + 'interim/rush_selected.csv', index_col=0)

    # create 'game_play' column as unique identifier
    us = pd.Series(['_'] * df.shape[0]).astype(str)
    df['game_play'] = df["gameId"].map(str) + us + df["playId"].map(str)

    # select data for tsfresh feature extraction
    ts = df[['game_play', 'frame_num', 's']].copy()
    ts = ts[ts['s'].notnull()]

    print 'extracting features'
    #features = extract_features(ts, column_id='game_play', column_sort='frame_num')
    # save to temporary file
    f = 'tmp/speed_features_tmp.csv'
    print 'writing %s' % (f)
    #features.to_csv(data_dir + f)
    features = pd.read_csv(data_dir + f, index_col=0)

    feature_count = features.shape[1]
    print '%s features extracted' % (feature_count)

    print 'importing target data'
    # import target data for feature filtering
    t = pd.read_csv(data_dir + 'interim/epa_rush.csv', index_col=0)

    # limit target data to training set
    # this to avoid contaminating test set during feature selection process
    train = df[df['testset'] == 0]
    train_game_plays = train[['gameId', 'playId', 'game_play']].drop_duplicates()
    tsy = pd.merge(t, train_game_plays, on=['gameId', 'playId'], how='inner')
    tsy = tsy.set_index('game_play')

    # %%

    train_features = features.loc[train_game_plays['game_play'], :]
    mrg = pd.merge(features, tsy, left_index=True, right_index=True, how='inner')
    mrg = mrg.sort_index()
    features = mrg[features.columns].copy()

    print 'selecting target variable'
    ycol = 'success'
    y = mrg[ycol].values

    print 'filtering features'
    # %%
    # drop features that have missing values, cannot be used in models
    features = features.dropna(axis=1)
    ff = select_features(features, y, n_jobs=8)

    feature_count = ff.shape[1]
    print '%s features retained' % (feature_count)

    features_all = features[ff.columns]

    features_all = features_all.reset_index().rename(columns={'index': 'game_play'})
    features_all = pd.merge(features_all, game_plays, on='game_play', how='inner')

    #save to file
    f = 'interim/speed_features.csv'
    print 'writing %s' % (f)
    features_all.to_csv(data_dir + f)

if __name__ == '__main__':
    run()
