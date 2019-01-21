import pandas as pd
from tsfresh import extract_features, select_features

def extract_speed_features(rush_file):

    print 'importing data for speed feature extraction'

    # import data file
    data_dir = 'C:/Users/mworley/nfl_tracking/data/'
    #rush_file = data_dir + 'interim/' + rush_file + 'csv'
    rush_file_frames = data_dir + 'interim/' + rush_file + '_frames.csv'
    print 'importing %s' % (rush_file_frames)
    df = pd.read_csv(rush_file_frames, index_col=0).reset_index(drop=True)

    # create 'game_play' column as unique identifier
    us = pd.Series(['_'] * df.shape[0]).astype(str)
    df['game_play'] = df["gameId"].map(str) + us + df["playId"].map(str)

    # select data for tsfresh feature extraction
    ts = df[['game_play', 'frame_num', 's']].copy()
    ts = ts[ts['s'].notnull()]

    print 'extracting features'
    features = extract_features(ts, column_id='game_play',
                                column_sort='frame_num')

    feature_count = features.shape[1]
    print '%s features extracted' % (feature_count)

    # save to temporary file
    f = 'tmp/' + rush_file + '_speed_features.csv'
    print 'writing %s' % (f)
    features.to_csv(data_dir + f)

def filter_speed_features(rush_file, target):
    data_dir = 'C:/Users/mworley/nfl_tracking/data/'
    feature_file = data_dir + 'tmp/' + rush_file + '_speed_features.csv'
    rushes = data_dir + 'interim/' + rush_file + '.csv'
    features = pd.read_csv(feature_file, index_col=0)
    targets = pd.read_csv(data_dir + 'interim/targets.csv', index_col=0)
    df = pd.read_csv(rushes, index_col=0)

    print 'converting length-dependent features'

    x = 's'
    to_convert = ['abs_energy',
                  'absolute_sum_of_changes',
                  'count_above_mean',
                  'count_below_mean',
                  'sum_of_reoccurring_data_points',
                  'sum_of_reoccurring_values',
                  'sum_values']
    to_convert = [x + '__' + c for c in to_convert]
    length_col = x + '__length'

    for c in to_convert:
        features[c] = features[c] / features[length_col]

    # identify list of game plays from train set to filter features and targets
    train = df[df['testset'] == 0]
    game_plays = train.index.values

    print '%s training set rows selected' % (len(game_plays))
    # filter rows to include only training set
    #train_targets = targets.loc[game_plays, :]
    train_features = features.loc[game_plays, :]

    # drop features that have missing values, cannot be used in models
    print 'dropping rows missing all features'
    print 'from %s rows' % (train_features.shape[0])
    train_features = train_features.dropna(how='all')
    print '%s rows with feature data retained' % (train_features.shape[0])

    print 'dropping features with missing values'
    print 'from %s extracted features' % (train_features.shape[1])
    train_features = train_features.dropna(axis=1)
    print '%s features with full data retained' % (train_features.shape[1])

    targets_clean = targets[targets[target].notnull()]

    # merge to align indexes
    mrg = pd.merge(train_features, targets_clean,
                   left_index=True, right_index=True,
                   how='inner')
    mrg = mrg.sort_index()

    print '%s training set rows with features and targets' % (mrg.shape[0])
    features_clean = mrg[train_features.columns].copy()

    print 'selecting target variable'
    y = mrg[target].values

    print 'filtering features using %s' % (target)
    features_filtered = select_features(features_clean, y, n_jobs=8)

    feature_count = features_filtered.shape[1]
    print '%s features retained' % (feature_count)

    # get training and test rows, use filtered feature columns to select
    features = features[features_filtered.columns]

    #save to file
    f = 'interim/speed_' + rush_file + '_' + target + '.csv'
    print 'writing %s' % (f)
    features_filtered.to_csv(data_dir + f)
