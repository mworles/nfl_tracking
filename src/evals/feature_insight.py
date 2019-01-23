import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.externals import joblib
import ast
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, 'c:/users/mworley/nfl_tracking/src/features')
from feature_functions import combine_features, game_play_index, write_file

data_dir = 'C:/Users/mworley/nfl_tracking/data/'
out_dir = data_dir + 'out/'
model_dir = 'C:/Users/mworley/nfl_tracking/models/'
plot_dir = 'C:/Users/mworley/nfl_tracking/plots/'
modfiles = [m for m in os.listdir(model_dir)]

# import full tracking file
print 'importing full tracking data'
rush_file = data_dir + 'interim/rush_features'
rush = pd.read_csv(rush_file + '.csv', index_col=0)
rush = rush[rush['madeacross'] ==1]
rush = game_play_index(rush)


rush_types = ['toLOS', 'tocontact', 'contact']
target_types = ['success', 'EPA', 'yards_aftercont', 'yards']
rush_targets = [(x, y) for y in target_types for x in rush_types]

rush_targets
# %%
test_results_all = []

for rt in rush_targets:
    if rt[1] in ['yards', 'EPA', 'yards_aftercont']:
        alg_names = ['Lasso', 'Ridge']
    else:
        alg_names = ['LogisticRegression']
    rush_target = "".join([rt[0], '_', rt[1], '_'])
    file_stems = ["".join([rush_target, a]) for a in alg_names]
    files_to_import = []
    for fs in file_stems:
        files_to_import.extend([f for f in modfiles if f.startswith(fs)])
    if len(files_to_import) == 1:
        modfile = model_dir + files_to_import[0]
    else:
        print 'more than one model file'
        continue

    print 'importing training data'
    # import data
    data_dir = 'C:/Users/mworley/nfl_tracking/data/'
    rush_type = rt[0]
    target_type = rt[1]
    dfx = combine_features(rush_type, target_type)
    dfy = pd.read_csv(data_dir + 'interim/targets.csv', index_col=0)

    # select non-missing targets
    dfy = dfy[dfy[target_type].notnull()]

    # select training set rows
    dfx_train = dfx[dfx['testset'] == 0]
    dfx_train = dfx_train.drop(columns='testset')

    # x and y for training set
    mrg_train = pd.merge(dfx_train, dfy, left_index=True, right_index=True,
                         how='inner')
    dfx_train = mrg_train.loc[:, dfx.columns[1:]]
    y_train = mrg_train.loc[:, target_type].values

    # scale x
    scaler = StandardScaler()
    x_train = scaler.fit_transform(dfx_train)

    # all feature names
    feature_names = np.array(dfx_train.columns.tolist()).flatten()

    # import model
    mod = joblib.load(modfile)

    print 'reducing features with SelectFromModel'
    # use selectfrommodel with default threshold
    # remove irrelevant features
    sfm = SelectFromModel(mod, prefit=True)
    x_train_sel = sfm.transform(x_train)
    features_sup = sfm.get_support()
    features_sel = feature_names[features_sup]

    # PCA for feature dimension reduction
    '''
    n_comp_max = 4
    n_comps = range(1, n_comp_max)
    for n in n_comps:
        pca = PCA(n_components=n)
        fit = pca.fit(x_train_sel)
        print("Explained Variance: %s") % fit.explained_variance_ratio_
    '''
    print 'running PCA'
    n_comp = 3
    pca = PCA(n_components=n_comp)
    pca.fit(x_train_sel)
    pca_var = map(lambda x: round(x*100, 0), pca.explained_variance_ratio_)
    x_pca = pca.transform(x_train_sel)
    df_pca = pd.DataFrame(x_pca, index=dfx_train.index)

    #
    print 'identifying extreme rows on PCA components'
    df_compx = df_pca.copy()
    compx = []
    for n in range(0, n_comp):
        comp_n = n
        p90 = np.percentile(df_compx[comp_n], 90)
        p10 = np.percentile(df_compx[comp_n], 10)
        df_compx['high'] = np.where(df_compx[comp_n] >= p90, 1, 0)
        df_compx['low'] = np.where(df_compx[comp_n] <= p10, 1, 0)
        df_compx_high = df_compx[df_compx['high'] == 1].index.values
        df_compx_low = df_compx[df_compx['low'] == 1].index.values
        compx.extend([df_compx_high, df_compx_low])
    x_list = ['high', 'low'] * 3
    c_list = ['comp1'] * 2 + ['comp2'] * 2 + ['comp3'] * 2
    var_list = [pca_var[0]] * 2 + [pca_var[1]] * 2 + [pca_var[2]] * 2
    xc_list = zip(c_list, x_list, var_list)

    print 'creating plots for PCA components'
    # plot speed over time
    n = 0
    for vals in compx:
        rush_compx = rush[rush.index.isin(vals)]
        hl = xc_list[n][0]
        pc = xc_list[n][1]
        var = str(xc_list[n][2])
        fig = plt.figure(figsize=(10, 8))
        for i, d in rush_compx.groupby(rush_compx.index):
            x = d['time_cum']
            y = d['x_fromscrim']
            c = d['s']
            plt.scatter(x = x, y = y,  marker='8', s = 40, c = c,
                        cmap='magma', alpha=0.25)
        title = "".join([rush_target, hl, '_', pc, '_', var])
        plt.title(title)
        plt.clim(0,8)
        plt.colorbar()
        plt.ylim(-10, 20)
        plt.xlim(0, 8)
        plt.tight_layout()
        #plt.show()
        #plt.close()
        fig.savefig(plot_dir + 'pca/' + title + '_speed.jpg')

    # plot speed in x y space on field
        rush_compx = rush[rush.index.isin(vals)]
        hl = xc_list[n][0]
        pc = xc_list[n][1]
        fig = plt.figure(figsize=(10, 8))
        for i, d in rush_compx.groupby(rush_compx.index):
            x = d['y']
            y = d['x_fromscrim']
            c = d['s']
            ax = plt.scatter(x = x, y = y, marker='8', s = 40, c = c,
                             cmap='magma', alpha=0.25)
        plt.title(title)
        plt.clim(0,8)
        plt.colorbar()
        plt.ylim(-10, 40)
        plt.tight_layout()
        #plt.show()
        #plt.close()
        fig.savefig(plot_dir + 'pca/' + title + '_field.jpg' )
        n += 1

    '''
    sfm = SelectFromModel(mod, prefit=True, threshold=0.50)
    n_features = sfm.transform(x_train).shape[1]
    while n_features > 5:
        sfm.threshold += 0.1
        x_train_sel = sfm.transform(x_train)
        n_features = x_train_sel.shape[1]
        print n_features

    features_sup = sfm.get_support()
    features_sel = feature_names[features_sup]
    print features_sel

    # cross validate score using selected features
    shuffle =  StratifiedKFold(n_splits=5, shuffle=True, random_state=42041)
    scores = cross_val_score(mod, x_train_sel, y_train, scoring='accuracy',
                            cv=shuffle) #, n_jobs=-1)
    print np.mean(scores)

    dfx_train_sel = dfx_train.loc[:, features_sel]
    dfx_train_sel['target'] = y_train
    dfx_train_sel.groupby('target').mean()
    '''
