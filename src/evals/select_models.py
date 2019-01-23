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

rush_types = ['toLOS', 'tocontact', 'contact']
target_types = ['success', 'EPA', 'yards_aftercont', 'yards']
rush_targets = [(x, y) for y in target_types for x in rush_types]
outfiles = [f for f in os.listdir(data_dir + 'out/')]
rush_targets
# %%
test_results_all = []

for rt in rush_targets[6:7]:
    test_results = []
    print rt
    print 'importing hyperparameter search results'
    out_data = []
    if rt[1] in ['yards', 'EPA', 'yards_aftercont']:
        alg_names = ['Lasso', 'Ridge']
    else:
        alg_names = ['LogisticRegression']
    rush_target = "".join([rt[0], '_', rt[1], '_'])
    file_stems = ["".join([rush_target, a]) for a in alg_names]
    files_to_import = []
    for fs in file_stems:
        files_to_import.extend([f for f in outfiles if f.startswith(fs)])

    if len(files_to_import) == 0:
        continue
    else:
        files = ["".join([data_dir, 'out/', x]) for x in files_to_import]

    for f in files:
        try:
            df = pd.read_csv(f, index_col=0)
            out_data.append(df)
        except:
            pass

    df = pd.concat(out_data, sort=True)

    if 'success' in rush_target:
        score_col = 'mean_test_score'
        classify = True
    else:
        score_col = ['mean_test_neg_mean_squared_error', 'mean_test_r2']
        task = 'regression'
        classify = False

    n_features_min = df['n_features'].unique().min()
    dfns = df[df['n_features'] == n_features_min]
    dfns = dfns.sort_values(score_col, ascending=False)
    dfns_best = dfns.iloc[0, :]
    print 'best cross-validation, no speed features'
    print dfns_best[score_col]

    df = df.sort_values(score_col, ascending=False)
    best = df.iloc[0]
    print 'best cross-validation score all features'
    score_valid = best[score_col]
    print score_valid

    params = ast.literal_eval(best['params'])
    alg = best['alg_name']

    if alg=='LogisticRegression':
        mod = LogisticRegression(penalty=params['penalty'],
                                 C=params['C'],
                                 random_state=0,
                                 n_jobs=1,
                                 solver='liblinear')
    elif alg=='Lasso':
        mod = Lasso(alpha=params['alpha'],
                    random_state=0)
    elif alg=='Ridge':
        mod = Ridge(alpha=params['alpha'],
                    random_state=0)
    else:
        pass

    print 'importing training and test data'
    # import data
    data_dir = 'C:/Users/mworley/nfl_tracking/data/'
    rush_type = rt[0]
    target_type = rt[1]
    dfx = combine_features(rush_type, target_type)
    dfy = pd.read_csv(data_dir + 'interim/targets.csv', index_col=0)

    # select non-missing targets
    dfy = dfy[dfy[target_type].notnull()]

    # select training and test rows
    dfx_train = dfx[dfx['testset'] == 0]
    dfx_test = dfx[dfx['testset'] == 1]
    dfx_train = dfx_train.drop(columns='testset')
    dfx_test = dfx_test.drop(columns='testset')

    # x and y for training set
    mrg_train = pd.merge(dfx_train, dfy, left_index=True, right_index=True,
                         how='inner')
    dfx_train = mrg_train.loc[:, dfx.columns[1:]]
    y_train = mrg_train.loc[:, target_type].values

    # x and y for test set
    mrg_test= pd.merge(dfx_test, dfy, left_index=True, right_index=True,
                         how='inner')
    dfx_test = mrg_train.loc[:, dfx.columns[1:]]
    y_test = mrg_train.loc[:, target_type].values

    # scale features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(dfx_train)
    x_test = scaler.transform(dfx_test)

    print 'fitting model'
    # fit model to full training set
    mod.fit(x_train, y_train)
    filename = model_dir + rush_target + alg + '.sav'
    joblib.dump(mod, filename)

    # predict on test set
    preds = mod.predict(x_test)

    if classify:
        score_test = accuracy_score(y_test, preds)
    else:
        score_test1 = mean_squared_error(y_test, preds)
        score_test2 = r2_score(y_test, preds)
        score_test = [score_test1, score_test2]

    print 'test score: %s' % (score_test)

    if classify:
        test_results = [rush_target, alg, params, dfns_best[score_col], '',
                        best[score_col], '', score_test, '']
    else:
        test_results = [rush_target, alg, params]
        test_results.extend(dfns_best[score_col].values)
        test_results.extend(best[score_col].values)
        test_results.extend(score_test)

    test_results_all.append(test_results)

test_df_columns = ['rush_target', 'alg', 'params', 'score1_valid_ns', 'score2_valid_ns',
                   'score_1_valid', 'score2_valid', 'score1_test', 'score2_test']
test_df = pd.DataFrame(test_results_all, columns=test_df_columns)
test_df = test_df.set_index('rush_target')
write_file('out/summary.csv', test_df)

'''
# all feature names
feature_names = np.array(dfx.columns[1:].tolist()).flatten()

# %%
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

# use selectfrommodel with default threshold
# %%
sfm = SelectFromModel(mod, prefit=True)
x_train_sel = sfm.transform(x_train)
features_sup = sfm.get_support()
features_sel = feature_names[features_sup]

# PCA for feature extraction
n_comps = range(1, 4)
for n in n_comps:
    pca = PCA(n_components=n)
    fit = pca.fit(x_train)
    print("Explained Variance: %s") % fit.explained_variance_ratio_


pca = PCA(n_components=3)
pca.fit(x_train_sel)
x_pca = pca.transform(x_train_sel)
pc = pca.components_
df_pca = pd.DataFrame(x_pca, index=dfx_train.index)
df_loads = pd.DataFrame(pc.T, index=features_sel)
df_pca['target'] = y_train
df_pca.groupby('target').mean()

# %%
# import tracking file
rush_file = data_dir + 'interim/rush_features'
rush = pd.read_csv(rush_file + '.csv', index_col=0)
rush = rush[rush['madeacross'] ==1]
rush = game_play_index(rush)

#%%
compx = []
for n in range(1, 3):
    comp_n = n
    p90 = np.percentile(df_pca[comp_n], 90)
    p10 = np.percentile(df_pca[comp_n], 10)
    df_compx['high'] = np.where(df_compx[comp_n] >= p90, 1, 0)
    df_compx['low'] = np.where(df_compx[comp_n] <= p10, 1, 0)
    df_compx_high = df_compx[df_compx['high'] == 1].index.values
    df_compx_low = df_compx[df_compx['low'] == 1].index.values
    compx.extend([df_compx_high, df_compx_low])
x_list = ['high', 'low'] * 2
c_list = ['comp1'] * 2 + ['comp2'] * 2
xc_list = zip(c_list, x_list)

# %%
# plot speed over time
n = 0
for vals in compx:
    rush_compx = rush[rush.index.isin(vals)]
    fig = plt.figure(figsize=(10, 8))
    for i, d in rush_compx.groupby(rush_compx.index):
        x = d['time_cum']
        y = d['x_fromscrim']
        c = d['s']
        plt.scatter(x = x, y = y,  marker='8', s = 40, c = c,
                    cmap='magma', alpha=0.25)
    plt.title(xc_list[n])
    plt.clim(0,8)
    plt.colorbar()
    plt.ylim(-10, 20)
    plt.xlim(0, 8)
    plt.tight_layout()
    plt.show()
    plt.close()

# plot speed in x y space on field
    rush_compx = rush[rush.index.isin(vals)]
    fig = plt.figure(figsize=(10, 8))
    for i, d in rush_compx.groupby(rush_compx.index):
        x = d['y']
        y = d['x_fromscrim']
        c = d['s']
        ax = plt.scatter(x = x, y = y, marker='8', s = 40, c = c,
                         cmap='magma', alpha=0.25)
    plt.title(xc_list[n])
    plt.clim(0,8)
    plt.colorbar()
    plt.ylim(-10, 40)
    plt.tight_layout()
    plt.show()
    plt.close()
    n += 1
'''
