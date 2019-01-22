import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, KFold
import ast
import sys
sys.path.insert(0, 'c:/users/mworley/nfl_tracking/src/features')
from feature_functions import combine_features

data_dir = 'C:/Users/mworley/nfl_tracking/data/'
out_dir = data_dir + 'out/'

rush_types = ['toLOS', 'tocontact', 'contact']
target_types = ['success', 'EPA', 'yards_aftercont', 'yards']
rush_targets = [(x, y) for y in target_types for x in rush_types]
outfiles = [f for f in os.listdir(data_dir + 'out/')]

# %%

for rt in rush_targets[0:1]:
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

    df = pd.concat(out_data)


df = df.sort_values('mean_test_score', ascending=False)
best = df.iloc[0]
print best['mean_test_score']
params = ast.literal_eval(best['params'])
alg = best['alg_name']
if alg=='LogisticRegression':
    mod = LogisticRegression(penalty=params['penalty'],
                             C=params['C'],
                             random_state=0,
                             n_jobs=1)
else:
    pass


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

# fit model to full training set
mod.fit(x_train, y_train)

# predict on test set
preds = mod.predict(x_test)
accuracy_score(y_test, preds)

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

# transform training data using feature selector
shuffle =  StratifiedKFold(n_splits=5, shuffle=True, random_state=42041)
score = cross_val_score(mod, x_train_sel, y_train, scoring='accuracy',
                        cv=shuffle) #, n_jobs=-1)
print score

sfm_matnp.vstack((np.array(feature_names)[x_sup], x_sfm)).T
[dfx.columns[1:][x_sup], x_sfm]
n_features = x_sfm.shape[1]

x_sfm
