# import packages
import pandas as pd
import numpy as np
from math import exp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy import stats
import sys

sys.path.insert(0, 'c:/users/mworley/nfl_tracking/src/features')
from feature_functions import combine_features

def random_search_cv(rush_type, target_type, n_iters = 10):
    print ''
    print ''
    print "".join(['rush_', rush_type, '_', target_type])
    print ''
    print ''
    # import data
    print 'importing data'

    try:
        dfx = combine_features(rush_type, target_type)
    except:
        print 'invalid feature-target pair'
        return

    data_dir = 'C:/Users/mworley/nfl_tracking/data/'
    dfy = pd.read_csv(data_dir + 'interim/targets.csv', index_col=0)

    # select training rows
    dfx = dfx[dfx['testset'] == 0]
    dfx = dfx.drop(columns='testset')

    # select non-missing targets aligned with feature training rows
    dfy = dfy[dfy[target_type].notnull()]

    if dfy.shape[0] < dfx.shape[0]:
        dfx = dfx.loc[dfy.index.values]
    elif dfy.shape[0] > dfx.shape[0]:
        dfy = dfy.loc[dfx.index.values]
    else:
        pass

    y = dfy[target_type].values

    # get dataset without speed features
    cnames = dfx.columns.tolist()
    speed_features = [c for c in cnames if 's__' in c]
    speed_first = dfx.columns.tolist().index(speed_features[0])
    xns = dfx[dfx.columns[0:speed_first]]

    # %%
    print 'scaling features'
    scaler = StandardScaler()
    xns = scaler.fit_transform(xns)
    x = scaler.fit_transform(dfx)

    # identify model, based on target

    if len(set(y)) == 2:
        alg = LogisticRegression(random_state=0)
        algs = [alg]
        task = 'classification'
        shuffle = StratifiedKFold(n_splits=5, shuffle=True, random_state=42041)

    else:
        alg1 = Lasso(max_iter=100000, random_state=0)
        alg2 = Ridge(max_iter=1000, random_state=0)
        algs = [alg1, alg2]
        task = 'regression'
        shuffle = KFold(n_splits=5, shuffle=True, random_state=42041)

    feature_frames = [xns, x]

    for alg in algs:

        random_iters = n_iters

        alg_name = str(alg).split('(')[0]

        results_frames = []

        n = 0

        for x_train in feature_frames:

            feature_count = x_train.shape[1]
            print 'feature set %s' % (n)
            print '%s features'% (feature_count)

            param2 = [0.001, 0.01, 0.1,1, 10]

            if task == 'classification':
                param1 = ['l1', 'l2']
                params_grid = {'penalty': param1,
                               'C': param2}
                score = 'accuracy'

            else:
                if alg_name == 'Lasso':
                    param2 = param2[1:]
                    param2.append(50)
                params_grid = {'alpha': param2}
                score = ['neg_mean_squared_error', 'r2']

            cv_grid = GridSearchCV(alg, params_grid,
                                   cv=shuffle,
                                   scoring=score,
                                   n_jobs=-1,
                                   refit=score[0])

            print 'fitting grid search for %s' % (alg_name)
            cv_grid.fit(x_train, y)

            print 'grid search best score'
            print cv_grid.best_score_

            results_grid = pd.DataFrame(cv_grid.cv_results_)

            if task != 'classification':
                sort_col = 'mean_test_' + score[0]
            else:
                sort_col = 'mean_test_score'

            results_grid.sort_values(sort_col,
                                     ascending=False,
                                     inplace=True)
            best_grid = results_grid.head(6)

            # set up random search

            if task == 'classification':
                param1 = best_grid['param_penalty'].unique().tolist()
                param2 = best_grid['param_C'].unique().tolist()
            else:
                param2 = best_grid['param_alpha'].unique().tolist()

            param2_range= [np.min(param2), np.max(param2)]
            param2_min = np.log(param2_range[0])
            param2_max = np.log(param2_range[1])
            param2_values = np.random.uniform(param2_min, param2_max, random_iters)
            param2_values = map(lambda x: round(exp(x), 5), param2_values)
            param2_values.extend(param2)
            random_iters += len(param2)

            if task == 'classification':
                params_random= {'penalty': param1,
                                'C': param2_values}

            else:
                params_random = {'alpha': param2_values}

            cv_random = RandomizedSearchCV(alg, params_random,
                                           n_iter=random_iters,
                                           scoring=score,
                                           cv=shuffle,
                                           n_jobs=-1,
                                           verbose=1,
                                           refit=score[0])

            print 'fitting random search for %s' % (alg_name)
            cv_random.fit(x_train, y)

            print 'random search best score'
            print cv_random.best_score_


            results_random = pd.DataFrame(cv_random.cv_results_)
            cols = results_random.columns
            cols_tokeep = [c for c in cols if 'param' in c]
            score_cols = [c for c in cols if 'mean' in c]
            cols_tokeep.extend(score_cols)
            results_random = results_random[cols_tokeep]
            results_random.sort_values(sort_col,
                                       ascending=False,
                                       inplace=True)
            results_random['alg_name'] = str(alg).split('(')[0]
            results_random['n_features'] = x_train.shape[1]
            results_frames.append(results_random)
            n += 1

        results = pd.concat(results_frames).reset_index(drop=True)

        f = 'out/' + rush_type + '_' + target_type + '_' + alg_name + '.csv'

        write_file(f, results)

rush_types = ['toLOS', 'contact', 'tocontact']
target_types = ['success', 'EPA', 'yards_aftercont', 'yards']
rush_targets = [(x, y) for x in rush_types for y in target_types]

if __name__ == '__main__':
    for rt in rush_targets:
        try:
            random_search_cv(rt[0], rt[1], 5)
        except:
            pass
