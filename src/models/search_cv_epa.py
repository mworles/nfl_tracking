# import packages
import pandas as pd
import numpy as np
from math import exp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy import stats
import sys
sys.path.insert(0, 'c:/users/mworley/nfl_tracking/src/features')
from feature_functions import *

def run():

    # import data
    print 'importing data'
    data_dir = 'C:/Users/mworley/nfl_tracking/data/'

    dfx = pd.read_csv(data_dir + 'processed/features.csv', index_col=0)
    dfy = pd.read_csv(data_dir + 'processed/targets.csv', index_col=0)

    # select training rows
    dfx = dfx[dfx['testset'] == 0]
    dfy = dfy.loc[dfx.index.values, :]

    # get dataset without speed features
    cnames = dfx.columns.tolist()
    speed_features = [c for c in cnames if 's__' in c]
    speed_first = dfx.columns.tolist().index(speed_features[0])
    xns = dfx[dfx.columns[0:speed_first]]
    target = 'EPA'
    y = dfy[target].values

    # %%
    print 'scaling features'
    scaler = StandardScaler()
    xns = scaler.fit_transform(xns)
    x = scaler.fit_transform(dfx)
    shuffle = StratifiedKFold(n_splits=5, shuffle=True, random_state=42041)
    ridge = Ridge(random_state=0)
    lasso = Lasso(random_state=0)

    feature_frames = [xns, x]

    n = 0
    rs_iters = 5

    for reg in [ridge, lasso]:

    results_frames = []

        model_name = str(reg).split('(')[0]
        print 'tuning %s' % (model_name)

        for x_train in feature_frames:

            print 'feature set %s' % (n)

            print 'setting up grid search'
            params_grid = {'alpha':[0.0001, 0.001, 0.01, 0.1,1, 10]}
            score = MSE
            cv_grid = GridSearchCV(reg, params_grid, cv=shuffle,
                                   scoring=['neg_mean_squared_error', 'r2'],
                                   n_jobs=-1, refit=False)
            print 'fitting grid search'
            cv_grid.fit(x_train, y)
            results_grid = pd.DataFrame(cv_grid.cv_results_)
            col_sort = 'mean_test_neg_mean_squared_error'
            best_grid = results_grid.sort_values(col_sort,
                                                 ascending=False).head(6)

            #
            print 'setting up random search'
            param1 = best_grid['param_alpha'].unique().tolist()
            param1_min = np.log(np.min(param1))
            param1_max = np.log(np.max(param1))
            param1_values = np.random.uniform(param1_min, param1_max, rs_iters)
            param1_values = map(lambda x: round(exp(x), 5), param1_values)
            param1_values.extend(param1)
            rs_iters += len(param1)

            params_random = {'alpha': param1_values}
            cv_random = RandomizedSearchCV(reg, params_random, n_iter=rs_iters,
                                           scoring=['neg_mean_squared_error', 'r2'],
                                           cv=shuffle, n_jobs=-1,
                                           verbose=1, refit=False)

            print 'fitting random search'
            cv_random.fit(x_train, y)
            results_random = pd.DataFrame(cv_random.cv_results_)
            #cols_tokeep = [c for c in results_random.columns if 'param' in c]
            #cols_tokeep.extend(['mean_test_score', 'mean_train_score'])
            #results_random = results_random[cols_tokeep]
            results_random = results_random.sort_values(col_sort,
                                                        ascending=False)
            print results_random[col_sort]
            results_random['model'] = model_name
            results_random['n_features'] = x_train.shape[1]
            results_frames.append(results_random)
            n += 1

        results = pd.concat(results_frames).reset_index(drop=True)


        f = 'out/' + model_name + '_' + target + '_random.csv'
        write_file(f, results)

if __name__ == '__main__':
    run()

# example of nested grid for SVC
#tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     #'C': [1, 10, 100, 1000]},
                    #{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
