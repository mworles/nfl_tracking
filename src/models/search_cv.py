# import packages
import pandas as pd
import numpy as np
from math import exp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict, cross_val_score
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
    y = dfy['success'].values

    # %%
    print 'scaling features'
    scaler = StandardScaler()
    xns = scaler.fit_transform(xns)
    x = scaler.fit_transform(dfx)
    shuffle = StratifiedKFold(n_splits=5, shuffle=True, random_state=42041)
    clf = LogisticRegression(random_state=0)
    model = str(clf).split('(')[0]


    feature_frames = [xns, x]
    results_frames = []

    n = 0
    rs_iters = 50

    for x_train in feature_frames:

        print 'feature set %s' % (n)

        print 'setting up grid search'
        params_grid = {'penalty': ['l1', 'l2'], 'C':[0.0001, 0.001, 0.01, 0.1,1, 10]}
        score = 'accuracy'
        cv_grid = GridSearchCV(clf, params_grid, cv=shuffle, scoring=score, n_jobs=-1)
        print 'fitting grid search'
        cv_grid.fit(x_train, y)
        results_grid = pd.DataFrame(cv_grid.cv_results_)
        best_grid = results_grid.sort_values('mean_test_score',
                                             ascending=False).head(6)

        #
        print 'setting up random search'
        param1 = best_grid['param_penalty'].unique().tolist()
        param2 = best_grid['param_C'].unique().tolist()
        #param2_min = np.min(param2)
        param2_min = np.log(0.01)
        param2_max = np.log(10)
        param2_values = np.random.uniform(param2_min, param2_max, rs_iters)
        param2_values = map(lambda x: round(exp(x), 5), param2_values)
        param2_values.extend(param2)
        rs_iters += len(param2)

        params_random = {'penalty': param1,
                         'C': param2_values}
                         #'C': np.log(stats.uniform(param2_min, param2_max))}
        cv_random = RandomizedSearchCV(clf, params_random, n_iter = rs_iters,
                                       scoring=score,
                                       cv=shuffle, n_jobs=-1,
                                       verbose=1)

        print 'fitting random search'
        cv_random.fit(x_train, y)
        results_random = pd.DataFrame(cv_random.cv_results_)
        cols_tokeep = [c for c in results_random.columns if 'param' in c]
        cols_tokeep.extend(['mean_test_score', 'mean_train_score'])
        results_random = results_random[cols_tokeep]
        results_random = results_random.sort_values('mean_test_score',
                                                    ascending=False)
        results_random['model'] = str(clf).split('(')[0]
        results_random['n_features'] = x_train.shape[1]
        results_frames.append(results_random)
        n += 1

    results = pd.concat(results_frames).reset_index(drop=True)


    f = 'out/' + model + '_random.csv'
    write_file(f, results)

if __name__ == '__main__':
    run()

# example of nested grid for SVC
#tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     #'C': [1, 10, 100, 1000]},
                    #{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
