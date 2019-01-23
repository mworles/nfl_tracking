# import packages
import pandas as pd
import sys
sys.path.insert(0, 'c:/users/mworley/nfl_tracking/src/models')
from param_searcher import (random_search_cv, guided_search_classify,
                            guided_search_regression)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso

rush_types = ['toLOS', 'tocontact', 'contact']
target_types = ['success', 'EPA', 'yards_aftercont', 'yards']
rush_targets = [(x, y) for y in target_types for x in rush_types]
# remove invalid pairs

random_iters = 50

if __name__ == '__main__':
    '''
    #map(lambda x: random_search_cv(x[0], x[1], random_iters), rush_targets)
    clf = LogisticRegression(random_state=0)
    lm_lasso = Lasso(max_iter=10000, random_state=0)
    lm_ridge = Ridge(max_iter=10000, random_state=0)

    # rush toLOS target success
    x = rush_targets[0]
    param_dict = {'penalty': ['l1', 'l2'], 'C': [0.0001, 0.2]}
    guided_search_classify(x[0], x[1], clf, param_dict, 50)

    # rush tocontact target success
    x = rush_targets[1]
    param_dict = {'penalty': ['l2'], 'C': [0.01, 0.5]}
    guided_search_classify(x[0], x[1], clf, param_dict, 50)

    # rush contact target success
    x = rush_targets[2]
    param_dict = {'penalty': ['l2'], 'C': [0.01, 1.0]}
    guided_search_classify(x[0], x[1], clf, param_dict, 50)

    # rush toLOS target EPA
    x = rush_targets[3]
    param_dict = {'alpha': [0.001, 0.50]}
    guided_search_regression(x[0], x[1], lm_lasso, param_dict, 100)

    # rush tocontact target EPA
    x = rush_targets[4]
    param_dict = {'alpha': [0.001, 0.50]}
    guided_search_regression(x[0], x[1], lm_lasso, param_dict, 100)

    # rush contact target EPA
    x = rush_targets[5]
    param_dict = {'alpha': [0.001, 0.50]}
    guided_search_regression(x[0], x[1], lm_lasso, param_dict, 100)
    '''

    # rush toLOS target yards_aftercont
    x = rush_targets[6]
    random_search_cv(x[0], x[1], random_iters)

    #param_dict = {'alpha': [0.01, 0.2]}
    #guided_search_regression(x[0], x[1], lm_lasso, param_dict, 50)
    '''
    # rush tocontact target yards_aftercont
    x = rush_targets[7]
    param_dict = {'alpha': [0.01, 0.2]}
    guided_search_regression(x[0], x[1], lm_lasso, param_dict, 50)

    # rush contact target yards_aftercont
    x = rush_targets[8]
    alpha_values = [0.01, 20]
    param_dict = {'alpha': alpha_values}
    guided_search_regression(x[0], x[1], lm_ridge, param_dict, 50)

    # rush toLOS target yards
    x = rush_targets[9]
    alpha_values = [0.005, 0.05]
    param_dict = {'alpha': alpha_values}
    guided_search_regression(x[0], x[1], lm_lasso, param_dict, 100)

    # rush tocontact target yards
    x = rush_targets[10]
    alpha_values = [0.005, 0.1]
    param_dict = {'alpha': alpha_values}
    guided_search_regression(x[0], x[1], lm_lasso, param_dict, 100)

    # rush contact target yards
    x = rush_targets[11]
    alpha_values = [0.005, 0.1]
    param_dict = {'alpha': alpha_values}
    guided_search_regression(x[0], x[1], lm_lasso, param_dict, 100)
    '''
