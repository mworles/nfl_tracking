# import packages
import pandas as pd
import sys
sys.path.insert(0, 'c:/users/mworley/nfl_tracking/src/models')
from param_searcher import random_search_cv

rush_types = ['toLOS', 'contact', 'tocontact']
target_types = ['success', 'EPA', 'yards_aftercont', 'yards']
rush_targets = [(x, y) for x in rush_types for y in target_types]
random_iters = 50

if __name__ == '__main__':

    map(lambda x: random_search_cv(x[0], x[1], random_iters), rush_targets)
