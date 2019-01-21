import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'c:/users/mworley/nfl_tracking/src/features')
from feature_functions import *

rush_types = ['toLOS', 'contact', 'tocontact']
target_types = ['success', 'EPA', 'yards_aftercont', 'yards']
#rush_targets = [(x, y) for x in rush_types, y in target_types] #for y in target_types]

rush_targets = [(x, y) for x in rush_types for y in target_types]

rush_targets

rush_type = rush_types[0]
target_type = target_types[0]

dfx = combine_features(rush_type, target_type)
data_dir = 'C:/Users/mworley/nfl_tracking/data/'
targets = pd.read_csv(data_dir + 'interim/targets.csv', index_col=0)

targets[~targets.index.duplicated()].shape
targets.shape
