# import packages
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'c:/users/mworley/nfl_tracking/src/features')
from feature_functions import *

# import data
data_dir = 'C:/Users/mworley/nfl_tracking/data/'
rush = pd.read_csv(data_dir + 'interim/rush_contact.csv', index_col=0)
epa = pd.read_csv(data_dir + 'interim/epa_rush.csv', index_col=0)

cols_tokeep = ['x_aftercont', 'x_fromscrim_atcont']
rush = rush.loc[:, cols_tokeep]

cols_tokeep = ['EPA', 'WPA', 'success', 'Yards.Gained']
epa = epa.loc[:, cols_tokeep]

targets = pd.merge(rush, epa, left_index=True, right_index=True, how='outer')

# save to file
write_file('processed/targets.csv', targets)
