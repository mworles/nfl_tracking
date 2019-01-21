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
rush = rush.rename(columns={'x_aftercont': 'yards_aftercont',
                            'x_fromscrim_atcont': 'yards_tocont'})

cols_tokeep = ['EPA', 'WPA', 'success', 'Yards.Gained']
epa = epa.loc[:, cols_tokeep]

targets = pd.merge(rush, epa, left_index=True, right_index=True, how='outer')

targets = targets.rename(columns={'Yards.Gained': 'yards'})

# remove extreme values of EPA
targets['EPA'] = np.where(targets['EPA'] > 4, np.nan, targets['EPA'])
targets['EPA'] = np.where(targets['EPA'] < -4, np.nan, targets['EPA'])

# cap extreme positive values of yards
targets['yards'] = np.where(targets['yards'] > 45, 45, targets['yards'])

# drop duplicates by index
targets = targets[~targets.index.duplicated(keep='first')]

# save to file
write_file('interim/targets.csv', targets)
