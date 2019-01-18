# import packages
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'c:/users/mworley/nfl_tracking/src/features')
from feature_functions import *

# import data
data_dir = 'C:/Users/mworley/nfl_tracking/data/'
rush_ft = pd.read_csv(data_dir + 'interim/rush_selected.csv', index_col=0)
play_ft = pd.read_csv(data_dir + 'interim/play_features.csv', index_col=0)
speed_ft = pd.read_csv(data_dir + 'interim/speed_features.csv', index_col=0)
tgts = pd.read_csv(data_dir + 'interim/epa_rush.csv', index_col=0)

x1cols = ['testset', 'early_con']
x1 = rush_ft[x1cols]

x2cols = play_ft.columns.tolist()
x2cols.remove('down')
x2 = play_ft[x2cols]

x = pd.merge(x1, x2, left_index=True, right_index=True, how='inner')
x = pd.merge(x, speed_ft, left_index=True, right_index=True, how='inner')
x = x.dropna()

y = tgts[['EPA', 'WPA', 'success']]

xy = pd.merge(x, y, left_index=True, right_index=True, how='inner')
xy = xy.drop_duplicates()
xy = xy.dropna()

y = xy[y.columns]
x = xy[x.columns]

# save to file
write_file('processed/features.csv', x)
write_file('processed/targets.csv', y)
