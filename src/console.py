import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'c:/users/mworley/nfl_tracking/src/features')
from feature_functions import *
data_dir = 'C:/Users/mworley/nfl_tracking/data/'


rush_types = ['toLOS', 'tocontact']
target_types = ['yards', 'yards_aftercont', 'EPA', 'success']
rush_targets = [(x, y) for y in target_types for x in rush_types]
# remove invalid pairs
rush_targets.remove(('toLOS', 'yards_aftercont'))

common_feats = []

for x in rush_targets:
    print x
    dfx = combine_features(x[0], x[1])
    s_cols = [c for c in dfx.columns if 's__' in c]
    print dfx[s_cols].shape[1]
    if len(common_feats) == 0:
        common_feats = s_cols
    else:
        common_feats = list(set(common_feats).intersection(s_cols))

for x in pd.Series(common_feats):
    print x

#
dfx = combine_features('toLOS', 'yards')
dfx.sort_values('s__abs_energy', ascending=False).head()


sum = pd.read_csv(data_dir + 'out/summary.csv')
sum.head()
for x in sum.iterrows():
    print x.values




list(set(list1).intersection(list2))

dfy = pd.read_csv(data_dir + 'interim/targets.csv', index_col=0)
