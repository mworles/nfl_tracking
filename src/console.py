import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'c:/users/mworley/nfl_tracking/src/features')
from feature_functions import *

data_dir = 'C:/Users/mworley/nfl_tracking/data/'
df = pd.read_csv(data_dir + 'out/contact_yards_Lasso.csv', index_col=0)
df = df[df['n_features'] == 8]
df = df[df['alg_name'] == 'Lasso']
