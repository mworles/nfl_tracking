import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'c:/users/mworley/nfl_tracking/src/features')
from feature_functions import *

df = pd.read_csv(data_dir + 'interim/contact_features.csv', index_col=0)

df.head()
