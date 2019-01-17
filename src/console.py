import pandas as pd

# import data
data_dir = 'C:/Users/mworley/nfl_tracking/data/'
df = pd.read_csv(data_dir + 'interim/speed_features.csv', index_col=0)
