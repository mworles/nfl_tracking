import pandas as pd
import numpy as np

# import data
data_dir = 'C:/Users/mworley/nfl_tracking/data/'
df = pd.read_csv(data_dir + 'interim/rushes_clean.csv', index_col=0)

df = df[df['event'] == 'ball_snap']

df = df.sort_values('s', ascending=False)
df[['playDescription']].head(20)
