import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import data
data_dir = 'C:/Users/mworley/nfl_tracking/data/'
df = pd.read_csv(data_dir + 'interim/rushes_clean.csv', index_col=0)
dfd = pd.read_csv(data_dir + 'interim/defenders.csv', index_col=0)
df.columns
ends = df[df['end'] == 1]
gids = ends['gameId'].drop_duplicates().values
gid = gids[0]

ends = ends[ends['gameId'] == gid]
