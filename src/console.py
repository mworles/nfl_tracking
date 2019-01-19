import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import data
data_dir = 'C:/Users/mworley/nfl_tracking/data/'

dfx = pd.read_csv(data_dir + 'processed/features.csv', index_col=0)
dfy = pd.read_csv(data_dir + 'processed/targets.csv', index_col=0)
df = pd.merge(dfx, dfy, left_index=True, right_index=True, how='inner')

# select training rows
df = df[df['testset'] == 0]

x = df['dfndis_LOS']
y = df['EPA']

plt.scatter(x, y)
plt.show()


# %%
df = pd.read_csv(data_dir + 'interim/contact_features.csv', index_col=0)
df.head(50)

# %%
df = pd.read_csv("C:/Users/mworley/nfl_tracking/data/raw/scrapR2017.csv")
df = df[df['RushAttempt'] == 1]
for c in df.columns:
    print c

df = df.sort_values('EPA', ascending=False)

df[['EPA', 'qtr', 'yrdline100', 'ydstogo', 'GoalToGo', 'down', 'Yards.Gained', 'Touchdown', 'WPA']]
