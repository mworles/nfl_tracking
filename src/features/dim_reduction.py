import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# import data
data_dir = 'C:/Users/mworley/nfl_tracking/data/'
df = pd.read_csv(data_dir + 'interim/speed_features.csv', index_col=0)
df = df.dropna(axis=1)
df = df.iloc[:, 1:-2]
vals = df.values

# feature extraction
n_comps = range(1, 4)
for n in n_comps:
    pca = PCA(n_components=n)
    fit = pca.fit(vals)
    print("Explained Variance: %s") % fit.explained_variance_ratio_

fit = PCA(n_components=1).fit(vals)
pc = fit.components_.flatten()
df = pd.DataFrame({'vars': df.columns, 'comps': pc})
df.sort_values('comps', ascending=False).head(10)
