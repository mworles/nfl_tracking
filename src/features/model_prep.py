# import packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict, cross_val_score

# import data
data_dir = 'C:/Users/mworley/nfl_tracking/data/'
dfx = pd.read_csv(data_dir + 'processed/features.csv', index_col=0)
dfy = pd.read_csv(data_dir + 'processed/targets.csv', index_col=0)

# select training rows
dfx = dfx[dfx['testset'] == 0]
dfy = dfy.loc[dfx.index.values, :]

# get dataset without speed features
cnames = dfx.columns.tolist()
speed_features = [c for c in cnames if 's__' in c]
speed_first = dfx.columns.tolist().index(speed_features[0])
xns = dfx[dfx.columns[0:speed_first]]
y = dfy['success'].values
epa = dfy['EPA'].values

# %%
scaler = StandardScaler()
xns = scaler.fit_transform(xns)
x = scaler.fit_transform(dfx)
shuffle = StratifiedKFold(n_splits=5, shuffle=True)
clf = LogisticRegression(random_state=0)

# %%
probs_cv_ns = cross_val_predict(clf, xns, y, cv=shuffle,
                             n_jobs=-1, method='predict_proba')
probs_ns = probs_cv_ns[:, 1]
preds_ns = np.where(probs_ns > 0.5, 1, 0)
accuracy_ns = accuracy_score(y, preds_ns)
print accuracy_ns

# %%
probs_cv_ws = cross_val_predict(clf, x, y, cv=shuffle,
                                n_jobs=-1, method='predict_proba')
probs_ws = probs_cv_ws[:, 1]
preds_ws = np.where(probs_ws > 0.5, 1, 0)
accuracy_ws = accuracy_score(y, preds_ws)
print accuracy_ws

# %%
reg = LinearRegression()
score = cross_val_score(reg, xns, epa, scoring='r2',
                       cv=shuffle, n_jobs=-1)
print 'R2: %s' % (np.mean(score))
score = cross_val_score(reg, x, epa, scoring='r2',
                       cv=shuffle, n_jobs=-1)
print 'R2: %s' % (np.mean(score))

# %%
# use RFE to select speed features
from sklearn.feature_selection import RFE
# create the RFE model and select features
rfe = RFE(reg, 5)
# filter data down to speed features
xspd = x[:, speed_first:]
rfe = rfe.fit(xspd, epa)

# summarize the selection of the attributes
rfe_bool = rfe.support_
rfe_ranks = rfe.ranking_
rfe_cols = dfx.columns[speed_first:][rfe_bool]
df_rfe = dfx[rfe_cols]
df_rfe.head()


# %%
import matplotlib.pyplot as plt
for col in df_rfe.columns:
    x = dfx[col]
    y = epa
    plt.scatter(x, y)
    plt.title(col)
    plt.show()



# %%
df_rfe.to_csv(data_dir + 'out/speed_rfe.csv')


# Feature Importance with Extra Trees Classifier
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(x, y)
print(model.feature_importances_)
fi = pd.DataFrame({'features': dfx.columns, 'imp': model.feature_importances_})
fi.sort_values('imp', ascending=False)



'''
# fit model to all data points
clf.fit(x, y)
clf.coef_

coef_speed = clf.coef_[:, speed_first:]
coef_rfe = coef_speed[:, rfe_bool].flatten()
for r in zip(df_rfe.columns, coef_rfe):
    print r
'''