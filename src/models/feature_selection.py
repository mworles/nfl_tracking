
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
df_rfe.to_csv(data_dir + 'out/speed_rfe.csv')


# Feature Importance with Extra Trees Classifier
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(x, y)
print(model.feature_importances_)
fi = pd.DataFrame({'features': dfx.columns, 'imp': model.feature_importances_})
fi.sort_values('imp', ascending=False)
