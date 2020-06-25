import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
ggcol = []
for j, c in zip(range(15), plt.rcParams['axes.prop_cycle']):
    ggcol.append((j, c)[1].get('color'))

# import rush tracking file
report_plots = 'C:/Users/mworley/nfl_tracking/reports/plots/'
data_dir = 'C:/Users/mworley/nfl_tracking/data/'
df = pd.read_csv(data_dir + 'interim/rush_features.csv', index_col=0)
print(df.head())
"""
# select runs where rusher crossed line of scrimmage
df = df[df['madeacross'] == 1]

player_name = 'Melvin Gordon'
dfp = df[df['displayName'] == player_name]


# %%
fig, axs = plt.subplots(figsize=(6, 4))
dfsub = dfp[['gameId', 'playId', 'time_cum', 's']]
dfsub.set_index('time_cum', inplace=True)
ax = dfsub.groupby(['gameId', 'playId'])['s'].plot(alpha=0.50, c=ggcol[0])
plt.xlim(0, 6)
plt.xlabel('Time from snap')
plt.ylabel('Speed (yards/sec)')
plt.title(player_name + 'time series of speed for each run')
fig.savefig(report_plots + 'reportplot_0.jpg')
plt.close()

# %%
fig, axs = plt.subplots(figsize=(6, 4))
for i, d in dfp.groupby(['gameId', 'playId']):
    d2 = dfp # d[d['x_fromscrim'] > 4]
    x = d2['time_cum']
    y = d2['x_fromscrim']
    c = d2['s']
    plt.scatter(x = x, y = y,  marker='8', s = 5, c = c, alpha=0.25)
cbar = plt.colorbar()
cbar.set_label('Speed', rotation=270)
plt.clim(0,8)
#plt.ylim(-10, 20)
plt.xlim(0, 6.5)
plt.title(player_name + ' - Yards gained and speed for each run')
plt.xlabel('Time from snap')
plt.ylabel('Yards gained')
plt.tight_layout()
fig.savefig(report_plots + 'reportplot_1.jpg')
plt.close()

# %%
dfp['across_sum'] = dfp.groupby(['gameId', 'playId'])['across'].transform('cumsum')
dfs = dfp[dfp['across_sum'] < 2]

fig, axs = plt.subplots(figsize=(5, 4))
dfsub = dfs[['gameId', 'playId', 'time_cum', 's']]
dfsub.set_index('time_cum', inplace=True)
ax = dfsub.groupby(['gameId', 'playId'])['s'].plot(alpha=0.50, c=ggcol[0])
plt.xlim(0, 5)
plt.xlabel('Time from snap')
plt.ylabel('Speed (yards/sec)')
plt.title(player_name + ' - snap to scrimmage')
plt.show()
fig.savefig(report_plots + 'reportplot_2.jpg')
plt.close()

# %%
dfp['across_sum'] = dfp.groupby(['gameId', 'playId'])['across'].transform('cumsum')

# select runs where contact ocurred
dfs = dfp[dfp['x_atcont'].notnull()]
# keep runs where:
# contact occurred 20 yards or less past LOS
dfs = dfs[dfs['x_fromscrim_atcont'] <=20]
# yards after contact was 20 yards or less
dfs = dfs[dfs['x_aftercont'] <=20]
# select frames occuring up to point of first contact
dfs = dfs[dfs['after_cont'] != 1]

fig, axs = plt.subplots(figsize=(5, 4))

dfsub = dfs[['gameId', 'playId', 'time_cum', 's']]
dfsub.set_index('time_cum', inplace=True)
ax = dfsub.groupby(['gameId', 'playId'])['s'].plot(alpha=0.50, c=ggcol[1])
#plt.xlim(0, 5)
plt.xlabel('Time from snap')
plt.ylabel('Speed (yards/sec)')
plt.title(player_name + ' - snap to contact')
plt.show()
fig.savefig(report_plots + 'reportplot_3.jpg')
plt.close()




# %%
player_count = df[df['event'] == 'ball_snap'].groupby('displayName')['event'].count()
df.groupby(['gameId', 'playId'])['time'].count().describe()
player_count.sort_values(ascending=False).head(50)
"""
