import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# set plot style
plt.style.use('ggplot')

# %%
data_dir = 'C:/Users/mworley/nfl_tracking/data/'
plot_dir = 'C:/Users/mworley/nfl_tracking/plots/'
rush_type = 'toLOS'
target_type = 'success'
alg_name = 'LogisticRegression'

rush_types = ['toLOS', 'contact', 'tocontact']
target_types = ['success', 'EPA', 'yards_aftercont', 'yards']
rush_targets = [(x, y) for x in rush_types for y in target_types]

# %%

for rt in rush_targets[2:3]:
    outfiles = [f for f in os.listdir(data_dir + 'out/')]
    if rt[1] in ['yards', 'EPA', 'yards_aftercont']:
        alg_names = ['Lasso', 'Ridge']
    else:
        alg_names = ['LogisticRegression']
    rush_target = "".join([rt[0], '_', rt[1], '_'])
    file_stems = ["".join([rush_target, a]) for a in alg_names]
    files_to_import = []
    for fs in file_stems:
        files_to_import.extend([f for f in outfiles if f.startswith(fs)])
    if len(files_to_import) == 0:
        continue
    else:
        files = ["".join([data_dir, 'out/', x]) for x in files_to_import]
    recent_file = files[-1]
    df = pd.read_csv(recent_file)
    #out_data = map(lambda x: pd.read_csv(x, index_col=0), files)
    #df = pd.concat(out_data)

    rt_list = [rush_target]

    if 'LogisticRegression' in alg_names:
        score = 'accuracy'
        alg = 'regression_'
        pplots = [p for p in os.listdir(plot_dir)]
        name_elements = [rush_target, alg, score]
        fig_name = "".join(name_elements)
        pfigs = [p for p in pplots if fig_name in p]

        if len(pfigs) == 0:
            name_elements.append('.pdf')
            fig_name = "".join(name_elements)
            fig_loc = plot_dir + fig_name
        else:
            f_count = str(len([p for p in pplots if fig_name in p]))
            name_elements_add = "".join(['_', f_count, '.pdf'])
            name_elements.extend(name_elements_add)
            fig_name = "".join(name_elements)
            fig_loc = plot_dir + fig_name

        n = 1
        fig = plt.figure(figsize=(10, 8))
        score_train = 'mean_train_score'
        score_val ='mean_test_score'
        ymin = df.describe().loc['min', score_val]
        ymax = df.describe().loc['max', score_train]

        for i, d in df.groupby(['param_penalty', 'n_features']):
            ax = plt.subplot(2, 2, n)
            i = list(map(lambda x: str(x), i))
            title = " ".join(rt_list + i)
            d = d.sort_values('param_C')
            x = d['param_C']
            y = d[score_train]
            plt.plot(x, y, linestyle='--', alpha=0.7, label=score_train)
            x = d['param_C']
            y = d[score_val]
            plt.plot(x, y, linestyle='-', alpha=0.7, label=score_val)
            plt.xlabel('C')
            axmin = d.describe().loc['min', 'param_C']
            axmax = d.describe().loc['max', 'param_C']
            plt.ylim(ymin, ymax)
            plt.xscale('log')
            plt.legend()
            plt.title(title)
            plt.ylabel('score')
            n += 1
        plt.tight_layout()
        #plt.show()
        print('saving %s' % (fig_loc)
        fig.savefig(fig_loc)

    else:
        df = df.rename(columns={'mean_test_neg_mean_squared_error':
                                'mean_test_MSE',
                                'mean_train_neg_mean_squared_error':
                                'mean_train_MSE'})
        score = 'mean_test_MSE'
        alg = 'regression_'
        name_elements = [rush_target, alg, score, '.pdf']
        fig_name = "".join(name_elements)
        fig_loc = plot_dir + fig_name

        n = 1
        fig = plt.figure(figsize=(10, 8))
        score_train = 'mean_train_MSE'
        score_val ='mean_test_MSE'
        ymin = df.describe().loc['min', score_val]
        ymax = df.describe().loc['max', score_train]

        for i, d in df.groupby(['alg_name', 'n_features']):
            ax = plt.subplot(2, 2, n)
            i = list(map(lambda x: str(x), i))
            title = " ".join(rt_list + i)
            d = d.sort_values('param_alpha')
            x = d['param_alpha']
            y = d[score_train]
            plt.plot(x, y, linestyle='--', alpha=0.7, label=score_train)
            x = d['param_alpha']
            y = d[score_val]
            plt.plot(x, y, linestyle='-', alpha=0.7, label=score_val)
            plt.xlabel('alpha')
            plt.ylim(ymin, ymax)
            plt.xscale('log')
            plt.title(title)
            plt.legend()
            n += 1
        plt.tight_layout()
        #plt.show()
        print('saving %s' % (fig_loc)
        fig.savefig(fig_loc)

        score = 'r2'
        alg = 'regression_'
        name_elements = [rush_target, alg, score, '.pdf']
        fig_name = "".join(name_elements)
        fig_loc = plot_dir + fig_name

        n = 1
        fig = plt.figure(figsize=(10, 8))
        ymin = df.describe().loc['min', 'mean_test_r2']
        ymax = df.describe().loc['max', 'mean_train_r2']
        for i, d in df.groupby(['alg_name', 'n_features']):
            ax = plt.subplot(2, 2, n)
            i = list(map(lambda x: str(x), i))
            title = " ".join(rt_list + i)
            score_train = 'mean_train_r2'
            score_val ='mean_test_r2'
            d = d.sort_values('param_alpha')
            x = d['param_alpha']
            y = d[score_train]
            plt.plot(x, y, linestyle='--', alpha=0.7, label=score_train)
            x = d['param_alpha']
            y = d[score_val]
            plt.plot(x, y, linestyle='-', alpha=0.7, label=score_val)
            plt.xlabel('alpha')
            plt.ylim(ymin, ymax)
            plt.xscale('log')
            plt.title(title)
            plt.legend()
            n += 1
        plt.tight_layout()
        #plt.show()
        print('saving %s' % (fig_loc)
        fig.savefig(fig_loc)
plt.close('all')

# %%
