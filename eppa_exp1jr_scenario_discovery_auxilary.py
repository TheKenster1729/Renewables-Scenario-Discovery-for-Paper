from eppa_exp1jr_scenario_discovery_main import GenerateDataframe
import pandas as pd
import operator
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

"""
-----------------------------------
SET PARAMETERS
-----------------------------------
"""

input_case = 'GLB_RAW'
output_case = 'REF_GLB_RENEW_SHARE'
year = 2050

# import share of renewables, absolute renewable production, and absolute totals

# s = Analyzer('EPPA_renewables_elec_share', 'pol_GLB', input_file_path = 'EPPA_ensembles_inputs_full.csv', nrows = 400)
# r = Analyzer('EPPA_renewables_elec_prod', 'pol_GLB', input_file_path = 'EPPA_ensembles_inputs_full.csv', nrows = 400)
# t = Analyzer('EPPA_renewables_total_elec_prod', 'pol_GLB', input_file_path = 'EPPA_ensembles_inputs_full.csv', nrows = 400)

# TODO: runs become the new indices of the dataframes, makes filtering by crashed runs easier
# TODO: fix concatenation issue

# output_runs = set(s.df['Unnamed: 3'])
# input_runs = set(s.input_df['run'])
# valid_runs = list(input_runs.intersection(output_runs))
# valid_indices = [x - 1 for x in valid_runs]
master_df = 5
input_var_columns = 1
output_var_column = 2
inputs = 1

generator = GenerateDataframe('GLB_RAW', 'REF_GLB_RENEW_SHARE')

generator.CART(2050)

"""
-----------------------------------
FUNCTIONS
-----------------------------------
"""

def correlations():
    # check correlations
    print(master_df.corr())

def regression(master_df = master_df):
    ax = master_df.plot(x = 'wind', y = 'Renew. Pen.', kind = 'scatter', legend = False, xlabel = 'Price of Wind', ylabel = 'Percent of Energy From Renewables')
    d = np.polyfit(master_df['wind'], master_df['Renew. Pen.'], 1)
    f = np.poly1d(d)
    master_df.insert(len(master_df.columns), 'Reg', f(master_df['wind']))
    master_df.plot(x = 'wind', y = 'Reg', color = 'teal', legend = False, ax = ax)
    plt.show()
    #plt.savefig()

def lower_30_values():
    # check association between 30 lowest values of each variable and renewables production
    master_df_percentiles = master_df.rank(pct = True)
    master_df_percentiles = pd.concat([s.input_df['run'], master_df_percentiles], axis = 1)
    for input in inputs.columns:
        b = master_df[master_df_percentiles[input] < np.percentile(master_df_percentiles[input], 8)] # 30 lowest values of input
        overall_median = master_df['abs_renew'].median()
        print(input, ':', b['abs_renew'].median() - overall_median) # take the difference between overall and lowest 30 medians

def second_order_effects():
    # checking for second order effects by looking at scenarios with low wind, high penetration
    second_order_df = master_df[(master_df['share'] > np.percentile(master_df['share'], 75)) & (master_df['WIND'] < np.percentile(master_df['WIND'], 50))]
    second_order_df.plot(kind = 'scatter', x = 'WIND', y = 'SOLAR')
    plt.show()

def logistic_regression():
    # logistic regression
    X = sm.add_constant(master_df[master_df.columns[:9]])
    y = np.where(master_df['share'] < np.percentile(master_df['share'], 80), 1, 0)
    log_reg = sm.Logit(y, X).fit()
    print(log_reg.summary())

def CART(input_df = master_df, input_columns = input_var_columns, output_column = output_var_column, max_leaf_nodes = 15):
    # feature importances using absolute total as response variable
    X = input_df[input_columns].values
    y = input_df[output_column].values
    
    lower_percentiles = [15, 16, 17, 18, 19, 20]
    upper_percentiles = [80, 81, 82, 83, 84, 85]
    percentiles_dict = {}
    for percentile in lower_percentiles:
        metric = operator.lt(y, np.percentile(y, percentile))
        metric_clean = np.where(metric == True, 1, 0)
        clf = BaggingClassifier(base_estimator = DecisionTreeClassifier(random_state = 0, max_leaf_nodes = max_leaf_nodes), n_estimators = 1000
            ).fit(X, metric_clean)
        percentiles_dict[str(percentile)] = np.mean([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    for percentile in upper_percentiles:
        metric = operator.gt(y, np.percentile(y, percentile))
        metric_clean = np.where(metric == True, 1, 0)
        clf = BaggingClassifier(base_estimator = DecisionTreeClassifier(random_state = 0, max_leaf_nodes = max_leaf_nodes), n_estimators = 1000
            ).fit(X, metric_clean)
        percentiles_dict[str(percentile)] = np.mean([tree.feature_importances_ for tree in clf.estimators_], axis=0)

    max_length_str = max([len(a) for a in input_columns])
    print(''.ljust(max_length_str + 2) + '    '.join(['{:<2}'.format(a) for a in percentiles_dict.keys()]))
    importance_scores = np.array(list(percentiles_dict.values())).T
    for i in range(len(input_columns)):
        name = input_columns[i]
        print(name.ljust(max_length_str + 2) + 
        '  '.join(["{:<4}".format(str(a)[:4]) for a in importance_scores[i]]))

# helper function - implements k fold CV for a given model and dataset
def k_fold_CV(X, y, model, k = 10):
    from sklearn.model_selection import KFold
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score

    kf = KFold(n_splits = k, random_state = None)

    cf_matrices = []
    acc_score = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        pred_values = model.predict(X_test)
        
        cf = confusion_matrix(y_test, pred_values)
        cf_matrices.append(cf)
        acc = accuracy_score(y_test, pred_values)
        acc_score.append(acc)
    
    cf_matrix = sum(cf_matrices)/k
    score = sum(acc_score)/k
    return cf_matrix, score

def CART_validation(input_df = master_df, input_columns = input_var_columns, output_column = output_var_column, max_leaf_nodes = 10):
    X = input_df[input_columns]
    y = input_df[output_column]
    
    lower_percentiles = [40, 20, 10]
    upper_percentiles = [60, 70, 90]
    percentiles_dict = {}
    for percentile in lower_percentiles:
        metric = operator.lt(y, np.percentile(y, percentile))
        metric_clean = np.where(metric == True, 1, 0)
        clf = BaggingClassifier(base_estimator = DecisionTreeClassifier(random_state = 0, 
        max_leaf_nodes = max_leaf_nodes, criterion = 'entropy'), n_estimators = 1000)
        cf_matrix, acc_score = k_fold_CV(X, metric_clean, clf)
        print('Percentile:', '{}'.format(percentile), acc_score, cf_matrix)
        # percentiles_dict[str(percentile)] = np.mean([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    for percentile in upper_percentiles:
        metric = operator.gt(y, np.percentile(y, percentile))
        metric_clean = np.where(metric == True, 1, 0)
        clf = BaggingClassifier(base_estimator = DecisionTreeClassifier(random_state = 0, 
        max_leaf_nodes = max_leaf_nodes, criterion = 'entropy'), n_estimators = 1000)
        cf_matrix, acc_score = k_fold_CV(X, metric_clean, clf)
        print('Percentile:', '{}'.format(percentile), acc_score, cf_matrix)
        # percentiles_dict[str(percentile)] = np.mean([tree.feature_importances_ for tree in clf.estimators_], axis=0)

def accuracy_frontier(input_df = master_df, input_columns = input_var_columns, output_column = output_var_column, max_leaf_nodes = 10):
    X = input_df[input_columns]
    y = input_df[output_column]

    accs = []
    
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    for percentile in percentiles:
        metric = operator.gt(y, np.percentile(y, percentile))
        metric_clean = np.where(metric == True, 1, 0)
        clf = BaggingClassifier(base_estimator = DecisionTreeClassifier(random_state = 0, 
        max_leaf_nodes = max_leaf_nodes, criterion = "entropy"), n_estimators = 1000)
        acc_score = k_fold_CV(X, metric_clean, clf)
        accs.append(acc_score[1])
    
    plt.plot(percentiles, accs)
    plt.xlabel('Percentile')
    plt.ylabel('Accuracy')
    plt.show()

def upper_half_corr():
    # checking correlations for >50th percentile absolute total
    master_df_upper_half = master_df[master_df['total'] > np.percentile(master_df['total'], 50)]
    print(master_df_upper_half.corr())

def lower_half_corr():
    # checking correlations for <50th percentile absolute total
    master_df_lower_half = master_df[master_df['total'] < np.percentile(master_df['total'], 50)]
    print(master_df_lower_half.corr())

def upper_75th_corr():
    # checking correlations for >25% renewable penetration
    master_df[master_df['WIND'] > 0.25].corr().to_csv('greaterthan25windcorr.csv')

def feature_importances(master_df):
    # print feature importance scores
    import operator
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn import tree
    import graphviz

    lower_50th_demand = master_df[master_df['EN DEM'] < master_df['EN DEM'].median()]
    upper_50th_demand = master_df[master_df['EN DEM'] > master_df['EN DEM'].median()]

    # X = lower_50th_demand[lower_50th_demand.columns[:-3]]
    # Y_df = lower_50th_demand[lower_50th_demand.columns[-3]]

    # X = upper_50th_demand[upper_50th_demand.columns[:-3]]
    # Y_df = upper_50th_demand[upper_50th_demand.columns[-3]]

    X = master_df[master_df.columns[:-3]]
    Y_df = master_df['abs_renew']
    X_train, X_test, y_train, y_test = train_test_split(X, Y_df)

    percentiles = [90, 85, 80, 20, 15, 10]
    importance_df = pd.DataFrame([], columns = percentiles, index = master_df.columns[:-3])
    for percentile in percentiles:
        if percentile < 50:
            metric = operator.lt(y_train, np.percentile(y_train, percentile))
            y_test = np.where(operator.lt(y_test, np.percentile(y_test, percentile)) == True, 1, 0)
            metric_clean = np.where(metric == True, 1, 0)
            clf = BaggingClassifier(base_estimator = DecisionTreeClassifier(random_state = 0), n_estimators = 1000).fit(X_train, metric_clean)
            importance_df[percentile] = np.mean([tree.feature_importances_ for tree in clf.estimators_], axis=0)
            score = clf.score(X_test, y_test)
            print(score)
        else:
            metric = operator.gt(y_train, np.percentile(y_train, percentile))
            y_test = np.where(operator.gt(y_test, np.percentile(y_test, percentile)) == True, 1, 0)
            metric_clean = np.where(metric == True, 1, 0)
            clf = BaggingClassifier(base_estimator = DecisionTreeClassifier(random_state = 0), n_estimators = 1000).fit(X_train, metric_clean)
            importance_df[percentile] = np.mean([tree.feature_importances_ for tree in clf.estimators_], axis=0)
            score = clf.score(X_test, y_test)
            print(score)

    importance_df.to_csv('absolute_renewable_featureimportances_2050_reference.csv')

def generate_upperlower10th_parallelplot(master_df):
    # generate parallel plot
    master_df = master_df.rank(pct = True)
    master_df_parallelaxis = master_df[(master_df['Renew. Pen.']
    < 0.1) | (master_df['Renew. Pen.'] > 0.9)]
    master_df_parallelaxis['Label'] = np.where(master_df_parallelaxis['Renew. Pen.'] <= 0.1, 'Low Renewable Penetration', 'High Renewable Penetration')
    master_df_parallelaxis.sort_values(by = 'Label', inplace = True)

    fig, ax = plt.subplots(figsize = (12, 7.5), tight_layout = True)
    pd.plotting.parallel_coordinates(master_df_parallelaxis, class_column = 'Label', 
    cols = master_df_parallelaxis.columns[:-3], color=('b', 'r'), ax = ax, alpha = 0.25)
    ax.set_xticklabels(master_df_parallelaxis.columns[:-3], rotation = 15)
    ax.set_title('High/Low Renewable Penetration Parallel Axis Plot')
    ax.set_ylabel('Percentile')
    ax.legend(loc = 'lower left')
    # plt.savefig(fname = 'parallel_axis_plot_upper_lower_10th_for_summary.png', dpi = 300)
    plt.show()

def generate_upperlower10th_parallelplot_presentable(master_df, year = 2050):
    # generate parallel plot with better ordering of x-axis labels and min/max values
    master_df = master_df.rank(pct = True)
    master_df_parallelaxis = master_df[(master_df['Renew. Pen.']
    < 0.1) | (master_df['Renew. Pen.'] > 0.9)]
    master_df_parallelaxis['Label'] = np.where(master_df_parallelaxis['Renew. Pen.'] <= 0.1, 'Low Renewable Penetration', 'High Renewable Penetration')
    
    # housekeeping: sort and rename
    master_df_parallelaxis.sort_values(by = 'Label', inplace = True)
    master_df_parallelaxis.rename(columns = {'Renew. Pen.': 'Global Share of Renewables', 'WIND': 'Wind Cost', 'SOLAR': 'Solar Cost', 
    'NUCLEAR': 'Nuclear Cost', 'CCS': 'CCS Cost', 'BIOENERGY': 'Bioenergy Cost', 'FF COST': 'Fossil Fuels Cost', 'EN DEM': 'Energy Demand',
    'POPULATION': 'Population'}, inplace = True)
    plotting_df = master_df_parallelaxis.reindex(columns = ['Global Share of Renewables', 'Wind Cost', 'Solar Cost', 'Nuclear Cost', 'CCS Cost',
    'Bioenergy Cost', 'Fossil Fuels Cost', 'Energy Demand', 'GDP', 'Population', 'Label'])

    fig, ax = plt.subplots(figsize = (12, 7.5), tight_layout = True)
    pd.plotting.parallel_coordinates(plotting_df, class_column = 'Label', color=('b', 'r'), ax = ax, alpha = 0.75)
    ax.set_xticklabels(plotting_df.columns, rotation = 15)
    ax.set_title('High/Low Renewable Penetration Parallel Axis Plot, {}'.format(year))
    ax.set_ylabel('Percentile')
    ax.legend(loc = 'lower left')
    plt.show()
    # plt.savefig(fname = 'parallel_axis_plot_upper_lower_10th_for_summary.png', dpi = 300)

def cooler_parallel_plot(master_df = master_df, year = 2050):
    from matplotlib.path import Path
    import matplotlib.patches as patches
    from sklearn import datasets
    import seaborn as sns

    renew_pen_10th_perc = np.percentile(master_df['Renew. Pen.'], 10)
    renew_pen_90th_perc = np.percentile(master_df['Renew. Pen.'], 90)

    master_df_parallelaxis = master_df[(master_df['Renew. Pen.']
    < renew_pen_10th_perc) | (master_df['Renew. Pen.'] > renew_pen_90th_perc)]
    master_df_parallelaxis['Label'] = np.where(master_df_parallelaxis['Renew. Pen.'] <= renew_pen_10th_perc, 0, 1)
    
    # housekeeping: sort and rename
    master_df_parallelaxis.sort_values(by = 'Label', inplace = True)
    master_df_parallelaxis.rename(columns = {'Renew. Pen.': 'Global Share of Renewables', 'WIND': 'Wind Cost', 'SOLAR': 'Solar Cost', 
    'NUCLEAR': 'Nuclear Cost', 'CCS': 'CCS Cost', 'BIOENERGY': 'Bioenergy Cost', 'FF COST': 'Fossil Fuels Cost', 'EN DEM': 'Energy Demand',
    'POPULATION': 'Population'}, inplace = True)
    plotting_df = master_df_parallelaxis.reindex(columns = ['Global Share of Renewables', 'Wind Cost', 'Solar Cost', 'Nuclear Cost', 'CCS Cost',
    'Bioenergy Cost', 'Fossil Fuels Cost', 'Energy Demand', 'GDP', 'Population', 'Label'])

    iris = datasets.load_iris()
    ynames = plotting_df.columns[:-1]
    ys = plotting_df[plotting_df.columns[:-1]].values
    ymins = ys.min(axis=0)
    ymaxs = ys.max(axis=0)
    dys = ymaxs - ymins
    ymins -= dys * 0.05  # add 5% padding below and above
    ymaxs += dys * 0.05

    dys = ymaxs - ymins

    # transform all data to be compatible with the main axis
    zs = np.zeros_like(ys)
    zs[:, 0] = ys[:, 0]
    zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

    fig, host = plt.subplots(figsize=(10,4))

    axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
    for i, ax in enumerate(axes):
        ax.set_ylim(ymins[i], ymaxs[i])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if ax != host:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('right')
            ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))

    host.set_xlim(0, ys.shape[1] - 1)
    host.set_xticks(range(ys.shape[1]))
    host.set_xticklabels(ynames, fontsize=14, rotation = 15)
    host.tick_params(axis='x', which='major', pad=7)
    host.spines['right'].set_visible(False)
    host.xaxis.tick_top()
    host.set_title('High/Low Renewable Penetration Parallel Axis Plot, {}'.format(year), fontsize=18, pad=12)

    colors = ['gray', 'orange'] # plt.cm.Set2.colors
    labels = ['Low Renewable Penetration', 'High Renewable Penetration']
    legend_handles = [None for _ in labels]
    for j in range(ys.shape[0]):
        # create bezier curves
        verts = list(zip([x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
                        np.repeat(zs[j, :], 3)[1:-1]))
        codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', lw=2, alpha=0.4, edgecolor=colors[plotting_df['Label'].iloc[j]])
        legend_handles[plotting_df['Label'].iloc[j]] = patch
        host.add_patch(patch)
    host.legend(legend_handles, labels,
                loc='upper center', bbox_to_anchor=(0.5, 1.18),
                ncol=len(labels), fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(fname = 'bezier_plot_ref_2050.png', dpi = 300)

def plot_PDFs(master_df, year = 2050):
    import seaborn as sns

    plt.rc('axes.spines', **{'bottom': True, 'left': False, 'right': False, 'top': False})
    
    for var in master_df.columns:
        p = sns.kdeplot(master_df[var], color = 'red', label = year, shade = True)
        p.set(yticklabels = [])
        p.tick_params(left = False)
        p.set(ylabel = None)
        plt.show()

def give_me_a_boost(input_df, output_column):
    from sklearn.ensemble import GradientBoostingClassifier
    X = input_df[input_columns].values
    y = input_df[output_column].values

    lower_percentiles = [15, 16, 17, 18, 19, 20]
    for percentile in lower_percentiles:
        if percentile == 15:
            metric = operator.lt(y, np.percentile(y, percentile))
            metric_clean = np.where(metric == True, 1, 0)
            clf = GradientBoostingClassifier(n_estimators = 1000, learning_rate = 0.01,
                max_depth = 4, random_state = 0).fit(X, metric_clean)
            print(np.mean([est[0].feature_importances_ for est in clf.estimators_], axis = 0))

def give_me_a_boost_validation(input_df = master_df, input_columns = input_var_columns, output_column = output_var_column):
    from sklearn.ensemble import GradientBoostingClassifier
    X = input_df[input_columns]
    y = input_df[output_column]

    lower_percentiles = [15, 16, 17, 18, 19, 20]
    for percentile in lower_percentiles:
        if percentile == 15:
            metric = operator.lt(y, np.percentile(y, percentile))
            metric_clean = np.where(metric == True, 1, 0)
            clf = GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.01,
                max_depth = 4, random_state = 0)
            cf_matrix = k_fold_CV(X, metric_clean, clf)
            print(cf_matrix)
            # print(np.mean([est[0].feature_importances_ for est in clf.estimators_], axis = 0))

# give_me_a_boost_validation(master_df, 'Renew. Pen.')

# random forest classifier
def see_forest_not_trees(input_df, output_column):
    from sklearn.ensemble import RandomForestClassifier

    X = input_df[input_df.columns[:9]]
    y = input_df[output_column]

    lower_percentiles = [15, 16, 17, 18, 19, 20]
    for percentile in lower_percentiles:
        if percentile == 15:
            metric = operator.lt(y, np.percentile(y, percentile))
            metric_clean = np.where(metric == True, 1, 0)
            clf = RandomForestClassifier(n_estimators = 500, 
                    n_jobs = -1, oob_score = True, bootstrap = True)
            cf_matrix = k_fold_CV(X, metric_clean, clf)
            print('OOB score:', clf.fit(X, metric_clean).oob_score_)
            print(cf_matrix)

# see_forest_not_trees(master_df, 'Renew. Pen.')

# get a single cart tree for testing purposes
def cart_single_tree(input_df, output_column):
    from sklearn.model_selection import train_test_split

    X = input_df[input_df.columns[:9]]
    y = input_df[output_column]
    
    lower_percentiles = [15, 16, 17, 18, 19, 20]
    upper_percentiles = [80, 81, 82, 83, 84, 85]
    for percentile in lower_percentiles:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
        metric_train = operator.lt(y_train, np.percentile(y_train, percentile))
        metric_clean_train = np.where(metric_train == True, 1, 0)
        metric_test = operator.lt(y_test, np.percentile(y_train, percentile))
        metric_clean_test = np.where(metric_test == True, 1, 0)
        clf = DecisionTreeClassifier(random_state = 0).fit(X_train, metric_clean_train)
        cf_matrix, score = k_fold_CV(X_test, metric_clean_test, clf, k = 8)
        print(score)
        print(cf_matrix)
    for percentile in upper_percentiles:
        metric = operator.gt(y, np.percentile(y, percentile))
        metric_clean = np.where(metric == True, 1, 0)
        clf = DecisionTreeClassifier(random_state = 0).fit(X, metric_clean)

def bagged_regression_tree(input_df, output_column):
    from sklearn.ensemble import BaggingRegressor
    from sklearn.model_selection import train_test_split

    X = input_df[input_df.columns[:9]]
    y = input_df[output_column]
    
    lower_percentiles = [15, 16, 17, 18, 19, 20]
    upper_percentiles = [80, 81, 82, 83, 84, 85]
    percentiles_dict = {}
    for percentile in lower_percentiles:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
        clf = BaggingRegressor(n_estimators = 1000, n_jobs = -1).fit(X_train, y_train)
        print(clf.score(X_test, y_test))
        # print('Percentile:', '{}'.format(percentile), acc_score)
        # percentiles_dict[str(percentile)] = np.mean([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    # for percentile in upper_percentiles:
    #     metric = operator.gt(y, np.percentile(y, percentile))
    #     metric_clean = np.where(metric == True, 1, 0)
    #     clf = BaggingRegressor(n_estimators = 1000)
        # acc_score = k_fold_CV(X, metric_clean, clf)
        # print('Percentile:', '{}'.format(percentile), acc_score)    
# bagged_regression_tree(master_df, 'Renew. Pen.')

def random_forest_regressor(input_df = master_df, input_columns = input_var_columns, output_column = output_var_column):
    from sklearn.ensemble import RandomForestRegressor
    from itertools import product
    from seaborn import heatmap
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    X = input_df[input_columns]
    y = input_df[output_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

    lower_percentiles = [15, 16, 17, 18, 19, 20]
    for percentile in lower_percentiles:
        if percentile == 20:
            clf = RandomForestRegressor(n_estimators = 100, random_state = 0)
            model = clf.fit(X_train, y_train)
            print(model.score(X_test, y_test))

def random_forest_regressor_full_timeseries():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    X = inputs
    y = full_timeseries

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

    clf = RandomForestRegressor(n_estimators = 100, random_state = 0, max_features = 'sqrt')
    model = clf.fit(X_train, y_train)
    print(model.predict(X_test))
    print(y_test)
    print(model.score(X_test, y_test))

# generates a heatmap of accuracy relative to guessing the same thing every time,
# for various combinations of CART parameters
def hyper_parameterization_boosted_trees(input_df = master_df, input_columns = input_var_columns, output_column = output_var_column):
    from sklearn.ensemble import GradientBoostingClassifier
    from itertools import product
    from seaborn import heatmap
    import matplotlib.pyplot as plt

    # parameters to adjust
    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.0075, 0.01]
    max_depths = [2, 3, 4, 5, 6]
    combos = product(learning_rates, max_depths)
    acc_df = pd.DataFrame(data = np.zeros((len(learning_rates), len(max_depths))), 
        index = learning_rates, columns = max_depths)
    
    X = input_df[input_columns]
    y = input_df[output_column]

    lower_percentiles = [15, 16, 17, 18, 19, 20]
    for percentile in lower_percentiles:
        if percentile == 20:
            metric = operator.lt(y, np.percentile(y, percentile))
            metric_clean = np.where(metric == True, 1, 0)
            for combo in combos:
                clf = GradientBoostingClassifier(n_estimators = 100, learning_rate = combo[0],
                    max_depth = combo[1], random_state = 0)
                k = 5 # parameter for CV, hold constant
                cf_matrix, acc_score = k_fold_CV(X, metric_clean, clf, k = 10)
                rel_acc = acc_score*100 - input_df.shape[0]/k
                acc_df.loc[combo[0], combo[1]] = rel_acc
    print(acc_df)
    ax = heatmap(acc_df)
    plt.show()

def hyper_parameterization_forest_trees(input_df = inputs, input_columns = input_var_columns, output_column = output_var_column, oob_score = True):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from itertools import product
    from seaborn import heatmap
    import matplotlib.pyplot as plt

    # parameters to adjust
    min_samples_splits = [2, 3, 4, 8, 12, 16, 20]
    max_depths = [2, 4, 6, 8, 10, 12, 15]
    combos = product(min_samples_splits, max_depths)
    acc_df = pd.DataFrame(data = np.zeros((len(min_samples_splits), len(max_depths))), 
        index = min_samples_splits, columns = max_depths)
    
    X = input_df
    y = s.df[year]

    lower_percentiles = [15, 16, 17, 18, 19, 70]
    for percentile in lower_percentiles:
        if percentile == 70:
            metric = operator.gt(y, np.percentile(y, percentile))
            metric_clean = np.where(metric == True, 1, 0)
            for combo in combos:
                clf = RandomForestClassifier(n_estimators = 200, min_samples_split = combo[0],
                    max_depth = combo[1], random_state = 0, oob_score = oob_score)
                k = 5 # parameter for CV, hold constant
                cf_matrix, acc_score = k_fold_CV(X, metric_clean, clf, k = k)
                rel_acc = acc_score*100 - percentile
                acc_df.loc[combo[0], combo[1]] = rel_acc

    ax = heatmap(acc_df)
    ax.set_xlabel('max depth')
    ax.set_ylabel('min samples split')
    ax.set_title('Forest Classifier - Max Depth and Minimum Samples to Split a Node')
    plt.show()

def feature_importances_forest_trees(input_df = master_df, input_columns = input_var_columns, output_column = output_var_column, oob_score = True):
    from sklearn.ensemble import RandomForestClassifier
    import matplotlib.pyplot as plt
    
    X = input_df[input_columns]
    print(X)
    y = s.df[year]
    percentile = 70

    metric = operator.gt(y, np.percentile(y, percentile))
    metric_clean = np.where(metric == True, 1, 0)
    print(metric_clean)
    clf = RandomForestClassifier(min_samples_split = 12,
        max_depth = 12, random_state = 0, oob_score = oob_score)

    model = clf.fit(X, metric_clean)
    feature_importances = [estimator.feature_importances_ for estimator in model.estimators_]
    avg_importances = sum(feature_importances)/len(feature_importances)
    labeled_importances = [str(x) + ': ' + str(y) for x, y in zip(input_columns, avg_importances)]
    most_important_df = master_df
    for imp in labeled_importances:
        print(imp)

    most_important_df = pd.concat([input_df[['WindGas', 'WindBio', 'PC']], pd.DataFrame(metric_clean, columns = ['Target'])], axis = 1)
    pd.plotting.parallel_coordinates(most_important_df.sort_values(by = 'Target'), class_column = 'Target', color = ('#556270', '#C7F464'))
    plt.show()

def hyper_parameterization_bagged_trees(input_df = master_df, input_columns = input_var_columns, output_column = output_var_column):
    from sklearn.ensemble import BaggingClassifier
    from itertools import product
    from seaborn import heatmap
    import matplotlib.pyplot as plt

    # parameters to adjust
    max_samples = [10, 20, 50, 100, 200, 300]
    max_features = [1.0, int(2), int(5), int(7), int(8)]
    combos = product(max_samples, max_features)
    acc_df = pd.DataFrame(data = np.zeros((len(max_samples), len(max_features))), 
        index = max_samples, columns = max_features)
    
    X = input_df[input_columns]
    y = input_df[output_column]

    lower_percentiles = [15, 16, 17, 18, 19, 20]
    for percentile in lower_percentiles:
        if percentile == 20:
            metric = operator.lt(y, np.percentile(y, percentile))
            metric_clean = np.where(metric == True, 1, 0)
            for combo in combos:
                clf = BaggingClassifier(n_estimators = 100, max_samples = combo[0],
                    max_features = combo[1], random_state = 0)
                k = 5 # parameter for CV, hold constant
                cf_matrix, acc_score = k_fold_CV(X, metric_clean, clf, k = 10)
                rel_acc = acc_score*100 - input_df.shape[0]/k
                acc_df.loc[combo[0], combo[1]] = rel_acc
    print(acc_df)
    ax = heatmap(acc_df)
    ax.set_xlabel('max samples')
    ax.set_ylabel('max features')
    ax.set_title('Bagging Classifier - Max Samples and Max Features')
    plt.show()

def hyper_parameterization_single_tree(input_df = master_df, input_columns = input_var_columns, output_column = output_var_column):
    from sklearn.tree import DecisionTreeClassifier
    from itertools import product
    from seaborn import heatmap
    import matplotlib.pyplot as plt

    # parameters to adjust
    max_depth = [1, 5, 10, 20, 30, 50, 100]
    min_samples_splits = [2, 5, 10, 15, 20]
    combos = product(max_depth, min_samples_splits)
    acc_df = pd.DataFrame(data = np.zeros((len(max_depth), len(min_samples_splits))), 
        index = max_depth, columns = min_samples_splits)
    
    X = input_df[input_columns]
    y = input_df[output_column]

    lower_percentiles = [15, 16, 17, 18, 19, 20]
    for percentile in lower_percentiles:
        if percentile == 20:
            metric = operator.lt(y, np.percentile(y, percentile))
            metric_clean = np.where(metric == True, 1, 0)
            for combo in combos:
                clf = DecisionTreeClassifier(max_depth = combo[0],
                    min_samples_split = combo[1], random_state = 0)
                k = 5 # parameter for CV, hold constant
                cf_matrix, acc_score = k_fold_CV(X, metric_clean, clf, k = 10)
                rel_acc = acc_score*100 - input_df.shape[0]/k
                acc_df.loc[combo[0], combo[1]] = rel_acc
    print(acc_df)
    ax = heatmap(acc_df)
    ax.set_xlabel('min samples split')
    ax.set_ylabel('max depth')
    ax.set_title('Single Tree - Max Tree Depth and Min Split')
    plt.show()

def scenario_identification(input_case = input_case, output_case = output_case, year = year):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    generator_ref_glb_share = GenerateDataframe(input_case, output_case)
    generator_ref_glb_abs_renew = GenerateDataframe(input_case, 'REF_GLB_RENEW')
    generator_ref_glb_tot_elec = GenerateDataframe(input_case, 'REF_GLB_TOT')

    y_share = generator_ref_glb_share.get_y_by_year(year)
    y_abs_renew = generator_ref_glb_abs_renew.get_y_by_year(year)
    y_tot_elec = generator_ref_glb_tot_elec.get_y_by_year(year)

    cutoff = 70
    x_for_plot = np.linspace(min(y_abs_renew), 14000, 100)
    slope = np.percentile(y_share, cutoff)
    plt.scatter(y_abs_renew, y_tot_elec)
    plt.plot(x_for_plot, x_for_plot/slope, color = 'red')
    
    # high_renewables = np.where(y > np.percentile(y, cutoff), 1, 0)
    # plt.hist(y)
    # fix, ax = plt.subplots()
    # raw_data[raw_data['target'] == 1].plot.scatter(x = 'abs_renew', y = 'total', c = 'green', ax = ax)
    # raw_data[raw_data['target'] == 0].plot.scatter(x = 'abs_renew', y = 'total', c = 'red', ax = ax)

    # pca = PCA(n_components = 2)
    # principalComponents = pca.fit_transform(X)
    # print(pca.fit(X).explained_variance_ratio_)
    # principalDf = pd.concat([pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2']), 
    #     pd.DataFrame(high_renewables, columns = ['target'])], axis = 1)
    # fix, ax = plt.subplots()
    # principalDf[principalDf['target'] == 1].plot(kind = 'scatter', x = 'principal component 1', y = 'principal component 2', c = 'red', ax = ax)
    # principalDf[principalDf['target'] == 0].plot(kind = 'scatter', x = 'principal component 1', y = 'principal component 2', c = 'blue', ax = ax)
    plt.title("Scenario Identification")
    plt.xlabel('Absolute Renewable Production')
    plt.ylabel('Total Renewable Production')
    plt.show()

def time_series_clustering(input_df = master_df, input_columns = input_var_columns, output_columns = s.df.columns[1:]):
    from tslearn.clustering import TimeSeriesKMeans

    full_timeseries_vals = full_timeseries.drop(columns = ['Unnamed: 3'], axis = 1).to_numpy()
    full_timeseries_vals_reshaped = full_timeseries_vals.reshape(full_timeseries_vals.shape[0], full_timeseries_vals.shape[1], 1)

    model = TimeSeriesKMeans(n_clusters = 3, metric = 'dtw', max_iter = 10)
    fit = model.fit(full_timeseries_vals_reshaped)
    cluster_classifications = model.predict(full_timeseries_vals_reshaped)

    fig, ax = plt.subplots(1, 3, figsize = (12, 5), sharey = True)
    fig.tight_layout()
    for i in range(3):
        for x in full_timeseries_vals_reshaped[cluster_classifications == i]:
            ax[i].plot(full_timeseries.columns[1:].values.astype(int), x.ravel(), "k-", alpha = 0.2)
        ax[i].plot(full_timeseries.columns[1:].values.astype(int), fit.cluster_centers_[i].ravel(), "r-")
        if i == 0:
            ax[i].set_ylabel('Renewable Penetration')
        if i == 1:
            ax[i].set_xlabel('Year')
            ax[i].set_title('Reference Scenario Clustering')

    plt.show()

def cart_with_clusters(input_df = inputs, input_columns = input_var_columns, output_column = output_var_column, oob_score = True):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from itertools import product
    from seaborn import heatmap
    import matplotlib.pyplot as plt
    from tslearn.clustering import TimeSeriesKMeans
    from scipy.stats import mode
    from sklearn.model_selection import train_test_split

    # get the clusters
    full_timeseries_vals = full_timeseries.drop(columns = ['Unnamed: 3']).to_numpy()
    full_timeseries_vals_reshaped = full_timeseries_vals.reshape(full_timeseries_vals.shape[0], full_timeseries_vals.shape[1], 1)

    model = TimeSeriesKMeans(n_clusters = 6, metric = 'dtw', max_iter = 10)
    fit = model.fit(full_timeseries_vals_reshaped)
    cluster_classifications = model.predict(full_timeseries_vals_reshaped)

    # do the model thing
    X = master_df[input_columns]
    y = cluster_classifications
    y_mode = mode(y)[1][0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

    samp_splits = np.linspace(2, 20, num = 19, dtype = int)
    model_scores = []
    for samp_split in samp_splits:
        clf = RandomForestClassifier(n_estimators = 300, min_samples_split = samp_split,
                        random_state = 0, oob_score = oob_score).fit(X_train, y_train)
        rel_acc = clf.score(X_test, y_test) - y_mode/y.size
        model_scores.append(rel_acc)

    plt.plot(samp_splits, model_scores)
    plt.xlabel('Min Samples Split')
    plt.ylabel('Relative Accuracy')
    plt.title('Cluster Classification Validation')
    plt.show()
    # k = 5 # parameter for CV, hold constant
    # cf_matrix, acc_score = k_fold_CV(X, y, clf, k = k)
    # print(acc_score)
    # rel_acc = acc_score * 100 - mode(y).mode[0]/y.size
    # print(rel_acc)


    # parameters to adjust
    # min_samples_splits = [2, 3, 4, 8, 12, 16, 20]
    # max_depths = [2, 4, 6, 8, 10, 12, 15]
    # combos = product(min_samples_splits, max_depths)
    # acc_df = pd.DataFrame(data = np.zeros((len(min_samples_splits), len(max_depths))), 
    #     index = min_samples_splits, columns = max_depths)
    
    # X = input_df
    # y = s.df[year]

    # lower_percentiles = [15, 16, 17, 18, 19, 70]
    # for percentile in lower_percentiles:
    #     if percentile == 70:
    #         metric = operator.gt(y, np.percentile(y, percentile))
    #         metric_clean = np.where(metric == True, 1, 0)
    #         for combo in combos:
    #             clf = RandomForestClassifier(n_estimators = 200, min_samples_split = combo[0],
    #                 max_depth = combo[1], random_state = 0, oob_score = oob_score)
    #             k = 5 # parameter for CV, hold constant
    #             cf_matrix, acc_score = k_fold_CV(X, metric_clean, clf, k = k)
    #             rel_acc = acc_score*100 - percentile
    #             acc_df.loc[combo[0], combo[1]] = rel_acc

    # ax = heatmap(acc_df)
    # ax.set_xlabel('max depth')
    # ax.set_ylabel('min samples split')
    # ax.set_title('Forest Classifier - Max Depth and Minimum Samples to Split a Node')
    # plt.show()

"""
------------------
TEST CASES
------------------
"""
# random_input_array = np.random.normal(1, 1, (400,5))
# output_array = 5*random_input_array[:,0] + random_input_array[:,1] + random_input_array[:,2] + random_input_array[:,3] + random_input_array[:,4]
# test_df = pd.concat([pd.DataFrame(random_input_array, columns = ['Var1', 'Var2', 'Var3', 'Var4', 'Var5']), pd.DataFrame(output_array, columns = ['output'])], axis = 1)

# CART_validation(input_df = test_df, input_columns = test_df.columns[:5], output_column = 'output')
