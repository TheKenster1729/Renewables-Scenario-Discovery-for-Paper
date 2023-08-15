import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import operator
from textwrap import wrap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from itertools import product
from seaborn import heatmap
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn import tree
import graphviz
import os

SMALL_SIZE = 15
MEDIUM_SIZE = 18
BIGGER_SIZE = 12

plt.rc('font', size = SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize = SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize = MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize = SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize = SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize = SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize = BIGGER_SIZE)  # fontsize of the figure title
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

class SD:
    def __init__(self, input_case, output_case):
        """Creates an object that stores the input and output dataframes that contain simulation results. These objects have a number of
        methods that are helpful for scenario discovery analysis. Be careful not to mix incompatible input/output cases together (e.g.
        using U.S. input data to analyze share of renewables in China).

        Args:
            input_case (str): the input data to be used (supported: GLB_GRP_NORM (global, grouped, and normed); GLB_RAW
            (global, full number of variables, no normalization); USA (full number of variables with GDP + Pop specific to US);
            CHN (full number of variables with GDP + Pop specific to China); EUR (full number of variables with GDP + Pop
            specific to EU); CHN_ENDOG_RENEW (output-output mapping as shown in paper section 4.1); CHN_ENDOG_EMISSIONS (output-output
            mapping as shown in paper section 4.2))

            output_case (str): the output metric (supported: REF_GLB_RENEW_SHARE (global share of renewables under
            the reference scenario); REF_USA_RENEW_SHARE (US share of renewables under the reference scenario); REF_CHN_RENEW_SHARE
            (China share of renewables under the reference scenario); REF_EUR_RENEW_SHARE (EU share of renewables under the reference
            scenario); 2C_GLB_RENEW_SHARE (global share of renewables under the policy scenario); 2C_USA_RENEW_SHARE (US share of
            renewables under the policy scenario); 2C_CHN_RENEW_SHARE (China share of renewables under the policy scenario);
            2C_EUR_RENEW_SHARE (EU share of renewables under the policy scenario); REF_GLB_RENEW (global renewable energy
            production in Twh); REF_GLB_TOT (total global energy production in Twh); 2C_GLB_RENEW (global renewable energy production
            in Twh under policy); 2C_GLB_TOT (total global energy production in Twh under policy); 2C_CHN_ENDOG_RENEW (output-output 
            mapping as shown in paper section 4.1); REF_CHN_EMISSIONS (output-output mapping as shown in paper section 4.1); CHN_ENDOG_EMISSIONS 
            (output-output mapping as shown in paper section 4.2))

        Raises:
            ValueError: if an invalid input case is passed
            ValueError: if an invalid output case is passed
        """
        self.input_case = input_case
        self.output_case = output_case

        self.supported_input_scenarios = ['GLB_GRP_NORM', 'GLB_RAW', 'USA', 'CHN', 'EUR', 'CHN_ENDOG_RENEW', 'CHN_ENDOG_EMISSIONS']
        self.natural_to_code_conversions_dict_inputs = {'GLB_GRP_NORM': ['samples-norm+groupedav', 'A:J'], 'GLB_RAW': ['samples', 'A:BB'], 'USA': ['samples', 'A:AZ, BC:BF'],
            'CHN': ['samples', 'A:AZ, BG:BJ'], 'EUR': ['samples', 'A:AZ, BK: BN'], 'CHN_ENDOG_RENEW': ['2C_CHN_renew_outputs_inputs', 'A:G'],
            'CHN_ENDOG_EMISSIONS': ['REF_CHN_emissions_inputs', 'A:H']}
        if self.input_case in self.supported_input_scenarios:
            sheetname = self.natural_to_code_conversions_dict_inputs[self.input_case][0]
            columns = self.natural_to_code_conversions_dict_inputs[self.input_case][1]
            if self.input_case == "GLB_GRP_NORM":
                self.input_df = pd.read_excel('Full Data for Paper.xlsx', sheetname, usecols = columns, nrows = 400, engine = 'openpyxl')
            else:
                self.input_df = pd.read_excel('Full Data for Paper.xlsx', sheetname, usecols = columns, nrows = 400, header = 2, engine = 'openpyxl')
            
            if self.input_case == 'CHN_ENDOG_RENEW':
                # these scenarios already have the crashed runs removed
                pass
            else:
                if '2C' in output_case:
                    # indicates a policy scenario in which runs crashed
                    crashed_run_numbers = [3, 14, 74, 116, 130, 337]
                    self.input_df = self.input_df.drop(index = [i - 1 for i in crashed_run_numbers])
        else:
            raise ValueError('This input scenario is not supported. Supported scenarios are {}'.format(self.supported_input_scenarios))

        self.supported_output_scenarios = ['REF_GLB_RENEW_SHARE', 'REF_USA_RENEW_SHARE', 'REF_CHN_RENEW_SHARE', 'REF_EUR_RENEW_SHARE',
            '2C_GLB_RENEW_SHARE', '2C_USA_RENEW_SHARE', '2C_CHN_RENEW_SHARE', '2C_EUR_RENEW_SHARE', 'REF_GLB_RENEW', 'REF_GLB_TOT', '2C_GLB_RENEW', '2C_GLB_TOT',
            '2C_CHN_ENDOG_RENEW', 'REF_CHN_EMISSIONS']
        self.natural_to_code_conversions_dict_outputs = {'REF_GLB_RENEW_SHARE': 'ref_GLB_renew_share', 'REF_USA_RENEW_SHARE': 'ref_USA_renew_share',
            'REF_CHN_RENEW_SHARE': 'ref_CHN_renew_share', 'REF_EUR_RENEW_SHARE': 'ref_EUR_renew_share', '2C_GLB_RENEW_SHARE': '2C_GLB_renew_share',
            '2C_USA_RENEW_SHARE': '2C_USA_renew_share', '2C_CHN_RENEW_SHARE': '2C_CHN_renew_share', '2C_EUR_RENEW_SHARE': '2C_EUR_renew_share',
            'REF_GLB_RENEW': 'ref_GLB_renew', 'REF_GLB_TOT': 'ref_GLB_total_elec', '2C_GLB_RENEW': '2C_GLB_renew', '2C_GLB_TOT': '2C_GLB_total_elec',
            '2C_CHN_ENDOG_RENEW': '2C_CHN_renew_outputs_output', 'REF_CHN_EMISSIONS': 'REF_CHN_emissions_output'}
        if self.output_case in self.supported_output_scenarios:
            sheetname = self.natural_to_code_conversions_dict_outputs[output_case]
            self.output_df = pd.read_excel('Full Data for Paper.xlsx', sheetname, usecols = 'D:X', nrows = 400, engine = 'openpyxl')
            if self.output_case == '2C_CHN_ENDOG_RENEW' or self.output_case == 'REF_CHN_EMISSIONS':
                self.output_df = pd.read_excel('Full Data for Paper.xlsx', sheetname, usecols = 'A:B', nrows = 400, engine = 'openpyxl')
                print('Note: some methods are not supported for this output scenario because it contains data from only one year of the simulation (2050).')
            else:
                if '2C' in output_case:
                    # indicates a policy scenario in which runs crashed
                    crashed_run_numbers = [3, 14, 74, 116, 130, 337]
                    self.output_df = self.output_df.drop(index = [i - 1 for i in crashed_run_numbers])
        else:
            raise ValueError('This output scenario is not supported. Supported scenarios are {}'.format(self.supported_output_scenarios))

    def get_X(self, runs = False):
        """Get the exogenous dataset (does not include run numbers).

        Returns:
            DataFrame: Input variables and their values
        """
        if runs:
            return self.input_df
        else:
            return self.input_df[self.input_df.columns[1:]]

    def get_y(self, runs = False):
        """Get the endogenous dataset (does not include run numbers).

        Returns:
            DataFrame: Output timeseries
        """
        if runs:
            return self.output_df
        else:
            return self.output_df[self.output_df.columns[1:]]

    def get_y_by_year(self, year):
        """Get the series for an individual year.

        Args:
            year (int): A year included in the dataset (options:
            2007, and 2010-2100 in 5-year increments)

        Returns:
            Series: A pandas Series object with the data from the given year
        """
        return self.output_df[str(year)]

    def parallel_plot_grp_norm(self, year, percentile = 70):
        """Generate a parallel plot for the given output scenario, using the global grouped and normed inputs. Note that the parallel
        plot will only use the global grouped and normed inputs, so this function should only be used with global output data. Populates
        current working directory with SVG image of the plot for editing.

        Args:
            year (int): A year included in the dataset (options:
            2007, and 2010-2100 in 5-year increments)
            percentile (float, optional): The percentile defining the threshold between "high renewable energy penetration" and
            "low renewable energy penetration". Values above this threshold will be categorized as "high renewable energy
            penetration" and values below this threshold will be categorized as "low renewable energy penetration". Defaults to 70.
        """
        import plotly.graph_objects as go

        generator_for_plot = SD("GLB_GRP_NORM", self.output_case)
        X = generator_for_plot.get_X()
        y = generator_for_plot.get_y_by_year(year)
        perc = np.percentile(y, percentile)

        dataframe_for_plot = X.copy()
        dataframe_for_plot['Share'] = y
        dataframe_for_plot_sorted = dataframe_for_plot.sort_values(by = ['Share'])
        y_discrete = np.where(dataframe_for_plot_sorted['Share'] > perc, 1, 0)

        dimensions_list = []
        for name, data in dataframe_for_plot_sorted.iteritems():
            data_max = max(data)
            data_min = min(data)
            series_dict = dict(range = [data_min, data_max], label = name, values = data)
            dimensions_list.append(series_dict)

        fig = go.Figure(data = go.Parcoords(line = dict(color = y_discrete,
                        colorscale = [[0, 'green'], [1, 'blue']], showscale = True), dimensions = dimensions_list))

        fig.update_layout(
            plot_bgcolor = 'white',
            paper_bgcolor = 'white'
        )

        fig.show()
        fig.write_image("{} {} Parallel Plot.svg".format(generator_for_plot.input_case, generator_for_plot.output_case), width = 1300, height = 500)

    def CART(self, year, percentile = 70, gt = True, max_depth = 5):
        """Make an example CART tree similar to the ones used in the random forest aggregation. Populates current working directory with SVG tree image for editing.

        Args:
            year (int): A year included in the dataset (options:
            2007, and 2010-2100 in 5-year increments)
            percentile (float, optional): The percentile defining the threshold between "high renewable energy penetration" and
            "low renewably energy penetration". Use with gt below. Defaults to 70.
            gt (bool, optional): Whether the target class should be greater than (default) percentile, or less than percentile.
            If True, then runs with renewable penetration higher than the percentile value are labeled as the target
            class (1). If False, then runs with renewable penetration lower than the percentile value are labeled as the target class
            (1). Defaults to True.
            max_depth (int, optional): Controls the size of the tree. Defaults to 5.
        """
        from sklearn.tree import export_graphviz

        X = self.get_X()
        if self.output_case == '2C_CHN_ENDOG_RENEW' or self.output_case == 'REF_CHN_EMISSIONS':
            y = self.get_y()
        else:
            y = self.get_y_by_year(year)

        perc = np.percentile(y, percentile)
        if gt == True:
            y_discrete = np.where(y > perc, 1, 0)
        elif gt == False:
            y_discrete = np.where(y < perc, 1, 0)
        else:
            raise ValueError("Supported values of gt are True and False")

        wrapped_columns = ['\n'.join(wrap(i, 10)) for i in X.columns]
        tree = DecisionTreeClassifier(max_depth = max_depth, random_state = 42)
        tree_fit = tree.fit(X, y_discrete)

        fig, ax = plt.subplots()
        visual = plot_tree(tree_fit, feature_names = wrapped_columns, rounded = True, filled = True, fontsize = 10, ax = ax)
        # plt.savefig(fname = 'test_tree.png', dpi = 300)
        # print("FEATURE IMPORTANCES\n")
        # for name, importance in zip(X.columns, tree_fit.feature_importances_):
        #     print(name + ': %0.2f' % importance)

        dot_data = export_graphviz(tree_fit,
                  feature_names = X.columns,
                  class_names = ['Low Renew.', 'High Renew.'],
                  filled = True, rounded = True,
                  special_characters = True,
                   out_file = None,
                           )
        graph = graphviz.Source(dot_data, format = 'svg')
        if gt == True:
            graph.render(filename = 'Plots/' + self.input_case + '; ' + self.output_case + ' ' + str(year) + ', ' + 'Target High Renewables CART')
        elif gt == False:
            graph.render(filename = 'Plots/' + self.input_case + '; ' + self.output_case + ' ' + str(year) + ', ' + 'Target Low Renewables CART')
        else:
            raise ValueError("Supported values of gt are True and False")
        plt.show()

    def k_fold_CV(self, X, y, model, k = 10):
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

    def hyper_parameterization_random_forest_classsifier(self, year):

        # parameters to adjust
        min_samples_splits = [2, 3, 4, 8, 12, 16, 20]
        max_depths = [2, 4, 6, 8, 10, 12, 15]

        X = self.get_X()
        y = self.get_y_by_year(year)

        fig, axs = plt.subplots(2, 3, sharex = 'all', sharey = 'all')
        percentiles = [10, 20, 30, 70, 80, 90]
        for i, percentile in enumerate(percentiles):
            acc_df = pd.DataFrame(data = np.zeros((len(min_samples_splits), len(max_depths))),
            index = min_samples_splits, columns = max_depths)

            if percentile < 50:
                metric = operator.lt(y, np.percentile(y, percentile))
                metric_clean = np.where(metric == True, 1, 0)
                combos = product(min_samples_splits, max_depths)
                for combo in combos:
                    clf = RandomForestClassifier(n_estimators = 200, min_samples_split = combo[0],
                        max_depth = combo[1], random_state = 0)
                    k = 5 # parameter for CV, hold constant
                    cf_matrix, acc_score = self.k_fold_CV(X, metric_clean, clf, k = k)
                    rel_acc = acc_score*100 - (100 - percentile)
                    acc_df.loc[combo[0], combo[1]] = rel_acc

            if percentile > 50:
                metric = operator.gt(y, np.percentile(y, percentile))
                metric_clean = np.where(metric == True, 1, 0)
                combos = product(min_samples_splits, max_depths)
                for combo in combos:
                    clf = RandomForestClassifier(n_estimators = 200, min_samples_split = combo[0],
                        max_depth = combo[1], random_state = 0)
                    k = 5 # parameter for CV, hold constant
                    cf_matrix, acc_score = self.k_fold_CV(X, metric_clean, clf, k = k)
                    rel_acc = acc_score*100 - percentile
                    acc_df.loc[combo[0], combo[1]] = rel_acc

            ax_obj = axs.flat[i]
            heatmap(acc_df, ax = ax_obj)
            ax_obj.set_title('Percentile = {}'.format(percentile))

        fig.suptitle('RFC Hyperparameterization - {}, {} {}'.format(self.input_case, self.output_case, year))
        fig.supxlabel('Max Depth')
        fig.supylabel('Min Samples to Split a Node')
        plt.savefig('Plots/{} {} {} RFC Hyper.png'.format(self.input_case, self.output_case, year), dpi = 300)
        plt.show()

    def rfc_model_wit_top_n(self, year, X, y_discrete, num_to_plot = 4):
        clf = RandomForestClassifier(n_estimators = 200, min_samples_split = 8,
                        max_depth = 6, random_state = 0)
        model_fit = clf.fit(X, y_discrete)

        feature_importances = pd.DataFrame([estimator.feature_importances_ for estimator in model_fit.estimators_], columns = self.get_X().columns)
        sorted_labeled_importances = feature_importances.mean().sort_values(ascending = False)
        top_n = sorted_labeled_importances.index[:num_to_plot].to_list()

        return feature_importances, sorted_labeled_importances, top_n


    def parallel_plot_most_important_inputs(self, year, gt = True, percentile = 70, num_to_plot = 4):
        from sklearn.ensemble import RandomForestClassifier
        import matplotlib.pyplot as plt
        import plotly.graph_objects as go
        import plotly.express as px

        X = self.get_X()
        if self.output_case == '2C_CHN_ENDOG_RENEW' or self.output_case == 'REF_CHN_EMISSIONS':
            y = self.get_y()
        else:
            y = self.get_y_by_year(year)

        perc = np.percentile(y, percentile)
        if gt == True:
            y_discrete = np.where(y > perc, 1, 0)
        elif gt == False:
            y_discrete = np.where(y < perc, 1, 0)
        else:
            raise ValueError("Supported values of gt are True and False")

        # train classifier
        feature_importances, sorted_labeled_importances, top_n = self.rfc_model_wit_top_n(year, X, y_discrete)

        # bar graph to display importances of top 4
        figbar, axbar = plt.subplots(figsize = (12, 7))
        sorted_labeled_importances.iloc[:num_to_plot].plot(kind = 'bar', ax = axbar, rot = 0)
        axbar.set_ylabel('Avg. Feature Importance')
        axbar.set_xlabel('Four Most Important Features')
        axbar.set_title('Most Important Features for {}, {} {}'.format(self.input_case, self.output_case, year))

        # parallel axis plot
        X_for_plot = X[top_n]
        dataframe_for_plot = pd.concat([X_for_plot, y], axis = 1).rename({str(year): 'Share'}, axis = 'columns')
        dataframe_for_plot_sorted = dataframe_for_plot.sort_values(by = ['Share'])
        if gt == True:
            y_discrete_for_plot = np.where(dataframe_for_plot_sorted['Share'] > perc, 1, 0)
        elif gt == False:
            y_discrete_for_plot = np.where(dataframe_for_plot_sorted['Share'] < perc, 1, 0)

        if self.output_case =='REF_CHN_EMISSIONS':
            dataframe_for_plot_sorted = dataframe_for_plot_sorted.rename({'Share': 'China CO2eq Emissions'}, axis = 'columns')

        fig = px.parallel_coordinates(dataframe_for_plot_sorted[top_n + [dataframe_for_plot_sorted.columns[-1]]],
            color = y_discrete_for_plot, color_continuous_scale = [(0.00, "rgb(124, 181, 24, 0.25)"), (0.5, "rgb(124, 181, 24, 0.25)"), (0.5, "rgb(253, 174, 97, 0.25)"),  (1.00, "rgb(253, 174, 97, 0.25)")],
            dimensions = top_n + [dataframe_for_plot_sorted.columns[-1]])
        fig.update_layout(coloraxis_colorbar = dict(
            title = dataframe_for_plot_sorted.columns[-1],
            tickvals = [0.25, 0.75],
            ticktext = ['< {}th Percentile {}'.format(percentile, dataframe_for_plot_sorted.columns[-1]), 
            '> {}th Percentile {}'.format(percentile, dataframe_for_plot_sorted.columns[-1])],
            lenmode = 'pixels', len = 100, yanchor = 'middle'), width = 1200)

        fig.write_image('Plots/{} {} {} Parallel Coordinates SVG.svg'.format(self.input_case, self.output_case, year))
        plt.savefig('Plots/{} {} {} 4 Most Important Inputs, gt = {}, Percentile = {}.png'.format(self.input_case, self.output_case, year, gt, percentile), dpi = 300)
        # fig.write_image("Plots/{} {} Parallel Plot Top 4.svg".format(self.input_case, self.output_case), width = 1300, height = 500)

    def time_series_clustering(self, n_clusters = 3, show = False, save = False, print_cluster_centers = False):
        from tslearn.clustering import TimeSeriesKMeans

        full_timeseries = self.get_y()

        full_timeseries_vals = full_timeseries.to_numpy()
        full_timeseries_vals_reshaped = full_timeseries_vals.reshape(full_timeseries_vals.shape[0], full_timeseries_vals.shape[1], 1)

        model = TimeSeriesKMeans(n_clusters = n_clusters, max_iter = 10)
        fit = model.fit(full_timeseries_vals_reshaped)
        cluster_classifications = model.predict(full_timeseries_vals_reshaped)

        if print_cluster_centers:
            for i in range(n_clusters):
                cluster_center = fit.cluster_centers_[i].ravel()
                for index, series in self.get_y(runs = True).iterrows():
                    # if index < 3:
                    #     print(np.round(series.iloc[1:].values, 7), np.round(cluster_center, 7))
                    #     print(np.equal(np.round(series.iloc[1:].values, 9), np.round(cluster_center, 9)))
                    if np.allclose(np.round(series.iloc[1:].values, 7), np.round(cluster_center, 7), rtol = 0, atol = 1e-6):
                        print('click:', i)
                        print('run #', series.iloc[0])
        
        if show:
            fig, ax = plt.subplots()
            colors = ['#AE9C45', '#6073B1', '#052955']
            for i in range(n_clusters):
                # the j variable is used to avoid redundant labeling
                for j, x in enumerate(full_timeseries_vals_reshaped[cluster_classifications == i]):
                    if j == 0:
                        ax.plot(full_timeseries.columns.values.astype(int), x.ravel(), alpha = 0.2, color = colors[i], label = 'Cluster {}'.format(i))
                    else:
                        ax.plot(full_timeseries.columns.values.astype(int), x.ravel(), alpha = 0.2, color = colors[i])
            # for i in range(n_clusters):
            #     ax.plot(full_timeseries.columns.values.astype(int), fit.cluster_centers_[i].ravel(), color = colors[i], marker = '*')
            ax.set_xlabel('Year')
            ax.set_ylabel('Fraction of Energy from Renewables')
            ax.legend()
            ax.set_title('{} {} Time Series Clustering'.format(self.input_case, self.output_case))

            plt.show()

        if save:
            fig, ax = plt.subplots()
            colors = ['#AE9C45', '#6073B1', 'red']
            for i in range(n_clusters):
                # the j variable is used to avoid redundant labeling
                for j, x in enumerate(full_timeseries_vals_reshaped[cluster_classifications == i]):
                    if j == 0:
                        ax.plot(full_timeseries.columns.values.astype(int), x.ravel(), alpha = 0.2, color = colors[i], label = 'Cluster {}'.format(i))
                    else:
                        ax.plot(full_timeseries.columns.values.astype(int), x.ravel(), alpha = 0.2, color = colors[i])
            for i in range(n_clusters):
                ax.plot(full_timeseries.columns.values.astype(int), fit.cluster_centers_[i].ravel(), color = colors[i], marker = '*')
            ax.set_xlabel('Year')
            ax.set_ylabel('Fraction of Energy from Renewables')
            ax.legend()
            ax.set_title('{} {} Time Series Clustering'.format(self.input_case, self.output_case))

            plt.savefig('Plots/{} {} Time Series Clustering.png'.format(self.input_case, self.output_case), dpi = 300)

        return full_timeseries_vals_reshaped, cluster_classifications

    def classification_with_time_series_clusters(self, max_depth = 5):
        from sklearn.tree import export_graphviz

        full_timeseries, cluster_classifications = self.time_series_clustering()
        cluster_classifications_set = sorted(set(cluster_classifications))

        X = self.get_X()
        tree = DecisionTreeClassifier(max_depth = max_depth).fit(X, cluster_classifications)
        dot_data = export_graphviz(tree,
                  feature_names = X.columns,
                  class_names = ['Cluster {}'.format(i) for i in cluster_classifications_set],
                  filled = True, rounded = True,
                  special_characters = True,
                   out_file = None,
                           )
        graph = graphviz.Source(dot_data, format = 'svg')
        graph.render(filename = 'Plots/' + self.input_case + '; ' + self.output_case + ', ' + 'Time Series Cluster CART')

    def parallel_plot_time_series_clusters(self):
        from sklearn.ensemble import RandomForestClassifier
        import matplotlib.pyplot as plt
        import plotly.graph_objects as go

        X = self.get_X()
        timeseries_vals, timeseries_classes = self.time_series_clustering()

        # train classifier
        clf = RandomForestClassifier(n_estimators = 200, min_samples_split = 8,
                        max_depth = 6, random_state = 0)
        model_fit = clf.fit(X, timeseries_classes)

        feature_importances = [estimator.feature_importances_ for estimator in model_fit.estimators_]
        avg_importances = sum(feature_importances)/len(feature_importances)
        labeled_importances = dict((name, score) for name, score in zip(X.columns, avg_importances))

        sorted_labeled_importances = sorted(labeled_importances.items(), key = lambda x: x[1], reverse = True)
        top_4 = sorted_labeled_importances[:4]

        # bar graph to display importances of top 4
        figbar, axbar = plt.subplots(figsize = (10, 7))
        axbar.bar(x = [i[0] for i in top_4], height = [k[1] for k in top_4])
        axbar.set_ylabel('Avg. Feature Importance')
        axbar.set_xlabel('Four Most Important Features')
        axbar.set_title('Most Important Features for {} {} Timeseries Clustering'.format(self.input_case, self.output_case))

        # parallel axis plot
        X_for_plot = X[[i[0] for i in top_4]]
        dataframe_for_plot = X_for_plot.copy()
        dataframe_for_plot['Cluster #'] = timeseries_classes
        dataframe_for_plot_sorted = dataframe_for_plot.sort_values(by = ['Cluster #'])

        dimensions_list = []
        for name, data in dataframe_for_plot_sorted[[i[0] for i in top_4] + ['Cluster #']].items():
            data_max = max(data)
            data_min = min(data)
            series_dict = dict(range = [data_min, data_max], label = name, values = data)
            dimensions_list.append(series_dict)

        fig = go.Figure(data = go.Parcoords(line = dict(color = dataframe_for_plot_sorted['Cluster #'],
                        colorscale = [[0, 'darkorange'], [1, 'blue']], showscale = True), dimensions = dimensions_list))

        fig.update_layout(
            plot_bgcolor = 'white',
            paper_bgcolor = 'white'
        )
        fig.update_traces(labelfont = {'size': 28}, tickfont = {'size': 20})

    def dist(self, year, gt = True, percentile = 70):
        from sklearn.ensemble import RandomForestClassifier
        import matplotlib.pyplot as plt

        X = self.get_X()
        if self.output_case == '2C_CHN_ENDOG_RENEW' or self.output_case == 'REF_CHN_EMISSIONS':
            y = self.get_y()
        else:
            y = self.get_y_by_year(year)

        perc = np.percentile(y, percentile)
        if gt == True:
            y_discrete = np.where(y > perc, 1, 0)
        elif gt == False:
            y_discrete = np.where(y < perc, 1, 0)
        else:
            raise ValueError("Supported values of gt are True and False")

        # train classifier
        clf = RandomForestClassifier(n_estimators = 200, min_samples_split = 8,
                        max_depth = 6, random_state = 0)
        model_fit = clf.fit(X, y_discrete)

        feature_importances = [estimator.feature_importances_ for estimator in model_fit.estimators_]
        avg_importances = sum(feature_importances)/len(feature_importances)
        labeled_importances = dict((name, score) for name, score in zip(X.columns, avg_importances))

        sorted_labeled_importances = sorted(labeled_importances.items(), key = lambda x: x[1], reverse = True)
        top_4 = sorted_labeled_importances[:4]

        # parallel axis plot
        X_for_plot = X[[i[0] for i in top_4]]
        dataframe_for_plot = pd.concat([X_for_plot, y], axis = 1).rename({str(year): 'Share'}, axis = 'columns')
        if gt == True:
            y_discrete_for_plot = np.where(dataframe_for_plot['Share'] > perc, 'High Renew', 'Low Renew')
        elif gt == False:
            y_discrete_for_plot = np.where(dataframe_for_plot['Share'] < perc, 'Low Renew', 'High Renew')

        dataframe_for_plot['y_discrete'] = y_discrete_for_plot
        dataframe_for_plot_sorted = dataframe_for_plot.sort_values(by = ['Share'])

        if self.output_case =='REF_CHN_EMISSIONS':
            dataframe_for_plot_sorted = dataframe_for_plot_sorted.rename({'Share': 'China CO2eq Emissions'}, axis = 'columns')

        for col in dataframe_for_plot_sorted.columns[:-1]:
            fig_dist, ax_dist = plt.subplots()
            sns.histplot(x = dataframe_for_plot_sorted[col], stat = 'density', fill = True, ax = ax_dist)
            ax_dist.set_axis_off()
            fig_dist.savefig('Plots\\{} hist svg.svg'.format(col).replace('/', '-'))

# a few plots that aren't included in the methods above
if __name__ == '__main__':
    # cwd = r'G:\My Drive\School\College\Senior Year\Senior Spring\UROP\Renewables Scenario Discovery for Paper'
    cwd = r'C:\Users\kenny\My Drive\School\College\Senior Year\Senior Spring\UROP\Renewables Scenario Discovery for Paper'
    os.chdir(cwd)

    def scenario_classification_scatterplot(total, renew, share, save = False):
        # rewrite this so args are objects, that will allow the figure to be renamed properly each time
        total_elec_cutoff = renew/np.percentile(share, 70)
        renew_cutoff = total * np.percentile(share, 70)

        above_cutoff_total = total[total < total_elec_cutoff]
        above_cutoff_renew = renew[renew > renew_cutoff]

        below_cutoff_total = total[total >= total_elec_cutoff]
        below_cutoff_renew = renew[renew <= renew_cutoff]

        fig_ref_scatter, ax_ref_scatter = plt.subplots()
        ax_ref_scatter.scatter(below_cutoff_renew, below_cutoff_total, color = 'red', label = 'Below 70th Percentile Share')
        ax_ref_scatter.scatter(above_cutoff_renew, above_cutoff_total, color = 'green', label = 'Above 70th Percentile Share')
        ax_ref_scatter.legend()
        ax_ref_scatter.set_xlabel('Renewable Energy Production, Twh')
        ax_ref_scatter.set_ylabel('Total Energy Production, Twh')
        ax_ref_scatter.set_title('Natural Scenario Classification Scheme, Policy Global 2050')

        if save:
            plt.savefig('2C_glb_scenario_classification_2050', dpi = 300)
        plt.show()

    # renew = InputOutputSD('GLB_GRP_NORM', '2C_GLB_RENEW').get_y_by_year(2050)
    # total = InputOutputSD('GLB_GRP_NORM', '2C_GLB_TOT').get_y_by_year(2050)
    # share = InputOutputSD('GLB_GRP_NORM', '2C_GLB_RENEW_SHARE').get_y_by_year(2050)

    def regional_importance_heatmap(year, r1: SD, r2: SD, r3: SD, r4: SD, gt = True, percentile = 70):

        inputs1 = r1.get_X().columns
        inputs2 = r2.get_X().columns
        inputs3 = r3.get_X().columns
        inputs4 = r4.get_X().columns

        all_inputs = inputs1.union(inputs2).union(inputs3).union(inputs4)
        importance_df = pd.DataFrame(data = [], index = all_inputs, columns = [r1.output_case, r2.output_case, r3.output_case, r4.output_case])
        for t in [r1, r2, r3, r4]:
            X = t.get_X()
            y = t.get_y_by_year(year)

            perc = np.percentile(y, percentile)
            if gt == True:
                y_discrete = np.where(y > perc, 1, 0)
            elif gt == False:
                y_discrete = np.where(y < perc, 1, 0)
            else:
                raise ValueError("Supported values of gt are True and False")

            clf = RandomForestRegressor(n_estimators = 200, min_samples_split = 8,
                            max_depth = 6, random_state = 0)
            model_fit = clf.fit(X, y)

            feature_importances = [estimator.feature_importances_ for estimator in model_fit.estimators_]
            avg_importances = sum(feature_importances)/len(feature_importances)
            labeled_importances = dict((name, score) for name, score in zip(X.columns, avg_importances))

            sorted_labeled_importances = sorted(labeled_importances.items(), key = lambda x: x[1], reverse = True)
            top = sorted_labeled_importances[:10]
            regional_importance_df = pd.DataFrame(data = [k[1] for k in top], index = [k[0] for k in top], columns = [t.output_case])
            importance_df = importance_df.combine_first(regional_importance_df)
        importance_df = importance_df.dropna(how = 'all').fillna(-1)
        print(importance_df)

        fig, ax = plt.subplots(figsize = (10, 10))
        labels = [i.split('_')[1] for i in importance_df.columns] # ['\n'.join(wrap(i, 10)) for i in importance_df.columns]
        heatmap(importance_df, xticklabels = True, yticklabels = True, mask = importance_df == -1, square = True, ax = ax)
        ax.set_xticklabels(labels, rotation = 'horizontal')
        plt.show()

    def output_dist():
        g = SD('GLB_RAW', '2C_GLB_RENEW_SHARE')
        y = g.get_y()
        y_5th = y.quantile(q = 0.05, axis = 0)
        y_med = y.median()
        y_95th = y.quantile(q = 0.95)

        # ax = sns.lineplot(x = y.columns, y = y_5th, color = 'red', label = '5th/9th Percentile')
        ax = sns.lineplot(x = y.columns, y = y_med, color = 'blue',  label = 'Median')
        # sns.lineplot(x = y.columns, y = y_95th, color = 'red', ax = ax)
        ax.fill_between(y.columns, y_5th, y_95th, color = 'red', alpha = 0.25)
        ax.set_ylabel(r'% Energy from Renewables')
        ax.set_xticks(ax.get_xticks()[0::2])
        ax.set_xlabel('Year')

        plt.savefig('2C_glb_renew_share_dist.png', dpi = 300)

    def regional_importance_over_time_figure(regions):

        def make_heatmap(SD_obj: SD, percentile = 70):
            years = np.arange(2020, 2105, 5)
            X = SD_obj.get_X()

            all_top_n = []
            importances_df = pd.DataFrame(index = sorted(X.columns))
            for year in years:
                y = SD_obj.get_y_by_year(year)
                y_discrete = np.where(y > np.percentile(y, percentile), 1, 0)
                importances, sorted_importances, top_n = SD_obj.rfc_model_wit_top_n(year, X, y_discrete, num_to_plot = 3)
                
                # sort in alphabetical order to ensure compatibility with dataframe index
                alphabetized_importances = sorted_importances.sort_index()
                importances_df[year] = alphabetized_importances

                all_top_n += top_n
            all_top_n = list(set(all_top_n))
            
            heatmap_df = importances_df.loc[all_top_n]
            cpalette = sns.color_palette("viridis", as_cmap = True)
            fig, ax = plt.subplots(tight_layout = True, figsize = (13, 13))
            sns.heatmap(heatmap_df, cmap = cpalette, square = True, ax = ax, cbar_kws = {'orientation': 'horizontal', 'pad': 0.05})
            ax.set_title('{}'.format(SD_obj.output_case))
            plt.savefig('Plots/{} {} timeseries heatmap.svg'.format(SD_obj.input_case, SD_obj.output_case))

        inputs = regions
        outputs = [('REF_' + x + '_RENEW_SHARE', '2C_' + x + '_RENEW_SHARE') for x in inputs]

        for inp, out in zip(inputs, outputs):
            ref_scenario = out[0]
            pol_scenario = out[1]
            SD_obj_ref = SD(inp, ref_scenario)
            SD_obj_pol = SD(inp, pol_scenario)

            make_heatmap(SD_obj_ref)
            make_heatmap(SD_obj_pol)

    regional_importance_over_time_figure(['USA', 'EUR', 'CHN'])