import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# plotly.io.orca.config.executable = r'C:\Users\kenny\anaconda3\pkgs\plotly-orca-1.3.1-1\orca_app\orca.exe'

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
        self.hyperparams = None
        self.colors = ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000'] # colorblind-friendly palette from IBM
                    # blue         purple      magenta   orange      tan
        # "translates" between the abbreviations used in the code and the long forms used in the paper
        self.readability_dict = {'GLB_RAW': 'Global', 'REF': 'Share of Renewables Under Reference', 'POL': 'Share of Renewables Under Policy',
                                'CHN': 'China', 'USA': 'USA', 'EUR': 'Europe'}
        self.ref_or_pol = 'REF' if 'REF' in self.output_case else 'POL'

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

    def categorize(self, year, percentile = 70, gt = True):
        continuous_data = self.get_y_by_year(year)
        percentile_val = np.percentile(continuous_data, percentile)
        if gt == True:
            categorical_data = np.where(continuous_data >= percentile_val, 1, 0)
        elif gt == False:
            categorical_data = np.where(continuous_data < percentile_val, 1, 0)
        else:
            raise ValueError("Supported values of gt are True, False")

        return categorical_data

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
        from sklearn.tree import DecisionTreeClassifier, plot_tree
        import graphviz
        from textwrap import wrap

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

    def percentile_validation(self, year):
        X = self.get_X()
        percentiles_to_test = [70, 75, 80, 85, 90]
        validation_df = pd.DataFrame(columns = percentiles_to_test, index = X.columns)
        for perc in percentiles_to_test:
            y = self.categorize(year, percentile = perc)
            feature_importances, sorted_labeled_importances, top_n = self.rfc_model_with_top_n(X, y, num_to_plot = len(X.columns))
            validation_df[perc] = feature_importances.mean()

        validation_df.to_csv("Percentile Validation.csv")

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
        import operator
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from itertools import product
        from seaborn import heatmap

        # parameters to adjust
        min_samples_splits = [2, 3, 4, 8, 12, 16, 20]
        max_depths = [2, 4, 6, 8, 10, 12, 15]

        X = self.get_X()
        y = self.get_y_by_year(year)

        fig, axs = plt.subplots(2, 3, sharex = 'all', sharey = 'all')
        percentiles = [70]
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

    def rfc_model_with_top_n(self, X, y_discrete, num_to_plot = 4):
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators = 200, min_samples_split = 8,
                        max_depth = 6, random_state = 0)
        model_fit = clf.fit(X, y_discrete)

        feature_importances = pd.DataFrame([estimator.feature_importances_ for estimator in model_fit.estimators_], columns = self.get_X().columns)
        sorted_labeled_importances = feature_importances.mean().sort_values(ascending = False)
        top_n = sorted_labeled_importances.index[:num_to_plot].to_list()

        return feature_importances, sorted_labeled_importances, top_n

    def parallel_plot_most_important_inputs(self, year, gt = True, percentile = 70, num_to_plot = 4, save = False, show = True):
        import matplotlib.pyplot as plt
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
        feature_importances, sorted_labeled_importances, top_n = self.rfc_model_with_top_n(X, y_discrete)

        # bar graph to display importances of top 4
        figbar, axbar = plt.subplots(figsize = (15, 10))
        sorted_labeled_importances.iloc[:num_to_plot].plot(kind = 'bar', ax = axbar, rot = 0, color = self.colors[0])
        axbar.set_ylabel('Avg. Feature Importance')
        axbar.set_xlabel('Four Most Important Features')
        axbar.set_title('Most Important Features for {} {} {}'.format(self.readability_dict[self.input_case], self.readability_dict[self.ref_or_pol], year))
        print(sorted_labeled_importances)

        # parallel axis plot
        X_for_plot = X[top_n]
        dataframe_for_plot = pd.concat([X_for_plot, y], axis = 1).rename({str(year): 'Share'}, axis = 'columns')
        dataframe_for_plot_sorted = dataframe_for_plot.sort_values(by = ['Share'])
        print(dataframe_for_plot_sorted)
        if gt == True:
            y_discrete_for_plot = np.where(dataframe_for_plot_sorted['Share'] > perc, 1, 0)
        elif gt == False:
            y_discrete_for_plot = np.where(dataframe_for_plot_sorted['Share'] < perc, 1, 0)

        if self.output_case =='REF_CHN_EMISSIONS':
            dataframe_for_plot_sorted = dataframe_for_plot_sorted.rename({'Share': 'China CO2eq Emissions'}, axis = 'columns')

        color_scale = [(0.00, self.colors[1]), (0.5, self.colors[1]), (0.5, self.colors[4]),  (1.00, self.colors[4])]
        fig = px.parallel_coordinates(dataframe_for_plot_sorted[top_n + [dataframe_for_plot_sorted.columns[-1]]],
            color = y_discrete_for_plot, color_continuous_scale = color_scale,
            dimensions = top_n + [dataframe_for_plot_sorted.columns[-1]])
        fig.update_layout(coloraxis_colorbar = dict(
            title = dataframe_for_plot_sorted.columns[-1],
            tickvals = [0.25, 0.75],
            ticktext = ['< {}th Percentile {}'.format(percentile, dataframe_for_plot_sorted.columns[-1]),
            '> {}th Percentile {}'.format(percentile, dataframe_for_plot_sorted.columns[-1])],
            lenmode = 'pixels', len = 100, yanchor = 'middle'), width = 1200)

        if save:
            plt.savefig('Figures Second Draft/{} {} {} 4 Most Important Inputs, gt = {}, Percentile = {}.png'.format(self.input_case, self.output_case, year, gt, percentile), dpi = 300)
            fig.write_image('Plots/{} {} {} Parallel Coordinates SVG.svg'.format(self.input_case, self.output_case, year), engine = "orca")

        if show:
            plt.show()
            fig.show()
        # fig.write_image("Plots/{} {} Parallel Plot Top 4.svg".format(self.input_case, self.output_case), width = 1300, height = 500)

    def time_series_clustering(self, n_clusters = 3):
        from tslearn.clustering import TimeSeriesKMeans
        import seaborn as sns

        if not os.path.exists("TS Clusters"):
            os.mkdir("TS Clusters")

        filename = "{} {} {} clusters.csv".format(self.output_case, self.input_case, n_clusters)
        path_to_file = os.path.join("TS Clusters", filename)

        # if these timeseries already exist, simply return them
        if os.path.exists(path_to_file):
            dataframe_to_return = pd.read_csv(path_to_file)
        # if not, then make them
        else:
            print("Timeseries for this input/output/number of clusters combination not found, creating one now.")
            full_timeseries = self.get_y(runs = True)
            fit_model = TimeSeriesKMeans(n_clusters = n_clusters, max_iter = 10).fit(full_timeseries.loc[:, "2007":])

            # processing and so on
            dataframe_to_return_wide = full_timeseries.copy()
            dataframe_to_return_wide["Cluster #"] = fit_model.labels_
            dataframe_to_return_wide.rename(columns = {"Unnamed: 3": "Run #"}, inplace = True)
            dataframe_to_return = dataframe_to_return_wide.melt(id_vars = ["Run #", "Cluster #"], var_name = "Year", value_name = self.output_case)

            # rename clusters
            sns.lineplot(data = dataframe_to_return, x = "Year", y = self.output_case, hue = "Cluster #", units = "Run #", estimator = None)
            plt.title("Current Cluster Labels")
            plt.show()
            print("Rename clusters if desired")
            mapping = {}
            for i in range(n_clusters):
                renamed_cluster_number = input("Cluster {} becomes:".format(str(i)))
                mapping[i] = renamed_cluster_number
            dataframe_to_return["Cluster #"] = dataframe_to_return["Cluster #"].replace(mapping)
            dataframe_to_return.to_csv(path_to_file)

        return dataframe_to_return

    def plot_timeseries_clusters(self, n_clusters = 3, xlabel = None, ylabel = None, title = None, show = True, save = False):
        from tslearn.barycenters import euclidean_barycenter
        import seaborn as sns

        rcParams = {"font.sans-serif": "Palatino Linotype", "font.size": 15, "axes.titlesize": 28, "axes.labelsize": 21,
                    "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15, "figure.titlesize": 28, "patch.edgecolor": "gray",
                    "axes.edgecolor": "lightgray"}
        sns.set_theme(style = "white", rc = rcParams)

        clusters = self.time_series_clustering(n_clusters = n_clusters).sort_values(by = "Cluster #")

        # create plot from long-form dataframe with seaborn
        fig, ax = plt.subplots(figsize = (13, 9))
        sns.lineplot(data = clusters, x = "Year", y = self.output_case, hue = "Cluster #", units = "Run #", estimator = None, alpha = 0.33, palette = self.colors[::2], ax = ax)
        for i in range(n_clusters):
            cluster_points = clusters[clusters["Cluster #"] == "Cluster " + str(i + 1)]
            cluster_pivot = cluster_points.pivot(columns = "Year", index = "Run #", values = self.output_case)
            barycenter = euclidean_barycenter(cluster_pivot).ravel()
            sns.lineplot(x = cluster_pivot.columns, y = barycenter, color = self.colors[::2][i], marker = "o", ax = ax)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        if save:
            fig.savefig(fname = "Plots/{} {} {} clusters.png".format(self.input_case, self.output_case, n_clusters), dpi = 400)
        if show:
            plt.show()

    def get_timeseries_cluster_fractions(self, n_clusters = 3):        
        time_series_clusters = self.time_series_clustering(n_clusters = n_clusters)
        time_series_clusters_proportions = time_series_clusters["Cluster #"].value_counts(normalize = True)
        print("Cluster Fractions for {}, n_clusters = {}\n".format(self.output_case, n_clusters))
        print(time_series_clusters_proportions)

    def get_cluster_barycenters(self, n_clusters = 3):
        from tslearn.barycenters import euclidean_barycenter

        time_series_clusters = self.time_series_clustering(n_clusters = n_clusters)
        barycenter_df = pd.DataFrame()
        barycenter_df["Year"] = self.get_y().columns
        for i in range(n_clusters):
            cluster_points = time_series_clusters[time_series_clusters["Cluster #"] == "Cluster " + str(i + 1)]
            cluster_pivot = cluster_points.pivot(columns = "Year", index = "Run #", values = self.output_case)
            barycenter = euclidean_barycenter(cluster_pivot).ravel()
            barycenter_df["Cluster " + str(i + 1)] = barycenter
        print("Barycenters for {}".format(self.output_case))
        print(barycenter_df)

    def parallel_plot_time_series_clusters(self):
        from sklearn.ensemble import RandomForestClassifier
        import matplotlib.pyplot as plt
        import plotly.express as px

        X = self.get_X()
        time_series_df = self.time_series_clustering()
        just_cluster_labels = time_series_df.pivot(index = "Run #", columns = "Year", values = "Cluster #")[2007].values # any year will do, cluster labels are the same across years

        # train classifier
        _, sorted_labeled_importances, top_4 = self.rfc_model_with_top_n(X, just_cluster_labels)

        # bar graph to display importances of top 4
        figbar, axbar = plt.subplots(figsize = (10, 7))
        sorted_labeled_importances[top_4].plot(kind = 'bar', ax = axbar, rot = 0)
        axbar.set_ylabel('Avg. Feature Importance')
        axbar.set_xlabel('Four Most Important Features')
        axbar.set_title('Most Important Features for {} {} Timeseries Clustering'.format(self.input_case, self.output_case))
        plt.show()
        # figbar.savefig('Plots/{} {} TSC Important Inputs.png'.format(self.input_case, self.output_case), dpi = 300)

        # parallel axis plot
        X_for_plot = X[top_4]
        dataframe_for_plot = X_for_plot.copy()
        dataframe_for_plot['Cluster #'] = just_cluster_labels
        dataframe_for_plot_sorted = dataframe_for_plot.sort_values(by = ['Cluster #'])

        fig = px.parallel_coordinates(dataframe_for_plot_sorted[top_4 + [dataframe_for_plot_sorted.columns[-1]]],
            color = dataframe_for_plot_sorted.columns[-1], color_continuous_scale = [(0.00, "#648FFF"), (0.33, "#648FFF"), (0.33, "#FE6100"),  (0.66, "#FE6100"), (0.66, "#DC267F"), (1, "#DC267F")],
            dimensions = top_4)
        fig.update_layout(coloraxis_colorbar = dict(
            title = dataframe_for_plot_sorted.columns[-1],
            tickvals = [0.33, 0.66 + 0.66/2, 0.66*2 + 0.66/2],
            ticktext = ['Cluster 1', 'Cluster 2', 'Cluster 3'],
            lenmode = 'pixels', len = 100, yanchor = 'middle'), width = 800)
        fig.show()
        # fig.write_image('Plots/{} {} Time Series Clustering Importances.png'.format(self.input_case, self.output_case), scale = 2)

    def dist(self, year, gt = True, percentile = 70):
        from sklearn.ensemble import RandomForestClassifier
        import seaborn as sns

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

        for col in dataframe_for_plot_sorted.columns:
            fig_dist, ax_dist = plt.subplots()
            sns.histplot(x = dataframe_for_plot_sorted[col], stat = 'density', fill = True, ax = ax_dist)
            ax_dist.set_axis_off()
            fig_dist.savefig('Plots\\{} hist svg.svg'.format(col).replace('/', '-'))

    def predict_individual_clusters(self, n_clusters = 3, save = False, show = True):
        import seaborn as sns
        rcParams = {"font.sans-serif": "Palatino Linotype", "font.size": 15, "axes.titlesize": 28, "axes.labelsize": 21,
                    "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15, "figure.titlesize": 28, "patch.edgecolor": "gray",
                    "axes.edgecolor": "lightgray"}
        sns.set_theme(style = "white", rc = rcParams)

        time_series_df = self.time_series_clustering(n_clusters = n_clusters)
        cluster_options = np.unique(time_series_df["Cluster #"])
        title = "Reference" if self.ref_or_pol == 'REF' else "Policy"
        for i, cluster in enumerate(cluster_options):
            X = self.get_X()
            just_cluster_labels = time_series_df.pivot(index = "Run #", columns = "Year", values = "Cluster #")[2007].values
            y = np.where(cluster == just_cluster_labels, 1, 0)

            # bar graph to display importances of top 4
            _, sorted_labeled_importances, top_4 = self.rfc_model_with_top_n(X, y)
            figbar, axbar = plt.subplots(1, 2, figsize = (16, 9))
            sorted_labeled_importances[top_4].plot(kind = 'bar', ax = axbar.flat[0], rot = 0, color = self.colors[::2][i])
            axbar.flat[0].set_ylabel('Avg. Feature Importance')
            axbar.flat[0].set_xlabel('Four Most Important Features')
            figbar.suptitle('Most Important Features for to Predict {}, Global Share of Renewables Under {}'.format(cluster, title))
            # figbar.savefig('Plots/{} {} TSC Important Inputs.png'.format(self.input_case, self.output_case), dpi = 300)

            # also add a plot of this cluster for context
            cluster_plot_df = time_series_df.copy()
            cluster_plot_df["Hue"] = np.where(cluster_plot_df["Cluster #"] == cluster, 1, 0) # create a column to track current cluster
            cluster_plot_df = cluster_plot_df.sort_values(by = "Cluster #") # sort for color consistency
            palette = ["lightgray", self.colors[::2][i]]
            sns.lineplot(data = cluster_plot_df, x = "Year", y = self.output_case, hue = "Hue", units = "Run #", estimator = None, legend = False, palette = palette)
            axbar.flat[1].set_ylabel("Fraction of Renewables")
            axbar.flat[1].set_xlabel("Year")
            figbar.suptitle("Most Important Features to Predict {}, {} {}".format(cluster, self.readability_dict[self.input_case], self.readability_dict[self.ref_or_pol]))

            if save:
                plt.savefig("Figures Second Draft/{}_{}_clusters_pol.png".format(cluster, n_clusters), dpi = 400)
            if show:
                plt.show()

    def permutation_importance(self, year):
        from sklearn.inspection import permutation_importance
        from sklearn.linear_model import LogisticRegression

        X = self.get_X()
        y_discrete = self.categorize(year)

        # logistic regression R^2 for reference
        logit = LogisticRegression().fit(X, y_discrete)
        rsquare = logit.score(X, y_discrete)

        clf = RandomForestClassifier(n_estimators = 200, min_samples_split = 8,
                        max_depth = 6, random_state = 0).fit(X, y_discrete)
        score = clf.score(X, y_discrete)
        print("Accuracy:", f"{score: .2f}")

        perm_importance = permutation_importance(clf, X, y_discrete, n_repeats = 30)

        num_important = 0
        for i in perm_importance.importances_mean.argsort()[::-1]:
            if perm_importance.importances_mean[i] - 3 * perm_importance.importances_std[i] > 0:
                print(f"{X.columns[i]:<15}"
                f"{perm_importance.importances_mean[i]:.3f}"
                f" +/- {perm_importance.importances_std[i]:.3f}")
                num_important += 1

        print(num_important)

# a few plots that aren't included in the methods above
if __name__ == '__main__':
    try:
        cwd = r'C:\Users\kenny\My Drive\School\College\Senior Year\Senior Spring\UROP\Renewables Scenario Discovery for Paper'
        os.chdir(cwd)
    except FileNotFoundError:
        cwd = r'G:\My Drive\School\College\Senior Year\Senior Spring\UROP\Renewables Scenario Discovery for Paper'
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
        ax_ref_scatter.grid(True)
        ax_ref_scatter.scatter(below_cutoff_renew, below_cutoff_total, color = '#648FFF', label = 'Below 70th Percentile Share')
        ax_ref_scatter.scatter(above_cutoff_renew, above_cutoff_total, color = '#FFB000', label = 'Above 70th Percentile Share')
        ax_ref_scatter.legend()
        ax_ref_scatter.set_xlabel('Renewable Energy Production, Twh')
        ax_ref_scatter.set_ylabel('Total Energy Production, Twh')
        ax_ref_scatter.set_title('Scenario Classification Scheme - 70th Percentile, Policy Global 2050')

        if save:
            plt.savefig('Figures Second Draft/2C GLB Classification.png', dpi = 300)
        plt.show()

    def regional_importance_heatmap(year, r1: SD, r2: SD, r3: SD, r4: SD, gt = True, percentile = 70):
        from seaborn import heatmap

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
        import seaborn as sns

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
        plt.show()

        # plt.savefig('2C_glb_renew_share_dist.png', dpi = 300)

    def regional_importance_over_time_figure(regions):
        import seaborn as sns

        def make_heatmap(SD_obj: SD, percentile = 70):
            years = np.arange(2025, 2105, 5)
            X = SD_obj.get_X()

            all_top_n = []
            importances_df = pd.DataFrame(index = sorted(X.columns))
            for year in years:
                y = SD_obj.get_y_by_year(year)
                y_discrete = np.where(y > np.percentile(y, percentile), 1, 0)
                importances, sorted_importances, top_n = SD_obj.rfc_model_wit_top_n(X, y_discrete, num_to_plot = 3)

                # sort in alphabetical order to ensure compatibility with dataframe index
                alphabetized_importances = sorted_importances.sort_index()
                importances_df[year] = alphabetized_importances

                all_top_n += top_n
            all_top_n = list(set(all_top_n))

            heatmap_df = importances_df.loc[all_top_n]
            cpalette = sns.color_palette("viridis", as_cmap = True)
            fig, ax = plt.subplots(tight_layout = True, figsize = (20, 20))
            sns.heatmap(heatmap_df, cmap = cpalette, square = True, ax = ax, cbar_kws = {'orientation': 'horizontal', 'pad': 0.05})
            ax.set_title('{}'.format(SD_obj.output_case))
            plt.savefig('Plots/{} {} timeseries heatmap.png'.format(SD_obj.input_case, SD_obj.output_case), dpi = 300)

        inputs = regions
        outputs = [('REF_' + x + '_RENEW_SHARE', '2C_' + x + '_RENEW_SHARE') for x in inputs]

        for inp, out in zip(inputs, outputs):
            ref_scenario = out[0]
            pol_scenario = out[1]
            SD_obj_ref = SD(inp, ref_scenario)
            SD_obj_pol = SD(inp, pol_scenario)

            make_heatmap(SD_obj_ref)
            make_heatmap(SD_obj_pol)

    def global_outputs_clustering(year = 2050):
        import plotly.graph_objects as go

        input_sd_obj = SD("GLB_RAW", "2C_GLB_RENEW_SHARE")
        df_to_plot = pd.read_excel("Full Data for Paper.xlsx", sheet_name = "2C_GLB_renew_inputs_{}".format(str(year)), header = 2)
        crashed_runs = [3, 14, 74, 116, 130, 337] # these will have to be updated for any crashed runs in other ensembles
        df_to_plot = df_to_plot[~df_to_plot["Run #"].isin(crashed_runs)]

        time_series_df = input_sd_obj.time_series_clustering()
        just_cluster_labels = time_series_df.pivot(index = "Run #", columns = "Year", values = "Cluster #")[2007].values
        share = time_series_df.pivot(index = "Run #", columns = "Year", values = input_sd_obj.output_case)[year].values
        df_to_plot["{} Share".format(year)] = share
        df_to_plot["Cluster #"] = just_cluster_labels
        df_to_plot = df_to_plot.sort_values(by = "Cluster #")

        colors = input_sd_obj.colors[::2]
        color_scale = [(0.00, colors[0]), (0.33, colors[0]), (0.33, colors[1]),  (0.66, colors[1]), (0.66, colors[2]),  (1, colors[2])]
        dimensions = [dict(label = name, values = series) for (name, series) in df_to_plot.items()][1:-1]

        fig = go.Figure(
            data = go.Parcoords(
                line = dict(color = df_to_plot["Cluster #"].apply(lambda x: int(x[-1])), # this is a somewhat sloppy workaround to how plotly does parallel axis plots, and it will break for n_clusters > 9
                            colorscale = color_scale,
                            colorbar = dict(showticklabels = True, tickvals = [1.33, 2, 2.66], ticktext = ["Cluster #1", "Cluster #2", "Cluster #3"], title = "Policy Global Outputs with Clusters, {}".format(year), len = 0.33),
                            showscale = True),
                dimensions = list(
                    dimensions
                )
            )
        )

        # title = "test"
        # fig.update_layout(coloraxis_colorbar = dict(
        #     title = title,
        #     tickvals = [0.33, 0.66 + 0.66/2, 0.66*2 + 0.66/2],
        #     ticktext = ['Cluster 1', 'Cluster 2', 'Cluster 3'],
        #     lenmode = 'pixels', len = 100, yanchor = 'middle'), width = 1200)

        # fig.show()
        fig.write_image('Plots/Policy Global Outputs with Clusters {}.svg'.format(year), width = 1200)

    def heatmap_for_all_regions(save = False, show = True, policy = "ref"):
        from seaborn import heatmap
        import seaborn as sns

        def rfc_model_with_top_n(input_df, output_df, year, num_to_plot = 4, percentile = 70):

            X = input_df[input_df.columns[1:]]
            y = output_df[year]
            perc = np.percentile(y, percentile)
            y_discrete = np.where(y > perc, 1, 0)

            clf = RandomForestClassifier(n_estimators = 200, min_samples_split = 8,
                            max_depth = 6, random_state = 0)
            model_fit = clf.fit(X, y_discrete)

            feature_importances = pd.DataFrame([estimator.feature_importances_ for estimator in model_fit.estimators_], columns = X.columns)
            sorted_labeled_importances = feature_importances.mean().sort_values(ascending = False)
            top_n = sorted_labeled_importances.index[:num_to_plot].to_list()

            return feature_importances, sorted_labeled_importances, top_n

        def make_heatmap(intput_df, output_df, title, percentile = 70):
            years = np.arange(2025, 2105, 5)

            importances_df = pd.DataFrame(index = sorted(input_df.columns))
            for year in years:
                year = str(year)
                importances, sorted_importances, top_n = rfc_model_with_top_n(input_df, output_df, year, num_to_plot = 5)

                alphabetized_importances = sorted_importances.sort_index()
                importances_df[year] = alphabetized_importances

            sum_of_importances = importances_df.sum(axis = 1).sort_values(ascending = False)
            top_5_importances_by_sum = sum_of_importances[:5].index

            heatmap_df = importances_df.loc[top_5_importances_by_sum]

            return heatmap_df
            # plt.savefig('Plots/{} {} timeseries heatmap.png'.format(SD_obj.input_case, SD_obj.output_case), dpi = 300)

        # main_fig, main_ax = plt.subplots(4, 5, sharex = True)

        regions = ['GLB', 'USA', 'CHN', 'EUR', 'CAN', 'MEX', 'JPN', 'ANZ', 'ROE', 'RUS', 'ASI', 'IND', 'BRA', 'AFR', 'MES', 'LAM',
                        'REA', 'KOR', 'IDZ']
        specific_columns = ['BA:BB', 'BC:BF', 'BG:BJ', 'BK:BN', 'BO:BR', 'BS:BV', 'BW:BZ', 'CA:CD', 'CE:CH', 'CI:CL', 'CM:CP',
                        'CQ:CT', 'CU:CX', 'CY:DB', 'DC:DF', 'DG:DJ', 'DK:DN', 'DO:DR', 'DS:DV']
        regions = [regions[0]]
        specific_columns = [specific_columns[0]]
        assert len(regions) == len(specific_columns)

        colors = ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000']
        if policy == "ref":
            color = colors[2]
        elif policy == "2C":
            color = colors[1]
        else:
            raise ValueError("Supported values for policy are ref and 2C")
        
        for region, columns in zip(regions, specific_columns):
            fig, ax = plt.subplots(figsize = (10, 8))
            input_df = pd.read_excel("Full Data for Paper.xlsx", "samples", header = 2, nrows = 400, usecols = 'A:BB, {}'.format(columns))
            output_df = pd.read_excel("Full Data for Paper.xlsx", "{}_{}_renew_share".format(policy, region), header = 0, nrows = 400, usecols = 'E:X')

            heatmap_df = make_heatmap(input_df, output_df, region)
            cpalette = sns.light_palette(color, as_cmap = True)
            sns.heatmap(heatmap_df, cmap = cpalette, vmin = 0, vmax = 0.4, square = True, ax = ax, cbar_kws = {"orientation": "horizontal"}, cbar = True)
            ax.set_title('{}'.format(region))

            if save:
                fig.savefig("Global Heatmap/{} region heatmap {}.svg".format(region, policy))
            if show:
                plt.show()

    def k_means(show = True):
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans

        policy = "REF"
        colors = ['#648FFF', '#785EF0', '#FE6100', '#FFB000', '#DC267F']
        if policy == '2C':
            X = pd.read_excel("Full Data for Paper.xlsx", "{}_GLB_renew_inputs".format(policy), header = 2).drop(columns = "run#").drop(index = [2, 13, 73, 115, 129, 336])
        else:
            X = pd.read_excel("Full Data for Paper.xlsx", "{}_GLB_renew_inputs".format(policy), header = 2).drop(columns = "run#")
        pca_fit = PCA(n_components = 2).fit(X)
        pca = pca_fit.transform(X)
        components = abs(pca_fit.components_)

        n_clusters = 3
        kmeans = KMeans(n_clusters = n_clusters, random_state = 42)
        clusters = kmeans.fit_predict(X)
        colors_from_clusters = {key: colors[key] for key in np.unique(clusters)}
        colors_for_plot = np.vectorize(colors_from_clusters.get)(clusters)

        fig, ax = plt.subplots()
        for i in range(n_clusters):
            condition = colors_for_plot == colors[i]
            ax.scatter(pca[:, 0][condition], pca[:, 1][condition], c = colors_for_plot[condition], label = 'Cluster {}'.format(i + 1))
        ax.legend()
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        if show:
            plt.show()

    def cost_vs_production():
        colors = ['#648FFF', '#785EF0', '#FE6100', '#FFB000', '#DC267F']
        policy = 'REF'
        if policy == '2C':
            sd = SD('GLB_RAW', '2C_GLB_RENEW_SHARE')
            X = sd.get_X()
            y_df = pd.read_excel("Full Data for Paper.xlsx", "{}_GLB_renew_inputs".format(policy), header = 2).drop(columns = "run#").drop(index = [2, 13, 73, 115, 129, 336])
            y_nuc = y_df['2100 Nuclear (TWh)']
            y_wind = y_df['2100 Renewables (TWh)']
        else:
            sd = SD('GLB_RAW', 'REF_GLB_RENEW_SHARE')
            X = sd.get_X()
            y_df = pd.read_excel("Full Data for Paper.xlsx", "{}_GLB_renew_inputs".format(policy), header = 2).drop(columns = "run#")
            y_nuc = y_df['Nuclear (TWh)']
            y_wind = y_df['Renewables (TWh)']

        percentile_threshold_wind = np.percentile(y_wind, 70)
        fig, ax = plt.subplots(1, 2)
        ax[0].scatter(X['Nuclear'], y_nuc, label = 'Nuclear', c = colors[0])
        ax[0].scatter(X['WindBio'], y_wind, label = 'Wind', c = colors[1])
        ax[0].hlines(percentile_threshold_wind, xmin = X['wind'].min(), xmax = X['wind'].max(), color = colors[2], label = '70th Percentile Renew')
        ax[0].legend()

        totalax = ax[1].twinx()
        renew_share_2100 = sd.get_y_by_year(2100)
        ax[1].scatter(X['wind'], renew_share_2100, color = colors[0], label = 'Share')
        totalax.scatter(X['wind'], y_wind, label = 'Total Renew', color = colors[1])
        ax[1].legend()
        plt.show()

    def change_cluster_numbers(path_to_cluster_file, mapping):
        """WARNING: THIS FUNCTION IS DANGEROUS, MAKE SURE YOU KNOW WHAT YOU'RE DOING BEFORE RUNNING IT
        
        This function will rename the clusters produced by the time series clustering function. The use case for this function
        is interpretability of figures: the name of the cluster doesn't matter, but some orderings of clusters make more sense
        when these clusters are plotted. In other words, the role of this function is completely aesthetic.

        However, because the clusters will be renamed, any new figures generated after the re-naming process will describe the
        same clusters, but the names will be incompatible. Therefore, it is important to run this function immediately after
        generating a new set of clusters to avoid any future issues, because there is no surefire way to restore the initial cluster labels.

        Args:
            path_to_cluster_file (str): path to csv file containing the cluster numbers
        
        Returns:
            None
        """
        pass

    # regional_importance_over_time_figure(['USA', 'EUR', 'CHN'])
    # parallel_time_series = SD('GLB_RAW', 'REF_GLB_RENEW_SHARE')
    # parallel_time_series.parallel_plot_time_series_clusters()
    # china = SD('CHN_ENDOG_EMISSIONS', 'CHN_ENDOG_EMISSIONS')
    # china.parallel_plot_most_important_inputs(2050)
    # nuke = SD('GLB_RAW', '2C_GLB_RENEW_SHARE')
    # nuke.parallel_plot_most_important_inputs(2090)
    sd = SD('GLB_RAW', '2C_GLB_RENEW_SHARE')
    # sd.get_cluster_barycenters()
    global_outputs_clustering(year = 2100)
    # sd.plot_timeseries_clusters(xlabel = "Year", ylabel = "Fraction of Energy from Renewables", title = 
    #                             "Time Series Clustering, Global Share of Renewables Under Policy", show = False, save = True)

    # sd.permutation_importance(2050)
    # heatmap_for_all_regions()
    # sd.time_series_clustering(show = False, export = True)
    # sd.predict_individual_clusters()
    # sd = SD("GLB_RAW", "REF_GLB_RENEW_SHARE")
    # sd.parallel_plot_most_important_inputs(2050)
    # sd.predict_individual_clusters()
    # global_outputs_clustering(case = "POL")
    # heatmap_for_all_regions(show = False, save = True, policy = "2C")
    # total = SD("GLB_RAW", "2C_GLB_TOT").get_y_by_year(2050)
    # renew = SD("GLB_RAW", "2C_GLB_RENEW").get_y_by_year(2050)
    # share = SD("GLB_RAW", "2C_GLB_RENEW_SHARE").get_y_by_year(2050)

    # scenario_classification_scatterplot(total, renew, share, save = True)