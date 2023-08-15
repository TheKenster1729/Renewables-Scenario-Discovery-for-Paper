import os
import pandas as pd
from eppa_exp1jr_scenario_discovery_main import SD
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from itertools import product

try:
    cwd = r'C:\Users\kenny\My Drive\School\College\Senior Year\Senior Spring\UROP\Renewables Scenario Discovery for Paper'
    os.chdir(cwd)
except FileNotFoundError:
    cwd = r'G:\My Drive\School\College\Senior Year\Senior Spring\UROP\Renewables Scenario Discovery for Paper'
    os.chdir(cwd)

input_case = 'GLB_RAW'
output_case = 'REF_GLB_RENEW_SHARE'
data = SD(input_case, output_case)
X = data.get_X()
y = X.pop('wind')
perc = 70
num_to_plot = 4
target = np.where(y > np.percentile(y, perc), 1, 0)
param_grid = {'max_depth': [2, 4, 6, 8, 10, 15], 'min_samples_leaf': [2, 8, 16, 20]}

def hyperparam(X, target, params_dict):
    hyperparameterization = GridSearchCV(RandomForestClassifier(), params_dict).fit(X, target)
    best_estimator = hyperparameterization.best_estimator_.fit(X, target)

    return best_estimator

def rfc_with_top_n(X, target, model):
    feature_importances = pd.DataFrame([estimator.feature_importances_ for estimator in model.estimators_], columns = X.columns)
    sorted_labeled_importances = feature_importances.mean().sort_values(ascending = False)
    top_n = sorted_labeled_importances.index[:num_to_plot].to_list()

    return feature_importances, sorted_labeled_importances, top_n

model = hyperparam(X, target, param_grid)
print(model)
feature_importances, sorted_labeled_importances, top_n = rfc_with_top_n(X, target, model)

# bar graph to display importances of top 4
figbar, axbar = plt.subplots(figsize = (16, 7))
sorted_labeled_importances.iloc[:num_to_plot].plot(kind = 'bar', ax = axbar, rot = 0)
axbar.set_ylabel('Avg. Feature Importance')
axbar.set_xlabel('Four Most Important Features')
axbar.set_title('Input SD for {}, {}'.format(input_case, output_case))

# parallel axis plot
# X_for_plot = X[top_n]
# dataframe_for_plot = pd.concat([X_for_plot, y], axis = 1)
# dataframe_for_plot_sorted = dataframe_for_plot.sort_values(by = ['Share'])
plt.show()