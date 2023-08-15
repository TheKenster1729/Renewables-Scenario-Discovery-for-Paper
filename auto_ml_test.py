import os
import pandas as pd
from eppa_exp1jr_scenario_discovery_main import SD
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from itertools import product
import tensorflow as tf

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
y = X.pop("WindBio")
perc = 70
num_to_plot = 4
target = np.where(y > np.percentile(y, perc), 1, 0)

# Initialize the structured data classifier.
clf = ak.StructuredDataClassifier(
    overwrite = True, max_trials = 5
)  # It tries 3 different models.
# Feed the structured data classifier with training data.
clf.fit(x= X, y = target, epochs = 10)
# # Predict with the best model.
predicted_y = clf.predict(X)
# # Evaluate the best model with testing data.
print(clf.evaluate(X, target))
print(clf.export_model().metrics_names)
print(clf.export_model().summary())