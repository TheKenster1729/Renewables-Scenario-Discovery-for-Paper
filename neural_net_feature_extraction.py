import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from eppa_exp1jr_scenario_discovery_main import SD
import eli5
from eli5.sklearn import PermutationImportance
from scikeras.wrappers import KerasClassifier, KerasRegressor
sns.set()

sd = SD('GLB_RAW', '2C_GLB_RENEW_SHARE')
X = sd.get_X()
year = sd.get_y_by_year(2050)
percentile = 70
percentile_val = np.percentile(year, percentile)
y = np.where(year > percentile_val, 1, 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the data for better training results
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a simple neural network
model = Sequential()
model.add(Dense(256, input_dim=X.shape[1], activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
classifier_model = KerasClassifier(build_fn = model)
classifier_model.fit(X, y)

perm = PermutationImportance(classifier_model, random_state=1).fit(X,y)
eli5.show_weights(perm, feature_names = X.columns.to_list())

'''
# Extract feature importances from the first layer's weights
feature_importances = np.mean(np.abs(model.layers[0].get_weights()[0]), axis=1)

# Normalize feature importances
feature_importances_normalized = feature_importances / np.sum(feature_importances)

# Select top five and get their names
named_feature_importances_dict = {"Feature": X.columns, "Importance": feature_importances_normalized}
named_feature_importances_df = pd.DataFrame(data = named_feature_importances_dict)
named_feature_importances_df = named_feature_importances_df.sort_values(by = "Importance", ascending = False)
print(named_feature_importances_df)

# Visualize feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=named_feature_importances_df["Feature"].iloc[:5], y=named_feature_importances_df["Importance"].iloc[:5])
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.show()
'''