import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# Read the excel file
data = pd.read_excel('get.xlsx')

# Extract the features and labels
X = data.drop(['True label'], axis=1)
y = data[['True label']].values.ravel()

# Define the parameter grid to search
param_grid = {'random_state': [0.6, 0.7, 1],
              'n_estimators': [100, 200, 300],
              'max_depth': [4, 6, 8],
              'min_samples_leaf': [3, 5, 9],
              'min_samples_split': [3, 4, 5]
             }

# Initialize the classifier
gb = GradientBoostingClassifier()

# Perform grid search
grid_search = GridSearchCV(gb, param_grid, cv=5)
grid_search.fit(X, y)

# Print the best parameters and the corresponding score
print("Best parameters: {}".format(grid_search.best_params_))
print("Best score: {:.2f}".format(grid_search.best_score_))