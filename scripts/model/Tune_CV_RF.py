# Input: model, random search parameters, data
# CV
# Random search
# output: return accuracy metrix and best parameters

# Input: model, random search parameters, data
# CV
# Grid search
# output: return accuracy metrix and best parameters

# Write CV function - reuse code

# But let's write one script for now
# Compartmentalize

import pickle
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV# import matplotlib.pylab as pl
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn import svm
# from sklearn.ensemble import RandomForestClassifier

# from sklearn.tree import export_graphviz

###############################################################################
# Set up folders and variables
###############################################################################
filename = 'neural_network_v1.py'

save_folder = '../../data/data_for_model/'
load_folder = '../../data/data_for_model/'
model_folder = '../../data/model/'
figure_folder = '../../images/model/nn1/'

###############################################################################
# Load
###############################################################################

with open(save_folder + 'train_test_sel_features.pkl', 'rb') as f:
    [features,
        excluded_features,
        x_train_sel_df,
        x_test_sel_df,
        y_train_sel_df,
        y_test_sel_df,
        train_idx,
        test_idx,
        train_flag,
        test_flag] = pickle.load(f)

###############################################################################
# Set up hyperparameters
###############################################################################

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

###############################################################################
# Set up functions
###############################################################################

def evaluate(model, test_features, test_labels):
    y_pred_dev = model.predict(test_features)
    accuracy_dev = accuracy_score(test_labels, y_pred_dev)
    print('Model Performance')
    print("Accuracy Dev: %.2f%%" % (accuracy_dev * 100.0))

    return accuracy_dev


###############################################################################
# Set up CV random search
###############################################################################


# Use the random grid to search for best hyperparameters
# First create the base model to tune
model = RandomForestClassifier()
# Random search of parameters, using 5 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(xgb_Trainset_X, xgb_Trainset_y)


rf_random.best_params_

###############################################################################
# Evaluate CV random search
###############################################################################

base_model = RandomForestClassifier(n_estimators = 10, random_state = 42)

base_model.fit(xgb_Trainset_X, xgb_Trainset_y)
base_accuracy = evaluate(base_model, xgb_Devset_X, xgb_Devset_y)
# Model Performance
# Accuracy Dev: 85.07%
best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, xgb_Devset_X, xgb_Devset_y)
# Model Performance
# Accuracy Dev: 91.54%
print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))
# Improvement of 7.60%.
print('Best random model:')
print(best_random)


###############################################################################
# Grid search around the best parameters
###############################################################################


# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [False],
    'max_depth': [80, 85, 90, 95, 100],
    'max_features': ['sqrt'],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [4, 5, 6],
    'n_estimators': [700, 800, 900]
}

model = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = model, param_grid = param_grid,
                          cv = 5, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(xgb_Trainset_X, xgb_Trainset_y)
print(grid_search.best_params_)
best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, xgb_Devset_X, xgb_Devset_y)
# Model Performance
# Accuracy = 91.04%
print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))
# Improvement of 7.02%.

# Final result
# best_random is better but not much different so trust CV and go with best_grid

# RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
#                        max_depth=95, max_features='sqrt', max_leaf_nodes=None,
#                        min_impurity_decrease=0.0, min_impurity_split=None,
#                        min_samples_leaf=1, min_samples_split=5,
#                        min_weight_fraction_leaf=0.0, n_estimators=800,
#                        n_jobs=None, oob_score=False, random_state=None,
#                        verbose=0, warm_start=False)


###############################################################################
# Final test of the model
###############################################################################

model = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
                        max_depth=95, max_features='sqrt', max_leaf_nodes=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=1, min_samples_split=5,
                        min_weight_fraction_leaf=0.0, n_estimators=800,
                        n_jobs=None, oob_score=False, random_state=None,
                        verbose=0, warm_start=False)
# Fit the grid search to the data
model.fit(xgb_Trainset_X, xgb_Trainset_y)

with open('RF_hyper_tune_bestmodel_full_fea.pickle', 'wb') as handle:
    pickle.dump([model], handle)

with open('RF_hyper_tune_bestmodel_full_fea.pickle', 'rb') as handle:
    [model] = pickle.load(handle)


y_pred_train = model.predict(xgb_Trainset_X)
y_pred_dev = model.predict(xgb_Devset_X)
y_pred_test = model.predict(xgb_Testset_X)


# evaluate predictions
accuracy_train = accuracy_score(xgb_Trainset_y, y_pred_train)
print("Accuracy Train: %.2f%%" % (accuracy_train * 100.0))

accuracy_dev = accuracy_score(xgb_Devset_y, y_pred_dev)
print("Accuracy Dev: %.2f%%" % (accuracy_dev * 100.0))

accuracy_test = accuracy_score(xgb_Testset_y, y_pred_test)
print("Accuracy Test: %.2f%%" % (accuracy_test * 100.0))

# =============================================================================
# Accuracy Train: 100.00%
# Accuracy Dev: 91.04%
# Accuracy Test: 88.56%
# =============================================================================




###############################################################################
# Load data
###############################################################################

# # Source
# Load_folder = 'C:/Users/diman/OneDrive/Work_temp/Insight Fellows Demo 2019/WorkSave/'
# homefolder = 'C:/Users/diman/OneDrive/Work_temp/Insight/NHANESDietStudyPractice'
# filename = 'DietData_NHANES_SVM.py'

# os.chdir(homefolder)

# with open(Load_folder + 'df_diet.pkl', 'rb') as f:
#     [df_diet, \
#         df_diet_total_nonan, \
#         df_diet_y, \
#         df_diet_y_raw, \
#         class_0_flag, \
#         class_1_flag, \
#         class_2_flag, \
#         Class_0_df_index, \
#         Class_1_df_index, \
#         Class_2_df_index, \
#         df_index_Devset, \
#         df_index_Testset, \
#         df_index_Trainset] = pickle.load(f)

# with open('xgb_feature_selection.pickle', 'rb') as handle:
#         [features,
#         excluded_feature,
#         xgb_Trainset_reduc_y,
#         xgb_Trainset_reduc_X,
#         xgb_Devset_reduc_y,
#         xgb_Devset_reduc_X,
#         xgb_Testset_reduc_y,
#         xgb_Testset_reduc_X,] = pickle.load(handle)

# with open('xgb_train_dev_test_full_features.pickle', 'rb') as handle:
#         [xgb_Trainset_y,
#         xgb_Trainset_X,
#         xgb_Devset_y,
#         xgb_Devset_X,
#         xgb_Testset_y,
#         xgb_Testset_X,] = pickle.load(handle)






###############################################################################
# RF Specific
###############################################################################


###############################################################################
# Single tree visualization
###############################################################################

# Pull out one tree from the forest
tree = grid_search.best_estimator_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = df_diet_total_nonan.columns, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')


# Extract the small tree
tree_small =grid_search.best_estimator_[5]
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = df_diet_total_nonan.columns, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('small_tree.png');

###############################################################################
# Feature importance
###############################################################################

# Get numerical feature importances
importances = list(best_grid.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(df_diet_total_nonan.columns, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

feature_list = df_diet_total_nonan.columns

# Import matplotlib for plotting and use magic command for Jupyter Notebooks
# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');



