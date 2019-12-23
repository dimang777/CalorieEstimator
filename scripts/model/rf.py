import time
import pickle
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

###############################################################################
# Set up folders and variables
###############################################################################
filename = 'rf.py'

save_folder = '../../data/data_for_model/'
load_folder = '../../data/data_for_model/'
model_folder = '../../data/model/'

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
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]  # Create the random grid
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
    y_pred = model.predict(test_features)
    accuracy = accuracy_score(test_labels, y_pred)
    print('Model Performance')
    print("Accuracy test: %.2f%%" % (accuracy * 100.0))

    return accuracy*100


###############################################################################
# Set up CV random search
###############################################################################


# Use the random grid to search for best hyperparameters
# Use the random grid to search for best hyperparameters
# First create the base model to tune
model = RandomForestClassifier()

start = time.time()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=model,
                               param_distributions=random_grid,
                               n_iter=100,
                               cv=3,
                               verbose=2,
                               random_state=42,
                               n_jobs=-1)
# Fit the random search model
rf_random.fit(x_train_sel_df.values, y_train_sel_df.values)

print(rf_random.best_params_)
end = time.time()
print(end - start)

# _ tasks
# Using multicore - _s
# Using single core - _s

model = rf_random.best_estimator_
random_accuracy_train = evaluate(model,
                                 x_train_sel_df.values,
                                 y_train_sel_df.values)
random_accuracy_test = evaluate(model,
                                x_test_sel_df.values,
                                y_test_sel_df.values)

###############################################################################
# Grid search around the best parameters
###############################################################################

# Number of trees in random forest
n_estimators = [1580, 1590, 1600, 1610, 1620]
# Number of features to consider at every split
max_features = ['sqrt']
# Maximum number of levels in tree
max_depth = [68, 69, 70, 71, 72]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [4, 5, 6]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 3]
# Method of selecting samples for training each tree
bootstrap = [False]  # Create the random grid

param_grid = {
    'bootstrap': bootstrap,
    'max_depth': max_depth,
    'max_features': max_features,
    'min_samples_leaf': min_samples_leaf,
    'min_samples_split': min_samples_split,
    'n_estimators': n_estimators
}

model = RandomForestClassifier()

start = time.time()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator=model,
                           param_grid=param_grid,
                           cv=3,
                           n_jobs=-1,
                           verbose=2)

# Fit the grid search to the data
grid_search.fit(x_train_sel_df.values, y_train_sel_df.values)
print(grid_search.best_params_)

end = time.time()
print(end - start)
# _ tasks
# Using multicore - _ hours
# Using single core - _s

###############################################################################
# Final model
###############################################################################

model = grid_search.best_estimator_
grid_accuracy_train = evaluate(model,
                               x_train_sel_df.values,
                               y_train_sel_df.values)
grid_accuracy_test = evaluate(model,
                              x_test_sel_df.values,
                              y_test_sel_df.values)

# =============================================================================
# Accuracy Test: 100.00%
# Accuracy Test: 93.05%
# =============================================================================

###############################################################################
# Save model
###############################################################################

with open(model_folder+'rf_hyper_tune_bestmodel_sel_fea.pickle', 'wb') \
  as handle:
    pickle.dump([model], handle)

with open(model_folder+'rf_hyper_tune_bestmodel_sel_fea.pickle', 'rb') \
  as handle:
    [model] = pickle.load(handle)
