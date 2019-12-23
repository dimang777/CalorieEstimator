import time
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

###############################################################################
# Set up folders and variables
###############################################################################
filename = 'nbc.py'

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

# No parameters to tune for NBC - same priors
random_grid = {}
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
# Set up CV random search - skip for NBC
###############################################################################

###############################################################################
# Grid search around the best parameters
###############################################################################
# Perform CV - no parameter to tune for NBC

param_grid = {}

model = GaussianNB()

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
# Accuracy Test: 83.35%
# Accuracy Test: 83.87%
# =============================================================================

###############################################################################
# Save model
###############################################################################

with open(model_folder+'nbc_hyper_tune_bestmodel_sel_fea.pickle', 'wb') \
  as handle:
    pickle.dump([model], handle)

with open(model_folder+'nbc_hyper_tune_bestmodel_sel_fea.pickle', 'rb') \
  as handle:
    [model] = pickle.load(handle)
