import time
import pickle
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import xgboost
import matplotlib.pyplot as plt

###############################################################################
# Set up folders and variables
###############################################################################
filename = 'xgb.py'

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

n_estimators = np.arange(100, 1000, 100)
max_depth = range(3, 11)
learning_rate = np.arange(0.01, 0.1+0.01, 0.01)
subsample = np.arange(0.8, 1+0.04, 0.04)
colsample_bytree = np.arange(0.4, 1+0.1, 0.1)
gamma = [0, 1, 5]
early_stopping_rounds = np.arange(5, 25, 5)
eval_metric = ['merror']
eval_set_plot = [(x_train_sel_df.values, y_train_sel_df.values)]

random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'learning_rate': learning_rate,
               'subsample': subsample,
               'colsample_bytree': colsample_bytree,
               'gamma': gamma,
               'early_stopping_rounds': early_stopping_rounds,
               'eval_metric': eval_metric,
               'eval_set_plot': eval_set_plot}
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
# add tree_method = 'gpu_hist' for GPU - but doesn't seem to be 
# compatible with CV and multicore deployment
model = xgboost.XGBClassifier(objective='multi:softprob',
                              num_class=3)

start = time.time()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
xgb_random = RandomizedSearchCV(estimator=model,
                                param_distributions=random_grid,
                                n_iter=100,
                                cv=3,
                                verbose=2,
                                random_state=42,
                                n_jobs=-1)
# Fit the random search model
xgb_random.fit(x_train_sel_df.values, y_train_sel_df.values)

print(xgb_random.best_params_)
end = time.time()
print(end - start)

# _ tasks
# Using multicore - _s
# Using single core - _s

model = xgb_random.best_estimator_
random_accuracy_train = evaluate(model,
                               x_train_sel_df.values,
                               y_train_sel_df.values)
random_accuracy_test = evaluate(model,
                              x_test_sel_df.values,
                              y_test_sel_df.values)

###############################################################################
# Grid search around the best parameters
###############################################################################

# Create the parameter grid based on the results of random search
n_estimators = np.arange(880, 920, 10)
max_depth = [4]
learning_rate = np.arange(0.048, 0.052+0.001, 0.001)
subsample = np.arange(0.82, 0.86+0.01, 0.01)
colsample_bytree = np.arange(0.68, 0.72+0.01, 0.01)
gamma = [1]
early_stopping_rounds = [20]
eval_metric = ['merror']
eval_set_plot = [(x_train_sel_df.values, y_train_sel_df.values)]

param_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'learning_rate': learning_rate,
               'subsample': subsample,
               'colsample_bytree': colsample_bytree,
               'gamma': gamma,
               'early_stopping_rounds': early_stopping_rounds,
               'eval_metric': eval_metric,
               'eval_set_plot': eval_set_plot}
print(param_grid)


model = xgboost.XGBClassifier(objective='multi:softprob',
                              num_class=3)
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
# {'C': 1040, 'gamma': 0.0002, 'kernel': 'linear'}
end = time.time()
print(end - start)
# 1800 tasks
# Using multicore - ~2 hours
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
# Accuracy Test: 99.88%
# Accuracy Test: 95.29%
# =============================================================================

###############################################################################
# Save model
###############################################################################

with open(model_folder+'xgb_hyper_tune_bestmodel_sel_fea.pickle', 'wb') \
  as handle:
    pickle.dump([model], handle)

with open(model_folder+'xgb_hyper_tune_bestmodel_sel_fea.pickle', 'rb') \
  as handle:
    [model] = pickle.load(handle)
