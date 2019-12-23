import time
import pickle
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import svm

###############################################################################
# Set up folders and variables
###############################################################################
filename = 'svm.py'

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


C = [int(x) for x in np.logspace(start=0, stop=3, num=4)]
kernel = ['linear', 'rbf']
gamma = [(x) for x in np.logspace(start=-3, stop=2, num=6)]
random_grid = {'C': C,
               'kernel': kernel,
               'gamma': gamma}
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
model = svm.SVC()

start = time.time()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
svm_random = RandomizedSearchCV(estimator=model,
                                param_distributions=random_grid,
                                n_iter=100,
                                cv=3,
                                verbose=0,
                                random_state=42,
                                n_jobs=-1)
# Fit the random search model
svm_random.fit(x_train_sel_df.values, y_train_sel_df.values)

print(svm_random.best_params_)
end = time.time()
print(end - start)

# 48 tasks
# Using multicore - 5.15s
# Using single core - 3.5s

###############################################################################
# Grid search around the best parameters
###############################################################################

# Create the parameter grid based on the results of random search
C = [int(x) for x in np.linspace(start=900, stop=1100, num=21)]
kernel = ['linear']
gamma = [(x) for x in np.linspace(start=0.0002, stop=0.002, num=10)]
param_grid = {'C': C,
              'kernel': kernel,
              'gamma': gamma}
print(param_grid)

model = svm.SVC()
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
# 630 tasks
# Using multicore - 7.5s
# Using single core - 9.93s

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
# Accuracy Test: 98.88%
# Accuracy Test: 98.51%
# =============================================================================

###############################################################################
# Save model
###############################################################################

with open(model_folder+'svm_hyper_tune_bestmodel_sel_fea.pickle', 'wb') \
  as handle:
    pickle.dump([model], handle)

with open(model_folder+'svm_hyper_tune_bestmodel_sel_fea.pickle', 'rb') \
  as handle:
    [model] = pickle.load(handle)
