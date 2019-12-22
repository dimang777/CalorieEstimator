import time
import pickle
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import xgboost

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


max_depth = range(3, 11)
learning_rate = np.arange(0.01, 0.1+0.01, 0.01)
subsample = np.arange(0.8, 1+0.04, 0.04)
colsample_bytree = np.arange(0.4, 1+0.1, 0.1)
gamma = [0, 1, 5]
eval_metric = ['merror']
random_grid = {'max_depth': max_depth,
               'learning_rate': learning_rate,
               'subsample': subsample,
               'colsample_bytree': colsample_bytree,
               'gamma': gamma,
               'eval_metric': eval_metric}
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
# add tree_method = 'gpu_hist' for GPU
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
#!@@$!#@!#!$QTQ$%@#%!@#!$#@$!#@&^*%(&^)&* WORKED UNTIL HERE
#!@@$!#@!#!$QTQ$%@#%!@#!$#@$!#@&^*%(&^)&* WORKED UNTIL HERE
#!@@$!#@!#!$QTQ$%@#%!@#!$#@$!#@&^*%(&^)&* WORKED UNTIL HERE
#!@@$!#@!#!$QTQ$%@#%!@#!$#@$!#@&^*%(&^)&* WORKED UNTIL HERE
#!@@$!#@!#!$QTQ$%@#%!@#!$#@$!#@&^*%(&^)&* WORKED UNTIL HERE
#!@@$!#@!#!$QTQ$%@#%!@#!$#@$!#@&^*%(&^)&* WORKED UNTIL HERE
#!@@$!#@!#!$QTQ$%@#%!@#!$#@$!#@&^*%(&^)&* WORKED UNTIL HERE
#!@@$!#@!#!$QTQ$%@#%!@#!$#@$!#@&^*%(&^)&* WORKED UNTIL HERE

# 48 tasks
# Using multicore - 67.8s
# Using single core - 3.5s

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
