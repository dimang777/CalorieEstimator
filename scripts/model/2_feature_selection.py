# Feature selection
# Wrapper based method (Backward elimination)
# Source: https://towardsdatascience.com/feature-selection-using-wrapper-methods-in-python-f0d352b346f
# Here the training set was used to select features since the selected features
# are eventually used to modeling. Feature selection in a way is part of the
# modeling

import pickle
import statsmodels.api as sm
import pandas as pd
from sklearn.model_selection import train_test_split


###############################################################################
# Set up folders and variables
###############################################################################
filename = '2_feature_selection.py'

save_folder = '../../data/data_for_model/'
load_folder = '../../data/data_for_model/'


with open(load_folder + 'df_diet.pkl', 'rb') as f:
    [df_diet_total,
        df_diet_y,
        df_diet_y_raw,
        class_0_flag,
        class_1_flag,
        class_2_flag,
        Class_0_df_index,
        Class_1_df_index,
        Class_2_df_index] = pickle.load(f)

with open(save_folder + 'train_test_full_features.pkl', 'rb') as f:
    [x_train_ful_df,
        x_test_ful_df,
        y_train_ful_df,
        y_test_ful_df,
        train_idx,
        test_idx,
        train_flag,
        test_flag] = pickle.load(f)


###############################################################################
# Set up functions
###############################################################################


def backward_elimination(data, target, significance_level=0.05):
    features = data.columns.tolist()
    excluded_features = []
    while(len(features) > 0):
        features_with_constant = sm.add_constant(data[features])
        p_values = sm.OLS(target, features_with_constant).fit().pvalues[1:]
        max_p_value = p_values.max()
        if(max_p_value >= significance_level):
            excluded_feature = p_values.idxmax()
            excluded_features.append(excluded_feature)
            features.remove(excluded_feature)
        else:
            break
    return features, excluded_features


###############################################################################
# Select features
###############################################################################

[features, excluded_features] = \
  backward_elimination(pd.DataFrame(x_train_ful_df), y_train_ful_df.values)

df_diet_total_sel = df_diet_total.loc[:, features].copy()


###############################################################################
# Divide into train and test set
###############################################################################

indices = range(df_diet_total_sel.shape[0])
x_train_sel_df, x_test_sel_df, y_train_sel_df, y_test_sel_df, \
  train_idx, test_idx = train_test_split(df_diet_total_sel,
                                         df_diet_y, indices,
                                         test_size=0.2,
                                         random_state=42,
                                         stratify=df_diet_y.to_numpy())
train_flag = train_idx in indices
test_flag = train_idx in indices


###############################################################################
# Save
###############################################################################


with open(save_folder + 'train_test_sel_features.pkl', 'wb') as f:
    pickle.dump([features,
                 excluded_features,
                 x_train_sel_df,
                 x_test_sel_df,
                 y_train_sel_df,
                 y_test_sel_df,
                 train_idx,
                 test_idx,
                 train_flag,
                 test_flag], f)


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

with open(save_folder + 'df_diet_sel.pkl', 'wb') as f:
    pickle.dump([df_diet_total_sel,
                 df_diet_y,
                 df_diet_y_raw,
                 class_0_flag,
                 class_1_flag,
                 class_2_flag,
                 Class_0_df_index,
                 Class_1_df_index,
                 Class_2_df_index], f)

with open(save_folder + 'df_diet_sel.pkl', 'rb') as f:
    [df_diet_total_sel,
        df_diet_y,
        df_diet_y_raw,
        class_0_flag,
        class_1_flag,
        class_2_flag,
        Class_0_df_index,
        Class_1_df_index,
        Class_2_df_index] = pickle.load(f)
