# Feature reduction

import numpy as np
import pandas as pd
import pickle


###############################################################################
# Set up folders and variables
###############################################################################
filename = 'data_setup_full_features.py'

load_folder = '../../data/cleaned_df/'


###############################################################################
# Load
###############################################################################

with open(load_folder + 'df.pkl', 'rb') as f:
    [df] = pickle.load(f)

with open(load_folder + 'df_bfr_demo_filter.pkl', 'rb') as f:
    [_, _, \
                 _, \
                 _, \
                 diet_filenames, \
                 diet_filename_varname_pd_dict, \
                 _, \
                 _, \
                 _, \
                 _, \
                     ] = pickle.load(f)

with open(load_folder + 'corr.pkl', 'rb') as f:
    [df_corr_triu] = pickle.load(f)

###############################################################################
# Isolate diet data
###############################################################################


len(diet_filename_varname_pd_dict[diet_filenames[0]])
# 75

df_diet = df[diet_filename_varname_pd_dict[diet_filenames[0]][2:]].copy()




###############################################################################
# Remove nans
###############################################################################


valid_sample_count = 0
valid_sample_flag = np.zeros(len(df_diet.index), dtype=bool)
for i in range(0,len(df_diet.index)):
    if np.sum(df_diet.iloc[i,:].isna().values) == 0:
        valid_sample_count += 1
        valid_sample_flag[i] = True


df_diet_total_nonan = df_diet.iloc[valid_sample_flag,:].copy()

###############################################################################
# Separate label and convert continuous data to classes - total calorie
###############################################################################
# Get total calorie values
df_diet_y_raw = df_diet_total_nonan.loc[:,'I0_DR1TKCAL'].copy()
# Remove the label from the Data
df_diet_total_nonan = df_diet_total_nonan.drop(columns='I0_DR1TKCAL')

# Convert to classes
numofeachclass = int(df_diet_y_raw.sort_values(ascending=False, na_position='last').count()/3)

diet_y = np.zeros(df_diet_y_raw.count(), 'int')
diet_y[:numofeachclass] = 2
diet_y[numofeachclass:2*numofeachclass] = 1

df_diet_y_raw_sort = df_diet_y_raw.sort_values(ascending=False, na_position='last').copy()

df_diet_cla = pd.DataFrame({'Class':diet_y})
df_diet_cla = df_diet_cla.set_index(df_diet_y_raw_sort.index)
print(df_diet_cla.describe())

df_diet_y_sorted = df_diet_cla.join(df_diet_y_raw_sort).loc[:,'Class']

df_diet_total_label = df_diet_total_nonan.join(df_diet_y_sorted)

class_0_flag = df_diet_total_label.loc[:,'Class'].values == 0
class_1_flag = df_diet_total_label.loc[:,'Class'].values == 1
class_2_flag = df_diet_total_label.loc[:,'Class'].values == 2

Class_0_df_index = df_diet_total_label.iloc[class_0_flag,:].index
Class_1_df_index = df_diet_total_label.iloc[class_1_flag,:].index
Class_2_df_index = df_diet_total_label.iloc[class_2_flag,:].index

df_diet_y = df_diet_total_label.loc[:,'Class'].copy()

# use this to split
from sklearn.model_selection import train_test_split



