import numpy as np
import pandas as pd
import pickle


###############################################################################
# Set up folders and variables
###############################################################################
filename = 'data_setup.py'
save_folder = '../../data/data_for_model/'
load_folder = '../../data/cleaned_df/'

figure_folder = '../../images/'
sheet_folder = '../../sheets/'


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

len(diet_filename_varname_pd_dict[diet_filenames[0]])
# 75

df_diet = df[diet_filename_varname_pd_dict[diet_filenames[0]][2:]].copy()
df_diet.describe()



valid_sam_count = 0
valid_sam_flag = np.zeros(len(df_diet.index), dtype=bool)
for i in range(0,len(df_diet.index)):
    if np.sum(df_diet.iloc[i,:].isna().values) == 0:
        valid_sam_count += 1
        valid_sam_flag[i] = True 

df_diet_total_nonan = df_diet.iloc[valid_sam_flag,:].copy()
df_diet_total_nonan.describe()

# Remove the label from the Data
df_diet_total_nonan = df_diet_total_nonan.drop(columns='I0_DR1TKCAL') # Total calorie

d_num = 10
df_corr_triu['I0_DR1TKCAL'].sort_values(ascending=False, na_position='last').index[:d_num].values
Selected_X_vars_nn = df_corr_triu['I0_DR1TKCAL'].sort_values(ascending=False, na_position='last').index[:d_num].values

df_diet_6ip = df_diet.loc[:,Selected_X_vars_nn].copy()
len(df_diet_6ip.columns)

Valid_sam_6ip_count = 0
Valid_sam_6ip_flag = np.zeros(len(df_diet_6ip.index), dtype=bool)
for i in range(0,len(df_diet_6ip.index)):
    if np.sum(df_diet.iloc[i,:].isna().values) == 0:
        Valid_sam_6ip_count += 1
        Valid_sam_6ip_flag[i] = True 

np.log2(Valid_sam_6ip_count)
df_diet_6ip_nonan = df_diet_6ip.iloc[valid_sam_flag,:].copy()
df_diet_6ip_nonan.describe()

# 2^d = 2013
# dlog2(2) = log2(2013)
# d = log2(2013)
# np.log2(2013) - Valid number of inputs based on sample size
# Source: http://www.simafore.com/blog/3-key-considerations-for-machine-learning-on-big-data

if np.sum(df.loc[:,'I0_DR1TKCAL'][Valid_sam_6ip_flag].isna().values) == 0:
    print('I0_DR1TKCAL has no nans along with other 11 variables')

# Copy label
df_diet_y_raw = df.loc[:,'I0_DR1TKCAL'][Valid_sam_6ip_flag].copy()

NumofEachClass = int(df_diet_y_raw.sort_values(ascending=False, na_position='last').count()/3)

diet_y = np.zeros(df_diet_y_raw.count(), 'int')
diet_y[:NumofEachClass] = 2
diet_y[NumofEachClass:2*NumofEachClass] = 1

df_diet_y_raw_sort = df_diet_y_raw.sort_values(ascending=False, na_position='last').copy()

df_diet_cla = pd.DataFrame({'Class':diet_y})
df_diet_cla = df_diet_cla.set_index(df_diet_y_raw_sort.index)
df_diet_cla.describe()

df_diet_y = df_diet_cla.join(df_diet_y_raw_sort).loc[:,'Class']

df_diet_nn_selectedip = df_diet_6ip_nonan.join(df_diet_y)
df_diet_nn_selectedip.describe()

class_0_flag = df_diet_nn_selectedip.loc[:,'Class'].values == 0
class_1_flag = df_diet_nn_selectedip.loc[:,'Class'].values == 1
class_2_flag = df_diet_nn_selectedip.loc[:,'Class'].values == 2

Class_0_df_index = df_diet_nn_selectedip.iloc[class_0_flag,:].index
Class_1_df_index = df_diet_nn_selectedip.iloc[class_1_flag,:].index
Class_2_df_index = df_diet_nn_selectedip.iloc[class_2_flag,:].index

TrainingSetRatio = 0.8
DevSetRatio = 0.1
TestSetRatio = 0.1

DevSetNum = int(DevSetRatio * NumofEachClass) # 68
TestSetNum = int(DevSetRatio * NumofEachClass) # 67
TrainingSetNum = NumofEachClass - DevSetNum - TestSetNum # 537

Class_0_df_index_rand = np.array(Class_0_df_index).ravel().copy()
np.random.shuffle(Class_0_df_index_rand)
Class_0_df_index_Devset = Class_0_df_index_rand[:DevSetNum]
Class_0_df_index_Testset = Class_0_df_index_rand[DevSetNum:DevSetNum+TestSetNum]
Class_0_df_index_Trainset = Class_0_df_index_rand[DevSetNum+TestSetNum:]
len(Class_0_df_index_Devset)
len(Class_0_df_index_Testset)
len(Class_0_df_index_Trainset)

Class_1_df_index_rand = np.array(Class_1_df_index).ravel().copy()
np.random.shuffle(Class_1_df_index_rand)
Class_1_df_index_Devset = Class_1_df_index_rand[:DevSetNum]
Class_1_df_index_Testset = Class_1_df_index_rand[DevSetNum:DevSetNum+TestSetNum]
Class_1_df_index_Trainset = Class_1_df_index_rand[DevSetNum+TestSetNum:]

Class_2_df_index_rand = np.array(Class_2_df_index).ravel().copy()
np.random.shuffle(Class_2_df_index_rand)
Class_2_df_index_Devset = Class_2_df_index_rand[:DevSetNum]
Class_2_df_index_Testset = Class_2_df_index_rand[DevSetNum:DevSetNum+TestSetNum]
Class_2_df_index_Trainset = Class_2_df_index_rand[DevSetNum+TestSetNum:]


df_index_Devset = np.concatenate((Class_0_df_index_Devset, Class_1_df_index_Devset, Class_2_df_index_Devset), axis=None)
len(df_index_Devset)
df_index_Testset = np.concatenate((Class_0_df_index_Testset, Class_1_df_index_Testset, Class_2_df_index_Testset), axis=None)
len(df_index_Testset)
df_index_Trainset = np.concatenate((Class_0_df_index_Trainset, Class_1_df_index_Trainset, Class_2_df_index_Trainset), axis=None)
np.log2(len(df_index_Trainset))

# Data format
# x1=  [1;2;3;4;5;6] column vector
# y1 = a number belonging to {0,1,2}
# X = [x1columnvec x2columnvec…xmtrain]
# nx x mtrain
# Y = [y1 y2 …ymtrain]
# 1 x mtrain
# One-hot
# Y = [[0;0;0;1] [0;1;0;0] [0;0;1;0] …[1;0;0;0]]
# C x mtrain


tf_Devset_y = np.array(df_diet_nn_selectedip.loc[df_index_Devset,'Class'].values).ravel().copy()
tf_Devset_X = np.array(df_diet_nn_selectedip.loc[df_index_Devset,df_diet_nn_selectedip.columns[:10].values].values).transpose().copy()

# Check - Complete
# tf_Defset_X.shape
# tf_Defset_y[0]
# tf_Defset_X[:,0]
# df_diet_nn_selectedip.loc[df_index_Devset[0],:]

tf_Testset_y = np.array(df_diet_nn_selectedip.loc[df_index_Testset,'Class'].values).ravel().copy()
tf_Testset_X = np.array(df_diet_nn_selectedip.loc[df_index_Testset,df_diet_nn_selectedip.columns[:10].values].values).transpose().copy()
tf_Testset_X.shape

tf_Trainset_y = np.array(df_diet_nn_selectedip.loc[df_index_Trainset,'Class'].values).ravel().copy()
tf_Trainset_X = np.array(df_diet_nn_selectedip.loc[df_index_Trainset,df_diet_nn_selectedip.columns[:10].values].values).transpose().copy()
tf_Trainset_X.shape

# Final set
tf_Devset_y
tf_Devset_X
tf_Testset_y
tf_Testset_X
tf_Trainset_y
tf_Trainset_X

# Normalization
Divisors = np.array(df_diet_6ip_nonan.max().values).ravel().copy()

tf_Devset_norm_X = tf_Devset_X.copy()
tf_Testset_norm_X = tf_Testset_X.copy()
tf_Trainset_norm_X = tf_Trainset_X.copy()
for i in range(tf_Devset_X.shape[0]):
    tf_Devset_norm_X[i,:] = np.divide(tf_Devset_norm_X[i,:], Divisors[i])
    tf_Testset_norm_X[i,:] = np.divide(tf_Testset_norm_X[i,:], Divisors[i])
    tf_Trainset_norm_X[i,:] = np.divide(tf_Trainset_norm_X[i,:], Divisors[i])

# Final set normalized
tf_Devset_y
tf_Devset_norm_X
tf_Testset_y
tf_Testset_norm_X
tf_Trainset_y
tf_Trainset_norm_X
tf_Testset_norm_X.shape
tf_Trainset_norm_X.shape
# 221 + 62 + 155 +115 + 79 + 150 + 170 + 300 + 100 + 106 + 87 + 250 + 180 + 41
# 2016 lines of code so far - being conservative (i.e., didn't count variable list, constants.)
# ~2000 lines of code
# ~2300 including the neural network code

with open(Save_folder + 'tf.pkl', 'wb') as f:
    pickle.dump([tf_Devset_y, \
                 tf_Devset_norm_X, \
                 tf_Testset_y, \
                 tf_Testset_norm_X, \
                 tf_Trainset_y, \
                 tf_Trainset_norm_X], f)


with open(Save_folder + 'tf.pkl', 'rb') as f:
    [tf_Devset_y, \
        tf_Devset_norm_X, \
        tf_Testset_y, \
        tf_Testset_norm_X, \
        tf_Trainset_y, \
        tf_Trainset_norm_X] = pickle.load(f)

with open(Save_folder + 'df_diet.pkl', 'wb') as f:
    pickle.dump([df_diet, \
        df_diet_total_nonan, \
        df_diet_y, \
        df_diet_y_raw, \
        class_0_flag, \
        class_1_flag, \
        class_2_flag, \
        Class_0_df_index, \
        Class_1_df_index, \
        Class_2_df_index, \
        df_index_Devset, \
        df_index_Testset, \
        df_index_Trainset], f)

with open(Save_folder + 'df_diet.pkl', 'rb') as f:
    [df_diet, \
        df_diet_total_nonan, \
        df_diet_y, \
        df_diet_y_raw, \
        class_0_flag, \
        class_1_flag, \
        class_2_flag, \
        Class_0_df_index, \
        Class_1_df_index, \
        Class_2_df_index, \
        df_index_Devset, \
        df_index_Testset, \
        df_index_Trainset] = pickle.load(f)




