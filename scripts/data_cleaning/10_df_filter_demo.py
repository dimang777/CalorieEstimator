# Prepare the data to put everything into pandas dataframe
import pickle
import pandas as pd
import numpy as np

###############################################################################
# Set up folders and variables
###############################################################################
filename = '10_df_filter_demo.py'
save_folder = '../../data/cleaned_df/'
load_folder = '../../data/cleaned_df/'

with open(load_folder + 'df_bfr_demo_filter.pkl', 'rb') as f:
    [df_bfr_demo_filter, df_collection_key, \
                 demo_filenames, \
                 demo_filename_varname_pd_dict, \
                 diet_filenames, \
                 diet_filename_varname_pd_dict, \
                 exam_filenames, \
                 exam_filename_varname_pd_dict, \
                 lab_filenames, \
                 lab_filename_varname_pd_dict, \
                     ] = pickle.load(f)

# D0_DMDCITZN - 7 to nan, 9 to nan
# D0_DMDMARTL - 77 to nan, 99 to nan

# D0_RIDEXPRG - find subjects who are 1 and then remove them from the dataframe. And then remove this variable
# This is removing pregnant subjects

# D0_WTMEC2YR - 0 to nan
# D0_INDHHIN2 - 77 to nan and 99 to nan
# D0_INDFMIN2 - 77 to nan and 99 to nan


df = df_bfr_demo_filter.copy()
df['D0_DMDCITZN'] = df['D0_DMDCITZN'].replace([7, 9], [np.nan, np.nan])
np.sum(df['D0_DMDCITZN']==9)

df['D0_DMDMARTL'] = df['D0_DMDMARTL'].replace([77, 99], [np.nan, np.nan])
np.sum(df['D0_DMDMARTL']==77)

df['D0_WTMEC2YR'] = df['D0_WTMEC2YR'].replace([0], [np.nan])
np.sum(df['D0_WTMEC2YR']==0)

df['D0_INDHHIN2'] = df['D0_INDHHIN2'].replace([77, 99], [np.nan, np.nan])
np.sum(df['D0_INDHHIN2']==99)

df['D0_INDFMIN2'] = df['D0_INDFMIN2'].replace([77, 99], [np.nan, np.nan])
np.sum(df['D0_INDFMIN2']==77)

pregnant_droplist_idx = list(np.where(df['D0_RIDEXPRG'] == 1)[0])
pregnant_droplist = df.index[pregnant_droplist_idx]
        
a = set(pregnant_droplist)
b = set(df.index)
c = a.intersection(b)

df = df.drop(c)
df = df.drop(columns = ['D0_RIDEXPRG'])


temp = demo_filename_varname_pd_dict['DEMO_I']
temp.remove('D0_RIDEXPRG')
demo_filename_varname_pd_dict['DEMO_I'] = temp

demo_filename_varname_pd_dict #Fixed
df # Fixed


#----------df----------- THE variable to use

with open(save_folder + 'df.pkl', 'wb') as f:
    pickle.dump([df], f)

with open(save_folder + 'df.pkl', 'rb') as f:
    [df] = pickle.load(f)