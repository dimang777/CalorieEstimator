# This file requires Sub1 to run first to load data
import xport
import os
import matplotlib.pyplot as plt
# %matplotlib qt
import numpy as np
from os import walk
import h5py
import math
import pickle

savefolder = 'C:\\Users\\diman\\OneDrive\\Work_temp\\Insight Fellows Demo 2019\\WorkSave'
homefolder = 'C:\\Users\\diman\\OneDrive\\Work_temp\\Insight Fellows Demo 2019'
filename = 'Sub3_Data_Clean_Diet.py'

Demo_var_str = 'Demo'
Diet_var_str = 'Diet'
Exam_var_str = 'Exam'
Lab_var_str = 'Lab'
Ques_var_str = 'Ques'

ext = '.xpt'
years = ['2015', '2013', '2011', '2009', '2007', '2005']

year = '2015'
Save_folder = 'C:/Users/diman/OneDrive/Work_temp/Insight Fellows Demo 2019/WorkSave/'

with open(Save_folder + Diet_var_str + '_' + year + '_raw_loadstr.pkl', 'rb') as f:
    Diet_save_variables_str = pickle.load(f)

with open(Save_folder + Diet_var_str + '_' + year + '_raw.pkl', 'rb') as f:
    exec(Diet_save_variables_str + '= pickle.load(f)')

# Ignore error - above code loads the variables


for Condense in [0]:
    Cols_toRemove = ['WTDR2D', 
    'DR1EXMER', 
    'DRABF',
    'DR1DBIH',
    'DR1DAY',
    'DR1LANG',
    'DR1MRESP',
    'DR1HELP',
    'DR1SKY',
    'DRQSDT1',
    'DRQSDT2',
    'DRQSDT3',
    'DRQSDT4',
    'DRQSDT5',
    'DRQSDT6',
    'DRQSDT7',
    'DRQSDT8',
    'DRQSDT9',
    'DRQSDT10',
    'DRQSDT11',
    'DRQSDT12',
    'DRQSDT91',
    'DR1TNUMF',
    'DR1TWS',
    'DRD340',
    'DRD350A',
    'DRD350AQ',
    'DRD350B',
    'DRD350BQ',
    'DRD350C',
    'DRD350CQ',
    'DRD350D',
    'DRD350DQ',
    'DRD350E',
    'DRD350EQ',
    'DRD350F',
    'DRD350FQ',
    'DRD350G',
    'DRD350GQ',
    'DRD350H',
    'DRD350HQ',
    'DRD350I',
    'DRD350IQ',
    'DRD350J',
    'DRD350JQ',
    'DRD350K',
    'DRD370A',
    'DRD370AQ',
    'DRD370B',
    'DRD370BQ',
    'DRD370C',
    'DRD370CQ',
    'DRD370D',
    'DRD370DQ',
    'DRD370E',
    'DRD370EQ',
    'DRD370F',
    'DRD370FQ',
    'DRD370G',
    'DRD370GQ',
    'DRD370H',
    'DRD370HQ',
    'DRD370I',
    'DRD370IQ',
    'DRD370J',
    'DRD370JQ',
    'DRD370K',
    'DRD370KQ',
    'DRD370L',
    'DRD370LQ',
    'DRD370M',
    'DRD370MQ',
    'DRD370N',
    'DRD370NQ',
    'DRD370O',
    'DRD370OQ',
    'DRD370P',
    'DRD370PQ',
    'DRD370Q',
    'DRD370QQ',
    'DRD370R',
    'DRD370RQ',
    'DRD370S',
    'DRD370SQ',
    'DRD370T',
    'DRD370TQ',
    'DRD370U',
    'DRD370UQ',
    'DRD370V']

# Remove variables
for Condense in [0]:
    Cols_toRemove_Idx = []
    for i in range(0,len(Cols_toRemove)):
        Cols_toRemove_Idx.append(int(Diet_2015_DR1TOT_I_raw_lbl.index(Cols_toRemove[i])))
    
    Cols_toKeep_idx = []
    for i in range(0,Diet_2015_DR1TOT_I_raw.shape[1]):
        if i not in Cols_toRemove_Idx:
            Cols_toKeep_idx.append(int(i))
    
    # VariablestoSave_flag = [True]*Diet_2015_DR1TOT_I_raw.shape[1] # To generate a list of trues - don't know how to manipulate yet
    
    Diet_2015_DR1TOT_I_clean1 = np.copy(Diet_2015_DR1TOT_I_raw[:,Cols_toKeep_idx])
    
    Diet_2015_DR1TOT_I_clean1.shape
    
    Diet_2015_DR1TOT_I_clean1_lbl = []
    for i in Cols_toKeep_idx:
        Diet_2015_DR1TOT_I_clean1_lbl.append(Diet_2015_DR1TOT_I_raw_lbl[i])

# Filtering
for Condense in [0]:
# WTDRD1 - remove 0 and . (no nans) - other are continuous variable
# DR1DRSTZ - remove 2,4,5,. (no nans) - keep only 1
# DRDINT - remove . (nans) - keep 1 and 2
# DRQSDIET - remove 1, 9, . - keep only 2
# DR1_300 - remove 1, 3, 7, 9, . - keep only 2

    FilterVars = ['WTDRD1', 'DR1DRSTZ', 'DRDINT', 'DRQSDIET', 'DR1_300']  
    
    FilterVars_idx = []
    for i_str in FilterVars:
        FilterVars_idx.append(Diet_2015_DR1TOT_I_clean1_lbl.index(i_str))
    
    i = 0
    # np.sum(np.isnan(Diet_2015_DR1TOT_I_clean1[:, FilterVars_idx[i]])*1) # find nan
    Keep_flag_clean1 = Diet_2015_DR1TOT_I_clean1[:, FilterVars_idx[i]] != 0
    
    i = 1
    Keep_flag_clean1 = np.logical_and(Keep_flag_clean1, Diet_2015_DR1TOT_I_clean1[:, FilterVars_idx[i]] == 1)
    
    i = 2
    Keep_flag_clean1 = np.logical_and(Keep_flag_clean1, np.logical_or(Diet_2015_DR1TOT_I_clean1[:, FilterVars_idx[i]] == 1, Diet_2015_DR1TOT_I_clean1[:, FilterVars_idx[i]] == 2))
    # np.isnan(Diet_2015_DR1TOT_I_clean1[:, FilterVars_idx[i]]) # alternative approach - code not complete
    
    i = 3
    Keep_flag_clean1 = np.logical_and(Keep_flag_clean1, Diet_2015_DR1TOT_I_clean1[:, FilterVars_idx[i]] == 2)
    
    i = 4
    Keep_flag_clean1 = np.logical_and(Keep_flag_clean1, Diet_2015_DR1TOT_I_clean1[:, FilterVars_idx[i]] == 2)
    
    np.sum(Keep_flag_clean1)


# remove filtered variables
for Condense in [0]:
    FilterVars_toRemove_idx = []
    for i in range(1,len(FilterVars)): # should keep WTDRD1
        FilterVars_toRemove_idx.append(int(Diet_2015_DR1TOT_I_clean1_lbl.index(FilterVars[i])))
    
    Cols_toKeep_aftfilt_idx = []
    for i in range(0,Diet_2015_DR1TOT_I_clean1.shape[1]):
        if i not in FilterVars_toRemove_idx:
            Cols_toKeep_aftfilt_idx.append(int(i))
    
    # remove columns and rows at the same time
    Diet_2015_DR1TOT_I_clean2_temp = np.copy(Diet_2015_DR1TOT_I_clean1[Keep_flag_clean1, :])
    Diet_2015_DR1TOT_I_clean2 = np.copy(Diet_2015_DR1TOT_I_clean2_temp[:, Cols_toKeep_aftfilt_idx])
    
    Diet_2015_DR1TOT_I_clean2.shape
    
    Diet_2015_DR1TOT_I_clean2_lbl = []
    for i in Cols_toKeep_aftfilt_idx:
        Diet_2015_DR1TOT_I_clean2_lbl.append(Diet_2015_DR1TOT_I_clean1_lbl[i])
    
    len(Diet_2015_DR1TOT_I_clean2_lbl)

# Replace with nans
for Condense in [0]:
# DBQ095Z - set 2, 3, 91, 99 to nan - keep 1,2,3
# DRQSPREP - set 9 to nan
# DR1STY - set 9 to nan
# DRD360 - set 7, 9 to nan

    Diet_2015_DR1TOT_I_clean3 = np.copy(Diet_2015_DR1TOT_I_clean2)
    
    ChangeVars = ['DBQ095Z', 'DRQSPREP', 'DR1STY', 'DRD360']  
    
    ChangeVars_idx = []
    for i_str in ChangeVars:
        ChangeVars_idx.append(Diet_2015_DR1TOT_I_clean2_lbl.index(i_str))
    
    i = 0
    # np.sum(np.isnan(Diet_2015_DR1TOT_I_clean1[:, FilterVars_idx[i]])*1) # find nan
    bool_array0 = Diet_2015_DR1TOT_I_clean3[:, ChangeVars_idx[i]] == 2
    bool_array1 = Diet_2015_DR1TOT_I_clean3[:, ChangeVars_idx[i]] == 3
    bool_array2 = Diet_2015_DR1TOT_I_clean3[:, ChangeVars_idx[i]] == 91
    bool_array3 = Diet_2015_DR1TOT_I_clean3[:, ChangeVars_idx[i]] == 99
    bool_array_tot = np.logical_or(np.logical_or(np.logical_or(bool_array0, bool_array1), bool_array2), bool_array3)
    np.sum(bool_array_tot)
    Diet_2015_DR1TOT_I_clean3[bool_array_tot, ChangeVars_idx[i]] = float('nan')
    
    i = 1
    bool_array0 = Diet_2015_DR1TOT_I_clean3[:, ChangeVars_idx[i]] == 9
    np.sum(bool_array0)
    Diet_2015_DR1TOT_I_clean3[bool_array0, ChangeVars_idx[i]] = float('nan')
    
    i = 2
    bool_array0 = Diet_2015_DR1TOT_I_clean3[:, ChangeVars_idx[i]] == 9
    np.sum(bool_array0)
    Diet_2015_DR1TOT_I_clean3[bool_array0, ChangeVars_idx[i]] = float('nan')
    
    i = 3
    bool_array0 = Diet_2015_DR1TOT_I_clean3[:, ChangeVars_idx[i]] == 7
    bool_array1 = Diet_2015_DR1TOT_I_clean3[:, ChangeVars_idx[i]] == 9
    bool_array_tot = np.logical_or(bool_array0, bool_array1)
    np.sum(bool_array_tot)
    Diet_2015_DR1TOT_I_clean3[bool_array_tot, ChangeVars_idx[i]] = float('nan')
    
    Diet_2015_DR1TOT_I_clean3.shape
    
    Diet_2015_DR1TOT_I_clean3_lbl = Diet_2015_DR1TOT_I_clean2_lbl

# Save

with open(Save_folder + Diet_var_str + '_' + year + '_clean3.pkl', 'wb') as f:
    pickle.dump([Diet_2015_DR1TOT_I_clean3, Diet_2015_DR1TOT_I_clean3_lbl], f)

with open(Save_folder + Diet_var_str + '_' + year + '_clean3.pkl', 'rb') as f:
    [Diet_2015_DR1TOT_I_clean3, Diet_2015_DR1TOT_I_clean3_lbl] = pickle.load(f)

# This code for saving doesn't work for now.
if 0:
    for Condense in [0]:
        os.chdir(savefolder)
        Source = homefolder+filename
        hf = h5py.File('Diet_2015_DR1TOT_I_clean3.h5', 'w')
        # hf.create_dataset(Source, (200,), dtype="S10")
        hf.create_dataset('Diet_2015_DR1TOT_I_clean3', data=Diet_2015_DR1TOT_I_clean3)
        hf.create_dataset('Diet_2015_DR1TOT_I_clean3_lbl', data=Diet_2015_DR1TOT_I_clean3_lbl)
        
        hf.close()
        
        type(Diet_2015_DR1TOT_I_clean3)
        type(Diet_2015_DR1TOT_I_clean3_lbl)
