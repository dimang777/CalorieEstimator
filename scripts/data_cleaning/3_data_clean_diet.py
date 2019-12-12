import os
import numpy as np
import pickle

###############################################################################
# Set up folders and variables
###############################################################################
filename = '3_data_clean_diet.py'
load_folder = '../../data/raw_formatted/'
save_folder = '../../data/clean/'

demo_var_str = 'demo'
diet_var_str = 'diet'
exam_var_str = 'exam'
lab_var_str = 'lab'

ext = '.xpt'

totalvars_num = 0

year = '2015'

with open(load_folder + diet_var_str + '_' + year + '_raw_loadstr.pkl', 'rb') as f:
    demo_save_variables_str = pickle.load(f)

with open(load_folder + diet_var_str + '_' + year + '_raw.pkl', 'rb') as f:
    exec(demo_save_variables_str + '= pickle.load(f)')



for condense in [0]: # Code folding
    cols_toremove = ['WTDR2D',
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


###############################################################################
# Remove features
###############################################################################

for condense in [0]:
    cols_toremove_idx = []
    for i in range(0,len(cols_toremove)):
        cols_toremove_idx.append(int(diet_2015_DR1TOT_I_raw_lbl.index(cols_toremove[i])))

    cols_tokeep_idx = []
    for i in range(0,diet_2015_DR1TOT_I_raw.shape[1]):
        if i not in cols_toremove_idx:
            cols_tokeep_idx.append(int(i))

    diet_2015_DR1TOT_I_clean1 = np.copy(diet_2015_DR1TOT_I_raw[:,cols_tokeep_idx])
    
    print(diet_2015_DR1TOT_I_clean1.shape)
    
    diet_2015_DR1TOT_I_clean1_lbl = []
    for i in cols_tokeep_idx:
        diet_2015_DR1TOT_I_clean1_lbl.append(diet_2015_DR1TOT_I_raw_lbl[i])

###############################################################################
# Filtering - replace or remove meaningless numbers
###############################################################################
for condense in [0]:
# WTDRD1 - remove 0 and . (no nans) - other are continuous variable
# DR1DRSTZ - remove 2,4,5,. (no nans) - keep only 1
# DRDINT - remove . (nans) - keep 1 and 2
# DRQSDIET - remove 1, 9, . - keep only 2
# DR1_300 - remove 1, 3, 7, 9, . - keep only 2

    filtervars = ['WTDRD1', 'DR1DRSTZ', 'DRDINT', 'DRQSDIET', 'DR1_300']  
    
    filtervars_idx = []
    for i_str in filtervars:
        filtervars_idx.append(diet_2015_DR1TOT_I_clean1_lbl.index(i_str))
    
    i = 0
    keep_flag_clean1 = diet_2015_DR1TOT_I_clean1[:, filtervars_idx[i]] != 0
    
    i = 1
    keep_flag_clean1 = np.logical_and(keep_flag_clean1, diet_2015_DR1TOT_I_clean1[:, filtervars_idx[i]] == 1)
    
    i = 2
    keep_flag_clean1 = np.logical_and(keep_flag_clean1, np.logical_or(diet_2015_DR1TOT_I_clean1[:, filtervars_idx[i]] == 1, diet_2015_DR1TOT_I_clean1[:, filtervars_idx[i]] == 2))
    # np.isnan(diet_2015_DR1TOT_I_clean1[:, filtervars_idx[i]]) # alternative approach - code not complete
    
    i = 3
    keep_flag_clean1 = np.logical_and(keep_flag_clean1, diet_2015_DR1TOT_I_clean1[:, filtervars_idx[i]] == 2)
    
    i = 4
    keep_flag_clean1 = np.logical_and(keep_flag_clean1, diet_2015_DR1TOT_I_clean1[:, filtervars_idx[i]] == 2)
    
    np.sum(keep_flag_clean1)

###############################################################################
# remove filtered variables
###############################################################################
for condense in [0]:
    filtervars_toRemove_idx = []
    for i in range(1,len(filtervars)): # should keep WTDRD1
        filtervars_toRemove_idx.append(int(diet_2015_DR1TOT_I_clean1_lbl.index(filtervars[i])))
    
    cols_tokeep_aftfilt_idx = []
    for i in range(0,diet_2015_DR1TOT_I_clean1.shape[1]):
        if i not in filtervars_toRemove_idx:
            cols_tokeep_aftfilt_idx.append(int(i))
    
    # remove columns and rows at the same time
    diet_2015_DR1TOT_I_clean2_temp = np.copy(diet_2015_DR1TOT_I_clean1[keep_flag_clean1, :])
    diet_2015_DR1TOT_I_clean2 = np.copy(diet_2015_DR1TOT_I_clean2_temp[:, cols_tokeep_aftfilt_idx])
    
    diet_2015_DR1TOT_I_clean2.shape
    
    diet_2015_DR1TOT_I_clean2_lbl = []
    for i in cols_tokeep_aftfilt_idx:
        diet_2015_DR1TOT_I_clean2_lbl.append(diet_2015_DR1TOT_I_clean1_lbl[i])
    
    len(diet_2015_DR1TOT_I_clean2_lbl)

###############################################################################
# Replace with nans
###############################################################################
for condense in [0]:
# DBQ095Z - set 2, 3, 91, 99 to nan - keep 1,2,3
# DRQSPREP - set 9 to nan
# DR1STY - set 9 to nan
# DRD360 - set 7, 9 to nan

    diet_2015_DR1TOT_I_clean3 = np.copy(diet_2015_DR1TOT_I_clean2)
    
    changevars = ['DBQ095Z', 'DRQSPREP', 'DR1STY', 'DRD360']  
    
    changevars_idx = []
    for i_str in changevars:
        changevars_idx.append(diet_2015_DR1TOT_I_clean2_lbl.index(i_str))
    
    i = 0
    # np.sum(np.isnan(diet_2015_DR1TOT_I_clean1[:, filtervars_idx[i]])*1) # find nan
    bool_array0 = diet_2015_DR1TOT_I_clean3[:, changevars_idx[i]] == 2
    bool_array1 = diet_2015_DR1TOT_I_clean3[:, changevars_idx[i]] == 3
    bool_array2 = diet_2015_DR1TOT_I_clean3[:, changevars_idx[i]] == 91
    bool_array3 = diet_2015_DR1TOT_I_clean3[:, changevars_idx[i]] == 99
    bool_array_tot = np.logical_or(np.logical_or(np.logical_or(bool_array0, bool_array1), bool_array2), bool_array3)
    np.sum(bool_array_tot)
    diet_2015_DR1TOT_I_clean3[bool_array_tot, changevars_idx[i]] = float('nan')
    
    i = 1
    bool_array0 = diet_2015_DR1TOT_I_clean3[:, changevars_idx[i]] == 9
    np.sum(bool_array0)
    diet_2015_DR1TOT_I_clean3[bool_array0, changevars_idx[i]] = float('nan')
    
    i = 2
    bool_array0 = diet_2015_DR1TOT_I_clean3[:, changevars_idx[i]] == 9
    np.sum(bool_array0)
    diet_2015_DR1TOT_I_clean3[bool_array0, changevars_idx[i]] = float('nan')
    
    i = 3
    bool_array0 = diet_2015_DR1TOT_I_clean3[:, changevars_idx[i]] == 7
    bool_array1 = diet_2015_DR1TOT_I_clean3[:, changevars_idx[i]] == 9
    bool_array_tot = np.logical_or(bool_array0, bool_array1)
    np.sum(bool_array_tot)
    diet_2015_DR1TOT_I_clean3[bool_array_tot, changevars_idx[i]] = float('nan')
    
    diet_2015_DR1TOT_I_clean3.shape
    
    diet_2015_DR1TOT_I_clean3_lbl = diet_2015_DR1TOT_I_clean2_lbl

###############################################################################
# Save
###############################################################################

with open(save_folder + diet_var_str + '_' + year + '_clean3.pkl', 'wb') as f:
    pickle.dump([diet_2015_DR1TOT_I_clean3, diet_2015_DR1TOT_I_clean3_lbl], f)

with open(save_folder + diet_var_str + '_' + year + '_clean3.pkl', 'rb') as f:
    [diet_2015_DR1TOT_I_clean3, diet_2015_DR1TOT_I_clean3_lbl] = pickle.load(f)
