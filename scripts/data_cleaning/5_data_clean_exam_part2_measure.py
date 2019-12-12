import numpy as np
import pickle
from utils import removeval

###############################################################################
# Set up folders and variables
###############################################################################
filename = '5_data_clean_exam_part2_measure.py'
load_folder = '../../data/raw_formatted/'
save_folder = '../../data/clean/'

demo_var_str = 'demo'
diet_var_str = 'diet'
exam_var_str = 'exam'
lab_var_str = 'lab'

year = '2015'

with open(load_folder + exam_var_str + '_' + year + '_raw_loadstr.pkl', 'rb') as f:
    demo_save_variables_str = pickle.load(f)

with open(load_folder + exam_var_str + '_' + year + '_raw.pkl', 'rb') as f:
    exec(demo_save_variables_str + '= pickle.load(f)')

# Part 1 - BP
# -Part 2 - measurements
# Part 3 - dental


for condense in [0]:
    cols_toremove = ['BMIWT', 
    'BMXRECUM', 
    'BMIRECUM',
    'BMXHEAD',
    'BMIHEAD',
    'BMIHT',
    'BMDBMIC', 
    'BMILEG',
    'BMIARML',
    'BMIARMC',
    'BMIWAIST',
    'BMXSAD1',
    'BMXSAD2',
    'BMXSAD3',
    'BMXSAD4',
    'BMDSADCM']


# Remove variables
for condense in [0]:
    [exam_2015_BMX_I_clean1, exam_2015_BMX_I_clean1_lbl] = removeval(cols_toremove, exam_2015_BMX_I_raw, exam_2015_BMX_I_raw_lbl)

    print(exam_2015_BMX_I_clean1.shape)
    print(exam_2015_BMX_I_clean1_lbl)


# Filter
for condense in [0]:
# BMDSTATS - remove 3, 4, and nan - keep 1 and 2 

    filtervars = ['BMDSTATS']  
    
    filtervars_idx = []
    for i_str in filtervars:
        filtervars_idx.append(exam_2015_BMX_I_clean1_lbl.index(i_str))
    
    i = 0
    keep_flag_clean1 = ~np.isnan(exam_2015_BMX_I_clean1[:, filtervars_idx[i]])
        
    keep_flag_clean1 = np.logical_and(keep_flag_clean1, np.logical_or(exam_2015_BMX_I_clean1[:, filtervars_idx[i]] == 1, exam_2015_BMX_I_clean1[:, filtervars_idx[i]] == 2))
    
    np.sum(keep_flag_clean1)
    
    exam_2015_BMX_I_clean2 = np.copy(exam_2015_BMX_I_clean1[keep_flag_clean1, :])
    
    exam_2015_BMX_I_clean2_lbl = (exam_2015_BMX_I_clean1_lbl)

    print(exam_2015_BMX_I_clean2.shape)
    print(exam_2015_BMX_I_clean2_lbl)

     # Save
    with open(save_folder + exam_var_str + '_' + year + '_BMX_clean3.pkl', 'wb') as f:
        pickle.dump([exam_2015_BMX_I_clean2, exam_2015_BMX_I_clean2_lbl], f)
    
    with open(save_folder + exam_var_str + '_' + year + '_BMX_clean3.pkl', 'rb') as f:
        [exam_2015_BMX_I_clean2, exam_2015_BMX_I_clean2_lbl] = pickle.load(f)
