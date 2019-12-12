import numpy as np
import pickle
from utils import removeval

###############################################################################
# Set up folders and variables
###############################################################################
filename = '4_data_clean_exam_part1_bp.py'
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
# Part 2 - measurements
# Part 3 - dental

for Condense in [0]:
    cols_toremove = ['PEASCCT1', 
    'BPXCHR', 
    'BPAARM',
    'BPACSZ',
    'BPXPULS',
    'BPXPTY',
    'BPXML1', 
    'BPXSY1',
    'BPXDI1',
    'BPAEN1',
    'BPAEN2',
    'BPAEN3',
    'BPXSY4',
    'BPXDI4',
    'BPAEN4']

# Remove variables
for Condense in [0]:
    [exam_2015_BPX_I_clean1, exam_2015_BPX_I_clean1_lbl] = removeval(cols_toremove, exam_2015_BPX_I_raw, exam_2015_BPX_I_raw_lbl)

    print(exam_2015_BPX_I_clean1.shape)
    print(exam_2015_BPX_I_clean1_lbl)


# Average and replace second measurements - BPXSY2 and BPXDI2 (remove third afterwards - BPXSY3 and BPXDI3)
for Condense in [0]:

    exam_2015_BPX_I_clean2_temp = np.copy(exam_2015_BPX_I_clean1)
    
    # SBP
    idx_2 = exam_2015_BPX_I_clean1_lbl.index('BPXSY2')
    idx_3 = exam_2015_BPX_I_clean1_lbl.index('BPXSY3')
    
    avg_flag = np.logical_and(~np.isnan(exam_2015_BPX_I_clean1[:, idx_2]),
                              ~np.isnan(exam_2015_BPX_I_clean1[:, idx_3]))
    
    exam_2015_BPX_I_clean2_temp[avg_flag, idx_2] = np.round((exam_2015_BPX_I_clean1[avg_flag, idx_2] + exam_2015_BPX_I_clean1[avg_flag, idx_3])/2)
    
    
    # DBP
    idx_2 = exam_2015_BPX_I_clean1_lbl.index('BPXDI2')
    idx_3 = exam_2015_BPX_I_clean1_lbl.index('BPXDI3')
    
    avg_flag = np.logical_and(~np.isnan(exam_2015_BPX_I_clean1[:, idx_2]),
                              ~np.isnan(exam_2015_BPX_I_clean1[:, idx_3]))
    
    exam_2015_BPX_I_clean2_temp[avg_flag, idx_2] = np.round((exam_2015_BPX_I_clean1[avg_flag, idx_2] + exam_2015_BPX_I_clean1[avg_flag, idx_3])/2)
    
    cols_aftavg_toremove = ['BPXSY3', 
        'BPXDI3']
    
    [exam_2015_BPX_I_clean2, exam_2015_BPX_I_clean2_lbl] = removeval(cols_aftavg_toremove, exam_2015_BPX_I_clean2_temp, exam_2015_BPX_I_clean1_lbl)
 
    print(exam_2015_BPX_I_clean2.shape)
    print(exam_2015_BPX_I_clean2_lbl)

# Filter
for Condense in [0]:
# BPXPLS - remove nans - other are continuous variable
# BPXSY2 - remove nans - 
# BPXDI2 - remove nans -

    filtervars = ['BPXPLS', 'BPXSY2', 'BPXDI2']  
    
    filtervars_idx = []
    for i_str in filtervars:
        filtervars_idx.append(exam_2015_BPX_I_clean2_lbl.index(i_str))
    
    i = 0
    keep_flag_clean2 = ~np.isnan(exam_2015_BPX_I_clean2[:, filtervars_idx[i]])
    
    i = 1
    keep_flag_clean2 = np.logical_and(keep_flag_clean2, ~np.isnan(exam_2015_BPX_I_clean2[:, filtervars_idx[i]]))
    
    i = 2
    keep_flag_clean2 = np.logical_and(keep_flag_clean2, ~np.isnan(exam_2015_BPX_I_clean2[:, filtervars_idx[i]]))
    
    print(np.sum(keep_flag_clean2))
    
    exam_2015_BPX_I_clean3 = np.copy(exam_2015_BPX_I_clean2[keep_flag_clean2, :])
    
    exam_2015_BPX_I_clean3_lbl = exam_2015_BPX_I_clean2_lbl

    print(exam_2015_BPX_I_clean3.shape)
    print(exam_2015_BPX_I_clean3_lbl)

    # Save
    with open(save_folder + exam_var_str + '_' + year + '_BPX_clean3.pkl', 'wb') as f:
        pickle.dump([exam_2015_BPX_I_clean3, exam_2015_BPX_I_clean3_lbl], f)
    
    with open(save_folder + exam_var_str + '_' + year + '_BPX_clean3.pkl', 'rb') as f:
        [exam_2015_BPX_I_clean3, exam_2015_BPX_I_clean3_lbl] = pickle.load(f)
