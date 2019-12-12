import numpy as np
import pickle
from utils import removeval, str2num_1darray

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

ext = '.xpt'

totalvars_num = 0

year = '2015'

with open(load_folder + exam_var_str + '_' + year + '_raw_loadstr.pkl', 'rb') as f:
    demo_save_variables_str = pickle.load(f)

with open(load_folder + exam_var_str + '_' + year + '_raw.pkl', 'rb') as f:
    exec(demo_save_variables_str + '= pickle.load(f)')

# Part 1 - BP
# Part 2 - measurements
# -Part 3 - dental


cols_toremove = ['OHXIMP',
    'OHX02CSC',
    'OHX03CSC',
    'OHX04CSC',
    'OHX05CSC',
    'OHX06CSC',
    'OHX07CSC',
    'OHX08CSC',
    'OHX09CSC',
    'OHX10CSC',
    'OHX11CSC',
    'OHX12CSC',
    'OHX13CSC',
    'OHX14CSC',
    'OHX15CSC',
    'OHX18CSC',
    'OHX19CSC',
    'OHX20CSC',
    'OHX21CSC',
    'OHX22CSC',
    'OHX23CSC',
    'OHX24CSC',
    'OHX25CSC',
    'OHX26CSC',
    'OHX27CSC',
    'OHX28CSC',
    'OHX29CSC',
    'OHX30CSC',
    'OHX31CSC',
    'OHXRCAR',
    'OHXRCARO',
    'OHXRRES',
    'OHXRRESO',
    'OHX02SE',
    'OHX03SE',
    'OHX04SE',
    'OHX05SE',
    'OHX07SE',
    'OHX10SE',
    'OHX12SE',
    'OHX13SE',
    'OHX14SE',
    'OHX15SE',
    'OHX18SE',
    'OHX19SE',
    'OHX20SE',
    'OHX21SE',
    'OHX28SE',
    'OHX29SE',
    'OHX30SE',
    'OHX31SE']
    

# Remove variables
for condense in [0]:
    [exam_2015_OHXDEN_I_clean1, exam_2015_OHXDEN_I_clean1_lbl] = removeval(cols_toremove, exam_2015_OHXDEN_I_raw, exam_2015_OHXDEN_I_raw_lbl)

    print(exam_2015_OHXDEN_I_clean1.shape)
    print(exam_2015_OHXDEN_I_clean1_lbl)
    
    


# Average

cols_toavg = ['OHX01TC',
         'OHX02TC',
         'OHX03TC',
         'OHX04TC',
         'OHX05TC',
         'OHX06TC',
         'OHX07TC',
         'OHX08TC',
         'OHX09TC',
         'OHX10TC',
         'OHX11TC',
         'OHX12TC',
         'OHX13TC',
         'OHX14TC',
         'OHX15TC',
         'OHX16TC',
         'OHX17TC',
         'OHX18TC',
         'OHX19TC',
         'OHX20TC',
         'OHX21TC',
         'OHX22TC',
         'OHX23TC',
         'OHX24TC',
         'OHX25TC',
         'OHX26TC',
         'OHX27TC',
         'OHX28TC',
         'OHX29TC',
         'OHX30TC',
         'OHX31TC',
         'OHX32TC']
    
        
cols_toavg_2 = ['OHX02CTC',
         'OHX03CTC',
         'OHX04CTC',
         'OHX05CTC',
         'OHX06CTC',
         'OHX07CTC',
         'OHX08CTC',
         'OHX09CTC',
         'OHX10CTC',
         'OHX11CTC',
         'OHX12CTC',
         'OHX13CTC',
         'OHX14CTC',
         'OHX15CTC',
         'OHX18CTC',
         'OHX19CTC',
         'OHX20CTC',
         'OHX21CTC',
         'OHX22CTC',
         'OHX23CTC',
         'OHX24CTC',
         'OHX25CTC',
         'OHX26CTC',
         'OHX27CTC',
         'OHX28CTC',
         'OHX29CTC',
         'OHX30CTC',
         'OHX31CTC']

# Remaining permanent teeth count

cols_toavg_idx = []
for i in range(0,len(cols_toavg)):
    cols_toavg_idx.append(int(exam_2015_OHXDEN_I_clean1_lbl.index(cols_toavg[i])))

intactteeth_count = np.zeros(exam_2015_OHXDEN_I_clean1.shape[0], 'int')

for j in range(0,len(cols_toavg_idx)):

    array_num = str2num_1darray(exam_2015_OHXDEN_I_clean1[:, cols_toavg_idx[j]])
    
    intactteeth_count = intactteeth_count + (array_num == 2)*1

# Replace the first variable with teeth count
exam_2015_OHXDEN_I_clean2 = np.zeros([exam_2015_OHXDEN_I_clean1.shape[0], 7])
exam_2015_OHXDEN_I_clean2.shape

for i in range(0,3):
    exam_2015_OHXDEN_I_clean2[:, i] = str2num_1darray(exam_2015_OHXDEN_I_clean1[:, i])
    
exam_2015_OHXDEN_I_clean2[:, 3] = np.copy(intactteeth_count)


# damanged, undamaged - get a score based on damaged, undamaged, and missing - so perhaps score of dental deterioration
# damanged - 0.5. missing - 1 - something like that
# Damaged permanent teeth count
# need to calculate missing number of teeth - 31 or 30 minus remaining teeth

# Damaged teeth count
cols_toavg_2_idx = []
for i in range(0,len(cols_toavg_2)):
    cols_toavg_2_idx.append(int(exam_2015_OHXDEN_I_clean1_lbl.index(cols_toavg_2[i])))

damagedteeth_count = np.zeros(exam_2015_OHXDEN_I_clean1.shape[0], 'int')

for j in range(0,len(cols_toavg_2_idx)):

    single_count_array = np.logical_or(exam_2015_OHXDEN_I_clean1[:, cols_toavg_2_idx[0]] == 'F',
                                       exam_2015_OHXDEN_I_clean1[:, cols_toavg_2_idx[0]] == 'J')
    single_count_array = np.logical_or(single_count_array,
                                       exam_2015_OHXDEN_I_clean1[:, cols_toavg_2_idx[0]] == 'T')
    single_count_array = np.logical_or(single_count_array,
                                       exam_2015_OHXDEN_I_clean1[:, cols_toavg_2_idx[0]] == 'Z')
    
    damagedteeth_count = damagedteeth_count + single_count_array*1

exam_2015_OHXDEN_I_clean2[:, 4] = np.copy(damagedteeth_count)

missingteeth_count = 32 - intactteeth_count

exam_2015_OHXDEN_I_clean2[:, 5] = np.copy(missingteeth_count)

TeethDeterioration_Score = missingteeth_count + 0.5*damagedteeth_count

exam_2015_OHXDEN_I_clean2[:, 6] = np.copy(TeethDeterioration_Score)

exam_2015_OHXDEN_I_clean2_lbl = exam_2015_OHXDEN_I_clean1_lbl[0:3]
exam_2015_OHXDEN_I_clean2_lbl.append('CUS_INTTEETH')
exam_2015_OHXDEN_I_clean2_lbl.append('CUS_DMGTEETH')
exam_2015_OHXDEN_I_clean2_lbl.append('CUS_MISTEETH')
exam_2015_OHXDEN_I_clean2_lbl.append('CUS_DETERSCORE')
# Filter data
for condense in [0]:
# OHDEXSTS - keep only 1
# OHDDESTS - keep only 1

    filtervars = ['OHDEXSTS', 'OHDDESTS']  
    
    filtervars_idx = []
    for i_str in filtervars:
        filtervars_idx.append(exam_2015_OHXDEN_I_clean2_lbl.index(i_str))
    
    i = 0
    # np.sum(np.isnan(Diet_2015_DR1TOT_I_clean1[:, filtervars_idx[i]])*1) # find nan
    keep_flag_clean2 = exam_2015_OHXDEN_I_clean2[:, filtervars_idx[i]] == 1

    i = 1
    keep_flag_clean2 = np.logical_and(keep_flag_clean2, exam_2015_OHXDEN_I_clean2[:, filtervars_idx[i]] == 1)
    
    np.sum(keep_flag_clean2)
    
    # remove filtered variables
for condense in [0]:
    filtervars_toremove_idx = []
    for i in range(0,len(filtervars)):
        filtervars_toremove_idx.append(int(exam_2015_OHXDEN_I_clean2_lbl.index(filtervars[i])))
    
    cols_tokeep_aftfilt_idx = []
    for i in range(0,exam_2015_OHXDEN_I_clean2.shape[1]):
        if i not in filtervars_toremove_idx:
            cols_tokeep_aftfilt_idx.append(int(i))
    
    # remove columns and rows at the same time
    exam_2015_OHXDEN_I_clean3_temp = np.copy(exam_2015_OHXDEN_I_clean2[keep_flag_clean2, :])
    exam_2015_OHXDEN_I_clean3 = np.copy(exam_2015_OHXDEN_I_clean3_temp[:, cols_tokeep_aftfilt_idx])
    
    exam_2015_OHXDEN_I_clean3.shape
    
    exam_2015_OHXDEN_I_clean3_lbl = []
    for i in cols_tokeep_aftfilt_idx:
        exam_2015_OHXDEN_I_clean3_lbl.append(exam_2015_OHXDEN_I_clean2_lbl[i])
    
    len(exam_2015_OHXDEN_I_clean3_lbl)
    
    # Save
    with open(save_folder + exam_var_str + '_' + year + '_OHXDEN_clean3.pkl', 'wb') as f:
        pickle.dump([exam_2015_OHXDEN_I_clean3, exam_2015_OHXDEN_I_clean3_lbl], f)
    
    with open(save_folder + exam_var_str + '_' + year + '_OHXDEN_clean3.pkl', 'rb') as f:
        [exam_2015_OHXDEN_I_clean3, exam_2015_OHXDEN_I_clean3_lbl] = pickle.load(f)


