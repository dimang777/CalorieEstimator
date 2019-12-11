import numpy as np
import pickle

###############################################################################
# Set up folders and variables
###############################################################################
filename = '2_data_clean_demo.py'
load_folder = '../../data/raw_formatted/'
save_folder = '../../data/clean/'

demo_var_str = 'demo'
diet_var_str = 'diet'
exam_var_str = 'exam'
lab_var_str = 'lab'

ext = '.xpt'

totalvars_num = 0

year = '2015'

with open(load_folder + demo_var_str + '_' + year + '_raw_loadstr.pkl', 'rb') as f:
    demo_save_variables_str = pickle.load(f)

with open(load_folder + demo_var_str + '_' + year + '_raw.pkl', 'rb') as f:
    exec(demo_save_variables_str + '= pickle.load(f)')

# Remove subjects without examination
demo_2015_DEMO_I_clean1 = np.copy(demo_2015_DEMO_I_raw[demo_2015_DEMO_I_raw[:,2] == 2, :])

# Record deleted subject numbers
seqn_del_flag = demo_2015_DEMO_I_raw[:,2] != 2

# Remove age below or equal to 20
# 4149 subjects under 20 removed
age_flag = demo_2015_DEMO_I_raw[:,4] <= 20
seqn_del_flag = np.logical_or(seqn_del_flag, age_flag)
seqn_del_idx = np.where(seqn_del_flag)[0].tolist()
seqn_del_temp = list(demo_2015_DEMO_I_raw[seqn_del_idx,0])
seqn_del = [int(i) for i in seqn_del_temp]

demo_2015_DEMO_I_clean2 = np.copy(demo_2015_DEMO_I_clean1[demo_2015_DEMO_I_clean1[:,4] > 20, :])
demo_2015_DEMO_I_clean2.shape # 5395, 47

# Features to keep - manually determined using documents
cols_tokeep = [0, 4, 6, 12, 13, 17, 18, 40, 41, 44, 45, 46]

demo_2015_DEMO_I_clean3 = np.copy(demo_2015_DEMO_I_clean2[:,cols_tokeep])
demo_2015_DEMO_I_clean3_lbl = []
for i in cols_tokeep:
    demo_2015_DEMO_I_clean3_lbl.append(demo_2015_DEMO_I_raw_lbl[i])

# Save

with open(save_folder + demo_var_str + '_' + year + '_clean3.pkl', 'wb') as f:
    pickle.dump([demo_2015_DEMO_I_clean3, demo_2015_DEMO_I_clean3_lbl, seqn_del], f)

with open(save_folder + demo_var_str + '_' + year + '_clean3.pkl', 'rb') as f:
    [demo_2015_DEMO_I_clean3, demo_2015_DEMO_I_clean3_lbl, seqn_del] = pickle.load(f)

