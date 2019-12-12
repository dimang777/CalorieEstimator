# Prepare the data to put everything into pandas dataframe
import pickle
import pandas as pd

###############################################################################
# Set up folders and variables
###############################################################################
filename = '9_data_clean_formulate_pandas_df.py'
save_folder = '../../data/cleaned_df/'
load_folder = '../../data/clean/'

demo_var_str = 'demo'
diet_var_str = 'diet'
exam_var_str = 'exam'
lab_var_str = 'lab'

year = '2015'

with open(load_folder + demo_var_str + '_' + year + '_clean3.pkl', 'rb') as f:
    [demo_2015_DEMO_I_clean3, demo_2015_DEMO_I_clean3_lbl, seqn_del] = pickle.load(f)

with open(load_folder + diet_var_str + '_' + year + '_clean3.pkl', 'rb') as f:
    [diet_2015_DR1TOT_I_clean3, diet_2015_DR1TOT_I_clean3_lbl] = pickle.load(f)

with open(load_folder + exam_var_str + '_' + year + '_BPX_clean3.pkl', 'rb') as f:
    [exam_2015_BPX_I_clean3, exam_2015_BPX_I_clean3_lbl] = pickle.load(f)

with open(load_folder + exam_var_str + '_' + year + '_BMX_clean3.pkl', 'rb') as f:
    [exam_2015_BMX_I_clean2, exam_2015_BMX_I_clean2_lbl] = pickle.load(f)

with open(load_folder + exam_var_str + '_' + year + '_OHXDEN_clean3.pkl', 'rb') as f:
    [exam_2015_OHXDEN_I_clean3, exam_2015_OHXDEN_I_clean3_lbl] = pickle.load(f)

with open(load_folder + lab_var_str + '_' + year + '_clean3_loadstr.pkl', 'rb') as f:
    lab_save_variables_str = pickle.load(f)

with open(load_folder + lab_var_str + '_' + year + '_clean3.pkl', 'rb') as f:
    exec(lab_save_variables_str + ' = pickle.load(f)')


with open(load_folder + 'prepfordf_' + year + '_rename.pkl', 'rb') as f:
    [demo_filenames, \
                    varnames_demo_2015_clean3_dict, \
                    demo_filename_varname_pd_dict, \
                    diet_filenames, \
                    varnames_diet_2015_clean3_dict, \
                    diet_filename_varname_pd_dict, \
                    exam_filenames, \
                    varnames_exam_2015_clean3_dict, \
                    exam_filename_varname_pd_dict, \
                    lab_filenames, \
                    varnames_lab_2015_clean2_dict, \
                    lab_filename_varname_pd_dict] = pickle.load(f)




df_collection = {}
# demo
letter = 'D'
for i,i_str in enumerate(demo_filenames):
    exec('df_collection[\'' + letter  + '\'+str(i)] = pd.DataFrame(' + varnames_demo_2015_clean3_dict[i_str] + 
                        '[:,1:], columns = '+ demo_var_str +'_filename_varname_pd_dict[\''+ i_str +'\'][1:], index = ' + 
                        varnames_demo_2015_clean3_dict[i_str] + '[:,0].astype(int))')

# diet
letter = 'I'
for i,i_str in enumerate(diet_filenames):
    exec('df_collection[\'' + letter  + '\'+str(i)] = pd.DataFrame(' + varnames_diet_2015_clean3_dict[i_str] + 
                        '[:,1:], columns = '+ diet_var_str + '_filename_varname_pd_dict[\'' + i_str + '\'][1:], index = ' + 
                        varnames_diet_2015_clean3_dict[i_str] + '[:,0].astype(int))')

# exam
letter = 'E'
for i,i_str in enumerate(exam_filenames):
    exec('df_collection[\'' + letter  + '\'+str(i)] = pd.DataFrame(' + varnames_exam_2015_clean3_dict[i_str] + 
                        '[:,1:], columns = '+ exam_var_str +'_filename_varname_pd_dict[\'' + i_str + '\'][1:], index = ' + 
                        varnames_exam_2015_clean3_dict[i_str] + '[:,0].astype(int))')

# lab
letter = 'L'
for i,i_str in enumerate(lab_filenames):
    exec('df_collection[\'' + letter  + '\'+str(i)] = pd.DataFrame(' + varnames_lab_2015_clean2_dict[i_str] + \
                        '[:,1:], columns = ' + lab_var_str + '_filename_varname_pd_dict[\'' + i_str + '\'][1:], index = ' + \
                        varnames_lab_2015_clean2_dict[i_str] + '[:,0].astype(int))')

df_collection_key = ['D0', 'I0', 'E0', 'E1', 'E2']
for i in range(0, len(lab_filenames)):
    df_collection_key.append('L'+str(i))




df_collection['I0']

df_bfr_filter = df_collection[df_collection_key[0]].join(df_collection[df_collection_key[1]], how='outer')

for i_str in df_collection_key[2:]:
    df_bfr_filter = df_bfr_filter.join(df_collection[i_str], how='outer')

seqn_df = list(df_bfr_filter.index.values)

# Intersection Usage
# a=set([1,2,3,4])
# b=set([3,4,5,6])
# a.intersection(b)

a =set(seqn_df)
len(a)
b =set(seqn_del)
len(b)
c = a.intersection(b)
len(c)

df_bfr_demo_filter = df_bfr_filter.copy()
df_bfr_demo_filter = df_bfr_demo_filter.drop(c)
# df_bfr_demo_filter
# df_bfr_filter

with open(save_folder + 'df_bfr_demo_filter.pkl', 'wb') as f:
    pickle.dump([df_bfr_demo_filter, df_collection_key, \
                 demo_filenames, \
                 demo_filename_varname_pd_dict, \
                 diet_filenames, \
                 diet_filename_varname_pd_dict, \
                 exam_filenames, \
                 exam_filename_varname_pd_dict, \
                 lab_filenames, \
                 lab_filename_varname_pd_dict, \
                     ], f)

with open(save_folder + 'df_bfr_demo_filter.pkl', 'rb') as f:
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


# df_bfr_demo_filter






