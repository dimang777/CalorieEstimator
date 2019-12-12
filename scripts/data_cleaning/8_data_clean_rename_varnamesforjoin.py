# Prepare the data to put everything into pandas dataframe
import pickle


###############################################################################
# Set up folders and variables
###############################################################################
filename = '8_data_clean_rename_varnamesforjoin.py'
save_folder = '../../data/clean/'
load_folder = '../../data/clean/'
load_folder2 = '../../data/raw_formatted/'

demo_var_str = 'demo'
diet_var_str = 'diet'
exam_var_str = 'exam'
lab_var_str = 'lab'

year = '2015'

with open(load_folder2 + lab_var_str + '_' + year + '_raw_loadstr.pkl', 'rb') as f:
    lab_save_variables_str = pickle.load(f)

with open(load_folder2 + lab_var_str + '_' + year + '_raw.pkl', 'rb') as f:
    exec(lab_save_variables_str + '= pickle.load(f)')

with open(load_folder + demo_var_str + '_' + year + '_clean3.pkl', 'rb') as f:
    [demo_2015_DEMO_I_clean3, demo_2015_DEMO_I_clean3_lbl, SEQN_del] = pickle.load(f)

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

# Renamed variables to indicate type and file incase the variable names overlap
# e.g., L1_RYAGE
# L is lab
# 1 is the first file in dict
# The rest RYAGE is the variable name



demo_filenames = ['DEMO_I']

varnames_demo_2015_clean3_dict = {'DEMO_I': 'demo_2015_DEMO_I_clean3'}

varnames_demo_2015_clean3_lbl_dict = {'DEMO_I': 'demo_2015_DEMO_I_clean3_lbl'}

demo_filename_varname_clean3_dict = {'DEMO_I': demo_2015_DEMO_I_clean3_lbl}


diet_filenames = ['DR1TOT_I']

varnames_diet_2015_clean3_dict = {'DR1TOT_I': 'diet_2015_DR1TOT_I_clean3'}

varnames_diet_2015_clean3_lbl_dict = {'DR1TOT_I': 'diet_2015_DR1TOT_I_clean3_lbl'}

diet_filename_varname_clean3_dict = {'DR1TOT_I': diet_2015_DR1TOT_I_clean3_lbl}



exam_filenames = ['BPX_I', 'BMX_I', 'OHXDEN_I']
varnames_exam_2015_clean3_dict = {'BPX_I': 'exam_2015_BPX_I_clean3', \
                                  'BMX_I': 'exam_2015_BMX_I_clean2', \
                                  'OHXDEN_I': 'exam_2015_OHXDEN_I_clean3'}

varnames_exam_2015_clean3_lbl_dict = {'BPX_I': 'exam_2015_BPX_I_clean3_lbl', \
                                  'BMX_I': 'exam_2015_BMX_I_clean2_lbl', \
                                  'OHXDEN_I': 'exam_2015_OHXDEN_I_clean3_lbl'}

exam_filename_varname_clean3_dict = {'BPX_I': exam_2015_BPX_I_clean3_lbl, \
                                  'BMX_I': exam_2015_BMX_I_clean2_lbl, \
                                  'OHXDEN_I': exam_2015_OHXDEN_I_clean3_lbl}


files_not_used = ['FASTQX_I',
        'HEPC_I',
        'HIV_I',
        'UCFLOW_I']
lab_filenames = []
for i_str in lab_filenames_pre:
    if i_str not in files_not_used:
        lab_filenames.append(i_str)




# Change the names of columns
def renameforpd(filenames, filename_varname_dict, Letter):
    # Letter is:
    # D for demo
    # I for diet
    # E for exam
    # L for lab
    filename_varname_pd_dict = {}
    for i, i_str in enumerate(filenames):
        modified_names = []
        for j in range(0,len(filename_varname_dict[i_str])):
            if j == 0:
                modified_names.append(filename_varname_dict[i_str][j])
            else:
                modified_names.append(Letter+str(i)+'_'+filename_varname_dict[i_str][j])
        filename_varname_pd_dict[i_str] = modified_names
    return filename_varname_pd_dict


demo_filename_varname_pd_dict = renameforpd(demo_filenames, demo_filename_varname_clean3_dict, 'D')

diet_filename_varname_pd_dict = renameforpd(diet_filenames, diet_filename_varname_clean3_dict, 'I')

exam_filename_varname_pd_dict = renameforpd(exam_filenames, exam_filename_varname_clean3_dict, 'E')

lab_filename_varname_pd_dict = renameforpd(lab_filenames, lab_filename_varname_clean2_dict, 'L')


print(demo_filenames)
print(varnames_demo_2015_clean3_dict)
print(demo_filename_varname_pd_dict)

print(diet_filenames)
print(varnames_diet_2015_clean3_dict)
print(diet_filename_varname_pd_dict)

print(exam_filenames)
print(varnames_exam_2015_clean3_dict)
print(exam_filename_varname_pd_dict)

print(lab_filenames)
print(varnames_lab_2015_clean2_dict)
print(lab_filename_varname_pd_dict)

    

# Save
with open(save_folder + 'prepfordf_' + year + '_rename.pkl', 'wb') as f:
    pickle.dump([demo_filenames, \
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
                    lab_filename_varname_pd_dict], f)

with open(save_folder + 'prepfordf_' + year + '_rename.pkl', 'rb') as f:
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




