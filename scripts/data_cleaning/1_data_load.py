import xport
import os
from os import walk
import pickle
import numpy as np


###############################################################################
# Set up folders and variables
###############################################################################
filename = '1_data_load.py'
save_folder = '../../data/raw_formatted/'
load_folder = '../../data/raw/NHANES_2015_2016/'

demo_folder_str = 'Demographics Data'
diet_folder_str = 'Dietary Data'
exam_folder_str = 'Examination Data'
lab_folder_str = 'Laboratory Data'

demo_var_str = 'Demo'
diet_var_str = 'Diet'
exam_var_str = 'Exam'
lab_var_str = 'Lab'

ext = '.xpt'

TotalVars_num = 0

year = '2015'

###############################################################################
# Misc
###############################################################################


# Worked
with open(save_folder+'test.pickle', 'wb') as handle:
    pickle.dump([filename], handle)

with open(save_folder+'test.pickle', 'rb') as handle:
    [filename] = pickle.load(handle)


###############################################################################
# Functions
###############################################################################

def get_filenames(mypath):
    # Get relevant file names in a folder. Files with an extension XPT.

    # Scan the folder for files
    f = []
    for (dirpath, dirnames, filenames_temp) in walk(mypath):
        f.extend(filenames_temp)

    # Get files with an extension XPT
    file_idx = []
    for i in range(len(filenames_temp)):
        if filenames_temp[i][-3:].upper() == 'XPT':
            file_idx.append(i) # Case 2: worked

    # Save and return file names
    filenames= []
    for i in range(len(file_idx)):
        filenames.append(filenames_temp[file_idx[i]][:-4])

    return filenames, file_idx

def load_files(mypath, ext, var_str, year, file_idx, filenames):
    # Load files unto the workspace. Each variable will have data from each file

    # Keep track of the variable names
    varnames = []

    for i in range(len(file_idx)):    
        print(i)
        # Load each file unto a designated variable
        with open(mypath+'/'+filenames[i]+ext, 'rb') as f:
            exec(var_str + '_' + year + '_' + filenames[i] + '_raw = xport.to_numpy(f)') # Worked
            # Count the total number of variables
            exec('TotalVars_num += ' + var_str + '_' + year + '_' + filenames[i] + '_raw.shape[1]')
        # Store loaded file information
        with open(mypath+'/'+filenames[i]+ext, 'rb') as f:
            exec(var_str + '_' + year + '_' + filenames[i] + '_raw_lbl = xport.Reader(f).fields')

        # Keep track of the variable names and the file information
        exec('varnames.append(\'' + var_str + '_' + year + '_' + filenames[i] + '_raw\')')
        exec('varnames.append(\'' + var_str + '_' + year + '_' + filenames[i] + '_raw_lbl\')')

    return varnames



# Demographic
if 1:
    print(demo_folder_str)
    mypath = load_folder + demo_folder_str

    # Get file names in the folder
    [filenames, file_idx] = get_filenames(mypath)


    # Load files unto the workspace.
    temp = load_files(mypath, ext, diet_var_str, year, file_idx, filenames)
    exec('varnames_' + diet_var_str + '_' + year + ' = temp')




    # Save variables
    Demo_save_code_str = 'pickle.dump('

    Demo_save_variables_str = '['
    for i in range(len(file_idx)):
        Demo_save_variables_str = Demo_save_variables_str + \
            demo_var_str + '_' + year + '_' + filenames[i] + '_raw, ' + \
            demo_var_str + '_' + year + '_' + filenames[i] + '_raw_lbl, '
    Demo_save_variables_str = Demo_save_variables_str + ']'

    Demo_save_code_str = Demo_save_code_str + Demo_save_variables_str + ', f)'


    with open(save_folder + demo_var_str + '_' + year + '_raw.pkl', 'wb') as f:
        exec(Demo_save_code_str)
    with open(save_folder + demo_var_str + '_' + year + '_raw_loadstr.pkl', 'wb') as f:
        pickle.dump(Demo_save_variables_str, f)


    with open(save_folder + demo_var_str + '_' + year + '_raw_loadstr.pkl', 'rb') as f:
        Demo_save_variables_str = pickle.load(f)

    with open(save_folder + demo_var_str + '_' + year + '_raw.pkl', 'rb') as f:
        exec(Demo_save_variables_str + '= pickle.load(f)')

# Diet
if 1:
    print(diet_folder_str)
    mypath = load_folder + diet_folder_str

    # Get file names in the folder
    [filenames, file_idx] = get_filenames(mypath)


    # Load files unto the workspace.
    temp = load_files(mypath, ext, diet_var_str, year, file_idx, filenames)
    exec('varnames_' + diet_var_str + '_' + year + ' = temp')



    # Save variables
    Diet_save_code_str = 'pickle.dump('

    Diet_save_variables_str = '['
    for i in range(len(file_idx)):
        Diet_save_variables_str = Diet_save_variables_str + \
            diet_var_str + '_' + year + '_' + filenames[i] + '_raw, ' + \
            diet_var_str + '_' + year + '_' + filenames[i] + '_raw_lbl, '
    Diet_save_variables_str = Diet_save_variables_str + ']'

    Diet_save_code_str = Diet_save_code_str + Diet_save_variables_str + ', f)'

    with open(save_folder + diet_var_str + '_' + year + '_raw.pkl', 'wb') as f:
        exec(Diet_save_code_str)
    with open(save_folder + diet_var_str + '_' + year + '_raw_loadstr.pkl', 'wb') as f:
        pickle.dump(Diet_save_variables_str, f)

    with open(save_folder + diet_var_str + '_' + year + '_raw_loadstr.pkl', 'rb') as f:
        Diet_save_variables_str = pickle.load(f)

    with open(save_folder + diet_var_str + '_' + year + '_raw.pkl', 'rb') as f:
        exec(Diet_save_variables_str + '= pickle.load(f)')

# Exam
if 1:
    print(exam_folder_str)
    mypath = load_folder + exam_folder_str

    # Get file names in the folder
    [filenames, file_idx] = get_filenames(mypath)

    # Load files unto the workspace.
    temp = load_files(mypath, ext, exam_var_str, year, file_idx, filenames)
    exec('varnames_' + exam_var_str + '_' + year + ' = temp')



    # Save variables
    Exam_save_code_str = 'pickle.dump('

    Exam_save_variables_str = '['
    for i in range(len(file_idx)):
        Exam_save_variables_str = Exam_save_variables_str + \
            exam_var_str + '_' + year + '_' + filenames[i] + '_raw, ' + \
            exam_var_str + '_' + year + '_' + filenames[i] + '_raw_lbl, '
    Exam_save_variables_str = Exam_save_variables_str + ']'

    Exam_save_code_str = Exam_save_code_str + Exam_save_variables_str + ', f)'

    with open(save_folder + exam_var_str + '_' + year + '_raw.pkl', 'wb') as f:
        exec(Exam_save_code_str)
    with open(save_folder + exam_var_str + '_' + year + '_raw_loadstr.pkl', 'wb') as f:
        pickle.dump(Exam_save_variables_str, f)

    with open(save_folder + exam_var_str + '_' + year + '_raw_loadstr.pkl', 'rb') as f:
        Exam_save_variables_str = pickle.load(f)

    with open(save_folder + exam_var_str + '_' + year + '_raw.pkl', 'rb') as f:
        exec(Exam_save_variables_str + '= pickle.load(f)')

# Lab - NOTE: HIV_I is missing from the beginning. That's okay. It's not used. But is different from other files not used.
if 1:

    print(lab_folder_str)
    mypath = load_folder + lab_folder_str

    # Get file names in the folder
    [filenames, file_idx] = get_filenames(mypath)

    # Load files unto the workspace.
    temp = load_files(mypath, ext, lab_var_str, year, file_idx, filenames)
    exec('varnames_' + lab_var_str + '_' + year + ' = temp')



    # Save variables
    Lab_save_code_str = 'pickle.dump('

    Lab_save_variables_str = '[Lab_filenames_Pre, '
    for i in range(len(file_idx)):
        Lab_save_variables_str = Lab_save_variables_str + \
            lab_var_str + '_' + year + '_' + filenames[i] + '_raw, ' + \
            lab_var_str + '_' + year + '_' + filenames[i] + '_raw_lbl, '
    Lab_save_variables_str = Lab_save_variables_str + ']'

    Lab_save_code_str = Lab_save_code_str + Lab_save_variables_str + ', f)'

    with open(save_folder + lab_var_str + '_' + year + '_raw.pkl', 'wb') as f:
        exec(Lab_save_code_str)
    with open(save_folder + lab_var_str + '_' + year + '_raw_loadstr.pkl', 'wb') as f:
        pickle.dump(Lab_save_variables_str, f)

    with open(save_folder + lab_var_str + '_' + year + '_raw_loadstr.pkl', 'rb') as f:
        Lab_save_variables_str = pickle.load(f)

    with open(save_folder + lab_var_str + '_' + year + '_raw.pkl', 'rb') as f:
        exec(Lab_save_variables_str + '= pickle.load(f)')

print(TotalVars_num)


# Deleted files: All extensions .xpt
# Lab: HIV_I
# Ques: ACQ_I
# Ques: BPQ_I