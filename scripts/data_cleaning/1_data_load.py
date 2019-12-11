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

demo_var_str = 'demo'
diet_var_str = 'diet'
exam_var_str = 'exam'
lab_var_str = 'lab'

ext = '.xpt'

totalvars_num = 0

year = '2015'

# Deleted files: All extensions .xpt
# Lab: HIV_I
# Ques: ACQ_I
# Ques: BPQ_I

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

def generate_save_code(var_str, file_idx, filenames, year):
    save_code_str = 'pickle.dump('

    if var_str == 'lab':
        save_variables_str = '[lab_filenames_pre, '
    else:
        save_variables_str = '['

    for i in range(len(file_idx)):
        save_variables_str = save_variables_str + \
            var_str + '_' + year + '_' + filenames[i] + '_raw, ' + \
            var_str + '_' + year + '_' + filenames[i] + '_raw_lbl, '
    save_variables_str = save_variables_str + ']'

    save_code_str = save_code_str + save_variables_str + ', f)'

    return save_code_str, save_variables_str


###############################################################################
# Demographic
###############################################################################
if 1:
    print(demo_folder_str)
    mypath = load_folder + demo_folder_str

    # Get file names in the folder
    [filenames, file_idx] = get_filenames(mypath)


    # Load files unto the workspace. Each variable will have data from each file
    # Although repeating, this section could not be put into Functions
    # since direct loading to workspace was required.
    var_str = demo_var_str

    # Keep track of the variable names
    varnames = []

    for i in range(len(file_idx)):    
        print(i)
        # Load each file unto a designated variable
        with open(mypath+'/'+filenames[i]+ext, 'rb') as f:
            exec(var_str + '_' + year + '_' + filenames[i] + '_raw = xport.to_numpy(f)') # Worked
            # Count the total number of variables
            exec('totalvars_num += ' + var_str + '_' + year + '_' + filenames[i] + '_raw.shape[1]')
        # Store loaded file information
        with open(mypath+'/'+filenames[i]+ext, 'rb') as f:
            exec(var_str + '_' + year + '_' + filenames[i] + '_raw_lbl = xport.Reader(f).fields')

        # Keep track of the variable names and the file information
        exec('varnames.append(\'' + var_str + '_' + year + '_' + filenames[i] + '_raw\')')
        exec('varnames.append(\'' + var_str + '_' + year + '_' + filenames[i] + '_raw_lbl\')')

    exec('varnames_' + var_str + '_' + year + ' = varnames')


    # Generate code for saving
    demo_save_code_str, demo_save_variables_str = generate_save_code(demo_var_str, file_idx, filenames, year)

    # Save
    with open(save_folder + demo_var_str + '_' + year + '_raw.pkl', 'wb') as f:
        exec(demo_save_code_str)
    with open(save_folder + demo_var_str + '_' + year + '_raw_loadstr.pkl', 'wb') as f:
        pickle.dump(demo_save_variables_str, f)


    with open(save_folder + demo_var_str + '_' + year + '_raw_loadstr.pkl', 'rb') as f:
        demo_save_variables_str = pickle.load(f)

    with open(save_folder + demo_var_str + '_' + year + '_raw.pkl', 'rb') as f:
        exec(demo_save_variables_str + '= pickle.load(f)')

###############################################################################
# Diet
###############################################################################
if 1:
    print(diet_folder_str)
    mypath = load_folder + diet_folder_str

    # Get file names in the folder
    [filenames, file_idx] = get_filenames(mypath)


    # Load files unto the workspace. Each variable will have data from each file
    var_str = diet_var_str

    # Keep track of the variable names
    varnames = []

    for i in range(len(file_idx)):    
        print(i)
        # Load each file unto a designated variable
        with open(mypath+'/'+filenames[i]+ext, 'rb') as f:
            exec(var_str + '_' + year + '_' + filenames[i] + '_raw = xport.to_numpy(f)') # Worked
            # Count the total number of variables
            exec('totalvars_num += ' + var_str + '_' + year + '_' + filenames[i] + '_raw.shape[1]')
        # Store loaded file information
        with open(mypath+'/'+filenames[i]+ext, 'rb') as f:
            exec(var_str + '_' + year + '_' + filenames[i] + '_raw_lbl = xport.Reader(f).fields')

        # Keep track of the variable names and the file information
        exec('varnames.append(\'' + var_str + '_' + year + '_' + filenames[i] + '_raw\')')
        exec('varnames.append(\'' + var_str + '_' + year + '_' + filenames[i] + '_raw_lbl\')')

    exec('varnames_' + var_str + '_' + year + ' = varnames')

    # Generate code for saving
    [diet_save_code_str, diet_save_variables_str] = generate_save_code(diet_var_str, file_idx, filenames, year)

    # Save
    with open(save_folder + diet_var_str + '_' + year + '_raw.pkl', 'wb') as f:
        exec(diet_save_code_str)
    with open(save_folder + diet_var_str + '_' + year + '_raw_loadstr.pkl', 'wb') as f:
        pickle.dump(diet_save_variables_str, f)

    with open(save_folder + diet_var_str + '_' + year + '_raw_loadstr.pkl', 'rb') as f:
        diet_save_variables_str = pickle.load(f)

    with open(save_folder + diet_var_str + '_' + year + '_raw.pkl', 'rb') as f:
        exec(diet_save_variables_str + '= pickle.load(f)')

###############################################################################
# Exam
###############################################################################
if 1:
    print(exam_folder_str)
    mypath = load_folder + exam_folder_str

    # Get file names in the folder
    [filenames, file_idx] = get_filenames(mypath)

    # Load files unto the workspace. Each variable will have data from each file
    var_str = exam_var_str

    # Keep track of the variable names
    varnames = []

    for i in range(len(file_idx)):    
        print(i)
        # Load each file unto a designated variable
        with open(mypath+'/'+filenames[i]+ext, 'rb') as f:
            exec(var_str + '_' + year + '_' + filenames[i] + '_raw = xport.to_numpy(f)') # Worked
            # Count the total number of variables
            exec('totalvars_num += ' + var_str + '_' + year + '_' + filenames[i] + '_raw.shape[1]')
        # Store loaded file information
        with open(mypath+'/'+filenames[i]+ext, 'rb') as f:
            exec(var_str + '_' + year + '_' + filenames[i] + '_raw_lbl = xport.Reader(f).fields')

        # Keep track of the variable names and the file information
        exec('varnames.append(\'' + var_str + '_' + year + '_' + filenames[i] + '_raw\')')
        exec('varnames.append(\'' + var_str + '_' + year + '_' + filenames[i] + '_raw_lbl\')')

    exec('varnames_' + var_str + '_' + year + ' = varnames')

    # Generate code for saving
    [exam_save_code_str, exam_save_variables_str] = generate_save_code(exam_var_str, file_idx, filenames, year)

    # Save
    with open(save_folder + exam_var_str + '_' + year + '_raw.pkl', 'wb') as f:
        exec(exam_save_code_str)
    with open(save_folder + exam_var_str + '_' + year + '_raw_loadstr.pkl', 'wb') as f:
        pickle.dump(exam_save_variables_str, f)

    with open(save_folder + exam_var_str + '_' + year + '_raw_loadstr.pkl', 'rb') as f:
        exam_save_variables_str = pickle.load(f)

    with open(save_folder + exam_var_str + '_' + year + '_raw.pkl', 'rb') as f:
        exec(exam_save_variables_str + '= pickle.load(f)')

###############################################################################
# Lab
###############################################################################
if 1:

    print(lab_folder_str)
    mypath = load_folder + lab_folder_str

    # Get file names in the folder
    [filenames, file_idx] = get_filenames(mypath)

    # Load files unto the workspace. Each variable will have data from each file
    var_str = lab_var_str

    # Keep track of the variable names
    varnames = []

    for i in range(len(file_idx)):    
        print(i)
        # Load each file unto a designated variable
        with open(mypath+'/'+filenames[i]+ext, 'rb') as f:
            exec(var_str + '_' + year + '_' + filenames[i] + '_raw = xport.to_numpy(f)') # Worked
            # Count the total number of variables
            exec('totalvars_num += ' + var_str + '_' + year + '_' + filenames[i] + '_raw.shape[1]')
        # Store loaded file information
        with open(mypath+'/'+filenames[i]+ext, 'rb') as f:
            exec(var_str + '_' + year + '_' + filenames[i] + '_raw_lbl = xport.Reader(f).fields')

        # Keep track of the variable names and the file information
        exec('varnames.append(\'' + var_str + '_' + year + '_' + filenames[i] + '_raw\')')
        exec('varnames.append(\'' + var_str + '_' + year + '_' + filenames[i] + '_raw_lbl\')')

    exec('varnames_' + var_str + '_' + year + ' = varnames')

    # Special for lab
    lab_filenames_pre = filenames

    # Generate code for saving
    [lab_save_code_str, lab_save_variables_str] = generate_save_code(lab_var_str, file_idx, filenames, year)

    # Save
    with open(save_folder + lab_var_str + '_' + year + '_raw.pkl', 'wb') as f:
        exec(lab_save_code_str)
    with open(save_folder + lab_var_str + '_' + year + '_raw_loadstr.pkl', 'wb') as f:
        pickle.dump(lab_save_variables_str, f)

    with open(save_folder + lab_var_str + '_' + year + '_raw_loadstr.pkl', 'rb') as f:
        lab_save_variables_str = pickle.load(f)

    with open(save_folder + lab_var_str + '_' + year + '_raw.pkl', 'rb') as f:
        exec(lab_save_variables_str + '= pickle.load(f)')

print(totalvars_num)

