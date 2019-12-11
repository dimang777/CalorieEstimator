import os
from os import walk
import pickle

filename = '1_data_load.py'

save_folder = '../../data/raw_formatted/'

demo_folder_str = 'Demographics Data'
Diet_folder_str = 'Dietary Data'
Exam_folder_str = 'Examination Data'
Lab_folder_str = 'Laboratory Data'

Demo_var_str = 'Demo'
Diet_var_str = 'Diet'
Exam_var_str = 'Exam'
Lab_var_str = 'Lab'

ext = '.xpt'

# Worked
with open(save_folder+'test.pickle', 'wb') as handle:
    pickle.dump([filename], handle)

with open(save_folder+'test.pickle', 'rb') as handle:
    [filename] = pickle.load(handle)


# Count the amount of data. The total number of data.

TotalVars_num = 0

year = 2015
Folder = '../../data/raw/NHANES_2015_2016'

if 1:
    for Condense in [0]: # Demographic
        print(demo_folder_str)
        mypath = Folder + demo_folder_str
        os.chdir(mypath)
        
        # Get file names in the folder
        f = []
        for (dirpath, dirnames, filenames_temp) in walk(mypath):
            f.extend(filenames_temp)
        
        file_idx = [] 
        for i in range(len(filenames_temp)):
            if filenames_temp[i][-3:].upper() == 'XPT':
                file_idx.append(i)
        
        filenames= []
        exec('varnames_' + Demo_var_str + '_' + year + ' = []')
        for i in range(len(file_idx)):
            filenames.append(filenames_temp[file_idx[i]][:-4])
        
        
        for i in range(len(file_idx)):    
            print(i)
            with open(filenames[i]+ext, 'rb') as f:
                exec(Demo_var_str + '_' + year + '_' + filenames[i] + '_raw = xport.to_numpy(f)') # Worked
                exec('TotalVars_num += ' + Demo_var_str + '_' + year + '_' + filenames[i] + '_raw.shape[1]')
            with open(filenames[i]+ext, 'rb') as f:
                exec(Demo_var_str + '_' + year + '_' + filenames[i] + '_raw_lbl = xport.Reader(f).fields')
            #varnames_Demo_2015
            exec('varnames_' + Demo_var_str + '_' + year + '.append(\'' + Demo_var_str + '_' + year + '_' + filenames[i] + '_raw\')')
            exec('varnames_' + Demo_var_str + '_' + year + '.append(\'' + Demo_var_str + '_' + year + '_' + filenames[i] + '_raw_lbl\')')

        # Save variables
        Demo_save_code_str = 'pickle.dump('

        Demo_save_variables_str = '['
        for i in range(len(file_idx)):
            Demo_save_variables_str = Demo_save_variables_str + \
                Demo_var_str + '_' + year + '_' + filenames[i] + '_raw, ' + \
                Demo_var_str + '_' + year + '_' + filenames[i] + '_raw_lbl, '
        Demo_save_variables_str = Demo_save_variables_str + ']'

        Demo_save_code_str = Demo_save_code_str + Demo_save_variables_str + ', f)'

        with open(save_folder + Demo_var_str + '_' + year + '_raw.pkl', 'wb') as f:
            exec(Demo_save_code_str)
        with open(save_folder + Demo_var_str + '_' + year + '_raw_loadstr.pkl', 'wb') as f:
            pickle.dump(Demo_save_variables_str, f)


        with open(save_folder + Demo_var_str + '_' + year + '_raw_loadstr.pkl', 'rb') as f:
            Demo_save_variables_str = pickle.load(f)

        with open(save_folder + Demo_var_str + '_' + year + '_raw.pkl', 'rb') as f:
            exec(Demo_save_variables_str + '= pickle.load(f)')



if 1:
    for Condense in [0]: # Diet - orignal practice code- DO NOT ERASE COMMENTS IN THIS CELL
        print(Diet_folder_str)
        mypath = Folder + Diet_folder_str
        os.chdir(mypath)
        
        f = []
        for (dirpath, dirnames, filenames_temp) in walk(mypath):
            f.extend(filenames_temp)
        #    break
        
        # Both case 1 and 2 worked. used 2 since simpler
        #file_idx = np.array([], 'int64') # Case 1: worked
        file_idx = [] # Case 2: worked
        for i in range(len(filenames_temp)):
            if filenames_temp[i][-3:].upper() == 'XPT':
        #        file_idx = np.append(file_idx, int(i)) # Case 1: worked
                file_idx.append(i) # Case 2: worked
        
        #filenames[file_idx] # This didn't work
        filenames_temp[file_idx[0]] # This worked
        
        
        filenames= []
        exec('varnames_' + Diet_var_str + '_' + year + ' = []')
        for i in range(len(file_idx)):
            filenames.append(filenames_temp[file_idx[i]][:-4])
        
        for i in range(len(file_idx)):    
            print(i)
            with open(filenames[i]+ext, 'rb') as f:
        #        print('Diet_' + year + '_' + filenames[i] + '_raw = xport.to_numpy(f)')
        #        Diet_2015_DR1IFF_I_raw = xport.to_numpy(f)
        #        eval('Diet_' + year + '_' + filenames[i] + '_raw = xport.to_numpy(f)') # eval doesn't work when passing assignment. Use exec
                exec(Diet_var_str + '_' + year + '_' + filenames[i] + '_raw = xport.to_numpy(f)') # Worked
                exec('TotalVars_num += ' + Diet_var_str + '_' + year + '_' + filenames[i] + '_raw.shape[1]')
            with open(filenames[i]+ext, 'rb') as f:
                exec(Diet_var_str + '_' + year + '_' + filenames[i] + '_raw_lbl = xport.Reader(f).fields')
        #        Diet_2015_DR1IFF_I_raw = xport.to_numpy(f)    
           
            
            # varnames_Diet_2015.append('Diet_2015_DRXFCD_I_raw')
            exec('varnames_' + Diet_var_str + '_' + year + '.append(\'' + Diet_var_str + '_' + year + '_' + filenames[i] + '_raw\')')
            exec('varnames_' + Diet_var_str + '_' + year + '.append(\'' + Diet_var_str + '_' + year + '_' + filenames[i] + '_raw_lbl\')')
        # Diet_2015_DR1IFF_I_raw[0:5,0:5]



        # Save variables
        Diet_save_code_str = 'pickle.dump('

        Diet_save_variables_str = '['
        for i in range(len(file_idx)):
            Diet_save_variables_str = Diet_save_variables_str + \
                Diet_var_str + '_' + year + '_' + filenames[i] + '_raw, ' + \
                Diet_var_str + '_' + year + '_' + filenames[i] + '_raw_lbl, '
        Diet_save_variables_str = Diet_save_variables_str + ']'

        Diet_save_code_str = Diet_save_code_str + Diet_save_variables_str + ', f)'

        with open(save_folder + Diet_var_str + '_' + year + '_raw.pkl', 'wb') as f:
            exec(Diet_save_code_str)
        with open(save_folder + Diet_var_str + '_' + year + '_raw_loadstr.pkl', 'wb') as f:
            pickle.dump(Diet_save_variables_str, f)

        with open(save_folder + Diet_var_str + '_' + year + '_raw_loadstr.pkl', 'rb') as f:
            Diet_save_variables_str = pickle.load(f)

        with open(save_folder + Diet_var_str + '_' + year + '_raw.pkl', 'rb') as f:
            exec(Diet_save_variables_str + '= pickle.load(f)')

if 1:
    for Condense in [0]: # Exam
        print(Exam_folder_str)
        mypath = Folder + Exam_folder_str
        os.chdir(mypath)
        
        f = []
        for (dirpath, dirnames, filenames_temp) in walk(mypath):
            f.extend(filenames_temp)
        
        file_idx = []
        for i in range(len(filenames_temp)):
            if filenames_temp[i][-3:].upper() == 'XPT':
                file_idx.append(i)
        
        filenames= []
        exec('varnames_' + Exam_var_str + '_' + year + ' = []')
        for i in range(len(file_idx)):
            filenames.append(filenames_temp[file_idx[i]][:-4])
        
        for i in range(len(file_idx)):    
            print(i)
            with open(filenames[i]+ext, 'rb') as f:
                exec(Exam_var_str + '_' + year + '_' + filenames[i] + '_raw = xport.to_numpy(f)')
                exec('TotalVars_num += ' + Exam_var_str + '_' + year + '_' + filenames[i] + '_raw.shape[1]')
            with open(filenames[i]+ext, 'rb') as f:
                exec(Exam_var_str + '_' + year + '_' + filenames[i] + '_raw_lbl = xport.Reader(f).fields')
            #varnames_Exam_2015
            exec('varnames_' + Exam_var_str + '_' + year + '.append(\'' + Exam_var_str + '_' + year + '_' + filenames[i] + '_raw\')')
            exec('varnames_' + Exam_var_str + '_' + year + '.append(\'' + Exam_var_str + '_' + year + '_' + filenames[i] + '_raw_lbl\')')

        # Save variables
        Exam_save_code_str = 'pickle.dump('

        Exam_save_variables_str = '['
        for i in range(len(file_idx)):
            Exam_save_variables_str = Exam_save_variables_str + \
                Exam_var_str + '_' + year + '_' + filenames[i] + '_raw, ' + \
                Exam_var_str + '_' + year + '_' + filenames[i] + '_raw_lbl, '
        Exam_save_variables_str = Exam_save_variables_str + ']'

        Exam_save_code_str = Exam_save_code_str + Exam_save_variables_str + ', f)'

        with open(save_folder + Exam_var_str + '_' + year + '_raw.pkl', 'wb') as f:
            exec(Exam_save_code_str)
        with open(save_folder + Exam_var_str + '_' + year + '_raw_loadstr.pkl', 'wb') as f:
            pickle.dump(Exam_save_variables_str, f)

        with open(save_folder + Exam_var_str + '_' + year + '_raw_loadstr.pkl', 'rb') as f:
            Exam_save_variables_str = pickle.load(f)

        with open(save_folder + Exam_var_str + '_' + year + '_raw.pkl', 'rb') as f:
            exec(Exam_save_variables_str + '= pickle.load(f)')

if 1:
    for Condense in [0]: # Lab - NOTE: HIV_I is missing from the beginning. That's okay. It's not used. But is different from other files not used.
        print(Lab_folder_str)
        mypath = Folder + Lab_folder_str
        os.chdir(mypath)
        
        f = []
        for (dirpath, dirnames, filenames_temp) in walk(mypath):
            f.extend(filenames_temp)
        
        file_idx = []
        for i in range(len(filenames_temp)):
            if filenames_temp[i][-3:].upper() == 'XPT':
                file_idx.append(i)
        
        # Sort out xpt files among others
        filenames= []
        exec('varnames_' + Lab_var_str + '_' + year + ' = []')
        exec('varnames_' + Lab_var_str + '_' + year + '_raw_dict = {}')
        exec('varnames_' + Lab_var_str + '_' + year + '_raw_lbl_dict = {}')
        for i in range(len(file_idx)):
            filenames.append(filenames_temp[file_idx[i]][:-4])
        Lab_filenames_Pre = filenames
        
        Lab_filename_varname_dict = {}
        for i in range(len(file_idx)):    
            print(i)
            with open(filenames[i]+ext, 'rb') as f:
                exec(Lab_var_str + '_' + year + '_' + filenames[i] + '_raw = xport.to_numpy(f)')
                exec('TotalVars_num += ' + Lab_var_str + '_' + year + '_' + filenames[i] + '_raw.shape[1]')
            with open(filenames[i]+ext, 'rb') as f:
                exec(Lab_var_str + '_' + year + '_' + filenames[i] + '_raw_lbl = xport.Reader(f).fields')
            #varnames_Exam_2015
            exec('varnames_' + Lab_var_str + '_' + year + '.append(\'' + Lab_var_str + '_' + year + '_' + filenames[i] + '_raw\')')
            exec('varnames_' + Lab_var_str + '_' + year + '.append(\'' + Lab_var_str + '_' + year + '_' + filenames[i] + '_raw_lbl\')')
            exec('varnames_' + Lab_var_str + '_' + year + '_raw_dict[filenames[i]] = \'' + Lab_var_str + '_' + year + '_' + filenames[i] + '_raw\'')
            exec('varnames_' + Lab_var_str + '_' + year + '_raw_lbl_dict[filenames[i]] = \'' + Lab_var_str + '_' + year + '_' + filenames[i] + '_raw_lbl\'')
            exec('Lab_filename_varname_dict[filenames[i]] = ' + Lab_var_str + '_' + year + '_' + filenames[i] + '_raw_lbl')
            # varnames_Lab_2015_raw_dict
            # varnames_Lab_2015_raw_lbl_dict
            # np.shape(Lab_filebranch_varnames[0])

        # Save variables
        Lab_save_code_str = 'pickle.dump('

        Lab_save_variables_str = '[Lab_filenames_Pre, '
        for i in range(len(file_idx)):
            Lab_save_variables_str = Lab_save_variables_str + \
                Lab_var_str + '_' + year + '_' + filenames[i] + '_raw, ' + \
                Lab_var_str + '_' + year + '_' + filenames[i] + '_raw_lbl, '
        Lab_save_variables_str = Lab_save_variables_str + ']'

        Lab_save_code_str = Lab_save_code_str + Lab_save_variables_str + ', f)'

        with open(save_folder + Lab_var_str + '_' + year + '_raw.pkl', 'wb') as f:
            exec(Lab_save_code_str)
        with open(save_folder + Lab_var_str + '_' + year + '_raw_loadstr.pkl', 'wb') as f:
            pickle.dump(Lab_save_variables_str, f)

        with open(save_folder + Lab_var_str + '_' + year + '_raw_loadstr.pkl', 'rb') as f:
            Lab_save_variables_str = pickle.load(f)

        with open(save_folder + Lab_var_str + '_' + year + '_raw.pkl', 'rb') as f:
            exec(Lab_save_variables_str + '= pickle.load(f)')

print(TotalVars_num)


# Deleted files: All extensions .xpt
# 2015
# Lab: HIV_I
# Ques: ACQ_I
# Ques: BPQ_I
# 2013
# Lab: ALDS_H
# Lab: APOB_H
# Lab: FOLFMS_H
# Lab: GHB_H
# Lab: SSFLRT_H
# 2011
# Diet: DSPI
# Exam: AUX_G
# Lab: HEPBD_G
# Ques: DEQ_G
# Ques: RXQASA_G
# 2009
# Lab: DEET_F
# Lab: DOXPOL_F
# Lab: UAS_F
# 2007
# Lab: CBC_E
# Lab: CHLMDA_E
# Lab: HEPB_S_E
# Lab: HSV_E
# Lab: PBCD_E
# Lab: UAM_E
# Ques: KIQ_P_E
# 2005
# Lab: EPP_D
# Lab: PHYTO_D
# Lab: TCHOL_D




# Steps
# Load data. Remove empty files
#%% Figure


# plt.figure(1)
# plt.plot(a[:,0], a[:,1], 'x')
# plt.ylim(8.99, 9.01)
# plt.show()

# plt.figure(2)
# plt.plot(a[:,0], a[:,2], 'x')
# #plt.ylim(8.99, 9.01)
# plt.show()

