import numpy as np

def removeval(Cols_toRemove, Raw, Raw_lbl):
    # Original primitive code - kept it in case something goes wrong
    Cols_toRemove_idx = []
    for i in range(0,len(Cols_toRemove)):
        Cols_toRemove_idx.append(int(Raw_lbl.index(Cols_toRemove[i])))

    Cols_toKeep_idx = []
    for i in range(0, Raw.shape[1]):
        if i not in Cols_toRemove_idx:
            Cols_toKeep_idx.append(int(i))

    Clean = np.copy(Raw[:,Cols_toKeep_idx])

    Clean_lbl = []
    for i in Cols_toKeep_idx:
        Clean_lbl.append(Raw_lbl[i])

    return [Clean, Clean_lbl];   

def str2num_1darray(Array_str):
# Convert "string" of numbers to numbers
    Array_num = np.zeros(len(Array_str))
    for i, i_str in enumerate(Array_str):
        if i_str.lower() == 'nan':
            Array_num[i] = float('nan')
        else:
            Array_num[i] = round(float(i_str))

    return Array_num

def replace_w_nan(Var, Col, Value):
    Replace = np.copy(Var)
    Replace[Var[:,Col] == Value, Col] = float('nan')
    return Replace