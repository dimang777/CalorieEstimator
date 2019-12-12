# Prepare the data to put everything into pandas dataframe
import pickle
import pandas as pd
import numpy as np

###############################################################################
# Set up folders and variables
###############################################################################
filename = 'df_basic_stats.py'
save_folder = '../../data/cleaned_df/'
load_folder = '../../data/cleaned_df/'

figure_folder = '../../images/eda/'

###############################################################################
# Load
###############################################################################

with open(load_folder + 'df.pkl', 'rb') as f:
    [df] = pickle.load(f)

with open(load_folder + 'df_bfr_demo_filter.pkl', 'rb') as f:
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

###############################################################################
# Generate a list of keys to use
###############################################################################

df_key = demo_filename_varname_pd_dict[demo_filenames[0]][1:]
for i_str in diet_filename_varname_pd_dict[diet_filenames[0]][1:]:
    df_key.append(i_str)
for j_str in exam_filenames:
    for i_str in exam_filename_varname_pd_dict[j_str][1:]:
        df_key.append(i_str)
for j_str in lab_filenames:
    for i_str in lab_filename_varname_pd_dict[j_str][1:]:
        df_key.append(i_str)

# Total data after phase 1 - 1310842
# non-NA/null observations
print(df.count().sum())


###############################################################################
# Sample Stats
###############################################################################


# 1 - Age
df['D0_RIDAGEYR'].describe()

df['D0_RIDAGEYR'].hist()
ax = df['D0_RIDAGEYR'].hist(bins=6)
fig = ax.get_figure()
fig.savefig(figure_folder+'figure.jpg')
# make a function with standard format so I can use it repeatedly

# 2 - Income
key = 'D0_INDHHIN2'
np.where(df[key] == 77)
df[key].describe()
df[key].hist(bins=100, range = (0,20))
df[key][df[key]<60].hist() # Skewed. A lot of 15 (higher than 100,000 lumped together)

np.sum((df[key]<60)) # 5033 - correct

# Correlation
df_corr = df.corr()
df_corr.shape
ax = sns.heatmap(abs(df_corr))
type(df_corr)

# SBP
key = 'E0_BPXSY2'
df[key].describe()
df[key].hist()
df_corr[key].hist()
df_corr[key].describe()
df_corr[key].sort_values()

# DBP
key = 'E0_BPXDI2'
df[key].describe()
df[key].hist()
df_corr[key].hist()
df_corr[key].describe()
df_corr[key].sort_values()

# Teeth_DETERSCORE
key = 'E2_CUS_DETERSCORE'
df[key].describe()
df[key].hist()
df_corr[key].hist()
df_corr[key].describe()
df_corr[key].sort_values()

# missing teeth
key = 'E2_CUS_MISTEETH'
df[key].describe()
df[key].hist()
df_corr[key].hist()
df_corr[key].describe()
df_corr[key].sort_values()



# df.values.flatten()

# df_corr_flat = df_corr.values.flatten()

mask_triu = np.triu(np.ones((460, 460), dtype=bool))
# np.fill_diagonal(mask_triu, False)

df_corr_abs = df_corr.abs()

df_corr_abs_triu = df_corr_abs.mask(mask_triu, np.nan)

df_corr_triu = df_corr.mask(mask_triu, np.nan)

df_corr_triu

with open(Save_folder + 'corr.pkl', 'wb') as f:
    pickle.dump([df_corr_triu], f)

with open(Save_folder + 'corr.pkl', 'rb') as f:
    [df_corr_triu] = pickle.load(f)


# 0.7 and up considered high correlation - 
# Source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3576830/
VarstoIgnore = lab_filename_varname_pd_dict[lab_filenames[51]][1:].copy()
for i_str in lab_filename_varname_pd_dict[lab_filenames[49]][1:].copy():
    VarstoIgnore.append(i_str)


WeightVars = ['WTSA2YR',
    'WTSAF2YR',
    'WTSH2YR',
    'WTSOG2YR',
    'WTFSM',
    'WTINT2YR',
    'WTMEC2YR',
    'WTDRD1']
    
    
VarstoIgnore 
for i_str in df_key:
    for j_str in WeightVars:
        if j_str in i_str:
            VarstoIgnore.append(i_str)
            break # only break out of the inner loop - confirmed below


CorrelatedPairs = []
for i_str in df_key:
    if i_str not in VarstoIgnore:
        result_local = df_corr_abs_triu.index[np.logical_and(df_corr_abs_triu[i_str].values>0.5, df_corr_abs_triu[i_str].values<0.9)].tolist()
        if result_local != []:
            for j_str in result_local:
                Sample_count_twovar = np.sum(np.logical_and(df[i_str].notna(), df[j_str].notna())*1)
                CorrelatedPairs.append([i_str, j_str, df_corr_abs_triu.loc[j_str, i_str], Sample_count_twovar])
        
len(CorrelatedPairs)


# for i_str in [2,3,4,5,6]:
#     print('i - ' + str(i_str))
#     for j_str in [3,5]:
#         print('j - ' + str(j_str))
#         if j_str == i_str:
#             print('broke')
#             break
   
# 'WTSA2YR' in 'L43_WTSA2YR'
Col_list = []
for i_str in diet_filename_varname_pd_dict[diet_filenames[0]][1:]:
    if i_str not in VarstoIgnore:
        Col_list.append(i_str)

colnum = 0        
with pd.ExcelWriter('output2.xlsx') as writer:
    for i_str in Col_list:
        df_corr_abs_triu_toExcel_temp = df_corr_abs_triu.loc[:, i_str].mask(df_corr_abs_triu.loc[:, i_str].values < 0.8, np.nan).sort_values(ascending=False, na_position='last')
        if np.sum(df_corr_abs_triu_toExcel_temp.isna().values*1) != len(df_corr_abs_triu_toExcel_temp.values):
            df_corr_abs_triu_toExcel_temp.to_excel(writer, startcol = colnum)
            colnum = colnum+3




# df_corr_abs_05up = temp.mask(temp < 0.7, np.nan)


lab_filename_varname_pd_dict['TCHOL_I']


r_bloodchol_cholintake = df_corr_abs_triu['I0_DR1TCHOL'].sort_values(ascending=False, na_position='last')['L34_LBXTC']
Sample_count_bchol_and_ichol = np.sum(np.logical_and(df['I0_DR1TCHOL'].notna(), df['L34_LBXTC'].notna())*1)
ax1 = df.plot.scatter(x='I0_DR1TCHOL', y='L34_LBXTC')



lab_filename_varname_pd_dict['HDL_I']
r_bloodhddchol_cholintake = df_corr_abs_triu['I0_DR1TCHOL'].sort_values(ascending=False, na_position='last')['L16_LBDHDD']
Sample_count_bhddchol_and_ichol = np.sum(np.logical_and(df['I0_DR1TCHOL'].notna(), df['L16_LBDHDD'].notna())*1)
ax1 = df.plot.scatter(x='I0_DR1TCHOL', y='L16_LBDHDD')


lab_filename_varname_pd_dict['TRIGLY_I']
r_bloodlddchol_cholintake = df_corr_abs_triu['I0_DR1TCHOL'].sort_values(ascending=False, na_position='last')['L37_LBDLDL']
Sample_count_blddchol_and_ichol = np.sum(np.logical_and(df['I0_DR1TCHOL'].notna(), df['L37_LBDLDL'].notna())*1)
ax1 = df.plot.scatter(x='I0_DR1TCHOL', y='L37_LBDLDL')



#I0_DR1TLZ, I0_DR1TVK - n = 3031
r = df_corr_triu['I0_DR1TLZ'].sort_values(ascending=False, na_position='last')['I0_DR1TVK']
Sample_count = np.sum(np.logical_and(df['I0_DR1TLZ'].notna(), df['I0_DR1TVK'].notna())*1)
ax1 = df.plot.scatter(x='I0_DR1TLZ', y='I0_DR1TVK')

#I0_DR1TFA, I0_DR1TFDFE - n = 3031
r = df_corr_triu['I0_DR1TFA'].sort_values(ascending=False, na_position='last')['I0_DR1TFDFE']
Sample_count = np.sum(np.logical_and(df['I0_DR1TFA'].notna(), df['I0_DR1TFDFE'].notna())*1)
ax1 = df.plot.scatter(x='I0_DR1TFA', y='I0_DR1TFDFE')

#I0_DR1TPROT, I0_DR1TSELE - n = 3031
r = df_corr_triu['I0_DR1TPROT'].sort_values(ascending=False, na_position='last')['I0_DR1TSELE']
Sample_count = np.sum(np.logical_and(df['I0_DR1TPROT'].notna(), df['I0_DR1TSELE'].notna())*1)
ax1 = df.plot.scatter(x='I0_DR1TPROT', y='I0_DR1TSELE')

#I0_DR1TPROT, I0_DR1TPHOS - n = 3031
r = df_corr_triu['I0_DR1TPROT'].sort_values(ascending=False, na_position='last')['I0_DR1TPHOS']
Sample_count = np.sum(np.logical_and(df['I0_DR1TPROT'].notna(), df['I0_DR1TPHOS'].notna())*1)
ax1 = df.plot.scatter(x='I0_DR1TPROT', y='I0_DR1TPHOS')

#I0_DR1TCHOL, I0_DR1TP204 - n = 3031
r = df_corr_triu['I0_DR1TCHOL'].sort_values(ascending=False, na_position='last')['I0_DR1TP204']
Sample_count = np.sum(np.logical_and(df['I0_DR1TCHOL'].notna(), df['I0_DR1TP204'].notna())*1)
ax1 = df.plot.scatter(x='I0_DR1TCHOL', y='I0_DR1TP204')

#I0_DR1TCHOL, I0_DR1TCHL - n = 3031
r = df_corr_triu['I0_DR1TCHOL'].sort_values(ascending=False, na_position='last')['I0_DR1TCHL']
Sample_count = np.sum(np.logical_and(df['I0_DR1TCHOL'].notna(), df['I0_DR1TCHL'].notna())*1)
ax1 = df.plot.scatter(x='I0_DR1TCHOL', y='I0_DR1TCHL')

#I0_DR1TTFAT, I0_DR1TKCAL - n = 3031
r = df_corr_triu['I0_DR1TKCAL'].sort_values(ascending=False, na_position='last')['I0_DR1TTFAT']
Sample_count = np.sum(np.logical_and(df['I0_DR1TKCAL'].notna(), df['I0_DR1TTFAT'].notna())*1)
ax1 = df.plot.scatter(x='I0_DR1TKCAL', y='I0_DR1TTFAT')

# Fat list of variables correlated
order = np.where(df_corr_triu['I0_DR1TTFAT'].sort_values(ascending=False, na_position='last').index.values == 'I0_DR1TP204')
print('Order of correlation: ' + str(order[0][0]))
r = df_corr_triu['I0_DR1TTFAT'].sort_values(ascending=False, na_position='last')['I0_DR1TP204']
print('r: ' + str(r))
# 22nd in the list with r = 0.54



df_corr.to_pickle('df_corr.pkl')

df.to_pickle('df.pkl')

df_corr_triu.to_pickle('df_corr_triu.pkl')






# I0_DR1TTFAT



# diet_filenames[0]
# diet_numVars = len(diet_filename_varname_pd_dict[diet_filenames[0]])

# 2^d = 3000
# dlog2(2) = log2(3000)
# d = log2(3000)
np.log2(3000)


# df_corr.abs()
# mask_triu = np.triu(np.ones((460, 460), dtype=bool))
# np.fill_diagonal(mask_triu, False)
# mask_flat = mask.flatten()

# df_corr.values[mask]

# mask_05 = (df_corr.values[mask]>0.5)
# df_corr_05up = df_corr[mask_05]
