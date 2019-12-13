import pickle
import pandas as pd
import numpy as np
import seaborn as sns

###############################################################################
# Set up folders and variables
###############################################################################
filename = 'exploratory_data_analysis.py'
save_folder = '../../data/cleaned_df/'
load_folder = '../../data/cleaned_df/'

figure_folder = '../../images/eda/'
sheet_folder = '../../sheets/eda/'

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
# Set up functions
###############################################################################

def decorate_fig_hist(ax, title):
    ax.set_xlabel(title)
    ax.set_ylabel('Count')
    ax.set_title(title)
    fig = ax.get_figure()
    fig.savefig(figure_folder+title+'_hist.jpg')
    fig.clf()

def decorate_fig_corr(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig = ax.get_figure()
    fig.savefig(figure_folder+title+'_corr.jpg')
    fig.clf()

def decorate_fig_others(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig = ax.get_figure()
    fig.savefig(figure_folder+title+'.jpg')
    fig.clf()

###############################################################################
# Sample Stats - histograms and correlations
###############################################################################

# 1 - Age
df['D0_RIDAGEYR'].describe()

df['D0_RIDAGEYR'].hist()
ax1 = df['D0_RIDAGEYR'].hist(bins=6)
decorate_fig_hist(ax1, 'Age')

# 2 - Income
key = 'D0_INDHHIN2'
np.where(df[key] == 77)
df[key].describe()
# df[key].hist(bins=100, range = (0,20))
ax2 = df[key][df[key]<60].hist() # Skewed. A lot of 15 (higher than 100,000 lumped together)
decorate_fig_hist(ax2, 'Income')

np.sum((df[key]<60)) # 5033 - correct

# Correlation
df_corr = df.corr()
df_corr.shape
ax3 = sns.heatmap(abs(df_corr))
type(df_corr)
decorate_fig_others(ax3, '', '', 'Correlation Heatmap')


# SBP
key = 'E0_BPXSY2'
df[key].describe()
ax4 = df[key].hist()
decorate_fig_hist(ax4, 'SBP')
ax5 = df_corr[key].hist()
decorate_fig_hist(ax5, 'SBP Correlations')
df_corr[key].describe()
df_corr[key].sort_values()


# DBP
key = 'E0_BPXDI2'
df[key].describe()
ax6 = df[key].hist()
decorate_fig_hist(ax6, 'DBP')
ax6 = df_corr[key].hist()
decorate_fig_hist(ax6, 'DBP Correlations')
df_corr[key].describe()
df_corr[key].sort_values()

# Teeth_DETERSCORE
key = 'E2_CUS_DETERSCORE'
df[key].describe()
ax7 = df[key].hist()
decorate_fig_hist(ax7, 'Teeth Deterioration Score')
ax21 = df_corr[key].hist()
decorate_fig_hist(ax21, 'Teeth Deter Score Correlations')
df_corr[key].describe()
df_corr[key].sort_values()

# missing teeth
key = 'E2_CUS_MISTEETH'
df[key].describe()
ax8 = df[key].hist()
decorate_fig_hist(ax8, 'Missing Teeth')
ax9 = df_corr[key].hist()
decorate_fig_hist(ax9, 'Missing Teeth Correlations')
df_corr[key].describe()
df_corr[key].sort_values()


# Upper triangle for absolute valued correlations matrix
mask_triu = np.triu(np.ones((460, 460), dtype=bool))
df_corr_abs = df_corr.abs()
df_corr_abs_triu = df_corr_abs.mask(mask_triu, np.nan)
df_corr_triu = df_corr.mask(mask_triu, np.nan)

with open(save_folder + 'corr.pkl', 'wb') as f:
    pickle.dump([df_corr_triu], f)

with open(save_folder + 'corr.pkl', 'rb') as f:
    [df_corr_triu] = pickle.load(f)


###############################################################################
# Sample Stats
###############################################################################

# 0.7 and up considered high correlation - 
# Source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3576830/
varstoignore = lab_filename_varname_pd_dict[lab_filenames[51]][1:].copy()
for i_str in lab_filename_varname_pd_dict[lab_filenames[49]][1:].copy():
    varstoignore.append(i_str)

# Weight variables that are not data and should be ignored
weightvars = ['WTSA2YR',
    'WTSAF2YR',
    'WTSH2YR',
    'WTSOG2YR',
    'WTFSM',
    'WTINT2YR',
    'WTMEC2YR',
    'WTDRD1']
    
    
varstoignore 
for i_str in df_key:
    for j_str in weightvars:
        if j_str in i_str:
            varstoignore.append(i_str)
            break # only break out of the inner loop - confirmed below

# Exception
varstoignore.append('D0_RIDEXPRG')

# Take correlations between 0.5 and 0.9 (0.9 is used since too high correlations
# can be suspected of being the same variable)
correlatedpairs = []
for i_str in df_key:
    if i_str not in varstoignore:
        result_local = df_corr_abs_triu.index[np.logical_and(df_corr_abs_triu[i_str].values>0.5, df_corr_abs_triu[i_str].values<0.9)].tolist()
        if result_local != []:
            for j_str in result_local:
                sample_count_twovar = np.sum(np.logical_and(df[i_str].notna(), df[j_str].notna())*1)
                correlatedpairs.append([i_str, j_str, df_corr_abs_triu.loc[j_str, i_str], sample_count_twovar])
        
print(len(correlatedpairs))


# Exception: 'WTSA2YR' in 'L43_WTSA2YR'
col_list = []
for i_str in diet_filename_varname_pd_dict[diet_filenames[0]][1:]:
    if i_str not in varstoignore:
        col_list.append(i_str)

# Write to an excel sheet - variables with higher correlations
colnum = 0
with pd.ExcelWriter(sheet_folder+'High Correlations.xlsx') as writer:
    for i_str in col_list:
        df_corr_abs_triu_toexcel_temp = df_corr_abs_triu.loc[:, i_str].mask(df_corr_abs_triu.loc[:, i_str].values < 0.8, np.nan).sort_values(ascending=False, na_position='last')
        if np.sum(df_corr_abs_triu_toexcel_temp.isna().values*1) != len(df_corr_abs_triu_toexcel_temp.values):
            df_corr_abs_triu_toexcel_temp.to_excel(writer, startcol = colnum)
            colnum = colnum+2

###############################################################################
# Selected correlation analysis
###############################################################################

lab_filename_varname_pd_dict['TCHOL_I']


r_bloodchol_cholintake = df_corr_abs_triu['I0_DR1TCHOL'].sort_values(ascending=False, na_position='last')['L34_LBXTC']
sample_count_bchol_and_ichol = np.sum(np.logical_and(df['I0_DR1TCHOL'].notna(), df['L34_LBXTC'].notna())*1)
ax11 = df.plot.scatter(x='I0_DR1TCHOL', y='L34_LBXTC')
decorate_fig_corr(ax11, 'Cholesterol in Diet', 'Cholesterol in Blood', 'Cholesterol Diet vs Lab')


lab_filename_varname_pd_dict['HDL_I']
r_bloodhddchol_cholintake = df_corr_abs_triu['I0_DR1TCHOL'].sort_values(ascending=False, na_position='last')['L16_LBDHDD']
Sample_count_bhddchol_and_ichol = np.sum(np.logical_and(df['I0_DR1TCHOL'].notna(), df['L16_LBDHDD'].notna())*1)
ax12 = df.plot.scatter(x='I0_DR1TCHOL', y='L16_LBDHDD')
decorate_fig_corr(ax12, 'Cholesterol in Diet', 'High Density Lipid Cholesterol in Blood', 'Cholesterol Diet vs HDL Cholesterol Lab')

lab_filename_varname_pd_dict['TRIGLY_I']
r_bloodlddchol_cholintake = df_corr_abs_triu['I0_DR1TCHOL'].sort_values(ascending=False, na_position='last')['L37_LBDLDL']
Sample_count_blddchol_and_ichol = np.sum(np.logical_and(df['I0_DR1TCHOL'].notna(), df['L37_LBDLDL'].notna())*1)
ax13 = df.plot.scatter(x='I0_DR1TCHOL', y='L37_LBDLDL')
decorate_fig_corr(ax13, 'Cholesterol in Diet', 'High Density Lipid Cholesterol in Blood', 'Cholesterol Diet vs HDL Cholesterol Lab')


# Look at variables in diet only
#I0_DR1TLZ, I0_DR1TVK - n = 3031
r = df_corr_triu['I0_DR1TLZ'].sort_values(ascending=False, na_position='last')['I0_DR1TVK']
Sample_count = np.sum(np.logical_and(df['I0_DR1TLZ'].notna(), df['I0_DR1TVK'].notna())*1)
ax14 = df.plot.scatter(x='I0_DR1TLZ', y='I0_DR1TVK')
decorate_fig_corr(ax14, 'Lutein and Zeaxanthin', 'Vitamin K', 'Lutein and Zeaxanthin vs Vitamin K')

#I0_DR1TFA, I0_DR1TFDFE - n = 3031
r = df_corr_triu['I0_DR1TFA'].sort_values(ascending=False, na_position='last')['I0_DR1TFDFE']
Sample_count = np.sum(np.logical_and(df['I0_DR1TFA'].notna(), df['I0_DR1TFDFE'].notna())*1)
ax15 = df.plot.scatter(x='I0_DR1TFA', y='I0_DR1TFDFE')
decorate_fig_corr(ax15, 'Folic acid', 'Folate', 'Folic acid vs Folate')

#I0_DR1TPROT, I0_DR1TSELE - n = 3031
r = df_corr_triu['I0_DR1TPROT'].sort_values(ascending=False, na_position='last')['I0_DR1TSELE']
Sample_count = np.sum(np.logical_and(df['I0_DR1TPROT'].notna(), df['I0_DR1TSELE'].notna())*1)
ax16 = df.plot.scatter(x='I0_DR1TPROT', y='I0_DR1TSELE')
decorate_fig_corr(ax16, 'Protein', 'Selenium', 'Protein vs Selenium')

#I0_DR1TPROT, I0_DR1TPHOS - n = 3031
r = df_corr_triu['I0_DR1TPROT'].sort_values(ascending=False, na_position='last')['I0_DR1TPHOS']
Sample_count = np.sum(np.logical_and(df['I0_DR1TPROT'].notna(), df['I0_DR1TPHOS'].notna())*1)
ax17 = df.plot.scatter(x='I0_DR1TPROT', y='I0_DR1TPHOS')
decorate_fig_corr(ax17, 'Protein', 'Phosphorus', 'Protein vs Phosphorus')

#I0_DR1TCHOL, I0_DR1TP204 - n = 3031
r = df_corr_triu['I0_DR1TCHOL'].sort_values(ascending=False, na_position='last')['I0_DR1TP204']
Sample_count = np.sum(np.logical_and(df['I0_DR1TCHOL'].notna(), df['I0_DR1TP204'].notna())*1)
ax18 = df.plot.scatter(x='I0_DR1TCHOL', y='I0_DR1TP204')
decorate_fig_corr(ax18, 'Cholesterol', 'Eicosatetraenoic PFA20-4', 'Cholesterol vs Eicosatetraenoic PFA20-4')

#I0_DR1TCHOL, I0_DR1TCHL - n = 3031
r = df_corr_triu['I0_DR1TCHOL'].sort_values(ascending=False, na_position='last')['I0_DR1TCHL']
Sample_count = np.sum(np.logical_and(df['I0_DR1TCHOL'].notna(), df['I0_DR1TCHL'].notna())*1)
ax19 = df.plot.scatter(x='I0_DR1TCHOL', y='I0_DR1TCHL')
decorate_fig_corr(ax19, 'Cholesterol', 'Total choline', 'Cholesterol vs Total choline')

#I0_DR1TTFAT, I0_DR1TKCAL - n = 3031
r = df_corr_triu['I0_DR1TKCAL'].sort_values(ascending=False, na_position='last')['I0_DR1TTFAT']
Sample_count = np.sum(np.logical_and(df['I0_DR1TKCAL'].notna(), df['I0_DR1TTFAT'].notna())*1)
ax20 = df.plot.scatter(x='I0_DR1TKCAL', y='I0_DR1TTFAT')
decorate_fig_corr(ax20, 'Total fat', 'Total calorie', 'Total fat vs Total calorie')

# Fat list of variables correlated
order = np.where(df_corr_triu['I0_DR1TTFAT'].sort_values(ascending=False, na_position='last').index.values == 'I0_DR1TP204')
print('Order of correlation: ' + str(order[0][0]))
r = df_corr_triu['I0_DR1TTFAT'].sort_values(ascending=False, na_position='last')['I0_DR1TP204']
print('r: ' + str(r))
# 22nd in the list with r = 0.54

###############################################################################
# Save
###############################################################################

df_corr.to_pickle(save_folder+'df_corr.pkl')

df_corr_triu.to_pickle(save_folder+'df_corr_triu.pkl')

