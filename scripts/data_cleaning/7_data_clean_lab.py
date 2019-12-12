import numpy as np
import pickle
from utils import removeval, replace_w_nan

###############################################################################
# Set up folders and variables
###############################################################################
filename = '7_data_clean_lab.py'
load_folder = '../../data/raw_formatted/'
save_folder = '../../data/clean/'

demo_var_str = 'demo'
diet_var_str = 'diet'
exam_var_str = 'exam'
lab_var_str = 'lab'

year = '2015'

with open(load_folder + lab_var_str + '_' + year + '_raw_loadstr.pkl', 'rb') as f:
    demo_save_variables_str = pickle.load(f)

with open(load_folder + lab_var_str + '_' + year + '_raw.pkl', 'rb') as f:
    exec(demo_save_variables_str + '= pickle.load(f)')


files_tocol2nan = ['APOB_I',
    'UADM_I',
    'UTASS_I',
    'UTAS_I',
    'PBCD_I',
    'IHGEM_I',
    'TRIGLY_I',
    'CUSEZN_I',
    'INS_I',
    'UIO_I',
    'UHG_I',
    'UM_I',
    'UMS_I',
    'OGTT_I',
    'PFAS_I',
    'EPHPP_I',
    'PHTHTE_I',
    'GLU_I',
    'UASS_I',
    'UAS_I',
    'UVOC_I',
    'UVOCS_I',
    'VOCWBS_I']


# replace_w_nan(Var, Col, Value)
for i_str in files_tocol2nan:
    exec('lab_2015_' + i_str + '_clean1 = replace_w_nan(lab_2015_' + i_str + '_raw, 1, 0)')
    exec('lab_2015_' + i_str + '_clean1_lbl = lab_2015_' + i_str + '_raw_lbl')


for i_str in files_tocol2nan:
    exec('print(type(lab_2015_' + i_str + '_clean1_lbl))')


files_3tonan = ['HEPA_I',
    'HSV_I',
    'ORHPV_I',
    'HPVSWC_I',
    'HPVSWR_I',
    'UCPREG_I']

for i_str in files_3tonan:
    exec('print(type(lab_2015_' + i_str + '_raw_lbl))')


i = 0
varname = 'LBXHA'
exec('colnum = lab_2015_' + files_3tonan[i] + '_raw_lbl.index(\''+varname+'\')')
# replace_w_nan(Var, Col, Value)
exec('lab_2015_' + files_3tonan[i] + '_clean1 = replace_w_nan(lab_2015_' + files_3tonan[i] + '_raw, ' + str(colnum) + ', 3)')
exec('lab_2015_' + files_3tonan[i] + '_clean1_lbl = (lab_2015_' + files_3tonan[i] + '_raw_lbl)')

i = 1
varname = 'LBXHE1'
exec('colnum = lab_2015_' + files_3tonan[i] + '_raw_lbl.index(\''+varname+'\')')
# replace_w_nan(Var, Col, Value)
exec('lab_2015_' + files_3tonan[i] + '_clean1 = replace_w_nan(lab_2015_' + files_3tonan[i] + '_raw, ' + str(colnum) + ', 3)')
exec('lab_2015_' + files_3tonan[i] + '_clean1_lbl = (lab_2015_' + files_3tonan[i] + '_raw_lbl)')

varname = 'LBXHE2'
exec('colnum = lab_2015_' + files_3tonan[i] + '_raw_lbl.index(\''+varname+'\')')
# replace_w_nan(Var, Col, Value)
exec('lab_2015_' + files_3tonan[i] + '_clean1 = replace_w_nan(lab_2015_' + files_3tonan[i] + '_clean1, ' + str(colnum) + ', 3)')
exec('lab_2015_' + files_3tonan[i] + '_clean1_lbl = (lab_2015_' + files_3tonan[i] + '_raw_lbl)')

i = 2
varname = 'ORXHPV'
exec('colnum = lab_2015_' + files_3tonan[i] + '_raw_lbl.index(\''+varname+'\')')
# replace_w_nan(Var, Col, Value)
exec('lab_2015_' + files_3tonan[i] + '_clean1 = replace_w_nan(lab_2015_' + files_3tonan[i] + '_raw, ' + str(colnum) + ', 3)')
exec('lab_2015_' + files_3tonan[i] + '_clean1_lbl = (lab_2015_' + files_3tonan[i] + '_raw_lbl)')

i = 3
varname = 'LBXHP2C'
exec('colnum = lab_2015_' + files_3tonan[i] + '_raw_lbl.index(\''+varname+'\')')
# replace_w_nan(Var, Col, Value)
exec('lab_2015_' + files_3tonan[i] + '_clean1 = replace_w_nan(lab_2015_' + files_3tonan[i] + '_raw, ' + str(colnum) + ', 3)')
exec('lab_2015_' + files_3tonan[i] + '_clean1_lbl = (lab_2015_' + files_3tonan[i] + '_raw_lbl)')

i = 4
varname = 'LBDRPCR'
exec('colnum = lab_2015_' + files_3tonan[i] + '_raw_lbl.index(\''+varname+'\')')
# replace_w_nan(Var, Col, Value)
exec('lab_2015_' + files_3tonan[i] + '_clean1 = replace_w_nan(lab_2015_' + files_3tonan[i] + '_raw, ' + str(colnum) + ', 3)')
exec('lab_2015_' + files_3tonan[i] + '_clean1_lbl = (lab_2015_' + files_3tonan[i] + '_raw_lbl)')

i = 5
varname = 'URXPREG'
exec('colnum = lab_2015_' + files_3tonan[i] + '_raw_lbl.index(\''+varname+'\')')
# replace_w_nan(Var, Col, Value)
exec('lab_2015_' + files_3tonan[i] + '_clean1 = replace_w_nan(lab_2015_' + files_3tonan[i] + '_raw, ' + str(colnum) + ', 3)')
exec('lab_2015_' + files_3tonan[i] + '_clean1_lbl = (lab_2015_' + files_3tonan[i] + '_raw_lbl)')



files_nonetonan = ['ALB_CR_I',
'CHLMDA_I',
'HDL_I',
'TCHOL_I',
'CRCO_I',
'CBC_I',
'COT_I',
'FERTIN_I',
'FLDEP_I',
'FLDEW_I',
'FOLATE_I',
'FOLFMS_I',
'GHB_I',
'HEPBD_I',
'HEPB_S_I',
'HEPE_I',
'HSCRP_I',
'SSNEON_I',
'SSMHHT_I',
'TST_I',
'BIOPRO_I',
'TFR_I',
'TRICH_I',
'VOCWB_I']

for i_str in files_nonetonan:
    exec('print(type(lab_2015_' + i_str + '_raw_lbl))')

# Transfer files
for i_str in files_nonetonan:
    exec('lab_2015_' + i_str + '_clean1 = np.copy(lab_2015_' + i_str + '_raw)')
    exec('lab_2015_' + i_str + '_clean1_lbl = (lab_2015_' + i_str + '_raw_lbl)')



files_not_used = ['FASTQX_I',
        'HEPC_I',
        'HIV_I',
        'UCFLOW_I']

exec('varnames_' + lab_var_str + '_' + year + '_clean1_dict = {}')
exec('varnames_' + lab_var_str + '_' + year + '_clean1_lbl_dict = {}')
    
for i_str in lab_filenames_pre:
    if i_str not in files_not_used:
        exec('varnames_' + lab_var_str + '_' + year + '_clean1_dict[i_str] = \'' + lab_var_str + '_' + year + '_' + i_str + '_clean1\'')
        exec('varnames_' + lab_var_str + '_' + year + '_clean1_lbl_dict[i_str] = \'' + lab_var_str + '_' + year + '_' + i_str + '_clean1_lbl\'')



# Util codes
# lab_2015_APOB_I_raw.shape
# lab_2015_APOB_I_clean1.shape
# np.sum((lab_2015_HSV_I_raw[:,colnum] == 3)*1)
# np.sum((lab_2015_HSV_I_clean1[:,colnum] == 3)*1)
# np.sum(np.isnan(lab_2015_HSV_I_raw[:,colnum])*1)
# np.sum(np.isnan(lab_2015_HSV_I_clean1[:,colnum])*1)
# a = np.argwhere(np.isnan(lab_2015_APOB_I_clean1[:,1]))
# a = np.argwhere(np.isnan(lab_2015_APOB_I_raw[:,1]))


# Long variable declaration
for condense in [1]:

    files_not_used = ['FASTQX_I',
        'HEPC_I',
        'HIV_I',
        'UCFLOW_I']
    
    
    
    lab_filename_remvarname_dict = {}
    i = 0
    print(lab_filenames_pre[i]) #ALB_CR_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['URDUMALC',
        'URDUCRLC',
        'URXCRS',
        'URXUMS']
    
    i = 1
    print(lab_filenames_pre[i]) #APOB_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['LBDAPBSI']
    
    i = 2
    print(lab_filenames_pre[i]) #BIOPRO_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['LBDSALSI',
        'LBDSBUSI',
        'LBDSCASI',
        'LBDSCHSI',
        'LBDSCRSI',
        'LBDSGBSI',
        'LBDSGLSI',
        'LBDSIRSI',
        'LBDSPHSI',
        'LBDSTBSI',
        'LBDSTPSI',
        'LBDSTRSI',
        'LBDSUASI']
    
    i = 3
    print(lab_filenames_pre[i]) #CBC_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = []
    
    i = 4
    print(lab_filenames_pre[i]) #CHLMDA_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = []
    
    i = 5
    print(lab_filenames_pre[i]) #COT_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['LBDCOTLC',
        'LBDHCTLC']
    
    i = 6
    print(lab_filenames_pre[i]) #CRCO_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['LBDBCRSI',
        'LBDBCRLC',
        'LBDBCOSI',
        'LBDBCOLC']
    
    i = 7
    print(lab_filenames_pre[i]) #CUSEZN_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['LBDSCUSI',
        'LBDSSESI',
        'LBDSZNSI']
    
    i = 8
    print(lab_filenames_pre[i]) #EPHPP_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['URDBP3LC',
        'URDBPHLC',
        'URDBPFLC',
        'URDBPSLC',
        'URDTLCLC',
        'URDTRSLC',
        'URDBUPLC',
        'URDEPBLC',
        'URDMPBLC',
        'URDPPBLC',
        'URD14DLC',
        'URDDCBLC']
    
    i = 9
    print(lab_filenames_pre[i]) #FASTQX_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = []
    
    i = 10
    print(lab_filenames_pre[i]) #FERTIN_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['LBDFERSI']
    
    i = 11
    print(lab_filenames_pre[i]) #FLDEP_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = []
    
    i = 12
    print(lab_filenames_pre[i]) #FLDEW_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = []
    
    i = 13
    print(lab_filenames_pre[i]) #FOLATE_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['LBDRFOSI']
    
    i = 14
    print(lab_filenames_pre[i]) #FOLFMS_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['LBDFOT',
        'LBDSF1LC',
        'LBDSF2LC',
        'LBDSF3LC',
        'LBDSF4LC',
        'LBDSF5LC',
        'LBDSF6LC']
    
    i = 15
    print(lab_filenames_pre[i]) #GHB_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = []
    
    i = 16
    print(lab_filenames_pre[i]) #GLU_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['LBDGLUSI']
    
    i = 17
    print(lab_filenames_pre[i]) #HDL_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['LBDHDDSI']
    
    i = 18
    print(lab_filenames_pre[i]) #HEPA_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = []
    
    i = 19
    print(lab_filenames_pre[i]) #HEPBD_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['LBDHBG',
        'LBDHD']
    
    i = 20
    print(lab_filenames_pre[i]) #HEPB_S_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = []
    
    i = 21
    print(lab_filenames_pre[i]) #HEPC_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = []
    
    i = 22
    print(lab_filenames_pre[i]) #HEPE_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['LBDHEM']
    
    i = 23
    print(lab_filenames_pre[i]) #HPVSWC_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = []
    
    i = 24
    print(lab_filenames_pre[i]) #HPVSWR_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['LBDRHP',
         'LBDRLP',
         'LBDR06',
         'LBDR11',
         'LBDR16',
         'LBDR18',
         'LBDR26',
         'LBDR31',
         'LBDR33',
         'LBDR35',
         'LBDR39',
         'LBDR40',
         'LBDR42',
         'LBDR45',
         'LBDR51',
         'LBDR52',
         'LBDR53',
         'LBDR54',
         'LBDR55',
         'LBDR56',
         'LBDR58',
         'LBDR59',
         'LBDR61',
         'LBDR62',
         'LBDR64',
         'LBDR66',
         'LBDR67',
         'LBDR68',
         'LBDR69',
         'LBDR70',
         'LBDR71',
         'LBDR72',
         'LBDR73',
         'LBDR81',
         'LBDR82',
         'LBDR83',
         'LBDR84',
         'LBDR89',
         'LBDRPI']
    
    i = 25
    print(lab_filenames_pre[i]) #HSCRP_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['LBDHRPLC']
    
    i = 26
    print(lab_filenames_pre[i]) #HSV_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = []
    
    i = 27
    print(lab_filenames_pre[i]) #IHGEM_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['LBDIHGSI',
        'LBDIHGLC',
        'LBDBGELC',
        'LBDBGMLC']
    
    i = 28
    print(lab_filenames_pre[i]) #INS_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['LBDINSI',
        'LBDINLC',
        'PHAFSTMN']
    
    i = 29
    print(lab_filenames_pre[i]) #OGTT_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['LBDGLTSI',
        'GTDSCMMN',
        'GTDDR1MN',
        'GTDBL2MN',
        'GTDDR2MN',
        'GTXDRANK',
        'GTDCODE']
    
    i = 30
    print(lab_filenames_pre[i]) #ORHPV_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['ORXGH',
        'ORXGL',
        'ORXH06',
        'ORXH11',
        'ORXH16',
        'ORXH18',
        'ORXH26',
        'ORXH31',
        'ORXH33',
        'ORXH35',
        'ORXH39',
        'ORXH40',
        'ORXH42',
        'ORXH45',
        'ORXH51',
        'ORXH52',
        'ORXH53',
        'ORXH54',
        'ORXH55',
        'ORXH56',
        'ORXH58',
        'ORXH59',
        'ORXH61',
        'ORXH62',
        'ORXH64',
        'ORXH66',
        'ORXH67',
        'ORXH68',
        'ORXH69',
        'ORXH70',
        'ORXH71',
        'ORXH72',
        'ORXH73',
        'ORXH81',
        'ORXH82',
        'ORXH83',
        'ORXH84',
        'ORXHPC',
        'ORXHPI',]
    
    i = 31
    print(lab_filenames_pre[i]) #PBCD_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['LBDBPBSI',
        'LBDBPBLC',
        'LBDBCDSI',
        'LBDBCDLC',
        'LBDTHGSI',
        'LBDTHGLC',
        'LBDBSESI',
        'LBDBSELC',
        'LBDBMNSI',
        'LBDBMNLC']
    
    i = 32
    print(lab_filenames_pre[i]) #PFAS_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['LBDPFDEL',
        'LBDPFHSL',
        'LBDMPAHL',
        'LBDPFNAL',
        'LBDPFUAL',
        'LBDPFDOL',
        'LBDNFOAL',
        'LBDBFOAL',
        'LBDNFOSL',
        'LBDMFOSL']
    
    i = 33
    print(lab_filenames_pre[i]) #PHTHTE_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['URDCNPLC',
        'URDCOPLC',
        'URDECPLC',
        'URDHIBLC',
        'URDMBPLC',
        'URDMC1LC',
        'URDMCOLC',
        'URDMEPLC',
        'URDMHBLC',
        'URDMHHLC',
        'URDMCHLC',
        'URDMHPLC',
        'URDMIBLC',
        'URDMNPLC',
        'URDMOHLC',
        'URDMZPLC']
    
    i = 34
    print(lab_filenames_pre[i]) #SSMHHT_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['SSMHHTL',
        'SSECPTL',
        'SSMONPL']
    
    i = 35
    print(lab_filenames_pre[i]) #SSNEON_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['SSIMIDLC',
        'SSACETLC',
        'SSCLOTLC',
        'SSTHIALC',
        'SSOHIMLC',
        'SSANDLC']
    
    i = 36
    print(lab_filenames_pre[i]) #TCHOL_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['LBDTCSI']
    
    i = 37
    print(lab_filenames_pre[i]) #TFR_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['LBDTFRSI']
    
    i = 38
    print(lab_filenames_pre[i]) #TRICH_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = []
    
    i = 39
    print(lab_filenames_pre[i]) #TRIGLY_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['LBDTRSI',
        'LBDLDLSI']
    
    i = 40
    print(lab_filenames_pre[i]) #TST_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['LBDTSTLC',
        'LBDESTLC',
        'LBDSHGLC']
    
    i = 41
    print(lab_filenames_pre[i]) #UADM_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['URD4DALC',
        'URD6DALC',
        'URD4MALC',
        'URD5NALC',
        'URDPDALC']
    
    i = 42
    print(lab_filenames_pre[i]) #UASS_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['URDUA3LC',
        'URDUA5LC',
        'URDUABLC',
        'URDUACLC',
        'URDUDALC',
        'URDUMMAL']
    
    i = 43
    print(lab_filenames_pre[i]) #UAS_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['URDUA3LC',
        'URDUA5LC',
        'URDUABLC',
        'URDUACLC',
        'URDUDALC',
        'URDUMMAL']
    
    i = 44
    print(lab_filenames_pre[i]) #UCFLOW_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = []
    
    i = 45
    print(lab_filenames_pre[i]) #UCPREG_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = []
    
    i = 46
    print(lab_filenames_pre[i]) #UHG_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['URDUHGLC']
    
    i = 47
    print(lab_filenames_pre[i]) #UIO_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['URDUIOLC']
    
    i = 48
    print(lab_filenames_pre[i]) #UMS_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['URDUBALC',
        'URDUCDLC',
        'URDUCOLC',
        'URDUCSLC',
        'URDUMOLC',
        'URDUMNLC',
        'URDUPBLC',
        'URDUSBLC',
        'URDUSNLC',
        'URDUSRLC',
        'URDUTLLC',
        'URDUTULC',
        'URDUURLC']
    
    i = 49
    print(lab_filenames_pre[i]) #UM_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['URDUBALC',
        'URDUCDLC',
        'URDUCOLC',
        'URDUCSLC',
        'URDUMOLC',
        'URDUMNLC',
        'URDUPBLC',
        'URDUSBLC',
        'URDUSNLC',
        'URDUSRLC',
        'URDUTLLC',
        'URDUTULC',
        'URDUURLC']
    
    i = 50
    print(lab_filenames_pre[i]) #UTASS_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = []
    
    i = 51
    print(lab_filenames_pre[i]) #UTAS_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = []
    
    i = 52
    print(lab_filenames_pre[i]) #UVOCS_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['URD1DCLC',
        'URD2DCLC',
        'URD2MHLC',
        'URD34MLC',
        'URDAAMLC',
        'URDAMCLC',
        'URDATCLC',
        'URDBMALC',
        'URDBPMLC',
        'URDCEMLC',
        'URDCYALC',
        'URDCYMLC',
        'URDDHBLC',
        'URDDPMLC',
        'URDGAMLC',
        'URDHEMLC',
        'URDHP2LC',
        'URDHPMLC',
        'URDPM1LC',
        'URDPM3LC',
        'URDMADLC',
        'URDMB1LC',
        'URDMB2LC',
        'URDMB3LC',
        'URDPHELC',
        'URDPHGLC',
        'URDPMALC',
        'URDPMMLC',
        'URDTCVLC']
    
    i = 53
    print(lab_filenames_pre[i]) #UVOC_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['URD1DCLC',
        'URD2DCLC',
        'URD2MHLC',
        'URD34MLC',
        'URDAAMLC',
        'URDAMCLC',
        'URDATCLC',
        'URDBMALC',
        'URDBPMLC',
        'URDCEMLC',
        'URDCYALC',
        'URDCYMLC',
        'URDDHBLC',
        'URDDPMLC',
        'URDGAMLC',
        'URDHEMLC',
        'URDHP2LC',
        'URDHPMLC',
        'URDPM1LC',
        'URDPM3LC',
        'URDMADLC',
        'URDMB1LC',
        'URDMB2LC',
        'URDMB3LC',
        'URDPHELC',
        'URDPHGLC',
        'URDPMALC',
        'URDPMMLC',
        'URDTCVLC']
    
    i = 54
    print(lab_filenames_pre[i]) #VOCWBS_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['LBD2DFLC',
        'LBD4CELC',
        'LBDV06LC',
        'LBDV07LC',
        'LBDV08LC',
        'LBDV1DLC',
        'LBDV2ALC',
        'LBDV3BLC',
        'LBDV4CLC',
        'LBDVBFLC',
        'LBDVBMLC',
        'LBDVBZLC',
        'LBDVZBLC',
        'LBDVC6LC',
        'LBDVCBLC',
        'LBDVCFLC',
        'LBDVCMLC',
        'LBDVCTLC',
        'LBDVDBLC',
        'LBDVDELC',
        'LBDVEELC',
        'LBDVDXLC',
        'LBDVEALC',
        'LBDVEBLC',
        'LBDVECLC',
        'LBDVFNLC',
        'LBDVIBLC',
        'LBDVIPLC',
        'LBDVMCLC',
        'LBDVMELC',
        'LBDVMPLC',
        'LBDVNBLC',
        'LBDVOXLC',
        'LBDVTCLC',
        'LBDVTELC',
        'LBDVFTLC',
        'LBDVHTLC',
        'LBDVTOLC',
        'LBDVTPLC',
        'LBDVVBLC',
        'LBDVXYLC']
    
    i = 55
    print(lab_filenames_pre[i]) #VOCWB_I
    lab_filename_remvarname_dict[lab_filenames_pre[i]] = ['LBD2DFLC',
        'LBD4CELC',
        'LBDV06LC',
        'LBDV07LC',
        'LBDV08LC',
        'LBDV1DLC',
        'LBDV2ALC',
        'LBDV3BLC',
        'LBDV4CLC',
        'LBDVBFLC',
        'LBDVBMLC',
        'LBDVBZLC',
        'LBDVZBLC',
        'LBDVC6LC',
        'LBDVCBLC',
        'LBDVCFLC',
        'LBDVCMLC',
        'LBDVCTLC',
        'LBDVDBLC',
        'LBDVDELC',
        'LBDVEELC',
        'LBDVDXLC',
        'LBDVEALC',
        'LBDVEBLC',
        'LBDVECLC',
        'LBDVFNLC',
        'LBDVIBLC',
        'LBDVIPLC',
        'LBDVMCLC',
        'LBDVMELC',
        'LBDVMPLC',
        'LBDVNBLC',
        'LBDVOXLC',
        'LBDVTCLC',
        'LBDVTELC',
        'LBDVFTLC',
        'LBDVHTLC',
        'LBDVTOLC',
        'LBDVTPLC',
        'LBDVVBLC',
        'LBDVXYLC']

# files_not_used
# lab_filenames_pre
# lab_filename_varname_dict
# varnames_lab_2015_clean1_dict
# varnames_lab_2015_clean1_lbl_dict 
# varnames_lab_2015_raw_dict
# varnames_lab_2015_raw_lbl_dict 
exec('varnames_' + lab_var_str + '_' + year + '_clean2_dict = {}')
exec('varnames_' + lab_var_str + '_' + year + '_clean2_lbl_dict = {}')
lab_filename_varname_clean2_dict = {}
    
for i, i_str in enumerate(lab_filenames_pre):   
    if i_str not in files_not_used:
        exec('varnames_' + lab_var_str + '_' + year + '_clean2_dict[i_str] = \'' + lab_var_str + '_' + year + '_' + i_str + '_clean2\'')
        exec('varnames_' + lab_var_str + '_' + year + '_clean2_lbl_dict[i_str] = \'' + lab_var_str + '_' + year + '_' + i_str + '_clean2_lbl\'')

        cols_toremove = lab_filename_remvarname_dict[i_str]
        
        exec('[' + lab_var_str + '_' + year + '_' + i_str + '_clean2, ' + 
               lab_var_str + '_' + year + '_' + i_str + '_clean2_lbl] = ' +
             'removeval(' +
             'cols_toremove, ' +
             varnames_lab_2015_clean1_dict[i_str] + ', ' +
             varnames_lab_2015_clean1_lbl_dict[i_str] + ')')
        exec('print(' + varnames_lab_2015_clean2_dict[i_str] + '.shape)')
        exec('lab_filename_varname_clean2_dict[i_str] = ' + varnames_lab_2015_clean2_lbl_dict[i_str]) # filename lists all column names

lab_save_variables_str = '[lab_filename_varname_clean2_dict, varnames_lab_2015_clean2_dict, '
for i, i_str in enumerate(lab_filenames_pre):   
    if i_str not in files_not_used:
        lab_save_variables_str = lab_save_variables_str + \
            lab_var_str + '_' + year + '_' + i_str + '_clean2, ' + \
            lab_var_str + '_' + year + '_' + i_str + '_clean2_lbl, '

lab_save_variables_str = lab_save_variables_str + ']'

type(cols_toremove)
type(lab_2015_ALB_CR_I_clean1_lbl)

# Save
with open(save_folder + lab_var_str + '_' + year + '_clean3_loadstr.pkl', 'wb') as f:
    pickle.dump(lab_save_variables_str, f)

with open(save_folder + lab_var_str + '_' + year + '_clean3.pkl', 'wb') as f:
    exec('pickle.dump(' + lab_save_variables_str + ', f)')


with open(save_folder + lab_var_str + '_' + year + '_clean3_loadstr.pkl', 'rb') as f:
    lab_save_variables_str = pickle.load(f)

with open(save_folder + lab_var_str + '_' + year + '_clean3.pkl', 'rb') as f:
    exec(lab_save_variables_str + ' = pickle.load(f)')





