#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_wqm.py

Created on Sat Dec 21 16:29:17 2019
@author: rtsearcy

Description: Used to test wq_modeling package
"""

import wq_modeling as wqm
import os 
import pandas as pd
import numpy as np

# folder = '/Users/rtsearcy/data/water_quality_modeling/thfs/preliminary_analysis/traditional_nowcast'
folder = '/Users/rtsearcy/Box/water_quality_modeling/thfs/traditional_nowcast/modeling_datasets'
file = 'LP_trad_modeling_dataset_20000101_20200301.csv'
f = 'ENT'

# LOAD
df = wqm.load_data(os.path.join(folder, file), years=[2013,2018], season='a')

coef_var = abs(df.describe().loc['std']/df.describe().loc['mean']).sort_values() # CV
FC_corr = abs(df.corr()['logFC']).sort_values()
ENT_corr = abs(df.corr()['logENT']).sort_values()

#%% CLEAN
ptd = 0.05  # Percent missing data allowed before variable drop
sv = []  # Variables to save from being dropped

df_clean = wqm.clean(df, percent_to_drop=ptd, save_vars=sv)  

#%% PARSE
y_train, X_train, y_test, X_test = wqm.parse(df_clean, 
                                             fib=f, 
                                             #season='s', 
                                             parse_type='c', 
                                             test_percentage= 0.01,
                                             save_dir = os.path.join(folder,'test_dir'))

df_cal = y_train.to_frame().merge(X_train, left_index=True, right_index=True)
cal_corr = abs(df_cal.corr()['log'+f]).sort_values()
#%% CURRENT METHOD
cm_train = wqm.current_method(X_train, fib=f)
cm_test = wqm.current_method(X_test, fib=f)

train_perf_df = pd.DataFrame(cm_train, index=['Current Method'])
test_perf_df = pd.DataFrame(cm_test, index=['Current Method'])

#%% SELECT VARS
no_model = ['sample_time', 'TC', 'FC', 'ENT', 'TC1', 'FC1', 'ENT1', 
            'TC_exc', 'FC_exc', 'ENT_exc', 'TC1_exc','FC1_exc', 'ENT1_exc', 
            'logTC1', 'logFC1', 'logENT1', # previous sample typically not included
            'wet', 'rad_2h', 'rad_3h', 'rad_4h',
            'MWD','MWD1','MWD1_b','MWD1_b_max', 'MWD1_b_min','MWD_b',
            'SIN_MWD1_b', 'SIN_MWD1_b_min','SIN_MWD1_b_max',
            'lograin4','lograin5','lograin6','lograin7']

# Traditional NC
#to_model= ['WVHT1_max','WVHT','WVHT1', 'SIN_MWD1_b', 'Wtemp_B1', 'Wtemp_L1_min','DPD1_max',
#           #'rad1',
#           'temp1_max', 'pres1', 'lograin1','lograin5', 'lograin2T','lograin3T','lograin5T','lograin14T',
#           'TideMin','TideR','TideLT_0','TideGT_2', 'weekend1']
#
#to_model = ['Wtemp_L1_min', 'TideLT_0', 'lograin1'] # 'all season, c'
#
#X_train_vs = X_train[to_model]
#X_test_vs = X_test[to_model]

# HF (2016)1
#to_model = [
            #'tide',
            #'atemp',
            #'rad',
            #'wspd',
            #'relhum',
            #'wtemp',
            #'APD',
            #'WVHT',
            #'MWD',
            #'pres',
#            ] #ENT
to_model = ['tide', 'wspd', 'atemp'] # FC
#to_model = ['tide']
X_train_vs = X_train[to_model]
#X_train_vs = wqm.multicollinearity_check(X_train_vs)

#X_train_vs = wqm.select_vars(y_train, X_train, 
#                             method='rfe', no_model=no_model, corr_thresh=0.95, vif=2.5)

X_test_vs = X_test[X_train_vs.columns]  # reshape X_test

#%% FIT MLR/BLR
mlr, mlr_perf = wqm.fit(y_train, X_train_vs, model_type='mlr')
n = len(y_train)
k = len(X_train_vs.columns)
r2a = 1 - ((1-mlr.score(X_train_vs, y_train))*(n-1))/(n-k-1)
print('\nR2_adj = ' + str(round(r2a,3)))
blr, blr_perf = wqm.fit(y_train, X_train_vs, model_type='blr')

#%% TUNE
tune_mlr = wqm.tune(y_train, X_train_vs, model=mlr, cm_perf=cm_train)
tune_blr = wqm.tune(y_train, X_train_vs, model=blr, cm_perf=cm_train)

#%% TRAIN/TEST PERFORMANCE
print('\n\n- - - | Metrics | - - -')
if np.isnan(tune_mlr):
    tune_mlr=1
if np.isnan(tune_blr):
    tune_blr=0.5

mlr_t_perf = wqm.pred_eval(y_train, mlr.predict(X_train_vs)*tune_mlr, thresh=np.log10(wqm.fib_thresh(f)))
train_perf_df = train_perf_df.append(pd.DataFrame(mlr_t_perf, index=['MLR-T']))

mlr_t_perf_test = wqm.pred_eval(y_test, mlr.predict(X_test_vs)*tune_mlr, thresh=np.log10(wqm.fib_thresh(f)))
test_perf_df = test_perf_df.append(pd.DataFrame(mlr_t_perf_test, index=['MLR-T']))

blr_t_perf = wqm.pred_eval(y_train > np.log10(wqm.fib_thresh(f)) , blr.predict_proba(X_train_vs)[:, 1] > tune_blr)
train_perf_df = train_perf_df.append(pd.DataFrame(blr_t_perf, index=['BLR-T']))

blr_t_perf_test = wqm.pred_eval(y_test > np.log10(wqm.fib_thresh(f)) , blr.predict_proba(X_test_vs)[:, 1] > tune_blr)
test_perf_df = test_perf_df.append(pd.DataFrame(blr_t_perf_test, index=['BLR-T']))

print('\nCalibration Metrics: ')
print(train_perf_df)
print('\nValidation Metrics: ')
print(test_perf_df)

#%%
print('\n- - | Test on 2016 data | - -')
df_new = wqm.load_data(os.path.join(folder,'modeling_dataset_10min_april2016.csv'), years=[2016,2018], season='a')
df_new = wqm.clean(df_new, percent_to_drop=.05, save_vars=sv)
cm_new = wqm.current_method(df_new, fib=f)
X_new = df_new[X_train_vs.columns]
y_new = df_new['log'+f]

new_perf = wqm.pred_eval(y_new, mlr.predict(X_new)*tune_mlr, thresh=np.log10(wqm.fib_thresh(f)))
print('\nCM')
print(cm_new)
print('MLR-HF')
print(new_perf)
new_perf_blr = wqm.pred_eval(y_new > np.log10(wqm.fib_thresh(f)) , blr.predict_proba(X_new)[:, 1] > tune_blr)
print('BLR-HF')
print(new_perf_blr)

#%%
print('\n- - | Test on season data | - -')
'''
Need to find proxy variables for now
'''
season_folder = '/Users/rtsearcy/data/water_quality_modeling/thfs/preliminary_analysis/traditional_nowcast'
season_file= 'Lovers_Point_modeling_dataset_20090101_20191031.csv'
df_season = wqm.load_data(os.path.join(season_folder, season_file), years=[2019,2019], season='s')
df_season = wqm.clean(df_season, percent_to_drop=.05, save_vars=[])

cm_season = wqm.current_method(df_season , fib=f)  

# Traditional NowCast
if f == 'FC':
    trad_folder = '/Users/rtsearcy/data/water_quality_modeling/thfs/preliminary_analysis/traditional_nowcast/good_models/FC/2009-2015'
    trad_pkl = 'model_LP_FC_09_15_all_jack_3var_MLR-T.pkl'
    trad_coef = 'coefficients_LP_FC_09_15_all_jack_3var_MLR-T.csv'
else:
    trad_folder = '/Users/rtsearcy/data/water_quality_modeling/thfs/preliminary_analysis/traditional_nowcast/good_models/ENT/2009-2015'
    trad_pkl = 'model_LP_ENT_09_15_summer_jack_lasso_5var_MLR-T.pkl'
    trad_coef = 'coefficients_LP_ENT_09_15_summer_jack_lasso_5var_MLR-T.csv'
    
trad_model = joblib.load(os.path.join(trad_folder, trad_pkl))
df_coef = pd.read_csv(os.path.join(trad_folder, trad_coef), header=None)
df_coef.columns = ['Variable', 'Coefficient']
df_coef.set_index('Variable', inplace=True)
tuner = float(df_coef.loc['tune'])  # extract PM/thresh tuner and constant from coef dataframe
constant = float(df_coef.loc['constant'])
df_coef.drop(['tune', 'constant'], inplace=True)
model_vars = list(df_coef.index)
print(df_coef)

trad_model_vars = df_season[model_vars]

df_season.rename(columns = {'wspd1':'wspd',
                     'temp1':'atemp',
                     'rad1':'rad',
                     'Tide6':'tide',
                     'Wtemp_B':'wtemp',
                     'pres1':'pres'},
          inplace=True)  

X_season = df_season[X_train_vs.columns]
y_season = df_season['log'+f]
trad_perf = wqm.pred_eval(y_season, tuner * trad_model.predict(trad_model_vars.values), thresh=np.log10(wqm.fib_thresh(f)))

season_perf = wqm.pred_eval(y_season, mlr.predict(X_season)*tune_mlr, thresh=np.log10(wqm.fib_thresh(f)))
print('\nCM')
print(cm_season)
print('MLR-HF')
print(season_perf)
season_perf_blr = wqm.pred_eval(y_season > np.log10(wqm.fib_thresh(f)) , blr.predict_proba(X_season)[:, 1] > tune_blr)
print('BLR-HF')
print(season_perf_blr)
print('Traditional')
print(trad_perf)






