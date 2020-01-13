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

folder = '/Users/rtsearcy/data/water_quality_modeling/thfs/preliminary_analysis/traditional_nowcast'
file = 'Lovers_Point_modeling_dataset_20090101_20191031.csv'

# LOAD
df = wqm.load_data(os.path.join(folder, file), years=[2009,2019], season='a')

#%% CLEAN
ptd = 0.05  # Percent missing data allowed before variable drop
sv = []  # Variables to save from being dropped

df_clean = wqm.clean(df, percent_to_drop=ptd, save_vars=sv)  

#%% PARSE
y_train, X_train, y_test, X_test = wqm.parse(df_clean, 
                                             fib='ENT', 
                                             season='a', 
                                             parse_type='c', 
                                             test_percentage= 0.20,
                                             save_dir = os.path.join(folder,'test_dir'))

#%% CURRENT METHOD
cm_train = wqm.current_method(X_train, fib='ENT')
cm_test = wqm.current_method(X_test, fib='ENT')


#%% SELECT VARS
no_model = ['sample_time', 'TC', 'FC', 'ENT', 'TC1', 'FC1', 'ENT1', 
            'TC_exc', 'FC_exc', 'ENT_exc', 'TC1_exc','FC1_exc', 'ENT1_exc', 
            'logTC1', 'logFC1', 'logENT1', # previous sample typically not included
            'wet','rad_1h', 'rad_2h', 'rad_3h', 'rad_4h',
            'MWD','MWD1','MWD1_b','MWD1_b_max', 'MWD1_b_min']

X_train = wqm.select_vars(y_train, X_train, method='rfe', no_model=no_model)

#%% FIT
model, df_perf = wqm.fit(y_train, X_train)