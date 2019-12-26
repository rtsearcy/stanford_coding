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

folder = '/Users/rtsearcy/Documents/Stanford/Projects/Targeted High Frequency Sampling/preliminary_analysis/raw_data/'
file = 'modeling_dataset_10min_april2013.csv'
file = 'modeling_dataset_agency_samples_2001_2018.csv'

# LOAD
df = wqm.load_data(os.path.join(folder, file), years=[2005,2019], season='a')

# %% CLEAN
ptd = 0.05  # Percent missing data allowed before variable drop
sv = []  # Variables to save from being dropped

df_clean = wqm.clean(df, percent_to_drop=ptd, save_vars=sv)  

#%% PARSE
y_train, X_train, y_test, X_test = wqm.parse(df_clean, 
                                             fib='ENT', 
                                             season='a', 
                                             parse_type='j', 
                                             test_percentage= 0.25,
                                             save_dir = os.path.join(folder,'test_dir'))

#%% FIT
no_model = ['sample_time', 'TC', 'FC', 'ENT', 'TC1', 'FC1', 'ENT1', 
            'TC_exc', 'FC_exc', 'ENT_exc', 'TC1_exc','FC1_exc', 'ENT1_exc', 
            'logTC1', 'logFC1', 'log ENT1', # previous sample typically not included
            'wet','rad_1h', 'rad_2h', 'rad_3h', 'rad_4h',]

model, df_perf = wqm.fit(y_train, X_train, cm=True, no_model = )