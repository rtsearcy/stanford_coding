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

folder = '/Users/rtsearcy/coding/water_quality_modeling/data_poor_beaches/thfs/'
file = 'modeling_dataset_agency_samples_2001_2013.csv'

df = wqm.load_data(os.path.join(folder, file), years=[2005,2019], season='a')

#%% 
y_train, X_train, y_test, X_test = wqm.parse(df, 
                                             fib='ENT', 
                                             season='a', 
                                             parse_type='j', 
                                             test_percentage= 0.25,
                                             save_dir = os.path.join(folder,'test_dir'))
