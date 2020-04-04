#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test save
Created on Mon Jan 20 16:27:34 2020

@author: rtsearcy
"""

import os
import pandas as pd
import wq_modeling as wqm

#%% SAVE MODEL

# Coef
model = blr
coef = model.coef_.reshape(-1)  # variable coefficients
intercept = float(model.intercept_)
df_coef = pd.Series(coef, index= X_train_vs.columns)
df_coef.loc['constant'] = intercept
df_coef.loc['tune'] = tune_blr

# Perf
df_perf = pd.DataFrame()
df_perf = df_perf.append(train_perf_df.loc[['Current Method', 'BLR-T']])
df_perf = df_perf.append(test_perf_df.loc[['Current Method', 'BLR-T']])

wqm.save_model(folder='/Users/rtsearcy/data/water_quality_modeling/thfs/preliminary_analysis/traditional_nowcast/good_models/FC/2009-2015',
           name='LP_FC_09_15_all_jack_3var', 
           model=model, 
           df_coef=df_coef,
           df_perf=df_perf)