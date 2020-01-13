#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vars_combo_2.py
Created on Wed Jan  8 19:37:57 2020

@author: rtsearcy
"""

import pandas as pd
import os
from numpy import sin, cos, pi, isnan, nan







folder = '/Users/rtsearcy/data/water_quality_modeling/thfs/preliminary_analysis/raw_data/traditional_nowcast'
fib_file = 'Lovers_Point_variables_fib.csv'

sd = '2009-01-01'
ed = '2019-10-31'
ang = 45 # Lover's Point

out_file = 'Lovers_Point_modeling_dataset_' + sd.replace('-','') + '_' + ed.replace('-','') + '.csv'

var_files = [f for f in os.listdir(folder) if f != fib_file and '.csv' in f]
df = pd.read_csv(os.path.join(folder,fib_file))
assert 'dt' in df.columns, 'No datetime column named \'dt\' found'
df['dt'] = pd.to_datetime(df['dt'])
df.set_index('dt', inplace=True)
# Sort data into ascending time index (Earliest sample first)
df.sort_index(ascending=True, inplace=True) 

for f in var_files:
    df_var = pd.read_csv(os.path.join(folder,f))
    if 'date' in df_var.columns:
        df_var['dt'] = df_var['date']
        df_var.drop(['date'],axis=1, inplace=True)
    elif 'dt' not in df_var.columns:
        print('No \'dt\' column in ' + f)
        continue
    df_var['dt'] = pd.to_datetime(df_var['dt'])
    df_var.set_index('dt', inplace=True)
    df_var.sort_index(ascending=True, inplace=True)
    df = df.merge(df_var, how = 'left', left_index=True, right_index=True)
    print(f + ' merged...')
    
if 'wspd1' in df.columns and 'wdir1' in df.columns:  # Wind speed/direction
    df['awind1'] = df['wspd1'] * round(sin(((df['wdir1'] - ang) / 180) * pi),1)
    df['owind1'] = df['wspd1'] * round(cos(((df['wdir1'] - ang) / 180) * pi),1)

if 'wspd_L1' in df.columns and 'wdir_L1' in df.columns:  # Local wind speed/direction
    df['awind_L1'] = df['wspd_L1'] * round(sin(((df['wdir_L1'] - ang) / 180) * pi),1)
    df['owind_L1'] = df['wspd_L1'] * round(cos(((df['wdir_L1'] - ang) / 180) * pi),1)

for w in ['MWD1', 'MWD1_max', 'MWD1_min']:  # Wave direction
    if w in df.columns:
        df['MWD1_b' + w.replace('MWD1', '')] = df[w] - ang  # Direction relative to beach
        df['SIN_MWD1_b' + w.replace('MWD1', '')] = \
            round(sin(((df['MWD1_b' + w.replace('MWD1', '')]) / 180) * pi), 3)
        df.drop([w], axis=1, inplace=True)

df = df[sd:ed]
df_out = df
df_out.to_csv(os.path.join(folder,out_file))
