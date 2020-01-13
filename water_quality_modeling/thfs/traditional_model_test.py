#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
traditional_model_test.py

Created on Thu Dec 19 10:03:41 2019
@author: rtsearcy

Description: Creates a traditional nowcast model from years of FIB and environmental data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.metrics import confusion_matrix, roc_curve, r2_score
import os
import sys

#%% INPUT
file = 'modeling_dataset_agency_samples_2001_2013.csv'  # dataset file (path is relative)
file = 'modeling_dataset_10min_april2013.csv'
season = 'a' # a - all data, s - summer (Apr-Oct), w - winter (Nov-Mar)
years = [2001] # list of min and max years [st_yr, end_yr]; 
# input start year only if you want to use up until the end of the dataset

fib = 'ENT' # fib - TC, FC, or ENT
parse_type = 'c' # c - chronological; j - jackknife
test_percentage = 0.3 
#chronological: round to the nearest whole year; jackknife: fraction of whole into test

no_model = ['sample_time', 'TC', 'FC', 'ENT', 'TC1', 'FC1', 'ENT1', 
            'TC_exc', 'FC_exc', 'ENT_exc', 'TC1_exc','FC1_exc', 'ENT1_exc', 
            'logTC1', 'logFC1', 'logENT1', # previous sample typically not included
            'wet','rad_1h', 'rad_2h', 'rad_3h', 'rad_4h',]

#%% LOAD DATA INTO DF
df = pd.read_csv(file, parse_dates=['dt'], index_col=['dt'])

# Remove years not in desired range
df = df[df.index.year >= years[0]] # start year
if len(years) == 2:
    df = df[df.index.year <= years[1]]

# If seasonal, remove other season data
if season == 's':
    df = df[(df.index.month >= 4) & (df.index.month < 11)]
elif season == 'w':
    df = df[(df.index.month <= 3) | (df.index.month >= 11)]
    
# TODO Remove rows with missing data/IMPUTE

# FIB Statistics and Plots
print('\n- - | FIB Statistics | - -\n')
print('Start Year: ' + str(df.index.year[0]) + '; End Year: ' + str(df.index.year[-1]))        
if season == 's':
    print('Season: Summer')
elif season == 'w':
    print('Season: Winter')
else: print('Season: All Data')
print('Number of Samples: ' + str(len(df)))
print(df[['TC', 'FC', 'ENT']].describe())

# Exceedances
print('\nExceedances:')

fib_thresh = {'TC': 10000, 'FC': 400, 'ENT': 104}
for f in fib_thresh:
    if f + '_exc' not in df.columns:
        df[f + '_exc'] = (df[f] > fib_thresh[f]).astype(int)
        df[f + '1_exc'] = (df[f+'1'] > fib_thresh[f]).astype(int)
        print(f + ' : ' + str(sum(df[f + '_exc'])))

## TODO initial removal over overly correlated variables

#%% PARSE INTO TRAIN AND TEST SETS
# Remove other FIB variables
print('\n- - | Dataset Parsing | - -')
other_fib = [x for x in ['TC','FC','ENT'] if x != fib]
cols = df.columns
for i in range(0, len(other_fib)):
    cols = [x for x in cols if other_fib[i] not in x]
df = df[cols]
print('Splitting dataset for: ' + fib)
df.sort_index(ascending=True, inplace=True) # Sort data into ascending time index

# Split
if parse_type == 'c':
    if len(np.unique(df.index.year)) == 1:
        num_test_sam = int(len(df)*test_percentage)
        test_data = df.iloc[-num_test_sam:]  # Select last % of samples for test set
    else:
        num_test_yrs = round(test_percentage * (max(df.index.year) - min(df.index.year)))
        test_yr_e = str(max(df.index.year))  # End year
        if season in ['a', 's']:
            test_yr_s = str(int(test_yr_e) - num_test_yrs + 1)  # Start year
            test_data = df[test_yr_s:test_yr_e].sort_index(ascending=False)
    
        elif season == 'w':
            test_yr_s = str(int(test_yr_e) - num_test_yrs)  # Start year
            temp_test = df[test_yr_s:test_yr_e]
            test_data = temp_test[~((temp_test.index.month.isin([1, 2, 3])) &
                                    (temp_test.index.year.isin([test_yr_s])))].sort_index(ascending=False)
        # Ensure winter seasons (which cross over years) are bundled together        
    train_data = df[~df.index.isin(test_data.index)].sort_index(ascending=False)

    y_test = test_data['log' + fib]
    X_test = test_data.drop('log' + fib, axis=1)
    y_train = train_data['log' + fib]
    X_train = train_data.drop('log' + fib, axis=1)
    print('Parse Method: Chronological')
    print('   Training Set: ' + str(min(y_train.index.year)) + ' - '
          + str(max(y_train.index.year)))
    print('   Test Set: ' + str(min(y_test.index.year)) + ' - '
          + str(max(y_test.index.year)))

else:
    y = df['log' + fib]  # Separate into mother dependent and independent datasets
    X = df.drop('log' + fib, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage,
                                                        random_state=0)
    print('\nParse Method: Jackknife')
    print('   Test Set Percentage: ' + str(test_percentage*100) + '%')

# Account for NA samples
y_test = y_test.dropna()
X_test = X_test.reindex(y_test.index)
y_train = y_train.dropna()
X_train = X_train.reindex(y_train.index)

# Check Exceedances
train_exc = X_train[f + '_exc'].sum() # Assune 'FIB_exc' vars previously calculated
test_exc = X_test[f + '_exc'].sum()
print('\nTraining (calibration) subset:\n' + '  Samples - ' + str(len(X_train))  
+ '\n  Exc. - ' + str(train_exc) + ' (' + str(100*train_exc/len(X_train)) + '%)')
print('Test (validation) subset:\n' + '  Samples - ' + str(len(X_test)) 
+ '\n  Exc. - ' + str(test_exc)+ ' (' + str(100*test_exc/len(X_train)) + '%)')

## TODO Save train and test sets seperately


#%% FIT MODEL
# Inputs: 
# y_train, X_train, y_test, X_train (training and testing subset)
# fib (TC,FC, or ENT)
# model_type (MLR (mlr), BLR (blr), Random Forest (rf), Neural Net (nn))
# current_method (True/Fals)

def model_eval(true, predicted, thresh=0.5, tune=0):  # Evaluate Model Performance
    if true.dtype == 'float':
        true = (true > thresh).astype(int)  # Convert to binary
    if predicted.dtype == 'float':
        predicted = (predicted > thresh).astype(int)

    cm = confusion_matrix(true, predicted)  
    # Lists number of true positives, true negatives,false pos,and false negs.
    sens = cm[1, 1] / (cm[1, 1] + cm[1, 0])  # sensitivity - TP / TP + FN
    spec = cm[0, 0] / (cm[0, 1] + cm[0, 0])  # specificity - TN / TN + FP
    acc = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 1] + cm[0, 1] + cm[1, 0])
    samples = len(true)  # number of samples
    exc = true.sum()  # number of exceedances

    if tune == 0:
        out = {'Sensitivity': round(sens, 3), 'Specificity': round(spec, 3), 
               'Accuracy': round(acc, 3), 'Samples': samples, 'Exceedances': exc}
    else:
        out = [round(sens, 3), round(spec, 3)]

    return out

print('\n- - | Modeling (' + fib + ') | - -')
cols_perf = ['Sensitivity', 'Specificity', 'Accuracy', 'Exceedances', 'Samples']
df_perf = pd.DataFrame(columns = ['Model','Dataset'] + cols_perf)
df_perf = df_perf.set_index(['Model', 'Dataset']) 

# Check Exceedances
train_exc = X_train[f + '_exc'].sum()  # repeated from above
test_exc = X_test[f + '_exc'].sum()
if (train_exc < 2) | (test_exc == 0):  # If insufficient exceedances in cal/val sets, use new split method
    sys.exit('* Insufficient amount of exceedances in each dataset.Program terminated *')

# Current Method
cm_train = X_train[[fib + '_exc', fib + '1_exc' ]]  #Assune 'FIB_exc' vars previously calculated
cm_test = X_test[[fib + '_exc', fib + '1_exc' ]]

df_perf = df_perf.append(pd.DataFrame(model_eval(cm_train[f + '_exc'], cm_train[f + '1_exc']),
                                      index=[['Current Method'], ['Calibration']]),sort=False)  # CM performance
df_perf = df_perf.append(pd.DataFrame(model_eval(cm_test[f + '_exc'], cm_test[f + '1_exc']),
                                      index=[['Current Method'], ['Validation']]),sort=False)
#df_perf.index.names = ['Model', 'Dataset']  # Name multiindex
df_perf = df_perf[cols_perf]
cm_perf = df_perf.loc['Current Method']
print('\n- Current (Persistence Method -\n')
print(cm_perf)

# Drop variables NOT to be modeled
# no_model = [] default to drop
to_model = [x for x in X_train.columns if x not in no_model]  # Drop excluded variables
X_train = X_train[to_model]
X_test = X_test[to_model]

# Linear Regression
#lm = LinearRegression()
#lm = Ridge()
lm = Lasso()
lm.fit(X_train,y_train)

#df_perf = df_perf.append(pd.DataFrame(model_eval(y_train, lm.predict(X_train), np.log10(fib_thresh[fib])),
#                                      index=[['Lasso Regression'], ['Calibration']]),sort=False)  # CM performance
#df_perf = df_perf.append(pd.DataFrame(model_eval(y_test, lm.predict(X_test), np.log10(fib_thresh[fib])),
#                                      index=[['Lasso Regression'], ['Validation']]),sort=False)
#lasso_perf = df_perf.loc['Lasso Regression']
#print('\n- Lasso Regression -\n')
#print(lasso_perf)

# Logistic Regression

#TODO Random Forests

#TODO Neural Networks


#%% TUNE MODEL (MLR, BLR)


#%% TEST MODEL