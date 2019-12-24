#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wq_modeling.py

Created on Thu Dec 19 10:03:41 2019
@author: rtsearcy


Package of functions to create statistical water quality nowcast models from FIB and 
environmental data. Functions include:
    
    - fib_thresh: FIB exceedance thresholds
    - load_data: Load modeling datasets
    - clean: Cleans modeling datasets by imputing missing values, and removing missing 
      rows and columns
    - parse: Parse data into training and test subsets
    - pred_eval: Evaluates predictions statistics 
    - current_method: Computes performance metrics for the current method
    - fit: Fit model on training set
    - tune: TBD, tunes regression models to a certain set of performance standards
    - test: TBD

TODO List:
    
TODO - clean missing data - impute or delete columns/rows
TODO - fit: correlation, var selection, add more model types or create seperate functions
TODO - tune: add tuning function, create performance standard function with default
TODO - test: add testing function


"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.metrics import confusion_matrix, roc_curve, r2_score
import os
import sys

# %% DATA

no_model = ['sample_time', 'TC', 'FC', 'ENT', 'TC1', 'FC1', 'ENT1', 
            'TC_exc', 'FC_exc', 'ENT_exc', 'TC1_exc','FC1_exc', 'ENT1_exc', 
            'logTC1', 'logFC1', 'logENT1', # previous sample typically not included
            'wet','rad_1h', 'rad_2h', 'rad_3h', 'rad_4h',]


# %%
def fib_thresh(f):
    '''
    FIB exceedance thresholds for TC, FC/E. coli, ENT
    
    Parameters:
        - f = 'TC', 'FC', or 'ENT'
        
    Output:
        - integer of the exceedance threshold
        
    '''
    
    assert f in ['TC','FC','ENT'], 'FIB type must be either TC, FC, or ENT'
    if f == 'TC':
        return 10000
    elif f == 'FC':
        return 400
    elif f == 'ENT':
        return 104
        
    
# %%
def load_data(file, years=[1991], season='a'):
    '''
    Load modeling dataset and returns a Pandas DataFrame.
    - Set datetime index
    - Remove undesired modeling years
    - Prints FIB statistics in the dataset
    
    Parameters:
        - file = full file path with modeling dataset. Must:
                - BE A .CSV FILE
                - Contain a datetime column named 'dt' for the index
                - Contain FIB data (TC,FC, and/or ENT samples)
                
        - years = list of length 1 or 2 containing range of integer years to be loaded
                - No need to define for High Frequency sampling (i.e. one day - less than 
                  one year of data)
        - season = 'a', 's', or 'w', if data is to be modeled for a specific season
                - 'a' - All data/do not split by season       
                - 's' - Summer data only (Apr-Oct)
                - 'w' - Winter data only (Nov-Mar)
                
    Output:
        - df = Pandas DataFrame containing modeling dataset with a sorted datetime index
        
    '''
    
    assert type(years) == list, 'years paramater must be a list of size 1 or 2'
    assert years[0] in range(1991,2100), 'years parameter list must contain an integer year greater than 1991'
    if len(years) == 2:
        assert years[1] in range(1991,2100) and years[1] >= years[0], 'second years parameter must be an integer year >= to the first years parameter'
    assert season in ['a','s','w'], 'season paramter must be either \'a\', \'s\', or \'w\''
    
    df = pd.read_csv(file)
    assert 'dt' in df.columns, 'No datetime column named \'dt\' found'
    df['dt'] = pd.to_datetime(df['dt'])
    df.set_index('dt', inplace=True)
    # Sort data into ascending time index (Earliest sample first)
    df.sort_index(ascending=True, inplace=True) 
    
    # Remove years not in desired range
    df = df[df.index.year >= years[0]] # start year
    if len(years) == 2:
        df = df[df.index.year <= years[1]]
    
    # If seasonal, remove other season data
    if season == 's':
        df = df[(df.index.month >= 4) & (df.index.month < 11)]
    elif season == 'w':
        df = df[(df.index.month <= 3) | (df.index.month >= 11)]
        
    # TODO Remove rows with missing data/IMPUTE OR CREATE CLEAN FUNCTION
    
    # FIB Statistics and Plots
    print('\n- - | FIB Statistics | - -\n')
    print('Dataset: ' + file)
    print('\nStart Year: ' + str(df.index.year[0]) + '; End Year: ' + str(df.index.year[-1]))        
    if season == 's':
        print('Season: Summer')
    elif season == 'w':
        print('Season: Winter')
    else: print('Season: All Data')
    print('Number of Samples: ' + str(df.shape[0]))
    
    fib = []
    for f in ['TC','FC','ENT']:
        fib.append(f) if f in df.columns else print(f + ' not in dataset')
    assert len(fib) > 1, '- - No FIB data in this dataset --'
    print(df[fib].describe())
    
    # Previous FIB / Exceedances / Log10 Transform
    print('\nExceedances:')
    # fib_thresh = {'TC': 10000, 'FC': 400, 'ENT': 104}
    for f in fib:
        if f + '1' not in df.columns: # Previous FIB sample
            df[f + '1'] = df[f].shift(1)
        if f + '_exc' not in df.columns: # Exceedance variables
            df[f + '_exc'] = (df[f] > fib_thresh(f)).astype(int)
            df[f + '1_exc'] = (df[f+'1'] > fib_thresh(f)).astype(int)
            print(f + ' : ' + str(sum(df[f + '_exc'])))
        if 'log' + f not in df.columns: # log10 transformed FIB variables
            df['log' + f] = np.log10(df[f])
            df['log' + f + '1'] = np.log10(df[f + '1'])
            
    print('\nNumber of Columns: ' + str(df.shape[1]))
    
    return df


#%%
def clean(df):
    '''
    Cleans modeling datasets by imputing missing values, and removing missing 
    rows and columns
    
    Parameters:
        - df = Input dataframe (result from load_data)
        
    Output:
        - df_out = Cleaned Dataframe without missing values (Ready for modeling)
    '''
    
    df_out = df
    
    return df_out


#%% 
def parse(df, season='a', fib='ENT', parse_type='c', test_percentage=0.3, save_dir=None):
    '''
    Parse dataset into training and test subset to be used for model fitting
    and evaluation.
    
    Parameters:
        - df = DataFrame containing modeling dataset. Must run load_dataset function first
        
        - season = 'a', 's', or 'w', if data is to be modeled for a specific season
                - 'a' - All data/do not split by season       
                - 's' - Summer data only (Apr-Oct)
                - 'w' - Winter data only (Nov-Mar)
        
        - fib = FIB type to be modeled (TC, FC, or ENT). Default ENT
        
        - parse_type = 'c' or 'j' (Chronological or Jackknife methods, respectively)
            - 'c': Splits dataset chronologically (i.e. the last xx years/% of data into 
              the test set; the remaining into the training set)
            - 'j': Splits dataset using a random xx% of data for the test subset, and the
              remaining for the training subset
              
        - test_percentage = Percent of dataset that will go into the test subset
            - If parse_type = 'c', then the nearest whole year/season will be used
            
        - save_dir = name of the directory for the training and testing subsets to be saved
            - Will create a new directory if it doesn't exist
            
    Output:
        - y_train, X_train, y_test, X_test = Training and test subsets
            - y - Pandas Series, X - Pandas Dataframes
            - y and X DFs have matching indices
        
    '''
    
    # assert statements
    assert fib in ['TC','FC','ENT'], 'FIB type must be either TC, FC, or ENT'
    assert parse_type in ['c','j'], 'parse_type must be either \'c\' or \'j\''
    assert type(test_percentage) == float and 0 <= test_percentage < 1.0, 'test_percentage must be a fraction value'
    
    # Remove other FIB variables
    print('\n- - | Parsing Dataset | - -')
    other_fib = [x for x in ['TC','FC','ENT'] if x != fib]
    cols = df.columns
    for i in range(0, len(other_fib)):
        cols = [x for x in cols if other_fib[i] not in x]
    df = df[cols]  
    # TODO - check for previous FIB and logFIB vars, add if not there
    
    print('FIB ' + fib)
    df.sort_index(ascending=True, inplace=True) # Sort data into ascending time index
    # Split
    if parse_type == 'c':
        if len(np.unique(df.index.year)) == 1: #If High freq. data or less than one year
            num_test_sam = int(len(df)*test_percentage)
            test_data = df.iloc[-num_test_sam:]  # Select last % of samples for test set
        else:
            test_yr_e = str(max(df.index.year))  # End year
            num_test_yrs = round(test_percentage * (max(df.index.year) - min(df.index.year)))
            if any(x in np.unique(df.index.month)for x in [1,2,3,11,12]) & any(x not in np.unique(df.index.month)for x in [4,5,6,7,8,9,10]):
                test_yr_s = str(int(test_yr_e) - num_test_yrs)  # Start year
                temp_test = df[test_yr_s:test_yr_e]
                test_data = temp_test[~((temp_test.index.month.isin([1, 2, 3])) &
                                        (temp_test.index.year.isin([test_yr_s])))].sort_index(ascending=True)
            # Ensure winter seasons (which cross over years) are bundled together   
            else:
                test_yr_s = str(int(test_yr_e) - num_test_yrs + 1)  # Start year
                test_data = df[test_yr_s:test_yr_e].sort_index(ascending=True)  
        #Set remaining data to training subset    
        train_data = df[~df.index.isin(test_data.index)].sort_index(ascending=True)
    
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
        print('Parse Method: Jackknife')
        print('   Test Set Percentage: ' + str(test_percentage*100) + '%')
    
    # Account for NA samples
    y_test = y_test.dropna()
    X_test = X_test.reindex(y_test.index)
    y_train = y_train.dropna()
    X_train = X_train.reindex(y_train.index)
    
    # Check Exceedances
    train_exc = X_train[fib + '_exc'].sum() # Assune 'FIB_exc' vars previously calculated
    test_exc = X_test[fib + '_exc'].sum()
    print('\nTraining (calibration) subset:\n' + '  Samples - ' + str(len(X_train))  
    + '\n  Exc. - ' + str(train_exc) + ' (' + str(round(100*train_exc/len(X_train),1)) + '%)')
    print('\nTest (validation) subset:\n' + '  Samples - ' + str(len(X_test)) 
    + '\n  Exc. - ' + str(test_exc)+ ' (' + str(round(100*test_exc/len(X_test),1)) + '%)')
    
    print('\nNumber of Variables: ' + str(X_train.shape[1]))
    
    # Save train and test sets seperately
    if save_dir != None:
        try:
            os.makedirs(save_dir, exist_ok=True)  # Create dir if doesn't exist
            train_fname = 'training_subset_' + fib + '_' + parse_type + '_' + str(min(df.index.year)) + '_' + str(max(df.index.year)) + '.csv' # Training set filename
            test_fname = train_fname.replace('training', 'test') # Test set filename
            test_data = y_test.to_frame().merge(X_test, left_index=True, right_index=True)
            train_data = y_train.to_frame().merge(X_train, left_index=True, right_index=True)
            train_data.to_csv(os.path.join(save_dir, train_fname))
            test_data.to_csv(os.path.join(save_dir, test_fname))
            print('\nTraining and test subsets saved to: \n' + save_dir)
            
        except Exception as exc:
            print('\nERROR (There was a problem saving the parsed files: %s)' % exc)
            # continue

    return y_train, X_train, y_test, X_test
    

# %%
def pred_eval(true, predicted, thresh=0.5, tune=False):  # Evaluate Model Predictions
    '''
    Evaluates model sensitivity, specificity, and accuracy for a given set of predictions
    
    Parameters:
        - true = Pandas or Numpy Series of true values
        
        - predicted = Pandas or Numpy Series of model predictions
        
        - thresh = threshold above which a positive outcome is predicted (i.e. FIB 
          exceedance)
        
        - tune = True if model tuning (see below function)
        
    Output:
        - out = Dictionary of performance statistics
            - Sensitivity (True Positive Rate)
            - Specificity (True Negative Rate)
            - Accuracy (Total correctly Predicted)
            - Samples
            - Exceedances
    '''
#    if true.dtype == 'float':
#        true = (true > thresh).astype(int)  # Convert to binary
#    if predicted.dtype == 'float':
#        predicted = (predicted > thresh).astype(int)
        
#    true = (true > thresh).astype(int)  # Convert to binary
#    predicted = (predicted > thresh).astype(int)

#    cm = confusion_matrix(true, predicted)  
#    # Lists number of true positives, true negatives,false pos,and false negs.
#    sens = cm[1, 1] / (cm[1, 1] + cm[1, 0])  # sensitivity - TP / TP + FN
#    spec = cm[0, 0] / (cm[0, 1] + cm[0, 0])  # specificity - TN / TN + FP
#    acc = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 1] + cm[0, 1] + cm[1, 0])
    
    samples = len(true)  # number of samples
    exc = (true>thresh).sum()  # number of exceedances
    
    tp = np.sum((true > thresh) & (predicted > thresh))  # True positives
    tn = np.sum((true < thresh) & (predicted < thresh))  # True negatives
    fp = np.sum((true < thresh) & (predicted > thresh))  # False positives
    fn = np.sum((true > thresh) & (predicted < thresh))  # False negative

    sens = tp / (tp + fn)  # Sensitivity
    spec = tn / (tn + fp)  # Specificity
    acc = (tn + tp) / samples  # Accuracy

    if tune == False:
        out = {'Sensitivity': round(sens, 3), 'Specificity': round(spec, 3), 
               'Accuracy': round(acc, 3), 'Samples': samples, 'Exceedances': exc}
    else:
        out = [round(sens, 3), round(spec, 3)]

    return out


# %%
def current_method(df, fib='ENT'):
    '''
    Computes prediction metrics for the current method / persistence method:
        FIB = FIB1 [Predicted FIB = FIB from previous sample]
        
    Parameters:
        - df = DataFrame containing FIB data (must have current and previous sample 
          variables)
        
        - fib = 'TC', 'FC', or 'ENT'
        
    Output:
        - Dictionary containing the performance metrics for the current method of the
          dataset. Returns NoneType if the function cannot find FIB data
    '''
    
    if all(f in df.columns for f in [fib, fib + '1']):
        return pred_eval(df[fib], df[fib+'1'], thresh=fib_thresh(fib))
    elif all(f in df.columns for f in ['log' + fib, 'log' + fib + '1']):
        return pred_eval(df['log'+fib], df['log'+fib+'1'], thresh=np.log10(fib_thresh(fib)))
    elif all(f in df.columns for f in [fib + '_exc', fib + '1_exc']):
        return pred_eval(df[fib+'_exc'], df[fib+'1_exc'])
    else:
        print('Cannot compute the current method performance for this dataset')
        return
    
#%% FIT MODEL
    
## Inputs: 
## y_train, X_train, y_test, X_train (training and testing subset)
## fib (TC,FC, or ENT)
## model_type (MLR (mlr), BLR (blr), Random Forest (rf), Neural Net (nn))
## current_method (True/Fals)
#

#
#print('\n- - | Modeling (' + fib + ') | - -')
#cols_perf = ['Sensitivity', 'Specificity', 'Accuracy', 'Exceedances', 'Samples']
#df_perf = pd.DataFrame(columns = ['Model','Dataset'] + cols_perf)
#df_perf = df_perf.set_index(['Model', 'Dataset']) 
#
## Check Exceedances
#train_exc = X_train[f + '_exc'].sum()  # repeated from above
#test_exc = X_test[f + '_exc'].sum()
#if (train_exc < 2) | (test_exc == 0):  # If insufficient exceedances in cal/val sets, use new split method
#    sys.exit('* Insufficient amount of exceedances in each dataset.Program terminated *')
#
    
# Current Method
#cm_train = X_train[[fib + '_exc', fib + '1_exc' ]]  #Assune 'FIB_exc' vars previously calculated
#cm_test = X_test[[fib + '_exc', fib + '1_exc' ]]
#
#df_perf = df_perf.append(pd.DataFrame(model_eval(cm_train[f + '_exc'], cm_train[f + '1_exc']),
#                                      index=[['Current Method'], ['Calibration']]),sort=False)  # CM performance
#df_perf = df_perf.append(pd.DataFrame(model_eval(cm_test[f + '_exc'], cm_test[f + '1_exc']),
#                                      index=[['Current Method'], ['Validation']]),sort=False)
##df_perf.index.names = ['Model', 'Dataset']  # Name multiindex
#df_perf = df_perf[cols_perf]
#cm_perf = df_perf.loc['Current Method']
#print('\n- Current (Persistence Method -\n')
#print(cm_perf)


#
## Drop variables NOT to be modeled
## no_model = [] default to drop
#to_model = [x for x in X_train.columns if x not in no_model]  # Drop excluded variables
#X_train = X_train[to_model]
#X_test = X_test[to_model]
#
## Linear Regression
##lm = LinearRegression()
##lm = Ridge()
#lm = Lasso()
#lm.fit(X_train,y_train)
#
##df_perf = df_perf.append(pd.DataFrame(model_eval(y_train, lm.predict(X_train), np.log10(fib_thresh[fib])),
##                                      index=[['Lasso Regression'], ['Calibration']]),sort=False)  # CM performance
##df_perf = df_perf.append(pd.DataFrame(model_eval(y_test, lm.predict(X_test), np.log10(fib_thresh[fib])),
##                                      index=[['Lasso Regression'], ['Validation']]),sort=False)
##lasso_perf = df_perf.loc['Lasso Regression']
##print('\n- Lasso Regression -\n')
##print(lasso_perf)
#
## Logistic Regression
#
##TODO Random Forests
#
##TODO Neural Networks
#
#
##%% TUNE MODEL (MLR, BLR)
#
#
##%% TEST MODEL