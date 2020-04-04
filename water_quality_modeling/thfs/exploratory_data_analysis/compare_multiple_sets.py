# -*- coding: utf-8 -*-
'''
compare_multiple_sets.py
RTS - 4.4.2020

- Loads ALL modeling datasets containing FIB samples and Environmental Variables (EV)
    - 3 HF sampling events
        - (6/hr, 3/hr, 2/hr, 1/hr intervals for 2013)
    - traditional calibration (summers 2009-2013)
        - test (indiv. summers 2013-onward, indiv months)
    
- Create comparison tables
    - N, mean, median, max/min
    - exceedances, BLOQ
    - Autocorrelation (lag 1-3)
    - Correlation w/ top_vars (atemp, Wtemp_B, tide, wspd, wdir, WVHT, DPD, rad)

- TODO: Calculate comparison stats:

- Plots:
    - FIB boxplots
    - EV boxplots (atemp, Wtemp_B, tide, wspd, wdir, WVHT, DPD, rad)
    - 

'''

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
    

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf
import warnings
warnings.filterwarnings("ignore")

# %% Import Data

#Dir/files
folder = '/Users/rtsearcy/Box/water_quality_modeling/thfs/'
trad_file = 'traditional_nowcast/modeling_datasets/LP_trad_modeling_dataset_2009_2016.csv'
hf13_file = 'hf/modeling_datasets/LP_hf_modeling_dataset_20130420.csv'
hf16_file = 'hf/modeling_datasets/LP_hf_modeling_dataset_20160409.csv'
hf18_file = 'hf/modeling_datasets/LP_hf_modeling_dataset_20180421.csv'

save = 1
season = 'summer'
top_vars = ['tide','rad','Wtemp_B','atemp','wspd','wdir','WVHT','DPD']

trad_dates = {
        season + '_2009_2016': ['04-01-2009','10-31-2016'],  # All summer data
        season + '_2009_2013': ['04-01-2009','10-31-2013'],  # Calibration set for 2013 season
        season + '_2013': ['04-01-2013','10-31-2013'],  # 2013 season
        season + '_2014': ['04-01-2014','10-31-2014'],  # 2014 season
        'april_2013': ['04-01-2013','04-30-2013'],  # April in 2013 season
        'may_2013': ['05-01-2013','05-31-2013'],  # May in 2013 season
        }

df = []

## High Frequency
hf_13 = pd.read_csv(os.path.join(folder, hf13_file), parse_dates=['dt'], index_col=['dt'])
hf_13.index.rename('hf_2013', inplace=True)
df.append(hf_13)
hf_16 = pd.read_csv(os.path.join(folder, hf16_file), parse_dates=['dt'], index_col=['dt'])
hf_16.index.rename('hf_2016', inplace=True)
df.append(hf_16)
hf_18 = pd.read_csv(os.path.join(folder, hf18_file), parse_dates=['dt'], index_col=['dt'])
hf_18.index.rename('hf_2018', inplace=True)
df.append(hf_18)

# Sub-sampling frequency testing for HF 2013
for d in [hf_13]: # [hf_13, hf_16, hf_18]
    for p in [3,2,1]:
#        if p == 3:
#            df_temp = hf_13[(hf_13.index.minute == 0) | (hf_13.index.minute == 20) | (hf_13.index.minute == 40)]
#        elif p == 2:
#            df_temp = hf_13[(hf_13.index.minute == 0) | (hf_13.index.minute == 30)]
#        elif p == 1:        
#            df_temp = hf_13[(hf_13.index.minute == 0)]
        if p == 3:
            df_temp = d[(d.index.minute == 0) | (d.index.minute == 20) | (d.index.minute == 40)]
        elif p == 2:
            df_temp = d[(d.index.minute == 0) | (d.index.minute == 30)]
        elif p == 1:        
            df_temp = d[(d.index.minute == 0)]
        df_temp.index.rename(d.index.name + '_' + str(p) + '_per_hour', inplace=True)
        df.append(df_temp)
    
## Traditional
df_trad = pd.read_csv(os.path.join(folder, trad_file), parse_dates=['dt'], index_col=['dt'])
# If seasonal, remove other season data
if season == 'summer':
    df_trad = df_trad[(df_trad.index.month >= 4) & (df_trad.index.month < 11)]
elif season == 'winter':
    df_trad = df_trad[(df_trad.index.month <= 3) | (df_trad.index.month >= 11)]

for k in list(trad_dates.keys()):
    sd = trad_dates[k][0]
    ed = trad_dates[k][1]
    df_temp = df_trad[sd:ed].sort_index()
    df_temp.index.rename('trad_' + k, inplace=True)
    df.append(df_temp)


print([d.index.name for d in df])
#print([d['ENT'].mean() for d in df])

# %% FIB Statistics, Correlations
#print('\n- - | FIB Statistics | - -\n')
#
#df_fib = df
#
##df_fib = np.log10(df_fib)
#df_stats = df_fib.describe()
#df_stats.rename(index={'count':'N'}, inplace=True)
#
## CV
#cv = df.describe().loc['std'] / np.abs(df.describe().loc['mean'])
#cv.name = 'CV'
#df_stats = df_stats.append(cv.T)
#    
## Skewness/Kurtosis
#skew = df_fib.skew()
#skew.name = 'skewness'
#df_stats = df_stats.append(skew.T)
#
#kurt = df_fib.kurtosis()
#kurt.name = 'kurtosis'
#df_stats = df_stats.append(kurt.T)
#
## At or Below Level of Quantification
#bloq = (df_fib[['TC','FC','ENT']] == 10).sum()
#bloq.name = 'abloq'
#df_stats = df_stats.append(bloq.T)
#
## Exceedances
#exc = pd.Series()
#for f in ['TC','FC','ENT']:
#    exc[f] = (df_fib[f] > fib_thresh(f)).sum()
#exc.name = 'exceedances'
#df_stats = df_stats.append(exc.T)
#
#print(df_stats[['TC','FC','ENT']])
#
## Correlationns
#print('\n- - | FIB Correlations | - -\n')
#df_corr = df.corr()  # PCC
#print('log10(FC)')
#print(df_corr['logFC'].loc[top_vars])
#print('\nlog10(ENT)')
#print(df_corr['logENT'].loc[top_vars])
#    
## %% FIB Plots
#
## FIB boxplots
#fig1 = plt.figure(1)
#ax1 = np.log10(df[['FC','ENT']]).boxplot()
#ax1.set_ylabel(r'$log_{10}(MPN/100 ml)$')
#ax1.set_title(set_name)
#
## FIB histograms
#fig2 = plt.figure(2)
#plt.subplot(1,2,1)
#plt.hist(np.log10(df_fib['FC']),bins=20)
#plt.axvline(np.log10(400), c='r', ls='--')
#plt.title('FC')
#plt.ylabel('Count')
#plt.xlabel(r'$log_{10}(MPN/100 ml)$')
#
#plt.subplot(1,2,2)
#plt.hist(np.log10(df_fib['ENT']),bins=20)
#plt.axvline(np.log10(104), c='r', ls='--')
#plt.title('ENT')
#plt.xlabel(r'$log_{10}(MPN/100 ml)$')
#
#plt.suptitle(set_name)
#
## logFIB vs. time
#fig3 = plt.figure(3)
#plt.plot(np.log10(df_fib['FC']), c='b', ls='', marker='.', label='FC')
#plt.plot(np.log10(df_fib['ENT']), c='r', ls='', marker='.', label='ENT')
#
#plt.xlabel('')
#plt.ylabel('logFIB')
#plt.axhline(np.log10(104), color = 'b')
#plt.axhline(np.log10(400),xmin=0,xmax=1, color = 'r')
#plt.legend()
#plt.suptitle(set_name + '\n(horizontal lines indicate FIB SSS)')
#
## Autocorrelation
#ts = np.median([t.total_seconds() for t in (df.index[1:]- df.index[0:-1])])
#tm = int(ts/60)
#th = int(tm/60)
#td = round(th/24,2)
#if td == 0 & th == 0:
#    interval = str(tm) + ' min.'
#elif td == 0 & th > 0:
#    interval = str(th) + ' hr.'
#else:
#    interval = str(td) + ' days'
#
#if len(df)<10:
#    n = len(df) - 1
#else:
#    n = 10
#
#fig4 = plot_acf(df['logENT'], zero=False, lags=n)
#plt.title('ENT Autocorrelation - ' + set_name)
#plt.ylim(-1,1)
#plt.xlabel('Lag')
#if n ==10:
#    plt.xticks(ticks=[1,2,3,4,5,6,7,8,9,10])
#plt.text(6.5,0.8, 'Avg. interval: ' + interval, fontsize=10)
#
#fig5 = plot_acf(df['logFC'], zero=False, lags=n)
#plt.title('FC Autocorrelation - ' + set_name)
#plt.ylim(-1,1)
#plt.xlabel('Lag')
#if n ==10:
#    plt.xticks(ticks=[1,2,3,4,5,6,7,8,9,10])
#plt.text(6.5,0.8, 'Avg. interval: ' + interval, fontsize=10)
#    
## %% Enviro Variables Stats and Plots
##try:
##    df.drop(['TC', 'FC', 'ENT'], axis=1, inplace=True)  # drop untransformed FIB
##except KeyError:
##    pass
##
##print('\n- - | Environmental Variable (EV) Statistics | - -\n')
##
##for f in ['ENT', 'FC']:
##    PCC = df.corr()['log' + f]  # Pearson correlations (PCC)
##    PCC_a = np.abs(PCC).sort_values(ascending=False)  # absolute value or PCC, sorted
##
##    print('log' + f + ' - Strong Linear Correlation(absolute values)\n')
##    strong_corr = PCC_a[PCC_a >= 0.68]
##    print(strong_corr)
##    print('\nlog' + f + ' - Weak Linear Correlation(absolute values)\n')
##    print(PCC_a[PCC_a < 0.36])
##    # Categories from Taylor 1990 - Interpretation of the Correlation Coefficient: A Basic Review
##    #   0 < |r| < 0.35 -> Poorly correlated
##    #   0.36 < |r| < 0.67 -> Moderately correlated
##    #   0.68 < |r| < 1.0 -> Strongly correlated
##
##    print('\n')
##
##print('Coefficients of Variation for EVs (\u03C3/\u03BC):')  # CV = sigma/mu
##print((df.describe().loc['std'] / np.abs(df.describe().loc['mean'])).sort_values(ascending=False))
#
##TODO - Plot logFIB vs. rad/tide/wind (stacked subplots)
## Plot rad/rad_1h/rad2_h vs. FIB (stacked subplots) --> Is there a lag?
#
#plot_vars = ['logENT','tide']
#if 'wspd' in df.columns:
#    plot_vars += ['wspd']
#elif 'wspd_coops' in df.columns:
#    plot_vars += ['wspd_coops']
#    
#if 'rad' in df.columns:
#    plot_vars += ['rad']
#
#axes = df[plot_vars].plot(subplots=True, figsize=(10, 8), style = ['r.-','b.-','g.-','k.-'])
#axes[0].axhline(np.log10(104),xmin=0,xmax=1, color = 'r')
##labels
#axes[0].set_ylabel('logFIB')
#axes[1].set_ylabel('water level (m)')
#axes[2].set_ylabel('wind speed (m/s)')
#if 'rad' in plot_vars:
#    axes[3].set_ylabel('W/m^2')
#plt.suptitle(set_name)
#
## %% Save
#if save==1:
#    save_folder = os.path.join(folder,'EDA',set_name)
#    os.makedirs(save_folder, exist_ok=True)
#    
#    # Basic Stats
#    df_stats.to_csv(os.path.join(save_folder, 'basic_stats_' + set_name + '.csv'), index=True)
#    
#    # Basic Stats
#    df_corr.to_csv(os.path.join(save_folder, 'pearson_correlation_' + set_name + '.csv'), index=True)
#    
#    # FIB Figures
#    plt.savefig(os.path.join(save_folder, 'vars_time_series_' + set_name + '.png'))
#    
#    # Boxplots
#    plt.figure(1)
#    plt.savefig(os.path.join(save_folder, 'FIB_boxplots_' + set_name + '.png'))
#    
#    # Histograms
#    plt.figure(2)
#    plt.savefig(os.path.join(save_folder, 'FIB_histograms_' + set_name + '.png'))
#    
#    # Boxplots
#    plt.figure(3)
#    plt.savefig(os.path.join(save_folder, 'FIB_time_series_' + set_name + '.png'))
#    
#    # Autocorrelations
#    plt.figure(4)
#    plt.savefig(os.path.join(save_folder, 'ENT_autocorr_' + set_name + '.png'))
#    
#    plt.figure(5)
#    plt.savefig(os.path.join(save_folder, 'FC_autocorr_' + set_name + '.png'))
#    
#    
#    
