# -*- coding: utf-8 -*-
'''
basic_stats_and_corr.py
RTS - 10.26.2019

- Loads modeling datasets containing FIB samples and Environmental Variables (EV)
    - 3 HF (6/hr, 3/hr, 2/hr)
    - traditional calibration (summers 2009-2013)
    - test (indiv. summers 2013-onward, indiv months)
    
- Calculates basic stats 
    - FIB - N, mean, median, min/max, SD,skewness/kurtosis, 
            % samples exceeding limits, % samples at/below LOQ
            - Plots histograms, boxplots, time series
            
    - EV’s - All, including lagged and instantaneous (at time of sample): 
        - atemp / atemp1
        - wtemp / wtemp1
        - wspd / wspd1
        - rad / rad1
        - tide / TideMax, TideMin, TideR
        - WVHT / WVHT1
        - Rainfall @ time lags prior to samples? 
    
- Calculates correlations between FIB and EVs

- Plots:
    - FIB boxplots
    - FIB histograms
    - FIB time series
    - Autocorrelation function for ENT and FC
    - Variable time series

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
#sd = '01-01-2000'  # All data
#ed = '03-31-2020'

#sd = '01-01-2009'  # Calibration set for 2013 season
#ed = '03-31-2013'

#sd = '04-01-2013'  # 2013 Season
#ed = '10-31-2013'

sd = '04-01-2013'  # April of 2013 Season
ed = '04-30-2013'

#sd = '05-01-2013'  # May of 2013 Season
#ed = '05-31-2013'

#sd = '04-01-2014'  # 2014 Season
#ed = '10-31-2014'

save = 1
season = 'summer'  # 's' (summer;apr-oct), 'w'(winter;nov-mar), or 'all' (all data)
per_hour = 6  # 6 - 1/10 min, 3 - 1/20 min, 2 - 1/30 min, 1 - hourly

folder = '/Users/rtsearcy/Box/water_quality_modeling/thfs/'

#file = 'traditional_nowcast/modeling_datasets/LP_trad_modeling_dataset_2009_2016.csv'
file = 'hf/modeling_datasets/LP_hf_modeling_dataset_20160409.csv'

top_vars = ['tide','rad','Wtemp_B','atemp','wspd','wdir','WVHT','DPD']

df = pd.read_csv(os.path.join(folder, file), parse_dates=['dt'], index_col=['dt'])
#df_hf = pd.read_csv(os.path.join(folder, hf_file), parse_dates=['dt'], index_col=['dt'])
if len(np.unique(df.index.year)) > 1:    
    df = df[sd:ed].sort_index()
    
    # If seasonal, remove other season data
    if season == 'summer':
        df = df[(df.index.month >= 4) & (df.index.month < 11)]
    elif season == 'winter':
        df = df[(df.index.month <= 3) | (df.index.month >= 11)]
else:
    # Sampling frequency testing
    if per_hour == 3:
        df = df[(df.index.minute == 0) | (df.index.minute == 20) | (df.index.minute == 40)]
    elif per_hour == 2:
        df = df[(df.index.minute == 0) | (df.index.minute == 30)]
    elif per_hour == 1:        
        df = df[(df.index.minute == 0)]

if 'hf' in file:
    set_name = 'hf_'+df.index[0].strftime(format='%m%d%Y')+'_'+str(per_hour)+'_per_hour'
else:
    sy = str(df.index[0].year)
    sm = '0'*(2-len(str(df.index[0].month))) + str(df.index[0].month)
    ey = str(df.index[-1].year)
    em = '0'*(2-len(str(df.index[-1].month))) + str(df.index[-1].month)
    
    set_name = 'trad_' + season + '_data_' + sy + sm + '_' + ey + em

print('Dataset: ' + set_name)


# %% FIB Statistics, Correlations
print('\n- - | FIB Statistics | - -\n')

df_fib = df

#df_fib = np.log10(df_fib)
df_stats = df_fib.describe()
df_stats.rename(index={'count':'N'}, inplace=True)

# CV
cv = df.describe().loc['std'] / np.abs(df.describe().loc['mean'])
cv.name = 'CV'
df_stats = df_stats.append(cv.T)
    
# Skewness/Kurtosis
skew = df_fib.skew()
skew.name = 'skewness'
df_stats = df_stats.append(skew.T)

kurt = df_fib.kurtosis()
kurt.name = 'kurtosis'
df_stats = df_stats.append(kurt.T)

# At or Below Level of Quantification
bloq = (df_fib[['TC','FC','ENT']] == 10).sum()
bloq.name = 'abloq'
df_stats = df_stats.append(bloq.T)

# Exceedances
exc = pd.Series()
for f in ['TC','FC','ENT']:
    exc[f] = (df_fib[f] > fib_thresh(f)).sum()
exc.name = 'exceedances'
df_stats = df_stats.append(exc.T)

print(df_stats[['TC','FC','ENT']])

# Correlationns
print('\n- - | FIB Correlations | - -\n')
df_corr = df.corr()  # PCC
print('log10(FC)')
print(df_corr['logFC'].loc[top_vars])
print('\nlog10(ENT)')
print(df_corr['logENT'].loc[top_vars])
    
# %% FIB Plots

# FIB boxplots
fig1 = plt.figure(1)
ax1 = np.log10(df[['FC','ENT']]).boxplot()
ax1.set_ylabel(r'$log_{10}(MPN/100 ml)$')
ax1.set_title(set_name)

# FIB histograms
fig2 = plt.figure(2)
plt.subplot(1,2,1)
plt.hist(np.log10(df_fib['FC']),bins=20)
plt.axvline(np.log10(400), c='r', ls='--')
plt.title('FC')
plt.ylabel('Count')
plt.xlabel(r'$log_{10}(MPN/100 ml)$')

plt.subplot(1,2,2)
plt.hist(np.log10(df_fib['ENT']),bins=20)
plt.axvline(np.log10(104), c='r', ls='--')
plt.title('ENT')
plt.xlabel(r'$log_{10}(MPN/100 ml)$')

plt.suptitle(set_name)

# logFIB vs. time
fig3 = plt.figure(3)
plt.plot(np.log10(df_fib['FC']), c='b', ls='', marker='.', label='FC')
plt.plot(np.log10(df_fib['ENT']), c='r', ls='', marker='.', label='ENT')

plt.xlabel('')
plt.ylabel('logFIB')
plt.axhline(np.log10(104), color = 'b')
plt.axhline(np.log10(400),xmin=0,xmax=1, color = 'r')
plt.legend()
plt.suptitle(set_name + '\n(horizontal lines indicate FIB SSS)')

# Autocorrelation
ts = np.median([t.total_seconds() for t in (df.index[1:]- df.index[0:-1])])
tm = int(ts/60)
th = int(tm/60)
td = round(th/24,2)
if td == 0 & th == 0:
    interval = str(tm) + ' min.'
elif td == 0 & th > 0:
    interval = str(th) + ' hr.'
else:
    interval = str(td) + ' days'

if len(df)<10:
    n = len(df) - 1
else:
    n = 10

fig4 = plot_acf(df['logENT'], zero=False, lags=n)
plt.title('ENT Autocorrelation - ' + set_name)
plt.ylim(-1,1)
plt.xlabel('Lag')
if n ==10:
    plt.xticks(ticks=[1,2,3,4,5,6,7,8,9,10])
plt.text(6.5,0.8, 'Avg. interval: ' + interval, fontsize=10)

fig5 = plot_acf(df['logFC'], zero=False, lags=n)
plt.title('FC Autocorrelation - ' + set_name)
plt.ylim(-1,1)
plt.xlabel('Lag')
if n ==10:
    plt.xticks(ticks=[1,2,3,4,5,6,7,8,9,10])
plt.text(6.5,0.8, 'Avg. interval: ' + interval, fontsize=10)
    
# %% Enviro Variables Stats and Plots
#try:
#    df.drop(['TC', 'FC', 'ENT'], axis=1, inplace=True)  # drop untransformed FIB
#except KeyError:
#    pass
#
#print('\n- - | Environmental Variable (EV) Statistics | - -\n')
#
#for f in ['ENT', 'FC']:
#    PCC = df.corr()['log' + f]  # Pearson correlations (PCC)
#    PCC_a = np.abs(PCC).sort_values(ascending=False)  # absolute value or PCC, sorted
#
#    print('log' + f + ' - Strong Linear Correlation(absolute values)\n')
#    strong_corr = PCC_a[PCC_a >= 0.68]
#    print(strong_corr)
#    print('\nlog' + f + ' - Weak Linear Correlation(absolute values)\n')
#    print(PCC_a[PCC_a < 0.36])
#    # Categories from Taylor 1990 - Interpretation of the Correlation Coefficient: A Basic Review
#    #   0 < |r| < 0.35 -> Poorly correlated
#    #   0.36 < |r| < 0.67 -> Moderately correlated
#    #   0.68 < |r| < 1.0 -> Strongly correlated
#
#    print('\n')
#
#print('Coefficients of Variation for EVs (\u03C3/\u03BC):')  # CV = sigma/mu
#print((df.describe().loc['std'] / np.abs(df.describe().loc['mean'])).sort_values(ascending=False))

#TODO - Plot logFIB vs. rad/tide/wind (stacked subplots)
# Plot rad/rad_1h/rad2_h vs. FIB (stacked subplots) --> Is there a lag?

plot_vars = ['logENT','tide']
if 'wspd' in df.columns:
    plot_vars += ['wspd']
elif 'wspd_coops' in df.columns:
    plot_vars += ['wspd_coops']
    
if 'rad' in df.columns:
    plot_vars += ['rad']

axes = df[plot_vars].plot(subplots=True, figsize=(10, 8), style = ['r.-','b.-','g.-','k.-'])
axes[0].axhline(np.log10(104),xmin=0,xmax=1, color = 'r')
#labels
axes[0].set_ylabel('logFIB')
axes[1].set_ylabel('water level (m)')
axes[2].set_ylabel('wind speed (m/s)')
if 'rad' in plot_vars:
    axes[3].set_ylabel('W/m^2')
plt.suptitle(set_name)

# %% Save
if save==1:
    save_folder = os.path.join(folder,'EDA',set_name)
    os.makedirs(save_folder, exist_ok=True)
    
    # Basic Stats
    df_stats.to_csv(os.path.join(save_folder, 'basic_stats_' + set_name + '.csv'), index=True)
    
    # Basic Stats
    df_corr.to_csv(os.path.join(save_folder, 'pearson_correlation_' + set_name + '.csv'), index=True)
    
    # FIB Figures
    plt.savefig(os.path.join(save_folder, 'vars_time_series_' + set_name + '.png'))
    
    # Boxplots
    plt.figure(1)
    plt.savefig(os.path.join(save_folder, 'FIB_boxplots_' + set_name + '.png'))
    
    # Histograms
    plt.figure(2)
    plt.savefig(os.path.join(save_folder, 'FIB_histograms_' + set_name + '.png'))
    
    # Boxplots
    plt.figure(3)
    plt.savefig(os.path.join(save_folder, 'FIB_time_series_' + set_name + '.png'))
    
    # Autocorrelations
    plt.figure(4)
    plt.savefig(os.path.join(save_folder, 'ENT_autocorr_' + set_name + '.png'))
    
    plt.figure(5)
    plt.savefig(os.path.join(save_folder, 'FC_autocorr_' + set_name + '.png'))
    
    
    
