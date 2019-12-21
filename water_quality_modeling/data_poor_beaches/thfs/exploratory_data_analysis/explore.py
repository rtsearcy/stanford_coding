# -*- coding: utf-8 -*-
# %% explore.py - perusing the data
# RTS - 10.26.2019

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# %% Import Data
folder = '/Users/rtsearcy/Documents/Stanford/Projects/Targeted High Frequency Sampling/preliminary_analysis/raw_data'
file = 'modeling_dataset_10min_april2013.csv'
file = 'modeling_dataset_10min_april2016.csv'
#file = 'modeling_dataset_10min_april2018_siteM.csv'
print('Dataset: ' + file)

df = pd.read_csv(os.path.join(folder, file), parse_dates=['dt'], index_col=['dt'])

# %% FIB Statistics and Plots
print('Number of Samples: ' + str(len(df)))
print('Sampling Interval: ' + str(df.index[1] - df.index[0]) + ' minutes')

print('\n- - | FIB Statistics | - -\n')
print(df[['TC', 'FC', 'ENT']].describe())
print('\nExceedances:')
print('  TC: ' + str(sum(df['TC'] > 10000)))
print('  FC: ' + str(sum(df['FC'] > 400)))
print('  ENT: ' + str(sum(df['ENT'] > 104)))

# TODO - Plot logFIB vs. time
# Label x axis with hour/minute
ax = df[['logENT','logFC']].plot(title=file[:-4] + '\n(horizontal lines indicate FIB SSS)', style =['b','orange'])
ax.set_xlabel('')
ax.set_ylabel('logFIB')
ax.axhline(np.log10(104),xmin=0,xmax=1, color = 'b')
ax.axhline(np.log10(400),xmin=0,xmax=1, color = 'orange')

# %% Enviro Variables Stats and Plots
try:
    df.drop(['TC', 'FC', 'ENT'], axis=1, inplace=True)  # drop untransfored FIB
except KeyError:
    pass

print('\n- - | Environmental Variable (EV) Statistics | - -\n')

for f in ['ENT', 'FC']:
    PCC = df.corr()['log' + f]  # Pearson correlations (PCC)
    PCC_a = np.abs(PCC).sort_values(ascending=False)  # absolute value or PCC, sorted

    print('log' + f + ' - Strong Linear Correlation(absolute values)\n')
    strong_corr = PCC_a[PCC_a >= 0.68]
    print(strong_corr)
    print('\nlog' + f + ' - Weak Linear Correlation(absolute values)\n')
    print(PCC_a[PCC_a < 0.36])
    # Categories from Taylor 1990 - Interpretation of the Correlation Coefficient: A Basic Review
    #   0 < |r| < 0.35 -> Poorly correlated
    #   0.36 < |r| < 0.67 -> Moderately correlated
    #   0.68 < |r| < 1.0 -> Strongly correlated

    print('\n')

print('Coefficients of Variation for EVs (\u03C3/\u03BC):')  # CV = sigma/mu
print((df.describe().loc['std'] / np.abs(df.describe().loc['mean'])).sort_values(ascending=False))

#TODO - Plot logFIB vs. rad/tide/wind (stacked subplots)
# Plot rad/rad_1h/rad2_h vs. FIB (stacked subplots) --> Is there a lag?

plot_vars = ['logENT','tide']
if 'wspd' in df.columns:
    plot_vars += ['wspd']
elif 'wspd_coops' in df.columns:
    plot_vars += ['wspd_coops']
    
if 'rad' in df.columns:
    plot_vars += ['rad']

axes = df[plot_vars].plot(subplots=True, figsize=(10, 8), style = ['k','b','g','orange'])
axes[0].axhline(np.log10(104),xmin=0,xmax=1, color = 'r')
#labels
axes[0].set_ylabel('logFIB')
axes[1].set_ylabel('water level (m)')
axes[2].set_ylabel('wind speed (m/s)')
if 'rad' in plot_vars:
    axes[3].set_ylabel('W/m^2')



