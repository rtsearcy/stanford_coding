# beach_init.py
# RTS - March 2018

# 1.Creates directory for beaches on modeling list (if not already created)
#  In directory - variables folder, models folder, and saves raw FIB sample csv.
# 2.Creates FIB variable dataset (save in var folder, date and time will be used to match enviro vars)
#  Using raw FIB, create following variables:
#     date
#     sample_time
#     FIB
#     FIB1 (previous sample)
#     FIB_exc
#     FIB1_exc
#     logFIB (log_10 transform)
#     logFIB1
#     weekend1
#     laborday (if summer)

import pandas as pd
from numpy import log10
import os
import shutil
from datetime import date


def labor_day(x):  # x must be a datetime.date; dates up through 2020
    lbd = [date(2002, 9, 2),
           date(2003, 9, 1),
           date(2004, 9, 6),
           date(2005, 9, 5),
           date(2006, 9, 4),
           date(2007, 9, 3),
           date(2008, 9, 1),
           date(2009, 9, 7),
           date(2010, 9, 6),
           date(2011, 9, 5),
           date(2012, 9, 3),
           date(2013, 9, 2),
           date(2014, 9, 1),
           date(2015, 9, 7),
           date(2016, 9, 5),
           date(2017, 9, 4),
           date(2018, 9, 3),
           date(2019, 9, 2),
           date(2020, 9, 7),
           date(2021, 9, 6),
           date(2022, 9, 5)]

    year = x.year
    short = [f for f in lbd if f.year in [year, year+1]]
    if short[0] <= x < short[1]:
        result = 1
    else:
        result = 0
    return result


# Inputs #
base_folder = '/Users/rtsearcy/data/water_quality_modeling/thfs/preliminary_analysis/raw_data/FIB'  # directory with all raw FIB files
fib_file = os.path.join(base_folder, 'FIB_samples_LP_2001_2019.csv')

beach = 'Lovers Point'  # all - all beaches in init file; list of beach names
skip_check = 1  # If 1, skip if directory is already created; if 0, replace directory, saving to old directory

season = 'All'  # Summer, Winter , All
sd = '20011001'  # Start date (will parse out depending on season
ed = '20191031'  # end date

# FIB Variables
fib = ['TC', 'FC', 'ENT']
thresholds = {'TC': 10000, 'FC': 400, 'ENT': 104}
df_raw = pd.read_csv(fib_file, encoding='latin1')
df_raw['dt'] = pd.to_datetime(df_raw['dt'])
df_raw.set_index('dt', inplace=True)
df_raw.sort_index(inplace=True)  # ascending

df_vars = df_raw.copy()
df_vars = df_vars[['sample_time'] + fib]
for f in fib:
    df_vars[f + '1'] = df_vars[f].dropna().shift(1)  # previous sample, skipping any missed samples in dataset
    df_vars[f + '_exc'] = (df_vars[f] > thresholds[f]).astype(int)  # exceeds threshold? (binary)
    df_vars[f + '1_exc'] = (df_vars[f + '1'] > thresholds[f]).astype(int)
    # previous day exceeds threshold? (binary)
    df_vars['log' + f] = round(log10(df_vars[f]), 5)
    df_vars['log' + f + '1'] = round(log10(df_vars[f + '1']), 5)

var_order = ['sample_time'] + fib + [f + '1' for f in fib] + [f + '_exc' for f in fib] \
    + [f + '1_exc' for f in fib] + ['log' + f for f in fib] + ['log' + f + '1'for f in fib]
df_vars = df_vars[var_order]
df_vars['weekend1'] = ((df_vars.index.weekday == 0) | (df_vars.index.weekday == 6) |
                       (df_vars.index.weekday == 7)).astype(int)
# Was yesterday a weekend (Fr/Sat/Sun)? (binary) Monday = 0, Sunday = 7

# Account for time range and season
df_vars = df_vars[sd:ed]
if season == 'Summer':
    df_vars = df_vars[(df_vars.index.month >= 4) & (df_vars.index.month < 11)]
    df_vars['laborday'] = [labor_day(x) for x in df_vars.index.date]  # Binary - has Labor Day passed?
elif season == 'Winter':
    df_vars = df_vars[(df_vars.index.month <= 3) | (df_vars.index.month >= 11)]

# Save variables
print(' Variables (Season: ' + season + ', Sample Range: ' + str(df_vars.index.year[0]) + ' to '
      + str(df_vars.index.year[-1]) + ')')
print('   Number of Samples: ' + str(len(df_vars)))
var_file = os.path.join(base_folder, beach.replace(' ', '_') + '_variables_fib.csv')
df_vars.to_csv(var_file)
print(' Saved to : ' + var_file + '\n')
