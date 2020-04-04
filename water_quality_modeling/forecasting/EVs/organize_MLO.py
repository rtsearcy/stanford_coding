# getMet_NCDC.py - Process raw met data from Hopkins MLO
# RTS - 3/31/2020 


import pandas as pd
import numpy as np
import os

# Inputs
folder = '/Users/rtsearcy/Box/water_quality_modeling/data/met/MLO/raw/'
station = 'bird_rock' # 'west_beach', 'bird_rock'

#df = pd.DataFrame()
for file in os.listdir(os.path.join(folder,station)):
    if not file.endswith('.txt'):
        continue
    print(file)
    
    try:
        ff = os.path.join(folder,station,file)
        df_temp = pd.read_csv(ff)
        if ' year' in df_temp.columns:
            df_temp.rename(columns={' year': 'year'}, inplace=True)
        if 'dayofyear' in df_temp.columns:
            df_temp.rename(columns={"dayofyear": "day"}, inplace=True)
        B = pd.to_datetime(df_temp['day'],format="%j")
        df_temp['month'] = B.dt.month
        df_temp['day'] = B.dt.day
            #df_temp['day'] = pd.to_datetime(df['day'], unit='D', origin='julian')
        
        #df_temp['date'] = pd.to_datetime((df_temp['year'].astype(int).astype(str) + df_temp['day'].astype(int).astype(str)), format="%Y%j")
        #df.drop(['year','day'], inplace=True)
        if 'time' in df_temp.columns:
            df_temp.rename(columns={"time": "hourmin"}, inplace=True)
            #df_temp['hourmin'] = [((4-len(str(x)))*'0')+str(x) for x in df_temp['hourmin']]
        #df_temp['hourmin'] = pd.to_datetime(df_temp['hourmin'], format='%H%m').dt.time
        
        #df_temp.set_index([' year','dayofyear','hourmin'], inplace=True)
        df_temp.to_csv(ff.replace('.txt','.csv'), index=False)
        #A = [x for x in ['year','day','hourmin'] if x in df_temp.columns[0:3]]
#        if len(A) == 3:
#            #df = df.append(df_temp)
#        else: 
#            print('   Could not read txt file properly')
    except:
        print('   Could not read txt file properly')
        continue
    
# %% Combine
#if 'time' in df.columns:
#    df['hourmin'] = df['hourmin'].combine_first(df['time'])
#    df.drop(labels=['time'], axis=1, inplace=True)
#        

#sd = '2007-12-31'  # start date, in YYYY-MM-DD format (account for previous day)
#ed = '2019-10-31'  # end date, account for 8hr UTC shift

# Save to file
#outfile = a.replace(' ', '_') + '_Met_Variables_' + sd_new.replace('-', '') + '_' + ed_new.replace('-', '') + '.csv'
#df_vars.index.rename('date', inplace=True)
#df_vars.to_csv(os.path.join(outfolder, outfile))  # Save vars file
#print('Meteorological variables saved to: ' + outfile)
