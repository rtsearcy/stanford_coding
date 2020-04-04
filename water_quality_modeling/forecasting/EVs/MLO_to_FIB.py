# getMet_NCDC.py - Process raw met data from Hopkins MLO
# RTS - 3/31/2020 


import pandas as pd
import numpy as np
import os

# Inputs
mlo_station = 'bird_rock'  # 'west_beach' , 'bird_rock'
mlo_folder = '/Users/rtsearcy/Box/water_quality_modeling/data/met/MLO/raw/' + mlo_station # 'west_beach', 'bird_rock'

fib_folder = '/Users/rtsearcy/Box/water_quality_modeling/thfs/traditional_nowcast/variables'
fib_file = 'Lovers_Point_variables_fib_2000_2020.csv'

#fib_folder = '/Users/rtsearcy/Box/water_quality_modeling/thfs/traditional_nowcast/modeling_datasets/'
#fib_file = 'LP_trad_modeling_dataset_20000101_20200301.csv'

df_fib = pd.read_csv(os.path.join(fib_folder,fib_file))

#if 'sample_time' in df_fib.columns:
#    df_fib['dt'] = pd.to_datetime(df_fib['dt'] + ' ' + df_fib['sample_time'], format='%Y-%m-%d %H:%M')
#else:
#    df_fib['dt'] = pd.to_datetime(df_fib['dt'])
df_fib['dt'] = pd.to_datetime(df_fib['dt'])
df_fib.set_index('dt', inplace=True)

#df_mlo = pd.DataFrame(index=df_fib.index)
df_mlo = pd.DataFrame()
if mlo_station == 'bird_rock':
    pref = 'BR'
else:
    pref = ''

old_file = ''
for i in df_fib.index:
    try: 
        print(i)
        year = str(i.year)
        month = '0'*(2-len(str(i.month))) + str(i.month)
        day = i.day
        hour = i.hour
        minute = i.minute
        if minute == 0:
            #minute == '00'
            hm = hour*100
        else:
            hm = round(int(str(hour) + str(minute)),-1)  # NEED TO ROUND HOUR AND MIN
            if int(str(hm)[-2:]) == 60:
                hm += 40
                
        file = pref + year + '-' + month + '.csv'
        if file not in os.listdir(mlo_folder):
            print('  ' + file + ' not found!')
        
        if file != old_file:
            df_temp = pd.read_csv(os.path.join(mlo_folder, file))
        else:
            df_temp = df_temp
        print(file)
        print(str(hm))
        
        if df_temp['month'].dtype == 'O':
            month = str(month)
        else:
            month = int(month)
            
        if df_temp['day'].dtype == 'O':
            day = str(month)
        else:
            day = int(day)
        
        if df_temp['hourmin'].dtype == 'O':
            hm = str(hm)
        else:
            hm = int(hm)
            
        obs = df_temp[(df_temp['month'] == month) & 
                      (df_temp['day'] == day) & 
                      (df_temp['hourmin'] == hm)]  # Row of MLO observations  
        obs.index = [i]
        #print(str(hm))
        df_mlo = df_mlo.append(obs)
        
    except Exception as exc:
        print('   There was a problem: %s' % exc)
        obs = pd.DataFrame(index = [i])
        df_mlo = df_mlo.append(obs)
        continue
            
# %%
# Save to file
df_mlo.index.rename('dt',inplace=True)
outfile = fib_file.replace('.csv', '_mlo_data.csv')
df_mlo.to_csv(os.path.join(fib_folder, outfile))  # Save vars file
#print('Meteorological variables saved to: ' + outfile)
