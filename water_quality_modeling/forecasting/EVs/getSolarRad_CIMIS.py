# getSolarRad_CIMIS.py
# RTS 12082017
# RTS 03152018 Update

# Grabs bulk solar radiation data [W/m^2] from CIMIS stations around CA, stores in csv.
# http://www.cimis.water.ca.gov/

# Note: CIMIS API often times out calls quickly at certain times of day,
# so this code may need to be rerun for specific stations

import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import time
import os

# Inputs #
out_folder = '/Users/rtsearcy/Box/water_quality_modeling/data/rad'

start_date = '2012-01-01'  # in YYYY-MM-DD format, build in previous day
end_date = '2020-03-31'

api_key = '6216de17-d2ad-4f0f-b3d5-65ec3638c7c4'
#api_key = '25b6d300-a073-46e1-9b35-6628451c99f4'
units = 'M'  # 'E' English, 'M' Metric
item = 'day-sol-rad-avg'
item = 'hly-sol-rad'

stations = {
#    'Santa Rosa': 83,  # ~10 miles from beach
#    #'Pescadero': 253, # Active only since 2017
#    'Santa Cruz': 104,  # De Lavega station
#    'Watsonville West': 209,
#    'Castroville': 19,
    'Monterey': 193,  # Pacific Grove station
#    'Carmel': 210,
#    'San Luis Obispo': 160,  # Close to Morro Bay
#    'Nipomo': 202,
#    'Lompoc': 231,  # Active 2010
#    'Santa Barbara': 107,
#    'Santa Monica': 99,
#    'Long Beach': 174,
#    'Irvine': 75,
#    #'San Clemente':, 241, Active only since 2016
#    'Torrey Pines': 173
}

print('CIMIS Solar Radiation Data\nDirectory: ' + out_folder)
for s in stations:
    print('\nGrabbing ' + item + ' data for ' + s)
    df_out = pd.DataFrame()
    sd = datetime.strptime(start_date, '%Y-%m-%d')
    ed = sd + timedelta(days=70) #timedelta(days=4*365)  # CIMIS limits calls to 1750 records
    break_val = 0
    c = 1

    while break_val == 0:
        if ed > datetime.strptime(end_date, '%Y-%m-%d'):
            ed = datetime.strptime(end_date, '%Y-%m-%d')
            break_val = 1

        print('  Finding data from ' + sd.strftime('%Y-%m-%d') + ' to ' + ed.strftime('%Y-%m-%d'))
        time.sleep(1)

        url = 'http://et.water.ca.gov/api/data?' \
              + 'appKey=' + api_key \
              + '&targets=' + str(stations[s]) \
              + '&startDate=' + sd.strftime('%Y-%m-%d') \
              + '&endDate=' + ed.strftime('%Y-%m-%d') \
              + '&dataItems=' + item \
              + '&unitOfMeasure=' + units  # + '&unitOfMeasure=\'' + units + '\'' quotations seems to work for now

        web = requests.get(url)
        try:
            web.raise_for_status()
            d = json.loads(web.text)
            d = d['Data']['Providers'][0]['Records']
            if item == 'day-sol-rad-avg':    
                #d = d['Data']['Providers'][0]['Records']
                df = pd.DataFrame(d)
                df['rad_avg'] = [s['Value'] for s in df['DaySolRadAvg']]  # Average solar radiation for day of
                df['rad1'] = df['rad_avg'].shift(1, freq='D')  # Previous day
                df = df[['Date', 'rad1']]
            elif item == 'hly-sol-rad':
                df = pd.DataFrame(d)
                df =  df[['Date','Hour','HlySolRad']]
                df['HlySolRad'] =  [x['Value'] for x in df['HlySolRad']]
            df_out = df_out.append(df, ignore_index=True)
            sd = ed + timedelta(days=1)  # new sd due to call limit
            ed = sd + timedelta(days=70) #timedelta(days=4 * 365)

        except Exception as exc:
            print('   There was a problem: %s' % exc)
            time.sleep(5)
            continue

    df_out['date'] = pd.to_datetime(df_out['Date'])
    df_out.drop('Date', axis=1, inplace=True)
    df_out.set_index('date', inplace=True)
    
    
#    df_out['rad_avg'] = [s['Value'] for s in df_out['DaySolRadAvg']]  # Average solar radiation for day of
#    df_out['rad1'] = df_out['rad_avg'].shift(1, freq='D')  # Previous day
#    df_out = df_out['rad1']
    
    # Save to file
    outfile = os.path.join(out_folder, s.replace(' ', '_') + '_CIMIS_' + item.replace('-','_') + '_'
                           + start_date.replace('-', '') + '_' + end_date.replace('-', '') + '.csv')
    df_out.to_csv(outfile, header=True)  # PD Series
    missing = df_out.isnull().sum()
    print(str(missing) + ' days of missing data')
    print('Data for ' + s + ' saved.\n')
