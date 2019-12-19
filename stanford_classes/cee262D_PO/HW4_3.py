# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_file = '/Users/rtsearcy/coding/stanford_classes/cee262D_PO/rho_data.csv'

df = pd.read_csv(data_file)
df.set_index('Depth',inplace=True)
df = 1000 + 1000*df

#%% Problem 1

g = 9.81
pres = pd.DataFrame(index = df.columns)

for i in df.columns:
    p = df[i][0.0]
    for j in df.index[1:]:
        p += df[i][j] * g * (j - df.index[j-1])
    pres.loc[i] = p