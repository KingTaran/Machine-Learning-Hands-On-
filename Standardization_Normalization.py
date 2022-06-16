# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 13:28:07 2021

@author: taran
"""
#Standardization
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
d = pd.read_csv("D:/Data/Seeds_data.csv")
d.drop(['Type'], axis = 1, inplace = True)

a = d.describe()
# Initialise the Scaler
scaler = StandardScaler()
# To scale data
df = scaler.fit_transform(d)
# Convert the array back to a dataframe
dataset = pd.DataFrame(df)
res = dataset.describe()

#Normalization
df = pd.read_csv("D:/Data/Seeds_data.csv")
df.columns
df.drop(['Type'], axis = 1, inplace = True)

a1 = df.describe()

# get dummies
df = pd.get_dummies(df, drop_first = True)

### Normalization function - Custom Function
# Range converts to: 0 to 1
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

df_norm = norm_func(df)
b = df_norm.describe()


