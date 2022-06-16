# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 06:07:10 2021

@author: taran
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("D:/Data/boston_data.csv")
df.dtypes
df.head()
df.describe()

#Dropping Unneccessary Columns
df = df.drop(['zn', 'chas'],axis=1)
#Checking Null Values
df.isnull().sum()

#Plotting boxplots to see if there are any outliers in our data (considering data betwen 25th and 75th percentile as non outlier)
fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(15, 5))
ax = ax.flatten()
index = 0
for i in df.columns:
  sns.boxplot(y=i, data=df, ax=ax[index])
  index +=1
plt.tight_layout(pad=0.4)
plt.show()

#Checking percentage/amount of Outliers
for i in df.columns:
  df.sort_values(by=i, ascending=True, na_position='last')
  q1, q3 = np.nanpercentile(df[i], [25,75])
  iqr = q3-q1
  lower_bound = q1-(1.5*iqr)
  upper_bound = q3+(1.5*iqr)
  outlier_data = df[i][(df[i] < lower_bound) | (df[i] > upper_bound)] #creating a series of outlier data
  perc = (outlier_data.count()/df[i].count())*100
  print('Outliers in %s is %.2f%% with count %.f' %(i, perc, outlier_data.count()))
#For Replacing outliers with mean
for i in df.columns:
  df.sort_values(by=i, ascending=True, na_position='last')
  q1, q3 = np.nanpercentile(df[i], [25,75])
  iqr = q3-q1
  lower_bound = q1-(1.5*iqr)
  upper_bound = q3+(1.5*iqr)
  mean = df[i].mean()
  if i != 'MEDV':
    df.loc[df[i] < lower_bound, [i]] = mean
    df.loc[df[i] > upper_bound, [i]] = mean
  else:
    df.loc[df[i] < lower_bound, [i]] = mean
    df.loc[df[i] > upper_bound, [i]] = 50
    
df.describe
   