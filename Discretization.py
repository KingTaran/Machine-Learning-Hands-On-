# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 09:41:44 2021

@author: taran
"""


import pandas as pd
data = pd.read_csv("D:/Data/iris.csv")
data.head()
data.describe()

data = data.rename(columns={'Sepal.Length': 'SLength', 'Sepal.Width': 'SWidth','Petal.Length': 'PLength','Petal.Width': 'PWidth'})

data['SLength'] = pd.cut(data['SLength'], bins=[min(data.SLength) - 1, 
                                                  data.SLength.mean(), max(data.SLength)], labels=["Low","High"])
data['SWidth'] = pd.cut(data['SWidth'], bins=[min(data.SWidth) - 1, 
                                                  data.SWidth.mean(), max(data.SWidth)], labels=["Low","High"])
data['PLength'] = pd.cut(data['PLength'], bins=[min(data.PLength) - 1, 
                                                  data.PLength.mean(), max(data.PLength)], labels=["Low","High"])
data['PWidth'] = pd.cut(data['PWidth'], bins=[min(data.PWidth) - 1, 
                                                  data.PWidth.mean(), max(data.PWidth)], labels=["Low","High"])

data.head()
data.SLength.value_counts()
data.SWidth.value_counts()
data.PLength.value_counts()
data.PWidth.value_counts()