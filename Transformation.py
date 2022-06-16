# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 17:36:33 2021

@author: taran
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("D:/Data/calories_consumed.csv")

df.head()

df=df.rename(columns={'Weight gained (grams)': 'wg','Calories Consumed': 'cc' })

df.head()

df.corr()

#Made a graph to infer towards linear regression
plt.scatter(x=df, y=df, color='red')
plt.xlabel("Calories Consumed")
plt.ylabel("Weight gained (grams)")

df.describe()
