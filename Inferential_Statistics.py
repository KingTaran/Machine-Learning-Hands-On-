# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 19:20:04 2021

@author: taran
"""

import pandas as pd

import matplotlib.pyplot as plot

df = pd.read_excel(r"D:\data\Assignment_module02 (1).xlsx")

df.describe()

df.shape

#Inference on comparing points with score and Weigh)
df.plot(x="Points", y=["Score", "Weigh"], kind="bar")

df.var()

#If we see there variances weight column should only be taken 
#for future purposes..