# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 11:29:56 2021

@author: taran
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# we use animal category dataset
df = pd.read_csv("D:/Data/animal_category.csv")

df.columns # column names
df.shape # will give u shape of the dataframe

df_new = pd.get_dummies(df)
df_new_1 = pd.get_dummies(df, drop_first = True)

