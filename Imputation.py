# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 09:26:52 2021

@author: taran
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# load the dataset
# use modified ethnic dataset
df = pd.read_csv('D:/Data/claimants.csv') # for doing modifications

# check for count of NA'sin each column
df.isna().sum()

mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df["CLMSEX"] = pd.DataFrame(mode_imputer.fit_transform(df[["CLMSEX"]]))
df["SEATBELT"] = pd.DataFrame(mode_imputer.fit_transform(df[["SEATBELT"]]))
df["CLMINSUR"] = pd.DataFrame(mode_imputer.fit_transform(df[["CLMINSUR"]]))
df.isnull().sum()  # all Sex,Seatbelt,Insurance records replaced by mode

median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
df["CLMAGE"] = pd.DataFrame(median_imputer.fit_transform(df[["CLMAGE"]]))
df["CLMAGE"].isna().sum() 