# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 18:36:50 2021

@author: taran
"""

#1a

import pandas as pd

df = pd.read_csv(r"D:\Data\Q1_a.csv")

df.speed.skew()
#We infer that it is having a negative skewness
#Thus it will have left long tail(Mean<Median)
df.dist.skew()
#We infer that it is having a positive skewness
#Thus it will have right long tail(Mean>Median)

df.speed.kurt()
#We infer that it is having a negative kurtosis
#Thus it will have short tails and increased spread
#and wide area in graph no peaks
df.dist.kurt()
#We infer that it is having a positive kurtosis
#Thus it will have long tails and decreased spread
#and no wide area in graph and sharp peaks

#1b

df = pd.read_csv(r"D:\Data\Q1_b.csv")

df.SP.skew()
#We infer that it is having a positive skewness
#Thus it will have right long tail(Mean>Median)

df.WT.skew()
#We infer that it is having a negative skewness
#Thus it will have left long tail(Mean<Median)

df.SP.kurt()
#We infer that it is having a positive kurtosis
#Thus it will have long tails and decreased spread
#and no wide area in graph and sharp peaks

df.WT.kurt()
#We infer that it is having a close to zero kurtosis
#Perfectly Normally Distributed

#2

#Histogram and Boxplot analysis
#Firstly it is a Right SKewed Histogram
#Mode is closer to the left of the graph and smaller than
#both median and mode
#Thus there are some outliers greater than mode
#Data is more Spread out


#3
import pandas as pd
 
df = pd.read_excel(r"D:\data\Q3.xlsx")

df.Marks.mean()

df.Marks.median()

df.Marks.var()

df.Marks.std()

import matplotlib.pyplot as plt

plt.boxplot(df.Marks)
#We can infer that mean is very close to median
#Thus it is most probably a symmetrical distribution 
#2 outliers are present-49,56

#5
#When mean=median then,
#distribution is symmetric and it has 0 skewness


#6
#When Mean>Median then,
#Positive Skewed

#7
#When median>mean then,
#Negative Skewness
 
#8
#Positive Kurtosis Data indicates
#Long Tails
#Sharp Peaks
#Less Spread
#Called Leptykurtic


#9
#Negative Kurtosis Data indicates
#Short Tails
#Large Spread
#Wide Divided Data
#Called Platykurtic

#10
#Asymmetric Distribution of Data
#Left Skewed Data
#IQR=18-10=8











