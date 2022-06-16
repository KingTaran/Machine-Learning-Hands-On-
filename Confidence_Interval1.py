# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 15:58:14 2022

@author: taran
"""

#Q1

import pandas as pd

data = pd.read_csv("D:/data/Cars.csv", encoding = 'utf8')

data1 = pd.read_csv("D:/data/wc-at.csv", encoding = 'utf8')

data.loc[data['MPG'] > 38 , 'greater_than_38'] = 'True'

data.greater_than_38.count()

data.loc[data['MPG'] < 40 , 'less_than_40'] = 'True'

data.less_than_40.count()

subset_df = data[(data["MPG"] > 20) & (data["MPG"] < 50)]

subset_df.count()

#Q2

data['MPG'].hist()

data['MPG'].plot(kind = 'box')

data1['Waist'].hist()

data1['AT'].hist()

data1['Waist'].plot(kind = 'box')

data1['AT'].plot(kind = 'box')

#Q3

#90%= 1.645

#94%= 1.880

#60%= 0.253

from scipy import stats

from scipy.stats import norm

#z-score of 90% confidence interval

stats.norm.ppf(0.95)

#z-score of 94% confidence interval

stats.norm.ppf(.97)

#z-score of 60% confidence interval

stats.norm.ppf(.60)

#Q4

# +/- 2.059539 - 99%

# +/- 1.316345 - 96%

#  +/- 1.177716 - 95%


#Q5

#First we need to find T here that is -0.471.
#For probability calculations, the number of degrees of freedom is n - 1,
#so here you need the t-distribution with 17 degrees of freedom.

#The probability that t < - 0.471 with 17 degrees of freedom assuming the population mean is true,
#the t-value is less than the t-value obtained With 17 degrees of freedom and a t score of - 0.471,
#the probability of the bulbs lasting less than 260 days on average of 0.3218 assuming the mean life of the bulbs is 300 days.

#Q6

# Z value is calculated first that is Z=0.625

import scipy.stats as stats

#Calculation of the p-value for the standard normal distribution

1 - stats.norm.cdf(0.625)
stats.norm.sf(0.625)

# B) 0.2676


#Q7

#Mean = 38

#SD = 6

#Z score = (Value - Mean)/SD  

#Z score for 44  = (44 - 38)/6  = 1  =>  84.13 %  

#People above 44 age = 100 - 84.13 =  15.87%  = 63    out of 400

#Z score for 38  = (38 - 38)/6 = 0 => 50%

#Hence People between 38 & 44  age = 84.13 - 50  = 34.13 % = 137 out of 400

#Hence More employees at the processing center are older than 44 than between 38 and 44. is FALSE

#Z score for 30  = (30 - 38)/6 =  -1.33  =  9.15  %   = 36 out of 400

#Hence A training program for employees under the age of 30 at the center would be expected to attract about 36 employees - TRUE


#Q8

#Q9

# p(a<x<b) = 0.99 , mean =100 , standardDeviation = 20
#Z(0.5)
stats.norm.ppf(0.005)

#Z(99.5)
stats.norm.ppf(0.995) 

#Now Z=(x - mean)/std dev
#So, x = 20Z + 100
#Now putting values of Z and getting ans as 
# D) 48.5,151.5


#Q10

#For Company's Profit N(7+5,3^2 + 4^2)
#Net Profit N(12,5^2)

#Z=95%=1.96

#Range = (12 - 1.96 * 5, 12 + 1.96 * 5)

#Range = (2.2 , 22.8) = (Rs 99 M , Rs 1026M)

#5th percentile profit

#p = 0.05 so  Z = 1.64
 
#Hence Profit is 3.78 i.e 170.1M Rs

#Loss is when profit < 0

#The first division of company, thus have larger probability of making a loss in a given year.















