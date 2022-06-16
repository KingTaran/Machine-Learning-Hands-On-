# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 00:28:18 2022

@author: taran
"""
#Q1


import pandas as pd

from scipy import stats


data = pd.read_csv("D:\data\Cutlets.csv")
data
data.drop(data.index[35:51], inplace=True)
data.columns = "Unit_A", "Unit_B"

# Normal Q-Q plot
import pylab

# Checking Whether data is normally distributed
stats.probplot(data.Unit_A, dist="norm", plot=pylab)
stats.probplot(data.Unit_B, dist="norm", plot=pylab)

# Normality Test
stats.shapiro(data.Unit_A) # Shapiro Test
stats.shapiro(data.Unit_B)

#Thus we can infer here that both the columns or data is normally distributed
#As their p value is greater than 0.05

############### Paired T-Test ##############
# A univariate test that tests for a significant difference between 2 related variables.

import seaborn as sns
sns.boxplot(data=data)

# Assuming the external Conditions are same for both the samples
# Paired T-Test
ttest, pval = stats.ttest_rel(data['Unit_A'], data['Unit_B'] )
print(pval)

#As pval which we are getting is greater than 0.05 so we will go with null hypothesis 

#Q2

import pandas as pd

from scipy import stats
import scipy

data = pd.read_csv("D:\data\lab_tat_updated.csv")
data

# Normal Q-Q plot
import pylab

# Checking Whether data is normally distributed
stats.probplot(data.Laboratory_1, dist="norm", plot=pylab)
stats.probplot(data.Laboratory_2, dist="norm", plot=pylab)
stats.probplot(data.Laboratory_3, dist="norm", plot=pylab)
stats.probplot(data.Laboratory_4, dist="norm", plot=pylab)

# Normality Test
stats.shapiro(data.Laboratory_1) # Shapiro Test
stats.shapiro(data.Laboratory_2)
stats.shapiro(data.Laboratory_3)
stats.shapiro(data.Laboratory_4)

#Thus we can infer here that both the columns or data is normally distributed
#As pvalue is greater than 0.05

# All 4 suppliers are being checked for variances
scipy.stats.levene(data.Laboratory_1,data.Laboratory_2, data.Laboratory_3, data.Laboratory_4)
#P value is greater than 0.05 so they have all in all equal variance

# One - Way Anova
F, p = stats.f_oneway(data.Laboratory_1, data.Laboratory_2, data.Laboratory_3,data.Laboratory_4)

# p value
p  


#Q3

############### Chi-Square Test ################
import pandas as pd
import scipy

data = pd.read_csv("D:/data/BuyerRatio.csv")
data

df_table=data.iloc[:,1:6]
df_table

df_table.values

val=stats.chi2_contingency(df_table)

no_of_rows=len(df_table.iloc[0:2,0])
no_of_columns=len(df_table.iloc[0,0:4])
degree_of_f=(no_of_rows-1)*(no_of_columns-1)
print('Degree of Freedom=',degree_of_f)

Expected_value=val[3]

from scipy.stats import chi2
chi_square=sum([(o-e)**2/e for o,e in zip(df_table.values,Expected_value)])
chi_square_statestic=chi_square[0]+chi_square[1]
chi_square_statestic

critical_value=chi2.ppf(0.95,3)
critical_value

if chi_square_statestic >= critical_value:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')

pvalue=1-chi2.cdf(chi_square_statestic,3)
pvalue

if pvalue <= 0.05:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')

df_table = pd.crosstab(data['East'],data['Observed Values'])
df_table


#Q4


import pandas as pd

from scipy import stats
import scipy

data = pd.read_csv("D:\data\CustomerOrderform.csv")
data

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

data["Phillippines"] = le.fit_transform(data["Phillippines"])
data["Indonesia"] = le.fit_transform(data["Indonesia"])
data["Malta"] = le.fit_transform(data["Malta"])
data["India"] = le.fit_transform(data["Phillippines"])

df_table = data
df_table


df_table.values

val=stats.chi2_contingency(df_table)

no_of_rows=len(df_table.iloc[0:2,0])
no_of_columns=len(df_table.iloc[0,0:4])
degree_of_f=(no_of_rows-1)*(no_of_columns-1)
print('Degree of Freedom=',degree_of_f)

Expected_value=val[3]

from scipy.stats import chi2
chi_square=sum([(o-e)**2/e for o,e in zip(df_table.values,Expected_value)])
chi_square_statestic=chi_square[0]+chi_square[1]
chi_square_statestic

critical_value=chi2.ppf(0.95,3)
critical_value

if chi_square_statestic >= critical_value:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')

pvalue=1-chi2.cdf(chi_square_statestic,3)
pvalue

if pvalue <= 0.05:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')

#Q5

import numpy as np

two_prop_test = pd.read_csv("D:/data/Fantaloons.csv")

from statsmodels.stats.proportion import proportions_ztest

tab1 = two_prop_test.Weekdays.value_counts()
tab1
tab2 = two_prop_test.Weekend.value_counts()
tab2

# crosstable table
pd.crosstab(two_prop_test.Weekdays, two_prop_test.Weekend)

count = np.array([58, 152])
nobs = np.array([480, 740])

stats, pval = proportions_ztest(count, nobs, alternative = 'two-sided') 
print(pval) # Pvalue 0.000

stats, pval = proportions_ztest(count, nobs, alternative = 'larger')
print(pval)  # Pvalue 0.999  























































