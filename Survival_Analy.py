# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 22:57:43 2022

@author: taran
"""

#Q1

#pip install lifelines
#import lifelines

import pandas as pd
# Loading the the survival un-employment data
df = pd.read_csv("D:\data\Patient.csv")
df.head()
df.describe()

df = df.drop(['Scenario'], axis = 1)
df = df.drop(['PatientID'], axis = 1)

# Spell is referring to time 
T = df.Followup

from lifelines import KaplanMeierFitter

# Initiating the KaplanMeierFitter model
kmf = KaplanMeierFitter()

# Fitting KaplanMeierFitter model on Time and Events for death 
kmf.fit(T, event_observed=df.Eventtype)

# Time-line estimations plot 
kmf.plot()

#Q2


import pandas as pd
# Loading the the survival un-employment data
df = pd.read_excel("D:\data\ECG_Surv.xlsx")
df.head()
df.describe()

# Spell is referring to time 
T = df.survival_time_hr

from lifelines import KaplanMeierFitter

# Initiating the KaplanMeierFitter model
kmf = KaplanMeierFitter()

# Fitting KaplanMeierFitter model on Time and Events for death 
kmf.fit(T, event_observed=df.alive)

# Time-line estimations plot 
kmf.plot()

# Over Multiple groups 
# For each group, here group is ui
df.group.value_counts()

# Applying KaplanMeierFitter model on Time and Events for the group "1"
kmf.fit(T[df.group==1], df.alive[df.group==1], label='1')
ax = kmf.plot()

# Applying KaplanMeierFitter model on Time and Events for the group "0"
kmf.fit(T[df.group==2], df.alive[df.group==2], label='2')
kmf.plot(ax=ax)

# Applying KaplanMeierFitter model on Time and Events for the group "1"
kmf.fit(T[df.group==3], df.alive[df.group==3], label='3')
ax = kmf.plot()

#hence group 3 have least time

#likewise we can make graphs for any factor and see the response



















