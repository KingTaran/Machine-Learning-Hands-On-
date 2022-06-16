# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 01:00:26 2022

@author: taran
"""

#Q1

### Multinomial Regression ####
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mode = pd.read_csv("D:\data\mdata.csv")
mode.head(10)

mode.describe()

mode.drop(mode.columns[[0, 1, 2 , 4]], axis = 1, inplace = True)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

mode.ses = le.fit_transform(mode['ses'])
mode.prog = le.fit_transform(mode['prog'])
mode.honors = le.fit_transform(mode['honors'])

# Rearrange the order of the variables
mode = mode.iloc[:, [6, 0,1,2, 3, 4,5]]


mode.corr()
#For seeing the correaltion

train, test = train_test_split(mode, test_size = 0.25)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(train.iloc[:, 1:], train.iloc[:, 0])
help(LogisticRegression)

test_predict = model.predict(test.iloc[:, 1:]) # Test predictions

# Test accuracy 
accuracy_score(test.iloc[:,0], test_predict)
#0.5

train_predict = model.predict(train.iloc[:, 1:]) # Train predictions 
# Train accuracy 
accuracy_score(train.iloc[:,0], train_predict) 
#0.56

#Q2

### Multinomial Regression ####
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("D:\data\loan.csv")

#Here we see there are a lot of columns given so we slice down the importatnt columns
#And join them and the make the required data frame and then preprocess it
#So,that we can perform operation on it.
c1=df.iloc[:, 2:8] 
c2=df.iloc[:, 12:14] 
c4=df.iloc[:,16:17]
c3 = c1.join(c2)
c4 = c3.join(c4)

mode = c4.iloc[:, [8,0,1,2,3,4,5,6,7]]

from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
mode["loan_status"] = lb.fit_transform(mode["loan_status"])
mode["term"] = lb.fit_transform(mode["term"])
mode["home_ownership"] = lb.fit_transform(mode["home_ownership"])

mode['int_rate'] = (mode['int_rate']).str.replace('%' , ' ')
mode['int_rate']=mode['int_rate'].astype()

# Correlation values between each independent features
mode.corr()

train, test = train_test_split(mode, test_size = 0.2)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(train.iloc[:, 1:], train.iloc[:, 0])
help(LogisticRegression)

test_predict = model.predict(test.iloc[:, 1:]) # Test predictions

# Test accuracy 
accuracy_score(test.iloc[:,0], test_predict)
#0.8348439073514602

train_predict = model.predict(train.iloc[:, 1:]) # Train predictions 
# Train accuracy 
accuracy_score(train.iloc[:,0], train_predict) 
#0.8282189280206464



















