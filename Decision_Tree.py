# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 19:46:20 2022

@author: taran
"""
#q1

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv("D:/data/Company_Data.csv")

data.isnull().sum()
data.dropna()
data.columns


# Converting into binary
lb = LabelEncoder()
data["ShelveLoc"] = lb.fit_transform(data["ShelveLoc"])
data["Urban"] = lb.fit_transform(data["Urban"])
data["US"] = lb.fit_transform(data["US"])

data['Sales'].unique()
data['Sales'].value_counts()
colnames = list(data.columns)

data['Sales'] = pd.cut(data['Sales'], bins=[min(data.Sales) - 1, 
                                                  data.Sales.mean(), max(data.Sales)], labels=["Low","High"])


predictors = colnames[1:15]
target = colnames[0]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])

# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

#42+48=90
#90/120=0.75

np.mean(preds == test[target]) # Test Data Accuracy 
#Overfit Model

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy
#1

#DT-Regression

# Input and Output Split
predictors = data.loc[:, data.columns!="Sales"]
type(predictors)

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
data["Sales"] = lb.fit_transform(data["Sales"])
target = data["Sales"]
type(target)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)

# Train the Regression DT
from sklearn import tree
regtree = tree.DecisionTreeRegressor(max_depth = 3)
regtree.fit(x_train, y_train)

# Prediction
test_pred = regtree.predict(x_test)
train_pred = regtree.predict(x_train)

# Measuring accuracy
from sklearn.metrics import mean_squared_error, r2_score

# Error on test dataset
mean_squared_error(y_test, test_pred)
#0.16764984964226146
r2_score(y_test, test_pred)
#0.31875616653303274

# Error on train dataset
mean_squared_error(y_train, train_pred)
#0.1618784330678912

r2_score(y_train, train_pred)
#0.35157441925551314


# Plot the DT
#dot_data = tree.export_graphviz(regtree, out_file=None)
#from IPython.display import Image
#import pydotplus
#graph = pydotplus.graph_from_dot_data(dot_data)
#Image(graph.create_png())


# Pruning the Tree
# Minimum observations at the internal node approach
regtree2 = tree.DecisionTreeRegressor(min_samples_split = 3)
regtree2.fit(x_train, y_train)

# Prediction
test_pred2 = regtree2.predict(x_test)
train_pred2 = regtree2.predict(x_train)

# Error on test dataset
mean_squared_error(y_test, test_pred2)
#0.278125
r2_score(y_test, test_pred2)
#-0.13015873015873014

# Error on train dataset
mean_squared_error(y_train, train_pred2)
#0.0015625
r2_score(y_train, train_pred2)
#0.9937411985604757

###########
## Minimum observations at the leaf node approach
regtree3 = tree.DecisionTreeRegressor(min_samples_leaf = 3)
regtree3.fit(x_train, y_train)

# Prediction
test_pred3 = regtree3.predict(x_test)
train_pred3 = regtree3.predict(x_train)

# measure of error on test dataset
mean_squared_error(y_test, test_pred3)
#0.20068055555555553
r2_score(y_test, test_pred3)

# measure of error on train dataset
mean_squared_error(y_train, train_pred3)
#0.20068055555555553
r2_score(y_train, train_pred3)
#0.18453615520282196



#q2


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("D:/data/Diabetes.csv")

data.isnull().sum()
data.dropna()
data.columns


# Converting into binary
lb = LabelEncoder()
data["Outcome"] = lb.fit_transform(data["Outcome"])

data['Outcome'].unique()
data['Outcome'].value_counts()
colnames = list(data.columns)

predictors = colnames[0:8]
target = colnames[8]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.2)

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])

# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])
#83+24=107
#107/154=0.69

np.mean(preds == test[target]) # Test Data Accuracy 
#Overfit Model

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy


#DT Regression

# Input and Output Split
predictors = data.loc[:, data.columns!="Outcome"]
type(predictors)

target = data["Outcome"]
type(target)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)

# Train the Regression DT
from sklearn import tree
regtree = tree.DecisionTreeRegressor(max_depth = 3)
regtree.fit(x_train, y_train)

# Prediction
test_pred = regtree.predict(x_test)
train_pred = regtree.predict(x_train)

# Measuring accuracy
from sklearn.metrics import mean_squared_error, r2_score

# Error on test dataset
mean_squared_error(y_test, test_pred)
# 0.15562875215194477
r2_score(y_test, test_pred)
#0.2660784477956806
# Error on train dataset
mean_squared_error(y_train, train_pred)
#0.1519283364818788
r2_score(y_train, train_pred)
#0.3405365947022857

# Plot the DT
#dot_data = tree.export_graphviz(regtree, out_file=None)
#from IPython.display import Image
#import pydotplus
#graph = pydotplus.graph_from_dot_data(dot_data)
#Image(graph.create_png())


# Pruning the Tree
# Minimum observations at the internal node approach
regtree2 = tree.DecisionTreeRegressor(min_samples_split = 3)
regtree2.fit(x_train, y_train)

# Prediction
test_pred2 = regtree2.predict(x_test)
train_pred2 = regtree2.predict(x_train)

# Error on test dataset
mean_squared_error(y_test, test_pred2)
#0.23376623376623376
r2_score(y_test, test_pred2)
#-0.10240604493935179

# Error on train dataset
mean_squared_error(y_train, train_pred2)
#0.003257328990228013
r2_score(y_train, train_pred2)
#0.9858611677201709

###########
## Minimum observations at the leaf node approach
regtree3 = tree.DecisionTreeRegressor(min_samples_leaf = 3)
regtree3.fit(x_train, y_train)

# Prediction
test_pred3 = regtree3.predict(x_test)
train_pred3 = regtree3.predict(x_train)

# measure of error on test dataset
mean_squared_error(y_test, test_pred3)
#0.16281024531024532
r2_score(y_test, test_pred3)
#0.23221161706546467

# measure of error on train dataset
mean_squared_error(y_train, train_pred3)
#0.048099891422366994
r2_score(y_train, train_pred3)
#0.7912165766678565



#q3

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("D:/data/Fraud_check.csv")

data.isnull().sum()
data.dropna()
data.columns

#Creating dummy vairables for ['Undergrad','Marital.Status','Urban'] dropping first dummy variable
data=pd.get_dummies(data,columns=['Undergrad','Marital.Status','Urban'], drop_first=True)

data["TaxInc"] = pd.cut(data["Taxable.Income"], bins = [10002,30000,99620], labels = ["Risky", "Good"])

#After creation of new col. TaxInc also made its dummies var concating right side of df
data = pd.get_dummies(data,columns = ["TaxInc"],drop_first=True)

data = data.drop("Marital.Status_Single",axis=1)

# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
# Normalized data frame (considering the numerical part of data)
data_norm = norm_func(data.iloc[:,1:])

colnames = list(data.columns)

predictors = colnames[:6]
target = colnames[6]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.2)

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])


# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target]) # Test Data Accuracy 
#0.9916666666666667

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy
#1

#DT Regression

# Input and Output Split
predictors = data_norm.loc[:, data_norm.columns!="TaxInc_Good"]
type(predictors)

target = data["TaxInc_Good"]
type(target)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)

# Train the Regression DT
from sklearn import tree
regtree = tree.DecisionTreeRegressor(max_depth = 3)
regtree.fit(x_train, y_train)

# Prediction
test_pred = regtree.predict(x_test)
train_pred = regtree.predict(x_train)

# Measuring accuracy
from sklearn.metrics import mean_squared_error, r2_score

# Error on test dataset
mean_squared_error(y_test, test_pred)
# 0.174026479378094
r2_score(y_test, test_pred)
#-0.05515002233454913


# Error on train dataset
mean_squared_error(y_train, train_pred)
#0.15574903541014604
r2_score(y_train, train_pred)
#0.04863390443814375

# Plot the DT
#dot_data = tree.export_graphviz(regtree, out_file=None)
#from IPython.display import Image
#import pydotplus
#graph = pydotplus.graph_from_dot_data(dot_data)
#Image(graph.create_png())


# Pruning the Tree
# Minimum observations at the internal node approach
regtree2 = tree.DecisionTreeRegressor(min_samples_split = 3)
regtree2.fit(x_train, y_train)

# Prediction
test_pred2 = regtree2.predict(x_test)
train_pred2 = regtree2.predict(x_train)

# Error on test dataset
mean_squared_error(y_test, test_pred2)
#0.3145833333333333
r2_score(y_test, test_pred2)
#-0.9073684210526318

# Error on train dataset
mean_squared_error(y_train, train_pred2)
#0.013541666666666667
r2_score(y_train, train_pred2)
#0.9172830668893661

###########
## Minimum observations at the leaf node approach
regtree3 = tree.DecisionTreeRegressor(min_samples_leaf = 3)
regtree3.fit(x_train, y_train)

# Prediction
test_pred3 = regtree3.predict(x_test)
train_pred3 = regtree3.predict(x_train)

# measure of error on test dataset
mean_squared_error(y_test, test_pred3)
#0.23987962962962964
r2_score(y_test, test_pred3)
#-0.45442807017543885

# measure of error on train dataset
mean_squared_error(y_train, train_pred3)
#0.07645833333333334
r2_score(y_train, train_pred3)
#0.5329674699753438





















