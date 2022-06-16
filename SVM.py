# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 03:42:25 2022

@author: taran
"""

#Q1


import numpy as np


# Loading the data set using pandas as data frame format 
import pandas as pd
train = pd.read_csv("D:\\data\\SalaryData_Train1.csv")
train = train.drop(columns=['relationship','workclass','education','age','maritalstatus','occupation','race','sex','native'],)
test = pd.read_csv("D:\\data\\SalaryData_Test1.csv")
test = test.drop(columns=['relationship','workclass','education','age','maritalstatus','occupation','race','sex','native'],)


# fucntion to convert categorical features using Label Encoder
def convert_categorical(df):
    categorical_feature_mask = df.dtypes==object
    categorical_cols = df.columns[categorical_feature_mask].tolist()

    # import labelencoder
    from sklearn.preprocessing import LabelEncoder
    # instantiate labelencoder object
    le = LabelEncoder()
    # apply le on categorical feature columns
    df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))

convert_categorical(train)
convert_categorical(test)

from sklearn.svm import SVC

train_X = train.iloc[:, :-1]
train_y = train.iloc[:, -1]
test_X  = test.iloc[:, :-1]
test_y  = test.iloc[:, -1]



# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X, train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y)
#0.7964143426294821












#Q2
import pandas as pd
import numpy as np

df = pd.read_csv("D:\\data\\forestfires.csv")
df.describe()

df.drop(df.columns[[0, 1]], axis = 1, inplace = True)

df.size_category.unique()

# fucntion to convert categorical features using Label Encoder
def convert_categorical(df):
    categorical_feature_mask = df.dtypes==object
    categorical_cols = df.columns[categorical_feature_mask].tolist()

    # import labelencoder
    from sklearn.preprocessing import LabelEncoder
    # instantiate labelencoder object
    le = LabelEncoder()
    # apply le on categorical feature columns
    df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))

convert_categorical(df)



from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

train,test = train_test_split(df, test_size = 0.20)


train_X = train.iloc[:, :-1]
train_y = train.iloc[:, -1]
test_X  = test.iloc[:, :-1]
test_y  = test.iloc[:, -1]


# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X, train_y)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear == test_y)
#0.9807692307692307

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X, train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y)
#0.7692307692307693