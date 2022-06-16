# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 00:31:00 2022

@author: taran
"""
#Q1

# Multilinear Regression with Regularization using L1 and L2 norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# loading the data
car = pd.read_csv("D:/data/Computer_Data (1).csv")
car = car.drop(['Unnamed: 0'], axis=1)

from sklearn import preprocessing
 
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
 
# Encode labels in column 'species'.
car['cd']= label_encoder.fit_transform(car['cd'])
car['multi']= label_encoder.fit_transform(car['multi'])
car['premium']= label_encoder.fit_transform(car['premium'])

car.columns

# Correlation matrix 
a = car.corr()
a

# EDA
a1 = car.describe()

# Sctter plot and histogram between variables
sns.pairplot(car) # sp-hp, wt-vol multicolinearity issue

# Preparing the model on train data 
model_train = smf.ols("price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend", data = car).fit()
model_train.summary()
#R^2 = 0.776

# Prediction
pred = model_train.predict(car)
# Error
resid  = pred - car.price
# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse
#275.07266261641524


# To overcome the issues, LASSO and RIDGE regression are used
################
###LASSO MODEL###
from sklearn.linear_model import Lasso
help(Lasso)

lasso = Lasso(alpha = 0.13, normalize = True)

lasso.fit(car.iloc[:, 1:], car.price)

# Coefficient values for all independent variables#
lasso.coef_
#Here we interpret that ads and hd columns are having coefficients close to 0

lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(car.columns[1:]))

lasso.alpha

pred_lasso = lasso.predict(car.iloc[:, 1:])

# Adjusted r-square
lasso.score(car.iloc[:, 1:], car.price)
# 0.7716633012279069

# RMSE
np.sqrt(np.mean((pred_lasso - car.price)**2))
#277.50064677426957

### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
help(Ridge)
rm = Ridge(alpha = 0.4, normalize = True)

rm.fit(car.iloc[:, 1:], car.price)

# Coefficients values for all the independent vairbales
rm.coef_
#Here we interpret that ads and hd columns are having coefficients close to 0

rm.intercept_

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(car.columns[1:]))

rm.alpha

pred_rm = rm.predict(car.iloc[:, 1:])

# Adjusted r-square
rm.score(car.iloc[:, 1:], car.price)
#0.6787983799363293

# RMSE
np.sqrt(np.mean((pred_rm - car.price)**2))
#329.1281315375888

### ELASTIC NET REGRESSION ###
from sklearn.linear_model import ElasticNet 
help(ElasticNet)
enet = ElasticNet(alpha = 0.4)

enet.fit(car.iloc[:, 1:], car.price) 

# Coefficients values for all the independent vairbales
enet.coef_
enet.intercept_

plt.bar(height = pd.Series(enet.coef_), x = pd.Series(car.columns[1:]))

enet.alpha

pred_enet = enet.predict(car.iloc[:, 1:])

# Adjusted r-square
enet.score(car.iloc[:, 1:], car.price)
#0.741154820628258

# RMSE
np.sqrt(np.mean((pred_enet - car.price)**2))
#295.458290962069



#Q2

# Multilinear Regression with Regularization using L1 and L2 norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# loading the data
df1 = pd.read_csv("D:/data/ToyotaCorolla.csv")

df = df1[['Price', 'Age_08_04', 'KM', 'HP', 'cc', 'Doors', 'Gears', 'Quarterly_Tax', 'Weight']]
df.describe()

# Correlation matrix 
a = df.corr()
a

# EDA
a1 = df.describe()

# Sctter plot and histogram between variables
sns.pairplot(df) # sp-hp, wt-vol multicolinearity issue

# Preparing the model on train data 
model_train = smf.ols("Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight", data = df).fit()
model_train.summary()
#R^2 =  0.864

# Prediction
pred = model_train.predict(df)
# Error
resid  = pred - df.Price
# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse
#1338.2584236201496

# To overcome the issues, LASSO and RIDGE regression are used
################
###LASSO MODEL###
from sklearn.linear_model import Lasso
help(Lasso)

lasso = Lasso(alpha = 0.13, normalize = True)

lasso.fit(df.iloc[:, 1:], df.Price)

# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(df.columns[1:]))
#Here we intercept KM,cc,doors,quaterly_Tax close to 0 intercept values

lasso.alpha

pred_lasso = lasso.predict(df.iloc[:, 1:])

# Adjusted r-square
lasso.score(df.iloc[:, 1:], df.Price)
#0.8637502278874631

# RMSE
np.sqrt(np.mean((pred_lasso - df.Price)**2))
#1338.3199065958215


### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
help(Ridge)
rm = Ridge(alpha = 0.4, normalize = True)

rm.fit(df.iloc[:, 1:], df.Price)

# Coefficients values for all the independent vairbales
rm.coef_
rm.intercept_

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(df.columns[1:]))
#Here km,cc,quarterly tax has coefficients value close to  0
rm.alpha

pred_rm = rm.predict(df.iloc[:, 1:])

# Adjusted r-square
rm.score(df.iloc[:, 1:], df.Price)
#0.8306009245941786

# RMSE
np.sqrt(np.mean((pred_rm - df.Price)**2))
#1492.2705331587526

### ELASTIC NET REGRESSION ###
from sklearn.linear_model import ElasticNet 
help(ElasticNet)
enet = ElasticNet(alpha = 0.4)

enet.fit(df.iloc[:, 1:], df.Price) 

# Coefficients values for all the independent vairbales
enet.coef_
enet.intercept_

plt.bar(height = pd.Series(enet.coef_), x = pd.Series(df.columns[1:]))
#Here km and cc have value of coefficients close to 0 
enet.alpha

pred_enet = enet.predict(df.iloc[:, 1:])

# Adjusted r-square
enet.score(df.iloc[:, 1:], df.Price)
#0.8631173040588218

# RMSE
np.sqrt(np.mean((pred_enet - df.Price)**2))

#1341.42476730272

####################

#Here lasso regression is best

#Q3

# Multilinear Regression with Regularization using L1 and L2 norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# loading the data
data = pd.read_csv("D:/data/Life_expectencey_LR.csv")

data = data.drop(['Country','Year','Status'], axis=1)

a = data.corr()
a

# EDA
a1 = data.describe()

# Sctter plot and histogram between variables
sns.pairplot(data) 

# Preparing the model on train data 
model_train = smf.ols("Life_expectancy ~ Adult_Mortality + infant_deaths + Alcohol + percentage_expenditure + Hepatitis_B + Measles + BMI + under_five_deaths + Polio + Total_expenditure + Diphtheria + HIV_AIDS + GDP + Population + thinness + thinness_yr + Income_composition + Schooling", data = data).fit()
model_train.summary()
#R^2 = 0.835

# Prediction
pred = model_train.predict(data)
# Error
resid  = pred - data.Life_expectancy
# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse
#3.5750455750827412

# To overcome the issues, LASSO and RIDGE regression are used
################
###LASSO MODEL###
from sklearn.linear_model import Lasso
help(Lasso)

lasso = Lasso(alpha = 0.13, normalize = True)

#For removing NAN values
data = data[data['Life_expectancy'].notna()]
data = data[data['Adult_Mortality'].notna()]
data = data[data['infant_deaths'].notna()]
data = data[data['Alcohol'].notna()]
data = data[data['percentage_expenditure'].notna()]
data = data[data['Hepatitis_B'].notna()]
data = data[data['Measles'].notna()]
data = data[data['BMI'].notna()]
data = data[data['under_five_deaths'].notna()]
data = data[data['Polio'].notna()]
data = data[data['Total_expenditure'].notna()]
data = data[data['Diphtheria'].notna()]
data = data[data['HIV_AIDS'].notna()]
data = data[data['GDP'].notna()]
data = data[data['Population'].notna()]
data = data[data['thinness'].notna()]
data = data[data['thinness_yr'].notna()]
data = data[data['Income_composition'].notna()]
data = data[data['Schooling'].notna()]

lasso.fit(data.iloc[:, 1:], data.Life_expectancy)

# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(data.columns[1:]))
#Here we see only schooling and income composition are most important
lasso.alpha

pred_lasso = lasso.predict(data.iloc[:, 1:])

# Adjusted r-square
lasso.score(data.iloc[:, 1:], data.Life_expectancy)
#0.2199259591432765

# RMSE
np.sqrt(np.mean((pred_lasso -  data.Life_expectancy)**2))
#7.767166092080508

### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
help(Ridge)
rm = Ridge(alpha = 0.4, normalize = True)

rm.fit(data.iloc[:, 1:], data.Life_expectancy)

# Coefficients values for all the independent vairbales
rm.coef_
rm.intercept_

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(data.columns[1:]))

rm.alpha

pred_rm = rm.predict(data.iloc[:, 1:])

# Adjusted r-square
rm.score(data.iloc[:, 1:], data.Life_expectancy)
# 0.8094282484500525

# RMSE
np.sqrt(np.mean((pred_rm - data.Life_expectancy)**2))
#3.839051534875406

### ELASTIC NET REGRESSION ###
from sklearn.linear_model import ElasticNet 
help(ElasticNet)
enet = ElasticNet(alpha = 0.4)

enet.fit(data.iloc[:, 1:], data.Life_expectancy) 

# Coefficients values for all the independent vairbales
enet.coef_
enet.intercept_

plt.bar(height = pd.Series(enet.coef_), x = pd.Series(data.columns[1:]))

enet.alpha

pred_enet = enet.predict(data.iloc[:, 1:])

# Adjusted r-square
enet.score(data.iloc[:, 1:], data.Life_expectancy)
#0.8199219492431894

# RMSE
np.sqrt(np.mean((pred_enet - data.Life_expectancy)**2))
#3.731857652987986

####################

#Here Elactic net is the best











