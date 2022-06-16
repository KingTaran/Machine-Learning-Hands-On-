# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 04:29:54 2022

@author: taran
"""
#Q1


# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

data = pd.read_csv("D:/data/calories_consumed.csv")

data.rename(columns = {'Weight gained (grams)':'wtgain','Calories Consumed' : 'calc'}, inplace = True)

data.describe()

import matplotlib.pyplot as plt # mostly used for visualization purposes 


plt.hist(data.calc) #histogram
plt.boxplot(data.calc) #boxplot


plt.hist(data.wtgain) #histogram
plt.boxplot(data.wtgain) #boxplot

# Scatter plot
plt.scatter(x = data['wtgain'], y = data['calc'], color = 'green') 

# correlation
np.corrcoef(data.wtgain, data.calc) 

#0.94 so good correlation

cov_output = np.cov(data.wtgain, data.calc)[0, 1]
cov_output

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('calc ~ wtgain', data = data).fit()
model.summary()

#From model summary we can see betas are 1577.2 and 2.13 so we can make a atraight line
#R square value is greater than 0.8 so strong correlation 
#Plus p is also less than 0.05


pred1 = model.predict(pd.DataFrame(data['wtgain']))

# Regression Line
plt.scatter(data.wtgain, data.calc)
plt.plot(data.wtgain, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = data.calc - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

#RMSE value is 232.8

######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(data['wtgain']), y = data['calc'], color = 'brown')
np.corrcoef(np.log(data.wtgain), data.wtgain) #correlation
# r = 0.95

model2 = smf.ols('calc ~ np.log(wtgain)', data = data).fit()
model2.summary()

#coefficents values are -1911.1244 and 774.1735 
#R^2=0.878

pred2 = model2.predict(pd.DataFrame(data['wtgain']))

# Regression Line
plt.scatter(np.log(data.wtgain), data.calc)
plt.plot(np.log(data.wtgain), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = data.calc - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

#RMSE = 253.55


#### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = data['wtgain'], y = np.log(data['calc']), color = 'orange')
np.corrcoef(data.wtgain, np.log(data.calc)) #correlation
#r= 0.89872528

model3 = smf.ols('np.log(calc) ~ wtgain', data = data).fit()
model3.summary()
#the coefficients values are 7.4068 and 0.0009
#R^2=0.808

pred3 = model3.predict(pd.DataFrame(data['wtgain']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(data.wtgain, np.log(data.calc))
plt.plot(data.wtgain, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = data.calc - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3

#RMSE = 272.4

#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(calc) ~ wtgain + I(wtgain*wtgain)', data = data).fit()
model4.summary()
#coefficient values are 7.2892,0.0017 and -7.689e-07
#R^2= 0.852

pred4 = model4.predict(pd.DataFrame(data))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = data.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = wcat.iloc[:, 1].values


plt.scatter(data.wtgain, np.log(data.calc))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = data.calc - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4

#RMSE = 240.82
# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

#Lowest RMSE is 232.85 of the SLR model


# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size = 0.2)

finalmodel = smf.ols('calc ~ wtgain', data = train).fit()
finalmodel.summary()
#Coefficients are 1503.5714,2.2258 
#R^2 =  0.909

# Predict on test data
pred_test_AT = finalmodel.predict(pd.DataFrame(test))


# Model Evaluation on Test data
test_res = test.calc - pred_test_AT
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse

#RMSE on test data is 284.5 

# Prediction on train data
pred_train_AT = finalmodel.predict(pd.DataFrame(train))


# Model Evaluation on train data
train_res = train.calc - pred_train_AT
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse


#RMSE on training data is 223.9



#Q2

# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

df = pd.read_csv("D:/data/delivery_time.csv")

df.rename(columns = {'Delivery Time':'DT','Sorting Time' : 'ST'}, inplace = True)


df.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

plt.bar(height = df.DT, x = np.arange(1, 21, 1))
plt.hist(df.DT) #histogram
plt.boxplot(df.DT) #boxplot

plt.bar(height = df.ST, x = np.arange(1, 21, 1))
plt.hist(df.ST) #histogram
plt.boxplot(df.ST) #boxplot

# Scatter plot
plt.scatter(x = df['ST'], y = df['DT'], color = 'green') 

# correlation
np.corrcoef(df.ST,df.DT) 
#Value is 0.83 so good

# Covariance

cov_output = np.cov(df.ST, df.DT)[0, 1]
cov_output

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('DT ~ ST', data = df).fit()
model.summary()
#Coefficients are 6.5827 and 1.6490
#R^2 = 0.682 not good

pred1 = model.predict(pd.DataFrame(df['ST']))

# Regression Line
plt.scatter(df.ST, df.DT)
plt.plot(df.ST, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = df.DT - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

#RMSE1 = 2.8

######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(df['ST']), y = df['DT'], color = 'brown')
np.corrcoef(np.log(df.ST), df.DT) #correlation
# r = 0.83

model2 = smf.ols('DT ~ np.log(ST)', data = df).fit()
model2.summary()
#Coefficients are 1.1597 and 9.0434
#R^2 = 0.695

pred2 = model2.predict(pd.DataFrame(df['ST']))

# Regression Line
plt.scatter(np.log(df.ST), df.DT)
plt.plot(np.log(df.ST), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = df.DT - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

#RMSE2 = 2.73

#### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = df['ST'], y = np.log(df['DT']), color = 'orange')
np.corrcoef(df.ST, np.log(df.DT)) #correlation
#r= 0.84

model3 = smf.ols('np.log(DT) ~ ST', data = df).fit()
model3.summary()
#Coefficients are 2.1214 and 0.1056
#R^2 = 0.711

pred3 = model3.predict(pd.DataFrame(df['ST']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(df.ST, np.log(df.DT))
plt.plot(df.ST, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = df.DT - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3

#RMSE3 = 2.94

#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(DT) ~ ST + I(ST*ST)', data = df).fit()
model4.summary()
#Coefficients are 1.6997,0.2659 and -0.0128 
#R^2 are 0.765

pred4 = model4.predict(pd.DataFrame(df))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = df.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = wcat.iloc[:, 1].values


plt.scatter(df.ST, np.log(df.DT))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = df.DT - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4

#RMSE4 = 2.8


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

###################
# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size = 0.2)

#Here we have logm model as best as it has minimum error

finalmodel = smf.ols('DT ~ np.log(ST)', data = train).fit()
finalmodel.summary()
#Coefficients are 2.3037 and 8.1025 
#R^2 = 0.676 (Not as good)

# Predict on test data
pred_test_AT = finalmodel.predict(pd.DataFrame(test))


# Model Evaluation on Test data
test_res = test.DT - pred_test_AT
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse

#RMSE = 3.6

# Prediction on train data
pred_train_AT = finalmodel.predict(pd.DataFrame(train))


# Model Evaluation on train data
train_res = train.DT - pred_train_AT
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

#RMSE = 2.5


#Q3

# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

data = pd.read_csv("D:/data/emp_data.csv")

data.rename(columns = {'Salary_hike':'SH','Churn_out_rate' : 'COR'}, inplace = True)

data.describe()

import matplotlib.pyplot as plt # mostly used for visualization purposes 


plt.hist(data.COR) #histogram
plt.boxplot(data.COR) #boxplot


plt.hist(data.SH) #histogram
plt.boxplot(data.SH) #boxplot

# Scatter plot
plt.scatter(x = data['SH'], y = data['COR'], color = 'green') 

# correlation
np.corrcoef(data.SH, data.COR) 

# -0.91172162 so good correlation

cov_output = np.cov(data.SH, data.COR)[0, 1]
cov_output

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('COR ~ SH', data = data).fit()
model.summary()

#From model summary we can see betas are 244.3649 and  -0.1015  so we can make a atraight line
#R square value is greater than 0.8 so strong correlation i.e 0.831
#Plus p is also less than 0.05


pred1 = model.predict(pd.DataFrame(data['SH']))

# Regression Line
plt.scatter(data.SH, data.COR)
plt.plot(data.SH, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = data.COR - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

#RMSE value is 3.99

######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(data['SH']), y = data['COR'], color = 'brown')
np.corrcoef(np.log(data.SH), data.COR) #correlation
# r = -0.92

model2 = smf.ols('COR ~ np.log(SH)', data = data).fit()
model2.summary()

#coefficents values are 1381.4562 and -176.1097
#R^2=0.849

pred2 = model2.predict(pd.DataFrame(data['SH']))

# Regression Line
plt.scatter(np.log(data.SH), data.COR)
plt.plot(np.log(data.SH), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = data.COR - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

#RMSE = 3.78


#### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = data['SH'], y = np.log(data['COR']), color = 'orange')
np.corrcoef(data.SH, np.log(data.COR)) #correlation
#r= -0.93463607

model3 = smf.ols('np.log(COR) ~ SH', data = data).fit()
model3.summary()
#the coefficients values are 6.6383 and -0.0014 
#R^2=0.874

pred3 = model3.predict(pd.DataFrame(data['SH']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(data.SH, np.log(data.COR))
plt.plot(data.SH, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = data.COR - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3

#RMSE = 3.54

#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(COR) ~ SH + I(SH*SH)', data = data).fit()
model4.summary()
#coefficient values are 23.1762,-0.0207 and 5.605e-06
#R^2= 0.984

pred4 = model4.predict(pd.DataFrame(data))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = data.iloc[:, 1:2].values
X_poly = poly_reg.fit_transform(X)
# y = wcat.iloc[:, 1].values


plt.scatter(data.SH, np.log(data.COR))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = data.COR - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4

#RMSE = 1.33
# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

#Lowest RMSE is 1.326790 of the Poly model

###################
# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size = 0.2)

finalmodel = smf.ols('np.log(COR) ~ SH + I(SH*SH)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_AT = np.exp(test_pred)
pred_test_AT

# Model Evaluation on Test data
test_res = test.COR - pred_test_AT
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse
#RMSE  = 3.4011744491688516

# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_AT = np.exp(train_pred)
pred_train_AT

# Model Evaluation on train data
train_res = train.COR - pred_train_AT
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

#RMSE = 1.1992570434696765



#Q4

# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

df = pd.read_csv("D:/data/Salary_Data.csv")

df.rename(columns = {'YearsExperience':'YE','Salary' : 'S'}, inplace = True)


df.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 


plt.hist(df.S) #histogram
plt.boxplot(df.S) #boxplot


plt.hist(df.YE) #histogram
plt.boxplot(df.YE) #boxplot

# Scatter plot
plt.scatter(x = df['YE'], y = df['S'], color = 'green') 

# correlation
np.corrcoef(df.YE , df.S) 
# r = 0.978

# Covariance

cov_output = np.cov(df.YE , df.S)[0, 1]
cov_output

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('S ~ YE', data = df).fit()
model.summary()
#Coefficients are 2.579e+04 and 9449.9623
#R^2 =  0.957 very good

pred1 = model.predict(pd.DataFrame(df['YE']))

# Regression Line
plt.scatter(df.YE, df.S)
plt.plot(df.YE, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = df.S - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

#RMSE1 = 5592.043608760662


######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(df['YE']), y = df['S'], color = 'brown')
np.corrcoef(np.log(df.YE), df.S) #correlation
# r = 0.92406108

model2 = smf.ols('S ~ np.log(YE)', data = df).fit()
model2.summary()
#Coefficients are 1.493e+04 and 4.058e+04
#R^2 = 0.854

pred2 = model2.predict(pd.DataFrame(df['YE']))

# Regression Line
plt.scatter(np.log(df.YE), df.S)
plt.plot(np.log(df.YE), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = df.S - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

#RMSE2 = 10302.893706228304

#### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = df['YE'], y = np.log(df['S']), color = 'orange')
np.corrcoef(df.YE, np.log(df.S)) #correlation
#r= 0.96538444

model3 = smf.ols('np.log(S) ~ YE', data = df).fit()
model3.summary()
#Coefficients are 10.5074 and 0.1255
#R^2 = 0.932

pred3 = model3.predict(pd.DataFrame(df['YE']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(df.YE, np.log(df.S))
plt.plot(df.YE, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = df.S - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3

#RMSE3 = 7213.235076620233


#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(S) ~ YE + I(YE*YE)', data = df).fit()
model4.summary()
#Coefficients are 10.3369,0.2024 and -0.0066 
#R^2 are  0.949

pred4 = model4.predict(pd.DataFrame(df))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = df.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = wcat.iloc[:, 1].values


plt.scatter(df.YE, np.log(df.S))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = df.S - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4

#RMSE4 = 5391.081582693625


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

###################
# The best model
#Here it is the Poly Model

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size = 0.2)

#Here we have poly model as best as it has minimum error

finalmodel = smf.ols('np.log(S) ~ YE + I(YE*YE)', data = train).fit()
finalmodel.summary()
#Coefficients are 10.3452 and 0.2009 and -0.0065
#R^2 = 0.941 

# Predict on test data
pred_test_AT = finalmodel.predict(pd.DataFrame(test))


# Model Evaluation on Test data
test_res = test.S - pred_test_AT
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse

#RMSE = 91295.25325269996

# Prediction on train data
pred_train_AT = finalmodel.predict(pd.DataFrame(train))


# Model Evaluation on train data
train_res = train.S - pred_train_AT
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

#RMSE = 77735.71544884978


#Q5


# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

df = pd.read_csv("D:/data/SAT_GPA.csv")

df.rename(columns = {'SAT_Scores':'S'}, inplace = True)


df.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 


plt.hist(df.S) #histogram
plt.boxplot(df.S) #boxplot


plt.hist(df.GPA) #histogram
plt.boxplot(df.GPA) #boxplot

# Scatter plot
plt.scatter(x = df['S'], y = df['GPA'], color = 'green') 

# correlation
np.corrcoef(df.S , df.GPA) 
# r = 0.978

# Covariance

cov_output = np.cov(df.S , df.GPA)[0, 1]
cov_output

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('GPA ~ S', data = df).fit()
model.summary()
#Coefficients are 2.4029 and 0.0009
#R^2 =   0.086 poor

pred1 = model.predict(pd.DataFrame(df['S']))

# Regression Line
plt.scatter(df.S, df.GPA)
plt.plot(df.S, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = df.GPA - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

#RMSE1 = 0.5159457227723684


######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(df['S']), y = df['GPA'], color = 'brown')
np.corrcoef(np.log(df.S), df.GPA) #correlation
# r = 0.27771976

model2 = smf.ols('GPA ~ np.log(S)', data = df).fit()
model2.summary()
#Coefficients are 0.4796 and 0.3868
#R^2 = 0.077

pred2 = model2.predict(pd.DataFrame(df['S']))

# Regression Line
plt.scatter(np.log(df.S), df.GPA)
plt.plot(np.log(df.S), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = df.GPA - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

#RMSE2 = 0.518490410108067

#### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = df['S'], y = np.log(df['GPA']), color = 'orange')
np.corrcoef(df.S, np.log(df.GPA)) #correlation
#r= 0.29408419

model3 = smf.ols('np.log(GPA) ~ S', data = df).fit()
model3.summary()
#Coefficients are 0.8727 and 0.0003
#R^2 =  0.086

pred3 = model3.predict(pd.DataFrame(df['S']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(df.S, np.log(df.GPA))
plt.plot(df.S, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = df.GPA - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3

#RMSE3 = 0.5175875893834133


#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(GPA) ~ S + I(S*S)', data = df).fit()
model4.summary()

#Here in the poly model value of p is greater than 0.05 so not acceptable




# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3])}
table_rmse = pd.DataFrame(data)
table_rmse

###################
# The best model
#Here it is the SLR Model

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size = 0.2)

#Here we have SLR model as best as it has minimum error

finalmodel = smf.ols('GPA ~ S', data = train).fit()
finalmodel.summary()
#Coefficients are  2.3714 and 0.0010 
#R^2 = 0.118

# Predict on test data
pred_test_AT = finalmodel.predict(pd.DataFrame(test))


# Model Evaluation on Test data
test_res = test.GPA - pred_test_AT
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse

#RMSE = 0.5761031020000748

# Prediction on train data
pred_train_AT = finalmodel.predict(pd.DataFrame(train))


# Model Evaluation on train data
train_res = train.GPA - pred_train_AT
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

#RMSE = 0.5018100621623087














































