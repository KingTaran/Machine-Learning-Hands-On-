# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 04:46:11 2022

@author: taran
"""

import pandas as pd
import numpy as np
# import seaborn as sb
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

#Importing Data
df = pd.read_csv("D:/data/Affairs.csv", sep = ",")

df1 = pd.get_dummies(df["naffairs"])
print(df1)

df_two = pd.concat((df1, df), axis=1)
df_two = df_two.drop(["naffairs"], axis=1)

c1 = df_two.drop('Unnamed: 0', axis = 1)

c1.head(11)
c1.describe()
c1.isna().sum()

mapping = {c1.columns[0]: 'nf0', c1.columns[1]: 'nf1' ,c1.columns[2]: 'nf2' ,c1.columns[3]: 'nf3' ,c1.columns[4]: 'nf7' ,c1.columns[5]: 'nf12'}
su = c1.rename(columns=mapping)

#As here we are talking about affair their or not so we take nf0 column and
#see the probability of nffairs to be 0

#First model on measures of happiness
logit_model1 = sm.logit('nf0 ~ vryunhap +unhap +avgmarr + hapavg +vryhap ', data = su).fit()

#summary
logit_model1.summary2() # for AIC
#AIC value = 646.1504

logit_model1.summary()

pred = logit_model1.predict(su.iloc[ :, 1: ])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(su.nf0, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold
#0.7525773195876289

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)
#Area under the curve is 0.653415

# filling all the cells with zeroes
su["pred"] = np.zeros(601)
# taking threshold value and above the prob value will be treated as correct value 
su.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(su["pred"], su["nf0"])
classification
#Accuracy = 0.52

#Second model on measures of relation
logit_model2 = sm.logit('nf0 ~ antirel + notrel + slghtrel + smerel +vryrel ', data = su).fit()

#summary
logit_model2.summary2() # for AIC
#AIC value = 666.5004

logit_model2.summary()

pred = logit_model2.predict(su.iloc[ :, 1: ])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(su.nf0, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold
#0.8142857142857143

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)
#Area under the curve is 0.611641

# filling all the cells with zeroes
su["pred"] = np.zeros(601)
# taking threshold value and above the prob value will be treated as correct value 
su.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(su["pred"], su["nf0"])
classification


#Third model on measures of years of marriage
logit_model3 = sm.logit('nf0 ~ yrsmarr1 + yrsmarr2 + yrsmarr3 + yrsmarr4+ yrsmarr5+ yrsmarr6', data = su).fit()

#summary
logit_model3.summary2() # for AIC
#AIC value = 668.1473

logit_model3.summary()

pred = logit_model3.predict(su.iloc[ :, 1: ])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(su.nf0, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold
#0.8636363636363636

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)
#Area under the curve is 0.596415

# filling all the cells with zeroes
su["pred"] = np.zeros(601)
# taking threshold value and above the prob value will be treated as correct value 
su.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(su["pred"], su["nf0"])
classification

#Fourth model on measures of kids
logit_model4 = sm.logit('nf0 ~ kids ', data = su).fit()

#summary
logit_model2.summary2() # for AIC
#AIC value = 667.9409

logit_model2.summary()

pred = logit_model2.predict(su.iloc[ :, 1: ])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(su.nf0, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold
#0.842105263157894

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)
#Area under the curve is 0.569645

# filling all the cells with zeroes
su["pred"] = np.zeros(601)
# taking threshold value and above the prob value will be treated as correct value 
su.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(su["pred"], su["nf0"])
classification

#Final Model is the first one as it has minimum AIC value
### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(su, test_size = 0.3) # 30% test data

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('nf0 ~ vryunhap +unhap +avgmarr + hapavg +vryhap ', data = train_data).fit()

#summary
model.summary2() # for AIC
#445.9394
model.summary()

# Prediction on Test data set
test_pred = model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(181)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['nf0'])
confusion_matrix

accuracy_test = (39 + 66)/(181) 
accuracy_test
#0.580110497237569

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["nf0"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["nf0"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test
#0.6494720965309201

# prediction on train data
train_pred = model.predict(train_data.iloc[ :, 1: ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(420)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['nf0'])
confusion_matrx

accuracy_train = (77 + 132)/(420)
print(accuracy_train)
#0.4976190476190476

#Q2

import pandas as pd
import numpy as np
# import seaborn as sb
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

#Importing Data
df = pd.read_csv("D:/data/advertising.csv", sep = ",")

c1 = df.drop(['Ad_Topic_Line','City','Male','Country','Timestamp','Age'], axis = 1)
c1.head(11)
c1.describe()
c1.isna().sum()

mapping = {c1.columns[0]: 'DTS', c1.columns[1]: 'INC' ,c1.columns[2]: 'IU' ,c1.columns[3]: 'COADD' }
su = c1.rename(columns=mapping)

# Model building 
# import statsmodels.formula.api as sm
logit_model = sm.logit('COADD ~ DTS + INC + IU', data = su).fit()

#summary
logit_model.summary2() # for AIC
#256.0049
logit_model.summary()

pred = logit_model.predict(su.iloc[ :, 0:3 ])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(su.COADD, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold
#0.7162878695323616

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)
#0.988148

# filling all the cells with zeroes
su["pred"] = np.zeros(1000)
# taking threshold value and above the prob value will be treated as correct value 
su.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(su["pred"], su["COADD"])
classification
#ACCURACY IS 0.96

### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(su, test_size = 0.3) # 30% test data

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('COADD ~ DTS + INC + IU', data = train_data).fit()

#summary
model.summary2() # for AIC
#212.2886
model.summary()

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(300)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['COADD'])
confusion_matrix

accuracy_test = (155 + 139)/(300) 
accuracy_test
#0.98

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["COADD"])
classification_test
#ACCURACY IS 0.98

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["COADD"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test
#0.9980413105413106

# prediction on train data
train_pred = model.predict(train_data.iloc[ :, 0:3])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(700)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['COADD'])
confusion_matrx

accuracy_train = (340 + 325)/(700)
print(accuracy_train)
#0.95


#Q3

import pandas as pd
import numpy as np
# import seaborn as sb
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

#Importing Data
df = pd.read_csv("D:/data/election_data.csv", sep = ",")

c1 = df.drop(['Election-id','Year'], axis = 1)
c1.head(11)
c1.describe()
c1.isna().sum()
c1 = c1.drop([0])

mapping = {c1.columns[0]: 'RES', c1.columns[1]: 'AMTS' ,c1.columns[2]: 'PR'  }
su = c1.rename(columns=mapping)

# Model building 
# import statsmodels.formula.api as sm
logit_model = sm.logit('RES ~ AMTS + PR', data = su).fit()

#summary
logit_model.summary2() # for AIC
#9.8177
logit_model.summary()

pred = logit_model.predict(su.iloc[ :, 1: ])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(su.RES, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold
#0.679359445509941

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)
#0.958333


# filling all the cells with zeroes
su["pred"] = np.zeros(10)
# taking threshold value and above the prob value will be treated as correct value 
su.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(su["pred"], su["RES"])
classification
#ACCURACY =  0.80 

### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(su, test_size = 0.3) # 30% test data

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('RES ~ AMTS + PR', data = train_data).fit()

#summary
model.summary2() # for AIC
#9.8177
model.summary()

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(3)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['RES'])
confusion_matrix

accuracy_test = (1 + 2)/(3) 
accuracy_test
#1.0

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["RES"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["RES"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test


# prediction on train data
train_pred = model.predict(train_data.iloc[ :, 1: ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(7)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['RES'])
confusion_matrx

accuracy_train = (3 + 3)/(7)
print(accuracy_train)
#0.8571428571428571

#Q4

import pandas as pd
import numpy as np
# import seaborn as sb
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

#Importing Data
data = pd.read_csv("D:/data/bank_data.csv", sep = ",")
d = pd.read_csv("D:/data/bank_data.csv", sep = ",")

df = data[['divorced', 'married' ,'single']]
df["martial"] = df.idxmax(axis=1)

df1 = data[['poutfailure', 'poutother' ,'poutsuccess','poutunknown']]
df1["pout"] = df1.idxmax(axis=1)

df2 = data[['joadmin.','joblue.collar','joentrepreneur','johousemaid','jomanagement','joretired','joself.employed','joservices','jostudent','jotechnician','jounemployed','jounknown']]
df2["job"] = df2.idxmax(axis=1)

df3 = data[['con_cellular','con_telephone','con_unknown']]
df3["contact"] = df3.idxmax(axis=1)

from sklearn import preprocessing
 
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
 
# Encode labels in column 'species'.
df['martial']= label_encoder.fit_transform(df['martial'])
df1['pout']= label_encoder.fit_transform(df1['pout'])
df2['job']= label_encoder.fit_transform(df2['job'])
df3['contact']= label_encoder.fit_transform(df3['contact'])

data.drop(data.iloc[:, 9:32], inplace = True, axis = 1)

data = pd.concat([data, df.martial], axis=1)
data = pd.concat([data, df1.pout], axis=1)
data = pd.concat([data, df2.job], axis=1)
data = pd.concat([data, df3.contact], axis=1)
data = pd.concat([data, d.y], axis=1)

data.drop(['age','pdays','previous'], axis = 1 ,inplace = True)

#Data dataframe is ready 
#First I combined likely binary columns into single categorical column,
#Then I label encoded the single categorical column and
#finally appended those columns and removed single binary columns
#And selected the final columns which need to be taken for analysis  

# Model building 
# import statsmodels.formula.api as sm
logit_model = sm.logit('y ~ default + balance + housing + loan + duration + campaign + martial + pout + job + contact ', data = data).fit()

#summary
logit_model.summary2() # for AIC
#24692.0678
logit_model.summary()

pred = logit_model.predict(data.iloc[ :, 0:10])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(data.y, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold
#0.12432558087227445

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)
#0.863667

# filling all the cells with zeroes
data["pred"] = np.zeros(45211)
# taking threshold value and above the prob value will be treated as correct value 
data.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(data["pred"], data["y"])
classification
#Accuracy = 0.79

### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size = 0.3) # 30% test data

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('y ~ default + balance + housing + loan + duration + campaign + martial + pout + job + contact ', data = train_data).fit()

#summary
model.summary2() # for AIC
#17116.6203
model.summary()

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(13564)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['y'])
confusion_matrix

accuracy_test = (9355 + 1303)/(13564) 
accuracy_test
#0.7857564140371572

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["y"])
classification_test
#0.79

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["y"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test
#0.8613319639964256

# prediction on train data
train_pred = model.predict(train_data.iloc[ :, 0:10 ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(31647)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['y'])
confusion_matrx

accuracy_train = (22289 + 2872)/(31647)
print(accuracy_train)
#0.7950516636648024










