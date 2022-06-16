# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 00:18:43 2022

@author: taran
"""

#Q1
import pandas as pd

df = pd.read_csv("D:/data/SalaryData_Test.csv",encoding = "ISO-8859-1")

df1 = df.drop(columns=['relationship','workclass','education','age','maritalstatus','occupation','race','sex','native'],)


from sklearn.model_selection import train_test_split

X = df1.drop(['Salary'], axis=1)

y = df1['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_pred

from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

y_pred_train = model.predict(X_train)

y_pred_train

print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

print('Training set score: {:.4f}'.format(model.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(model.score(X_test, y_test)))

#########

df2 = pd.read_csv("D:/data/SalaryData_Train.csv",encoding = "ISO-8859-1")

df3 = df2.drop(columns=['relationship','workclass','education','age','maritalstatus','occupation','race','sex','native'],)

from sklearn.model_selection import train_test_split

X = df3.drop(['Salary'], axis=1)

y = df3['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_pred

from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

y_pred_train = model.predict(X_train)

y_pred_train

print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

print('Training set score: {:.4f}'.format(model.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(model.score(X_test, y_test)))







#Q2
import pandas as pd


df = pd.read_csv("D:/data/NB_Car_Ad.csv",encoding = "ISO-8859-1")

df1 = df.drop(columns=['User ID','Gender','Age'],)

X = df1.drop(['Purchased'], axis=1)

y = df['Purchased']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_pred

from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

y_pred_train = model.predict(X_train)

y_pred_train

print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

print('Training set score: {:.4f}'.format(model.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(model.score(X_test, y_test)))


#Q3
import pandas as pd

df1 = pd.read_csv("D:/data/Disaster_tweets_NB.csv",encoding = "ISO-8859-1")

df = df1.drop(columns=['id','location','keyword'],)

import re
stop_words = []
# Load the custom built Stopwords
with open("D:/data/stopwords_en.txt","r") as sw:
    stop_words = sw.read()

stop_words = stop_words.split("\n")
   
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))

df.text = df.text.apply(cleaning_text)

# removing empty rows
df = df.loc[df.text != " ",:]

from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size = 0.3)

# creating a matrix of token counts for the entire text document 
def split_into_words(i):
    return [word for word in i.split(" ")]

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

df_bow = CountVectorizer(analyzer = split_into_words).fit(df.text)

# Defining BOW for all messages
all_df_matrix = df_bow.transform(df.text)

# For training messages
train_df_matrix = df_bow.transform(df_train.text)

# For testing messages
test_df_matrix = df_bow.transform(df_test.text)

# Learning Term weighting and normalizing on entire emails
tfidf_transformer = TfidfTransformer().fit(all_df_matrix)

# Preparing TFIDF for train emails
train_tfidf = tfidf_transformer.transform(train_df_matrix)
train_tfidf.shape # (row, column)

# Preparing TFIDF for test emails
test_tfidf = tfidf_transformer.transform(test_df_matrix)
test_tfidf.shape #  (row, column)

# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf, df_train.target)

import numpy as np

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m == df_test.target)
accuracy_test_m

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m, df_test.target) 

pd.crosstab(test_pred_m, df_test.target)

# Training Data accuracy
train_pred_m = classifier_mb.predict(train_tfidf)
accuracy_train_m = np.mean(train_pred_m == df_train.target)
accuracy_train_m




