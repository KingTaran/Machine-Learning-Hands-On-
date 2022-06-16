# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 16:53:59 2022

@author: taran
"""
#Q1


#Voting


# Import the required libraries
from sklearn import datasets, linear_model, svm, neighbors, naive_bayes
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
diabetes = pd.read_csv("D:/data/diabetes.csv", encoding = 'utf8')

lb = LabelEncoder()
diabetes["Outcome"] = lb.fit_transform(diabetes["Outcome"])

colnames = list(diabetes.columns)

x = diabetes.iloc[:, 0:8]
y = diabetes.iloc[:,8:9]

# Split the train and test samples
test_samples = 100
x_train, y_train = x[:-test_samples], y[:-test_samples]
x_test, y_test = x[-test_samples:], y[-test_samples:]

# Instantiate the learners (classifiers)
learner_1 = neighbors.KNeighborsClassifier(n_neighbors=5)
learner_2 = linear_model.Perceptron(tol=1e-2, random_state=0)
learner_3 = svm.SVC(gamma=0.001)

# Instantiate the voting classifier
voting = VotingClassifier([('KNN', learner_1),
                           ('Prc', learner_2),
                           ('SVM', learner_3)])


# Fit classifier with the training data
voting.fit(x_train, y_train)

# Predict the most voted class
hard_predictions = voting.predict(x_test)

print('Hard Voting:', accuracy_score(y_test, hard_predictions))
#0.75

# Soft Voting # 
# Instantiate the learners (classifiers)
learner_4 = neighbors.KNeighborsClassifier(n_neighbors = 5)
learner_5 = naive_bayes.GaussianNB()
learner_6 = svm.SVC(gamma = 0.001, probability = True)

# Instantiate the voting classifier
voting = VotingClassifier([('KNN', learner_4),
                           ('NB', learner_5),
                           ('SVM', learner_6)],
                            voting = 'soft')

# Fit classifier with the training data
voting.fit(x_train, y_train)
learner_4.fit(x_train, y_train)
learner_5.fit(x_train, y_train)
learner_6.fit(x_train, y_train)

# Predict the most probable class
soft_predictions = voting.predict(x_test)

# Get the base learner predictions
predictions_4 = learner_4.predict(x_test)
predictions_5 = learner_5.predict(x_test)
predictions_6 = learner_6.predict(x_test)

# Accuracies of base learners
print('L4:', accuracy_score(y_test, predictions_4))
print('L5:', accuracy_score(y_test, predictions_5))
print('L6:', accuracy_score(y_test, predictions_6))

# Accuracy of Soft voting
print('Soft Voting:', accuracy_score(y_test, soft_predictions))
#0.76


# Stacking (Regression)

#importing required libraries for ensembel models
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

#splitting the data into train and test 

predictors = diabetes.iloc[:, 0:8]
target = diabetes.iloc[:,8:9]


trainx, trainy = predictors[:500], diabetes.Outcome[:500]
testx, testy = predictors[500:], diabetes.Outcome[500:]


#stacking model
#creating meta learner using logistic regression
meta_learner = LogisticRegression(solver='lbfgs')
#importing stacking  classifier
from sklearn.ensemble import StackingClassifier
estimators = [('clf1', KNeighborsClassifier(n_neighbors=4)),
              ('clf2', DecisionTreeClassifier(criterion='entropy', max_depth = 4)),
              ('clf3', MLPClassifier(hidden_layer_sizes =(100, ), solver='lbfgs', random_state=123456))]
stack_clf = StackingClassifier(estimators=estimators, final_estimator=meta_learner)
stack_clf.fit(trainx, trainy)
#test accuracy
np.mean(testy == stack_clf.predict(testx))
#0.78
#train accuracy
np.mean(trainy == stack_clf.predict(trainx))


#Bagging

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the dataset
diabetes = pd.read_csv("D:/data/diabetes.csv", encoding = 'utf8')

lb = LabelEncoder()
diabetes["Outcome"] = lb.fit_transform(diabetes["Outcome"])

predictors = diabetes.iloc[:, 0:8]
type(predictors)

target = diabetes["Outcome"]
type(target)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)


from sklearn.ensemble import BaggingClassifier
dt = DecisionTreeClassifier(max_depth = 1)
bag_clf = BaggingClassifier(dt, n_estimators = 500, max_samples = 0.5, max_features = 0.5)
bag_clf = bag_clf.fit(x_train, y_train)
#test acuuracy for bagging model
np.mean((bag_clf.predict(x_test) == y_test))
#train acuuracy for bagging model
np.mean((bag_clf.predict(x_train) == y_train))
#for improving the model performance altering the parameters by using gridsearchcv.
param_grid = {'base_estimator__max_depth' : [1, 2, 3, 4, 5],'max_samples' : [0.05, 0.1, 0.2, 0.5, 0.6, 0.8]}
gridsearch_bag = GridSearchCV(BaggingClassifier(DecisionTreeClassifier(),n_estimators = 100, max_features = 0.5),param_grid, scoring = "accuracy")
gridsearch_bag.fit(x_train, y_train)
model_bag = gridsearch_bag.best_estimator_
#fitting the best estimator of gridsearchcv to the data
model_bag.fit(x_train,y_train)
np.mean((model_bag.predict(x_test) == y_test))
#train acuuracy for bagging model
np.mean((model_bag.predict(x_train) == y_train))


#boosting model
#XGB boosting method
from xgboost import XGBClassifier
xgb_clf = XGBClassifier(max_depths = 5, n_estimators = 500, learning_rate = 0.3, n_jobs = -1)



xgb_clf.fit(x_train,y_train)
np.mean((xgb_clf.predict(x_test) == y_test))
#train acuuracy for bagging model
np.mean((xgb_clf.predict(x_train) == y_train))
#check the accuracy at different parameters by using gridsearch cv
xgb_clf = XGBClassifier(n_estimators = 500, learning_rate = 0.1, random_state = 42)
param_grid1 = {'max_depth': [2, 3, 4, 5], 'gamma': [0.1, 0.2, 0.3],
               'subsample': [0.1, 0.2], 'colsample_bytree': [0.1, 0,2],
               'rag_alpha': [1e-2, 0.1, 1]}
gridsearch_xgb = GridSearchCV(xgb_clf, param_grid1, n_jobs=-1, cv=5, scoring='accuracy')
gridsearch_xgb.fit(x_train, y_train)
model_XGB = gridsearch_xgb.best_estimator_
#test accuracy
np.mean(y_test == model_XGB.predict(x_test))
#train accuracy
np.mean(y_train == model_XGB.predict(x_train))


#Q2

#Voting

# Import the required libraries
from sklearn import datasets, linear_model, svm, neighbors, naive_bayes
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Load the dataset
Tumor = pd.read_csv("D:/data/Tumor_Ensemble.csv", encoding = 'utf8')

lb = LabelEncoder()
Tumor["diagnosis"] = lb.fit_transform(Tumor["diagnosis"])


from sklearn import preprocessing

x = Tumor.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

estimators = [('clf1', KNeighborsClassifier(n_neighbors=4)),
              ('clf2', DecisionTreeClassifier(criterion='entropy', max_depth = 4)),
              ('clf3', MLPClassifier(hidden_layer_sizes =(100, ), solver='lbfgs', random_state=123456))]


x = df.iloc[:, 0:31]
y = df.iloc[:,31:32]

#splitting the data into train and test
target = df.iloc[:, 1]
predictors = df.iloc[:, 2:32]
trainx, trainy = predictors[:400], target[:400]
testx, testy = predictors[400:], target[400:]


#voting model
from sklearn.ensemble import VotingClassifier
import numpy as np

vote_clf = VotingClassifier(estimators, voting='soft')
vote_clf.fit(trainx, trainy)
np.mean(testy == vote_clf.predict(testx))
np.mean(trainy == vote_clf.predict(trainx))

#Stacking
from sklearn.linear_model import LogisticRegression
#creating meta learner using logistic regression
meta_learner = LogisticRegression(solver='lbfgs')
#importing stacking  classifier
from sklearn.ensemble import StackingClassifier
estimators = [('clf1', KNeighborsClassifier(n_neighbors=4)),
              ('clf2', DecisionTreeClassifier(criterion='entropy', max_depth = 4)),
              ('clf3', MLPClassifier(hidden_layer_sizes =(100, ), solver='lbfgs', random_state=123456))]
stack_clf = StackingClassifier(estimators=estimators, final_estimator=meta_learner)
stack_clf.fit(trainx, trainy)
#test accuracy
np.mean(testy == stack_clf.predict(testx))
#0.988
#train accuracy
np.mean(trainy == stack_clf.predict(trainx))
#0.995

#Bagging

from sklearn.ensemble import BaggingClassifier
dt = DecisionTreeClassifier(max_depth = 1)
bag_clf = BaggingClassifier(dt, n_estimators = 500, max_samples = 0.5, max_features = 0.5)
bag_clf = bag_clf.fit(trainx, trainy)
#test acuuracy for bagging model
np.mean((bag_clf.predict(testx) == testy))
#0.92
#train acuuracy for bagging model
np.mean((bag_clf.predict(trainx) == trainy))
#0.95

#boosting model
#XGB boosting method
from xgboost import XGBClassifier
xgb_clf = XGBClassifier(max_depths = 5, n_estimators = 500, learning_rate = 0.3, n_jobs = -1)
xgb_clf.fit(trainx, trainy)
np.mean((xgb_clf.predict(testx) == testy))
#0.9585798816568047
#train acuuracy for bagging model
np.mean((xgb_clf.predict(trainx) == trainy))
#check the accuracy at different parameters by using gridsearch cv
xgb_clf = XGBClassifier(n_estimators = 500, learning_rate = 0.1, random_state = 42)
param_grid1 = {'max_depth': [2, 3, 4, 5], 'gamma': [0.1, 0.2, 0.3],
               'subsample': [0.1, 0.2], 'colsample_bytree': [0.1, 0,2],
               'rag_alpha': [1e-2, 0.1, 1]}
gridsearch_xgb = GridSearchCV(xgb_clf, param_grid1, n_jobs=-1, cv=5, scoring='accuracy')
gridsearch_xgb.fit(trainx, trainy)
model_XGB = gridsearch_xgb.best_estimator_
#test accuracy
np.mean(testy == model_XGB.predict(testx))
#train accuracy
np.mean(trainy == model_XGB.predict(trainx))


#Q3


##### Bagging ######

import pandas as pd

df = pd.read_excel("D:\data\Coca_Rating_Ensemble.xlsx")
df.info()
# Dummy variables
df.head()
df.info()

df= df.drop(["Origin","Bean_Type","Name","Company_Location"],axis=1)
# n-1 dummy variables will be created for n categories
df = pd.get_dummies(df, columns = ["Company"], drop_first = True)

df.head()


# Input and Output Split
predictors = df.loc[:, df.columns!="Rating"]
type(predictors)

target = df["Rating"]
type(target)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)


from sklearn import tree
clftree = tree.DecisionTreeRegressor()
from sklearn.ensemble import BaggingRegressor


bag_clf = BaggingRegressor(base_estimator = clftree, 
                            bootstrap = True, n_jobs = 1, random_state = 0)

bag_clf.fit(x_train, y_train)



# Prediction
test_pred = bag_clf.predict(x_test)
train_pred = bag_clf.predict(x_train)

# Measuring accuracy
from sklearn.metrics import mean_squared_error, r2_score

# Error on test dataset
mean_squared_error(y_test, test_pred)
#0.19178525219070272
r2_score(y_test, test_pred)
#0.07949221333271395

# Error on train dataset
mean_squared_error(y_train, train_pred)
#0.04522666626610958
r2_score(y_train, train_pred)
#0.8062139272574314



####### Boosting ###############





# Input and Output Split
predictors = df.loc[:, df.columns!="Rating"]
type(predictors)

target = df["Rating"]
type(target)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)

# Refer to the links
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble

from sklearn.ensemble import GradientBoostingRegressor

boost_clf = GradientBoostingRegressor()

boost_clf.fit(x_train, y_train)

# Prediction
test_pred = boost_clf.predict(x_test)
train_pred = boost_clf.predict(x_train)

# Measuring accuracy
from sklearn.metrics import mean_squared_error, r2_score

# Error on test dataset
mean_squared_error(y_test, test_pred)
#0.18070230077350813
r2_score(y_test, test_pred)
#0.13268683055301245

# Error on train dataset
mean_squared_error(y_train, train_pred)
#0.15227675688362582
r2_score(y_train, train_pred)
#0.34752841359511644

# Hyperparameters
boost_clf2 = GradientBoostingRegressor(learning_rate = 0.02, n_estimators = 1000, max_depth = 1)
boost_clf2.fit(x_train, y_train)

# Prediction
test_pred = boost_clf2.predict(x_test)
train_pred = boost_clf2.predict(x_train)

# Measuring accuracy
from sklearn.metrics import mean_squared_error, r2_score

# Error on test dataset
mean_squared_error(y_test, test_pred)
#0.18403203320327344
r2_score(y_test, test_pred)
#0.11670518131717966

# Error on train dataset
mean_squared_error(y_train, train_pred)
#0.18167311444753226
r2_score(y_train, train_pred)
#0.2215716461489503



#Q4


#importing required libraries for ensembel models
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
#importing data
password = pd.read_excel("D:/data/Ensemble_Password_Strength.xlsx")
#check the data has null values, duplicate records in the data
password.isna().sum()
password.duplicated().sum()
password.info()
#characters column has some int and other data types so convert all of them to str
password.characters = password.characters.astype('str')
#converting dataframe into tuple
password_tuple=list(password.to_records(index=False))
x=[labels[0] for labels in password_tuple]
y=[labels[1] for labels in password_tuple]
#defining custom fun to split the data by letter wise
def word_divide_char(inputs):
    character=[]
    for i in inputs:
        character.append(i)
    return character
#converting the data numaric form and normalizing the data by using tfidf vectorizer.
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(tokenizer=word_divide_char, lowercase=False)
matrix = tfidf.fit_transform(x)
#converting sparse matrix to dense matrix
df = pd.DataFrame(matrix.todense())
#splitting the data into train and test
target = password.iloc[:, 1]
trainx, trainy = df[:1600], target[:1600]
testx, testy = df[1600:], target[1600:]
#diffeternt ensembel models are used below to find strength accuracy.
#bagging model
from sklearn.ensemble import BaggingClassifier
dt = DecisionTreeClassifier(max_depth = 1)
bag_clf = BaggingClassifier(dt, n_estimators = 500, max_samples = 0.5, max_features = 0.5)
bag_clf = bag_clf.fit(trainx, trainy)

#test acuuracy for bagging model
np.mean((bag_clf.predict(testx) == testy))
#0.8370927318295739

#train acuuracy for bagging model
np.mean((bag_clf.predict(trainx) == trainy))
# 0.863125

#for improving the model performance altering the parameters by using gridsearchcv.
param_grid = {'base_estimator__max_depth' : [1, 2, 3, 4, 5],'max_samples' : [0.05, 0.1, 0.2, 0.5, 0.6, 0.8]}
gridsearch_bag = GridSearchCV(BaggingClassifier(DecisionTreeClassifier(),n_estimators = 100, max_features = 0.5),param_grid, scoring = "accuracy")
gridsearch_bag.fit(trainx, trainy)
model_bag = gridsearch_bag.best_estimator_
#fitting the best estimator of gridsearchcv to the data
model_bag.fit(trainx, trainy)
np.mean((model_bag.predict(testx) == testy))
#train acuuracy for bagging model
np.mean((model_bag.predict(trainx) == trainy))
#0.89125
 
#boosting model
#XGB boosting method
from xgboost import XGBClassifier
xgb_clf = XGBClassifier(max_depths = 5, n_estimators = 500, learning_rate = 0.3, n_jobs = -1)
xgb_clf.fit(trainx, trainy)
np.mean((xgb_clf.predict(testx) == testy))
#train acuuracy for bagging model
#0.9298245614035088
np.mean((xgb_clf.predict(trainx) == trainy))
#check the accuracy at different parameters by using gridsearch cv
xgb_clf = XGBClassifier(n_estimators = 500, learning_rate = 0.1, random_state = 42)
param_grid1 = {'max_depth': [2, 3, 4, 5], 'gamma': [0.1, 0.2, 0.3],
               'subsample': [0.1, 0.2], 'colsample_bytree': [0.1, 0,2],
               'rag_alpha': [1e-2, 0.1, 1]}
gridsearch_xgb = GridSearchCV(xgb_clf, param_grid1, n_jobs=-1, cv=5, scoring='accuracy')
gridsearch_xgb.fit(trainx, trainy)
model_XGB = gridsearch_xgb.best_estimator_
#test accuracy
np.mean(testy == model_XGB.predict(testx))
#0.924812030075188
#train accuracy
np.mean(trainy == model_XGB.predict(trainx))
#0.948125

#stacking model
#creating meta learner using logistic regression
meta_learner = LogisticRegression(solver='lbfgs')
#importing stacking  classifier
from sklearn.ensemble import StackingClassifier
estimators = [('clf1', KNeighborsClassifier(n_neighbors=4)),
              ('clf2', DecisionTreeClassifier(criterion='entropy', max_depth = 4)),
              ('clf3', MLPClassifier(hidden_layer_sizes =(100, ), solver='lbfgs', random_state=123456))]
stack_clf = StackingClassifier(estimators=estimators, final_estimator=meta_learner)
stack_clf.fit(trainx, trainy)
#test accuracy
np.mean(testy == stack_clf.predict(testx))
#0.8446115288220551
#train accuracy
np.mean(trainy == stack_clf.predict(trainx))
#0.939375

##check the accuracy at different parameters by using gridsearch cv
param_grid2 = {'clf2__min_samples_split': [4, 5, 6, 7, 8], 'clf2__max_depth': [3, 4, 5, 6], 'clf3__hidden_layer_sizes': [200, 300]}
gridsearch_stack = GridSearchCV(stack_clf, param_grid2, n_jobs=-1, cv=5, scoring='accuracy')
gridsearch_stack.fit(trainx, trainy)
model_stack = gridsearch_stack.best_estimator_
np.mean(testy == model_stack.predict(testx))
np.mean(trainy == model_stack.predict(trainx))

#voting model
from sklearn.ensemble import VotingClassifier
vote_clf = VotingClassifier(estimators, voting='soft')
vote_clf.fit(trainx, trainy)
np.mean(testy == vote_clf.predict(testx))
#0.8320802005012531
np.mean(trainy == vote_clf.predict(trainx))
#0.961875
##check the accuracy at different parameters by using gridsearch cv
param_grid3 = {'clf2__min_samples_split': [4, 5, 6, 7, 8], 'clf2__max_depth': [3, 4, 5, 6], 'clf3__hidden_layer_sizes': [200, 300]}
gridsearch_vote = GridSearchCV(vote_clf, param_grid3, n_jobs=-1, cv = 5 , scoring = 'accuracy')
gridsearch_vote.fit(trainx, trainy)
model_vote = gridsearch_stack.best_estimator_
np.mean(testy == model_vote.predict(testx))
np.mean(trainy == model_vote.predict(trainx))





