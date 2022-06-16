# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 02:52:11 2022

@author: taran
"""
#Q1

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


np.random.seed(10)

startups = pd.read_csv(r"D:\data\50_Startups.csv")

startups.head()

le = LabelEncoder()
startups["State"] = le.fit_transform(startups["State"])
startups.head()

startups.info()

startups.describe()

x = startups.values #returns a numpy array
x
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
startups_scaled = pd.DataFrame(x_scaled, columns=["R&D Spend", "Administration", "Marketing Spend", "State", "Profit"])

startups_scaled.describe()

train_data, test_data = train_test_split(startups_scaled, test_size = 0.3)

x_train = train_data.iloc[:,:-1].values.astype("float32")
x_test = test_data.iloc[:,:-1].values.astype("float32")
y_train = train_data.Profit.values.astype("float32")
y_test = test_data.Profit.values.astype("float32")

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = x_train.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='relu'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()


prediction_train = NN_model.fit(x_train, y_train, epochs=500, batch_size=32, validation_split = 0.2)


# accuracy score on train data 
eval_score_train = NN_model.evaluate(x_train,y_train,verbose=0)
eval_score_train
print ("Accuracy: %.3f%%" %(eval_score_train[1]*100)) 
#2.145%

prediction_test = NN_model.fit(x_test, y_test, epochs=500, batch_size=32, validation_split = 0.2)


# Evaluating the model on test data  
eval_score_test = NN_model.evaluate(x_test,y_test,verbose = 1)
print ("Accuracy: %.3f%%" %(eval_score_test[1]*100)) 
#0.863%


#Q2



# Import necessary libraries for MLP and reshaping the data structres
import numpy as np
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
# from keras.layers import Dropout,Flatten
from keras.utils import np_utils

np.random.seed(10)

# Loading the data set using pandas as data frame format 
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("D:\\data\\forestfires.csv")
df.describe()

df.drop(df.columns[[0, 1]], axis = 1, inplace = True)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

df['size_category']= le.fit_transform(df['size_category'])

train,test = train_test_split(df, test_size = 0.20)

x_train = train.iloc[:, 0:28].values.astype("float32")
y_train = train.iloc[:, 28:29].values.astype("float32")
x_test = test.iloc[:, 0:28].values.astype("float32")
y_test  = test.iloc[:,28:29].values.astype("float32")
# Separating the data set into 2 parts - all the inputs and label columns
# converting the integer type into float32 format 

# Normalizing the inputs to fall under 0-1 by 
# diving the entire data with 255 (max pixel value)
x_train = x_train/255
x_test = x_test/255

# one hot encoding outputs for both train and test data sets 
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Storing the number of classes into the variable num_of_classes 
num_of_classes = y_test.shape[1]
x_train.shape
y_train.shape
x_test.shape
y_test.shape

# Creating a user defined function to return the model for which we are
# giving the input to train the ANN mode
def design_mlp():
    # Initializing the model 
    model = Sequential()
    model.add(Dense(150,input_dim =28,activation="relu"))
    model.add(Dense(200,activation="tanh"))
    model.add(Dense(500,activation="tanh"))
    model.add(Dense(num_of_classes,activation="softmax"))
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
    return model

# building a cnn model using train data set and validating on test data set
model = design_mlp()

# fitting model on train data
model.fit(x=x_train,y=y_train,batch_size=5,epochs=8)

# Evaluating the model on test data  
eval_score_test = model.evaluate(x_test,y_test,verbose = 1)
print ("Accuracy: %.3f%%" %(eval_score_test[1]*100)) 
#95.192%
# accuracy on test data set

# accuracy score on train data 
eval_score_train = model.evaluate(x_train,y_train,verbose=0)
print ("Accuracy: %.3f%%" %(eval_score_train[1]*100)) 
#96.126%
# accuracy on train data set

#Q3

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

np.random.seed(10)

startups = pd.read_csv(r"D:\data\concrete.csv")
startups.drop(startups.columns[[ 7]], axis = 1, inplace = True)
startups.head()

startups.info()

startups.describe()

x = startups.values #returns a numpy array
x
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
startups_scaled = pd.DataFrame(x_scaled, columns=["cement", "slag", "ash", "water", "superplastic","coarsegg","fineagg","strength"])

startups_scaled.describe()

train_data, test_data = train_test_split(startups_scaled, test_size = 0.3)

x_train = train_data.iloc[:,:-1].values.astype("float32")
x_test = test_data.iloc[:,:-1].values.astype("float32")
y_train = train_data.strength.values.astype("float32")
y_test = test_data.strength.values.astype("float32")

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = x_train.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='relu'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()


prediction_train = NN_model.fit(x_train, y_train, epochs=500, batch_size=32, validation_split = 0.2)


# accuracy score on train data 
eval_score_train = NN_model.evaluate(x_train,y_train,verbose=0)
eval_score_train
print ("Accuracy: %.3f%%" %(eval_score_train[1]*100)) 
#Accuracy: 8.537%
prediction_test = NN_model.fit(x_test, y_test, epochs=500, batch_size=32, validation_split = 0.2)


# Evaluating the model on test data  
eval_score_test = NN_model.evaluate(x_test,y_test,verbose = 1)
print ("Accuracy: %.3f%%" %(eval_score_test[1]*100)) 
#6.960%


#Q4

# Import necessary libraries for MLP and reshaping the data structres
import numpy as np
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
# from keras.layers import Dropout,Flatten
from keras.utils import np_utils

np.random.seed(10)

# Loading the data set using pandas as data frame format 
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("D:\\data\\RPL.csv")
df.describe()

df.drop(df.columns[[0, 1,2,4,5,6]], axis = 1, inplace = True)


train,test = train_test_split(df, test_size = 0.20)

x_train = train.iloc[:, 0:7].values.astype("float32")
y_train = train.iloc[:, 7:8].values.astype("float32")
x_test = test.iloc[:, 0:7].values.astype("float32")
y_test  = test.iloc[:,7:8].values.astype("float32")
# Separating the data set into 2 parts - all the inputs and label columns
# converting the integer type into float32 format 

# Normalizing the inputs to fall under 0-1 by 
# diving the entire data with 255 (max pixel value)
x_train = x_train/255
x_test = x_test/255

# one hot encoding outputs for both train and test data sets 
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Storing the number of classes into the variable num_of_classes 
num_of_classes = y_test.shape[1]
x_train.shape
y_train.shape
x_test.shape
y_test.shape

# Creating a user defined function to return the model for which we are
# giving the input to train the ANN mode
def design_mlp():
    # Initializing the model 
    model = Sequential()
    model.add(Dense(150,input_dim =7,activation="relu"))
    model.add(Dense(200,activation="tanh"))
    model.add(Dense(500,activation="tanh"))
    model.add(Dense(num_of_classes,activation="softmax"))
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
    return model

# building a cnn model using train data set and validating on test data set
model = design_mlp()

# fitting model on train data
model.fit(x=x_train,y=y_train,batch_size=1000,epochs=8)

# Evaluating the model on test data  
eval_score_test = model.evaluate(x_test,y_test,verbose = 1)
print ("Accuracy: %.3f%%" %(eval_score_test[1]*100)) 
#78.900%
# accuracy on test data set

# accuracy score on train data 
eval_score_train = model.evaluate(x_train,y_train,verbose=0)
print ("Accuracy: %.3f%%" %(eval_score_train[1]*100)) 
#79.813%
# accuracy on train data set













