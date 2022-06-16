# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 16:58:11 2021

@author: taran
"""
#Q1
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

flight= pd.read_csv("D:\\Data\\EastWestAirlines.csv")

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)
    
df_norm=norm_func(flight.iloc[:,1:])

k=list(range(10,20))
k
TWSS=[]
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
TWSS

plt.plot(k,TWSS, 'ro-');plt.xlabel('number of clusters');plt.ylabel('total within sum of squares');plt.xticks(k)
    
model1=KMeans(n_clusters=14)
model1.fit(df_norm)

model1.cluster_centers_
model1.labels_
model=pd.Series(model1.labels_)
model
flight['clust']=model



flightfinal=flight.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]

fly=flight.iloc[:,1:13].groupby(flightfinal.clust).mean()


#Q2

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from numpy import random, float, array
import numpy as np
import seaborn as sns

crime = pd.read_csv("D:\\Data\\crime_data.csv")

crime.head()

# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(crime.iloc[:,1:])
df_norm.describe()

###### screw plot or elbow curve ############
k = list(range(2,15))

from sklearn.cluster import	KMeans
from scipy.spatial.distance import cdist 
import numpy as np
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))

plt.figure(figsize=(16,6))
plt.plot(k,TWSS,'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=4) 
model.fit(df_norm)


model.cluster_centers_
model.labels_
model=pd.Series(model.labels_)
model
crime['clust']=model

model

crime.iloc[:, 1:8].groupby(crime.clust).mean()


#Q3
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

insurance = pd.read_csv("D:\\Data\\Insurance Dataset.csv")

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)

df_norm=norm_func(insurance.iloc[:,1:])

k=list(range(10,20))
k
TWSS=[]
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
TWSS

plt.plot(k,TWSS, 'ro-');plt.xlabel('number of clusters');plt.ylabel('total within sum of squares');plt.xticks(k)
    
model1=KMeans(n_clusters=10)
model1.fit(df_norm)

model1.cluster_centers_
model1.labels_
model=pd.Series(model1.labels_)
model
insurance['clust']=model



insurancefinal=insurance.iloc[:,[5,0,1,2,3,4]]

insurance1=insurance.iloc[:,1:13].groupby(insurancefinal.clust).mean()

insurance1


#Q4
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

tel = pd.read_csv("D:\\Data\\Telco_customer_churn.csv")

tel.info()

tel_1 = pd.get_dummies(tel, drop_first = True)

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)

df_norm=norm_func(tel_1.iloc[:,1:])

k=list(range(10,20))
k
TWSS=[]
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
TWSS

plt.plot(k,TWSS, 'ro-');plt.xlabel('number of clusters');plt.ylabel('total within sum of squares');plt.xticks(k)
    
model1=KMeans(n_clusters=14)
model1.fit(df_norm)

model1.cluster_centers_
model1.labels_
model=pd.Series(model1.labels_)
model
tel['clust']=model



telfinal=tel.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]

tel2=tel.iloc[:,1:13].groupby(telfinal.clust).mean()

























