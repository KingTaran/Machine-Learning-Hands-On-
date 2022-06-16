# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 04:47:06 2021

@author: taran
"""

#Q1

import pandas as pd               # for Data Manipulation
import matplotlib.pyplot as plt   # for Visualization
import numpy as np                #for Mathematical calculations
import seaborn as sns             #for Advanced visualizations

air = pd.read_csv("D:\\Data\\EastWestAirlines.csv")

air

# We see the columns in the dataset
air.columns

# As a part of the Data cleansing we check the data for any missing/ na values
air.isna().sum()


#Outlier Treatment
#Now we all know that data can have outliers which are the values that may effect the analysis in a big way. 
#we can see the outliers being outshined out of the plot.

# We now plot the boxplot for the data using each feature independently and check for Outliers
plt.boxplot(air.Balance);plt.title('Boxplot');plt.show()

plt.boxplot(air.Qual_miles);plt.title('Boxplot');plt.show()  # outliers present

plt.boxplot(air.cc1_miles);plt.title('Boxplot');plt.show()  # No outliers 

plt.boxplot(air.cc2_miles);plt.title('Boxplot');plt.show()  # outliers present

plt.boxplot(air.cc3_miles);plt.title('Boxplot');plt.show()  # outliers present

plt.boxplot(air.Bonus_miles);plt.title('Boxplot');plt.show()  # outliers present

plt.boxplot(air.Bonus_trans);plt.title('Boxplot');plt.show()  # outliers present

plt.boxplot(air.Flight_miles_12mo);plt.title('Boxplot');plt.show()  # outliers present

plt.boxplot(air.Flight_trans_12);plt.title('Boxplot');plt.show()  # outliers present

#Now in order to deal with the outliers we can use one of "3R Techniques" viz:
      # 1. Rectify
      # 2. Retain
       #3. Remove

#In our example we choose let say to keep or retain the data, but how should we keep outliers in the data and work forward?
#We use the Winsorization techniue.This technique helps modify the sample distribution of random variables by 
#removing the outliers and replacing them with the values that constitute the 5th percentile and 95th percentile values,
#i.e, all data below 5th percentile gets replaced with the value of the 5th percentile and all the values above 
#95th percentile, with the 95th percentile value.


from scipy.stats.mstats import winsorize

air['Balance']=winsorize(air.Balance,limits=[0.07, 0.093])   
plt.boxplot(air['Balance']);plt.title('Boxplot');plt.show()

air['Qual_miles']=winsorize(air.Qual_miles,limits=[0.06, 0.094])   
plt.boxplot(air['Qual_miles']);plt.title('Boxplot');plt.show()

air['cc2_miles']=winsorize(air.cc2_miles,limits=[0.02, 0.098])   
plt.boxplot(air['cc2_miles']);plt.title('Boxplot');plt.show()

air['cc3_miles']=winsorize(air.cc3_miles,limits=[0.01, 0.099])   
plt.boxplot(air['cc3_miles']);plt.title('Boxplot');plt.show()

air['Bonus_miles']=winsorize(air.Bonus_miles,limits=[0.08, 0.092])   
plt.boxplot(air['Bonus_miles']);plt.title('Boxplot');plt.show()

air['Bonus_trans']=winsorize(air.Bonus_trans,limits=[0.01, 0.099])   
plt.boxplot(air['Bonus_trans']);plt.title('Boxplot');plt.show()

air['Flight_miles_12mo']=winsorize(air.Flight_miles_12mo,limits=[0.15, 0.85])   
plt.boxplot(air['Flight_miles_12mo']);plt.title('Boxplot');plt.show()

air['Flight_trans_12']=winsorize(air.Flight_trans_12,limits=[0.15, 0.85])   
plt.boxplot(air['Flight_trans_12']);plt.title('Boxplot');plt.show()


# Now we check the data for zero variance values
(air == 0).all()

# We drop the features that have zero variance values
air1 = air
air1.drop(["Qual_miles","Flight_miles_12mo","Flight_trans_12"], axis=1,inplace = True)
air1.columns

# We see the data again now to check whether the data is in scale
air1.describe

# we notice that the data needs to be normalise, using normalization

from sklearn import preprocessing   #package for normalize
air_normalized = preprocessing.normalize(air1)
print(air_normalized)

##########################Univariate, Bivariate################
plt.hist(air1["Balance"])   #Univariate

plt.hist(air1["Days_since_enroll"])

plt.scatter(air1["Balance"], air["Days_since_enroll"]);plt.xlabel('Days_since_enroll');plt.ylabel('Balance')   #Bivariate

air1.skew(axis = 0, skipna = True) 

air1.kurtosis(axis = 0, skipna = True)

#After all the analyses we now start with the Heirarchical Clustering procedure which would require us with 
#building the dendogram

#Now, one of the advantages of hierarchical clustering is that we do not have to specify the number of clusters. 
#In order to determine the optimal number of clusters we plot the dendogram, which is a diagram representation 
#of the tree based approach.

# in order to create a dendogram we need to define the linkage and create a linkage matrix
# we would need the appropriate library for the same
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

###Finding the Distance using Euclidean Distance with 
# Complete (maximum) linkage: Maximum distance among all data points in two clusters
z = linkage(air_normalized, method = "complete", metric = "euclidean")

# Now we plot the dendogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, leaf_rotation = 0,  leaf_font_size = 10 )  
plt.show()

# We select the number of clusters from the dendogram as "4"

#Hierarchical clustering means creating a tree of clusters by iteratively grouping or separating data points. 
    #There are two types of hierarchical clustering:
        #Agglomerative clustering
        #Divisive clustering
#We now apply the Agglomerative clustering technique:Agglomerative clustering is kind of a bottom-up approach. 
#Each data point is assumed to be a separate cluster at first. Then the similar clusters are iteratively combined

from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = "euclidean").fit(air_normalized) 
h_complete.labels_

# These labels for the clusters so formed are in the array form, which needs to be changed to incorporate into the dataset
# In order to do so, we use the type casting and change the array to a series which will be easier to incorporate
cluster_labels = pd.Series(h_complete.labels_)

# We now incorporate the labels into the dataset as a new feature that will fit as per the records
air['clust'] = cluster_labels

# In order to check the same we use head function
air.head()

# We can clearly see that we have the labels in the dataset in the form of a column called "clust", symbolizing the clusters
# In order to see the clusters we aggregate the records within the clusters and group them by the clusters to visualize the 
# 4 nos of clear cluster formed
air.iloc[:, 0:].groupby(air.clust).mean()

#We can now clearly see the 4 number of clusters formed, which can be described as under
        #1. Cluster1 = "0" = Defines the group of fliers that earn the most out of the card and are frequent fliers
        #2. Cluster2 = "1" = Defines the group of fliers that earn but are the third most frequent fliers
        #3. Cluster3 = "2" = Defines the group of fliers that earn the least and are the least frequent fliers
        #4. Cluster4 = "3" = Defines the group of fliers that earn but and are the second most frequent fliers'''


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

crime.shape

# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(crime.iloc[:,1:])
df_norm.describe()

from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch # for creating dendrogram 

z = linkage(df_norm, method="complete",metric="euclidean")

plt.figure(figsize=(15, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Features')
plt.ylabel('Crime')
sch.dendrogram(z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

df_norm['clust'] = cluster_labels

df_norm.iloc[:, 0:].groupby(df_norm.clust).mean()

#1. Cluster1 = "0" = Defines the moderate urbanpop with heavy crime rates
#2. Cluster2 = "1" = Defines the more civilized urbanpop with moderate crime rates
#3. Cluster3 = "2" = Defines the less urbanpop with least crime rate


#Q3

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from numpy import random, float, array
import numpy as np
import seaborn as sns

df = pd.read_excel("D:\\Data\\Telco_customer_churn.xlsx")
df.head()

df.shape

df.info()
#Here we understand which are categorical and which are numerical

df1 = pd.get_dummies(df, drop_first = True)

df1.info()

# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df1.iloc[:,1:])
df_norm.describe()

from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch # for creating dendrogram 

z = linkage(df_norm, method="complete",metric="euclidean")

plt.figure(figsize=(15, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Features')
plt.ylabel('Churned Out Customers')
sch.dendrogram(z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

df_norm['clust'] = cluster_labels

df_norm.iloc[:, 0:].groupby(df_norm.clust).mean()


#Q4

import pandas as pd               # for Data Manipulation
import matplotlib.pyplot as plt   # for Visualization
import numpy as np                #for Mathematical calculations
import seaborn as sns             #for Advanced visualizations

air = pd.read_csv("D:\\Data\\AutoInsurance.csv")

air

# We see the columns in the dataset
air.columns

# As a part of the Data cleansing we check the data for any missing/ na values
air.isna().sum()

air_new = pd.get_dummies(air)

from sklearn import preprocessing   #package for normalize
air_normalized = preprocessing.normalize(air_new)
print(air_normalized)


from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

###Finding the Distance using Euclidean Distance with 
# Complete (maximum) linkage: Maximum distance among all data points in two clusters
z = linkage(air_normalized, method = "complete", metric = "euclidean")

# Now we plot the dendogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Insurance')
sch.dendrogram(z, leaf_rotation = 0,  leaf_font_size = 10 )  
plt.show()





















