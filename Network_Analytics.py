# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 16:20:45 2022

@author: taran
"""
#Q1

import pandas as pd
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt

# Degree Centrality
G = pd.read_csv("D:\data\connecting_routes.csv")
G = G.iloc[:, 1:10]

G.columns =['Airline ID', 'Source Airport', 'Source Airport ID', 'Destination Airport','Destination Airport ID','Stops','Equipment','NA']

g = nx.Graph()

g = nx.from_pandas_edgelist(G, source = 'Source Airport', target = 'Destination Airport')


print(nx.info(g))


b = nx.degree_centrality(g)  # Degree Centrality
print(b) 

pos = nx.spring_layout(g, k = 0.15)
nx.draw_networkx(g, pos, node_size = 25, node_color = 'blue')


# closeness centrality
closeness = nx.closeness_centrality(g)
print(closeness)

## Betweeness Centrality 
b = nx.betweenness_centrality(g) # Betweeness_Centrality
print(b)

## Eigen-Vector Centrality
evg = nx.eigenvector_centrality(g) # Eigen vector centrality
print(evg)

# cluster coefficient
cluster_coeff = nx.clustering(g)
print(cluster_coeff)

# Average clustering
cc = nx.average_clustering(g) 
print(cc)

#From this dataset we can see AAE airport carries the most 
#important value in our network


import pandas as pd
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt

# Degree Centrality
G = pd.read_csv("D:\\data\\flight_hault.csv")


G.columns =['Airport ID', 'Airport Name', 'City', 'Country','IATA','ICAO','Latitude','Longitude','Altitude','Timezone','DST','Tz']

g = nx.Graph()

g = nx.from_pandas_edgelist(G, source = 'City', target = 'Country')


print(nx.info(g))


b = nx.degree_centrality(g)  # Degree Centrality
print(b) 

pos = nx.spring_layout(g, k = 0.15)
nx.draw_networkx(g, pos, node_size = 25, node_color = 'blue')

# closeness centrality
closeness = nx.closeness_centrality(g)
print(closeness)

## Betweeness Centrality 
b = nx.betweenness_centrality(g) # Betweeness_Centrality
print(b)

## Eigen-Vector Centrality
evg = nx.eigenvector_centrality(g) # Eigen vector centrality
print(evg)

# cluster coefficient
cluster_coeff = nx.clustering(g)
print(cluster_coeff)

# Average clustering
cc = nx.average_clustering(g) 
print(cc)

#From this dataset we can see 108 mile ranch airport carries the most 
#important value in our network

#Q2


import pandas as pd
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt

#Visualizing Facebook Circular Network
G = pd.read_csv("D:\\data\\facebook.csv")

G = nx.DiGraph(G.values)

nx.draw(G)

#Visualizing Instagram Star Network
G = pd.read_csv("D:\\data\\instagram.csv")

G = nx.DiGraph(G.values)

nx.draw(G)


#Visualizing Linkedin Star Network
G = pd.read_csv("D:\\data\\linkedin.csv")

G = nx.DiGraph(G.values)

nx.draw(G)