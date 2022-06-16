# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 14:41:37 2022

@author: taran
"""
import os
import pandas as pd

df = pd.read_csv(r"D:/data/game.csv", encoding = 'utf8')
df.shape 
df.columns
df.describe 
df.dtypes
df.game

from sklearn.feature_extraction.text import TfidfVectorizer #term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus

####### Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words = "english")    # taking stop words from tfid vectorizer 

######### Checking the NaN values in overview column with empty string
df["game"].isnull().sum()  ### No Nan values found
 
######### Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(df.game)   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape #12294, 46

from sklearn.metrics.pairwise import linear_kernel

#### Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

###### creating a mapping of df game to index number 
df_index = pd.Series(df.index, index = df['game']).drop_duplicates()

df_id = df_index["ICO"]
df_id
topN = 10 ## Assigning Top 10 to get Top 10 scores 
def get_recommendations(Name, topN):    
    
    df_id = df_index[Name]
    
    # Getting the pair wise similarity score 
    cosine_scores = list(enumerate(cosine_sim_matrix[df_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the movie index 
    df_idx  =  [i[0] for i in cosine_scores_N]
    df_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    df_similar_show = pd.DataFrame(columns=["game", "rating"])
    df_similar_show["game"] = df.loc[df_idx, "game"]
    df_similar_show["rating"] = df_scores
    df_similar_show.reset_index(inplace = True)  
    
    print (df_similar_show)
   

    
##### Will enter the df (Game data) and number of df's that is to be recommended 
get_recommendations("Grand Theft Auto IV", topN = 10)
df_index["Grand Theft Auto IV"]

#Q2

#Loading some imp library and dataset
import pandas as pd

entertainment = pd.read_csv("D:/data/Entertainment.csv")

entertainment.shape
entertainment.columns
entertainment.Category

#Checking for null values in Category
entertainment["Category"].isnull().sum()

#importing tfdifvectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#To remove all stop words
tfidf = TfidfVectorizer(stop_words="english")

#Converting categorical to numerical format
tfidf_matrix = tfidf.fit_transform(entertainment.Category)
tfidf_matrix.shape

#Cosine similariry

from sklearn.metrics.pairwise import linear_kernel

cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

enter_index = pd.Series(entertainment.index, index=entertainment["Titles"]).drop_duplicates()


enter_id = enter_index["Heat (1995)"]
enter_id

def get_recommendations(Name, topN):    
    # topN = 10
    # Getting the movie index using its title 
    enter_id = enter_index[Name]
   
   
    cosine_scores = list(enumerate(cosine_sim_matrix[enter_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the movie index 
    enter_idx  =  [i[0] for i in cosine_scores_N]
    enter_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    enter_similar_show = pd.DataFrame(columns=["name", "Score"])
    enter_similar_show["name"] = entertainment.loc[enter_idx, "Titles"]
    enter_similar_show["Score"] = enter_scores
    enter_similar_show.reset_index(inplace = True)  
    # enter_similar_show.drop(["index"], axis=1, inplace=True)
    print (enter_similar_show)
    

get_recommendations("Now and Then (1995)", topN = 10)
enter_index["Now and Then (1995)"]



