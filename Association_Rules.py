# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 17:47:24 2022

@author: taran
"""
#Q1

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("D:\\data\\book.csv")
    

frequent_itemsets = apriori(df, min_support = 0.15, max_len = 4, use_colnames = True)

frequent_itemsets.sort_values('support', ascending = False, inplace = True)

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)

rules.head(20)
rules.sort_values('lift', ascending = False).head(10)

import matplotlib.pyplot as plt

x = [rules.support]
y = [rules.confidence]

plt.scatter(x, y)
plt.show()

def to_list(i):
    return (sorted(list(i)))

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 3 rules 
rules_no_redudancy.sort_values('lift', ascending = False).head(3)

import matplotlib.pyplot as plt

x = [rules_no_redudancy.support]
y = [rules_no_redudancy.confidence]

plt.scatter(x, y)
plt.show()

#Insights:
 #From the books dataset, 18 rules were created using the apriori algorithm.
   #There are 3 different variables plotted in the scatterplot; confidence, support and lift. 
#The rules with the highest lift ratio lie within the constraint of support value of 0.09 to 0.198 and
#confidence value from 0.28 to 1

#Q2

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

groceries = []
with open("D:\\data\\groceries.csv") as f:
    groceries = f.read()

# splitting the data into separate transactions using separator as "\n"
groceries = groceries.split("\n")

groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))

all_groceries_list = [i for item in groceries_list for i in item]

from collections import Counter 

item_frequencies = Counter(all_groceries_list)

# after sorting
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])

# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))

# barplot of top 10 
import matplotlib.pyplot as plt

plt.bar(height = frequencies[0:11], x = list(range(0, 11)), color = 'rgbkymc')
plt.xticks(list(range(0, 11), ), items[0:11])
plt.xlabel("items")
plt.ylabel("Count")
plt.show()


# Creating Data Frame for the transactions data
groceries_series = pd.DataFrame(pd.Series(groceries_list))
groceries_series = groceries_series.iloc[:9835, :] # removing the last empty transaction

groceries_series.columns = ["transactions"]

# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = groceries_series['transactions'].str.join(sep = '*').str.get_dummies(sep = '*')

frequent_itemsets = apriori(X, min_support = 0.050, max_len = 4, use_colnames = True)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)

plt.bar(x = list(range(0, 11)), height = frequent_itemsets.support[0:11], color ='rgmyk')
plt.xticks(list(range(0, 11)), frequent_itemsets.itemsets[0:11], rotation=20)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)

def to_list(i):
    return (sorted(list(i)))

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 3 rules 
rules_no_redudancy.sort_values('lift', ascending = False).head(3)

import matplotlib.pyplot as plt

x = [rules_no_redudancy.support]
y = [rules_no_redudancy.confidence]

plt.scatter(x, y)
plt.show()

#Insights:
 #From the books dataset, 6 rules were created using the apriori algorithm.
   #There are 3 different variables plotted in the scatterplot; confidence, support and lift. 
#The rules with the highest lift ratio lie within the constraint of support value of 0.56 to 0.74 and
#confidence value from 0.21 to 0.39
#We can suggest 3 combos as in the rules_no_redundancy object


#Q3

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("D:\\data\\my_movies.csv")

df1 = df.drop(['V1', 'V2','V3','V4','V5'], axis = 1)

frequent_itemsets = apriori(df1, min_support = 0.20, max_len = 4, use_colnames = True)

frequent_itemsets.sort_values('support', ascending = False, inplace = True)

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)

rules.head(20)
rules.sort_values('lift', ascending = False).head(10)

import matplotlib.pyplot as plt

x = [rules.support]
y = [rules.confidence]

plt.scatter(x, y)
plt.show()

def to_list(i):
    return (sorted(list(i)))

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 3 rules 
rules_no_redudancy.sort_values('lift', ascending = False).head(3)

import matplotlib.pyplot as plt

x = [rules_no_redudancy.support]
y = [rules_no_redudancy.confidence]

plt.scatter(x, y)
plt.show()


#Insights:
 #From the books dataset, 16 rules were created using the apriori algorithm.
   #There are 3 different variables plotted in the scatterplot; confidence, support and lift. 
#The rule with the highest lift ratio of 5 is people watch LOTR2 first then they watch LOTR1.
#Rest of the data can be used to create combos or suggestions accordingly

#Q4
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("D:\\data\\myphonedata.csv")

df1 = df.drop(['V1', 'V2','V3'], axis = 1)

frequent_itemsets = apriori(df1, min_support = 0.05, max_len = 4, use_colnames = True)

frequent_itemsets.sort_values('support', ascending = False, inplace = True)

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)

rules.head(20)
rules.sort_values('lift', ascending = False).head(10)

import matplotlib.pyplot as plt

x = [rules.support]
y = [rules.confidence]

plt.scatter(x, y)
plt.show()

def to_list(i):
    return (sorted(list(i)))

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 3 rules 
rules_no_redudancy.sort_values('lift', ascending = False).head(3)

import matplotlib.pyplot as plt

x = [rules_no_redudancy.support]
y = [rules_no_redudancy.confidence]

plt.scatter(x, y)
plt.show()

#Insights:
 #From the books dataset, 18 rules were created using the apriori algorithm.
   #There are 3 different variables plotted in the scatterplot; confidence, support and lift. 
#The Top 3 colours should be White,Orange and Red

#Q5
#Question not clear
#Will submit the anwswer later













