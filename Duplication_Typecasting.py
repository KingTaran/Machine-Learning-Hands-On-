# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 04:57:21 2021

@author: taran
"""

import pandas as pd

data = pd.read_csv("C:/Data/OnlineRetail.csv")
data.dtypes

#Typecasting
data.UnitPrice = data.UnitPrice.astype('int64')

data = data.fillna(0)
data.CustomerID = data.CustomerID.astype('int64')

#Duplication
duplicate = data.duplicated()

data1 = data.drop_duplicates()

data1


#EDA
#Grouping countries by Total Amount of sales

country_price = data.groupby('Country')['Quantity'].sum().sort_values(ascending = False)
country_price
# Top 5 Companies with high number of purchase
country_price[:5].plot(kind = 'bar')
# Top 5 Compaies with least number of purchase
country_price[33:].plot(kind = 'bar')

# Sales for different months

# For converting invoice date from data type object to date time
data[["InvoiceDate"]] = data[["InvoiceDate"]].apply(pd.to_datetime)
# Adding year feature to the dataset 
timest = data['InvoiceDate'].dt.year
data['Year'] = timest
# Adding Total Amount Column
TotalAmount = data['Quantity'] * data['UnitPrice']
data.insert(loc=5,column='TotalAmount',value=TotalAmount)


data['Mon'] = data['InvoiceDate'].dt.month
data['month'] = data['InvoiceDate'].dt.month_name() 
data.groupby(['Mon','Year'])['TotalAmount'].sum().plot(kind = 'bar', title = 'Sales month wise')
