# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 17:59:04 2021

@author: taran
"""
#1
word = "Grow Grattitude"

word

letter=word[0]

letter

len(word)

print(word.count('G'))

#2
word="Being aware of a single shortcoming within yourself is far more useful than being aware of a thousand in someone else"
count=0

#Counts each character except space  
for i in range(0, len(word)):  
    if(word[i] != ' '):  
        count = count + 1;  
   
#Displays the total number of characters present in the given string  
print("Total number of characters in a string: " + str(count));

#3
word="Idealistic as it may sound,altruism should be the driving force in business,not just competition and a desire for wealth"


print(word[0]) #get any char from string

print (word[:3]) #get the first three char

print (word[-3:]) #get the last three char

#4

word = "stay positive and optimistic"

result = word.startswith('H')
print (result)

result = word.endswith('d')
print (result)

result = word.endswith('c')
print (result)

#5

print('[]' * 108)

#7

word = "Grow Grattitude"

word.replace('Grow','Growth of')

#8

def printInSortedOrder(arr, n):
    index = [0] * n
     
    # Initially the index of the strings
    # are assigned to the 'index[]'
    for i in range(n):
        index[i] = i
     
    # selection sort technique is applied
    for i in range(n - 1):
        min = i
        for j in range(i + 1, n):
             
            # with the help of 'index[]'
            # strings are being compared
            if (arr[index[min]] > arr[index[j]]):
                min = j
         
        # index of the smallest string is placed
        # at the ith index of 'index[]'
        if (min != i):
            index[min], index[i] = index[i], index[min]
     
    # printing strings in sorted order
    for i in range(n):
        print(arr[index[i]], end = " ")
 
#Driver code
    arr=[]
    n = 4
    printInSortedOrder(arr, n)    















