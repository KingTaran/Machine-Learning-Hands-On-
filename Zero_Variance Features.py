# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 13:47:34 2021

@author: taran
"""

import pandas as pd

df = pd.read_csv('D:/Data/Z_dataset.csv')

df.var()

#HERE SQUARE_LENGTH,SQUARE_BREADTH,REC_BREADTH
#HAS VARIANCES CLOSE TO 0 BUT REC_LENGTH HAS HIGH VARIANCE
#MAKING IT IMPORTANT TO ANALYZE