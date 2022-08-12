#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 13:19:17 2022

@author: davor
"""
import io
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import pingouin as pg
from scipy import stats
from tensorflow.python.client import device_lib
from statsmodels.tsa.stattools import adfuller

companies = ['Комерцијална банка Скопје', 'Алкалоид Скопје','Гранит Скопје','Макпетрол Скопје','Македонијатурист Скопје']
companies_short_names= ['KMB','ALK','GRNT','MPT','MTUR']
info =[]
description_columns = ['Company','count','mean','min','max','std','ADF Test(p)','Statistic:','Critical Value:1pc'  ]

for i in range(len(companies)):    
    df = pd.read_csv(companies[i]+'.csv')
    description=[]
    description.append(companies_short_names[i])

    des= df['open'].describe()
    description.append(des['count'])
    description.append(des['mean'])
    description.append(des['min'])
    description.append(des['max'])
    description.append(des['std'])
    
    result = adfuller(df['open'].values)
    description.append('%.3f' % result[1])    
    description.append('%.3f' % result[0])
    description.append('%.3f' % result[4]['1%'])    
    
    info.append(description)
frame = pd.DataFrame(info, columns = description_columns)
frame.to_csv("descriptive.csv", encoding='utf-8')            

print(frame)
