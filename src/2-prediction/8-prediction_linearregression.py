#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 04:18:20 2018

"""

#1. Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Data Pre-Processing

#2.1. Data Loading
data = pd.read_csv('../../data/sales_dataset.csv')

#get only monht column
months = data[["Month"]]

#get only Sales column
sales = data[["Sales"]]


#test


#Split data as train and test
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(months,sales,test_size=0.33, random_state=0)

# build model (linear regression)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)

prediction = lr.predict(x_test)
print(prediction)

#Sort for sequential display on the chart, sorts by index of record
x_train = x_train.sort_index()
y_train = y_train.sort_index()
plt.plot(x_train,y_train)

plt.plot(x_test,prediction)
plt.title("Monthly Sales")
plt.xlabel("Months")
plt.ylabel("Sales")
plt.show()

    

