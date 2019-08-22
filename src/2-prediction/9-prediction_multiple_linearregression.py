"""
@author Mehmet Emin PACA
"""

# 1 - Load Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2 - Data Preprocessing Steps

# 2.1 - Load data set
data = pd.read_csv('../../data/sample_dataset.csv')




# 2.3 Encode Categorical Values -> Numeric Values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
countries = data.iloc[:,0:1].values
countries[:,0] = le.fit_transform(countries[:,0])


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features="all")
countries = ohe.fit_transform(countries).toarray()



genders = data.iloc[:,-1:].values
genders[:,0] = le.fit_transform(genders[:,0])
ohe = OneHotEncoder(categorical_features="all")
genders = ohe.fit_transform(genders).toarray()

Age = data.iloc[:,1:4].values

# 2.4 Create Data Frames
result = pd.DataFrame(data=countries, index=range(22), columns=['fr','tr','us'])

result2 = pd.DataFrame(data=Age, index=range(22), columns=['Height','Weight','Age'])

result3 = pd.DataFrame(data=genders[:,:1], index=range(22),columns=['Gender'])

# 2.5 Concatenate Data Frames
s = pd.concat([result,result2],axis=1)
s2 = pd.concat([s,result3],axis=1)

# 2.6 Split data as test and train data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s,result3, test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

heights = s2.iloc[:,3:4].values
print(heights)

leftSide = s2.iloc[:,:3]
rightSide = s2.iloc[:,4:]

newData = pd.concat([leftSide,rightSide], axis=1)

x_train,x_test,y_train,y_test = train_test_split(newData,heights,test_size=0.33,random_state=0)

regressor2 = LinearRegression()
regressor2.fit(x_train,y_train)

y_pred2 = regressor2.predict(x_test)


import statsmodels.api as sm
X = np.append(arr=np.ones((22,1)).astype(int),values=newData, axis=1)
X_l = newData.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog=heights, exog=X_l)
r = r_ols.fit()
print(r.summary())

# Remove column 4 because P-Value = 0.717 > 0.05
# [look previous summary-backward elimination requires removing big P values> SL]
# if we use forward selection  we add P < SL columns one by one
X_l = newData.iloc[:,[0,1,2,3,5]].values
r_ols = sm.OLS(endog=heights, exog=X_l)
r = r_ols.fit()
print(r.summary())