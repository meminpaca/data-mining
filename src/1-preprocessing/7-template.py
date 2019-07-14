"""
@author Mehmet Emin PACA
"""

# 1 - Load Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer

# 2 - Data Preprocessing Steps

# 2.1 - Load data set
data = pd.read_csv('../../data/missing_values.csv')

# 2.2 Fill Missing Values
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
Age = data.iloc[:,1:4].values
imputer = imputer.fit(Age[:,1:4])
Age[:,1:4]=imputer.transform(Age[:,1:4])
countries = data.iloc[:,0:1].values

# 2.3 Encode Categorical Values -> Numeric Values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
countries[:,0] = le.fit_transform(countries[:,0])


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features="all")
countries = ohe.fit_transform(countries).toarray()

# 2.4 Create Data Frames
result = pd.DataFrame(data=countries, index=range(22), columns=['fr','tr','us'])

result2 = pd.DataFrame(data=Age, index=range(22), columns=['Height','Weight','Age'])

gender = data.iloc[:,-1].values
result3 = pd.DataFrame(data=gender, index=range(22),columns=['Gender'])

# 2.5 Concatenate Data Frames
s = pd.concat([result,result2],axis=1)
s2 = pd.concat([s,result3],axis=1)

# 2.6 Split data as test and train data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s,result3, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# 2.7 Scale data using Standardisation method
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

print(X_train)

print(X_test)