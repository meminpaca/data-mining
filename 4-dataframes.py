"""
@author Mehmet Emin PACA
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer

#codes
# load data set
data = pd.read_csv('data/missing_values.csv')

imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)

Age = data.iloc[:,1:4].values
#print(Age)

imputer = imputer.fit(Age[:,1:4])

Age[:,1:4]=imputer.transform(Age[:,1:4])

#print(Age)


countries = data.iloc[:,0:1].values

print(countries)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
countries[:,0] = le.fit_transform(countries[:,0])
print(countries)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features="all")
countries = ohe.fit_transform(countries).toarray()
print(countries)

result = pd.DataFrame(data=countries, index=range(22), columns=['fr','tr','us'])
print(result)

result2 = pd.DataFrame(data=Age, index=range(22), columns=['Height','Weight','Age'])
print(result2)

gender = data.iloc[:,-1].values
print(gender)

result3 = pd.DataFrame(data=gender, index=range(22),columns=['Gender'])
print(result3)

s = pd.concat([result,result2],axis=1)

s2 = pd.concat([s,result3],axis=1)

