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
data = pd.read_csv('../../data/homework_tennis_playing_state.csv')

# 2.2 Encode Categorical Values -> Numerical Values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
outlook = data.iloc[:,0:1].values
outlook[:,0] = le.fit_transform(outlook[:,0])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features="all")
outlook = ohe.fit_transform(outlook).toarray()


nwindies = data.iloc[:,3:4].values
nwindies[:,0] = le.fit_transform(nwindies[:,0])


states = data.iloc[:,-1:].values
states[:,0] = le.fit_transform(states[:,0])

# 2.3 Create Dataframes
result = pd.DataFrame(data=outlook,index=range(14),columns=['Overcast','Rainy','Sunny'])


result2 = pd.DataFrame(data=data.iloc[:,1:3].values,index=range(14),columns=['Temperature','Humidity'])

result3 = pd.DataFrame(data=nwindies[:,0],index=range(14),columns=['Windy'])
