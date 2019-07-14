"""
@author Mehmet Emin PACA
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer

#codes
# load data set
data = pd.read_csv('../../data/missing_values.csv')

imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)

Age = data.iloc[:,1:4].values
print(Age)

imputer = imputer.fit(Age[:,1:4])

Age[:,1:4]=imputer.transform(Age[:,1:4])

print(Age)




