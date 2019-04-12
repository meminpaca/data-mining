"""
@author Mehmet Emin PACA
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#codes
# load data set
data = pd.read_csv('data/sample_dataset.csv')

print(data)

#data preprocessing
heightWeight = data[['height','weight']]
print(heightWeight)


