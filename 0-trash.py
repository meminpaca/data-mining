import pandas as pd
import numpy as np

data = pd.read_csv("data/missing_values.csv")

from sklearn.impute import SimpleImputer



my_imputer = SimpleImputer(missing_values="NaN",strategy="mean")

Age = data.iloc[:,1:4].values()

#Age[:,1:4] = my_imputer.fit_transform(Age[:,1:4])
print(Age)