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
data = pd.read_csv('../../data/salaries.csv')

# 2.2 Split data as test and train data
from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(s,result3, test_size=0.33, random_state=0)

x = data.iloc[:,1:2]
y = data.iloc[:,2:]

X = x.values
Y = y.values

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, Y)




#plt.scatter(X,Y,color='red')
#plt.plot(X, lin_reg.predict(X), color='blue')
#plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_reg  = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color='red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)),color="blue")
plt.show()


# sample predictions
print(lin_reg2.predict(poly_reg.fit_transform(np.array([[6]]))))
print(lin_reg2.predict(poly_reg.fit_transform(np.array([[11]]))))



