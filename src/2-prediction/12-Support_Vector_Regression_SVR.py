"""
@author Mehmet Emin PACA
"""

# 1 - Load Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2 - Data Preprocessing Steps

# 2.1 - Load data set
data = pd.read_csv('../../data/salaries.csv')

# 2.2 Slice Data Frames
# get education level column
x = data.iloc[:,1:2]
# get salary column
y = data.iloc[:,2:]

# convert dataframe to numpy array
X = x.values
Y = y.values

# linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)
plt.scatter(X,Y,color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.show()

# polynomial regression(2nd degree)
from sklearn.preprocessing import PolynomialFeatures
poly_reg2  = PolynomialFeatures(degree=2)
x_poly2 = poly_reg2.fit_transform(X)
#print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly2,y)
plt.scatter(X,Y,color='red')
plt.plot(X, lin_reg2.predict(poly_reg2.fit_transform(X)),color="blue")
plt.show()


# polynomial regression (4th degree)
poly_reg4  = PolynomialFeatures(degree=4)
x_poly4 = poly_reg4.fit_transform(X)
#print(x_poly)
lin_reg4 = LinearRegression()
lin_reg4.fit(x_poly4,y)
plt.scatter(X,Y,color='red')
plt.plot(X, lin_reg4.predict(poly_reg4.fit_transform(X)),color="blue")
plt.show()


# sample predictions
print(lin_reg2.predict(poly_reg2.fit_transform(np.array([[6]]))))
print(lin_reg2.predict(poly_reg2.fit_transform(np.array([[11]]))))

# sample predictions
print(lin_reg4.predict(poly_reg4.fit_transform(np.array([[6]]))))
print(lin_reg4.predict(poly_reg4.fit_transform(np.array([[11]]))))

from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

from sklearn.svm import  SVR
svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')
plt.title('SVR')
plt.show()

print(svr_reg.predict(np.array([[11]])))
print(svr_reg.predict(np.array([[6.6]])))
