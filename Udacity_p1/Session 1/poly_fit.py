# TODO: Add import statements
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Assign the data to predictor and outcome variables
# TODO: Load the data
train_data = pd.read_csv("/home/hzq/Udacity_p1/data/data.csv")
X = train_data[['Var_X']]
y = train_data[['Var_Y']]
# X = train_data['Var_X'].values.reshape(-1, 1)
# y = train_data['Var_Y'].values

# Create polynomial features
# TODO: Create a PolynomialFeatures object, then fit and transform the
# predictor feature
poly_feat = PolynomialFeatures(degree = 4)
X_poly = poly_feat.fit_transform(X)

# Make and fit the polynomial regression model
# TODO: Create a LinearRegression object and fit it to the polynomial predictor
# features
poly_model = LinearRegression().fit(X_poly, y)
y_fit = poly_model.predict(X_poly)
# plt.plot(X, y, 'bo-')
plt.plot(X, y, 'bo', X, y_fit, 'r+')
plt.show()
