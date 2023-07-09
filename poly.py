import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Read the data from a CSV file
data = pd.read_csv('/Users/vivienyu/Desktop/academics/2B/stats_project/ml_data.csv')

# Replace -1111.1 with 0 in the dataset
data.replace(-1111.1, 0, inplace=True)
data.fillna(0, inplace=True)

column_ranges = data.describe().loc[['min', 'max']]
print("Column Ranges:")
print(column_ranges)

# Extract the input features (columns 0, 1, 2) and target values (column 3)
X = data.iloc[:, 0].values
y1 = data.iloc[:, 1].values.reshape(-1, 1)
y2 = data.iloc[:, 2].values.reshape(-1, 1)
y3 = data.iloc[:, 3].values.reshape(-1, 1)

# Perform stratified train-test split
X_train1, X_test1, y1_train, y1_test = train_test_split(y1, X, test_size=0.2, random_state=42)
X_train2, X_test2, y2_train, y2_test = train_test_split(y2, X, test_size=0.2, random_state=42)
X_train3, X_test3, y3_train, y3_test = train_test_split(y3, X, test_size=0.2, random_state=42)

# Apply polynomial features transformation
degree = 2  # Set the degree of the polynomial
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train1)
X_test_poly = poly.transform(X_test1)

# Create and train polynomial regression models for each target variable
model1 = LinearRegression()
model1.fit(X_train_poly, y1_train)

model2 = LinearRegression()
model2.fit(X_train_poly, y2_train)

model3 = LinearRegression()
model3.fit(X_train_poly, y3_train)

# Make predictions for the test data
predictions1 = model1.predict(X_test_poly)
predictions2 = model2.predict(X_test_poly)
predictions3 = model3.predict(X_test_poly)

# Calculate the mean squared error
mse1 = mean_squared_error(y1_test, predictions1)
mse2 = mean_squared_error(y2_test, predictions2)
mse3 = mean_squared_error(y3_test, predictions3)

# Calculate the R-squared
r21 = r2_score(y1_test, predictions1)
r22 = r2_score(y2_test, predictions2)
r23 = r2_score(y3_test, predictions3)

print("High Blood Pressure Mean Squared Error:", mse1)
print("High Blood Pressure R-squared:", r21)
print("Smoker Mean Squared Error:", mse2)
print("Smoker R-squared:", r22)
print("Diabetes Mean Squared Error:", mse3)
print("Diabetes R-squared:", r23)
