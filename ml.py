import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Read the data 
data = pd.read_csv('/Users/vivienyu/Desktop/ml_data.csv') # File Path

# Replace -1111.1 with 0 in the dataset
data.replace(-1111.1, 0, inplace=True)
data.fillna(0, inplace=True)

column_ranges = data.describe().loc[['min', 'max']]
print("Column Ranges:")
print(column_ranges)

# Extract the input and target values
X = data.iloc[:, 0].values
y1 = data.iloc[:, 1].values.reshape(-1, 1)
y2 = data.iloc[:, 2].values.reshape(-1, 1)
y3 = data.iloc[:, 3].values.reshape(-1, 1)

# Perform stratified train-test split
X_train1, X_test1, y1_train, y1_test = train_test_split(y1, X, test_size=0.2, random_state=42)
X_train2, X_test2, y2_train, y2_test = train_test_split(y2, X, test_size=0.2, random_state=42)
X_train3, X_test3, y3_train, y3_test = train_test_split(y3, X, test_size=0.2, random_state=42)

######### HIGH BLOOD PRESSURE ##########

# Create a linear regression model
model1 = LinearRegression()

# Train the model
model1 = model1.fit(X_train1, y1_train)

# Make predictions for the test data
predictions1 = model1.predict(X_test1)

# Calculate the mean squared error and r-squared value
mse1 = mean_squared_error(y1_test, predictions1)
print("High Blood Pressure Mean Squared Error:", mse1)

r21 = r2_score(y1_test, predictions1)
print("R-squared:", r21)

######### ACCESS TO PRIMARY CARE ##########

# Create a linear regression model
model2 = LinearRegression()

# Train the model
model2.fit(X_train2, y2_train)

# Make predictions for the test data
predictions2 = model2.predict(X_test2)

# Calculate the mean squared error and r-squared value
mse2 = mean_squared_error(y2_test, predictions2)
print("Physician Mean Squared Error:", mse2)

r22 = r2_score(y2_test, predictions2)
print("R-squared:", r22)

######### DIABETES ##########

model3 = LinearRegression()

# Train the model
model3.fit(X_train3, y3_train)

# Make predictions for the test data
predictions3 = model3.predict(X_test3)

# Calculate the mean squared error and r-squared value
mse3 = mean_squared_error(y3_test, predictions3)
print("Diabetes Mean Squared Error:", mse3)

r23 = r2_score(y3_test, predictions3)
print("R-squared:", r23)
