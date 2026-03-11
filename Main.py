# Step 1 – Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2 – Load dataset
df = pd.read_csv("House_Data.csv")

# Step 3 – Check data
df.head()
df.isnull().sum()

# Step 4 – Remove missing values
df.dropna(inplace=True)
df.isnull().sum()
df.shape

# Step 5 – Select columns
# X = Size of house (sqft)
# Y = Price of house
x = df[['Area']]      # Example column: Area (sq ft)
y = df['Price']       # Example column: Price

# Step 6 – Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Step 7 – Linear Regression model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Step 8 – Prediction
y_pred = lr.predict(X_test)
y_pred
X_test
y_test

# Step 9 – Accuracy (R² score)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
r2

# Step 10 – Equation of line
m = lr.coef_
m

c = lr.intercept_
c

# Example: price for 1500 sq ft house
m * 1500 + c

# Step 11 – Visualization
sns.scatterplot(x='Area', y='Price', data=df)
plt.plot(X_test, lr.predict(X_test), color='red')
plt.show()

# Step 12 – Error
from sklearn.metrics import mean_squared_error
print("MSE:", mean_squared_error(y_test, y_pred))
