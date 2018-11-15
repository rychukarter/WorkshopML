import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utilities import utilities as utl
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# Read data from file (into DataFrame)
data = pd.read_excel("mlr03.xls")
# data = shuffle(data)
# Print data
print(data)

# Plot data and show the plot
data.plot()


# Advance plotting
plt.figure()
Y = data["FINAL"]
X = data.drop("FINAL", axis=1)
plt.plot(X["EXAM1"], Y, "ro")
plt.plot(X["EXAM2"], Y, "go")
plt.plot(X["EXAM3"], Y, "bo")
plt.show()


# Data scaling
X_scaled = utl.scale_df(X)
print(X_scaled)
column_names = X.columns.values
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=column_names, index=data.index)
print(X_scaled)

# Data split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, shuffle=False)


theta = np.array([1.0, 1.0, 1.0])
num_iters = 1000
alpha = 0.00001

J = utl.cost_function(X_train, Y_train, theta)
print("Starting theta: ", theta)
print("Starting J = ", J)
theta = utl.gradient_descent(X_train, Y_train, theta, alpha, num_iters)
J = utl.cost_function(X_train, Y_train, theta)
print("After GD theta: ", theta)
print("After GD J = ", J)
Y_predicted = np.dot(X_test, theta)
print(Y_test.values)
print(Y_predicted)
print(r2_score(Y_test, Y_predicted))

# Linear Regression training
regression = LinearRegression(normalize=False)
regression.fit(X_train, Y_train)
J = utl.cost_function(X_train, Y_train, regression.coef_)
print("Theta calculated by sklear: ", regression.coef_)
print("J of sklearn: ", J)
# Linear Regression test
Y_predicted_sk = regression.predict(X_test)
plt.figure()
plt.plot(X_test["EXAM1"], Y_test, "go")
plt.plot(X_test["EXAM1"], Y_predicted_sk, "bx")
plt.plot(X_test["EXAM1"], Y_predicted, "rx")
plt.show()
print(r2_score(Y_test, Y_predicted_sk))
utl.plot_error_histogram(Y_test, Y_predicted_sk)
plt.show()
