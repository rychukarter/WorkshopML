import pandas as pd
import matplotlib.pyplot as plt
from utilities import utilities as utl
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Read data from file (into DataFrame)
data = pd.read_excel("mlr03.xls")
data = shuffle(data)
# Print data
print(data)

# Plot data and show the plot
data.plot()
plt.show()

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

# Linear Regression training
regression = LinearRegression(normalize=False)
regression.fit(X_train, Y_train)

# Linear Regression test
Y_predicted = regression.predict(X_test)
plt.figure()
plt.plot(X_test["EXAM1"], Y_test, "ro")
plt.plot(X_test["EXAM1"], Y_predicted, "bx")
plt.show()