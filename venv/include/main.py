import pandas as pd
import matplotlib.pyplot as plt
from utilities import utilities as utl
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression



print(data)
utl.scale_df(data)

# Read data from file (into DataFrame)
data = pd.read_excel("mlr03.xls")
# Print data
print(data)
# Plot data and show the plot
data.plot()
pirnt('dupa')


# Advance plotting
plt.figure()
Y = data["FINAL"]
X = data.drop("FINAL", axis=1)
plt.plot(X["EXAM1"], Y, "ro")
plt.plot(X["EXAM2"], Y, "go")
plt.plot(X["EXAM3"], Y, "bo")



# Data split
column_names = X.columns.values
scaler = MinMaxScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(X), columns=column_names, index=data.index)
print(data_scaled)
data_scaled.plot()


#plt.show()