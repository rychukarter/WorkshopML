import pandas as pd
import matplotlib.pyplot as plt

# Read data from file (into DataFrame)
data = pd.read_excel("mlr03.xls")
# Print data
print(data)
# Plot data and show the plot
#plt.figure(1)
data.plot()

# Advance plotting
plt.figure(2)
X = data["EXAM1"]
Y = data["FINAL"]
plt.plot(X, Y, "ro")
plt.show()
