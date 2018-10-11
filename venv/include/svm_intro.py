import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

data = np.array([[1, 2], [2, 1], [2, 2], [3, 4], [4, 3], [3, 3]])
labels = ["red", "red", "red", "blue", "blue", "blue"]

plt.scatter(data[:, 0], data[:, 1], c=labels)
# plt.show()

svm_classifier = LinearSVC()
svm_classifier.fit(data, labels)
print(svm_classifier.predict([[3, 1]]))

# Model explanation
print(svm_classifier.coef_)
print(svm_classifier.intercept_)

w = svm_classifier.coef_[0]
a = -w[0] / w[1]
x = np.linspace(0, 5)
y = a * x - svm_classifier.intercept_[0] / w[1]
plt.plot(x, y)
plt.show()
