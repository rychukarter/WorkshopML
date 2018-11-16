import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

# 1. Prepare some fake data
data = np.array([[1, 2], [2, 1], [2, 2], [3, 4], [4, 3], [3, 3]])
labels = ["red", "red", "red", "blue", "blue", "blue"]
plt.scatter(data[:, 0], data[:, 1], c=labels)
plt.show()

# # 2. Create and fit linear SVM Classifier
# svm_classifier = LinearSVC(C=100, verbose=1)
# svm_classifier.fit(data, labels)
# print(svm_classifier.get_params())

# # 3. Make some test prediction
# print(svm_classifier.predict([[3, 1]]))

# # 4. Model explanation
# print(svm_classifier.coef_)
# print(svm_classifier.intercept_)

# # 5. Plot decision plane
# w = svm_classifier.coef_[0]
# a = -w[0] / w[1]
# x = np.linspace(0, 5)
# y = a * x - svm_classifier.intercept_[0] / w[1]
# plt.plot(x, y)
# plt.show()
