# 0. Imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import time
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# 1. Import data as data frame
print("Step 1: import data as data frame")

df = pd.read_csv("data/poker/poker-hand-data.csv")
df = df.sample(n=10000, random_state=100)
print(df.info())
print(df.head())


# 2. Data normalization and scaling
print("\nStep 2: Data normalization and scaling")

scaler = MinMaxScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=list(df.columns.values))
print(scaled_df.head())


# 3. Train test split
print("\nStep 3: split data by train test split")

X = np.array(scaled_df.drop('Class', axis=1))
y = np.array(df['Class'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
print(X_train.size)
print(X_test.size)

# # 4. Linear SVM method
# from sklearn.svm import LinearSVC
# print("\nStep 4: linear SVM classifier training")
#
# start = time.time()
# lin_cls = LinearSVC(C=5.0, random_state=100)
# lin_cls.fit(X_train, y_train)
# end = time.time()
# print(lin_cls.get_params())
# print("Learning time: " + str(int(end - start)) + " seconds")

# # 4a. Linear SVM accuracy evaluation
# print("\nStep 4a: linear SVM classifier accuracy evaluation")
# y_predict = lin_cls.predict(X_test)
# errors = 0
# for i in range(0, y_test.size):
#     if y_predict[i] != y_test[i]:
#         errors += 1
# print("Accuracy: " + str(1 - errors/np.size(y_test)))
# print("Accuracy by library: " + str(lin_cls.score(X_test, y_test)))

# # 5. Kernel based SVM
# print("\nStep 5: nonlinear SVM classifier training")
#
# start = time.time()
# rbf_cls = SVC(C=5.0, kernel='rbf', random_state=100)
# rbf_cls.fit(X_train, y_train)
# end = time.time()
# print("Learning time: " + str(int(end - start)) + " seconds")


# # 5a. Kernel based SVM evaluation
# print("\nStep 5a: nonlinear SVM classifier evaluation")
# y_predict = rbf_cls.predict(X_test)
# errors = 0
# for i in range(0, y_test.size):
#     if y_predict[i] != y_test[i]:
#         errors += 1
# print("Accuracy: " + str(1 - errors/np.size(y_test)))
# print("Accuracy by library: " + str(rbf_cls.score(X_test, y_test)))


# 6. Multilayer perceptron
print("\n6. Multilayer perceptron")
start = time.time()
mlp_cls = MLPClassifier()
mlp_cls.fit(X_train, y_train)
end = time.time()
print("Learning time: " + str(int(end - start)) + " seconds")

print(mlp_cls.get_params())

print("Accuracy: " + str(mlp_cls.score(X_test, y_test)))