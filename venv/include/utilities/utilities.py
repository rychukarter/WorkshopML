import pandas as pd
import numpy as np


def scale_df(data):
    data = pd.DataFrame(data)
    return data.apply(lambda x: (x - data.min())/(data.max() - data.min()), axis=1)


def cost_function(X, y, theta):
    return (1/(2*y.size)) * np.sum((np.dot(X.values, theta).T - y.values) ** 2)


def gradient_descent(X, y, theta, alpha, num_iters):
    theta_temp = theta
    for i in range(num_iters):
        for n in range(theta.size):
            delta = np.sum((np.dot(X.values, theta).T - y.values) * X.values[:, n])
            theta_temp[n] = theta[n] - (alpha/y.size) * delta
        theta = theta_temp
    return theta


def linear_regression():

    pass
