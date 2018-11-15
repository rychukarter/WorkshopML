import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


def plot_error_histogram(y_test, y_pred, title="Prediction error histogram"):
    error = y_test - y_pred
    plt.hist(error, 5, facecolor='g')
    plt.xlabel('Prediction error')
    plt.ylabel('Count')
    plt.title(title)
    plt.grid(True)
    return plt


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, '-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, '-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
