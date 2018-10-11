import pandas as pd


def scale_df(data):
    data = pd.DataFrame(data)
    return data.apply(lambda x: (x - data.min())/(data.max() - data.min()), axis=1)


def linear_regression():
    pass
