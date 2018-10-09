import pandas as pd


def scale(x):
    return (x - x.min) / (x.max - x.min)


def scale_df(data):
    data = pd.DataFrame(data)
    print(data.apply(lambda x: (x - data.min())/(data.max() - data.min()), axis=1))


#X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
#X_scaled = X_std * (max - min) + min