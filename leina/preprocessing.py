import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler


def split_data(data, target_col=None):
    data = data.dropna()
    if target_col is None:
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
    else:
        X = data.drop(target_col, axis=1)
        y = data[target_col]
    return X, y


def label_encode(target, mode='sklearn'):
    if mode == 'sklearn':
        return LabelEncoder().fit_transform(target)

def one_hot_encode(X, mode='pandas', **kwargs):
    if mode == 'pandas':
        return pd.get_dummies(X, **kwargs)
    if mode == 'sklearn':
        return OneHotEncoder().fit_transform(X, **kwargs)

def ordinal_encode(X, mode='sklearn', **kwargs):
    if mode == 'sklearn':
        return OrdinalEncoder().fit_transform(X, **kwargs)

def standard_scale(X, mode='sklearn', **kwargs):
    if mode == 'sklearn':
        return StandardScaler().fit_transform(X, **kwargs)