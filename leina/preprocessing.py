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


def full_clean(df):
    df['is_default'] = df['default'].apply(lambda row: 1 if row == 'yes' else 0)
    df['is_housing'] = df['housing'].apply(lambda row: 1 if row == 'yes' else 0)
    df['is_loan'] = df['loan'].apply(lambda row: 1 if row == 'yes' else 0)
    df['target'] = df['deposit'].apply(lambda row: 1 if row == 'yes' else 0)

    marital_dummies = pd.get_dummies(df['marital'], prefix='marital')
    marital_dummies.drop('marital_divorced', axis=1, inplace=True)
    df = pd.concat([df, marital_dummies], axis=1)

    job_dummies = pd.get_dummies(df['job'], prefix='job')
    job_dummies.drop('job_unknown', axis=1, inplace=True)
    df = pd.concat([df, job_dummies], axis=1)

    education_dummies = pd.get_dummies(df['education'], prefix='education')
    education_dummies.drop('education_unknown', axis=1, inplace=True)
    df = pd.concat([df, education_dummies], axis=1)

    contact_dummies = pd.get_dummies(df['contact'], prefix='contact')
    contact_dummies.drop('contact_unknown', axis=1, inplace=True)
    df = pd.concat([df, contact_dummies], axis=1)

    poutcome_dummies = pd.get_dummies(df['poutcome'], prefix='poutcome')
    poutcome_dummies.drop('poutcome_unknown', axis=1, inplace=True)
    df = pd.concat([df, poutcome_dummies], axis=1)

    months = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10,
              'nov': 11, 'dec': 12}
    df['month'] = df['month'].map(months)

    df.drop(['job', 'education', 'marital', 'default', 'housing', 'loan', 'contact', 'poutcome', 'deposit'],
            axis=1, inplace=True)

    return df
