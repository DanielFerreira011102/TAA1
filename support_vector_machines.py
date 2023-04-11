import pandas as pd
from sklearn.model_selection import train_test_split
from analytics import print_report
from models import train
from preprocessing import split_data, one_hot_encode, label_encode

data = pd.read_csv('bank.csv')

X, y = split_data(data)

X = one_hot_encode(X, mode='pandas')
y = label_encode(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)

gbm = train(X_train, y_train, name='support_vector_machines')

y_pred = gbm.predict(X_test)

print_report(y_test, y_pred)
