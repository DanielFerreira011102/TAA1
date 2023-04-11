import pandas as pd
from sklearn.model_selection import train_test_split
from analytics import print_report, plot_confusion_matrix, plot_roc
from models import train
from preprocessing import split_data, one_hot_encode, label_encode

data = pd.read_csv('bank.csv')

X, y = split_data(data)

X = one_hot_encode(X, mode='pandas')
y = label_encode(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)

model = train(X_train, y_train, name='logistic_regression')

y_pred = model.predict(X_test)

print_report(y_test, y_pred)

plot_confusion_matrix(y_test, y_pred)

plot_roc(model, X_test, y_test)

