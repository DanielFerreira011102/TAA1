import pandas as pd
from sklearn.model_selection import train_test_split
from analytics import print_report, plot_decision_tree
from models import train
from preprocessing import split_data, one_hot_encode, label_encode

data = pd.read_csv('bank.csv')

X, y = split_data(data)

X = one_hot_encode(X, mode='pandas')
y = label_encode(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)

clf = train(X_train, y_train, name='decision_tree')

y_pred = clf.predict(X_test)

print_report(y_test, y_pred)

plot_decision_tree(clf, X.columns.values)
