import pandas as pd
from sklearn.model_selection import train_test_split
from leina.analytics import get_report, plot_confusion_matrix, plot_roc, plot_decision_tree, get_best
from leina.models import train
from leina.preprocessing import split_data, one_hot_encode, label_encode
from utils import Logger, LogLevel

logger = Logger(level=LogLevel.INFO)

data = pd.read_csv('bank.csv')
X, y = split_data(data)
X = one_hot_encode(X, mode='pandas')
y = label_encode(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)

accuracy_map = {}


# Logistic Regression
def run_logistic_regression():
    logger.info('Running Logistic Regression')

    lr = train(X_train, y_train, name='logistic_regression', max_iter=10000)
    y_pred = lr.predict(X_test)

    accuracy_score, confusion_matrix, classification_report = get_report(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc(lr, X_test, y_test)

    accuracy_map['logistic_regression'] = accuracy_score


# Decision Tree
def run_decision_tree():
    logger.info('Running Decision Tree')

    clf = train(X_train, y_train, name='decision_tree')
    y_pred = clf.predict(X_test)

    accuracy_score, confusion_matrix, classification_report = get_report(y_test, y_pred)
    plot_decision_tree(clf, X.columns.values)

    accuracy_map['decision_tree'] = accuracy_score


# Gradient Boosting Machines
def run_gradient_boosting_machines():
    logger.info('Running Gradient Boosting Machines')

    gbm = train(X_train, y_train, name='gradient_boosting_machines')
    y_pred = gbm.predict(X_test)

    accuracy_score, confusion_matrix, classification_report = get_report(y_test, y_pred)

    accuracy_map['gradient_boosting_machines'] = accuracy_score


# Support Vector Machines
def run_support_vector_machines():
    logger.info('Running Support Vector Machines')

    svm = train(X_train, y_train, name='support_vector_machines')
    y_pred = svm.predict(X_test)

    accuracy_score, confusion_matrix, classification_report = get_report(y_test, y_pred)

    accuracy_map['support_vector_machines'] = accuracy_score


# Random Forest
def run_random_forest():
    logger.info('Running Random Forest')

    rfc = train(X_train, y_train, name='random_forest')
    y_pred = rfc.predict(X_test)

    accuracy_score, confusion_matrix, classification_report = get_report(y_test, y_pred)

    accuracy_map['random_forest'] = accuracy_score


# Naive Bayes
def run_naive_bayes():
    logger.info('Running Naive Bayes')

    nb = train(X_train, y_train, name='naive_bayes')
    y_pred = nb.predict(X_test)

    accuracy_score, confusion_matrix, classification_report = get_report(y_test, y_pred)

    accuracy_map['naive_bayes'] = accuracy_score


# K-Nearest Neighbours
def run_k_nearest_neighbours():
    logger.info('Running K-Nearest Neighbours')

    knn = train(X_train, y_train, name='k_nearest_neighbours')
    y_pred = knn.predict(X_test)

    accuracy_score, confusion_matrix, classification_report = get_report(y_test, y_pred)

    accuracy_map['k_nearest_neighbours'] = accuracy_score


# Ranking
def run_ranking():
    logger.info('Ranking')

    get_best(accuracy_map)


# Loop Logistic Regression
def run_loop_logistic_regression():
    logger.info('Running Loop Logistic Regression')

    penalty = "l2"
    C = [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1, 2, 5]
    lr_accuracy_map = {}
    for c in C:
        lr = train(X_train, y_train, name='logistic_regression', max_iter=10000, penalty=penalty, C=c)
        y_pred = lr.predict(X_test)
        accuracy_score, confusion_matrix, classification_report = get_report(y_test, y_pred, out=False)
        lr_accuracy_map[c] = accuracy_score

    get_best(lr_accuracy_map)


# Loop K-Nearest Neighbours
def run_loop_k_nearest_neighbours():
    logger.info('Running Loop K-Nearest Neighbours')

    knn_accuracy_map = {}
    for k in range(1, 30):
        knn = train(X_train, y_train, name='k_nearest_neighbours', n_neighbors=k)
        y_pred = knn.predict(X_test)
        accuracy_score, confusion_matrix, classification_report = get_report(y_test, y_pred, out=False)
        knn_accuracy_map[k] = accuracy_score

    get_best(knn_accuracy_map)


if __name__ == "__main__":
    run_logistic_regression()
    run_decision_tree()
    run_gradient_boosting_machines()
    run_support_vector_machines()
    run_random_forest()
    run_naive_bayes()
    run_k_nearest_neighbours()
    run_ranking()
    run_loop_logistic_regression()
    run_loop_k_nearest_neighbours()
