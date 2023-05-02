import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier

from leina.analytics import get_report, plot_confusion_matrix, plot_roc, plot_decision_tree, get_best, \
    plot_feature_importance
from leina.models import train
from leina.preprocessing import split_data, one_hot_encode, label_encode, ordinal_encode, standard_scale, full_clean
from utils import Logger, LogLevel

logger = Logger(level=LogLevel.INFO)

data = pd.read_csv('bank.csv')
new_data = full_clean(data)
X, y = split_data(new_data, target_col='target')
print(X, y)
# X = one_hot_encode(X)
features = X.columns
# y = label_encode(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)

accuracy_map = {}


# Logistic Regression
def run_logistic_regression():
    logger.info('Running Logistic Regression')

    lr = train(X_train, y_train, name='LogisticRegression', max_iter=10000)
    y_pred = lr.predict(X_test)

    accuracy_score, confusion_matrix, classification_report = get_report(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc(lr, X_test, y_test)

    accuracy_map['logistic_regression'] = accuracy_score


# Decision Tree
def run_decision_tree():
    logger.info('Running Decision Tree')

    clf = train(X_train, y_train, name='DecisionTreeClassifier')
    y_pred = clf.predict(X_test)

    accuracy_score, confusion_matrix, classification_report = get_report(y_test, y_pred)
    plot_decision_tree(clf, X.columns.values)

    accuracy_map['decision_tree'] = accuracy_score


# Gradient Boosting Machines
def run_gradient_boosting_machines():
    logger.info('Running Gradient Boosting Machines')

    gbm = train(X_train, y_train, name='GradientBoostingClassifier')
    y_pred = gbm.predict(X_test)

    accuracy_score, confusion_matrix, classification_report = get_report(y_test, y_pred)

    accuracy_map['gradient_boosting_machines'] = accuracy_score


# Support Vector Machines
def run_support_vector_machines():
    logger.info('Running Support Vector Machines')

    svm = train(X_train, y_train, name='SVC')
    y_pred = svm.predict(X_test)

    accuracy_score, confusion_matrix, classification_report = get_report(y_test, y_pred)

    accuracy_map['support_vector_machines'] = accuracy_score


# Random Forest
def run_random_forest():
    logger.info('Running Random Forest')

    rfc = train(X_train, y_train, name='RandomForestClassifier')
    y_pred = rfc.predict(X_test)

    accuracy_score, confusion_matrix, classification_report = get_report(y_test, y_pred)

    accuracy_map['random_forest'] = accuracy_score


# Naive Bayes
def run_naive_bayes():
    logger.info('Running Naive Bayes')

    nb = train(X_train, y_train, name='GaussianNB')
    y_pred = nb.predict(X_test)

    accuracy_score, confusion_matrix, classification_report = get_report(y_test, y_pred)

    accuracy_map['naive_bayes'] = accuracy_score


# K-Nearest Neighbours
def run_k_nearest_neighbours():
    logger.info('Running K-Nearest Neighbours')

    knn = train(X_train, y_train, name='KNeighborsClassifier')
    y_pred = knn.predict(X_test)

    accuracy_score, confusion_matrix, classification_report = get_report(y_test, y_pred)

    accuracy_map['k_nearest_neighbours'] = accuracy_score


# Adaboost
def run_adaboost():
    logger.info('Running Adaboost')

    ada = train(X_train, y_train, name='AdaBoostClassifier')
    y_pred = ada.predict(X_test)

    accuracy_score, confusion_matrix, classification_report = get_report(y_test, y_pred)

    accuracy_map['adaboost'] = accuracy_score


# Xgboost
def run_xgboost():
    logger.info('Running Xgboost')

    xgb = train(X_train, y_train, name='XGBClassifier')
    y_pred = xgb.predict(X_test)

    accuracy_score, confusion_matrix, classification_report = get_report(y_test, y_pred)

    accuracy_map['xgboost'] = accuracy_score

    plot_feature_importance(xgb, features)


# Multi Layer Perceptron
def run_multi_layer_perceptron():
    logger.info('Running Multi Layer Perceptron')

    mlp = train(X_train, y_train, name='MLPClassifier')
    y_pred = mlp.predict(X_test)

    accuracy_score, confusion_matrix, classification_report = get_report(y_test, y_pred)

    accuracy_map['multi_layer_perceptron'] = accuracy_score


# Ridge Classifier
def run_ridge_classifier():
    logger.info('Running Ridge Classifier')

    rc = train(X_train, y_train, name='RidgeClassifier')
    y_pred = rc.predict(X_test)

    accuracy_score, confusion_matrix, classification_report = get_report(y_test, y_pred)

    accuracy_map['ridge_classifier'] = accuracy_score


# Passive Aggressive Classifier
def run_passive_aggressive_classifier():
    logger.info('Running Passive Aggressive Classifier')

    pac = train(X_train, y_train, name='PassiveAggressiveClassifier')
    y_pred = pac.predict(X_test)

    accuracy_score, confusion_matrix, classification_report = get_report(y_test, y_pred)

    accuracy_map['passive_aggressive_classifier'] = accuracy_score


# Extremely Randomized Trees
def run_extremely_randomized_trees():
    logger.info('Running Extremely Randomized Trees')

    ert = train(X_train, y_train, name='ExtraTreesClassifier')
    y_pred = ert.predict(X_test)

    accuracy_score, confusion_matrix, classification_report = get_report(y_test, y_pred)

    accuracy_map['extremely_randomized_trees'] = accuracy_score


# LightGBM
def run_lightgbm_classifier():
    logger.info('Running LightGBM Classifier')

    lgbm = train(X_train, y_train, name='LGBMClassifier')
    y_pred = lgbm.predict(X_test)

    accuracy_score, confusion_matrix, classification_report = get_report(y_test, y_pred)

    accuracy_map['lightgbm_classifier'] = accuracy_score


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
        lr = train(X_train, y_train, name='LogisticRegression', max_iter=10000, penalty=penalty, C=c)
        y_pred = lr.predict(X_test)
        accuracy_score, confusion_matrix, classification_report = get_report(y_test, y_pred, out=False)
        lr_accuracy_map[c] = accuracy_score

    get_best(lr_accuracy_map)


# Loop K-Nearest Neighbours
def run_loop_k_nearest_neighbours():
    logger.info('Running Loop K-Nearest Neighbours')

    knn_accuracy_map = {}
    for k in range(1, 30):
        knn = train(X_train, y_train, name='KNeighborsClassifier', n_neighbors=k)
        y_pred = knn.predict(X_test)
        accuracy_score, confusion_matrix, classification_report = get_report(y_test, y_pred, out=False)
        knn_accuracy_map[k] = accuracy_score

    get_best(knn_accuracy_map)

# Best Xgboost
def run_best_xgboost():
    logger.info('Running Best Xgboost')

    # Step 1: Model selection
    xgb = XGBClassifier()
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 5, 7],
        'n_estimators': [50, 100, 200],
    }
    grid_search = GridSearchCV(xgb, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Step 2: Hyperparameter tuning
    best_params = grid_search.best_params_
    xgb = train(X_train, y_train, name="XGBClassifier", **best_params)

    # Step 3: Ensemble learning (if needed)

    # Step 4: Model evaluation
    y_pred = xgb.predict(X_test)
    accuracy_score, confusion_matrix, classification_report = get_report(y_test, y_pred)

    accuracy_map['best_xgboost_classifier'] = accuracy_score


if __name__ == "__main__":
    run_logistic_regression()
    run_decision_tree()
    run_gradient_boosting_machines()
    run_support_vector_machines()
    run_random_forest()
    run_naive_bayes()
    run_k_nearest_neighbours()
    run_adaboost()
    run_xgboost()
    run_multi_layer_perceptron()
    run_ridge_classifier()
    run_passive_aggressive_classifier()
    run_extremely_randomized_trees()
    run_lightgbm_classifier()
    # run_best_xgboost()
    run_ranking()
    run_loop_logistic_regression()
    run_loop_k_nearest_neighbours()
