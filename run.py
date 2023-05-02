import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier

from leina.analytics import get_report, plot_confusion_matrix, plot_roc, plot_decision_tree, get_best, \
    plot_feature_importance
from leina.models import train, return_function_name
from leina.preprocessing import split_data, one_hot_encode, label_encode, ordinal_encode, standard_scale, full_clean
from utils import Logger, LogLevel
from sklearn.model_selection import GridSearchCV

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

# General Function
def run_model(name, natural_name=None, **kwargs):
    if not natural_name:
        natural_name = name
    logger.info(f'Running {natural_name}')


    lr = train(X_train, y_train, name=name, **kwargs)
    y_pred = lr.predict(X_test)

    accuracy_score, confusion_matrix, classification_report = get_report(y_test, y_pred)
    #plot_confusion_matrix(y_test, y_pred)
    #plot_roc(lr, X_test, y_test)

    accuracy_map[natural_name] = accuracy_score

# Logistic Regression
def run_logistic_regression(**kwargs):
    logger.info('Running Logistic Regression')

    lr = train(X_train, y_train, name='LogisticRegression', max_iter=10000, **kwargs)
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

# Xgboost
def run_xgboost():
    logger.info('Running Xgboost')

    xgb = train(X_train, y_train, name='XGBClassifier')
    y_pred = xgb.predict(X_test)

    accuracy_score, confusion_matrix, classification_report = get_report(y_test, y_pred)

    accuracy_map['xgboost'] = accuracy_score

    plot_feature_importance(xgb, features)


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


# Grid Search
def grid_search(name, grid_vals):
    grid_lr = GridSearchCV(estimator=name, param_grid=grid_vals, scoring='accuracy',
                           cv=6, refit=True, return_train_score=True)

    # Training and Prediction
    grid_lr.fit(X_train, y_train)
    best_params = grid_lr.best_params_
    #preds = grid_lr.best_estimator_.predict(X_test)
    print(f"Grid Search preds: {best_params}")

    return best_params


def write_best_parameters(function_name, best_args):
    file_name = "best_params.txt"
    file_write = open(file_name, 'w')

    file_write.write(function_name)
    for arg, value in best_args.items():
        file_write.write("," + arg + "=" + str(value))

    file_write.close()

def try_float(num):
    try:
        return float(num)
    except ValueError:
        return num

def read_best_parameters(function_name):
    file_name = "best_params.txt"
    file_read = open(file_name, 'r')

    for line in file_read.readlines():
        if function_name in line:
            line = line.strip()
            file_read.close()
            split_args = line.split(',')
            split_args.pop(0)
            arg_dict = {}
            for arg_and_value in split_args:
                split_arg_and_value = arg_and_value.split('=')
                arg_name = split_arg_and_value[0]
                value = split_arg_and_value[1]
                arg_dict[arg_name] = try_float(value)

            return arg_dict
    file_read.close()
    return {}


if __name__ == "__main__":
    function_names = [
        {'name': "LogisticRegression", 'max_iter': 10000, 'penalty': 'l2'},
        {'name': "LogisticRegression", 'max_iter': 10000, 'penalty': None, 'natural_name': "LogisticsRegressionUnregularized"},
        {'name': "DecisionTreeClassifier"},
        {'name': "GradientBoostingClassifier"},
        {'name': "SVC"},
        {'name': "RandomForestClassifier"},
        {'name': "GaussianNB"},
        {'name': "KNeighborsClassifier"},
        {'name': "AdaBoostClassifier"},
        {'name': "XGBClassifier"},
        {'name': "MLPClassifier"},
        {'name': "RidgeClassifier"},
        {'name': "PassiveAggressiveClassifier"},
        {'name': "ExtraTreesClassifier"},
        {'name': "LGBMClassifier"},
    ]

    for args in function_names:
        run_model(**args)

    grid_vals = {'penalty': ['l2'], 'C': [0.001, 0.01, 0.1, 1, 2, 5, 10]}

    for args in function_names:
        func_name = args['name']
        best_args:dict = grid_search(return_function_name(**args), grid_vals)
        write_best_parameters(func_name, best_args)

    # run_best_xgboost()
    #run_ranking()
    #run_loop_logistic_regression()
    #run_loop_k_nearest_neighbours()

    #function_name = "LogisticRegression"
    #best_args: dict = grid_search(return_function_name(function_name, max_iter=10000), grid_vals)
    #a = read_best_parameters(function_name)
    #run_logistic_regression()
    #run_logistic_regression(**a)
