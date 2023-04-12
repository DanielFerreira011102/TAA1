from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, \
    ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def train(X_train, y_train, name='logistic_regression', **kwargs):
    methods = {
        'logistic_regression': _logistic_regression,
        'decision_tree': _decision_tree,
        'gradient_boosting_machines': _gradient_boosting_machines,
        'naive_bayes': _naive_bayes,
        'random_forest': _random_forest,
        'support_vector_machines': _support_vector_machines,
        'k_nearest_neighbours': _k_nearest_neighbours,
        'adaboost': _adaboost,
        'xgboost': _xgboost,
        'multi_layer_perceptron': _multi_layer_perceptron,
        'ridge_classifier': _ridge_classifier,
        'passive_aggressive_classifier': _passive_aggressive_classifier,
        'extremely_randomized_trees': _extremely_randomized_trees,
        'lightgbm_classifier': _lightgbm_classifier
    }
    return methods[name](X_train, y_train, **kwargs)


def _logistic_regression(X_train, y_train, **kwargs):
    lr = LogisticRegression(**kwargs)
    lr.fit(X_train, y_train)
    return lr


def _decision_tree(X_train, y_train, **kwargs):
    clf = DecisionTreeClassifier(**kwargs)
    clf.fit(X_train, y_train)
    return clf


def _gradient_boosting_machines(X_train, y_train, **kwargs):
    gbm = GradientBoostingClassifier(**kwargs)
    gbm.fit(X_train, y_train)
    return gbm


def _naive_bayes(X_train, y_train, **kwargs):
    nb = GaussianNB(**kwargs)
    nb.fit(X_train, y_train)
    return nb


def _random_forest(X_train, y_train, **kwargs):
    rfc = RandomForestClassifier(**kwargs)
    rfc.fit(X_train, y_train)
    return rfc


def _support_vector_machines(X_train, y_train, **kwargs):
    svc = SVC(**kwargs)
    svc.fit(X_train, y_train)
    return svc


def _k_nearest_neighbours(X_train, y_train, **kwargs):
    knn = KNeighborsClassifier(**kwargs)
    knn.fit(X_train, y_train)
    return knn


def _adaboost(X_train, y_train, **kwargs):
    ada = AdaBoostClassifier(**kwargs)
    ada.fit(X_train, y_train)
    return ada


def _xgboost(X_train, y_train, **kwargs):
    xgb = XGBClassifier(**kwargs)
    xgb.fit(X_train, y_train)
    return xgb


def _multi_layer_perceptron(X_train, y_train, **kwargs):
    mlp = MLPClassifier(**kwargs)
    mlp.fit(X_train, y_train)
    return mlp


def _ridge_classifier(X_train, y_train, **kwargs):
    ridge = RidgeClassifier(**kwargs)
    ridge.fit(X_train, y_train)
    return ridge


def _passive_aggressive_classifier(X_train, y_train, **kwargs):
    pac = PassiveAggressiveClassifier(**kwargs)
    pac.fit(X_train, y_train)
    return pac


def _extremely_randomized_trees(X_train, y_train, **kwargs):
    etc = ExtraTreesClassifier(**kwargs)
    etc.fit(X_train, y_train)
    return etc

def _lightgbm_classifier(X_train, y_train, **kwargs):
    lgbm = LGBMClassifier(**kwargs)
    lgbm.fit(X_train, y_train)
    return lgbm
