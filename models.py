from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def train(X_train, y_train, name='logistic_regression', **kwargs):
    methods = {
        'logistic_regression': _logistic_regression,
        'decision_tree': _decision_tree,
        'gradient_boosting_machines': _gradient_boosting_machines,
        'naive_bayes': _naive_bayes,
        'random_forest': _random_forest,
        'support_vector_machines': _support_vector_machines,
        'k_nearest_neighbours': _k_nearest_neighbours
    }
    return methods[name](X_train, y_train, **kwargs)

def _logistic_regression(X_train, y_train, **kwargs):
    model = LogisticRegression(**kwargs)
    model.fit(X_train, y_train)
    return model


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