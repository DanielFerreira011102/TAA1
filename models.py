from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def train(X_train, y_train, name='logistic_regression', **kwargs):
    if name == 'logistic_regression':
        return _logistic_regression(X_train, y_train, **kwargs)
    if name == 'decision_tree':
        return _decision_tree(X_train, y_train, **kwargs)
    if name == 'gradient_boosting_machines':
        return _gradient_boosting_machines(X_train, y_train, **kwargs)
    if name == 'naive_bayes':
        return _naive_bayes(X_train, y_train, **kwargs)
    if name == 'random_forest':
        return _random_forest(X_train, y_train, **kwargs)
    if name == 'support_vector_machines':
        return _support_vector_machines(X_train, y_train, **kwargs)
    if name == 'k_nearest_neighbours':
        return _k_nearest_neighbours(X_train, y_train, **kwargs)


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