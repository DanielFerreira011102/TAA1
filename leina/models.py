from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier # noqa
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier # noqa
from sklearn.naive_bayes import GaussianNB # noqa
from sklearn.neighbors import KNeighborsClassifier # noqa
from sklearn.neural_network import MLPClassifier # noqa
from sklearn.svm import SVC # noqa
from sklearn.tree import DecisionTreeClassifier # noqa
from xgboost import XGBClassifier # noqa
from lightgbm import LGBMClassifier # noqa

def return_function_name(function_name:str, **kwargs):
    return eval(function_name + "(**kwargs)")

def train(X_train, y_train, name='LogisticRegression', **kwargs):
    model = return_function_name(name, **kwargs)
    model.fit(X_train, y_train)
    return model