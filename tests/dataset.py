import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import learning_curve, ShuffleSplit, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, make_scorer, \
    roc_curve
from sklearn.tree import DecisionTreeClassifier

from utils import Logger, LogLevel
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

"""
https://github.com/emekaefidi/Bank-Marketing-with-Machine-Learning/blob/master/Bank%20Marketing%20with%20Machine%20Learning.ipynb
"""

logger = Logger(level=LogLevel.INFO)

# Load the dataset
df = pd.read_csv('../bank.csv')

# Print the shape of the dataset
logger.info('Dataset shape', nl=True)
print(df.shape)

# Print the column names
logger.info('Columns', nl=True)
print(df.columns)

# Print the data types of the columns
logger.info('Data types', nl=True)
print(df.dtypes)

# Print the first few rows of the dataset
logger.info('Dataset preview', nl=True)
print(df.head())

# Print the summary statistics of numerical columns
logger.info('Summary statistics', nl=True)
print(df.describe())

# Count the number of rows for each type
logger.info('Types for deposit', nl=True)
print(df.groupby('deposit').size())

df['OUTPUT_LABEL'] = (df.deposit == 'yes').astype('int')


def calc_prevalence(y_actual):
    # this function calculates the prevalence of the positive class (label = 1)
    return sum(y_actual) / len(y_actual)


print('Prevalence of the positive class: %.3f' % calc_prevalence(df['OUTPUT_LABEL'].values))

# Print the number of unique values for each categorical column
logger.info('Unique values', nl=True)
for col in df.columns:
    print(col, ':', df[col].nunique(), 'unique values')

logger.info('Remove duration', nl=True)
print("""Important note: this attribute highly affects the output target (e.g., if duration=0 then y=’no’).
Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known.
Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have
a realistic predictive model.""")

df = df.drop('duration', axis=1)

# Print the count of missing values in each column
logger.info('Missing values', nl=True)
print(df.isnull().sum())

# Select the numerical columns
logger.info('Numerical features', nl=True)
num_cols = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']

for col in num_cols:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(df[col], bins=20, density=True)
    ax.set_xlabel(col)
    ax.set_ylabel('Density')
    ax.axvline(df[col].mean(), color='red', linestyle='dashed', linewidth=1)
    ax.axvline(df[col].median(), color='green', linestyle='dashed', linewidth=1)
    ax.legend(['Mean', 'Median'])
    plt.show()

# Create dummy variables for the categorical features
logger.info('Categorical features', nl=True)
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
cols_new_cat = pd.get_dummies(df[cat_cols], drop_first=False)

# Bar plot of job feature
plt.figure(figsize=(10, 6))
sns.countplot(x='job', data=df)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Pie chart of marital status feature
df['marital'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.ylabel('')
plt.show()

# Bar plot of education feature
sns.countplot(x='education', data=df)
plt.show()

# Pie chart of default feature
df['default'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.ylabel('default')
plt.show()

# Bar plot of housing loan feature
sns.countplot(x='housing', data=df)
plt.show()

# Bar plot of personal loan feature
sns.countplot(x='loan', data=df)
plt.show()

# Bar plot of contact feature
sns.countplot(x='contact', data=df)
plt.show()

# Bar plot of month feature
sns.countplot(x='month', data=df,
              order=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
plt.show()

# Bar plot of poutcome feature
sns.countplot(x='poutcome', data=df)
plt.show()

# Deposit target variable
logger.info('Target', nl=True)
df['deposit'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.ylabel('deposit')
plt.show()

logger.info('Summary of Features Engineering', nl=True)
df = pd.concat([df, cols_new_cat], axis=1)
cols_all_cat = list(cols_new_cat.columns)
print('Total number of features:', len(cols_all_cat + num_cols))
print('Numerical Features:', len(num_cols))
print('Categorical Features:', len(cols_all_cat))

df[num_cols + cols_all_cat].isnull().sum().sort_values(ascending=False)

cols_input = num_cols + cols_all_cat
df_data = df[cols_input + ['OUTPUT_LABEL']]

print(cols_input)

logger.info('Building Training, Validation & Test Samples', nl=True)
# shuffle the samples
df_data = df_data.sample(n=len(df_data), random_state=42)
df_data = df_data.reset_index(drop=True)

# 30% of the validation and test samples
df_valid_test = df_data.sample(frac=0.30, random_state=42)
print('Split size: %.3f' % (len(df_valid_test) / len(df_data)))

# Split into test and validation samples by 50% which makes 15% of test and 15% of validation samples
df_test = df_valid_test.sample(frac=0.5, random_state=42)
df_valid = df_valid_test.drop(df_test.index)

# Use the rest of the data as training data
df_train_all = df_data.drop(df_valid_test.index)

# Check the prevalence of each
print('Test prevalence(n = %d):%.3f' % (len(df_test), calc_prevalence(df_test.OUTPUT_LABEL.values)))
print('Valid prevalence(n = %d):%.3f' % (len(df_valid), calc_prevalence(df_valid.OUTPUT_LABEL.values)))
print('Train all prevalence(n = %d):%.3f' % (len(df_train_all), calc_prevalence(df_train_all.OUTPUT_LABEL.values)))

# We need to balance the data set because if we use the training data as the predictive model the accuracy
# is going to be very high because we haven't caught any of the y output which states whether a person will buy
# a term deposit or not. There are more negatives than positive so the predictive models assigns negatives to much
# of the samples. Creating a balance sheet will allow 50% of the samples to be both positive and negative.

logger.info('Balance de dataset', nl=True)
# split the training data into positive and negative
rows_pos = df_train_all.OUTPUT_LABEL == 1
df_train_pos = df_train_all.loc[rows_pos]
df_train_neg = df_train_all.loc[~rows_pos]

# merge the balanced data
df_train = pd.concat([df_train_pos, df_train_neg.sample(n=len(df_train_pos), random_state=42)], axis=0)

# shuffle the order of training samples
df_train = df_train.sample(n=len(df_train), random_state=42).reset_index(drop=True)

print('Train balanced prevalence(n = %d):%.3f' % (len(df_train), calc_prevalence(df_train.OUTPUT_LABEL.values)))

# split the validation into positive and negative
rows_pos = df_valid.OUTPUT_LABEL == 1
df_valid_pos = df_valid.loc[rows_pos]
df_valid_neg = df_valid.loc[~rows_pos]

# merge the balanced data
df_valid = pd.concat([df_valid_pos, df_valid_neg.sample(n=len(df_valid_pos), random_state=42)], axis=0)

# shuffle the order of training samples
df_valid = df_valid.sample(n=len(df_valid), random_state=42).reset_index(drop=True)

print('Valid balanced prevalence(n = %d):%.3f' % (len(df_valid), calc_prevalence(df_train.OUTPUT_LABEL.values)))

# split the test into positive and negative
rows_pos = df_test.OUTPUT_LABEL == 1
df_test_pos = df_test.loc[rows_pos]
df_test_neg = df_test.loc[~rows_pos]

# merge the balanced data
df_test = pd.concat([df_test_pos, df_test_neg.sample(n=len(df_test_pos), random_state=42)], axis=0)

# shuffle the order of training samples
df_test = df_test.sample(n=len(df_test), random_state=42).reset_index(drop=True)

print('Test balanced prevalence(n = %d):%.3f' % (len(df_test), calc_prevalence(df_train.OUTPUT_LABEL.values)))

logger.info('Save datasets', nl=True)

df_train_all.to_csv('df_train_all.csv', index=False)
df_train.to_csv('df_train.csv', index=False)
df_valid.to_csv('df_valid.csv', index=False)
df_test.to_csv('df_test.csv', index=False)

pickle.dump(cols_input, open('cols_input.sav', 'wb'))


def fill_my_missing(df_, df_mean_, col2use):
    # This function fills the missing values

    # check the columns are present
    for c in col2use:
        assert c in df_.columns, c + ' not in df'
        assert c in df_mean_.col.values, c + 'not in df_mean'

    # replace the mean
    for c in col2use:
        mean_value = df_mean_.loc[df_mean_.col == c, 'mean_val'].values[0]
        df_[c] = df_[c].fillna(mean_value)
    return df_


logger.info('Fill missing values', nl=True)
df_mean = df_train_all[cols_input].mean(axis=0)
# save the means
df_mean.to_csv('df_mean.csv', index=True)

df_mean_in = pd.read_csv('df_mean.csv', names=['col', 'mean_val'])
df_mean_in.head()

df_train_all = fill_my_missing(df_train_all, df_mean_in, cols_input)
df_train = fill_my_missing(df_train, df_mean_in, cols_input)
df_valid = fill_my_missing(df_valid, df_mean_in, cols_input)

# create the X and y matrices
X_train = df_train[cols_input].values
X_train_all = df_train_all[cols_input].values
X_valid = df_valid[cols_input].values

y_train = df_train['OUTPUT_LABEL'].values
y_valid = df_valid['OUTPUT_LABEL'].values

print('Training All shapes:', X_train_all.shape)
print('Training shapes:', X_train.shape, y_train.shape)
print('Validation shapes:', X_valid.shape, y_valid.shape)

scaler = StandardScaler()
scaler.fit(X_train_all)

scaler_file = 'scaler.sav'
pickle.dump(scaler, open(scaler_file, 'wb'))

# load it back
scaler = pickle.load(open(scaler_file, 'rb'))

# transform our data matrices
X_train_tf = scaler.transform(X_train)
X_valid_tf = scaler.transform(X_valid)

logger.info('Model selection', nl=True)


def calc_specificity(y_actual_, y_pred_, thresh_):
    # calculates specificity
    return sum((y_pred_ < thresh_) & (y_actual_ == 0)) / sum(y_actual_ == 0)


def print_report(y_actual_, y_pred_, thresh_):
    auc = roc_auc_score(y_actual_, y_pred_)
    accuracy = accuracy_score(y_actual_, (y_pred_ > thresh_))
    recall = recall_score(y_actual_, (y_pred_ > thresh_))
    precision = precision_score(y_actual_, (y_pred_ > thresh_))
    specificity = calc_specificity(y_actual_, y_pred_, thresh_)
    f1 = 2 * (precision * recall) / (precision + recall)

    print('AUC:%.3f' % auc)
    print('accuracy:%.3f' % accuracy)
    print('recall:%.3f' % recall)
    print('precision:%.3f' % precision)
    print('specificity:%.3f' % specificity)
    print('prevalence:%.3f' % calc_prevalence(y_actual_))
    print('f1:%.3f' % f1)
    print(' ')
    return auc, accuracy, recall, precision, specificity, f1


logger.info('K nearest neighbors (KNN)', nl=True)
print("""K Nearest Neighbors looks at the k closest datapoints and probability sample that has positive labels.
It is easy to implement, and you don't need an assumption for the data structure. KNN is also good for multivariate
analysis.""")

# Model Selection: baseline models
# In this section, we will first compare the performance of the following 7 machine learning models using default hyperparameters:
# -K-nearest neighbors
# -Logistic regression
# -Stochastic gradient descent
# -Naive Bayes
# -Decision tree
# -Random forest
# -Gradient boosting classifier

thresh = 0.5

knn = KNeighborsClassifier(n_neighbors=100)
knn.fit(X_train_tf, y_train)

y_train_preds = knn.predict_proba(X_train_tf)[:, 1]
y_valid_preds = knn.predict_proba(X_valid_tf)[:, 1]

print('KNN')
print('Training:')
knn_train_auc, knn_train_accuracy, knn_train_recall, knn_train_precision, knn_train_specificity, knn_train_f1 = print_report(
    y_train, y_train_preds, thresh)
print('Validation:')
knn_valid_auc, knn_valid_accuracy, knn_valid_recall, knn_valid_precision, knn_valid_specificity, knn_valid_f1 = print_report(
    y_valid, y_valid_preds, thresh)

logger.info('Logistic Regression', nl=True)
print("""Logistic regression uses a line (Sigmoid function) in the form of an "S" to predict if the dependent variable
is true or false based on the independent variables. The "S-shaped" curve (on the line graph) will show the probability
of the dependent variable occurring based on where the points of the independent variables lands on the curve.
In this case, the output (y) is predicted by the numerical and categorical variables defined as "x" such as age,
education and so on. Logistic regression is best used for classifying samples.""")

lr = LogisticRegression(random_state=42)
lr.fit(X_train_tf, y_train)

y_train_preds = lr.predict_proba(X_train_tf)[:, 1]
y_valid_preds = lr.predict_proba(X_valid_tf)[:, 1]

print('Logistic Regression')
print('Training:')
lr_train_auc, lr_train_accuracy, lr_train_recall, lr_train_precision, lr_train_specificity, lr_train_f1 = print_report(
    y_train, y_train_preds, thresh)
print('Validation:')
lr_valid_auc, lr_valid_accuracy, lr_valid_recall, lr_valid_precision, lr_valid_specificity, lr_valid_f1 = print_report(
    y_valid, y_valid_preds, thresh)

logger.info('Stochastic Gradient Descent', nl=True)
print("""Stochastic Gradient Descent analyzes various sections of the data instead of the data as a whole and predicts
the output using the independent variables. Stochastic Gradient Descent is faster than logistic regression in the sense
that it doesn't run the whole dataset but instead looks at different parts of the dataset.""")

sgdc = SGDClassifier(loss='log', alpha=0.1, random_state=42)
sgdc.fit(X_train_tf, y_train)

y_train_preds = sgdc.predict_proba(X_train_tf)[:, 1]
y_valid_preds = sgdc.predict_proba(X_valid_tf)[:, 1]

print('Stochastic Gradient Descent')
print('Training:')
sgdc_train_auc, sgdc_train_accuracy, sgdc_train_recall, sgdc_train_precision, sgdc_train_specificity, sgdc_train_f1 = print_report(
    y_train, y_train_preds, thresh)
print('Validation:')
sgdc_valid_auc, sgdc_valid_accuracy, sgdc_valid_recall, sgdc_valid_precision, sgdc_valid_specificity, sgdc_valid_f1 = print_report(
    y_valid, y_valid_preds, thresh)

logger.info('Naive Bayes', nl=True)
print("""Naive Bayes assumes that all variables in the dataset are independent of each other. Meaning that there are
no dependent variables or output. This algorithm uses Bayes rule which calculated the probability of an event related
to previous knowledge of the variables concerning the event. 
This won't really work in this case since we have an output of the bank customers who will get a bank deposit.
This process is better for tasks such as image processing.""")

nb = GaussianNB()
nb.fit(X_train_tf, y_train)

y_train_preds = nb.predict_proba(X_train_tf)[:, 1]
y_valid_preds = nb.predict_proba(X_valid_tf)[:, 1]

print('Naive Bayes')
print('Training:')
nb_train_auc, nb_train_accuracy, nb_train_recall, nb_train_precision, nb_train_specificity, nb_train_f1 = print_report(
    y_train, y_train_preds, thresh)
print('Validation:')
nb_valid_auc, nb_valid_accuracy, nb_valid_recall, nb_valid_precision, nb_valid_specificity, nb_valid_f1 = print_report(
    y_valid, y_valid_preds, thresh)

logger.info('Decision Tree Classifier', nl=True)
print("""Decision trees works through the data to decide if one action occurs, what will then be the
result of a "yes" and a "no". It works each data making the decision of which path to take based on the answer.
Because of this decision making process, this algorithm has no assumptions about the structure
of the data, but instead decides on the path to take through each decision the algorithm performs.""")

tree = DecisionTreeClassifier(max_depth=10, random_state=42)
tree.fit(X_train_tf, y_train)

y_train_preds = tree.predict_proba(X_train_tf)[:, 1]
y_valid_preds = tree.predict_proba(X_valid_tf)[:, 1]

print('Decision Tree')
print('Training:')
tree_train_auc, tree_train_accuracy, tree_train_recall, tree_train_precision, tree_train_specificity, tree_train_f1 = print_report(
    y_train, y_train_preds, thresh)
print('Validation:')
tree_valid_auc, tree_valid_accuracy, tree_valid_recall, tree_valid_precision, tree_valid_specificity, tree_valid_f1 = print_report(
    y_valid, y_valid_preds, thresh)

logger.info('Random Forest', nl=True)
print("""Random forest works like a decision tree algorithm but it performs various decision tree analysis
on the dataset as a whole. That is, it is the bigger version of the decision tree; a forest is
bigger than a tree, you can think of it that way. Random forest takes random samples of trees and it works
faster than the decision tree algorithm.""")

rf = RandomForestClassifier(max_depth=6, random_state=42)
rf.fit(X_train_tf, y_train)

y_train_preds = rf.predict_proba(X_train_tf)[:, 1]
y_valid_preds = rf.predict_proba(X_valid_tf)[:, 1]

print('Random Forest')
print('Training:')
rf_train_auc, rf_train_accuracy, rf_train_recall, rf_train_precision, rf_train_specificity, rf_train_f1 = print_report(
    y_train, y_train_preds, thresh)
print('Validation:')
rf_valid_auc, rf_valid_accuracy, rf_valid_recall, rf_valid_precision, rf_valid_specificity, rf_valid_f1 = print_report(
    y_valid, y_valid_preds, thresh)

logger.info('Gradient Boosting Classifier', nl=True)
print("""Boosting is a technique that builds a new decision tree algorithm that focuses on the errors on the datase
to fix them. This learns the whole model in other to fix it and improve the prediction of the model.
Aside from being related to decision trees, it also relates to gradient descent algorithm as the name signifies.
Gradient boosting analyzes different parts of the dataset and builds trees that focuses and corrects those errors.
The XGBoost library is also the determining factor in winning a lot of Kaggle data science competitions.""")

gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                 max_depth=3, random_state=42)
gbc.fit(X_train_tf, y_train)

y_train_preds = gbc.predict_proba(X_train_tf)[:, 1]
y_valid_preds = gbc.predict_proba(X_valid_tf)[:, 1]

print('Gradient Boosting Classifier')
print('Training:')
gbc_train_auc, gbc_train_accuracy, gbc_train_recall, gbc_train_precision, gbc_train_specificity, gbc_train_f1 = print_report(
    y_train, y_train_preds, thresh)
print('Validation:')
gbc_valid_auc, gbc_valid_accuracy, gbc_valid_recall, gbc_valid_precision, gbc_valid_specificity, gbc_valid_f1 = print_report(
    y_valid, y_valid_preds, thresh)

logger.info('Analyze results baseline models', nl=True)
df_results = pd.DataFrame(
    {'classifier': ['KNN', 'KNN', 'LR', 'LR', 'SGD', 'SGD', 'NB', 'NB', 'DT', 'DT', 'RF', 'RF', 'GB', 'GB'],
     'data_set': ['train', 'valid'] * 7,
     'auc': [knn_train_auc, knn_valid_auc, lr_train_auc, lr_valid_auc, sgdc_train_auc, sgdc_valid_auc, nb_train_auc,
             nb_valid_auc, tree_train_auc, tree_valid_auc, rf_train_auc, rf_valid_auc, gbc_train_auc, gbc_valid_auc, ],
     'accuracy': [knn_train_accuracy, knn_valid_accuracy, lr_train_accuracy, lr_valid_accuracy, sgdc_train_accuracy,
                  sgdc_valid_accuracy, nb_train_accuracy, nb_valid_accuracy, tree_train_accuracy, tree_valid_accuracy,
                  rf_train_accuracy, rf_valid_accuracy, gbc_train_accuracy, gbc_valid_accuracy, ],
     'recall': [knn_train_recall, knn_valid_recall, lr_train_recall, lr_valid_recall, sgdc_train_recall,
                sgdc_valid_recall, nb_train_recall, nb_valid_recall, tree_train_recall, tree_valid_recall,
                rf_train_recall, rf_valid_recall, gbc_train_recall, gbc_valid_recall, ],
     'precision': [knn_train_precision, knn_valid_precision, lr_train_precision, lr_valid_precision,
                   sgdc_train_precision, sgdc_valid_precision, nb_train_precision, nb_valid_precision,
                   tree_train_precision, tree_valid_precision, rf_train_precision, rf_valid_precision,
                   gbc_train_precision, gbc_valid_precision, ],
     'specificity': [knn_train_specificity, knn_valid_specificity, lr_train_specificity, lr_valid_specificity,
                     sgdc_train_specificity, sgdc_valid_specificity, nb_train_specificity, nb_valid_specificity,
                     tree_train_specificity, tree_valid_specificity, rf_train_specificity, rf_valid_specificity,
                     gbc_train_specificity, gbc_valid_specificity, ],
     'f1': [knn_train_f1, knn_valid_f1, lr_train_f1, lr_valid_f1, sgdc_train_f1, sgdc_valid_f1, nb_train_f1,
            nb_valid_f1, tree_train_f1, tree_valid_f1, rf_train_f1, rf_valid_f1, gbc_train_f1, gbc_valid_f1, ],
     })

sns.set(style="darkgrid")
plt.figure(figsize=(10, 6))
ax = sns.barplot(x="classifier", y="auc", hue="data_set", data=df_results)
ax.set_xlabel('Classifier', fontsize=15)
ax.set_ylabel('AUC', fontsize=15)
ax.tick_params(labelsize=15)

# Put the legend out of the figure
plt.legend(loc=2, borderaxespad=0., fontsize=15)
plt.show()

logger.info('Learning Curves', nl=True)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("AUC")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='roc_auc')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="b")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="b",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


title = "Learning Curves (Random Forest)"
# Cross validation with 5 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
estimator = RandomForestClassifier(max_depth=20, random_state=42)
plot_learning_curve(estimator, title, X_train_tf, y_train, ylim=(0.2, 1.01), cv=cv, n_jobs=4)

plt.show()

logger.info('Variance and Bias', nl=True)
# In the case of random forest, we can see the model has high variance because the training and cross-validation scores
# show data points which are very spread out from one another. High variance would cause an algorithm to model th
# e noise in the training set (overfitting).
# Depending on the learning curve, there are a few strategies we can employ to improve the models

# High Variance:
# - Add more samples
# - Add regularization
# - Reduce number of features
# - Decrease model complexity
# - Add better features
# - Change model architecture
# - High Bias:

# Add new features:
# - Increase model complexity
# - Reduce regularization
# - Change model architecture

logger.info('Feature importance', nl=True)

logger.info('Logistic Regression', nl=True)
lr = LogisticRegression(random_state=42)
lr.fit(X_train_tf, y_train)

feature_importances = pd.DataFrame(lr.coef_[0],
                                   index=cols_input,
                                   columns=['importance']).sort_values('importance',
                                                                       ascending=False)

num = np.min([50, len(cols_input)])
ylocs = np.arange(num)
# get the feature importance for top num and sort in reverse order
values_to_plot = feature_importances.iloc[:num].values.ravel()[::-1]
feature_labels = list(feature_importances.iloc[:num].index)[::-1]

plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k');
plt.barh(ylocs, values_to_plot, align='center')
plt.ylabel('Features')
plt.xlabel('Importance Score')
plt.title('Positive Feature Importance Score - Logistic Regression')
plt.yticks(ylocs, feature_labels)
plt.show()

values_to_plot = feature_importances.iloc[-num:].values.ravel()
feature_labels = list(feature_importances.iloc[-num:].index)

plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k');
plt.barh(ylocs, values_to_plot, align='center')
plt.ylabel('Features')
plt.xlabel('Importance Score')
plt.title('Negative Feature Importance Score - Logistic Regression')
plt.yticks(ylocs, feature_labels)
plt.show()

logger.info('Random Forest', nl=True)

rf = RandomForestClassifier(max_depth=6, random_state=42)
rf.fit(X_train_tf, y_train)

feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index=cols_input,
                                   columns=['importance']).sort_values('importance',
                                                                       ascending=False)

num = np.min([50, len(cols_input)])
ylocs = np.arange(num)
# get the feature importance for top num and sort in reverse order
values_to_plot = feature_importances.iloc[:num].values.ravel()[::-1]
feature_labels = list(feature_importances.iloc[:num].index)[::-1]

plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
plt.barh(ylocs, values_to_plot, align='center')
plt.ylabel('Features')
plt.xlabel('Importance Score')
plt.title('Feature Importance Score - Random Forest')
plt.yticks(ylocs, feature_labels)
plt.show()

logger.info('Gradient Boosting Classifier', nl=True)

gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                 max_depth=3, random_state=42)
gbc.fit(X_train_tf, y_train)

feature_importances = pd.DataFrame(gbc.feature_importances_,
                                   index=cols_input,
                                   columns=['importance']).sort_values('importance',
                                                                       ascending=False)

num = np.min([50, len(cols_input)])
ylocs = np.arange(num)
# get the feature importance for top num and sort in reverse order
values_to_plot = feature_importances.iloc[:num].values.ravel()[::-1]
feature_labels = list(feature_importances.iloc[:num].index)[::-1]

plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k');
plt.barh(ylocs, values_to_plot, align='center')
plt.ylabel('Features')
plt.xlabel('Importance Score')
plt.title('Feature Importance Score - Gradient Boosting Classifier')
plt.yticks(ylocs, feature_labels)
plt.show()

logger.info('Decision Trees', nl=True)

tree = DecisionTreeClassifier(max_depth=10, random_state=42)
tree.fit(X_train_tf, y_train)

feature_importances = pd.DataFrame(tree.feature_importances_,
                                   index=cols_input,
                                   columns=['importance']).sort_values('importance',
                                                                       ascending=False)

num = np.min([50, len(cols_input)])
ylocs = np.arange(num)
# get the feature importance for top num and sort in reverse order
values_to_plot = feature_importances.iloc[:num].values.ravel()[::-1]
feature_labels = list(feature_importances.iloc[:num].index)[::-1]

plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k');
plt.barh(ylocs, values_to_plot, align='center')
plt.ylabel('Features')
plt.xlabel('Importance Score')
plt.title('Feature Importance Score - Decision Trees')
plt.yticks(ylocs, feature_labels)
plt.show()

logger.info('Hyperparameter tuning', nl=True)

# train a model for each max_depth in a list. Store the auc for the training and validation set

# max depths
max_depths = np.arange(2, 20, 2)

train_aucs = np.zeros(len(max_depths))
valid_aucs = np.zeros(len(max_depths))

for jj in range(len(max_depths)):
    max_depth = max_depths[jj]

    # fit model
    rf = RandomForestClassifier(n_estimators=100, max_depth=max_depth, random_state=42)
    rf.fit(X_train_tf, y_train)
    # get predictions
    y_train_preds = rf.predict_proba(X_train_tf)[:, 1]
    y_valid_preds = rf.predict_proba(X_valid_tf)[:, 1]

    # calculate auc
    auc_train = roc_auc_score(y_train, y_train_preds)
    auc_valid = roc_auc_score(y_valid, y_valid_preds)

    # save aucs
    train_aucs[jj] = auc_train
    valid_aucs[jj] = auc_valid

plt.figure()
plt.plot(max_depths, train_aucs, 'o-', label='train')
plt.plot(max_depths, valid_aucs, 'o-', label='valid')
plt.xlabel('max_depth')
plt.ylabel('AUC')
plt.legend()
plt.show()

# number of trees
n_estimators = range(200, 1000, 200)
# maximum number of features to use at each split
max_features = [None, 'sqrt']
# maximum depth of the tree
max_depth = range(2, 20, 2)
# minimum number of samples to split a node
min_samples_split = range(2, 10, 2)
# criterion for evaluating a split
criterion = ['gini', 'entropy']

# random grid

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'criterion': criterion}

print(random_grid)

auc_scoring = make_scorer(roc_auc_score)

# create a baseline model
rf = RandomForestClassifier()

# create the randomized search cross-validation
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                               n_iter=20, cv=2,
                               scoring=auc_scoring, verbose=1, random_state=42)

# fit the random search model (this will take a few minutes)
t1 = time.time()
rf_random.fit(X_train_tf, y_train)
t2 = time.time()
print(t2 - t1)

print("Best params:" % rf_random.best_params_)

rf = RandomForestClassifier(max_depth=6, random_state=42)
rf.fit(X_train_tf, y_train)

y_train_preds = rf.predict_proba(X_train_tf)[:, 1]
y_valid_preds = rf.predict_proba(X_valid_tf)[:, 1]

thresh = 0.5

print('Baseline Random Forest')
rf_train_base_auc = roc_auc_score(y_train, y_train_preds)
rf_valid_base_auc = roc_auc_score(y_valid, y_valid_preds)

print('Training AUC:%.3f' % rf_train_base_auc)
print('Validation AUC:%.3f' % rf_valid_base_auc)

print('Optimized Random Forest')
y_train_preds_random = rf_random.best_estimator_.predict_proba(X_train_tf)[:, 1]
y_valid_preds_random = rf_random.best_estimator_.predict_proba(X_valid_tf)[:, 1]

rf_train_opt_auc = roc_auc_score(y_train, y_train_preds_random)
rf_valid_opt_auc = roc_auc_score(y_valid, y_valid_preds_random)

print('Training AUC:%.3f' % rf_train_opt_auc)
print('Validation AUC:%.3f' % rf_valid_opt_auc)

logger.info('Optimized SGDClassifier', nl=True)

sgdc = SGDClassifier(loss='log', alpha=0.1, random_state=42)
sgdc.fit(X_train_tf, y_train)

penalty = ['none', 'l2', 'l1']
max_iter = range(200, 1000, 200)
alpha = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
random_grid_sgdc = {'penalty': penalty,
                    'max_iter': max_iter,
                    'alpha': alpha}
# create the randomized search cross-validation
sgdc_random = RandomizedSearchCV(estimator=sgdc, param_distributions=random_grid_sgdc, n_iter=20, cv=2,
                                 scoring=auc_scoring, verbose=0, random_state=42)

t1 = time.time()
sgdc_random.fit(X_train_tf, y_train)
t2 = time.time()
print(t2 - t1)

print("Best params:" % sgdc_random.best_params_)

y_train_preds = sgdc.predict_proba(X_train_tf)[:, 1]
y_valid_preds = sgdc.predict_proba(X_valid_tf)[:, 1]

print('Baseline sgdc')
sgdc_train_base_auc = roc_auc_score(y_train, y_train_preds)
sgdc_valid_base_auc = roc_auc_score(y_valid, y_valid_preds)

print('Training AUC:%.3f' % sgdc_train_base_auc)
print('Validation AUC:%.3f' % sgdc_valid_base_auc)

print('Optimized sgdc')
y_train_preds_random = sgdc_random.best_estimator_.predict_proba(X_train_tf)[:, 1]
y_valid_preds_random = sgdc_random.best_estimator_.predict_proba(X_valid_tf)[:, 1]
sgdc_train_opt_auc = roc_auc_score(y_train, y_train_preds_random)
sgdc_valid_opt_auc = roc_auc_score(y_valid, y_valid_preds_random)

print('Training AUC:%.3f' % sgdc_train_opt_auc)
print('Validation AUC:%.3f' % sgdc_valid_opt_auc)

logger.info('Optimized Gradient Boosting Classifier', nl=True)

gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                 max_depth=3, random_state=42)
gbc.fit(X_train_tf, y_train)

# number of trees
n_estimators = range(50, 200, 50)

# maximum depth of the tree
max_depth = range(1, 5, 1)

# learning rate
learning_rate = [0.001, 0.01, 0.1]

# random grid

random_grid_gbc = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'learning_rate': learning_rate}

# create the randomized search cross-validation
gbc_random = RandomizedSearchCV(estimator=gbc, param_distributions=random_grid_gbc, n_iter=20, cv=2,
                                scoring=auc_scoring, verbose=0, random_state=42)

t1 = time.time()
gbc_random.fit(X_train_tf, y_train)
t2 = time.time()
print(t2 - t1)

print("Best params:" % gbc_random.best_params_)

y_train_preds = gbc.predict_proba(X_train_tf)[:, 1]
y_valid_preds = gbc.predict_proba(X_valid_tf)[:, 1]

print('Baseline gbc')
gbc_train_base_auc = roc_auc_score(y_train, y_train_preds)
gbc_valid_base_auc = roc_auc_score(y_valid, y_valid_preds)

print('Training AUC:%.3f' % gbc_train_base_auc)
print('Validation AUC:%.3f' % gbc_valid_base_auc)
print('Optimized gbc')
y_train_preds_random = gbc_random.best_estimator_.predict_proba(X_train_tf)[:, 1]
y_valid_preds_random = gbc_random.best_estimator_.predict_proba(X_valid_tf)[:, 1]
gbc_train_opt_auc = roc_auc_score(y_train, y_train_preds_random)
gbc_valid_opt_auc = roc_auc_score(y_valid, y_valid_preds_random)

print('Training AUC:%.3f' % gbc_train_opt_auc)
print('Validation AUC:%.3f' % gbc_valid_opt_auc)

df_results = pd.DataFrame({'classifier': ['SGD', 'SGD', 'RF', 'RF', 'GB', 'GB'],
                           'data_set': ['baseline', 'optimized'] * 3,
                           'auc': [sgdc_valid_base_auc, sgdc_valid_opt_auc,
                                   rf_valid_base_auc, rf_valid_opt_auc,
                                   gbc_valid_base_auc, gbc_valid_opt_auc],
                           })

sns.set(style="darkgrid")

ax = sns.barplot(x="classifier", y="auc", hue="data_set", data=df_results)
ax.set_xlabel('Classifier', fontsize=15)
ax.set_ylabel('AUC', fontsize=15)
ax.tick_params(labelsize=15)
# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=15)

plt.show()

logger.info('Picking the best model', nl=True)

pickle.dump(gbc_random.best_estimator_, open('best_classifier.pkl', 'wb'), protocol=4)

logger.info('Model evaluation', nl=True)

# load the model, columns, mean values, and scaler
best_model = pickle.load(open('best_classifier.pkl', 'rb'))
cols_input = pickle.load(open('cols_input.sav', 'rb'))
df_mean_in = pd.read_csv('df_mean.csv', names=['col', 'mean_val'])
scaler = pickle.load(open('scaler.sav', 'rb'))

# load the data
df_train = pd.read_csv('df_train.csv')
df_valid = pd.read_csv('df_valid.csv')
df_test = pd.read_csv('df_test.csv')

# fill missing
df_train = fill_my_missing(df_train, df_mean_in, cols_input)
df_valid = fill_my_missing(df_valid, df_mean_in, cols_input)
df_test = fill_my_missing(df_test, df_mean_in, cols_input)

# create X and y matrices
X_train = df_train[cols_input].values
X_valid = df_valid[cols_input].values
X_test = df_test[cols_input].values

y_train = df_train['OUTPUT_LABEL'].values
y_valid = df_valid['OUTPUT_LABEL'].values
y_test = df_test['OUTPUT_LABEL'].values

# transform our data matrices
X_train_tf = scaler.transform(X_train)
X_valid_tf = scaler.transform(X_valid)
X_test_tf = scaler.transform(X_test)

y_train_preds = best_model.predict_proba(X_train_tf)[:, 1]
y_valid_preds = best_model.predict_proba(X_valid_tf)[:, 1]
y_test_preds = best_model.predict_proba(X_test_tf)[:, 1]

print('Training:')
train_auc, train_accuracy, train_recall, train_precision, train_specificity, train_f1 = print_report(y_train,
                                                                                                     y_train_preds,
                                                                                                     thresh)
print('Validation:')
valid_auc, valid_accuracy, valid_recall, valid_precision, valid_specificity, valid_f1 = print_report(y_valid,
                                                                                                     y_valid_preds,
                                                                                                     thresh)
print('Test:')
test_auc, test_accuracy, test_recall, test_precision, test_specificity, test_f1 = print_report(y_test, y_test_preds,
                                                                                               thresh)
logger.info('ROC curve', nl=True)

fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_preds)
auc_train = roc_auc_score(y_train, y_train_preds)

fpr_valid, tpr_valid, thresholds_valid = roc_curve(y_valid, y_valid_preds)
auc_valid = roc_auc_score(y_valid, y_valid_preds)

fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_preds)
auc_test = roc_auc_score(y_test, y_test_preds)

plt.plot(fpr_train, tpr_train, 'r-', label='Train AUC:%.3f' % auc_train)
plt.plot(fpr_valid, tpr_valid, 'b-', label='Valid AUC:%.3f' % auc_valid)
plt.plot(fpr_test, tpr_test, 'g-', label='Test AUC:%.3f' % auc_test)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
