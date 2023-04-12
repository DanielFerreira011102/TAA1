import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from leina.analytics import get_report, plot_feature_importance
from leina.models import train

"""https://medium.com/@nutanbhogendrasharma/deal-banking-marketing-campaign-dataset-with-machine-learning-9c1f84ad285d"""

df = pd.read_csv('../bank.csv')
df.head()

sns.set_theme(style="darkgrid")

job_count = df['job'].value_counts()
plt.figure(figsize=(10, 6))
job_count.plot(kind="bar")
plt.title("Type of Job Distribution")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

default_count = df['default'].value_counts()
plt.figure(figsize=(10, 6))
default_count.plot(kind='bar').set(title='Default Column Distribution')
plt.tight_layout()
plt.show()

marital_count = df['marital'].value_counts()
plt.figure(figsize=(10, 6))
marital_count.plot(kind="bar").set(title="Marital Distribution")
plt.tight_layout()
plt.show()

loan_count = df['loan'].value_counts()
plt.figure(figsize=(10, 6))
loan_count.plot(kind="bar").set(title="Loan Distribution")
plt.tight_layout()
plt.show()

housing_count = df['housing'].value_counts()
plt.figure(figsize=(10, 6))
housing_count.plot(kind="bar").set(title="Housing Loan Distribution")
plt.tight_layout()
plt.show()

education_count = df['education'].value_counts()
plt.figure(figsize=(10, 6))
education_count.plot(kind="bar").set(title="Education Column Distribution")
plt.tight_layout()
plt.show()

contact_count = df['contact'].value_counts()
plt.figure(figsize=(10, 6))
contact_count.plot(kind="bar").set(title="Contact Column Distribution")
plt.tight_layout()
plt.show()

month_count = df['month'].value_counts()
plt.figure(figsize=(10, 6))
month_count.plot(kind="bar").set(title="Month Data Distribution")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
# Plot pdays whose value is greater than 0
df[df['pdays'] > 0]['pdays'].hist(bins=50).set(title="Pdays Data Distribution")
plt.tight_layout()
plt.show()

target_count = df['deposit'].value_counts()
plt.figure(figsize=(10, 6))
target_count.plot(kind="bar").set(title="Target Distribution")
plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 20))
df[df['deposit'] == 'yes'].hist()
plt.title('Client has subscribed a term deposit')
plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 20))
df[df['deposit'] == 'no'].hist()
plt.title('Client has not subscribed a term deposit')
plt.tight_layout()
plt.show()

# Preprocessing
df['is_default'] = df['default'].apply(lambda row: 1 if row == 'yes' else 0)
df['is_housing'] = df['housing'].apply(lambda row: 1 if row == 'yes' else 0)
df['is_loan'] = df['loan'].apply(lambda row: 1 if row == 'yes' else 0)
df['target'] = df['deposit'].apply(lambda row: 1 if row == 'yes' else 0)

marital_dummies = pd.get_dummies(df['marital'], prefix='marital')
marital_dummies.drop('marital_divorced', axis=1, inplace=True)
df = pd.concat([df, marital_dummies], axis=1)

job_dummies = pd.get_dummies(df['job'], prefix='job')
job_dummies.drop('job_unknown', axis=1, inplace=True)
df = pd.concat([df, job_dummies], axis=1)

education_dummies = pd.get_dummies(df['education'], prefix='education')
education_dummies.drop('education_unknown', axis=1, inplace=True)
df = pd.concat([df, education_dummies], axis=1)

contact_dummies = pd.get_dummies(df['contact'], prefix='contact')
contact_dummies.drop('contact_unknown', axis=1, inplace=True)
df = pd.concat([df, contact_dummies], axis=1)

poutcome_dummies = pd.get_dummies(df['poutcome'], prefix='poutcome')
poutcome_dummies.drop('poutcome_unknown', axis=1, inplace=True)
df = pd.concat([df, poutcome_dummies], axis=1)

months = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10,
          'nov': 11, 'dec': 12}
df['month'] = df['month'].map(months)

df.drop(['job', 'education', 'marital', 'default', 'housing', 'loan', 'contact', 'poutcome', 'deposit'],
        axis=1, inplace=True)

print(df.dtypes)
print(df.head(10))

# The axis=1 argument drop columns
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

print("Logistic Regression")
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Predicted value: ", y_pred[:10])
print("Actual value: ", y_test[:10])

accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)
print(f'Accuracy of the model Logistic Regression is {accuracy * 100:.2f}%')

print("Random Forest")
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print("Predicted value: ", y_pred[:10])
print("Actual value: ", y_test[:10])

accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)
print(f'Accuracy of the Random Forest Classifier model is {accuracy * 100:.2f}%')

print("SVC")
svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print("Predicted value: ", y_pred[:10])
print("Actual value: ", y_test[:10])

accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)
print(f'Accuracy of the SVC model is {accuracy * 100:.2f}%')

print("Decision Tree")
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
print("Predicted value: ", y_pred[:10])
print("Actual value: ", y_test[:10])

accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)
print(f'Accuracy of the Decision Tree Classifier model is {accuracy * 100:.2f}%')

print('Xgboost')

xgb = train(X_train, y_train, name='xgboost')
y_pred = xgb.predict(X_test)
accuracy_score, confusion_matrix, classification_report = get_report(y_test, y_pred)
plot_feature_importance(xgb, features=X.columns)
