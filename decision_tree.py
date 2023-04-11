import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("bank.csv")

# Remove any rows with missing values
data = data.dropna()

# Encode the categorical variables
# It transforms each unique value of the categorical variable into a separate binary variable.
# Multiple columns of True/False values
data_encoded = pd.get_dummies(data)

# Split the data into training and testing sets
X = data_encoded.drop(['deposit_yes', 'deposit_no'], axis=1)
y = data_encoded['deposit_yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)

# Initialize decision tree classifier
clf = DecisionTreeClassifier(random_state=60)

# Train decision tree classifier
clf.fit(X_train, y_train)

# Make predictions on test data
y_pred = clf.predict(X_test)

# Evaluating the performance of the model
print("Accuracy score:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

# Visualize decision tree
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns.values, class_names=["no", "yes"], filled=True)
plt.show()
