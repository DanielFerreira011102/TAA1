# Importing necessary libraries
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the data
data = pd.read_csv('bank.csv')

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

# Creating the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)

# Fitting the classifier to the training data
knn.fit(X_train, y_train)

# Making predictions on the testing data
y_pred = knn.predict(X_test)

# Evaluating the performance of the classifier
print("Accuracy score:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))