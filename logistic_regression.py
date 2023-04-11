import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, \
    classification_report, roc_curve, auc
import seaborn as sns

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

# Fit a logistic regression model to the training set
# increase the maximum number of iterations and specify the solver
model = LogisticRegression(max_iter=10000, solver='lbfgs')
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluating the performance of the model
print("Accuracy score:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

# Get the predicted probabilities for the testing set
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Calculate the false positive rate, true positive rate, and threshold values
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Calculate the area under the ROC curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # Add a dashed diagonal line for comparison
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

