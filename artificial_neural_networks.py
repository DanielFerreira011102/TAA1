# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout

# Load the data
data = pd.read_csv('bank.csv')

# Remove any rows with missing values
data = data.dropna()

# Encode the categorical variables
# It transforms each unique value of the categorical variable into a separate binary variable.
# Multiple columns of 0/1 values
data_encoded = pd.get_dummies(data, dtype=int)

# Split the data into training and testing sets
X = data_encoded.drop(['deposit_yes', 'deposit_no'], axis=1)
y = data_encoded['deposit_yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)

# Creating the ANN model
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(16, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# Fitting the model to the training data
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Making predictions on the testing data
y_pred = np.round(model.predict(X_test)).astype(int)

# Evaluating the performance of the model
print("Accuracy score:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))
