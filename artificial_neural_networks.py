import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from leina.analytics import print_report
from leina.preprocessing import one_hot_encode, split_data, label_encode

data = pd.read_csv('bank.csv')

X, y = split_data(data)

X = one_hot_encode(X, mode='pandas', dtype=int)
y = label_encode(y)

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

# Fitting the model to the training data
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Making predictions on the testing data
y_pred = np.round(model.predict(X_test)).astype(int)

# Evaluating the performance of the model
print_report(y_test, y_pred)
