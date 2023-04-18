import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

observations = pd.read_csv('observations.csv')

X = observations.drop(['outcome'], axis='columns')
y = observations['outcome']
le = LabelEncoder()# Encode categorical variables
X['AwayTeam'] = le.fit_transform(X['AwayTeam'])
X['HomeTeam'] = le.fit_transform(X['HomeTeam'])
X['IsClosed'] = le.fit_transform(X['IsClosed'])
y = le.fit_transform(y)
X = X.astype({'awaywon': 'float32', 'awaylost': 'float32', 'homewon': 'float32', 'homelost': 'float32', 'AwayTeamMoneyLine': 'float32',
              'HomeTeamMoneyLine': 'float32', 'PointSpread': 'float32', 'OverUnder': 'float32', 'PointSpreadAwayTeamMoneyLine': 'float32',
              'PointSpreadHomeTeamMoneyLine': 'float32','HomeRotationNumber': 'float32', 'AwayRotationNumber': 'float32', 'OverPayout': 'float32',
              'UnderPayout':'float32'})
X_train = X[:1000]
X_test = X[1000:]
y_train = y[:1000]
y_test = y[1000:]
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
  tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print('Test accuracy:', test_acc)
