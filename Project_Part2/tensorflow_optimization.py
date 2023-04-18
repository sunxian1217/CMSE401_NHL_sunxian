import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping

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
early_stopping = EarlyStopping('val_acc',patience=3)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test),callbacks=[early_stopping])

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print('Test accuracy:', test_acc)
