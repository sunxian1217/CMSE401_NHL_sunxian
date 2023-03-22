#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
data = pd.read_json('2021.json')
data = data.drop(['GameKey', 'SeasonType', 'Season', 'Date', 'Channel', 'Quarter', 'TimeRemaining',
                  'Possession', 'Down', 'Distance', 'YardLine', 'YardLineTerritory', 'RedZone', 
                  'HasStarted', 'IsInProgress', 'IsOver', 'Has1stQuarterStarted', 'Has2ndQuarterStarted',
                  'Has3rdQuarterStarted', 'Has4thQuarterStarted', 'IsOvertime', 'DownAndDistance',
                  'QuarterDescription', 'StadiumID', 'LastUpdated', 'GeoLat', 'GeoLong', 'ForecastTempLow',
                  'ForecastTempHigh', 'ForecastDescription', 'ForecastWindChill', 'ForecastWindSpeed',
                  'Canceled', 'Closed', 'LastPlay', 'Day', 'DateTime', 'GlobalGameID', 'ScoreID', 'Status',
                  'GameEndDateTime', 'HomeRotationNumber', 'AwayRotationNumber', 'NeutralVenue', 'RefereeID',
                  'OverPayout', 'UnderPayout', 'HomeTimeouts', 'AwayTimeouts', 'DateTimeUTC', 'Attendance',
                  'StadiumDetails'], axis=1)
data = data.dropna()

le = LabelEncoder()# Encode categorical variables
data['AwayTeam'] = le.fit_transform(data['AwayTeam'])
data['HomeTeam'] = le.fit_transform(data['HomeTeam'])

X = data.drop(['HomeScore', 'AwayScore'], axis=1)
y = data['HomeScore'] < data['AwayScore']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)# Split the data into training and testing sets

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)#evaluate the model
print('Test loss:', loss)
print('Test accuracy:', accuracy)


# In[ ]:




