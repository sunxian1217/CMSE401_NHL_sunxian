#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[46]:


df = pd.read_json('2022_schedules.json')
df = df.drop(['GameID', 'Season', 'SeasonType', 'Status', 'Day','AwayTeamID',
       'HomeTeamID', 'StadiumID', 'Channel','Updated','Period','TimeRemainingMinutes','TimeRemainingSeconds','LastPlay',
             'GameEndDateTime','DateTimeUTC','SeriesInfo','Periods', 'GlobalGameID',
       'GlobalAwayTeamID', 'GlobalHomeTeamID', 'NeutralVenue','Attendance'],axis=1)
df = df.dropna()
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['Date'] = df['DateTime'].dt.date
df = df.drop('DateTime',axis='columns')
df['outcome'] = 'home'
for index, row in df.iterrows():
    if row['AwayTeamScore'] > row['HomeTeamScore']:
        df.at[index, 'outcome'] = 'away'
df = df.dropna()
standings = pd.read_json('2022_standings.json')
standings.drop(['Season','SeasonType','GlobalTeamID','TeamID','City','Name','Conference','Division'],axis='columns',inplace=True)
data = pd.DataFrame()
data.loc['won',list(df['HomeTeam'].unique())] = 0
data.loc['lost',list(df['HomeTeam'].unique())] = 0
data = data.unstack()
data = pd.DataFrame(data,columns=['time']).T
for index, row in df.iterrows():
    if row['AwayTeam'] > row['HomeTeam']:
        winner = row['AwayTeam']
        loser = row['HomeTeam']
    elif row['AwayTeam'] < row['HomeTeam']:
        winner = row['HomeTeam']
        loser = row['AwayTeam']
    data.loc[index,(winner,'won')] = data[(winner,'won')].max()+1
    data.loc[index,(loser,'lost')] = data[(loser,'lost')].max()+1
data = data.fillna(method='ffill').drop(index='time')


# In[47]:


def create_features(row):
    features={}
    features['awaywon'] = data.loc[row.name,(row['AwayTeam'],'won')]
    features['awaylost'] = data.loc[row.name,(row['AwayTeam'],'lost')]
    features['homewon'] = data.loc[row.name,(row['HomeTeam'],'won')]
    features['homelost'] = data.loc[row.name,(row['HomeTeam'],'lost')]
    
    
    home_standing = standings.query(f"Key=='{row['HomeTeam']}'").add_prefix('home')
    away_standing = standings.query(f"Key=='{row['AwayTeam']}'").add_prefix('away')
    
    if len(home_standing) > 0:
        home_standing = home_standing.iloc[0].to_dict()
    else:
        home_standing = {}
    if len(away_standing) > 0:
        away_standing = away_standing.iloc[0].to_dict()
    else:
        away_standing = {}
    return pd.Series({**features,**home_standing,**away_standing,**row})
observations = df.apply(create_features,axis='columns')
observations.drop(['AwayTeamScore','HomeTeamScore','Date','awayKey','homeKey'],axis='columns',inplace=True)


# In[51]:


observations['date'] = df['Date']
X = observations.drop(['outcome','date','homePercentage','awayPercentage'], axis='columns')
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


# In[ ]:




