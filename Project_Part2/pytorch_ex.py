#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


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
observations.drop(['AwayTeamScore','HomeTeamScore','Date','awayKey','homeKey','homePercentage','awayPercentage'],axis='columns',inplace=True)




X = observations.drop(['outcome'], axis='columns')
y = observations['outcome']
le = LabelEncoder()# Encode categorical variables
X['AwayTeam'] = le.fit_transform(X['AwayTeam'])
X['HomeTeam'] = le.fit_transform(X['HomeTeam'])
y = le.fit_transform(y)
X['IsClosed'] = le.fit_transform(X['IsClosed'])
X = X.astype({'awaywon': 'float32', 'awaylost': 'float32', 'homewon': 'float32', 'homelost': 'float32', 'AwayTeamMoneyLine': 'float32',
              'HomeTeamMoneyLine': 'float32', 'PointSpread': 'float32', 'OverUnder': 'float32', 'PointSpreadAwayTeamMoneyLine': 'float32',
              'PointSpreadHomeTeamMoneyLine': 'float32','HomeRotationNumber': 'float32', 'AwayRotationNumber': 'float32', 'OverPayout': 'float32',
              'UnderPayout':'float32'})

# Convert the data to PyTorch tensors
X_tensor = torch.Tensor(X.values)
y_tensor = torch.Tensor(y.reshape(-1,1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Normalize the input data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the PyTorch model
model = nn.Linear(X_tensor.shape[1], 1)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
batch_size = 32
train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Train the model
for epoch in range(100):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        X_batch_tensor = torch.Tensor(X_batch)
        y_pred = model(X_batch_tensor)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

# Evaluate the model on the test set
X_test_tensor = torch.Tensor(X_test)
y_pred = model(X_test_tensor)
predictions = (y_pred > 0).float()
accuracy = (predictions == y_test).sum().item() / len(y_test)
print("Accuracy:", accuracy)






