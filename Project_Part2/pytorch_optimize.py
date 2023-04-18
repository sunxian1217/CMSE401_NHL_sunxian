import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

observations = pd.read_csv('observations.csv')
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


# Define the validation data
X_val, y_val = torch.Tensor(X.values), torch.Tensor(y.reshape(-1,1))

# Initialize early stopping variables
best_loss = float('inf')
best_epoch = 0
patience = 10
counter = 0

# Train the model
for epoch in range(100):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        X_batch_tensor = torch.Tensor(X_batch)
        y_pred = model(X_batch_tensor)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        
    # Evaluate the model on the validation set
    with torch.no_grad():
        y_val_pred = model(X_val)
        val_loss = criterion(y_val_pred, y_val)
    
    # Check for improvement in validation loss
    if val_loss < best_loss:
        best_loss = val_loss
        best_epoch = epoch
        counter = 0
    else:
        counter += 1
        
    # Stop training if validation loss has not improved for `patience` epochs
    if counter >= patience:
        print(f'Early stopping: validation loss has not improved for {patience} epochs')
        break

# Evaluate the model on the test set using the best model from early stopping
X_test_tensor = torch.Tensor(X_test)
y_pred = model(X_test_tensor)
predictions = (y_pred > 0).float()
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
