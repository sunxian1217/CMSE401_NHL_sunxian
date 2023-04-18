import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
torch.set_num_threads(4)

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