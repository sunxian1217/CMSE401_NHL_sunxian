import tensorflow as tf
import tensorflow_model_optimization as tfmot
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tempfile

# Load the dataset
observations = pd.read_csv('observations.csv')

# Preprocess the data
X = observations.drop(['outcome'], axis='columns')
y = observations['outcome']
le = LabelEncoder()
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

# Define the model architecture
model = tf.keras.Sequential([
  tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])


# Apply weight pruning to the trained model
pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.0, final_sparsity=0.5, begin_step=2000, end_step=4000)
pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, pruning_schedule=pruning_schedule)

# Fine-tune the pruned model
pruned_model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

logdir = tempfile.mkdtemp()

callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]

pruned_model.fit(X_train, y_train,
                  batch_size=128, epochs=10, validation_split=0.2,
                  callbacks=callbacks)
pruned_model.fit(X_train, y_train, epochs=10,batch_size=32128, validation_data=(X_test, y_test))

# Evaluate the pruned and fine-tuned model on the test set
test_loss, test_acc = pruned_model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', test_acc)