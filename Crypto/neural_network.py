import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
import os

def preprocess_and_scale(df):
    # Shift the 'c' column to predict the future price direction
    shift_rows = 12
    df['c_future'] = df['c'].shift(-shift_rows)

    # Create a binary target variable for price direction
    df['c_future_direction'] = (df['c_future'] > df['c']).astype(int)

    # Drop the last 'shift_rows' rows where the future price is NaN
    df = df.dropna()

    # Define features (X) and target (y)
    X = df.drop(columns=['c_future', 'c_future_direction', 't'])  # Assuming 't' is a timestamp column
    y = df['c_future_direction']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# Load and preprocess the data
X_btc, y_btc = preprocess_and_scale(pd.read_csv('Crypto/data/coinglass_BTC_ta.csv'))
X_eth, y_eth = preprocess_and_scale(pd.read_csv('Crypto/data/coinglass_ETH_ta.csv'))
X_sol, y_sol = preprocess_and_scale(pd.read_csv('Crypto/data/coinglass_SOL_ta.csv'))

# Concatenate all scaled dataframes
X_combined = np.concatenate([X_btc, X_eth, X_sol])
y_combined = pd.concat([y_btc, y_eth, y_sol])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.01, random_state=42)

# Define the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.1),  # Dropout layer for regularization
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0015)  # Increase the learning rate

# Compile the model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=80, batch_size=1, validation_split=0.1)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')

# Create the directory for saving the model if it doesn't exist
model_dir = 'Crypto/models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the model
model.save(f'{model_dir}/crypto_h4_model')

print("Model saved successfully!")

# Further use:
# reloaded_model = tf.keras.models.load_model(f'{model_dir}/crypto_model')