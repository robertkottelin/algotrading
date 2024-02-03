import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
import os

def preprocess_and_scale(df):
    # Shift the 'c' column to predict the future price direction
    shift_rows = 1
    df['price_future'] = df['BTC price'].shift(-shift_rows)

    # Create a binary target variable for price direction
    df['price_future_direction'] = (df['price_future'] > df['BTC price']).astype(int)

    # Drop the last 'shift_rows' rows where the future price is NaN
    df = df.dropna()

    # Define features (X) and target (y)
    X = df.drop(columns=['price_future', 'price_future_direction', 't'])  # Assuming 't' is a timestamp column
    y = df['price_future_direction']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

file_pattern = 'Crypto/ta_data/coinglass_*.csv'
data_dict = {}
# Iterate over each file that matches the pattern
# Iterate over each file that matches the pattern
for file_path in glob.glob(file_pattern):
    # Extract the exchange and cryptocurrency pair from the file path
    filename = os.path.basename(file_path)
    parts = filename.split('_')
    exchange = parts[1]
    pair = parts[2].replace('.csv', '')
    
    # Generate a unique key for the dictionary
    key = f"{exchange}_{pair}"
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Preprocess and scale the data
    X, y = preprocess_and_scale(df)
    
    # Store the processed data in the dictionary
    data_dict[key] = (X, y)

# Initialize empty lists to collect the X and y data
X_list = []
y_list = []

# Iterate over the dictionary and append the data to the lists
for key in data_dict:
    X, y = data_dict[key]
    X_list.append(X)
    y_list.append(y)

# Concatenate all the X dataframes
X_combined = np.concatenate(X_list, axis=0)

# Concatenate all the y series
y_combined = pd.concat(y_list, ignore_index=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.01, random_state=42)

# Define the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.1),  # Dropout layer for regularization
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0015)  # Increase the learning rate

# Compile the model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with early stopping
model.fit(X_train, y_train, epochs=200, batch_size=1, validation_split=0.1, callbacks=[early_stopping])

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')

# Create the directory for saving the model if it doesn't exist
model_dir = 'Crypto/models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the model
model.save(f'{model_dir}/crypto_1d_model')

print("Model saved successfully!")

# Further use:
# reloaded_model = tf.keras.models.load_model(f'{model_dir}/crypto_model')