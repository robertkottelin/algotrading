import pandas as pd
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

# Load and preprocess the data
X_aave, y_aave = preprocess_and_scale(pd.read_csv('Crypto/data/coinglass_AAVE_ta.csv'))
X_ada, y_ada = preprocess_and_scale(pd.read_csv('Crypto/data/coinglass_ADA_ta.csv'))
X_avax, y_avax = preprocess_and_scale(pd.read_csv('Crypto/data/coinglass_AVAX_ta.csv'))
X_bnb, y_bnb = preprocess_and_scale(pd.read_csv('Crypto/data/coinglass_BNB_ta.csv'))
X_btc, y_btc = preprocess_and_scale(pd.read_csv('Crypto/data/coinglass_BTC_ta.csv'))
X_dot, y_dot = preprocess_and_scale(pd.read_csv('Crypto/data/coinglass_DOT_ta.csv'))
X_etc, y_etc = preprocess_and_scale(pd.read_csv('Crypto/data/coinglass_ETC_ta.csv'))
X_eth, y_eth = preprocess_and_scale(pd.read_csv('Crypto/data/coinglass_ETH_ta.csv'))
X_link, y_link = preprocess_and_scale(pd.read_csv('Crypto/data/coinglass_LINK_ta.csv'))
X_ltc, y_ltc = preprocess_and_scale(pd.read_csv('Crypto/data/coinglass_LTC_ta.csv'))
X_matic, y_matic = preprocess_and_scale(pd.read_csv('Crypto/data/coinglass_MATIC_ta.csv'))
X_sol, y_sol = preprocess_and_scale(pd.read_csv('Crypto/data/coinglass_SOL_ta.csv'))
X_xrp, y_xrp = preprocess_and_scale(pd.read_csv('Crypto/data/coinglass_XRP_ta.csv'))

# Concatenate all scaled dataframes
X_combined = np.concatenate([X_aave, X_ada, X_avax, X_bnb, X_btc, X_dot, X_etc, X_eth, X_link, X_ltc, X_matic, X_sol, X_xrp ])
y_combined = pd.concat([y_aave, y_ada, y_avax, y_bnb, y_btc, y_dot, y_etc, y_eth, y_link, y_ltc, y_matic, y_sol, y_xrp])

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