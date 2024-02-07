
# Load the processed data
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Coinglass/ta_data/coinglass_Binance_BTCUSDT.csv')

# Sort the DataFrame by date, descending
# df = df.sort_values(by='t', ascending=False)

import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Function to preprocess and scale the data
def preprocess_data_for_prediction(df, scaler):
    # Assuming the same preprocessing as during training
    X = df.drop(columns=['t'])  # Drop columns that are not features
    X_scaled = scaler.transform(X)  # Scale the features
    return X_scaled

# Load the saved model
model = tf.keras.models.load_model('/content/drive/MyDrive/Colab Notebooks/Coinglass/models/crypto_1d_model')

# Preprocess the new data
scaler = StandardScaler()  # Initialize a new scaler
scaler.fit(df.drop(columns=['t']))  # Fit the scaler to the new data
X_new = preprocess_data_for_prediction(df, scaler)

# Predict the future price direction
predictions = model.predict(X_new)
predicted_direction = (predictions > 0.5).astype(int)  # Convert probabilities to binary predictions

# Add predictions to the DataFrame
df['predicted_direction'] = predicted_direction.flatten()

# Display the DataFrame with predictions
# sort newest to oldest
df = df.sort_values(by='t', ascending=True)
# drop all columns except for t, BTC price and predicted_direction
df = df[['t', 'BTC price', 'predicted_direction']]



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming df is already defined and sorted by 't'
trade_amount = 1000.0  # Fixed trade size
initial_account_balance = 1000.0  # Starting account balance
account_value = initial_account_balance
max_loss = 0.0
last_direction = df.iloc[0]['predicted_direction'] - 1  # Initialize to trigger a trade on the first row
account_values = [account_value]  # Track account value over time

# Variables for managing positions
btc_held = 0.0
trade_in_progress = False
last_trade_price = 0.0

# Iterate over the DataFrame rows
for index, row in df.iterrows():
    current_price = row['prices']
    predicted_direction = row['predicted_direction']
    
    # Check if we need to close the current position and open a new one
    if predicted_direction != last_direction:
        # Close existing position if any
        if trade_in_progress:
            if last_direction == 1:  # If last trade was long, sell BTC
                account_value += (current_price - last_trade_price) * btc_held
            elif last_direction == 0:  # If last trade was short, cover BTC
                account_value += (last_trade_price - current_price) * btc_held
            
            # Calculate and update max loss after closing position
            change_in_value = account_value - initial_account_balance
            max_loss = min(max_loss, change_in_value)
            initial_account_balance = account_value  # Reset initial balance for the next trade
        
        # Open new position with fixed trade amount of 1000 USD
        btc_held = trade_amount / current_price
        last_trade_price = current_price
        trade_in_progress = True
        
        last_direction = predicted_direction
    
    # Append current account value to the list after every iteration
    account_values.append(account_value)

# Ensure df and account_values have the same length
if len(account_values) > len(df):
    account_values = account_values[:-1]  # Adjust if off by one due to additional append

df['account_value'] = account_values

# Plotting
plt.figure(figsize=(14, 7))

# Prices plot
plt.plot(df['t'], df['prices'], label='BTC Price', color='blue')
plt.xlabel('Time')
plt.ylabel('BTC Price', color='blue')
plt.tick_params(axis='y', labelcolor='blue')

# Account value plot with secondary y-axis
ax2 = plt.gca().twinx()
ax2.plot(df['t'], df['account_value'], label='Account Value', color='green')
ax2.set_ylabel('Account Value', color='green')
ax2.tick_params(axis='y', labelcolor='green')

plt.title('BTC Price and Account Value Over Time')
plt.legend(loc='upper left')
plt.show()
