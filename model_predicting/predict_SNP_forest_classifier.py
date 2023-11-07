import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime, timedelta
import os

RISK_FREE_RATE = 0.04 / 252  # Assuming a risk-free rate of 5% per annum, daily

# Load the pre-trained Random Forest model
model_path = 'models/final_optimized_random_forest.joblib'
model = joblib.load(model_path)

# Load and sort the new dataset by date
df_path = 'data/SNP/SNPFinal.csv'
df = pd.read_csv(df_path)
df.sort_values('Date', inplace=True)

# drop all data before 2020-01-01
df = df[df['Date'] >= '2020-01-01']

# Separate the last row before dropping NaN values
last_row = df.iloc[-1:]
df = df.iloc[:-1]

# Drop rows with NaN values in the main dataset
df = df.dropna()

# Preprocess the dataset: Drop the 'Date' column and any target label column if present
X = df.drop(['Date', 'Next_Higher'], axis=1, errors='ignore')

# Make predictions with the model on the dataset without the last row
predictions = model.predict(X)
probability_predictions = model.predict_proba(X)

# Process the last row separately:
# Handle NaN values for the last row before prediction
# Replace NaN values with the median of each column in the main dataset
medians = df.median(numeric_only=True)
X_last_row_filled = last_row.drop(['Date', 'Next_Higher'], axis=1, errors='ignore').fillna(medians)

# Predict on the last row with NaNs handled
last_row_prediction = model.predict(X_last_row_filled)
last_row_probability = model.predict_proba(X_last_row_filled)

# Get the probability for the predicted class in the last row
last_row_probability_predicted_class = last_row_probability[0][np.argmax(last_row_probability)]

# Format the date to include the day of the week for the last row
date_with_day_last_row = pd.to_datetime(last_row['Date']).dt.strftime('%A, %Y-%m-%d').values[0]
prediction_text = "Bullish" if last_row_prediction[0] == 1 else "Bearish"

# Combine the statements to print both the prediction and its probability for the last row
print(f"The prediction for the SNP {date_with_day_last_row} is: {prediction_text} with a probability of {last_row_probability_predicted_class:.2%}")

# Simulate investing starting one year ago with 1000 USD
starting_capital = 1000
portfolio_value = [starting_capital]

# Assuming 'Next_Higher' is the column with actual outcomes
if 'Next_Higher' in df:
    # Get actual outcomes
    y = df['Next_Higher'].astype(int)

    # Define gain and loss factors
    gain_factor = 1.03  # 3% gain per correct prediction
    loss_factor = 0.99  # 1% loss per incorrect prediction

    # Calculate portfolio value over time
    for prediction, actual in zip(predictions, y):
        if prediction == actual:
            # Apply gain if prediction is correct
            portfolio_value.append(portfolio_value[-1] * gain_factor)
        else:
            # Apply penalty if prediction is wrong
            portfolio_value.append(portfolio_value[-1] * loss_factor)

    # Convert to a numpy array for easier slicing
    portfolio_value = np.array(portfolio_value)

    # Get the date 1 year ago from the last date in the dataset
    last_date = datetime.strptime(df.iloc[-1]['Date'], '%Y-%m-%d')
    date_1_year_ago = last_date - timedelta(days=365)

    # Filter the dataframe to the last year
    df_last_1_year = df[df['Date'] >= date_1_year_ago.strftime('%Y-%m-%d')]
    # Ensure the length of the portfolio array matches the date array
    portfolio_value_last_1_year = portfolio_value[-len(df_last_1_year['Date']):]

    # Calculate daily returns
    daily_returns = np.diff(portfolio_value_last_1_year) / portfolio_value_last_1_year[:-1]

    # Calculate gross profits and gross losses
    gross_profits = np.sum(daily_returns[daily_returns > 0])
    gross_losses = -np.sum(daily_returns[daily_returns < 0])
    profit_factor = gross_profits / gross_losses if gross_losses != 0 else np.inf

    # Calculate Sharpe Ratio
    excess_returns = daily_returns - RISK_FREE_RATE
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else np.inf

    # print(f"Final simulated portfolio value: {portfolio_value_last_1_year[-1]} USD")

    print(os.path.basename(__file__))
    print(f"Simulated with a 1/3 risk-reward (RR) ratio.")
    print(f"Last year's Profit Factor: {profit_factor:.2f}")
    print(f"Last year's Sharpe Ratio: {sharpe_ratio:.2f}")

    plt.figure(figsize=(14, 7))
    plt.plot(df_last_1_year['Date'], portfolio_value_last_1_year, label='Portfolio Value')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (USD)')
    plt.title('Simulated Investment Portfolio Value Over the Last Year')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

else:
    print("Cannot simulate investment as 'Next_Higher' column is not present in the dataset.")
