import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime, timedelta

# Load the pre-trained Random Forest model
model_path = 'models/final_optimized_random_forest.joblib'
model = joblib.load(model_path)

# Load and sort the new dataset by date
df_path = 'data/SNP/SNPFinal.csv'
df = pd.read_csv(df_path)
df.sort_values('Date', inplace=True)

# drop all data before 2020-01-01
df = df[df['Date'] >= '2020-01-01']

# Preprocess the dataset: Drop the 'Date' column and any target label column if present
X = df.drop(['Date', 'Next_Higher'], axis=1, errors='ignore')

# Make predictions with the model
predictions = model.predict(X)

# Get the last prediction (for the most recent day)
most_recent_prediction = predictions[-1]
print(f"The prediction for the most recent day ({df.iloc[-1]['Date']}) is: {most_recent_prediction}")

# Simulate investing starting one year ago with 1000 USD
starting_capital = 1000
portfolio_value = [starting_capital]

# Assuming 'Next_Higher' is the column with actual outcomes
if 'Next_Higher' in df:
    # Get actual outcomes
    y = df['Next_Higher'].astype(int)

    # Define gain and loss factors
    gain_factor = 1.02  # 1% gain per correct prediction
    loss_factor = 0.98  # 1% loss per incorrect prediction

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

    print(f"Final portfolio value: {portfolio_value_last_1_year[-1]} USD")

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
