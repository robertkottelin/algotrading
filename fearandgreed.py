import pandas as pd

# Load and process the fear and greed index data
data = pd.read_csv('data/fearandgreed.csv')
data['Date'] = pd.to_datetime(data['Date'], format='%m-%d-%y')
data['Date'] = pd.to_datetime(data['Date'].dt.strftime('%Y-%m-%d'))

# Remove '%' and convert to float
columns_to_update = ['Bullish', 'Neutral', 'Bearish']
for col in columns_to_update:
    data[col] = data[col].str.rstrip('%').astype('float') / 100.0

# Create a complete date range and reindex the DataFrame
start_date = data['Date'].min()
end_date = data['Date'].max()
all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
data.set_index('Date', inplace=True)
data = data.reindex(all_dates)

# Forward-fill the missing values
data.fillna(method='ffill', inplace=True)

# Reset the index
data.reset_index(inplace=True)
data.rename(columns={'index': 'Date'}, inplace=True)

# Load the macroeconomic data (replace 'macrodata.csv' with your actual filename)
macro_data = pd.read_csv('data/macrodata.csv')
macro_data['Date'] = pd.to_datetime(macro_data['Date'])  # assuming date format here is recognized by pandas

# Merge the two datasets on the 'Date' column
merged_data = pd.merge(data, macro_data, on='Date', how='inner')

# Check the result
print(merged_data.head())

# Save the merged dataset to a new CSV file
merged_data.to_csv('data/macrotechnicalfearandgreed.csv', index=False)
