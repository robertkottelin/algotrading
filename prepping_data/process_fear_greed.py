import pandas as pd

# Define the path to the Excel file
file_path = 'data/fearandgreed/sentiment.xls'

# Load the necessary columns from the Excel file
df = pd.read_excel(file_path, usecols="A:D", skiprows=4)

# Rename the columns for convenience
df.columns = ['Date', 'Bullish', 'Neutral', 'Bearish']

# Convert the 'Date' column to datetime, ignoring parsing errors
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Remove the rows where the date could not be parsed (they will have NaT in the 'Date' column)
df = df.dropna(subset=['Date'])

# Ensure the 'Date' column is of dtype datetime64[ns]
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' as the DataFrame index
df.set_index('Date', inplace=True)

# Create a new DataFrame that includes every day up to today
full_date_range = pd.date_range(start=df.index.min(), end=pd.Timestamp('today').normalize())

# Reindex the dataframe with the full date range, filling forward the sentiment values
df = df.reindex(full_date_range).fillna(method='ffill').reset_index()

# Rename the 'index' column back to 'Date'
df.rename(columns={'index': 'Date'}, inplace=True)

# Convert the 'Date' back to the string format YYYY-MM-DD if necessary
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

# Export the DataFrame to a CSV file
csv_file_path = 'data/fearandgreed/fearandgreed.csv'
df.to_csv(csv_file_path, index=False, date_format='%Y-%m-%d')

print(f'CSV file saved to {csv_file_path}')
