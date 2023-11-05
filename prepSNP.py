import os
import pandas as pd

def process_file(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Load the dataframe
    df = pd.read_csv(file_path)

    # Check if 'Close' column exists
    if 'Close' not in df.columns:
        raise ValueError("The 'Close' column is not present in the data.")

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        raise ValueError("The 'Date' column is not present in the data.")

    cutoff_date = pd.to_datetime('1987-07-24')
    df = df[df['Date'] > cutoff_date]

    # Create a new column based on the comparison of 'Close' prices between rows
    df['Next_Higher'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Add the last 5 days' close prices as new columns
    for i in range(1, 10):
        df[f'Close_{i}d_ago'] = df['Close'].shift(i)

    # drop rows with NaN values
    df.dropna(inplace=True)

    df.drop(columns=['Open', 'Dividends', 'Stock Splits'], inplace=True)

    return df

def main():
    input_file_path = 'data/SNP/SNPMacroTechnicalFearNGreed.csv'
    output_file_path = 'data/SNP/SNPMacroTechnicalFearNGreedPrepped.csv'

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Process the file
    try:
        new_df = process_file(input_file_path)

        # Save the updated dataframe to the new location
        new_df.to_csv(output_file_path, index=False)
        print(f"Processed and saved to {output_file_path}")

    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)

# Check if the script is being run directly (and not imported)
if __name__ == "__main__":
    main()
