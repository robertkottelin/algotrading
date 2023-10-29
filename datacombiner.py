# data_combiner.py

import os
import pandas as pd

# Directory containing your CSV files
data_dir = 'data/preppeddata/'
output_dir = 'data/preppedconcatdata/'

# Check if output directory exists, if not, create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_and_combine_data(data_directory):
    """
    Loads and combines data from multiple CSV files located within
    a specific directory into a single DataFrame.

    :param data_directory: Path to the directory containing the CSV files.
    :return: A DataFrame consisting of all combined data.
    """
    combined_data = pd.DataFrame()
    for filename in os.listdir(data_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_directory, filename)
            df = pd.read_csv(file_path)
            combined_data = pd.concat([combined_data, df], ignore_index=True)

    return combined_data

def main():
    # Load and combine data
    combined_data = load_and_combine_data(data_dir)

    # Save the combined data to a new CSV file
    combined_data.to_csv(os.path.join(output_dir, 'combined_data.csv'), index=False)

if __name__ == "__main__":
    main()
