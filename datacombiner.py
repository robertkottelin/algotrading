import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Directory containing your CSV files
data_dir = 'data/preppeddata/'
output_dir = 'data/preppedconcatdata/'

# Check if output directory exists, if not, create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_and_combine_data(data_directory):
    combined_data = pd.DataFrame()
    scaler = StandardScaler()

    # Save the scaler
    scaler_filepath = "models/scaler.save"
    dump(scaler, scaler_filepath)

    for filename in os.listdir(data_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_directory, filename)
            df = pd.read_csv(file_path)

            # Extract X without the target column
            X = df.drop('Next_Higher', axis=1)
            # Normalize the data
            X_scaled = scaler.fit_transform(X)

            # Merge the scaled data back with the target column
            df_scaled = pd.concat([pd.DataFrame(X_scaled, columns=X.columns), df['Next_Higher'].reset_index(drop=True)], axis=1)
            
            combined_data = pd.concat([combined_data, df_scaled], ignore_index=True)

    return combined_data

def main():
    # Load and combine data
    combined_data = load_and_combine_data(data_dir)

    # Save the combined and normalized data to a new CSV file
    combined_data.to_csv(os.path.join(output_dir, 'combined_data.csv'), index=False)

if __name__ == "__main__":
    main()
