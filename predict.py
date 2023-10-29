import os
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path, scaler):
    # Load the data from the CSV file
    df = pd.read_csv(file_path)
    
    # We will predict using the latest entry
    latest_entry = df.iloc[-1:]  # Selecting the last row
    
    X = latest_entry

    # Apply the same preprocessing applied when training the model
    X_scaled = scaler.transform(X)

    return X_scaled

def main():
    data_dir = 'data/macrotechnicalfearandgreed/'
    models_dir = 'models/'
    model_name = 'combined_model.h5' 

    # Load the model
    model_path = os.path.join(models_dir, model_name)
    model = tf.keras.models.load_model(model_path)

    # Prepare the data scaler - assuming you have the same scaler used during training
    # Ideally, you want to save your scaler to a file after training and load it here.
    # For demonstration, we are initializing a new one, but in practice, this should be the same scaler you used during training.
    scaler = StandardScaler()

    # Assume you have one or multiple CSV files in the directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_dir, filename)
            
            # Preprocess the data
            X_latest = load_and_preprocess_data(file_path, scaler)

            # Make a prediction
            prediction = model.predict(X_latest)
            
            # Since it's a binary classification, the model outputs probabilities
            # You might want to convert this to a class label
            predicted_class = (prediction > 0.5).astype("int32")

            if predicted_class == 1:
                print(f"The model predicts that the market in {filename} will go up.")
            else:
                print(f"The model predicts that the market in {filename} will go down.")

if __name__ == "__main__":
    main()
