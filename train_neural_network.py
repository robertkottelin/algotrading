import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization

# Directory containing your CSV files
data_dir = 'data/preppeddata/'

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Assuming the 'Next_Higher' column is your target
    X = df.drop('Next_Higher', axis=1)
    y = df['Next_Higher']

    # Standardize the features (important for neural network training)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    return train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# Define the neural network model
def build_advanced_model(input_dim):
    model = Sequential()

    # Input layer
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(BatchNormalization())  # Normalize the activations of the previous layer at each batch
    model.add(Dropout(0.2))  # Randomly sets a fraction rate of input units to 0 at each update during training
    # Hidden layer 1
    model.add(Dense(256, activation='relu'))  # Larger layer
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    # Hidden layer 2
    model.add(Dense(128, activation='relu'))  # Intermediate layer
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    # Hidden layer 3
    model.add(Dense(64, activation='relu'))  # Smaller layer, potentially nearing output size
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    # Output layer: Since we are doing binary classification, we use a single neuron with sigmoid activation.
    model.add(Dense(1, activation='sigmoid'))  # binary classification
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def build_very_large_model(input_dim):
    model = Sequential()

    # Input layer
    model.add(Dense(512, input_dim=input_dim, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))  # less dropout in the first layer

    # Hidden layer 1
    model.add(Dense(1024, activation='relu'))  # very large layer
    model.add(BatchNormalization())
    model.add(Dropout(0.2))  # increasing dropout for subsequent layers

    # Hidden layer 2
    model.add(Dense(1024, activation='relu'))  # very large layer
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Hidden layer 3
    model.add(Dense(512, activation='relu'))  # reducing the size to start preparing for final classification
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Hidden layer 4
    model.add(Dense(256, activation='relu'))  # intermediary layer
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Hidden layer 5
    model.add(Dense(128, activation='relu'))  # smaller layer
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))  # for binary classification

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Iterate over files and train a model for each
for filename in os.listdir(data_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(data_dir, filename)
        
        # Load and preprocess the data from the file
        X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

        # Build the model
        model = build_advanced_model(X_train.shape[1])  # input_dim is the number of features

        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

        # After training, you might want to save the model, evaluate it, or further fine-tune it.
        # Save model
        model.save(f"models/model_for_{filename}.h5")  # or another appropriate path

        # Evaluate model
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"Model for {filename} - Test Loss: {loss} - Test Accuracy: {accuracy}")
