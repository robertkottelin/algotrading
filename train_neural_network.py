import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt

# Directory containing your CSV files
data_dir = 'data/preppeddata/'

# Function to load and preprocess data
def load_and_combine_data(data_directory):
    combined_data = pd.DataFrame()
    for filename in os.listdir(data_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_directory, filename)
            df = pd.read_csv(file_path)
            combined_data = pd.concat([combined_data, df], ignore_index=True)
    return combined_data

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

def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.tight_layout()
    plt.show()

# Load and combine data
combined_data = load_and_combine_data(data_dir)

# Assuming the 'Next_Higher' column is your target
X = combined_data.drop('Next_Higher', axis=1)
y = combined_data['Next_Higher']

# Standardize the features (important for neural network training)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# Prepare a learning rate scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=3, min_lr=1e-4)

# Set higher batch_size and epochs due to the availability of substantial computational resources
batch_size = 512 # decrease? increase?
epochs = 10 

# Build the model (using the more complex version here)
model = build_very_large_model(X_train.shape[1])  # input_dim is the number of features

# Train the model with our new parameters
history = model.fit(
    X_train, 
    y_train, 
    epochs=epochs, 
    batch_size=batch_size, 
    validation_data=(X_test, y_test), 
    callbacks=[lr_scheduler]  # Using learning rate scheduler
)

# Save model
model.save("models/combined_model.h5")  # or another appropriate path

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Model for combined dataset - Test Loss: {loss} - Test Accuracy: {accuracy}")

# Plot the training history
plot_training_history(history)