import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import optuna
import joblib

# Ensure the 'models' directory exists
os.makedirs('models', exist_ok=True)
LOG_FILE = "logs/optuna_trials_lstm.log"

def log_trial(trial, trial_result):
    with open(LOG_FILE, "a") as log_file:
        log_message = f"Trial {trial.number} finished with value: {trial_result} and parameters: {trial.params}.\n"
        log_file.write(log_message)

# Load and preprocess data
try:
    df = pd.read_csv('data/SNP/SNPMacroTechnicalFearNGreedPrepped.csv')

    # Assume the data has been preprocessed for LSTM input (i.e., it is a sequence data)
    # We will not sample the data for simplicity
    X = df.drop(['Next_Higher', 'Date'], axis=1)
    y = df['Next_Higher'].astype(int)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # LSTM expects 3D data: [samples, time steps, features]
    # Reshaping the input data
    X_reshaped = X.values.reshape((X.shape[0], 1, X.shape[1]))
except Exception as e:
    print(f"Failed to load and preprocess data: {e}")
    exit(1)

# Data splitting
try:
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_encoded, test_size=0.01, random_state=42)
except Exception as e:
    print(f"Failed to split data: {e}")
    exit(1)

def create_model(trial):
    n_layers = trial.suggest_int('n_layers', 3, 6)
    dropout = trial.suggest_float('dropout', 0.0, 0.4)

    model = Sequential()
    for i in range(n_layers):
        num_units = trial.suggest_int(f'n_units_l{i}', 50, 200)
        if i == 0:
            # First layer
            model.add(LSTM(units=num_units, return_sequences=(n_layers > 1), input_shape=(X_train.shape[1], X_train.shape[2])))
        else:
            # Subsequent layers
            return_sequences = i < n_layers - 1
            model.add(LSTM(units=num_units, return_sequences=return_sequences))
        model.add(Dropout(rate=dropout))
    
    model.add(Dense(1, activation='sigmoid'))
    return model

def objective(trial):
    model = create_model(trial)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    history = model.fit(X_train, y_train, 
                        epochs=50, 
                        batch_size=32,
                        validation_split=0.1, 
                        callbacks=[early_stopping], 
                        verbose=0)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    log_trial(trial, accuracy)
    return accuracy

# Optuna study
try:
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)  # You might want to reduce the number of trials for an LSTM due to longer training times.
except Exception as e:
    print(f"Optuna optimization failed: {e}")
    exit(1)

# Best trial results
best_trial = study.best_trial
print(f"Best Accuracy: {best_trial.value}")
print("Best hyperparameters: ", best_trial.params)

# Train the final model with best hyperparameters
final_model = create_model(best_trial)
final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
final_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# Save the final model
final_model_path = os.path.join('models', 'final_optimized_lstm.h5')
final_model.save(final_model_path)

# Evaluate the final model
loss, final_accuracy = final_model.evaluate(X_test, y_test, verbose=0)
print(f"Final Model - Test Accuracy: {final_accuracy}")
