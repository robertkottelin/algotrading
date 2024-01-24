import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from joblib import dump
import optuna
import datetime

# Directory containing your CSV files
data_dir = 'data/preppedconcatdata/combined_data.csv'

# Ensure the 'models' directory exists
os.makedirs('models', exist_ok=True)
LOG_FILE = "optuna_trials.log"
def log_trial(trial, trial_result):
    with open(LOG_FILE, "a") as log_file:  # "a" means append mode, which won't overwrite existing content
        log_message = f"Trial {trial.number} finished with value: {trial_result} and parameters: {trial.params}.\n"
        log_file.write(log_message)

def build_very_large_model(input_dim, lr):
    model = Sequential([
        # Input layer
        Dense(512, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        # Hidden layers
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        # Output layer
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=lr),
        metrics=['accuracy']
    )

    return model

def plot_training_history(history):
    plt.figure(figsize=(12, 6))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.tight_layout()
    plt.show()

def objective(trial):
    # Hyperparameters to be tuned by Optuna
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])
    epochs = trial.suggest_int('epochs', 100, 200)

    # Architecture hyperparameters
    num_layers = trial.suggest_int("num_layers", 4, 6)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.2)
    
    # ReduceLROnPlateau parameters
    reduction_factor = trial.suggest_float("reduction_factor", 0.8, 0.95)
    patience = trial.suggest_int("patience", 2, 4)
    min_lr = trial.suggest_float("min_lr", 1e-5, 1e-2, log=True)

    # Model building
    model = Sequential()
    model.add(Dense(trial.suggest_int('units_layer_0', 64, 1024), input_dim=X_train.shape[1], activation=trial.suggest_categorical('activation_layer_0', ['relu', 'tanh', 'sigmoid'])))
    model.add(Dropout(rate=dropout_rate))

    # Adding variable layers based on the trial suggestion
    for i in range(1, num_layers):
        model.add(Dense(trial.suggest_int(f'units_layer_{i}', 64, 1024), activation=trial.suggest_categorical(f'activation_layer_{i}', ['relu', 'tanh', 'sigmoid'])))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=lr),
        metrics=['accuracy']
    )

    # Callbacks
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=reduction_factor, patience=patience, min_lr=min_lr)
    early_stopping = EarlyStopping(monitor='val_loss', patience=4)

    # Training the model
    history = model.fit(
        X_train, 
        y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=(X_test, y_test), 
        callbacks=[early_stopping, lr_scheduler],
        verbose=2
    )

    # Evaluate the model
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)

    log_trial(trial, accuracy)

    return accuracy


# Load and preprocess data
df = pd.read_csv(data_dir)
X = df.drop('Next_Higher', axis=1)
y = df['Next_Higher']

# Data splitting and scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Best trial results
best_trial = study.best_trial
print(f"Best Accuracy: {best_trial.value}")
print("Best hyperparameters: ", best_trial.params)

# Log the best trial
log_trial(best_trial)

# Retrieve the best hyperparameters
best_lr = best_trial.params["lr"]
best_batch_size = best_trial.params["batch_size"]
best_epochs = best_trial.params["epochs"]
best_params = study.best_trial.params

# Final model training
final_model = build_very_large_model(X_train.shape[1], best_lr)
final_history = final_model.fit(
    X_train, 
    y_train, 
    epochs=best_epochs, 
    batch_size=best_batch_size, 
    validation_data=(X_test, y_test), 
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=3),
        ReduceLROnPlateau(monitor='val_loss', factor=best_params["reduction_factor"], patience=best_params["patience"], min_lr=best_params["min_lr"])
    ],
    verbose=2
)

# Save and evaluate the final model
final_model.save("models/final_optimized_model.h5")
final_loss, final_accuracy = final_model.evaluate(X_test, y_test)
print(f"Final Model - Test Loss: {final_loss} - Test Accuracy: {final_accuracy}")

# Training history plot
plot_training_history(final_history)
