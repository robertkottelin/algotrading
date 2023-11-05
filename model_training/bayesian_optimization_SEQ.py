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
from sklearn.preprocessing import StandardScaler
import optuna
import datetime

# Directory containing your CSV files
data_dir = '/content/drive/MyDrive/Algotrading/SNPMacroTechnicalFearNGreedPrepped.csv'
# Ensure the 'models' directory exists
os.makedirs('models', exist_ok=True)
LOG_FILE = "/content/drive/MyDrive/Algotrading/optuna_trials_DNN_SNP.log"

def log_trial(trial, trial_result):
    try:
        with open(LOG_FILE, "a") as log_file:
            log_message = f"Trial {trial.number} finished with value: {trial_result} and parameters: {trial.params}.\n"
            log_file.write(log_message)
    except Exception as e:
        print(f"Error logging trial {trial.number}: {e}")

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
    try:
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
    except Exception as e:
        print(f"Failed to plot training history: {e}")

def objective(trial):
    # Hyperparameters to be tuned by Optuna
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512, 1024])
    epochs = trial.suggest_int('epochs', 100, 400)

    # Architecture hyperparameters
    num_layers = trial.suggest_int("num_layers", 4, 8, step=1)
    dropout_rate = trial.suggest_float("dropout_rate", 0.01, 0.2)

    # ReduceLROnPlateau parameters
    reduction_factor = trial.suggest_float("reduction_factor", 0.8, 0.99)
    patience = trial.suggest_int("patience", 10, 20)
    min_lr = trial.suggest_float("min_lr", 1e-6, 1e-2, log=True)

    # Model building
    model = Sequential()
    model.add(Dense(trial.suggest_int('units_layer_0', 256, 2048, step=256), input_dim=X_train.shape[1], activation=trial.suggest_categorical('activation_layer_0', ['relu', 'tanh', 'sigmoid'])))
    model.add(Dropout(rate=dropout_rate))

    # Adding variable layers based on the trial suggestion
    for i in range(1, num_layers):
        model.add(Dense(trial.suggest_int(f'units_layer_{i}', 256, 2048, step=256), activation=trial.suggest_categorical(f'activation_layer_{i}', ['relu', 'tanh', 'sigmoid'])))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=lr),
        metrics=['accuracy']
    )

    # Callbacks
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=reduction_factor, patience=patience, min_lr=min_lr)
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    # Training the model
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )

    # Evaluate the model
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)

    log_trial(trial, accuracy)

    return accuracy

# Load and preprocess data
try:
    df = pd.read_csv(data_dir)
    X = df.drop(['Next_Higher', 'Date'], axis=1)
    y = df['Next_Higher']
    # Preprocessing steps (scaling, encoding, etc.)
except Exception as e:
    print(f"Failed to load or preprocess data: {e}")
    exit(1)

# Data splitting and scaling
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    # Initialize the StandardScaler
    scaler = StandardScaler()
    # Fit only on training data
    scaler.fit(X_train)
    # Apply the transformations to the data:
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
except Exception as e:
    print(f"Failed to split or scale data: {e}")
    exit(1)

# Optuna study
try:
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10000)
except Exception as e:
    print(f"Optuna optimization failed: {e}")
    exit(1)
    
import joblib
try:
    joblib.dump(scaler, 'models/scaler.gz')
except Exception as e:
    print(f"Failed to save the scaler: {e}")

# After the study, we extract the best parameters
best_params = study.best_trial.params
print(f"Best hyperparameters: {best_params}")

# Training the final model with the best parameters from the study
final_model = build_very_large_model(input_dim=X_train.shape[1], lr=best_params['lr'])

try:
    final_history = final_model.fit(
        X_train, y_train,
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        validation_data=(X_test, y_test),
        verbose=2
    )
except Exception as e:
    print(f"Failed to train the final model: {e}")

try:
    final_model.save('models/final_model.h5')
    # Save scaler or any preprocessing objects if needed
except Exception as e:
    print(f"Failed to save the final model: {e}")

# Evaluate the final model
try:
    final_loss, final_accuracy = final_model.evaluate(X_test, y_test, verbose=0)
    print(f"Final Model - Test Loss: {final_loss} - Test Accuracy: {final_accuracy}")
except Exception as e:
    print(f"Failed to evaluate the final model: {e}")

# Plot training history
try:
    plot_training_history(final_history)
except Exception as e:
    print(f"Failed to plot training history: {e}")

