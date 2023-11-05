import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import optuna
import joblib

# Directory containing your CSV files
# data_dir = os.path.join('data', 'SNP', 'SNPMacroTechnicalFearNGreedPrepped.csv')

# Ensure the 'models' directory exists
os.makedirs('models', exist_ok=True)
LOG_FILE = "logs/optuna_trials_randomforest_SNP.log"

def log_trial(trial, trial_result):
    try:
        with open(LOG_FILE, "a") as log_file:
            log_message = f"Trial {trial.number} finished with value: {trial_result} and parameters: {trial.params}.\n"
            log_file.write(log_message)
    except Exception as e:
        print(f"Error logging trial {trial.number}: {e}")

# Load and preprocess data
try:
    df = pd.read_csv('data/SNP/SNPMacroTechnicalFearNGreedPrepped.csv')

    # smaller dataset for testing
    # df_sampled = df.sample(frac=0.1, random_state=42)
    X = df.drop('Next_Higher', axis=1)
    X = X.drop('Date', axis=1)
    y = df['Next_Higher'].astype(int)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
except Exception as e:
    print(f"Failed to load and preprocess data: {e}")
    exit(1)

# Data splitting
try:
    # test_size_proportion = 0.01 * (len(df_sampled) / len(df))
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_proportion, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
except Exception as e:
    print(f"Failed to split data: {e}")
    exit(1)

def objective(trial):
    try:
        print(f"Starting trial {trial.number}")
        # Hyperparameters to be tuned by Optuna
        n_estimators = trial.suggest_int('n_estimators', 100, 1000)
        max_depth = trial.suggest_int('max_depth', 5, 100)         
        min_samples_split = trial.suggest_int('min_samples_split', 2, 200)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 100)
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

        print(f"Trial {trial.number} parameters: "
              f"n_estimators={n_estimators}, "
              f"max_depth={max_depth}, "
              f"min_samples_split={min_samples_split}, "
              f"min_samples_leaf={min_samples_leaf}, "
              f"criterion={criterion}")

        # Create a random forest classifier
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=42,
            n_jobs=-1,
            verbose=1      
        )

        # Train the model
        clf.fit(X_train, y_train)

        # Make predictions
        predictions = clf.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)

        print(f"Trial {trial.number} finished with accuracy: {accuracy}")

        # Log the trial result
        log_trial(trial, accuracy)

        return accuracy
    except Exception as e:
        print(f"Error in trial {trial.number}: {e}")
        return None

# Optuna study
try:
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=10000)
except Exception as e:
    print(f"Optuna optimization failed: {e}")
    exit(1)

# Best trial results
best_trial = study.best_trial
print(f"Best Accuracy: {best_trial.value}")
print("Best hyperparameters: ", best_trial.params)

# Log the best trial result
log_trial(best_trial, best_trial.value)

# Retrieve the best hyperparameters
best_params = best_trial.params

# Final model training with best hyperparameters
try:
    final_model = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        criterion=best_params["criterion"],
        random_state=42
    )

    final_model.fit(X_train, y_train)
except Exception as e:
    print(f"Failed to train the final model: {e}")
    exit(1)

# Save the final model
try:
    joblib.dump(final_model, os.path.join('models', 'final_optimized_random_forest.joblib'))
except Exception as e:
    print(f"Failed to save the final model: {e}")
    exit(1)

# Evaluate the final model
try:
    final_predictions = final_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, final_predictions)
    print(f"Final Model - Test Accuracy: {final_accuracy}")
except Exception as e:
    print(f"Failed to evaluate the final model: {e}")
    exit(1)
