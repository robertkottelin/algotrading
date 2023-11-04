
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import optuna

# Directory containing your CSV files
data_dir = 'data/preppedconcatdata/combined_data.csv'

# Ensure the 'models' directory exists
os.makedirs('models', exist_ok=True)
LOG_FILE = "optuna_trials_randomforest.log"

def log_trial(trial, trial_result):
    with open(LOG_FILE, "a") as log_file:
        log_message = f"Trial {trial.number} finished with value: {trial_result} and parameters: {trial.params}.\n"
        log_file.write(log_message)

def objective(trial):
    # Hyperparameters to be tuned by Optuna
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_int('max_depth', 10, 100)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 15)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

    # Create a random forest classifier
    # Create a random forest classifier
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=42
    )


    # Train the model
    clf.fit(X_train, y_train)

    # Make predictions
    predictions = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    log_trial(trial, accuracy)

    return accuracy

# Load and preprocess data
df = pd.read_csv(data_dir)
X = df.drop('Next_Higher', axis=1)
y = df['Next_Higher']
y = y.astype(int)
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Data splitting
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
best_params = best_trial.params

# Final model training with best hyperparameters
final_model = RandomForestClassifier(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    min_samples_split=best_params["min_samples_split"],
    min_samples_leaf=best_params["min_samples_leaf"],
    max_features=best_params["max_features"],
    criterion=best_params["criterion"],
    random_state=42
)

final_model.fit(X_train, y_train)

# Save the final model
import joblib
joblib.dump(final_model, 'models/final_optimized_random_forest.joblib')

# Evaluate the final model
final_predictions = final_model.predict(X_test)
final_accuracy = accuracy_score(y_test, final_predictions)
print(f"Final Model - Test Accuracy: {final_accuracy}")
