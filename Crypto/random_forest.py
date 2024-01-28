import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_csv('Crypto/coinglass_ta.csv')

# Shift the 'c' column to predict the future price direction
shift_rows = 6
df['c_future'] = df['c'].shift(-shift_rows)

# Create a binary target variable for price direction
df['c_future_direction'] = (df['c_future'] > df['c']).astype(int)

# Drop the last 'shift_rows' rows where the future price is NaN
df = df.dropna()

# Define features (X) and target (y)
X = df.drop(columns=['c_future', 'c_future_direction', 't'])  # Assuming 't' is a timestamp column
y = df['c_future_direction']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.01, random_state=42)

# Initialize the Random Forest Classifier with GridSearch for hyperparameter tuning
param_grid = {
    'n_estimators': [10, 20, 40, 80, 100, 150, 200],  # Number of trees in the forest
    'max_depth': [5, 10, 20, 40, 50, None],  # Maximum depth of the tree
    # Add more parameters here if desired
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=2, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best estimator found by grid search
best_model = grid_search.best_estimator_

# Predict and evaluate the model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
