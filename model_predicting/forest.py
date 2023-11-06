from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming X_train is your training data and y_train are the labels
# And your best parameters have been determined as mentioned in your message
model = RandomForestClassifier(
    n_estimators=225,
    max_depth=12,
    min_samples_split=53,
    min_samples_leaf=55,
    criterion='gini',
    # concurrent workers
    n_jobs=-1,
    random_state=42, # for reproducibility
    verbose=1
)

df = pd.read_csv('data/SNP/SNPFinal.csv')

# smaller dataset for testing
# df_sampled = df.sample(frac=0.1, random_state=42)
X = df.drop('Next_Higher', axis=1)
X = X.drop('Date', axis=1)
y = df['Next_Higher'].astype(int)

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy}")

# Get feature importances
importances = model.feature_importances_

# To make it easier to interpret, you can sort the features by importance
sorted_indices = np.argsort(importances)[::-1]

# Assuming X_train is a DataFrame, this will print the feature names and their importance
for index in sorted_indices:
    print(f"{X_train.columns[index]}: {importances[index]}")

