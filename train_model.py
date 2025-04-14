import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor
import joblib

# Load data
train_data = pd.read_csv("TrainingData.csv")

# Rename target column to match what's in the CSV
target_column = 'Optimized_Performance'

# Separate features (X) and target (y)
X = train_data.drop(columns=[target_column])
y = train_data[target_column]

# Encode categorical columns
categorical_columns = ['Make', 'Model', 'Version', 'Grade']
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le  # save encoder for test data

# Drop VIN since it’s just an identifier
X = X.drop(columns=['VIN'])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define and train the model
model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='logistic',
    solver='adam',
    max_iter=2000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42
)

model.fit(X_scaled, y)

# Save the model, scaler, and encoders
joblib.dump(model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(label_encoders, "encoders.joblib")

print("✅ Model training complete and saved successfully.")
