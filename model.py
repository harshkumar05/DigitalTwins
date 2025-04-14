import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Load Training Data
train_data = pd.read_csv("TrainingData.csv")  # Ensure the correct path

# df=pd.DataFrame(train_data)
# print(df)
print(train_data.columns.tolist())



# Drop any duplicate rows or irrelevant columns
train_data = train_data.drop_duplicates().reset_index(drop=True)

# Identify Feature Columns and Target Column
feature_columns = train_data.columns.difference(["Optimized_Performance"])  # All except target
target_column = "Optimized_Performance"

# Extract Features (X) and Target (y)
X = train_data[feature_columns]
y = train_data[target_column]

# Standardize Feature Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.iloc[:, 5:])  # Ignore non-numeric VIN, Make, Model, etc.

# Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the Neural Network Model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.2),  # Prevents Overfitting
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='linear')  # Output layer for regression
])

# Compile the Model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Train the Model
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_val, y_val), verbose=1)

# Save the trained model and scaler
model.save("car_performance_model.h5")
np.save("scaler.npy", scaler.scale_)

print("Model training completed and saved.")
