import pandas as pd
import joblib
import numpy as np

# Load model, scaler, and encoders
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")
label_encoders = joblib.load("encoders.joblib")

# Load test data
test_data = pd.read_csv("TestData.csv")

# Save VINs for logging
vins = test_data['VIN']

# Drop target column if exists and prepare input
if 'Optimized_Performance' in test_data.columns:
    test_data = test_data.drop(columns=['Optimized_Performance'])

# Encode categorical columns
categorical_columns = ['Make', 'Model', 'Version', 'Grade']
for col in categorical_columns:
    le = label_encoders[col]
    test_data[col] = le.transform(test_data[col])

# Drop VIN before feeding to model
X_test = test_data.drop(columns=['VIN'])

# Scale features
X_test_scaled = scaler.transform(X_test)

# Predict optimized performance
predictions = model.predict(X_test_scaled)

# Attach predictions to VINs and log performance drops
for vin, pred in zip(vins, predictions):
    print(f"🔧 VIN: {vin} — Predicted Performance: {pred:.2f}%")
    if pred < 60:
        print(f"⚠️ WARNING: Performance dropped below 60% for VIN: {vin} ({pred:.2f}%)")

# Optional: Save results to CSV
output = pd.DataFrame({
    'VIN': vins,
    'Predicted_Optimized_Performance': predictions
})
output.to_csv("predicted_results.csv", index=False)

print("\n✅ Testing complete. Results saved to 'predicted_results.csv'")
