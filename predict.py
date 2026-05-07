import joblib
import pandas as pd

# Load saved model
model = joblib.load("model.pkl")

# Sample transaction input
data = {
    "step": [1],
    "type": [4],  # TRANSFER = 4
    "amount": [181.00],
    "oldbalanceOrg": [181.00],
    "newbalanceOrig": [0.0],
    "oldbalanceDest": [0.0],
    "newbalanceDest": [0.0],
    "isFlaggedFraud": [0]
}

# Convert input into DataFrame
input_data = pd.DataFrame(data)

# Predict
prediction = model.predict(input_data)

if prediction[0] == 1:
    print("Fraud Transaction Detected")
else:
    print("Normal Transaction")