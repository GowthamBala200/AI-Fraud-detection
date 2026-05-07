import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv("Fraud detection dataset.csv")

# Remove unnecessary columns
df = df.drop(["nameOrig", "nameDest"], axis=1)

# Encode transaction type
encoder = LabelEncoder()
df["type"] = encoder.fit_transform(df["type"])

# Features and target
X = df.drop("isFraud", axis=1)
y = df["isFraud"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=50,
    random_state=42
)

# Train model
print("Training Model...")
model.fit(X_train, y_train)



# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save trained model
joblib.dump(model, "model.pkl")

print("\nModel Saved Successfully as model.pkl")