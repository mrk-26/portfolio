import requests
import pandas as pd
import joblib

# Load and preprocess test sample locally (optional)
DATASET_PATH = "UCI HAR Dataset"
SCALER_PATH = "artifacts/scaler.joblib"

# Load raw X_test
X_test = pd.read_csv(
    f"{DATASET_PATH}/test/X_test.txt",
    delim_whitespace=True,
    header=None
).values

# Load scaler and scale sample
scaler = joblib.load(SCALER_PATH)
X_test_scaled = scaler.transform(X_test)

# Select sample
sample_features = list(X_test_scaled[0])  # first test sample

# Send request
API_URL = "http://localhost:8000/predict"
response = requests.post(API_URL, json={"features": sample_features})

if response.status_code == 200:
    data = response.json()
    print("Predicted Activity:", data["predicted_label"])
    print("Class Probabilities:")
    for label, prob in data["probabilities"].items():
        print(f"  {label}: {prob:.4f}")
else:
    print(f"Error {response.status_code}: {response.text}")
