import requests
import numpy as np

# Generate dummy 10-second accelerometer data at 30Hz (300 samples)
dummy_data = np.random.randn(300, 3).tolist()

payload = {"data": dummy_data}
response = requests.post("http://localhost:8000/predict", json=payload)

if response.status_code == 200:
    print("Predicted activity:", response.json().get("activity"))
else:
    print("Error:", response.text)
