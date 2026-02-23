from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np

# Initialize FastAPI app
title = "HARNet10 Activity Recognition API"
app = FastAPI(title=title)

# Configure CORS (allow your React frontend origin)
origins = [
    "http://localhost:3000",  # React dev server
    # Add other origins as needed
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pretrained HARNet10 model from Torch Hub with trust
# Adjust class_num to the number of activity classes in your use case
REPO = 'OxWearables/ssl-wearables'
model = torch.hub.load(
    REPO,            # GitHub repo path
    'harnet10',      # Model name
    class_num=5,     # Number of output classes
    pretrained=True, # Load pretrained weights
    trust_repo=True  # Trust and execute code without prompt
)
model.eval()

# Define request schema
class SensorData(BaseModel):
    # Expect a 2D list: [ [x, y, z], [x, y, z], ... ] of length 300
    data: list[list[float]]

# Prediction endpoint
@app.post("/predict")
async def predict_activity(sensor: SensorData):
    # Convert list to NumPy array and then to PyTorch tensor
    input_array = np.array(sensor.data, dtype=np.float32)
    if input_array.shape != (300, 3):
        return {"error": f"Invalid input shape {input_array.shape}, expected (300, 3)"}

    # Torch Hub models expect input shape (batch, channels, time)
    # Here: channels = 3 axes, time = 300 samples
    input_tensor = torch.from_numpy(input_array).permute(1, 0).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        logits = model(input_tensor)
        pred_class = torch.argmax(logits, dim=1).item()

    return {"activity": pred_class}

# Run with: uvicorn app:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
