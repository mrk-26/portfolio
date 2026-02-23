import os
import joblib
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import uvicorn

# --- Model architecture ---
class HARNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LeakyReLU(), nn.BatchNorm1d(512), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.LeakyReLU(), nn.BatchNorm1d(256), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.LeakyReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.LeakyReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# --- Load artifacts ---
ARTIFACT_DIR = 'artifacts'
SCALER_PATH = os.path.join(ARTIFACT_DIR, 'scaler.joblib')
ENCODER_PATH = os.path.join(ARTIFACT_DIR, 'label_encoder.joblib')
MODEL_PATH = os.path.join(ARTIFACT_DIR, 'best_model.pth')

# Ensure artifacts exist
for path in (SCALER_PATH, ENCODER_PATH, MODEL_PATH):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required artifact not found: {path}")

scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# Initialize model and load weights
def load_model():
    input_dim = scaler.scale_.shape[0]
    num_classes = len(label_encoder.classes_)
    model = HARNet(input_dim, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

model = load_model()

# FastAPI App
app = FastAPI(title='HAR 32-Feature Classifier')

class Features(BaseModel):
    features: list[float]

@app.post('/predict')
def predict(request: Features):
    x = np.array(request.features, dtype=np.float32)
    expected = scaler.scale_.shape[0]
    if x.size != expected:
        raise HTTPException(status_code=400, detail=f'Expected {expected} features, got {x.size}')
    x_scaled = scaler.transform([x])
    with torch.no_grad():
        logits = model(torch.from_numpy(x_scaled))
        idx = int(torch.argmax(logits, dim=1).item())
    activity = label_encoder.inverse_transform([idx])[0]
    return {'activity': activity, 'index': idx}

if __name__ == '__main__':
    # Run Uvicorn server
    try:
        # Use 0.0.0.0 to accept all interfaces and avoid permission errors
        uvicorn.run(app, host='0.0.0.0', port=int(os.getenv('PORT', 8000)), reload=False)
    except PermissionError as e:
        print(f"PermissionError: {e}. Try running with a higher port or elevated permissions.")