from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# Define request and response schemas
class PredictionRequest(BaseModel):
    features: list  # list of floats length num_features

class PredictionResponse(BaseModel):
    predicted_label: str
    probabilities: dict

# Initialize FastAPI app
app = FastAPI(
    title="HAR Prediction API",
    description="API for Human Activity Recognition model predictions",
    version="1.0.0"
)

# Load artifacts on startup using lifespan events
@app.on_event("startup")
async def load_artifacts():
    global scaler, label_encoder, model, activity_labels
    scaler = joblib.load("artifacts/scaler.joblib")
    label_encoder = joblib.load("artifacts/label_encoder.joblib")
    model = load_model("artifacts/best_model.h5")
    activity_labels = pd.read_csv(
        "artifacts/activity_labels.txt",
        sep=" ", header=None, names=["code", "label"]
    )

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    num_features = model.input_shape[1]
    if len(req.features) != num_features:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {num_features} features, got {len(req.features)}"
        )
    X = np.array(req.features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    probs = model.predict(X_scaled)[0]
    class_idx = int(np.argmax(probs))
    code = class_idx + 1
    label = activity_labels.loc[activity_labels.code == code, 'label'].values[0]
    prob_dict = {activity_labels.loc[i, 'label']: float(probs[i]) for i in range(len(probs))}
    return PredictionResponse(predicted_label=label, probabilities=prob_dict)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Entry point for local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)