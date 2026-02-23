# server.py
# Clean single-model API for anti-spoofing
import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from flask import Flask, request, jsonify

from anti_spoof_predict import Detection, AntiSpoofPredict
import transform as trans


# --------------------------
# Config
# --------------------------
MODEL_PATH = "./resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"  # <--- update with your actual model
DEVICE_ID = 0  # GPU id, or will fallback to CPU
INPUT_SIZE = (80, 80)  # must match your model
app = Flask(__name__)

# --------------------------
# Load Model
# --------------------------
predictor = AntiSpoofPredict(device_id=DEVICE_ID)
predictor._load_model(MODEL_PATH)
predictor.model.eval()

# face detector
detector = Detection()

# preprocessing (from transform.py)
to_tensor = trans.Compose([trans.ToTensor()])


def preprocess(img, bbox):
    """Crop, resize, tensorize"""
    x, y, w, h = bbox
    face = img[y:y + h, x:x + w]
    face = cv2.resize(face, INPUT_SIZE)
    tensor = to_tensor(face)
    tensor = tensor.unsqueeze(0).to(predictor.device)
    return tensor


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    # detect face
    try:
        bbox = detector.get_bbox(img)
    except Exception as e:
        return jsonify({"error": f"Face detection failed: {str(e)}"}), 500

    # preprocess
    tensor = preprocess(img, bbox)

    # predict
    with torch.no_grad():
        result = predictor.model(tensor)
        result = F.softmax(result, dim=1).cpu().numpy().flatten()

    label = int(np.argmax(result))
    confidence = float(result[label])

    return jsonify({
        "label": label,         # 0 = spoof, 1 = live
        "confidence": confidence,
        "raw_scores": result.tolist()
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
