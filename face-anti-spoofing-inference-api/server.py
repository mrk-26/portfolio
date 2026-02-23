from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from Model_def import MiniFASNetV2
import base64
import io
from PIL import Image

# Initialize FastAPI app
app = FastAPI(title="Face Liveness Detection API")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Load Model
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model 
model = MiniFASNetV2(
    embedding_size=128,
    conv6_kernel=(5, 5),
    drop_p=0.2,
    num_classes=3,
    img_channel=3
)
model.to(device)
model.eval()

# Load checkpoint
model_path = "/mnt/d/proj-face/api/model.pth"  # Update with your path
state_dict = torch.load(model_path, map_location=device)

# Remove 'module.' prefix if checkpoint was trained with DataParallel
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
print("Model loaded successfully!")

# Face detection model (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(image):
    """Preprocess image for model input"""
    # Convert to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    image = cv2.resize(image, (80, 80))
    
    # Convert to tensor and normalize
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    return image_tensor.to(device)

@app.post("/predict")
async def predict_liveness(file: UploadFile = File(...)):
    """Endpoint to predict liveness from an image"""
    # Read image file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Detect faces
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    results = []
    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = image[y:y+h, x:x+w]
        
        # Preprocess and predict
        input_tensor = preprocess_image(face_roi)
        
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            live_prob = probs[0, 1].item()  # Assuming class 1 is "live"
        
        # Determine result
        is_live = live_prob > 0.5
        label = "REAL" if is_live else "FAKE"
        color = (0, 255, 0) if is_live else (0, 0, 255)
        
        # Add to results
        results.append({
            "bbox": [int(x), int(y), int(w), int(h)],
            "label": label,
            "confidence": live_prob,
            "color": color
        })
    
    return {"results": results}

@app.get("/")
async def root():
    return {"message": "Face Liveness Detection API is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)