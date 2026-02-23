import cv2
import torch
import torch.nn.functional as F
import numpy as np
from Model_def import MiniFASNetV2


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
model_path = "/mnt/d/proj-face/api/model.pth"  # replace with your path
state_dict = torch.load(model_path, map_location=device)

# Remove 'module.' prefix if checkpoint was trained with DataParallel
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
print("Model loaded successfully!")

# -------------------------
# Video Frame Processing
# -------------------------
def extract_frames(video_path, num_frames=100):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // num_frames, 1)
    frames = []
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (80, 80))  # input size expected by MiniFASNet
        frames.append(frame)
        if len(frames) >= num_frames:
            break
    cap.release()
    return np.array(frames)

# -------------------------
# Liveness Prediction
# -------------------------
def predict_liveness(video_path):
    frames = extract_frames(video_path)
    frames = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    frames = frames.to(device)
    
    with torch.no_grad():
        outputs = model(frames)
        probs = F.softmax(outputs, dim=1)[:, 1]  # assuming class 1 = live
        score = probs.mean().item()
    return score

# -------------------------
# Example Usage
# -------------------------
if __name__ == "__main__":
    video_file = "/mnt/d/proj-face/api/video.mp4"  # replace with your video
    score = predict_liveness(video_file)
    print(f"Liveness score: {score:.4f}")
    if score > 0.5:
        print("Video is LIVE")
    else:
        print("Video is SPOOF")
