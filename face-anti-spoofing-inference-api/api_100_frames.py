import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -------------------------
# MiniFASNetSE Model
# -------------------------
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, stride=1, padding=1, use_se=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)
        self.se = SELayer(out_c) if use_se else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        x = self.se(x)
        return x

class MiniFASNetSE(nn.Module):
    def __init__(self, num_classes=2):
        super(MiniFASNetSE, self).__init__()
        self.layer1 = ConvBlock(3, 8)
        self.layer2 = ConvBlock(8, 16)
        self.layer3 = ConvBlock(16, 32, use_se=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x

# -------------------------
# Load Model
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MiniFASNetSE(num_classes=2).to(device)
model.eval()

model_path = "/mnt/d/facel/Silent-Face-Anti-Spoofing/api/model.pth"  # replace with your actual model path
state_dict = torch.load(model_path, map_location=device)

# Fix DataParallel keys if needed
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
        frame = cv2.resize(frame, (80, 80))  # adjust input size if needed
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
    video_file = "/path/to/test/video.mp4"  # replace with your video
    score = predict_liveness(video_file)
    print(f"Liveness score: {score:.4f}")
    if score > 0.5:
        print("Video is LIVE")
    else:
        print("Video is SPOOF")
