# backend/deepfake_detector.py

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

# -------------------------
# Config
# -------------------------
MODEL_PATH = "deepfake_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load ResNet18 Model
# -------------------------
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # 0 = Real, 1 = Fake
model = model.to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"✅ Loaded model from {MODEL_PATH}")
else:
    print(f"⚠️ Model file not found at {MODEL_PATH}. Run train.py first.")

# -------------------------
# Preprocessing
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------
# Detection Function
# -------------------------
def detect_deepfake(image_path: str) -> dict:
    """
    Detect whether an image is Real or Fake using ResNet18.
    Returns prediction + confidence.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_class = torch.max(probs, dim=0)

        label = "FAKE" if predicted_class.item() == 1 else "REAL"

        return {
            "file": os.path.basename(image_path),
            "prediction": label,
            "confidence": float(confidence.item())
        }

    except Exception as e:
        return {"error": str(e)}