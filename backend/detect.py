# backend/detect.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import subprocess

MODEL_PATH = "deepfake_model.pth"

# -------------------------------
# 1. Auto-train if model missing
# -------------------------------
if not os.path.exists(MODEL_PATH):
    try:
        print("⚠️ deepfake_model.pth not found. Training model automatically...")
        subprocess.run(["python", "train.py"], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Training failed: {str(e)}")

# -------------------------------
# 2. Load Model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # 0=real, 1=fake

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model = model.to(device)

# -------------------------------
# 3. Image Preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -------------------------------
# 4. Inference Function
# -------------------------------
def detect_deepfake(image_path: str) -> dict:
    """
    Predict whether an image is Real or Fake (deepfake).
    Returns {"prediction": "Real/Fake", "confidence": 95.34}
    """
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        label = "Real" if pred == 0 else "Fake"
        confidence = round(probs[0][pred].item() * 100, 2)  # percentage

        return {"prediction": label, "confidence": confidence}

    except Exception as e:
        return {"error": str(e)}