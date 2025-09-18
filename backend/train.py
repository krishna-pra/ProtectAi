# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import requests
import json
import time

# -------------------------------
# 1. Training Config
# -------------------------------
DATA_DIR = "dataset"  # must have subfolders: dataset/train/real, dataset/train/fake
MODEL_SAVE_PATH = "deepfake_model.pth"
BATCH_SIZE = 8
EPOCHS = 3  # keep low for testing
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------
# 2. Helper: Download small dataset if missing
# -------------------------------
def setup_sample_dataset():
    print("⚠️ No dataset found. Downloading a small sample dataset...")

    folders = [
        "dataset/train/real",
        "dataset/train/fake",
        "dataset/val/real",
        "dataset/val/fake",
    ]
    for f in folders:
        os.makedirs(f, exist_ok=True)

    real_images = [
        ("https://upload.wikimedia.org/wikipedia/commons/3/37/Albert_Einstein_Head.jpg", "einstein.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/9/99/Barack_Obama.jpg", "obama.jpg"),
    ]

    fake_images = [
        ("https://thispersondoesnotexist.com/image", "fake1.jpg"),
        ("https://thispersondoesnotexist.com/image", "fake2.jpg"),
    ]

    def download(url, path):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(path, "wb") as f:
                    f.write(r.content)
                print(f"✅ Downloaded {path}")
            else:
                print(f"⚠️ Failed {url}")
        except Exception as e:
            print(f"Error: {e}")

    # Download reals
    for url, name in real_images:
        download(url, f"dataset/train/real/{name}")
        download(url, f"dataset/val/real/val_{name}")

    # Download fakes
    for url, name in fake_images:
        download(url, f"dataset/train/fake/{name}")
        download(url, f"dataset/val/fake/val_{name}")


# Check dataset
if not os.path.exists(os.path.join(DATA_DIR, "train")):
    setup_sample_dataset()


# -------------------------------
# 3. Data Loading & Augmentation
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# -------------------------------
# 4. Model Definition
# -------------------------------
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # 0=real, 1=fake
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# -------------------------------
# 5. Training & Validation
# -------------------------------
def validate() -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"🔎 Validation Accuracy: {acc:.2f}%")
    return acc


def train() -> float:
    final_train_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        final_train_acc = 100 * correct / total
        print(f"📚 Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}, Train Acc: {final_train_acc:.2f}%")
        validate()

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Model saved as {MODEL_SAVE_PATH}")
    return final_train_acc


if __name__ == "__main__":
    final_train_acc = train()

    # Save metadata
    metadata = {
        "last_trained": time.strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": EPOCHS,
        "final_train_acc": round(final_train_acc, 2),
    }
    with open("model_metadata.json", "w") as f:
        json.dump(metadata, f)
    print("📝 Training metadata saved as model_metadata.json")