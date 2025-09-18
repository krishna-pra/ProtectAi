# setup_sample_dataset.py
import os
import requests
from PIL import Image
from io import BytesIO
import time

# Folders to create
FOLDERS = [
    "dataset/train/real",
    "dataset/train/fake",
    "dataset/val/real",
    "dataset/val/fake",
]

for folder in FOLDERS:
    os.makedirs(folder, exist_ok=True)

# Sample images (replace or extend these with your own sources)
REAL_IMAGES = [
    "https://upload.wikimedia.org/wikipedia/commons/3/37/Albert_Einstein_Head.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/9/99/Barack_Obama.jpg",
]

FAKE_IMAGES = [
    "https://thispersondoesnotexist.com/image",
    "https://thispersondoesnotexist.com/image",
]

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ProtectAI/1.0)"}


def download_image(url: str, dest_path: str, timeout: int = 15, retries: int = 1) -> bool:
    """
    Download an image and validate it with Pillow before saving.
    Returns True on success, False otherwise.
    """
    attempt = 0
    while attempt <= retries:
        try:
            resp = requests.get(url, timeout=timeout, headers=HEADERS)
            resp.raise_for_status()

            # Quick content-size check
            if len(resp.content) < 200:
                print(f"⚠️ {url} returned very small content ({len(resp.content)} bytes). Skipping.")
                return False

            # Validate image bytes with Pillow
            try:
                bio = BytesIO(resp.content)
                img = Image.open(bio)
                img.verify()  # raises if not a valid image
            except Exception:
                print(f"⚠️ Downloaded content from {url} is not a valid image.")
                return False

            # Save to disk
            with open(dest_path, "wb") as f:
                f.write(resp.content)

            print(f"✅ Downloaded {dest_path}")
            return True

        except requests.RequestException as e:
            print(f"Error downloading {url}: {e}. Attempt {attempt+1}/{retries+1}")
            attempt += 1
            time.sleep(1)  # small backoff

    print(f"❌ Failed to download {url} after {retries+1} attempts.")
    return False


# Download real images
for i, url in enumerate(REAL_IMAGES):
    train_path = f"dataset/train/real/real_{i}.jpg"
    val_path = f"dataset/val/real/real_val_{i}.jpg"
    download_image(url, train_path)
    download_image(url, val_path)

# Download fake images
for i, url in enumerate(FAKE_IMAGES):
    train_path = f"dataset/train/fake/fake_{i}.jpg"
    val_path = f"dataset/val/fake/fake_val_{i}.jpg"
    download_image(url, train_path)
    download_image(url, val_path)

print("\nDataset setup complete. Check the `dataset/` folder.")