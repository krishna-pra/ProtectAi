# backend/fingerprint.py

import hashlib
from PIL import Image
import numpy as np
from typing import Union


def generate_fingerprint(image_path: str, raise_error: bool = False) -> Union[str, None]:
    """
    Generate a unique fingerprint (SHA256 hash) of an image based on its pixel values.

    Args:
        image_path (str): Path to the image file.
        raise_error (bool): If True, raise exceptions instead of returning an error string.

    Returns:
        str: SHA256 hex digest of the image pixels, or error message if failed.
    """
    try:
        # Open image, convert to grayscale, normalize size
        img = Image.open(image_path).convert("L")
        img = img.resize((128, 128))
        
        # Convert pixels to bytes
        pixels = np.array(img).flatten()
        pixel_bytes = pixels.tobytes()
        
        # Generate SHA256 hash
        fingerprint = hashlib.sha256(pixel_bytes).hexdigest()
        return fingerprint

    except Exception as e:
        if raise_error:
            raise
        return f"Error generating fingerprint: {str(e)}"


# Optional test code
if __name__ == "__main__":
    test_image = "uploads/test.jpg"
    print(generate_fingerprint(test_image))
