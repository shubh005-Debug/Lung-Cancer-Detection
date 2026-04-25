import numpy as np
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

TARGET_SIZE = (224, 224)


def preprocess_image(file) -> np.ndarray:
    """
    Full preprocessing pipeline for CT scan images.

    Steps:
    1. Read image bytes
    2. Validate & open image
    3. Convert to RGB
    4. Resize to 224x224
    5. Normalize pixel values to [0, 1]
    6. Expand batch dimension → shape (1, 224, 224, 3)
    """
    # Step 1: Read bytes
    image_bytes = file.read()
    if not image_bytes:
        raise ValueError("Uploaded file is empty.")

    # Step 2: Open image
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise ValueError("Cannot open image. Please upload a valid image file (PNG, JPG, TIFF, BMP).")

    # Step 3: Convert to RGB (handles grayscale, RGBA, CMYK, etc.)
    if image.mode != "RGB":
        image = image.convert("RGB")
        logger.info(f"Converted image mode to RGB.")

    # Step 4: Resize
    image = image.resize(TARGET_SIZE, Image.LANCZOS)
    logger.info(f"Resized image to {TARGET_SIZE}.")

    # Step 5: Normalize
    img_array = np.array(image, dtype=np.float32) / 255.0

    # Step 6: Expand batch dimension
    img_array = np.expand_dims(img_array, axis=0)  # → (1, 224, 224, 3)

    assert img_array.shape == (1, 224, 224, 3), f"Unexpected shape: {img_array.shape}"
    logger.info(f"Preprocessed image shape: {img_array.shape}")

    return img_array