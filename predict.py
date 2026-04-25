import numpy as np
import logging
from preprocessing import preprocess_image
from gradcam import generate_gradcam

logger = logging.getLogger(__name__)

# Alphabetical order — matches Colab ImageDataGenerator
CLASS_NAMES = [
    "adenocarcinoma",
    "large_cell_carcinoma",
    "normal",
    "squamous_cell_carcinoma"
]

RISK_MAP = {
    "adenocarcinoma": "High",
    "large_cell_carcinoma": "High",
    "squamous_cell_carcinoma": "High",
    "normal": "Low",
}

RISK_COLORS = {
    "High": "#dc2626",
    "Medium": "#ea580c",
    "Low": "#16a34a",
}


def predict_image(file, model) -> dict:
    if model is None:
        raise RuntimeError("Model not loaded.")

    img_array = preprocess_image(file)

    logger.info(f"Running inference, input shape: {img_array.shape}")
    raw_preds = model.predict(img_array, verbose=0)
    logger.info(f"Raw model output: {raw_preds}")

    probabilities = raw_preds[0].tolist()
    prob_sum = sum(probabilities)
    if prob_sum > 0:
        probabilities = [p / prob_sum for p in probabilities]

    pred_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[pred_index]
    confidence = round(probabilities[pred_index] * 100, 2)
    risk_level = RISK_MAP.get(predicted_class, "Unknown")

    prob_dict = {
        cls: round(prob, 4)
        for cls, prob in zip(CLASS_NAMES, probabilities)
    }

    # Generate Grad-CAM heatmap
    gradcam_image = generate_gradcam(model, img_array, pred_index)

    logger.info(f"✅ Prediction: {predicted_class} ({confidence}%) | Risk: {risk_level}")

    return {
        "prediction": predicted_class,
        "confidence": confidence,
        "probabilities": prob_dict,
        "risk_level": risk_level,
        "risk_color": RISK_COLORS.get(risk_level, "#6b7280"),
        "gradcam": gradcam_image,  # base64 encoded PNG
    }