
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import logging
import traceback
import os
from dotenv import load_dotenv

from model_loader import ModelLoader
from predict import predict_image
from ai_advisor import get_recommendations
from report_generator import generate_pdf_report

load_dotenv()

FRONTEND_URL = os.getenv("FRONTEND_URL")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

CORS(app, origins=[FRONTEND_URL])

# Load model once at startup
model_loader = ModelLoader()
model_loader.load()


# Root route
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "LungAI Backend API is running",
        "health_endpoint": "/health"
    })


# Health route
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model_loader.is_loaded(),
        "real_model_loaded": model_loader.is_real_model(),
        "message": "LungAI API is running"
    })


# Model info
@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model_type": "CNN (VGG16 Transfer Learning)",
        "framework": "TensorFlow / Keras",
        "input_size": "224x224x3",
        "num_classes": 4,
        "classes": [
            "adenocarcinoma",
            "large_cell_carcinoma",
            "normal",
            "squamous_cell_carcinoma"
        ],
        "accuracy": 92.0,
        "description": (
            "VGG16-based CNN trained on CT scan images "
            "for lung cancer classification"
        )
    })


# Prediction
@app.route("/predict", methods=["POST"])
def predict():

    if not model_loader.is_loaded():
        return jsonify({"error": "Model not loaded."}), 503

    # Prevent predictions using mock model
    if not model_loader.is_real_model():
        return jsonify({
            "error": "Trained model is not available. Prediction disabled."
        }), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    allowed_extensions = {"png", "jpg", "jpeg", "bmp", "tiff"}

    ext = (
        file.filename.rsplit(".", 1)[-1].lower()
        if "." in file.filename
        else ""
    )

    if ext not in allowed_extensions:
        return jsonify({
            "error": "Invalid file type. Allowed: png, jpg, jpeg, bmp, tiff"
        }), 400

    try:
        result = predict_image(file, model_loader.get_model())

        recommendations = get_recommendations(
            result["prediction"]
        )

        result["recommendations"] = recommendations

        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    except Exception:
        logger.error(f"Prediction error: {traceback.format_exc()}")

        return jsonify({
            "error": "Prediction failed. Please try again."
        }), 500


# PDF report
@app.route("/generate-report", methods=["POST"])
def generate_report():

    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        prediction_result = data.get("prediction_result", {})
        recommendations = data.get("recommendations", {})
        patient_name = data.get("patient_name", "Anonymous")

        pdf_buffer = generate_pdf_report(
            prediction_result,
            recommendations,
            patient_name
        )

        return send_file(
            pdf_buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=(
                f"LungAI_Report_{patient_name.replace(' ', '_')}.pdf"
            )
        )

    except Exception:
        logger.error(f"Report error: {traceback.format_exc()}")

        return jsonify({
            "error": "Report generation failed."
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        debug=False,
        host="0.0.0.0",
        port=port
    )
