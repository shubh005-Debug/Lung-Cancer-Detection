from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import logging
import traceback
import os

from model_loader import ModelLoader
from predict import predict_image
from ai_advisor import get_recommendations
from report_generator import generate_pdf_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Build folder path ─────────────────────────────
# Works both locally and on Render
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR  = os.path.join(BASE_DIR, '..', 'frontend', 'build')
BUILD_DIR  = os.path.realpath(BUILD_DIR)

app = Flask(
    __name__,
    static_folder=BUILD_DIR if os.path.exists(BUILD_DIR) else None,
    static_url_path='/'
)

# ── CORS ──────────────────────────────────────────
CORS(app, origins=[
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://*.vercel.app",
    "https://*.onrender.com",
    "*"          # allow all during debugging — tighten later
])

# ── Load model once ───────────────────────────────
model_loader = ModelLoader()
model_loader.load()


# ══════════════════════════════════════════════════
# API ROUTES  (must come BEFORE the React catch-all)
# ══════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model_loader.is_loaded(),
        "message": "LungAI API is running"
    })


@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model_type": "CNN (VGG16 Transfer Learning)",
        "framework": "TensorFlow / Keras",
        "input_size": "224x224x3",
        "num_classes": 4,
        "classes": ["adenocarcinoma", "large_cell_carcinoma",
                    "normal", "squamous_cell_carcinoma"],
        "accuracy": 92.0,
        "description": "VGG16 CNN trained on CT scan images for lung cancer classification"
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    allowed = {"png", "jpg", "jpeg", "bmp", "tiff"}
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in allowed:
        return jsonify({"error": "Invalid file type. Allowed: png, jpg, jpeg, bmp, tiff"}), 400

    if not model_loader.is_loaded():
        return jsonify({"error": "Model not loaded."}), 503

    try:
        result = predict_image(file, model_loader.get_model())
        result["recommendations"] = get_recommendations(result["prediction"])
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {traceback.format_exc()}")
        return jsonify({"error": "Prediction failed. Please try again."}), 500


@app.route("/generate-report", methods=["POST"])
def generate_report():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        pdf_buffer = generate_pdf_report(
            data.get("prediction_result", {}),
            data.get("recommendations", {}),
            data.get("patient_name", "Anonymous")
        )
        return send_file(
            pdf_buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="LungAI_Report.pdf"
        )
    except Exception as e:
        logger.error(f"Report error: {traceback.format_exc()}")
        return jsonify({"error": "Report generation failed."}), 500


# ══════════════════════════════════════════════════
# SERVE REACT  (catch-all — must be LAST)
# ══════════════════════════════════════════════════

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    # If React build exists, serve it
    if os.path.exists(BUILD_DIR):
        # Serve existing static file (JS, CSS, images…)
        target = os.path.join(BUILD_DIR, path)
        if path and os.path.exists(target):
            return send_from_directory(BUILD_DIR, path)
        # Fall back to index.html for client-side routing
        index = os.path.join(BUILD_DIR, 'index.html')
        if os.path.exists(index):
            return send_from_directory(BUILD_DIR, 'index.html')

    # No build folder → API-only mode
    return jsonify({
        "status": "ok",
        "message": "LungAI API is running. React build not found.",
        "endpoints": ["/health", "/model-info", "/predict", "/generate-report"]
    })


# ══════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"BUILD_DIR = {BUILD_DIR} (exists={os.path.exists(BUILD_DIR)})")
    app.run(debug=False, host="0.0.0.0", port=port)
