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

app = Flask(__name__,
    static_folder=os.path.join(os.path.dirname(__file__), '..', 'frontend', 'build'),
    static_url_path='/'
)
CORS(app)

# Load model once at startup
model_loader = ModelLoader()
model_loader.load()


# ── Serve React Frontend ──────────────────────
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    build_folder = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'build')
    if path != "" and os.path.exists(os.path.join(build_folder, path)):
        return send_from_directory(build_folder, path)
    return send_from_directory(build_folder, 'index.html')


# ── API Routes ────────────────────────────────
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
        "classes": ["adenocarcinoma", "large_cell_carcinoma", "normal", "squamous_cell_carcinoma"],
        "accuracy": 92.0,
        "description": "VGG16-based CNN trained on CT scan images for lung cancer classification"
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    allowed_extensions = {"png", "jpg", "jpeg", "bmp", "tiff"}
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in allowed_extensions:
        return jsonify({"error": f"Invalid file type. Allowed: png, jpg, jpeg, bmp, tiff"}), 400

    if not model_loader.is_loaded():
        return jsonify({"error": "Model not loaded."}), 503

    try:
        result = predict_image(file, model_loader.get_model())
        recommendations = get_recommendations(result["prediction"])
        result["recommendations"] = recommendations
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

        prediction_result = data.get("prediction_result", {})
        recommendations = data.get("recommendations", {})
        patient_name = data.get("patient_name", "Anonymous")

        pdf_buffer = generate_pdf_report(prediction_result, recommendations, patient_name)
        return send_file(
            pdf_buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"LungAI_Report_{patient_name.replace(' ', '_')}.pdf"
        )
    except Exception as e:
        logger.error(f"Report error: {traceback.format_exc()}")
        return jsonify({"error": "Report generation failed."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
