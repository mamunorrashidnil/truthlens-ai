import os
import shutil

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ---------------- CONFIG ----------------------------------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "deepfake_xception_model.keras")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
DATASET_BASE = os.path.join(BASE_DIR, "..", "dataset")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "gif", "bmp"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["FEEDBACK_ENABLED"] = True

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- LOAD MODEL ------------------------------------- #

print("Loading TruthLens AI model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at: {MODEL_PATH}\n"
        "Please run  python src/train.py  to train and save the model first."
    )

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully!")

# ---------------- HELPERS ---------------------------------------- #

def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def predict_image(img_path: str) -> tuple[str, float]:
    """Return (label, confidence_pct) for the given image path."""

    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        label = "Real"
        confidence = float(prediction)
    else:
        label = "Fake"
        confidence = float(1 - prediction)

    return label, round(confidence * 100, 2)

# ---------------- ROUTES ----------------------------------------- #

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename == "":
            return render_template(
                "index.html",
                error="No file selected.",
                feedback_enabled=app.config["FEEDBACK_ENABLED"]
            )

        if not allowed_file(file.filename):
            return render_template(
                "index.html",
                error="Unsupported file type. Please upload a JPG, PNG, WEBP, or GIF.",
                feedback_enabled=app.config["FEEDBACK_ENABLED"]
            )

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        label, conf = predict_image(filepath)

        return render_template(
            "index.html",
            prediction=label,
            confidence=conf,
            filename=filename,
            feedback_enabled=app.config["FEEDBACK_ENABLED"],
        )

    return render_template(
        "index.html",
        feedback_enabled=app.config["FEEDBACK_ENABLED"]
    )

# ---------------- FEEDBACK --------------------------------------- #

@app.route("/feedback", methods=["POST"])
def feedback():

    if not app.config["FEEDBACK_ENABLED"]:
        return jsonify({"status": "error", "message": "Feedback is disabled."}), 403

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "error", "message": "Invalid JSON payload."}), 400

    filename = data.get("filename")
    predicted_label = data.get("label")
    is_correct = data.get("is_correct")

    if not filename or predicted_label not in ("Real", "Fake") or is_correct is None:
        return jsonify({"status": "error", "message": "Missing or invalid fields."}), 400

    # Sanitize filename to prevent path traversal
    filename = secure_filename(filename)
    source = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    if not os.path.isfile(source):
        return jsonify({"status": "error", "message": "Source file not found."}), 404

    true_label = predicted_label.lower() if is_correct else (
        "fake" if predicted_label == "Real" else "real"
    )

    dest_dir = os.path.join(DATASET_BASE, true_label)
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy(source, os.path.join(dest_dir, filename))

    return jsonify({"status": "ok", "message": f"Thank you! Image added to '{true_label}' dataset."})

# ---------------- RUN -------------------------------------------- #

if __name__ == "__main__":
    app.run(debug=True)
