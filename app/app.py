import os
import io
import base64
import uuid
import threading

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request, send_from_directory
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# ---------------- CONFIG ----------------------------------------- #

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "..", "models", "deepfake_xception_model.keras")
DATASET_BASE = os.path.join(BASE_DIR, "..", "dataset")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "gif", "bmp"}

app.config["FEEDBACK_ENABLED"] = True

# ---------------- LOAD MODEL ------------------------------------- #

print("Loading TruthLens AI model...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at: {MODEL_PATH}\n"
        "Please run  python src/train.py  first."
    )
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
model_lock = threading.Lock()
print("Model loaded successfully!")

# ---------------- HELPERS ---------------------------------------- #

def allowed_ext(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_from_bytes(img_bytes: bytes) -> tuple:
    img = image.load_img(io.BytesIO(img_bytes), target_size=(299, 299))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    with model_lock:
        prediction = model.predict(arr, verbose=0)[0][0]
    if prediction > 0.5:
        return "Real", round(float(prediction) * 100, 2)
    return "Fake", round(float(1 - prediction) * 100, 2)


def background_train():
    try:
        import sys
        sys.path.insert(0, os.path.join(BASE_DIR, "..", "src"))
        from preprocess import get_generators
        from train import build_model
        real_n = len(os.listdir(os.path.join(DATASET_BASE, "real"))) if os.path.exists(os.path.join(DATASET_BASE, "real")) else 0
        fake_n = len(os.listdir(os.path.join(DATASET_BASE, "fake"))) if os.path.exists(os.path.join(DATASET_BASE, "fake")) else 0
        if real_n < 10 or fake_n < 10:
            return
        print(f"[BG Train] real={real_n} fake={fake_n}")
        tg, vg = get_generators(DATASET_BASE)
        m = build_model()
        m.fit(tg, validation_data=vg, epochs=3, verbose=0)
        tmp = MODEL_PATH + ".tmp.keras"
        m.save(tmp)
        global model
        with model_lock:
            model = tf.keras.models.load_model(tmp, compile=False)
            os.replace(tmp, MODEL_PATH)
        print("[BG Train] Done.")
    except Exception as e:
        print(f"[BG Train] Error: {e}")

# ---------------- ROUTES ----------------------------------------- #

@app.route("/")
def index():
    return render_template("index.html", feedback_enabled=app.config["FEEDBACK_ENABLED"])


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, "static"),
        "favicon.png", mimetype="image/png"
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("file")
    try:
        if file and file.filename != "":
            if not allowed_ext(file.filename):
                return jsonify({"error": "Unsupported file type. Use JPG, PNG, WEBP, GIF, or BMP."}), 400
            img_bytes = file.read()
        else:
            return jsonify({"error": "No image provided."}), 400

        label, conf = predict_from_bytes(img_bytes)
        return jsonify({"prediction": label, "confidence": conf})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {e}"}), 500


@app.route("/feedback", methods=["POST"])
def feedback():
    if not app.config["FEEDBACK_ENABLED"]:
        return jsonify({"status": "error"}), 403
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "error"}), 400

    predicted_label = data.get("label")
    is_correct      = data.get("is_correct")
    image_data_url  = data.get("image_data", "")

    if predicted_label not in ("Real", "Fake") or is_correct is None:
        return jsonify({"status": "error"}), 400

    true_label = predicted_label.lower() if is_correct else (
        "fake" if predicted_label == "Real" else "real"
    )
    dest_dir = os.path.join(DATASET_BASE, true_label)
    os.makedirs(dest_dir, exist_ok=True)

    try:
        if image_data_url.startswith("data:image"):
            _, b64 = image_data_url.split(",", 1)
            img_bytes = base64.b64decode(b64)
            fname = f"fb_{uuid.uuid4().hex[:8]}.jpg"
            with open(os.path.join(dest_dir, fname), "wb") as f:
                f.write(img_bytes)
    except Exception as e:
        print(f"[Feedback] save error: {e}")

    threading.Thread(target=background_train, daemon=True).start()
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
