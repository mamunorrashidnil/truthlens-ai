import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input


def predict(model_path: str, img_path: str) -> tuple[str, float]:
    """
    Run deepfake detection on a single image.

    Args:
        model_path: Path to the saved .keras model file.
        img_path:   Path to the image to analyze.

    Returns:
        A tuple of (label, confidence) where label is "Real" or "Fake"
        and confidence is a float in [0, 1].
    """

    model = tf.keras.models.load_model(model_path)

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

    print(f"Prediction: {label} ({confidence:.2%})")
    return label, confidence


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TruthLens AI — single-image prediction")
    parser.add_argument("--model", required=True, help="Path to .keras model file")
    parser.add_argument("--image", required=True, help="Path to image file")
    args = parser.parse_args()

    label, conf = predict(args.model, args.image)
    print(f"Result: {label}  |  Confidence: {conf:.2%}")
