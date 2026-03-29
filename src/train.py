import os
import sys

# Ensure src/ is on the path so preprocess can be imported from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers, models
from preprocess import get_generators


def build_model():
    """Build and compile the Xception-based deepfake detection model."""

    base_model = Xception(
        weights="imagenet",
        include_top=False,
        input_shape=(299, 299, 3)
    )

    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":

    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset")

    train_gen, val_gen = get_generators(dataset_path)

    model = build_model()

    print("Training TruthLens AI...")

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10
    )

    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    os.makedirs(model_dir, exist_ok=True)

    # Save in the modern .keras format (replaces deprecated .h5)
    save_path = os.path.join(model_dir, "deepfake_xception_model.keras")
    model.save(save_path)

    print(f"Model saved to: {save_path}")
