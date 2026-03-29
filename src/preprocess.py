import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input


def get_generators(dataset_path: str):
    """
    Build training and validation data generators from a directory structured as:
        dataset_path/
            real/   ← real images
            fake/   ← AI-generated / deepfake images

    Args:
        dataset_path: Absolute or relative path to the dataset root directory.

    Returns:
        (train_gen, val_gen): Keras DirectoryIterator objects.
    """

    dataset_path = os.path.abspath(dataset_path)

    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        horizontal_flip=True,
        rotation_range=10,
        zoom_range=0.1,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=(299, 299),
        batch_size=32,
        class_mode="binary",
        subset="training",
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=(299, 299),
        batch_size=32,
        class_mode="binary",
        subset="validation",
        shuffle=False
    )

    return train_gen, val_gen
