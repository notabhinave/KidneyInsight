import os
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils.config import IMG_SIZE, TUMOUR_LABEL, NON_TUMOUR_LABEL


DATA_DIR = "E:/kidneyinsight/data/processed_clahe"


def load_images(folder, label):
    images = []
    labels = []

    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)

        image = np.load(file_path)

        # Normalize to [0,1]
        image = image.astype("float32") / 255.0

        # Convert grayscale → 3 channels
        image = np.stack([image, image, image], axis=-1)

        images.append(image)
        labels.append(label)

    return images, labels


def load_dataset():
    images = []
    labels = []

    # Tumour images
    tumour_imgs, tumour_labels = load_images(
        os.path.join(DATA_DIR, "tumour"),
        TUMOUR_LABEL
    )

    # Non-tumour images
    non_tumour_imgs, non_tumour_labels = load_images(
        os.path.join(DATA_DIR, "non_tumour"),
        NON_TUMOUR_LABEL
    )

    images.extend(tumour_imgs)
    labels.extend(tumour_labels)

    images.extend(non_tumour_imgs)
    labels.extend(non_tumour_labels)

    X = np.array(images)
    y = np.array(labels)

    return X, y


def get_train_val_data(test_size=0.2):
    X, y = load_dataset()

    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=42
    )
