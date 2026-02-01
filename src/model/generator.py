import os
import numpy as np
from tensorflow.keras.utils import Sequence
from src.utils.config import IMG_SIZE, BATCH_SIZE, TUMOUR_LABEL, NON_TUMOUR_LABEL


DATA_DIR = "data/processed_clahe"


class KidneyDataGenerator(Sequence):
    def __init__(self, file_paths, labels, batch_size=BATCH_SIZE, shuffle=True):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.file_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[
            index * self.batch_size:(index + 1) * self.batch_size
        ]

        X, y = [], []

        for i in batch_indices:
            image = np.load(self.file_paths[i])

            # Normalize
            image = image.astype("float32") / 255.0

            # Convert to 3 channels
            image = np.stack([image, image, image], axis=-1)

            X.append(image)
            y.append(self.labels[i])

        return np.array(X), np.array(y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
