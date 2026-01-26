import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from src.model.build_model import build_model
from src.model.dataloader import get_train_val_data
from src.utils.config import EPOCHS, BATCH_SIZE


def train():
    # Load data
    X_train, X_val, y_train, y_val = get_train_val_data()

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights = dict(enumerate(class_weights))

    print("Class weights:", class_weights)

    # Build model
    model = build_model()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="best_model.h5",
            monitor="val_loss",
            save_best_only=True
        )
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks
    )

    return model, history


if __name__ == "__main__":
    train()
