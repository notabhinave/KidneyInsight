import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from src.model.dataloader import get_train_val_data


def evaluate():
    # Load validation data
    _, X_val, _, y_val = get_train_val_data()

    # Load trained model
    model = tf.keras.models.load_model("best_model.h5")

    # Predict
    y_pred_prob = model.predict(X_val)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    print("Confusion Matrix:\n", cm)

    # Classification report
    print("\nClassification Report:\n")
    print(classification_report(y_val, y_pred, target_names=["Non-Tumour", "Tumour"]))


if __name__ == "__main__":
    evaluate()
