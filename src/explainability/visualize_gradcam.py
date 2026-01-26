import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.explainability.grad_cam import make_gradcam_heatmap, overlay_heatmap
from src.utils.config import IMG_SIZE


MODEL_PATH = "best_model.h5"


def load_image(image_path):
    # Load image (grayscale)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize
    image = cv2.resize(image, IMG_SIZE)

    # Normalize
    image_norm = image.astype("float32") / 255.0

    # Convert to 3 channels
    image_rgb = np.stack([image_norm]*3, axis=-1)

    return image, np.expand_dims(image_rgb, axis=0)


def visualize(image_path):
    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load image
    original, processed = load_image(image_path)

    # Predict
    prob = model.predict(processed)[0][0]

    print(f"Tumour Probability: {prob:.4f}")

    if prob < 0.5:
        print("Prediction: NON-TUMOUR → Grad-CAM skipped")
        return

    print("Prediction: TUMOUR → Applying Grad-CAM")

    # Generate Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(processed, model)

    # Overlay heatmap
    overlay = overlay_heatmap(
        cv2.cvtColor(original, cv2.COLOR_GRAY2BGR),
        heatmap
    )

    # Display
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original CT")
    plt.imshow(original, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Grad-CAM Heatmap")
    plt.imshow(heatmap, cmap="jet")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    # Example image path (change this)
    visualize("E:/kidneyinsight/data/raw/Tumour/Tumor- (22).jpg")
