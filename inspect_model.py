
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image, ImageOps

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "mnist_cnn.keras"
OUTPUT_DIR = "model_inspection"
CUSTOM_IMAGE_PATH = ""  # Example: "my_digit.png"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. INSPECT THE .KERAS ARCHIVE
# ============================================================

print("\n" + "=" * 60)
print("1. KERAS ARCHIVE CONTENTS")
print("=" * 60)

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Place the model in this folder or update MODEL_PATH."
    )

if zipfile.is_zipfile(MODEL_PATH):
    with zipfile.ZipFile(MODEL_PATH, "r") as archive:
        for info in archive.infolist():
            print(f"{info.filename} - {info.file_size:,} bytes")

        # Export archive contents for optional manual inspection.
        archive.extractall(os.path.join(OUTPUT_DIR, "archive_contents"))

    print("\nArchive extracted to:", OUTPUT_DIR)
else:
    print("This is not a standard ZIP-based .keras archive.")
    print("It may be a legacy or differently saved model.")


# ============================================================
# 2. LOAD THE TRAINED MODEL
# ============================================================

print("\n" + "=" * 60)
print("2. MODEL ARCHITECTURE")
print("=" * 60)

model = tf.keras.models.load_model(MODEL_PATH)

model.summary()

print("\nInput shape:", model.input_shape)
print("Output shape:", model.output_shape)
print("Number of layers:", len(model.layers))
print("Total parameters:", model.count_params())

# Save architecture to JSON.
with open(os.path.join(OUTPUT_DIR, "architecture.json"), "w") as f:
    f.write(model.to_json())

# Save readable layer summary.
with open(os.path.join(OUTPUT_DIR, "model_summary.txt"), "w") as f:
    model.summary(print_fn=lambda line: f.write(line + "\n"))

# Visualize model layers if supported.
try:
    tf.keras.utils.plot_model(
        model,
        to_file=os.path.join(OUTPUT_DIR, "architecture.png"),
        show_shapes=True,
        show_layer_names=True
    )
    print("Architecture image saved.")
except Exception as exc:
    print("Could not generate architecture image:", exc)
    print("The text summary is still available.")


# ============================================================
# 3. INSPECT LEARNED WEIGHTS
# ============================================================

print("\n" + "=" * 60)
print("3. TRAINED WEIGHTS")
print("=" * 60)

weight_report = []

for layer in model.layers:
    print(f"\nLayer: {layer.name}")

    if not layer.weights:
        print("  No trainable or non-trainable weight tensors.")
        continue

    for variable in layer.weights:
        values = variable.numpy()

        stats = {
            "layer": layer.name,
            "weight": variable.name,
            "shape": str(values.shape),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values))
        }

        weight_report.append(stats)

        print(f"  Tensor: {variable.name}")
        print(f"  Shape: {values.shape}")
        print(f"  Min: {stats['min']:.6f}")
        print(f"  Max: {stats['max']:.6f}")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std: {stats['std']:.6f}")

# Export numerical statistics.
import csv

weight_csv = os.path.join(OUTPUT_DIR, "weight_statistics.csv")

with open(weight_csv, "w", newline="", encoding="utf-8") as f:
    columns = [
        "layer", "weight", "shape",
        "min", "max", "mean", "std"
    ]
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(weight_report)

print("\nWeight statistics saved to:", weight_csv)

# Plot a histogram of the first non-empty weight tensor.
for layer in model.layers:
    if layer.weights:
        first_weights = layer.weights[0].numpy().flatten()

        plt.figure(figsize=(8, 4))
        plt.hist(first_weights, bins=50)
        plt.title(f"Weight Distribution: {layer.name}")
        plt.xlabel("Weight value")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, "weight_distribution.png")
        )
        plt.close()
        print("Weight histogram saved.")
        break


# ============================================================
# 4. LOAD THE MNIST DATASET
# ============================================================

print("\n" + "=" * 60)
print("4. MNIST TEST DATASET")
print("=" * 60)

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Test images shape:", x_test.shape)
print("Test labels shape:", y_test.shape)
print("Image data type:", x_test.dtype)
print("Pixel range before normalization:",
      x_test.min(), "to", x_test.max())

# MNIST images normally have pixel values from 0 to 255.
x_test_normalized = x_test.astype("float32") / 255.0

print("Pixel range after normalization:",
      x_test_normalized.min(), "to", x_test_normalized.max())


# ============================================================
# 5. ADAPT INPUT TO THE MODEL
# ============================================================

def prepare_images(images):
    """Prepare MNIST images according to the model input shape."""
    shape = model.input_shape

    if isinstance(shape, list):
        raise ValueError("This script expects a single-input model.")

    expected_h = shape[1]
    expected_w = shape[2]

    if expected_h is not None and images.shape[1] != expected_h:
        raise ValueError(
            f"Model expects height {expected_h}; "
            f"MNIST images have height {images.shape[1]}."
        )

    if expected_w is not None and images.shape[2] != expected_w:
        raise ValueError(
            f"Model expects width {expected_w}; "
            f"MNIST images have width {images.shape[2]}."
        )

    # Add a channel dimension for CNNs using (H, W, 1).
    if len(shape) == 4:
        if shape[-1] == 1:
            images = images[..., np.newaxis]
        elif shape[1] == 1:
            images = images[:, np.newaxis, :, :]
        elif shape[-1] is not None:
            raise ValueError(
                f"Unexpected model input shape: {shape}"
            )

    return images


x_test_model = prepare_images(x_test_normalized)


# ============================================================
# 6. PREDICT SAMPLE DIGITS
# ============================================================

print("\n" + "=" * 60)
print("5. SAMPLE PREDICTIONS")
print("=" * 60)

sample_count = 10

probabilities = model.predict(
    x_test_model[:sample_count], verbose=0
)

predictions = np.argmax(probabilities, axis=1)
confidence = np.max(probabilities, axis=1)

fig, axes = plt.subplots(2, 5, figsize=(12, 6))

for i, ax in enumerate(axes.flat):
    ax.imshow(x_test[i], cmap="gray")
    ax.set_title(
        f"Actual: {y_test[i]}\n"
        f"Predicted: {predictions[i]} "
        f"({confidence[i]:.1%})"
    )
    ax.axis("off")

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "sample_predictions.png"),
    dpi=150
)
plt.show()

for i in range(sample_count):
    print(
        f"Image {i}: Actual={y_test[i]}, "
        f"Predicted={predictions[i]}, "
        f"Confidence={confidence[i]:.2%}"
    )


# ============================================================
# 7. EVALUATE ON THE COMPLETE MNIST TEST SET
# ============================================================

print("\n" + "=" * 60)
print("6. MODEL EVALUATION")
print("=" * 60)

test_predictions = model.predict(x_test_model, verbose=0)

predicted_labels = np.argmax(test_predictions, axis=1)
test_accuracy = np.mean(predicted_labels == y_test)

print(f"Test accuracy: {test_accuracy:.4%}")
print(f"Correct predictions: {np.sum(predicted_labels == y_test)}")
print(f"Incorrect predictions: {np.sum(predicted_labels != y_test)}")

# Save incorrect examples for analysis.
wrong_indices = np.where(predicted_labels != y_test)[0]

if len(wrong_indices) > 0:
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))

    for ax, index in zip(axes.flat, wrong_indices[:10]):
        ax.imshow(x_test[index], cmap="gray")
        ax.set_title(
            f"Actual: {y_test[index]}\n"
            f"Predicted: {predicted_labels[index]}"
        )
        ax.axis("off")

    for ax in axes.flat[len(wrong_indices[:10]):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "incorrect_predictions.png"),
        dpi=150
    )
    plt.show()

# Save all prediction results.
with open(
    os.path.join(OUTPUT_DIR, "predictions.csv"),
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.writer(f)
    writer.writerow(["index", "actual", "predicted", "confidence"])

    for i in range(len(y_test)):
        writer.writerow([
            i,
            int(y_test[i]),
            int(predicted_labels[i]),
            float(np.max(test_predictions[i]))
        ])


# ============================================================
# 8. PREDICT YOUR OWN DIGIT IMAGE (OPTIONAL)
# ============================================================

def predict_custom_image(image_path):
    """
    Predict a single digit from an image.

    For best results, use a centered digit with a dark background
    and a bright digit, similar to standard MNIST images.
    """
    image = Image.open(image_path).convert("L")
    image = ImageOps.fit(image, (28, 28))

    pixels = np.asarray(image).astype("float32") / 255.0

    # Convert white-background / dark-digit images to the
    # usual MNIST style of black background / bright digit.
    if pixels.mean() > 0.5:
        pixels = 1.0 - pixels

    pixels = pixels[np.newaxis, ...]

    # Apply the expected model input layout.
    pixels = prepare_images(pixels)

    output = model.predict(pixels, verbose=0)[0]
    digit = int(np.argmax(output))

    print("\nCustom image:", image_path)
    print("Predicted digit:", digit)
    print(f"Confidence: {output[digit]:.2%}")

    plt.figure(figsize=(4, 4))
    plt.imshow(np.squeeze(pixels[0]), cmap="gray")
    plt.title(f"Predicted: {digit} ({output[digit]:.1%})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "custom_prediction.png"),
        dpi=150
    )
    plt.show()

    return digit, output


if CUSTOM_IMAGE_PATH and os.path.isfile(CUSTOM_IMAGE_PATH):
    predict_custom_image(CUSTOM_IMAGE_PATH)
elif CUSTOM_IMAGE_PATH:
    print("\nCustom image not found:", CUSTOM_IMAGE_PATH)
else:
    print(
        "\nTo test your own image, set CUSTOM_IMAGE_PATH "
        "near the top of this script."
    )


print("\nInspection complete!")
print("Results folder:", os.path.abspath(OUTPUT_DIR))
